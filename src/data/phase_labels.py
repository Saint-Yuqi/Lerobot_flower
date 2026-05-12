"""Per-frame pre/post-grasp phase labels for SO-101 datasets.

Reads v3.0 LeRobotDataset parquets directly (no lerobot import — works in
both the lerobot env and the flower env) and returns int8 labels aligned to
the iteration order both `FlowerSO101Dataset` and `lerobot.LeRobotDataset`
use: episode_index ascending, frame_index within each episode.

The detector is per-episode adaptive (close_frac and open_frac of
`g_max - g_min`, not absolute thresholds). The probe at
`scripts/spikes/probe_phase_detector.py` showed absolute thresholds break on
eval3 — its "closed" state sits at g≈23, not g≈1.5. Adaptive thresholds
work across all three tasks.

Public API:
    PhaseLabelResult                   — labels + episode/frame indices + summary
    compute_phase_labels(...)          — main entry: returns PhaseLabelResult, cached on disk
    find_close_frame(...)              — episode-level open->closed detector
    summarize(result, *, label="")     — small dict with pregrasp_frac, n_failed_close
    episode_grasp_frames(...)          — per-episode t* index for diagnostics

Cache key: (repo_id, revision, open_frac, close_frac, min_amplitude,
post_close_margin, sorted(episodes) hash). Action parquets are immutable per
dataset revision so we don't need an mtime check.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

LOG = logging.getLogger(__name__)

INFO_PATH = "meta/info.json"
EPISODES_DIR = "meta/episodes"

DEFAULT_OPEN_FRAC = 0.6
DEFAULT_CLOSE_FRAC = 0.4
DEFAULT_MIN_AMPLITUDE = 5.0
DEFAULT_POST_CLOSE_MARGIN = 3
DEFAULT_GRIPPER_COL = 5  # action[:, 5] for SO-101

LABEL_PREGRASP = 0
LABEL_POSTGRASP = 1


# --------------------------------------------------------------------- detector

def find_close_frame(
    g: np.ndarray,
    *,
    open_frac: float = DEFAULT_OPEN_FRAC,
    close_frac: float = DEFAULT_CLOSE_FRAC,
    min_amplitude: float = DEFAULT_MIN_AMPLITUDE,
    post_close_margin: int = DEFAULT_POST_CLOSE_MARGIN,
) -> int | None:
    """Find the first stable open->closed transition frame in `g` (gripper signal).

    Adaptive thresholds: close = g_min + close_frac * (g_max - g_min),
    open = g_min + open_frac * (g_max - g_min). Episodes where the gripper
    moves less than `min_amplitude` are treated as degenerate (returns None).

    Retries past transient dips: if the first close-candidate doesn't stay
    closed for `post_close_margin` frames, keep looking past it.
    """
    g = np.asarray(g, dtype=np.float32)
    if len(g) == 0:
        return None
    g_min, g_max = float(g.min()), float(g.max())
    amplitude = g_max - g_min
    if amplitude < min_amplitude:
        return None
    open_threshold = g_min + open_frac * amplitude
    close_threshold = g_min + close_frac * amplitude
    open_idxs = np.where(g > open_threshold)[0]
    if len(open_idxs) == 0:
        return None
    t_open = int(open_idxs[0])
    t_search = t_open
    n = len(g)
    while True:
        close_idxs = np.where(g[t_search:] < close_threshold)[0]
        if len(close_idxs) == 0:
            return None
        t_star = t_search + int(close_idxs[0])
        end = min(t_star + post_close_margin, n - 1)
        if g[t_star : end + 1].max() < close_threshold:
            return t_star
        t_search = t_star + 1


# ----------------------------------------------------------------- result type

@dataclass
class PhaseLabelResult:
    """Container for phase labels aligned to a dataset's iteration order.

    `labels[i]`, `episode_indices[i]`, `frame_indices[i]` all describe the
    same dataset row (idx `i`). `assert_dataset_alignment` uses
    episode_indices + frame_indices to verify the alignment holds on the
    in-memory dataset.
    """
    labels: np.ndarray            # int8 (N,) — 0=pre_grasp, 1=post_grasp
    episode_indices: np.ndarray   # int32 (N,)
    frame_indices: np.ndarray     # int32 (N,)
    n_episodes: int
    n_failed_close: int
    grasp_frames: dict[int, int | None] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    @property
    def pregrasp_frac(self) -> float:
        if len(self.labels) == 0:
            return 0.0
        return float((self.labels == LABEL_PREGRASP).mean())

    def slice(self, start: int, stop: int) -> "PhaseLabelResult":
        """Slice [start, stop) — for ConcatDataset offset-based wiring."""
        sub_eps = self.episode_indices[start:stop]
        unique_eps = np.unique(sub_eps).tolist()
        sub_grasp = {int(e): self.grasp_frames.get(int(e)) for e in unique_eps}
        n_failed = sum(1 for v in sub_grasp.values() if v is None)
        return PhaseLabelResult(
            labels=self.labels[start:stop].copy(),
            episode_indices=sub_eps.copy(),
            frame_indices=self.frame_indices[start:stop].copy(),
            n_episodes=len(unique_eps),
            n_failed_close=n_failed,
            grasp_frames=sub_grasp,
        )


# ------------------------------------------------------------ episode metadata

def _read_info(root: Path) -> dict[str, Any]:
    with open(root / INFO_PATH) as f:
        return json.load(f)


def _load_episode_meta(root: Path) -> list[dict]:
    """Read meta/episodes/chunk-*/file-*.parquet rows for every episode."""
    ep_root = root / EPISODES_DIR
    files = sorted(ep_root.rglob("file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata parquets under {ep_root}")
    out: list[dict] = []
    for f in files:
        df = pq.read_table(f).to_pandas()
        for _, row in df.iterrows():
            out.append({
                "episode_index": int(row["episode_index"]),
                "dataset_from_index": int(row["dataset_from_index"]),
                "dataset_to_index": int(row["dataset_to_index"]),
                "data_chunk": int(row["data/chunk_index"]),
                "data_file": int(row["data/file_index"]),
            })
    out.sort(key=lambda e: e["episode_index"])
    return out


def _load_gripper(root: Path, ep_meta: dict, gripper_col: int) -> np.ndarray:
    """Read the gripper column (action[:, gripper_col]) for one episode."""
    info = _read_info(root)
    data_path_fmt: str = info["data_path"]
    path = root / data_path_fmt.format(
        chunk_index=ep_meta["data_chunk"], file_index=ep_meta["data_file"],
    )
    tbl = pq.read_table(path, columns=["episode_index", "action", "frame_index"])
    df = tbl.to_pandas()
    df = df[df["episode_index"] == ep_meta["episode_index"]].sort_values("frame_index")
    actions = np.stack([np.asarray(a, dtype=np.float32) for a in df["action"].values])
    return actions[:, gripper_col]


# ------------------------------------------------------------ cache helpers

def _sanitize(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _default_cache_dir() -> Path:
    env = os.environ.get("LEROBOT_PHASE_CACHE")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache/lerobot_phase"


def _cache_path(
    cache_dir: Path,
    repo_id: str,
    revision: str,
    open_frac: float,
    close_frac: float,
    min_amplitude: float,
    post_close_margin: int,
    episodes: list[int] | None,
) -> Path:
    eps_key = "all" if episodes is None else hashlib.sha1(
        ",".join(str(e) for e in sorted(int(x) for x in episodes)).encode()
    ).hexdigest()[:10]
    name = (
        f"{_sanitize(repo_id)}__{_sanitize(revision)}"
        f"__of{open_frac:g}_cf{close_frac:g}_a{min_amplitude:g}_m{post_close_margin}"
        f"__eps{eps_key}.npz"
    )
    return cache_dir / name


# ------------------------------------------------------------------- main API

def compute_phase_labels(
    repo_id: str,
    root: Path | None,
    episodes: list[int] | None = None,
    *,
    revision: str = "v3.0",
    open_frac: float = DEFAULT_OPEN_FRAC,
    close_frac: float = DEFAULT_CLOSE_FRAC,
    min_amplitude: float = DEFAULT_MIN_AMPLITUDE,
    post_close_margin: int = DEFAULT_POST_CLOSE_MARGIN,
    gripper_col: int = DEFAULT_GRIPPER_COL,
    cache_dir: Path | None = None,
    force: bool = False,
) -> PhaseLabelResult:
    """Compute per-frame phase labels for a dataset.

    Iteration order matches FlowerSO101Dataset and lerobot.LeRobotDataset:
    episode_index ascending, then frame_index within each episode. The
    resulting `labels` array has the same length and order as the dataset's
    `__getitem__` indexing.

    Args:
        repo_id: HF Hub dataset id (for cache key + logs).
        root: Local dataset root. If None, expects an HF snapshot already
            cached on disk; we won't trigger a download.
        episodes: Subset of episode indices to include. None = all in dataset.
        revision: HF revision (for cache key).
        open_frac, close_frac, min_amplitude, post_close_margin: detector
            params; see `find_close_frame`.
        gripper_col: which column of the `action` vector holds the gripper
            signal (5 for SO-101).
        cache_dir: where to read/write the cached npz. Defaults to
            $LEROBOT_PHASE_CACHE or ~/.cache/lerobot_phase.
        force: ignore cache, recompute.
    """
    if root is None:
        raise ValueError("compute_phase_labels: root must be provided (already-cached snapshot path)")
    root = Path(root)
    cache_dir = Path(cache_dir).expanduser() if cache_dir else _default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path(
        cache_dir, repo_id, revision, open_frac, close_frac,
        min_amplitude, post_close_margin, episodes,
    )

    if cache_path.exists() and not force:
        try:
            data = np.load(cache_path, allow_pickle=False)
            labels = data["labels"]
            episode_indices = data["episode_indices"]
            frame_indices = data["frame_indices"]
            grasp_eps = data["grasp_eps"]
            grasp_ts = data["grasp_ts"]   # -1 sentinel = None
            grasp_frames = {
                int(e): (int(t) if int(t) >= 0 else None)
                for e, t in zip(grasp_eps, grasp_ts)
            }
            n_failed = sum(1 for v in grasp_frames.values() if v is None)
            return PhaseLabelResult(
                labels=labels,
                episode_indices=episode_indices,
                frame_indices=frame_indices,
                n_episodes=len(grasp_frames),
                n_failed_close=n_failed,
                grasp_frames=grasp_frames,
            )
        except Exception as e:
            LOG.warning("phase-label cache load failed (%s); recomputing.", e)

    all_eps = _load_episode_meta(root)
    if episodes is not None:
        keep = set(int(e) for e in episodes)
        all_eps = [e for e in all_eps if e["episode_index"] in keep]
        if not all_eps:
            raise ValueError(
                f"compute_phase_labels: none of {episodes} match available episode indices"
            )

    label_chunks: list[np.ndarray] = []
    ep_chunks: list[np.ndarray] = []
    frame_chunks: list[np.ndarray] = []
    grasp_frames: dict[int, int | None] = {}
    n_failed = 0

    for ep in all_eps:
        g = _load_gripper(root, ep, gripper_col)
        ep_len = len(g)
        t_star = find_close_frame(
            g,
            open_frac=open_frac, close_frac=close_frac,
            min_amplitude=min_amplitude,
            post_close_margin=post_close_margin,
        )
        grasp_frames[ep["episode_index"]] = t_star
        if t_star is None:
            n_failed += 1
            pre_end = ep_len  # all-pregrasp
        else:
            pre_end = min(t_star + post_close_margin + 1, ep_len)
        labels = np.full(ep_len, LABEL_POSTGRASP, dtype=np.int8)
        labels[:pre_end] = LABEL_PREGRASP
        label_chunks.append(labels)
        ep_chunks.append(np.full(ep_len, ep["episode_index"], dtype=np.int32))
        frame_chunks.append(np.arange(ep_len, dtype=np.int32))

    labels_arr = np.concatenate(label_chunks) if label_chunks else np.zeros(0, dtype=np.int8)
    ep_arr = np.concatenate(ep_chunks) if ep_chunks else np.zeros(0, dtype=np.int32)
    frame_arr = np.concatenate(frame_chunks) if frame_chunks else np.zeros(0, dtype=np.int32)

    # Cache (-1 sentinel for None t*).
    grasp_eps_arr = np.array(list(grasp_frames.keys()), dtype=np.int32)
    grasp_ts_arr = np.array(
        [v if v is not None else -1 for v in grasp_frames.values()], dtype=np.int32,
    )
    # np.savez auto-appends .npz; write to a sibling .tmp then rename.
    tmp_path = cache_path.with_name(cache_path.name + ".tmp")
    with open(tmp_path, "wb") as fh:
        np.savez_compressed(
            fh,
            labels=labels_arr, episode_indices=ep_arr, frame_indices=frame_arr,
            grasp_eps=grasp_eps_arr, grasp_ts=grasp_ts_arr,
        )
    os.replace(tmp_path, cache_path)

    return PhaseLabelResult(
        labels=labels_arr,
        episode_indices=ep_arr,
        frame_indices=frame_arr,
        n_episodes=len(all_eps),
        n_failed_close=n_failed,
        grasp_frames=grasp_frames,
    )


def summarize(result: PhaseLabelResult, *, label: str = "") -> dict:
    """Small summary dict; logs a warning when failed-close rate exceeds 10%."""
    n_total = int(len(result))
    n_pregrasp = int((result.labels == LABEL_PREGRASP).sum())
    n_eps = max(int(result.n_episodes), 1)
    fail_rate = result.n_failed_close / n_eps
    out = {
        "label": label,
        "n_total": n_total,
        "n_pregrasp": n_pregrasp,
        "pregrasp_frac": result.pregrasp_frac,
        "n_failed_close": int(result.n_failed_close),
        "n_episodes": int(result.n_episodes),
        "failed_close_rate": fail_rate,
    }
    if fail_rate > 0.10:
        LOG.warning(
            "phase_labels: %.1f%% of episodes failed grasp-close detection (%d/%d) "
            "for %r — consider checking the dataset or detector params.",
            fail_rate * 100, result.n_failed_close, result.n_episodes, label,
        )
    return out


def episode_grasp_frames(
    repo_id: str,
    root: Path | None,
    episodes: list[int] | None = None,
    **kwargs,
) -> dict[int, int | None]:
    """Per-episode t* index (None if no stable close detected)."""
    return compute_phase_labels(repo_id, root, episodes, **kwargs).grasp_frames


# ------------------------------------------------------------------ CLI debug

def _main():
    import argparse
    ap = argparse.ArgumentParser(description="Debug: compute phase labels on a dataset.")
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--open-frac", type=float, default=DEFAULT_OPEN_FRAC)
    ap.add_argument("--close-frac", type=float, default=DEFAULT_CLOSE_FRAC)
    ap.add_argument("--min-amplitude", type=float, default=DEFAULT_MIN_AMPLITUDE)
    ap.add_argument("--post-close-margin", type=int, default=DEFAULT_POST_CLOSE_MARGIN)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--print-histogram", action="store_true")
    args = ap.parse_args()

    root = Path(args.root) if args.root else None
    if root is None:
        try:
            from huggingface_hub import snapshot_download
            root = Path(snapshot_download(
                repo_id=args.repo_id, repo_type="dataset", revision="v3.0",
                allow_patterns=["meta/*", "data/**"],
            ))
        except Exception as e:
            raise SystemExit(f"need --root (snapshot_download failed: {e})")

    result = compute_phase_labels(
        repo_id=args.repo_id, root=root, episodes=args.episodes,
        open_frac=args.open_frac, close_frac=args.close_frac,
        min_amplitude=args.min_amplitude, post_close_margin=args.post_close_margin,
        force=args.force,
    )
    summ = summarize(result, label=args.repo_id)
    print(json.dumps(summ, indent=2))

    if args.print_histogram:
        ts = [v for v in result.grasp_frames.values() if v is not None]
        if ts:
            ts_arr = np.array(ts)
            print(f"\nt* histogram (n={len(ts)}):")
            print(f"  min={ts_arr.min()}, mean={ts_arr.mean():.1f}, max={ts_arr.max()}")
            print(f"  quartiles: 25%={np.percentile(ts_arr,25):.0f} "
                  f"50%={np.percentile(ts_arr,50):.0f} "
                  f"75%={np.percentile(ts_arr,75):.0f}")


if __name__ == "__main__":
    _main()
