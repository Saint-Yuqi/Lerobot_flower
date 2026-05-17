"""SO-101 LeRobotDataset v3.0 reader for the flower-env training stack.

Mirrors what `lerobot.datasets.lerobot_dataset.LeRobotDataset` produces — frames
with action chunks at requested delta_timestamps — but does not import lerobot.
Lerobot 0.5.x pins Python>=3.12; the flower env is pinned at 3.10 to keep
torch==2.2.2 and pytorch-lightning==2.0.8 happy, so we cannot install lerobot
in this env. Instead we read the v3.0 on-disk format directly with pyarrow +
PyAV (already installed) and `huggingface_hub.snapshot_download` for fetching.

Dataset v3.0 layout (HF Hub repos):
    meta/info.json
    meta/stats.json
    meta/tasks.parquet
    meta/episodes/chunk-NNN/file-NNN.parquet   (episode metadata, may be 1 file)
    data/chunk-NNN/file-NNN.parquet            (frame rows: state, action, task_index, ...)
    videos/<video_key>/chunk-NNN/file-NNN.mp4  (av1-encoded video)

A single parquet "file" can hold many episodes (lerobot rolls 100 MB per file
as `data_files_size_in_mb` in info.json). Each episode's row in `meta/episodes`
points at the (chunk_index, file_index) it lives in for both data and each
video stream.

Batch shape returned by `__getitem__`:
    "observation.images.main": Tensor (C, H, W) float32 in [0, 1], square if resize_hw set
    "observation.state":       Tensor (state_dim,) float32
    "action":                  Tensor (chunk_size, action_dim) float32
    "action_is_pad":           Tensor (chunk_size,) bool  — True where last action repeated past episode end
    "task":                    str
    "episode_index":           int
    "frame_index":             int  (within episode)
    "index":                   int  (global, dataset-wide)
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from huggingface_hub import snapshot_download


INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
TASKS_PATH = "meta/tasks.parquet"
EPISODES_DIR = "meta/episodes"


@dataclass
class _EpisodeIndex:
    """One episode's address: where its frames live on disk."""
    episode_index: int
    dataset_from_index: int   # global frame index, inclusive
    dataset_to_index: int     # global frame index, exclusive
    data_chunk: int
    data_file: int
    video_chunks: dict[str, int]  # video_key -> chunk_index
    video_files: dict[str, int]   # video_key -> file_index
    tasks: list[str]              # task strings present in this episode


class FlowerSO101Dataset(torch.utils.data.Dataset):
    """Read a v3.0 LeRobotDataset (SO-101 follower) without depending on lerobot.

    Args:
        repo_id: HF Hub dataset id (e.g. "ethrl2026/so101_pickup_..._task3"). When
            ``root`` is provided, ``repo_id`` is only used for cache keys / logs.
        root: Optional local directory. If None, the dataset is snapshot_download'd
            into the HF cache.
        revision: HF git revision/tag (default "v3.0" — matches our datasets).
        episodes: Optional list of episode indices (zero-based) to keep. None = all.
        chunk_size: How many consecutive actions to bundle per sample (default 50).
        video_key: Which video feature to read (default "observation.images.main").
        resize_hw: If set, bilinearly resize images to (resize_hw, resize_hw) — needed
            for Florence-2 which asserts square feature maps.
        max_decoded_videos: LRU cap on open PyAV containers per worker (default 8).
        video_backend: "pyav" (default; works for av1) or "opencv".
        image_transforms: Optional callable applied to the image tensor post-resize,
            pre-return — see `src.data.image_transforms.build_image_transforms`.

    Notes on action padding:
        When chunk_size > 1 and idx+k exceeds the episode end, we repeat the last
        valid action and set ``action_is_pad[k] = True``. This matches what
        `lerobot.datasets.factory.resolve_delta_timestamps` produces and gives the
        policy a chance to mask padded targets in its loss.
    """

    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str = "v3.0",
        episodes: list[int] | None = None,
        chunk_size: int = 50,
        video_key: str = "observation.images.main",
        resize_hw: int | None = 224,
        max_decoded_videos: int = 8,
        video_backend: str = "pyav",
        image_transforms: Callable[[torch.Tensor], torch.Tensor] | None = None,
        frame_cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.repo_id = repo_id
        self.revision = revision
        self.video_key = video_key
        self.chunk_size = int(chunk_size)
        self.frame_cache_dir = frame_cache_dir
        self.resize_hw = int(resize_hw) if resize_hw else None
        self.max_decoded_videos = int(max_decoded_videos)
        self.video_backend = video_backend
        self.image_transforms = image_transforms

        self.root = Path(root).expanduser() if root else None
        if self.root is None:
            self.root = Path(snapshot_download(
                repo_id=repo_id, repo_type="dataset", revision=revision,
                allow_patterns=["meta/*", "data/**", "videos/**"],
            ))
        if not (self.root / INFO_PATH).exists():
            raise FileNotFoundError(f"{self.root}/{INFO_PATH} missing — bad root?")

        with open(self.root / INFO_PATH) as f:
            self.info: dict[str, Any] = json.load(f)
        if self.info.get("codebase_version", "") != "v3.0":
            raise ValueError(
                f"Only v3.0 datasets are supported here; got "
                f"codebase_version={self.info.get('codebase_version')!r}"
            )
        self.fps: int = int(self.info["fps"])
        self.data_path_fmt: str = self.info["data_path"]
        self.video_path_fmt: str = self.info["video_path"]
        if self.video_key not in self.info["features"]:
            raise KeyError(
                f"video_key {self.video_key!r} not in features "
                f"{list(self.info['features'])}"
            )

        # Tasks: task_index → task string. v3.0 stores task as the parquet index
        # and task_index as the only column (see lerobot.datasets.io_utils.load_tasks).
        tasks_table = pq.read_table(self.root / TASKS_PATH)
        tasks_df = tasks_table.to_pandas().reset_index()
        if "task_index" not in tasks_df.columns or "task" not in tasks_df.columns:
            raise ValueError(
                f"meta/tasks.parquet has unexpected columns {list(tasks_df.columns)} "
                "(expected 'task_index' + 'task' after reset_index)"
            )
        self._task_lookup = dict(zip(tasks_df["task_index"], tasks_df["task"]))

        self._episodes = self._load_episodes()
        if episodes is not None:
            ep_keep = set(int(e) for e in episodes)
            self._episodes = [e for e in self._episodes if e.episode_index in ep_keep]
            if not self._episodes:
                raise ValueError(f"None of {episodes} match available episode indices.")

        # Total frames after filtering.
        self._total_frames = sum(
            e.dataset_to_index - e.dataset_from_index for e in self._episodes
        )

        # Build cumulative offsets so we can binary-search idx → episode quickly.
        self._cum_starts = np.zeros(len(self._episodes) + 1, dtype=np.int64)
        for i, e in enumerate(self._episodes):
            self._cum_starts[i + 1] = (
                self._cum_starts[i] + (e.dataset_to_index - e.dataset_from_index)
            )

        # Per-worker LRU for parquet files and PyAV containers; init lazily so the
        # dataset is fork-safe (each DataLoader worker rebuilds these on first use).
        self._parquet_cache: dict[str, "pq.ParquetFile"] = {}
        self._video_cache: dict[str, av.container.InputContainer] = {}
        self._video_cache_order: list[str] = []
        self._tls = threading.local()

        # State/action dims from info schema.
        self.state_dim: int = int(self.info["features"]["observation.state"]["shape"][0])
        self.action_dim: int = int(self.info["features"]["action"]["shape"][0])

        # Stats (optional but training will want them).
        stats_path = self.root / STATS_PATH
        self.stats: dict[str, dict[str, np.ndarray]] | None = None
        if stats_path.exists():
            with open(stats_path) as f:
                raw_stats = json.load(f)
            self.stats = {
                k: {sk: np.asarray(sv, dtype=np.float32) for sk, sv in v.items()}
                for k, v in raw_stats.items()
            }

        # Optional decode-once frame cache (native uint8 memmap). When present
        # __getitem__ reads the exact bytes _decode_frame_at would produce,
        # skipping PyAV — bit-identical, just faster. None until built.
        self._maybe_open_frame_cache()

    # ------------------------------------------------- native frame cache

    def _frame_cache_paths(self) -> tuple[Path, Path]:
        safe = self.repo_id.replace("/", "__")
        stem = f"{safe}__{self.revision}"
        base = Path(self.frame_cache_dir).expanduser()
        return base / f"{stem}.u8", base / f"{stem}.json"

    def _maybe_open_frame_cache(self) -> None:
        self._frame_cache = None
        self._frame_cache_path = None
        self._frame_cache_shape = None
        if self.frame_cache_dir is None:
            return
        mm_path, meta_path = self._frame_cache_paths()
        if not (mm_path.exists() and meta_path.exists()):
            return  # not built yet — decode path used until build_frame_cache runs
        meta = json.loads(meta_path.read_text())
        if meta.get("repo_id") != self.repo_id or meta.get("revision") != self.revision:
            raise ValueError(f"frame cache {meta_path} repo/revision mismatch: {meta}")
        shape = (int(meta["n_frames"]), int(meta["height"]),
                 int(meta["width"]), int(meta["channels"]))
        self._frame_cache = np.memmap(mm_path, dtype=np.uint8, mode="r", shape=shape)
        self._frame_cache_path = str(mm_path)
        self._frame_cache_shape = shape

    def _frame_addr(self, idx: int) -> tuple["_EpisodeIndex", int, float]:
        """(episode, global_dataset_idx, timestamp) for a flat idx — image only.

        Mirrors the locate logic in __getitem__ but reads just the timestamp;
        used by build_frame_cache so __getitem__'s hot path stays untouched.
        """
        ep_i = int(np.searchsorted(self._cum_starts, idx, side="right") - 1)
        ep = self._episodes[ep_i]
        local_idx = idx - self._cum_starts[ep_i]
        global_dataset_idx = int(ep.dataset_from_index + local_idx)
        data_table = self._get_parquet_table(ep.data_chunk, ep.data_file)
        ep_mask = data_table.column("episode_index").to_numpy() == ep.episode_index
        ep_rows = data_table.filter(ep_mask).slice(local_idx, length=1)
        timestamp = float(ep_rows.column("timestamp").to_pylist()[0])
        return ep, global_dataset_idx, timestamp

    def build_frame_cache(self, progress_every: int = 2000, verify_n: int = 16) -> None:
        """Decode every frame once into a native-uint8 memmap keyed by global
        dataset index. Must run on an UNfiltered dataset (episodes=None) so all
        global indices are written. Verifies bit-equality vs fresh decode."""
        if self.frame_cache_dir is None:
            raise ValueError("frame_cache_dir not set")
        mm_path, meta_path = self._frame_cache_paths()
        mm_path.parent.mkdir(parents=True, exist_ok=True)

        n_frames = int(max(int(e.dataset_to_index) for e in self._episodes))
        ep0, _g0, ts0 = self._frame_addr(0)
        probe = self._decode_frame_at(ep0, ts0)
        h, w, c = int(probe.shape[0]), int(probe.shape[1]), int(probe.shape[2])

        mm = np.memmap(mm_path, dtype=np.uint8, mode="w+", shape=(n_frames, h, w, c))
        filled = np.zeros(n_frames, dtype=bool)
        total = self._total_frames
        for idx in range(total):
            ep, gidx, ts = self._frame_addr(idx)
            frame = self._decode_frame_at(ep, ts)
            if frame.shape != (h, w, c):
                raise ValueError(f"frame {idx} shape {frame.shape} != ({h},{w},{c})")
            mm[gidx] = frame
            filled[gidx] = True
            if progress_every and idx % progress_every == 0:
                print(f"[frame-cache] {idx}/{total} (gidx={gidx})", flush=True)
        mm.flush()

        n_missing = int((~filled).sum())
        if n_missing:
            raise RuntimeError(
                f"frame cache has {n_missing} unfilled slots — gaps in global index"
            )

        rng = np.random.default_rng(0)
        sample = rng.choice(total, size=min(verify_n, total), replace=False)
        for idx in sample:
            ep, gidx, ts = self._frame_addr(int(idx))
            fresh = self._decode_frame_at(ep, ts)
            if not np.array_equal(fresh, np.asarray(mm[gidx])):
                raise RuntimeError(
                    f"VERIFY FAIL idx={idx} gidx={gidx} — cache != fresh decode"
                )
        meta = {"repo_id": self.repo_id, "revision": self.revision,
                "n_frames": n_frames, "height": h, "width": w, "channels": c}
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[frame-cache] built {n_frames} frames {h}x{w}x{c} -> {mm_path} "
              f"(verified {len(sample)} samples bit-identical)", flush=True)

    # ----------------------------------------------------- pickling

    # PyAV InputContainer and threading.local are not picklable. DataLoader
    # workers with multiprocessing_context='spawn' (required under DDP) need
    # to pickle the dataset; strip the caches and rebuild them in __setstate__
    # so each worker starts fresh.
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_parquet_cache"] = {}
        state["_video_cache"] = {}
        state["_video_cache_order"] = []
        state["_tls"] = None
        state["_frame_cache"] = None  # np.memmap reopened in __setstate__
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._tls = threading.local()
        if self._frame_cache_path is not None and self._frame_cache_shape is not None:
            self._frame_cache = np.memmap(
                self._frame_cache_path, dtype=np.uint8, mode="r",
                shape=tuple(self._frame_cache_shape),
            )

    # ----------------------------------------------------- pytorch Dataset API

    def __len__(self) -> int:
        return self._total_frames

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self._total_frames:
            raise IndexError(idx)

        # Locate episode for this idx.
        ep_idx_in_filtered = int(np.searchsorted(self._cum_starts, idx, side="right") - 1)
        ep = self._episodes[ep_idx_in_filtered]
        local_idx = idx - self._cum_starts[ep_idx_in_filtered]  # 0-based within ep
        global_dataset_idx = ep.dataset_from_index + local_idx

        # Read frame row.
        data_table = self._get_parquet_table(ep.data_chunk, ep.data_file)
        # Filter to this episode in case multiple episodes share the file.
        ep_mask = data_table.column("episode_index").to_numpy() == ep.episode_index
        ep_rows = data_table.filter(ep_mask).slice(local_idx, length=self.chunk_size)
        # The first row of ep_rows is our anchor frame.
        state_arr = np.asarray(ep_rows.column("observation.state").to_pylist()[0], dtype=np.float32)
        action_full = ep_rows.column("action").to_pylist()  # list of lists, len <= chunk_size
        task_index = int(ep_rows.column("task_index").to_pylist()[0])
        timestamp = float(ep_rows.column("timestamp").to_pylist()[0])
        frame_index = int(ep_rows.column("frame_index").to_pylist()[0])

        # Action chunk — pad by repeating the last valid action.
        actions = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
        is_pad = np.zeros((self.chunk_size,), dtype=bool)
        n_valid = len(action_full)
        for i in range(self.chunk_size):
            src_i = min(i, n_valid - 1)
            actions[i] = np.asarray(action_full[src_i], dtype=np.float32)
            is_pad[i] = i >= n_valid

        # Image frame at timestamp. The memmap cache (when built) holds the
        # exact bytes _decode_frame_at returns — bit-identical, skips PyAV.
        if self._frame_cache is not None:
            img = np.asarray(self._frame_cache[global_dataset_idx])
        else:
            img = self._decode_frame_at(ep, timestamp)
        # HWC uint8 -> CHW float32 in [0,1].
        img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float().div_(255.0)
        if self.resize_hw is not None and (img_t.shape[-1] != self.resize_hw or img_t.shape[-2] != self.resize_hw):
            img_t = torch.nn.functional.interpolate(
                img_t.unsqueeze(0), size=(self.resize_hw, self.resize_hw),
                mode="bilinear", align_corners=False, antialias=True,
            ).squeeze(0)
        if self.image_transforms is not None:
            img_t = self.image_transforms(img_t)

        task_str = self._task_lookup.get(task_index, "")

        return {
            self.video_key: img_t,
            "observation.state": torch.from_numpy(state_arr),
            "action": torch.from_numpy(actions),
            "action_is_pad": torch.from_numpy(is_pad),
            "task": task_str,
            "task_index": task_index,
            "episode_index": int(ep.episode_index),
            "frame_index": frame_index,
            "index": int(global_dataset_idx),
            "timestamp": timestamp,
        }

    # ----------------------------------------------------------- helpers

    def _load_episodes(self) -> list[_EpisodeIndex]:
        """Read meta/episodes/chunk-*/file-*.parquet and build per-episode entries."""
        ep_root = self.root / EPISODES_DIR
        files = sorted(ep_root.rglob("file-*.parquet"))
        if not files:
            raise FileNotFoundError(f"No episode metadata parquets under {ep_root}")
        eps: list[_EpisodeIndex] = []
        # Find all video keys present (so we can grab per-episode chunk/file refs).
        video_keys = [
            k for k, ft in self.info["features"].items() if ft.get("dtype") == "video"
        ]
        for f in files:
            tbl = pq.read_table(f)
            df = tbl.to_pandas()
            # Episode rows include: dataset_from_index, dataset_to_index, length,
            # data/chunk_index, data/file_index, videos/<key>/chunk_index, videos/<key>/file_index,
            # tasks (list of str), and optionally per-episode stats prefixed "stats/...".
            for _, row in df.iterrows():
                tasks = list(row.get("tasks") or [])
                if isinstance(tasks, np.ndarray):
                    tasks = tasks.tolist()
                vid_chunks = {}
                vid_files = {}
                for vk in video_keys:
                    vid_chunks[vk] = int(row[f"videos/{vk}/chunk_index"])
                    vid_files[vk] = int(row[f"videos/{vk}/file_index"])
                eps.append(_EpisodeIndex(
                    episode_index=int(row["episode_index"]),
                    dataset_from_index=int(row["dataset_from_index"]),
                    dataset_to_index=int(row["dataset_to_index"]),
                    data_chunk=int(row["data/chunk_index"]),
                    data_file=int(row["data/file_index"]),
                    video_chunks=vid_chunks,
                    video_files=vid_files,
                    tasks=tasks,
                ))
        eps.sort(key=lambda e: e.episode_index)
        return eps

    def _get_parquet_table(self, chunk_idx: int, file_idx: int):
        """Cache the full data parquet file in-memory (~100MB; trades RAM for speed)."""
        key = f"data:{chunk_idx}:{file_idx}"
        if key not in self._parquet_cache:
            path = self.root / self.data_path_fmt.format(
                chunk_index=chunk_idx, file_index=file_idx,
            )
            if not path.exists():
                raise FileNotFoundError(path)
            self._parquet_cache[key] = pq.read_table(str(path))
        return self._parquet_cache[key]

    def _decode_frame_at(self, ep: _EpisodeIndex, timestamp: float) -> np.ndarray:
        """Return one HWC uint8 frame at the requested timestamp.

        Uses PyAV's keyframe seek + linear forward decode (the same pattern as
        lerobot's `decode_video_frames_torchvision`). For our 30 fps SO-101 videos
        the GOP is small so this is fast in practice.
        """
        video_rel = self.video_path_fmt.format(
            video_key=self.video_key,
            chunk_index=ep.video_chunks[self.video_key],
            file_index=ep.video_files[self.video_key],
        )
        video_path = str(self.root / video_rel)

        # A single parquet "file" can hold multiple episodes. Episode boundary inside
        # the mp4 is timestamp-based; the timestamp we just pulled from the parquet
        # is GLOBAL (in seconds from the start of the video file, which == the start
        # of the first episode it contains). PyAV seeks on the same scale.
        container = self._get_video_container(video_path)
        stream = container.streams.video[0]

        target_pts = int(round(timestamp / float(stream.time_base)))
        try:
            container.seek(target_pts, stream=stream, backward=True, any_frame=False)
        except av.AVError:
            container.seek(0)

        best = None
        best_pts_diff = None
        # Decode forward from keyframe. Stop a few frames past target to be safe.
        for frame in container.decode(stream):
            frame_pts = int(frame.pts) if frame.pts is not None else 0
            diff = abs(frame_pts - target_pts)
            if best is None or diff < best_pts_diff:
                best = frame
                best_pts_diff = diff
            if frame_pts >= target_pts:
                break

        if best is None:
            raise RuntimeError(
                f"Failed to decode any frame from {video_path} at t={timestamp}s"
            )
        # Convert to HxWxC uint8 RGB.
        return best.to_ndarray(format="rgb24")

    def _get_video_container(self, video_path: str) -> av.container.InputContainer:
        if video_path in self._video_cache:
            # Move to MRU end.
            try:
                self._video_cache_order.remove(video_path)
            except ValueError:
                pass
            self._video_cache_order.append(video_path)
            return self._video_cache[video_path]
        # Open new; LRU evict oldest if over capacity.
        container = av.open(video_path)
        # threading_type 'AUTO' (PyAV default) lets ffmpeg use multiple threads.
        self._video_cache[video_path] = container
        self._video_cache_order.append(video_path)
        if len(self._video_cache_order) > self.max_decoded_videos:
            old = self._video_cache_order.pop(0)
            try:
                self._video_cache[old].close()
            except Exception:
                pass
            del self._video_cache[old]
        return container

    # ------------------------------------------------------ introspection

    def episodes_indices(self) -> list[int]:
        return [e.episode_index for e in self._episodes]

    def __repr__(self) -> str:
        return (
            f"FlowerSO101Dataset(repo_id={self.repo_id!r}, frames={self._total_frames}, "
            f"episodes={len(self._episodes)}, chunk={self.chunk_size}, "
            f"video_key={self.video_key!r}, resize_hw={self.resize_hw})"
        )
