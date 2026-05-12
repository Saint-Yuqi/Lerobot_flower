"""Color-stratified train/val episode splitting for LeRobot v3.0 datasets.

The split is by *episode index*, not by frame, because frames within one
episode are correlated and a frame-level split would leak future actions
into the val set.

Source of truth for the per-episode prompt is
`<root>/meta/episodes/chunk-*/file-*.parquet`'s `tasks` column (list[str]).
We bucket each episode by which color word (`blue`/`red`/`green`) appears
in its task strings; episodes with none, or with more than one, go in
`other` with a warning.
"""
from __future__ import annotations

import glob
import logging
import math
import re
from pathlib import Path

import pyarrow.parquet as pq

LOG = logging.getLogger(__name__)

COLORS = ("blue", "red", "green")
# Word-boundary regex so "red" doesn't match "colo*red*" — substring match
# would mis-bucket every blue/green prompt as containing red.
_COLOR_RE = {c: re.compile(rf"\b{c}\b", re.IGNORECASE) for c in COLORS}


def _classify_tasks(tasks) -> str:
    """Pick the single color label for one episode's task list, or 'other'."""
    found = {c for c in COLORS for t in tasks if _COLOR_RE[c].search(t)}
    if len(found) == 1:
        return next(iter(found))
    return "other"


def episodes_by_color(repo_id: str, root: str | Path | None) -> dict[str, list[int]]:
    """Group episode indices by which color word appears in their task prompt.

    Args:
        repo_id: HF dataset id (e.g. "ethrl2026/so101_..."). Used to resolve
            the local cache path when `root` is None.
        root: dataset root containing `meta/episodes/chunk-*/file-*.parquet`.
            If None (HF auto-download), `LeRobotDatasetMetadata` is used to
            locate the cache dir. The dataset must already be downloaded —
            this function only reads parquet, never fetches.

    Returns:
        Dict like `{"blue": [0,3,...], "red": [...], "green": [...], "other": [...]}`.
        Empty buckets are *omitted* so callers can iterate without filtering.
    """
    if root is None:
        from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
        root = LeRobotDatasetMetadata(repo_id=repo_id).root
    root = Path(root)
    files = sorted(glob.glob(str(root / "meta/episodes/chunk-*/file-*.parquet")))
    if not files:
        raise FileNotFoundError(
            f"no episode parquet files under {root}/meta/episodes/ — is repo_id={repo_id!r} initialized?"
        )

    by: dict[str, list[int]] = {}
    n_other = 0
    for f in files:
        df = pq.read_table(f, columns=["episode_index", "tasks"]).to_pandas()
        for ep_idx, tasks in zip(df["episode_index"], df["tasks"]):
            label = _classify_tasks(tasks)
            by.setdefault(label, []).append(int(ep_idx))
            if label == "other":
                n_other += 1
                LOG.warning(
                    "episode %d has no single color match in tasks=%r — bucketing as 'other'",
                    ep_idx, list(tasks),
                )

    for k in by:
        by[k] = sorted(by[k])

    LOG.info(
        "episodes_by_color(%s): %s",
        root,
        {k: len(v) for k, v in by.items()},
    )
    return by


def train_val_episode_split(
    by_color: dict[str, list[int]],
    *,
    per_color: int | None = None,
    fraction: float | None = None,
    min_train_per_color: int = 3,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """Stratified train/val split over episode indices.

    Exactly one of `per_color` (absolute count) or `fraction` (relative) must
    be supplied. For each non-empty color bucket, we hold out
    `n_val = per_color` or `max(1, ceil(fraction * n))` episodes; the
    remaining go to train. If any color would be left with fewer than
    `min_train_per_color` training episodes, we abort with `ValueError`
    naming the offending color and counts.

    Args:
        by_color: output of `episodes_by_color`. Buckets are processed in
            sorted-key order for stability across runs.
        per_color: held out per color, integer. Mutually exclusive with `fraction`.
        fraction: per-color fraction in (0, 1). Mutually exclusive with `per_color`.
        min_train_per_color: safety floor on training episodes per color.
        seed: shuffle seed for picking which episodes land in val.

    Returns:
        `(train_ids, val_ids)`, each sorted. Both lists are disjoint and their
        union equals all episode indices in `by_color`.
    """
    if (per_color is None) == (fraction is None):
        raise ValueError("supply exactly one of per_color or fraction")
    if fraction is not None and not 0.0 < fraction < 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction!r}")
    if per_color is not None and per_color < 1:
        raise ValueError(f"per_color must be >= 1, got {per_color!r}")

    import random
    rng = random.Random(seed)

    train_ids: list[int] = []
    val_ids: list[int] = []
    for color in sorted(by_color):
        episodes = list(by_color[color])
        n = len(episodes)
        if n == 0:
            continue
        if per_color is not None:
            n_val = per_color
        else:
            n_val = max(1, math.ceil(fraction * n))

        n_train = n - n_val
        if n_train < min_train_per_color:
            raise ValueError(
                f"color {color!r} has only {n} episodes; holding out {n_val} "
                f"would leave {n_train} training episodes (< min_train_per_color={min_train_per_color}). "
                f"Lower per_color/fraction or record more episodes for this color."
            )

        shuffled = episodes[:]
        rng.shuffle(shuffled)
        val_for_color = sorted(shuffled[:n_val])
        train_for_color = sorted(shuffled[n_val:])
        val_ids.extend(val_for_color)
        train_ids.extend(train_for_color)
        LOG.info("split[%s]: train=%d val=%d (val_ids=%s)",
                 color, len(train_for_color), len(val_for_color), val_for_color)

    return sorted(train_ids), sorted(val_ids)
