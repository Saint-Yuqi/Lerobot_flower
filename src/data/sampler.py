"""Phase-weighted sampler + dataset/label alignment asserts.

`make_phase_weighted_sampler` wraps `torch.utils.data.WeightedRandomSampler`
with sane defaults for the pre/post-grasp binary labelling:
  - replacement=True (the whole point: pre-grasp frames seen multiple times)
  - num_samples=len(weights) by default, so step count per epoch is unchanged
  - explicit `torch.Generator` so the per-run seed is honored

The alignment asserts (`assert_dataset_alignment`,
`assert_concat_alignment`) turn a silent labels-to-dataset misalignment into
a loud startup error: they pick indices spread across episode boundaries,
read `dataset[i]['episode_index']` and `frame_index`, and compare against
the PhaseLabelResult's per-frame metadata.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import ConcatDataset, WeightedRandomSampler

from src.data.phase_labels import PhaseLabelResult, LABEL_PREGRASP


def make_phase_weighted_sampler(
    phase_labels: np.ndarray | PhaseLabelResult,
    weight_pregrasp: float,
    *,
    num_samples: int | None = None,
    replacement: bool = True,
    seed: int = 42,
) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler that upweights pre-grasp frames.

    Args:
        phase_labels: int8 array (or PhaseLabelResult) where 0 = pre_grasp,
            1 = post_grasp.
        weight_pregrasp: relative weight for pre-grasp frames (post-grasp = 1).
            weight_pregrasp=1.0 is *not* identical to shuffle=True because
            this sampler uses replacement.
        num_samples: per-"epoch" draw count. Defaults to len(weights), so
            step-count per epoch (and any per-step LR schedule) is unchanged.
        replacement: True (correct here — point is to see pre-grasp multiple
            times per epoch). False is provided for the "fully separate the
            weighting effect from the sampling-with-replacement effect"
            A/B control mentioned in the plan.
        seed: feeds an explicit torch.Generator; WeightedRandomSampler has
            no `seed` kwarg of its own.
    """
    if isinstance(phase_labels, PhaseLabelResult):
        labels = phase_labels.labels
    else:
        labels = phase_labels
    labels = np.asarray(labels)
    weights = np.where(labels == LABEL_PREGRASP, float(weight_pregrasp), 1.0)
    weights = weights.astype(np.float64)
    g = torch.Generator()
    g.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(num_samples if num_samples is not None else len(weights)),
        replacement=bool(replacement),
        generator=g,
    )


def concat_phase_labels(parts: list[PhaseLabelResult]) -> PhaseLabelResult:
    """Concat per-source PhaseLabelResults in the same order they were
    passed to ConcatDataset.

    Note: episode_indices may collide across sources (each source numbers
    episodes 0..N-1 independently). The concatenated result keeps the
    original per-source episode_index; alignment asserts must therefore be
    run *per source slice*, not against the concatenated label array, which
    is what `assert_concat_alignment` does.
    """
    if not parts:
        raise ValueError("concat_phase_labels: empty parts")
    labels = np.concatenate([p.labels for p in parts])
    eps = np.concatenate([p.episode_indices for p in parts])
    frames = np.concatenate([p.frame_indices for p in parts])
    n_eps = sum(p.n_episodes for p in parts)
    n_failed = sum(p.n_failed_close for p in parts)
    # Per-source episode index numbering may collide; we don't promise a
    # global grasp_frames dict in the concat path — slice() per source
    # provides the per-source view.
    return PhaseLabelResult(
        labels=labels,
        episode_indices=eps,
        frame_indices=frames,
        n_episodes=n_eps,
        n_failed_close=n_failed,
        grasp_frames={},  # not meaningful at concat scope
    )


# ----------------------------------------------------------- alignment asserts

def _pick_check_indices(n: int, n_check: int, episode_indices: np.ndarray) -> np.ndarray:
    """Pick `n_check` indices spread across episode boundaries (not all in ep 0)."""
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    n_check = min(int(n_check), n)
    # Boundary-anchored: include the first/last index of every episode (so we
    # catch off-by-one alignment bugs that only show up at boundaries) plus
    # a sprinkling of mid-episode indices.
    diffs = np.diff(episode_indices, prepend=episode_indices[0] - 1)
    starts = np.where(diffs != 0)[0]
    ends = np.concatenate([starts[1:] - 1, [n - 1]])
    boundary_idxs = np.unique(np.concatenate([starts, ends]))
    if len(boundary_idxs) >= n_check:
        rs = np.random.default_rng(0)
        return np.sort(rs.choice(boundary_idxs, size=n_check, replace=False))
    extra_needed = n_check - len(boundary_idxs)
    extra = np.linspace(0, n - 1, num=extra_needed + 2, dtype=np.int64)[1:-1]
    return np.unique(np.concatenate([boundary_idxs, extra]))


def assert_dataset_alignment(
    dataset,
    result: PhaseLabelResult,
    n_check: int = 64,
) -> None:
    """Verify the in-memory dataset's episode/frame indices match `result`.

    Calls `dataset[i]` for `n_check` indices (spread across episode
    boundaries) and asserts `episode_index` and `frame_index` in the
    returned sample dict match `result.episode_indices[i]` /
    `result.frame_indices[i]`. Cheap (one __getitem__ each); turns silent
    misalignment into a loud startup error.
    """
    n = len(dataset)
    if len(result.labels) != n:
        raise AssertionError(
            f"phase_labels length {len(result.labels)} != dataset length {n}; "
            "iteration order disagreement."
        )
    idxs = _pick_check_indices(n, n_check, result.episode_indices)
    mismatches: list[str] = []
    for i in idxs:
        sample = dataset[int(i)]
        ds_ep = int(sample["episode_index"]) if "episode_index" in sample else None
        ds_frame = int(sample["frame_index"]) if "frame_index" in sample else None
        if ds_ep is None or ds_frame is None:
            raise AssertionError(
                f"dataset[{i}] missing episode_index/frame_index; "
                f"keys={list(sample.keys())}"
            )
        exp_ep = int(result.episode_indices[i])
        exp_frame = int(result.frame_indices[i])
        if ds_ep != exp_ep or ds_frame != exp_frame:
            mismatches.append(
                f"  idx={i}: dataset=(ep{ds_ep}, frame{ds_frame})  "
                f"labels=(ep{exp_ep}, frame{exp_frame})"
            )
    if mismatches:
        raise AssertionError(
            "phase_labels iteration order != dataset iteration order:\n"
            + "\n".join(mismatches[:10])
        )


def assert_concat_alignment(
    dataset: ConcatDataset,
    parts: list[PhaseLabelResult],
    n_check: int = 32,
) -> None:
    """Per-source alignment assert for ConcatDataset.

    Walks `dataset.datasets[k]` and `parts[k]` together and runs
    `assert_dataset_alignment` on each (k, part) pair. Must be called with
    the parts in the same order as ConcatDataset was constructed.
    """
    if len(dataset.datasets) != len(parts):
        raise AssertionError(
            f"ConcatDataset has {len(dataset.datasets)} parts but {len(parts)} "
            "PhaseLabelResults provided."
        )
    for k, (part_ds, part_lbl) in enumerate(zip(dataset.datasets, parts)):
        try:
            assert_dataset_alignment(part_ds, part_lbl, n_check=n_check)
        except AssertionError as e:
            raise AssertionError(f"ConcatDataset part {k}: {e}") from None
