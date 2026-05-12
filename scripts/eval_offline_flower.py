"""Offline FlowerVLA evaluation — flower-env equivalent of `scripts/eval_offline.py`.

Loads a FlowerVLA checkpoint and a dataset, samples K frames per episode, and
reports two cheap-but-meaningful gates before risking the real SO-101:

1. **Test A — Open-loop replay**: for each sampled frame, predict the action
   chunk and compute per-joint MAE against the ground-truth chunk. Records
   median + mean MAE per joint across the sample.

2. **Test C — OOD prompts**: feed a fixed set of nonsensical prompts and check
   for NaN/Inf or actions that walk outside the training joint range. Catches
   catastrophic blow-ups before they reach hardware.

Test B (prompt-equivalence consistency, the SmolVLA acid test) is intentionally
omitted here — it relies on prompt_aug's color-prompt pool which only applies
to task1/task2 datasets, and adds significant complexity. Add when needed.

Usage:
    /shares/feldmann.ics.mnf.uzh/Yuqi/conda/envs/flower/bin/python \\
        scripts/eval_offline_flower.py \\
        --checkpoint ethrl2026/so101-eval3-flower-v1 \\
        --dataset ethrl2026/so101_pickup_20260509_185350_task3 \\
        --frames-per-episode 4 \\
        --out reports/eval3_flower_offline.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


# Test C — OOD prompts. Picked to stress the tokenizer + policy head without
# being solvable. Mirrors the SmolVLA offline harness.
OOD_PROMPTS = [
    "Put the banana in the purple colored bowl.",
    "What is the meaning of life?",
    "",
    "asdf qwerty zxcv banana bowl table",
]


def _resolve_checkpoint(arg: str) -> str:
    """Accept a local path or HF Hub repo id; return a usable local dir."""
    p = Path(arg)
    if p.is_dir() and (p / "config.json").exists():
        return str(p)
    if p.exists():
        raise SystemExit(
            f"--checkpoint {arg!r} exists but has no config.json (not a flower ckpt)."
        )
    print(f"[eval] pulling checkpoint {arg!r} from HuggingFace Hub...")
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=arg, repo_type="model")


def _sample_global_indices(ds, frames_per_episode: int, seed: int) -> list[int]:
    """Pick K frames per episode uniformly. Returns sorted global indices."""
    rng = np.random.default_rng(seed)
    out: list[int] = []
    for ep in ds._episodes:
        ep_len = ep.dataset_to_index - ep.dataset_from_index
        if ep_len <= 0:
            continue
        # Avoid sampling the very last frames where the action chunk gets
        # heavily padded — keeps the MAE metric honest.
        usable = max(1, ep_len - ds.chunk_size)
        n = min(int(frames_per_episode), usable)
        picks = rng.choice(usable, size=n, replace=False)
        out.extend(int(ep.dataset_from_index + p) for p in picks)
    out.sort()
    return out


def _global_to_filtered_idx(ds, global_idx: int) -> int:
    """Translate a dataset-global index to the position inside the (episode-
    filtered) dataset. With our default ds (no episode filter), this is the
    identity, but we keep the helper so that future filtered eval datasets work."""
    for i, ep in enumerate(ds._episodes):
        if ep.dataset_from_index <= global_idx < ep.dataset_to_index:
            local = global_idx - ep.dataset_from_index
            return int(ds._cum_starts[i] + local)
    raise IndexError(global_idx)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True, help="HF Hub repo id or local root.")
    parser.add_argument("--root", default=None, help="Optional local dataset root override.")
    parser.add_argument("--frames-per-episode", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=None, help="Optional JSON output path.")
    parser.add_argument("--max-frames", type=int, default=400,
                        help="Hard cap on frames to evaluate; full sweep on big sets is slow.")
    args = parser.parse_args()

    import torch

    from src.flower.dataset import FlowerSO101Dataset
    from src.flower.policy import FlowerVLAPolicy

    # ---- Checkpoint ----
    ckpt_path = _resolve_checkpoint(args.checkpoint)
    policy = FlowerVLAPolicy.from_pretrained(ckpt_path, device=args.device)
    policy.eval()
    chunk_size = int(policy.config.chunk_size)
    print(f"[eval] policy: chunk_size={chunk_size} action_dim={policy.config.action_dim}")

    # ---- Dataset ----
    ds = FlowerSO101Dataset(
        repo_id=args.dataset, root=args.root,
        chunk_size=chunk_size,
        video_key=policy.config.video_key,
        resize_hw=int(policy.config.image_hw),
    )
    print(f"[eval] dataset: {ds}")

    # ---- Sample frames ----
    idxs = _sample_global_indices(ds, args.frames_per_episode, args.seed)
    if len(idxs) > args.max_frames:
        rng = np.random.default_rng(args.seed)
        idxs = sorted(int(i) for i in rng.choice(idxs, size=int(args.max_frames), replace=False))
    print(f"[eval] sampling {len(idxs)} frames from {len(ds._episodes)} episodes")

    # ---- Phase labels (for per-phase MAE breakdown) ----
    # Source of truth: src/data/phase_labels.py. Cached on disk; reads only
    # the action column so even on big datasets this is fast.
    from src.data.phase_labels import compute_phase_labels, summarize, LABEL_PREGRASP
    phase = compute_phase_labels(
        repo_id=args.dataset, root=ds.root,
        episodes=[e.episode_index for e in ds._episodes],
    )
    phase_summary = summarize(phase, label=args.dataset)
    print(f"[eval] phase labels: {phase_summary}")

    # ---- Test A: open-loop replay ----
    per_joint_abs: list[np.ndarray] = []  # per-frame per-joint MAE
    chunk_mse: list[float] = []
    per_frame_phase: list[int] = []        # 0=pre_grasp, 1=post_grasp per kept frame
    failed = 0
    t0 = time.time()
    for k, global_idx in enumerate(idxs):
        try:
            filtered_idx = _global_to_filtered_idx(ds, global_idx)
            sample = ds[filtered_idx]
        except Exception as e:
            failed += 1
            print(f"[eval] WARN sample fetch failed at idx={global_idx}: {e!r}")
            continue
        obs = {
            policy.config.video_key: sample[policy.config.video_key],
            "observation.state": sample["observation.state"],
            "task": sample["task"],
        }
        policy.reset()
        with torch.no_grad():
            pred = policy.sample_chunk(obs)  # (1, T, A)
        if pred.dim() == 3:
            pred = pred[0]
        gt = sample["action"]  # (T, A)
        is_pad = sample["action_is_pad"]  # (T,)
        valid_mask = ~is_pad
        # Only evaluate on non-padded steps so MAE is honest.
        if valid_mask.sum() == 0:
            continue
        diff = (pred - gt).abs().numpy()[valid_mask.numpy()]  # (V, A)
        per_joint_abs.append(diff.mean(axis=0))               # (A,)
        chunk_mse.append(float(((pred - gt) ** 2)[valid_mask].mean().item()))
        per_frame_phase.append(int(phase.labels[filtered_idx]))

        if (k + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (k + 1) / max(elapsed, 1e-6)
            eta = (len(idxs) - (k + 1)) / max(rate, 1e-6)
            print(f"[eval] testA: {k+1}/{len(idxs)} frames  "
                  f"{rate:.2f} f/s  ETA={eta:.1f}s")

    per_joint = np.stack(per_joint_abs) if per_joint_abs else np.zeros((0, ds.action_dim))
    feat_names = ds.info["features"]["action"].get("names")
    joint_names = feat_names if isinstance(feat_names, list) else [f"joint_{i}" for i in range(ds.action_dim)]
    testA = {
        "n_frames_used": int(per_joint.shape[0]),
        "n_frames_failed": int(failed),
        "per_joint_mae_mean": dict(zip(joint_names, per_joint.mean(axis=0).round(4).tolist()))
            if per_joint.shape[0] else {},
        "per_joint_mae_median": dict(zip(joint_names, np.median(per_joint, axis=0).round(4).tolist()))
            if per_joint.shape[0] else {},
        "overall_mae_mean": float(per_joint.mean()) if per_joint.shape[0] else None,
        "chunk_mse_mean": float(np.mean(chunk_mse)) if chunk_mse else None,
    }
    # Per-phase breakdown — the A/B headline metric. Anchors are bucketed by
    # the phase label of the *anchor* frame (consistent with the training-side
    # weighting). Per-joint slices help pinpoint where pre-grasp error
    # concentrates (likely shoulder/elbow during approach, gripper at closure).
    phase_arr = np.asarray(per_frame_phase, dtype=np.int8)
    if per_joint.shape[0] and len(phase_arr) == per_joint.shape[0]:
        mask_pre = phase_arr == LABEL_PREGRASP
        mask_post = ~mask_pre
        def _safe_mean(a):
            return float(a.mean()) if a.size else None
        def _safe_pj(a):
            return dict(zip(joint_names, a.mean(axis=0).round(4).tolist())) if a.size else {}
        testA["phase_label_summary"] = phase_summary
        testA["n_frames_pregrasp"] = int(mask_pre.sum())
        testA["n_frames_postgrasp"] = int(mask_post.sum())
        testA["mae_pregrasp_anchor"] = _safe_mean(per_joint[mask_pre])
        testA["mae_postgrasp_anchor"] = _safe_mean(per_joint[mask_post])
        testA["per_joint_mae_pregrasp"] = _safe_pj(per_joint[mask_pre])
        testA["per_joint_mae_postgrasp"] = _safe_pj(per_joint[mask_post])
    print(f"[eval] Test A: mean overall MAE = {testA['overall_mae_mean']}")
    print(f"[eval] Test A: per-joint mean MAE = {testA['per_joint_mae_mean']}")
    if "mae_pregrasp_anchor" in testA:
        print(f"[eval] Test A: mae_pregrasp_anchor={testA['mae_pregrasp_anchor']}  "
              f"mae_postgrasp_anchor={testA['mae_postgrasp_anchor']}  "
              f"(n_pre={testA['n_frames_pregrasp']} n_post={testA['n_frames_postgrasp']})")

    # ---- Test C: OOD prompts ----
    # Take the first valid frame as the observation and vary the prompt.
    if not idxs:
        print("[eval] WARN no idxs sampled; skipping Test C")
        testC = {"prompts": [], "any_nonfinite": False, "any_oor": False}
    else:
        ref_sample = ds[_global_to_filtered_idx(ds, idxs[0])]
        action_min = torch.tensor(ds.stats["action"]["min"], dtype=torch.float32)
        action_max = torch.tensor(ds.stats["action"]["max"], dtype=torch.float32)
        # Allow a small margin outside training min/max; if actions blow up far past
        # this, they would damage the arm.
        margin = (action_max - action_min) * 0.5
        lower = action_min - margin
        upper = action_max + margin

        prompt_results: list[dict] = []
        any_nonfinite = False
        any_oor = False
        for prompt in OOD_PROMPTS:
            obs = {
                policy.config.video_key: ref_sample[policy.config.video_key],
                "observation.state": ref_sample["observation.state"],
                "task": prompt,
            }
            policy.reset()
            with torch.no_grad():
                pred = policy.sample_chunk(obs)
            if pred.dim() == 3:
                pred = pred[0]
            finite = bool(torch.isfinite(pred).all().item())
            oor = bool(((pred < lower) | (pred > upper)).any().item())
            any_nonfinite |= not finite
            any_oor |= oor
            prompt_results.append({
                "prompt": prompt,
                "finite": finite,
                "out_of_range": oor,
                "min_per_joint": pred.min(dim=0).values.tolist(),
                "max_per_joint": pred.max(dim=0).values.tolist(),
            })
        testC = {
            "prompts": prompt_results,
            "any_nonfinite": any_nonfinite,
            "any_oor": any_oor,
        }
        print(f"[eval] Test C: any_nonfinite={any_nonfinite}  any_oor={any_oor}")

    report = {
        "checkpoint": args.checkpoint,
        "checkpoint_path": str(ckpt_path),
        "dataset": args.dataset,
        "frames_per_episode": int(args.frames_per_episode),
        "chunk_size": chunk_size,
        "test_a_open_loop_replay": testA,
        "test_c_ood_prompts": testC,
        "wall_seconds": round(time.time() - t0, 2),
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"[eval] wrote {args.out}")
    else:
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
