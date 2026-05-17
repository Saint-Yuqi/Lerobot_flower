"""Standalone no-train eval: a checkpoint's mean val loss on a dataset.

Replicates train_flower_accelerate.py's val construction EXACTLY (same
episodes_by_color + train_val_episode_split + seed + clean val + frame
cache) so the number is directly comparable to that run's `eval/loss`
curve. Single process / single GPU (a pure no-grad pass needs no DDP).

Use to get the true step-0 baseline (raw checkpoint, zero gradient steps),
which the training loop never logs (it only evals at step>0).

  python scripts/eval_checkpoint.py \\
    --init-from ethrl2026/so101-eval1-flower-v1-aug-mgpu \\
    --repo-id ethrl2026/so101_pickup_20260515_200119_task1_extra2 \\
    --revision main --cache-dir /home/yiyyan/.cache/flower_frames
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--init-from", required=True, help="HF model id or local dir")
    p.add_argument("--repo-id", required=True)
    p.add_argument("--revision", default="main")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--min-train-per-color", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3,
                   help="Full val passes (rf_loss is stochastic — average).")
    args = p.parse_args()

    import multiprocessing as _mp

    import torch
    from huggingface_hub import snapshot_download
    from torch.utils.data import DataLoader

    from src.data.splits import episodes_by_color, train_val_episode_split
    from src.flower.dataset import FlowerSO101Dataset
    from src.flower.policy import FlowerVLAPolicy

    ckpt = (args.init_from if os.path.isdir(args.init_from)
            else snapshot_download(repo_id=args.init_from, repo_type="model"))
    policy = FlowerVLAPolicy.from_pretrained(ckpt, device="cuda")
    policy.eval()
    cs = int(policy.config.chunk_size)
    hw = int(policy.config.image_hw)

    ds_probe = FlowerSO101Dataset(
        repo_id=args.repo_id, revision=args.revision, chunk_size=cs,
        video_key="observation.images.main", resize_hw=hw,
    )
    by = episodes_by_color(args.repo_id, root=ds_probe.root)
    train_eps, val_eps = train_val_episode_split(
        by, fraction=args.val_fraction,
        min_train_per_color=args.min_train_per_color, seed=args.seed,
    )
    val_ds = FlowerSO101Dataset(
        repo_id=args.repo_id, root=ds_probe.root, episodes=val_eps,
        revision=args.revision, chunk_size=cs,
        video_key="observation.images.main", resize_hw=hw,
        frame_cache_dir=args.cache_dir,
    )
    nw = args.num_workers
    mp_ctx = _mp.get_context("spawn") if nw > 0 else None
    loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=nw,
        drop_last=False, pin_memory=True, persistent_workers=nw > 0,
        multiprocessing_context=mp_ctx,
    )

    pass_means = []
    with torch.no_grad():
        for r in range(args.repeats):
            losses = []
            for b in loader:
                loss, _ = policy(b)
                losses.append(float(loss.detach().cpu()))
            pass_means.append(sum(losses) / max(1, len(losses)))
            print(f"[eval] pass {r + 1}/{args.repeats} mean={pass_means[-1]:.5f}",
                  flush=True)

    mean = sum(pass_means) / len(pass_means)
    print(f"[eval] {args.init_from}  on  {args.repo_id} val")
    print(f"[eval] val episodes={len(val_eps)} frames={len(val_ds)}  "
          f"per-pass means={[round(m, 5) for m in pass_means]}")
    print(f"[eval] RAW-CHECKPOINT mean val loss = {mean:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
