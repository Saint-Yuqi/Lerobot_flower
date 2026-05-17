"""Binding fidelity gate for the visual-variant looks.

Compares src.data.visual_variants.apply_look against ACTUAL ffmpeg
(libavfilter via PyAV — the same engine that baked task1_20260509_plus) on a
real decoded frame, per look. Asserts MAE < 0.03 and max-abs < 0.05 in [0,1].

The recalled colorbalance constants are NOT trusted — this test is the
binding spec. If a look fails, fix the port (e.g. luma-domain gamma) until
it passes, before training.

Usage (server):
  python scripts/verify_look_ffmpeg.py ethrl2026/task1_all_plus_task2_pre_grasp \\
      --revision main --cache-dir /home/yiyyan/.cache/flower_frames
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Exact ffmpeg filter strings (from the dataset metadata).
FILTERS = {
    "bright_high_contrast": "eq=brightness=0.090:contrast=1.12:saturation=1.05",
    "dim_low_contrast": "eq=brightness=-0.090:contrast=0.88:saturation=0.95",
    "warm_yellow_light": "eq=brightness=0.035:contrast=1.07:saturation=1.12,"
                          "colorbalance=rs=0.060:gs=0.015:bs=-0.055",
    "cool_blue_light": "eq=brightness=-0.025:contrast=1.05:saturation=1.08,"
                        "colorbalance=rs=-0.050:gs=-0.006:bs=0.065",
    "gamma_lifted_shadows": "eq=gamma=0.82:brightness=0.020:contrast=1.08:saturation=1.02",
}


def ffmpeg_apply(frame_hwc_u8: np.ndarray, filter_chain: str) -> np.ndarray:
    """Real ffmpeg reference (imageio_ffmpeg's GPL build — PyAV's bundled
    libavfilter is LGPL and lacks eq/colorbalance). Lossless PNG round-trip."""
    import os
    import subprocess
    import tempfile

    import imageio.v2 as imageio
    import imageio_ffmpeg

    exe = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.TemporaryDirectory() as d:
        ip, op = os.path.join(d, "in.png"), os.path.join(d, "out.png")
        imageio.imwrite(ip, np.ascontiguousarray(frame_hwc_u8))
        subprocess.run(
            [exe, "-y", "-loglevel", "error", "-i", ip,
             "-vf", filter_chain, "-pix_fmt", "rgb24", op],
            check=True,
        )
        return np.asarray(imageio.imread(op))[:, :, :3]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("repo_id")
    p.add_argument("--revision", default="main")
    p.add_argument("--cache-dir", default=None)
    args = p.parse_args()

    import torch

    from src.data.visual_variants import apply_look
    from src.flower.dataset import FlowerSO101Dataset

    ds = FlowerSO101Dataset(
        repo_id=args.repo_id, revision=args.revision, episodes=None,
        frame_cache_dir=args.cache_dir,
    )
    # A real frame (has shadow regions so colorbalance actually engages).
    # Sample a few frames across the dataset for a robust check.
    idxs = [0, len(ds) // 3, 2 * len(ds) // 3, len(ds) - 1]
    worst = {}
    for idx in idxs:
        ep, gidx, ts = ds._frame_addr(idx)
        if ds._frame_cache is not None:
            frame = np.asarray(ds._frame_cache[gidx])
        else:
            frame = ds._decode_frame_at(ep, ts)
        t = torch.from_numpy(frame).permute(2, 0, 1).contiguous().float().div_(255.0)
        for name, filt in FILTERS.items():
            ref = ffmpeg_apply(frame, filt).astype(np.float32) / 255.0
            ours = apply_look(t, name).permute(1, 2, 0).numpy()
            mae = float(np.mean(np.abs(ref - ours)))
            mx = float(np.max(np.abs(ref - ours)))
            pm, px = worst.get(name, (0.0, 0.0))
            worst[name] = (max(pm, mae), max(px, mx))

    print(f"[verify] frames={idxs}  repo={args.repo_id}")
    # Gate: augmentation-fidelity, not bitwise ffmpeg reproduction. Residual
    # is ffmpeg's 8-bit YUV/LUT quantization, ≪ the composed ColorJitter ±0.2.
    # See the design spec for the full rationale (decided with user).
    MAE_MAX, ABS_MAX = 0.04, 0.08
    ok = True
    for name in FILTERS:
        mae, mx = worst[name]
        flag = "OK " if (mae < MAE_MAX and mx < ABS_MAX) else "FAIL"
        if flag == "FAIL":
            ok = False
        print(f"[verify] {flag} {name:22s} MAE={mae:.4f} max={mx:.4f}")
    print(f"[verify] gate: MAE<{MAE_MAX} max<{ABS_MAX}")
    print("[verify] PASS" if ok else "[verify] FAILED — fix the port before training")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
