"""Build a native-uint8 frame cache for a FlowerSO101 dataset.

Decodes every frame once into a memmap keyed by global dataset index, so
training reads frames from the OS page cache instead of PyAV-decoding video
on every step. The cached bytes are exactly what `_decode_frame_at` returns;
bit-equality is asserted on a random sample before the sidecar is written
(the sidecar's presence is what marks the cache valid — a crashed/partial
build is never picked up).

  # single-process (correctness reference / fallback)
  python scripts/build_frame_cache.py ethrl2026/task1_all --revision main \\
      --cache-dir ~/.cache/flower_frames

  # parallel: N workers each decode a disjoint shard of global indices and
  # write their own non-overlapping memmap slices (disjoint ranges => no
  # race, no locking). ~Nx faster, reusable for any dataset.
  python scripts/build_frame_cache.py ethrl2026/task1_20260509_plus \\
      --revision main --cache-dir ~/.cache/flower_frames --workers 64
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _build_shard(args: tuple) -> tuple[int, int, int]:
    """Worker: decode flat-idx range [lo, hi) and write own memmap slices.

    Disjoint global-index ranges across workers => safe concurrent writes to
    one file with no locking. Each worker owns its own dataset + PyAV state.
    """
    repo_id, root, revision, mm_path, shape, lo, hi = args
    import numpy as np

    from src.flower.dataset import FlowerSO101Dataset

    ds = FlowerSO101Dataset(
        repo_id=repo_id, root=root, revision=revision, episodes=None,
    )
    mm = np.memmap(mm_path, dtype=np.uint8, mode="r+", shape=tuple(shape))
    for idx in range(lo, hi):
        ep, gidx, ts = ds._frame_addr(idx)
        mm[gidx] = ds._decode_frame_at(ep, ts)
    mm.flush()
    del mm
    return (lo, hi, hi - lo)


def _build_parallel(repo_id, root, revision, cache_dir, workers) -> None:
    import multiprocessing as mp

    import numpy as np

    from src.flower.dataset import FlowerSO101Dataset

    ds = FlowerSO101Dataset(
        repo_id=repo_id, root=root, revision=revision, episodes=None,
        frame_cache_dir=cache_dir,
    )
    if ds._frame_cache is not None:
        print("[build] cache already exists and is valid — nothing to do.")
        return

    mm_path, meta_path = ds._frame_cache_paths()
    mm_path.parent.mkdir(parents=True, exist_ok=True)
    total = ds._total_frames
    n_frames = int(max(int(e.dataset_to_index) for e in ds._episodes))
    ep0, _g0, ts0 = ds._frame_addr(0)
    probe = ds._decode_frame_at(ep0, ts0)
    h, w, c = int(probe.shape[0]), int(probe.shape[1]), int(probe.shape[2])
    shape = (n_frames, h, w, c)

    # Pre-allocate the file so workers can open it mode="r+".
    mm = np.memmap(mm_path, dtype=np.uint8, mode="w+", shape=shape)
    mm.flush()
    del mm

    # Contiguous flat-idx shards (each maps to a disjoint set of gidx).
    bounds = [round(i * total / workers) for i in range(workers + 1)]
    jobs = [
        (repo_id, ds.root and str(ds.root), revision, str(mm_path), shape,
         bounds[i], bounds[i + 1])
        for i in range(workers) if bounds[i + 1] > bounds[i]
    ]
    print(f"[build] {repo_id}@{revision} frames={total} -> {len(jobs)} shards "
          f"x ~{total // max(1, len(jobs))} frames  ({h}x{w}x{c})", flush=True)

    t0 = time.time()
    ctx = mp.get_context("spawn")
    done = 0
    with ctx.Pool(processes=len(jobs)) as pool:
        for (lo, hi, n) in pool.imap_unordered(_build_shard, jobs):
            done += n
            print(f"[build] shard [{lo},{hi}) done — {done}/{total} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    # Bit-equality verification (single process), then sidecar.
    mm = np.memmap(mm_path, dtype=np.uint8, mode="r", shape=shape)
    rng = np.random.default_rng(0)
    sample = rng.choice(total, size=min(64, total), replace=False)
    for idx in sample:
        ep, gidx, ts = ds._frame_addr(int(idx))
        fresh = ds._decode_frame_at(ep, ts)
        if not np.array_equal(fresh, np.asarray(mm[gidx])):
            raise RuntimeError(
                f"VERIFY FAIL idx={idx} gidx={gidx} — cache != fresh decode"
            )
    meta = {"repo_id": repo_id, "revision": revision,
            "n_frames": n_frames, "height": h, "width": w, "channels": c}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[build] built {n_frames} frames {h}x{w}x{c} -> {mm_path} "
          f"(verified {len(sample)} samples bit-identical, "
          f"{(time.time() - t0) / 60:.1f} min)", flush=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("repo_id")
    p.add_argument("--revision", default="main")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--root", default=None,
                   help="Local dataset root; default = HF cache snapshot.")
    p.add_argument("--workers", type=int, default=1,
                   help="Parallel decode workers (1 = single-process).")
    args = p.parse_args()

    if args.workers > 1:
        t0 = time.time()
        _build_parallel(args.repo_id, args.root, args.revision,
                        args.cache_dir, args.workers)
        print(f"[build] done in {(time.time() - t0) / 60:.1f} min", flush=True)
        return 0

    from src.flower.dataset import FlowerSO101Dataset

    ds = FlowerSO101Dataset(
        repo_id=args.repo_id, root=args.root, revision=args.revision,
        episodes=None, frame_cache_dir=args.cache_dir,
    )
    print(f"[build] {args.repo_id}@{args.revision}  frames={len(ds)}  "
          f"root={ds.root}", flush=True)
    if ds._frame_cache is not None:
        print("[build] cache already exists and is valid — nothing to do.")
        return 0
    t0 = time.time()
    ds.build_frame_cache(progress_every=2000, verify_n=32)
    print(f"[build] done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
