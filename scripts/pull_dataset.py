"""Pre-download a LeRobot v3.0 dataset into the HF cache.

Run this before training so the job doesn't pay the download cost and so 4
DDP ranks don't hammer the HF Hub concurrently (429s). Uses the same
allow_patterns as FlowerSO101Dataset, so a subsequent `root: null` run
finds the cached snapshot instantly.

Usage:
    python scripts/pull_dataset.py ethrl2026/task1_all
    python scripts/pull_dataset.py ethrl2026/task1_all --revision main
"""
from __future__ import annotations

import argparse

from huggingface_hub import snapshot_download


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("repo_id")
    p.add_argument("--revision", default="main")
    p.add_argument("--local-dir", default=None,
                   help="Optional explicit dir; default = HF cache.")
    args = p.parse_args()

    path = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        allow_patterns=["meta/*", "data/**", "videos/**"],
        local_dir=args.local_dir,
    )
    print(f"[pull] {args.repo_id}@{args.revision} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
