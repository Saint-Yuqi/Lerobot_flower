"""Capture host/git/runtime metadata for telemetry sidecars.

Used by both `scripts/train.py` (when writing the wandb_metadata.json
sidecar next to a checkpoint) and `scripts/run_inference.py` (when
building the rollout meta.json). Centralised here so the two sides
stay schema-consistent.
"""
from __future__ import annotations

import getpass
import os
import platform
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str, cwd: Path | str | None = None) -> str:
    """Run a git command anchored at the repo root, return stripped stdout.

    The cwd is anchored to the repo root, NOT Path.cwd() (which on SLURM
    may point elsewhere) and NOT to the checkpoint dir (which under the
    HF cache is not a git repo at all).
    """
    cwd = Path(cwd) if cwd is not None else REPO_ROOT
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            check=False,
        )
        return out.stdout.decode("utf-8", "replace").strip()
    except Exception:
        return ""


def git_sha(cwd: Path | str | None = None) -> str:
    return _git("rev-parse", "HEAD", cwd=cwd)


def git_branch(cwd: Path | str | None = None) -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD", cwd=cwd)


def git_dirty(cwd: Path | str | None = None) -> bool:
    cwd = Path(cwd) if cwd is not None else REPO_ROOT
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(cwd),
            capture_output=True,
            check=False,
        )
        return out.stdout.strip() != b""
    except Exception:
        return False


def _try_version(modname: str) -> str | None:
    try:
        import importlib
        m = importlib.import_module(modname)
        v = getattr(m, "__version__", None)
        return str(v) if v is not None else None
    except Exception:
        return None


def cuda_version() -> str | None:
    try:
        import torch
        return getattr(torch.version, "cuda", None)
    except Exception:
        return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def capture_runtime_metadata(cwd: Path | str | None = None) -> dict:
    """Snapshot host/git/library state. Cheap; safe to call once per process."""
    return {
        "code_git_sha": git_sha(cwd),
        "code_git_dirty": git_dirty(cwd),
        "code_branch": git_branch(cwd),
        "hostname": socket.gethostname(),
        "user": getpass.getuser() if hasattr(os, "getuid") else os.environ.get("USER", ""),
        "python_version": platform.python_version(),
        "torch_version": _try_version("torch"),
        "lerobot_version": _try_version("lerobot"),
        "cuda_version": cuda_version(),
        "captured_at": utc_now_iso(),
    }
