"""Checkpoint <-> training-run linkage helpers.

Two halves:
- `write_checkpoint_meta` is called by `scripts/train.py` after every
  `policy.save_pretrained(...)` to leave a `wandb_metadata.json` next to
  the weights. This file is what survives the `api.upload_folder` ->
  `snapshot_download` round-trip and lets the inference side recover the
  training run id/url even when the checkpoint came back via the HF Hub.
- `resolve_training_run` is called by `scripts/run_inference.py` to read
  that sidecar back. Returns `None` (never raises) when the sidecar is
  missing — every checkpoint uploaded before this code landed is in that
  state.

`checkpoint_short_id` produces a filesystem-safe, length-bounded
identifier that's reused both for the rollout's `inference_run_id` and
as a wandb tag.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.utils.run_metadata import utc_now_iso


def checkpoint_short_id(repo_id_or_dirname: str, max_len: int = 24) -> str:
    """Sanitize a HF repo id or local dir name into a wandb-tag-safe slug.

    HF repo ids contain `/`; local dirnames may contain dates with `:` or
    similar. Collapse anything outside `[A-Za-z0-9-]` to `-` and bound
    the length so the result stays under wandb's per-tag limit.
    """
    safe = re.sub(r"[^A-Za-z0-9]", "-", repo_id_or_dirname)
    safe = re.sub(r"-+", "-", safe).strip("-")
    return safe[:max_len] or "ckpt"


def write_checkpoint_meta(
    ckpt_dir: Path | str,
    wandb_run,
    cfg: dict,
    step: int,
    git_sha: str,
    extra: dict | None = None,
) -> Path:
    """Write `wandb_metadata.json` into `ckpt_dir`.

    `wandb_run` may be None (wandb disabled): we still write the file with
    null fields so a downstream reader always finds a sidecar with a
    consistent schema. `cfg` is the loaded training YAML; we pull
    `experiment_name` and the logging block from it.

    `extra` is for fields that are useful to remember in `best/` but not
    in periodic saves — typically `{"val_loss": float, "is_best": True}`.
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_cfg = cfg.get("logging") or {}

    payload = {
        "wandb_run_id": getattr(wandb_run, "id", None),
        "wandb_project": getattr(wandb_run, "project", None) or log_cfg.get("project"),
        "wandb_entity": getattr(wandb_run, "entity", None) or log_cfg.get("entity"),
        "wandb_url": getattr(wandb_run, "url", None),
        "wandb_name": getattr(wandb_run, "name", None),
        "experiment_name": cfg.get("experiment_name"),
        "git_sha": git_sha,
        "saved_at": utc_now_iso(),
        "step": int(step),
    }
    if extra:
        payload.update(extra)

    out = ckpt_dir / "wandb_metadata.json"
    out.write_text(json.dumps(payload, indent=2))
    return out


def resolve_training_run(ckpt_path: Path | str) -> dict | None:
    """Read `wandb_metadata.json` from a checkpoint dir.

    Contract: returns the full sidecar dict on success, `None` when the
    file is missing or unparseable. NEVER raises — legacy HF checkpoints
    uploaded before the sidecar landed must still produce a working
    rollout, just without the back-link.
    """
    p = Path(ckpt_path)
    candidate = p / "wandb_metadata.json"
    if not candidate.exists():
        return None
    try:
        data = json.loads(candidate.read_text())
        if not isinstance(data, dict):
            return None
        return data
    except (OSError, json.JSONDecodeError):
        return None
