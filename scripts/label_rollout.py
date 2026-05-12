"""Post-hoc relabel of a rollout's verdict / notes / tags.

Most rollouts run unattended on SLURM and finish with verdict='unset'.
This CLI is the realistic operator-labeling loop:

    python scripts/label_rollout.py <inference_run_id> \\
        --verdict success --notes "banana on first try" \\
        [--tags wrong_bowl,recovered]

It updates `outcome.json` in place (appending to `label_history`), and
when the rollout's wandb run is online, also pushes the new verdict
to the rollout's wandb summary AND to the matching entry inside the
training run's `summary.rollouts[]`.

`--push-only` skips the local `outcome.json` mutation (useful after
`wandb sync` to push verdicts that were captured offline).
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _resolve_run_dir(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir() and (p / "outcome.json").exists():
        return p
    candidate = REPO_ROOT / "logs" / "inference" / arg
    if candidate.is_dir() and (candidate / "outcome.json").exists():
        return candidate
    raise SystemExit(
        f"[label] cannot find a rollout dir for '{arg}'. "
        f"Tried '{p}' and '{candidate}'."
    )


def _parse_tags(s: str) -> list[str]:
    return [t.strip() for t in s.split(",") if t.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="inference_run_id or path to logs/inference/<id>")
    parser.add_argument(
        "--verdict",
        choices=["success", "partial", "failure", "abort", "unset"],
        required=True,
    )
    parser.add_argument("--notes", default="")
    parser.add_argument(
        "--tags",
        type=_parse_tags,
        default=[],
        help="comma-separated tags (e.g. dropped_object,recovered)",
    )
    parser.add_argument("--by", default="operator")
    parser.add_argument(
        "--push-only", action="store_true",
        help="do not modify outcome.json; only push to wandb",
    )
    args = parser.parse_args()

    run_dir = _resolve_run_dir(args.run_id)
    outcome_path = run_dir / "outcome.json"
    meta_path = run_dir / "meta.json"

    # 1. Local outcome.json update.
    if not args.push_only:
        try:
            outcome = json.loads(outcome_path.read_text())
        except Exception as e:
            raise SystemExit(f"[label] cannot read {outcome_path}: {e!r}")
        outcome["verdict"] = args.verdict
        if args.notes:
            outcome["notes"] = args.notes
        if args.tags:
            existing = list(outcome.get("tags") or [])
            for t in args.tags:
                if t not in existing:
                    existing.append(t)
            outcome["tags"] = existing
        history = list(outcome.get("label_history") or [])
        history.append({
            "ts_wall_iso": _utc_iso(),
            "by": str(args.by),
            "verdict": args.verdict,
            "notes": args.notes or "",
        })
        outcome["label_history"] = history
        outcome_path.write_text(json.dumps(outcome, indent=2))
        print(f"[label] outcome.json updated -> {outcome_path}")

    # 2. Read meta.json to recover wandb pointers.
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        print(f"[label] cannot read {meta_path}: {e!r}; skipping wandb push")
        return

    cli_args = meta.get("cli_args") or {}
    no_wandb = bool(cli_args.get("no_wandb"))
    wandb_mode = cli_args.get("wandb_mode", "online")
    if no_wandb or wandb_mode != "online":
        print("[label] wandb skipped (offline)")
        return

    # 3. Wandb push.
    rollout_id = meta.get("inference_run_id") or args.run_id
    rollout_project = cli_args.get("wandb_project", "Lerobot-rollouts")
    training = meta.get("training_run") or {}
    training_entity = training.get("wandb_entity")
    training_project = training.get("wandb_project")
    training_rid = training.get("wandb_run_id")

    try:
        import wandb
        api = wandb.Api()
    except Exception as e:
        print(f"[label] wandb import/init failed: {e!r}")
        return

    # 3a. Update rollout run summary.
    rollout_entity = training_entity  # rollouts inherit entity from training
    if rollout_entity is None:
        # Fall back to the wandb default entity for this user.
        try:
            rollout_entity = wandb.api.default_entity  # type: ignore[attr-defined]
        except Exception:
            rollout_entity = None
    if rollout_entity:
        try:
            rr = api.run(f"{rollout_entity}/{rollout_project}/{rollout_id}")
            rr.summary["verdict"] = args.verdict
            if args.notes:
                rr.summary["notes"] = args.notes
            rr.summary.update({})  # commit
            print(f"[label] rollout wandb summary updated -> {rr.url}")
        except Exception as e:
            print(f"[label] rollout summary push skipped: {e!r}")

    # 3b. Update entry in training run's summary.rollouts[].
    if training_entity and training_project and training_rid:
        try:
            tr = api.run(f"{training_entity}/{training_project}/{training_rid}")
            tr_rollouts = list(tr.summary.get("rollouts") or [])
            updated = False
            for r in tr_rollouts:
                if r.get("id") == rollout_id:
                    r["verdict"] = args.verdict
                    if args.notes:
                        r["notes"] = args.notes
                    updated = True
            if updated:
                tr.summary["rollouts"] = tr_rollouts
                tr.summary.update({})
                print(f"[label] training-run rollouts[] updated -> {tr.url}")
            else:
                print(f"[label] no entry for {rollout_id} in training run")
        except Exception as e:
            print(f"[label] training run push skipped: {e!r}")
    else:
        print("[label] training-run pointers missing (legacy no-sidecar checkpoint)")


if __name__ == "__main__":
    main()
