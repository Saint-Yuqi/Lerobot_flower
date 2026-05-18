"""Run a trained FlowerVLA policy on the real SO-101 robot.

Flower-env equivalent of `scripts/run_inference.py` (the SmolVLA inference script).
Reuses the same telemetry stack (`src.inference.rollout_logger`, etc.) so
post-hoc analysis works the same way.

Lazy-imports the live robot driver from the vendored
`third_party/lerobot_hw/robots/so_follower/`, so this script works in dry-run
mode on any machine.

Usage:
    # HF Hub checkpoint, dry-run (no hardware), no wandb:
    /shares/feldmann.ics.mnf.uzh/Yuqi/conda/envs/flower/bin/python \\
        scripts/run_inference_flower.py \\
        --checkpoint ethrl2026/so101-eval3-flower-v1 \\
        --prompt "Y+L" --max-seconds 20 --dry-run --no-wandb

    # Live arm on the lab machine:
    /shares/feldmann.ics.mnf.uzh/Yuqi/conda/envs/flower/bin/python \\
        scripts/run_inference_flower.py \\
        --checkpoint ethrl2026/so101-eval3-flower-v1 \\
        --prompt "Y+L" --max-seconds 20 \\
        --robot-port /dev/tty.usbmodem5B141136551 --robot-id follower_111
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.flower.policy import FlowerVLAPolicy
from src.flower.runner import DryRunRobot, make_live_robot, run_rollout, JOINT_KEYS
from src.inference.rollout_logger import NoopLogger, RolloutLogger
from src.utils.checkpoint_meta import checkpoint_short_id, resolve_training_run
from src.utils.run_metadata import capture_runtime_metadata


def _resolve_checkpoint(arg: str) -> tuple[str, str, str]:
    p = Path(arg)
    if p.is_dir() and (p / "config.json").exists():
        return str(p), "local", str(p)
    if p.exists():
        raise SystemExit(
            f"--checkpoint {arg!r} exists but is not a flower ckpt (no config.json)."
        )
    if "/" not in arg or arg.count("/") > 1:
        raise SystemExit(
            f"--checkpoint {arg!r} is neither a local dir nor a HF repo id."
        )
    print(f"[infer] pulling checkpoint {arg!r} from HuggingFace Hub...")
    from huggingface_hub import snapshot_download
    local = snapshot_download(repo_id=arg, repo_type="model")
    return local, "hf", arg


def _sanitize_tag(prefix: str, body: str, max_body: int = 32) -> str:
    safe = re.sub(r"[^A-Za-z0-9_\-]", "_", body)[:max_body]
    return f"{prefix}_{safe}"


def _build_logger(args, meta: dict, action_dim: int, chunk_size: int):
    if args.no_log:
        return NoopLogger()
    return RolloutLogger(
        log_dir=args.log_dir,
        inference_run_id=meta["inference_run_id"],
        meta=meta,
        action_dim=int(action_dim),
        state_dim=int(action_dim),
        control_hz=float(args.control_hz),
        chunk_size=int(chunk_size),
        frame_every=int(args.frame_every),
        video=bool(args.video),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Local flower checkpoint dir OR HuggingFace repo id (user/repo).")
    parser.add_argument("--prompt", required=True, help="Task instruction.")
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Receding-horizon execution length: execute only the first N "
             "actions of each predicted chunk, then re-query the policy. "
             "Default: execute the full model-native chunk. The model always "
             "predicts policy.config.chunk_size actions (fixed at train time); "
             "this only changes how often we re-plan. Clamped to "
             "[1, policy chunk_size].",
    )
    parser.add_argument(
        "--sampling-steps", type=int, default=None,
        help="Number of rectified-flow Euler integration steps the DiT runs "
             "to denoise each action chunk (e.g. --sampling-steps 6). "
             "Default: use the checkpoint's config value. Not weight-coupled "
             "(safe to change at inference): fewer = faster/coarser, "
             "more = slower/finer. Must be >= 1.",
    )
    parser.add_argument("--device", default="auto",
                        help="Torch device: auto | cuda | mps | cpu (auto picks cuda > mps > cpu)")

    parser.add_argument("--robot-port", default="/dev/tty.usbmodem5B141136551")
    parser.add_argument("--robot-id", default="follower_111")
    parser.add_argument("--camera-key", default="main")
    parser.add_argument("--camera-index", type=int, default=0)

    parser.add_argument("--dry-run", action="store_true",
                        help="Synthetic robot; print actions instead of sending.")
    parser.add_argument("--log-dir", default="logs/inference")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--frame-every", type=int, default=10)
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--crop", default=None,
        help="Override the checkpoint's image crop as 'x0,y0,x1,y1'. By "
             "default the crop is read from the checkpoint config "
             "(policy.config.crop) so inference framing always matches "
             "training automatically. Use 'none' to force full-frame.",
    )

    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="Lerobot-rollouts")
    parser.add_argument("--wandb-mode", choices=["online", "offline"], default="online")
    args = parser.parse_args()

    if args.device == "auto":
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            args.device = "mps"
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        else:
            args.device = "cpu"
        print(f"[infer] auto-selected device: {args.device}")

    ckpt_path, ckpt_source, ckpt_origin = _resolve_checkpoint(args.checkpoint)
    runtime = capture_runtime_metadata()
    ckpt_short = checkpoint_short_id(args.checkpoint)
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    inference_run_id = (
        f"{time.strftime('%Y%m%d-%H%M%S')}_job{job_id}_{ckpt_short}_flower"
    )
    training = resolve_training_run(ckpt_path)

    meta_dict: dict = {
        "inference_run_id": inference_run_id,
        **runtime,
        "cli_args": vars(args),
        "policy_family": "flowervla",
        "checkpoint_source": ckpt_source,
        "checkpoint_path": ckpt_path,
        "checkpoint_origin": ckpt_origin,
        "training_run": training,
        "started_at": datetime.now().isoformat(),
        "robot": {
            "robot_type": "so101_follower_dryrun" if args.dry_run else "so101_follower",
            "port": args.robot_port,
            "id": args.robot_id,
            "camera_key": args.camera_key,
            "camera_index": args.camera_index,
        },
    }

    if args.video and args.no_log:
        print("[infer] WARNING: --video is a no-op with --no-log; skipping mp4.")

    # Provisional dims; refresh after robot connects.
    logger = _build_logger(args, meta_dict, action_dim=6, chunk_size=50)
    if isinstance(logger, RolloutLogger):
        print(f"[infer] inference_run_id: {inference_run_id}")
        print(f"[infer] log_dir: {logger.run_dir}")

    termination_reason = "init_fault"
    wandb_run = None
    robot = None
    policy = None
    t_start = time.perf_counter()

    try:
        # ---- Policy ----
        print(f"[infer] loading flower policy from {ckpt_path} on {args.device}...")
        policy = FlowerVLAPolicy.from_pretrained(ckpt_path, device=args.device)
        policy.eval()
        chunk_size = int(policy.config.chunk_size)
        action_dim = int(policy.config.action_dim)
        print(f"[infer] policy: chunk_size={chunk_size} action_dim={action_dim}")

        # Inference crop defaults to the crop baked into the checkpoint
        # (policy.config.crop) so framing always matches training. --crop
        # overrides; '--crop none' forces full frame.
        crop = getattr(policy.config, "crop", None)
        if args.crop is not None:
            if str(args.crop).strip().lower() in ("none", "off", ""):
                crop = None
            else:
                crop = [int(v) for v in str(args.crop).split(",")]
                if len(crop) != 4:
                    raise SystemExit("--crop must be 'x0,y0,x1,y1' or 'none'")
        print(f"[infer] image crop: "
              f"{crop if crop is not None else 'OFF (full frame)'} "
              f"(source: {'--crop override' if args.crop is not None else 'checkpoint config'})")
        meta_dict["inference_crop"] = crop

        exec_chunk_size = chunk_size
        if args.chunk_size is not None:
            exec_chunk_size = max(1, min(int(args.chunk_size), chunk_size))
            policy.exec_chunk_size = exec_chunk_size
            if exec_chunk_size != args.chunk_size:
                print(f"[infer] --chunk-size {args.chunk_size} clamped to "
                      f"{exec_chunk_size} (model predicts {chunk_size}-step chunks).")
            print(f"[infer] receding horizon: executing {exec_chunk_size}/"
                  f"{chunk_size} actions per chunk before re-planning.")

        ckpt_sampling_steps = int(policy.config.num_sampling_steps)
        sampling_steps = ckpt_sampling_steps
        if args.sampling_steps is not None:
            sampling_steps = max(1, int(args.sampling_steps))
            policy.config.num_sampling_steps = sampling_steps
            policy.model.num_sampling_steps = sampling_steps
            print(f"[infer] sampling steps overridden: {sampling_steps} "
                  f"(checkpoint default was {ckpt_sampling_steps}).")
        else:
            print(f"[infer] sampling steps: {sampling_steps} (checkpoint default).")
        meta_dict["policy_num_sampling_steps"] = sampling_steps
        try:
            param_count = sum(int(p.numel()) for p in policy.parameters() if p.requires_grad)
            meta_dict["policy_active_param_count"] = param_count
            meta_dict["policy_chunk_size"] = chunk_size
            print(f"[infer] policy loaded. active params: {param_count:,}")
        except Exception:
            pass

        # ---- Robot ----
        if args.dry_run:
            robot = DryRunRobot(camera_key=args.camera_key)
            print("[infer] DRY RUN — synthetic robot, no hardware.")
        else:
            robot = make_live_robot(
                port=args.robot_port,
                robot_id=args.robot_id,
                camera_key=args.camera_key,
                camera_index=args.camera_index,
            )
        robot.connect()

        if isinstance(logger, RolloutLogger):
            logger.action_dim = action_dim
            logger.state_dim = action_dim
            logger.chunk_size = exec_chunk_size
            (logger.run_dir / "meta.json").write_text(
                json.dumps(meta_dict, indent=2, default=str)
            )

        # ---- Wandb ----
        if not args.no_wandb:
            try:
                import wandb
                tags = [ckpt_short]
                tags.append(_sanitize_tag("robot", str(getattr(robot, "robot_type", "unknown"))))
                tags.append(_sanitize_tag("prompt", str(args.prompt)))
                training_group = (training or {}).get("wandb_run_id") or "no-sidecar"
                training_entity = (training or {}).get("wandb_entity")
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=training_entity,
                    id=inference_run_id,
                    name=inference_run_id,
                    group=training_group,
                    tags=tags + ["flowervla"],
                    config=meta_dict,
                    mode=args.wandb_mode,
                    dir=str(Path(args.log_dir) / inference_run_id),
                )
                print(f"[infer] wandb: {getattr(wandb_run, 'url', None) or '(offline)'}")
            except Exception as e:
                print(f"[infer] WARN wandb.init failed ({e!r}); continuing offline")
                wandb_run = None
        logger.attach_wandb(wandb_run)

        # ---- Callbacks for telemetry ----
        def _on_chunk(payload: dict) -> None:
            obs = payload["observation"]
            actions_arr = payload["actions"]
            if actions_arr is None:
                return
            state = np.array([float(obs[k]) for k in JOINT_KEYS], dtype=np.float32)
            images = {}
            img = obs.get(args.camera_key)
            if isinstance(img, np.ndarray) and img.ndim == 3:
                images[args.camera_key] = img
            chunk_obj = type("C", (), {})()  # tiny duck-typed object
            chunk_obj.actions = actions_arr
            chunk_obj.chunk_size = int(actions_arr.shape[0])
            try:
                logger.log_chunk(
                    chunk_idx=payload["chunk_idx"],
                    t0=payload["t0"],
                    t1=payload["t1"],
                    state=state,
                    prompt=args.prompt,
                    images=images,
                    chunk=chunk_obj,
                )
            except Exception as e:
                logger.note_event("warn.log_chunk", repr(e))

        def _on_step(payload: dict) -> None:
            images = {}
            img = payload.get("image")
            if isinstance(img, np.ndarray) and img.ndim == 3:
                images[args.camera_key] = img
            try:
                logger.log_step(
                    step=payload["step"],
                    chunk_idx=payload["chunk_idx"],
                    chunk_step=payload["chunk_step"],
                    inferred_this_step=payload["inferred_this_step"],
                    queue_depth_after=0,
                    state=payload["state"],
                    action_raw=payload["action_raw"],
                    action_sent=payload["action_sent"],
                    clamped_mask=np.zeros(action_dim, dtype=np.uint8),
                    period_actual_ms=payload.get("period_actual_ms"),
                    frame_path=logger.maybe_save_frame(images, payload["step"]),
                )
            except Exception as e:
                logger.note_event("warn.log_step", repr(e))

        # ---- Rollout ----
        summary = run_rollout(
            policy=policy,
            robot=robot,
            prompt=args.prompt,
            max_seconds=float(args.max_seconds),
            control_hz=float(args.control_hz),
            camera_key=args.camera_key,
            image_hw=int(policy.config.image_hw),
            video_key=policy.config.video_key,
            crop=crop,
            on_step=_on_step,
            on_chunk=_on_chunk,
        )
        termination_reason = summary["termination_reason"]
        print(f"[infer] rollout summary: {summary}")

    except KeyboardInterrupt:
        termination_reason = "user_abort"
    except Exception as e:
        termination_reason = f"exception:{type(e).__name__}"
        print(f"[infer] EXCEPTION: {e!r}")
        traceback.print_exc()
    finally:
        try:
            logger.close(verdict="unset", notes="", reason=termination_reason)
        except Exception as e:
            print(f"[infer] WARN logger.close failed: {e!r}")
        if wandb_run is not None:
            try:
                import wandb
                wandb.finish()
            except Exception:
                pass
        if robot is not None:
            try:
                robot.disconnect()
            except Exception as e:
                print(f"[infer] WARN robot.disconnect failed: {e!r}")
        wall = time.perf_counter() - t_start
        print(f"[infer] done. termination={termination_reason} wall={wall:.2f}s")
        if isinstance(logger, RolloutLogger):
            print(f"[infer] logs at {logger.run_dir}")
    sys.exit(0 if termination_reason in ("success", "timeout", "user_abort") else 2)


if __name__ == "__main__":
    main()
