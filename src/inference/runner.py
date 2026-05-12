"""Pure control-loop runner.

Owns the loop, NaN guard, clamp call, CUDA-sync, and `try/finally`.
Tests call it directly with mocks; `scripts/run_inference.py:main()` is
a thin wrapper that builds the deps (policy, robot, logger, clamp_fn)
and forwards.

Contract: `run_rollout(...)` returns the `termination_reason` string
that the wrapper writes into `episode.json` and uses to dispatch the
verdict.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from src.models.base_vla import Observation


def _read_obs(robot, camera_key: str) -> tuple[dict, np.ndarray]:
    """Always linear: capture, unpack, return (images_dict, state).

    The dry-run path is handled at the wrapper level by passing a
    `_DryRunRobot` shim — `_DryRunRobot.capture_observation()` returns
    synthetic zeros. This keeps the runner branch-free.
    """
    cam_frames = robot.capture_observation()
    images = {camera_key: cam_frames[f"observation.images.{camera_key}"]}
    state = cam_frames["observation.state"]
    return images, np.asarray(state, dtype=np.float32).reshape(-1)


def run_rollout(
    *,
    policy,
    robot,
    logger,
    control_hz: float,
    max_seconds: float,
    clamp_fn,
    prompt: str,
    camera_key: str,
    _skip_sync: bool = False,
) -> str:
    """Main control loop. Returns the `termination_reason` string.

    `_skip_sync` is a test hook for the CUDA-sync contrast (verification
    step 7) — when True, the latency timer skips `torch.cuda.synchronize`
    so we can prove the synced version actually catches GPU-async work.
    """
    # Lazy import torch so smoke tests on no-CUDA boxes don't pay the
    # cost. Falls back gracefully if torch isn't available at all.
    try:
        import torch
        _have_torch = True
    except Exception:
        torch = None  # type: ignore
        _have_torch = False

    period = 1.0 / max(control_hz, 1e-6)
    deadline = time.monotonic() + max_seconds
    action_queue: list[np.ndarray] = []
    chunk_idx = -1
    chunk_step = 0
    step = 0
    prev_loop_start: float | None = None
    termination_reason = "timeout"

    try:
        while time.monotonic() < deadline:
            loop_start = time.monotonic()
            period_actual_ms = (
                None if prev_loop_start is None
                else (loop_start - prev_loop_start) * 1000.0
            )
            try:
                images, state = _read_obs(robot, camera_key)
            except Exception as e:
                termination_reason = "robot_fault"
                logger.note_event("error.robot", repr(e))
                break

            just_refilled = False
            if not action_queue:
                just_refilled = True
                chunk_idx += 1
                t0 = time.perf_counter()
                try:
                    chunk = policy.predict(
                        Observation(images=images, state=state, prompt=prompt)
                    )
                except Exception as e:
                    termination_reason = "policy_fault"
                    logger.note_event("error.policy", repr(e))
                    break
                if (not _skip_sync) and _have_torch and torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                logger.log_chunk(
                    chunk_idx=chunk_idx,
                    t0=t0,
                    t1=t1,
                    state=state,
                    prompt=prompt,
                    images=images,
                    chunk=chunk,
                )
                action_queue = list(chunk.actions)
                chunk_step = 0
            else:
                chunk_step += 1

            action_raw = np.asarray(action_queue.pop(0))
            if not np.isfinite(action_raw).all():
                logger.note_event(
                    "nan",
                    f"step={step} action_raw={action_raw.tolist()}",
                )
                termination_reason = "policy_nan"
                break

            action_sent, mask = clamp_fn(action_raw)
            try:
                robot.send_action(action_sent)
            except Exception as e:
                termination_reason = "robot_fault"
                logger.note_event("error.robot", repr(e))
                break

            logger.log_step(
                step=step,
                chunk_idx=chunk_idx,
                chunk_step=chunk_step,
                inferred_this_step=just_refilled,
                queue_depth_after=len(action_queue),
                state=state,
                action_raw=action_raw,
                action_sent=action_sent,
                clamped_mask=mask,
                period_actual_ms=period_actual_ms,
                frame_path=logger.maybe_save_frame(images, step),
            )
            step += 1
            prev_loop_start = loop_start

            elapsed = time.monotonic() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
        else:
            # while-loop ran to its deadline without break.
            termination_reason = "timeout"
    except KeyboardInterrupt:
        termination_reason = "user_abort"

    return termination_reason
