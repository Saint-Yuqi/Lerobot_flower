"""Closed-loop runner bridging FlowerVLAPolicy and an SO-101 robot.

What this module owns:

  * `build_obs_tensor(...)`: turn a raw robot observation dict
    ({"joint.pos": float, ..., "main": (H, W, 3) uint8 RGB}) into the tensor
    inputs FlowerVLAPolicy.sample_chunk wants ({"observation.images.main":
    CHW float32 in [0,1], "observation.state": (S,) float32, "task": str}).
  * `action_tensor_to_robot_dict(...)`: invert the joint ordering convention
    so the robot's `send_action({...})` receives a properly keyed dict.
  * `DryRunRobot`: a synthetic robot for offline smoke-tests of the loop
    itself, without needing the physical arm.
  * `run_rollout(...)`: the actual control loop. Mirrors
    `scripts/run_inference.py`'s structure for SmolVLA but uses our policy
    + telemetry; no lerobot dep at runtime (lazy-imports lerobot_hw for the
    live robot).

The control-rate, max-seconds, NaN-guard, and chunk-capture pieces are all
identical to the SmolVLA inference path so analysis tooling (rollout_logger
output format, etc.) carries over.
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "third_party") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "third_party"))

from src.flower.policy import FlowerVLAPolicy  # noqa: E402


JOINT_KEYS = [
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
]


# ----------------------------------------------------------- observation glue

def build_obs_tensor(
    obs: dict[str, Any],
    *,
    camera_key: str,
    image_hw: int,
    video_key: str = "observation.images.main",
    joint_keys: Iterable[str] = tuple(JOINT_KEYS),
) -> dict[str, Any]:
    """Convert one robot observation into the dict FlowerVLAPolicy expects.

    Args:
        obs: dict from `robot.get_observation()` — joint floats + raw HWC uint8
             RGB image under ``camera_key``.
        camera_key: key under which the camera image lives in ``obs``.
        image_hw: target square HxW (224 for Florence-2). Resize happens inside
            the policy but we centre-crop+resize HERE so the network never sees
            squashed frames. We mirror lerobot's bilinear-resize-to-square.
        video_key: name FlowerVLAPolicy uses internally for the camera input.
        joint_keys: ordered list of joint feature keys.

    Returns:
        Dict ready to hand to `policy.sample_chunk(...)`.
    """
    img = obs.get(camera_key)
    if not isinstance(img, np.ndarray):
        raise TypeError(f"obs[{camera_key!r}] is not an ndarray; got {type(img)}")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"camera image must be HxWx3 uint8 RGB; got {img.shape}")
    # HWC uint8 -> CHW float32 in [0, 1]. Policy._build_flower_batch handles
    # the bilinear resize to image_hw; we just hand it the raw aspect-true image.
    img_t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float().div_(255.0)

    # State: ordered concat of joint positions.
    state = np.array(
        [float(obs[k]) for k in joint_keys], dtype=np.float32,
    )

    return {
        video_key: img_t,
        "observation.state": torch.from_numpy(state),
        "task": "",  # caller overwrites with the prompt
    }


def action_tensor_to_robot_dict(
    action: torch.Tensor | np.ndarray,
    joint_keys: Iterable[str] = tuple(JOINT_KEYS),
) -> dict[str, float]:
    """Convert a flat action tensor back into the {joint.pos: float} dict the robot wants."""
    if isinstance(action, torch.Tensor):
        action = action.detach().to("cpu").float().numpy()
    if action.shape[-1] != len(joint_keys):
        raise ValueError(
            f"action has {action.shape[-1]} dims but {len(joint_keys)} joint_keys."
        )
    return {k: float(action[i]) for i, k in enumerate(joint_keys)}


# ----------------------------------------------------------- dry-run robot

class DryRunRobot:
    """Synthetic robot for closed-loop smoke tests. Mirrors SO101Follower's interface."""

    robot_type = "so101_follower_dryrun"
    name = "so101_follower_dryrun"

    def __init__(self, camera_key: str = "main"):
        self._camera_key = camera_key
        self._joints = [k.split(".")[0] for k in JOINT_KEYS]

    @property
    def action_features(self) -> dict[str, type]:
        return {f"{j}.pos": float for j in self._joints}

    @property
    def observation_features(self) -> dict:
        out: dict = {f"{j}.pos": float for j in self._joints}
        out[self._camera_key] = (480, 640, 3)
        return out

    @property
    def cameras(self) -> dict:
        return {self._camera_key: None}

    def connect(self) -> None:
        return

    def disconnect(self) -> None:
        return

    def get_observation(self) -> dict:
        out: dict = {f"{j}.pos": 0.0 for j in self._joints}
        out[self._camera_key] = np.zeros((480, 640, 3), dtype=np.uint8)
        return out

    def send_action(self, action: dict) -> dict:
        print(f"[runner] dry-run action: "
              f"{ {k: round(float(v), 3) for k, v in action.items()} }")
        return action


def make_live_robot(*, port: str, robot_id: str, camera_key: str, camera_index: int):
    """Build a live SO101Follower from the vendored lerobot_hw package.

    Lazy import — only triggers serial / opencv code when actually called.
    """
    from lerobot_hw.cameras.opencv import OpenCVCameraConfig
    from lerobot_hw.robots.so_follower.config_so_follower import SO101FollowerConfig
    from lerobot_hw.robots.so_follower.so_follower import SO101Follower
    cfg = SO101FollowerConfig(
        port=port,
        id=robot_id,
        cameras={camera_key: OpenCVCameraConfig(
            index_or_path=camera_index, width=640, height=480, fps=30,
        )},
    )
    return SO101Follower(cfg)


# ----------------------------------------------------------- control loop

def install_chunk_capture(policy: FlowerVLAPolicy) -> dict:
    """Wrap policy.sample_chunk to capture every chunk produced, for the logger.

    Mirrors `_install_chunk_capture` in scripts/run_inference.py; here we hook
    sample_chunk because that's what fills the chunk queue.
    """
    slot: dict = {"count": 0, "actions": None, "t0": None, "t1": None}
    orig = policy.sample_chunk

    def _wrapped(observation, *args, **kwargs):
        t0 = time.perf_counter()
        out = orig(observation, *args, **kwargs)
        try:
            arr = out.detach().to("cpu").float().numpy()
            if arr.ndim == 3 and arr.shape[0] == 1:
                arr = arr[0]
        except Exception:
            arr = None
        slot["actions"] = arr
        slot["t0"] = t0
        slot["t1"] = time.perf_counter()
        slot["count"] += 1
        return out

    policy.sample_chunk = _wrapped  # type: ignore[assignment]
    return slot


def run_rollout(
    *,
    policy: FlowerVLAPolicy,
    robot,
    prompt: str,
    max_seconds: float,
    control_hz: float,
    camera_key: str = "main",
    image_hw: int = 224,
    video_key: str = "observation.images.main",
    on_step: Callable[[dict], None] | None = None,
    on_chunk: Callable[[dict], None] | None = None,
) -> dict:
    """Run a closed-loop rollout. Returns a small summary dict.

    Args:
        policy: a `FlowerVLAPolicy` already on the inference device.
        robot: anything with the SO101Follower interface (real or `DryRunRobot`).
        prompt: task instruction passed at every step.
        max_seconds: wall-clock deadline for the rollout.
        control_hz: target loop rate; the loop sleeps to honour it.
        camera_key: which camera key to read off the observation.
        image_hw: target square HxW for the policy's input.
        video_key: dataset-style key the policy expects.
        on_step: optional callback that receives a per-step telemetry dict.
        on_chunk: optional callback that receives a per-chunk dict on chunk refills.

    Returns:
        Summary dict with steps, termination_reason, wall_seconds.
    """
    chunk_slot = install_chunk_capture(policy)
    policy.eval()
    policy.reset()

    period = 1.0 / max(float(control_hz), 1e-6)
    deadline = time.perf_counter() + float(max_seconds)
    step = 0
    chunk_idx = -1
    chunk_step = 0
    last_chunk_count = chunk_slot["count"]
    termination_reason = "timeout"
    prev_loop_start: float | None = None
    t_start = time.perf_counter()

    try:
        while time.perf_counter() < deadline:
            loop_start = time.perf_counter()
            period_actual_ms = (
                None if prev_loop_start is None
                else (loop_start - prev_loop_start) * 1000.0
            )

            try:
                obs = robot.get_observation()
            except Exception as e:
                termination_reason = "robot_fault"
                print(f"[runner] EXCEPTION during get_observation: {e!r}")
                traceback.print_exc()
                break

            obs_tensor = build_obs_tensor(
                obs, camera_key=camera_key, image_hw=image_hw, video_key=video_key,
            )
            obs_tensor["task"] = prompt

            try:
                action_t = policy.select_action(obs_tensor)
            except Exception as e:
                termination_reason = "policy_fault"
                print(f"[runner] EXCEPTION during select_action: {e!r}")
                traceback.print_exc()
                break

            inferred_this_step = chunk_slot["count"] > last_chunk_count
            if inferred_this_step:
                chunk_idx += 1
                chunk_step = 0
                last_chunk_count = chunk_slot["count"]
                if on_chunk is not None and chunk_slot["actions"] is not None:
                    on_chunk({
                        "chunk_idx": chunk_idx,
                        "actions": chunk_slot["actions"],
                        "t0": chunk_slot["t0"],
                        "t1": chunk_slot["t1"],
                        "observation": obs,
                    })
            else:
                chunk_step += 1

            action_dict = action_tensor_to_robot_dict(action_t)

            # NaN/Inf guard.
            nan_dim = None
            for k, v in action_dict.items():
                if not np.isfinite(float(v)):
                    nan_dim = k
                    break
            if nan_dim is not None:
                termination_reason = "policy_nan"
                print(f"[runner] non-finite action on {nan_dim}: {action_dict[nan_dim]}")
                break

            try:
                sent = robot.send_action(action_dict)
            except Exception as e:
                termination_reason = "robot_fault"
                print(f"[runner] EXCEPTION during send_action: {e!r}")
                traceback.print_exc()
                break

            if on_step is not None:
                on_step({
                    "step": step,
                    "chunk_idx": max(chunk_idx, 0),
                    "chunk_step": chunk_step,
                    "inferred_this_step": inferred_this_step,
                    "action_raw": action_t.detach().to("cpu").float().numpy(),
                    "action_sent": np.array(
                        [float(sent.get(k, action_dict.get(k, np.nan))) for k in JOINT_KEYS],
                        dtype=np.float32,
                    ),
                    "state": np.array(
                        [float(obs[k]) for k in JOINT_KEYS],
                        dtype=np.float32,
                    ),
                    "image": obs.get(camera_key),
                    "period_actual_ms": period_actual_ms,
                })

            step += 1
            prev_loop_start = loop_start
            elapsed = time.perf_counter() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        termination_reason = "user_abort"
    return {
        "steps": step,
        "termination_reason": termination_reason,
        "wall_seconds": round(time.perf_counter() - t_start, 3),
        "final_chunk_idx": chunk_idx,
    }
