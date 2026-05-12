"""Per-step action clamp.

Referenced by `configs/robot/so101.yaml` line 32 ("clamp_actions: true")
but never previously implemented. Adding it here completes the
`action_raw -> NaN/Inf check (terminate) -> clamp -> send` pipeline.

Implementation MUST use `np.clip` (which preserves NaN), not min/max
fallbacks. The runner's NaN guard runs on `action_raw` BEFORE this
clamp; if a buggy clamp impl silently replaced NaN with a limit value,
the guard would become a no-op and bad actions would reach the robot.
"""
from __future__ import annotations

import numpy as np


def clamp_action(
    action: np.ndarray,
    joint_limits_deg: dict[str, list[float]],
    joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Clamp `action` to per-joint limits.

    Args:
        action: shape (action_dim,), float32. Caller has already
            confirmed `np.isfinite(action).all()`.
        joint_limits_deg: ordered dict-like, mapping joint name ->
            [low, high]. Loaded from the robot YAML.
        joint_names: ordered list of joint names of length action_dim.
            Used to look up each joint's limits.

    Returns:
        (clamped_action, clipped_mask) where `clipped_mask` is a uint8
        array of shape (action_dim,), 1 where the action was at or beyond
        the joint limit (i.e. the clamp actually moved the value).
    """
    if action.shape[0] != len(joint_names):
        raise ValueError(
            f"clamp_action: action dim {action.shape[0]} != "
            f"len(joint_names)={len(joint_names)}"
        )

    lows = np.empty(len(joint_names), dtype=np.float32)
    highs = np.empty(len(joint_names), dtype=np.float32)
    for i, name in enumerate(joint_names):
        if name not in joint_limits_deg:
            raise KeyError(
                f"clamp_action: joint '{name}' missing from joint_limits_deg "
                f"(have: {list(joint_limits_deg.keys())})"
            )
        lo, hi = joint_limits_deg[name]
        lows[i] = float(lo)
        highs[i] = float(hi)

    # np.clip preserves NaN by design (NaN in -> NaN out). This is the
    # invariant that makes the upstream NaN guard load-bearing rather
    # than redundant.
    clamped = np.clip(action.astype(np.float32, copy=False), lows, highs)
    # mask: 1 where the input violated a limit (so clamp had to move it).
    # Comparing the original action handles the edge case where the
    # input is exactly at the limit -> mask stays 0 (no clipping happened).
    mask = ((action < lows) | (action > highs)).astype(np.uint8)
    return clamped, mask
