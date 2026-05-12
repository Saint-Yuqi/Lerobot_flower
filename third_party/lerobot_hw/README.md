# Vendored lerobot hardware drivers (`lerobot_hw`)

Source: a minimal subset of `lerobot==0.5.1`'s installable wheel — only the
hardware code paths needed to talk to the SO-101 follower arm and an OpenCV
USB camera. The original `lerobot` package requires Python 3.12; this vendored
copy is patched to run under Python 3.10 inside the `flower` conda env.

Used by `scripts/run_inference_flower.py` via `src.flower.runner.make_live_robot`.
Lazy-imported at robot-connect time, so dry-run inference and offline eval do
not load any of this.

## Subpackages

| Path | Purpose |
|---|---|
| `robots/so_follower/` | `SO101Follower`, `SO101FollowerConfig`, robot kinematics |
| `motors/` | Generic motor bus + Feetech (SO-101 servos) + Dynamixel drivers |
| `cameras/` | OpenCV camera config + driver |
| `utils/` | `decorators`, `errors`, `utils`, `constants`, `import_utils` |
| `types.py` | `RobotAction`, `RobotObservation` aliases |

## Patches applied to upstream

1. **Module rename**: every `from lerobot.X` / `import lerobot.X` rewritten to
   `from lerobot_hw.X` / `import lerobot_hw.X`. Lets us drop the package under
   `third_party/` without colliding with a real lerobot install.

2. **PEP 695 type aliases**: upstream uses `type NameOrID = str | int` (Python
   3.12+). Rewritten to plain `NameOrID = str | int` (works on 3.10 with
   `from __future__ import annotations`, already in those files).

## Required pip packages in `flower` env

- `pyserial` — SO-101 serial bus
- `feetech-servo-sdk` — Feetech SCS-series servo protocol
- `draccus` — config parser for `RobotConfig.ChoiceRegistry`
- `deepdiff` — used by lerobot's utility code
- `jsonlines`, `pyarrow`, `pandas`, `av` — already required by our dataset reader

## Re-syncing
If lerobot 0.5.x is updated upstream and you want to refresh the vendored code,
re-copy the same subtree, then re-run the patch sed (see the README at the top
of this repo). Don't forget to re-patch `motors_bus.py` PEP 695 syntax.
