# Vendored FlowerVLA (flower_calvin variant)

Source: https://github.com/intuitive-robots/flower_vla (calvin variant), copied verbatim
into `flower/` then locally patched. Imported by adding `third_party/flower_vla/` to
`sys.path` and writing `from flower.models.flower import FlowerVLA`.

This is **vendored**, not installed. It does not appear on the lerobot env at all.
It runs only inside the dedicated `flower` conda env (Python 3.10, torch 2.2.2,
transformers 4.46, pytorch-lightning 2.0.8, hydra-core 1.1.1).

## Local patches applied

| Tag | File | What changed |
|----|------|------|
| F3 | `flower/models/flower.py` (`encode_observations`) | `embed_tensor = torch.zeros(B, 1, 1)` → `torch.zeros(B)`. `FreqEmbedder.timestep_embedding` does `t[:, None]` and expects a 1-D tensor; the original 3-D shape broke the broadcast. |
| F4 | `flower/models/utils.py` (`ActionIndex`) | Registered `so101` as action type 3 with action_dim 6 and `('SO101', 'position', 1)` → 3. SO-101 follower has 6 DoF (no gripper-as-7th-dim convention here). |
| F5 | `flower/models/flower.py` (`__init__` + `encode_observations`) | Added `self.default_action_type = 1` attribute (preserves upstream behavior). `encode_observations` now uses `torch.full_like(action_type_tensor, self.default_action_type)` instead of hardcoded 1. Set `model.default_action_type = 3` for SO-101. |

F1 (image resize to 224 before Florence-2) and F2 (`model.obs_modalities = "state_obs"` after construction) are applied at the policy-wrapper level in `src/flower/policy.py`, not in the vendored model.

## Touch points
- `flower.models.flower.FlowerVLA` — main module
- `flower.callbacks.ema.EMA` — exponential moving average (not used yet)
- `flower.utils.lr_schedulers.*` — optional, may be replaced by our own scheduler

## Re-syncing
If upstream changes meaningfully, re-copy from `/tmp/flower_calvin/flower/` and re-apply F3+F4 by reading this README.
