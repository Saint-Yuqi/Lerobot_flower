# FlowerVLA feasibility spike

**Branch:** `spike/flowervla-feasibility`
**Date:** 2026-05-11
**Status:** GO — full forward + backward verified on task3 with finite loss and non-zero grad. Adapter scope is larger than the plan estimated.

## Verdict

**SPIKE_VERDICT: GO** — with caveats.

Final stage-B run on one task3 batch (B=2, chunk=50, padded action=7):

```
[spike-B] forward OK. loss=3084.293213  info_keys=['diff_min', 'diff_max', 'diff_mean', 'loss']
[spike-B] mean grad norm: 8.2542e+01  finite_params=258
[spike-B] SPIKE_VERDICT: GO
```

Loss is large (random init + dataset-stat mismatch — expected at this
stage) but **finite and positive**, gradient norm is **non-zero**, and 258
parameters carry finite gradients through the DiT backbone. Florence-2
loads and runs; the rectified-flow loss path is correct.

**Caveat: the Phase 2 adapter is more involved than the plan estimated.**
Four concrete upstream-code-rot issues surfaced during the spike (see "Adapter scope adjustments" below); each adds work to Phase 2.2 beyond what the plan describes.

## Upstream choice: flower_calvin (not flower_pret)

| Aspect | flower_pret (OXE) | flower_calvin (CALVIN/LIBERO) | Pick |
|--------|-------------------|-------------------------------|------|
| Purpose | pretraining on OXE-RLDS | finetune on a single embodiment | flower_calvin (closer to what we want) |
| Framework | plain `nn.Module` + `accelerate` | PyTorch Lightning | flower_calvin (cleaner forward) |
| Language input | pre-tokenized `input_ids` / `attention_mask` | raw `lang_text: List[str]` (tokenized inside `construct_prompts`) | flower_calvin (lerobot gives us raw strings) |
| Multi-action-space | `ActionIndex` for OXE diversity | single action space | flower_calvin (simpler for SO-101) |
| VLM default | Florence-2-large | Florence-2-base | flower_calvin (smaller; less VRAM) |
| Cameras | primary + optional wrist | primary `rgb_static` + optional `rgb_gripper` | tied (we use primary only) |

## Pre-spike sanity (verified)

- `lerobot==0.5.1` installed; `from lerobot.optim.optimizers import AdamConfig` imports cleanly.
- All three target HF datasets exist and are accessible:
  - `ethrl2026/task1_20260509_prompt_lighting_augmented_360` (private, 360 eps, 165828 frames, natural-language prompts)
  - `ethrl2026/task2_20260509_stage2_random_lighting_augmented_2160` (private, 2160 eps, 870828 frames, natural-language prompts)
  - `ethrl2026/so101_pickup_20260509_185350_task3` (private, 45 eps, 18959 frames, **encoded labels** `Y+L`, `T+M`, etc.)
- HF whoami: `PrajnaYang`, member of `ethrl2026` org → can push to the eval{1,2,3}-flower-v1 repos.
- **Hub tags:** all three new datasets are now tagged `v3.0` (done with
  `HfApi().create_tag(repo, tag="v3.0", repo_type="dataset")` on 2026-05-11).
  `LeRobotDataset(repo_id=...)` resolves cleanly from a cold cache;
  no `snapshot_download` workaround needed.

## Batch-shape gaps and adapter contract

Spike output (`scripts/spikes/flower_spike.py`) confirms the following with concrete numbers:

| # | Aspect | LeRobotDataset (SO-101) | FlowerVLA-CALVIN expects | Adapter fix |
|---|--------|-------------------------|--------------------------|-------------|
| G1 | image | `observation.images.main` shape `(B, 3, 480, 640)` float32 in [0,1] | `batch["rgb_obs"]["rgb_static"]` shape `(B, T, 3, H, W)` | unsqueeze T=1; Florence-2's `_encode_image` does its own internal resize/normalize |
| G2 | action_dim | `action` shape `(B, 50, 6)` (6-DoF joints) | default 7 (CALVIN delta-EE) | set `FlowerVLAConfig.action_dim=6` |
| G3 | chunk_size | 50 | default `act_window_size=10` | set `act_window_size=50`; verify DiT positional buffers grow |
| G4 | proprio | `observation.state` shape `(B, 6)` (joint positions, deg) | `batch[obs_modalities]["proprio"]` if `use_proprio=True`; default `lowdim_obs_dim=7` | set `lowdim_obs_dim=6`, `use_proprio=True`; pack under `obs_modalities` (CALVIN config uses `"state_obs"`) |
| G5 | language | `task` = `List[str]`, raw natural language (or encoded `'Y+L'` on task3) | `batch["lang_text"]` = `List[str]`, tokenized inside `construct_prompts` via Florence-2 processor | pass through; no work needed |
| G6 | action norm | lerobot's `NormalizerProcessor` normalizes before policy, unnorms after | flow-matching expects targets in a stable range (usually [-1,1] or [0,1]) | run lerobot's normalizer; verify the resulting range is in the band flow-matching trains on (decide during forward-pass spike) |

`observation_delta_indices` value for the FlowerVLAConfig (Phase 2.2):
**None** (single-frame obs, like SmolVLA). FlowerVLA-CALVIN treats `T` as
1 unless `use_second_view=True` with separate temporal windows — for SO-101's
single `main` camera and our current setup, `T=1` is correct.

## Concrete `FlowerVLAConfig` initial knobs (filled from spike, no guessing)

```python
@PreTrainedConfig.register_subclass("flowervla")
@dataclass
class FlowerVLAConfig(PreTrainedConfig):
    pretrained_path: str | None = None       # upstream HF id for a pretrained ckpt (TBD)
    chunk_size: int = 50
    n_action_steps: int = 50
    n_obs_steps: int = 1                     # CALVIN default; single-frame
    device: str = "cuda"
    use_amp: bool = False

    # FlowerVLA-specific (verified from /tmp/flower_calvin/flower/models/flower.py):
    vlm_path: str = "microsoft/Florence-2-base"
    action_dim: int = 6                      # SO-101
    lowdim_obs_dim: int = 6                  # SO-101 joint state
    act_window_size: int = 50                # matches chunk_size
    use_second_view: bool = False            # SO-101 single camera
    use_proprio: bool = True                 # we have observation.state
    freeze_florence: bool = False            # spike default; consider True for cheap finetune
    freeze_vision_tower: bool = False

    num_sampling_steps: int = 4              # FlowerVLA published default for inference
    sampling_type: str = "ln"                # rectified-flow default

    def get_optimizer_preset(self):
        # lerobot.optim.optimizers.AdamConfig imports OK in v0.5.1.
        # Either return AdamConfig(lr=1e-4) or None (train.py builds AdamW directly).
        return None

    def get_scheduler_preset(self):
        return None

    def validate_features(self):
        required = {"observation.state", "action"}
        present = set(self.input_features) | set(self.output_features)
        missing = required - present
        if missing:
            raise ValueError(f"FlowerVLAConfig: missing features {missing}")

    @property
    def observation_delta_indices(self):
        return None   # single-frame obs; matches SmolVLA, matches FlowerVLA T=1

    @property
    def action_delta_indices(self):
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self):
        return None
```

## Adapter scope adjustments (forward-pass findings)

The spike's `--stage B` forward surfaced four upstream-code issues that
the Phase 2 adapter has to absorb. None of them block the integration —
they all have clear fixes — but they add to the adapter's surface area.

| # | Upstream behavior | Adapter fix |
|---|-------------------|-------------|
| F1 | Florence-2's `_encode_image` asserts square feature maps (`h * w == num_tokens`). SO-101's 480×640 4:3 frames fail. | Resize to 224×224 inside `FlowerVLAPolicy.forward` / `predict_action_chunk` before passing to the VLM. The spike confirms this works. |
| F2 | `FLOWERVLA.__init__` hardcodes `self.obs_modalities = []`. `encode_observations` later does `batch[self.obs_modalities]` → `TypeError: unhashable type: 'list'`. | Set `model.obs_modalities = "state_obs"` after construction in `FlowerVLAPolicy.__init__`. Also pack proprio under that key in the forward-batch builder. |
| F3 | `encode_observations` builds `embed_tensor = torch.zeros(B, 1, 1)` and passes that to `FreqEmbedder`, which assumes 1-D input → the resulting `frequency_embeds.shape` is `(B, 1, 1, dim)`, breaking the proprio path's `batch_size, _ = output_shape` unpack downstream (4-tuple vs expected 2-tuple). | Either (a) monkey-patch `frequency_embedder` to flatten its input first, or (b) override `encode_observations` in the adapter to feed a properly-shaped 1-D scalar. We chose `use_proprio=False` for the spike forward; Phase 2.2 needs to enable proprio properly. |
| F4 | `ActionIndex` hardcodes three action spaces (`joint_single=8`, `eef_delta=7`, `bimanual_nav=16`) at module-construction time. The DiT's `action_encoders` / `action_decoders` are built from those dims. SO-101's 6-DoF doesn't fit any. | Either pad SO-101 actions to 7 (gripper-velocity-style ghost dim) for inference compatibility, **or** (better) monkey-patch `ActionIndex` to register a new `so101: 6` space and rebuild the affected encoders. Spike used the pad to reach a finite loss; Phase 2.2 should add the SO-101 entry. |

These are tractable but real. Initial estimate for Phase 2.2 was "a few hundred lines"; with F2/F3/F4 each requiring a focused override, expect closer to ~600-800 LOC across `modeling_flower.py` + `configuration_flower.py`, with at least one of those overrides being a careful monkey-patch on the upstream module.

## Phase 2 readiness checklist (post-spike)

- [x] Decide upstream base: `flower_vla_calvin`
- [x] Confirm Florence-2 loads + DiT runs on SO-101 imagery (224×224 resize)
- [x] Confirm `rf_loss` produces finite loss + non-zero grad
- [x] Build conda env `flower` with the required deps (torch 2.2.2 + pytorch-lightning 2.0.8 + transformers 4.46 + hydra 1.1.1)
- [ ] Choose pretrained init source — random init produces loss ~3000 on a single batch (expected for warm-up). For a real run, point at one of:
  - `intuitive-robots`'s HF org for FlowerVLA-pretrained weights (TBD which exact repo)
  - CALVIN-finetuned ckpts from `flower_vla_calvin`'s README
- [ ] Decide how `train.py` invokes FlowerVLA — the conda env split means `train.py` (lerobot env) can't directly import FlowerVLA. Either:
  - Move `train.py` over to the `flower` env and `pip install -e lerobot` there (heavy)
  - Vendor a *trimmed* `flower_vla` module (just the model + deps it needs) under `third_party/` and write a fresh, lerobot-compatible model class that re-implements `rf_loss` + Florence-2 conditioning without pulling in PyTorch Lightning
  - Run two-stage training: lerobot-side dataloader + adapter in flower env, with the lerobot policy interface bridged via a thin RPC or shared-memory ring
  The cleanest is option 2; option 3 is the "do-no-harm" path that keeps both envs untouched. **This is the next major decision.**

2. **Decide on pretrained init.** FlowerVLA's value vs SmolVLA depends on
   the pretrained checkpoint being a strong prior. Need to point
   `pretrained_path` at one of:
   - the OXE pretrain (from `flower_vla_pret`)
   - a CALVIN-finetuned ckpt (from `flower_vla_calvin`)
   - random init (likely much worse — only worth it as a control)
   Browse https://huggingface.co/intuitive-robots for an actual HF weights repo.

3. **Action normalization sanity.** SO-101 actions are joint angles in degrees,
   range roughly ±90°. lerobot's per-dim normalizer maps them to a learned range
   from dataset stats. FlowerVLA's `rf_loss` interpolates between targets and
   `randn_like` noise (mean 0, std 1) — targets that aren't roughly unit-scale
   will train poorly. Confirm at first forward.

## Files this spike leaves on the branch

- `scripts/spikes/flower_spike.py` — the throwaway batch-shape script (to be deleted in the Phase 2 landing commit, explicit in the commit message)
- `docs/flowervla_spike.md` — this file (kept; basis for Phase 2 adapter code)

`third_party/flower_vla/` (the submodule) is added in Phase 2.1, not here —
the spike clones to `/tmp/` to avoid polluting the working tree until we
commit to integration.
