# Architecture

This repo and its sibling
`/shares/feldmann.ics.mnf.uzh/Yuqi/Lerobot` (SmolVLA) share the same
training and eval skeleton — `task1/task2/task3` configs, phase-weighted
sampling, color-stratified splits, wandb sidecar, early-stop logic,
rollout logger, offline eval gates. Only the `model.*` config block and
the policy implementation differ.

## Repo layout

```
.
├── environment.yml             ← conda env (name: flower, Python 3.10)
├── configs/
│   ├── train/
│   │   ├── full_eval{1,2,3}.yaml    ← three task configs, same schema as SmolVLA
│   │   ├── overfit.yaml             ← 200-step regression baseline
│   │   └── overfit_phase.yaml       ← phase-sampler variant
│   └── robot/so101.yaml
├── scripts/
│   ├── train_flower.py              ← training entrypoint
│   ├── train_flower.slurm
│   ├── overfit_flower.py
│   ├── eval_offline_flower.py       ← Tests A + C
│   ├── run_inference_flower.py      ← live / dry-run rollouts
│   ├── label_rollout.py             ← post-hoc verdict labeller
│   └── spikes/                      ← feasibility-spike scripts (historical)
├── src/
│   ├── flower/                      ← FlowerVLA-specific (policy, dataset, normalizer, runner)
│   ├── data/                        ← splits, phase_labels, sampler (copied from SmolVLA repo)
│   ├── inference/                   ← runner, rollout_logger, safety (copied)
│   ├── utils/                       ← checkpoint_meta, run_metadata, gpu_metrics (copied)
│   └── models/base_vla.py           ← Observation / ActionChunk / BaseVLA (copied)
└── third_party/
    ├── flower_vla/                  ← vendored upstream (FLOWERVLA + Florence-2 wiring)
    └── lerobot_hw/                  ← vendored SO-101 drivers (lazy-imported)
```

## What changed when this repo was split out of `Lerobot`

1. **Source of truth for shared code**: `src/data/{splits,phase_labels,sampler}.py`,
   `src/utils/*`, `src/inference/*`, `src/models/base_vla.py` were **copied** here.
   The two repos may now diverge; if a fix lands in one side, port it manually if
   it applies to the other.
2. **Config naming**: dropped the `_flower` suffix (e.g. `full_eval1_flower.yaml`
   → `full_eval1.yaml`). The on-wandb `experiment_name` still contains `_flower_v1`
   to disambiguate runs from the SmolVLA side.
3. **Data paths**: `configs/train/full_eval{1,2}.yaml` use absolute paths into the
   SmolVLA repo's `data/hf/...` dir. The HF snapshots themselves are physically
   stored once and shared by both repos.
4. **Inference loop file**: `scripts/run_inference.py` is NOT here — it's the
   SmolVLA inference script. FlowerVLA uses `scripts/run_inference_flower.py`.
5. **Removed**: `src/models/smolvla_wrapper.py`, `src/data/prompt_aug.py`,
   `configs/data/arrangements.json` (all SmolVLA-only).

## F-patch series (upstream patches applied to `third_party/flower_vla/`)

| ID | What | Where |
|---|---|---|
| F1 | Input image resized to 224×224 before Florence-2 | `src/flower/policy.py` (preprocess) |
| F2 | `obs_modalities="state_obs"` (avoids second-camera path) | `src/flower/policy.py` |
| F3 | `FreqEmbedder` accepts 1-D input (upstream expected 2-D) | `third_party/flower_vla/flower/models/utils.py` |
| F4 | Action type 3 = `so101` (6-DoF: shoulder×2, elbow, wrist×2, gripper) | `third_party/flower_vla/flower/models/flower.py` |
| F5 | `default_action_type` class attribute for serialization | `third_party/flower_vla/flower/models/flower.py` |

Don't fork upstream — patches are in-tree and documented in
[third_party/flower_vla/README.md](../third_party/flower_vla/README.md).

## Shared training skeleton (matched with SmolVLA repo)

| Concern | File | Notes |
|---|---|---|
| Train entrypoint | `scripts/train_flower.py` | Mirrors `scripts/train.py` in SmolVLA repo |
| Config schema | `configs/train/*.yaml` | Identical top-level keys; only `model.*` block diverges |
| Color-stratified split | `src/data/splits.py` | `episodes_by_color`, `train_val_episode_split` |
| Phase labels | `src/data/phase_labels.py` | Gripper-based, NPZ cached, alignment-checked |
| Phase sampling | `src/data/sampler.py` | `make_phase_weighted_sampler`, `assert_dataset_alignment` |
| Checkpoint sidecar | `src/utils/checkpoint_meta.py` | wandb metadata next to ckpt |
| Run metadata | `src/utils/run_metadata.py` | git SHA, host, Python ver |
| Rollout telemetry | `src/inference/rollout_logger.py` | steps.csv, chunks.jsonl, outcome.json |
| Rollout labeling | `scripts/label_rollout.py` | post-hoc verdict/notes/tags, updates wandb |

If you change any of these here, ask yourself whether the SmolVLA side
needs the same change — there's no automated sync.

## How `lerobot_hw` import works

`src/flower/runner.py` adds `<repo>/third_party/` to `sys.path` so
`from lerobot_hw.cameras.opencv import OpenCVCameraConfig` resolves to
`third_party/lerobot_hw/cameras/opencv/__init__.py`. Imports are **lazy**
(inside `make_live_robot`) so dry-run mode and training never touch
serial / opencv. See [src/flower/runner.py:30-40](../src/flower/runner.py).

## How vendored `flower_vla` imports work

`src/flower/policy.py` adds `<repo>/third_party/flower_vla/` to
`sys.path` so `from flower.models.flower import FLOWERVLA` resolves
without conflicting with our own `src/flower/` package (different
top-level entry: `flower` vs `src.flower`). See
[src/flower/policy.py:40-50](../src/flower/policy.py).

## Historical: `scripts/spikes/flower_spike.py`

This spike script has a dual-env workflow — branch A dumps a sample
batch using the `lerobot` package (must run in the SmolVLA env), branch B
loads the dumped batch and runs a FlowerVLA forward (runs in the flower
env). Branch A will fail in this repo's env. Use it from the SmolVLA
sibling repo if needed, or just keep it as reference; it was used once
during Phase 0 spike verification.
