# FlowerVLA — SO-101 Pick & Place

Florence-2 + DiT VLA policy for the SO-101 robot. Split out from the
SmolVLA-focused `Lerobot` repo so the two backbones can evolve
independently.

Sibling repo: `/shares/feldmann.ics.mnf.uzh/Yuqi/Lerobot` (SmolVLA, Python
3.12, lerobot 0.5.x). This repo's `data/hf/...` snapshots point back to
that sibling — see [data layout](#data-layout) below.

## Quickstart

**Environment.** Python 3.10 / torch 2.2.2 / transformers 4.46
(Florence-2 is broken on transformers ≥5 due to an upstream
`forced_bos_token_id` bug we can't patch). All deps pinned in
`environment.yml`.

```bash
conda env create -f environment.yml          # creates env named "flower"
conda activate flower
python --version                             # 3.10.x
```

If the env already exists from the sibling repo (you ran Phase 2 of the
migration), `conda env update -f environment.yml --prune` keeps it in
sync.

## Run training

```bash
# Overfit smoke (200 steps, single episode) — the baseline regression check.
python scripts/overfit_flower.py --config configs/train/overfit.yaml

# Phase-sampling variant of the overfit smoke.
python scripts/overfit_flower.py --config configs/train/overfit_phase.yaml

# Full task-1 fine-tune via SLURM.
sbatch scripts/train_flower.slurm configs/train/full_eval1.yaml
# task-2 and task-3:
sbatch scripts/train_flower.slurm configs/train/full_eval2.yaml
sbatch scripts/train_flower.slurm configs/train/full_eval3.yaml
```

Configs share schema with the SmolVLA repo (`experiment_name`, `data`,
`model`, `train.{batch_size,lr,warmup_steps,phase_sampling,early_stop}`,
`hf`, `logging`). Only the `model.*` block differs (Florence-2 specifics:
`vlm_path`, `freeze_florence`, `dit_dim`, `num_sampling_steps`,
`default_action_type: 3` for SO-101).

## Run offline eval

```bash
python scripts/eval_offline_flower.py \
    --checkpoint <ckpt dir or HF id> \
    --dataset ethrl2026/task1_20260509_plus \
    --out reports/eval1_step20000.json
```

Tests A (open-loop MAE) + C (OOD prompts). Test B (prompt equivalence) is
not run — it needs the SmolVLA-only PromptAugmentingDataset color pool.

## Run inference (real robot or dry-run)

```bash
# Dry-run smoke (no robot connected).
python scripts/run_inference_flower.py \
    --checkpoint <ckpt> \
    --prompt "Put the banana in the blue colored bowl." \
    --max-seconds 5 \
    --dry-run

# Real-robot rollout.
python scripts/run_inference_flower.py \
    --checkpoint <ckpt> \
    --prompt "Put the banana in the blue colored bowl." \
    --max-seconds 20
```

## Probe Testing
Test the model's understanding of color and spatial language by running inference on a frame in which the arm is grasping the banana and to see if the predicted actions are consistent with the prompts. This can help identify if the model has learned to associate certain colors or spatial positions with the correct actions.
```bash
python scripts/probe_testing_flower.py   --checkpoint ethrl2026/so101-eval2-flower-reasoning-enhanced   --dataset ethrl2026/task2_20260509_stage2_random_lighting_augmented_2160   --episode 12   --frame 258   --prompt "Put the banana in the blue colored bowl."   --prompt "Put the banana in the red colored bowl."   --prompt "Put the banana in the green colored bowl."   --prompt "Put the banana in the left bowl."   --prompt "Put the banana in the middle bowl."   --prompt "Put the banana in the right bowl."
```

## Data layout

The HuggingFace dataset snapshots are **not** in this repo. The configs
default to `root: null`, which triggers `snapshot_download` from the HF
Hub on first run and caches to `~/.cache/huggingface/`.

If multiple jobs launch concurrently and you start seeing 429s, pre-cache
the snapshot once and point `root` at it:

```bash
huggingface-cli download --repo-type dataset --revision v3.0 \
    ethrl2026/task1_20260509_plus \
    --local-dir /shares/feldmann.ics.mnf.uzh/Yuqi/Lerobot/data/hf/task1_20260509_plus
```

```yaml
# configs/train/full_eval1.yaml
data:
  root: /shares/feldmann.ics.mnf.uzh/Yuqi/Lerobot/data/hf/task1_20260509_plus
```

## Phase-weighted sampling

Per-frame `pre_grasp` / `post_grasp` labels from the gripper signal,
upweighting pre-grasp frames in a `WeightedRandomSampler`. Single config
knob in `train.phase_sampling.enabled`. Default `enabled: true,
weight_pregrasp: 2.0` across all 5 configs. Implementation at
[src/data/phase_labels.py](src/data/phase_labels.py) +
[src/data/sampler.py](src/data/sampler.py) — same code as in the
SmolVLA repo (copied at split time, will diverge as needed).

## Vendored upstream

| Path | Source | Patches |
|---|---|---|
| `third_party/flower_vla/` | `flower_calvin` upstream | F1 (224 resize), F2 (`obs_modalities="state_obs"`), F3 (FreqEmbedder 1-D input), F4 (`so101` action type 3, dim 6), F5 (`default_action_type` attribute), F6 (SDPA: drop illegal `is_causal=True` + explicit `attn_mask` combo so DiT runs on torch MPS/CPU backends), F7 (`so101` real proprio MLP — encode joint state; upstream ZeroEncoder'd it even with `use_proprio=True`), F8 (`encode_proprio`: collapse 3-D `action_type` mask for 2-D proprio + bf16/fp32 `index_put` dtype fix) |
| `third_party/lerobot_hw/` | `lerobot 0.5.1` (subset) | PEP 695 `type X = ...` → plain assignment; all `from lerobot.X` rewritten to `from lerobot_hw.X` |

`lerobot_hw` is **lazy-imported** by
[src/flower/runner.py](src/flower/runner.py) — dry-run mode never touches
hardware deps (serial / opencv).

## Architecture & migration history

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the relationship to
the SmolVLA sibling repo, the F-patch series, and the shared training
skeleton. See [docs/flowervla_spike.md](docs/flowervla_spike.md) for the
original spike verdict.
