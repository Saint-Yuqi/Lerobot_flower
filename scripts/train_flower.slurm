#!/bin/bash
#SBATCH --job-name=flower_train
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1                 # any GPU; pin (gpu:a100:1 / gpu:h100:1) only for big runs
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Usage:
#   sbatch scripts/train_flower.slurm <config.yaml> [<script.py>]
# Examples:
#   sbatch scripts/train_flower.slurm configs/train/overfit_flower.yaml scripts/overfit_flower.py
#   sbatch scripts/train_flower.slurm configs/train/full_eval1_flower.yaml scripts/train_flower.py
#
# This uses the `flower` conda env (Python 3.10, torch 2.2.2, transformers 4.46,
# pytorch-lightning 2.0.8). Lerobot is NOT installed in this env — the training
# stack reads dataset files directly with pyarrow + PyAV.

set -euo pipefail

CONFIG=${1:?"usage: sbatch train_flower.slurm <config.yaml> [<script.py>]"}
SCRIPT=${2:-scripts/train_flower.py}

# --- Conda env (flower-env-native) ---
source /apps/u24/opt/x86_64_v3/miniforge3-25.3.0-3-parbelp6xzi5ketwmb6iavf6xur4z55o/etc/profile.d/conda.sh
conda activate /shares/feldmann.ics.mnf.uzh/Yuqi/conda/envs/flower

# Ignore ~/.local/lib user-site (any torch installed there shadows the conda copy).
export PYTHONNOUSERSITE=1

# PyAV needs FFmpeg shared libs from the conda env, not the system ones.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# HF cache + pip cache on scratch.
export HF_HOME=/home/yuqyan/scratch/Lerobot/hf_cache
export PIP_CACHE_DIR=/home/yuqyan/scratch/Lerobot/pip_cache

# HF_TOKEN fallback for private repos on compute nodes.
if [ -z "${HF_TOKEN:-}" ] && [ -s "$HOME/.cache/huggingface/token" ]; then
    export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

mkdir -p logs

echo "=== job ${SLURM_JOB_ID:-local}  config=$CONFIG  script=$SCRIPT  host=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-?} ==="
nvidia-smi
python --version
python -c "import torch, transformers, pytorch_lightning, av; print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} transformers={transformers.__version__} pl={pytorch_lightning.__version__} av={av.__version__}')"

python -u "$SCRIPT" --config "$CONFIG"

echo "=== done ==="
