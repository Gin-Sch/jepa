#!/bin/bash
#SBATCH --job-name=PlotVJEPAProbe
#SBATCH --output=logs/plot_vjepa_probe_diagnostics_%j.out
#SBATCH --error=logs/plot_vjepa_probe_diagnostics_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:a6000:1

# Post-hoc t-SNE + confusion matrix for the official attentive probe.
# Requires a finished probe run (classifier checkpoint under
# logs/vjepa_pretrain_heldout/video_classification_frozen/...).

set -euo pipefail

module load python3/anaconda3-2024.06-py3.12
module load nvidia-hpc-sdk
module load nvhpc-hpcx-cuda12/24.7

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TRAINING_ENV=server

JEPA_DIR="${JEPA_DIR:-/rhomes/gschum/jepa}"
EVAL_CONFIG="${EVAL_CONFIG:-${JEPA_DIR}/configs/vjepa/eval_attentive_probe_vitl16.yaml}"

conda activate vjepa_pretrain

cd "${JEPA_DIR}"
mkdir -p logs logs/vjepa_probe_diagnostics

export PYTHONPATH="${JEPA_DIR}:${PYTHONPATH:-}"

python scripts/plot_vjepa_probe_diagnostics.py \
  --eval-config "${EVAL_CONFIG}" \
  --split val \
  --device cuda:0 \
  "$@"
