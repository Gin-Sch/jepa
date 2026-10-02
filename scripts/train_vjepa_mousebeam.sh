#!/bin/bash
#SBATCH --job-name=TrainVJEPA
#SBATCH --output=logs/train_vjepa_mousebeam_%j.out
#SBATCH --error=logs/train_vjepa_mousebeam_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:h200:1

# Pretrain on mouse-disjoint vjepa_eval_train.csv, then run attentive probe
# (probe-train / probe-val on vjepa_eval_{train,val}.csv) against the new checkpoint.

set -euo pipefail

module load python3/anaconda3-2024.06-py3.12
module load nvidia-hpc-sdk
module load nvhpc-hpcx-cuda12/24.7

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TRAINING_ENV=server

JEPA_DIR="${JEPA_DIR:-/rhomes/gschum/jepa}"
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-${JEPA_DIR}/configs/vjepa/pretrain_vitl16_h200.yaml}"
EVAL_CONFIG="${EVAL_CONFIG:-${JEPA_DIR}/configs/vjepa/eval_attentive_probe_vitl16.yaml}"

conda activate vjepa_pretrain

cd "${JEPA_DIR}"
mkdir -p logs logs/vjepa_pretrain_heldout

python -m app.main \
  --fname "${PRETRAIN_CONFIG}" \
  --devices cuda:0

python -m evals.main \
  --fname "${EVAL_CONFIG}" \
  --devices cuda:0
