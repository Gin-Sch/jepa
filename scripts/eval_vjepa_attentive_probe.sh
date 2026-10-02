#!/bin/bash
#SBATCH --job-name=EvalVJEPAProbe
#SBATCH --output=logs/eval_vjepa_attentive_probe_%j.out
#SBATCH --error=logs/eval_vjepa_attentive_probe_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:a6000:1

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
mkdir -p logs

python -m evals.main \
  --fname "${EVAL_CONFIG}" \
  --devices cuda:0
