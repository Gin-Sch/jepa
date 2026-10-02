#!/bin/bash
#SBATCH --job-name=ExportEmbVJEPA
#SBATCH --output=logs/export_emb_vjepa_%j.out
#SBATCH --error=logs/export_emb_vjepa_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a6000:1

set -euo pipefail

module load python3/anaconda3-2024.06-py3.12
module load nvidia-hpc-sdk
module load nvhpc-hpcx-cuda12/24.7

export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Frozen self-pretrained V-JEPA ViT-L/16 -> one embedding per video.
#
# Stages 1 and 3 need myenv_mini + MouseBeamWalk helpers (MBW_DIR).
# Stage 2 needs vjepa_pretrain and this jepa checkout as CWD.

JEPA_DIR="${JEPA_DIR:-/rhomes/gschum/jepa}"
MBW_DIR="${MBW_DIR:-/rhomes/gschum/MouseBeamWalk}"
export MBW_DIR
MANIFEST="${MANIFEST:-${MBW_DIR}/manifests/manifest.jsonl}"
INDEX_CSV="${INDEX_CSV:-${MBW_DIR}/manifests/vjepa_index_camera6.csv}"
RAW_NPZ="${RAW_NPZ:-${MBW_DIR}/outputs/embeddings/vjepa_raw_camera6.npz}"
OUT_PREFIX="${OUT_PREFIX:-${MBW_DIR}/outputs/embeddings/vjepa_camera6}"
EVAL_CONFIG="${EVAL_CONFIG:-${JEPA_DIR}/configs/vjepa/eval_attentive_probe_vitl16.yaml}"

mkdir -p "${JEPA_DIR}/logs" "${MBW_DIR}/outputs/embeddings"

# Stage 1: index CSV labelled by manifest row index (keeps features joinable).
conda activate myenv_mini
cd "${JEPA_DIR}"
python scripts/export_embeddings_vjepa.py write-index \
  --manifest-path "${MANIFEST}" \
  --index-csv "${INDEX_CSV}" \
  --camera 6

# Stage 2: frozen encoder forward pass.
conda activate vjepa_pretrain
cd "${JEPA_DIR}"
export PYTHONPATH="${JEPA_DIR}:${PYTHONPATH:-}"
python scripts/export_embeddings_vjepa.py extract \
  --eval-config "${EVAL_CONFIG}" \
  --index-csv "${INDEX_CSV}" \
  --raw-output "${RAW_NPZ}" \
  --num-workers 4 \
  "$@"

# Stage 3: join manifest metadata into the embedding table.
conda activate myenv_mini
cd "${JEPA_DIR}"
python scripts/export_embeddings_vjepa.py finalize \
  --raw-input "${RAW_NPZ}" \
  --index-csv "${INDEX_CSV}" \
  --manifest-path "${MANIFEST}" \
  --output-prefix "${OUT_PREFIX}" \
  --model-name "V-JEPA ViT-L/16 (ours)"
