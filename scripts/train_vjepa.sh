#!/bin/bash
#SBATCH --job-name=TrainVJEPA
#SBATCH --output=logs/train_vjepa_%j.out
#SBATCH --error=logs/train_vjepa_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:h200:1

# Pretrain V-JEPA from a user YAML+CSV, plot training metrics, extract frozen
# encoder embeddings, and save a label-colored t-SNE. No attentive probe.
#
# Usage:
#   PRETRAIN_CONFIG=configs/vjepa/pretrain_vitl16_h200.yaml \
#   INDEX_CSV=/path/to/videos.csv \
#   sbatch scripts/train_vjepa.sh
#
# INDEX_CSV defaults to the first entry under data.datasets in PRETRAIN_CONFIG.
# OUT_DIR defaults to logging.folder from PRETRAIN_CONFIG.

set -euo pipefail

module load python3/anaconda3-2024.06-py3.12
module load nvidia-hpc-sdk
module load nvhpc-hpcx-cuda12/24.7

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TRAINING_ENV=server

JEPA_DIR="${JEPA_DIR:-/rhomes/gschum/jepa}"
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-${JEPA_DIR}/configs/vjepa/pretrain_vitl16_h200.yaml}"

conda activate vjepa_pretrain

cd "${JEPA_DIR}"
export PYTHONPATH="${JEPA_DIR}:${PYTHONPATH:-}"
mkdir -p "${JEPA_DIR}/logs"

# Resolve logging.folder, write_tag, and default INDEX_CSV from the pretrain YAML.
eval "$(
python - "${PRETRAIN_CONFIG}" <<'PY'
import sys
from pathlib import Path
import yaml

cfg_path = Path(sys.argv[1])
with cfg_path.open("r", encoding="utf-8") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

folder = cfg["logging"]["folder"]
write_tag = cfg["logging"].get("write_tag", "jepa")
datasets = cfg.get("data", {}).get("datasets") or []
index_csv = datasets[0] if datasets else ""

def sh_quote(value: str) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"

print(f"RESOLVED_OUT_DIR={sh_quote(folder)}")
print(f"RESOLVED_WRITE_TAG={sh_quote(write_tag)}")
print(f"RESOLVED_INDEX_CSV={sh_quote(index_csv)}")
PY
)"

OUT_DIR="${OUT_DIR:-${RESOLVED_OUT_DIR}}"
WRITE_TAG="${WRITE_TAG:-${RESOLVED_WRITE_TAG}}"
INDEX_CSV="${INDEX_CSV:-${RESOLVED_INDEX_CSV}}"

if [[ -z "${INDEX_CSV}" ]]; then
  echo "INDEX_CSV is empty; set INDEX_CSV or data.datasets in ${PRETRAIN_CONFIG}" >&2
  exit 1
fi

LOG_CSV="${LOG_CSV:-${OUT_DIR}/${WRITE_TAG}_r0.csv}"
RAW_NPZ="${RAW_NPZ:-${OUT_DIR}/embeddings_raw.npz}"
METRICS_PNG="${METRICS_PNG:-${OUT_DIR}/train_metrics.png}"
TSNE_PNG="${TSNE_PNG:-${OUT_DIR}/embeddings_tsne.png}"

mkdir -p "${OUT_DIR}"

echo "=== [1/4] Pretrain ==="
echo "config=${PRETRAIN_CONFIG}"
echo "out_dir=${OUT_DIR}"
echo "index_csv=${INDEX_CSV}"

python -m app.main \
  --fname "${PRETRAIN_CONFIG}" \
  --devices cuda:0

echo "=== [2/4] Plot training metrics ==="
python scripts/plot_vjepa_train_metrics.py \
  --log-csv "${LOG_CSV}" \
  --output "${METRICS_PNG}" \
  --title "V-JEPA pretrain (${WRITE_TAG})"

echo "=== [3/4] Extract frozen embeddings ==="
python scripts/export_embeddings_vjepa.py extract \
  --pretrain-config "${PRETRAIN_CONFIG}" \
  --index-csv "${INDEX_CSV}" \
  --raw-output "${RAW_NPZ}" \
  --num-workers 4

echo "=== [4/4] t-SNE of embeddings ==="
python scripts/plot_vjepa_embeddings_tsne.py \
  --raw-npz "${RAW_NPZ}" \
  --index-csv "${INDEX_CSV}" \
  --output "${TSNE_PNG}" \
  --title "V-JEPA encoder t-SNE (${WRITE_TAG})"

echo "Done."
echo "  checkpoint: ${OUT_DIR}/${WRITE_TAG}-latest.pth.tar"
echo "  metrics:    ${METRICS_PNG}"
echo "  embeddings: ${RAW_NPZ}"
echo "  t-SNE:      ${TSNE_PNG}"
