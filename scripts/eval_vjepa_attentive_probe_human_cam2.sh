#!/bin/bash
#SBATCH --job-name=EvalHumanCam2Probe
#SBATCH --output=logs/eval_vjepa_attentive_probe_human_cam2_%j.out
#SBATCH --error=logs/eval_vjepa_attentive_probe_human_cam2_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=99:00:00
#SBATCH --gres=gpu:a6000:1

# Attentive probe on the human_cam2 V-JEPA checkpoint.
# Ported from MouseBeamWalk's eval_vjepa_attentive_probe flow; points at
# outputs/vjepa_pretrain_human_cam2 and trainCSV/human_cam2.csv.
#
# Usage (cluster):
#   sbatch scripts/eval_vjepa_attentive_probe_human_cam2.sh
#
# Optional overrides:
#   INDEX_CSV=/path/to/human_cam2.csv \
#   PRETRAIN_FOLDER=/path/to/outputs/vjepa_pretrain_human_cam2 \
#   RUN_DIAGNOSTICS=1 \
#   sbatch scripts/eval_vjepa_attentive_probe_human_cam2.sh

set -euo pipefail

module load python3/anaconda3-2024.06-py3.12
module load nvidia-hpc-sdk
module load nvhpc-hpcx-cuda12/24.7

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export TRAINING_ENV=server

JEPA_DIR="${JEPA_DIR:-/rhomes/gschum/jepa}"
INDEX_CSV="${INDEX_CSV:-${JEPA_DIR}/trainCSV/human_cam2.csv}"
PRETRAIN_FOLDER="${PRETRAIN_FOLDER:-${JEPA_DIR}/outputs/vjepa_pretrain_human_cam2}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-human_cam2_vjepa_vitl16-latest.pth.tar}"
WRITE_TAG="${WRITE_TAG:-human_cam2_vjepa_vitl16}"
TRAIN_CSV="${TRAIN_CSV:-${JEPA_DIR}/trainCSV/human_cam2_eval_train.csv}"
VAL_CSV="${VAL_CSV:-${JEPA_DIR}/trainCSV/human_cam2_eval_val.csv}"
EVAL_CONFIG="${EVAL_CONFIG:-${JEPA_DIR}/configs/vjepa/eval_attentive_probe_human_cam2.yaml}"
VAL_FRACTION="${VAL_FRACTION:-0.2}"
SEED="${SEED:-0}"
GROUP_REGEX="${GROUP_REGEX:-^([0-9]+)}"
RUN_DIAGNOSTICS="${RUN_DIAGNOSTICS:-0}"

conda activate vjepa_pretrain

cd "${JEPA_DIR}"
export PYTHONPATH="${JEPA_DIR}:${PYTHONPATH:-}"
mkdir -p logs "${JEPA_DIR}/trainCSV" "${PRETRAIN_FOLDER}"

if [[ ! -f "${INDEX_CSV}" ]]; then
  echo "Index CSV not found: ${INDEX_CSV}" >&2
  exit 1
fi
if [[ ! -f "${PRETRAIN_FOLDER}/${CHECKPOINT_NAME}" ]]; then
  echo "Checkpoint not found: ${PRETRAIN_FOLDER}/${CHECKPOINT_NAME}" >&2
  exit 1
fi

echo "=== [1/3] Prepare train/val CSVs (subject-disjoint when possible) ==="
python scripts/prepare_vjepa_eval_split.py \
  --index-csv "${INDEX_CSV}" \
  --train-output "${TRAIN_CSV}" \
  --val-output "${VAL_CSV}" \
  --val-fraction "${VAL_FRACTION}" \
  --seed "${SEED}" \
  --group-regex "${GROUP_REGEX}" \
  --num-classes-file "${PRETRAIN_FOLDER}/human_cam2_num_classes.txt"

NUM_CLASSES="$(tr -d '[:space:]' < "${PRETRAIN_FOLDER}/human_cam2_num_classes.txt")"
if [[ -z "${NUM_CLASSES}" ]]; then
  echo "Failed to resolve num_classes from split helper." >&2
  exit 1
fi

echo "=== [2/3] Write resolved attentive-probe config (num_classes=${NUM_CLASSES}) ==="
RESOLVED_CONFIG="${PRETRAIN_FOLDER}/eval_attentive_probe_human_cam2.resolved.yaml"
python - "${EVAL_CONFIG}" "${RESOLVED_CONFIG}" \
  "${TRAIN_CSV}" "${VAL_CSV}" "${NUM_CLASSES}" \
  "${PRETRAIN_FOLDER}" "${CHECKPOINT_NAME}" "${WRITE_TAG}" <<'PY'
import sys
from pathlib import Path
import yaml

src, dst, train_csv, val_csv, num_classes, folder, ckpt, write_tag = sys.argv[1:]
with Path(src).open("r", encoding="utf-8") as handle:
    cfg = yaml.load(handle, Loader=yaml.FullLoader)

cfg["data"]["dataset_train"] = train_csv
cfg["data"]["dataset_val"] = val_csv
cfg["data"]["num_classes"] = int(num_classes)
cfg["pretrain"]["folder"] = folder
cfg["pretrain"]["checkpoint"] = ckpt
cfg["pretrain"]["write_tag"] = write_tag

dst_path = Path(dst)
dst_path.parent.mkdir(parents=True, exist_ok=True)
with dst_path.open("w", encoding="utf-8") as handle:
    yaml.dump(cfg, handle, default_flow_style=False, sort_keys=False)
print(f"Wrote {dst_path}")
PY

echo "=== [3/3] Run attentive probe ==="
python -m evals.main \
  --fname "${RESOLVED_CONFIG}" \
  --devices cuda:0

echo "Probe finished. Classifier checkpoint under:"
echo "  ${PRETRAIN_FOLDER}/video_classification_frozen/human-cam2-attn-probe-16x2x3/"

if [[ "${RUN_DIAGNOSTICS}" == "1" ]]; then
  echo "=== Diagnostics (confusion matrix + encoder t-SNE) ==="
  python scripts/plot_vjepa_probe_diagnostics.py \
    --eval-config "${RESOLVED_CONFIG}" \
    --split val \
    --device cuda:0
fi

echo "Done."
