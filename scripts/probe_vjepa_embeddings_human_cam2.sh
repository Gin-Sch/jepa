#!/bin/bash
#SBATCH --job-name=ProbeHumanEmb
#SBATCH --output=logs/probe_vjepa_embeddings_human_cam2_%j.out
#SBATCH --error=logs/probe_vjepa_embeddings_human_cam2_%j.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00

# Linear probe on already-exported human_cam2 embeddings (embeddings_raw.npz).
# No GPU / videos / attentive-probe checkpoint needed.
#
# Local:
#   conda activate myenv
#   bash scripts/probe_vjepa_embeddings_human_cam2.sh
#
# Cluster:
#   sbatch scripts/probe_vjepa_embeddings_human_cam2.sh

set -euo pipefail

JEPA_DIR="${JEPA_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RAW_NPZ="${RAW_NPZ:-${JEPA_DIR}/outputs/vjepa_pretrain_human_cam2/embeddings_raw.npz}"
OUT_DIR="${OUT_DIR:-${JEPA_DIR}/outputs/vjepa_pretrain_human_cam2/probe}"
INDEX_CSV="${INDEX_CSV:-${JEPA_DIR}/trainCSV/human_cam2.csv}"

cd "${JEPA_DIR}"
mkdir -p logs "${OUT_DIR}"

# Prefer myenv locally; fall back to whatever is active on the cluster.
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx myenv; then
    conda activate myenv
  elif conda env list | awk '{print $1}' | grep -qx myenv_mini; then
    conda activate myenv_mini
  fi
fi

EXTRA_ARGS=()
if [[ -f "${INDEX_CSV}" ]]; then
  EXTRA_ARGS+=(--index-csv "${INDEX_CSV}")
  echo "Using subject groups from ${INDEX_CSV}"
else
  echo "No index CSV at ${INDEX_CSV}; running video-level CV"
fi

python scripts/probe_vjepa_embeddings.py \
  --raw-npz "${RAW_NPZ}" \
  --output-dir "${OUT_DIR}" \
  "${EXTRA_ARGS[@]}" \
  "$@"

echo "Done. Results in ${OUT_DIR}"
