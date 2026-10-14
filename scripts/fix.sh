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
cd /rhomes/gschum/jepa
conda activate vjepa_pretrain

CSV=outputs/vjepa_pretrain_human_cam2_bouts_2026-10-12/human_cam2_bouts_vjepa_vitl16_r0.csv
OUT=outputs/vjepa_pretrain_human_cam2_bouts_2026-10-12

# Keep the last contiguous block after the final header (the 195982 run)
python - <<'PY'
from pathlib import Path
p = Path("outputs/vjepa_pretrain_human_cam2_bouts_2026-10-12/human_cam2_bouts_vjepa_vitl16_r0.csv")
lines = p.read_text().splitlines()
# last header line index
hdr = max(i for i, L in enumerate(lines) if L.startswith("epoch,"))
clean = lines[hdr:]
backup = p.with_suffix(".csv.bak")
backup.write_text(p.read_text())
p.write_text("\n".join(clean) + "\n")
print(f"backed up -> {backup}; kept {len(clean)-1} data rows from header at line {hdr}")
PY

# Re-run post steps only
python scripts/plot_vjepa_train_metrics.py \
  --log-csv "$CSV" \
  --output "$OUT/train_metrics.png" \
  --title "V-JEPA pretrain (human_cam2_bouts_vjepa_vitl16)"

python scripts/export_embeddings_vjepa.py extract \
  --pretrain-config configs/vjepa/pretrain_vitl16_human_cam2_bouts.yaml \
  --index-csv trainCSV/human_cam2_bouts.csv \
  --raw-output "$OUT/embeddings_raw.npz" \
  --num-workers 4

python scripts/plot_vjepa_embeddings_tsne.py \
  --raw-npz "$OUT/embeddings_raw.npz" \
  --index-csv trainCSV/human_cam2_bouts.csv \
  --output "$OUT/embeddings_tsne.png" \
  --title "V-JEPA encoder t-SNE (human_cam2_bouts_vjepa_vitl16)"