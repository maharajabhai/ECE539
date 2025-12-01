#!/usr/bin/env bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --job-name="ECG_UTILS"
#SBATCH --gres=gpu:1

set -e

cd "$(dirname "$0")/.."
module purge || true

echo "Starting ECG utility training (diagnosis/BPM)..."
python3 scripts/train_ecg_utils.py

echo "Done."
