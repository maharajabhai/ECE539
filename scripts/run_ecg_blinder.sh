#!/usr/bin/env bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH --job-name="ECG_BLINDER"
# Align GPU request with working PrivDiffuser config
#SBATCH --gres=gpu:1
#SBATCH -C V100|A100

set -e
cd "$(dirname "$0")/.."
module purge || true

echo "Starting ECG Blinder pipeline..."
python3 /home/simran/ECE539/scripts/ecg_blinder_pipeline_with_overlay.py

echo "Done."
