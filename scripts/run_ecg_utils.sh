#!/usr/bin/env bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --job-name="ECG_UTILS"
#SBATCH --gres=gpu:1

cd /home/simran/ECE539
export PYTHONPATH="$PWD/ecg_ptbxl_benchmarking/code:$PYTHONPATH"

echo "Starting ECG utility training (diagnosis/BPM)..."
python3 scripts/train_ecg_utils.py

echo "Done."
