#!/usr/bin/env bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH --mem=12G
#SBATCH --job-name="PrivDiffuser_EMG"
#SBATCH -C A100

python scripts/emg_privdiffuser_with_overlay.py