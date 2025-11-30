#!/usr/bin/env bash
#SBATCH -A cs551
#SBATCH -p academic
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -C A30
#SBATCH -t 24:00:00
#SBATCH --mem=12G
#SBATCH --job-name="PrivDiffuser_EMG"

source activate myenv

python scripts/emg_privdiffuser_with_overlay.py