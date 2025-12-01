#!/usr/bin/env bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --mem=24G
#SBATCH --job-name="PrivDiffuser_ECG"
#SBATCH -C A100

python scripts/ecg_privdiffuser_with_overlay.py
