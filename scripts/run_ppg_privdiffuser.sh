#!/usr/bin/env bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH --mem=24G
#SBATCH --job-name="PPG_PrivDiffuser"

python3 scripts/ppg_privdiffuser_with_quality.py
