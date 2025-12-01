#!/usr/bin/env bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH --mem=24G
#SBATCH --job-name="PPG_Blinder"

python3 scripts/ppg_blinder_with_quality.py
