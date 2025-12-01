#!/usr/bin/env bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH --mem=24G
#SBATCH --job-name="PPG_extract"

# Run the seeing-red extraction / preprocessing pipeline to produce PPG waveforms.
# Expects videos under data/videos and writes outputs into seeing-red/data/.

source /home/simran/ECE539/env_privacy/bin/activate
cd /home/simran/ECE539/seeing-red/code
python3 signal_run_all.py
