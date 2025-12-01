#!/usr/bin/env bash
#SBATCH -p short
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH --mem=16G
#SBATCH --job-name="PPG_Seeingred_ID"

set -e

# 1) Export anonymized signals (expects data/ppg_anon.npz + data/ppg_meta.pkl)
python3 scripts/ppg_export_anonymized_csv.py

# 2) Run seeing-red feature pipeline on anonymized signals using the alternate root
export SEEINGRED_DATA_ROOT="$(pwd)/seeing-red/data_anon"
export SEEINGRED_ANON_ROOT="$SEEINGRED_DATA_ROOT"
export SEEINGRED_RAW_ROOT="$(pwd)/seeing-red/data"
cd "$(dirname "$0")/../seeing-red/code"
if [ ! -d "../data_anon/extracted" ]; then
  echo "Missing anonymized extracted dir at ../data_anon/extracted"
  exit 1
fi
python3 signal_preprocessor.py -d ../data_anon/extracted -o ../data_anon/preprocessed
python3 signal_beat_separation.py -d ../data_anon/preprocessed -o ../data_anon/beats
python3 signal_fiducial_points_detection.py -d ../data_anon/beats -o ../data_anon/fiducial_points
python3 fta_average_beat.py -o ../data_anon/beat_visuals
python3 signal_beat_fta.py -d ../data_anon/beats -o ../data_anon/beats-post-FTA
python3 feature_extractor.py -o ../data_anon/features
python3 feature_selection1.py -f ../data_anon/features -o ../data_anon/features-selected1
python3 feature_selection2.py -f ../data_anon/features-selected1 -o ../data_anon/features-selected2

# 3) Evaluate identity on anonymized features (20-beat aggregation)
cd ..
python3 ../scripts/ppg_seeingred_identity_eval.py
