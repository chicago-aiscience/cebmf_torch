#!/bin/bash
#SBATCH --job-name=tiled-clustering
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/<data-path>/cebmf-data/logs/tiled_clustering_%j.out
#SBATCH --error=/<data-path>/cebmf-data/logs/tiled_clustering_%j.err

set -euo pipefail

# Paths
EXE=/<execution-path>/.venv/bin/python3
SCRIPT=/<execution-path>/scripts/run_tiled_clustering.py
OUT=/<data-path>/cebmf-data
PROFILE=/<data-path>/cebmf-data/profiles

# Check GPU availability
nvidia-smi || true

# Execute script
$EXE $SCRIPT \
    --out-dir $OUT \
    --profile-output-dir $PROFILE \
    --profile \
    --profile-iterations 1