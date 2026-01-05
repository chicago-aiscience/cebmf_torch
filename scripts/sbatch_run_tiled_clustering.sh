#!/bin/bash
#SBATCH --job-name=tiled-clustering
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/net/scratch/ntebaldi/cebmf-data/logs/tiled_clustering_%j.out
#SBATCH --error=/net/scratch/ntebaldi/cebmf-data/logs/tiled_clustering_%j.err

set -euo pipefail

# Paths
EXE=/net/scratch/ntebaldi/cebmf/cebmf_torch/.venv/bin/python3
SCRIPT=/net/scratch/ntebaldi/cebmf/cebmf_torch/scripts/run_tiled_clustering.py
OUT=/net/scratch/ntebaldi/cebmf-data
PROFILE=/net/scratch/ntebaldi/cebmf-data/profiles

# Check GPU availability
nvidia-smi || true

# Execute script
$EXE $SCRIPT \
    --out-dir $OUT \
    --profile-output-dir $PROFILE \
    --profile \
    --profile-iterations 1