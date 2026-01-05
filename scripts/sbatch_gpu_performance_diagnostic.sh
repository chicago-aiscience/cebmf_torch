#!/bin/bash
#SBATCH --job-name=gpu-performance-diagnostic
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/net/scratch/ntebaldi/cebmf-data/logs/gpu_performance_diagnostic_%j.out
#SBATCH --error=/net/scratch/ntebaldi/cebmf-data/logs/gpu_performance_diagnostic_%j.err

set -euo pipefail

# Paths
EXE=/net/scratch/ntebaldi/cebmf/cebmf_torch/.venv/bin/python3
SCRIPT=/net/scratch/ntebaldi/cebmf/cebmf_torch/scripts/gpu_performance_diagnostic.py
OUT=/net/scratch/ntebaldi/cebmf-data/benchmark

# Execute script
$EXE $SCRIPT $OUT