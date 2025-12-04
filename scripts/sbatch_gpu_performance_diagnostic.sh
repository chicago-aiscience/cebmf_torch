#!/bin/bash
#SBATCH --job-name=gpu-performance-diagnostic
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=/<data-path>/logs/gpu_performance_diagnostic_%j.out
#SBATCH --error=/<data-path>/logs/gpu_performance_diagnostic_%j.err

set -euo pipefail

# Paths
EXE=/<execution-path>/.venv/bin/python3
SCRIPT=/<execution-path>/scripts/gpu_performance_diagnostic.py
OUT=/<data-path>/benchmark

# Execute script
$EXE $SCRIPT $OUT