#!/bin/bash
#SBATCH --job-name=spiked-emdn-profile-spiked-emdn
#SBATCH --partition=general
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/net/scratch/ntebaldi/cebmf-data/logs/spiked_emdn_profile_spiked_emdn_%j.out
#SBATCH --error=/net/scratch/ntebaldi/cebmf-data/logs/spiked_emdn_profile_spiked_emdn_%j.err

set -euo pipefail

# Paths
EXE=/net/scratch/ntebaldi/cebmf/cebmf_torch/.venv/bin/python
SCRIPT=/net/scratch/ntebaldi/cebmf/cebmf_torch/scripts/run_spiked_emdn.py
ENV_FILE=/net/scratch/ntebaldi/cebmf/cebmf_torch/scripts/.env.profile
PLOTS=/net/scratch/ntebaldi/cebmf-data/plots
PROFILE=/net/scratch/ntebaldi/cebmf-data/profiles

PRIOR_NAME=spiked_emdn

# Check GPU availability
nvidia-smi || true

# Execute script
$EXE $SCRIPT \
    --env-file $ENV_FILE \
    --prior-name $PRIOR_NAME \
    --profile-output-dir $PROFILE \
    --plots-output-dir $PLOTS
