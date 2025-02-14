#!/bin/bash
# Job name:
#SBATCH --job-name=Marcenv
#
# Partition for a100 GPU:
#SBATCH --partition=gpua100
#
# Number of tasks (one for each GPU desired):
#SBATCH --ntasks=1
#
# Number of CPUs per task:
#SBATCH --cpus-per-task=4
#
# Number of GPUs:
#SBATCH --gres=gpu:1
#
# Wall clock limit:
#SBATCH --time=40:00:00
#
# Set environment variables for Hugging Face, temporary files, and PyTorch cache
export HF_HOME=/scratch/6538142/huggingface
export TRANSFORMERS_CACHE=/scratch/6538142/huggingface
export TMPDIR=/scratch/6538142/tmp
export TORCH_HOME=/scratch/6538142/torch

# Create directories if they don't exist
mkdir -p /scratch/6538142/huggingface
mkdir -p /scratch/6538142/tmp
mkdir -p /scratch/6538142/torch

# Verify environment variables (optional, for debugging)
echo "HF_HOME: $HF_HOME"
echo "TRANSFORMERS_CACHE: $TRANSFORMERS_CACHE"
echo "TMPDIR: $TMPDIR"
echo "TORCH_HOME: $TORCH_HOME"

## Command(s) to run:
python3.9 Setup1.py