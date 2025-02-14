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

# Run your Python script
python3.9 Comparesetup1.py
