#!/bin/bash
#SBATCH --job-name=floodlm-transfer
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/users/admarkowitz/FloodLM/logs/slurm_%j.out
#SBATCH --error=/users/admarkowitz/FloodLM/logs/slurm_%j.err

# Usage:
#   sbatch slurm/submit_slurm_transfer.sh
#
# Runs the full transfer learning pipeline:
#   1. Train Model_1 from scratch
#   2. Train Model_2 warm-started from Model_1's best h=64 checkpoint
#   3. Inference, RMSE evaluation, Kaggle submission

mkdir -p /users/admarkowitz/FloodLM/logs

source /opt/local/miniconda3/etc/profile.d/conda.sh
conda activate floodlm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /users/admarkowitz/FloodLM

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║              FloodLM Transfer Learning Pipeline                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID:   $SLURM_JOB_ID"
echo "Node:     $SLURMD_NODENAME"
echo "GPUs:     $CUDA_VISIBLE_DEVICES"
echo "Started:  $(date)"
echo ""

bash run/pipeline_transfer.sh auto

echo ""
echo "Finished: $(date)"
