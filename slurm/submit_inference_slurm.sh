#!/bin/bash
#SBATCH --job-name=floodlm
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=/users/admarkowitz/FloodLM/logs/slurm_%j.out
#SBATCH --error=/users/admarkowitz/FloodLM/logs/slurm_%j.err

# Usage:
#   sbatch submit_slurm.sh        # trains both Model_1 and Model_2

# Ensure log directory exists (must happen before SLURM tries to write output files,
# so run `mkdir -p /users/admarkowitz/FloodLM/logs` once before your first sbatch)
mkdir -p /users/admarkowitz/FloodLM/logs

# Activate conda environment
source /opt/local/miniconda3/etc/profile.d/conda.sh
conda activate floodlm

# Move to project root
cd /users/admarkowitz/FloodLM

echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Started:       $(date)"
echo ""

bash run/pipeline_inference.sh auto all

echo ""
echo "Finished: $(date)"
