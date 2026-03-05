#!/bin/bash
#SBATCH --job-name=floodlm
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
#   sbatch slurm/submit_slurm.sh        # trains both Model_1 and Model_2 from scratch or resumes if interrupted
#
# For resuming from checkpoint with mixed precision (RECOMMENDED):
#   sbatch slurm/submit_slurm_resume.sh
#
# NOTE: Both scripts now use PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#       and mixed precision for better GPU memory management

# Ensure log directory exists (must happen before SLURM tries to write output files,
# so run `mkdir -p /users/admarkowitz/FloodLM/logs` once before your first sbatch)
mkdir -p /users/admarkowitz/FloodLM/logs

# Activate conda environment
source /opt/local/miniconda3/etc/profile.d/conda.sh
conda activate floodlm

# Configure CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Move to project root
cd /users/admarkowitz/FloodLM

echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Started:       $(date)"
echo ""

bash run/pipeline.sh auto all

echo ""
echo "Finished: $(date)"
