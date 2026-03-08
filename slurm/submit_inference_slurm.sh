#!/bin/bash
#SBATCH --job-name=floodlm
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=/users/admarkowitz/FloodLM/logs/slurm_%j.out
#SBATCH --error=/users/admarkowitz/FloodLM/logs/slurm_%j.err

# Usage:
#   sbatch slurm/submit_inference_slurm.sh
#       # default: infer both models from checkpoints/latest, prefer best_h64
#
#   sbatch slurm/submit_inference_slurm.sh auto all \
#       --model1-dir checkpoints/Model_1_20260307_123456 \
#       --model2-dir checkpoints/Model_2_20260308_113327 \
#       --select best_h64
#       # use specific run dirs; pick best_h64 checkpoint from each
#
#   sbatch slurm/submit_inference_slurm.sh auto all --select val_loss
#       # scan all .pt files in checkpoints/latest and pick lowest val_loss per model

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

bash run/pipeline_inference.sh auto all "$@"

echo ""
echo "Finished: $(date)"
