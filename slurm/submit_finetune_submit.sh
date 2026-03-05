#!/bin/bash
#SBATCH --job-name=floodlm-finetune
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/users/admarkowitz/FloodLM/logs/slurm_%j.out
#SBATCH --error=/users/admarkowitz/FloodLM/logs/slurm_%j.err

# Fine-tune both models on train+val data at h=64, then infer and submit to Kaggle.
#
# Usage:
#   sbatch slurm/submit_finetune_submit.sh
#
# Tune these env vars before submitting if desired:
#   FINETUNE_EPOCHS (default: 4)  — number of h=64 fine-tune epochs per model
#   FINETUNE_LR    (default: 3e-4) — learning rate for fine-tuning
#   KAGGLE_MESSAGE                 — custom submission message

mkdir -p /users/admarkowitz/FloodLM/logs

source /opt/local/miniconda3/etc/profile.d/conda.sh
conda activate floodlm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fine-tune hyperparameters (override defaults here if desired)
export FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-4}"
export FINETUNE_LR="${FINETUNE_LR:-3e-4}"
export KAGGLE_MESSAGE="${KAGGLE_MESSAGE:-FloodLM finetune+val $(date +%Y%m%d_%H%M%S)}"

cd /users/admarkowitz/FloodLM

echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Started:       $(date)"
echo "FINETUNE_EPOCHS: ${FINETUNE_EPOCHS}"
echo "FINETUNE_LR:     ${FINETUNE_LR}"
echo "KAGGLE_MESSAGE:  ${KAGGLE_MESSAGE}"
echo ""

bash run/pipeline_finetune_submit.sh auto all

echo ""
echo "Finished: $(date)"
