#!/bin/bash
#SBATCH --job-name=floodlm-fullevent-ft
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=/users/admarkowitz/FloodLM/logs/slurm_%j.out
#SBATCH --error=/users/admarkowitz/FloodLM/logs/slurm_%j.err

# Fine-tune a trained h=64 checkpoint with expanding-horizon curriculum
# starting at eff_T=64 and growing to T_max, with random warm-start history.
#
# Usage:
#   sbatch slurm/submit_fullevent_finetune.sh --pretrain-from checkpoints/latest/Model_1_best.pt
#   SELECTED_MODEL=Model_2 sbatch slurm/submit_fullevent_finetune.sh --pretrain-from checkpoints/latest/Model_2_best.pt

mkdir -p /users/admarkowitz/FloodLM/logs
mkdir -p /users/admarkowitz/FloodLM/checkpoints/fullevent_finetune

source /opt/local/miniconda3/etc/profile.d/conda.sh
conda activate floodlm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SELECTED_MODEL="${SELECTED_MODEL:-Model_1}"

cd /users/admarkowitz/FloodLM

echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Model:         $SELECTED_MODEL"
echo "Extra args:    $@"
echo "Started:       $(date)"
echo ""

python -u src/fullevent/train.py --finetune --mixed-precision "$@"

echo ""
echo "Finished: $(date)"
