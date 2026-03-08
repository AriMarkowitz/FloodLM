#!/bin/bash
#SBATCH --job-name=floodlm-model3
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
#   sbatch slurm/submit_slurm_model3.sh                                     # resume from checkpoints/latest (default)
#   sbatch slurm/submit_slurm_model3.sh scratch                             # train from scratch
#   sbatch slurm/submit_slurm_model3.sh checkpoints/Model_3_20260308_XXX   # resume from specific dir
#

mkdir -p /users/admarkowitz/FloodLM/logs

source /opt/local/miniconda3/etc/profile.d/conda.sh
conda activate floodlm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /users/admarkowitz/FloodLM

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║         FloodLM Model_3 Training (Encoder-Decoder, AMP)           ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Started:       $(date)"
echo ""

PROJECT_DIR="/users/admarkowitz/FloodLM"
PYTHON="python3"

log_info()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]  $1"; }
log_error() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2; }

log_info "═════════════════════════════════════════════════════════════════"
RESUME_ARG="${1:-}"
if [ "${RESUME_ARG}" = "scratch" ]; then
    RESUME_FLAG=""
    log_info "Training Model_3 — from scratch (no resume)"
elif [ -n "${RESUME_ARG}" ]; then
    RESUME_FLAG="--resume ${RESUME_ARG}"
    log_info "Training Model_3 — resuming from: ${RESUME_ARG}"
else
    RESUME_FLAG="--resume checkpoints/latest"
    log_info "Training Model_3 — resuming from latest checkpoint"
fi
log_info "═════════════════════════════════════════════════════════════════"

CMD="cd \"${PROJECT_DIR}\" && ${PYTHON} -u src/model3/train.py ${RESUME_FLAG} --mixed-precision"

log_info "Command: $CMD"
log_info ""

if eval "$CMD"; then
    log_info "✓ Model_3 training completed successfully"
    EXIT_CODE=0
else
    EXIT_CODE=$?
    log_error "✗ Model_3 training failed with exit code $EXIT_CODE"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
if [ $EXIT_CODE -eq 0 ]; then
    echo "║                    ✓ Model_3 trained successfully                 ║"
else
    echo "║                    ✗ Model_3 training failed                      ║"
fi
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Finished: $(date)"
echo ""

exit $EXIT_CODE
