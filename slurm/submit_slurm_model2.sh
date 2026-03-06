#!/bin/bash
#SBATCH --job-name=floodlm-model2
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
#   sbatch slurm/submit_slurm_model2.sh            # resume from checkpoints/latest (default)
#   sbatch slurm/submit_slurm_model2.sh scratch    # train from scratch (use after arch changes)
#
# Trains Model_2 only. Defaults to resume.
# Applies stability fixes: gradient clipping + LR reduction at curriculum jumps.
#

mkdir -p /users/admarkowitz/FloodLM/logs

source /opt/local/miniconda3/etc/profile.d/conda.sh
conda activate floodlm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /users/admarkowitz/FloodLM

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║              FloodLM Model_2 Training (Mixed Precision)            ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Started:       $(date)"
echo "Memory config: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
echo ""

PROJECT_DIR="/users/admarkowitz/FloodLM"
PYTHON="python3"

log_info() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]  $1"; }
log_error() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2; }

log_info "═════════════════════════════════════════════════════════════════"
RESUME_ARG="${1:-}"
if [ "${RESUME_ARG}" = "scratch" ]; then
    RESUME_FLAG=""
    log_info "Training Model_2 — from scratch (no resume)"
else
    RESUME_FLAG="--resume checkpoints/latest"
    log_info "Training Model_2 — resuming from latest checkpoint"
fi
log_info "═════════════════════════════════════════════════════════════════"

CMD="cd \"${PROJECT_DIR}\" && SELECTED_MODEL=\"Model_2\" ${PYTHON} -u src/train.py ${RESUME_FLAG} --mixed-precision"

log_info "Command: $CMD"
log_info ""

if eval "$CMD"; then
    log_info "✓ Model_2 training completed successfully"
    EXIT_CODE=0
else
    EXIT_CODE=$?
    log_error "✗ Model_2 training failed with exit code $EXIT_CODE"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
if [ $EXIT_CODE -eq 0 ]; then
    echo "║                    ✓ Model_2 trained successfully                 ║"
else
    echo "║                    ✗ Model_2 training failed                      ║"
fi
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Finished: $(date)"
echo ""

exit $EXIT_CODE
