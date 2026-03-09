#!/bin/bash
#SBATCH --job-name=floodlm-coldstart
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=/users/admarkowitz/FloodLM/logs/slurm_%j.out
#SBATCH --error=/users/admarkowitz/FloodLM/logs/slurm_%j.err

# Cold-start fine-tuning: loads a chosen h64 checkpoint and fine-tunes exclusively on
# full-event rollouts from t=0 (matching inference conditions) for N shuffled passes.
#
# Usage:
#   sbatch slurm/submit_slurm_coldstart.sh [MODEL] [CHECKPOINT] [PASSES] [LR]
#
#   MODEL       Model_1 or Model_2 (default: Model_2)
#   CHECKPOINT  Path to checkpoint dir or .pt file (default: checkpoints/latest)
#   PASSES      Number of shuffled passes over the event pool (default: 500)
#   LR          Learning rate (default: 1e-4)
#
# Examples:
#   sbatch slurm/submit_slurm_coldstart.sh Model_2
#   sbatch slurm/submit_slurm_coldstart.sh Model_2 checkpoints/latest 500 1e-4
#   sbatch slurm/submit_slurm_coldstart.sh Model_2 checkpoints/Model_2_20260307_123456 200 3e-5
#   sbatch slurm/submit_slurm_coldstart.sh Model_1 checkpoints/latest/Model_1_best_h64.pt 500 1e-4

mkdir -p /users/admarkowitz/FloodLM/logs

source /opt/local/miniconda3/etc/profile.d/conda.sh
conda activate floodlm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /users/admarkowitz/FloodLM

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║           FloodLM Cold-Start Fine-Tuning (Mixed Precision)         ║"
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

log_info()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]  $1"; }
log_error() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2; }

MODEL="${1:-Model_2}"
CHECKPOINT="${2:-checkpoints/latest}"
PASSES="${3:-500}"
LR="${4:-1e-4}"

log_info "═════════════════════════════════════════════════════════════════"
log_info "Model:      ${MODEL}"
log_info "Checkpoint: ${CHECKPOINT}"
log_info "Passes:     ${PASSES}"
log_info "LR:         ${LR}"
log_info "═════════════════════════════════════════════════════════════════"

CMD="cd \"${PROJECT_DIR}\" && SELECTED_MODEL=\"${MODEL}\" ${PYTHON} -u src/train.py \
    --resume \"${CHECKPOINT}\" \
    --cold-start-passes ${PASSES} \
    --learning-rate ${LR} \
    --no-val \
    --mixed-precision"

log_info "Command: $CMD"
log_info ""

if eval "$CMD"; then
    log_info "✓ Cold-start fine-tuning completed successfully"
    EXIT_CODE=0
else
    EXIT_CODE=$?
    log_error "✗ Cold-start fine-tuning failed with exit code $EXIT_CODE"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
if [ $EXIT_CODE -eq 0 ]; then
    echo "║              ✓ Cold-start fine-tuning completed                    ║"
else
    echo "║              ✗ Cold-start fine-tuning failed                       ║"
fi
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Finished: $(date)"
echo ""

exit $EXIT_CODE
