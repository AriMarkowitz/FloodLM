#!/bin/bash
#SBATCH --job-name=floodlm-extend
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
#   sbatch slurm/submit_slurm_extend.sh <Model_1|Model_2> <n_epochs> [checkpoint_dir]
#
#   <Model_1|Model_2>  — which model to extend
#   <n_epochs>         — number of additional epochs to run at h=64
#   [checkpoint_dir]   — optional: path to checkpoint dir (default: checkpoints/latest)
#
# Examples:
#   sbatch slurm/submit_slurm_extend.sh Model_2 4
#   sbatch slurm/submit_slurm_extend.sh Model_2 6 checkpoints/Model_2_20260307_203050
#   sbatch slurm/submit_slurm_extend.sh Model_1 4
#

mkdir -p /users/admarkowitz/FloodLM/logs

source /opt/local/miniconda3/etc/profile.d/conda.sh
conda activate floodlm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /users/admarkowitz/FloodLM

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║              FloodLM Extended Training (Mixed Precision)           ║"
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

MODEL_ARG="${1:-}"
N_EPOCHS="${2:-}"
RESUME_ARG="${3:-checkpoints/latest}"

# Validate args
if [ -z "${MODEL_ARG}" ] || [ -z "${N_EPOCHS}" ]; then
    log_error "Usage: sbatch slurm/submit_slurm_extend.sh <Model_1|Model_2> <n_epochs> [checkpoint_dir]"
    exit 1
fi

if [ "${MODEL_ARG}" != "Model_1" ] && [ "${MODEL_ARG}" != "Model_2" ]; then
    log_error "MODEL must be Model_1 or Model_2, got: ${MODEL_ARG}"
    exit 1
fi

if ! [[ "${N_EPOCHS}" =~ ^[0-9]+$ ]] || [ "${N_EPOCHS}" -lt 1 ]; then
    log_error "n_epochs must be a positive integer, got: ${N_EPOCHS}"
    exit 1
fi

log_info "═════════════════════════════════════════════════════════════════"
log_info "Model:         ${MODEL_ARG}"
log_info "Extra epochs:  ${N_EPOCHS} (all at h=64)"
log_info "Resume from:   ${RESUME_ARG}"
log_info "═════════════════════════════════════════════════════════════════"

CMD="cd \"${PROJECT_DIR}\" && SELECTED_MODEL=\"${MODEL_ARG}\" ${PYTHON} -u src/train.py \
    --resume ${RESUME_ARG} \
    --epochs ${N_EPOCHS} \
    --max-h 64 \
    --mixed-precision"

log_info "Command: $CMD"
log_info ""

if eval "$CMD"; then
    log_info "✓ ${MODEL_ARG} extended training completed successfully"
    EXIT_CODE=0
else
    EXIT_CODE=$?
    log_error "✗ ${MODEL_ARG} extended training failed with exit code $EXIT_CODE"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
if [ $EXIT_CODE -eq 0 ]; then
    echo "║                 ✓ Extended training completed                      ║"
else
    echo "║                 ✗ Extended training failed                         ║"
fi
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Finished: $(date)"
echo ""

exit $EXIT_CODE
