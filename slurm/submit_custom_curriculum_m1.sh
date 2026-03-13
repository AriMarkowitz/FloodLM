#!/bin/bash
#SBATCH --job-name=floodlm-m1-curriculum
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=18:00:00
#SBATCH --output=/users/admarkowitz/FloodLM/logs/slurm_%j.out
#SBATCH --error=/users/admarkowitz/FloodLM/logs/slurm_%j.err

# Train Model_1 from scratch or resume from a checkpoint with a custom horizon curriculum.
# Args:
#   $1 — checkpoint path, or "scratch" to train from scratch (required)
#   $2 — curriculum string, e.g. "8:2,16:4,24:4,32:4,48:4,64:4" (required)
#   $3 — learning rate (optional, default 1e-3)
#   $4 — no-mirror-latest flag: "no-mirror" to skip (optional)
#
# Examples:
#   sbatch slurm/submit_custom_curriculum_m1.sh scratch "1:2,2:2,4:2,8:2,16:4,32:4,64:4"
#   sbatch slurm/submit_custom_curriculum_m1.sh checkpoints/Model_1_XXXXXXXX/Model_1_best.pt "8:2,16:4,24:4,32:4,48:4,64:4"
#   sbatch slurm/submit_custom_curriculum_m1.sh checkpoints/Model_1_XXXXXXXX/Model_1_best.pt "8:2,16:4,32:4,64:4" 1e-3 no-mirror

CKPT="${1}"
CURRICULUM="${2}"
LR="${3:-1e-3}"
NO_MIRROR="${4:-}"

if [ -z "$CKPT" ] || [ -z "$CURRICULUM" ]; then
    echo "Usage: sbatch submit_custom_curriculum_m1.sh <checkpoint|scratch> <curriculum> [lr] [no-mirror]"
    exit 1
fi

if [ "$CKPT" = "scratch" ]; then
    RESUME_FLAG=""
else
    RESUME_FLAG="--resume ${CKPT}"
fi

mkdir -p /users/admarkowitz/FloodLM/logs

source /opt/local/miniconda3/etc/profile.d/conda.sh
conda activate floodlm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /users/admarkowitz/FloodLM

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     FloodLM Model_1 Custom Curriculum                             ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURMD_NODENAME"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "Started:     $(date)"
echo "Checkpoint:  $CKPT"
echo "Curriculum:  $CURRICULUM"
echo "LR:          $LR"
echo ""

PROJECT_DIR="/users/admarkowitz/FloodLM"
PYTHON="python3"

log_info()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]  $1"; }
log_error() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2; }

MIRROR_FLAG=""
if [ "$NO_MIRROR" = "no-mirror" ]; then
    MIRROR_FLAG="--no-mirror-latest"
    echo "NOTE: checkpoints/latest/ will NOT be touched (--no-mirror-latest)"
fi
echo ""

CMD="cd \"${PROJECT_DIR}\" && SELECTED_MODEL=\"Model_1\" ${PYTHON} -u src/train.py \
    ${RESUME_FLAG} \
    --mixed-precision \
    --learning-rate ${LR} \
    --curriculum \"${CURRICULUM}\" \
    ${MIRROR_FLAG}"

log_info "Command: $CMD"
log_info ""

if eval "$CMD"; then
    log_info "Training complete."
    EXIT_CODE=0
else
    EXIT_CODE=$?
    log_error "Training failed with exit code $EXIT_CODE"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
if [ $EXIT_CODE -eq 0 ]; then
    echo "║                    Finished successfully                           ║"
else
    echo "║                    Failed                                          ║"
fi
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Finished: $(date)"
echo ""

exit $EXIT_CODE
