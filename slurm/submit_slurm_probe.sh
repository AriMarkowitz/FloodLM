#!/bin/bash
#SBATCH --job-name=floodlm-probe
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=/users/admarkowitz/FloodLM/logs/slurm_%j.out
#SBATCH --error=/users/admarkowitz/FloodLM/logs/slurm_%j.err

# Quick probe for architecture iteration — works for Model_1 or Model_2.
# Always trains from scratch. Runs 8 epochs per the model's curriculum schedule.
# Does NOT touch checkpoints/latest/ — safe to run without clobbering the best model.
#
# Usage:
#   sbatch slurm/submit_slurm_probe.sh [Model_1|Model_2]   (default: Model_2)

MODEL="${1:-Model_2}"

if [[ "$MODEL" != "Model_1" && "$MODEL" != "Model_2" ]]; then
    echo "ERROR: Invalid model '$MODEL'. Must be Model_1 or Model_2."
    exit 1
fi

mkdir -p /users/admarkowitz/FloodLM/logs

source /opt/local/miniconda3/etc/profile.d/conda.sh
conda activate floodlm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /users/admarkowitz/FloodLM

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║       FloodLM ${MODEL} Architecture Probe (8 epochs)              ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Model:         $MODEL"
echo "Started:       $(date)"
echo "NOTE: checkpoints/latest/ will NOT be touched (--no-mirror-latest)"
echo ""

PROJECT_DIR="/users/admarkowitz/FloodLM"
PYTHON="python3"

log_info()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]  $1"; }
log_error() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2; }

log_info "Training ${MODEL} from scratch — 8 epochs, no-mirror-latest"

CMD="cd \"${PROJECT_DIR}\" && SELECTED_MODEL=\"${MODEL}\" ${PYTHON} -u src/train.py --mixed-precision --epochs 8 --learning-rate 1e-3 --no-mirror-latest"

log_info "Command: $CMD"
log_info ""

if eval "$CMD"; then
    log_info "Probe complete. Check wandb val loss vs baseline to decide if arch is worth full training."
    EXIT_CODE=0
else
    EXIT_CODE=$?
    log_error "Probe failed with exit code $EXIT_CODE"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
if [ $EXIT_CODE -eq 0 ]; then
    echo "║                    Probe finished successfully                     ║"
else
    echo "║                    Probe failed                                    ║"
fi
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Finished: $(date)"
echo ""

exit $EXIT_CODE
