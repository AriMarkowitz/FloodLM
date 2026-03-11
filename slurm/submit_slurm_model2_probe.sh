#!/bin/bash
#SBATCH --job-name=floodlm-m2-probe
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output=/users/admarkowitz/FloodLM/logs/slurm_%j.out
#SBATCH --error=/users/admarkowitz/FloodLM/logs/slurm_%j.err

# Quick probe for Model_2 noise-injection curriculum.
# Always trains from scratch. Runs 10 epochs:
#   Epochs 1-2:  K=0 (10 clean warm-start, h=1 standard)
#   Epochs 3-4:  K=1 (10 clean + 1 perturbed → predict step 12)
#   Epochs 5-6:  K=2 (10 clean + 2 perturbed → predict step 13)
#   Epochs 7-8:  K=3 (10 clean + 3 perturbed → predict step 14)
#   Epochs 9-10: K=4 (10 clean + 4 perturbed → predict step 15)
# Does NOT touch checkpoints/latest/ — safe to run without clobbering the best model.
# Compare val loss trajectory vs baseline to decide if curriculum helps.
#
# Usage:
#   sbatch slurm/submit_slurm_model2_probe.sh

mkdir -p /users/admarkowitz/FloodLM/logs

source /opt/local/miniconda3/etc/profile.d/conda.sh
conda activate floodlm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /users/admarkowitz/FloodLM

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     FloodLM Model_2 Noise Curriculum Probe (10 epochs, K=0..4)    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURMD_NODENAME"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Started:       $(date)"
echo "NOTE: checkpoints/latest/ will NOT be touched (--no-mirror-latest)"
echo ""

PROJECT_DIR="/users/admarkowitz/FloodLM"
PYTHON="python3"

log_info()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]  $1"; }
log_error() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2; }

log_info "Training Model_2 from scratch — 10 epochs (noise curriculum K=0..4, 2 epochs/stage), no-mirror-latest"

CMD="cd \"${PROJECT_DIR}\" && SELECTED_MODEL=\"Model_2\" ${PYTHON} -u src/train.py --mixed-precision --epochs 10 --learning-rate 1e-3 --no-mirror-latest"

log_info "Command: $CMD"
log_info ""

if eval "$CMD"; then
    log_info "Probe complete. Check wandb noise_curriculum/* and val loss trajectory vs baseline."
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
