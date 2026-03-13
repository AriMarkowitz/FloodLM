#!/bin/bash
#SBATCH --job-name=floodlm-ni-probe
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/users/admarkowitz/FloodLM/logs/slurm_%j.out
#SBATCH --error=/users/admarkowitz/FloodLM/logs/slurm_%j.err

# Noise-injection training probe for Model_2.
# Incremental curriculum: K advances by k_per_advance every epochs_per_k epochs.
# At each K, collects AR noise stats from the current model, then trains with
# noise-injected GT inputs.
#
# Usage:
#   sbatch slurm/submit_noise_injection.sh <pretrained_ckpt> [epochs_per_k] [k_per_advance] [epochs]
#
# Examples:
#   sbatch slurm/submit_noise_injection.sh checkpoints/latest/Model_2_best.pt             # defaults: 1 epoch/K, +1 K/advance, 20 epochs
#   sbatch slurm/submit_noise_injection.sh checkpoints/latest/Model_2_best.pt 2 1 40      # 2 epochs per K, +1 per advance, 40 epochs
#   sbatch slurm/submit_noise_injection.sh checkpoints/latest/Model_2_best.pt auto 1 100  # auto-advance on convergence, +1 per advance, 100 epochs
#   sbatch slurm/submit_noise_injection.sh checkpoints/latest/Model_2_best.pt auto 5 200  # auto + jump 5 K per advance

mkdir -p /users/admarkowitz/FloodLM/logs

source /opt/local/miniconda3/etc/profile.d/conda.sh
conda activate floodlm

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /users/admarkowitz/FloodLM

PRETRAINED="${1:?Usage: sbatch submit_noise_injection.sh <pretrained_ckpt> [epochs_per_k] [k_per_advance] [epochs]}"
EPOCHS_PER_K="${2:-1}"
K_PER_ADVANCE="${3:-1}"
EPOCHS="${4:-20}"

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║     FloodLM Model_2 Noise-Injection Probe                        ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Job ID:         $SLURM_JOB_ID"
echo "Node:           $SLURMD_NODENAME"
echo "GPUs:           $CUDA_VISIBLE_DEVICES"
echo "Started:        $(date)"
echo "Pretrained:     ${PRETRAINED}"
echo "epochs_per_k:   ${EPOCHS_PER_K}"
echo "k_per_advance:  ${K_PER_ADVANCE}"
echo "epochs:         ${EPOCHS}"
echo "NOTE: checkpoints/latest/ will NOT be touched (--no-mirror-latest)"
echo ""

log_info()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]  $1"; }
log_error() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2; }

CMD="SELECTED_MODEL=\"Model_2\" python3 -u src/train_noise_injection.py \
    --pretrained ${PRETRAINED} \
    --mixed-precision \
    --no-mirror-latest \
    --epochs ${EPOCHS} \
    --epochs-per-k ${EPOCHS_PER_K} \
    --k-per-advance ${K_PER_ADVANCE}"

log_info "Command: $CMD"

if eval "$CMD"; then
    EXIT_CODE=0
    log_info "Probe complete. Check wandb train/loss and train/K trajectories."
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

exit $EXIT_CODE
