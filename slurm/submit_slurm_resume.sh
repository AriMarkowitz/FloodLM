#!/bin/bash
#SBATCH --job-name=floodlm-resume
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
#   sbatch slurm/submit_slurm_resume.sh
#
# Resumes training from the latest checkpoint for each model with mixed precision
# - Detects existing checkpoints in checkpoints/latest/
# - Resumes Model_1 and Model_2 with --mixed-precision to reduce GPU memory
# - If no checkpoint exists, trains from scratch
#

# Ensure log directory exists
mkdir -p /users/admarkowitz/FloodLM/logs

# Activate conda environment
source /opt/local/miniconda3/etc/profile.d/conda.sh
conda activate floodlm

# Configure CUDA memory management to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Move to project root
cd /users/admarkowitz/FloodLM

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                    FloodLM Resume Training (Mixed Precision)       ║"
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
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Helper functions
log_info() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]  $1"
}

log_error() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2
}

# Function to get the latest checkpoint for a model
get_latest_checkpoint() {
    local model_name="$1"
    local latest_dir="checkpoints/latest"
    
    # Check if latest directory has this model's best checkpoint
    if [ -f "${latest_dir}/${model_name}_best.pt" ]; then
        echo "${latest_dir}"
        return 0
    fi
    
    # Otherwise look for the most recent run directory for this model
    local recent=$(ls -td checkpoints/${model_name}_* 2>/dev/null | head -1)
    if [ -n "$recent" ] && [ -d "$recent" ]; then
        echo "$recent"
        return 0
    fi
    
    return 1
}

# Function to train a model
train_model() {
    local model_name="$1"
    local checkpoint_path="$2"
    
    log_info "═════════════════════════════════════════════════════════════════"
    log_info "Training $model_name"
    log_info "═════════════════════════════════════════════════════════════════"
    
    if [ -n "$checkpoint_path" ]; then
        log_info "Resuming from checkpoint: $checkpoint_path"
        log_info "Mixed precision: ENABLED"
        
        CMD="cd \"${PROJECT_DIR}\" && SELECTED_MODEL=\"${model_name}\" ${PYTHON} -u src/train.py --resume \"${checkpoint_path}\" --mixed-precision"
    else
        log_info "No checkpoint found. Training from scratch."
        log_info "Mixed precision: ENABLED"
        
        CMD="cd \"${PROJECT_DIR}\" && SELECTED_MODEL=\"${model_name}\" ${PYTHON} -u src/train.py --mixed-precision"
    fi
    
    log_info "Command: $CMD"
    log_info ""
    
    if eval "$CMD"; then
        log_info "✓ $model_name training completed successfully"
        return 0
    else
        local exit_code=$?
        log_error "✗ $model_name training failed with exit code $exit_code"
        return $exit_code
    fi
}

# Main training loop
overall_success=0

for model in Model_2 Model_1; do
    log_info ""
    
    # Get latest checkpoint for this model
    checkpoint=$(get_latest_checkpoint "$model" 2>/dev/null) || checkpoint=""
    
    # Train the model
    if train_model "$model" "$checkpoint"; then
        log_info "✓ $model completed"
    else
        log_error "✗ $model failed"
        overall_success=1
    fi
    
    log_info ""
done

echo ""
echo "╔════════════════════════════════════════════════════════════════════╗"
if [ $overall_success -eq 0 ]; then
    echo "║                    ✓ All models trained successfully             ║"
else
    echo "║                    ✗ Some models failed training                 ║"
fi
echo "╚════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Finished: $(date)"
echo ""

exit $overall_success
