#!/bin/bash

################################################################################
# Resume Training Script
# Continues training from a checkpoint with optional modifications
# 
# Usage:
#   bash resume_training.sh <checkpoint_path> [options]
#   bash resume_training.sh checkpoints/latest/ --mixed-precision --max-h 48
#   bash resume_training.sh checkpoints/Model_2_20260303_003721/ --max-h 48
################################################################################

set -euo pipefail

# Configure CUDA memory management to avoid fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/logs"

# Default parameters
CHECKPOINT_PATH="${1:-checkpoints/latest/}"
MIXED_PRECISION=false
MAX_H=""
LEARNING_RATE=""

# Parse arguments
while [[ $# -gt 1 ]]; do
    case "$2" in
        --mixed-precision)
            MIXED_PRECISION=true
            shift
            ;;
        --max-h)
            MAX_H="$3"
            shift 2
            ;;
        --lr|--learning-rate)
            LEARNING_RATE="$3"
            shift 2
            ;;
        *)
            echo "Unknown option: $2"
            echo "Supported: --mixed-precision, --max-h, --learning-rate"
            exit 1
            ;;
    esac
done

# Helper functions
log_info() {
    local msg="$1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]  ${msg}"
}

log_error() {
    local msg="$1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] ${msg}" >&2
}

# Validate checkpoint
if [ ! -d "${PROJECT_DIR}/${CHECKPOINT_PATH}" ]; then
    log_error "Checkpoint directory not found: ${PROJECT_DIR}/${CHECKPOINT_PATH}"
    exit 1
fi

mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/resume_training_${TIMESTAMP}.log"

# Build command
CMD="cd \"${PROJECT_DIR}\" && ${PYTHON} -u src/train.py --resume \"${CHECKPOINT_PATH}\""

if [ "$MIXED_PRECISION" = true ]; then
    CMD="${CMD} --mixed-precision"
    log_info "✓ Mixed precision (float16) enabled — reduces GPU memory by ~40-50%"
fi

if [ -n "$MAX_H" ]; then
    CMD="${CMD} --max-h ${MAX_H}"
    log_info "Max horizon capped at: ${MAX_H}"
fi

if [ -n "$LEARNING_RATE" ]; then
    CMD="${CMD} --learning-rate ${LEARNING_RATE}"
    log_info "Learning rate: ${LEARNING_RATE}"
fi

log_info "Resume training from: ${PROJECT_DIR}/${CHECKPOINT_PATH}"
log_info "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (reduces fragmentation)"
log_info "Log file: ${LOG_FILE}"
log_info "Command: ${CMD}"
log_info ""
log_info "Starting training..."

if eval "${CMD}" 2>&1 | tee "${LOG_FILE}"; then
    log_info "✓ Training completed successfully"
    exit 0
else
    EXIT_CODE=$?
    log_error "✗ Training failed with exit code ${EXIT_CODE}"
    exit ${EXIT_CODE}
fi
