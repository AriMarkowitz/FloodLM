#!/bin/bash

################################################################################
# FloodLM Pipeline Script

################################################################################

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs"

# Arguments
GPU_ID="${1:-auto}"
MODEL_SELECTION="${2:-Model_1}"

# Validate model selection
case "${MODEL_SELECTION}" in
    Model_1|Model_2)
        MODELS=("${MODEL_SELECTION}")
        ;;
    all)
        MODELS=("Model_1" "Model_2")
        ;;
    *)
        echo "Invalid model selection: ${MODEL_SELECTION}"
        echo "Valid options: Model_1, Model_2, or all"
        exit 1
        ;;
esac

# Create log directory
mkdir -p "${LOG_DIR}"

################################################################################
# Helper Functions
################################################################################

log_info() {
    local msg="$1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]  ${msg}"
    if [ -n "${LOG_FILE:-}" ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]  ${msg}" >> "${LOG_FILE}"
    fi
}

log_warn() {
    local msg="$1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [WARN]  ${msg}"
    if [ -n "${LOG_FILE:-}" ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] [WARN]  ${msg}" >> "${LOG_FILE}"
    fi
}

log_error() {
    local msg="$1"
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] ${msg}" >&2
    if [ -n "${LOG_FILE:-}" ]; then
        echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] ${msg}" >> "${LOG_FILE}"
    fi
}

header() {
    local text="$1"
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  ${text}"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
}

update_data_config() {
    local MODEL_NAME="$1"
    export SELECTED_MODEL="${MODEL_NAME}"
    log_info "Using SELECTED_MODEL=${SELECTED_MODEL}"
}

################################################################################
# Initialization
################################################################################

header "FloodLM Training & Inference Pipeline"

log_info "Script directory: ${SCRIPT_DIR}"
log_info "GPU ID: ${GPU_ID}"
log_info "Models to train: ${MODELS[*]}"
log_info "Number of models: ${#MODELS[@]}"

# Check dependencies
log_info "Checking dependencies..."

if ! command -v "${PYTHON}" &> /dev/null; then
    log_error "Python not found: ${PYTHON}"
    exit 1
fi

PYTHON_VERSION=$(${PYTHON} --version 2>&1)
log_info "Python: ${PYTHON_VERSION}"

# List available GPUs
log_info "Available GPUs:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader 2>/dev/null | while read line; do
        log_info "  $line"
    done
else
    log_warn "  Could not query GPUs"
fi

################################################################################
# GPU Selection
################################################################################

log_info "Selecting GPU..."

if [ "${GPU_ID}" = "auto" ]; then
    log_info "Auto-selecting GPU with most free memory..."
    if command -v nvidia-smi &> /dev/null; then
        GPU_ID=$(${PYTHON} << 'EOF'
import subprocess
try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.free', 
         '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=5
    )
    lines = result.stdout.strip().split('\n')
    if lines and lines[0]:
        gpu_data = [(int(line.split(',')[0].strip()), 
                    int(line.split(',')[1].strip())) for line in lines if line.strip()]
        gpu_id = max(gpu_data, key=lambda x: x[1])[0]
        print(gpu_id)
    else:
        print(0)
except:
    print(0)
EOF
)
    else
        GPU_ID=0
    fi
    log_info "Selected GPU: ${GPU_ID}"
fi

################################################################################
# Environment Setup
################################################################################

log_info "Setting up environment..."

# Set GPU
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
log_info "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

# GPU optimization flags
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_LAUNCH_BLOCKING=0

# PyTorch specific optimizations
if command -v nproc &> /dev/null; then
    export OMP_NUM_THREADS=$(nproc)
elif command -v sysctl &> /dev/null; then
    export OMP_NUM_THREADS=$(sysctl -n hw.ncpu)
else
    export OMP_NUM_THREADS=4
fi
log_info "OMP_NUM_THREADS=${OMP_NUM_THREADS}"

################################################################################
# Run Training for Each Model
################################################################################

OVERALL_SUCCESS=0
RESULTS_SUMMARY=""

header "FloodLM Full Pipeline: Train → Inference → Evaluate"

# ============================================================================
# Stage 1: Training
# ============================================================================

header "Stage 1: Training Models"

for MODEL in "${MODELS[@]}"; do
    LOG_FILE="${LOG_DIR}/train_${MODEL}_${TIMESTAMP}.log"
    
    log_info "Training ${MODEL}..."
    log_info "Log file: ${LOG_FILE}"
    log_info "Command: SELECTED_MODEL=${MODEL} ${PYTHON} src/train.py"
    
    if cd "${SCRIPT_DIR}" && SELECTED_MODEL="${MODEL}" ${PYTHON} -u src/train.py 2>&1 | tee "${LOG_FILE}"; then
        log_info "✓ ${MODEL} training completed successfully"
        RESULTS_SUMMARY="${RESULTS_SUMMARY}
✓ ${MODEL}: Training passed"
    else
        TRAIN_EXIT_CODE=$?
        log_error "✗ ${MODEL} training failed with exit code ${TRAIN_EXIT_CODE}"
        log_error "Check log: ${LOG_FILE}"
        OVERALL_SUCCESS=1
        RESULTS_SUMMARY="${RESULTS_SUMMARY}
✗ ${MODEL}: Training failed (exit code ${TRAIN_EXIT_CODE})"
    fi
    
    echo ""
done

# ============================================================================
# Stage 2: Inference
# ============================================================================

header "Stage 2: Running Autoregressive Inference"

LOG_FILE="${LOG_DIR}/inference_${TIMESTAMP}.log"

log_info "Running inference for all trained models..."
log_info "Log file: ${LOG_FILE}"
log_info "Command: ${PYTHON} src/autoregressive_inference.py --checkpoint-dir checkpoints --output submission.csv"

if cd "${SCRIPT_DIR}" && ${PYTHON} -u src/autoregressive_inference.py --checkpoint-dir checkpoints --output submission.csv 2>&1 | tee "${LOG_FILE}"; then
    log_info "✓ Inference completed successfully"
    RESULTS_SUMMARY="${RESULTS_SUMMARY}
✓ Inference: Passed"
    
    # Check if submission file was created
    if [ -f "${SCRIPT_DIR}/submission.csv" ]; then
        SUBMISSION_SIZE=$(wc -l < "${SCRIPT_DIR}/submission.csv")
        log_info "✓ Submission file created: submission.csv (${SUBMISSION_SIZE} rows)"
    fi
else
    INFERENCE_EXIT_CODE=$?
    log_error "✗ Inference failed with exit code ${INFERENCE_EXIT_CODE}"
    log_error "Check log: ${LOG_FILE}"
    OVERALL_SUCCESS=1
    RESULTS_SUMMARY="${RESULTS_SUMMARY}
✗ Inference: Failed (exit code ${INFERENCE_EXIT_CODE})"
fi

echo ""

# ============================================================================
# Stage 3: Evaluation (Calculate RMSE)
# ============================================================================

header "Stage 3: Calculating RMSE"

LOG_FILE="${LOG_DIR}/rmse_${TIMESTAMP}.log"

log_info "Calculating RMSE metrics..."
log_info "Log file: ${LOG_FILE}"
log_info "Command: ${PYTHON} calculate_rmse.py"

if cd "${SCRIPT_DIR}" && ${PYTHON} -u calculate_rmse.py 2>&1 | tee "${LOG_FILE}"; then
    log_info "✓ RMSE calculation completed successfully"
    RESULTS_SUMMARY="${RESULTS_SUMMARY}
✓ RMSE Calculation: Passed"
else
    RMSE_EXIT_CODE=$?
    log_error "✗ RMSE calculation failed with exit code ${RMSE_EXIT_CODE}"
    log_error "Check log: ${LOG_FILE}"
    OVERALL_SUCCESS=1
    RESULTS_SUMMARY="${RESULTS_SUMMARY}
✗ RMSE Calculation: Failed (exit code ${RMSE_EXIT_CODE})"
fi

echo ""

################################################################################
# Summary
################################################################################

header "Pipeline Complete - Summary"

log_info "Pipeline stages executed:"
log_info "  1. Training: ${MODELS[*]}"
log_info "  2. Inference: All models combined"
log_info "  3. Evaluation: RMSE calculation"
log_info ""
log_info "Results:"
echo "${RESULTS_SUMMARY}"

if [ ${OVERALL_SUCCESS} -eq 0 ]; then
    log_info "✓ All pipeline stages passed!"
    log_info "Output files:"
    [ -f "${SCRIPT_DIR}/submission.csv" ] && log_info "  - submission.csv"
else
    log_error "✗ Some pipeline stages failed"
fi

log_info ""
log_info "Log files saved to: ${LOG_DIR}/"
log_info "View logs with: tail -f ${LOG_DIR}/*.log"

exit ${OVERALL_SUCCESS}


