#!/bin/bash

################################################################################
# FloodLM Pipeline Script

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

header "FloodLM Inference and Submit Pipeline: Inference → Evaluate → Submit"

# ============================================================================
# Stage 1: Inference
# ============================================================================

header "Stage 2: Running Autoregressive Inference"

LOG_FILE="${LOG_DIR}/inference_${TIMESTAMP}.log"

# checkpoints/latest/ is a real directory maintained by train.py containing the
# most recent best checkpoint + normalizers for every model.
if [ -d "${PROJECT_DIR}/checkpoints/latest" ]; then
    CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/latest"
else
    CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"
    log_warn "No checkpoints/latest directory found — using ${CHECKPOINT_DIR}"
fi
log_info "Checkpoint dir for inference: ${CHECKPOINT_DIR}"

SUBMISSION_FILE="submission_${TIMESTAMP}.csv"

log_info "Running inference for all trained models..."
log_info "Output file: ${SUBMISSION_FILE}"
log_info "Log file: ${LOG_FILE}"
log_info "Command: ${PYTHON} src/autoregressive_inference.py --checkpoint-dir ${CHECKPOINT_DIR} --output ${SUBMISSION_FILE}"

if cd "${PROJECT_DIR}" && ${PYTHON} -u src/autoregressive_inference.py --checkpoint-dir "${CHECKPOINT_DIR}" --output "${SUBMISSION_FILE}" 2>&1 | tee "${LOG_FILE}"; then
    log_info "✓ Inference completed successfully"
    RESULTS_SUMMARY="${RESULTS_SUMMARY}
✓ Inference: Passed"

    if [ -f "${PROJECT_DIR}/${SUBMISSION_FILE}" ]; then
        SUBMISSION_SIZE=$(wc -l < "${PROJECT_DIR}/${SUBMISSION_FILE}")
        log_info "✓ Submission file created: ${SUBMISSION_FILE} (${SUBMISSION_SIZE} rows)"
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
# Stage 2: Evaluation (Calculate RMSE)
# ============================================================================

header "Stage 2: Calculating RMSE"

LOG_FILE="${LOG_DIR}/rmse_${TIMESTAMP}.log"

log_info "Calculating RMSE metrics..."
log_info "Log file: ${LOG_FILE}"
log_info "Command: ${PYTHON} kaggle/calculate_rmse.py ${SUBMISSION_FILE} submission_firsttry.csv"

if cd "${PROJECT_DIR}" && ${PYTHON} -u kaggle/calculate_rmse.py "${SUBMISSION_FILE}" submission_firsttry.csv 2>&1 | tee "${LOG_FILE}"; then
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

# ============================================================================
# Stage 3: Kaggle Submission
# ============================================================================

header "Stage 4: Submitting to Kaggle"

LOG_FILE="${LOG_DIR}/kaggle_${TIMESTAMP}.log"

SUBMISSION_MSG="${KAGGLE_MESSAGE:-FloodLM ${TIMESTAMP}}"

log_info "Submitting ${SUBMISSION_FILE} to Kaggle..."
log_info "Message: ${SUBMISSION_MSG}"
log_info "Log file: ${LOG_FILE}"

if [ -f "${PROJECT_DIR}/${SUBMISSION_FILE}" ]; then
    if cd "${PROJECT_DIR}" && ${PYTHON} -u kaggle/submit_to_kaggle.py "${SUBMISSION_FILE}" \
            --message "${SUBMISSION_MSG}" --yes 2>&1 | tee "${LOG_FILE}"; then
        log_info "✓ Kaggle submission completed"
        RESULTS_SUMMARY="${RESULTS_SUMMARY}
✓ Kaggle Submission: Passed"
    else
        KAGGLE_EXIT_CODE=$?
        log_error "✗ Kaggle submission failed with exit code ${KAGGLE_EXIT_CODE}"
        log_error "Check log: ${LOG_FILE}"
        OVERALL_SUCCESS=1
        RESULTS_SUMMARY="${RESULTS_SUMMARY}
✗ Kaggle Submission: Failed (exit code ${KAGGLE_EXIT_CODE})"
    fi
else
    log_warn "${SUBMISSION_FILE} not found — skipping Kaggle submission"
    RESULTS_SUMMARY="${RESULTS_SUMMARY}
⚠ Kaggle Submission: Skipped (no ${SUBMISSION_FILE})"
fi

echo ""

################################################################################
# Summary
################################################################################

header "Pipeline Complete - Summary"

log_info "Pipeline stages executed:"
log_info "  1. Training: ${MODELS[*]}"
log_info "  2. Inference: All models combined (checkpoints/latest)"
log_info "  3. Evaluation: RMSE calculation"
log_info "  4. Kaggle Submission"
log_info ""
log_info "Results:"
echo "${RESULTS_SUMMARY}"

if [ ${OVERALL_SUCCESS} -eq 0 ]; then
    log_info "✓ All pipeline stages passed!"
    log_info "Output files:"
    [ -f "${PROJECT_DIR}/${SUBMISSION_FILE}" ] && log_info "  - ${SUBMISSION_FILE}"
else
    log_error "✗ Some pipeline stages failed"
fi

log_info ""
log_info "Log files saved to: ${LOG_DIR}/"
log_info "View logs with: tail -f ${LOG_DIR}/*.log"

exit ${OVERALL_SUCCESS}


