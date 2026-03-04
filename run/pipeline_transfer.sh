#!/bin/bash

################################################################################
# FloodLM Transfer Learning Pipeline
#
# Same as pipeline.sh but Model_2 is warm-started from Model_1's best h=64
# checkpoint instead of random weights.
#
# Stage 1a: Train Model_1 from scratch
# Stage 1b: Train Model_2 using Model_1 weights (--pretrain checkpoints/latest)
# Stage 2:  Inference (both models)
# Stage 3:  RMSE evaluation
# Stage 4:  Kaggle submission
################################################################################

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"

GPU_ID="${1:-auto}"

log_info()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]  $1"; }
log_warn()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [WARN]  $1"; }
log_error() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2; }
header()    { echo ""; echo "════════════════════════════════════════════════════════════════════"; echo "  $1"; echo "════════════════════════════════════════════════════════════════════"; echo ""; }

header "FloodLM Transfer Learning Pipeline"
log_info "Script directory: ${SCRIPT_DIR}"
log_info "GPU ID: ${GPU_ID}"

# GPU selection
if [ "${GPU_ID}" = "auto" ]; then
    if command -v nvidia-smi &> /dev/null; then
        GPU_ID=$(${PYTHON} << 'EOF'
import subprocess
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=5)
    lines = result.stdout.strip().split('\n')
    if lines and lines[0]:
        gpu_data = [(int(l.split(',')[0].strip()), int(l.split(',')[1].strip())) for l in lines if l.strip()]
        print(max(gpu_data, key=lambda x: x[1])[0])
    else:
        print(0)
except:
    print(0)
EOF
)
    else
        GPU_ID=0
    fi
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_LAUNCH_BLOCKING=0
log_info "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

OVERALL_SUCCESS=0
RESULTS_SUMMARY=""

# ============================================================================
# Stage 1a: Train Model_1 from scratch
# ============================================================================
header "Stage 1a: Training Model_1 (from scratch)"

LOG_FILE="${LOG_DIR}/train_Model_1_${TIMESTAMP}.log"
log_info "Log file: ${LOG_FILE}"

if cd "${PROJECT_DIR}" && SELECTED_MODEL="Model_1" ${PYTHON} -u src/train.py --mixed-precision 2>&1 | tee "${LOG_FILE}"; then
    log_info "✓ Model_1 training completed"
    RESULTS_SUMMARY="${RESULTS_SUMMARY}
✓ Model_1: Training passed"
else
    EXIT_CODE=$?
    log_error "✗ Model_1 training failed (exit code ${EXIT_CODE})"
    OVERALL_SUCCESS=1
    RESULTS_SUMMARY="${RESULTS_SUMMARY}
✗ Model_1: Training failed (exit code ${EXIT_CODE})"
fi

# ============================================================================
# Stage 1b: Train Model_2 warm-started from Model_1 weights
# ============================================================================
header "Stage 1b: Training Model_2 (transfer from Model_1)"

PRETRAIN_DIR="${PROJECT_DIR}/checkpoints/latest"
if [ ! -f "${PRETRAIN_DIR}/Model_1_best_h64.pt" ] && [ ! -f "${PRETRAIN_DIR}/Model_1_best.pt" ]; then
    log_warn "No Model_1 checkpoint found in ${PRETRAIN_DIR} — training Model_2 from scratch"
    PRETRAIN_FLAG=""
else
    log_info "Using Model_1 weights from: ${PRETRAIN_DIR}"
    PRETRAIN_FLAG="--pretrain ${PRETRAIN_DIR}"
fi

LOG_FILE="${LOG_DIR}/train_Model_2_${TIMESTAMP}.log"
log_info "Log file: ${LOG_FILE}"

if cd "${PROJECT_DIR}" && SELECTED_MODEL="Model_2" ${PYTHON} -u src/train.py --mixed-precision ${PRETRAIN_FLAG} 2>&1 | tee "${LOG_FILE}"; then
    log_info "✓ Model_2 training completed"
    RESULTS_SUMMARY="${RESULTS_SUMMARY}
✓ Model_2: Training passed (transfer from Model_1)"
else
    EXIT_CODE=$?
    log_error "✗ Model_2 training failed (exit code ${EXIT_CODE})"
    OVERALL_SUCCESS=1
    RESULTS_SUMMARY="${RESULTS_SUMMARY}
✗ Model_2: Training failed (exit code ${EXIT_CODE})"
fi

# ============================================================================
# Stage 2: Inference
# ============================================================================
header "Stage 2: Running Autoregressive Inference"

LOG_FILE="${LOG_DIR}/inference_${TIMESTAMP}.log"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/latest"
SUBMISSION_FILE="submission_transfer_${TIMESTAMP}.csv"

log_info "Checkpoint dir: ${CHECKPOINT_DIR}"
log_info "Output: ${SUBMISSION_FILE}"

if cd "${PROJECT_DIR}" && ${PYTHON} -u src/autoregressive_inference.py --checkpoint-dir "${CHECKPOINT_DIR}" --output "${SUBMISSION_FILE}" 2>&1 | tee "${LOG_FILE}"; then
    log_info "✓ Inference completed"
    RESULTS_SUMMARY="${RESULTS_SUMMARY}
✓ Inference: Passed"
else
    EXIT_CODE=$?
    log_error "✗ Inference failed (exit code ${EXIT_CODE})"
    OVERALL_SUCCESS=1
    RESULTS_SUMMARY="${RESULTS_SUMMARY}
✗ Inference: Failed (exit code ${EXIT_CODE})"
fi

# ============================================================================
# Stage 3: RMSE Evaluation
# ============================================================================
header "Stage 3: Calculating RMSE"

LOG_FILE="${LOG_DIR}/rmse_${TIMESTAMP}.log"

if cd "${PROJECT_DIR}" && ${PYTHON} -u kaggle/calculate_rmse.py "${SUBMISSION_FILE}" submission_firsttry.csv 2>&1 | tee "${LOG_FILE}"; then
    log_info "✓ RMSE calculation completed"
    RESULTS_SUMMARY="${RESULTS_SUMMARY}
✓ RMSE Calculation: Passed"
else
    EXIT_CODE=$?
    log_error "✗ RMSE calculation failed (exit code ${EXIT_CODE})"
    OVERALL_SUCCESS=1
    RESULTS_SUMMARY="${RESULTS_SUMMARY}
✗ RMSE Calculation: Failed (exit code ${EXIT_CODE})"
fi

# ============================================================================
# Stage 4: Kaggle Submission
# ============================================================================
header "Stage 4: Submitting to Kaggle"

LOG_FILE="${LOG_DIR}/kaggle_${TIMESTAMP}.log"
SUBMISSION_MSG="${KAGGLE_MESSAGE:-FloodLM transfer ${TIMESTAMP}}"

if [ -f "${PROJECT_DIR}/${SUBMISSION_FILE}" ]; then
    if cd "${PROJECT_DIR}" && ${PYTHON} -u kaggle/submit_to_kaggle.py "${SUBMISSION_FILE}" \
            --message "${SUBMISSION_MSG}" --yes 2>&1 | tee "${LOG_FILE}"; then
        log_info "✓ Kaggle submission completed"
        RESULTS_SUMMARY="${RESULTS_SUMMARY}
✓ Kaggle Submission: Passed"
    else
        EXIT_CODE=$?
        log_error "✗ Kaggle submission failed (exit code ${EXIT_CODE})"
        OVERALL_SUCCESS=1
        RESULTS_SUMMARY="${RESULTS_SUMMARY}
✗ Kaggle Submission: Failed (exit code ${EXIT_CODE})"
    fi
else
    log_warn "${SUBMISSION_FILE} not found — skipping Kaggle submission"
    RESULTS_SUMMARY="${RESULTS_SUMMARY}
⚠ Kaggle Submission: Skipped"
fi

# ============================================================================
# Summary
# ============================================================================
header "Pipeline Complete - Summary"
echo "${RESULTS_SUMMARY}"

if [ ${OVERALL_SUCCESS} -eq 0 ]; then
    log_info "✓ All pipeline stages passed!"
else
    log_error "✗ Some pipeline stages failed"
fi

log_info "Log files: ${LOG_DIR}/"
exit ${OVERALL_SUCCESS}
