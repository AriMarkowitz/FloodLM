#!/bin/bash
#
# Fine-tune pipeline: resume both models on train+val data at h=64, then infer + submit.
#
# Usage:
#   bash run/pipeline_finetune_submit.sh [GPU_ID] [Model_1|Model_2|all]
#
# This should only be run AFTER the main training run is complete and you are
# satisfied with the architecture/hyperparameters. It adds val events back into
# training for a few final h=64 epochs at a reduced LR before the Kaggle submission.

set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON="${PYTHON:-python3}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${PROJECT_DIR}/logs"

GPU_ID="${1:-auto}"
MODEL_SELECTION="${2:-all}"

case "${MODEL_SELECTION}" in
    Model_1|Model_2)
        MODELS=("${MODEL_SELECTION}")
        ;;
    all)
        MODELS=("Model_1" "Model_2")
        ;;
    *)
        echo "Invalid model selection: ${MODEL_SELECTION}. Use Model_1, Model_2, or all."
        exit 1
        ;;
esac

mkdir -p "${LOG_DIR}"

log_info()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [INFO]  $1"; }
log_warn()  { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [WARN]  $1"; }
log_error() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] [ERROR] $1" >&2; }

header() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════"
    echo "  $1"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
}

header "FloodLM Fine-tune + Submit Pipeline"
log_info "Models: ${MODELS[*]}"
log_info "GPU_ID: ${GPU_ID}"

# GPU selection
if [ "${GPU_ID}" = "auto" ]; then
    if command -v nvidia-smi &> /dev/null; then
        GPU_ID=$(${PYTHON} << 'EOF'
import subprocess
try:
    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, timeout=5
    )
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
    log_info "Auto-selected GPU: ${GPU_ID}"
fi

export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export TF_FORCE_GPU_ALLOW_GROWTH=true
export CUDA_LAUNCH_BLOCKING=0

CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/latest"
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    log_error "checkpoints/latest not found — run main training first."
    exit 1
fi

OVERALL_SUCCESS=0
RESULTS_SUMMARY=""

# ============================================================================
# Stage 1: Fine-tune each model on train+val data
# ============================================================================
header "Stage 1: Fine-tuning on train+val (all) data at h=64"

# Fine-tune config:
#   --epochs 4      → 4 additional h=64 epochs (adjust via FINETUNE_EPOCHS env var)
#   --lr 3e-4       → 1/3 of original 1e-3
#   --max-h 64      → skip curriculum entirely, train at h=64 from the start
#   --train-split all → train+val events combined
#   --no-val        → no validation (all data is now in training)
#   --mixed-precision → AMP + gradient checkpointing to avoid OOM at h=64

FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-4}"
FINETUNE_LR="${FINETUNE_LR:-3e-4}"

for MODEL in "${MODELS[@]}"; do
    LOG_FILE="${LOG_DIR}/finetune_${MODEL}_${TIMESTAMP}.log"
    log_info "Fine-tuning ${MODEL} for ${FINETUNE_EPOCHS} epochs at h=64 (train+val data, lr=${FINETUNE_LR})..."
    log_info "Log: ${LOG_FILE}"

    if cd "${PROJECT_DIR}" && SELECTED_MODEL="${MODEL}" ${PYTHON} -u src/train.py \
            --resume "${CHECKPOINT_DIR}" \
            --epochs "${FINETUNE_EPOCHS}" \
            --learning-rate "${FINETUNE_LR}" \
            --max-h 64 \
            --train-split all \
            --no-val \
            --mixed-precision \
            2>&1 | tee "${LOG_FILE}"; then
        log_info "✓ ${MODEL} fine-tuning completed"
        # Overwrite best_h64.pt with the new fine-tuned checkpoint so inference picks it up.
        # (best_h64 is only written during val-enabled training; without --no-val it stays stale.)
        if [ -f "${CHECKPOINT_DIR}/${MODEL}_best.pt" ]; then
            cp "${CHECKPOINT_DIR}/${MODEL}_best.pt" "${CHECKPOINT_DIR}/${MODEL}_best_h64.pt"
            log_info "✓ Copied ${MODEL}_best.pt → ${MODEL}_best_h64.pt in ${CHECKPOINT_DIR}"
        fi
        RESULTS_SUMMARY="${RESULTS_SUMMARY}
✓ ${MODEL}: Fine-tune passed"
    else
        EXIT_CODE=$?
        log_error "✗ ${MODEL} fine-tuning failed (exit code ${EXIT_CODE})"
        OVERALL_SUCCESS=1
        RESULTS_SUMMARY="${RESULTS_SUMMARY}
✗ ${MODEL}: Fine-tune failed (exit code ${EXIT_CODE})"
    fi
    echo ""
done

# ============================================================================
# Stage 2: Inference
# ============================================================================
header "Stage 2: Autoregressive Inference"

LOG_FILE="${LOG_DIR}/inference_finetune_${TIMESTAMP}.log"
SUBMISSION_FILE="submission_finetune_${TIMESTAMP}.csv"

log_info "Running inference from ${CHECKPOINT_DIR}..."
log_info "Output: ${SUBMISSION_FILE}"

if cd "${PROJECT_DIR}" && ${PYTHON} -u src/autoregressive_inference.py \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --output "${SUBMISSION_FILE}" \
        2>&1 | tee "${LOG_FILE}"; then
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

echo ""

# ============================================================================
# Stage 3: RMSE calculation
# ============================================================================
header "Stage 3: RMSE Calculation"

LOG_FILE="${LOG_DIR}/rmse_finetune_${TIMESTAMP}.log"
log_info "Calculating RMSE..."

if cd "${PROJECT_DIR}" && ${PYTHON} -u kaggle/calculate_rmse.py "${SUBMISSION_FILE}" submission_firsttry.csv \
        2>&1 | tee "${LOG_FILE}"; then
    log_info "✓ RMSE calculation completed"
    RESULTS_SUMMARY="${RESULTS_SUMMARY}
✓ RMSE: Passed"
else
    EXIT_CODE=$?
    log_warn "✗ RMSE calculation failed (exit code ${EXIT_CODE}) — continuing to Kaggle submission"
    RESULTS_SUMMARY="${RESULTS_SUMMARY}
⚠ RMSE: Failed (exit code ${EXIT_CODE})"
fi

echo ""

# ============================================================================
# Stage 4: Architecture snapshot
# ============================================================================
header "Stage 4: Architecture Snapshot"

SUBMISSION_BASE="${SUBMISSION_FILE%.csv}"
log_info "Snapshotting code + checkpoints → snapshots/${SUBMISSION_BASE}/"
bash "${SCRIPT_DIR}/snapshot_arch.sh" "${SUBMISSION_BASE}" "${PROJECT_DIR}" \
    && log_info "✓ Snapshot saved" \
    || log_warn "✗ Snapshot failed (non-fatal)"

echo ""

# ============================================================================
# Stage 5: Kaggle submission
# ============================================================================
header "Stage 5: Kaggle Submission"

LOG_FILE="${LOG_DIR}/kaggle_finetune_${TIMESTAMP}.log"
SUBMISSION_MSG="${KAGGLE_MESSAGE:-FloodLM finetune+val ${TIMESTAMP}}"
log_info "Submitting ${SUBMISSION_FILE} — message: '${SUBMISSION_MSG}'"

if [ -f "${PROJECT_DIR}/${SUBMISSION_FILE}" ]; then
    if cd "${PROJECT_DIR}" && ${PYTHON} -u kaggle/submit_to_kaggle.py "${SUBMISSION_FILE}" \
            --message "${SUBMISSION_MSG}" --yes \
            2>&1 | tee "${LOG_FILE}"; then
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
⚠ Kaggle Submission: Skipped (no submission file)"
fi

echo ""

# ============================================================================
# Summary
# ============================================================================
header "Fine-tune Pipeline Complete"
log_info "Results:"
echo "${RESULTS_SUMMARY}"

if [ ${OVERALL_SUCCESS} -eq 0 ]; then
    log_info "✓ All stages passed!"
else
    log_error "✗ Some stages failed — check logs in ${LOG_DIR}/"
fi

exit ${OVERALL_SUCCESS}
