#!/bin/bash

################################################################################
# FloodLM Test Pipeline Script
# 
# Executes the complete training and inference pipeline on a data subset
# with GPU optimizations and performance monitoring
#
# Supports testing both Model_1 and Model_2 with separate graph structures
#
# Usage:
#   ./test_pipeline.sh [GPU_ID] [MODEL]
#
# Examples:
#   ./test_pipeline.sh           # Auto GPU, test Model_1
#   ./test_pipeline.sh 0         # GPU 0, test Model_1
#   ./test_pipeline.sh 0 Model_1 # GPU 0, test Model_1
#   ./test_pipeline.sh 0 Model_2 # GPU 0, test Model_2
#   ./test_pipeline.sh 0 all     # GPU 0, test both models
#
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

header "FloodLM Multi-Model Test Pipeline"

log_info "Script directory: ${SCRIPT_DIR}"
log_info "GPU ID: ${GPU_ID}"
log_info "Models to test: ${MODELS[*]}"
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
# Pre-test Checks
################################################################################

log_info "Performing pre-test checks..."

# Check data directory
if [ ! -d "${SCRIPT_DIR}/data" ]; then
    log_error "Data directory not found: ${SCRIPT_DIR}/data"
    exit 1
fi

# Check source files
for file in src/model.py src/data.py data_config.py example_usage.py; do
    if [ ! -f "${SCRIPT_DIR}/${file}" ]; then
        log_error "File not found: ${SCRIPT_DIR}/${file}"
        exit 1
    fi
done

log_info "Pre-test checks passed"

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
# Run Tests for Each Model
################################################################################

OVERALL_SUCCESS=0
RESULTS_SUMMARY=""

for MODEL in "${MODELS[@]}"; do
    LOG_FILE="${LOG_DIR}/test_${MODEL}_${TIMESTAMP}.log"
    
    header "Testing ${MODEL}"
    
    log_info "Log file: ${LOG_FILE}"
    
    # Update data configuration
    update_data_config "${MODEL}"
    
    # Verify data configuration
    log_info "Verifying data configuration for ${MODEL}..."
    if ! SELECTED_MODEL="${MODEL}" ${PYTHON} data_config.py >> "${LOG_FILE}" 2>&1; then
        log_error "Data validation failed for ${MODEL}"
        log_error "Check log: ${LOG_FILE}"
        OVERALL_SUCCESS=1
        RESULTS_SUMMARY="${RESULTS_SUMMARY}
✗ ${MODEL}: Data validation failed"
        continue
    fi
    
    log_info "✓ Data validation successful"
    
    # Run test
    log_info "Starting test execution for ${MODEL}..."
    log_info "Command: cd ${SCRIPT_DIR} && SELECTED_MODEL=${MODEL} ${PYTHON} -u example_usage.py"
    
    if cd "${SCRIPT_DIR}" && SELECTED_MODEL="${MODEL}" ${PYTHON} -u example_usage.py 2>&1 | tee -a "${LOG_FILE}"; then
        TEST_EXIT_CODE=0
        log_info "✓ ${MODEL} test completed successfully"
        RESULTS_SUMMARY="${RESULTS_SUMMARY}
✓ ${MODEL}: Test passed"
    else
        TEST_EXIT_CODE=$?
        log_error "✗ ${MODEL} test failed with exit code ${TEST_EXIT_CODE}"
        log_error "Check log: ${LOG_FILE}"
        OVERALL_SUCCESS=1
        RESULTS_SUMMARY="${RESULTS_SUMMARY}
✗ ${MODEL}: Test failed (exit code ${TEST_EXIT_CODE})"
    fi
    
    # Post-test GPU status
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU Memory Status for ${MODEL}:"
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | while read line; do
            log_info "  $line"
        done
    fi
    
    echo ""
done

################################################################################
# Summary
################################################################################

header "Test Summary"

log_info "Models tested: ${MODELS[*]}"
log_info "Results:"
echo "${RESULTS_SUMMARY}"

if [ ${OVERALL_SUCCESS} -eq 0 ]; then
    log_info "✓ All tests passed!"
else
    log_error "✗ Some tests failed"
fi

log_info "Log files saved to: ${LOG_DIR}/"
log_info "View logs with: tail -f ${LOG_DIR}/test_*.log"

exit ${OVERALL_SUCCESS}


