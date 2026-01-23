#!/bin/bash
#SBATCH --job-name=skyrl-prebuild
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=${SLURM_PARTITION:-test}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=${CPUS_FOR_PREBUILD:-16}
#SBATCH --mem=${PREBUILD_MEMORY:-64G}
#SBATCH --time=${TIME_LIMIT:-12:00:00}

# ============================================================================
# SkyRL Image Pre-build Job
# ============================================================================
# Pre-builds OpenHands runtime images from base images (e.g., r2e images)
# so they're ready for training without on-demand build latency.
#
# Required Environment Variables:
#   DATASET_PATH - Path to parquet dataset file
#
# Optional Environment Variables:
#   IMAGE_PREFIX - Prefix for image names (default: docker.io/xingyaoww/)
#   MAX_WORKERS - Maximum concurrent builds (default: 2)
#   FORCE_REBUILD - Set to "true" to force rebuild existing images
# ============================================================================

set -euo pipefail

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

# ============================================================================
# Configuration
# ============================================================================

IMAGE_PREFIX="${IMAGE_PREFIX:-docker.io/xingyaoww/}"
MAX_WORKERS="${MAX_WORKERS:-2}"
FORCE_REBUILD="${FORCE_REBUILD:-false}"
SLURM_RUNTIME_DIR="${SKYRL_PROJECT_ROOT}/SkyRL/slurm-remote-runtime"

# ============================================================================
# Validation
# ============================================================================

if [[ -z "${DATASET_PATH:-}" ]]; then
    log_error "DATASET_PATH not set. Please specify the parquet dataset path."
    exit 1
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
    log_error "Dataset not found: ${DATASET_PATH}"
    exit 1
fi

if [[ ! -d "${SLURM_RUNTIME_DIR}" ]]; then
    log_error "slurm-remote-runtime directory not found: ${SLURM_RUNTIME_DIR}"
    exit 1
fi

# ============================================================================
# Main
# ============================================================================

main() {
    log_info "Starting SkyRL Image Pre-build"
    log_info "Job ID: ${SLURM_JOB_ID}"
    log_info "Node: $(hostname)"
    log_info "Dataset: ${DATASET_PATH}"
    log_info "Image prefix: ${IMAGE_PREFIX}"
    log_info "Max workers: ${MAX_WORKERS}"
    log_info "Force rebuild: ${FORCE_REBUILD}"

    # Activate virtual environment
    cd "${SLURM_RUNTIME_DIR}"
    if [[ -d "${SLURM_RUNTIME_DIR}/.venv" ]]; then
        source "${SLURM_RUNTIME_DIR}/.venv/bin/activate"
        log_info "Activated slurm-remote-runtime venv"
    else
        log_error "No venv found at ${SLURM_RUNTIME_DIR}/.venv"
        exit 1
    fi

    # Build arguments
    local args=(
        --dataset "${DATASET_PATH}"
        --image-prefix "${IMAGE_PREFIX}"
        --max-workers "${MAX_WORKERS}"
    )

    if [[ "${FORCE_REBUILD}" == "true" ]]; then
        args+=(--force-rebuild)
    fi

    # Output results to job directory
    local output_json="${SLURM_SUBMIT_DIR:-/tmp}/prebuild_results_${SLURM_JOB_ID}.json"
    args+=(--output-json "${output_json}")

    # Run pre-build
    log_info "Starting pre-build..."
    python -m slurm_runtime.prebuild_images "${args[@]}"

    log_info "Pre-build completed"
    log_info "Results saved to: ${output_json}"
}

main "$@"
