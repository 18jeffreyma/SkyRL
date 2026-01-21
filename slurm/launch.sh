#!/bin/bash
# ============================================================================
# SkyRL SLURM Job Launcher
# ============================================================================
# Coordinates launching of CPU runtime and GPU training jobs for SkyRL.
#
# Usage:
#   ./launch.sh --config dev              # Use development config
#   ./launch.sh --config production       # Use production config
#   ./launch.sh --config dev --dry-run    # Preview without submitting
#   ./launch.sh --config custom.conf      # Use custom config file
#
# The launcher:
# 1. Creates a coordination directory for inter-job communication
# 2. Submits the CPU runtime server job first
# 3. Submits the GPU training job with dependency on runtime
# 4. Reports job IDs and coordination directory
# ============================================================================

set -euo pipefail

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source common utilities
source "${SCRIPT_DIR}/lib/common.sh"

# ============================================================================
# Default Configuration
# ============================================================================

# Will be loaded from config file
CONFIG_FILE=""
DRY_RUN=false
SKIP_RUNTIME=false
RUNTIME_ONLY=false

# ============================================================================
# Argument Parsing
# ============================================================================

print_usage() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Launch SkyRL SLURM jobs for training with sandboxed code execution.

Options:
    --config FILE       Configuration file (e.g., dev, production, or path to .conf)
    --dry-run           Preview jobs without submitting
    --skip-runtime      Skip runtime job (use existing runtime)
    --runtime-only      Only launch runtime job
    --runtime-url URL   Use existing runtime at URL (implies --skip-runtime)
    --help              Show this help message

Examples:
    $(basename "$0") --config dev                     # Development run
    $(basename "$0") --config production              # Production run
    $(basename "$0") --config dev --dry-run           # Preview dev config
    $(basename "$0") --config /path/to/custom.conf    # Custom config
    $(basename "$0") --config dev --runtime-only      # Only start runtime

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --skip-runtime)
                SKIP_RUNTIME=true
                shift
                ;;
            --runtime-only)
                RUNTIME_ONLY=true
                shift
                ;;
            --runtime-url)
                EXISTING_RUNTIME_URL="$2"
                SKIP_RUNTIME=true
                shift 2
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

# ============================================================================
# Configuration Loading
# ============================================================================

load_config() {
    local config_arg="$1"

    # Check if it's a preset name (dev, production) or a path
    if [[ ! "${config_arg}" == *"/"* ]] && [[ ! "${config_arg}" == *".conf"* ]]; then
        # It's a preset name, look in config directory
        config_arg="${SCRIPT_DIR}/config/${config_arg}.conf"
    fi

    if [[ ! -f "${config_arg}" ]]; then
        log_error "Configuration file not found: ${config_arg}"
        exit 1
    fi

    log_info "Loading configuration from: ${config_arg}"
    source "${config_arg}"

    # Set defaults for any missing values
    GPU_PARTITION="${GPU_PARTITION:-gpu_test}"
    CPU_PARTITION="${CPU_PARTITION:-test}"
    GPU_NODES="${GPU_NODES:-1}"
    GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
    TRAINING_CPUS="${TRAINING_CPUS:-48}"
    TRAINING_MEMORY="${TRAINING_MEMORY:-500G}"
    CPUS_FOR_RUNTIME="${CPUS_FOR_RUNTIME:-64}"
    RUNTIME_MEMORY="${RUNTIME_MEMORY:-256G}"
    TIME_LIMIT="${TIME_LIMIT:-12:00:00}"
    MODEL="${MODEL:-Qwen/Qwen3-8B}"
    WANDB_EXPERIMENT="${WANDB_EXPERIMENT:-skyrl-$(date +%Y%m%d-%H%M%S)}"
}

# ============================================================================
# Job Submission Functions
# ============================================================================

create_runtime_sbatch_override() {
    local coord_dir="$1"
    local sbatch_file="${coord_dir}/runtime_server.sbatch"

    # Create a modified sbatch file with our config values
    cat > "${sbatch_file}" << EOF
#!/bin/bash
#SBATCH --job-name=skyrl-runtime
#SBATCH --output=${coord_dir}/runtime_%j.out
#SBATCH --error=${coord_dir}/runtime_%j.err
#SBATCH --partition=${CPU_PARTITION}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=${CPUS_FOR_RUNTIME}
#SBATCH --mem=${RUNTIME_MEMORY}
#SBATCH --time=${TIME_LIMIT}

# Export coordination directory
export SKYRL_COORD_DIR="${coord_dir}"
export SKYRL_PROJECT_ROOT="${SKYRL_PROJECT_ROOT}"
export CPUS_PER_WORKER="${CPUS_PER_WORKER:-8}"
export MEMORY_PER_WORKER="${MEMORY_PER_WORKER:-16G}"
export CONTAINER_IMAGE="${CONTAINER_IMAGE:-ghcr.io/opendevin/sandbox:main}"
export RUNTIME_PORT="${RUNTIME_PORT:-8000}"
export ALLHANDS_API_KEY="${ALLHANDS_API_KEY:-}"

# Run the actual runtime script
exec "${SCRIPT_DIR}/jobs/runtime_server.sbatch"
EOF

    echo "${sbatch_file}"
}

create_training_sbatch_override() {
    local coord_dir="$1"
    local sbatch_file="${coord_dir}/verl_training.sbatch"

    # Create a modified sbatch file with our config values
    cat > "${sbatch_file}" << EOF
#!/bin/bash
#SBATCH --job-name=skyrl-train
#SBATCH --output=${coord_dir}/training_%j.out
#SBATCH --error=${coord_dir}/training_%j.err
#SBATCH --partition=${GPU_PARTITION}
#SBATCH --nodes=${GPU_NODES}
#SBATCH --gres=gpu:${GPUS_PER_NODE}
#SBATCH --cpus-per-task=${TRAINING_CPUS}
#SBATCH --mem=${TRAINING_MEMORY}
#SBATCH --time=${TIME_LIMIT}

# Export coordination directory and all config
export SKYRL_COORD_DIR="${coord_dir}"
export SKYRL_PROJECT_ROOT="${SKYRL_PROJECT_ROOT}"

# Model config
export MODEL="${MODEL}"
export TP_SIZE="${TP_SIZE:-2}"
export SP_SIZE="${SP_SIZE:-2}"

# Training config
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export ROLLOUT_N="${ROLLOUT_N:-8}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8000}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"
export GPUS_PER_NODE="${GPUS_PER_NODE}"

# Data paths
export DATA_DIR="${DATA_DIR:-/mnt/shared_storage/datasets/r2e-all}"
export TRAIN_DATA="${TRAIN_DATA:-\${DATA_DIR}/train.parquet}"
export VAL_DATA="${VAL_DATA:-\${DATA_DIR}/validation.parquet}"

# Output paths
export CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/local_storage/ckpts/skyrl}"
export ROLLOUT_DIR="${ROLLOUT_DIR:-/mnt/local_storage/rollouts/skyrl}"

# Logging
export WANDB_PROJECT="${WANDB_PROJECT:-skyrl-training}"
export WANDB_EXPERIMENT="${WANDB_EXPERIMENT}"

# Runtime timeout
export RUNTIME_READY_TIMEOUT="${RUNTIME_READY_TIMEOUT:-600}"

# vLLM config
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

# EFA networking
export SKYRL_LD_LIBRARY_PATH_EXPORT="${SKYRL_LD_LIBRARY_PATH_EXPORT:-0}"
export FI_PROVIDER="${FI_PROVIDER:-efa}"

# Run the actual training script
exec "${SCRIPT_DIR}/jobs/verl_training.sbatch"
EOF

    echo "${sbatch_file}"
}

submit_runtime_job() {
    local coord_dir="$1"
    local sbatch_file

    sbatch_file=$(create_runtime_sbatch_override "${coord_dir}")

    if [[ "${DRY_RUN}" == true ]]; then
        log_info "[DRY RUN] Would submit runtime job:"
        log_info "  sbatch ${sbatch_file}"
        cat "${sbatch_file}"
        echo "RUNTIME_DRY_RUN"
        return 0
    fi

    local output
    output=$(sbatch "${sbatch_file}")

    # Extract job ID from output (format: "Submitted batch job XXXXXX")
    local job_id
    job_id=$(echo "${output}" | grep -oP '\d+$')

    if [[ -z "${job_id}" ]]; then
        log_error "Failed to submit runtime job: ${output}"
        return 1
    fi

    log_info "Submitted runtime job: ${job_id}"
    echo "${job_id}"
}

submit_training_job() {
    local coord_dir="$1"
    local dependency="${2:-}"
    local sbatch_file

    sbatch_file=$(create_training_sbatch_override "${coord_dir}")

    local sbatch_args=()
    if [[ -n "${dependency}" ]] && [[ "${dependency}" != "RUNTIME_DRY_RUN" ]]; then
        # Use afterany so training starts even if runtime fails (it will detect and exit)
        sbatch_args+=(--dependency="afterok:${dependency}")
    fi

    if [[ "${DRY_RUN}" == true ]]; then
        log_info "[DRY RUN] Would submit training job:"
        log_info "  sbatch ${sbatch_args[*]} ${sbatch_file}"
        cat "${sbatch_file}"
        echo "TRAINING_DRY_RUN"
        return 0
    fi

    local output
    output=$(sbatch "${sbatch_args[@]}" "${sbatch_file}")

    # Extract job ID
    local job_id
    job_id=$(echo "${output}" | grep -oP '\d+$')

    if [[ -z "${job_id}" ]]; then
        log_error "Failed to submit training job: ${output}"
        return 1
    fi

    log_info "Submitted training job: ${job_id}"
    echo "${job_id}"
}

# ============================================================================
# Main
# ============================================================================

main() {
    parse_args "$@"

    # Validate config
    if [[ -z "${CONFIG_FILE}" ]]; then
        log_error "Configuration file required. Use --config <name|path>"
        print_usage
        exit 1
    fi

    # Load configuration
    load_config "${CONFIG_FILE}"

    log_info "============================================"
    log_info "SkyRL SLURM Job Launcher"
    log_info "============================================"
    log_info "Configuration: ${CONFIG_FILE}"
    log_info "Model: ${MODEL}"
    log_info "GPU Nodes: ${GPU_NODES}"
    log_info "Time Limit: ${TIME_LIMIT}"

    if [[ "${DRY_RUN}" == true ]]; then
        log_info "Mode: DRY RUN (no jobs will be submitted)"
    fi

    # Create coordination directory
    local coord_dir
    coord_dir=$(create_job_coordination_dir "launch")
    export SKYRL_COORD_DIR="${coord_dir}"

    log_info "Coordination directory: ${coord_dir}"

    # Save configuration to coordination directory for reference
    cp "${SCRIPT_DIR}/config/"*.conf "${coord_dir}/" 2>/dev/null || true

    local runtime_job_id=""
    local training_job_id=""

    # Submit runtime job (unless skipped)
    if [[ "${SKIP_RUNTIME}" != true ]]; then
        log_info "Submitting runtime server job..."
        runtime_job_id=$(submit_runtime_job "${coord_dir}")

        if [[ "${RUNTIME_ONLY}" == true ]]; then
            log_info "Runtime-only mode, skipping training job"
            log_info "============================================"
            log_info "Runtime Job ID: ${runtime_job_id}"
            log_info "Coordination Dir: ${coord_dir}"
            log_info "============================================"
            exit 0
        fi
    else
        log_info "Skipping runtime job (using existing runtime)"
        if [[ -n "${EXISTING_RUNTIME_URL:-}" ]]; then
            write_runtime_url "${coord_dir}" "${EXISTING_RUNTIME_URL}"
            signal_runtime_ready "${coord_dir}"
        fi
    fi

    # Submit training job
    log_info "Submitting training job..."
    training_job_id=$(submit_training_job "${coord_dir}" "${runtime_job_id}")

    # Report results
    log_info "============================================"
    log_info "Jobs Submitted Successfully!"
    log_info "============================================"
    if [[ -n "${runtime_job_id}" ]]; then
        log_info "Runtime Job ID:  ${runtime_job_id}"
    fi
    log_info "Training Job ID: ${training_job_id}"
    log_info "Coordination:    ${coord_dir}"
    log_info ""
    log_info "Monitor with:"
    if [[ -n "${runtime_job_id}" ]] && [[ "${runtime_job_id}" != "RUNTIME_DRY_RUN" ]]; then
        log_info "  squeue -j ${runtime_job_id},${training_job_id}"
        log_info "  tail -f ${coord_dir}/runtime_*.out"
    fi
    if [[ "${training_job_id}" != "TRAINING_DRY_RUN" ]]; then
        log_info "  tail -f ${coord_dir}/training_*.out"
    fi
    log_info "============================================"
}

main "$@"
