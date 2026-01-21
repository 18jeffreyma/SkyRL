#!/bin/bash
# SkyRL SLURM Common Utilities
# Shared functions for inter-job coordination and logging

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Base directory for job coordination files
SKYRL_JOBS_BASE="${SKYRL_JOBS_BASE:-/n/holylabs/janapa_reddi_lab/Lab/jjma/coding-agent-rl/.skyrl_jobs}"

# Project root directory
SKYRL_PROJECT_ROOT="${SKYRL_PROJECT_ROOT:-/n/holylabs/janapa_reddi_lab/Lab/jjma/coding-agent-rl}"

# Default timeouts (in seconds)
RUNTIME_READY_TIMEOUT="${RUNTIME_READY_TIMEOUT:-600}"  # 10 minutes
TRAINING_COMPLETE_TIMEOUT="${TRAINING_COMPLETE_TIMEOUT:-43200}"  # 12 hours

# ============================================================================
# Logging Functions
# ============================================================================

log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $*"
}

log_warn() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $*" >&2
}

log_error() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $*" >&2
}

log_debug() {
    if [[ "${SKYRL_DEBUG:-0}" == "1" ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [DEBUG] $*"
    fi
}

# ============================================================================
# Job Coordination Directory Functions
# ============================================================================

# Create a new job coordination directory
# Returns the path to the coordination directory
create_job_coordination_dir() {
    local job_id="${1:-${SLURM_JOB_ID:-unknown}}"
    local timestamp
    timestamp=$(date '+%Y%m%d_%H%M%S')

    local coord_dir="${SKYRL_JOBS_BASE}/${timestamp}_${job_id}"
    mkdir -p "${coord_dir}"

    echo "${coord_dir}"
}

# Get or create coordination directory for current session
# Uses SKYRL_COORD_DIR if set, otherwise creates new
get_coordination_dir() {
    if [[ -n "${SKYRL_COORD_DIR:-}" ]] && [[ -d "${SKYRL_COORD_DIR}" ]]; then
        echo "${SKYRL_COORD_DIR}"
    else
        create_job_coordination_dir
    fi
}

# ============================================================================
# Inter-Job Communication Functions
# ============================================================================

# Write runtime URL to coordination file
# Args: $1 = coordination_dir, $2 = url
write_runtime_url() {
    local coord_dir="$1"
    local url="$2"

    echo "${url}" > "${coord_dir}/runtime_url.txt"
    log_info "Wrote runtime URL to ${coord_dir}/runtime_url.txt"
}

# Read runtime URL from coordination file with timeout
# Args: $1 = coordination_dir, $2 = timeout_seconds (optional, default 600)
# Returns: URL string or exits with error
wait_for_runtime_url() {
    local coord_dir="$1"
    local timeout="${2:-${RUNTIME_READY_TIMEOUT}}"
    local url_file="${coord_dir}/runtime_url.txt"
    local ready_file="${coord_dir}/runtime_ready.flag"

    local elapsed=0
    local interval=5

    log_info "Waiting for runtime URL at ${url_file} (timeout: ${timeout}s)"

    while [[ $elapsed -lt $timeout ]]; do
        if [[ -f "${ready_file}" ]] && [[ -f "${url_file}" ]]; then
            local url
            url=$(cat "${url_file}")
            if [[ -n "${url}" ]]; then
                log_info "Found runtime URL: ${url}"
                echo "${url}"
                return 0
            fi
        fi

        sleep "${interval}"
        elapsed=$((elapsed + interval))
        log_debug "Still waiting for runtime... (${elapsed}/${timeout}s)"
    done

    log_error "Timeout waiting for runtime URL after ${timeout}s"
    return 1
}

# Signal that runtime is ready
# Args: $1 = coordination_dir
signal_runtime_ready() {
    local coord_dir="$1"
    touch "${coord_dir}/runtime_ready.flag"
    log_info "Signaled runtime ready at ${coord_dir}/runtime_ready.flag"
}

# Signal that training is complete
# Args: $1 = coordination_dir
signal_training_complete() {
    local coord_dir="$1"
    touch "${coord_dir}/training_complete.flag"
    log_info "Signaled training complete at ${coord_dir}/training_complete.flag"
}

# Wait for training to complete (for runtime job to know when to shutdown)
# Args: $1 = coordination_dir, $2 = timeout_seconds (optional)
wait_for_training_complete() {
    local coord_dir="$1"
    local timeout="${2:-${TRAINING_COMPLETE_TIMEOUT}}"
    local flag_file="${coord_dir}/training_complete.flag"

    local elapsed=0
    local interval=30

    log_info "Waiting for training completion signal (timeout: ${timeout}s)"

    while [[ $elapsed -lt $timeout ]]; do
        if [[ -f "${flag_file}" ]]; then
            log_info "Training complete signal received"
            return 0
        fi

        sleep "${interval}"
        elapsed=$((elapsed + interval))

        # Log periodic status
        if [[ $((elapsed % 300)) -eq 0 ]]; then
            log_info "Still running, waiting for training... (${elapsed}/${timeout}s)"
        fi
    done

    log_warn "Timeout waiting for training completion after ${timeout}s"
    return 1
}

# ============================================================================
# Environment Setup Functions
# ============================================================================

# Load environment from .env file
# Args: $1 = path to .env file
load_env_file() {
    local env_file="$1"

    if [[ ! -f "${env_file}" ]]; then
        log_warn "Environment file not found: ${env_file}"
        return 1
    fi

    log_info "Loading environment from ${env_file}"

    # Export variables from .env file (skip comments and empty lines)
    while IFS= read -r line || [[ -n "$line" ]]; do
        # Skip comments and empty lines
        [[ -z "$line" ]] && continue
        [[ "$line" =~ ^[[:space:]]*# ]] && continue

        # Skip lines without '='
        [[ "$line" != *"="* ]] && continue

        # Export the variable (only if value is not <REPLACE>)
        if [[ "$line" != *"<REPLACE>"* ]]; then
            export "${line?}"
        fi
    done < "${env_file}"
}

# Set up LD_LIBRARY_PATH for FASRC EFA networking
setup_efa_networking() {
    if [[ "${SKYRL_LD_LIBRARY_PATH_EXPORT:-0}" == "1" ]]; then
        export FI_PROVIDER="${FI_PROVIDER:-efa}"
        log_info "EFA networking configured (FI_PROVIDER=${FI_PROVIDER})"
    fi
}

# Activate UV virtual environment
activate_uv_venv() {
    local venv_dir="${1:-${SKYRL_PROJECT_ROOT}/SkyRL/skyrl-agent/.venv}"

    if [[ -d "${venv_dir}" ]]; then
        source "${venv_dir}/bin/activate"
        log_info "Activated virtual environment: ${venv_dir}"
    else
        log_warn "Virtual environment not found: ${venv_dir}"
    fi
}

# ============================================================================
# SLURM Helper Functions
# ============================================================================

# Get the hostname of the first node in allocation
get_first_node() {
    if [[ -n "${SLURM_NODELIST:-}" ]]; then
        scontrol show hostnames "${SLURM_NODELIST}" | head -n1
    else
        hostname
    fi
}

# Get total number of nodes in allocation
get_node_count() {
    echo "${SLURM_NNODES:-1}"
}

# Get number of CPUs per node
get_cpus_per_node() {
    echo "${SLURM_CPUS_ON_NODE:-${SLURM_CPUS_PER_TASK:-1}}"
}

# Check if running inside a SLURM allocation
is_slurm_job() {
    [[ -n "${SLURM_JOB_ID:-}" ]]
}

# ============================================================================
# Cleanup Functions
# ============================================================================

# Clean up old coordination directories (older than N days)
# Args: $1 = days_old (default 7)
cleanup_old_coordination_dirs() {
    local days_old="${1:-7}"

    if [[ -d "${SKYRL_JOBS_BASE}" ]]; then
        log_info "Cleaning up coordination directories older than ${days_old} days"
        find "${SKYRL_JOBS_BASE}" -maxdepth 1 -type d -mtime +"${days_old}" -exec rm -rf {} \; 2>/dev/null || true
    fi
}

# Graceful shutdown handler
# Call this in a trap to handle SIGTERM/SIGINT
setup_graceful_shutdown() {
    local cleanup_fn="${1:-}"

    _shutdown_handler() {
        log_info "Received shutdown signal, cleaning up..."
        if [[ -n "${cleanup_fn}" ]] && declare -f "${cleanup_fn}" > /dev/null; then
            "${cleanup_fn}"
        fi
        exit 0
    }

    trap _shutdown_handler SIGTERM SIGINT
}
