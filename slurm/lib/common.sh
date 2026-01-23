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

# ============================================================================
# Podman Isolation Functions
# ============================================================================

# Set up isolated podman storage for this SLURM job
# This prevents state corruption between jobs by using job-specific directories
setup_isolated_podman_storage() {
    local job_id="${SLURM_JOB_ID:-$$}"
    local unique_id="${job_id}_${SLURM_ARRAY_TASK_ID:-0}"

    log_info "Setting up isolated podman environment for job ${unique_id}"

    # Create job-specific podman directories in /tmp (local storage) FIRST
    # We need these directories before any cleanup operations
    local podman_root="/tmp/podman-${USER}-${job_id}"
    mkdir -p "${podman_root}/storage"
    mkdir -p "${podman_root}/run"
    mkdir -p "${podman_root}/config"
    mkdir -p "${podman_root}/data/containers"  # For XDG_DATA_HOME redirect

    # CRITICAL: Set XDG_DATA_HOME to redirect ~/.local/share to /tmp
    # This prevents podman/buildx from using NFS-mounted home directory
    export XDG_DATA_HOME="${podman_root}/data"
    log_info "Set XDG_DATA_HOME=${XDG_DATA_HOME} (redirects ~/.local/share)"

    # Set TMPDIR to job-specific local storage
    export TMPDIR="/tmp/${USER}/${unique_id}"
    mkdir -p "${TMPDIR}"
    log_info "Set TMPDIR=${TMPDIR}"

    # Clean up stale podman state from previous jobs
    log_info "Cleaning up stale podman/container state..."
    rm -rf /scratch/${USER}/containers 2>/dev/null || true
    rm -rf /scratch/${USER}/podman* 2>/dev/null || true
    rm -rf /var/tmp/containers-user-$(id -u) 2>/dev/null || true
    rm -rf /tmp/containers-user-$(id -u) 2>/dev/null || true
    rm -rf /tmp/podman-run-$(id -u) 2>/dev/null || true
    # Don't remove our current job's directory
    for d in /tmp/podman-${USER}-*; do
        if [[ "$d" != "${podman_root}" ]]; then
            rm -rf "$d" 2>/dev/null || true
        fi
    done
    rm -rf /tmp/containers-${USER} 2>/dev/null || true
    # Critical: Clean up libpod local files that can override storage driver settings
    # Clean up the ENTIRE ~/.local/share/containers directory to prevent any cached state
    rm -rf "${HOME}/.local/share/containers" 2>/dev/null || true

    # Reset podman to clean state
    log_info "Resetting podman to clean state..."
    podman system migrate 2>/dev/null || true
    podman system reset -f 2>/dev/null || true
    podman system prune -f -a 2>/dev/null || true

    # Create storage.conf for this job pointing to /tmp (local disk)
    # Use overlay driver for better performance (copy-on-write vs full copies)
    # /tmp is local storage so xattr should be supported
    cat > "${podman_root}/config/storage.conf" << EOF
[storage]
driver = "overlay"
graphroot = "${podman_root}/storage"
runroot = "${podman_root}/run"

[storage.options.overlay]
mount_program = "/usr/bin/fuse-overlayfs"
EOF

    # Point podman to our config
    export CONTAINERS_STORAGE_CONF="${podman_root}/config/storage.conf"
    export XDG_RUNTIME_DIR="${podman_root}/run"

    # Socket for this job
    export PODMAN_SOCKET="${podman_root}/podman.sock"

    # Store the root for cleanup
    export _PODMAN_ISOLATION_ROOT="${podman_root}"

    log_info "Podman isolation configured:"
    log_info "  CONTAINERS_STORAGE_CONF=${CONTAINERS_STORAGE_CONF}"
    log_info "  XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR}"
    log_info "  XDG_DATA_HOME=${XDG_DATA_HOME}"
    log_info "  PODMAN_SOCKET=${PODMAN_SOCKET}"
    log_info "  TMPDIR=${TMPDIR}"
}

# Clean up isolated podman storage
# Call this in cleanup functions or traps
cleanup_isolated_podman_storage() {
    local podman_root="${_PODMAN_ISOLATION_ROOT:-}"

    if [[ -z "${podman_root}" ]]; then
        log_debug "No podman isolation root set, skipping cleanup"
        return 0
    fi

    log_info "Cleaning up podman storage at ${podman_root}"

    # Stop all containers gracefully
    podman stop --all --time 5 2>/dev/null || true

    # Remove all containers
    podman rm --all --force 2>/dev/null || true

    # Clean up the isolated directory
    rm -rf "${podman_root}" 2>/dev/null || true

    log_info "Podman storage cleanup complete"
}

# ============================================================================
# CUDA/Triton Setup Functions
# ============================================================================

# Set up CUDA environment for Triton JIT compilation
# This function probes for CUDA libraries and sets necessary environment variables
setup_cuda_for_triton() {
    log_info "Setting up CUDA for Triton JIT compilation..."

    local cuda_lib=""

    # Check common locations for libcuda.so
    for path in "/usr/lib64/libcuda.so.1" "/usr/lib64/libcuda.so" \
                "/usr/local/cuda/lib64/stubs/libcuda.so" \
                "/usr/local/cuda/lib64/libcuda.so"; do
        if [[ -f "${path}" ]]; then
            cuda_lib="$(dirname "${path}")"
            log_info "Found CUDA library at ${path}"
            break
        fi
    done

    # Try module load as fallback
    if [[ -z "${cuda_lib}" ]] && command -v module &>/dev/null; then
        log_info "CUDA not found in standard paths, trying module load..."
        module load cuda/12.4.0-fasrc01 2>/dev/null || \
        module load cuda/12.2.0-fasrc01 2>/dev/null || \
        module load cuda 2>/dev/null || true

        # Check if module load provided CUDA_HOME
        if [[ -n "${CUDA_HOME:-}" ]] && [[ -d "${CUDA_HOME}/lib64" ]]; then
            cuda_lib="${CUDA_HOME}/lib64"
            log_info "CUDA module loaded, using ${cuda_lib}"
        fi
    fi

    if [[ -n "${cuda_lib}" ]]; then
        export TRITON_LIBCUDA_PATH="${cuda_lib}"
        export LD_LIBRARY_PATH="${cuda_lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        export LIBRARY_PATH="${cuda_lib}${LIBRARY_PATH:+:$LIBRARY_PATH}"

        # Create symlink for Triton's hardcoded /lib64 lookup if needed
        if [[ ! -f "/lib64/libcuda.so" ]] && [[ -w "/tmp" ]]; then
            mkdir -p /tmp/triton_cuda_lib
            local libcuda_src=""
            if [[ -f "${cuda_lib}/libcuda.so.1" ]]; then
                libcuda_src="${cuda_lib}/libcuda.so.1"
            elif [[ -f "${cuda_lib}/libcuda.so" ]]; then
                libcuda_src="${cuda_lib}/libcuda.so"
            fi
            if [[ -n "${libcuda_src}" ]]; then
                ln -sf "${libcuda_src}" /tmp/triton_cuda_lib/libcuda.so 2>/dev/null || true
                export LIBRARY_PATH="/tmp/triton_cuda_lib:${LIBRARY_PATH}"
            fi
        fi

        log_info "CUDA configured for Triton:"
        log_info "  TRITON_LIBCUDA_PATH=${TRITON_LIBCUDA_PATH}"
        log_info "  LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"
    else
        log_warn "Could not find CUDA libraries - Triton JIT may fail"
        # Set fallback to disable Triton-based ops
        export VERL_DISABLE_FLASH_ATTN_CE=1
        log_info "Set VERL_DISABLE_FLASH_ATTN_CE=1 as fallback"
    fi
}
