#!/bin/bash
#SBATCH --job-name=skyrl-runtime
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=${SLURM_PARTITION:-test}
#SBATCH --nodes=1
#SBATCH --cpus-per-task=${CPUS_FOR_RUNTIME:-64}
#SBATCH --mem=${RUNTIME_MEMORY:-256G}
#SBATCH --time=${TIME_LIMIT:-12:00:00}
#SBATCH --exclude=${EXCLUDE_NODES:-}

# ============================================================================
# SkyRL Runtime Server Job
# ============================================================================
# Runs slurm-remote-runtime API server providing sandboxed Docker containers
# for agent code execution. This job should be started before the GPU training job.
#
# Required Environment Variables:
#   SKYRL_COORD_DIR - Path to coordination directory (set by launch.sh)
#
# Optional Environment Variables (from config):
#   CPUS_FOR_RUNTIME - Number of CPUs (default: 64)
#   CPUS_PER_WORKER - CPUs per agent container (default: 8)
#   RUNTIME_MEMORY - Total memory for runtime job (default: 256G)
#   RUNTIME_PORT - Port for API server (default: 8000)
#   CONTAINER_IMAGE - Docker image for agent containers
#   ALLHANDS_API_KEY - API key for runtime authentication
# ============================================================================

set -euo pipefail

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

# ============================================================================
# Configuration
# ============================================================================

# Runtime configuration
RUNTIME_PORT="${RUNTIME_PORT:-8000}"
CPUS_PER_WORKER="${CPUS_PER_WORKER:-8}"
MEMORY_PER_WORKER="${MEMORY_PER_WORKER:-16G}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-ghcr.io/opendevin/sandbox:main}"

# Paths
SLURM_RUNTIME_DIR="${SKYRL_PROJECT_ROOT}/slurm-remote-runtime"

# Disable API authentication for simplicity in SLURM environment
# Both runtime and training jobs run in same trusted environment
DISABLE_AUTH=true

# ============================================================================
# Validation
# ============================================================================

if [[ -z "${SKYRL_COORD_DIR:-}" ]]; then
    log_error "SKYRL_COORD_DIR not set. This job should be launched via launch.sh"
    exit 1
fi

if [[ ! -d "${SLURM_RUNTIME_DIR}" ]]; then
    log_error "slurm-remote-runtime directory not found: ${SLURM_RUNTIME_DIR}"
    exit 1
fi

# ============================================================================
# Setup Functions
# ============================================================================

cleanup() {
    log_info "Cleaning up runtime server..."

    # Kill any running API server processes
    if [[ -n "${API_SERVER_PID:-}" ]]; then
        kill "${API_SERVER_PID}" 2>/dev/null || true
        wait "${API_SERVER_PID}" 2>/dev/null || true
    fi

    # Stop Podman service if running
    if [[ -n "${PODMAN_SERVICE_PID:-}" ]]; then
        kill "${PODMAN_SERVICE_PID}" 2>/dev/null || true
        wait "${PODMAN_SERVICE_PID}" 2>/dev/null || true
    fi

    # Clean up isolated podman storage
    cleanup_isolated_podman_storage

    log_info "Runtime server cleanup complete"
}

# ============================================================================
# Main
# ============================================================================

main() {
    log_info "Starting SkyRL Runtime Server"
    log_info "Job ID: ${SLURM_JOB_ID}"
    log_info "Node: $(hostname)"
    log_info "Coordination dir: ${SKYRL_COORD_DIR}"

    # Set up graceful shutdown
    setup_graceful_shutdown cleanup

    # Load environment file if present
    local env_file="${SKYRL_PROJECT_ROOT}/SkyRL/skyrl-agent/.env"
    if [[ -f "${env_file}" ]]; then
        load_env_file "${env_file}"
    fi

    # Change to slurm-remote-runtime directory
    cd "${SLURM_RUNTIME_DIR}"

    # Activate virtual environment (slurm-remote-runtime has its own)
    if [[ -d "${SLURM_RUNTIME_DIR}/.venv" ]]; then
        source "${SLURM_RUNTIME_DIR}/.venv/bin/activate"
        log_info "Activated slurm-remote-runtime venv"
    else
        log_warn "No venv found, using system Python"
    fi

    # Set up isolated podman storage for this job
    # This prevents state corruption between SLURM jobs by using job-specific directories
    setup_isolated_podman_storage

    # Use the job-specific socket from isolation setup
    local podman_socket="${PODMAN_SOCKET}"

    # Clean up stale worker logs from previous jobs
    rm -f /scratch/worker_*.log 2>/dev/null || true
    rm -f "${SCRATCH:-/scratch}/worker_*.log" 2>/dev/null || true

    # Start podman system service in background (--time=0 means no timeout)
    podman system service --time=0 "unix://${podman_socket}" &
    PODMAN_SERVICE_PID=$!
    sleep 3  # Give it time to start (increased from 2s)

    if [[ -S "${podman_socket}" ]]; then
        export DOCKER_HOST="unix://${podman_socket}"
        log_info "Started Podman service at ${podman_socket}"
        log_info "Set DOCKER_HOST=${DOCKER_HOST}"

        # Verify Podman is actually working by listing images
        if podman images >/dev/null 2>&1; then
            log_info "Podman verification successful"
        else
            log_warn "Podman socket exists but 'podman images' failed - image building may have issues"
        fi

        # Authenticate with Docker Hub for pulling private r2e images
        local docker_token_file="${HOME}/dockerhub_token.txt"
        if [[ -f "${docker_token_file}" ]]; then
            # Strip newlines and authenticate
            tr -d '\n' < "${docker_token_file}" | podman login -u 18jeffreyma --password-stdin docker.io && \
                log_info "Authenticated with Docker Hub" || \
                log_warn "Failed to authenticate with Docker Hub"
        else
            log_warn "Docker token file not found at ${docker_token_file}, private images may fail to pull"
        fi
    else
        log_warn "Failed to start Podman service (socket not created), image building may not work"
        log_warn "Attempting to continue anyway - pre-built images may still work"
    fi

    # Determine the API URL
    local hostname
    hostname=$(hostname -f)
    local api_url="http://${hostname}:${RUNTIME_PORT}"

    log_info "Runtime API will be available at: ${api_url}"
    log_info "Authentication: DISABLED (trusted SLURM environment)"

    # Write the API URL to coordination file
    write_runtime_url "${SKYRL_COORD_DIR}" "${api_url}"

    # Start the API server
    log_info "Starting slurm-remote-runtime API server..."
    log_info "  Container image: ${CONTAINER_IMAGE}"
    log_info "  CPUs per worker: ${CPUS_PER_WORKER}"
    log_info "  Memory per worker: ${MEMORY_PER_WORKER}"

    # Run the API server in the background (no auth required in trusted SLURM environment)
    # Image building is enabled by default to handle base images that need openhands-agent-server
    "${SLURM_RUNTIME_DIR}/.venv/bin/python" -m slurm_runtime.api_server \
        --container-image "${CONTAINER_IMAGE}" \
        --host "0.0.0.0" \
        --port "${RUNTIME_PORT}" \
        --disable-auth \
        --build-reserved-cpus 4 \
        --cpus-per-worker "${CPUS_PER_WORKER}" \
        --memory-per-worker "${MEMORY_PER_WORKER}" \
        --cpu-pinning-method docker_cpuset \
        --worker-startup-timeout 600 \
        --verbose &

    API_SERVER_PID=$!
    log_info "API server started with PID: ${API_SERVER_PID}"

    # Wait for the server to be ready
    log_info "Waiting for API server to become ready..."
    local ready_timeout=60
    local elapsed=0

    while [[ $elapsed -lt $ready_timeout ]]; do
        if curl -s "http://localhost:${RUNTIME_PORT}/health" > /dev/null 2>&1; then
            log_info "API server is ready!"
            break
        fi

        # Check if process is still running
        if ! kill -0 "${API_SERVER_PID}" 2>/dev/null; then
            log_error "API server process died unexpectedly"
            exit 1
        fi

        sleep 2
        elapsed=$((elapsed + 2))
    done

    if [[ $elapsed -ge $ready_timeout ]]; then
        log_error "API server failed to become ready within ${ready_timeout}s"
        cleanup
        exit 1
    fi

    # Signal that runtime is ready
    signal_runtime_ready "${SKYRL_COORD_DIR}"

    log_info "Runtime server is running and ready"
    log_info "GPU training job can now connect to: ${api_url}"

    # Wait for training to complete or timeout
    # The training job will signal completion, or we'll timeout
    if wait_for_training_complete "${SKYRL_COORD_DIR}"; then
        log_info "Training completed successfully"
    else
        log_warn "Training did not signal completion (timeout or error)"
    fi

    # Cleanup
    cleanup
    log_info "Runtime server job finished"
}

main "$@"
