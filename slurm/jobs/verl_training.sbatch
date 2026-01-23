#!/bin/bash
#SBATCH --job-name=skyrl-train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=${GPU_PARTITION:-gpu_test}
#SBATCH --nodes=${GPU_NODES:-1}
#SBATCH --gres=gpu:${GPUS_PER_NODE:-4}
#SBATCH --cpus-per-task=${TRAINING_CPUS:-48}
#SBATCH --mem=${TRAINING_MEMORY:-500G}
#SBATCH --time=${TIME_LIMIT:-12:00:00}

# ============================================================================
# SkyRL VERL Training Job
# ============================================================================
# Runs VERL PPO training with vLLM inference backend. Connects to the
# runtime server job for sandboxed code execution.
#
# Required Environment Variables:
#   SKYRL_COORD_DIR - Path to coordination directory (set by launch.sh)
#
# Optional Environment Variables (from config):
#   GPU_NODES - Number of GPU nodes (default: 1)
#   GPUS_PER_NODE - GPUs per node (default: 4)
#   MODEL - Model path/name (default: Qwen/Qwen3-8B)
#   TP_SIZE - Tensor parallel size (default: 2)
#   SP_SIZE - Sequence parallel size (default: 2)
#   TRAIN_BATCH_SIZE - Training batch size (default: 64)
#   ROLLOUT_N - Number of rollouts per prompt (default: 8)
# ============================================================================

set -euo pipefail

# Source common utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"

# Note: CUDA/Triton setup is now handled by setup_cuda_for_triton() in main()
# This centralizes CUDA library detection and configuration

# ============================================================================
# Configuration
# ============================================================================

# Model configuration
MODEL="${MODEL:-Qwen/Qwen3-8B}"
TP_SIZE="${TP_SIZE:-2}"
SP_SIZE="${SP_SIZE:-2}"

# Training configuration
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-8000}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-32768}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-15}"

# Data paths
DATA_DIR="${DATA_DIR:-/mnt/shared_storage/datasets/r2e-all}"
TRAIN_DATA="${TRAIN_DATA:-${DATA_DIR}/train.parquet}"
VAL_DATA="${VAL_DATA:-${DATA_DIR}/validation.parquet}"

# Output directories
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/mnt/local_storage/ckpts/skyrl}"
ROLLOUT_DIR="${ROLLOUT_DIR:-/mnt/local_storage/rollouts/skyrl}"

# WandB configuration
WANDB_PROJECT="${WANDB_PROJECT:-skyrl-training}"
WANDB_EXPERIMENT="${WANDB_EXPERIMENT:-skyrl-$(date +%Y%m%d)}"

# Paths
SKYRL_AGENT_DIR="${SKYRL_PROJECT_ROOT}/SkyRL/skyrl-agent"

# ============================================================================
# Validation
# ============================================================================

if [[ -z "${SKYRL_COORD_DIR:-}" ]]; then
    log_error "SKYRL_COORD_DIR not set. This job should be launched via launch.sh"
    exit 1
fi

if [[ ! -d "${SKYRL_AGENT_DIR}" ]]; then
    log_error "skyrl-agent directory not found: ${SKYRL_AGENT_DIR}"
    exit 1
fi

# ============================================================================
# Setup Functions
# ============================================================================

cleanup() {
    log_info "Cleaning up training job..."

    # Signal training completion to runtime job
    if [[ -n "${SKYRL_COORD_DIR:-}" ]]; then
        signal_training_complete "${SKYRL_COORD_DIR}"
    fi

    log_info "Training cleanup complete"
}

# ============================================================================
# Main
# ============================================================================

main() {
    log_info "Starting SkyRL VERL Training Job"
    log_info "Job ID: ${SLURM_JOB_ID}"
    log_info "Nodes: ${SLURM_NNODES:-1}"
    log_info "GPUs per node: ${GPUS_PER_NODE:-4}"
    log_info "Coordination dir: ${SKYRL_COORD_DIR}"

    # Set up graceful shutdown
    setup_graceful_shutdown cleanup

    # Set up CUDA for Triton JIT compilation (before any Python runs)
    setup_cuda_for_triton

    # Wait for runtime URL
    log_info "Waiting for runtime server to be ready..."
    local runtime_url
    runtime_url=$(wait_for_runtime_url "${SKYRL_COORD_DIR}" "${RUNTIME_READY_TIMEOUT:-600}")

    if [[ -z "${runtime_url}" ]]; then
        log_error "Failed to get runtime URL"
        exit 1
    fi

    # Read the clean runtime URL directly from file (avoid log contamination)
    local clean_runtime_url
    clean_runtime_url=$(cat "${SKYRL_COORD_DIR}/runtime_url.txt")
    log_info "Runtime server ready at: ${clean_runtime_url}"

    # Read API key if available
    local api_key_file="${SKYRL_COORD_DIR}/runtime_api_key.txt"
    if [[ -f "${api_key_file}" ]]; then
        export ALLHANDS_API_KEY
        ALLHANDS_API_KEY=$(cat "${api_key_file}")
        log_info "Loaded runtime API key"
    fi

    # Set the runtime URL for OpenHands
    export SANDBOX_REMOTE_RUNTIME_API_URL="${clean_runtime_url}"
    log_info "Set SANDBOX_REMOTE_RUNTIME_API_URL=${clean_runtime_url}"

    # Change to skyrl-agent directory
    cd "${SKYRL_AGENT_DIR}"

    # Load environment file
    local base_env_file="${SKYRL_AGENT_DIR}/.env"
    local env_file="${SKYRL_COORD_DIR}/training.env"

    # Create a combined env file with runtime URL and API key
    # This ensures the environment variables are available to uv run subprocess
    if [[ -f "${base_env_file}" ]]; then
        cp "${base_env_file}" "${env_file}"
    else
        touch "${env_file}"
    fi

    # Add runtime URL to env file - read directly from file to avoid log contamination
    local runtime_url_from_file
    runtime_url_from_file=$(cat "${SKYRL_COORD_DIR}/runtime_url.txt")
    echo "" >> "${env_file}"
    echo "# Runtime configuration (added by verl_training.sbatch)" >> "${env_file}"
    echo "SANDBOX_REMOTE_RUNTIME_API_URL=${runtime_url_from_file}" >> "${env_file}"

    # Add API key if available
    local api_key_file="${SKYRL_COORD_DIR}/runtime_api_key.txt"
    if [[ -f "${api_key_file}" ]]; then
        echo "ALLHANDS_API_KEY=$(cat "${api_key_file}")" >> "${env_file}"
    fi

    log_info "Created combined env file at ${env_file}"

    # Load environment file
    if [[ -f "${base_env_file}" ]]; then
        load_env_file "${base_env_file}"
    fi

    # Override with runtime URL from coordination (use clean URL from file)
    export SANDBOX_REMOTE_RUNTIME_API_URL="${clean_runtime_url}"

    # Set vLLM configuration
    export VLLM_USE_V1="${VLLM_USE_V1:-1}"

    # Handle MIG GPU devices - vLLM requires integer device IDs
    # SLURM may set CUDA_VISIBLE_DEVICES to MIG UUIDs like "MIG-xxx,MIG-yyy"
    # Convert to sequential integers "0,1,..." for vLLM compatibility
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        if [[ "${CUDA_VISIBLE_DEVICES}" == *"MIG-"* ]]; then
            local num_gpus
            num_gpus=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
            local new_devices
            new_devices=$(seq -s',' 0 $((num_gpus - 1)))
            log_info "MIG GPUs detected, remapping CUDA_VISIBLE_DEVICES from UUIDs to integers: ${new_devices}"
            export CUDA_VISIBLE_DEVICES="${new_devices}"
        fi
    fi

    # Set up EFA networking if enabled
    setup_efa_networking

    # Log configuration
    log_info "Training Configuration:"
    log_info "  Model: ${MODEL}"
    log_info "  TP Size: ${TP_SIZE}"
    log_info "  SP Size: ${SP_SIZE}"
    log_info "  Batch Size: ${TRAIN_BATCH_SIZE}"
    log_info "  Rollout N: ${ROLLOUT_N}"
    log_info "  Train Data: ${TRAIN_DATA}"
    log_info "  Val Data: ${VAL_DATA}"
    log_info "  Checkpoint Dir: ${CHECKPOINT_DIR}"
    log_info "  WandB Project: ${WANDB_PROJECT}"

    # Create output directories
    mkdir -p "${CHECKPOINT_DIR}" "${ROLLOUT_DIR}"

    # Run VERL training
    log_info "Starting VERL PPO training..."

    uv run --frozen --extra verl --env-file "${env_file}" \
        -m skyrl_agent.integrations.verl.verl_main_ppo \
        algorithm.adv_estimator=loop \
        data.train_files="${TRAIN_DATA}" \
        data.val_files="${VAL_DATA}" \
        data.dataloader_num_workers=0 \
        data.train_batch_size="${TRAIN_BATCH_SIZE}" \
        data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
        data.max_response_length="${MAX_RESPONSE_LENGTH}" \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.return_raw_chat=true \
        actor_rollout_ref.model.path="${MODEL}" \
        actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
        actor_rollout_ref.actor.use_dynamic_bsz=False \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.clip_ratio_high=0.28 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}" \
        actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.mode=async \
        actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.8}" \
        actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN:-32768}" \
        actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=False \
        algorithm.norm_adv_by_std_in_grpo=False \
        trainer.val_before_train=False \
        trainer.critic_warmup=0 \
        trainer.logger='["console","wandb"]' \
        trainer.project_name="${WANDB_PROJECT}" \
        trainer.experiment_name="${WANDB_EXPERIMENT}" \
        trainer.n_gpus_per_node="${GPUS_PER_NODE:-8}" \
        trainer.nnodes="${SLURM_NNODES:-1}" \
        trainer.max_actor_ckpt_to_keep=10 \
        trainer.save_freq=1 \
        trainer.default_local_dir="${CHECKPOINT_DIR}/${WANDB_EXPERIMENT}" \
        trainer.test_freq=20 \
        trainer.total_epochs="${TOTAL_EPOCHS}" \
        trainer.rollout_data_dir="${ROLLOUT_DIR}/train" \
        trainer.validation_data_dir="${ROLLOUT_DIR}/val" \
        +skyrl_agent.task_yaml="./examples/run_verl/verl_oh.yaml" \
        +skyrl_agent.num_trajectories="${ROLLOUT_N}"

    local exit_code=$?

    # Signal training completion
    signal_training_complete "${SKYRL_COORD_DIR}"

    if [[ $exit_code -eq 0 ]]; then
        log_info "Training completed successfully"
    else
        log_error "Training failed with exit code: ${exit_code}"
    fi

    log_info "Training job finished"
    exit $exit_code
}

main "$@"
