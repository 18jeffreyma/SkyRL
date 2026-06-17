#!/usr/bin/env bash
# SkyRL + Arctic RL backend: Qwen3-1.7B BIRD-SQL GRPO recipe.
#
# Mirrors Snowflake-AI-Research/verl PR #6 (run id mafx904c, wandb
# `karthik-ganesan/arctic_rl_bird_sql/mafx904c`) hyperparameters so the
# resulting SkyRL step-1 metrics can be compared 1:1 with the validated
# verl baseline.
#
# Toggle to other backends is one CLI flag (`trainer.backend=fsdp` for
# stock SkyRL, `trainer.backend=arctic_rl` for this integration). No
# PYTHONPATH gymnastics — main_base discovers `integrations/arctic-rl/`
# via `_ensure_backend_importable`.
#
# Mirrors verl's:
#   - 8 GPUs colocated (training + sampling on the same GPUs via Arctic RL
#     server-side colocation)
#   - Qwen3-1.7B, BIRD parquet at <PATH>/open-source-text2sql/
#   - GRPO, no KL loss, no KL in reward, entropy_coeff=0, clip_ratio=0.2
#   - 32 prompts/batch × 16 rollouts (=512 trajectories/step), prompt len 32K,
#     response len 4K, lr=2e-6, ZeRO-3 with optimizer offload, 1 epoch
#
# Wire-protocol caveat: this PR's integration shim is `arctic_platform.rl`-
# compatible but does NOT yet ship the verl-shape meta dict / verl_grpo loss
# / post-processors / tied-embeddings weight-sync fix. Those fixes belong in
# `arctic_platform.rl` (see PR there). Expect SkyRL step-1 *invariants* to
# match verl (clipfrac=0, ppo_kl=0 on epoch 1) and the *shape* of the metric
# dict to overlap; absolute loss / grad_norm values will drift until the
# `arctic_platform.rl` PR lands.

set -euxo pipefail

SKYRL_DIR=<PATH>/sky-checkouts/SkyRL
DATA_DIR=${DATA_DIR:-"<PATH>/open-source-text2sql"}
PYBIN=/home/yak/miniconda3/envs/skyrl_v1/bin/python

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME="${HF_HOME:-<PATH>}"
export HF_HUB_OFFLINE=1
export TORCH_COMPILE_DISABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_CACHE_ROOT=<PATH>/vllm
export VLLM_LOGGING_LEVEL=INFO
export ARCTIC_CUDA_IPC_LOW_MEM=0

# Bypass the strict-weight-sync name check (Qwen3-1.7B has
# tie_word_embeddings=True, so DeepSpeed ships lm_head.weight while vLLM
# dedupes it). The clean fix lives in arctic_platform.rl; this env var is the
# temporary bypass until that PR lands.
export ARCTIC_WEIGHT_SYNC_STRICT_NAMES=0

# WandB (same project as verl PR #6 so the runs sit side-by-side)
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://<REDACTED_INTERNAL_URL>}"
export WANDB_API_KEY="${WANDB_API_KEY:-<REDACTED_WANDB_KEY>}"
export WANDB_PROJECT="${WANDB_PROJECT:-arctic_rl_bird_sql}"
export WANDB_DISABLE_CODE=True

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
MODEL=${MODEL:-"Qwen/Qwen3-1.7B"}
EXPERIMENT_NAME=skyrl_bird_grpo_${MODEL##*/}_${RUN_TS}
CHECKPOINT_DIR=${HOME}/ckpts/${EXPERIMENT_NAME}
mkdir -p "${CHECKPOINT_DIR}"

# Mirror verl PR #6: 8 GPUs, all colocated (training + sampling share the
# same GPUs via Arctic RL server-side colocation).
NUM_GPUS=8

# Verl PR #6: BSZ_PER_GPU=4, ROLL_N=16 -> BSZ=32 prompts/batch, 512 trajectories.
# SkyRL math (validate_batch_sizes):
#   train_per_gpu = TRAIN_BSZ * N_SAMPLES / NUM_GPUS = 32 * 16 / 8 = 64
#   mini_per_gpu  = MINI_BSZ  * N_SAMPLES / NUM_GPUS = 32 * 16 / 8 = 64
#   grad_accum    = 64 / 64 = 1  (matches verl's PPO_MINI_BSZ_PER_GPU == BSZ_PER_GPU)
TRAIN_BSZ=32
MINI_BSZ=32
N_SAMPLES=16
LR=2e-6
PROMPT_LEN=32768
RESPONSE_LEN=4096

cd "${SKYRL_DIR}"

"${PYBIN}" -m skyrl.train.entrypoints.main_base \
    trainer.backend=arctic_rl \
    trainer.arctic_rl={} \
    trainer.arctic_rl.colocate=true \
    trainer.arctic_rl.zero_stage=3 \
    trainer.arctic_rl.offload_optimizer=true \
    trainer.arctic_rl.log_prob_gpus=0 \
    trainer.arctic_rl.server_logs=true \
    trainer.arctic_rl.startup_timeout=1800 \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/val.parquet']" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.policy.model.path="${MODEL}" \
    trainer.placement.colocate_all=false \
    trainer.placement.policy_num_gpus_per_node=${NUM_GPUS} \
    trainer.placement.policy_num_nodes=1 \
    generator.inference_engine.num_engines=${NUM_GPUS} \
    generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.gpu_memory_utilization=0.5 \
    generator.inference_engine.async_engine=true \
    generator.batched=true \
    trainer.epochs=1 \
    trainer.eval_batch_size=32 \
    trainer.eval_before_train=false \
    trainer.eval_interval=10 \
    trainer.update_epochs_per_batch=1 \
    trainer.train_batch_size=${TRAIN_BSZ} \
    trainer.policy_mini_batch_size=${MINI_BSZ} \
    trainer.max_prompt_length=${PROMPT_LEN} \
    generator.sampling_params.max_generate_length=${RESPONSE_LEN} \
    generator.sampling_params.temperature=1.0 \
    generator.sampling_params.top_p=1.0 \
    trainer.policy.optimizer_config.lr=${LR} \
    trainer.policy.optimizer_config.max_grad_norm=1.0 \
    trainer.algorithm.use_kl_loss=false \
    trainer.algorithm.use_kl_in_reward=false \
    environment.env_class=bird \
    generator.n_samples_per_prompt=${N_SAMPLES} \
    trainer.logger=wandb \
    trainer.project_name="${WANDB_PROJECT}" \
    trainer.run_name="${EXPERIMENT_NAME}" \
    trainer.resume_mode=null \
    trainer.log_path="${CHECKPOINT_DIR}/logs" \
    trainer.ckpt_path="${CHECKPOINT_DIR}/ckpt" \
    trainer.ckpt_interval=-1 \
    "$@"
