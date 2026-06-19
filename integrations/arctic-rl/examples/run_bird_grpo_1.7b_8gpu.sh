#!/usr/bin/env bash
# SkyRL + Arctic RL backend: Qwen3-1.7B BIRD-SQL GRPO recipe.
#
# Mirrors Snowflake-AI-Research/arctic-verl xid2pl9f -- the verl
# converged-reference BIRD-1.7B run with ``remote_backend.colocate=True``:
# 269 steps, val reward ~0.59 (exec 0.93 / format 0.99), `ppo_kl=0` and
# `pg_clipfrac=0` every step. Source recipe:
# `<PATH>/launch_1.7b_newshape.sh`.
#
# Key memory knobs propagated to the Arctic DeepSpeed worker (without
# these the colocated training + vLLM share OOMs on H200 at 96K-token
# packed sequences -- see arctic_platform/rl/deepspeed_worker.py:147,155,202):
#   - attn_implementation=flash_attention_3   O(N) attention memory
#   - use_liger=true                          fused MLP/RMSNorm
#   - enable_gradient_checkpointing=true      O(sqrt N) activations
#   - ulysses_sequence_parallel_size=2        per-seq compute split 2-way
#   - logits_optimization=memory              chunked LM-head compute
#   - cuda_ipc_weight_sync=true               zero-copy weight push back to vLLM
#
# Toggle to other backends is one CLI flag (`trainer.backend=fsdp` for
# stock SkyRL, `trainer.backend=arctic_rl` for this integration). No
# PYTHONPATH gymnastics -- main_base discovers `integrations/arctic-rl/`
# via `_ensure_backend_importable`.
#
# Step-1 invariants we expect (matching xid2pl9f):
#   - actor/ppo_kl == 0           (single PPO epoch, single mini-batch:
#                                  old_log_probs == log_probs by construction)
#   - actor/pg_clipfrac == 0      (no clipping triggers when ratio == 1)
#   - actor/pg_clipfrac_lower == 0
#   - actor/loss == actor/pg_loss (no entropy/KL terms, both off in this recipe)
# Absolute loss / grad_norm magnitudes will differ run-to-run because
# rollouts differ; the invariants above are the deterministic baseline.

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
    trainer.arctic_rl.use_zorro=true \
    trainer.arctic_rl.use_liger=true \
    trainer.arctic_rl.attn_implementation=flash_attention_3 \
    trainer.arctic_rl.enable_gradient_checkpointing=true \
    trainer.arctic_rl.ulysses_sequence_parallel_size=2 \
    trainer.arctic_rl.logits_optimization=memory \
    trainer.arctic_rl.cuda_ipc_weight_sync=true \
    trainer.arctic_rl.lr_warmup_ratio=0.05 \
    'trainer.arctic_rl.optimizer_betas=[0.9,0.95]' \
    trainer.arctic_rl.vllm_enforce_eager=false \
    trainer.arctic_rl.vllm_enable_prefix_caching=true \
    trainer.arctic_rl.vllm_max_num_batched_tokens=40960 \
    trainer.arctic_rl.use_arctic_inference=true \
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
