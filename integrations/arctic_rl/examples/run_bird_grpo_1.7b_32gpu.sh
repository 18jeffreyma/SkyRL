#!/usr/bin/env bash
# SkyRL + Arctic RL backend: Qwen3-1.7B BIRD-SQL GRPO recipe — 4 nodes / 32 H200s.
#
# Multi-node scale-out of the converged single-node recipe in
# ``run_bird_grpo_1.7b_8gpu.sh``. Global batch geometry is kept *identical*
# to the 1-node run (32 prompts × 16 samples = 512 trajectories / step) so
# the gradient at each optimizer step is the same as st3ue30x / xid2pl9f
# (up to floating-point reduction order). Per-DP-rank work shrinks 4x,
# so wall-clock per step is expected to drop from ~180–200 s to ~50–70 s.
#
# DP math (validated in arctic_rl.config.build_rl_config):
#   training_gpus      = policy_num_gpus_per_node * policy_num_nodes = 32
#   ulysses_sp         = 2 -> dp_world = 32 / 2 = 16
#   mini_batch (global)= train_batch_size * n_samples = 32 * 16 = 512
#   mini_per_dp        = 512 / 16 = 32
#   micro_per_gpu      = n_samples = 16   (ZoRRo: each microbatch == one prompt group)
#   grad_accum         = 32 / 16 = 2
#   DS assertion:        16 * 2 * 16 = 512 == mini_batch  ✓
#
# Prereq: ray cluster already running across all 4 nodes (head + 3 workers).
# ``ray status`` should show ``32.0/32.0 GPU`` total. The Arctic client
# pre-init in arctic_rl/entrypoint.py calls ``ray.init(ignore_reinit_error=True)``
# and reuses the existing cluster.

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

# Bypass the strict-weight-sync name check (Qwen3-1.7B tie_word_embeddings=True).
export ARCTIC_WEIGHT_SYNC_STRICT_NAMES=0

# WandB
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://<REDACTED_INTERNAL_URL>}"
export WANDB_API_KEY="${WANDB_API_KEY:-<REDACTED_WANDB_KEY>}"
export WANDB_PROJECT="${WANDB_PROJECT:-arctic_rl_bird_sql}"
export WANDB_DISABLE_CODE=True

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
MODEL=${MODEL:-"Qwen/Qwen3-1.7B"}
EXPERIMENT_NAME=skyrl_bird_grpo_${MODEL##*/}_4node_${RUN_TS}
# Multi-node weight sync stages files under ``${CHECKPOINT_DIR}/ckpt/arctic_rl_job_*/weight_sync.pt``.
# The Arctic IPC weight-sync path on the head writes the file, then all 32
# vLLM InferenceWorkers across the 4 nodes mmap-read it. ``${HOME}/ckpts``
# is local SSD per node, so workers can't see it -- use Lustre (/data) which
# is mounted on all 4 nodes.
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/data/skyrl-runs/ckpts/${EXPERIMENT_NAME}}
mkdir -p "${CHECKPOINT_DIR}"

# 4 nodes × 8 H200s each.
NUM_NODES=4
GPUS_PER_NODE=8
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))   # = 32

# FCA + PIECEWISE cudagraph -- same arctic_inference_config plumbing as the 32B
# launcher so this run smoke-tests the client-side translation we ship for 32B.
# No speculative draft for 1.7B (no compatible spec checkpoint).
#
# pass_config.fuse_allreduce_rms=false: vLLM's fused AllReduce+RMSNorm pass uses
# FlashInfer's per-process IPC workspace, which collides between same-node
# replicas (TP>1 colocated). It's a no-op at TP=1 so disabling it here is free,
# and we match the 32B path bit-for-bit.
USE_FCA=${USE_FCA:-True}
AI_CFG_PARTS=()
if [[ "${USE_FCA}" == "True" ]]; then
    AI_CFG_PARTS+=('forest_cascade_attn_configs: "{}"')
    AI_CFG_PARTS+=('compilation_config: {cudagraph_mode: PIECEWISE, pass_config: {fuse_allreduce_rms: false}}')
fi
AI_CFG_OVERRIDE=()
if (( ${#AI_CFG_PARTS[@]} > 0 )); then
    IFS=, AI_CFG_BODY="${AI_CFG_PARTS[*]}" ; unset IFS
    AI_CFG_OVERRIDE+=("trainer.arctic_rl.arctic_inference_config={${AI_CFG_BODY}}")
fi

# Same global batch as st3ue30x / xid2pl9f.
TRAIN_BSZ=32
MINI_BSZ=32
N_SAMPLES=16
LR=2e-6
PROMPT_LEN=32768
RESPONSE_LEN=4096

cd "${SKYRL_DIR}"

"${PYBIN}" -m skyrl.train.entrypoints.main_base \
    trainer.override_entrypoint=integrations.arctic_rl.entrypoint \
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
    trainer.arctic_rl.vllm_max_num_seqs=256 \
    trainer.arctic_rl.low_memory_weight_sync=false \
    trainer.arctic_rl.use_arctic_inference=true \
    trainer.arctic_rl.server_logs=true \
    trainer.arctic_rl.startup_timeout=1800 \
    "${AI_CFG_OVERRIDE[@]}" \
    data.train_data="['${DATA_DIR}/train.parquet']" \
    data.val_data="['${DATA_DIR}/val.parquet']" \
    trainer.algorithm.advantage_estimator=grpo \
    trainer.policy.model.path="${MODEL}" \
    trainer.placement.colocate_all=false \
    trainer.placement.policy_num_gpus_per_node=${GPUS_PER_NODE} \
    trainer.placement.policy_num_nodes=${NUM_NODES} \
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
    generator.eval_sampling_params.max_generate_length=${RESPONSE_LEN} \
    generator.eval_sampling_params.temperature=0.0 \
    generator.eval_sampling_params.top_p=1.0 \
    generator.eval_sampling_params.top_k=-1 \
    generator.eval_n_samples_per_prompt=1 \
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
