#!/usr/bin/env bash
# SkyRL + Arctic RL backend: Qwen3-32B BIRD-SQL GRPO — 4 nodes / 32 H200s.
#
# Topology: NUM_NODES=4, GPUS_PER_NODE=8 (DP=32). vLLM TP=4, num_engines=8.
# train_batch=128 prompts x n_samples=16 = 2048 trajectories/step.
# Sibling: run_bird_grpo_32b_32gpu_fsdp.sh (same recipe, SkyRL FSDP-native).
#
# Prereq: 4-node ray cluster up on skyrl_v1 env; `ray status` shows 32/32 GPU.

set -euxo pipefail

SKYRL_DIR=${SKYRL_DIR:-<PATH>/sky-checkouts/SkyRL}
DATA_DIR=${DATA_DIR:-"<PATH>/open-source-text2sql"}
PYBIN=${PYBIN:-/home/yak/miniconda3/envs/skyrl_v1/bin/python}

export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export HF_HOME="${HF_HOME:-<PATH>}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_COMPILE_DISABLE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_CACHE_ROOT=<PATH>/vllm
export VLLM_LOGGING_LEVEL=INFO
# Match Tunji runtime behavior explicitly (his logs show vLLM selecting FLASH_ATTN).
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export ARCTIC_CUDA_IPC_LOW_MEM=0
# 32B + tie_word_embeddings is False, but keep the bypass on — it's a no-op
# when names match and a safety net if upstream Qwen3 adds new tied buffers.
export ARCTIC_WEIGHT_SYNC_STRICT_NAMES=0
# verl 32B recipe ships this; helps with the 32B optimizer-state CPU offload churn.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# WandB — same project as Tunji's verl 32B recipe so the runs sit side-by-side
# in skyrl_arctic_rl; override with PROJECT_NAME if you want to log to the
# arctic_rl_bird_sql project instead.
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://<REDACTED_INTERNAL_URL>}"
export WANDB_API_KEY="${WANDB_API_KEY:-<REDACTED_WANDB_KEY>}"
export WANDB_PROJECT="${WANDB_PROJECT:-skyrl_arctic_rl}"
export WANDB_DISABLE_CODE=True

# Resolve local Qwen3-32B HF snapshot — same dance as Tunji's verl recipe.
HF_REPO="models--Qwen--Qwen3-32B"
MODEL_REPO_DIR="${HF_HOME}/hub/${HF_REPO}"
if [[ ! -f "${MODEL_REPO_DIR}/refs/main" ]]; then
    echo "ERROR: missing ${MODEL_REPO_DIR}/refs/main — download Qwen3-32B to HF_HOME first"
    exit 1
fi
COMMIT=$(cat "${MODEL_REPO_DIR}/refs/main")
SNAPSHOT_PATH="${MODEL_REPO_DIR}/snapshots/${COMMIT}"
if [[ ! -d "${SNAPSHOT_PATH}" ]]; then
    echo "ERROR: missing snapshot ${SNAPSHOT_PATH}"
    exit 1
fi
echo "MODEL_SNAPSHOT=${SNAPSHOT_PATH}"
MODEL="${SNAPSHOT_PATH}"

RUN_TS=$(date -u +%Y%m%dT%H%M%SZ)
EXPERIMENT_NAME=skyrl_bird_grpo_Qwen3-32B_arctic_zorro_4node_${RUN_TS}
# Lustre — head writes weight_sync.pt, all 4 nodes mmap-read it (handoff §7).
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/data/skyrl-runs/ckpts/${EXPERIMENT_NAME}}
mkdir -p "${CHECKPOINT_DIR}"

NUM_NODES=4
GPUS_PER_NODE=8
NUM_GPUS=$((NUM_NODES * GPUS_PER_NODE))   # = 32

# Global batch: 128 prompts x 16 samples = 2048 trajectories. With DP=32 and
# ulysses_sp=1: per-DP mini=64, ZoRRo micro=16 (n_samples), grad_accum=4.
TRAIN_BSZ=128
MINI_BSZ=128
N_SAMPLES=16

LR=2e-6
PROMPT_LEN=32768
RESPONSE_LEN=4096

# vLLM sampling TP — verl recipe uses TP=4 (Qwen3-32B doesn't fit per-GPU at
# bf16 + 0.5 mem_util headroom on H200). 32 GPUs / TP=4 -> 8 engine replicas.
TP_SIZE=4
NUM_ENGINES=$((NUM_GPUS / TP_SIZE))

# Inference knobs forwarded to ArcticAsyncEngineArgs via
# trainer.arctic_rl.arctic_inference_config (raw passthrough):
#   FCA: forest_cascade_attn_configs={} + cudagraph_mode=PIECEWISE; the
#        fuse_allreduce_rms=false pass_config dodges a flashinfer 0.6.6
#        workspace-init assert during CUDA-graph capture.
#   Spec-dec: speculative_config={method: arctic, model: <path>, ...}.
#
# Flow-style dict values need a space after every `:` — OmegaConf.from_cli
# runs yaml.load on each rhs (Hydra's CLI parser is more lenient).
USE_FCA=${USE_FCA:-True}
SPEC_MODEL=${SPEC_MODEL:-/data-fast/qwen3-32b-bird-4096-3head}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-3}

AI_CFG_PARTS=()
if [[ "${USE_FCA}" == "True" ]]; then
    AI_CFG_PARTS+=('forest_cascade_attn_configs: "{}"')
    AI_CFG_PARTS+=('compilation_config: {cudagraph_mode: PIECEWISE, pass_config: {fuse_allreduce_rms: false}}')
fi
if [[ -n "${SPEC_MODEL}" && -d "${SPEC_MODEL}" ]]; then
    AI_CFG_PARTS+=("speculative_config: {method: arctic, model: ${SPEC_MODEL}, num_speculative_tokens: ${NUM_SPEC_TOKENS}}")
fi

AI_CFG_OVERRIDE=()
if (( ${#AI_CFG_PARTS[@]} > 0 )); then
    IFS=, AI_CFG_BODY="${AI_CFG_PARTS[*]}" ; unset IFS
    AI_CFG_OVERRIDE+=("trainer.arctic_rl.arctic_inference_config={${AI_CFG_BODY}}")
fi

cd "${SKYRL_DIR}"

"${PYBIN}" -m skyrl.train.entrypoints.main_base \
    trainer.override_entrypoint=integrations.arctic_rl.entrypoint \
    trainer.arctic_rl={} \
    trainer.arctic_rl.colocate=true \
    trainer.arctic_rl.zero_stage=3 \
    trainer.arctic_rl.offload_optimizer=true \
    trainer.arctic_rl.offload_param=false \
    trainer.arctic_rl.log_prob_gpus=0 \
    trainer.arctic_rl.use_zorro=true \
    trainer.arctic_rl.use_liger=true \
    trainer.arctic_rl.attn_implementation=flash_attention_3 \
    trainer.arctic_rl.enable_gradient_checkpointing=true \
    trainer.arctic_rl.ulysses_sequence_parallel_size=1 \
    trainer.arctic_rl.logits_optimization=memory \
    trainer.arctic_rl.cuda_ipc_weight_sync=true \
    trainer.arctic_rl.low_memory_weight_sync=true \
    trainer.arctic_rl.lr_warmup_ratio=0.05 \
    'trainer.arctic_rl.optimizer_betas=[0.9,0.95]' \
    trainer.arctic_rl.vllm_enforce_eager=false \
    trainer.arctic_rl.vllm_enable_prefix_caching=true \
    trainer.arctic_rl.vllm_max_num_batched_tokens=40960 \
    trainer.arctic_rl.vllm_max_num_seqs=256 \
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
    generator.inference_engine.num_engines=${NUM_ENGINES} \
    generator.inference_engine.tensor_parallel_size=${TP_SIZE} \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.gpu_memory_utilization=0.5 \
    generator.inference_engine.async_engine=true \
    generator.batched=true \
    trainer.epochs=1 \
    trainer.eval_batch_size=32 \
    trainer.eval_before_train=false \
    trainer.eval_interval=100 \
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
