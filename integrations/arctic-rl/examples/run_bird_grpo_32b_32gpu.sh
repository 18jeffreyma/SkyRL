#!/usr/bin/env bash
# SkyRL + Arctic RL backend: Qwen3-32B BIRD-SQL GRPO recipe — 4 nodes / 32 H200s.
#
# Adapted from Tunji's verl recipe
#   /code/users/truwase/scripts/arctic_setup/recipe_bird_sql/run_qwen3_32b_bird_grpo_arl_zorro_yes_local_ckpt.sh
# so the Arctic-RL training engine + ZoRRo dedup + colocated vLLM sampling get
# the same per-GPU compute as the verl baseline (W&B project skyrl_arctic_rl,
# converged run k1l3lue5).  The SkyRL knobs are wired through arctic_rl/config.py
# (build_rl_config); per-DP-rank micro-batch sizing is auto-derived from
# train_batch_size * n_samples_per_prompt and ulysses_sp.
#
# Purpose: head-to-head per-step E2E timing on 32B vs the stock-SkyRL FSDP
# baseline (run_bird_grpo_32b_32gpu_fsdp.sh, sibling).
#
# Topology (matches verl recipe):
#   NUM_NODES=4, GPUS_PER_NODE=8     => training_gpus = 32, sampling_gpus = 32
#   inference_engine.tensor_parallel_size=4, num_engines=8
#   train_batch_size=128, n_samples_per_prompt=16
#     => 2048 trajectories/step, 64 trajectories/GPU at DP=32 (ulysses_sp=1)
#
# Prereq: 4-node ray cluster on skyrl_v1 env already up (see ../../examples/run_bird_grpo_1.7b_32gpu.sh
# and HANDOFF_2026-06-22_SKYRL_BIRD.md). ``ray status`` should show 32/32 GPU.
#
# Step-1 invariants we still expect (matching the 1.7B run + xid2pl9f):
#   - actor/ppo_kl == 0, actor/pg_clipfrac == 0, actor/pg_clipfrac_lower == 0
#   - actor/loss == actor/pg_loss
# (use_kl_loss=false, use_kl_in_reward=false, update_epochs_per_batch=1)

set -euxo pipefail

SKYRL_DIR=<PATH>/sky-checkouts/SkyRL
DATA_DIR=${DATA_DIR:-"<PATH>/open-source-text2sql"}
PYBIN=/home/yak/miniconda3/envs/skyrl_v1/bin/python

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

# Tunji's verl recipe @ 32 GPUs:
#   BSZ_PER_GPU=4, PPO_MINI_BSZ_PER_GPU=4 -> BSZ=128, PPO_MINI_BSZ=128
#   ROLL_N=16 -> 2048 trajectories/step
# SkyRL: with ulysses_sp=1 + n_samples=16 + 32 training GPUs (DP=32),
#   train_per_gpu = 128 * 16 / 32 = 64  (mini == train, grad_accum=1 at SkyRL level)
#   DeepSpeed: micro=16 (n_samples, ZoRRo), grad_accum=64/16=4
#   _batch_assertion: 16 * 4 * 32 = 2048 == mini_batch  ✓
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

# FCA + speculative decoding — full inference knobs to match the verl recipe.
#
# Plumbing (SkyRL -> Arctic-Platform -> ArcticAsyncEngineArgs):
#   trainer.arctic_rl.arctic_inference_config  (dict, SkyRL dataclass field)
#       => ArcticRLClientConfig.arctic_inference_config
#       => build_model_config() merges into vllm engine kwargs (extra_engine_kwargs)
#       => ArcticAsyncEngineArgs(**kwargs)
#
# Tunji's verl recipe uses two convenience aliases on its own arctic_rl config
#   arctic_rl.use_fca=True
#   arctic_rl.spec_model=/data-fast/qwen3-32b-bird-4096-3head
# which arctic-verl/verl/trainer/ppo/arctic_rl_client.py:215 expands into THREE
# engine kwargs:
#   use_fca=True     -> compilation_config={cudagraph_mode: PIECEWISE}
#                       forest_cascade_attn_configs="{}"
#   spec_model=path  -> speculative_config={method: arctic, model: path,
#                                            num_speculative_tokens: 3}
# SkyRL's `arctic_rl` dataclass does NOT have those aliases; it only exposes
# `arctic_inference_config: dict` which is forwarded raw. So we replicate the
# same three engine kwargs here verbatim to stay apples-to-apples with verl.
# (num_speculative_tokens=3 is fixed at 3 by the verl bridge regardless of the
# draft model's n_predict; we mirror that.)
#
# The draft model lives on the local NVMe (/data-fast) on each of the 4 nodes,
# not on Lustre, so vLLM loads it per-replica from local SSD.
USE_FCA=${USE_FCA:-True}
SPEC_MODEL=${SPEC_MODEL:-/data-fast/qwen3-32b-bird-4096-3head}
NUM_SPEC_TOKENS=${NUM_SPEC_TOKENS:-3}

# SkyRL's main_base uses OmegaConf.from_cli, which yaml.load()s the value half
# of each `k=v` arg. YAML flow-style requires a SPACE after every `:` inside
# `{key: value, key: value}` -- without it the scanner treats `key:value` as a
# single bare-scalar token. (Hydra's CLI grammar tolerates `key:value`, which
# is what tripped us up earlier -- different parser, different rules.)
AI_CFG_PARTS=()
if [[ "${USE_FCA}" == "True" ]]; then
    AI_CFG_PARTS+=('forest_cascade_attn_configs: "{}"')
    # pass_config.fuse_allreduce_rms=false: dodge the
    # `Flashinfer workspace must be initialized when using flashinfer` assert in
    # vllm/compilation/passes/fusion/allreduce_rms_fusion.py:143 that fired
    # during CUDA graph capture on our stack. Default-True with cudagraph_mode
    # PIECEWISE + world_size>=2 rewrites every all_reduce->rms_norm into the
    # fused FlashInfer kernel, which our flashinfer 0.6.6 + ArcticInference
    # 6ec09b1 combo never initializes the workspace for.
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
    trainer.backend=arctic_rl \
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
