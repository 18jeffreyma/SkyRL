# Arctic RL Integration for SkyRL

Routes SkyRL's GRPO training loop through the Arctic RL server — **all GPU operations** (training, generation, log-probs, weight sync) happen on the server. The SkyRL client is CPU-only.

## Architecture

```
SkyRL Client (CPU-only, ray num_gpus=0)
  - Data loading, reward scoring (skyrl-gym)
  - Orchestration: generate → score → train
  - HTTP calls to Arctic RL server
        |
        | HTTP (torch-serialized batches)
        v
Arctic RL Server (own Ray cluster, all GPUs)
  - DeepSpeed Workers: forward/backward, GRPO loss, optimizer step
  - ArcticInference (vLLM) Replicas: generation, log-probs
  - NCCL weight sync between training and inference
```

## Validated Results

### Arctic RL Server Backend (this integration)

**Setup**: Qwen2.5-1.5B-Instruct, 4 DeepSpeed training GPUs + 2 ArcticInference (vLLM) sampling GPUs + 1 log-prob GPU (7x H200), GRPO with 5 samples/prompt.

| Step | GSM8K Eval (pass@1) | Training Reward |
|------|-------------------|-----------------|
| 0 (base) | 7.8% | — |
| 5 | 33% | 0.22 |
| 10 | 63% | 0.60 |
| 30 | 70% | 0.65 |
| 45 | 73% | 0.81 |
| 75 | **75.4%** | 0.82 |

### SkyRL Default (FSDP2 baseline)

| Model | GSM8K Accuracy (1,319 test) |
|---|---|
| Base (Qwen2.5-1.5B-Instruct) | 7.43% |
| Trained (step 59) | **79.00%** |

## Quick Start

### Install

```bash
uv sync --extra arctic-rl
```

This pulls in `arctic-platform` and `arctic-inference[vllm]` from public main
branches (vLLM and torch are pulled transitively via `arctic-inference[vllm]`,
SkyRL training core via `skyrl[skyrl-train]`).

Add `--extra fsdp` only if you also want to run the SkyRL native FSDP backend
side-by-side (e.g. for the comparison launcher
`integrations/arctic_rl/examples/run_bird_grpo_32b_32gpu_fsdp.sh`). The two
extras pin different vLLM versions, so `arctic-rl` alone is preferred for
pure Arctic runs.

### Use with any SkyRL recipe

Append a single CLI flag to **any** stock SkyRL recipe to route training
through Arctic RL:

```bash
bash examples/<your-recipe>/run.sh \
    trainer.override_entrypoint=integrations.arctic_rl.entrypoint
```

That's it. Optional Arctic-specific knobs go under `trainer.arctic_rl.*`
(e.g. `trainer.arctic_rl.colocate=true`, `trainer.arctic_rl.zero_stage=3`).
The integration's entrypoint auto-fills sensible defaults if you set none.

### Self-contained examples

```bash
# GSM8K, 4 H200s
bash integrations/arctic_rl/examples/run_gsm8k_grpo_4gpu.sh

# BIRD-SQL, Qwen3-32B, 32 H200s (4 nodes)
bash integrations/arctic_rl/examples/run_bird_grpo_32b_32gpu.sh
```

Set `LOGGER=wandb` for W&B logging (default is console).

## Configuration

### GPU Allocation

GPU layout is derived from standard SkyRL knobs:
- Training GPUs: `trainer.placement.policy_num_gpus_per_node * trainer.placement.policy_num_nodes`
- Sampling GPUs: `generator.inference_engine.num_engines * generator.inference_engine.tensor_parallel_size`
- Log-prob GPUs: `trainer.arctic_rl.log_prob_gpus` (default: 0 — log-probs colocate with sampling vLLM)
- Colocation between training and sampling: `trainer.arctic_rl.colocate` (default: false)

### Key Training Parameters

The launch script passes these to SkyRL via Hydra overrides:

| Parameter | Value | Notes |
|-----------|-------|-------|
| `trainer.train_batch_size` | 256 | Prompts per step |
| `trainer.policy_mini_batch_size` | 2 | Prompts per mini-batch |
| `generator.n_samples_per_prompt` | 5 | Completions per prompt |
| `trainer.policy.optimizer_config.lr` | 1e-6 | Learning rate |
| `trainer.epochs` | 20 | Training epochs |
| `trainer.eval_interval` | 5 | Eval every N steps |

DeepSpeed config is set automatically:
- `gradient_accumulation_steps` = `train_batch_size * n_samples / (policy_mini_batch_size * n_samples)` = 128
- `gradient_clipping` = 1.0
- `optimizer` = AdamW with lr from config

## File Structure

```
integrations/arctic_rl/                # importable as integrations.arctic_rl
├── README.md
├── __init__.py                       # exports ArcticPPOTrainer, ArcticGenerator
├── trainer.py                        # ArcticPPOTrainer: routes training to server
├── generator.py                      # ArcticGenerator: routes generation to server vLLM
├── config.py                         # ArcticRLTrainerConfig + build_rl_config
├── entrypoint.py                     # dispatched here from main_base
├── envs/                             # bird env registration
└── examples/
    └── run_*.sh                      # launchers (call main_base with override_entrypoint)
```

`integrations/` is a PEP 420 namespace package (no `__init__.py`); the
`arctic_rl` package below it is reachable as `integrations.arctic_rl` from
the SkyRL repo root with no `PYTHONPATH` setup. It is distinct from the
upstream `arctic_training` PyPI package's `arctic_training.arctic_rl`
sub-namespace — both coexist at import time without collision.

## How It Works

1. **`arctic_rl.entrypoint`** creates an `ArcticRLClient` which spawns the server as a subprocess with a clean environment (stripped `CUDA_VISIBLE_DEVICES` and `RAY_*` vars so the server gets its own GPU access)

2. **`ArcticPPOTrainer`** overrides the standard SkyRL training loop:
   - `fwd_logprobs_values_reward` → no-op (server computes old log-probs internally)
   - `compute_advantages_and_returns` → no-op (server computes GRPO advantages from rewards)
   - `train_critic_and_policy` → sends batches to server via HTTP, server runs GRPO loss + backward

3. **`ArcticGenerator`** routes generation to server vLLM and scores completions via `skyrl-gym`

4. **Server-side `grpo_loss`** (in `processors.py`) is self-contained:
   - Computes per-token log-probs with causal shift
   - Derives old log-probs by detaching (correct for `update_epochs_per_batch=1`)
   - Computes group-relative advantages from per-sequence rewards
   - Applies PPO clipped surrogate (eps_clip=0.2)
