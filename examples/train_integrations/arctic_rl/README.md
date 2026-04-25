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
  - vLLM Replicas: generation, log-probs
  - NCCL weight sync between training and inference
```

## Validated Results

### Arctic RL Server Backend (this integration)

**Setup**: Qwen2.5-1.5B-Instruct, 4 DeepSpeed training GPUs + 2 vLLM sampling GPUs + 1 log-prob GPU (7x H200), GRPO with 5 samples/prompt.

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

### Prerequisites

- 7+ GPUs (H200/A100 recommended)
- `arctic-skyrl` repo (this repo, `arctic-rl-integration` branch)
- `ArcticTraining-dss` repo (`arctic-rl-grpo-loss` branch)
- `arctic-inference` package installed
- GSM8K dataset prepared

### Step 1: Clone repos

```bash
# Clone arctic-skyrl (client)
git clone https://github.com/snowflake-eng/arctic-skyrl.git
cd arctic-skyrl
git checkout arctic-rl-integration
pip install -e ".[arctic-rl]"

# Clone ArcticTraining-dss (server) — in a separate directory
cd ..
git clone https://github.com/snowflake-eng/ArcticTraining-dss.git
cd ArcticTraining-dss
git checkout arctic-rl-grpo-loss
pip install --no-deps -e .
```

### Step 2: Prepare GSM8K dataset

```bash
cd arctic-skyrl
python examples/train/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
```

### Step 3: Run training

```bash
bash examples/train_integrations/arctic_rl/run_gsm8k_grpo_arctic.sh
```

This will:
1. Start an Arctic RL server (DeepSpeed + vLLM) on localhost
2. Initialize a CPU-only SkyRL client via Ray
3. Run GRPO training on GSM8K with eval every 5 steps
4. Log to console (set `LOGGER=wandb` for W&B)

### Step 4: Monitor

```bash
# Watch live metrics
tail -f /tmp/arctic_rl_training.log | grep -E "avg_raw_reward|global_step|pass_at_1"

# Check GPU usage
nvidia-smi
```

## Configuration

### GPU Allocation (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `ARCTIC_TRAINING_GPUS` | 4 | DeepSpeed training workers (DP) |
| `ARCTIC_SAMPLE_GPUS` | 2 | vLLM sampling replicas |
| `ARCTIC_LOG_PROB_GPUS` | 1 | vLLM log-prob engine |
| `ARCTIC_SERVER_PORT` | 7000 | Server HTTP port |
| `ARCTIC_SERVER_LOGS` | 0 | Set to 1 for verbose server output |
| `ARCTIC_STARTUP_TIMEOUT` | 600 | Server startup timeout (seconds) |

Total GPUs needed: `TRAINING + SAMPLE + LOG_PROB` (default: 7).

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
skyrl/backends/arctic_rl/
├── __init__.py                      # Exports ArcticPPOTrainer, ArcticGenerator
├── arctic_trainer.py                # ArcticPPOTrainer: routes training to server
├── arctic_generator.py              # ArcticGenerator: routes generation to server vLLM
└── config.py                        # ArcticRLClientConfig builder

skyrl/train/entrypoints/
├── main_arctic_rl.py                # Entrypoint: sets up client + server
└── main_arctic_rl_fully_async.py    # Fully-async entrypoint

examples/train_integrations/arctic_rl/
├── README.md                        # This file
└── run_gsm8k_grpo_arctic.sh         # Launch script for GSM8K GRPO
```

## How It Works

1. **`main_arctic_rl.py`** creates an `ArcticRLClient` which spawns the server as a subprocess with a clean environment (stripped `CUDA_VISIBLE_DEVICES` and `RAY_*` vars so the server gets its own GPU access)

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

## Companion PRs

- **Client (this repo)**: [`arctic-rl-integration`](https://github.com/snowflake-eng/arctic-skyrl/compare/arctic-rl-integration) branch
- **Server (ArcticTraining-dss)**: [`arctic-rl-grpo-loss`](https://github.com/snowflake-eng/ArcticTraining-dss/compare/arctic-rl-grpo-loss) branch — PR [#20](https://github.com/snowflake-eng/ArcticTraining-dss/pull/20)
