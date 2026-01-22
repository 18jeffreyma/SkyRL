# SkyRL SLURM Infrastructure

This directory contains the SLURM job orchestration system for running SkyRL reinforcement learning training on HPC clusters. It coordinates a two-job architecture: a CPU-based runtime server for sandboxed code execution and a GPU-based training job running VERL PPO.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SLURM Cluster                                      │
│                                                                              │
│  ┌─────────────────────────┐         ┌─────────────────────────────────┐    │
│  │   CPU Node (test)       │         │      GPU Node (gpu_test)        │    │
│  │                         │         │                                 │    │
│  │  ┌───────────────────┐  │         │  ┌───────────────────────────┐  │    │
│  │  │ Runtime Server    │  │  HTTP   │  │    VERL Training Job      │  │    │
│  │  │                   │◄─┼─────────┼──┤                           │  │    │
│  │  │ slurm-remote-     │  │ API     │  │  - vLLM inference         │  │    │
│  │  │ runtime           │  │         │  │  - PPO policy updates     │  │    │
│  │  │                   │  │         │  │  - Rollout generation     │  │    │
│  │  └───────────────────┘  │         │  └───────────────────────────┘  │    │
│  │           │             │         │              │                  │    │
│  │           ▼             │         │              │                  │    │
│  │  ┌───────────────────┐  │         │              │                  │    │
│  │  │ Podman Containers │  │         │              │                  │    │
│  │  │ (Agent Sandboxes) │  │         │              │                  │    │
│  │  └───────────────────┘  │         │              │                  │    │
│  └─────────────────────────┘         └──────────────┼──────────────────┘    │
│                                                     │                       │
│                                                     ▼                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Shared Storage (.skyrl_jobs/)                      │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌────────────┐ ┌─────────────────┐  │   │
│  │  │runtime_url  │ │runtime_ready│ │ training   │ │ Job logs        │  │   │
│  │  │.txt         │ │.flag        │ │ _complete  │ │ (*.out, *.err)  │  │   │
│  │  └─────────────┘ └─────────────┘ │ .flag      │ └─────────────────┘  │   │
│  │                                  └────────────┘                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Job Type | Description |
|-----------|----------|-------------|
| **Runtime Server** | CPU (`test`) | Runs `slurm-remote-runtime` API server providing sandboxed Podman containers for agent code execution |
| **Training Job** | GPU (`gpu_test`) | Runs VERL PPO training with vLLM inference backend, connects to runtime for environment interactions |
| **Coordination Dir** | Shared FS | `.skyrl_jobs/` directory for inter-job communication via flag files |

### Job Execution Flow

1. **Launch**: `launch.sh` creates coordination directory and submits both jobs
2. **Runtime Startup**: CPU job starts, launches API server, writes URL to coordination dir
3. **Runtime Ready**: CPU job signals ready via `runtime_ready.flag`
4. **Training Start**: GPU job (with dependency) starts, reads runtime URL, begins training
5. **Training Loop**: GPU job sends environment requests to runtime server via HTTP
6. **Completion**: Training signals completion via `training_complete.flag`, runtime shuts down

## Quick Start

### Prerequisites

1. **Start Podman socket** (FASRC uses Podman with Docker alias):
   ```bash
   podman system service unix:///tmp/podman.sock --time=0 &
   export DOCKER_HOST=unix:///tmp/podman.sock
   ```

2. **Verify container runtime**:
   ```bash
   docker run --rm hello-world
   ```

3. **Pre-pull container image** (optional but recommended):
   ```bash
   podman pull ghcr.io/opendevin/sandbox:main
   ```

4. **Ensure .env file exists** with required API keys:
   ```bash
   ls -la /n/holylabs/janapa_reddi_lab/Lab/jjma/coding-agent-rl/SkyRL/skyrl-agent/.env
   ```

5. **Verify data exists**:
   ```bash
   ls -la /n/netscratch/janapa_reddi_lab/Lab/jjma/data/r2e-all/
   ```

6. **Create output directories**:
   ```bash
   mkdir -p /n/netscratch/janapa_reddi_lab/Lab/jjma/ckpts/skyrl-dev
   mkdir -p /n/netscratch/janapa_reddi_lab/Lab/jjma/rollouts/skyrl-dev
   ```

### Launch Commands

```bash
cd /n/holylabs/janapa_reddi_lab/Lab/jjma/coding-agent-rl/SkyRL

# Dry run (preview jobs without submitting)
./slurm/launch.sh --config dev --dry-run

# Launch development run
./slurm/launch.sh --config dev

# Launch production run
./slurm/launch.sh --config production

# Launch only runtime server (for debugging)
./slurm/launch.sh --config dev --runtime-only

# Skip runtime, use existing runtime URL
./slurm/launch.sh --config dev --runtime-url http://existing-host:8000
```

### Monitoring

```bash
# Check job status
squeue -u $USER

# Monitor both jobs
squeue -j <runtime_job_id>,<training_job_id>

# Tail runtime logs
tail -f /path/to/coordination_dir/runtime_*.out

# Tail training logs
tail -f /path/to/coordination_dir/training_*.out

# Check GPU utilization (on GPU node)
nvidia-smi

# List recent coordination directories
ls -lt ~/.skyrl_jobs/ | head
```

## Configuration Reference

### Configuration Files

| File | Purpose |
|------|---------|
| `config/dev.conf` | Development/testing - small scale, FASRC test partitions |
| `config/production.conf` | Production runs - full scale resources |

### Key Configuration Options

#### Partition Settings
```bash
GPU_PARTITION="gpu_test"    # SLURM partition for GPU training
CPU_PARTITION="test"        # SLURM partition for runtime server
```

#### Resource Allocation
```bash
# GPU Training Job
GPU_NODES=1                 # Number of GPU nodes
GPUS_PER_NODE=2            # GPUs per node (MIG slices on FASRC)
TRAINING_CPUS=16           # CPUs for training job
TRAINING_MEMORY="64G"      # Memory for training job

# CPU Runtime Job
CPUS_FOR_RUNTIME=32        # CPUs for runtime server
RUNTIME_MEMORY="64G"       # Memory for runtime server

# Container Resources
CPUS_PER_WORKER=8          # CPUs per agent container
MEMORY_PER_WORKER="16G"    # Memory per agent container
MAX_PARALLEL_AGENTS=4      # Maximum concurrent agent containers
```

#### Model Configuration
```bash
MODEL="Qwen/Qwen3-1.7B"    # HuggingFace model path
TP_SIZE=1                  # Tensor parallelism (GPUs per model)
SP_SIZE=1                  # Sequence parallelism
```

#### Training Parameters
```bash
TRAIN_BATCH_SIZE=4         # Batch size for PPO updates
ROLLOUT_N=2                # Rollouts per prompt
MAX_PROMPT_LENGTH=4096     # Maximum input length
MAX_RESPONSE_LENGTH=8192   # Maximum output length
TOTAL_EPOCHS=1             # Training epochs
```

#### Time and Timeouts
```bash
TIME_LIMIT="12:00:00"      # SLURM job time limit
RUNTIME_READY_TIMEOUT=600  # Seconds to wait for runtime
```

#### Data Paths
```bash
DATA_DIR="/n/netscratch/.../data/r2e-all"
TRAIN_DATA="${DATA_DIR}/train.parquet"
VAL_DATA="${DATA_DIR}/validation.parquet"
CHECKPOINT_DIR="/n/netscratch/.../ckpts/skyrl-dev"
ROLLOUT_DIR="/n/netscratch/.../rollouts/skyrl-dev"
```

### FASRC-Specific Settings

#### MIG GPU Configuration
FASRC's `gpu_test` partition uses MIG (Multi-Instance GPU) where A100 40GB GPUs are split into 20GB slices:

```bash
# Each MIG slice appears as a separate GPU to SLURM
GPUS_PER_NODE=2            # 2 MIG slices (20GB each)
GPU_MEMORY_UTILIZATION=0.85 # Leave headroom for MIG overhead
TP_SIZE=1                  # One model per MIG slice
```

#### Network Configuration
Test partitions don't have EFA (Elastic Fabric Adapter):

```bash
FI_PROVIDER="tcp"          # Use TCP instead of EFA
SKYRL_LD_LIBRARY_PATH_EXPORT=0  # Disable EFA library setup
```

#### Podman/Docker
FASRC uses Podman with Docker CLI compatibility:

```bash
# Start Podman socket before job submission
podman system service unix:///tmp/podman.sock --time=0 &
export DOCKER_HOST=unix:///tmp/podman.sock
```

## Directory Structure

```
slurm/
├── README.md              # This file
├── launch.sh              # Main job submission orchestrator
├── config/
│   ├── dev.conf           # Development configuration
│   └── production.conf    # Production configuration
├── jobs/
│   ├── runtime_server.sbatch  # CPU runtime job script
│   └── verl_training.sbatch   # GPU training job script
└── lib/
    └── common.sh          # Shared utility functions

.skyrl_jobs/               # Coordination directory (created at runtime)
└── YYYYMMDD_HHMMSS_<job_id>/
    ├── runtime_url.txt        # Runtime server URL
    ├── runtime_api_key.txt    # API key for authentication
    ├── runtime_ready.flag     # Signal that runtime is ready
    ├── training_complete.flag # Signal that training finished
    ├── runtime_<id>.out       # Runtime stdout
    ├── runtime_<id>.err       # Runtime stderr
    ├── training_<id>.out      # Training stdout
    ├── training_<id>.err      # Training stderr
    └── *.conf                 # Copy of config files used
```

## Troubleshooting

### Common Issues

#### 1. "Podman socket not found" or container failures
```bash
# Ensure Podman socket is running
podman system service unix:///tmp/podman.sock --time=0 &
export DOCKER_HOST=unix:///tmp/podman.sock

# Test connectivity
docker ps
docker run --rm hello-world
```

#### 2. Training job stuck waiting for runtime
```bash
# Check if runtime job is running
squeue -u $USER

# Check runtime logs for errors
cat /path/to/coord_dir/runtime_*.err

# Manually check runtime URL file
cat /path/to/coord_dir/runtime_url.txt
cat /path/to/coord_dir/runtime_ready.flag
```

#### 3. CUDA out of memory on MIG GPUs
```bash
# Reduce memory usage in config
GPU_MEMORY_UTILIZATION=0.75  # Lower from 0.85
TRAIN_BATCH_SIZE=2           # Reduce batch size
MAX_RESPONSE_LENGTH=4096     # Reduce sequence length
```

#### 4. Job fails with "partition not available"
```bash
# Verify partition access
sinfo -p test,gpu_test

# Check your account permissions
sacctmgr show associations user=$USER
```

#### 5. Runtime server fails to start
```bash
# Check slurm-remote-runtime installation
ls -la /path/to/slurm-remote-runtime/.venv/

# Manually test the API server
cd /path/to/slurm-remote-runtime
source .venv/bin/activate
python -m slurm_runtime.api_server --help
```

### Debug Mode

Enable verbose logging:

```bash
# Set in config file
SKYRL_DEBUG=1

# Or export before launching
export SKYRL_DEBUG=1
./slurm/launch.sh --config dev
```

### Manual Component Testing

#### Test Runtime Server Standalone
```bash
# Allocate interactive CPU node
srun -p test --cpus-per-task=8 --mem=32G --time=1:00:00 --pty bash

# Start Podman socket
podman system service unix:///tmp/podman.sock --time=0 &
export DOCKER_HOST=unix:///tmp/podman.sock

# Start runtime server manually
cd /path/to/slurm-remote-runtime
source .venv/bin/activate
python -m slurm_runtime.api_server \
    --container-image ghcr.io/opendevin/sandbox:main \
    --host 0.0.0.0 \
    --port 8000 \
    --verbose

# Test from another terminal
curl http://$(hostname):8000/health
```

#### Test Training Script Locally
```bash
# Allocate interactive GPU node
srun -p gpu_test --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=1:00:00 --pty bash

# Set environment
export SANDBOX_REMOTE_RUNTIME_API_URL="http://runtime-host:8000"
export ALLHANDS_API_KEY="test-key"

# Run minimal training test
cd /path/to/SkyRL/skyrl-agent
uv run --frozen python -c "from skyrl_agent import *; print('Import OK')"
```

## Scaling Up

Once the end-to-end test succeeds on test partitions:

1. **Increase GPUs**: `GPUS_PER_NODE=4` or `GPUS_PER_NODE=8`
2. **Increase parallelism**: `MAX_PARALLEL_AGENTS=8`, `ROLLOUT_N=8`
3. **Increase batch size**: `TRAIN_BATCH_SIZE=16` or higher
4. **Use production partitions**: Update partition names in config
5. **Enable EFA**: Set `FI_PROVIDER="efa"` on EFA-enabled partitions

## Related Documentation

- [slurm-remote-runtime](../slurm-remote-runtime/README.md) - Runtime server implementation
- [skyrl-agent](skyrl-agent/README.md) - Agent implementation
- [VERL](https://github.com/volcengine/verl) - Training framework
- [OpenHands](https://github.com/All-Hands-AI/OpenHands) - Agent framework
