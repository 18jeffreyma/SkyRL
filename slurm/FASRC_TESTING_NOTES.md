# SkyRL FASRC End-to-End Testing Notes

**Date**: January 21, 2026
**Test Environment**: Harvard FASRC Cluster (test/gpu_test partitions)
**Status**: Successfully Running

---

## Executive Summary

Successfully configured and deployed SkyRL training on FASRC's test partitions with MIG (Multi-Instance GPU) support. The system coordinates a two-job architecture: a CPU-based runtime server for sandboxed code execution and a GPU-based VERL PPO training job. After resolving several compatibility issues, training is now running successfully.

**WandB Run**: https://wandb.ai/jeffreyma/skyrl-dev/runs/q9s5toh6

---

## Test Configuration

| Parameter | Value |
|-----------|-------|
| GPU Partition | `gpu_test` |
| CPU Partition | `test` |
| GPU Type | MIG A100 (20GB slices) |
| GPUs Requested | 2 MIG slices |
| Model | Qwen/Qwen3-1.7B |
| Batch Size | 4 |
| Rollout N | 8 |
| Max Prompt Length | 4096 |
| Max Response Length | 8192 |
| Time Limit | 12 hours |

---

## Issues Encountered and Resolutions

### 1. MIG GPU UUID Compatibility with vLLM

**Problem**: vLLM 0.8.5 expects integer CUDA device IDs, but SLURM exposes MIG devices with UUID strings like `MIG-250bc972-7bb0-5cb3-a6e6-e54a0fa3227f`.

**Error**:
```
ValueError: invalid literal for int() with base 10: 'MIG-250bc972-7bb0-5cb3-a6e6-e54a0fa3227f'
```

**Solution**: Added MIG detection and remapping in `verl_training.sbatch`:
```bash
# Handle MIG GPU devices - vLLM requires integer device IDs
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
```

**Location**: `slurm/jobs/verl_training.sbatch:148-161`

---

### 2. Triton/torch.compile Compilation Failure

**Problem**: torch._dynamo's Triton compilation backend fails to compile CUDA utilities on FASRC compute nodes due to missing development headers or incompatible gcc.

**Error**:
```
CalledProcessError: Command '['/usr/bin/gcc', ... '-lcuda', ...]' returned non-zero exit status 1.
torch._dynamo.exc.BackendCompilerFailed: backend='<vllm.compilation.backends.VllmBackend object at ...>' raised:
```

**Solution**: Changed `enforce_eager=False` to `enforce_eager=True` to bypass torch.compile:
```bash
actor_rollout_ref.rollout.enforce_eager=True \
```

**Location**: `slurm/jobs/verl_training.sbatch:217`

**Trade-off**: Eager mode is slower than compiled mode but avoids the compilation issue. For production, consider pre-compiling Triton kernels or using nodes with proper development tools.

---

### 3. Missing ResourcePoolManager Import

**Problem**: The `verl_trainer.py` imports `Role` from `verl.trainer.ppo.ray_trainer` but not `ResourcePoolManager`, which is needed by `verl_main_ppo.py`.

**Error**:
```
ImportError: cannot import name 'ResourcePoolManager' from 'skyrl_agent.integrations.verl.verl_trainer'
```

**Solution**: Added the missing import:
```python
from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role, compute_advantage, RayPPOTrainer
```

**Location**: `skyrl-agent/skyrl_agent/integrations/verl/verl_trainer.py:53`

---

### 4. Missing Abstract Method Implementation

**Problem**: `SWEBenchTask` class didn't implement the `initialize_runtime` abstract method from `BaseTask`.

**Error**:
```
TypeError: Can't instantiate abstract class SWEBenchTask without an implementation for abstract method 'initialize_runtime'
```

**Solution**: Added stub implementation:
```python
@classmethod
async def initialize_runtime(cls, *args, **kwargs):
    """Initialize the runtime for the task.

    For SWEBench, runtime initialization is handled by OpenHands SDK.
    """
    pass
```

**Location**: `skyrl-agent/skyrl_agent/tasks/swebench/utils.py:175-181`

---

### 5. ROLLOUT_N Configuration Mismatch

**Problem**: The `ROLLOUT_N` value in `dev.conf` (2) didn't match `generator.num_trajectories` in `verl_oh.yaml` (8), causing an assertion error.

**Error**:
```
AssertionError: Verl configuration received num trajectories per instance `num_trajectories` of 2 but the provided value in the `skyagent_task_yaml` is 8.
```

**Solution**: Updated `dev.conf` to use `ROLLOUT_N=8` to match the yaml configuration.

**Location**: `slurm/config/dev.conf`

**Note**: For future flexibility, consider making the yaml's `num_trajectories` configurable via environment variable or adding a dev-specific yaml.

---

### 6. Ray Worker Environment Recreation

**Problem**: Ray workers recreate virtual environments on each run because the VIRTUAL_ENV path doesn't match Ray's expected `.venv` location.

**Warning**:
```
(raylet) warning: `VIRTUAL_ENV=/path/to/skyrl-agent/.venv` does not match the project environment path `.venv` and will be ignored
(raylet) Creating virtual environment at: .venv
(raylet) Installed 351 packages in 4.38s
```

**Current Status**: Not blocking - Ray's venv creation takes ~4-25 seconds depending on caching. The packages are installed from cache so it's relatively fast.

**Potential Solutions** (for future optimization):
1. Use Ray's `runtime_env` with `pip` or `conda` configuration
2. Pre-build a Ray runtime environment
3. Use `--active` flag with uv (requires investigation)

---

## Files Modified

| File | Changes |
|------|---------|
| `slurm/config/dev.conf` | FASRC test partition settings, MIG configuration, ROLLOUT_N=8 |
| `slurm/README.md` | Comprehensive documentation |
| `slurm/jobs/verl_training.sbatch` | MIG UUID fix, enforce_eager=True, removed --isolated flag |
| `skyrl-agent/skyrl_agent/integrations/verl/verl_trainer.py` | Added ResourcePoolManager import |
| `skyrl-agent/skyrl_agent/tasks/swebench/utils.py` | Added initialize_runtime method |
| `skyrl-agent/.env` | Created with WANDB_API_KEY and network settings |

---

## How to Run

### Prerequisites

1. **Sync verl dependencies** (one-time):
   ```bash
   cd /n/holylabs/janapa_reddi_lab/Lab/jjma/coding-agent-rl/SkyRL/skyrl-agent
   uv sync --extra verl
   ```

2. **Create output directories**:
   ```bash
   mkdir -p /n/netscratch/janapa_reddi_lab/Lab/jjma/ckpts/skyrl-dev
   mkdir -p /n/netscratch/janapa_reddi_lab/Lab/jjma/rollouts/skyrl-dev
   ```

3. **Ensure .env file exists** at `skyrl-agent/.env` with:
   ```bash
   VLLM_USE_V1=1
   WANDB_API_KEY=your_key_here
   FI_PROVIDER=tcp
   ```

### Launch Commands

```bash
cd /n/holylabs/janapa_reddi_lab/Lab/jjma/coding-agent-rl/SkyRL

# Dry run (preview without submitting)
./slurm/launch.sh --config dev --dry-run

# Full launch (runtime + training)
./slurm/launch.sh --config dev

# Use existing runtime (faster iteration)
./slurm/launch.sh --config dev --skip-runtime --runtime-url "http://hostname:8000"

# Runtime only (for debugging)
./slurm/launch.sh --config dev --runtime-only
```

### Monitoring

```bash
# Check job status
squeue -u $USER

# Tail training logs
tail -f /path/to/coordination_dir/training_*.err

# Check runtime health
curl http://runtime-hostname:8000/health
```

---

## Key Architecture Notes

### Two-Job Coordination

```
┌─────────────────────┐         ┌─────────────────────────┐
│   CPU Node (test)   │         │   GPU Node (gpu_test)   │
│                     │         │                         │
│  Runtime Server     │◄────────│    VERL Training        │
│  (slurm-remote-     │  HTTP   │    - vLLM inference     │
│   runtime)          │  API    │    - PPO updates        │
│        │            │         │    - Rollouts           │
│        ▼            │         │                         │
│  Podman Containers  │         │                         │
│  (Agent Sandboxes)  │         │                         │
└─────────────────────┘         └─────────────────────────┘
            │                               │
            └───────────────┬───────────────┘
                            ▼
              Shared Storage (.skyrl_jobs/)
              - runtime_url.txt
              - runtime_ready.flag
              - Job logs
```

### MIG GPU Considerations

- FASRC's `gpu_test` uses MIG (Multi-Instance GPU) where A100 40GB GPUs are split into 20GB slices
- Each MIG slice appears as a separate device to SLURM
- CUDA_VISIBLE_DEVICES contains UUIDs, not integers
- vLLM and some CUDA libraries expect integer device IDs
- The MIG fix in `verl_training.sbatch` handles this transparently

### Network Configuration

- Test partitions don't have EFA (Elastic Fabric Adapter)
- Use TCP for distributed communication: `FI_PROVIDER=tcp`
- IPv6 socket warnings are harmless (system falls back to IPv4)

---

## Performance Observations

| Phase | Duration |
|-------|----------|
| Ray worker venv setup | ~4-25 seconds |
| Dataset loading (4578 train, 500 val) | ~2 minutes |
| Model loading (2 workers) | ~10 seconds each |
| vLLM server startup | ~20 seconds |
| Training step startup | ~1 minute |

---

## Known Warnings (Safe to Ignore)

1. **Flash Attention dtype warning**: The model loads in fp32 initially but training uses mixed precision
2. **FSDP deprecation warning**: PyTorch recommending new APIs, current code still works
3. **Socket IPv6 warning**: System successfully falls back to IPv4
4. **Ray blocking async warning**: Performance advisory, not an error
5. **Pydantic Field warnings**: API changes, functionality unaffected

---

## Future Improvements

### Short-term
- [ ] Create dev-specific `verl_oh_dev.yaml` with smaller `num_trajectories` for faster testing
- [ ] Pre-compile Triton kernels to enable `enforce_eager=False`
- [ ] Investigate Ray runtime_env to avoid venv recreation

### Medium-term
- [ ] Add health check endpoint monitoring to launch script
- [ ] Implement automatic retry on transient failures
- [ ] Add GPU memory monitoring and alerts

### Long-term
- [ ] Support multi-node training (EFA-enabled partitions)
- [ ] Implement checkpoint resume functionality
- [ ] Add distributed data parallel for larger models

---

## Troubleshooting Quick Reference

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| `ValueError: invalid literal for int()` with MIG UUID | vLLM MIG incompatibility | Verify MIG fix is in sbatch |
| `BackendCompilerFailed` with gcc | Triton compilation | Use `enforce_eager=True` |
| `ImportError: ResourcePoolManager` | Missing import | Check verl_trainer.py imports |
| `Can't instantiate abstract class` | Missing method | Check task class implementations |
| `AssertionError: num_trajectories` | Config mismatch | Match ROLLOUT_N with yaml |
| Job stuck at "Creating virtual environment" | Ray venv rebuild | Wait ~30s, this is normal |
| `QOSMaxSubmitJobPerUserLimit` | Too many pending jobs | Cancel old pending jobs |

---

## Contact and Resources

- **SkyRL Repository**: `/n/holylabs/janapa_reddi_lab/Lab/jjma/coding-agent-rl/SkyRL`
- **Runtime Repository**: `/n/holylabs/janapa_reddi_lab/Lab/jjma/coding-agent-rl/slurm-remote-runtime`
- **Data Location**: `/n/netscratch/janapa_reddi_lab/Lab/jjma/data/r2e-all/`
- **Checkpoint Location**: `/n/netscratch/janapa_reddi_lab/Lab/jjma/ckpts/`
- **VERL Documentation**: https://github.com/volcengine/verl
- **OpenHands Documentation**: https://github.com/All-Hands-AI/OpenHands
