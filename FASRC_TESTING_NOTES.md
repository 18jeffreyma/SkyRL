# FASRC Testing Notes - SkyRL with slurm-remote-runtime

## Test Environment

| Component | Configuration |
|-----------|---------------|
| GPU Partition | `gpu_test` - MIG GPUs (20GB A100 slices) |
| CPU Partition | `test` - Standard CPU nodes |
| Container Runtime | Podman with Docker alias |
| Model | Qwen/Qwen3-1.7B |
| Dataset | r2e-all at `/n/netscratch/janapa_reddi_lab/Lab/jjma/data/r2e-all/` |

## Issues Fixed

### 1. MIG GPU UUID Detection

**Problem:** vLLM's `check_mig_enabled()` failed on CUDA_VISIBLE_DEVICES containing UUIDs like `MIG-GPU-xxx`.

**Solution:** Added workaround in `verl_training.sbatch` to remap UUIDs to integer indices:
```bash
if [[ "$CUDA_VISIBLE_DEVICES" =~ ^MIG ]]; then
    log_info "MIG GPUs detected, remapping CUDA_VISIBLE_DEVICES from UUIDs to integers: 0"
    export CUDA_VISIBLE_DEVICES="0"
fi
```

### 2. Pydantic Frozen Instance Error

**Problem:** OHCodeActAgent couldn't set instance attributes after `super().__init__()` because the base Agent class is frozen.

**Solution:** Used `object.__setattr__()` to bypass Pydantic's frozen model restrictions in `codeact_agent.py`:
```python
object.__setattr__(self, '_infer_engine', infer_engine)
object.__setattr__(self, '_messages', [])
# ... etc for all instance attributes
```

### 3. API Key Authentication

**Problem:** Training jobs couldn't connect to runtime server due to API key mismatch.

**Solution:** Added `--disable-auth` flag to slurm-remote-runtime and updated `runtime_server.sbatch`:
```bash
--disable-auth \  # No API key required in trusted SLURM environment
```

### 4. Environment Variable Propagation

**Problem:** `SANDBOX_REMOTE_RUNTIME_API_URL` wasn't being passed to Ray workers.

**Solution:**
1. Read URL directly from file (avoid log contamination)
2. Export via Ray's runtime environment in `verl_main_ppo.py`:
```python
PPO_RAY_RUNTIME_ENV["env_vars"].update({
    "SANDBOX_REMOTE_RUNTIME_API_URL": os.environ["SANDBOX_REMOTE_RUNTIME_API_URL"]
})
```

## Current Blocker: Container Image Compatibility

### Problem

The OpenHands SDK's `APIRemoteWorkspace` starts containers with command:
```
/usr/local/bin/openhands-agent-server --port 60000
```

But r2e images (e.g., `docker.io/xingyaoww/r2e-namanjain12/pandas_final:xxx`) don't have `openhands-agent-server` installed.

### Evidence

From runtime logs:
```
Started container for runtime slurm-37564a5a3dca at http://holy8a24101:60001
Connection refused to http://holy8a24101:60001/alive, server still starting...
[... repeats for 60s ...]
Action server at http://holy8a24101:60001 did not become ready within 60.3s
Runtime slurm-37564a5a3dca failed to become ready within 60s
```

### Implemented Solution: On-Demand Image Building

Added `--enable-image-build` flag to slurm-remote-runtime that:
1. Detects when a base image (like r2e-...) is requested
2. Uses OpenHands' `build_runtime_image()` to build a runtime image with openhands-agent-server
3. Caches built images for reuse
4. Uses the built image for container startup

**Usage:**
```bash
python -m slurm_runtime.api_server \
    --enable-image-build \
    --build-reserved-cpus 4 \
    ...
```

**Tradeoffs:**
- First request for each base image adds ~2-5 min build time
- Built images are cached for subsequent requests
- Uses OpenHands' standard image build system

### Future: Pre-build Images (for production)

For production workloads, pre-building images is recommended:
- Script to batch-build all r2e images before training
- Upload to container registry for faster pulls
- No build latency during training

## Next Steps

1. Test on-demand image building with training
2. Verify images are cached correctly
3. Consider pre-building for production workloads

## Files Modified

- `SkyRL/slurm/jobs/verl_training.sbatch` - MIG workaround, env var fix
- `SkyRL/slurm/jobs/runtime_server.sbatch` - Disabled auth, increased timeout
- `SkyRL/skyrl-agent/skyrl_agent/agents/oh_codeact/codeact_agent.py` - Pydantic fix
- `SkyRL/skyrl-agent/skyrl_agent/integrations/verl/verl_main_ppo.py` - Ray env var propagation
- `slurm-remote-runtime/slurm_runtime/api_server.py` - Added --disable-auth flag
