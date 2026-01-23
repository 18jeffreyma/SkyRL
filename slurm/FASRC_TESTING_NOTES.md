# SkyRL FASRC End-to-End Testing Notes

**Last Updated**: January 23, 2026 (13:50 EST)
**Test Environment**: Harvard FASRC Cluster (test/gpu_test partitions)
**Status**: E2E Training Working - SDK Validation Errors Fixed

---

## Executive Summary

Successfully configured and deployed SkyRL training on FASRC's test partitions with MIG (Multi-Instance GPU) support. The system coordinates a two-job architecture: a CPU-based runtime server for sandboxed code execution and a GPU-based VERL PPO training job.

**What's Working:**
- Runtime server starts and initializes cleanly
- Image building from base images (e.g., `namanjain12/pandas_final`) works
- Container startup with built runtime images works
- Agent-server health checks pass
- Workspace initialization completes
- Agent rollouts complete successfully
- **PPO training step completes (backward pass)**
- **Checkpoints saved successfully (global_step_1 through global_step_19+)**
- **SDK Pydantic validation errors resolved (ActionEvent, MessageToolCall, FinishAction)**

---

## Test Configuration

### Current Working Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| GPU Partition | `gpu_test` | MIG slices (20GB A100) |
| CPU Partition | `test` | 32 CPUs, 64GB |
| Model | `Qwen/Qwen2.5-Coder-0.5B-Instruct` | Smaller model to fit in 20GB MIG |
| Batch Size | 1 | Reduced for testing |
| Rollout N | 2 | 2 trajectories per instance |
| Max Iterations | 20 | Reduced from 50 to fit in GPU memory |
| Max Prompt Length | 2048 | Reduced for backward pass |
| Max Response Length | 1024 | Reduced for backward pass |
| Max Model Length | 4096 | vLLM KV cache limit |
| GPU Memory Utilization | 0.5 | Leave room for training gradients |
| CPUs Per Worker | 4 | |
| Worker Slots | 7 | (32-4)/4 = 7 |
| Time Limit | 6 hours | |

### Memory Budget (20GB MIG)

```
Model weights (Qwen2.5-0.5B bf16):    ~1 GB
KV cache (4K context):                ~1 GB
Gradients + optimizer (FSDP offload): ~4 GB
Batch activations (1K response, 20 iter): ~2 GB
─────────────────────────────────────────────
Total:                                ~8 GB
Headroom:                            ~12 GB
```

---

## Architecture

```
┌─────────────────────┐         ┌─────────────────────────┐
│   CPU Node (test)   │         │   GPU Node (gpu_test)   │
│                     │         │                         │
│  Runtime Server     │◄────────│    VERL Training        │
│  (slurm-remote-     │  HTTP   │    - vLLM inference     │
│   runtime)          │  API    │    - PPO updates        │
│        │            │         │    - Agent rollouts     │
│        ▼            │         │                         │
│  Image Building     │         │                         │
│  (SDK build system) │         │                         │
│        │            │         │                         │
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

---

## Issues Fixed

### 1. MIG GPU UUID Compatibility with vLLM

**Problem**: vLLM expects integer CUDA device IDs, but SLURM exposes MIG devices with UUID strings.

**Solution**: Added MIG detection and remapping in `verl_training.sbatch`:
```bash
if [[ "${CUDA_VISIBLE_DEVICES}" == *"MIG-"* ]]; then
    export CUDA_VISIBLE_DEVICES="0"
fi
```

### 2. Podman State Corruption Between Jobs

**Problem**: When retrying runs, podman encountered "invalid internal status" errors from previous SLURM jobs.

**Solution**: Added `setup_isolated_podman_storage()` in `common.sh` that creates per-job storage:
```bash
# Creates /tmp/podman-$USER-$SLURM_JOB_ID/ with:
# - storage.conf using overlay driver with fuse-overlayfs
# - Isolated graphRoot, runRoot, XDG_RUNTIME_DIR
# - XDG_DATA_HOME redirect to prevent NFS issues
# - Full cleanup of stale state from previous jobs
# - podman system reset to clear any corrupted state
# - Cleanup trap on exit
```

The isolation is comprehensive and should handle any node without needing exclusions.

### 3. Triton JIT Compilation (Python.h Missing)

**Problem**: During PPO training step, Triton tries to JIT compile CUDA utilities but gcc fails because `Python.h` is not found.

**Error**:
```
/tmp/.../main.c:5:10: fatal error: Python.h: No such file or directory
```

**Root Cause**: Ray workers were using system Python which lacks development headers.

**Solution**: Added `setup_python_for_uv()` in `common.sh` that:
1. Loads the Python module (`module load python`)
2. Sets `UV_PYTHON` to point to the loaded Python
3. Exports it so Ray workers inherit it

**Location**: `slurm/lib/common.sh`

### 4. Pickle Serialization Error (Thread RLock)

**Problem**: After agent rollouts complete, `copy.deepcopy()` fails on Pydantic models containing thread locks.

**Error**:
```
TypeError: cannot pickle '_thread.RLock' object
```

**Solution**: Added `safe_deepcopy()` function in `skyrl_agent/agents/base.py`:
```python
def safe_deepcopy(obj: Any) -> Any:
    """Safely deepcopy an object, handling non-serializable Pydantic models."""
    try:
        return copy.deepcopy(obj)
    except (TypeError, pickle.PicklingError, RecursionError):
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        # ... other fallbacks
```

**Note**: Initial implementation had a bug that caused infinite recursion - make sure to call `copy.deepcopy(obj)` not `safe_deepcopy(obj)` in the try block!

### 5. Entry Script `/r2e_tests` Directory Missing

**Problem**: The container entry script (`instance_r2e_entry.sh`) assumed `/r2e_tests` exists, but some R2E images don't have this directory.

**Error**:
```
find: '/r2e_tests': No such file or directory
ln: failed to create symbolic link '/testbed/r2e_tests/r2e_tests': No such file or directory
```

**Solution**: Added conditional checks in `instance_r2e_entry.sh`:
```bash
# Delete *.pyc files and __pycache__ in /r2e_tests (if it exists)
if [ -d "/r2e_tests" ]; then
    find /r2e_tests -name '*.pyc' -delete 2>/dev/null || true
    # ...
fi
```

### 6. Empty Tensor Metrics Error

**Problem**: When some trajectories fail, the advantages tensor can be empty, causing `torch.max()` to fail.

**Error**:
```
RuntimeError: max(): Expected reduction dim to be specified for input.numel() == 0.
```

**Solution**: Added try-except handling in `verl_trainer.py`:
```python
try:
    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
except RuntimeError as e:
    if "numel() == 0" in str(e):
        logging.warning(f"Skipping data metrics due to empty tensors: {e}")
    else:
        raise
```

**Note**: Make sure to use Python's `logging` module, not the wandb `logger` variable which is a `Tracking` object.

### 7. OOM During Backward Pass

**Problem**: Even with a small model (0.5B), the backward pass runs out of memory on 20GB MIG GPUs due to long context from 50 agent iterations.

**Error**:
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate X GiB.
```

**Solution**: Aggressively reduce context lengths and iterations:

**In `slurm/config/dev.conf`:**
```bash
MAX_PROMPT_LENGTH=2048   # Reduced from 4096
MAX_RESPONSE_LENGTH=1024 # Reduced from 2048
MAX_MODEL_LEN=4096       # Reduced from 8192
```

**In `examples/run_verl/verl_oh.yaml`:**
```yaml
generator:
  max_iterations: 20     # Reduced from 50
  max_prompt_length: 2048
  sampling_params:
    max_tokens: 1024
```

### 8. API Key Authentication

**Problem**: Training jobs couldn't connect to runtime server due to API key mismatch.

**Solution**: Added `--disable-auth` flag for trusted SLURM environment.

### 9. Environment Variable Propagation to Ray Workers

**Problem**: `SANDBOX_REMOTE_RUNTIME_API_URL` wasn't passed to Ray workers.

**Solution**: Export via Ray's runtime environment in `verl_main_ppo.py`.

### 10. Image Building Issues

**Problem**: Runtime server wasn't building images - containers started with base images that lack agent-server.

**Root Causes Fixed**:
- Character-by-character image pull bug (double iteration in image check)
- Status mapping bug (`STARTING` → `"running"`, `RUNNING` → `"ready"`)
- Wrong Docker target (`source-minimal` → `binary-minimal`)
- ENTRYPOINT duplication (strip prefix from container commands)

### 11. Frozen Pydantic Model Attribute Assignment

**Problem**: `OHCodeActAgent` is frozen, can't set workspace attribute.

**Solution**: Use `object.__setattr__()` to bypass Pydantic protection.

### 12. LocalWorkspace vs RemoteWorkspace Compatibility

**Problem**: `LocalConversation` requires `LocalWorkspace`, fails with `APIRemoteWorkspace`.

**Solution**: Created `LocalWorkspaceAdapter` that wraps remote workspace and uses temp directory for local state.

### 13. Missing Prompt Templates in Ray Package

**Problem**: Agent prompts directory doesn't exist in Ray-packaged code.

**Solution**: Override `prompt_dir` property to return SDK's built-in prompts directory.

### 14. Entry Script Shell Compatibility

**Problem**: Container uses `/bin/sh` (dash), `source` command not found.

**Solution**: Use `/bin/bash script.sh` instead of `source script.sh`.

### 15. Tool Name Mismatch in Action Parser

**Problem**: The action parser only recognized legacy tool names (`execute_bash`, `str_replace_editor`), but the config loaded tools with new names (`terminal`, `file_editor`), causing "No function call detected" errors.

**Error**:
```
WARNING: No function call detected. Please use a tool to continue.
```

**Solution**: Added tool name aliases in `_create_action_from_parsed()`:
```python
# Handle bash/terminal/command execution
# Support: terminal (current config), execute_bash, bash, cmd_run (legacy names)
if fn_name in ("terminal", "execute_bash", "bash", "cmd_run"):
    from openhands.sdk.tool.builtins import CmdRunAction
    return CmdRunAction(command=params.get("command", ""))

# Handle file editing
# Support: file_editor (current config), str_replace_editor (legacy name)
if fn_name in ("file_editor", "str_replace_editor"):
    from openhands.sdk.tool.builtins import FileEditAction
    # ...
```

**Location**: `skyrl_agent/agents/oh_codeact/codeact_agent.py:510-529`

### 16. FinishAction Validation Error (Wrong Field Name)

**Problem**: `FinishAction` requires a `message` field, but code was passing `thought`.

**Error**:
```
2 validation errors for FinishAction
message
  Field required [type=missing, ...]
thought
  Extra inputs are not permitted [type=extra_forbidden, ...]
```

**Solution**: Changed `FinishAction(thought=...)` to `FinishAction(message=...)`:
```python
if fn_name == "finish":
    message = params.get("message", params.get("thought", "Task completed."))
    return FinishAction(message=message)
```

**Location**: `skyrl_agent/agents/oh_codeact/codeact_agent.py:506-508`

### 17. ActionEvent Validation Error (Missing Required Fields)

**Problem**: The SDK's `ActionEvent` requires many fields (`thought`, `tool_name`, `tool_call_id`, `tool_call`, `llm_response_id`) that weren't being provided.

**Error**:
```
5 validation errors for ActionEvent
thought
  Field required [type=missing, ...]
tool_name
  Field required [type=missing, ...]
tool_call_id
  Field required [type=missing, ...]
tool_call
  Field required [type=missing, ...]
llm_response_id
  Field required [type=missing, ...]
```

**Solution**: Created helper methods `_create_action_event()` and `_create_finish_action_event()`:
```python
def _create_action_event(
    self,
    action: "Action",
    tool_name: str,
    arguments: str = "{}",
    thought_text: str = "",
) -> ActionEvent:
    """Create a properly formatted ActionEvent for any action."""
    from uuid import uuid4

    tool_call_id = f"call_{uuid4().hex[:24]}"
    llm_response_id = f"resp_{uuid4().hex[:24]}"

    tool_call = MessageToolCall(
        id=tool_call_id,
        name=tool_name,
        arguments=arguments,
        origin="completion",
    )

    return ActionEvent(
        source="agent",
        thought=[TextContent(text=thought_text)] if thought_text else [],
        reasoning_content=None,
        thinking_blocks=[],
        responses_reasoning_item=None,
        tool_call=tool_call,
        tool_name=tool_name,
        tool_call_id=tool_call_id,
        llm_response_id=llm_response_id,
        action=action,
    )
```

**Location**: `skyrl_agent/agents/oh_codeact/codeact_agent.py:198-262`

### 18. MessageToolCall Validation Error (Missing `origin` Field)

**Problem**: `MessageToolCall` requires an `origin` field with value `"completion"` or `"responses"`.

**Error**:
```
1 validation error for MessageToolCall
origin
  Field required [type=missing, ...]
```

**Solution**: Added `origin="completion"` to MessageToolCall creation:
```python
tool_call = MessageToolCall(
    id=tool_call_id,
    name=tool_name,
    arguments=arguments,
    origin="completion",  # Required field: "completion" or "responses"
)
```

**Location**: `skyrl_agent/agents/oh_codeact/codeact_agent.py:226-231`

---

## Quick Reference

### Launch Commands

```bash
cd /n/holylabs/janapa_reddi_lab/Lab/jjma/coding-agent-rl/SkyRL

# Launch E2E training with dev config
./slurm/launch.sh --config dev

# Runtime only (for debugging)
./slurm/launch.sh --config dev --runtime-only

# Skip runtime (use existing)
./slurm/launch.sh --config dev --skip-runtime --runtime-url "http://hostname:8000"
```

### Monitoring

```bash
# Check job status
squeue -u $USER

# Find latest job directory
ls -lt .skyrl_jobs/ | head -5

# Tail runtime logs
tail -f .skyrl_jobs/<timestamp>/runtime_*.err

# Tail training logs
tail -f .skyrl_jobs/<timestamp>/training_*.err

# Check runtime health
curl http://<runtime-node>:8000/health

# Check saved checkpoints
ls -la /n/netscratch/janapa_reddi_lab/Lab/jjma/ckpts/skyrl-dev/skyrl-dev-test/
```

### Troubleshooting

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| `invalid internal status` | Stale podman state | Should auto-clean; if persists, run `podman system reset -f` |
| Container starts but `/alive` fails | Wrong Docker target | Ensure using `binary-minimal` |
| `No available worker slots` | Batch too large | Reduce BATCH_SIZE*ROLLOUT_N |
| Image building slow | First build for this base | ~5 min first time, then cached |
| `Python.h not found` | Ray workers using wrong Python | Check `UV_PYTHON` is set |
| `cannot pickle '_thread.RLock'` | deepcopy on Pydantic | Use `safe_deepcopy()` |
| `/r2e_tests: No such file` | Entry script issue | Fixed - uses conditional now |
| `numel() == 0` in metrics | Failed trajectories | Fixed - skips metrics gracefully |
| OOM during backward pass | Context too long | Reduce `max_iterations`, `max_tokens` |
| `No function call detected` | Tool name mismatch | Fixed - parser accepts both old/new names |
| `validation errors for FinishAction` | Wrong field name | Fixed - use `message` not `thought` |
| `validation errors for ActionEvent` | Missing required fields | Fixed - use `_create_action_event()` helper |
| `validation error for MessageToolCall` | Missing `origin` field | Fixed - add `origin="completion"` |

---

## Files Modified

### SkyRL/slurm/
- `lib/common.sh` - Added `setup_isolated_podman_storage()`, `setup_cuda_for_triton()`, `setup_python_for_uv()`
- `jobs/verl_training.sbatch` - MIG fix, CUDA setup, enforce_eager
- `jobs/runtime_server.sbatch` - Podman isolation, auth disabled, build config
- `config/dev.conf` - Test partition settings, reduced context lengths, excluded nodes

### SkyRL/skyrl-agent/
- `skyrl_agent/agents/base.py` - Added `safe_deepcopy()` function
- `skyrl_agent/agents/oh_codeact/codeact_agent.py` - Major fixes:
  - Added `_create_action_event()` helper for ActionEvent validation (lines 198-244)
  - Added `_create_finish_action_event()` helper for finish actions (lines 246-262)
  - Fixed tool name aliases in `_create_action_from_parsed()` (lines 510-529)
  - Fixed FinishAction to use `message` field (lines 506-508)
  - Fixed MessageToolCall to include `origin` field (line 230)
  - Fixed Pydantic frozen model attribute assignment
  - Fixed prompt_dir override for Ray packaging
- `skyrl_agent/agents/oh_codeact/codeact_runner.py` - LocalWorkspaceAdapter, conversation handling
- `skyrl_agent/tasks/swebench/utils.py` - CommandResult compatibility, entry script fixes
- `skyrl_agent/tasks/swebench/scripts/setup/instance_r2e_entry.sh` - Conditional `/r2e_tests` handling
- `skyrl_agent/integrations/verl/verl_trainer.py` - Import fixes, empty tensor handling
- `skyrl_agent/integrations/verl/verl_main_ppo.py` - Ray env var propagation
- `examples/run_verl/verl_oh.yaml` - Reduced iterations and context lengths

### slurm-remote-runtime/
- `slurm_runtime/api_server.py` - Status mapping, ENTRYPOINT handling, build integration
- `slurm_runtime/image_build_manager.py` - SDK build system, binary-minimal target, caching

---

## Configuration Files

### dev.conf Key Settings

```bash
# Model (smaller to fit in 20GB MIG)
MODEL="Qwen/Qwen2.5-Coder-0.5B-Instruct"

# Context lengths (reduced for backward pass memory)
MAX_PROMPT_LENGTH=2048
MAX_RESPONSE_LENGTH=1024
MAX_MODEL_LEN=4096

# GPU settings
GPU_MEMORY_UTILIZATION=0.5  # Leave room for gradients
GPUS_PER_NODE=1
```

### verl_oh.yaml Key Settings

```yaml
generator:
  num_trajectories: 2
  max_iterations: 20  # Reduced from 50
  max_prompt_length: 2048
  sampling_params:
    max_tokens: 1024
    stop: ["</function>"]  # XML-style function calling
```

---

## Next Steps

### High Priority
- [x] ~~Debug Triton/gcc compilation failure~~ (Fixed via UV_PYTHON)
- [x] ~~Verify full training step completes~~ (Working - checkpoints saving)
- [x] ~~Fix SDK Pydantic validation errors~~ (Fixed - ActionEvent, MessageToolCall, FinishAction)
- [ ] Evaluate agent behavior - check XML function call quality
- [ ] Tune system prompt for better tool usage

### Medium Priority
- [ ] Test with larger context once stable
- [ ] Increase `max_iterations` if memory allows
- [ ] Add image pre-building script for production
- [ ] Monitor and log build times for optimization
- [ ] Fix APIRemoteWorkspace cleanup warnings (non-critical)

### Low Priority
- [ ] Investigate Ray runtime_env to avoid venv recreation
- [ ] Create production config with larger batch sizes
- [ ] Support multi-node training (EFA-enabled partitions)

---

## Contact and Resources

- **SkyRL Repository**: `/n/holylabs/janapa_reddi_lab/Lab/jjma/coding-agent-rl/SkyRL`
- **Runtime Repository**: `/n/holylabs/janapa_reddi_lab/Lab/jjma/coding-agent-rl/SkyRL/slurm-remote-runtime`
- **Data Location**: `/n/netscratch/janapa_reddi_lab/Lab/jjma/data/r2e-all/`
- **Checkpoint Location**: `/n/netscratch/janapa_reddi_lab/Lab/jjma/ckpts/skyrl-dev/`
- **WandB Project**: https://wandb.ai/jeffreyma/skyrl-dev
