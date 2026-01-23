# SkyRL + FASRC Integration Handoff

**Date**: January 22, 2026
**Status**: Core Infrastructure Working - Training Running End-to-End

---

## Summary

SkyRL is being deployed on Harvard FASRC cluster for RL training on SWE-Bench tasks. The system uses a two-job architecture:
- **CPU Node** (test partition): Runtime server for sandboxed code execution via Podman containers
- **GPU Node** (gpu_test partition): VERL PPO training with vLLM inference

After resolving multiple SDK integration issues, the system is close to working end-to-end.

---

## Current Job Status

- **Runtime Job**: 56412282 (Running on holy8a24101)
- **Training Job**: 56413424 (Running on holygpu7c26106) - New run with prompt fix
- **Coordination Directory**: `.skyrl_jobs/20260122_180232_launch/`
- **WandB Run**: https://wandb.ai/jeffreyma/skyrl-dev/runs/giwyz17j

---

## Issues Fixed This Session

### 1. Podman Service Startup Failure
- **Problem**: Podman state corruption causing "invalid internal status" errors
- **Fix**: Added `podman system migrate` before service startup in `runtime_server.sbatch`

### 2. Entry Script Shell Compatibility
- **Problem**: Container uses `/bin/sh` (dash), not bash, so `source` command fails
- **Fix**: Changed to `/bin/bash /swe_util/instance_r2e_entry.sh` in `utils.py`

### 3. LocalConversation/RemoteConversation Incompatibility
- **Problem**: SDK's `LocalConversation` requires `LocalWorkspace`, but we use `APIRemoteWorkspace`
- **Problem**: SDK's `RemoteConversation` tries to serialize agent to server, which fails with custom agents
- **Fix**: Created `LocalWorkspaceAdapter` class that wraps `APIRemoteWorkspace` to satisfy the type check while delegating operations

### 4. LocalWorkspaceAdapter Permission Error
- **Problem**: Adapter used remote path (`/testbed`) for local directory, causing permission denied
- **Fix**: Use `tempfile.mkdtemp()` for local state management

### 5. Missing Prompt Templates (Session 3)
- **Problem**: SDK's base `Agent.prompt_dir` property looks for prompts in `skyrl_agent/agents/oh_codeact/prompts/` which doesn't exist
- **Error**: `Prompt file .../prompts/system_prompt.j2 not found`
- **Fix**: Override `prompt_dir` property in `OHCodeActAgent` to return SDK's built-in prompts directory:
  ```python
  @property
  def prompt_dir(self) -> str:
      import openhands.sdk.agent
      return os.path.join(os.path.dirname(openhands.sdk.agent.__file__), "prompts")
  ```

---

## Files Modified

| File | Changes |
|------|---------|
| `slurm/jobs/runtime_server.sbatch` | Podman migrate fix |
| `slurm/config/dev.conf` | 6h time limit for better GPU allocation |
| `skyrl-agent/skyrl_agent/tasks/swebench/utils.py` | Entry script bash fix |
| `skyrl-agent/skyrl_agent/agents/oh_codeact/codeact_runner.py` | LocalWorkspaceAdapter class |
| `skyrl-agent/skyrl_agent/agents/oh_codeact/codeact_agent.py` | Override `prompt_dir` property |

---

## Key Architecture Notes

### LocalWorkspaceAdapter Pattern

The SDK wasn't designed for our hybrid use case:
- Agent runs locally (GPU node)
- Workspace execution happens remotely (Runtime node)

Solution: `LocalWorkspaceAdapter` makes `APIRemoteWorkspace` look like `LocalWorkspace`:

```python
class LocalWorkspaceAdapter(LocalWorkspace):
    """Wraps APIRemoteWorkspace to satisfy LocalConversation's type check."""

    def __init__(self, remote_workspace: APIRemoteWorkspace):
        # Use temp dir for local state, delegate operations to remote
        local_temp_dir = tempfile.mkdtemp(prefix="skyrl_workspace_")
        super().__init__(working_dir=local_temp_dir)
        self._remote_workspace = remote_workspace

    def execute_command(self, command, timeout=120, **kwargs):
        return self._remote_workspace.execute_command(command, timeout=timeout, **kwargs)

    # ... other delegated methods
```

---

## How to Launch

```bash
cd /n/holylabs/janapa_reddi_lab/Lab/jjma/coding-agent-rl/SkyRL

# Full launch
./slurm/launch.sh --config dev

# Dry run
./slurm/launch.sh --config dev --dry-run

# Monitor
tail -f /path/to/.skyrl_jobs/*/training_*.err
tail -f /path/to/.skyrl_jobs/*/runtime_*.out
```

---

## Integration Tests

Created comprehensive tests in `slurm-remote-runtime/tests/integration/test_r2e_workflow.py`:
- R2E image building
- Container startup and health checks
- Command execution with stdout/stderr capture
- Entry script execution

Run with:
```bash
cd /n/holylabs/janapa_reddi_lab/Lab/jjma/coding-agent-rl/slurm-remote-runtime
./scripts/run_r2e_tests.sh
```

---

## Troubleshooting Quick Reference

| Error | Cause | Solution |
|-------|-------|----------|
| `invalid internal status` | Podman state corrupt | Run `podman system migrate` |
| `source: not found` | Container uses /bin/sh | Use `/bin/bash script.sh` |
| `workspace must be a LocalWorkspace` | Wrong conversation type | Use LocalWorkspaceAdapter |
| `Unknown kind 'OHCodeActAgent'` | RemoteConversation serializes agent | Use LocalConversation with adapter |
| `Permission denied: '/testbed'` | Adapter uses remote path locally | Use tempfile for local dir |
| `Prompt file .../system_prompt.j2 not found` | Agent uses wrong prompts dir | Override `prompt_dir` property |

---

## Next Steps

1. ~~**Verify Current Run**: Monitor jobs 56412282/56412283 to confirm end-to-end flow works~~ ✅ Done - Training running end-to-end
2. **Address Context Window Issues**: Pydantic validation errors on CONTEXT_WINDOW_EXCEEDED action
3. **Production Config**: Create config with larger batch sizes for full training
4. **Checkpoint Resume**: Implement checkpoint loading for continued training
5. **Multi-Node**: Test on EFA-enabled partitions for distributed training

---

## Detailed Notes

See `SkyRL/slurm/FASRC_TESTING_NOTES.md` for comprehensive documentation of all issues and fixes.

---

## Contact

- **Repository**: `/n/holylabs/janapa_reddi_lab/Lab/jjma/coding-agent-rl/SkyRL`
- **Runtime Repo**: `/n/holylabs/janapa_reddi_lab/Lab/jjma/coding-agent-rl/slurm-remote-runtime`
- **WandB**: https://wandb.ai/jeffreyma/skyrl-dev
