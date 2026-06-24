"""FSDP-native SkyRL entrypoint that registers the ``bird`` SQL env.

The vanilla ``skyrl.train.entrypoints.main_base`` only knows about envs that
are present in ``skyrl_gym``'s registry. ``bird`` lives in the Arctic-RL
integration tree at ``integrations/arctic-rl/arctic_rl/envs/`` and is only
registered as a side-effect of ``import arctic_rl.envs``. That side-effect
needs to fire on BOTH:

  1. The driver process (this script), and
  2. The ``skyrl_entrypoint`` Ray actor that owns the generator/env factory.

We accomplish (1) by adding the integration directory to ``sys.path`` and
importing ``arctic_rl.envs`` directly. We accomplish (2) by:

  - Adding ``PYTHONPATH`` for Ray workers via a monkey-patch on
    ``skyrl.train.utils.utils.prepare_runtime_environment`` so that the
    integration directory is on the worker's import path.
  - Monkey-patching ``main_base.skyrl_entrypoint`` to re-export the same
    Ray task body but with an extra ``import arctic_rl.envs`` at the top —
    this is the trigger that actually populates the registry on the worker
    process.

This wrapper is purely a compatibility shim. The SkyRL FSDP training code
path is untouched; only the env-registration plumbing is patched.
"""

import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve()
_ARCTIC_RL_INTEGRATION_DIR = _HERE.parents[1]
sys.path.insert(0, str(_ARCTIC_RL_INTEGRATION_DIR))
import arctic_rl.envs  # noqa: F401  side-effect: registers `bird` on driver

import ray  # noqa: E402

import skyrl.train.entrypoints.main_base as _mb  # noqa: E402
import skyrl.train.utils.utils as _skyrl_utils  # noqa: E402

# Liger fused linear-CE is enabled in the FSDP policy worker via
# ``SKYRL_USE_LIGER=1`` (see fsdp_worker.py); we propagate that env var to all
# Ray actors via the ``_patched_prepare`` monkey-patch below. The Arctic-RL
# integration enables Liger by default (``arctic_rl.use_liger=true``) — without
# it, a 32B Qwen3 model with vocab=151936 and packed-seq up to 36864 OOMs
# materializing the full LMHead logits tensor during loss computation.

_arctic_rl_path_str = str(_ARCTIC_RL_INTEGRATION_DIR)

_original_prepare = _skyrl_utils.prepare_runtime_environment


def _patched_prepare(cfg):
    env_vars = _original_prepare(cfg)
    existing_pp = env_vars.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    if _arctic_rl_path_str not in existing_pp.split(":"):
        env_vars["PYTHONPATH"] = (
            _arctic_rl_path_str + (":" + existing_pp if existing_pp else "")
        )
    # Forward SKYRL_USE_LIGER to Ray actors so fsdp_worker.py can opt into
    # Liger fused linear-CE. NOT forwarding PYTORCH_CUDA_ALLOC_CONF —
    # vLLM's CuMemAllocator is incompatible with expandable_segments and
    # asserts at startup.
    if "SKYRL_USE_LIGER" in os.environ:
        env_vars["SKYRL_USE_LIGER"] = os.environ["SKYRL_USE_LIGER"]
    return env_vars


_skyrl_utils.prepare_runtime_environment = _patched_prepare


@ray.remote(num_cpus=1)
def _skyrl_entrypoint_with_bird(cfg):
    import sys as _sys

    if _arctic_rl_path_str not in _sys.path:
        _sys.path.insert(0, _arctic_rl_path_str)
    import arctic_rl.envs  # noqa: F401  side-effect: registers `bird` on worker

    exp = _mb.BasePPOExp(cfg)
    exp.run()


_mb.skyrl_entrypoint = _skyrl_entrypoint_with_bird


if __name__ == "__main__":
    _mb.main()
