"""FSDP-native SkyRL entrypoint that registers the ``bird`` env on the driver
and on Ray workers.

``bird`` lives at ``integrations/arctic-rl/arctic_rl/envs/`` and registers via
``import arctic_rl.envs``. Stock ``main_base`` doesn't import the integration,
so this shim:
  1. imports ``arctic_rl.envs`` on the driver
  2. monkey-patches ``prepare_runtime_environment`` to forward
     ``PYTHONPATH`` + ``SKYRL_USE_LIGER`` to Ray workers
  3. monkey-patches ``main_base.skyrl_entrypoint`` to import
     ``arctic_rl.envs`` inside the Ray actor before running ``BasePPOExp``

No changes to the SkyRL training code itself — env-registration plumbing only.
Removed once ``trainer.override_entrypoint`` lands (PR #5 follow-up).
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

_arctic_rl_path_str = str(_ARCTIC_RL_INTEGRATION_DIR)

_original_prepare = _skyrl_utils.prepare_runtime_environment


def _patched_prepare(cfg):
    env_vars = _original_prepare(cfg)
    existing_pp = env_vars.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    if _arctic_rl_path_str not in existing_pp.split(":"):
        env_vars["PYTHONPATH"] = (
            _arctic_rl_path_str + (":" + existing_pp if existing_pp else "")
        )
    # Forward SKYRL_USE_LIGER (consumed by fsdp_worker.py).
    # NOT forwarding PYTORCH_CUDA_ALLOC_CONF — vLLM CuMemAllocator is
    # incompatible with expandable_segments and asserts at startup.
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
