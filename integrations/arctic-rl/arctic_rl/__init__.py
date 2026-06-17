"""Arctic RL backend for SkyRL.

Provides ``ArcticPPOTrainer`` and ``ArcticGenerator`` that route all GPU
work to an Arctic RL server, allowing any SkyRL recipe to switch backends
by setting ``trainer.arctic_rl={}`` (and ``colocate_all: false``).

Install::

    uv sync --extra arctic-rl

Usage in a recipe::

    from arctic_rl import ArcticPPOTrainer, ArcticGenerator

The folder on disk is ``integrations/arctic-rl/arctic_rl/`` — top-level sibling of
``skyrl/`` (matching the legacy ``skyrl-tx/`` placement). The Python
package is ``arctic_rl`` (top-level); it is distinct from the upstream
``arctic_platform`` package (which exposes the actual client under
``arctic_platform.rl``) — both coexist at import time without collision.

Dependencies:
    arctic_platform — pip package providing ArcticRLClient/Server
        (``from arctic_platform.rl import create_arctic_rl_client, ArcticRLClientConfig``)
"""

from arctic_rl.trainer import ArcticPPOTrainer
from arctic_rl.generator import ArcticGenerator

# Side-effect import: registers `bird` (and any future Arctic-RL-shipped
# envs) with skyrl_gym so that recipes can do
# `environment.env_class=bird` without modifying skyrl-gym's upstream
# registry. Importing `arctic_rl` is sufficient.
from arctic_rl import envs as _envs  # noqa: F401

__all__ = ["ArcticPPOTrainer", "ArcticGenerator"]
