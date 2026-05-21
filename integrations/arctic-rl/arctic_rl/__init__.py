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
``arctic_training`` package (which has its own ``arctic_training.arctic_rl``
sub-namespace) — both coexist at import time without collision.

Dependencies:
    arctic_training — pip package providing ArcticRLClient/Server
"""

from arctic_rl.trainer import ArcticPPOTrainer
from arctic_rl.generator import ArcticGenerator

__all__ = ["ArcticPPOTrainer", "ArcticGenerator"]
