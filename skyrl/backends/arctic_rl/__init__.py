"""Arctic RL backend for SkyRL.

Provides ``ArcticPPOTrainer`` and ``ArcticGenerator`` that route all GPU
work to an Arctic RL server, allowing any SkyRL recipe to switch backends
by changing one import and setting ``colocate_all: false``.

Install::

    uv sync --extra arctic-rl

Usage in a recipe::

    from skyrl.backends.arctic_rl import ArcticPPOTrainer, ArcticGenerator

Dependencies:
    arctic_training — pip package (arctic_rl_client sub-module)
"""

from skyrl.backends.arctic_rl.arctic_trainer import ArcticPPOTrainer
from skyrl.backends.arctic_rl.arctic_generator import ArcticGenerator

__all__ = ["ArcticPPOTrainer", "ArcticGenerator"]
