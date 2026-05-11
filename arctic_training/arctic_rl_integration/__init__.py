"""Arctic RL backend for SkyRL.

Provides ``ArcticPPOTrainer`` and ``ArcticGenerator`` that route all GPU
work to an Arctic RL server, allowing any SkyRL recipe to switch backends
by setting ``trainer.arctic_rl={}`` (and ``colocate_all: false``).

Install::

    uv sync --extra arctic-rl

Usage in a recipe::

    from arctic_rl_integration import ArcticPPOTrainer, ArcticGenerator

The folder on disk is ``arctic_training/arctic_rl_integration/`` — Charlie
asked for the integration code to sit at a top-level ``arctic_training/``
sibling of ``skyrl/`` (matching the legacy ``skyrl-tx/`` placement). The
inner module name ``arctic_rl_integration`` is intentionally distinct from
the upstream ``arctic_training`` Python package (the DSS server library),
which we depend on via pip — keeping the two namespaces separate avoids a
collision at import time.

Dependencies:
    arctic_training — pip package providing ArcticRLClient/Server
"""

from arctic_rl_integration.trainer import ArcticPPOTrainer
from arctic_rl_integration.generator import ArcticGenerator

__all__ = ["ArcticPPOTrainer", "ArcticGenerator"]
