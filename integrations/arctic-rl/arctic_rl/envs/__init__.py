"""Arctic RL env wrappers for skyrl-gym.

Registers domain envs that the upstream `skyrl-gym` package doesn't ship but
that the Arctic RL stack uses for verl-equivalent training recipes (BIRD/SQL).

Registration happens at integration-import time so any recipe that imports
the ``arctic_rl`` package — including the dispatched main_base entry path —
sees these envs without modifying skyrl-gym itself.
"""

from skyrl_gym.envs.registration import register

register(
    id="bird",
    entry_point="arctic_rl.envs.bird:BirdEnv",
)

__all__ = []
