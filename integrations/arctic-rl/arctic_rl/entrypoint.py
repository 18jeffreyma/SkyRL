"""Entrypoint for training with the Arctic RL backend.

Launches an ``ArcticRLClient`` (on-prem local server mode by default)
and wires it into SkyRL's trainer and generator.  No FSDP/Megatron GPU
workers are created — all GPU work happens on the Arctic RL server.

GPU layout and colocation are configured via standard SkyRL knobs;
ARL-specific settings live under ``trainer.arctic_rl`` (see
``ArcticRLTrainerConfig``).

This entrypoint is invoked indirectly by ``skyrl.train.entrypoints.main_base``
when ``trainer.arctic_rl is not None``. Researchers should use the standard
``main_base`` entrypoint and switch backends via config alone.
"""

import os
import sys
from typing import Optional

import ray
from loguru import logger

from arctic_training.arctic_rl.client import ArcticRLClient
from arctic_training.arctic_rl.config import ArcticRLClientConfig
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.utils import validate_cfg

from arctic_rl import ArcticGenerator, ArcticPPOTrainer
from arctic_rl.config import build_rl_config


class ArcticRLExp(BasePPOExp):

    def __init__(
        self,
        cfg: SkyRLTrainConfig,
        reconnect_config: Optional[ArcticRLClientConfig] = None,
    ):
        n_samples = cfg.generator.n_samples_per_prompt
        mini_batch_size = cfg.trainer.policy_mini_batch_size * n_samples
        train_batch_size = cfg.trainer.train_batch_size * n_samples
        grad_accum_steps = max(1, train_batch_size // mini_batch_size)
        lr = cfg.trainer.policy.optimizer_config.lr

        if reconnect_config is not None:
            self.arctic_client = ArcticRLClient(reconnect_config)
        else:
            self.arctic_client = ArcticRLClient(build_rl_config(cfg))

        logger.info(
            f"DeepSpeed config: lr={lr}, grad_accum_steps={grad_accum_steps}, "
            f"mini_batch={mini_batch_size}, train_batch={train_batch_size}"
        )
        logger.info(
            f"ArcticRLClient ready — "
            f"training_job={self.arctic_client.training_job_id}, "
            f"sample_job={self.arctic_client.sampling_job_id}, "
            f"log_prob_job={self.arctic_client.log_prob_job_id}"
        )
        super().__init__(cfg)

    def get_generator(self, cfg, tokenizer, inference_engine_client):
        return ArcticGenerator(
            arctic_client=self.arctic_client,
            tokenizer=tokenizer,
            sampling_params=cfg.generator.sampling_params,
        )

    def get_trainer(self, cfg, tracker, tokenizer, train_dataset, eval_dataset,
                    inference_engine_client, generator, colocate_pg):
        return ArcticPPOTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
            arctic_client=self.arctic_client,
        )

    def _setup_trainer(self):
        logger.info("Setting up ArcticRL trainer (GPU work delegated to on-prem server)")
        os.makedirs(self.cfg.trainer.export_path, exist_ok=True)
        os.makedirs(self.cfg.trainer.ckpt_path, exist_ok=True)

        from arctic_rl.trainer import _ArcticInferenceEngineStub
        ie_stub = _ArcticInferenceEngineStub(client=self.arctic_client)

        tracker = self.get_tracker()
        generator = self.get_generator(self.cfg, self.tokenizer, None)
        trainer = self.get_trainer(
            cfg=self.cfg,
            tracker=tracker,
            tokenizer=self.tokenizer,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            inference_engine_client=ie_stub,
            generator=generator,
            colocate_pg=self.colocate_pg,
        )
        trainer.build_models()
        if self.arctic_client.config.colocate:
            trainer.colocate_all = True
        return trainer


@ray.remote(num_cpus=1)
def skyrl_entrypoint(
    cfg: SkyRLTrainConfig,
    reconnect_config: Optional[ArcticRLClientConfig] = None,
):
    exp = ArcticRLExp(cfg, reconnect_config=reconnect_config)
    exp.run()


def main() -> None:
    """Arctic RL entrypoint. Reachable two ways: direct (``uv run -m
    arctic_rl.entrypoint``) or via core dispatch (``python -m
    skyrl.train.entrypoints.main_base trainer.backend=arctic_rl``).
    Both paths parse with ``ArcticSkyRLConfig`` here."""
    from arctic_rl.config import ArcticSkyRLConfig
    cfg = ArcticSkyRLConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)

    rl_config = build_rl_config(cfg)
    logger.info("Pre-initializing ArcticRL jobs (before ray.init)…")
    pre_client = ArcticRLClient(rl_config)
    reconnect_cfg = pre_client.reconnect_config()
    logger.info(
        f"ArcticRL jobs ready — training={pre_client.training_job_id}, "
        f"sample={pre_client.sampling_job_id}, log_prob={pre_client.log_prob_job_id}"
    )

    from skyrl.train.utils.utils import prepare_runtime_environment
    env_vars = prepare_runtime_environment(cfg)
    # Forward ARCTIC_* env vars to Ray workers — moved here from core utils per
    # reviewer feedback (core stays integration-agnostic).
    env_vars.update({k: v for k, v in os.environ.items() if k.startswith("ARCTIC_")})
    ray.init(num_gpus=0, runtime_env={"env_vars": env_vars})
    ray.get(skyrl_entrypoint.remote(cfg, reconnect_config=reconnect_cfg))


if __name__ == "__main__":
    main()
