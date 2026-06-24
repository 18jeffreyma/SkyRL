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
from typing import Any, Optional

import ray
from loguru import logger

from arctic_platform.rl import ArcticRLClientConfig, create_arctic_rl_client
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
        server_state: Optional[Any] = None,
    ):
        n_samples = cfg.generator.n_samples_per_prompt
        mini_batch_size = cfg.trainer.policy_mini_batch_size * n_samples
        train_batch_size = cfg.trainer.train_batch_size * n_samples
        grad_accum_steps = max(1, train_batch_size // mini_batch_size)
        lr = cfg.trainer.policy.optimizer_config.lr

        # arctic_platform.rl.create_arctic_rl_client takes an optional
        # server_state used by the ray-protocol client to reattach to an
        # already-initialized server actor (driver pre-init pattern, see main()).
        if reconnect_config is not None:
            self.arctic_client = create_arctic_rl_client(reconnect_config, server_state)
        else:
            self.arctic_client = create_arctic_rl_client(build_rl_config(cfg), server_state)

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

        # Source `colocate` from the main `cfg` (Hydra/OmegaConf full schema),
        # never from `self.arctic_client.config.colocate` — see
        # `_ArcticDispatch.__init__` comment for the reconnect_config() strip
        # rationale.
        arl_cfg = getattr(self.cfg.trainer, "arctic_rl", None)
        cfg_colocate = bool(getattr(arl_cfg, "colocate", False)) if arl_cfg else False

        from arctic_rl.trainer import _ArcticInferenceEngineStub
        ie_stub = _ArcticInferenceEngineStub(client=self.arctic_client, colocate=cfg_colocate)

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
        if cfg_colocate:
            trainer.colocate_all = True
        return trainer


@ray.remote(num_cpus=1)
def skyrl_entrypoint(
    cfg: SkyRLTrainConfig,
    reconnect_config: Optional[ArcticRLClientConfig] = None,
    server_state: Optional[Any] = None,
):
    exp = ArcticRLExp(cfg, reconnect_config=reconnect_config, server_state=server_state)
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
    pre_client = create_arctic_rl_client(rl_config)
    reconnect_cfg = pre_client.reconnect_config()
    # NOTE: Arctic-Platform's ArcticRLRayClient.reconnect_config() strips the
    # config down to {backend, model_name, *_job_id, comm_protocol} — it does
    # NOT carry policy flags like `colocate`. We DO NOT depend on
    # `_client.config.<flag>` anywhere in the integration; all policy decisions
    # read from the main `cfg` (Hydra/OmegaConf, full schema) directly. See
    # `_ArcticDispatch.__init__` and `_setup_trainer` for the source-of-truth.
    # For the ray comm protocol, the client owns an in-process server actor
    # state that must be handed to the reconnecting worker. For http it's None.
    server_state = (
        pre_client.get_server_state() if rl_config.comm_protocol == "ray" else None
    )
    logger.info(
        f"ArcticRL jobs ready — training={pre_client.training_job_id}, "
        f"sample={pre_client.sampling_job_id}, log_prob={pre_client.log_prob_job_id}"
    )

    from skyrl.train.utils.utils import prepare_runtime_environment
    env_vars = prepare_runtime_environment(cfg)
    # Forward ARCTIC_* env vars to Ray workers — moved here from core utils per
    # reviewer feedback (core stays integration-agnostic).
    env_vars.update({k: v for k, v in os.environ.items() if k.startswith("ARCTIC_")})
    # Forward WANDB_* env vars too. prepare_runtime_environment forwards only
    # WANDB_API_KEY; on a non-head landing of skyrl_entrypoint (Ray picks any
    # worker), missing WANDB_BASE_URL silently sends local-wandb keys to
    # api.wandb.ai (-> CommError 401) and missing WANDB_PROJECT etc. forks the
    # run to the wrong project. Mirror the ARCTIC_* policy.
    env_vars.update({k: v for k, v in os.environ.items() if k.startswith("WANDB_")})
    # Make the ``arctic_rl`` integration importable in Ray workers. The driver
    # discovers it via sys.path (added by main_base), but Ray workers inherit
    # only this runtime_env — without it, importing skyrl_entrypoint /
    # deserializing the ArcticSkyRLConfig fails with ``No module named
    # 'arctic_rl'``. ``parents[1]`` of this file is the dir holding the package.
    _integration_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _existing_pp = env_vars.get("PYTHONPATH") or os.environ.get("PYTHONPATH", "")
    env_vars["PYTHONPATH"] = _integration_root + (
        os.pathsep + _existing_pp if _existing_pp else ""
    )
    runtime_env = {"env_vars": env_vars}
    # arctic_platform.rl.create_arctic_rl_client(ray) starts a GPU Ray cluster
    # during pre-init above, so the driver-side ray.init must (a) reuse it
    # (ignore_reinit_error) and (b) pass runtime_env at TASK granularity
    # rather than via init — init's runtime_env is ignored on the second call
    # to an already-initialized cluster, which would silently drop our
    # PYTHONPATH and crash workers with `ModuleNotFoundError: arctic_rl`.
    ray.init(num_gpus=0, runtime_env=runtime_env, ignore_reinit_error=True)
    ray.get(
        skyrl_entrypoint.options(runtime_env=runtime_env).remote(
            cfg, reconnect_config=reconnect_cfg, server_state=server_state,
        )
    )


if __name__ == "__main__":
    main()
