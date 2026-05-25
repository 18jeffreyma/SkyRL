"""Arctic RL configuration types.

Defines:
- ``ArcticRLTrainerConfig``: backend-specific knobs (colocate, zero_stage, ...)
- ``ArcticTrainerConfig``: extends core ``TrainerConfig`` with ``arctic_rl`` field
- ``ArcticSkyRLConfig``: top-level config used by the integration's entrypoint
- ``build_rl_config(cfg)``: translates ``SkyRLTrainConfig`` → ``ArcticRLClientConfig``

These live in the integration to keep core SkyRL integration-agnostic — core only
knows about a generic ``trainer.backend: str`` field that lazily dispatches here.
All shared knobs (GPU counts, vLLM settings, colocation) are derived from existing
SkyRL config fields by ``build_rl_config``.
"""

from dataclasses import dataclass
from typing import Optional

from arctic_training.arctic_rl.config import ArcticRLClientConfig
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.config.config import BaseConfig, TrainerConfig, make_config


# ---------------------------------------------------------------------------
# Arctic RL backend configuration
# ---------------------------------------------------------------------------


@dataclass
class ArcticRLTrainerConfig(BaseConfig):
    """Arctic RL (DeepSpeed) backend settings.

    Only contains params unique to the Arctic RL server with no equivalent in
    the standard SkyRL config.  Shared knobs are derived in ``build_rl_config``.
    """

    colocate: bool = False
    """Share GPUs between training and inference on the ARL server.

    Distinct from ``trainer.placement.colocate_all`` which controls Ray placement
    groups.  ARL colocation is server-side GPU sharing — ``colocate_all`` must
    stay ``false`` when using the ARL backend.
    """
    use_zorro: bool = False
    """Enable ZoRRO (prompt deduplication) on the training server."""
    zero_stage: int = 0
    """DeepSpeed ZeRO stage (0, 2, or 3)."""
    log_prob_gpus: int = 0
    """Number of GPUs for log-prob computation (0 = skip separate log-prob)."""
    offload_optimizer: bool = False
    """Offload optimizer state to CPU when ``zero_stage >= 2``."""
    host: str = "localhost"
    """Server host for HTTP comm protocol; ignored for Ray."""
    port: int = 7000
    """Server port for HTTP comm protocol; ignored for Ray."""
    startup_timeout: float = 300.0
    """Seconds to wait for server jobs to come up."""
    server_logs: bool = False
    """Forward server logs to stdout for debugging."""


@dataclass
class ArcticTrainerConfig(TrainerConfig):
    """``TrainerConfig`` extended with the Arctic RL field.  Used only when
    ``trainer.backend == "arctic_rl"`` is set in the recipe."""

    arctic_rl: Optional[ArcticRLTrainerConfig] = None
    """Arctic RL backend settings.  ``None`` falls back to defaults."""


# Top-level config for arctic_rl recipes.  Used by the integration's entrypoint
# either directly (``uv run -m integrations.arctic_rl.entrypoint``) or via core
# dispatch (``trainer.backend=arctic_rl`` from ``main_base``).
ArcticSkyRLConfig = make_config(trainer_cls=ArcticTrainerConfig)


# ---------------------------------------------------------------------------
# Translation: SkyRLTrainConfig → ArcticRLClientConfig
# ---------------------------------------------------------------------------


def build_rl_config(cfg: SkyRLTrainConfig) -> ArcticRLClientConfig:
    """Build ``ArcticRLClientConfig`` from ``SkyRLTrainConfig``.

    Raises ``ValueError`` if ``cfg.trainer.arctic_rl`` is not set.
    """
    arl = cfg.trainer.arctic_rl
    if arl is None:
        raise ValueError(
            "trainer.arctic_rl must be set when using the Arctic RL entrypoint. "
            "Add 'trainer.arctic_rl={}' to your config overrides to enable it "
            "with defaults, or set individual fields like "
            "'trainer.arctic_rl.zero_stage=2'."
        )

    # -- Derived from existing SkyRL configs ---------------------------------
    training_gpus = (
        cfg.trainer.placement.policy_num_gpus_per_node
        * cfg.trainer.placement.policy_num_nodes
    )
    sampling_gpus = cfg.generator.inference_engine.num_engines
    colocate = arl.colocate
    vllm_gpu_mem = cfg.generator.inference_engine.gpu_memory_utilization
    tp_size = cfg.generator.inference_engine.tensor_parallel_size

    # -- From ARL-specific config --------------------------------------------
    lr = cfg.trainer.policy.optimizer_config.lr
    n_samples = cfg.generator.n_samples_per_prompt
    mini_batch_size = cfg.trainer.policy_mini_batch_size * n_samples
    train_batch_size = cfg.trainer.train_batch_size * n_samples
    grad_accum_steps = max(1, train_batch_size // mini_batch_size // training_gpus)

    # -- vLLM config (only for colocated or TP > 1) --------------------------
    vllm_cfg: dict | None = None
    if colocate:
        vllm_cfg = {
            "gpu_memory_utilization": vllm_gpu_mem,
            "enforce_eager": True,
        }
    if tp_size > 1:
        vllm_cfg = vllm_cfg or {}
        vllm_cfg["tensor_parallel_size"] = tp_size

    # -- DeepSpeed config ----------------------------------------------------
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": grad_accum_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": lr,
                "betas": list(cfg.trainer.policy.optimizer_config.betas)
                    if hasattr(cfg.trainer.policy.optimizer_config, "betas")
                    else [0.9, 0.999],
                "eps": getattr(cfg.trainer.policy.optimizer_config, "eps", 1e-8),
                "weight_decay": getattr(cfg.trainer.policy.optimizer_config, "weight_decay", 0.0),
            },
        },
        "gradient_clipping": cfg.trainer.policy.optimizer_config.max_grad_norm,
        "bf16": {"enabled": True},
    }

    zero_cfg: dict = {"stage": arl.zero_stage}
    if arl.zero_stage >= 2 and arl.offload_optimizer:
        zero_cfg["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
    ds_config["zero_optimization"] = zero_cfg

    # -- ZoRRO worker config -------------------------------------------------
    ds_worker_config = None
    if arl.use_zorro:
        ds_worker_config = {
            "use_zorro": True,
            "response_len": cfg.generator.sampling_params.max_generate_length,
            "max_token_len": (cfg.trainer.max_prompt_length + cfg.generator.sampling_params.max_generate_length)
                * cfg.trainer.policy_mini_batch_size * n_samples,
            "rollout_n": n_samples,
            "temperature": getattr(cfg.generator.sampling_params, "temperature", 1.0),
            "use_unpad": True,
        }

    return ArcticRLClientConfig(
        model_name=cfg.trainer.policy.model.path,
        backend="local",
        host=arl.host,
        port=arl.port,
        training_gpus=training_gpus,
        sampling_gpus=sampling_gpus,
        log_prob_gpus=arl.log_prob_gpus,
        log_prob_engine="vllm",
        colocate=colocate,
        vllm_config=vllm_cfg,
        ds_config=ds_config,
        ds_worker_config=ds_worker_config,
        training_config={
            "dtype": "bfloat16",
            "gradient_checkpointing": True,
            "gradient_accumulation_steps": grad_accum_steps,
            "optimizer": {
                "lr": lr,
                "weight_decay": 0.0,
                "beta1": 0.9,
                "beta2": 0.999,
                "lr_scheduler_type": "constant",
                "gradient_clipping": cfg.trainer.policy.optimizer_config.max_grad_norm,
                "warmup_steps_proportion": 0.0,
            },
        },
        startup_timeout=arl.startup_timeout,
        server_logs=arl.server_logs,
    )
