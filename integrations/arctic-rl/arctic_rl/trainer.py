"""ArcticPPOTrainer — subclass of RayPPOTrainer for the Arctic RL backend.

Overrides ``build_models()`` to route training operations (forward,
backward, optimizer step, weight sync) to the Arctic RL server via the
``arctic_rl_client`` package.

The server owns the full GRPO computation:
  - per-token log-probs (old + new)
  - group-relative advantage estimation from per-sequence rewards
  - clipped PPO surrogate loss + backward

The client is responsible only for:
  - rollout generation (via ArcticGenerator → server vLLM)
  - reward scoring (via skyrl-gym)
  - sending (sequences, rewards, loss_mask) to the server for training

Dependencies:
    arctic_platform  — on-prem-and-dss-platform Arctic RL client
"""

import asyncio
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch
from loguru import logger

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch, TrainingOutputBatch
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.config import SkyRLTrainConfig
from skyrl.backends.skyrl_train.workers.worker_utils import reduce_metrics


def _run(coro):
    """Run an awaitable from sync context.

    SkyRL's WorkerDispatch protocol is synchronous but the new
    ``arctic_platform.rl`` client methods are async, so the dispatch
    blocks on the coroutine here. There is no outer running loop in the
    Ray actor's main step path, so ``asyncio.run`` is safe; if a caller
    invokes us from inside a loop we fall back to a fresh loop in a worker
    thread.
    """
    try:
        return asyncio.run(coro)
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(
                lambda c=coro: asyncio.new_event_loop().run_until_complete(c)
            ).result()


class ArcticPPOTrainer(RayPPOTrainer):
    """PPO Trainer backed by Arctic RL server (DeepSpeed).

    Drop-in replacement for ``RayPPOTrainer``.  Requires
    ``colocate_all: false`` since all GPU work is on the Arctic RL server.
    """

    def __init__(self, *args, arctic_client, **kwargs):
        self._arctic_client = arctic_client
        self._stashed_rewards = None
        super().__init__(*args, **kwargs)

    def build_models(self, PolicyWorker=None, CriticWorker=None, RefWorker=None):
        """Replace GPU actor creation with a lightweight dispatch to Arctic RL."""
        self.dispatch = _ArcticDispatch(self.cfg, self._arctic_client)
        self.policy_model = _ArcticPolicyStub()
        self.ref_model = None
        self.critic_model = None
        logger.info("ArcticPPOTrainer: build_models → training routed to Arctic RL server")

    # ------------------------------------------------------------------
    # Override: skip separate old log-probs computation
    # ------------------------------------------------------------------

    def fwd_logprobs_values_reward(self, training_input: TrainingInputBatch):
        """No-op — the server computes old log-probs inline during fwd_bwd.

        The GRPO loss falls back to ``logprobs.detach()`` when
        ``old_log_probs_shifted`` is absent from the context, which is
        semantically correct for on-policy training (behavioral policy ==
        current policy at the start of each training step).
        """
        return training_input

    # ------------------------------------------------------------------
    # Override: stash rewards before parent pops them, then delegate
    # ------------------------------------------------------------------

    def compute_advantages_and_returns(self, training_input: TrainingInputBatch):
        """Compute GRPO advantages client-side, then send them to the server.

        We stash the raw rewards so train_critic_and_policy can restore them
        (parent's train() pops rewards before calling train_critic_and_policy),
        then delegate to the parent to compute group-relative advantages normally.
        """
        self._stashed_rewards = training_input["rewards"].clone()
        training_input["values"] = None
        return super().compute_advantages_and_returns(training_input)

    # ------------------------------------------------------------------
    # Override: train step sends raw data to server
    # ------------------------------------------------------------------

    def train_critic_and_policy(self, data: TrainingInputBatch):
        """Send the full batch to the Arctic RL server for training.

        The server handles gradient accumulation internally via
        ``_forward_maybe_backward`` (splitting into micro-batches based on
        DeepSpeed's ``gradient_accumulation_steps``).  Each epoch sends a
        single fwd_bwd + step pair; ``set_gradient_accumulation_boundary``
        ensures the step always triggers a real optimizer update.
        """
        if self._arctic_client.config.colocate:
            _run(self._arctic_client.wake_training())

        if self._stashed_rewards is not None:
            data["rewards"] = self._stashed_rewards
            self._stashed_rewards = None

        data.metadata["global_step"] = self.global_step
        n_samples = self.cfg.generator.n_samples_per_prompt

        all_metrics: Dict[str, List[float]] = defaultdict(list)

        for _epoch in range(self.cfg.trainer.update_epochs_per_batch):
            status = self.dispatch.forward_backward(
                "policy", data,
                loss_fn="grpo",
                loss_fn_config={"n_samples": n_samples},
            )
            for k, v in status.items():
                all_metrics[k].append(v)

            grad_norm = self.dispatch.optim_step("policy")
            if grad_norm is not None:
                all_metrics["grad_norm"].append(grad_norm)

        all_metrics.pop("loss_fn_outputs", None)
        all_metrics.pop("post_process_outputs", None)
        reduced = reduce_metrics(dict(all_metrics))

        for k, v in reduced.items():
            self.all_metrics[f"policy/{k}"] = v

        return reduced


# ---------------------------------------------------------------------------
# Dispatch — duck-types the WorkerDispatch interface used by RayPPOTrainer
# ---------------------------------------------------------------------------

class _ArcticDispatch:
    """Routes ``WorkerDispatch`` calls to the Arctic RL server."""

    def __init__(self, cfg: SkyRLTrainConfig, client):
        self.cfg = cfg
        self.client = client
        self._colocate = client.config.colocate
        arl = cfg.trainer.arctic_rl
        self._cuda_ipc = bool(getattr(arl, "cuda_ipc_weight_sync", False)) if arl is not None else False
        self._low_memory = bool(getattr(arl, "low_memory_weight_sync", False)) if arl is not None else False

    @staticmethod
    def _to_batch(data: TrainingInputBatch, start: int = 0, end: Optional[int] = None) -> dict:
        """Translate SkyRL's ``TrainingInputBatch`` to the
        ``arctic_platform.rl`` server's expected wire shape.

        The server's payload contract (shared with the verl adapter at
        ``verl/workers/remote_client/arctic_rl.py``) is:

          batch = {
            input_ids       [B, S],   # full prompt+response (left-padded)
            attention_mask  [B, S],   # 1 valid / 0 pad
            prompts         [B, P],   # prompt-only slice; the server reads
                                      # ONLY `.shape[1]` to derive
                                      # `response_lens = attention_mask[:, P:]`
                                      # (see processors/pipeline.py:236)
            responses       [B, A],   # response-only slice (consumed by
                                      # loss / advantage processors)
            response_mask   [B, A],   # response-only mask
            position_ids    [B, S],   # derived from attention_mask
          }

        SkyRL stores response-length tensors (``response_mask``,
        ``loss_mask``, ``rewards``, …) natively as ``[B, A]`` and
        ``sequences`` as ``[B, S]`` where S = max prompt + max response
        in the batch. The translation is purely adapter-side; the server
        is framework-agnostic.
        """
        if end is not None:
            data = data[start:end]

        def is_batch_tensor(t: Any) -> bool:
            return torch.is_tensor(t) and t.ndim >= 2

        sequences = data["sequences"]                   # [B, S]
        attention_mask = data["attention_mask"]         # [B, S]
        response_mask = data["response_mask"]           # [B, A] response-only
        full_seq_len = sequences.shape[1]
        response_len = response_mask.shape[1]
        prompt_len = full_seq_len - response_len        # uniform per batch:
        #   `convert_prompts_responses_to_batch_tensors` left-pads prompts
        #   to max_prompt_len and right-pads responses to max_response_len,
        #   so the split point is constant across rows.

        # Build the server-shape batch. We rename `sequences` -> `input_ids`
        # to match the server's expected key.
        batch: Dict[str, Any] = {
            "input_ids": sequences,
            "attention_mask": attention_mask,
            # Server uses `.shape[1]` only -> the slice is sufficient (token
            # values inside the prompt region are also correct here for any
            # future server-side use).
            "prompts": sequences[:, :prompt_len],
            "responses": sequences[:, prompt_len:],
            "response_mask": response_mask,
        }

        # position_ids derived from attention_mask (cumsum-1, with pad
        # positions filled with 1 so they round-trip safely through any
        # rotary-embedding lookup). Matches the verl adapter's behavior
        # when `drop_position_ids=False`.
        pos = attention_mask.long().cumsum(-1) - 1
        pos.masked_fill_(attention_mask == 0, 1)
        batch["position_ids"] = pos

        # Forward any other 2D batch tensors SkyRL produced (e.g.
        # `rollout_logprobs`, `rollout_expert_indices`, `loss_mask`,
        # `rewards`, ...). These keep their native SkyRL shape; the server
        # only reads ones it knows about (others are passed through as
        # context for post-processors / loss functions). We skip
        # `sequences` since it's already aliased to `input_ids` above.
        for k, v in data.items():
            if k in batch or k == "sequences":
                continue
            if is_batch_tensor(v):
                batch[k] = v

        meta = {k: v for k, v in data.items() if not is_batch_tensor(v)}
        return dict(batch=batch, meta=meta)


    # -- WorkerDispatch interface -------------------------------------------

    def forward(self, model: str, data: TrainingInputBatch) -> TrainingOutputBatch:
        batch = self._to_batch(data)
        # arctic_platform.rl: fwd_no_grad takes reference_model: bool (no
        # post_processors kwarg); response carries model outputs under "batch".
        result = _run(self.client.fwd_no_grad(batch, reference_model=False))
        out = TrainingOutputBatch()
        for k, v in result.get("batch", {}).items():
            out[k] = torch.tensor(v) if isinstance(v, list) else v
        out.metadata = {"model": model}
        return out

    def forward_backward(
        self,
        model: str,
        data: TrainingInputBatch,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, float]:
        batch = self._to_batch(data)
        result = _run(self.client.fwd_bwd(
            batch,
            processing={
                "loss_fn": loss_fn or "grpo",
                "config": loss_fn_config or {},
                "post": [],
            },
        ))
        result.pop("job_id", None)
        return result.get("metrics", result)

    def optim_step(self, model: str) -> Optional[float]:
        resp = _run(self.client.step())
        metrics = resp.get("metrics", resp)
        return metrics.get("grad_norm")

    def get_lcm_dp_size(self) -> int:
        return 1

    def save_checkpoint(self, model: str, ckpt_dir: str, tokenizer=None) -> None:
        _run(self.client.save_checkpoint())

    def load_checkpoint(self, model: str, ckpt_dir: str, **kwargs) -> None:
        logger.info(f"Arctic RL: load_checkpoint for {model} — delegated to server")

    def save_hf_model(self, model: str, export_dir: str, tokenizer) -> None:
        logger.info(f"Arctic RL: save_hf_model for {model} — delegated to server")

    def set_lr(self, model: str, learning_rate: float) -> None:
        pass

    def init_weight_sync_state(self, inference_engine_client) -> None:
        pass

    async def save_weights_for_sampler(self) -> None:
        if self._colocate:
            await self.client.empty_training_cache()
            await self.client.wake_training()
            await self.client.wake_inference()
            await self.client.sync_weights(cuda_ipc=self._cuda_ipc, low_memory=self._low_memory)
        else:
            await self.client.sync_weights(cuda_ipc=False, low_memory=self._low_memory)

    def mark_all_offloaded(self) -> None:
        pass

    def empty_cache(self, model=None) -> None:
        pass

    def get_node_ids(self) -> List[str]:
        return []


# ---------------------------------------------------------------------------
# Stub — satisfies self.policy_model usage in RayPPOTrainer / FullyAsync
# ---------------------------------------------------------------------------

class _ArcticPolicyStub:
    """No-op stub for ``self.policy_model``."""

    async def async_run_method(self, *args, **kwargs):
        pass

    def async_run_ray_method(self, *args, **kwargs):
        return []


class _ArcticInferenceEngineStub:
    """Stub for ``self.inference_engine_client``.

    Routes sleep/wake to the Arctic RL server for colocated mode.
    pause/resume are no-ops (server manages its own engine).
    """

    def __init__(self, client=None):
        self._client = client

    async def sleep(self, **kwargs):
        if self._client and self._client.config.colocate:
            await self._client.sleep_inference()

    async def wake_up(self, **kwargs):
        if self._client and self._client.config.colocate:
            tags = kwargs.get("tags")
            await self._client.wake_inference(tags=tags)

    async def pause_generation(self, **kwargs):
        pass

    async def resume_generation(self, **kwargs):
        pass


# ---------------------------------------------------------------------------
# Fully-async variant
# ---------------------------------------------------------------------------

def _make_arctic_fully_async_trainer_class():
    """Build the fully-async Arctic trainer class with a deferred import.

    The import of ``FullyAsyncRayPPOTrainer`` is deferred so that
    ``arctic_trainer.py`` can be imported without pulling in the
    fully-async module (and its extra dependencies) when only the
    sync trainer is needed.
    """
    from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer

    class _ArcticFullyAsyncPPOTrainer(FullyAsyncRayPPOTrainer):
        """Fully-async PPO Trainer backed by Arctic RL server (DeepSpeed).

        Drop-in replacement for ``FullyAsyncRayPPOTrainer``.  Applies the
        same Arctic RL overrides as ``ArcticPPOTrainer`` (no-op fwd_logprobs,
        server-side GRPO loss, DeepSpeed gradient accumulation).
        """

        def __init__(self, *args, arctic_client, **kwargs):
            self._arctic_client = arctic_client
            self._stashed_rewards = None
            super().__init__(*args, **kwargs)

        def build_models(self, PolicyWorker=None, CriticWorker=None, RefWorker=None):
            self.dispatch = _ArcticDispatch(self.cfg, self._arctic_client)
            self.policy_model = _ArcticPolicyStub()
            self.ref_model = None
            self.critic_model = None
            logger.info("ArcticFullyAsyncPPOTrainer: build_models → training routed to Arctic RL server")

        def fwd_logprobs_values_reward(self, training_input: TrainingInputBatch):
            """No-op — server computes old log-probs inline during fwd_bwd."""
            return training_input

        def compute_advantages_and_returns(self, training_input: TrainingInputBatch):
            self._stashed_rewards = training_input["rewards"].clone()
            training_input["values"] = None
            return super().compute_advantages_and_returns(training_input)

        def train_critic_and_policy(self, data: TrainingInputBatch):
            if self._stashed_rewards is not None:
                data["rewards"] = self._stashed_rewards
                self._stashed_rewards = None

            data.metadata["global_step"] = self.global_step
            n_samples = self.cfg.generator.n_samples_per_prompt

            all_metrics: Dict[str, List[float]] = defaultdict(list)

            for _epoch in range(self.cfg.trainer.update_epochs_per_batch):
                status = self.dispatch.forward_backward(
                    "policy", data,
                    loss_fn="grpo",
                    loss_fn_config={"n_samples": n_samples},
                )
                for k, v in status.items():
                    all_metrics[k].append(v)

                grad_norm = self.dispatch.optim_step("policy")
                if grad_norm is not None:
                    all_metrics["grad_norm"].append(grad_norm)

            all_metrics.pop("loss_fn_outputs", None)
            all_metrics.pop("post_process_outputs", None)
            reduced = reduce_metrics(dict(all_metrics))

            for k, v in reduced.items():
                self.all_metrics[f"policy/{k}"] = v

            return reduced

        async def async_sync_policy_weights_to_inference_engines(self):
            """Sync weights via Arctic RL server instead of policy_model."""
            await self._arctic_client.sync_weights()

    return _ArcticFullyAsyncPPOTrainer


def ArcticFullyAsyncPPOTrainer(*args, **kwargs):
    """Factory for the fully-async Arctic RL trainer.

    Defers the import of ``FullyAsyncRayPPOTrainer`` until first use.
    """
    cls = _make_arctic_fully_async_trainer_class()
    return cls(*args, **kwargs)
