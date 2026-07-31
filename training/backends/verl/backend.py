"""
veRL backend — binds veRL's upstream Tinker split-training primitives
(verl.workers.engine_workers_tinker) behind the TrainingBackend ABC.

Execution model: EAGER (Miles-shaped, NOT R9-buffered). veRL upstream ships
TinkerTrainingWorker with decoupled forward_backward (grad-accumulating,
returns per-token logprobs) + optimizer_step (accepts lr/betas/eps/
weight_decay) + optimizer_zero_grad. forward_backward here does real GPU
work and returns real logprobs; apply_optimizer_step applies AdamParams and
steps once.

M1 scope: training path only (create_model/fb/step/forward/get_logprobs/
checkpoints/delete). sample() raises UnsupportedFeatureError — rollout
engine + weight sync land in M2. Narrative: specs/006-verl-backend/design.md.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

from ..base import BackendError, BackendHandle, TrainingBackend, UnsupportedFeatureError

logger = logging.getLogger(__name__)


@dataclass
class VerlHandle(BackendHandle):
    """veRL-specific runtime state."""

    worker_group: Any = None          # RayWorkerGroup over TinkerTrainingWorker
    resource_pool: Any = None         # RayResourcePool
    config: Dict = field(default_factory=dict)
    hf_path: str = ""
    dp_size: int = 1
    loss_fn_name: str = ""            # currently installed loss fn on workers
    weight_version: int = 0
    created_at: str = ""
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)  # FIFO per handle


class VerlBackend(TrainingBackend):
    """veRL backend using upstream TinkerTrainingWorker split primitives."""

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        self.overrides = overrides or {}
        self._converter = None
        self._builder = None

    @property
    def converter(self):
        if self._converter is None:
            from .converter import VerlDataConverter
            self._converter = VerlDataConverter()
        return self._converter

    @property
    def builder(self):
        if self._builder is None:
            from .builder import VerlArgumentBuilder
            self._builder = VerlArgumentBuilder(overrides=self.overrides)
        return self._builder

    # ---------------------------------------------------------------- create

    async def create_model(
        self,
        model_id: str,
        request_id: str,
        base_model: str,
        num_gpus: int,
        lora_config: Optional[Dict[str, Any]] = None,
        parallelism: Optional[Dict[str, Any]] = None,
        rl_config: Optional[Dict[str, Any]] = None,
        rollout_config: Optional[Dict[str, Any]] = None,
        debug_train_only: bool = False,
        checkpoint_path: Optional[str] = None,
        max_batch_size: int = 4096,
        max_seq_len: int = 2048,
        rlve_config: Optional[Dict[str, Any]] = None,
        wandb_config: Optional[Dict[str, Any]] = None,
        objective: str = "language_modeling",
        num_labels: Optional[int] = None,
        head_config: Optional[Dict[str, Any]] = None,
    ) -> VerlHandle:
        if objective != "language_modeling":
            raise BackendError(
                f"verl is a language-modeling backend; objective {objective!r} unsupported",
                backend="verl", operation="create_model",
            )
        try:
            cfg = self.builder.build_args(
                base_model=base_model, num_gpus=num_gpus, lora_config=lora_config,
                parallelism=parallelism, rl_config=rl_config,
                rollout_config=rollout_config, max_seq_len=max_seq_len,
            )
            worker_group, resource_pool, dp_size = await asyncio.to_thread(
                _boot_worker_group, cfg, model_id,
            )
            handle = VerlHandle(
                model_id=model_id,
                backend_type="verl",
                worker_group=worker_group,
                resource_pool=resource_pool,
                config=cfg,
                hf_path=cfg["hf_path"],
                dp_size=dp_size,
                created_at=datetime.now().isoformat(),
            )
            if checkpoint_path:
                await self.load_checkpoint(handle, checkpoint_path)
            logger.info("[%s] verl model %s created (dp=%d)", request_id, model_id, dp_size)
            return handle
        except BackendError:
            raise
        except Exception as e:
            raise BackendError(str(e), backend="verl", operation="create_model", original_error=e) from e

    # ------------------------------------------------------------- train ops

    async def forward_backward(
        self,
        handle: BackendHandle,
        data: List[Dict],
        loss_fn: str,
    ) -> Dict[str, Any]:
        h: VerlHandle = handle  # type: ignore[assignment]
        async with h._lock:
            try:
                await asyncio.to_thread(self._ensure_loss_fn, h, loss_fn)
                td = await asyncio.to_thread(self._to_tensordict, h, data, loss_fn)
                out = await asyncio.to_thread(_wg_call, h.worker_group.forward_backward, td)
                logprobs = await asyncio.to_thread(self._model_output_logprobs, out, td)
                loss_fn_outputs = self.converter.extract_logprobs(logprobs, data)
                metrics = _scalar_metrics(out)
                return {
                    "loss_fn_output_type": loss_fn,
                    "loss_fn_outputs": loss_fn_outputs,
                    "metrics": metrics,
                    "deferred": False,
                }
            except BackendError:
                raise
            except Exception as e:
                raise BackendError(str(e), backend="verl", operation="forward_backward", original_error=e) from e

    async def apply_optimizer_step(
        self,
        handle: BackendHandle,
        learning_rate: Optional[float] = None,
        adam_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        h: VerlHandle = handle  # type: ignore[assignment]
        async with h._lock:
            try:
                params: Dict[str, Any] = {}
                if learning_rate is not None:
                    params["lr"] = float(learning_rate)
                ap = adam_params or {}
                if ap.get("beta1") is not None or ap.get("beta2") is not None:
                    params["betas"] = (float(ap.get("beta1", 0.9)), float(ap.get("beta2", 0.95)))
                if ap.get("eps") is not None:
                    params["eps"] = float(ap["eps"])
                if ap.get("weight_decay") is not None:
                    params["weight_decay"] = float(ap["weight_decay"])
                unsupported = {k: v for k, v in ap.items()
                               if k not in ("beta1", "beta2", "eps", "weight_decay", "grad_clip_norm") and v is not None}
                if unsupported:
                    logger.warning("verl optim_step ignoring AdamParams %s", sorted(unsupported))
                if ap.get("grad_clip_norm") is not None:
                    logger.warning("verl optim_step: grad_clip_norm set at engine build, not per step")

                metrics = await asyncio.to_thread(
                    _wg_scalar_call, h.worker_group.optimizer_step, params or None,
                )
                h.weight_version += 1
                grad_norm = metrics.get("grad_norm")
                return {"success": True, "grad_norm": grad_norm, "metrics": metrics}
            except BackendError:
                raise
            except Exception as e:
                raise BackendError(str(e), backend="verl", operation="apply_optimizer_step", original_error=e) from e

    async def forward(
        self,
        handle: BackendHandle,
        data: List[Dict],
        loss_fn: str,
    ) -> Dict[str, Any]:
        logprobs = await self.get_logprobs(handle, data)
        return {"loss_fn_output_type": loss_fn, "loss_fn_outputs": logprobs, "metrics": {}}

    async def get_logprobs(
        self,
        handle: BackendHandle,
        data: List[Dict],
    ) -> List[Any]:
        h: VerlHandle = handle  # type: ignore[assignment]
        async with h._lock:
            try:
                td = await asyncio.to_thread(self._to_tensordict, h, data, "cross_entropy")
                from verl.utils import tensordict_utils as tu
                tu.assign_non_tensor(td, compute_loss=False)
                out = await asyncio.to_thread(_wg_call, h.worker_group.infer_batch, td)
                logprobs = await asyncio.to_thread(self._model_output_logprobs, out, td)
                return self.converter.extract_logprobs(logprobs, data)
            except BackendError:
                raise
            except Exception as e:
                raise BackendError(str(e), backend="verl", operation="get_logprobs", original_error=e) from e

    # ------------------------------------------------------------ lifecycle

    async def update_inference_weights(self, handle: BackendHandle) -> None:
        # M1: no rollout engine attached; no-op until M2 wires
        # ActorRolloutRefWorker.update_weights.
        return None

    async def save_checkpoint(
        self,
        handle: BackendHandle,
        checkpoint_path: str,
        step_id: Optional[int] = None,
    ) -> str:
        h: VerlHandle = handle  # type: ignore[assignment]
        async with h._lock:
            try:
                await asyncio.to_thread(
                    h.worker_group.save_checkpoint, checkpoint_path, None, step_id or h.weight_version,
                )
                return checkpoint_path
            except Exception as e:
                raise BackendError(str(e), backend="verl", operation="save_checkpoint", original_error=e) from e

    async def load_checkpoint(self, handle: BackendHandle, checkpoint_path: str) -> None:
        h: VerlHandle = handle  # type: ignore[assignment]
        async with h._lock:
            try:
                await asyncio.to_thread(h.worker_group.load_checkpoint, checkpoint_path)
            except Exception as e:
                raise BackendError(str(e), backend="verl", operation="load_checkpoint", original_error=e) from e

    async def delete_model(self, handle: BackendHandle) -> None:
        h: VerlHandle = handle  # type: ignore[assignment]
        try:
            import ray
            for w in getattr(h.worker_group, "workers", []) or []:
                try:
                    ray.kill(w)
                except Exception:
                    pass
            if h.resource_pool is not None:
                pgs = getattr(h.resource_pool, "pgs", None) or []
                for pg in pgs:
                    try:
                        ray.util.remove_placement_group(pg)
                    except Exception:
                        pass
            h.worker_group = None
            h.resource_pool = None
        except Exception as e:
            raise BackendError(str(e), backend="verl", operation="delete_model", original_error=e) from e

    async def sample(
        self,
        handle: BackendHandle,
        request_id: str,
        prompt_tokens: List[int],
        num_samples: int,
        sampling_params: Optional[Dict[str, Any]] = None,
        prompt_logprobs: bool = False,
        pinned_version: Optional[int] = None,
    ) -> Dict[str, Any]:
        raise UnsupportedFeatureError(
            "sample", backend="verl",
            suggestion="M1 is training-path only; rollout engine lands in M2 (specs/006)",
        )

    async def prepare_for_generation(self, handle: BackendHandle) -> None:
        raise UnsupportedFeatureError(
            "prepare_for_generation", backend="verl",
            suggestion="M1 is training-path only; rollout engine lands in M2 (specs/006)",
        )

    # -------------------------------------------------------------- helpers

    def _ensure_loss_fn(self, h: VerlHandle, loss_fn: str) -> None:
        if h.loss_fn_name == loss_fn:
            return
        from .losses import get_loss_fn
        from functools import partial
        h.worker_group.set_loss_fn(partial(_loss_dispatch, loss_name=loss_fn))
        _ = get_loss_fn(loss_fn)  # validate name eagerly
        h.loss_fn_name = loss_fn

    def _to_tensordict(self, h: VerlHandle, data: List[Dict], loss_fn: str):
        from tensordict import TensorDict
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import left_right_2_no_padding

        padded = self.converter.forward_backward_to_backend(data, loss_fn, h.config)

        # Divisibility: engine needs per-DP-rank count % micro_batch_size == 0.
        # Pad with zero-weight clones (pure-sum => zero grad, E0-safe)
        # instead of rejecting or trimming.
        micro_bs = int(h.config.get("engine", {}).get("micro_batch_size_per_gpu", 1) or 1)
        multiple = h.dp_size * micro_bs
        b = padded["input_ids"].shape[0]
        remainder = b % multiple
        if remainder:
            pad_n = multiple - remainder
            for key, t in padded.items():
                clone = t[:1].repeat(pad_n, *([1] * (t.dim() - 1))).clone()
                if key in ("weights", "advantages"):
                    clone.zero_()
                padded[key] = torch.cat([t, clone], dim=0)

        global_token_num = padded["attention_mask"].sum(dim=-1).tolist()
        td = TensorDict(padded, batch_size=[padded["input_ids"].shape[0]])
        td = left_right_2_no_padding(td)
        # temperature: engine divides logits by it (logprob provenance);
        # training-side logprobs are untempered in the Tinker contract
        tu.assign_non_tensor(td, global_token_num=global_token_num, temperature=1.0)
        return td

    @staticmethod
    def _model_output_logprobs(out, td) -> torch.Tensor:
        """Gathered worker output -> padded [B, Rmax] logprobs."""
        from verl.utils import tensordict_utils as tu
        from verl.workers.utils.padding import no_padding_2_padding

        lp = tu.get(out, "log_probs")
        if lp is None:
            raise BackendError("worker returned no log_probs", backend="verl", operation="forward_backward")
        return no_padding_2_padding(lp, td)


def _loss_dispatch(config=None, model_output=None, data=None, dp_group=None, loss_name: str = "cross_entropy"):
    """Module-level picklable shim: resolves the loss by name inside the worker."""
    from .losses import get_loss_fn
    return get_loss_fn(loss_name)(config, model_output, data, dp_group)


def _wg_call(method, td):
    """Blocking worker-group data call: resolve DataProtoFuture if non-blocking."""
    out = method(td)
    if hasattr(out, "get"):
        out = out.get()
    return out


def _wg_scalar_call(method, *args):
    out = method(*args)
    if isinstance(out, list):
        # ONE_TO_ALL returns per-worker dicts; grad_norm identical post-clip
        merged: Dict[str, Any] = {}
        for d in out:
            if isinstance(d, dict):
                merged.update(d)
        return merged
    if hasattr(out, "get"):
        out = out.get()
    return out or {}


def _scalar_metrics(out) -> Dict[str, Any]:
    from verl.utils import tensordict_utils as tu
    try:
        metrics = tu.get(out, "metrics") or {}
    except Exception:
        metrics = {}
    out_metrics = {}
    for k, v in dict(metrics).items():
        if k.startswith("perf/"):
            continue
        if hasattr(v, "data"):
            v = v.data
        if isinstance(v, (int, float)):
            # SDK metric keys are "name:reduction" (chunked_fwdbwd_helpers)
            key = k.replace(":", "_").replace("/", "_")
            out_metrics[f"{key}:sum"] = float(v)
    return out_metrics


def _boot_worker_group(cfg: Dict[str, Any], model_id: str):
    """Boot a TinkerTrainingWorker RayWorkerGroup (training path only)."""
    import ray
    from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
    from verl.workers.config import FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
    from verl.workers.engine_workers import TrainingWorkerConfig
    from verl.workers.engine_workers_tinker import TinkerTrainingWorker

    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    model = cfg["model"]
    engine = cfg["engine"]
    model_config = HFModelConfig(
        path=model["path"],
        use_remove_padding=model["use_remove_padding"],
        lora_rank=model["lora_rank"],
        lora_alpha=model["lora_alpha"],
        target_modules=model["target_modules"],
    )
    engine_config = FSDPEngineConfig(
        forward_only=False,
        strategy=engine["strategy"],
        fsdp_size=engine["fsdp_size"],
        ulysses_sequence_parallel_size=engine["ulysses_sequence_parallel_size"],
        use_remove_padding=engine["use_remove_padding"],
        use_dynamic_bsz=engine["use_dynamic_bsz"],
        micro_batch_size_per_gpu=engine["micro_batch_size_per_gpu"],
        max_token_len_per_gpu=engine["max_token_len_per_gpu"],
        infer_micro_batch_size_per_gpu=engine["infer_micro_batch_size_per_gpu"],
        infer_max_token_len_per_gpu=engine["infer_max_token_len_per_gpu"],
        param_offload=engine["param_offload"],
        optimizer_offload=engine["optimizer_offload"],
        grad_offload=engine["grad_offload"],
    )
    optimizer_config = FSDPOptimizerConfig(
        lr=cfg["optim"]["lr"], lr_scheduler_type=cfg["optim"]["lr_scheduler_type"],
    )
    worker_config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=None,
    )

    ray_cls = RayClassWithInitArgs(cls=ray.remote(TinkerTrainingWorker), config=worker_config)
    resource_pool = RayResourcePool(process_on_nodes=[cfg["num_gpus"]], name_prefix=f"verl_{model_id}")
    wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls)
    wg.reset()

    sp = engine["ulysses_sequence_parallel_size"]
    dp_size = max(cfg["num_gpus"] // max(sp, 1), 1)
    return wg, resource_pool, dp_size
