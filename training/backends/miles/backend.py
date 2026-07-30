"""Miles backend — wraps TinkerTrainGroup/RolloutManager/SlimeArgumentBuilder
behind the TrainingBackend interface.

Targets the miles `tinker-seam` branch (upstream-based): async
TinkerTrainGroup fanout, decoupled
forward_backward_only / apply_optimizer_step, pure-sum loss via rollout keys
set in the converter."""
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import ray

from ..base import BackendError, BackendHandle, TrainingBackend

logger = logging.getLogger(__name__)


@dataclass
class MilesHandle(BackendHandle):
    """Miles-specific runtime state."""

    train_group: Any = None           # RayTrainGroup
    rollout_manager: Any = None       # RolloutManager (None for SFT)
    placement_group: Any = None       # Ray PlacementGroup
    args: Any = None                  # Megatron Namespace
    hf_path: str = ""
    router_ip: Optional[str] = None
    router_port: Optional[int] = None
    rlve_config: Optional[Dict[str, Any]] = None
    wandb_config: Optional[Dict[str, Any]] = None
    created_at: str = ""
    training_run_id: str = ""
    # Multi-LoRA pool mode (TINKERCLOUD_MILES_MULTILORA_SLOTS): this model is
    # an adapter slot on shared rails rather than a dedicated full model.
    controller: Any = None            # MultiLoRAController (named Ray actor)
    adapter_name: Optional[str] = None
    adapter_slot: Optional[int] = None
    # Serializes GPU-bound ops per model. The task manager runs request
    # handlers as concurrent asyncio tasks; without this, pipelined
    # fb/optim_step broadcasts interleave inconsistently across the DP actors
    # (mispaired collectives -> scrambled outputs; optim_step consuming a
    # later fb's grads). asyncio.Lock wakes waiters FIFO, so execution
    # follows submission order.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


class MilesBackend(TrainingBackend):
    """Thin adapter over existing Miles integration code (model_service.py / training_service.py)."""

    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        self.overrides = overrides or {}
        # Lazy-import converter to avoid import errors when Miles is not installed
        self._converter = None
        self._builder = None

    @property
    def converter(self):
        if self._converter is None:
            from .converter import MilesDataConverter
            self._converter = MilesDataConverter()
        return self._converter

    @property
    def builder(self):
        if self._builder is None:
            from .builder import MilesArgumentBuilder
            self._builder = MilesArgumentBuilder()
        return self._builder

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
    ) -> MilesHandle:
        if objective != "language_modeling":
            raise BackendError(
                f"Miles is a language-modeling backend; objective {objective!r} "
                f"requires a classification backend (automodel / megatron_bridge)",
                backend="miles", operation="create_model",
            )
        try:
            logger.info("[%s] Creating Miles model %s", request_id, model_id)

            # Build Slime arguments (blocking — run in thread pool)
            args, hf_path = await asyncio.to_thread(
                self.builder.build_args,
                base_model=base_model,
                lora_config=lora_config,
                debug_train_only=debug_train_only,
                checkpoint_path=checkpoint_path,
                parallelism_config=parallelism,
                max_batch_size=max_batch_size,
                max_seq_len=max_seq_len,
                rlve_config=rlve_config,
                wandb_config=wandb_config,
            )
            logger.info("[%s] Miles args built, hf_path=%s", request_id, hf_path)

            # Reuse upstream's own wiring (miles tinker-seam branch): placement
            # groups + RolloutManager from the factories train.py uses, and the
            # TinkerTrainGroup fanout for the decoupled train-step seam.
            from miles.ray.placement_group import create_placement_groups, create_rollout_manager
            from miles.ray.tinker_group import TinkerTrainGroup

            # Sync ray calls (pg.ready waits, actor allocation) — keep them off
            # the event loop or /retrieve_future polls stall and clients time out.
            pgs = await asyncio.to_thread(create_placement_groups, args)

            multi_lora = bool(getattr(args, "multi_lora", False))
            if multi_lora and debug_train_only:
                raise BackendError(
                    "Multi-LoRA pool mode requires rollout engines (adapter "
                    "weight push targets SGLang); debug_train_only unsupported",
                    backend="miles", operation="create_model",
                )

            rollout_manager = None
            router_ip = None
            router_port = None
            if not debug_train_only:
                rollout_manager, _ = await asyncio.to_thread(
                    create_rollout_manager, args, pgs["rollout"]
                )

            controller = None
            adapter_name = None
            adapter_slot = None
            if multi_lora:
                # Mirror the upstream driver boot: router -> controller
                # (named actor; reconcile on the actors resolves it by name).
                from miles.ray.multi_lora.controller import create_multilora_controller
                from miles.utils.adapter_config import TinkerAdapterConfig
                import re

                router_ip, router_port = await rollout_manager.get_router_address.remote()
                args.sglang_router_ip, args.sglang_router_port = router_ip, router_port
                controller = create_multilora_controller(
                    args, f"http://{router_ip}:{router_port}"
                )
                await controller.start.remote()

                adapter_name = re.sub(r"[^A-Za-z0-9._-]", "-", model_id)
                registration = await controller.register_adapter.remote(
                    adapter_name,
                    TinkerAdapterConfig(rank=args.lora_rank, alpha=args.lora_alpha),
                )
                adapter_slot = registration["slot"]
                logger.info(
                    "[%s] Multi-LoRA pool: adapter %s -> slot %d",
                    request_id, adapter_name, adapter_slot,
                )

            train_group = await asyncio.to_thread(lambda: TinkerTrainGroup(
                args=args,
                num_nodes=args.actor_num_nodes,
                num_gpus_per_node=args.actor_num_gpus_per_node,
                pg=pgs["actor"],
                num_gpus_per_actor=0.4,
                role="actor",
                with_ref=False,
                rollout_manager=rollout_manager,
            ))

            try:
                await asyncio.wait_for(train_group.init(), timeout=1800.0)
            except asyncio.TimeoutError:
                raise BackendError(
                    "Actor initialization timeout after 1800s",
                    backend="miles",
                    operation="create_model",
                )

            if rollout_manager is not None:
                await train_group.set_rollout_manager()

                # Pool mode: load the registered adapter into its slot before
                # the initial push (LoRA B=0 => zero-delta; PENDING->ACTIVE).
                if multi_lora:
                    await train_group.reconcile_adapters()

                # Mirror upstream train.py startup: load weights into SGLang
                # before anything samples, honoring rollout offload state.
                if args.offload_rollout:
                    await rollout_manager.onload_weights.remote()
                await train_group.update_weights()
                if args.offload_rollout:
                    await rollout_manager.onload_kv.remote()

                router_ip = getattr(args, "sglang_router_ip", None)
                router_port = getattr(args, "sglang_router_port", None)
                if not router_ip:
                    logger.error("[%s] SGLang router address missing from args", request_id)

            handle = MilesHandle(
                model_id=model_id,
                backend_type="miles",
                train_group=train_group,
                rollout_manager=rollout_manager,
                placement_group=pgs,
                args=args,
                hf_path=hf_path,
                router_ip=router_ip,
                router_port=router_port,
                rlve_config=rlve_config,
                wandb_config=wandb_config,
                created_at=datetime.now().isoformat(),
                training_run_id=model_id,
                controller=controller,
                adapter_name=adapter_name,
                adapter_slot=adapter_slot,
            )

            logger.info("[%s] Miles model %s created successfully", request_id, model_id)
            return handle

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="create_model", original_error=e,
            ) from e

    async def forward(
        self,
        handle: BackendHandle,
        data: List[Dict],
        loss_fn: str,
    ) -> Dict[str, Any]:
        h: MilesHandle = handle  # type: ignore[assignment]
        await h.lock.acquire()
        try:
            from miles.utils.ray_utils import Box

            if h.rollout_manager is not None and h.args.offload_rollout:
                await h.rollout_manager.offload.remote()

            rollout_data = self.converter.forward_to_backend(data, h.args, adapter_slot=h.adapter_slot)
            # TinkerTrainGroup returns per-sample logprob tensors already
            # merged into the client's submission order.
            logprobs = await h.train_group.forward_logprobs(0, Box(ray.put(rollout_data)))

            loss_fn_outputs = [
                {"logprobs": {"data": lp.tolist(), "shape": [len(lp)], "dtype": "float32"}}
                for lp in logprobs
            ]
            return {
                "type": "forward",
                "loss_fn_output_type": loss_fn,
                "loss_fn_outputs": loss_fn_outputs,
                "metrics": {},
            }

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="forward", original_error=e,
            ) from e
        finally:
            h.lock.release()

    async def forward_backward(
        self,
        handle: BackendHandle,
        data: List[Dict],
        loss_fn: str,
    ) -> Dict[str, Any]:
        h: MilesHandle = handle  # type: ignore[assignment]
        await h.lock.acquire()
        try:
            from miles.utils.ray_utils import Box

            if h.rollout_manager is not None and h.args.offload_rollout:
                await h.rollout_manager.offload.remote()

            is_rl = not h.args.debug_train_only

            from ...core.validators import RequestValidator
            from ...config import get_config

            config = get_config()
            allow_partial = getattr(config, "allow_partial_batches", False)
            validator = RequestValidator(h.args, allow_partial_batches=allow_partial)
            validation_error = validator.validate_forward_backward_request(
                data, is_rl=is_rl,
            )
            if validation_error:
                raise ValueError(
                    f"Request validation failed:\n{validation_error}\n\n"
                    f"{validator.get_config_summary()}"
                )

            rollout_data = self.converter.forward_backward_to_backend(
                data, loss_fn, h.args, adapter_slot=h.adapter_slot,
            )

            results = await h.train_group.forward_backward_only(0, Box(ray.put(rollout_data)))

            # Only pipeline-last-stage actors return metrics; average across
            # the DP ranks that did. Per-sample logprobs are not emitted by the
            # seam's fb pass itself (they ride a separate forward).
            summed: Dict[str, float] = {}
            reporting = 0
            for r in results or []:
                loss_dict = (r or {}).get("loss") or {}
                if loss_dict:
                    reporting += 1
                    for k, v in loss_dict.items():
                        summed[k] = summed.get(k, 0.0) + float(v)
            averaged = {k: v / reporting for k, v in summed.items()} if reporting else {}
            # SDK metric keys carry their cross-chunk reduction as ":<type>"
            # (chunked_fwdbwd_helpers._metrics_reduction splits on ":").
            metrics = {f"{k}:mean": v for k, v in averaged.items()}

            # Per-datum response logprobs in client order (the SDK weights its
            # metric reduction by len(loss_fn_outputs), and the cookbook
            # computes NLL from these).
            from miles.ray.tinker_group import merge_dp_sample_outputs
            logprobs_list = merge_dp_sample_outputs(results or [], key="log_probs")
            loss_fn_outputs = [
                {"logprobs": {"data": lp.tolist(), "shape": [len(lp)], "dtype": "float32"}}
                for lp in logprobs_list
            ]

            return {
                "loss_fn_output_type": loss_fn,
                "loss": averaged.get("loss"),
                "metrics": metrics,
                "loss_fn_outputs": loss_fn_outputs,
                "deferred": False,
            }

        except BackendError:
            raise
        except ValueError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="forward_backward", original_error=e,
            ) from e
        finally:
            h.lock.release()

    async def apply_optimizer_step(
        self,
        handle: BackendHandle,
        learning_rate: Optional[float] = None,
        adam_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # adam_params accepted for contract uniformity (P4); Miles applies lr
        # only — betas/eps are Megatron args fixed at creation.
        h: MilesHandle = handle  # type: ignore[assignment]
        await h.lock.acquire()
        try:
            # TinkerTrainGroup: apply_optimizer_step(learning_rate) fans out to
            # the actors; _and_sync additionally pushes weights to SGLang.
            offload_train = h.args.offload_train if h.args else True
            offload_rollout = h.args.offload_rollout if h.args else True

            step_kwargs = {"adapter_slot": h.adapter_slot, "adapter_name": h.adapter_name}
            if h.rollout_manager is None:
                results = await h.train_group.apply_optimizer_step(learning_rate, **step_kwargs)
            elif not offload_train and not offload_rollout:
                # Pool mode rides this arm: the sync pushes exactly the stepped
                # adapter (per-adapter upsert via the pending set).
                results = await h.train_group.apply_optimizer_step_and_sync(learning_rate, **step_kwargs)
            else:
                results = await h.train_group.apply_optimizer_step(learning_rate, **step_kwargs)

                # Mirror upstream train.py's offload dance around weight sync.
                if offload_train:
                    await h.train_group.offload()
                if offload_rollout:
                    await h.rollout_manager.onload_weights.remote()
                await h.train_group.update_weights()
                if offload_rollout:
                    await h.rollout_manager.onload_kv.remote()

            return {
                "success": results[0]["success"],
                "grad_norm": results[0]["grad_norm"],
                "learning_rates": [],
                "model_id": h.model_id,
            }

        except BackendError:
            raise
        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="apply_optimizer_step", original_error=e,
            ) from e
        finally:
            h.lock.release()

    async def update_inference_weights(self, handle: BackendHandle) -> None:
        h: MilesHandle = handle  # type: ignore[assignment]
        await h.lock.acquire()
        try:
            await h.train_group.update_weights()
        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="update_inference_weights", original_error=e,
            ) from e
        finally:
            h.lock.release()

    async def save_checkpoint(
        self,
        handle: BackendHandle,
        checkpoint_path: str,
        step_id: Optional[int] = None,
    ) -> str:
        h: MilesHandle = handle  # type: ignore[assignment]
        await h.lock.acquire()
        try:
            offload_train = h.args.offload_train if h.args else False
            if offload_train:
                logger.info("Skipping save_model (offload_train=True)")
                return checkpoint_path

            if step_id is None:
                step_id = 0

            await h.train_group.save_model(step_id)
            return checkpoint_path

        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="save_checkpoint", original_error=e,
            ) from e
        finally:
            h.lock.release()

    async def load_checkpoint(
        self,
        handle: BackendHandle,
        checkpoint_path: str,
    ) -> None:
        h: MilesHandle = handle  # type: ignore[assignment]
        await h.lock.acquire()
        try:
            await h.train_group.load_checkpoint(checkpoint_path)

            # Sync loaded weights to inference engine
            if h.rollout_manager is not None:
                await h.train_group.update_weights()

            logger.info("Miles checkpoint loaded from %s", checkpoint_path)

        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="load_checkpoint", original_error=e,
            ) from e
        finally:
            h.lock.release()

    async def delete_model(self, handle: BackendHandle) -> None:
        h: MilesHandle = handle  # type: ignore[assignment]
        # Hold the op lock so teardown can't interleave with an in-flight
        # fb/step (delete-during-optim_step crash class).
        await h.lock.acquire()
        try:
            resources_freed = []
            # Pool mode: the pool dies with this model (M1: one tenant per
            # pool). Kill the named controller so the next create_model can
            # register a fresh one.
            if h.controller is not None:
                try:
                    await h.controller.stop.remote()
                except Exception:
                    logger.warning("Multi-LoRA controller stop failed; killing", exc_info=True)
                ray.kill(h.controller, no_restart=True)
                resources_freed.append("multi_lora_controller")

            for actor in h.train_group._actor_handles:
                ray.kill(actor, no_restart=True)
                resources_freed.append("actor")

            if h.rollout_manager is not None:
                ray.kill(h.rollout_manager, no_restart=True)
                resources_freed.append("rollout_manager")

            # placement_group holds the create_placement_groups() dict of
            # (pg, bundle_indices, gpu_ids) tuples; pgs may be shared between
            # roles (colocate), so dedupe before removal.
            if h.placement_group:
                seen = set()
                for pg_tuple in h.placement_group.values():
                    pg_obj = pg_tuple[0] if isinstance(pg_tuple, tuple) else pg_tuple
                    # debug_train_only leaves the rollout entry as None
                    if pg_obj is not None and id(pg_obj) not in seen:
                        seen.add(id(pg_obj))
                        ray.util.remove_placement_group(pg_obj)
                        resources_freed.append("placement_group")

            logger.info("Miles model %s deleted, freed %d resources", h.model_id, len(resources_freed))

        except Exception as e:
            raise BackendError(
                str(e), backend="miles", operation="delete_model", original_error=e,
            ) from e
        finally:
            h.lock.release()

    async def get_logprobs(
        self,
        handle: BackendHandle,
        data: List[Dict],
    ) -> List[Any]:
        # Miles computes logprobs internally during forward_backward.
        # Expose via forward-only path for explicit logprob requests.
        result = await self.forward(handle, data, loss_fn="cross_entropy")
        logprobs_list = []
        for output in result.get("loss_fn_outputs", []):
            lp = output.get("logprobs", {})
            logprobs_list.append(lp.get("data", []))
        return logprobs_list

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
        """Sample via per-request HTTP calls to the SGLang router.

        pinned_version is accepted for contract uniformity but NOT honored:
        Miles serves from the live SGLang engine (same BUG-015 aliasing class;
        needs version-pinned sampling to fix).
        """
        from ...utils.sglang_client import SGLangClient

        h: MilesHandle = handle  # type: ignore[assignment]
        if not h.router_ip or not h.router_port:
            raise BackendError(
                "SGLang router not available",
                backend="miles", operation="sample",
            )
        client = SGLangClient(base_url=f"http://{h.router_ip}:{h.router_port}")

        # Pool mode: route to this model's adapter by engine-side slot name.
        # (pinned v0 -> base routing not wired yet; see 003 R6.)
        lora_path = None
        if h.adapter_slot is not None:
            from miles.utils.multi_lora import slot_lora_name

            lora_path = slot_lora_name(h.adapter_slot)

        sequences = []
        prompt_logprobs_result = None
        for _ in range(num_samples):
            result = await client.generate(
                input_ids=prompt_tokens,
                sampling_params=sampling_params or {},
                prompt_logprobs=prompt_logprobs,
                lora_path=lora_path,
            )
            sequences.append({
                "tokens": result["tokens"],
                "logprobs": result["logprobs"],
                "text": result.get("text"),
                "stop_reason": result.get("stop_reason", "length"),
            })
            if prompt_logprobs and prompt_logprobs_result is None:
                prompt_logprobs_result = result.get("prompt_logprobs")

        return {
            "sequences": sequences,
            "prompt_logprobs": prompt_logprobs_result,
        }

    async def prepare_for_generation(self, handle: BackendHandle) -> None:
        """SGLang router is always live for Miles — just validate it exists."""
        h: MilesHandle = handle  # type: ignore[assignment]
        if not h.router_ip or not h.router_port:
            raise BackendError(
                "SGLang router not available",
                backend="miles", operation="prepare_for_generation",
            )
