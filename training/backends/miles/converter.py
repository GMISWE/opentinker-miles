"""
Miles data converter — wraps TinkerDataConverter behind the
DataConverter ABC.

The actual logic remains in training/core/data_converter.py. This module
re-exports it through the backend interface.
"""
from typing import Any, Dict, List

from ..base import DataConverter
from ...core.data_converter import TinkerDataConverter


class MilesDataConverter(DataConverter):
    """Adapter: TinkerDataConverter → DataConverter ABC."""

    def __init__(self):
        self._inner = TinkerDataConverter()

    def forward_to_backend(
        self,
        data: List[Dict],
        args: Any,
        adapter_slot: Any = None,
    ) -> Any:
        """Convert Tinker data to Miles rollout_data for forward pass."""
        rollout_data = self._inner.forward_to_rollout(data)
        self._add_tinker_seam_keys(rollout_data, len(data), adapter_slot=adapter_slot)
        return rollout_data

    def forward_backward_to_backend(
        self,
        data: List[Dict],
        loss_fn: str,
        args: Any,
        adapter_slot: Any = None,
    ) -> Any:
        """Convert Tinker data to Miles rollout_data for training."""
        is_rl = not getattr(args, "debug_train_only", False)
        rollout_data = self._inner.forward_backward_to_rollout(data, is_rl=is_rl)
        self._add_tinker_seam_keys(rollout_data, len(data), adapter_slot=adapter_slot)
        # Per-request loss selection (upstream dispatches on args.loss_type at
        # startup; the seam overrides per batch). The inner converter already
        # sets sft_loss when it detects SFT-shaped data — don't override that.
        rollout_data.setdefault(
            "_loss_type_override",
            "sft_loss" if loss_fn == "cross_entropy" else "policy_loss",
        )
        return rollout_data

    @staticmethod
    def _add_tinker_seam_keys(rollout_data: Any, num_samples: int, adapter_slot: Any = None) -> None:
        """Keys the tinker-seam miles branch consumes.

        - dynamic_global_batch_size: actual request size, so upstream
          get_data_iterator schedules correctly for variable batches.
        - _loss_norm_total=1: pure-sum gradients — invariant to how a logical
          batch is split across forward_backward calls (G1 contract).
        - adapter_slots (pool mode): route every sample of this request to the
          model's slot; the shard converter injects n_adapters from args.
        """
        rollout_data["dynamic_global_batch_size"] = num_samples
        rollout_data["_loss_norm_total"] = 1
        if adapter_slot is not None:
            rollout_data["adapter_slots"] = [adapter_slot] * num_samples

    @staticmethod
    def merge_forward_backward_batches(batches: List[Any]) -> Any:
        """Concatenate per-request rollout_data dicts into one mixed-slot
        batch (M3 co-batching). Per-sample list keys concatenate in request
        order; scalars must agree across requests (same-loss_fn merge rule);
        dynamic_global_batch_size sums. The miles DP split slot-sorts each
        partition, so merged order need not be slot-sorted here."""
        if len(batches) == 1:
            return batches[0]
        keys = set(batches[0].keys())
        assert all(set(b.keys()) == keys for b in batches), \
            f"co-batch key mismatch: {[sorted(b.keys()) for b in batches]}"
        sizes = [b["dynamic_global_batch_size"] for b in batches]
        merged: Dict[str, Any] = {}
        for key in batches[0]:
            v0 = batches[0][key]
            if key == "dynamic_global_batch_size":
                merged[key] = sum(sizes)
            elif isinstance(v0, list) and all(
                isinstance(b[key], list) and len(b[key]) == n
                for b, n in zip(batches, sizes)
            ):
                merged[key] = [x for b in batches for x in b[key]]
            else:
                assert all(repr(b[key]) == repr(v0) for b in batches), \
                    f"co-batch scalar mismatch on {key!r}"
                merged[key] = v0
        return merged

    def backend_to_forward_result(
        self,
        result: Any,
        data: List[Dict],
    ) -> Dict[str, Any]:
        """Convert Miles forward result to Tinker format."""
        return self._inner.rollout_to_forward_result(
            result,
            loss_fn="cross_entropy",
            original_data=data,
        )

    def backend_to_forward_backward_result(
        self,
        result: Any,
        data: List[Dict],
    ) -> Dict[str, Any]:
        """Convert Miles forward_backward result to Tinker format."""
        return self._inner.rollout_to_forward_backward_result(
            result,
            loss_fn="cross_entropy",
            original_data=data,
        )
