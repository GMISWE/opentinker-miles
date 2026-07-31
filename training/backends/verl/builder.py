"""
veRL argument builder: HF model + tinker knobs -> config dict.

Returns a plain dict; VerlBackend materializes veRL dataclasses lazily on the
server (where verl is importable). FSDP2 strategy first; Megatron later.
"""
import os
from typing import Any, Dict, Optional

from ..base import ArgumentBuilder


class VerlArgumentBuilder(ArgumentBuilder):
    def __init__(self, overrides: Optional[Dict[str, Any]] = None):
        self.overrides = overrides or {}

    def build_args(
        self,
        base_model: str,
        num_gpus: int = 4,
        lora_config: Optional[Dict[str, Any]] = None,
        parallelism: Optional[Dict[str, Any]] = None,
        rl_config: Optional[Dict[str, Any]] = None,
        rollout_config: Optional[Dict[str, Any]] = None,
        checkpoint_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        parallelism = parallelism or {}
        lora_config = lora_config or {}
        max_seq_len = int(kwargs.get("max_seq_len", 2048))

        hf_path = self._resolve_hf_path(base_model)

        # request payloads carry explicit None values — coalesce, don't .get()-default
        lora_rank = int(lora_config.get("rank") or lora_config.get("lora_rank") or 0)
        lora_alpha = float(lora_config.get("alpha") or lora_config.get("lora_alpha") or 32)
        target_modules = lora_config.get("target_modules") or "all-linear"

        cfg: Dict[str, Any] = {
            "hf_path": hf_path,
            "num_gpus": num_gpus,
            "model": {
                "path": hf_path,
                "use_remove_padding": True,
                # FSDP LoRA namespace ONLY (model.lora_rank); the Megatron
                # namespace (model.lora.rank) must stay 0 — duplicated-knob
                # trap, specs/006 q2-plan.md
                "lora_rank": lora_rank,
                "lora_alpha": lora_alpha,
                "target_modules": target_modules,
            },
            "engine": {
                "strategy": self.overrides.get("strategy", "fsdp2"),
                "fsdp_size": num_gpus,
                "ulysses_sequence_parallel_size": int(parallelism.get("sp_size", 1)),
                "forward_only": False,
                "use_remove_padding": True,
                # fixed micro-batch (dynamic_bsz = batch-composition-dependent
                # micro-partitioning; pure-sum makes it E0-safe but keep the
                # default deterministic)
                "use_dynamic_bsz": bool(self.overrides.get("use_dynamic_bsz", False)),
                "micro_batch_size_per_gpu": int(self.overrides.get("micro_batch_size_per_gpu", 2)),
                "max_token_len_per_gpu": max_seq_len * 4,
                "infer_micro_batch_size_per_gpu": int(self.overrides.get("infer_micro_batch_size_per_gpu", 4)),
                "infer_max_token_len_per_gpu": max_seq_len * 8,
                "param_offload": bool(self.overrides.get("param_offload", False)),
                "optimizer_offload": bool(self.overrides.get("optimizer_offload", False)),
                "grad_offload": bool(self.overrides.get("grad_offload", False)),
            },
            "optim": {
                # placeholder; client owns LR per optim_step (OptimStepParams)
                "lr": 1e-6,
                "lr_scheduler_type": "constant",
            },
            "max_seq_len": max_seq_len,
        }
        return cfg

    @staticmethod
    def _resolve_hf_path(base_model: str) -> str:
        for root in filter(None, [os.environ.get("TINKERCLOUD_MODEL_ROOT"), "/data/models", "/root/models"]):
            candidate = os.path.join(root, base_model)
            if os.path.isdir(candidate):
                return candidate
        return base_model  # HF hub id / absolute path
