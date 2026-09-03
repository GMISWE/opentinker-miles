"""
Miles model setup: resolve a HF model to its torch_dist checkpoint, pick TP/PP/CP/DP
for the GPUs at hand, size SGLang's memory fraction, and map tinker:// checkpoint
URIs to Miles' save layout. Every rule here is Megatron/SGLang-specific.
"""
import logging
import os
from typing import Any, Dict, Optional

from ...utils.model_config import estimate_model_params

logger = logging.getLogger(__name__)


def auto_detect_all_parallelism(
    model_config: Dict[str, Any],
    num_gpus: int,
    max_seq_len: int = 2048,
    rlve_enabled: bool = False,
    model_name: str = "",
    default_tp: Optional[int] = None,
    default_cp: Optional[int] = None,
    rlve_tp: int = 2,
    rlve_cp: int = 2,
) -> Dict[str, int]:
    """
    Auto-detect all parallelism dimensions (TP, PP, CP, DP).

    Two modes:
    1. RLVE mode: Fixed parallelism optimized for long sequences (TP=2, CP=2)
    2. Standard mode: Full auto-detection based on model size and sequence length

    Args:
        model_config: HuggingFace model config dict
        num_gpus: Available GPUs
        max_seq_len: Maximum sequence length (affects CP decision)
        rlve_enabled: If True, use RLVE-optimized fixed config
        model_name: Optional model name for parameter estimation

    Returns:
        Dict with tp, pp, cp, dp values
    """
    if rlve_enabled:
        # RLVE MODE: Fixed parallelism for long sequences
        # TP=2, CP=2 → DP=1 on 4 GPUs (all GPUs work together on same batch)
        tp = rlve_tp
        pp = 1  # No pipeline parallel for RLVE (simpler, less latency)
        cp = rlve_cp
        dp = num_gpus // (tp * pp * cp)
        logger.info(f"RLVE parallelism: TP={tp}, PP={pp}, CP={cp}, DP={max(1, dp)}")
        return {'tp': tp, 'pp': pp, 'cp': cp, 'dp': max(1, dp)}

    # STANDARD MODE: Full auto-detection
    num_params = estimate_model_params(model_config, model_name)
    logger.info(f"Auto-detecting parallelism for {num_params:.2f}B params, {num_gpus} GPUs, max_seq_len={max_seq_len}")

    if default_tp:
        tp = default_tp
        logger.info(f"Using configured TP override: TP={tp}")
    # TP: based on model size
    elif num_params < 2.0:  # <2B params
        tp = 1
    elif num_params < 10.0:  # 2-10B params
        tp = min(2, num_gpus)
    elif num_params < 30.0:  # 10-30B params
        tp = min(4, num_gpus)
    else:  # >30B params
        tp = min(8, num_gpus)

    # PP: for very large models that don't fit with TP alone
    if num_params >= 30.0:
        pp = 2
    else:
        pp = 1

    # CP: based on sequence length requirements
    if default_cp:
        cp = default_cp
        logger.info(f"Using configured CP override: CP={cp}")
    elif max_seq_len > 8192:
        # CP caveat: the miles seam returns per-CP-rank logprob chunks for
        # fb/forward (no full-sequence reassembly yet) — clients would see
        # half-length per-sample logprobs at CP=2. Engage CP only when the
        # sequence length actually requires it; ≤8K fits without CP for the
        # supported model sizes.
        cp = 2
    else:
        cp = 1  # Short sequences don't need CP

    # DP: computed from remaining resources
    dp = num_gpus // (tp * pp * cp)

    # Fallback: ensure at least DP=1 by reducing CP/PP
    while dp < 1 and cp > 1:
        cp = 1
        dp = num_gpus // (tp * pp * cp)
        logger.warning("Reduced CP to 1 to ensure DP >= 1")
    while dp < 1 and pp > 1:
        pp = 1
        dp = num_gpus // (tp * pp * cp)
        logger.warning("Reduced PP to 1 to ensure DP >= 1")

    logger.info(f"Auto-detected parallelism: TP={tp}, PP={pp}, CP={cp}, DP={max(1, dp)}")
    return {'tp': tp, 'pp': pp, 'cp': cp, 'dp': max(1, dp)}


def detect_torch_dist_path(base_model: str) -> tuple[str, str]:
    """
    Auto-detect torch_dist model path from HF path or model name.

    Args:
        base_model: Base model path, HuggingFace model name (e.g., "Qwen/Qwen2.5-7B-Instruct"),
                   or torch_dist format path

    Returns:
        Tuple of (megatron_checkpoint_path, hf_model_path)
    """
    # Get model directory from environment (default: /data/models)
    model_dir = os.getenv("HF_HOME", "/data/models")

    # If base_model looks like a HuggingFace model name (contains "/"), resolve to local path
    # e.g., "Qwen/Qwen2.5-7B-Instruct" -> "/data/models/Qwen2.5-7B-Instruct"
    if "/" in base_model and not base_model.startswith("/"):
        model_name = base_model.split("/")[-1]  # Get "Qwen2.5-7B-Instruct" from "Qwen/Qwen2.5-7B-Instruct"
        local_path = os.path.join(model_dir, model_name)
        if os.path.exists(local_path):
            logger.info(f"Resolved HF model name to local path: {base_model} → {local_path}")
            base_model = local_path
        else:
            logger.warning(f"HF model {base_model} not found at {local_path}, will use as-is")

    if not base_model.endswith('_torch_dist'):
        torch_dist_path = f"{base_model}_torch_dist"
        if os.path.exists(torch_dist_path):
            logger.info(
                f"Auto-detected torch_dist model: {base_model} → {torch_dist_path}"
            )
            return torch_dist_path, base_model
        else:
            logger.warning(
                f"No torch_dist version found at {torch_dist_path}, "
                f"using {base_model} as-is"
            )
            return base_model, base_model
    else:
        # Already torch_dist format
        hf_path = base_model.replace('_torch_dist', '')
        logger.info(f"Using torch_dist model: {base_model}")
        return base_model, hf_path


def parse_checkpoint_uri(checkpoint_path: str, save_dir: str = "/data/checkpoints/tinker") -> str:
    """
    Parse tinker:// URI to filesystem checkpoint path.

    Args:
        checkpoint_path: Checkpoint path (tinker:// URI or filesystem path)
        save_dir: Base directory for checkpoints

    Returns:
        Filesystem checkpoint path

    Raises:
        ValueError: If tinker:// URI format is invalid
    """
    if checkpoint_path.startswith("tinker://"):
        import hashlib

        uri_parts = checkpoint_path.replace("tinker://", "").split("/")
        if len(uri_parts) >= 3 and uri_parts[1] == "weights":
            checkpoint_name = uri_parts[2]
            # Use same step_id calculation as save_weights
            step_id = int(
                hashlib.md5(checkpoint_name.encode()).hexdigest()[:8], 16
            ) % 100000
            filesystem_path = f"{save_dir}/iter_{step_id:07d}"
            logger.info(
                f"Checkpoint resume: {checkpoint_path} → {filesystem_path} "
                f"(step_id={step_id})"
            )
            return filesystem_path
        else:
            raise ValueError(f"Invalid tinker:// URI format: {checkpoint_path}")
    else:
        # Direct filesystem path
        logger.info(f"Checkpoint resume: loading from {checkpoint_path}")
        return checkpoint_path


def compute_sglang_mem_fraction(model_config: Dict[str, Any], model_name: str = "") -> float:
    """
    Compute SGLang memory fraction based on model size.

    Smaller models need less GPU memory for KV cache, so we can use a smaller fraction.
    This allows colocated training without offload for small models.

    Args:
        model_config: Model configuration dict from load_model_config()
        model_name: Optional model name/path for extracting size

    Returns:
        Memory fraction (0.0-1.0) for SGLang's mem_fraction_static
    """
    total_params = estimate_model_params(model_config, model_name)

    # Memory fraction based on model size:
    # - Larger models need more KV cache memory
    # - Smaller models can use less, leaving room for Megatron
    if total_params <= 1.0:       # <= 1B: 10% (~19GB on H200)
        mem_fraction = 0.10
    elif total_params <= 2.0:     # 1-2B: 15% (~28GB)
        mem_fraction = 0.15
    elif total_params <= 4.0:     # 2-4B: 20% (~38GB)
        mem_fraction = 0.20
    elif total_params <= 8.0:     # 4-8B: 30% (~57GB)
        mem_fraction = 0.30
    elif total_params <= 14.0:    # 8-14B: 40% (~76GB)
        mem_fraction = 0.40
    elif total_params <= 35.0:    # 14-35B: 50% (~95GB)
        mem_fraction = 0.50
    else:                          # > 35B: 70% (~132GB)
        mem_fraction = 0.70

    logger.info(
        f"SGLang mem_fraction={mem_fraction:.2f} for {total_params:.1f}B model"
    )
    return mem_fraction
