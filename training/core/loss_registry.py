"""
Loss-function registry: the names the API accepts and the `loss_fn_config`
keys each one takes. Validated at the HTTP boundary (400 on failure) so a
misspelt name or a stray key never silently degrades to a default.

Per-backend support is declared on TrainingBackend.SUPPORTED_LOSS_FNS and
checked by TrainingService.
"""
from typing import Dict, FrozenSet, Mapping, Optional

CLIP_KEYS = frozenset({"clip_low_threshold", "clip_high_threshold"})

# name -> allowed loss_fn_config keys
LOSS_FNS: Dict[str, FrozenSet[str]] = {
    "cross_entropy": frozenset(),
    "importance_sampling": frozenset(),
    "ppo": CLIP_KEYS,
    "cispo": CLIP_KEYS,
    "dro": frozenset({"beta"}),
    "classification_ce": frozenset(),
}

# Tinker's documented PPO/CISPO defaults (epsilon 0.2 either side).
DEFAULT_CLIP_LOW = 0.8
DEFAULT_CLIP_HIGH = 1.2


def validate(loss_fn: str, loss_fn_config: Optional[Mapping[str, float]] = None) -> None:
    """Raise ValueError if `loss_fn` is unknown or `loss_fn_config` carries a key it does not take."""
    allowed = LOSS_FNS.get(loss_fn)
    if allowed is None:
        raise ValueError(f"Unknown loss_fn {loss_fn!r}; supported: {', '.join(sorted(LOSS_FNS))}")
    for key, value in (loss_fn_config or {}).items():
        if key not in allowed:
            take = f"takes {', '.join(sorted(allowed))}" if allowed else "takes no loss_fn_config"
            raise ValueError(f"loss_fn_config key {key!r} is not valid for {loss_fn!r} ({take})")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"loss_fn_config[{key!r}] must be a number, got {type(value).__name__}")
    if loss_fn_config and CLIP_KEYS & set(loss_fn_config):
        low = loss_fn_config.get("clip_low_threshold", DEFAULT_CLIP_LOW)
        high = loss_fn_config.get("clip_high_threshold", DEFAULT_CLIP_HIGH)
        if not (0.0 <= low <= 1.0 <= high):
            raise ValueError(
                f"clip thresholds must satisfy 0 <= clip_low_threshold <= 1 <= clip_high_threshold, got {low}, {high}"
            )


def clip_thresholds(loss_fn_config: Optional[Mapping[str, float]]) -> tuple:
    """(low, high) with Tinker's defaults filled in."""
    cfg = loss_fn_config or {}
    return (float(cfg.get("clip_low_threshold", DEFAULT_CLIP_LOW)),
            float(cfg.get("clip_high_threshold", DEFAULT_CLIP_HIGH)))
