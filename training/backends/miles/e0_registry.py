"""E0 co-batching calibration registry.

The bit-exactness boundary that the co-batch admission guard enforces is a
MEASURED, per-(model x parallel config) constant, not a policy: at
Qwen2.5-0.5B/dp2 gradients diverge when a call's per-rank token total
crosses ~512, while at Qwen3-8B-Base/tp2 no boundary exists anywhere in the
swept range. The registry (e0_calibration.json) records those measurements;
this module resolves the guard threshold for a pool boot.

Resolution order:
  1. TINKERCLOUD_MILES_COBATCH_E0_TOKENS set and non-empty -> explicit
     override (an operator decision; logged as such).
  2. Registry entry for (model basename, tp, dp) -> its measured threshold;
     ``null`` in the registry means "no boundary found in the measured
     range" and resolves to guard-open (0).
  3. No entry -> UNCALIBRATED: the caller must disable co-batching
     entirely. Merging a configuration whose exactness behavior was never
     measured would turn the guard into an assertion.

New entries come from the calibration sweep
(`scripts/gates/g6b_cobatch_token_bucket.py --emit-entry`, guard off) and
land here by pull request — the entry is the reviewable certificate.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path(__file__).with_name("e0_calibration.json")


@dataclass(frozen=True)
class E0Resolution:
    source: str                    # "env" | "registry" | "uncalibrated"
    threshold: int                 # tokens/rank; 0 = guard open (no refusals)
    disable_cobatch: bool          # True only for "uncalibrated"


def _load_entries(path: Path = _REGISTRY_PATH) -> list:
    data = json.loads(path.read_text())
    if data.get("schema") != 1:
        raise ValueError(f"unsupported e0_calibration schema: {data.get('schema')!r}")
    return data["entries"]


def resolve_e0(base_model: str, tp: int, dp: int,
               registry_path: Path = _REGISTRY_PATH) -> E0Resolution:
    """Resolve the E0 guard threshold for a pool boot.

    ``base_model`` may be a hub id ("Qwen/Qwen2.5-0.5B") or a basename;
    registry keys are basenames.
    """
    env = os.environ.get("TINKERCLOUD_MILES_COBATCH_E0_TOKENS", "")
    if env.strip() != "":
        thr = int(env)
        logger.info(
            "E0 guard threshold from env override: %d tokens/rank "
            "(registry not consulted)", thr,
        )
        return E0Resolution(source="env", threshold=thr, disable_cobatch=False)

    basename = base_model.rsplit("/", 1)[-1]
    try:
        entries = _load_entries(registry_path)
    except (OSError, ValueError, KeyError) as e:
        logger.warning("E0 calibration registry unreadable (%s); treating "
                       "config as UNCALIBRATED", e)
        entries = []
    for ent in entries:
        if (ent.get("model") == basename
                and int(ent.get("tp", 0)) == int(tp)
                and int(ent.get("dp", 0)) == int(dp)):
            thr = ent.get("threshold_tokens_per_rank")
            if thr is None:
                logger.info(
                    "E0 calibration for (%s, tp=%d, dp=%d): no exactness "
                    "boundary in measured range %s — guard open",
                    basename, tp, dp, ent.get("measured_range_tokens_per_rank"),
                )
                return E0Resolution(source="registry", threshold=0,
                                    disable_cobatch=False)
            logger.info(
                "E0 calibration for (%s, tp=%d, dp=%d): threshold %d "
                "tokens/rank (measured %s)",
                basename, tp, dp, int(thr), ent.get("measured", {}).get("date"),
            )
            return E0Resolution(source="registry", threshold=int(thr),
                                disable_cobatch=False)

    logger.warning(
        "E0 calibration MISSING for (%s, tp=%d, dp=%d): co-batching DISABLED "
        "for this pool. To enable it, run the calibration sweep "
        "(scripts/gates/g6b_cobatch_token_bucket.py --emit-entry with the "
        "guard off on this exact model + parallel config) and add the "
        "emitted entry to e0_calibration.json.",
        basename, tp, dp,
    )
    return E0Resolution(source="uncalibrated", threshold=0, disable_cobatch=True)
