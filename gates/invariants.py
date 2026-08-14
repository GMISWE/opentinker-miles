"""Invariant registry: the paper's S3 contract as first-class code objects.

Each entry = (id, statement, claimed level, comparator, tolerance source,
paper pointer). Gates and RequestValidator rules derive from this registry;
verdict JSONs cite entries by id. Statements are one-line summaries — the
normative text is the cited paper section / spec.
"""

from dataclasses import dataclass

# Tolerance sources:
#   "exact"                       E0: bit-identity, no tolerance
#   "calibration:<key>"           E1: bar read from gates/calibration.json (P3)
#   "measured:rerun_envelope"     E2: envelope estimated from rerun population


@dataclass(frozen=True)
class Invariant:
    id: str
    statement: str
    level: str  # claimed equivalence level: "E0" | "E1" | "E2"
    comparator: str  # key into comparators.REGISTRY
    tolerance_source: str
    paper_ref: str  # label in paper/main.tex (monorepo)


REGISTRY: dict[str, Invariant] = {
    inv.id: inv
    for inv in [
        Invariant(
            id="SPLIT_INV",
            statement=(
                "fb(D1)..fb(DN) optim_step == fb(D) optim_step for every "
                "contiguous segmentation D1..DN of a round's data"
            ),
            level="E0",
            comparator="e0_bitwise",
            tolerance_source="exact",
            paper_ref="inv:split",
        ),
        Invariant(
            id="PARTITION_INV",
            statement=(
                "the round's gradient does not depend on which datums share a "
                "data-parallel rank: permuting the submission order of a "
                "single fb call leaves optim_step's gradient unchanged"
            ),
            level="E0",
            comparator="e0_bitwise",
            tolerance_source="exact",
            # strictly stronger than SPLIT_INV -- the DP-reduction defect
            # (tinker-cloud 39add0a) was invisible at SPLIT_INV's aligned arm
            # yet caught by a single permuted call. The manuscript has no
            # dedicated partition label yet; inv:split is the nearest.
            paper_ref="inv:split (no dedicated partition label; see HANDOFF 0-NEXT19)",
        ),
        Invariant(
            id="SEED_REPRO",
            statement=(
                "two models created with the same client-supplied seed give "
                "the same gradient on the same data, and a different seed "
                "gives a different one -- the seed is read, not defaulted"
            ),
            level="E0",
            comparator="e0_bitwise",
            tolerance_source="exact",
            paper_ref="sec:model:obs (client-supplied seed; no dedicated label)",
        ),
        Invariant(
            id="DEFER_OBS",
            statement=(
                "deferring fb execution to optim_step is the identity on obs: "
                "every eager key present, logprobs under pre-step weights W_k"
            ),
            level="E0",
            comparator="e0_bitwise",
            tolerance_source="exact",
            paper_ref="inv:defer",
        ),
        Invariant(
            id="ISOLATION",
            statement=(
                "co-located tenants observe what they would observe running "
                "serialized: per-slot grad_norm + logprobs unchanged by co-tenants"
            ),
            level="E0",
            comparator="e0_bitwise",
            tolerance_source="exact",
            paper_ref="sec:sched (multi-tenant co-location)",
        ),
        Invariant(
            id="PIN_DRIFT",
            statement=(
                "a sampler pinned to ver(S)=v returns bit-stable outputs across "
                "subsequent training steps; live samplers track post-step weights"
            ),
            level="E0",
            comparator="e0_bitwise",
            tolerance_source="exact",
            paper_ref="sec:model (observation function, ver(S))",
        ),
        Invariant(
            id="ADAPTER_ORACLE",
            statement=(
                "training-engine logprobs match an offline fp32 HF(+PEFT) oracle "
                "on the exported adapter, within the calibrated E1 bar"
            ),
            level="E1",
            comparator="e1_threshold",
            tolerance_source="calibration:e1_bar",
            paper_ref="sec:checking (model-derived metamorphic tests)",
        ),
        Invariant(
            id="ENDPOINT_NLL",
            statement=(
                "endpoint NLL of a fixed recipe lands inside the measured "
                "rerun envelope of the reference arm"
            ),
            level="E2",
            comparator="e2_envelope",
            tolerance_source="measured:rerun_envelope",
            paper_ref="sec:model:eq (equivalence hierarchy, E2)",
        ),
    ]
}


def get(inv_id: str) -> Invariant:
    return REGISTRY[inv_id]
