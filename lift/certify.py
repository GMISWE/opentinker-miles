"""Certificate assembly for a lifted program (translation validation,
specs/015-lift/DESIGN.md §5). Stage A (stream identity) is CPU-side and
lives in adapter.py; Stage B (training semantics on the frozen stream)
reuses the 008/013 gate machinery on pods. This module owns the fingerprint
utilities and the certificate record; the GPU driver binds at run time.

Rule: the certificate's measured level must equal the compilation report's
pass-derived guarantee — a mismatch is a compiler bug by definition.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StepFingerprint:
    num_tokens: int
    num_loss_tokens: int


def stream_fingerprint(batches: list[list[tuple[tuple[int, ...], tuple[int, ...]]]]):
    """Per-step (token count, loss-token count) — the step-0 entry is the
    38,055/27,067-style identity every arm must reproduce."""
    return [
        StepFingerprint(
            num_tokens=sum(len(ids) for ids, _ in b),
            num_loss_tokens=sum(sum(m) for _, m in b),
        )
        for b in batches
    ]


def fingerprint_digest(fps: list[StepFingerprint]) -> str:
    body = json.dumps([[f.num_tokens, f.num_loss_tokens] for f in fps])
    return hashlib.sha256(body.encode()).hexdigest()[:16]


@dataclass
class Check:
    name: str  # "stream-identity" | "step0-fingerprint" | "logprob-parity" | "endpoint"
    verdict: str  # "PASS" | "FAIL" | "DECLARED-T" | "SKIPPED"
    detail: str = ""


@dataclass
class Certificate:
    source: str
    program_hash: str
    pipeline: str
    claimed_level: str  # meet of pass levels, from the CompilationReport
    checks: list[Check] = field(default_factory=list)
    measured_level: str = ""  # filled by the GPU driver
    obligations_waived: list[str] = field(
        default_factory=list
    )  # each waiver is a decision

    @property
    def consistent(self) -> bool:
        """claimed == measured, and no FAIL. Inconsistency = compiler bug."""
        return self.measured_level == self.claimed_level and all(
            c.verdict != "FAIL" for c in self.checks
        )

    def to_json(self) -> str:
        return json.dumps(
            self.__dict__,
            default=lambda o: o.__dict__,
            indent=1,
            sort_keys=True,
        )
