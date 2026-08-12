"""Comparators discharge a claimed equivalence level on a pair of observables.

E0 = bit identity (compared on repr, so float64 round-trips exactly).
E1 = |a-b| within a calibrated bar (bar placement: specs/014 design S3.7).
E2 = value inside a measured rerun envelope.
All return a Comparison; rel_diff is always computed for diagnosis, even
when the verdict is bitwise.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Comparison:
    passed: bool
    delta: float | None  # observed deviation (relative for E0/E1)
    threshold: float | None  # bar / envelope half-width; None for E0


def rel_diff(a: float, b: float) -> float:
    if a == b:
        return 0.0
    if not (math.isfinite(a) and math.isfinite(b)) or a == 0:
        return float("inf")
    return abs(b - a) / abs(a)


def e0_bitwise(a: float, b: float) -> Comparison:
    return Comparison(passed=repr(a) == repr(b), delta=rel_diff(a, b), threshold=None)


def e1_threshold(a: float, b: float, bar: float) -> Comparison:
    d = abs(b - a)
    return Comparison(passed=d <= bar, delta=d, threshold=bar)


def e2_envelope(value: float, lo: float, hi: float) -> Comparison:
    mid = (lo + hi) / 2
    return Comparison(
        passed=lo <= value <= hi, delta=abs(value - mid), threshold=(hi - lo) / 2
    )


REGISTRY = {
    "e0_bitwise": e0_bitwise,
    "e1_threshold": e1_threshold,
    "e2_envelope": e2_envelope,
}
