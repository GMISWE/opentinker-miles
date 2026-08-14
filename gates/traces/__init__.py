"""Canonical trace generators — the admission clause L(ref) ⊆ L(I) made
executable. Each trace is pure data + an execute(driver) method; SDK/torch
imports stay inside execute so verdict logic loads without a GPU stack.
"""

from gates.traces.order_perm import OrderPerm  # noqa: F401
from gates.traces.seed_repro import SeedRepro  # noqa: F401
from gates.traces.seg_sweep import SegSweep  # noqa: F401
