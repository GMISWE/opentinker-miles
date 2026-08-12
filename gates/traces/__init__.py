"""Canonical trace generators — the admission clause L(ref) ⊆ L(I) made
executable. Each trace is pure data + an execute(driver) method; SDK/torch
imports stay inside execute so verdict logic loads without a GPU stack.
"""

from gates.traces.seg_sweep import SegSweep  # noqa: F401
