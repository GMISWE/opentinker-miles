"""Initial TrainIR passes. Semantics and refusal rules: specs/015-lift/IR.md §7.

norm-canon refuses on execution-defined groupings (the leak judgment) instead
of silently normalizing them away — that refusal IS the Q1 defect-class
detector. norm-repair is the declared, E2-graded semantic change.
"""

from __future__ import annotations

from dataclasses import replace

from lift import ir
from lift.pm import AnalysisManager, ELevel, Outcome


def _denom_leak(d: ir.Node) -> ir.Part | None:
    for x in ir.walk(d):
        if isinstance(x, ir.Part) and x.attr is not None:
            return x
    return None


class VerifyPass:
    name = "wf"
    level = ELevel.E0

    def run(self, program: ir.Program, am: AnalysisManager, opts: dict) -> Outcome:
        errs = am.get("wf", program)
        return Outcome.error(*errs) if errs else Outcome.unchanged()


class SchedCanonPass:
    name = "sched-canon"
    level = ELevel.E0

    def run(self, program: ir.Program, am: AnalysisManager, opts: dict) -> Outcome:
        lr = program.lr
        if isinstance(lr, ir.WarmupLR) and lr.warmup_steps == 0:
            return Outcome.changed(replace(program, lr=lr.inner))
        return Outcome.unchanged()


class NormCanonPass:
    """Mean/nested forms -> weighted pure-sum. Side condition per rewrite: the
    folded denominator/grouping must be grouping-invariant (no attr-defined
    partition). Violation -> Refuse with the witness subterm."""

    name = "norm-canon"
    level = ELevel.E0

    def run(self, program: ir.Program, am: AnalysisManager, opts: dict) -> Outcome:
        loss = program.loss

        if isinstance(loss, ir.Hole) or (
            isinstance(loss, ir.Reduce) and loss.agg is ir.Agg.SUM
        ):
            return Outcome.unchanged()

        if isinstance(loss, ir.Reduce):  # MEAN over Tokens
            bad = _denom_leak(loss.denom) if loss.denom else None
            if bad is not None:
                return Outcome.refused(
                    f"denominator depends on execution attribute '{bad.attr}'", loss
                )
            new = ir.Reduce(
                agg=ir.Agg.SUM,
                denom=None,
                over=loss.over,
                of=loss.of,
                weight=ir.WScaled(loss.weight, loss.denom),  # type: ignore[arg-type]
                loc=loss.loc,
            )
            return Outcome.changed(replace(program, loss=new))

        if isinstance(loss, ir.Nested):
            if loss.part.attr is not None:
                return Outcome.refused(
                    f"grouping depends on execution attribute '{loss.part.attr}'", loss
                )
            inner = loss.inner
            if not isinstance(inner, ir.Reduce) or not isinstance(
                inner.over, ir.CurGroup
            ):
                return Outcome.refused("unsupported nesting shape", loss)

            weight: ir.Weight = inner.weight
            if inner.agg is ir.Agg.MEAN:
                if isinstance(inner.denom, ir.Count):
                    weight = ir.WGrouped(weight, loss.part, "count")
                elif isinstance(inner.denom, ir.WSum):
                    weight = ir.WGrouped(weight, loss.part, "wsum")
                else:
                    return Outcome.refused("unsupported inner denominator", loss)
            if loss.agg is ir.Agg.MEAN:
                weight = ir.WScaled(weight, ir.CountGroups(loss.part))

            new = ir.Reduce(
                agg=ir.Agg.SUM,
                denom=None,
                over=ir.Tokens(),
                of=inner.of,
                weight=weight,
                loc=loss.loc,
            )
            return Outcome.changed(replace(program, loss=new))

        return Outcome.unchanged()


class NormRepairPass:
    """Execution-grouped mean -> global mean. A DECLARED semantic change,
    graded E2 by the certificate harness (never applied implicitly; only via
    an explicit policy=repair:norm-repair in the pipeline)."""

    name = "norm-repair"
    level = ELevel.E2

    def run(self, program: ir.Program, am: AnalysisManager, opts: dict) -> Outcome:
        loss = program.loss
        if isinstance(loss, ir.Nested) and loss.part.attr is not None:
            inner = loss.inner
            if isinstance(inner, ir.Reduce):
                new = ir.Reduce(
                    agg=ir.Agg.MEAN,
                    denom=ir.Count(ir.Tokens()),
                    over=ir.Tokens(),
                    of=inner.of,
                    weight=inner.weight,
                    loc=loss.loc,
                )
                return Outcome.changed(replace(program, loss=new))
        if isinstance(loss, ir.Reduce) and loss.denom is not None:
            bad = _denom_leak(loss.denom)
            if bad is not None:
                new = replace(loss, denom=ir.Count(ir.Tokens()))
                return Outcome.changed(replace(program, loss=new))
        return Outcome.unchanged()


class SchedApproxPass:
    """Fold a DEGENERATE warmup (init_frac ~= 1 for very few steps) into its
    inner schedule. Level E1 by declaration: the first `warmup_steps` steps see
    an LR off by at most (1-init_frac) relative. Bound: warmup_steps *
    (1-init_frac)/2 <= tol (default 1e-4 fractional step-LR mass). Anything
    larger stays distinct — no silent thresholding beyond the declared tol."""

    name = "sched-approx"
    level = ELevel.E1

    def run(self, program: ir.Program, am: AnalysisManager, opts: dict) -> Outcome:
        lr = program.lr
        tol = float(opts.get("tol", 1e-4))
        if isinstance(lr, ir.WarmupLR):
            mass = lr.warmup_steps * (1.0 - lr.init_frac) / 2.0
            if 0 <= mass <= tol:
                return Outcome.changed(replace(program, lr=lr.inner))
        return Outcome.unchanged()


class AnalysisPass:
    """No-op transform that forces an analysis into the AM cache (ledger use)."""

    level = ELevel.E0

    def __init__(self, analysis: str):
        self.name = analysis
        self.analysis = analysis

    def run(self, program: ir.Program, am: AnalysisManager, opts: dict) -> Outcome:
        am.get(self.analysis, program)
        return Outcome.unchanged()


def default_registry() -> dict:
    ps = [
        VerifyPass(),
        SchedCanonPass(),
        SchedApproxPass(),
        NormCanonPass(),
        NormRepairPass(),
        AnalysisPass("leaks"),
        AnalysisPass("hash"),
        AnalysisPass("holes"),
    ]
    return {p.name: p for p in ps}


DEFAULT_PIPELINE = "wf,sched-canon,norm-canon,wf,hash"
PERMISSIVE_PIPELINE = "wf,sched-canon,norm-canon<policy=repair:norm-repair>,wf,hash"
LEDGER_PIPELINE = "wf,leaks,hash"
COLLAPSE_PIPELINE = "wf,sched-canon,sched-approx,norm-canon,wf,hash"
