"""PassManager: pipelines, refusal policies, level accounting, hole conservation."""

from lift import ir
from lift.ir_test import NEMO, TINKER, VERL, _prog
from lift.passes import (
    DEFAULT_PIPELINE,
    PERMISSIVE_PIPELINE,
    default_registry,
)
from lift.pm import ELevel, PassManager, parse_pipeline


def _pm(pipeline: str) -> PassManager:
    return PassManager(pipeline, default_registry())


def test_parse_pipeline():
    assert parse_pipeline("a,b<k=v;j=w>,c") == [
        ("a", {}),
        ("b", {"k": "v", "j": "w"}),
        ("c", {}),
    ]


def test_tinker_sum_passes_unchanged_at_e0():
    out, rpt = _pm(DEFAULT_PIPELINE).run(_prog(TINKER))
    assert not rpt.stopped
    assert rpt.guarantee is ELevel.E0
    assert out.loss == TINKER


def test_verl_global_mean_canonicalizes_at_e0():
    out, rpt = _pm(DEFAULT_PIPELINE).run(_prog(VERL))
    assert not rpt.stopped
    assert rpt.guarantee is ELevel.E0
    loss = out.loss
    assert isinstance(loss, ir.Reduce) and loss.agg is ir.Agg.SUM
    assert isinstance(loss.weight, ir.WScaled)
    assert isinstance(loss.weight.denom, ir.Count)  # w/global-token-count


def test_nemo_leak_refused_with_witness_by_default():
    out, rpt = _pm(DEFAULT_PIPELINE).run(_prog(NEMO))
    assert rpt.stopped
    rec = next(r for r in rpt.records if r.name == "norm-canon")
    assert rec.status == "refused"
    assert "mb" in rec.refusal.reason
    assert rec.refusal.witness["_"] == "Nested"
    assert out.loss == NEMO  # untouched
    assert rpt.leaks == ("mb",)


def test_nemo_repair_policy_yields_global_mean_at_e2():
    out, rpt = _pm(PERMISSIVE_PIPELINE).run(_prog(NEMO))
    assert not rpt.stopped
    assert rpt.guarantee is ELevel.E2  # the declared semantic change costs E0
    loss = out.loss
    assert isinstance(loss, ir.Reduce)
    assert loss.agg is ir.Agg.MEAN and isinstance(loss.denom, ir.Count)
    assert isinstance(loss.denom.over, ir.Tokens)
    assert ir.leaks(out) == frozenset()  # repair removed the execution coupling


def test_data_defined_nested_canonicalizes_at_e0():
    per_seq = ir.Part("sequence")  # data-defined: TRL-style per-sample mean
    loss = ir.Nested(
        ir.Agg.SUM,
        per_seq,
        ir.Reduce(
            ir.Agg.MEAN, ir.Count(ir.CurGroup()), ir.CurGroup(), ir.PerTok("ce"), ir.W()
        ),
    )
    out, rpt = _pm(DEFAULT_PIPELINE).run(_prog(loss))
    assert not rpt.stopped and rpt.guarantee is ELevel.E0
    got = out.loss
    assert isinstance(got, ir.Reduce) and got.agg is ir.Agg.SUM
    assert isinstance(got.weight, ir.WGrouped) and got.weight.part == per_seq


def test_skip_policy_records_and_continues():
    out, rpt = _pm("wf,norm-canon<policy=skip>,hash").run(_prog(NEMO))
    assert not rpt.stopped
    assert out.loss == NEMO
    assert rpt.records[1].status == "refused"


def test_hole_conservation_guard():
    class BadPass:
        name = "bad"
        level = ELevel.E0

        def run(self, program, am, opts):
            from dataclasses import replace

            from lift.pm import Outcome

            return Outcome.changed(replace(program, loss=ir.Hole("U")))

    reg = default_registry() | {"bad": BadPass()}
    _, rpt = PassManager("bad", reg).run(_prog(TINKER))
    assert rpt.stopped and rpt.records[0].errors == ("hole count changed by pass",)


def test_report_serializes():
    _, rpt = _pm(PERMISSIVE_PIPELINE).run(_prog(NEMO), source="sft.yaml")
    s = rpt.to_json()
    assert "norm-repair" in s and "guarantee" in s
