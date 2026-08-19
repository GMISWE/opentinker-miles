"""TrainIR carrier + judgments. The three elaborations of IR.md Part II §4
are the fixture: tinker pure-sum / veRL global-token mean / NeMo RL
per-micro-batch mean (the leak(mb) term)."""

from lift import ir


def _prog(loss: ir.LossExpr) -> ir.Program:
    return ir.Program(
        model_ref="Qwen/Qwen2.5-0.5B",
        adapter=ir.Lora(r=32, alpha=32.0),
        stream=ir.BatchStream(
            128,
            1,
            0,
            ir.Truncate(
                4096, ir.Tokenize("qwen", ir.Render("chat", ir.SrcData("norobots")))
            ),
        ),
        loss=loss,
        opt=ir.AdamW(),
        lr=ir.LinearLR(2e-4),
        horizon=30,
        points=ir.Points(),
    )


TINKER = ir.Reduce(ir.Agg.SUM, None, ir.Tokens(), ir.PerTok("ce"), ir.W())
VERL = ir.Reduce(
    ir.Agg.MEAN, ir.Count(ir.Tokens()), ir.Tokens(), ir.PerTok("ce"), ir.W()
)
MB = ir.Part("microbatch", attr="mb")
NEMO = ir.Nested(
    ir.Agg.MEAN,
    MB,
    ir.Reduce(
        ir.Agg.MEAN, ir.Count(ir.CurGroup()), ir.CurGroup(), ir.PerTok("ce"), ir.W()
    ),
)


def test_wf_all_three():
    for loss in (TINKER, VERL, NEMO):
        assert ir.wf(_prog(loss)) == []


def test_curgroup_outside_nested_rejected():
    bad = ir.Reduce(ir.Agg.SUM, None, ir.CurGroup(), ir.PerTok("ce"), ir.W())
    assert any("CurGroup" in e for e in ir.wf(_prog(bad)))


def test_mean_needs_denominator():
    bad = ir.Reduce(ir.Agg.MEAN, None, ir.Tokens(), ir.PerTok("ce"), ir.W())
    assert any("denominator" in e for e in ir.wf(_prog(bad)))


def test_leaks_only_on_attr_partition():
    assert ir.leaks(_prog(TINKER)) == frozenset()
    assert ir.leaks(_prog(VERL)) == frozenset()
    assert ir.leaks(_prog(NEMO)) == frozenset({"mb"})


def test_hash_stable_and_loc_free():
    a = _prog(TINKER)
    b = _prog(
        ir.Reduce(
            ir.Agg.SUM,
            None,
            ir.Tokens(),
            ir.PerTok("ce"),
            ir.W(),
            loc=ir.Loc("sft.yaml", "loss"),
        )
    )
    assert a.hash == b.hash  # provenance never enters the hash
    assert a.hash != _prog(VERL).hash


def test_collapse_x_variants_same_hash():
    # megatron/fp8 variants differ only in Attrs, which live OUTSIDE the term:
    # identical Programs must hash identically (the collapse-ratio theorem).
    assert _prog(NEMO).hash == _prog(NEMO).hash


def test_holes_counted_and_typed():
    p = _prog(ir.Hole("ENV", ty="sample->batch+rewards"))
    hs = ir.holes(p)
    assert len(hs) == 1 and hs[0].kind == "ENV"
    assert ir.wf(p) == []
