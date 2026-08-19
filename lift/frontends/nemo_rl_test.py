"""nemo_rl front-end vs the actual upstream sft.yaml (fixture is a verbatim
copy of RL/examples/configs/sft.yaml)."""

import os

from lift import ir
from lift.frontends.nemo_rl import elaborate_sft, resolve
from lift.passes import DEFAULT_PIPELINE, default_registry
from lift.pm import ELevel, PassManager

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "testdata", "nemo_rl_sft.yaml")


def test_resolver_mul_and_refs():
    cfg = resolve(
        {
            "a": {"x": 4, "name": "m"},
            "b": "${mul:${a.x}, 2}",
            "c": "${a.name}",
            "d": "pre-${a.name}-post",
        }
    )
    assert cfg["b"] == 8 and cfg["c"] == "m" and cfg["d"] == "pre-m-post"


def test_elaborates_upstream_sft_yaml_completely():
    r = elaborate_sft(FIXTURE)
    assert r.unmapped == []  # every knob classified; U would block emission
    assert ir.wf(r.program) == []
    assert ir.leaks(r.program) == frozenset()


def test_program_fields_match_yaml():
    p = elaborate_sft(FIXTURE).program
    assert p.model_ref == "meta-llama/Llama-3.2-1B"
    assert p.adapter is None  # lora disabled in upstream sft.yaml
    assert p.horizon == 60
    assert p.opt == ir.AdamW(b1=0.9, b2=0.98, eps=1e-5, wd=0.1, clip=1.0)
    assert p.lr == ir.ConstLR(5.0e-6)
    assert p.points.eval_every == 10 and p.points.save_every == 10
    bs = p.stream
    assert isinstance(bs, ir.BatchStream) and bs.batch_size == 32 and bs.seed == 42
    tr = bs.inner
    assert isinstance(tr, ir.Truncate) and tr.max_len == 1024
    src = tr.inner.inner.inner
    assert isinstance(src, ir.SrcData) and src.ref == "squad:train"


def test_loss_is_global_token_mean_not_microbatch_mean():
    # NLLLoss normalizes by global_valid_toks (loss_functions.py:550-554):
    # grouping-invariant, so micro-batching classifies X and nothing leaks.
    p = elaborate_sft(FIXTURE).program
    assert isinstance(p.loss, ir.Reduce) and p.loss.agg is ir.Agg.MEAN
    assert isinstance(p.loss.denom, ir.Count)
    assert isinstance(p.loss.denom.over, ir.Tokens)


def test_profile_carries_execution_attrs():
    prof = elaborate_sft(FIXTURE).profile
    assert prof["policy.train_micro_batch_size"] == 1
    assert prof["policy.dynamic_batching.enabled"] is False
    assert prof["policy.dynamic_batching.train_mb_tokens"] == 1024  # ${mul:...}
    assert prof["policy.dtensor_cfg.tensor_parallel_size"] == 1
    assert prof["cluster.gpus_per_node"] == 1


def test_pipeline_canonicalizes_lifted_program_at_e0():
    r = elaborate_sft(FIXTURE)
    out, rpt = PassManager(DEFAULT_PIPELINE, default_registry()).run(
        r.program, source=r.source
    )
    assert not rpt.stopped and rpt.guarantee is ELevel.E0
    loss = out.loss
    assert isinstance(loss, ir.Reduce) and loss.agg is ir.Agg.SUM
    assert isinstance(loss.weight, ir.WScaled)


MEGATRON_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "testdata", "sft_openmathinstruct2_megatron.yaml"
)
BASE_FIXTURE = os.path.join(
    os.path.dirname(__file__), "..", "testdata", "sft_openmathinstruct2.yaml"
)


def test_megatron_variant_elaborates_completely():
    r = elaborate_sft(MEGATRON_FIXTURE)
    assert r.unmapped == [] and ir.wf(r.program) == []
    # megatron clip_grad 0.0 = disabled -> None (framework convention)
    assert r.program.opt.clip is None


def test_x_variant_collapse_under_declared_e1_pass():
    # Upstream's megatron twin differs at P-level only by a DEGENERATE 1-step
    # warmup (init_frac ~ 0.9999995). Strict E0 hashes differ; the explicit
    # sched-approx (E1) pass collapses the pair — the collapse-ratio theorem.
    from lift.passes import COLLAPSE_PIPELINE

    a = elaborate_sft(BASE_FIXTURE).program
    b = elaborate_sft(MEGATRON_FIXTURE).program
    assert a.hash != b.hash
    pm = PassManager(COLLAPSE_PIPELINE, default_registry())
    ca, ra = pm.run(a)
    cb, rb = pm.run(b)
    assert ca.hash == cb.hash
    assert ra.guarantee is ELevel.E0  # base needed no approximation
    assert rb.guarantee is ELevel.E1  # twin used the declared E1 fold
