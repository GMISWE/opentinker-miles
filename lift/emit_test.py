"""PM[Select]: select/emit over lifted programs, obligations included."""

import os

from lift import ir
from lift.emit import EMIT_PIPELINE, EmitResult, emit_registry
from lift.frontends.nemo_rl import elaborate_sft
from lift.pm import ELevel, PassManager

TD = os.path.join(os.path.dirname(__file__), "testdata")
SFT = os.path.join(TD, "nemo_rl_sft.yaml")
MEGA = os.path.join(TD, "sft_openmathinstruct2_megatron.yaml")


def _emit(path, pipeline=EMIT_PIPELINE):
    prog = elaborate_sft(path).program
    return PassManager(pipeline, emit_registry()).run(prog)


def test_emit_sft_yaml_kwargs_and_obligations():
    out, rpt = _emit(SFT)
    assert not rpt.stopped and rpt.guarantee is ELevel.E0
    assert isinstance(out, EmitResult)
    k = out.config_kwargs
    assert k["model_name"] == "meta-llama/Llama-3.2-1B"
    assert k["learning_rate"] == 5.0e-6 and k["lr_schedule"] == "constant"
    assert k["max_steps"] == 60 and k["save_every"] == 10 and k["eval_every"] == 10
    assert k["adam_beta2"] == 0.98 and k["adam_eps"] == 1e-5
    assert out.adam_params["weight_decay"] == 0.1
    assert out.adam_params["grad_clip_norm"] == 1.0
    kinds = {(o.kind, o.fieldname) for o in out.obligations}
    # full finetune -> surface obligation; wd/clip -> recipe gaps; dataset adapter
    assert ("surface", "adapter") in kinds
    assert ("recipe-gap", "weight_decay") in kinds
    assert ("recipe-gap", "grad_clip_norm") in kinds
    assert ("dataset", "dataset_builder") in kinds
    assert not out.runnable
    assert out.dataset_spec["source"] == "squad:train"
    assert out.dataset_spec["batch_size"] == 32


def test_emit_refuses_warmup_without_approx():
    out, rpt = _emit(MEGA)
    assert rpt.stopped
    rec = rpt.records[-1]
    assert rec.name == "emit" and rec.status == "refused"
    assert "WarmupLR" in rec.refusal.reason


def test_emit_after_sched_approx_succeeds_at_e1():
    out, rpt = _emit(MEGA, pipeline="sched-approx,select,emit")
    assert not rpt.stopped
    assert rpt.guarantee is ELevel.E1  # the fold is on the record
    assert isinstance(out, EmitResult)
    assert out.config_kwargs["lr_schedule"] == "constant"


def test_select_refuses_open_terms():
    prog = elaborate_sft(SFT).program
    from dataclasses import replace

    open_prog = replace(prog, loss=ir.Hole("ENV", ty="x"))
    _, rpt = PassManager("select", emit_registry()).run(open_prog)
    assert rpt.stopped and rpt.records[0].status == "refused"
