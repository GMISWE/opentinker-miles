"""Reverse direction: TrainIR -> sft.yaml, closed by the SEMANTIC round-trip —
re-elaborating the emitted yaml must reproduce the original program hash.
Byte-comparing yaml would be the wrong test; program identity is the claim."""

import os
from dataclasses import replace

from lift import ir
from lift.emit_native.nemo_rl import EmitNativeNemoRLPass, NativeEmission
from lift.frontends.nemo_rl import elaborate_sft
from lift.passes import default_registry
from lift.pm import PassManager

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "testdata", "nemo_rl_sft.yaml")


def _emit_native(lifted):
    reg = default_registry()
    reg["emit-native-nemo-rl"] = EmitNativeNemoRLPass(
        profile=lifted.profile, assets=lifted.assets
    )
    return PassManager("emit-native-nemo-rl", reg).run(lifted.program)


def test_semantic_round_trip(tmp_path):
    lifted = elaborate_sft(FIXTURE)
    out, rpt = _emit_native(lifted)
    assert not rpt.stopped
    assert isinstance(out, NativeEmission)

    emitted = tmp_path / "roundtrip.yaml"
    emitted.write_text(out.yaml_text)
    again = elaborate_sft(str(emitted))
    assert again.program.hash == lifted.program.hash
    assert again.unmapped == []


def test_rightward_i_ledger_linear_schedule_refused():
    lifted = elaborate_sft(FIXTURE)
    prog = replace(lifted.program, lr=ir.LinearLR(5e-6))
    reg = default_registry()
    reg["emit-native-nemo-rl"] = EmitNativeNemoRLPass(lifted.profile, lifted.assets)
    _, rpt = PassManager("emit-native-nemo-rl", reg).run(prog)
    assert rpt.stopped and "no LR scheduler" in rpt.records[0].refusal.reason


def test_rightward_i_ledger_weighted_loss_refused():
    lifted = elaborate_sft(FIXTURE)
    weighted = ir.Reduce(ir.Agg.SUM, None, ir.Tokens(), ir.PerTok("ce"), ir.W())
    prog = replace(lifted.program, loss=weighted)  # pure sum, no global-mean fold
    reg = default_registry()
    reg["emit-native-nemo-rl"] = EmitNativeNemoRLPass(lifted.profile, lifted.assets)
    _, rpt = PassManager("emit-native-nemo-rl", reg).run(prog)
    assert rpt.stopped and "NLLLoss" in rpt.records[0].refusal.reason


def test_canonicalized_loss_form_also_accepted():
    lifted = elaborate_sft(FIXTURE)
    prog, rpt0 = PassManager("norm-canon", default_registry()).run(lifted.program)
    assert not rpt0.stopped
    reg = default_registry()
    reg["emit-native-nemo-rl"] = EmitNativeNemoRLPass(lifted.profile, lifted.assets)
    out, rpt = PassManager("emit-native-nemo-rl", reg).run(prog)
    assert not rpt.stopped and isinstance(out, NativeEmission)


def test_truncate_policy_mismatch_refused():
    lifted = elaborate_sft(FIXTURE)
    bs = lifted.program.stream
    prog = replace(
        lifted.program, stream=replace(bs, inner=replace(bs.inner, policy="truncate"))
    )
    reg = default_registry()
    reg["emit-native-nemo-rl"] = EmitNativeNemoRLPass(lifted.profile, lifted.assets)
    _, rpt = PassManager("emit-native-nemo-rl", reg).run(prog)
    assert rpt.stopped and "overlength" in rpt.records[0].refusal.reason
