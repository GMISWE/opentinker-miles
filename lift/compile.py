"""Compiler driver: front-end -> PM[TrainIR] -> PM[Select] -> LiftBundle.

The bundle is the unit the certifier consumes: the final term, the emitted
binding (with executable assets attached), and every report. A bundle whose
emit stage refused or still carries obligations is a compilation RECORD, not
a runnable program.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as _dc_replace

from lift import ir
from lift.emit import EMIT_PIPELINE, EmitResult, emit_registry
from lift.frontends.base import LiftResult
from lift.frontends.nemo_rl import elaborate_sft
from lift.passes import DEFAULT_PIPELINE, default_registry
from lift.pm import CompilationReport, PassManager


@dataclass
class LiftBundle:
    lift: LiftResult
    program: ir.Program  # after the TrainIR pipeline
    emit: EmitResult | None  # None when emission refused/stopped
    reports: dict[str, CompilationReport]

    @property
    def runnable(self) -> bool:
        return self.emit is not None and self.emit.runnable


def compile_sft(
    path: str,
    ir_pipeline: str = DEFAULT_PIPELINE,
    emit_pipeline: str = EMIT_PIPELINE,
) -> LiftBundle:
    lifted = elaborate_sft(path)
    prog, ir_rpt = PassManager(ir_pipeline, default_registry()).run(
        lifted.program, source=path
    )
    assert isinstance(prog, ir.Program)
    reports = {"ir": ir_rpt}
    emit_res: EmitResult | None = None
    if not ir_rpt.stopped:
        out, emit_rpt = PassManager(emit_pipeline, emit_registry()).run(
            prog, source=path
        )
        reports["emit"] = emit_rpt
        if isinstance(out, EmitResult):
            spec = dict(out.dataset_spec)
            text = lifted.assets.get(spec.get("renderer", ""))
            if text is not None:
                spec["template_text"] = (
                    text  # digest stays the identity; text is executable
                )
            emit_res = _dc_replace(out, dataset_spec=spec, profile=dict(lifted.profile))
    return LiftBundle(lift=lifted, program=prog, emit=emit_res, reports=reports)
