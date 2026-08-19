"""PM[Select]: lower a hole-free Program onto the tinker surface and emit the
cookbook recipe binding (specs/015-lift/PASSMANAGER.md §6).

select: Program -> SelectResult (the op-stream template: what the reified
program will look like on the wire; R9 buffering is a legal schedule of it).
emit:   SelectResult -> EmitResult (supervised.train.Config kwargs +
tinker AdamParams + dataset spec + OBLIGATIONS). Obligation kinds:
  surface     the tinker op surface itself lacks it (e.g. full finetune)
  recipe-gap  AdamParams supports it but supervised.train.Config exposes no
              knob (weight_decay, grad_clip_norm) — cookbook backlog items
  dataset     needs a SupervisedDatasetBuilder adapter for the source
Emission with outstanding obligations is a report, not a runnable program;
the caller decides how each obligation is met and records that decision.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from lift import ir
from lift.passes import default_registry
from lift.pm import AnalysisManager, ELevel, Outcome


def _digest(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, default=str).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Obligation:
    kind: str  # "surface" | "recipe-gap" | "dataset"
    fieldname: str
    detail: str


@dataclass(frozen=True)
class OpStreamTemplate:
    horizon: int
    batch_size: int
    step_ops: tuple[str, ...]  # per training step, in submission order
    publish: str
    save_every: int
    eval_every: int
    note: str = (
        "fwd_bwd+optim_step submitted back-to-back; R9 buffering is a legal schedule"
    )


@dataclass(frozen=True)
class SelectResult:
    program: ir.Program
    template: OpStreamTemplate

    @property
    def hash(self) -> str:
        return _digest({"p": self.program.hash, "t": asdict(self.template)})


@dataclass(frozen=True)
class EmitResult:
    config_kwargs: dict
    adam_params: dict
    dataset_spec: dict
    obligations: tuple[Obligation, ...] = field(default_factory=tuple)
    profile: dict = field(default_factory=dict)

    @property
    def hash(self) -> str:
        return _digest(
            {
                "c": self.config_kwargs,
                "a": self.adam_params,
                "d": self.dataset_spec,
                "o": [asdict(o) for o in self.obligations],
            }
        )

    @property
    def runnable(self) -> bool:
        return not self.obligations


class SelectPass:
    name = "select"
    level = ELevel.E0

    def run(self, program: ir.Program, am: AnalysisManager, opts: dict) -> Outcome:
        hs = ir.holes(program)
        if hs:
            return Outcome.refused(
                f"{len(hs)} unfilled hole(s); selection needs a closed term", hs[0]
            )
        if program.horizon is None:
            return Outcome.refused("no horizon; the op stream is unbounded", program)
        bs = program.stream
        if not isinstance(bs, ir.BatchStream):
            return Outcome.refused("stream has no batch boundary", program)
        tmpl = OpStreamTemplate(
            horizon=program.horizon,
            batch_size=bs.batch_size,
            step_ops=("forward_backward", "optim_step"),
            publish=program.points.publish,
            save_every=program.points.save_every or 0,
            eval_every=program.points.eval_every or 0,
        )
        return Outcome.changed(SelectResult(program, tmpl))


_SCHED_NAMES = {ir.ConstLR: "constant", ir.LinearLR: "linear"}


class EmitPass:
    name = "emit"
    level = ELevel.E0

    def run(self, sel: SelectResult, am: AnalysisManager, opts: dict) -> Outcome:
        p = sel.program
        obl: list[Obligation] = []

        sched_name = _SCHED_NAMES.get(type(p.lr))
        if sched_name is None:
            return Outcome.refused(
                f"schedule {type(p.lr).__name__} not on the recipe surface "
                "(run sched-approx first if the warmup is degenerate)",
                p.lr,
            )
        base_lr = p.lr.base  # type: ignore[union-attr]

        if p.adapter is None:
            obl.append(
                Obligation(
                    "surface",
                    "adapter",
                    "full finetune: tinker clients are LoRA-only; supply a rank",
                )
            )
            lora_rank = None
        else:
            lora_rank = p.adapter.r
            if p.adapter.alpha != p.adapter.r:
                obl.append(
                    Obligation(
                        "surface",
                        "lora_alpha",
                        f"alpha={p.adapter.alpha} != r={p.adapter.r}; "
                        "tinker pins its own alpha convention",
                    )
                )
            if p.adapter.targets:
                obl.append(
                    Obligation(
                        "surface",
                        "lora_targets",
                        "arbitrary target-module lists not exposed; LoraConfig has only "
                        "coarse flags (train_mlp/train_attn/train_unembed)",
                    )
                )

        if p.opt.wd:
            obl.append(
                Obligation(
                    "recipe-gap",
                    "weight_decay",
                    "AdamParams.weight_decay exists; Config exposes no knob",
                )
            )
        if p.opt.clip is not None:
            obl.append(
                Obligation(
                    "recipe-gap",
                    "grad_clip_norm",
                    "AdamParams.grad_clip_norm exists; Config exposes no knob",
                )
            )

        bs = p.stream
        assert isinstance(bs, ir.BatchStream)
        tr = bs.inner
        assert isinstance(tr, ir.Truncate)
        tok = tr.inner
        assert isinstance(tok, ir.Tokenize)
        ren = tok.inner
        assert isinstance(ren, ir.Render)
        src = ren.inner
        assert isinstance(src, ir.SrcData)
        dataset_spec = {
            "source": src.ref,
            "renderer": ren.renderer,
            "train_on": ren.train_on,
            "tokenizer": tok.tokenizer,
            "add_bos": tok.add_bos,
            "add_eos": tok.add_eos,
            "max_length": tr.max_len,
            "overlength_policy": tr.policy,
            "batch_size": bs.batch_size,
            "epochs": bs.epochs,
            "seed": bs.seed,
        }
        obl.append(
            Obligation(
                "dataset",
                "dataset_builder",
                f"needs a SupervisedDatasetBuilder adapter for {src.ref}",
            )
        )

        kwargs = {
            "model_name": p.model_ref,
            "learning_rate": base_lr,
            "lr_schedule": sched_name,
            "num_epochs": bs.epochs,
            "max_steps": p.horizon,
            "save_every": sel.template.save_every,
            "eval_every": sel.template.eval_every,
            "adam_beta1": p.opt.b1,
            "adam_beta2": p.opt.b2,
            "adam_eps": p.opt.eps,
        }
        if lora_rank is not None:
            kwargs["lora_rank"] = lora_rank
        adam_params = {
            "beta1": p.opt.b1,
            "beta2": p.opt.b2,
            "eps": p.opt.eps,
            "weight_decay": p.opt.wd,
            "grad_clip_norm": p.opt.clip if p.opt.clip is not None else 0.0,
        }
        return Outcome.changed(
            EmitResult(
                kwargs,
                adam_params,
                dataset_spec,
                tuple(obl),
                dict(opts.get("profile", {})),
            )
        )


EMIT_PIPELINE = "select,emit"


def emit_registry() -> dict:
    reg = default_registry()
    for p in (SelectPass(), EmitPass()):
        reg[p.name] = p
    return reg
