"""TrainIR -> nemo_rl SFT yaml. The forward elaborator's table, read
right-to-left. Refusals here ARE the rightward I-ledger entries:

- loss must be what NLLLoss computes — the global-token mean (accepted in
  both its surface form and its norm-canon'd weighted-sum form); any other
  weighting is inexpressible in the fixed native loss.
- LR schedules: the dtensor SFT path has NO scheduler, so only a constant
  LR is expressible (cookbook's default "linear" does not map — a real
  rightward gap).
- Truncate.policy must be "mask_out": the framework's over-length behavior
  is inherent, a tinker-born "truncate" program differs on over-length data.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

from lift import ir
from lift.pm import AnalysisManager, ELevel, Outcome


@dataclass(frozen=True)
class NativeEmission:
    framework: str
    flat: dict
    yaml_text: str

    @property
    def hash(self) -> str:
        import hashlib

        return hashlib.sha256(self.yaml_text.encode()).hexdigest()


def _loss_is_native(loss: ir.LossExpr) -> bool:
    if not isinstance(loss, ir.Reduce) or not isinstance(loss.over, ir.Tokens):
        return False
    if not isinstance(loss.of, ir.PerTok) or loss.of.op != "ce":
        return False
    if loss.agg is ir.Agg.MEAN:
        return isinstance(loss.denom, ir.Count) and isinstance(
            loss.denom.over, ir.Tokens
        )
    # norm-canon'd equivalent: sum with weights W / Count(Tokens)
    if loss.agg is ir.Agg.SUM and isinstance(loss.weight, ir.WScaled):
        w = loss.weight
        return (
            isinstance(w.base, ir.W)
            and isinstance(w.denom, ir.Count)
            and isinstance(w.denom.over, ir.Tokens)
        )
    return False


def _nest(flat: dict) -> dict:
    out: dict = {}
    for key, v in flat.items():
        cur = out
        parts = key.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    return out


class EmitNativeNemoRLPass:
    name = "emit-native-nemo-rl"
    level = ELevel.E0

    def __init__(self, profile: dict | None = None, assets: dict | None = None):
        self.profile = dict(profile or {})
        self.assets = dict(assets or {})

    def run(self, program: ir.Program, am: AnalysisManager, opts: dict) -> Outcome:
        p = program
        if ir.holes(p):
            return Outcome.refused("open term", ir.holes(p)[0])
        if not _loss_is_native(p.loss):
            return Outcome.refused(
                "loss not expressible by NLLLoss (global-token mean); "
                "weighted per-token losses are a rightward I-ledger entry",
                p.loss,
            )
        if not isinstance(p.lr, ir.ConstLR):
            return Outcome.refused(
                f"{type(p.lr).__name__}: dtensor SFT path has no LR scheduler; "
                "only constant LR is expressible",
                p.lr,
            )

        bs = p.stream
        if not isinstance(bs, ir.BatchStream):
            return Outcome.refused("stream has no batch boundary", p)
        tr = bs.inner
        assert isinstance(tr, ir.Truncate)
        if tr.policy != "mask_out":
            return Outcome.refused(
                "overlength policy 'truncate' differs from the framework's "
                "inherent stub+mask_out behavior on over-length data",
                tr,
            )
        tok = tr.inner
        assert isinstance(tok, ir.Tokenize)
        ren = tok.inner
        assert isinstance(ren, ir.Render)
        src = ren.inner
        assert isinstance(src, ir.SrcData)
        ds_name, _, split = src.ref.rpartition(":")

        flat: dict = {
            "policy.model_name": p.model_ref,
            "policy.tokenizer.name": tok.tokenizer,
            "policy.train_global_batch_size": bs.batch_size,
            "policy.optimizer.name": "torch.optim.AdamW",
            "policy.optimizer.kwargs.lr": p.lr.base,
            "policy.optimizer.kwargs.weight_decay": p.opt.wd,
            "policy.optimizer.kwargs.betas": [p.opt.b1, p.opt.b2],
            "policy.optimizer.kwargs.eps": p.opt.eps,
            "policy.max_grad_norm": p.opt.clip,
            "data.train.dataset_name": ds_name,
            "data.train.split": split,
            "data.add_bos": tok.add_bos,
            "data.add_eos": tok.add_eos,
            "data.max_input_seq_length": tr.max_len,
            "data.shuffle": bs.seed is not None,
            "sft.max_num_epochs": bs.epochs,
            "sft.max_num_steps": p.horizon,
            "checkpointing.enabled": p.points.save_every is not None,
        }
        if bs.seed is not None:
            flat["sft.seed"] = bs.seed
        if p.points.eval_every is not None:
            flat["sft.val_period"] = p.points.eval_every
        if p.points.save_every is not None:
            flat["checkpointing.save_period"] = p.points.save_every
        if ren.renderer.startswith("jinja:"):
            text = self.assets.get(ren.renderer)
            if text is None:
                return Outcome.refused(f"template text missing for {ren.renderer}", ren)
            flat["policy.tokenizer.chat_template"] = text

        if p.adapter is not None:
            flat["policy.dtensor_cfg.lora_cfg.enabled"] = True
            flat["policy.dtensor_cfg.lora_cfg.dim"] = p.adapter.r
            flat["policy.dtensor_cfg.lora_cfg.alpha"] = p.adapter.alpha
            flat["policy.dtensor_cfg.lora_cfg.match_all_linear"] = not p.adapter.targets
            flat["policy.dtensor_cfg.lora_cfg.target_modules"] = list(p.adapter.targets)

        # X-values verbatim from the profile; program-derived keys win conflicts
        merged = {**self.profile, **flat}
        merged = {k: v for k, v in merged.items() if v is not None}
        text = yaml.safe_dump(_nest(merged), sort_keys=True, default_flow_style=False)
        return Outcome.changed(NativeEmission("nemo_rl", merged, text))
