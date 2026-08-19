"""NeMo RL front-end: examples/configs yaml (SFT family) -> TrainIR.

No nemo_rl import; the semantic table below is sourced from reading the
framework (file:line cited), per the front-end rule. Key facts:

- NLLLoss (nemo_rl/algorithms/loss_functions.py:484-561): token logprobs
  masked by token_mask[:,1:]*sample_mask, normalized by masked_mean(...,
  global_normalization_factor=global_valid_toks) — a GLOBAL-token mean.
  Each micro-batch contributes sum/global_valid_toks, so the loss is
  grouping-invariant: micro-batching classifies X constructively. (The
  per-micro-batch-mean pattern lives in RL loss paths, not here.)
- SFT step (nemo_rl/algorithms/sft.py): policy.train(BatchedDataDict{
  input_ids,input_lengths,token_mask,sample_mask}, NLLLoss()).
- horizon = min(max_num_epochs*len(loader), max_num_steps); len(loader) is
  not static, so horizon = max_num_steps with a ledger note.
- dtensor SFT path has no LR scheduler -> ConstLR. megatron_cfg.scheduler
  applies only when megatron_cfg.enabled.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import yaml

from lift import ir
from lift.frontends.base import LedgerEntry, LiftResult


class LiftError(Exception):
    pass


# --- ${...} resolution (OmegaConf-compatible subset: refs, mul, max) -----------

_INNER = re.compile(r"\$\{([^${}]+)\}")


def _lookup(root: dict, dotted: str):
    cur: Any = root
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise LiftError(f"unresolvable interpolation: ${{{dotted}}}")
        cur = cur[part]
    return cur


def _eval_expr(root: dict, expr: str):
    for fn in ("mul", "max"):
        if expr.startswith(fn + ":"):
            raw = expr[len(fn) + 1 :].split(",")
            vals = []
            for a in raw:
                a = a.strip()
                try:
                    vals.append(float(a) if "." in a else int(a))
                except ValueError:
                    vals.append(_lookup(root, a))
            out = vals[0] * vals[1] if fn == "mul" else max(vals)
            return int(out) if all(isinstance(v, int) for v in vals) else out
    return _lookup(root, expr)


def resolve(root: dict) -> dict:
    """Iterate innermost-first substitution to fixpoint."""

    def sub_str(s: str):
        m = _INNER.search(s)
        if not m:
            return s, False
        val = _eval_expr(root, m.group(1).strip())
        if m.span() == (0, len(s)):
            return val, True
        return s[: m.start()] + str(val) + s[m.end() :], True

    def go(node):
        changed = False
        if isinstance(node, dict):
            for k, v in node.items():
                nv, ch = go_val(v)
                node[k] = nv
                changed |= ch
        elif isinstance(node, list):
            for i, v in enumerate(node):
                nv, ch = go_val(v)
                node[i] = nv
                changed |= ch
        return changed

    def go_val(v):
        if isinstance(v, str):
            return sub_str(v)
        if isinstance(v, (dict, list)):
            return v, go(v)
        return v, False

    for _ in range(20):
        if not go(root):
            return root
    raise LiftError("interpolation did not reach fixpoint (cycle?)")


# --- config loading with `defaults:` includes ----------------------------------


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = dict(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> dict:
    """yaml + nemo_rl's `defaults:` include chain (string or list of relative
    paths), bases merged in order, the file's own keys overlaid last."""
    import os

    with open(path) as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise LiftError(f"{path}: not a mapping")
    bases = cfg.pop("defaults", None)
    if bases is None:
        return cfg
    if isinstance(bases, str):
        bases = [bases]
    merged: dict = {}
    for b in bases:
        merged = _deep_merge(
            merged, load_config(os.path.join(os.path.dirname(path), b))
        )
    return _deep_merge(merged, cfg)


# --- flattening ----------------------------------------------------------------


def _flatten(d: dict, prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


# --- the elaborator ------------------------------------------------------------


class _Elab:
    def __init__(self, source: str, cfg: dict):
        self.source = source
        self.cfg = cfg
        self.flat = _flatten(cfg)
        self.left = set(self.flat)
        self.ledger: list[LedgerEntry] = []
        self.profile: dict[str, Any] = {}
        self.assets: dict[str, str] = {}

    def _loc(self, key: str) -> ir.Loc:
        return ir.Loc(self.source, key)

    def take(self, key: str, default=None, required=False):
        if key in self.left:
            self.left.discard(key)
            return self.flat[key]
        if required:
            raise LiftError(f"{self.source}: missing required key {key}")
        return default

    def p(self, key: str, value, note: str = ""):
        self.ledger.append(LedgerEntry(key, "P", value, note))
        return value

    def x(self, key: str, value, note: str = "exec"):
        self.ledger.append(LedgerEntry(key, "X", value, note))
        self.profile[key] = value
        return value

    def t(self, key: str, value, note: str):
        self.ledger.append(LedgerEntry(key, "T", value, note))

    def i(self, key: str, value, note: str):
        self.ledger.append(LedgerEntry(key, "I", value, note))

    def take_x_subtree(self, prefix: str, note: str = "exec"):
        for k in sorted([k for k in self.left if k.startswith(prefix)]):
            self.x(k, self.take(k), note)


def elaborate_sft(path: str) -> LiftResult:
    cfg = resolve(load_config(path))
    if "sft" not in cfg:
        raise LiftError(f"{path}: not an SFT config (no top-level 'sft')")
    e = _Elab(path, cfg)

    # --- state / adapter
    model = e.p("policy.model_name", e.take("policy.model_name", required=True))
    megatron_on = bool(e.take("policy.megatron_cfg.enabled"))
    e.p(
        "policy.megatron_cfg.enabled",
        megatron_on,
        "megatron execution path" if megatron_on else "inert engine block",
    )
    adapter = None
    if megatron_on:
        if e.take("policy.megatron_cfg.peft.enabled"):
            e.p("policy.megatron_cfg.peft.enabled", True)
            dim = e.p(
                "policy.megatron_cfg.peft.dim", e.take("policy.megatron_cfg.peft.dim")
            )
            alpha = e.p(
                "policy.megatron_cfg.peft.alpha",
                e.take("policy.megatron_cfg.peft.alpha"),
            )
            targets = tuple(e.take("policy.megatron_cfg.peft.target_modules") or [])
            e.p("policy.megatron_cfg.peft.target_modules", list(targets))
            excl = e.take("policy.megatron_cfg.peft.exclude_modules") or []
            if excl:
                e.i(
                    "policy.megatron_cfg.peft.exclude_modules",
                    excl,
                    "no exclude list on surface",
                )
            dropout = e.take("policy.megatron_cfg.peft.dropout") or 0.0
            if dropout:
                e.i(
                    "policy.megatron_cfg.peft.dropout",
                    dropout,
                    "tinker LoRA has no dropout",
                )
            e.take_x_subtree("policy.megatron_cfg.peft.", "peft impl details")
            adapter = ir.Lora(r=int(dim), alpha=float(alpha), targets=targets)
        else:
            e.p("policy.megatron_cfg.peft.enabled", False, "full finetune")
            e.take_x_subtree("policy.megatron_cfg.peft.", "inert (peft disabled)")
        e.take_x_subtree("policy.dtensor_cfg.lora_cfg.", "inert (megatron path)")
    elif e.take("policy.dtensor_cfg.lora_cfg.enabled"):
        e.p("policy.dtensor_cfg.lora_cfg.enabled", True)
        dim = e.p(
            "policy.dtensor_cfg.lora_cfg.dim", e.take("policy.dtensor_cfg.lora_cfg.dim")
        )
        alpha = e.p(
            "policy.dtensor_cfg.lora_cfg.alpha",
            e.take("policy.dtensor_cfg.lora_cfg.alpha"),
        )
        match_all = e.take("policy.dtensor_cfg.lora_cfg.match_all_linear")
        targets = tuple(e.take("policy.dtensor_cfg.lora_cfg.target_modules") or [])
        e.p("policy.dtensor_cfg.lora_cfg.match_all_linear", match_all)
        e.p("policy.dtensor_cfg.lora_cfg.target_modules", list(targets))
        if match_all:
            targets = ()
        excl = e.take("policy.dtensor_cfg.lora_cfg.exclude_modules") or []
        if excl:
            e.i(
                "policy.dtensor_cfg.lora_cfg.exclude_modules",
                excl,
                "no exclude list on surface",
            )
        else:
            e.p("policy.dtensor_cfg.lora_cfg.exclude_modules", excl)
        dropout = e.take("policy.dtensor_cfg.lora_cfg.dropout") or 0.0
        if dropout:
            e.i(
                "policy.dtensor_cfg.lora_cfg.dropout",
                dropout,
                "tinker LoRA has no dropout",
            )
        else:
            e.p("policy.dtensor_cfg.lora_cfg.dropout", 0.0)
        e.x(
            "policy.dtensor_cfg.lora_cfg.dropout_position",
            e.take("policy.dtensor_cfg.lora_cfg.dropout_position"),
            "inert (dropout=0)" if not dropout else "exec",
        )
        e.t(
            "policy.dtensor_cfg.lora_cfg.lora_A_init",
            e.take("policy.dtensor_cfg.lora_cfg.lora_A_init"),
            "init handled by target runtime; endpoint-level effect, certifier grades",
        )
        e.x(
            "policy.dtensor_cfg.lora_cfg.use_triton",
            e.take("policy.dtensor_cfg.lora_cfg.use_triton"),
            "kernel choice",
        )
        adapter = ir.Lora(r=int(dim), alpha=float(alpha), targets=targets)
    else:
        e.p("policy.dtensor_cfg.lora_cfg.enabled", False, "full finetune")
        e.take_x_subtree("policy.dtensor_cfg.lora_cfg.", "inert (lora disabled)")

    # --- stream
    ds_name = e.take("data.train.dataset_name") or e.take("data.train.data_path")
    e.p("data.train.dataset_name", ds_name)
    split = e.p("data.train.split", e.take("data.train.split", default="train"))
    # ResponseDataset-style field mapping / derived-split knobs (docs/guides/sft.md)
    for k in ("data.train.input_key", "data.train.output_key", "data.train.seed"):
        v = e.take(k)
        if v is not None:
            e.p(k, v, "dataset field/split-seed mapping")
    svs = e.take("data.train.split_validation_size")
    if svs is not None:
        e.t(
            "data.train.split_validation_size",
            svs,
            "derived val split -> eval data spec",
        )
    if "data.validation" in e.left and e.flat["data.validation"] is None:
        e.take("data.validation")
        e.p("data.validation", None, "val derived from train split")
    tmpl = e.take("policy.tokenizer.chat_template")
    if tmpl:
        renderer = "jinja:" + hashlib.sha256(str(tmpl).encode()).hexdigest()[:8]
        e.assets[renderer] = str(
            tmpl
        )  # digest names the term; text rides the side table
    else:
        renderer = "model_default"
    e.p(
        "policy.tokenizer.chat_template",
        renderer,
        "renderer identity = template digest",
    )
    tk_kwargs = e.take("policy.tokenizer.chat_template_kwargs")
    if tk_kwargs:
        e.t(
            "policy.tokenizer.chat_template_kwargs",
            tk_kwargs,
            "template kwargs -> renderer cfg",
        )
    else:
        e.p("policy.tokenizer.chat_template_kwargs", None)
    tok = e.p("policy.tokenizer.name", e.take("policy.tokenizer.name", default=model))
    add_bos = e.p("data.add_bos", e.take("data.add_bos", default=True))
    add_eos = e.p("data.add_eos", e.take("data.add_eos", default=True))
    e.p(
        "data.add_generation_prompt",
        e.take("data.add_generation_prompt", default=False),
    )
    # Engine sequence CAPACITY (allocation bound), not data truncation: X.
    # Reclassified P->X when the native emitter's round-trip exposed that the
    # term never carries it — the denotation depends on data.max_input_seq_length.
    pol_max = e.x(
        "policy.max_total_sequence_length",
        e.take("policy.max_total_sequence_length"),
        "engine sequence capacity",
    )
    max_len = e.p(
        "data.max_input_seq_length",
        e.take("data.max_input_seq_length", default=pol_max),
        "defaults to policy.max_total_sequence_length",
    )
    if max_len is None:
        raise LiftError(f"{path}: no sequence-length bound found")
    gbs = e.p(
        "policy.train_global_batch_size",
        e.take("policy.train_global_batch_size", required=True),
    )
    epochs = e.p("sft.max_num_epochs", e.take("sft.max_num_epochs", default=1))
    shuffle = e.p("data.shuffle", e.take("data.shuffle", default=True))
    seed = e.take("sft.seed", default=42)
    e.p("sft.seed", seed)
    e.x("data.num_workers", e.take("data.num_workers"), "loader parallelism")

    proc = e.take("data.default.processor", default="sft_processor")
    if proc not in (None, "sft_processor"):
        e.ledger.append(
            LedgerEntry("data.default.processor", "U", proc, "unknown processor")
        )
    else:
        e.p("data.default.processor", proc)
    for k in ("data.default.prompt_file", "data.default.system_prompt_file"):
        v = e.take(k)
        if v:
            e.t(k, v, "prompt injection -> renderer cfg")
        else:
            e.p(k, None)

    stream: ir.StreamExpr = ir.BatchStream(
        int(gbs),
        int(epochs),
        int(seed) if shuffle else None,
        ir.Truncate(
            int(max_len),
            ir.Tokenize(
                str(tok),
                # roles_to_train_on defaults to ["assistant"] (llm_message_utils.py:150)
                ir.Render(
                    renderer,
                    ir.SrcData(f"{ds_name}:{split}", loc=e._loc("data.train")),
                    train_on="assistant",
                ),
                add_bos=bool(add_bos),
                add_eos=bool(add_eos),
            ),
            # sft_processor over-length: stub + loss_multiplier=0, NOT truncation
            # (processors.py:171-177) — a different program from truncate.
            policy="mask_out",
        ),
    )

    # --- loss: NLLLoss = global-token mean (loss_functions.py:550-554)
    loss = ir.Reduce(
        ir.Agg.MEAN,
        ir.Count(ir.Tokens()),
        ir.Tokens(),
        ir.PerTok("ce", loc=e._loc("nemo_rl.NLLLoss")),
        ir.W(),
        loc=e._loc("nemo_rl.NLLLoss"),
    )

    # --- optimizer / schedule
    sched: ir.Sched
    if megatron_on:
        e.take("policy.optimizer")  # explicitly nulled by megatron variants
        e.p("policy.optimizer", None, "disabled; megatron_cfg.optimizer governs")
        e.take_x_subtree("policy.optimizer.", "inert (megatron optimizer governs)")
        mk = "policy.megatron_cfg.optimizer."
        oname = e.take(mk + "optimizer", required=True)
        wd = e.p(mk + "weight_decay", e.take(mk + "weight_decay", default=0.0))
        if oname != "adam":
            e.i(mk + "optimizer", oname, "only AdamW on the tinker surface")
        else:
            e.p(mk + "optimizer", oname, "adam + weight_decay = AdamW (megatron docs)")
        lr = e.p(mk + "lr", e.take(mk + "lr", required=True))
        b1 = e.p(mk + "adam_beta1", e.take(mk + "adam_beta1", default=0.9))
        b2 = e.p(mk + "adam_beta2", e.take(mk + "adam_beta2", default=0.95))
        eps = e.p(mk + "adam_eps", e.take(mk + "adam_eps", default=1e-8))
        clip = e.take(mk + "clip_grad")
        if clip == 0.0:
            clip = None  # megatron convention: clip_grad 0.0 disables clipping
            e.p(mk + "clip_grad", 0.0, "0.0 = disabled (megatron convention) -> None")
        else:
            e.p(mk + "clip_grad", clip)
        e.take("policy.max_grad_norm")
        e.p("policy.max_grad_norm", clip, "interpolated into clip_grad")
        opt = ir.AdamW(
            b1=float(b1),
            b2=float(b2),
            eps=float(eps),
            wd=float(wd),
            clip=float(clip) if clip is not None else None,
        )
        sk = "policy.megatron_cfg.scheduler."
        style = e.take(sk + "lr_decay_style", default="constant")
        warmup = e.take(sk + "lr_warmup_iters", default=0) or 0
        winit = e.take(sk + "lr_warmup_init", default=0.0) or 0.0
        if style != "constant":
            e.ledger.append(
                LedgerEntry(
                    sk + "lr_decay_style", "U", style, "decay style not in table"
                )
            )
            sched = ir.ConstLR(float(lr))
        else:
            e.p(sk + "lr_decay_style", style)
            base = ir.ConstLR(float(lr))
            if warmup:
                sched = ir.WarmupLR(
                    int(warmup), base, init_frac=float(winit) / float(lr)
                )
                e.p(sk + "lr_warmup_iters", warmup)
                e.p(sk + "lr_warmup_init", winit, "warmup start (init_frac)")
            else:
                sched = base
                e.p(sk + "lr_warmup_iters", 0)
                e.take(sk + "lr_warmup_init")
        swd, ewd = e.take(sk + "start_weight_decay"), e.take(sk + "end_weight_decay")
        if swd is not None and swd != ewd:
            e.ledger.append(
                LedgerEntry(
                    sk + "start_weight_decay",
                    "U",
                    (swd, ewd),
                    "wd schedule not in table",
                )
            )
        e.take_x_subtree(mk, "precision/distribution of the optimizer")
        e.take_x_subtree(sk, "inert (constant decay)")
    else:
        opt_name = e.take("policy.optimizer.name", required=True)
        if opt_name != "torch.optim.AdamW":
            e.i("policy.optimizer.name", opt_name, "only AdamW on the tinker surface")
        else:
            e.p("policy.optimizer.name", opt_name)
        kw = "policy.optimizer.kwargs."
        lr = e.p(kw + "lr", e.take(kw + "lr", required=True))
        wd = e.p(kw + "weight_decay", e.take(kw + "weight_decay", default=0.0))
        betas = e.p(kw + "betas", e.take(kw + "betas", default=[0.9, 0.95]))
        eps = e.p(kw + "eps", e.take(kw + "eps", default=1e-8))
        for k in ("foreach", "fused"):
            e.x(kw + k, e.take(kw + k), "kernel choice")
        clip = e.p("policy.max_grad_norm", e.take("policy.max_grad_norm"))
        opt = ir.AdamW(
            b1=float(betas[0]),
            b2=float(betas[1]),
            eps=float(eps),
            wd=float(wd),
            clip=float(clip) if clip is not None else None,
        )
        sched = ir.ConstLR(float(lr))  # dtensor SFT path: no scheduler

    # --- horizon & points
    horizon = e.p(
        "sft.max_num_steps",
        e.take("sft.max_num_steps"),
        "min(epochs*len(loader), max_num_steps); len(loader) not static",
    )
    eval_every = e.p("sft.val_period", e.take("sft.val_period"))
    for k in (
        "sft.val_batches",
        "sft.val_global_batch_size",
        "sft.val_at_start",
        "sft.val_at_end",
    ):
        e.t(k, e.take(k), "eval workload mapping")
    e.x("sft.val_micro_batch_size", e.take("sft.val_micro_batch_size"))
    save_every = None
    if e.take("checkpointing.enabled"):
        e.p("checkpointing.enabled", True)
        save_every = e.p(
            "checkpointing.save_period", e.take("checkpointing.save_period")
        )
    else:
        e.p("checkpointing.enabled", False)
        e.take("checkpointing.save_period")
    e.take_x_subtree("checkpointing.", "observability")

    # --- execution attrs
    mbs = e.take("policy.train_micro_batch_size")
    e.x(
        "policy.train_micro_batch_size",
        mbs,
        "grad-accum granularity; loss is grouping-invariant (global_valid_toks)",
    )
    e.x("policy.precision", e.take("policy.precision"), "numerics; certifier grades")
    e.x(
        "policy.offload_optimizer_for_logprob",
        e.take("policy.offload_optimizer_for_logprob"),
    )
    e.take_x_subtree("policy.dtensor_cfg.", "parallelism/memory")
    e.take_x_subtree("policy.dynamic_batching.", "micro-batch shaping")
    e.take_x_subtree("policy.sequence_packing.", "micro-batch shaping")
    e.x(
        "policy.make_sequence_length_divisible_by",
        e.take("policy.make_sequence_length_divisible_by"),
        "padding rule",
    )
    e.take_x_subtree(
        "policy.megatron_cfg.",
        "engine block" if megatron_on else "inert (megatron disabled)",
    )
    e.take_x_subtree("cluster.", "topology")
    e.take_x_subtree("logger.", "observability")
    e.take_x_subtree("data.validation.", "eval data (T with sft.val_*)")

    # --- anything left is unknown to this front-end version: loud
    for k in sorted(e.left):
        e.ledger.append(LedgerEntry(k, "U", e.flat[k], "unmapped"))

    program = ir.Program(
        model_ref=str(model),
        adapter=adapter,
        stream=stream,
        loss=loss,
        opt=opt,
        lr=sched,
        horizon=int(horizon) if horizon is not None else None,
        points=ir.Points(
            eval_every=int(eval_every) if eval_every else None,
            save_every=int(save_every) if save_every else None,
            publish="none",
        ),
        loc=ir.Loc(path, ""),
    )
    return LiftResult(
        source=path,
        program=program,
        profile=e.profile,
        ledger=e.ledger,
        assets=e.assets,
    )
