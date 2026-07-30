#!/usr/bin/env python3
"""Pipelined-fb probe (2026-07-30): mimic the cookbook's submission pattern
fb(0), optim(0), fb(1), optim(1) all submitted before any await, then check
each fb result's per-sample logprob lengths against ITS OWN batch.

Batch 0 lengths: 40..421 (step 3); batch 1 lengths: 500..627 (step 1) —
disjoint ranges, so cross-wired futures show as bulk mismatches.
"""
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "tml-dev-key")

import torch
import tinker
from tinker import types

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LENS0 = [40 + 3 * i for i in range(128)]   # 40..421
LENS1 = [500 + i for i in range(128)]      # 500..627


def datum(seq_len, salt):
    toks = [(1000 + salt * 7919 + j * 104729) % 50000 for j in range(seq_len)]
    inp, tgt = toks[:-1], toks[1:]
    return types.Datum(
        model_input=types.ModelInput.from_ints(inp),
        loss_fn_inputs={
            "weights": types.TensorData.from_torch(torch.ones(len(inp), dtype=torch.float32)),
            "target_tokens": types.TensorData.from_torch(torch.tensor(tgt, dtype=torch.long)),
        },
    )


def check(tag, out, lens):
    lfo = out.loss_fn_outputs
    bad = 0
    first = None
    for i, (o, L) in enumerate(zip(lfo, lens)):
        lp = o["logprobs"].to_torch()
        if len(lp) != L - 1:
            bad += 1
            if first is None:
                first = (i, L - 1, len(lp))
    print(f"{tag}: n={len(lfo)} mismatches={bad}"
          + (f" first: datum {first[0]} expected {first[1]} got {first[2]}" if first else ""))
    return bad == 0


def main():
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=32)
    print("model_id:", tc.model_id, flush=True)
    ds0 = [datum(L, i) for i, L in enumerate(LENS0)]
    ds1 = [datum(L, 1000 + i) for i, L in enumerate(LENS1)]
    ap = types.AdamParams(learning_rate=2e-4)

    # Pipelined submission, cookbook-style: all four before any await.
    fb0 = tc.forward_backward(ds0, "cross_entropy")
    o0 = tc.optim_step(ap)
    fb1 = tc.forward_backward(ds1, "cross_entropy")
    o1 = tc.optim_step(ap)

    r0 = fb0.result()
    ro0 = o0.result()
    r1 = fb1.result()
    ro1 = o1.result()

    ok0 = check("fb0", r0, LENS0)
    ok1 = check("fb1", r1, LENS1)
    print("optim0 metrics:", (ro0.metrics or {}))
    print("optim1 metrics:", (ro1.metrics or {}))
    print("PIPELINED:", "PASS" if ok0 and ok1 else "FAIL")


if __name__ == "__main__":
    main()
