#!/usr/bin/env python3
"""G2 order probe (2026-07-30): fb with distinct-length datums; verify each
returned per-sample logprob vector has its own datum's length (client order).
Diagnoses the G3 failure (logprobs[218] paired with weights[432])."""
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "tml-dev-key")

import torch
import tinker
from tinker import types

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
LENS = [40, 55, 70, 85, 100, 115, 130, 145]  # distinct per-datum seq lens


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


def main():
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=32)
    print("model_id:", tc.model_id, flush=True)
    ds = [datum(L, i) for i, L in enumerate(LENS)]
    out = tc.forward_backward(ds, "cross_entropy").result()
    lfo = out.loss_fn_outputs
    print(f"num outputs: {len(lfo)} (expected {len(LENS)})")
    ok = True
    for i, (o, L) in enumerate(zip(lfo, LENS)):
        lp = o["logprobs"].to_torch()
        exp = L - 1  # weights length
        match = "OK" if len(lp) == exp else "MISMATCH"
        if len(lp) != exp:
            ok = False
        print(f"  datum {i}: expected {exp}, got {len(lp)}  {match}")
    print("G2 order:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
