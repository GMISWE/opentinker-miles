#!/usr/bin/env python3
"""G1 seam parity gate, v3 (2026-07-30 rewrite; v2 harness lost with the pod).

Gate (specs/005 design.md): N x forward_backward + 1 x optim_step must equal
1 x forward_backward(same total data) + 1 x optim_step -- grad_norm ratio
EXACTLY 1.0 under the pure-sum contract (_loss_norm_total=1).

Method: all optim steps use lr=0.0 (honored: seam checks `is not None`), so
weights never move and every arm sees identical parameters. Arms:
  A1: fb(8)        + step   -> grad_norm
  B : fb(4), fb(4) + step   -> grad_norm   (accumulation arm)
  A2: fb(8)        + step   -> grad_norm   (determinism check, == A1)
PASS = A1 == B == A2 bit-identical (repr equality on the float).
"""
import os, sys, time, json

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "tml-dev-key")
BASE = os.environ["TINKER_BASE_URL"]
KEY = os.environ["TINKER_API_KEY"]

import requests
import torch
import tinker
from tinker import types

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
RANK = 32
SEQ = 64
N = 8
HDRS = {"X-API-Key": KEY, "Content-Type": "application/json"}


def make_datums():
    out = []
    for i in range(N):
        toks = [(1000 + i * 7919 + j * 104729) % 50000 for j in range(SEQ)]
        inp, tgt = toks[:-1], toks[1:]
        out.append(types.Datum(
            model_input=types.ModelInput.from_ints(inp),
            loss_fn_inputs={
                "weights": types.TensorData.from_torch(
                    torch.ones(len(inp), dtype=torch.float32)),
                "target_tokens": types.TensorData.from_torch(
                    torch.tensor(tgt, dtype=torch.long)),
            },
        ))
    return out


def step_lr0(model_id):
    r = requests.post(f"{BASE}/api/v1/optim_step", headers=HDRS, json={
        "model_id": model_id,
        "adam_params": {"learning_rate": 0.0},
    })
    r.raise_for_status()
    rid = r.json()["request_id"]
    deadline = time.time() + 600
    while time.time() < deadline:
        fr = requests.post(f"{BASE}/api/v1/retrieve_future/{rid}", headers=HDRS)
        if fr.status_code == 200:
            return fr.json()          # raw backend result: {success, grad_norm}
        if fr.status_code == 408:     # still pending
            time.sleep(2)
            continue
        print(f"STEP FAILED ({fr.status_code}):", fr.text[:2000])
        sys.exit(2)
    print("STEP TIMEOUT")
    sys.exit(2)


def main():
    sc = tinker.ServiceClient()
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)
    model_id = tc.model_id
    print("model_id:", model_id, flush=True)
    ds = make_datums()

    grad_norms = {}
    for arm, splits in [("A1", [ds]), ("B", [ds[:4], ds[4:]]), ("A2", [ds])]:
        t0 = time.time()
        for part in splits:
            fut = tc.forward_backward(part, "cross_entropy")
            fut.result()
        res = step_lr0(model_id)
        gn = res.get("grad_norm")
        grad_norms[arm] = gn
        print(f"{arm}: grad_norm={gn!r} success={res.get('success')} "
              f"({time.time()-t0:.1f}s)", flush=True)

    a1, b, a2 = grad_norms["A1"], grad_norms["B"], grad_norms["A2"]
    ratio = b / a1 if a1 else float("nan")
    det = "BIT-IDENTICAL" if repr(a1) == repr(a2) else f"DRIFT ({a1!r} vs {a2!r})"
    split = "BIT-IDENTICAL" if repr(a1) == repr(b) else f"ratio={ratio!r}"
    print(f"\nG1 split-invariance : {split}")
    print(f"G1 determinism (A2) : {det}")
    ok = repr(a1) == repr(b) == repr(a2) and a1 and a1 > 0
    print("G1:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
