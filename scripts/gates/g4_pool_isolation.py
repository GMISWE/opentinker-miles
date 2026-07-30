#!/usr/bin/env python3
"""G4 pool isolation gate (M2): two tenants on ONE multi-LoRA pool; tenant A
training must not disturb tenant B's observables.

Server must be booted in pool mode (TINKERCLOUD_MILES_MULTILORA_SLOTS>0).
Phases, all through the public API:
  P1 join      : create A then B — B's create must JOIN A's pool (no boot)
  P2 isolation : B's probe logprobs + grad_norm bit-stable across A's REAL
                 steps (the BUG-015 pinned-probe monitor, cross-tenant)
  P3 interleave: fb(A),fb(B),step(A),fb(B),step(B) pipelined == serialized
                 per-tenant grad_norms (R2(ii) through the API; lr=0)
  P4 survive   : delete B; A's probe grad_norm still bit-equal (slot cleanup
                 does not disturb the co-tenant)
PASS = every comparison bit-identical (repr equality on floats/lists).
"""
import os, sys, time

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
HDRS = {"X-API-Key": KEY, "Content-Type": "application/json"}
FAILS = []


def check(name, ok, detail=""):
    print(f"  {name}: {'PASS' if ok else 'FAIL'}{' ' + detail if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)


def make_datums(n, salt):
    out = []
    for i in range(n):
        toks = [(1000 + (salt * 31 + i) * 7919 + j * 104729) % 50000 for j in range(SEQ)]
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


def submit_step(model_id, lr):
    r = requests.post(f"{BASE}/api/v1/optim_step", headers=HDRS, json={
        "model_id": model_id,
        "adam_params": {"learning_rate": lr},
    })
    r.raise_for_status()
    return r.json()["request_id"]


def await_future(rid, timeout=600):
    deadline = time.time() + timeout
    while time.time() < deadline:
        fr = requests.post(f"{BASE}/api/v1/retrieve_future/{rid}", headers=HDRS)
        if fr.status_code == 200:
            return fr.json()
        if fr.status_code == 408:
            time.sleep(2)
            continue
        print(f"FUTURE FAILED ({fr.status_code}):", fr.text[:2000])
        sys.exit(2)
    print("FUTURE TIMEOUT")
    sys.exit(2)


def step(model_id, lr):
    return await_future(submit_step(model_id, lr))


def probe(tc, model_id, ds):
    """fb(probe) + step(lr=0): returns (logprobs repr, grad_norm). lr=0 moves
    nothing and clears the slot's accumulated probe grads."""
    out = tc.forward_backward(ds, "cross_entropy").result()
    lps = [o["logprobs"].to_torch().tolist() for o in out.loss_fn_outputs]
    gn = step(model_id, 0.0).get("grad_norm")
    return repr(lps), gn


def main():
    sc = tinker.ServiceClient()

    print("P1 join: creating tenant A (pool boot)...", flush=True)
    t0 = time.time()
    tc_a = sc.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)
    boot_s = time.time() - t0
    print(f"  A={tc_a.model_id} ({boot_s:.0f}s)", flush=True)
    t0 = time.time()
    tc_b = sc.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)
    join_s = time.time() - t0
    print(f"  B={tc_b.model_id} ({join_s:.0f}s)", flush=True)
    # A join must not reboot the rails: order-of-magnitude faster than boot.
    check("P1 B joined (no second boot)", join_s < max(60.0, boot_s / 3),
          f"boot={boot_s:.0f}s join={join_s:.0f}s")

    probe_b = make_datums(4, salt=2)
    probe_a = make_datums(4, salt=1)
    train_a = make_datums(8, salt=3)

    print("P2 isolation: B pinned probe across A's real steps", flush=True)
    lp_b0, gn_b0 = probe(tc_b, tc_b.model_id, probe_b)
    print(f"  B baseline grad_norm={gn_b0!r}", flush=True)
    for i in range(3):
        tc_a.forward_backward(train_a, "cross_entropy").result()
        res = step(tc_a.model_id, 2e-4)
        print(f"  A real step {i}: grad_norm={res.get('grad_norm')!r}", flush=True)
    lp_b1, gn_b1 = probe(tc_b, tc_b.model_id, probe_b)
    check("P2 B logprobs bit-stable", lp_b0 == lp_b1)
    check("P2 B grad_norm bit-stable", repr(gn_b0) == repr(gn_b1),
          f"{gn_b0!r} vs {gn_b1!r}")

    print("P3 interleave: pipelined cross-tenant == serialized (lr=0)", flush=True)
    # Interleaved: ALL ops via the SDK — its per-client _take_turn preserves
    # submission order; mixing raw-HTTP steps with SDK fb races (the raw
    # step can reach the server first and consume an empty slot; observed
    # 2026-07-30: grad_norm 0.0 + exact 2x/4x leftover-grad multiples).
    # Verdict comes from post-round state probes: an out-of-order step
    # leaves unconsumed grads and the next probe's grad_norm doubles.
    lr0 = types.AdamParams(learning_rate=0.0)
    fut_a = tc_a.forward_backward(probe_a, "cross_entropy")
    fut_b1 = tc_b.forward_backward(probe_b, "cross_entropy")
    fut_sa = tc_a.optim_step(lr0)
    fut_b2 = tc_b.forward_backward(probe_b, "cross_entropy")
    fut_sb = tc_b.optim_step(lr0)
    for f in (fut_a, fut_b1, fut_sa, fut_b2, fut_sb):
        f.result()
    _, gn_a_i = probe(tc_a, tc_a.model_id, probe_a)
    _, gn_b_i = probe(tc_b, tc_b.model_id, probe_b)
    # Serialized per-tenant, same ops fully awaited (weights static: lr=0).
    tc_a.forward_backward(probe_a, "cross_entropy").result()
    step(tc_a.model_id, 0.0)
    tc_b.forward_backward(probe_b, "cross_entropy").result()
    tc_b.forward_backward(probe_b, "cross_entropy").result()
    step(tc_b.model_id, 0.0)
    _, gn_a_s = probe(tc_a, tc_a.model_id, probe_a)
    _, gn_b_s = probe(tc_b, tc_b.model_id, probe_b)
    check("P3 A interleaved == serialized", repr(gn_a_i) == repr(gn_a_s),
          f"{gn_a_i!r} vs {gn_a_s!r}")
    check("P3 B interleaved == serialized", repr(gn_b_i) == repr(gn_b_s),
          f"{gn_b_i!r} vs {gn_b_s!r}")

    print("P4 survive: delete B, A unchanged", flush=True)
    r = requests.post(f"{BASE}/api/v1/delete_model", headers=HDRS,
                      json={"model_id": tc_b.model_id})
    check("P4 delete B", r.status_code == 200, f"status={r.status_code}")
    _, gn_a_post = probe(tc_a, tc_a.model_id, probe_a)
    check("P4 A grad_norm after B's exit", repr(gn_a_post) == repr(gn_a_s),
          f"{gn_a_post!r} vs {gn_a_s!r}")

    requests.post(f"{BASE}/api/v1/delete_model", headers=HDRS,
                  json={"model_id": tc_a.model_id})
    print("\nG4:", "PASS" if not FAILS else f"FAIL ({', '.join(FAILS)})")
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
