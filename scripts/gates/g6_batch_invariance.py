#!/usr/bin/env python3
"""G6 batch-invariance gate (M3 co-batching, AC1).

Server must be in pool mode with merging on:
  TINKERCLOUD_MILES_MULTILORA_SLOTS>0, TINKERCLOUD_MILES_COBATCH_MAX_SAMPLES>=16.

Same tenant data under different co-tenant mixes must produce identical
per-tenant observables — the legality condition for cross-tenant co-batching.
  P1: tenants A and B on one pool.
  P2 solo baselines: fb alone + step(lr=0) per tenant -> grad_norm + logprobs.
  P3 co-batched: a queue-blocking forward makes A's and B's fb land in the
     dispatcher's merge window (verified via the co_batched_fb metric);
     each tenant's grad_norm + per-datum logprobs must equal its solo run
     bit-identically.
  P4 asymmetric mix: A co-batched with a DIFFERENT B batch (other content,
     other size) — A's observables still bit-equal (invariance to co-tenant
     content, not just presence).
PASS = every comparison bit-identical (repr equality).
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


def make_datums(n, salt, seq=SEQ):
    out = []
    for i in range(n):
        toks = [(1000 + (salt * 31 + i) * 7919 + j * 104729) % 50000 for j in range(seq)]
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
        "model_id": model_id, "adam_params": {"learning_rate": 0.0},
    })
    r.raise_for_status()
    rid = r.json()["request_id"]
    deadline = time.time() + 600
    while time.time() < deadline:
        fr = requests.post(f"{BASE}/api/v1/retrieve_future/{rid}", headers=HDRS)
        if fr.status_code == 200:
            return fr.json()
        if fr.status_code == 408:
            time.sleep(2)
            continue
        print(f"STEP FAILED ({fr.status_code}):", fr.text[:2000])
        sys.exit(2)
    print("STEP TIMEOUT")
    sys.exit(2)


def lps_of(fb_result):
    return repr([o["logprobs"].to_torch().tolist() for o in fb_result.loss_fn_outputs])


def main():
    sc = tinker.ServiceClient()

    print("P1: two tenants on one pool", flush=True)
    tc_a = sc.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)
    tc_b = sc.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)
    print(f"  A={tc_a.model_id} B={tc_b.model_id}", flush=True)

    probe_a = make_datums(8, salt=1)
    probe_b = make_datums(8, salt=2)
    probe_b2 = make_datums(4, salt=7, seq=48)   # different content AND shape
    blocker = make_datums(1, salt=9)

    print("P2 solo baselines", flush=True)
    ra = tc_a.forward_backward(probe_a, "cross_entropy").result()
    lp_a_solo = lps_of(ra)
    gn_a_solo = step_lr0(tc_a.model_id).get("grad_norm")
    rb = tc_b.forward_backward(probe_b, "cross_entropy").result()
    lp_b_solo = lps_of(rb)
    gn_b_solo = step_lr0(tc_b.model_id).get("grad_norm")
    print(f"  gnA_solo={gn_a_solo!r} gnB_solo={gn_b_solo!r}", flush=True)

    def co_round(tag, a_ds, b_ds):
        """Blocker forward occupies the dispatcher; both fb's queue behind it
        and merge. Returns (a_result, b_result, merged_flag)."""
        fut_blk = tc_a.forward(blocker, "cross_entropy")
        fut_a = tc_a.forward_backward(a_ds, "cross_entropy")
        fut_b = tc_b.forward_backward(b_ds, "cross_entropy")
        fut_blk.result()
        res_a, res_b = fut_a.result(), fut_b.result()
        merged = any(
            "co_batched_fb" in k for k in (res_a.metrics or {})
        ) and any(
            "co_batched_fb" in k for k in (res_b.metrics or {})
        )
        check(f"{tag} merge engaged", merged,
              f"(metrics A={sorted((res_a.metrics or {}))})" if not merged else "")
        return res_a, res_b

    print("P3 co-batched: same probes, merged into one train call", flush=True)
    res_a, res_b = co_round("P3", probe_a, probe_b)
    gn_a_co = step_lr0(tc_a.model_id).get("grad_norm")
    gn_b_co = step_lr0(tc_b.model_id).get("grad_norm")
    check("P3 A grad_norm co==solo", repr(gn_a_co) == repr(gn_a_solo),
          f"{gn_a_co!r} vs {gn_a_solo!r}")
    check("P3 B grad_norm co==solo", repr(gn_b_co) == repr(gn_b_solo),
          f"{gn_b_co!r} vs {gn_b_solo!r}")
    check("P3 A logprobs co==solo", lps_of(res_a) == lp_a_solo)
    check("P3 B logprobs co==solo", lps_of(res_b) == lp_b_solo)

    print("P4 asymmetric mix: A with a different co-tenant batch", flush=True)
    res_a, _ = co_round("P4", probe_a, probe_b2)
    gn_a_mix = step_lr0(tc_a.model_id).get("grad_norm")
    step_lr0(tc_b.model_id)   # clear B's probe_b2 grads
    check("P4 A grad_norm mix==solo", repr(gn_a_mix) == repr(gn_a_solo),
          f"{gn_a_mix!r} vs {gn_a_solo!r}")
    check("P4 A logprobs mix==solo", lps_of(res_a) == lp_a_solo)

    for m in (tc_b.model_id, tc_a.model_id):
        requests.post(f"{BASE}/api/v1/delete_model", headers=HDRS, json={"model_id": m})
    print("\nG6:", "PASS" if not FAILS else f"FAIL ({', '.join(FAILS)})")
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
