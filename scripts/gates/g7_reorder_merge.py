#!/usr/bin/env python3
"""G7 reorder-drain gate (M4).

Server must be in pool mode with merging AND reordering on:
  TINKERCLOUD_MILES_MULTILORA_SLOTS>0
  TINKERCLOUD_MILES_COBATCH_MAX_SAMPLES>=16
  TINKERCLOUD_MILES_COBATCH_REORDER=1

Pipelined recipe traffic (fb+step back-to-back per tenant) never merges
under the conservative drain: the tenant's own step closes the merge
window microseconds after its fb (measured: 0 merges across a full
2-tenant recipe arm). The reorder drain defers OTHER tenants' non-fb ops
past the window — legal because only per-tenant submission order is
contractual (isolation invariant, G4). This gate checks the relaxation:
  P1: tenants A and B on one pool.
  P2 solo baselines: fb alone + step(lr=0) per tenant -> grad_norm + logprobs.
  P3 pipelined+reordered: [fb_A, step_A, fb_B, step_B] submitted back-to-back
     behind a queue-blocking forward. Reorder drain must defer step_A and
     merge fb_A+fb_B (co_batched_fb metric on BOTH), then replay the steps.
     Logprobs must equal solo bit-identically; post-round probes (fb+lr0
     step) must reproduce the solo grad_norms bit-identically — leftover or
     mispaired grads would show as exact 2x norm multiples (pure-sum).
  P4 natural pipelined rounds: no blocker, 5 rounds of per-tenant fb+step(lr0)
     — logprobs must stay bit-stable every round; merge count reported
     (timing-dependent, not asserted).
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


def merged_flag(res):
    return any("co_batched_fb" in k for k in (res.metrics or {}))


LR0 = types.AdamParams(learning_rate=0.0)


def main():
    sc = tinker.ServiceClient()

    print("P1: two tenants on one pool", flush=True)
    tc_a = sc.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)
    tc_b = sc.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)
    print(f"  A={tc_a.model_id} B={tc_b.model_id}", flush=True)

    probe_a = make_datums(8, salt=1)
    probe_b = make_datums(8, salt=2)
    blocker = make_datums(2, salt=9)   # >= DP size (known gap, 003)

    print("P2 solo baselines", flush=True)
    ra = tc_a.forward_backward(probe_a, "cross_entropy").result()
    lp_a_solo = lps_of(ra)
    gn_a_solo = step_lr0(tc_a.model_id).get("grad_norm")
    rb = tc_b.forward_backward(probe_b, "cross_entropy").result()
    lp_b_solo = lps_of(rb)
    gn_b_solo = step_lr0(tc_b.model_id).get("grad_norm")
    print(f"  gnA_solo={gn_a_solo!r} gnB_solo={gn_b_solo!r}", flush=True)

    print("P3 pipelined fb+step per tenant, behind a blocker", flush=True)
    fut_blk = tc_a.forward(blocker, "cross_entropy")
    fut_a = tc_a.forward_backward(probe_a, "cross_entropy")
    fut_sa = tc_a.optim_step(LR0)
    fut_b = tc_b.forward_backward(probe_b, "cross_entropy")
    fut_sb = tc_b.optim_step(LR0)
    fut_blk.result()
    res_a, res_b = fut_a.result(), fut_b.result()
    fut_sa.result(), fut_sb.result()
    check("P3 merge engaged (A)", merged_flag(res_a),
          f"(metrics A={sorted((res_a.metrics or {}))})")
    check("P3 merge engaged (B)", merged_flag(res_b))
    check("P3 A logprobs pipelined==solo", lps_of(res_a) == lp_a_solo)
    check("P3 B logprobs pipelined==solo", lps_of(res_b) == lp_b_solo)
    # Post-round state probes: the deferred steps must have consumed exactly
    # their own fb's grads (mispairing shows as 2x probe norms).
    tc_a.forward_backward(probe_a, "cross_entropy").result()
    gn_a_post = step_lr0(tc_a.model_id).get("grad_norm")
    tc_b.forward_backward(probe_b, "cross_entropy").result()
    gn_b_post = step_lr0(tc_b.model_id).get("grad_norm")
    check("P3 A probe grad_norm post==solo", repr(gn_a_post) == repr(gn_a_solo),
          f"{gn_a_post!r} vs {gn_a_solo!r}")
    check("P3 B probe grad_norm post==solo", repr(gn_b_post) == repr(gn_b_solo),
          f"{gn_b_post!r} vs {gn_b_solo!r}")

    print("P4 natural pipelined rounds (no blocker), 5x", flush=True)
    merges = 0
    for rnd in range(5):
        fut_a = tc_a.forward_backward(probe_a, "cross_entropy")
        fut_sa = tc_a.optim_step(LR0)
        fut_b = tc_b.forward_backward(probe_b, "cross_entropy")
        fut_sb = tc_b.optim_step(LR0)
        res_a, res_b = fut_a.result(), fut_b.result()
        fut_sa.result(), fut_sb.result()
        merges += int(merged_flag(res_a) and merged_flag(res_b))
        check(f"P4 r{rnd} A logprobs stable", lps_of(res_a) == lp_a_solo)
        check(f"P4 r{rnd} B logprobs stable", lps_of(res_b) == lp_b_solo)
    print(f"  natural-traffic merge rounds: {merges}/5 (informational)", flush=True)

    for m in (tc_b.model_id, tc_a.model_id):
        requests.post(f"{BASE}/api/v1/delete_model", headers=HDRS, json={"model_id": m})
    print("\nG7:", "PASS" if not FAILS else f"FAIL ({', '.join(FAILS)})")
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
