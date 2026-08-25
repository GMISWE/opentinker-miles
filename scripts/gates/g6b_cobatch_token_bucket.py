#!/usr/bin/env python3
"""G6b — generalised co-batching E0 gate (the repair's gate).

G6 asserts that a tenant's batch is bit-identical run solo vs merged with
another tenant's, but tests exactly ONE size. The 2026-08-21 attribution showed
that size sits on the safe side of a threshold G6 does not know exists:

  a tenant's gradient is bit-identical solo vs co-batched IFF both calls'
  per-rank token totals fall on the same side of T ~ 512 tokens/rank.

Co-batching adds the co-tenant's tokens to the rank, so a merge can move the
call across T. The repair refuses exactly those merges
(``TINKERCLOUD_MILES_COBATCH_E0_TOKENS``, default 512;
``converter.cobatch_preserves_token_bucket``).

This gate sweeps n ACROSS the predicted band instead of sampling one point,
and asserts two things, because either alone is trivially satisfiable:

  (A) every arm is bit-identical  -- a guard that refuses everything passes A
  (B) the arms the law says are SAFE still merge -- a guard that is switched
      off passes B

PASS requires both. Run against a pool-mode server with merging on:
  TINKERCLOUD_MILES_MULTILORA_SLOTS>0, COBATCH_MAX_SAMPLES>=2048.

  python3 g6b_cobatch_token_bucket.py                 # guard on (expect PASS)
  python3 g6b_cobatch_token_bucket.py --expect-band   # guard OFF: assert the
                                                      # band REAPPEARS (control)
"""
import argparse
import os
import sys
import time

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "tml-dev-key")
BASE = os.environ["TINKER_BASE_URL"]
KEY = os.environ["TINKER_API_KEY"]

import requests
import torch

import tinker
from tinker import types

BASE_MODEL = os.environ.get("G6B_MODEL", "Qwen/Qwen2.5-0.5B")
RANK = int(os.environ.get("G6B_RANK", "32"))
SEQ = 64                 # -> 63 tokens per datum
THRESHOLD = 512
HDRS = {"X-API-Key": KEY, "Content-Type": "application/json"}


def make_datums(n, salt, seq=SEQ):
    out = []
    for i in range(n):
        toks = [(1000 + (salt * 31 + i) * 7919 + j * 104729) % 50000 for j in range(seq)]
        inp, tgt = toks[:-1], toks[1:]
        out.append(types.Datum(
            model_input=types.ModelInput.from_ints(inp),
            loss_fn_inputs={
                "weights": types.TensorData.from_torch(torch.ones(len(inp), dtype=torch.float32)),
                "target_tokens": types.TensorData.from_torch(torch.tensor(tgt, dtype=torch.long)),
            },
        ))
    return out


def step_lr0(model_id):
    r = requests.post(f"{BASE}/api/v1/optim_step", headers=HDRS,
                      json={"model_id": model_id, "adam_params": {"learning_rate": 0.0}})
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
        print(f"STEP FAILED ({fr.status_code}): {fr.text[:400]}")
        sys.exit(2)
    sys.exit(2)


def per_rank(n, dp, tok):
    """Same arithmetic the server guard uses: pad to a dp multiple, then stride."""
    lengths = [tok] * n
    if dp > 1 and n % dp:
        lengths += [tok] * (dp - n % dp)
    return sum(lengths[0::dp])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", type=int, default=2)
    ap.add_argument("--ns", default="6,8,9,10,12,16,20,24")
    ap.add_argument("--expect-band", action="store_true",
                    help="guard OFF control: require the predicted band to FAIL")
    cli = ap.parse_args()
    tok = SEQ - 1
    ns = [int(x) for x in cli.ns.split(",")]

    sc = tinker.ServiceClient()
    tc_a = sc.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)
    tc_b = sc.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)
    print(f"A={tc_a.model_id} B={tc_b.model_id}", flush=True)
    blocker = make_datums(int(os.environ.get("G6B_BLOCKER", "2")), salt=9)
    rows, fails = [], []

    try:
        make_datums(8, salt=99)  # warm-up shapes below
        for idx, n in enumerate(ns):
            a_ds, b_ds = make_datums(n, salt=1), make_datums(n, salt=2)
            solo_t, co_t = per_rank(n, cli.dp, tok), per_rank(2 * n, cli.dp, tok)
            safe = (solo_t <= THRESHOLD) == (co_t <= THRESHOLD)

            r_solo = tc_a.forward_backward(a_ds, "cross_entropy").result()
            gn_solo = step_lr0(tc_a.model_id).get("grad_norm")

            fut_blk = tc_a.forward(blocker, "cross_entropy")
            time.sleep(float(os.environ.get("G6B_SETTLE", "0.2")))
            fut_a = tc_a.forward_backward(a_ds, "cross_entropy")
            fut_b = tc_b.forward_backward(b_ds, "cross_entropy")
            fut_blk.result()
            r_co = fut_a.result(); fut_b.result()
            merged = any("co_batched_fb" in k for k in (r_co.metrics or {}))
            gn_co = step_lr0(tc_a.model_id).get("grad_norm")
            step_lr0(tc_b.model_id)

            identical = repr(gn_solo) == repr(gn_co)
            rows.append((n, solo_t, co_t, safe, merged, identical, gn_solo, gn_co))
            if idx == 0:
                continue                        # arm 0 absorbs kernel warm-up
            if cli.expect_band:
                if safe and not identical:
                    fails.append(f"n={n} is SAFE by the law but diverged")
                if not safe and identical and merged:
                    fails.append(f"n={n} should straddle and diverge, but was identical")
            else:
                if not identical:
                    fails.append(f"n={n} NOT bit-identical ({gn_solo!r} vs {gn_co!r})")
                if safe and not merged:
                    fails.append(f"n={n} is safe but did NOT merge (guard too strict / merging off)")
    finally:
        for m in (tc_b.model_id, tc_a.model_id):
            try:
                requests.post(f"{BASE}/api/v1/delete_model", headers=HDRS,
                              json={"model_id": m}, timeout=120)
            except Exception as e:  # noqa: BLE001
                print(f"DELETE FAILED {m}: {e}")

    print(f"\n{'n':>4} {'solo/rk':>8} {'co/rk':>7} {'law':>7} {'merged':>7} {'bit-id':>7}")
    for n, s, c, safe, merged, identical, _, _ in rows:
        print(f"{n:>4} {s:>8} {c:>7} {'safe' if safe else 'STRADDLE':>8} "
              f"{str(merged):>7} {str(identical):>7}")

    merged_any = any(r[4] for r in rows[1:])
    print(f"\n(A) all bit-identical : {all(r[5] for r in rows[1:])}")
    print(f"(B) safe arms merged  : {all(r[4] for r in rows[1:] if r[3])}  "
          f"(any merge at all: {merged_any})")
    if not cli.expect_band and not merged_any:
        fails.append("no arm merged at all — the gate would pass vacuously")

    if fails:
        print("\nG6b: FAIL")
        for f in fails:
            print(f"  {f}")
        return 1
    print("\nG6b: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
