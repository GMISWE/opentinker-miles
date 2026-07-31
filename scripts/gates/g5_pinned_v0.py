#!/usr/bin/env python3
"""G5 pinned-v0 + sampler routing gate (M2 leftover, 003 R6 second half).

Server must be in pool mode (TINKERCLOUD_MILES_MULTILORA_SLOTS>0).
Phases:
  P1: tenant A on pool; sampler saved at v0; greedy sample S0.
  P2: A trains 3 real steps (adapter moves, upserted each step); the
      pinned sampler re-samples S1 == S0 bit-identical — v0 routes to
      BASE weights, the frozen-reference property (DPO ref for free).
  P3: live sample of A (model_path routing, no session) — logprobs must
      DIFFER from S0 (the adapter actually moved; also proves the pinned
      match wasn't vacuous).
  P4: tenant B joins (fresh slot, zero delta); live sample of B ==
      S0 bit-identical (zero-delta ≡ base) while A's live differs —
      per-tenant sampler routing (find-first would have collapsed them).
PASS = every comparison as stated (repr equality on token/logprob lists).
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
PROMPT = [(1000 + j * 104729) % 50000 for j in range(24)]
HDRS = {"X-API-Key": KEY, "Content-Type": "application/json"}
GREEDY = {"temperature": 0.0, "max_tokens": 24, "top_p": 1.0, "top_k": -1}
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


def sdk_sample(sampler):
    fut = sampler.sample(
        prompt=types.ModelInput.from_ints(PROMPT),
        sampling_params=types.SamplingParams(temperature=0.0, max_tokens=24),
        num_samples=1,
    )
    seq = fut.result().sequences[0]
    return repr(list(seq.tokens)), repr([float(x) for x in seq.logprobs])


def raw_live_sample(model_id):
    """Live-engine sample routed to model_id via its tinker:// path."""
    r = requests.post(f"{BASE}/api/v1/asample", headers=HDRS, json={
        "num_samples": 1,
        "prompt": {"tokens": PROMPT},
        "sampling_params": GREEDY,
        "model_path": f"tinker://{model_id}/weights/live",
    })
    r.raise_for_status()
    res = await_future(r.json()["request_id"])
    seq = res["sequences"][0]
    return repr(list(seq["tokens"])), repr([float(x) for x in seq["logprobs"]])


def main():
    sc = tinker.ServiceClient()

    print("P1: tenant A + v0 sampler", flush=True)
    tc_a = sc.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)
    print(f"  A={tc_a.model_id}", flush=True)
    # MUST be the ephemeral (unnamed) flavor: only that path registers the
    # sampler session with a pinned weight_version — a named
    # save_weights_for_sampler + create_sampling_client(path) yields an
    # UNPINNED sampler served live (cost G5 run 1).
    pinned = tc_a.save_weights_and_get_sampling_client()
    s0_toks, s0_lps = sdk_sample(pinned)
    print(f"  S0 tokens={s0_toks[:60]}...", flush=True)

    print("P2: A trains 3 real steps; pinned sampler must not move", flush=True)
    train_a = make_datums(8, salt=3)
    for i in range(3):
        tc_a.forward_backward(train_a, "cross_entropy").result()
        tc_a.optim_step(types.AdamParams(learning_rate=2e-4)).result()
    s1_toks, s1_lps = sdk_sample(pinned)
    check("P2 pinned tokens bit-stable", s1_toks == s0_toks)
    check("P2 pinned logprobs bit-stable", s1_lps == s0_lps)

    print("P3: A's LIVE sample must differ from base", flush=True)
    a_toks, a_lps = raw_live_sample(tc_a.model_id)
    check("P3 A live != base", a_lps != s0_lps,
          "(logprobs identical — adapter never moved or route ignored)" if a_lps == s0_lps else "")

    print("P4: tenant B (fresh) live == base; routing is per-tenant", flush=True)
    tc_b = sc.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)
    print(f"  B={tc_b.model_id}", flush=True)
    b_toks, b_lps = raw_live_sample(tc_b.model_id)
    check("P4 B live == base (zero delta)", b_lps == s0_lps and b_toks == s0_toks)
    check("P4 A/B routes distinct", a_lps != b_lps)

    for m in (tc_b.model_id, tc_a.model_id):
        requests.post(f"{BASE}/api/v1/delete_model", headers=HDRS, json={"model_id": m})
    print("\nG5:", "PASS" if not FAILS else f"FAIL ({', '.join(FAILS)})")
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
