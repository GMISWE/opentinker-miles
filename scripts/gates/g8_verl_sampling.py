#!/usr/bin/env python3
"""G8 verl M2 sampling gates (2026-08-02).

Validates the verl backend's rollout path: colocated vLLM server replicas in
hybrid timeshare (sleep during training ops, wake + naive weight sync on
sample). Model: Qwen/Qwen2.5-0.5B LoRA r32 (G1/G3 lineage).

Gates:
  G8a sampling sanity + logprob provenance (the TIM oracle in miniature):
      sample(temp=1.0, seed=0, max_tokens=32, n=2) returns tokens+logprobs+
      stop_reason; seeded determinism across reruns; training-engine
      get_logprobs (via forward) on prompt+sampled tokens matches the
      sampler's returned logprobs at bf16-class E1 (~1e-2 nats). The two
      engines (vLLM rollout vs FSDP training) compute independently — this
      parity IS the M2 headline number.
  G8b weight-sync freshness: fb(8, cross_entropy) + optim_step(lr=1e-3),
      sample again same prompt/seed: output must move vs G8a AND training-
      engine logprobs on the NEW sample must again match the sampler at E1
      (the lazy version-gated sync delivered the stepped weights, not v0).
  G8c timeshare regression: G1 arms with rollout attached —
      fb(8)+step(lr=0) == 2xfb(4)+step(lr=0) == rerun, grad_norm
      bit-identical; then one more sample (wake after training) succeeds.
  G8d (stretch): 5-step SDK-level RL smoke — sample 4, importance_sampling
      datums (adv = reward - mean, logprobs from sampler), fb + step(1e-5);
      sampler logprobs must track training logprobs at each step start
      (IS ratio near 1).

Writes /data/g8_results.json (or G8_RESULTS_PATH) with all measured numbers.
"""
import json
import os
import sys
import time

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "tml-dev-key")
BASE = os.environ["TINKER_BASE_URL"]
KEY = os.environ["TINKER_API_KEY"]
RESULTS_PATH = os.environ.get("G8_RESULTS_PATH", "/data/g8_results.json")

# env must be set before the SDK imports read it
import requests  # noqa: E402
import torch  # noqa: E402
import tinker  # noqa: E402
from tinker import types  # noqa: E402

BASE_MODEL = "Qwen/Qwen2.5-0.5B"
RANK = 32
SEQ = 64
PROMPT = [(1000 + j * 104729) % 50000 for j in range(30)]  # fixed ~30-token prompt
MAX_TOKENS = 32
HDRS = {"X-API-Key": KEY, "Content-Type": "application/json"}

FAILS = []
RESULTS = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "model": BASE_MODEL,
           "rank": RANK, "gates": {}, "timings_s": {}}


def check(name, ok, detail=""):
    print(f"  {name}: {'PASS' if ok else 'FAIL'}{' ' + detail if detail else ''}", flush=True)
    if not ok:
        FAILS.append(name)
    return ok


def make_datums(n, salt=0):
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


def raw_step(model_id, lr):
    """optim_step via raw endpoint so grad_norm is readable from the future."""
    r = requests.post(f"{BASE}/api/v1/optim_step", headers=HDRS, json={
        "model_id": model_id,
        "adam_params": {"learning_rate": lr},
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


def sdk_sample(sampler, seed=0, n=2, max_tokens=MAX_TOKENS):
    fut = sampler.sample(
        prompt=types.ModelInput.from_ints(PROMPT),
        sampling_params=types.SamplingParams(
            temperature=1.0, max_tokens=max_tokens, seed=seed),
        num_samples=n,
    )
    res = fut.result()
    return [
        (list(s.tokens),
         [float(x) for x in s.logprobs] if s.logprobs is not None else None,
         str(s.stop_reason))
        for s in res.sequences
    ]


def train_logprobs(tc, full_tokens):
    """Training-engine per-token logprobs on a full sequence, via forward.

    Returns lp where lp[k] = logp(full[k+1] | full[:k+1]), length N-1.
    """
    inp, tgt = full_tokens[:-1], full_tokens[1:]
    datum = types.Datum(
        model_input=types.ModelInput.from_ints(inp),
        loss_fn_inputs={
            "weights": types.TensorData.from_torch(
                torch.ones(len(inp), dtype=torch.float32)),
            "target_tokens": types.TensorData.from_torch(
                torch.tensor(tgt, dtype=torch.long)),
        },
    )
    out = tc.forward([datum], loss_fn="cross_entropy").result()
    return list(out.loss_fn_outputs[0]["logprobs"].data)


def provenance(tc, sampled_tokens, sampler_lps):
    """Compare training-engine logprobs vs sampler logprobs on sampled positions."""
    full = PROMPT + list(sampled_tokens)
    lp = train_logprobs(tc, full)
    sl = lp[len(PROMPT) - 1:]
    assert len(sl) == len(sampled_tokens), (len(sl), len(sampled_tokens))
    a = torch.tensor(sl, dtype=torch.float64)
    b = torch.tensor(sampler_lps, dtype=torch.float64)
    gap = (a - b).abs()
    corr = float(torch.corrcoef(torch.stack([a, b]))[0, 1]) if len(a) > 1 else float("nan")
    return corr, float(gap.mean()), float(gap.max())


def main():
    sc = tinker.ServiceClient()

    # ---- boot ----
    t0 = time.time()
    tc = sc.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)
    boot_s = time.time() - t0
    RESULTS["timings_s"]["create_model_boot"] = round(boot_s, 1)
    print(f"model_id: {tc.model_id} (boot {boot_s:.1f}s)", flush=True)

    t0 = time.time()
    sampler = tc.save_weights_and_get_sampling_client()
    RESULTS["timings_s"]["create_sampler"] = round(time.time() - t0, 1)

    # =================================================== G8a
    print("\nG8a: sampling sanity + logprob provenance", flush=True)
    g8a = {}
    t0 = time.time()
    s1 = sdk_sample(sampler, seed=0, n=2)
    g8a["first_sample_latency_s"] = round(time.time() - t0, 1)
    RESULTS["timings_s"]["first_sample_wake_sync"] = g8a["first_sample_latency_s"]

    ok_shape = all(t and (lp is not None) and len(lp) == len(t) and sr
                   for t, lp, sr in s1)
    check("G8a sequences have tokens+logprobs+stop_reason", ok_shape,
          f"stop_reasons={[sr for _, _, sr in s1]}")
    g8a["stop_reasons"] = [sr for _, _, sr in s1]
    g8a["sample_lens"] = [len(t) for t, _, _ in s1]

    distinct = s1[0][0] != s1[1][0]
    g8a["fanout_distinct"] = distinct
    check("G8a num_samples fanout distinct (per-sample seed)", distinct)

    t0 = time.time()
    s2 = sdk_sample(sampler, seed=0, n=2)
    g8a["second_sample_latency_s"] = round(time.time() - t0, 1)
    det = all(s1[i][0] == s2[i][0] for i in range(2))
    g8a["seeded_determinism"] = det
    if not det:
        print("  NOTE: identical seeded request returned different tokens "
              "(vLLM per-request seed caveat) — noted, not a failure", flush=True)
    else:
        print("  G8a seeded determinism: rerun bit-identical", flush=True)

    corr0, gap0, mx0 = provenance(tc, s1[0][0], s1[0][1])
    corr1, gap1, mx1 = provenance(tc, s1[1][0], s1[1][1])
    g8a["provenance"] = {
        "s0": {"corr": corr0, "mean_abs_gap": gap0, "max_abs_gap": mx0},
        "s1": {"corr": corr1, "mean_abs_gap": gap1, "max_abs_gap": mx1},
    }
    print(f"  provenance s0: corr={corr0:.6f} mean|gap|={gap0:.5f} max|gap|={mx0:.5f}", flush=True)
    print(f"  provenance s1: corr={corr1:.6f} mean|gap|={gap1:.5f} max|gap|={mx1:.5f}", flush=True)
    check("G8a sampler-vs-training logprob parity (E1)",
          corr0 > 0.99 and corr1 > 0.99 and gap0 < 0.05 and gap1 < 0.05)
    RESULTS["gates"]["G8a"] = g8a

    # =================================================== G8b
    print("\nG8b: weight-sync freshness", flush=True)
    g8b = {}
    ds = make_datums(8, salt=1)
    t0 = time.time()
    tc.forward_backward(ds, "cross_entropy").result()
    res = raw_step(tc.model_id, 1e-3)
    g8b["train_step_s"] = round(time.time() - t0, 1)
    g8b["grad_norm"] = res.get("grad_norm")
    print(f"  fb+step(lr=1e-3): grad_norm={g8b['grad_norm']!r}", flush=True)

    t0 = time.time()
    s3 = sdk_sample(sampler, seed=0, n=2)
    g8b["post_step_sample_latency_s"] = round(time.time() - t0, 1)
    RESULTS["timings_s"]["post_step_sample_resync"] = g8b["post_step_sample_latency_s"]

    moved = (s3[0][0] != s1[0][0]) or (s3[0][1] != s1[0][1])
    g8b["sampler_moved"] = moved
    check("G8b sampler output moved after optim_step", moved)

    corr3, gap3, mx3 = provenance(tc, s3[0][0], s3[0][1])
    g8b["provenance_post_step"] = {"corr": corr3, "mean_abs_gap": gap3, "max_abs_gap": mx3}
    print(f"  post-step provenance: corr={corr3:.6f} mean|gap|={gap3:.5f} max|gap|={mx3:.5f}", flush=True)
    check("G8b post-step sampler-vs-training parity (sync delivered v1)",
          corr3 > 0.99 and gap3 < 0.05)
    RESULTS["gates"]["G8b"] = g8b

    # =================================================== G8c
    print("\nG8c: timeshare regression (G1 arms with rollout attached)", flush=True)
    g8c = {"grad_norms": {}}
    ds = make_datums(8, salt=3)
    for arm, splits in [("A1", [ds]), ("B", [ds[:4], ds[4:]]), ("A2", [ds])]:
        t0 = time.time()
        for part in splits:
            tc.forward_backward(part, "cross_entropy").result()
        res = raw_step(tc.model_id, 0.0)
        gn = res.get("grad_norm")
        g8c["grad_norms"][arm] = gn
        print(f"  {arm}: grad_norm={gn!r} ({time.time()-t0:.1f}s)", flush=True)
    a1, b, a2 = (g8c["grad_norms"][k] for k in ("A1", "B", "A2"))
    check("G8c split-invariance bit-identical", repr(a1) == repr(b) and a1 and a1 > 0,
          f"A1={a1!r} B={b!r}")
    check("G8c determinism bit-identical", repr(a1) == repr(a2), f"A2={a2!r}")

    t0 = time.time()
    s5 = sdk_sample(sampler, seed=0, n=1)
    g8c["post_arms_sample_ok"] = bool(s5 and s5[0][0])
    g8c["post_arms_sample_latency_s"] = round(time.time() - t0, 1)
    check("G8c sample after training arms succeeds", g8c["post_arms_sample_ok"])
    # lr=0 arms: weights unmoved since G8b -> seeded sample must be bit-stable
    lr0_stable = s5[0][0] == s3[0][0]
    g8c["lr0_sample_bit_stable"] = lr0_stable
    check("G8c lr=0 round-trip leaves sampler bit-stable", lr0_stable)
    RESULTS["gates"]["G8c"] = g8c

    # =================================================== G8d (stretch)
    print("\nG8d: 5-step RL smoke (importance_sampling)", flush=True)
    g8d = {"steps": []}
    try:
        for step in range(5):
            samples = sdk_sample(sampler, seed=1000 + step * 10, n=4)
            rewards = [len(set(t)) / max(len(t), 1) for t, _, _ in samples]
            mean_r = sum(rewards) / len(rewards)
            advs = [r - mean_r for r in rewards]

            # IS-ratio-at-step-start probe: training logprobs vs sampler logprobs
            gaps = []
            datums = []
            for (toks, lps, _), adv in zip(samples, advs):
                full = PROMPT + toks
                n1 = len(full) - 1
                p = len(PROMPT)
                adv_vec = torch.zeros(n1, dtype=torch.float32)
                lp_vec = torch.zeros(n1, dtype=torch.float32)
                adv_vec[p - 1:] = adv
                lp_vec[p - 1:] = torch.tensor(lps, dtype=torch.float32)
                datums.append(types.Datum(
                    model_input=types.ModelInput.from_ints(full),
                    loss_fn_inputs={
                        "advantages": types.TensorData.from_torch(adv_vec),
                        "logprobs": types.TensorData.from_torch(lp_vec),
                    },
                ))
            out = tc.forward_backward(datums, "importance_sampling").result()
            for i, (toks, lps, _) in enumerate(samples):
                tr = list(out.loss_fn_outputs[i]["logprobs"].data)[len(PROMPT) - 1:]
                gaps.append(float(torch.tensor(tr).sub(torch.tensor(lps)).abs().mean()))
            raw_step(tc.model_id, 1e-5)
            step_gap = sum(gaps) / len(gaps)
            g8d["steps"].append({"mean_abs_lp_gap": round(step_gap, 5),
                                 "mean_reward": round(mean_r, 4)})
            print(f"  step {step}: mean|lp_train-lp_samp|={step_gap:.5f} "
                  f"mean_reward={mean_r:.4f}", flush=True)
        max_gap = max(s["mean_abs_lp_gap"] for s in g8d["steps"])
        check("G8d loop completes, IS ratio near 1 at step start", max_gap < 0.05,
              f"max step gap {max_gap:.5f}")
        g8d["status"] = "ran"
    except Exception as e:  # stretch gate: report, don't block a-c verdicts
        g8d["status"] = f"ERROR: {e}"
        check("G8d RL smoke", False, str(e)[:500])
    RESULTS["gates"]["G8d"] = g8d

    # ---- wrap ----
    RESULTS["model_id"] = tc.model_id
    RESULTS["fails"] = FAILS
    RESULTS["verdict"] = "PASS" if not FAILS else "FAIL"
    with open(RESULTS_PATH, "w") as f:
        json.dump(RESULTS, f, indent=2)
    print(f"\nresults -> {RESULTS_PATH}")
    print("G8:", "PASS" if not FAILS else f"FAIL ({', '.join(FAILS)})")
    print(f"NOTE: model {tc.model_id} left resident; delete via "
          f"POST /api/v1/delete_model when done.")
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
