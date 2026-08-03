#!/usr/bin/env python3
"""G8e overlap check: 8 concurrent samples (no-lock generate path) + fb/step
submitted while samples are in flight, then a final sample (quiesce under
real overlap). Fresh model; deleted at the end."""
import json
import os
import time
os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "tml-dev-key")
import requests
import torch
import tinker
from tinker import types

BASE, KEY = os.environ["TINKER_BASE_URL"], os.environ["TINKER_API_KEY"]
PROMPT = [(1000 + j * 104729) % 50000 for j in range(30)]
OUT = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": {}}
fails = []

def check(name, ok, detail=""):
    print(f"  {name}: {'PASS' if ok else 'FAIL'} {detail}", flush=True)
    OUT["checks"][name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        fails.append(name)

def datums(n, salt=9):
    out = []
    for i in range(n):
        toks = [(1000 + (salt * 31 + i) * 7919 + j * 104729) % 50000 for j in range(64)]
        inp, tgt = toks[:-1], toks[1:]
        out.append(types.Datum(
            model_input=types.ModelInput.from_ints(inp),
            loss_fn_inputs={
                "weights": types.TensorData.from_torch(torch.ones(len(inp))),
                "target_tokens": types.TensorData.from_torch(torch.tensor(tgt, dtype=torch.long)),
            }))
    return out

sc = tinker.ServiceClient()
t0 = time.time()
tc = sc.create_lora_training_client(base_model="Qwen/Qwen2.5-0.5B", rank=32)
print(f"model {tc.model_id} (boot {time.time()-t0:.1f}s)", flush=True)
OUT["model_id"] = tc.model_id
sampler = tc.save_weights_and_get_sampling_client()

# warm-up sample (wake + first sync)
sampler.sample(prompt=types.ModelInput.from_ints(PROMPT),
               sampling_params=types.SamplingParams(temperature=1.0, max_tokens=8, seed=7),
               num_samples=1).result()
print("warm-up sample ok", flush=True)

# 8 concurrent samples, then fb+step while in flight
t0 = time.time()
sfuts = [sampler.sample(prompt=types.ModelInput.from_ints(PROMPT),
                        sampling_params=types.SamplingParams(temperature=1.0, max_tokens=64, seed=100 + i),
                        num_samples=1) for i in range(8)]
fbfut = tc.forward_backward(datums(4), "cross_entropy")
stfut = tc.optim_step(types.AdamParams(learning_rate=0.0))
print(f"submitted 8 samples + fb + step in {time.time()-t0:.2f}s", flush=True)

fb = fbfut.result()
ok_shapes = (len(fb.loss_fn_outputs) == 4 and
             all(len(o["logprobs"].data) == 63 for o in fb.loss_fn_outputs))
check("fb during in-flight samples returns correct shapes", ok_shapes,
      f"n_out={len(fb.loss_fn_outputs)}")
stfut.result()
check("optim_step(lr=0) completes", True)

sres = [f.result() for f in sfuts]
ok_s = all(r.sequences and r.sequences[0].tokens for r in sres)
check("all 8 concurrent samples return tokens", ok_s,
      f"lens={[len(r.sequences[0].tokens) for r in sres]}")

t0 = time.time()
fin = sampler.sample(prompt=types.ModelInput.from_ints(PROMPT),
                     sampling_params=types.SamplingParams(temperature=1.0, max_tokens=16, seed=7),
                     num_samples=1).result()
check("final sample after quiesce+train succeeds",
      bool(fin.sequences and fin.sequences[0].tokens), f"{time.time()-t0:.1f}s")

r = requests.post(f"{BASE}/api/v1/delete_model", json={"model_id": tc.model_id},
                  headers={"X-API-Key": KEY, "Content-Type": "application/json"})
check("delete_model", r.status_code == 200, f"http {r.status_code}")

OUT["verdict"] = "PASS" if not fails else f"FAIL: {fails}"
with open("/data/g8e_overlap_results.json", "w") as f:
    json.dump(OUT, f, indent=2)
print("G8e overlap:", OUT["verdict"], flush=True)
