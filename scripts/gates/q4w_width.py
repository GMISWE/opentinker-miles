#!/usr/bin/env python3
"""Q4-W: price merge WIDTH directly (closed-loop clients expose a merge
frontier of ~2; this measures what width w in {1,2,4,8} would buy if a scheduler exposed it).

Server must be pool mode, merging on, guard OFF (we price the move itself):
  TINKERCLOUD_MILES_MULTILORA_SLOTS=8  TINKERCLOUD_MILES_TRAIN_GPUS=2
  TINKERCLOUD_MILES_COBATCH_MAX_SAMPLES=2048
  TINKERCLOUD_MILES_COBATCH_REORDER=1  TINKERCLOUD_MILES_COBATCH_E0_TOKENS=0

Method (G7's blocker pattern, all SDK-submitted): per round, submit a fat
forward as a queue blocker, then w fb's (8 datums x seq 128 each — the Q4-B
S workload, ~1016 tok/call) from w distinct tenants back-to-back; they queue
behind the blocker and the drain merges all w into one call. T(w) = wall
from blocker completion to all fb results. The blocker term cancels because
w=1 is measured the same way. Rounds that fail to reach the target width
are discarded (achieved width read from the co_batched_fb result metric).

Emits one JSON line per arm and a summary.
"""

import json
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

BASE_MODEL = os.environ.get("Q4W_MODEL", "Qwen/Qwen2.5-0.5B")
RANK = int(os.environ.get("Q4W_RANK", "32"))
SEQ = 128          # datum length; input len SEQ-1 -> 8x127 = 1016 tok/call
N_DATUMS = 8
N_TENANTS = 8
WIDTHS = [int(w) for w in os.environ.get("Q4W_WIDTHS", "1,2,4,8").split(",")]
ROUNDS = int(os.environ.get("Q4W_ROUNDS", "14"))
WARM = 2
HDRS = {"X-API-Key": KEY, "Content-Type": "application/json"}
LR0 = types.AdamParams(learning_rate=0.0)


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


def achieved_width(results):
    return sum(1 for r in results if any("co_batched_fb" in k for k in (r.metrics or {})))


def main():
    sc = tinker.ServiceClient()
    print(f"creating {N_TENANTS} tenants", flush=True)
    tcs = [sc.create_lora_training_client(base_model=BASE_MODEL, rank=RANK)
           for _ in range(N_TENANTS)]
    for i, tc in enumerate(tcs):
        print(f"  t{i}={tc.model_id}", flush=True)

    probes = [make_datums(N_DATUMS, salt=10 + i) for i in range(N_TENANTS)]
    blocker = make_datums(256, salt=99)

    summary = {}
    for w in WIDTHS:
        walls, widths_seen, discarded = [], [], 0
        for rnd in range(ROUNDS):
            fut_blk = tcs[0].forward(blocker, "cross_entropy")
            time.sleep(0.05)  # let the blocker reach the drain head
            futs = [tcs[i].forward_backward(probes[i], "cross_entropy")
                    for i in range(w)]
            t_submitted = time.time()
            fut_blk.result()
            t_blk = time.time()
            results = [f.result() for f in futs]
            t_end = time.time()
            aw = achieved_width(results) if w > 1 else 1
            widths_seen.append(aw)
            # grads must not accumulate across rounds; await so no step
            # bleeds into the next round's timing window
            for f in [tcs[i].optim_step(LR0) for i in range(w)]:
                f.result()
            ok = (w == 1 or aw == w) and t_submitted < t_blk
            if rnd < WARM or not ok:
                discarded += int(rnd >= WARM)
                continue
            walls.append(t_end - t_blk)
        walls.sort()
        med = walls[len(walls) // 2] if walls else None
        summary[w] = {
            "median_s": med, "n": len(walls), "discarded": discarded,
            "min_s": walls[0] if walls else None,
            "widths_seen": widths_seen,
        }
        print(json.dumps({"width": w, **summary[w]}), flush=True)

    t1 = summary.get(1, {}).get("median_s")
    print("\nwidth  T(w) med    per-call    vs w x T(1)")
    for w in WIDTHS:
        med = summary[w]["median_s"]
        if med is None:
            print(f"{w:>5}  (no valid rounds)")
            continue
        line = f"{w:>5}  {med*1000:8.1f}ms  {med/w*1000:8.1f}ms"
        if t1 and w > 1:
            line += f"  {med/(w*t1):8.3f}x"
        print(line)

    for tc in tcs:
        requests.post(f"{BASE}/api/v1/delete_model", headers=HDRS,
                      json={"model_id": tc.model_id})
    print("MODELS_DELETED", flush=True)
    print("Q4W_SUMMARY " + json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
