#!/usr/bin/env python3
"""G3 data probe v2: use set_epoch(seed=0) like the training loop (shuffled
batch 0 = the batch that crashes), submit fb, find mismatched datums, and
introspect their ModelInput chunk structure."""
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "tml-dev-key")

import tinker
from tinker import types
from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

MODEL = "Qwen/Qwen2.5-0.5B"

builder = chat_datasets.NoRobotsBuilder(
    common_config=ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=MODEL,
        renderer_name="qwen3",
        max_length=8192,
        batch_size=128,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
)
dataset, _ = builder()
dataset.set_epoch(seed=0)  # exactly what train.main does before get_batch(0)
data = dataset.get_batch(0)

# Client-side shape audit BEFORE submitting
multi_chunk = []
for i, d in enumerate(data):
    chunks = d.model_input.chunks
    wl = len(d.loss_fn_inputs["weights"].data)
    il = d.model_input.length
    if len(chunks) != 1 or wl != il:
        multi_chunk.append((i, len(chunks), il, wl, [type(c).__name__ for c in chunks]))
print(f"batch: {len(data)} datums; irregular datums (chunks!=1 or weights!=input len): {len(multi_chunk)}")
for row in multi_chunk[:10]:
    print("  ", row)

sc = tinker.ServiceClient()
tc = sc.create_lora_training_client(base_model=MODEL, rank=32)
print("model_id:", tc.model_id, flush=True)

out = tc.forward_backward(data, "cross_entropy").result()
lfo = out.loss_fn_outputs
wlens = [len(d.loss_fn_inputs["weights"].data) for d in data]
got = [len(o["logprobs"].to_torch()) for o in lfo]
bad = [(i, wlens[i], got[i]) for i in range(min(len(got), len(wlens))) if got[i] != wlens[i]]
print(f"outputs: {len(lfo)}; mismatches: {len(bad)}")
for i, exp, g in bad[:10]:
    d = data[i]
    chunks = d.model_input.chunks
    print(f"  datum {i}: expected {exp}, got {g}; input_len={d.model_input.length} "
          f"chunks={len(chunks)} chunk_lens={[c.length for c in chunks]}")
print("G3-DATA2:", "PASS" if not bad else "FAIL")
