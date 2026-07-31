#!/usr/bin/env python3
"""Q4 P-mix RL tenant (M4).

rl_basic-equivalent GSM8K RL against the local TinkerCloud miles pool:
Qwen2.5-0.5B LoRA r32, 16 groups x 8 = 128 samples/batch, max_tokens 256,
lr 4e-5 (rl_basic default), importance_sampling loss. Sized so each fb
matches the SFT tenants' 128-datum batches. Phase structure is the point:
rollout generation occupies the shared SGLang engines while the train
GPUs idle -- the bubble a co-tenant's SFT training fills (P-mix).

Env: Q4RL_LOG_PATH (default /data/q4_rl), Q4RL_BATCHES (default 12).
"""
import asyncio
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "tml-dev-key")

from tinker_cookbook.recipes.math_rl.math_env import Gsm8kDatasetBuilder
from tinker_cookbook.rl import train

MODEL = "Qwen/Qwen2.5-0.5B"
RENDERER = "qwen3"
LOG_PATH = os.environ.get("Q4RL_LOG_PATH", "/data/q4_rl")
BATCHES = int(os.environ.get("Q4RL_BATCHES", "12"))

config = train.Config(
    log_path=LOG_PATH,
    model_name=MODEL,
    renderer_name=RENDERER,
    dataset_builder=Gsm8kDatasetBuilder(
        batch_size=16,
        group_size=8,
        renderer_name=RENDERER,
        model_name_for_tokenizer=MODEL,
    ),
    learning_rate=4e-5,
    max_tokens=256,
    lora_rank=32,
    max_steps=BATCHES,
    eval_every=0,
    save_every=0,
)

if __name__ == "__main__":
    asyncio.run(train.main(config))
