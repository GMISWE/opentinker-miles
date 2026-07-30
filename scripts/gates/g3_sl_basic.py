#!/usr/bin/env python3
"""G3 gate, v2 (2026-07-30 rewrite; v1 harness lost with the pod).

sl_basic-equivalent 30-step SFT run against the local TinkerCloud miles
backend: NoRobots chat data, Qwen2.5-0.5B, LoRA r32, lr 2e-4 linear
(sl_basic recipe hyperparams; model swapped to the 005 gate model).
PASS = run completes, grad_norm nonzero, train NLL decreases.
005 reference on 20260715 base: NLL 2.74 -> 2.34 over 30 steps (different
harness lineage -- trend is the gate, exact values re-baselined here).
"""
import asyncio
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "tml-dev-key")

from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

MODEL = "Qwen/Qwen2.5-0.5B"
# Env-overridable so two instances can run concurrently against a pool-mode
# server (the M2 per-tenant E2 gate).
LOG_PATH = os.environ.get("G3_LOG_PATH", "/data/g3_sl_basic_20260730")

config = train.Config(
    log_path=LOG_PATH,
    model_name=MODEL,
    renderer_name="qwen3",
    dataset_builder=chat_datasets.NoRobotsBuilder(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=MODEL,
            renderer_name="qwen3",
            max_length=8192,
            batch_size=128,
            train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )
    ),
    learning_rate=2e-4,
    lr_schedule="linear",
    num_epochs=1,
    max_steps=30,
    lora_rank=32,
    eval_every=0,
    save_every=0,
)

if __name__ == "__main__":
    asyncio.run(train.main(config))
