#!/usr/bin/env python3
"""Q4-B probe tenant: parameterized sl_basic (NoRobots, Qwen2.5-0.5B r32).

One tenant of the merge-pricing probe: the overhead-fraction axis is set by Q4B_BATCH x Q4B_MAXLEN (per-call tokens).
The dp2 E0-guard bucket boundary is batch*(maxlen-1) <= 1024 tokens/call
(<=512/rank): batch 8 x maxlen 128 = 1016 sits deterministically in the low
bucket, so under the guard every cross-tenant merge is refused; batch 128 x
maxlen 2048 sits far above it, so every merge is admitted.
"""

import asyncio
import os

os.environ.setdefault("TINKER_BASE_URL", "http://localhost:8000")
os.environ.setdefault("TINKER_API_KEY", "tml-dev-key")

from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

MODEL = os.environ.get("Q4B_MODEL", "Qwen/Qwen2.5-0.5B")
BATCH = int(os.environ.get("Q4B_BATCH", "8"))
MAXLEN = int(os.environ.get("Q4B_MAXLEN", "128"))
STEPS = int(os.environ.get("Q4B_STEPS", "30"))
LOG_PATH = os.environ["Q4B_LOG_PATH"]

config = train.Config(
    log_path=LOG_PATH,
    model_name=MODEL,
    renderer_name="qwen3",
    dataset_builder=chat_datasets.NoRobotsBuilder(
        common_config=ChatDatasetBuilderCommonConfig(
            model_name_for_tokenizer=MODEL,
            renderer_name="qwen3",
            max_length=MAXLEN,
            batch_size=BATCH,
            train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
        )
    ),
    learning_rate=2e-4,
    lr_schedule="linear",
    num_epochs=1,
    max_steps=STEPS,
    lora_rank=32,
    eval_every=0,
    save_every=0,
)

if __name__ == "__main__":
    asyncio.run(train.main(config))
