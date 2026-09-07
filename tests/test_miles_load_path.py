"""load_weights on Miles hands the actors a Megatron --load directory, resolved
from the tinker:// URI the same way create_model(checkpoint_path) resolves it;
the raw URI used to reach the actors and fail their directory check."""
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tinkercloud.training.backends.miles.backend import MilesBackend
from tinkercloud.training.backends.miles.model_setup import parse_checkpoint_uri


@pytest.mark.parametrize("optimizer", [False, True])
def test_load_checkpoint_resolves_uri_and_passes_the_flag(optimizer):
    backend = MilesBackend()
    h = SimpleNamespace(
        adapter_slot=None, lock=asyncio.Lock(), rollout_manager=None, created_from_checkpoint=False,
        args=SimpleNamespace(save="/tmp/save"), train_group=SimpleNamespace(load_checkpoint=AsyncMock()),
    )
    asyncio.run(backend.load_checkpoint(h, "tinker://model_a/weights/resume", optimizer=optimizer))
    h.train_group.load_checkpoint.assert_awaited_once_with(
        parse_checkpoint_uri("tinker://model_a/weights/resume", "/tmp/save"), load_optimizer=optimizer)
    assert h.created_from_checkpoint is True
