"""Miles resumes from the Megatron iter_* directory a checkpoint was published
from. The iter number is miles' own save counter, so it is recorded beside the
interchange adapter at publish time and resolved through that record — both
for load_weights (the train group) and create_model(checkpoint_path) (the
builder); the legacy hash-derived name is only a fallback for old checkpoints."""
import asyncio
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tinkercloud.training.backends import checkpoint_interchange
from tinkercloud.training.backends.miles import backend as miles_backend, model_setup
from tinkercloud.training.backends.miles.backend import MilesBackend

URI = "tinker://model_a/weights/resume"


@pytest.fixture
def ckpt_base(tmp_path, monkeypatch):
    monkeypatch.setattr(checkpoint_interchange, "CHECKPOINT_BASE", str(tmp_path / "ckpt"))
    return tmp_path


def _handle():
    return SimpleNamespace(
        adapter_slot=None, lock=asyncio.Lock(), rollout_manager=None, created_from_checkpoint=False,
        args=SimpleNamespace(save="/tmp/save"), train_group=SimpleNamespace(load_checkpoint=AsyncMock()),
    )


def test_publish_records_the_native_dir(ckpt_base, monkeypatch):
    save = ckpt_base / "save"
    old, new = save / "iter_0000001" / "adapter", save / "iter_0000002" / "adapter"
    for d in (old, new):
        d.mkdir(parents=True)
    os.utime(old, (1, 1))
    monkeypatch.setattr(miles_backend, "export_hf_adapter", lambda src, dst: None)
    miles_backend._publish_native_adapter(SimpleNamespace(save=str(save)), URI)
    root = checkpoint_interchange.resolve_checkpoint_root(URI)
    assert open(os.path.join(root, model_setup.NATIVE_CHECKPOINT_POINTER)).read() == str(new.parent)
    assert model_setup.resolve_native_checkpoint(URI, "/tmp/save") == str(new.parent)


@pytest.mark.parametrize("optimizer", [False, True])
def test_load_checkpoint_uses_the_record_and_passes_the_flag(ckpt_base, optimizer):
    native = ckpt_base / "save" / "iter_0000007"
    native.mkdir(parents=True)
    model_setup.record_native_checkpoint(URI, str(native))
    h = _handle()
    asyncio.run(MilesBackend().load_checkpoint(h, URI, optimizer=optimizer))
    h.train_group.load_checkpoint.assert_awaited_once_with(str(native), load_optimizer=optimizer)
    assert h.created_from_checkpoint is True


def test_missing_native_dir_is_an_error_not_a_guess(ckpt_base):
    model_setup.record_native_checkpoint(URI, str(ckpt_base / "gone"))
    with pytest.raises(Exception, match="is gone"):
        asyncio.run(MilesBackend().load_checkpoint(_handle(), URI))


def test_legacy_checkpoint_without_record_falls_back(ckpt_base):
    assert model_setup.resolve_native_checkpoint(URI, "/tmp/save") == model_setup.parse_checkpoint_uri(URI, "/tmp/save")
