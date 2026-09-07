"""create_model(checkpoint_path) on Miles is a weights-only load: the builder
must pair Megatron's --load with --no-load-optim / --no-load-rng / --finetune,
or the resume silently restores optimizer state (a full resume is
load_weights(optimizer=true), which goes through the train group instead)."""
from argparse import Namespace

import pytest

from test_miles_pack_length import MODEL_CONFIG, _load_builder_module


@pytest.fixture(scope="module")
def builder_mod():
    return _load_builder_module()


def _configure(builder_mod, checkpoint_path):
    b = builder_mod.MilesArgumentBuilder(default_save_dir="/tmp/ckpt")
    return b._configure_model_args(
        Namespace(), base_model="/tmp/model", megatron_checkpoint_path="/tmp/mcore",
        lora_config={"rank": 8}, debug_train_only=False, checkpoint_path=checkpoint_path,
        model_config=MODEL_CONFIG, parallel_config={"tp": 1, "pp": 1, "cp": 1},
    )


def test_checkpoint_path_at_create_is_weights_only(builder_mod):
    args = _configure(builder_mod, "/tmp/ckpt/run/weights/x")
    assert args.load == "/tmp/ckpt/run/weights/x"
    assert args.no_load_optim is True and args.no_load_rng is True and args.finetune is True


def test_no_checkpoint_path_sets_no_load(builder_mod):
    args = _configure(builder_mod, None)
    assert not getattr(args, "load", None)
