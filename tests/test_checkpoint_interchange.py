"""Unit tests for the cross-backend HF PEFT adapter interchange.

Filesystem-level contract only (no engine): URI resolution, publish/stage
round-trip, atomicity of publish, PEFT key normalization, and the "native
artifact wins" rule. Design: specs/007-q5-migration/HANDOFF.md §2.1.
"""
import json
import os

import pytest
import torch
from safetensors.torch import load_file, save_file

from tinkercloud.training.backends import checkpoint_interchange as ci

BARE_KEY = "model.layers.0.self_attn.q_proj.lora_A.weight"


def _write_adapter(d, r=32, alpha=32, key=BARE_KEY, value=1.0):
    """A minimal but REAL adapter: the publish path parses safetensors."""
    os.makedirs(d, exist_ok=True)
    save_file(
        {key: torch.full((2, 2), value)},
        os.path.join(d, ci.ADAPTER_WEIGHTS_FILE),
        metadata={"format": "pt"},
    )
    with open(os.path.join(d, ci.ADAPTER_CONFIG_FILE), "w") as f:
        json.dump({"peft_type": "LORA", "r": r, "lora_alpha": alpha}, f)
    return d


class TestResolveCheckpointRoot:
    def test_tinker_uri_maps_to_run_and_name(self):
        assert ci.resolve_checkpoint_root("tinker://run-abc/weights/ckpt-1") == (
            f"{ci.CHECKPOINT_BASE}/run-abc/ckpt-1"
        )

    def test_filesystem_path_passes_through(self, tmp_path):
        assert ci.resolve_checkpoint_root(str(tmp_path)) == str(tmp_path)

    def test_create_makes_the_directory(self, tmp_path):
        target = str(tmp_path / "nested" / "root")
        ci.resolve_checkpoint_root(target, create=True)
        assert os.path.isdir(target)


class TestPublishAndStage:
    def test_round_trip(self, tmp_path):
        src = _write_adapter(str(tmp_path / "native" / "model"))
        root = str(tmp_path / "ckpt")
        os.makedirs(root)

        published = ci.export_hf_adapter(src, root)
        assert published == os.path.join(root, ci.HF_ADAPTER_DIRNAME)
        assert ci.find_hf_adapter(root) == published
        assert ci.read_adapter_config(root)["r"] == 32

        dest = str(tmp_path / "other_backend" / "weights" / "model")
        assert ci.stage_hf_adapter(root, dest) == dest
        staged = load_file(os.path.join(dest, ci.ADAPTER_WEIGHTS_FILE))
        assert list(staged) == [ci.PEFT_PREFIX + BARE_KEY]

    def test_publish_without_adapter_is_a_no_op(self, tmp_path):
        src = str(tmp_path / "full_finetune")
        os.makedirs(src)
        root = str(tmp_path / "ckpt")
        os.makedirs(root)
        assert ci.export_hf_adapter(src, root) is None
        assert ci.find_hf_adapter(root) is None
        assert ci.read_adapter_config(root) is None

    def test_publish_replaces_a_previous_adapter(self, tmp_path):
        root = str(tmp_path / "ckpt")
        os.makedirs(root)
        ci.export_hf_adapter(_write_adapter(str(tmp_path / "s1"), value=1.0), root)
        ci.export_hf_adapter(_write_adapter(str(tmp_path / "s2"), value=2.0), root)
        published = load_file(os.path.join(root, ci.HF_ADAPTER_DIRNAME, ci.ADAPTER_WEIGHTS_FILE))
        assert float(next(iter(published.values()))[0, 0]) == 2.0
        # no staging tmp left behind for a cross-pod reader to trip on
        assert not os.path.exists(os.path.join(root, ci.HF_ADAPTER_DIRNAME + ".tmp"))

    def test_native_artifact_wins_over_interchange_copy(self, tmp_path):
        root = str(tmp_path / "ckpt")
        os.makedirs(root)
        ci.export_hf_adapter(_write_adapter(str(tmp_path / "src"), value=1.0), root)

        dest = _write_adapter(str(tmp_path / "native"), value=7.0)
        assert ci.stage_hf_adapter(root, dest) is None
        native = load_file(os.path.join(dest, ci.ADAPTER_WEIGHTS_FILE))
        assert float(next(iter(native.values()))[0, 0]) == 7.0

    def test_stage_without_published_adapter_is_a_no_op(self, tmp_path):
        root = str(tmp_path / "ckpt")
        os.makedirs(root)
        assert ci.stage_hf_adapter(root, str(tmp_path / "dest")) is None

    def test_missing_adapter_config_still_publishes_weights(self, tmp_path):
        src = _write_adapter(str(tmp_path / "src"))
        os.remove(os.path.join(src, ci.ADAPTER_CONFIG_FILE))
        root = str(tmp_path / "ckpt")
        os.makedirs(root)
        assert ci.export_hf_adapter(src, root) is not None
        assert ci.read_adapter_config(root) is None


class TestPeftKeyNormalization:
    def test_bare_megatron_keys_gain_the_peft_prefix(self, tmp_path):
        """Miles/Bridge exports bare module paths; PeftModel.from_pretrained
        silently loads NOTHING from those (measured: corr 0.806 vs 0.99964)."""
        root = str(tmp_path / "ckpt")
        os.makedirs(root)
        ci.export_hf_adapter(_write_adapter(str(tmp_path / "src")), root)
        keys = list(load_file(os.path.join(root, ci.HF_ADAPTER_DIRNAME, ci.ADAPTER_WEIGHTS_FILE)))
        assert keys == [ci.PEFT_PREFIX + BARE_KEY]

    def test_already_peft_keyed_weights_are_left_alone(self, tmp_path):
        root = str(tmp_path / "ckpt")
        os.makedirs(root)
        src = _write_adapter(str(tmp_path / "src"), key=ci.PEFT_PREFIX + BARE_KEY)
        ci.export_hf_adapter(src, root)
        keys = list(load_file(os.path.join(root, ci.HF_ADAPTER_DIRNAME, ci.ADAPTER_WEIGHTS_FILE)))
        assert keys == [ci.PEFT_PREFIX + BARE_KEY]   # not double-prefixed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
