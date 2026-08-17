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


class TestTorchBinFallback:
    """Miles' native export is `adapter_model.bin` (its fused-QKV lora_A is
    aliased across q/k/v and safetensors refuses shared storage). The publish
    path must convert it, cloning the aliased storages en route — regression
    for the conv_mig_main relay failure (2026-08-17)."""

    def _write_bin_adapter(self, d, aliased=True):
        os.makedirs(d, exist_ok=True)
        shared = torch.full((2, 2), 3.0)
        state = {
            "model.layers.0.self_attn.q_proj.lora_A.weight": shared,
            "model.layers.0.self_attn.k_proj.lora_A.weight": shared if aliased else shared.clone(),
            "model.layers.0.self_attn.q_proj.lora_B.weight": torch.full((2, 2), 4.0),
        }
        torch.save(state, os.path.join(d, ci.ADAPTER_WEIGHTS_TORCH_FILE))
        with open(os.path.join(d, ci.ADAPTER_CONFIG_FILE), "w") as f:
            json.dump({"peft_type": "LORA", "r": 32, "lora_alpha": 32}, f)
        return d

    def test_bin_with_aliased_tensors_publishes_as_safetensors(self, tmp_path):
        src = self._write_bin_adapter(str(tmp_path / "iter" / "adapter"))
        root = str(tmp_path / "ckpt")
        os.makedirs(root)
        published = ci.export_hf_adapter(src, root)
        assert published == os.path.join(root, ci.HF_ADAPTER_DIRNAME)
        out = load_file(os.path.join(published, ci.ADAPTER_WEIGHTS_FILE))
        assert sorted(out) == [
            ci.PEFT_PREFIX + "model.layers.0.self_attn.k_proj.lora_A.weight",
            ci.PEFT_PREFIX + "model.layers.0.self_attn.q_proj.lora_A.weight",
            ci.PEFT_PREFIX + "model.layers.0.self_attn.q_proj.lora_B.weight",
        ]
        assert float(out[ci.PEFT_PREFIX + "model.layers.0.self_attn.q_proj.lora_A.weight"][0, 0]) == 3.0
        # the interchange copy is now loadable by find/stage like any other
        assert ci.find_hf_adapter(root) == published

    def test_safetensors_still_wins_when_both_exist(self, tmp_path):
        src = self._write_bin_adapter(str(tmp_path / "src"))
        save_file(
            {BARE_KEY: torch.full((2, 2), 9.0)},
            os.path.join(src, ci.ADAPTER_WEIGHTS_FILE),
            metadata={"format": "pt"},
        )
        root = str(tmp_path / "ckpt")
        os.makedirs(root)
        ci.export_hf_adapter(src, root)
        out = load_file(os.path.join(root, ci.HF_ADAPTER_DIRNAME, ci.ADAPTER_WEIGHTS_FILE))
        assert list(out) == [ci.PEFT_PREFIX + BARE_KEY]
        assert float(out[ci.PEFT_PREFIX + BARE_KEY][0, 0]) == 9.0


class TestSingleTenantNativePublish:
    """save_checkpoint must publish the interchange adapter in single-tenant
    mode too — the hook rode the pool-only adapter_save_dir since the
    multi-LoRA merge, silently breaking the miles->X interchange edge."""

    def test_newest_iter_adapter_is_published(self, tmp_path):
        from tinkercloud.training.backends.miles.backend import _publish_native_adapter

        save_root = tmp_path / "save"
        for i, val in ((1, 1.0), (2, 2.0)):
            d = save_root / f"iter_{i:07d}" / "adapter"
            os.makedirs(d)
            torch.save(
                {"model.layers.0.self_attn.q_proj.lora_A.weight": torch.full((2, 2), val)},
                os.path.join(d, ci.ADAPTER_WEIGHTS_TORCH_FILE),
            )
            os.utime(d, (i * 1000, i * 1000))

        class Args:
            save = str(save_root)

        root = str(tmp_path / "ckpt")
        _publish_native_adapter(Args(), root)
        out = load_file(os.path.join(root, ci.HF_ADAPTER_DIRNAME, ci.ADAPTER_WEIGHTS_FILE))
        assert float(next(iter(out.values()))[0, 0]) == 2.0

    def test_missing_save_root_is_a_warning_not_a_crash(self, tmp_path):
        from tinkercloud.training.backends.miles.backend import _publish_native_adapter

        class Args:
            save = str(tmp_path / "nonexistent")

        _publish_native_adapter(Args(), str(tmp_path / "ckpt"))
        _publish_native_adapter(None, str(tmp_path / "ckpt"))
        assert not os.path.exists(os.path.join(tmp_path / "ckpt", ci.HF_ADAPTER_DIRNAME))


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
