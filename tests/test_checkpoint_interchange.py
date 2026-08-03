"""Unit tests for the cross-backend HF PEFT adapter interchange.

Filesystem-level contract only (no engine): URI resolution, publish/stage
round-trip, atomicity of publish, and the "native artifact wins" rule.
Design: specs/007-q5-migration/HANDOFF.md §2.1.
"""
import json
import os

import pytest

from tinkercloud.training.backends import checkpoint_interchange as ci


def _write_adapter(d, r=32, alpha=32, weights=b"fake-safetensors"):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, ci.ADAPTER_WEIGHTS_FILE), "wb") as f:
        f.write(weights)
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
        with open(os.path.join(dest, ci.ADAPTER_WEIGHTS_FILE), "rb") as f:
            assert f.read() == b"fake-safetensors"

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
        ci.export_hf_adapter(_write_adapter(str(tmp_path / "s1"), weights=b"v1"), root)
        ci.export_hf_adapter(_write_adapter(str(tmp_path / "s2"), weights=b"v2"), root)
        with open(os.path.join(root, ci.HF_ADAPTER_DIRNAME, ci.ADAPTER_WEIGHTS_FILE), "rb") as f:
            assert f.read() == b"v2"
        # no staging tmp left behind for a cross-pod reader to trip on
        assert not os.path.exists(os.path.join(root, ci.HF_ADAPTER_DIRNAME + ".tmp"))

    def test_native_artifact_wins_over_interchange_copy(self, tmp_path):
        root = str(tmp_path / "ckpt")
        os.makedirs(root)
        ci.export_hf_adapter(_write_adapter(str(tmp_path / "src"), weights=b"interchange"), root)

        dest = _write_adapter(str(tmp_path / "native"), weights=b"native")
        assert ci.stage_hf_adapter(root, dest) is None
        with open(os.path.join(dest, ci.ADAPTER_WEIGHTS_FILE), "rb") as f:
            assert f.read() == b"native"

    def test_stage_without_published_adapter_is_a_no_op(self, tmp_path):
        root = str(tmp_path / "ckpt")
        os.makedirs(root)
        assert ci.stage_hf_adapter(root, str(tmp_path / "dest")) is None

    def test_missing_adapter_config_still_publishes_weights(self, tmp_path):
        src = str(tmp_path / "src")
        os.makedirs(src)
        with open(os.path.join(src, ci.ADAPTER_WEIGHTS_FILE), "wb") as f:
            f.write(b"w")
        root = str(tmp_path / "ckpt")
        os.makedirs(root)
        assert ci.export_hf_adapter(src, root) is not None
        assert ci.read_adapter_config(root) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
