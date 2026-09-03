"""The DP gradient reduction must match the optimizer that consumes it.

miles builds the DDP config's `use_distributed_optimizer` from its own
predicate (`bridge_lora_helpers.py`: `"muon" not in args.optimizer`) while the
optimizer CLASS is chosen from `args.use_distributed_optimizer`. The builder
used to pin the latter to False, so for adam+LoRA the two disagreed:

    ddp_config.use_distributed_optimizer = True   -> DDP REDUCE-SCATTERs grads
    args.use_distributed_optimizer      = False   -> Float16OptimizerWithFloat16Params,
                                                     which needs a full ALL-REDUCE

After finalization each rank then held the summed gradient only on its own
1/dp shard of the buffer and its own *unreduced local* gradient everywhere
else -- and stepped the whole parameter set with it. The ranks ended up with
different gradients, different norms, different clip factors (clip_grad=1.0
scales every step by 1/||g||) and different weights.

Measured on the cluster at dp=2, Qwen2.5-0.5B, 8 datums, lr=0
(specs/014-gate-suite/INVESTIGATION-miles-split.md §DEFECT 2):

    rank0 local 131.530 | rank1 local 139.310 | true ||sum|| = 123.481
    post-finalize: rank0 102.787, rank1 96.936   <- the ranks DISAGREE

and the reported grad_norm ran 122.690 / 102.435 / 82.564 at dp = 1 / 2 / 4,
16.5% and 32.7% apart on identical data with a bit-identical forward. Removing
the override puts every width at 123.4814 +/- 4.2e-07 and makes a permuted
round bit-identical.

The regression this guards is narrow and source-level on purpose: the bug was
never a wrong VALUE, it was the service owning a flag the engine also derives.
A second derivation is exactly what must not come back.
"""
import importlib.util
import inspect
import os
import re
import sys
import types

import pytest


def _load_builder_module():
    """Import slime_builder.py without dragging FastAPI in via the package init."""
    try:  # in-container / installed layout
        from tinkercloud.training.core import slime_builder as m
        return m
    except ImportError:
        pass
    root = os.path.join(os.path.dirname(__file__), os.pardir, "training")

    def _pkg(name):
        p = types.ModuleType(name)
        p.__path__ = []
        sys.modules[name] = p
        return p

    def _load(dotted, path):
        spec = importlib.util.spec_from_file_location(dotted, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = mod
        spec.loader.exec_module(mod)
        return mod

    _pkg("_tcbdp")
    _pkg("_tcbdp.core")
    _pkg("_tcbdp.utils")
    _pkg("_tcbdp.backends")
    _pkg("_tcbdp.backends.miles")
    # slime_builder does `from ..backends.miles.config import MilesConfig`
    _load("_tcbdp.backends.env_config", os.path.join(root, "backends", "env_config.py"))
    _load("_tcbdp.backends.miles.config", os.path.join(root, "backends", "miles", "config.py"))
    _load("_tcbdp.utils.model_config", os.path.join(root, "utils", "model_config.py"))
    return _load("_tcbdp.core.slime_builder",
                 os.path.join(root, "core", "slime_builder.py"))


@pytest.fixture(scope="module")
def builder_src():
    return inspect.getsource(_load_builder_module())


def test_builder_does_not_pin_use_distributed_optimizer(builder_src):
    hits = re.findall(r"^\s*args\.use_distributed_optimizer\s*=.*$",
                      builder_src, flags=re.MULTILINE)
    assert not hits, (
        "the builder assigns use_distributed_optimizer:\n  "
        + "\n  ".join(h.strip() for h in hits)
        + "\nThe engine derives this flag for the DDP config separately, so a "
          "second derivation here can disagree with it -- and when it does, DDP "
          "reduce-scatters gradients for an optimizer that needs an all-reduce "
          "and every rank steps with a partly unreduced gradient."
    )


def test_builder_does_not_pass_the_flag_on_the_cli(builder_src):
    # The CLI is the other way the service could pin it, and it reaches the
    # actors (unlike post-parse args.X), so it would be just as wrong.
    assert "--use-distributed-optimizer" not in builder_src
    assert "--no-use-distributed-optimizer" not in builder_src


def test_client_seed_rides_the_cli(builder_src):
    # LoraConfig.seed was accepted by the API and read by nobody. Megatron's own
    # default is 1234, so miles was reproducible by accident while ignoring
    # whatever the client declared. specs/014-gate-suite §THE nemo_rl PEER PROBE.
    assert "'--seed'" in builder_src, "the client's seed must reach the engine"
    assert "lora_config or {}).get('seed')" in builder_src, (
        "--seed must carry the client's declared value, not a constant"
    )
