"""Re-apply the Megatron-Bridge `hide_adapters` guard on a miles pod.

UPSTREAM BUG: `hide_adapters()` pops the `adapters` submodule while the base
checkpoint loads, but `MultiLoRALinear.state_dict` and `.sharded_state_dict`
reach for `self.adapters` unguarded. So *any* torch_dist / megatron-path
base load of a multi-LoRA model dies with

    AttributeError: 'MultiLoRALinear' object has no attribute 'adapters'

Upstream never hits it because their driver loads the base via bridge-HF.
`MultiLoRAGroupedExpertLinear` subclasses `MultiLoRALinear`, so guarding the
base class covers both.

The fix lands in pod site-packages and is lost on every pod rebuild or
image re-pull — re-apply it after either, before pool-mode `create_model`
(pod_ready_miles.sh does this automatically).

    python3 patch_bridge_hide_adapters.py            # apply (idempotent)
    python3 patch_bridge_hide_adapters.py --check    # report only, rc=1 if unpatched
    python3 patch_bridge_hide_adapters.py --revert

Durable fix is a Megatron-Bridge PR + re-bake with a new pin; until then
this script is the record.
"""


import argparse
import sys

TARGET = "/usr/local/lib/python3.12/dist-packages/megatron/bridge/peft/multi_lora_layers.py"
MARKER = "# tinkercloud-patch: hide_adapters guard"

BEFORE_SD = """        self.to_wrap.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        self.adapters.state_dict(destination=destination, prefix=f"{prefix}adapters.", keep_vars=keep_vars)
        return destination"""

AFTER_SD = """        self.to_wrap.state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        _adapters = getattr(self, "adapters", None)  # tinkercloud-patch: hide_adapters guard
        if _adapters is not None:
            _adapters.state_dict(destination=destination, prefix=f"{prefix}adapters.", keep_vars=keep_vars)
        return destination"""

BEFORE_SSD = """        for i, adapter in enumerate(self.adapters):
            sharded_sd.update(adapter.sharded_state_dict(f"{prefix}adapters.{i}.", sharded_offsets, metadata))"""

AFTER_SSD = """        for i, adapter in enumerate(getattr(self, "adapters", None) or []):  # tinkercloud-patch: hide_adapters guard
            sharded_sd.update(adapter.sharded_state_dict(f"{prefix}adapters.{i}.", sharded_offsets, metadata))"""

PAIRS = ((BEFORE_SD, AFTER_SD), (BEFORE_SSD, AFTER_SSD))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--file", default=TARGET)
    a = ap.parse_args()

    src = open(a.file).read()
    patched = MARKER in src

    if a.check:
        print(f"{'PATCHED' if patched else 'UNPATCHED'}  {a.file}")
        return 0 if patched else 1

    if a.revert:
        if not patched:
            print("already unpatched; nothing to do")
            return 0
        for before, after in PAIRS:
            if after not in src:
                print(f"REFUSING: patched text not found, file diverged:\n{after[:80]}")
                return 2
            src = src.replace(after, before)
        open(a.file, "w").write(src)
        print(f"reverted {a.file}")
        return 0

    if patched:
        print("already patched; nothing to do")
        return 0
    for before, after in PAIRS:
        if before not in src:
            print(f"REFUSING: expected upstream text not found, file diverged:\n{before[:80]}")
            return 2
        src = src.replace(before, after)
    open(a.file, "w").write(src)
    print(f"patched {a.file} (2 sites)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
