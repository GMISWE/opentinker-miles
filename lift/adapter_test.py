"""Stage A adapter core: mask-by-role, mask_out overlength, stream identity."""

from lift.adapter import (
    BuiltSample,
    build_sample,
    build_stream,
    stream_identity,
)
from lift.certify import fingerprint_digest, stream_fingerprint
from lift.compile import compile_sft
from lift.frontends.nemo_rl_test import FIXTURE


def fake_render(messages):
    # 1 token per character, role tagged per message region
    return [(m["role"], [ord(c) for c in m["content"]]) for m in messages]


CHAT = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "content": "yo!"},
]


def test_mask_by_role_assistant():
    s = build_sample(CHAT, fake_render, "assistant", 100, "mask_out")
    assert s.token_ids == tuple(ord(c) for c in "hiyo!")
    assert s.loss_mask == (0, 0, 1, 1, 1)
    assert s.sample_ok


def test_mask_final_only():
    s = build_sample(CHAT, fake_render, "final", 100, "mask_out")
    assert s.loss_mask == (0, 0, 1, 1, 1)


def test_overlength_mask_out_stubs_and_drops_from_loss():
    # nemo_rl semantics: stub each message to min(4, L//n), sample leaves the loss
    long = [
        {"role": "user", "content": "x" * 10},
        {"role": "assistant", "content": "y" * 10},
    ]
    s = build_sample(long, fake_render, "assistant", 8, "mask_out")
    assert not s.sample_ok
    assert len(s.token_ids) == 8  # 4 + 4 stubs
    assert all(m == 0 for m in s.loss_mask)


def test_overlength_truncate_policy_differs():
    long = [
        {"role": "user", "content": "x" * 10},
        {"role": "assistant", "content": "y" * 10},
    ]
    s = build_sample(long, fake_render, "assistant", 8, "truncate")
    assert s.sample_ok and len(s.token_ids) == 8
    assert sum(s.loss_mask) == 0  # truncation ate the assistant region entirely


def test_stream_identity_verdicts():
    a = build_sample(CHAT, fake_render, "assistant", 100, "mask_out")
    b = build_sample(CHAT[:1], fake_render, "assistant", 100, "mask_out")
    native_same = [(a.token_ids, a.loss_mask), (b.token_ids, b.loss_mask)]
    assert stream_identity([a, b], native_same).verdict == "IDENTICAL"
    swapped = list(reversed(native_same))
    r = stream_identity([a, b], swapped)
    assert r.content_match and not r.order_match and "order_perm" in r.verdict
    r2 = stream_identity([a, a], native_same)
    assert r2.verdict == "MISMATCH" and r2.first_divergence


def test_fingerprint():
    a = BuiltSample((1, 2, 3), (0, 1, 1), True)
    fps = stream_fingerprint([[(a.token_ids, a.loss_mask)] * 2])
    assert fps[0].num_tokens == 6 and fps[0].num_loss_tokens == 4
    assert len(fingerprint_digest(fps)) == 16


def test_compile_bundle_carries_template_text_and_policies():
    b = compile_sft(FIXTURE)
    assert b.emit is not None
    spec = b.emit.dataset_spec
    assert spec["overlength_policy"] == "mask_out"
    assert spec["train_on"] == "assistant"
    assert spec["renderer"].startswith("jinja:")
    assert "Question:" in spec["template_text"]  # the executable template rides along
    assert not b.runnable  # obligations outstanding, honestly


def test_build_stream_batching():
    s = build_sample(CHAT, fake_render, "assistant", 100, "mask_out")
    assert [len(x) for x in build_stream([s] * 5, 2)] == [2, 2, 1]
