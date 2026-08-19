"""Stage A of certification: rebuild the data stream from an emitted
dataset_spec, independently of the native framework, so the two streams can
be compared BYTE-for-byte (specs/015-lift/OBLIGATIONS.md §3).

The core is dependency-injected (tokenize/render callables) so the semantics
are unit-testable without HF downloads; the real loaders bind tokenizer and
template at the edge. Content identity and ORDER are judged separately:
order depends on framework RNG internals, so an order-only mismatch is a
declared T graded by the order_perm gate, never a silent verdict.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

# A rendered message region: (role, token_ids). The per-message region
# structure mirrors nemo_rl's message_log so masks align per message.
Region = tuple[str, list[int]]
RenderFn = Callable[[list[dict]], list[Region]]  # messages -> regions


@dataclass(frozen=True)
class BuiltSample:
    token_ids: tuple[int, ...]
    loss_mask: tuple[int, ...]
    sample_ok: bool  # False = over-length under mask_out policy


def build_sample(
    messages: list[dict],
    render: RenderFn,
    train_on: str,
    max_length: int,
    overlength_policy: str,
) -> BuiltSample:
    regions = render(messages)
    length = sum(len(ids) for ids in (r[1] for r in regions))

    if length > max_length:
        if overlength_policy == "mask_out":
            # nemo_rl sft_processor:171-177 — stub each message, drop from loss
            stub = max(0, min(4, max_length // max(1, len(regions))))
            ids = [t for _, r in regions for t in r[:stub]]
            return BuiltSample(tuple(ids), tuple(0 for _ in ids), sample_ok=False)
        regions = _truncate_regions(regions, max_length)

    ids: list[int] = []
    mask: list[int] = []
    n = len(regions)
    for i, (role, r_ids) in enumerate(regions):
        if train_on == "final":
            on = i == n - 1
        elif train_on == "all":
            on = True
        else:
            on = role == train_on
        ids.extend(r_ids)
        mask.extend([1 if on else 0] * len(r_ids))
    return BuiltSample(tuple(ids), tuple(mask), sample_ok=True)


def _truncate_regions(regions: list[Region], max_length: int) -> list[Region]:
    out: list[Region] = []
    left = max_length
    for role, ids in regions:
        take = min(len(ids), left)
        out.append((role, ids[:take]))
        left -= take
        if left == 0:
            break
    return out


def build_stream(
    samples: list[BuiltSample], batch_size: int
) -> list[list[BuiltSample]]:
    return [samples[i : i + batch_size] for i in range(0, len(samples), batch_size)]


@dataclass
class IdentityReport:
    content_match: bool  # multiset of (token_ids, loss_mask) pairs identical
    order_match: bool  # additionally, identical order
    n_built: int
    n_native: int
    first_divergence: str = ""

    @property
    def verdict(self) -> str:
        if self.content_match and self.order_match:
            return "IDENTICAL"
        if self.content_match:
            return "CONTENT-IDENTICAL (order differs: declared T, grade via order_perm)"
        return "MISMATCH"


def stream_identity(
    built: list[BuiltSample], native: list[tuple[tuple[int, ...], tuple[int, ...]]]
) -> IdentityReport:
    """native: [(token_ids, loss_mask)] as exported from the framework side."""
    b = [(s.token_ids, s.loss_mask) for s in built]
    rpt = IdentityReport(
        content_match=sorted(b) == sorted(native),
        order_match=b == native,
        n_built=len(b),
        n_native=len(native),
    )
    if not rpt.content_match:
        b_set, n_set = set(b), set(native)
        only_b = next(iter(b_set - n_set), None)
        only_n = next(iter(n_set - b_set), None)
        rpt.first_divergence = (
            f"built-only sample len={len(only_b[0]) if only_b else '-'}; "
            f"native-only sample len={len(only_n[0]) if only_n else '-'}"
        )
    return rpt
