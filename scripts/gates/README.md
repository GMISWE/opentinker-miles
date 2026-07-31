# Seam parity gates (miles backend)

Standalone harnesses for the 005 validation gates (definitions:
`specs/005-miles-ray-interface/design.md`; current baselines:
`specs/003-multi-tenant-lora/HANDOFF.md`). Run ON the server pod with
`TINKER_BASE_URL=http://localhost:8000`; each creates its own model —
rotate the server between runs (models are not auto-reaped).

| Script | Gate | Pass criterion |
|---|---|---|
| `g1_seam_parity.py` | G1 grad-accum split-invariance | grad_norm bit-identical: 1×fb(8) ≡ 2×fb(4), + determinism rerun |
| `g2_client_order.py` | G2 client-order | per-datum logprob length == its weights length (distinct lens) |
| `g2_pipelined.py` | G2 under pipelining | same, with fb/optim/fb/optim submitted before any await |
| `g3_batch_probe.py` | G3 data-shape diagnostic | real shuffled NoRobots batch through fb, length audit |
| `g3_sl_basic.py` | G3 end-to-end | 30-step SFT completes, NLL decreasing (2026-07-30: 2.80→2.28) |
| `g4_pool_isolation.py` | G4 pool isolation (M2) | 2 tenants, 1 pool: join fast; B's pinned probe bit-stable across A's steps; interleaved ≡ serialized; A bit-equal after B's delete |
| `g5_pinned_v0.py` | G5 pinned-v0 + sampler routing | v0 sampler == base, bit-stable across training; live != base; per-tenant routing (B live == base while A live differs) |
| `g6_batch_invariance.py` | G6 batch-invariance (M3, AC1) | same tenant data solo vs co-batched with different co-tenant mixes ⇒ bit-identical per-tenant grad_norm + logprobs (needs `TINKERCLOUD_MILES_COBATCH_MAX_SAMPLES>=16`) |

G4/G5 need a pool-mode server (`TINKERCLOUD_MILES_MULTILORA_SLOTS>0`). The
per-tenant E₂ gate is two `g3_sl_basic.py` runs launched concurrently
against the same pool-mode server (each creates its own tenant).

Known gap (surfaced by G5 run 1): only the SDK's EPHEMERAL sampler flavor
(`save_weights_and_get_sampling_client()`, name=None) registers a
version-pinned sampler session. A NAMED `save_weights_for_sampler(name)` +
`create_sampling_client(path)` yields an unpinned sampler served LIVE on
both backends — snapshot-implying API, live semantics (BUG-015 class).
