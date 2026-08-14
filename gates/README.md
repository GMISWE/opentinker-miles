# Gate Suite

The checking layer for the TinkerCloud API as a shippable artifact.
Design, phasing (P1/P2/P3), and SOTA positioning:
`specs/014-gate-suite/design.md` in the monorepo.

A gate is one composition — **invariant × trace × arm → verdict** —
instead of a 300-line script:

```python
from gates import invariants, runner
from gates.traces import SegSweep

runner.run(
    invariants.get("SPLIT_INV"),
    SegSweep(model="Qwen/Qwen3-8B-Base", max_seq_len=8192, debug_train_only=True),
    tag="nemo_rl_tp2",
    out_dir="results",
)
```

0.5B and 8B are two parameter points of one gate, not two files.

```bash
# on the server pod (TINKER_BASE_URL=http://localhost:8000)
python -m gates.runner seg_sweep --tag nemo_rl --expect E0_ARBITRARY
python -m gates.runner seg_sweep --tag miles --segmentations '8;4,4;2,6;6,2;2,6;8' \
    --expect E0_ALIGNED_ONLY   # miles deadlocks on rank-imbalanced splits at dp>1
python -m gates.runner seg_sweep --tag nemo_rl_tp2 --model Qwen/Qwen3-8B-Base \
    --max-seq-len 8192 --debug-train-only --expect E0_ARBITRARY

# PARTITION_INV: one call, only the submission order changes (--debug-train-only
# is required on miles below NUM_GPUS=4 -- start_engines dies on
# reordered_gpu_ids[gpu_index] at NUM_GPUS=2)
python -m gates.runner order_perm --tag miles_dp4 --seed 7 --debug-train-only \
    --expect INVARIANT
python -m gates.runner order_perm --tag nemo_rl_dp4 --seed 7 --debug-train-only \
    --expect PARTITION_INVARIANT_ORDER_SENSITIVE

# SEED_REPRO: does the backend read LoraConfig.seed at all
python -m gates.runner seed_repro --tag nemo_rl_dp4 --debug-train-only \
    --expect SEED_HONORED
```

**DP width is not an arm anywhere in this suite.** Width is fixed by the
server's `NUM_GPUS` at launch, so one gate process cannot sweep it: run the
gate once per server configuration and diff the verdict JSONs. On a
buffered backend the width comparison is not certifiable through the API at
all — LoRA `B` is zero-initialized, so no step-0 observable can see `A`.

Exit code 0 iff `passed` (strict invariant at its claimed level) or, when
`--expect` is given, the outcome matches the prediction. Verdict JSONs
conform to `verdicts/schema.json`.

## Layout

| Path | Role |
|---|---|
| `invariants.py` | registry: the paper's §3 contract as first-class objects |
| `traces/` | canonical trace generators (the admission clause made executable) |
| `drivers/` | run a trace through the public SDK/HTTP surface only |
| `oracles/` | fp32 HF(+PEFT) oracle, rerun-envelope estimator (P1: pointers) |
| `comparators.py` | E0 bitwise / E1 calibrated bar / E2 measured envelope |
| `runner.py` | composition + uniform verdict JSON + CLI |
| `verdicts/schema.json` | verdict schema (history in, certificate out) |
| `calibration.json` | (P3) the two measured populations per scale |
| `fixtures/` | (P3) frozen streams, content-addressed |

## Migration map

Standing gates and their source probes. Migrated gates preserve datum
generators, arm naming, and verdict semantics token-for-token, so recorded
results JSONs under `specs/*/probes/results/` remain comparable.

| Gate | Invariant | Source | Status |
|---|---|---|---|
| G1b segmentation sweep (0.5B + 8B) | SPLIT_INV | `specs/008 probes/g1b_segmentation.py`, `specs/013 probes/g1b_8b.py` | **migrated** → `traces/seg_sweep.py` |
| Order permutation (partition sensitivity) | PARTITION_INV | `specs/014 probes/order_perm_probe.py` | **migrated** → `traces/order_perm.py` |
| Seed reproducibility | SEED_REPRO | (none — found by hand, tinker-cloud `995945e` + `GavinZhu-GMI/RL` `faba0bfc3`) | **new** → `traces/seed_repro.py` |
| G1 grad-accum split | SPLIT_INV | `scripts/gates/g1_seam_parity.py` | pending (subsumed by seg_sweep default arms) |
| G1c admission sweep | SPLIT_INV (admission clause) | `specs/008 probes/g1c_admission_sweep.py` | pending |
| G2 client-order / pipelined | DEFER_OBS | `scripts/gates/g2_client_order.py`, `g2_pipelined.py` | pending |
| G3 30-step SFT e2e | ENDPOINT_NLL | `scripts/gates/g3_sl_basic.py` | pending |
| G4 pool isolation | ISOLATION | `scripts/gates/g4_pool_isolation.py` | pending |
| G5 pinned-v0 + routing | PIN_DRIFT | `scripts/gates/g5_pinned_v0.py` | pending |
| G6 batch-invariance | ISOLATION | `scripts/gates/g6_batch_invariance.py` | pending |
| G7 reorder-drain | ISOLATION | `scripts/gates/g7_reorder_merge.py` | pending |
| G8/G8e sampling + overlap | DEFER_OBS | `scripts/gates/g8_verl_sampling.py`, `g8e_overlap_check.py` | pending |
| G9 adapter interchange | ADAPTER_ORACLE | `scripts/gates/g9_adapter_interchange.py` | pending |
| G10 staleness-k | PIN_DRIFT (ver(S)) | `scripts/gates/g10_staleness.py` | pending |
| logprob HF parity | ADAPTER_ORACLE | `scripts/gates/g_lp_hf_parity.py` | pending |

One-off attribution probes (E2/RESID arms, `q4_rl_tenant.py`, analyze
scripts) stay in their spec directories as historical record, per
design.md §6 Q2.

## CI tiers (P3)

Smoke (0.5B, minutes) on every PR touching translation/backend code; full
parity nightly; 8B weekly/manual; mandatory admission run before any
engine-submodule bump.
