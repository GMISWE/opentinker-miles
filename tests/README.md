# tinkercloud Tests

## Unit and protocol suites (CPU only, no server, no GPU)

Two suites run without Ray or a GPU and are the acceptance bar for every
protocol change:

- `tests/protocol/` — the real `tinker` SDK against a live server on the
  `fake` backend (`--backend fake`, deterministic CPU outputs). Covers auth,
  model info, loss registry, sampler routing, sampling params, checkpoints,
  and per-model ordering / seq_id idempotency. ~5 s.
- `tests/test_*.py` unit tests that import `tinkercloud.training` directly
  (ordering queue, backend interface contract, loss registry, checkpoint
  interchange, validators, converter layout).

The tests import the server as the `tinkercloud` package, so the checkout
must be importable under that name (a symlink named `tinkercloud` pointing at
this directory, with its parent on `PYTHONPATH`). On the cluster:

```bash
scripts/pod_pytest.sh                       # default pod + default suites
scripts/pod_pytest.sh tinkercloud-nemorl tests/protocol -q
```

The script pushes the local working tree (sha-verified) to a scratch dir on
the pod and runs pytest there; the deployed `/app` is untouched.

## Integration tests against a running server

Integration tests for a tinkercloud server running in Docker.

## Prerequisites

1. **tinkercloud server running:**
   ```bash
   cd /root/gavin/tinkercloud
   ALLOW_PARTIAL_BATCHES=true \
   PYTHONPATH=/root/gavin/tinkercloud:/root/Megatron-LM:/root/miles:$PYTHONPATH \
   python -m uvicorn training.api:app --host 0.0.0.0 --port 8000
   ```

2. **Model available:**
   ```bash
   # Download Qwen2.5-0.5B-Instruct
   huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct \
       --local-dir /data/models/Qwen2.5-0.5B-Instruct

   # Convert to torch_dist format (if needed for other tests)
   python /root/gavin/miles/tools/convert_hf_to_torch_dist.py \
       --model /data/models/Qwen2.5-0.5B-Instruct \
       --output /data/models/Qwen2.5-0.5B-Instruct_torch_dist \
       --model-args "--swiglu --num-layers 24 --hidden-size 896 ..."
   ```

3. **tinker-cookbook installed:**
   ```bash
   cd /root/gavin/tinker-cookbook && pip install -e .
   ```

## Running Tests

### Cleanup (always run before tests)
```bash
TINKER_BASE_URL=http://localhost:8000 TINKER_API_KEY=tml-dev-key \
    python tests/cleanup_test_env.py
```

### DPO Tests

**Shell script (quick):**
```bash
# Run 3 steps (default)
./tests/test_dpo_reduced.sh

# Run specific number of steps
./tests/test_dpo_reduced.sh 5
```

**Python (pytest compatible):**
```bash
# Run all DPO tests
pytest tests/test_dpo.py -v

# Run specific test
pytest tests/test_dpo.py::test_dpo_reduced -v

# Run directly
python tests/test_dpo.py --test reduced
python tests/test_dpo.py --test all
```

### RLVE Tests

**Shell script (quick, Tinker-only):**
```bash
# Run 1 batch (default)
./tests/test_rlve_reduced.sh

# Run specific number of batches
./tests/test_rlve_reduced.sh 3
```

**Python (pytest compatible):**
```bash
# Run all RLVE tests
PYTHONPATH=/root/gavin/miles:/root/gavin/tinker-cookbook pytest tests/test_rlve.py -v

# Run specific test
pytest tests/test_rlve.py::test_rlve_reduced -v

# Run directly
python tests/test_rlve.py --test reduced
python tests/test_rlve.py --test all
```

**Full comparison (both paths, ~15 min):**
```bash
# Runs Tinker RLVE, then Native Miles RLVE, compares advantages
./tests/test_rlve_both_paths.sh
```

**Note:** For DEBUG_ADVANTAGES to work, start server with:
```bash
DEBUG_ADVANTAGES=1 ALLOW_PARTIAL_BATCHES=true \
PYTHONPATH=/root/gavin/tinkercloud:/root/Megatron-LM:/root/miles:$PYTHONPATH \
python -m uvicorn training.api:app --host 0.0.0.0 --port 8000
```

### Other Tests

```bash
# Health check
pytest tests/test_health.py -v

# Model creation
pytest tests/test_model_creation.py -v

# Full HTTP API test
pytest tests/test_gmi_http.py -v

# RLVE advantage alignment unit test (mock rewards)
PYTHONPATH=/root/gavin/miles:/root/gavin/tinker-cookbook pytest tests/test_advantage_alignment.py -v
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TINKER_BASE_URL` | `http://localhost:8000` | tinkercloud server URL |
| `TINKER_API_KEY` | `tml-dev-key` | API key for authentication |
| `TEST_MODEL_PATH` | `/data/models/Qwen2.5-0.5B-Instruct` | Path to HF model |

## Test Files

| File | Description |
|------|-------------|
| `protocol/` | SDK-driven protocol suite on the fake backend (see above) |
| `test_ordering.py` | Per-model program order and barrier semantics (unit) |
| `test_backend_interface.py` | Backend ABC signature contract for every `SUPPORTED_BACKENDS` entry |
| `test_loss_registry.py` | `loss_fn` / `loss_fn_config` validation |
| `test_checkpoint_interchange.py` | Checkpoint URI <-> root resolution |
| `cleanup_test_env.py` | Cleanup script to free GPUs before tests |
| `test_dpo.py` | DPO training integration tests |
| `test_dpo_reduced.sh` | Shell script for quick DPO test |
| `test_health.py` | Server health check tests |
| `test_model_creation.py` | Model loading tests |
| `test_gmi_http.py` | Full HTTP API tests |
| `test_advantage_computation.py` | Advantage calculation unit tests (group centering) |
| `test_advantage_alignment.py` | RLVE advantage path alignment unit test (mock rewards) |
| `test_rlve.py` | RLVE training integration tests (Tinker path) |
| `test_rlve_reduced.sh` | Shell script for quick RLVE test (Tinker-only) |
| `test_rlve_both_paths.sh` | Full RLVE comparison (Tinker + Native Miles) |
| `test_kgateway_training.py` | Training flow tests |
