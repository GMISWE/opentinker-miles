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

The protocol suite passes under both SDK eras: the fork SDK installed on the
pod (JSON wire) and upstream `tinker >= 0.25` (proto wire, `client/config`
handshake, sequence ids). To run it under an upstream SDK without touching
the pod's install, put one in a scratch target and prepend it to `PYTHONPATH`
for the test process only (the server the suite spawns never imports the SDK):

```bash
kubectl exec tinkercloud-nemorl -- pip install --target /tmp/pytest-$USER/sdk027 "tinker==0.27.0"
kubectl exec tinkercloud-nemorl -- bash -c "cd /tmp/pytest-$USER/app && \
  PYTHONPATH=/tmp/pytest-$USER/sdk027:/tmp/pytest-$USER python -m pytest tests/protocol tests/test_proto_wire_sdk.py -q"
```

`tests/test_proto_wire_sdk.py` (the codec driven by the SDK's own
`request_conv` / `response_conv`) is skipped unless such an SDK is importable.

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

### Other Tests

```bash
# Health check
pytest tests/test_health.py -v

# Model creation
pytest tests/test_model_creation.py -v

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
| `test_checkpoint_store.py` | Checkpoint identity <-> directory, save counter, pending/completed/failed records |
| `test_checkpoint_interchange.py` | Cross-backend HF PEFT adapter publish/stage |
| `cleanup_test_env.py` | Cleanup script to free GPUs before tests |
| `test_e2e_nemo_rl.sh` | Bash smoke test of the seven core operations against a live server |
| `test_classification_backend_harness.py` | Classification backends (004) harness, GPU-free |
| `test_gates_*.py` | Verdict/ordering/seed contracts of the gate suite (`gates/`, specs/014) |
| `test_dpo.py` | DPO training integration tests |
| `test_dpo_reduced.sh` | Shell script for quick DPO test |
| `test_health.py` | Server health check tests |
| `test_model_creation.py` | Model loading tests |
| `test_advantage_computation.py` | Advantage calculation unit tests (group centering) |
| `test_advantage_alignment.py` | Advantage path alignment unit test (mock rewards) |
