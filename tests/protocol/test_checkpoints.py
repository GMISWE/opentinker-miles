"""Checkpoint listing, typed delete, and load_weights as a first request only."""
import requests
from tinker import types

from .conftest import API_KEY, make_datum


def _delete(server, model_id, checkpoint_id, **q):
    return requests.delete(f"{server.base_url}/api/v1/training_runs/{model_id}/checkpoints/{checkpoint_id}",
                           headers={"X-API-Key": API_KEY}, params=q or None, timeout=30)


def test_list_and_delete(service_client, server):
    tc = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    p = tc.save_state("c1").result().path
    sp = tc.save_weights_for_sampler("c1").result().path  # same id, other kind
    rest = service_client.create_rest_client()
    listed = rest.list_checkpoints(tc.model_id).result().checkpoints
    assert {(c.checkpoint_id, c.checkpoint_type, c.tinker_path) for c in listed} == {
        ("weights/c1", "training", p), ("sampler_weights/c1", "sampler", sp)}
    assert all(c.size_bytes and c.size_bytes > 0 for c in listed)

    # bare id is ambiguous -> 400; the SDK's own delete sends the typed form
    assert _delete(server, tc.model_id, "c1").status_code == 400
    rest.delete_checkpoint_from_tinker_path(sp).result()
    remaining = rest.list_checkpoints(tc.model_id).result().checkpoints
    assert [c.checkpoint_id for c in remaining] == ["weights/c1"]
    assert server.post("/api/v1/weights_info", {"tinker_path": sp}).status_code == 404
    assert server.post("/api/v1/weights_info", {"tinker_path": p}).status_code == 200
    # bytes are gone for the deleted kind only
    assert not (server.checkpoint_base / tc.model_id / "sampler_weights" / "c1").exists()
    assert (server.checkpoint_base / tc.model_id / "c1").exists()

    assert _delete(server, tc.model_id, "c1", checkpoint_type="training").status_code == 204
    assert rest.list_checkpoints(tc.model_id).result().checkpoints == []
    assert _delete(server, tc.model_id, "weights/c1").status_code == 404


def test_load_weights_first_request_rule(service_client, server):
    src = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    src.forward_backward([make_datum([1, 2, 3])], "cross_entropy")
    src.optim_step(types.AdamParams(learning_rate=0.5)).result()
    p = src.save_state("w").result().path

    fresh = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    r = server.post("/api/v1/load_weights", {"model_id": fresh.model_id, "path": p, "optimizer": False, "seq_id": 1})
    assert r.status_code == 200, r.text
    fut = server.post("/api/v1/retrieve_future", {"request_id": r.json()["request_id"]}, timeout=60)
    assert fut.status_code == 200 and fut.json()["path"] == p, fut.text
    # the loaded weights carry: w was 0.5 after src's step
    op = fresh.optim_step(types.AdamParams(learning_rate=0.0)).result()
    assert op.metrics["fake_w"] == 0.5

    r = server.post("/api/v1/load_weights", {"model_id": fresh.model_id, "path": p, "optimizer": False, "seq_id": 3})
    assert r.status_code == 400 and "not permitted" in r.json()["error"], r.text

    trained = service_client.create_lora_training_client(base_model="fake/tiny", rank=2)
    trained.forward_backward([make_datum([1, 2, 3])], "cross_entropy").result()
    r = server.post("/api/v1/load_weights", {"model_id": trained.model_id, "path": p, "optimizer": False})
    assert r.status_code == 400
    r = server.post("/api/v1/load_weights", {"model_id": fresh.model_id, "path": "tinker://nope/weights/x", "optimizer": False})
    assert r.status_code in (400, 404)
    r = server.post("/api/v1/load_weights", {"model_id": "model_missing", "path": p, "optimizer": False})
    assert r.status_code == 404
