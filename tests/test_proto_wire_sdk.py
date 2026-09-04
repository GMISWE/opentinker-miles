"""The proto codec driven by the tinker SDK's own converters (request_conv /
response_conv), so the wire conventions are checked against the exact code the
client runs. Skipped unless a tinker >= 0.25 is importable."""
import numpy as np
import pytest

from tinkercloud.training.models.requests import ForwardBackwardRequest
from tinkercloud.training.proto.wire import parse_forward_backward_request, serialize_result


sdk_req = pytest.importorskip("tinker.proto.request_conv", reason="tinker >= 0.25 not installed")
sdk_resp = pytest.importorskip("tinker.proto.response_conv")


def test_sdk_request_encoder_round_trips():
    import tinker.types as t
    request = t.ForwardBackwardRequest(
        model_id="model_abc", seq_id=9,
        forward_backward_input=t.ForwardBackwardInput(
            data=[t.Datum(model_input=t.ModelInput.from_ints([1, 2, 3, 4]),
                          loss_fn_inputs={"target_tokens": t.TensorData(data=[2, 3, 4, 5], dtype="int64", shape=[4]),
                                          "weights": t.TensorData(data=[1.0, 0.5, 1.0, 0.0], dtype="float32", shape=[4])})],
            loss_fn="ppo", loss_fn_config={"clip_low_threshold": 0.7}))
    msg = sdk_req.forward_backward_request_to_proto(request)
    msg.forward_only = True
    req, forward_only = parse_forward_backward_request(msg.SerializeToString())
    assert forward_only
    parsed = ForwardBackwardRequest.model_validate(req)
    assert parsed.seq_id == 9
    assert parsed.forward_backward_input.loss_fn_config == {"clip_low_threshold": 0.7}
    assert parsed.forward_backward_input.data[0].loss_fn_inputs["target_tokens"].data == [2, 3, 4, 5]


def test_sdk_response_decoder_reads_forward_backward():
    from tinker import ForwardBackwardOutput
    result = {"loss_fn_output_type": "cross_entropy", "metrics": {"loss:mean": 1.5}, "loss_fn_outputs": [
        {"logprobs": {"data": [-0.1, -0.2, -0.3], "shape": [3], "dtype": "float32"}},
        {"logprobs": {"data": [-0.4], "shape": [1], "dtype": "float32"}}]}
    out = sdk_resp.deserialize_proto_response(serialize_result("forward_backward", result), ForwardBackwardOutput)
    assert out.loss_fn_output_type == "cross_entropy" and out.metrics == {"loss:mean": 1.5}
    np.testing.assert_allclose(out.loss_fn_outputs[0]["logprobs"].tolist(), [-0.1, -0.2, -0.3], rtol=1e-6)
    assert out.loss_fn_outputs[1]["logprobs"].shape == [1]


def test_sdk_response_decoder_reads_sample():
    from tinker import SampleResponse
    result = {"sequences": [{"stop_reason": "stop", "tokens": [1, 2], "logprobs": [-0.5, -1.0]}],
              "prompt_logprobs": [None, -0.25],
              "topk_prompt_logprobs": [None, [(11, -0.1), (12, -0.2)], [(13, -0.3)]]}
    out = sdk_resp.deserialize_proto_response(serialize_result("sample", result), SampleResponse)
    assert out.sequences[0].tokens == [1, 2] and out.sequences[0].stop_reason == "stop"
    assert out.prompt_logprobs == [None, pytest.approx(-0.25)]
    assert out.topk_prompt_logprobs[0] is None
    assert out.topk_prompt_logprobs[2] == [(13, pytest.approx(-0.3))]
