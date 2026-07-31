"""
veRL data converter: Tinker Datum list <-> veRL padded/left-right TensorDict.

Layout: prompt = tokens[0:1] (prompt_len == 1 for every sample, satisfying
veRL's prompt_len > 0 requirement), responses = tokens[1:] right-padded.
Engine log_probs[t] = logp(tokens[t+1] | tokens[:t+1]) then align 1:1 with
client target_tokens / weights / advantages (all length N-1, Miles parity).

Datum schemas (from tinker-cookbook; arrives as Pydantic
ForwardBackwardDatum OR plain dict — handle both):
  SFT: model_input.tokens = tokens[:-1], target_tokens = tokens[1:],
       weights = weights[1:]  (full seq = tokens[:-1] + [target_tokens[-1]])
  RL:  model_input.tokens = full rollout, loss_fn_inputs = {advantages,
       logprobs (sampling), ...} aligned to tokens[1:].
"""
from typing import Any, Dict, List

import torch

from ..base import DataConverter


def _attr_or_key(obj, field: str):
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(field)
    return getattr(obj, field, None)


def _tensor_data(obj, dtype) -> torch.Tensor:
    """TensorData (pydantic .data / wire {'data': ...}) or raw list -> 1-D tensor."""
    if obj is None:
        return None
    data = obj
    if not isinstance(obj, (list, tuple, torch.Tensor)):
        inner = _attr_or_key(obj, "data")
        if inner is not None:
            data = inner
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().to(dtype).flatten()
    try:
        return torch.tensor(data, dtype=dtype).flatten()
    except (TypeError, ValueError):
        return None


def _loss_input(datum, *names):
    lfi = _attr_or_key(datum, "loss_fn_inputs")
    for name in names:
        v = _attr_or_key(lfi, name)
        if v is not None:
            return v
    return None


def _extract_tokens(datum) -> torch.Tensor:
    toks: List[int] = []
    model_input = _attr_or_key(datum, "model_input")
    if model_input is not None:
        chunks = _attr_or_key(model_input, "chunks")
        if chunks:
            for c in chunks:
                t = _attr_or_key(c, "tokens")
                if t is not None:
                    toks.extend(t.tolist() if isinstance(t, torch.Tensor) else list(t))
        else:
            t = _attr_or_key(model_input, "tokens")
            if t is not None:
                toks = t.tolist() if isinstance(t, torch.Tensor) else list(t)
    if not toks:
        t = _attr_or_key(datum, "tokens")
        if t is not None:
            toks = t.tolist() if isinstance(t, torch.Tensor) else list(t)
    assert toks, "datum has no tokens"
    return torch.tensor(toks, dtype=torch.long)


def _full_tokens(datum) -> torch.Tensor:
    tokens = _extract_tokens(datum)
    target = _tensor_data(_loss_input(datum, "target_tokens"), torch.long)
    if target is not None and target.numel() > 0:
        # SFT shape: model tokens are tokens[:-1]; append final target token
        return torch.cat([tokens, target[-1:]])
    return tokens


class VerlDataConverter(DataConverter):
    """Tinker Datum list -> padded left-right dict (backend nests it via
    verl's left_right_2_no_padding at dispatch time)."""

    def forward_backward_to_backend(
        self,
        data: List[Any],
        loss_fn: str,
        args: Any,
    ) -> Dict[str, torch.Tensor]:
        seqs = [_full_tokens(d) for d in data]
        n_lens = [int(s.numel()) for s in seqs]
        resp_lens = [n - 1 for n in n_lens]
        b, rmax = len(seqs), max(resp_lens)
        smax = rmax + 1

        input_ids = torch.zeros(b, smax, dtype=torch.long)
        attention_mask = torch.zeros(b, smax, dtype=torch.long)
        position_ids = torch.zeros(b, smax, dtype=torch.long)
        responses = torch.zeros(b, rmax, dtype=torch.long)
        response_mask = torch.zeros(b, rmax, dtype=torch.long)
        weights = torch.zeros(b, rmax, dtype=torch.float32)
        advantages = torch.zeros(b, rmax, dtype=torch.float32)
        old_log_probs = torch.zeros(b, rmax, dtype=torch.float32)

        for i, (seq, n) in enumerate(zip(seqs, n_lens)):
            input_ids[i, :n] = seq
            attention_mask[i, :n] = 1
            position_ids[i, :n] = torch.arange(n)
            responses[i, : n - 1] = seq[1:]
            response_mask[i, : n - 1] = 1

            w = _tensor_data(_loss_input(data[i], "weights"), torch.float32)
            if w is not None:
                weights[i, : min(len(w), n - 1)] = w[: n - 1]
            adv = _tensor_data(_loss_input(data[i], "advantages"), torch.float32)
            if adv is not None:
                advantages[i, : min(len(adv), n - 1)] = adv[: n - 1]
            lp = _tensor_data(_loss_input(data[i], "logprobs", "log_probs"), torch.float32)
            if lp is not None:
                old_log_probs[i, : min(len(lp), n - 1)] = lp[: n - 1]

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "prompts": input_ids[:, :1],
            "responses": responses,
            "response_mask": response_mask,
        }
        if loss_fn == "cross_entropy":
            out["weights"] = weights
        else:
            out["advantages"] = advantages
            out["old_log_probs"] = old_log_probs
        return out

    def forward_to_backend(self, data: List[Any], args: Any) -> Any:
        return self.forward_backward_to_backend(data, "cross_entropy", args)

    def backend_to_forward_result(self, result: Any, data: List[Any]) -> Dict[str, Any]:
        return {"loss_fn_outputs": self.extract_logprobs(result, data), "metrics": {}}

    def backend_to_forward_backward_result(self, result: Any, data: List[Any]) -> Dict[str, Any]:
        return {"loss_fn_outputs": self.extract_logprobs(result, data), "metrics": {}}

    @staticmethod
    def extract_logprobs(padded_logprobs: torch.Tensor, data: List[Any]) -> List[Dict[str, Any]]:
        """Datum-aligned per-sample logprobs (length N_i - 1 each)."""
        outputs = []
        for i, datum in enumerate(data):
            n = int(_full_tokens(datum).numel())
            lp = padded_logprobs[i, : n - 1].detach().float().cpu()
            outputs.append({"logprobs": {"data": lp.tolist(), "dtype": "float32", "shape": [n - 1]}})
        return outputs
