"""
veRL data converter: Tinker Datum list <-> veRL padded/left-right TensorDict.

Layout: prompt = tokens[0:1] (prompt_len == 1 for every sample, satisfying
veRL's prompt_len > 0 requirement), responses = tokens[1:] right-padded.
Engine log_probs[t] = logp(tokens[t+1] | tokens[:t+1]) then align 1:1 with
client target_tokens / weights / advantages (all length N-1, Miles parity).

Datum schemas (from tinker-cookbook):
  SFT: model_input.tokens = tokens[:-1], target_tokens = tokens[1:],
       weights = weights[1:]  (full seq = tokens[:-1] + [target_tokens[-1]])
  RL:  model_input.tokens = full rollout, loss_fn_inputs = {advantages,
       logprobs (sampling), ...} aligned to tokens[1:].
"""
from typing import Any, Dict, List

import torch

from ..base import DataConverter


def _get_field(datum: Dict, *names):
    """First present key among names, searching datum and loss_fn_inputs.

    hasattr-before-dict ordering matters: plain dicts expose .values (bound
    method) — always check dict membership first (BUG: values-collision).
    """
    for holder in (datum, datum.get("loss_fn_inputs") or {}):
        for name in names:
            if isinstance(holder, dict):
                if name in holder:
                    return holder[name]
            elif hasattr(holder, name):
                return getattr(holder, name)
    return None


def _as_1d_tensor(value, dtype):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype).flatten()
    if isinstance(value, dict) and "data" in value:  # TensorData wire format
        return torch.tensor(value["data"], dtype=dtype).flatten()
    return torch.tensor(value, dtype=dtype).flatten()


def _full_tokens(datum: Dict) -> torch.Tensor:
    model_input = datum.get("model_input") or {}
    tokens = _get_field({"model_input": None, **datum}, "tokens") or (
        model_input.get("tokens") if isinstance(model_input, dict) else None
    )
    if tokens is None and isinstance(model_input, dict):
        # chunked wire format: model_input {"chunks": [{"tokens": [...]}]}
        chunks = model_input.get("chunks") or []
        tokens = [t for c in chunks for t in c.get("tokens", [])]
    tokens = _as_1d_tensor(tokens, torch.long)
    assert tokens is not None and tokens.numel() > 0, "datum has no tokens"

    target = _as_1d_tensor(_get_field(datum, "target_tokens"), torch.long)
    if target is not None and target.numel() > 0:
        # SFT shape: model tokens are tokens[:-1]; append final target token
        return torch.cat([tokens, target[-1:]])
    return tokens


class VerlDataConverter(DataConverter):
    """Tinker Datum list -> padded left-right dict (backend nests it via
    verl's left_right_2_no_padding at dispatch time)."""

    def forward_backward_to_backend(
        self,
        data: List[Dict],
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

            w = _as_1d_tensor(_get_field(data[i], "weights"), torch.float32)
            if w is not None:
                weights[i, : min(len(w), n - 1)] = w[: n - 1]
            adv = _as_1d_tensor(_get_field(data[i], "advantages"), torch.float32)
            if adv is not None:
                advantages[i, : min(len(adv), n - 1)] = adv[: n - 1]
            lp = _as_1d_tensor(_get_field(data[i], "logprobs", "log_probs"), torch.float32)
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

    def forward_to_backend(self, data: List[Dict], args: Any) -> Any:
        return self.forward_backward_to_backend(data, "cross_entropy", args)

    def backend_to_forward_result(self, result: Any, data: List[Dict]) -> Dict[str, Any]:
        return {"loss_fn_outputs": self.extract_logprobs(result, data), "metrics": {}}

    def backend_to_forward_backward_result(self, result: Any, data: List[Dict]) -> Dict[str, Any]:
        return {"loss_fn_outputs": self.extract_logprobs(result, data), "metrics": {}}

    @staticmethod
    def extract_logprobs(padded_logprobs: torch.Tensor, data: List[Dict]) -> List[Dict[str, Any]]:
        """Datum-aligned per-sample logprobs (length N_i - 1 each)."""
        outputs = []
        for i, datum in enumerate(data):
            n = int(_full_tokens(datum).numel())
            lp = padded_logprobs[i, : n - 1].detach().cpu()
            outputs.append({"logprobs": {"data": lp.tolist(), "dtype": "float32", "shape": [n - 1]}})
        return outputs
