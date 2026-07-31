"""
Pure-sum loss functions for the veRL backend (Thm 1 contract).

Signature matches verl.workers.utils.losses: (config, model_output, data,
dp_group) -> (loss, metrics). Per-micro-batch partial sums against a
client-owned normalizer (1), scaled by dp_size to cancel FSDP's grad-averaging
all-reduce — split-invariant by construction. We deliberately do NOT use
veRL's sft_loss (flattened-roll boundary leak, see specs/006 q2-plan.md) or
its batch-measured normalizers.
"""
import torch

_METRIC_KEYS = ("weights", "advantages", "old_log_probs", "ref_log_prob")


def _padded_logprobs(model_output, data):
    from verl.workers.utils.padding import no_padding_2_padding

    return no_padding_2_padding(model_output["log_probs"], data)


def _dp_size(data):
    v = data.get("dp_size", None)
    return int(v) if v is not None else 1


def cross_entropy_loss(config, model_output, data, dp_group=None):
    """SFT: -sum(weights * logp(target)) * dp_size. Client owns normalization."""
    log_prob = _padded_logprobs(model_output, data)  # [B, Rmax]
    weights = data["weights"]
    if weights.is_nested:
        weights = weights.to_padded_tensor(0.0)
    loss = -(log_prob * weights).sum() * _dp_size(data)
    return loss, {"sum_weighted_logprob": -loss.detach().item() / max(_dp_size(data), 1)}


def importance_sampling_loss(config, model_output, data, dp_group=None):
    """RL: -sum(adv * exp(logp - logp_old)) * dp_size (Tinker importance_sampling).

    old_log_probs are the client-supplied sampling logprobs; per-token IS ratio,
    no clipping (clipping is client policy, not server policy).
    """
    log_prob = _padded_logprobs(model_output, data)
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    if old_log_prob.is_nested:
        old_log_prob = old_log_prob.to_padded_tensor(0.0)
    if advantages.is_nested:
        advantages = advantages.to_padded_tensor(0.0)
    mask = data["response_mask"]
    if mask.is_nested:
        mask = mask.to_padded_tensor(0)
    mask = mask.to(log_prob.dtype)

    ratio = torch.exp((log_prob - old_log_prob) * mask)
    loss = -(advantages * ratio * mask).sum() * _dp_size(data)

    with torch.no_grad():
        denom = mask.sum().clamp(min=1)
        ppo_kl = ((old_log_prob - log_prob) * mask).sum() / denom
    return loss, {"ppo_kl": ppo_kl.detach().item()}


LOSS_FNS = {
    "cross_entropy": cross_entropy_loss,
    "importance_sampling": importance_sampling_loss,
    "ppo": importance_sampling_loss,
}


def get_loss_fn(name: str):
    if name not in LOSS_FNS:
        raise ValueError(f"verl backend: unsupported loss_fn {name!r} (have {sorted(LOSS_FNS)})")
    return LOSS_FNS[name]
