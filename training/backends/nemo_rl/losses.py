"""Tinker-owned loss functions for the NeMo RL backend."""

import torch
from nemo_rl.algorithms.loss_functions import NLLLoss


class TinkerSumCELoss(NLLLoss):
    """Pure-sum cross-entropy: L = sum(-logprob * weight), no token-count mean.

    Tinker's `cross_entropy` contract (and forward_backward_custom, which routes
    DPO/custom losses through it) requires a pure sum so the client owns
    normalization; Miles honors this via `_loss_norm_total=1`. NeMo RL's NLLLoss
    divides by `global_valid_toks` (a mean), which mis-scales the gradient — and
    for the custom-loss path that divisor is a sum of real-valued gradient
    coefficients, not a token count (BUG-015). We reuse NLLLoss's logprob gather
    verbatim and only force the normalization factor to 1 (pure sum).
    """

    def __call__(
        self, next_token_logits, data, global_valid_seqs, global_valid_toks,
        *args, **kwargs,
    ):
        one = (
            torch.ones_like(global_valid_toks)
            if torch.is_tensor(global_valid_toks)
            else 1.0
        )
        return super().__call__(
            next_token_logits, data, global_valid_seqs, one, *args, **kwargs,
        )
