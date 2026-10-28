# SPDX-License-Identifier: Apache-2.0
"""Draft probability and categorical draw with graph-external uniform input."""

import torch


def sample_probability(logits, temperature, uniforms):
    """Return exact categorical proposal q and its sampled token IDs.

    Greedy rows keep argmax; their q is ignored by the standard greedy
    rejection branch. Float32 arithmetic matches target softmax policy.
    Randomness is supplied externally so graph replay never freezes a draw.
    """
    greedy = temperature < 1e-5
    safe_temperature = torch.where(greedy, 1.0, temperature)
    probs = (logits.float() / safe_temperature[:, None]).softmax(dim=-1)
    cdf = probs.cumsum(dim=-1)
    # Normalize the final cumulative roundoff for inverse-CDF sampling.
    draws = (uniforms * cdf[:, -1]).unsqueeze(-1).contiguous()
    sampled = torch.searchsorted(cdf, draws, right=True).squeeze(-1)
    sampled = sampled.clamp_max(probs.shape[-1] - 1)
    sampled = torch.where(greedy, logits.argmax(dim=-1), sampled)
    return sampled, probs
