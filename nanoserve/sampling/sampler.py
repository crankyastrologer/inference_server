"""Token sampling.

Greedy is the M1 default (deterministic, comparable). Temperature / top-k /
top-p are here for the batching work, where sampling diversity matters for
realistic output-length distributions.
"""

from __future__ import annotations

from typing import Optional

import torch


def sample(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> int:
    """Return one token id from a 1-D logits vector (or the last row of a 2-D)."""
    if logits.dim() > 1:
        logits = logits[-1]
    if temperature is None or temperature <= 0.0:
        return int(torch.argmax(logits).item())

    logits = logits.float() / temperature

    if top_k and top_k > 0:
        k = min(int(top_k), logits.shape[-1])
        kth_value = torch.topk(logits, k).values[-1]
        logits = torch.where(
            logits < kth_value,
            torch.full_like(logits, float("-inf")),
            logits,
        )

    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = (cumulative - probs) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())
