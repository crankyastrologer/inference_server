"""Rotary position embeddings, matching Qwen2's reference implementation.

Frequencies are computed in float32 regardless of the model dtype (the reference
disables autocast here), then cast to the activation dtype.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(nn.Module):
    """RoPE frequencies.

    ``inv_freq`` is deliberately *not* a buffer. The reference keeps it in
    float32 even when the model is bf16 (only activations are bf16); registering
    it as a buffer would let ``model.to(bfloat16)`` round it to bf16 and quietly
    shift every rotation. Recomputing 32 floats per forward is cheaper than the
    correctness bug.
    """

    def __init__(
        self,
        head_dim: int,
        rope_theta: float = 1_000_000.0,
        attention_scaling: float = 1.0,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.rope_theta = rope_theta
        self.attention_scaling = attention_scaling

    def _inv_freq(self, device: torch.device) -> torch.Tensor:
        exponent = torch.arange(0, self.head_dim, 2, dtype=torch.float, device=device)
        return 1.0 / (self.rope_theta ** (exponent / self.head_dim))

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self._inv_freq(x.device)
        inv_freq = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        position_ids = position_ids[:, None, :].float()
        freqs = (inv_freq @ position_ids).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
