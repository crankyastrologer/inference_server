"""Contiguous KV cache (baseline #3's memory model).

One preallocated buffer per layer of shape [1, num_kv_heads, max_len, head_dim].
Reserving ``max_len`` up front is deliberate: this is the "contiguous KV" a paged
block allocator later replaces, and the reservation waste is a thing we want to
measure, not hide.

Layout matches the attention tensors ([batch, heads, seq, head_dim]) so writes
and reads are plain slices on the sequence axis.
"""

from __future__ import annotations

import torch


class ContiguousKVCache:
    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_len: int,
        batch_size: int = 1,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_len = max_len
        self.batch_size = batch_size
        self.length = 0
        shape = (batch_size, num_kv_heads, max_len, head_dim)
        self.k = [torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)]
        self.v = [torch.empty(shape, dtype=dtype, device=device) for _ in range(num_layers)]

    @property
    def seq_len(self) -> int:
        return self.length

    def write(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        start = self.length
        s = k_new.shape[2]
        if start + s > self.max_len:
            raise RuntimeError(
                f"KV cache overflow: {start}+{s} > max_len={self.max_len}"
            )
        self.k[layer_idx][:, :, start : start + s] = k_new
        self.v[layer_idx][:, :, start : start + s] = v_new

    def read(
        self, layer_idx: int, total_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.k[layer_idx][:, :, :total_len],
            self.v[layer_idx][:, :, :total_len],
        )

    def advance(self, n: int) -> None:
        self.length += n

    def reset(self) -> None:
        self.length = 0
