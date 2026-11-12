"""Qwen2 forward pass, written from scratch (no HF module reuse).

Module names mirror the HF checkpoint (``model.layers.{i}.self_attn.q_proj`` etc.)
so weights load directly and the golden harness compares like-for-like. The
forward optionally emits the per-stage probe dict defined in ``ref/__init__``.

Numerics deliberately follow the HF reference op-for-op: RMSNorm upcast order,
RoPE in float32, ``softmax`` in float32 with downcast, GQA via repeat of KV
heads. That is what lets a probe check pass at fp32 tolerance instead of merely
"close enough".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanoserve.model.norm import RMSNorm
from nanoserve.model.rope import RotaryEmbedding, apply_rotary_pos_emb


@dataclass
class Qwen2Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    hidden_act: str = "silu"
    tie_word_embeddings: bool = True
    max_position_embeddings: int = 32768
    head_dim: Optional[int] = None

    def resolved_head_dim(self) -> int:
        return self.head_dim or self.hidden_size // self.num_attention_heads

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_hf(cls, c) -> "Qwen2Config":
        rope_params = getattr(c, "rope_parameters", None) or {}
        rope_theta = rope_params.get("rope_theta", getattr(c, "rope_theta", 10_000.0))
        try:
            head_dim = int(c.head_dim) if getattr(c, "head_dim", None) else None
        except Exception:
            head_dim = None
        return cls(
            vocab_size=c.vocab_size,
            hidden_size=c.hidden_size,
            intermediate_size=c.intermediate_size,
            num_hidden_layers=c.num_hidden_layers,
            num_attention_heads=c.num_attention_heads,
            num_key_value_heads=getattr(c, "num_key_value_heads", c.num_attention_heads),
            rms_norm_eps=getattr(c, "rms_norm_eps", 1e-6),
            rope_theta=rope_theta,
            hidden_act=getattr(c, "hidden_act", "silu"),
            tie_word_embeddings=getattr(c, "tie_word_embeddings", False),
            max_position_embeddings=getattr(c, "max_position_embeddings", 32768),
            head_dim=head_dim,
        )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class Qwen2Attention(nn.Module):
    def __init__(self, cfg: Qwen2Config) -> None:
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.resolved_head_dim()
        self.num_key_value_groups = cfg.num_key_value_groups
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache=None,
        layer_idx: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, s, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q_rope, k_rope = apply_rotary_pos_emb(q, k, cos, sin)

        if cache is not None:
            cache.write(layer_idx, k_rope, v)
            k_all, v_all = cache.read(layer_idx, cache.length + s)
        else:
            k_all, v_all = k_rope, v

        k_rep = repeat_kv(k_all, self.num_key_value_groups)
        v_rep = repeat_kv(v_all, self.num_key_value_groups)

        attn_weights = torch.matmul(q_rope, k_rep.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q_rope.dtype)

        attn_output = torch.matmul(attn_weights, v_rep)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(b, s, -1)
        attn_output = self.o_proj(attn_output)

        probes = {"q": q, "k": k, "v": v, "q_rope": q_rope, "k_rope": k_rope, "attn_out": attn_output}
        return attn_output, probes


class Qwen2MLP(nn.Module):
    def __init__(self, cfg: Qwen2Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)
        self.act_fn = nn.SiLU() if cfg.hidden_act == "silu" else nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen2Config) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Qwen2Attention(cfg)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.mlp = Qwen2MLP(cfg)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        cache=None,
        layer_idx: int = 0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        residual = hidden_states
        ln1 = self.input_layernorm(hidden_states)
        attn_out, probes = self.self_attn(
            ln1, cos, sin, attention_mask, cache=cache, layer_idx=layer_idx
        )
        hidden_states = residual + attn_out

        residual = hidden_states
        ln2 = self.post_attention_layernorm(hidden_states)
        mlp_out = self.mlp(ln2)
        hidden_states = residual + mlp_out

        probes.update({"ln1": ln1, "ln2": ln2, "mlp_out": mlp_out, "out": hidden_states})
        return hidden_states, probes


class Qwen2Model(nn.Module):
    def __init__(self, cfg: Qwen2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(cfg.resolved_head_dim(), cfg.rope_theta)

    @staticmethod
    def causal_mask(
        q_len: int,
        kv_len: int,
        q_offset: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        q_idx = torch.arange(q_len, device=device)[:, None] + q_offset
        k_idx = torch.arange(kv_len, device=device)[None, :]
        allowed = k_idx <= q_idx
        mask = torch.zeros((q_len, kv_len), dtype=dtype, device=device)
        mask = mask.masked_fill(~allowed, torch.finfo(dtype).min)
        return mask[None, None, :, :]

    def forward(
        self,
        input_ids: torch.Tensor,
        cache=None,
        position_ids: Optional[torch.Tensor] = None,
        collect_probes: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        b, s = input_ids.shape
        past_len = cache.length if cache is not None else 0
        hidden_states = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(
                past_len, past_len + s, device=input_ids.device
            )[None, :]
        cos, sin = self.rotary_emb(hidden_states, position_ids)

        kv_len = past_len + s
        attention_mask = self.causal_mask(
            s, kv_len, past_len, hidden_states.dtype, hidden_states.device
        )

        probes: Optional[dict[str, torch.Tensor]] = {"embed": hidden_states} if collect_probes else None
        for i, layer in enumerate(self.layers):
            hidden_states, layer_probes = layer(
                hidden_states, cos, sin, attention_mask, cache=cache, layer_idx=i
            )
            if probes is not None:
                for name, tensor in layer_probes.items():
                    probes[f"layer{i}.{name}"] = tensor

        if cache is not None:
            cache.advance(s)

        hidden_states = self.norm(hidden_states)
        if probes is not None:
            probes["final_norm"] = hidden_states
        return hidden_states, probes


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.model = Qwen2Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.tie_weights()

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        cache=None,
        position_ids: Optional[torch.Tensor] = None,
        collect_probes: bool = False,
    ) -> tuple[torch.Tensor, Optional[dict[str, torch.Tensor]]]:
        hidden_states, probes = self.model(
            input_ids, cache=cache, position_ids=position_ids, collect_probes=collect_probes
        )
        logits = self.lm_head(hidden_states)
        if probes is not None:
            probes = {name: tensor.detach().cpu() for name, tensor in probes.items()}
            probes["logits"] = logits.detach().cpu()
        return logits, probes
