"""Golden-activation harness.

Compares a from-scratch forward pass against a HuggingFace reference, one
internal probe at a time, so a bug is attributed to a stage instead of showing
up as "logits are wrong somewhere".

Probe contract (both reference and candidate must emit the same names/shapes):

    embed                       [B, S, H]
    layer{i}.ln1                [B, S, H]      RMSNorm before attention
    layer{i}.q                  [B, Hq, S, D]  query projection, pre-RoPE
    layer{i}.k                  [B, Hkv, S, D] key projection, pre-RoPE
    layer{i}.v                  [B, Hkv, S, D] value projection
    layer{i}.q_rope             [B, Hq, S, D]  post-RoPE (optional)
    layer{i}.k_rope             [B, Hkv, S, D] post-RoPE (optional)
    layer{i}.attn_out           [B, S, H]      attention output, after o_proj
    layer{i}.ln2                [B, S, H]      RMSNorm before MLP
    layer{i}.mlp_out            [B, S, H]
    layer{i}.out                [B, S, H]      decoder-layer output
    final_norm                  [B, S, H]
    logits                      [B, S, V]

The reference is captured with ``attn_implementation="eager"``: materialised
softmax is the mathematical reference a hand-written kernel is checked against,
not a fused SDPA/flash path that may differ at the 1e-5 level.
"""

__all__: list[str] = []
