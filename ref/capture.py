"""Capture reference activations from a HuggingFace Qwen2 model."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_PROMPT = (
    "The KV cache is the memory of attention: every token keeps one key and "
    "one value per layer, so context length and concurrency trade against the "
    "same fixed pool of blocks."
)

DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _core(model):
    """The transformer body (Qwen2Model) inside a *ForCausalLM, if present."""
    return getattr(model, "model", model)


def capture_activations(
    model,
    input_ids: list[int],
    *,
    compute_rope: bool = True,
) -> dict[str, torch.Tensor]:
    """Run one forward and return the probe dict described in ``ref/__init__``."""
    core = _core(model)
    cfg = model.config
    n_heads = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)

    raw: dict = {}
    handles = []

    def reg(module, name: str, unwrap: bool = False) -> None:
        def hook(_mod, _inp, out):
            if unwrap and isinstance(out, tuple):
                out = out[0]
            if torch.is_tensor(out):
                raw[name] = out.detach()
            elif isinstance(out, tuple):
                raw[name] = tuple(t.detach() for t in out)

        handles.append(module.register_forward_hook(hook))

    reg(core.embed_tokens, "embed")
    for i, layer in enumerate(core.layers):
        reg(layer.input_layernorm, f"layer{i}.ln1")
        reg(layer.self_attn.q_proj, f"layer{i}.q")
        reg(layer.self_attn.k_proj, f"layer{i}.k")
        reg(layer.self_attn.v_proj, f"layer{i}.v")
        reg(layer.self_attn, f"layer{i}.attn_out", unwrap=True)
        reg(layer.post_attention_layernorm, f"layer{i}.ln2")
        reg(layer.mlp, f"layer{i}.mlp_out")
        reg(layer, f"layer{i}.out", unwrap=True)
    reg(core.norm, "final_norm")
    if hasattr(model, "lm_head"):
        reg(model.lm_head, "logits")
    if hasattr(core, "rotary_emb"):
        reg(core.rotary_emb, "_rope")

    device = next(model.parameters()).device
    ids = torch.tensor([list(input_ids)], dtype=torch.long, device=device)
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        for h in handles:
            h.remove()

    b, s = raw["embed"].shape[0], raw["embed"].shape[1]

    def to_attn(t: torch.Tensor, heads: int) -> torch.Tensor:
        return t.view(b, s, heads, head_dim).transpose(1, 2)

    apply_rope = None
    if compute_rope and "_rope" in raw:
        try:
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

            apply_rope = apply_rotary_pos_emb
        except Exception:
            apply_rope = None

    out: dict[str, torch.Tensor] = {"embed": raw["embed"]}
    for i in range(int(cfg.num_hidden_layers)):
        out[f"layer{i}.ln1"] = raw[f"layer{i}.ln1"]
        q = to_attn(raw[f"layer{i}.q"], n_heads)
        k = to_attn(raw[f"layer{i}.k"], n_kv)
        v = to_attn(raw[f"layer{i}.v"], n_kv)
        out[f"layer{i}.q"] = q
        out[f"layer{i}.k"] = k
        out[f"layer{i}.v"] = v
        if apply_rope is not None:
            cos, sin = raw["_rope"]
            q_rope, k_rope = apply_rope(q, k, cos, sin)
            out[f"layer{i}.q_rope"] = q_rope
            out[f"layer{i}.k_rope"] = k_rope
        out[f"layer{i}.attn_out"] = raw[f"layer{i}.attn_out"]
        out[f"layer{i}.ln2"] = raw[f"layer{i}.ln2"]
        out[f"layer{i}.mlp_out"] = raw[f"layer{i}.mlp_out"]
        out[f"layer{i}.out"] = raw[f"layer{i}.out"]
    out["final_norm"] = raw["final_norm"]
    if "logits" in raw:
        out["logits"] = raw["logits"]

    return {k: v.contiguous() for k, v in out.items()}


def save_goldens(
    activations: dict[str, torch.Tensor],
    manifest: dict,
    out_dir: str | Path,
) -> Path:
    from safetensors.torch import save_file

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cpu = {k: v.detach().to("cpu").contiguous() for k, v in activations.items()}
    save_file(cpu, str(out_dir / "activations.safetensors"))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return out_dir


def load_goldens(
    dir_path: str | Path,
) -> tuple[dict[str, torch.Tensor], dict]:
    from safetensors.torch import load_file

    dir_path = Path(dir_path)
    tensors = load_file(str(dir_path / "activations.safetensors"))
    manifest = json.loads((dir_path / "manifest.json").read_text())
    return tensors, manifest


def build_manifest(
    model,
    model_id: str,
    input_ids: list[int],
    dtype: str,
    attn_implementation: str,
    activations: dict[str, torch.Tensor],
) -> dict:
    cfg = model.config
    return {
        "model_id": model_id,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "input_ids": list(input_ids),
        "seq_len": len(input_ids),
        "num_hidden_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
        "head_dim": getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads),
        "vocab_size": cfg.vocab_size,
        "probes": sorted(activations.keys()),
        "torch_version": torch.__version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _dir_tag(model_id: str) -> str:
    return model_id.replace("/", "__")


def main(argv=None) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = argparse.ArgumentParser(description="capture golden activations")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompt", default=DEFAULT_PROMPT)
    p.add_argument("--seq-len", type=int, default=64)
    p.add_argument(
        "--dtypes", nargs="+", default=["float32", "bfloat16"], choices=sorted(DTYPES)
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default="ref/goldens")
    args = p.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.model)
    enc = tok(args.prompt, return_tensors=None, truncation=True, max_length=args.seq_len)
    input_ids = enc.input_ids
    print(f"prompt -> {len(input_ids)} tokens")

    for dtype_name in args.dtypes:
        dtype = DTYPES[dtype_name]
        print(f"== {args.model} [{dtype_name}] ==")
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, attn_implementation="eager"
        )
        model.to(args.device)
        model.eval()
        acts = capture_activations(model, input_ids)
        manifest = build_manifest(
            model, args.model, input_ids, dtype_name, "eager", acts
        )
        out = save_goldens(
            acts, manifest, Path(args.out) / _dir_tag(args.model) / dtype_name
        )
        n_bytes = (out / "activations.safetensors").stat().st_size
        print(f"   {len(acts)} probes, {n_bytes / 1e6:.1f} MB -> {out}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
