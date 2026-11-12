from __future__ import annotations

from pathlib import Path

import pytest
import torch
from transformers import Qwen2Config as HFQwen2Config
from transformers import Qwen2ForCausalLM as HFQwen2ForCausalLM

from nanoserve.model.qwen2 import Qwen2Config, Qwen2ForCausalLM
from ref.capture import capture_activations, load_goldens
from ref.compare import TOLERANCES, assert_activations_close

GOLDEN_ROOT = Path("ref/goldens/Qwen__Qwen2.5-0.5B")
DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def _tiny_hf(seed: int = 0, attn: str = "eager") -> HFQwen2ForCausalLM:
    cfg = HFQwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    cfg._attn_implementation = attn
    torch.manual_seed(seed)
    return HFQwen2ForCausalLM(cfg).eval()


def _ours_from_hf(hf: HFQwen2ForCausalLM, dtype: torch.dtype, device: str = "cpu") -> Qwen2ForCausalLM:
    cfg = Qwen2Config.from_hf(hf.config)
    ours = Qwen2ForCausalLM(cfg)
    ours.load_state_dict(hf.state_dict(), strict=False)
    if cfg.tie_word_embeddings:
        ours.tie_weights()
    ours.to(device=device, dtype=dtype)
    ours.eval()
    return ours


def test_tiny_matches_hf_fp32():
    hf = _tiny_hf()
    ids = [1, 2, 3, 4, 5, 6, 7]
    golden = capture_activations(hf, ids)
    ours = _ours_from_hf(hf, torch.float32)
    with torch.no_grad():
        _, probes = ours(torch.tensor([ids]), collect_probes=True)
    assert_activations_close(probes, golden, dtype="float32")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 golden check needs CUDA")
def test_tiny_matches_hf_bf16():
    hf = _tiny_hf().to("cuda", dtype=torch.bfloat16)
    ids = [1, 2, 3, 4, 5, 6, 7]
    golden = capture_activations(hf, ids)
    ours = _ours_from_hf(_tiny_hf(), torch.bfloat16, device="cuda")
    with torch.no_grad():
        _, probes = ours(torch.tensor([ids], device="cuda"), collect_probes=True)
    assert_activations_close(probes, golden, dtype="bfloat16")


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
def test_matches_qwen25_goldens(dtype_name):
    golden_dir = GOLDEN_ROOT / dtype_name
    if not (golden_dir / "activations.safetensors").exists():
        pytest.skip("goldens not generated (run: python -m ref.capture)")
    if dtype_name == "bfloat16" and not torch.cuda.is_available():
        pytest.skip("bf16 check needs CUDA")

    from nanoserve.model.loader import load_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, _cfg, _info = load_model("Qwen/Qwen2.5-0.5B", dtype=DTYPES[dtype_name], device=device)
    except Exception as exc:  # weights not cached
        pytest.skip(f"model unavailable: {exc}")

    golden, meta = load_goldens(golden_dir)
    ids = torch.tensor([meta["input_ids"]], device=device)
    with torch.no_grad():
        _, probes = model(ids, collect_probes=True)

    assert_activations_close(probes, golden, dtype=dtype_name, rtol=TOLERANCES[dtype_name][0], atol=TOLERANCES[dtype_name][1])
