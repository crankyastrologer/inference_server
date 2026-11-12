from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers import Qwen2Config as HFQwen2Config
from transformers import Qwen2ForCausalLM as HFQwen2ForCausalLM

from nanoserve.detokenizer import IncrementalDetokenizer
from nanoserve.engine.generate import generate
from nanoserve.engine.sequence import SamplingParams
from nanoserve.memory import ContiguousKVCache
from nanoserve.model.qwen2 import Qwen2Config, Qwen2ForCausalLM
from nanoserve.sampling import sample


def tiny_model(seed: int = 0) -> Qwen2ForCausalLM:
    hf_cfg = HFQwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    hf_cfg._attn_implementation = "eager"
    torch.manual_seed(seed)
    hf = HFQwen2ForCausalLM(hf_cfg).eval()
    cfg = Qwen2Config.from_hf(hf.config)
    ours = Qwen2ForCausalLM(cfg)
    ours.load_state_dict(hf.state_dict(), strict=False)
    ours.tie_weights()
    ours.eval()
    return ours


def _cache_for(model, max_len: int) -> ContiguousKVCache:
    cfg = model.cfg
    return ContiguousKVCache(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.resolved_head_dim(),
        max_len=max_len,
        dtype=torch.float32,
    )


def test_cached_decode_matches_full_context():
    model = tiny_model()
    seq = [1, 2, 3, 4, 5, 6, 7, 8]
    prompt = seq[:5]
    with torch.no_grad():
        full_logits, _ = model(torch.tensor([seq]))  # [1, 8, V]

        cache = _cache_for(model, max_len=len(seq))
        prefill, _ = model(torch.tensor([prompt]), cache=cache)
        assert torch.allclose(prefill[0, -1], full_logits[0, len(prompt) - 1], atol=1e-5)

        for pos in range(len(prompt), len(seq)):
            step, _ = model(torch.tensor([[seq[pos]]]), cache=cache)
            assert torch.allclose(step[0, -1], full_logits[0, pos], atol=1e-5), pos
        assert cache.length == len(seq)


def test_greedy_generate_matches_recompute():
    model = tiny_model()
    prompt_ids = [3, 1, 4, 1, 5, 9, 2, 6]
    n = 12

    # reference: recompute the whole sequence each step
    ids = list(prompt_ids)
    expected = []
    with torch.no_grad():
        for _ in range(n):
            logits, _ = model(torch.tensor([ids]))
            tok = int(logits[0, -1].argmax())
            expected.append(tok)
            ids.append(tok)

    class StubTokenizer:
        eos_token_id = None

        def __call__(self, text, return_tensors=None):
            return SimpleNamespace(input_ids=torch.tensor([prompt_ids]))

        def decode(self, ids, **kwargs):
            return "".join(chr(i % 0x2000) for i in ids)

    params = SamplingParams(max_new_tokens=n, temperature=0.0)
    result = generate(model, StubTokenizer(), "ignored", params)
    assert result.token_ids == expected
    assert result.output_tokens == n


def test_sampler_greedy_and_topk():
    logits = torch.tensor([0.1, 2.0, -1.0, 0.5])
    assert sample(logits, temperature=0.0) == 1
    assert sample(logits, temperature=1.0, top_k=1) == 1


def test_incremental_detokenizer_roundtrip():
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    except Exception as exc:
        import pytest

        pytest.skip(f"tokenizer unavailable: {exc}")

    text = "Hello, 世界! café — 日本語 😀 done"
    ids = tok(text, add_special_tokens=False).input_ids
    detok = IncrementalDetokenizer(tok)
    streamed = "".join(detok.push(i) for i in ids) + detok.finish()
    expected = tok.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    assert streamed == expected
