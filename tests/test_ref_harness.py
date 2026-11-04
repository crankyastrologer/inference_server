from __future__ import annotations

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from ref.capture import capture_activations, load_goldens, save_goldens, build_manifest
from ref.compare import (
    assert_activations_close,
    compare_activations,
    first_failure,
    format_report,
)


def tiny_model(seed: int = 0) -> Qwen2ForCausalLM:
    cfg = Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    torch.manual_seed(seed)
    return Qwen2ForCausalLM(cfg).eval()


def _expected_probe_names(n_layers: int, rope: bool) -> set[str]:
    names = {"embed", "final_norm", "logits"}
    for i in range(n_layers):
        names |= {
            f"layer{i}.ln1",
            f"layer{i}.q",
            f"layer{i}.k",
            f"layer{i}.v",
            f"layer{i}.attn_out",
            f"layer{i}.ln2",
            f"layer{i}.mlp_out",
            f"layer{i}.out",
        }
        if rope:
            names |= {f"layer{i}.q_rope", f"layer{i}.k_rope"}
    return names


def test_capture_shapes():
    model = tiny_model()
    ids = [1, 2, 3, 4, 5, 6]
    acts = capture_activations(model, ids)

    # rope probes may be absent only if the HF helper moved; otherwise present
    rope = "layer0.q_rope" in acts
    assert set(acts) == _expected_probe_names(2, rope)

    cfg = model.config
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    assert acts["embed"].shape == (1, 6, cfg.hidden_size)
    assert acts["layer0.q"].shape == (1, cfg.num_attention_heads, 6, head_dim)
    assert acts["layer0.k"].shape == (1, cfg.num_key_value_heads, 6, head_dim)
    assert acts["layer0.attn_out"].shape == (1, 6, cfg.hidden_size)
    assert acts["layer1.out"].shape == (1, 6, cfg.hidden_size)
    assert acts["logits"].shape == (1, 6, cfg.vocab_size)


def test_identical_capture_compares_exact():
    model = tiny_model()
    ids = [1, 2, 3, 4]
    acts = capture_activations(model, ids)
    diffs = compare_activations(acts, acts, rtol=0.0, atol=0.0)
    assert all(d.ok for d in diffs), format_report(diffs)


def test_perturbation_is_detected_and_attributed():
    model = tiny_model()
    ids = [1, 2, 3, 4]
    golden = capture_activations(model, ids)

    cand = {k: v.clone() for k, v in golden.items()}
    cand["layer1.mlp_out"] = cand["layer1.mlp_out"] + 1.0

    diffs = compare_activations(cand, golden)
    bad = [d for d in diffs if not d.ok]
    assert [d.name for d in bad] == ["layer1.mlp_out"]
    assert first_failure(diffs).name == "layer1.mlp_out"


def test_missing_and_extra_probes_fail():
    model = tiny_model()
    acts = capture_activations(model, [1, 2, 3])
    cand = dict(acts)
    del cand["layer0.q"]
    cand["bogus"] = acts["embed"]

    diffs = compare_activations(cand, acts)
    notes = {d.name: d.note for d in diffs if not d.ok}
    assert notes["layer0.q"] == "missing in candidate"
    assert notes["bogus"] == "extra in candidate"


def test_save_load_roundtrip(tmp_path):
    model = tiny_model()
    ids = [5, 6, 7, 8]
    acts = capture_activations(model, ids)
    manifest = build_manifest(model, "tiny", ids, "float32", "eager", acts)

    out = save_goldens(acts, manifest, tmp_path / "golden")
    assert (out / "activations.safetensors").exists()
    assert (out / "manifest.json").exists()

    loaded, man = load_goldens(out)
    assert man["seq_len"] == 4 and man["model_id"] == "tiny"
    assert set(loaded) == set(acts)
    assert_activations_close(loaded, acts, rtol=0.0, atol=0.0)


def test_shape_mismatch_reported():
    model = tiny_model()
    acts = capture_activations(model, [1, 2, 3])
    cand = dict(acts)
    cand["embed"] = cand["embed"][:, :2, :]
    diffs = compare_activations(cand, acts)
    d = next(d for d in diffs if d.name == "embed")
    assert not d.ok and not d.shape_ok
    assert d.note == "shape mismatch"
