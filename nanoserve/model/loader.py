"""Load HF Qwen2 safetensors into the from-scratch model.

Module names mirror the checkpoint, so this is a plain ``load_state_dict``.
Tied embeddings mean the checkpoint omits ``lm_head.weight``; we re-tie after
loading rather than treating it as a missing parameter.
"""

from __future__ import annotations

import glob
import os

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoConfig

from nanoserve.model.qwen2 import Qwen2Config, Qwen2ForCausalLM


def load_state(model_id: str) -> dict[str, torch.Tensor]:
    snapshot = snapshot_download(model_id, allow_patterns=["*.safetensors"])
    files = sorted(glob.glob(os.path.join(snapshot, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no .safetensors found for {model_id} at {snapshot}")
    state: dict[str, torch.Tensor] = {}
    for path in files:
        state.update(load_file(path))
    return state


def load_model(
    model_id: str,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> tuple[Qwen2ForCausalLM, Qwen2Config, dict]:
    hf_cfg = AutoConfig.from_pretrained(model_id)
    cfg = Qwen2Config.from_hf(hf_cfg)

    model = Qwen2ForCausalLM(cfg)
    info = model.load_state_dict(load_state(model_id), strict=False)
    if cfg.tie_word_embeddings:
        model.tie_weights()

    model.to(device=device, dtype=dtype)
    model.eval()
    return model, cfg, {"missing": list(info.missing_keys), "unexpected": list(info.unexpected_keys)}
