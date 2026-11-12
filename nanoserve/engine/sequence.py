"""Request/config types for the generation loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SamplingParams:
    max_new_tokens: int = 64
    temperature: float = 0.0  # <= 0 means greedy
    top_k: int = 0
    top_p: float = 1.0
    stop_strings: list[str] = field(default_factory=list)
    seed: Optional[int] = None


@dataclass
class GenerationResult:
    token_ids: list[int]
    text: str
    prompt_tokens: int
    output_tokens: int
    finish_reason: str
    ttft_s: Optional[float] = None
    e2e_s: Optional[float] = None
