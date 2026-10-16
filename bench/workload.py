"""Workload generation: Poisson arrivals and prompt/output length sampling.

Arrivals are open-loop: request i has a precomputed arrival time, independent
of when the server is actually free. That is what makes the point on the
throughput-latency curve mean "the server at this offered load", rather than
"the server with this many clients in flight".

Lengths are sampled from a distribution, never fixed. Fixed lengths hide exactly
the ragged-completion behaviour continuous batching exists to handle.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


def poisson_arrivals(
    rate: float,
    n: int,
    rng: np.random.Generator,
    start: float = 0.0,
) -> np.ndarray:
    """n open-loop arrival times (seconds, sorted, relative to `start`).

    Inter-arrival gaps are Exponential(rate); the first request arrives at
    `start`. rate <= 0 puts every request at `start`.
    """
    if n <= 0:
        return np.empty(0, dtype=np.float64)
    if rate <= 0:
        return np.full(n, float(start), dtype=np.float64)
    gaps = rng.exponential(1.0 / rate, size=n)
    times = np.cumsum(gaps) - gaps[0] + start
    return times


def lognormal_lengths(
    n: int,
    mu: float,
    sigma: float,
    min_len: int,
    max_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """n lengths ~ LogNormal(mu, sigma) clipped to [min_len, max_len].

    mu/sigma are in log-space. Log-normal is the standard shape for request
    lengths: a bulk of short requests with a thin, long tail.
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    xs = rng.lognormal(mean=mu, sigma=sigma, size=n)
    return np.clip(np.rint(xs), min_len, max_len).astype(np.int64)


def sharegpt_lengths(
    path: str | Path,
    tokenizer,
    max_samples: Optional[int] = None,
    max_prompt: int = 3000,
    max_output: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    """Empirical (prompt, output) token lengths from a ShareGPT-style file.

    Accepts `.jsonl` (one conversation per line) or `.json` (a list). Each
    conversation is `{"conversations": [{"from": ..., "value": ...}, ...]}`;
    the first human turn is the prompt, the first assistant reply the output.
    """
    path = Path(path)
    prompt: list[int] = []
    output: list[int] = []
    with path.open("r", encoding="utf-8") as fh:
        if path.suffix == ".jsonl":
            records: Iterable[dict] = (json.loads(line) for line in fh if line.strip())
        else:
            records = json.load(fh)
        for rec in records:
            convs = rec.get("conversations") or []
            human = next((c for c in convs if c.get("from") in ("human", "user")), None)
            gpt = next((c for c in convs if c.get("from") in ("gpt", "assistant")), None)
            if not human or not gpt:
                continue
            p = len(tokenizer(human["value"]).input_ids)
            o = len(tokenizer(gpt["value"]).input_ids)
            if 0 < p <= max_prompt and 0 < o <= max_output:
                prompt.append(p)
                output.append(o)
            if max_samples and len(prompt) >= max_samples:
                break
    if not prompt:
        raise ValueError(f"no usable conversations found in {path}")
    return (
        np.asarray(prompt, dtype=np.int64),
        np.asarray(output, dtype=np.int64),
    )


@dataclass
class LengthSampler:
    """Samples (prompt_len, output_len) pairs from empirical length pools."""

    prompt_pool: np.ndarray
    output_pool: np.ndarray

    def sample(self, rng: np.random.Generator) -> tuple[int, int]:
        p = int(self.prompt_pool[rng.integers(len(self.prompt_pool))])
        o = int(self.output_pool[rng.integers(len(self.output_pool))])
        return p, o

    def sample_batch(
        self, n: int, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        pi = rng.integers(0, len(self.prompt_pool), size=n)
        oi = rng.integers(0, len(self.output_pool), size=n)
        return self.prompt_pool[pi], self.output_pool[oi]

    @classmethod
    def lognormal(
        cls,
        rng: np.random.Generator,
        pool_size: int = 20_000,
        prompt_mu: float = 5.1,
        prompt_sigma: float = 0.9,
        prompt_min: int = 8,
        prompt_max: int = 3000,
        output_mu: float = 4.9,
        output_sigma: float = 1.1,
        output_min: int = 8,
        output_max: int = 3000,
    ) -> "LengthSampler":
        """A stand-in distribution; swap for real ShareGPT lengths when handy.

        Medians land near 160 prompt / 130 output tokens with a long tail,
        roughly the shape of a chat workload.
        """
        return cls(
            prompt_pool=lognormal_lengths(
                pool_size, prompt_mu, prompt_sigma, prompt_min, prompt_max, rng
            ),
            output_pool=lognormal_lengths(
                pool_size, output_mu, output_sigma, output_min, output_max, rng
            ),
        )


def synthetic_prompt_ids(
    tokenizer,
    n_tokens: int,
    rng: np.random.Generator,
) -> list[int]:
    """A prompt of exactly `n_tokens` ids drawn uniformly from the vocab.

    Content is irrelevant to timing; only length matters for the baseline.
    Special ids are avoided so detokenisation and EOS behave.
    """
    vocab = int(getattr(tokenizer, "vocab_size", 0)) or len(tokenizer)
    lo = 1
    hi = max(lo + 1, vocab - 1)
    return rng.integers(lo, hi, size=n_tokens).tolist()
