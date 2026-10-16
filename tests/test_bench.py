from __future__ import annotations

import numpy as np
import pytest

from bench.metrics import SLO, RequestTrace, summarize
from bench.workload import LengthSampler, lognormal_lengths, poisson_arrivals


def test_poisson_arrivals_count_and_start():
    rng = np.random.default_rng(0)
    t = poisson_arrivals(rate=2.0, n=1000, rng=rng)
    assert t.shape == (1000,)
    assert t[0] == 0.0
    assert np.all(np.diff(t) >= 0)
    # mean gap ~ 1/rate
    assert abs(np.mean(np.diff(t)) - 0.5) < 0.05


def test_poisson_arrivals_zero_rate():
    rng = np.random.default_rng(0)
    t = poisson_arrivals(rate=0.0, n=5, rng=rng)
    assert np.all(t == 0.0)


def test_lognormal_clipped_to_bounds():
    rng = np.random.default_rng(1)
    x = lognormal_lengths(10_000, mu=5.0, sigma=2.0, min_len=8, max_len=64, rng=rng)
    assert x.min() >= 8 and x.max() <= 64


def test_length_sampler_batch():
    rng = np.random.default_rng(2)
    s = LengthSampler.lognormal(rng, pool_size=500)
    p, o = s.sample_batch(32, rng)
    assert p.shape == (32,) and o.shape == (32,)


def _trace(i, arrival, start, token_dts, end, ptok, otok):
    return RequestTrace(
        request_id=i,
        arrival_time=arrival,
        start_time=start,
        token_times=[start + d for d in token_dts],
        end_time=end,
        prompt_tokens=ptok,
        output_tokens=otok,
    )


def test_trace_ttft_and_itl():
    t = _trace(0, arrival=0.0, start=0.1, token_dts=[0.2, 0.3, 0.5], end=0.6, ptok=10, otok=3)
    assert abs(t.ttft - 0.3) < 1e-9          # arrival -> first token
    assert abs(t.ttft_service - 0.2) < 1e-9  # start -> first token
    assert t.itls == pytest.approx([0.1, 0.2])


def test_goodput_only_counts_slo_met():
    slo = SLO(ttft_ms=500, itl_p95_ms=50)
    fast = _trace(0, 0.0, 0.0, [0.01, 0.02, 0.03], 0.05, 5, 3)      # meets
    slow = _trace(1, 0.0, 0.0, [1.0, 2.0, 3.0], 4.0, 5, 3)          # ttft too high
    s = summarize([fast, slow], slo=slo, wall_time=4.0)
    assert s["num_requests"] == 2
    assert s["slo_met"] == 1
    assert s["output_tokens"] == 6
    assert abs(s["goodput_tok_per_s"] - 3 / 4.0) < 1e-6
