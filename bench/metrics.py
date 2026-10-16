"""Request traces and the metrics that summarise them.

TTFT is measured from request arrival, so it includes queueing. That is the
number a user feels and the one the SLO is written against. `ttft_service`
separates the server's own time-to-first-token from the wait, for diagnosing
whether a bad point on the curve is the queue or the engine.

ITL is taken as the gaps between successive output-token timestamps. Its p95 is
per-request, because a single slow request is a user-visible stall even when the
fleet-wide median is fine.

Goodput counts only tokens from requests that actually met the SLO. Raw
throughput can always be inflated by making everyone wait; goodput cannot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


def percentile(values: Sequence[float], q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


@dataclass
class SLO:
    """Service level objective used to separate throughput from goodput."""

    ttft_ms: float = 500.0
    itl_p95_ms: float = 50.0


@dataclass
class RequestTrace:
    request_id: int
    arrival_time: float
    start_time: float
    token_times: list[float] = field(default_factory=list)
    end_time: float = 0.0
    prompt_tokens: int = 0
    output_tokens: int = 0

    @property
    def queue_delay(self) -> float:
        return self.start_time - self.arrival_time

    @property
    def ttft(self) -> float:
        """Seconds from arrival to first output token."""
        if not self.token_times:
            return float("nan")
        return self.token_times[0] - self.arrival_time

    @property
    def ttft_service(self) -> float:
        """Seconds from service start to first output token (no queue)."""
        if not self.token_times:
            return float("nan")
        return self.token_times[0] - self.start_time

    @property
    def e2e(self) -> float:
        return self.end_time - self.arrival_time

    @property
    def generation_time(self) -> float:
        return self.end_time - self.start_time

    @property
    def itls(self) -> list[float]:
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]

    @property
    def itl_p95(self) -> float:
        if len(self.itls) < 2:
            return 0.0
        return percentile(self.itls, 95)

    @property
    def output_tokens_per_sec(self) -> float:
        gen = self.generation_time
        return self.output_tokens / gen if gen > 0 else float("nan")

    def meets_slo(self, slo: SLO) -> bool:
        ttft = self.ttft
        if ttft != ttft:  # NaN: no token produced
            return False
        return ttft * 1000.0 < slo.ttft_ms and self.itl_p95 * 1000.0 < slo.itl_p95_ms

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.output_tokens


def summarize(
    traces: Sequence[RequestTrace],
    slo: Optional[SLO] = None,
    wall_time: Optional[float] = None,
) -> dict:
    """Aggregate traces into a single flat dict of metrics."""
    slo = slo or SLO()
    n = len(traces)
    if n == 0:
        return {"num_requests": 0}

    if wall_time is None:
        wall_time = max(t.end_time for t in traces) - min(t.arrival_time for t in traces)

    ttft_ms = [t.ttft * 1000.0 for t in traces if t.token_times]
    queue_ms = [t.queue_delay * 1000.0 for t in traces]
    e2e_ms = [t.e2e * 1000.0 for t in traces]
    all_itl_ms = [gap * 1000.0 for t in traces for gap in t.itls]
    per_req_itl_p95_ms = [t.itl_p95 * 1000.0 for t in traces]

    total_output = sum(t.output_tokens for t in traces)
    total_prompt = sum(t.prompt_tokens for t in traces)

    accepted = [t for t in traces if t.meets_slo(slo)]
    good_output = sum(t.output_tokens for t in accepted)

    def ms(x: float) -> float:
        return round(x, 3)

    return {
        "num_requests": n,
        "wall_time_s": round(wall_time, 4),
        # latency
        "ttft_ms_mean": ms(float(np.mean(ttft_ms))) if ttft_ms else float("nan"),
        "ttft_ms_p50": ms(percentile(ttft_ms, 50)),
        "ttft_ms_p95": ms(percentile(ttft_ms, 95)),
        "ttft_ms_p99": ms(percentile(ttft_ms, 99)),
        "queue_ms_mean": ms(float(np.mean(queue_ms))) if queue_ms else float("nan"),
        "queue_ms_p95": ms(percentile(queue_ms, 95)),
        "itl_ms_p50": ms(percentile(all_itl_ms, 50)),
        "itl_ms_p95": ms(percentile(all_itl_ms, 95)),
        "itl_ms_p99": ms(percentile(all_itl_ms, 99)),
        "itl_p95_per_req_ms": ms(percentile(per_req_itl_p95_ms, 95)),
        "e2e_ms_mean": ms(float(np.mean(e2e_ms))) if e2e_ms else float("nan"),
        "e2e_ms_p95": ms(percentile(e2e_ms, 95)),
        # throughput
        "output_tokens": total_output,
        "prompt_tokens": total_prompt,
        "output_tok_per_s": round(total_output / wall_time, 2) if wall_time else float("nan"),
        "total_tok_per_s": (
            round((total_output + total_prompt) / wall_time, 2) if wall_time else float("nan")
        ),
        "requests_per_s": round(n / wall_time, 3) if wall_time else float("nan"),
        # goodput under SLO
        "slo_ttft_ms": slo.ttft_ms,
        "slo_itl_p95_ms": slo.itl_p95_ms,
        "slo_met": len(accepted),
        "slo_met_frac": round(len(accepted) / n, 4),
        "goodput_tok_per_s": (
            round(good_output / wall_time, 2) if wall_time else float("nan")
        ),
        "goodput_req_per_s": (
            round(len(accepted) / wall_time, 3) if wall_time else float("nan")
        ),
    }


def format_table(rows: Sequence[dict], columns: Sequence[str]) -> str:
    """Render summary dicts as a fixed-width text table."""
    if not rows:
        return "(no rows)"
    widths = {c: len(c) for c in columns}
    text = []
    for row in rows:
        cells = {c: _fmt(row.get(c)) for c in columns}
        text.append(cells)
        for c in columns:
            widths[c] = max(widths[c], len(cells[c]))
    header = "  ".join(c.rjust(widths[c]) for c in columns)
    sep = "  ".join("-" * widths[c] for c in columns)
    body = "\n".join(
        "  ".join(cells[c].rjust(widths[c]) for c in columns) for cells in text
    )
    return f"{header}\n{sep}\n{body}"


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if value != value:
            return "nan"
        return f"{value:.2f}"
    return str(value)
