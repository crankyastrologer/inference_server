"""Benchmark driver.

Sweeps offered request rate against a backend and reports the throughput-latency
curve. Run:

    python -m bench.run --model Qwen/Qwen2.5-0.5B --rates 0.25,0.5,1,2 --num-requests 16
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bench.backends import Backend, HFGenerateBackend
from bench.metrics import SLO, RequestTrace, format_table, summarize
from bench.workload import LengthSampler, poisson_arrivals, synthetic_prompt_ids


def run_point(
    backend: Backend,
    tokenizer,
    sampler: LengthSampler,
    request_rate: float,
    num_requests: int,
    max_new_tokens: int,
    seed: int,
    slo: SLO,
    record_requests: bool = False,
) -> tuple[dict, list[dict]]:
    rng = np.random.default_rng(seed)
    arrivals = poisson_arrivals(request_rate, num_requests, rng)
    prompt_lens, output_lens = sampler.sample_batch(num_requests, rng)

    t0 = time.perf_counter()
    traces: list[RequestTrace] = []
    for i in range(num_requests):
        target = t0 + arrivals[i]
        now = time.perf_counter()
        if now < target:
            time.sleep(target - now)

        prompt_ids = synthetic_prompt_ids(tokenizer, int(prompt_lens[i]), rng)
        start = time.perf_counter()
        result = backend.generate(prompt_ids, min(int(output_lens[i]), max_new_tokens))
        end = time.perf_counter()

        traces.append(
            RequestTrace(
                request_id=i,
                arrival_time=t0 + float(arrivals[i]),
                start_time=start,
                token_times=[start + dt for dt in result.token_times],
                end_time=end,
                prompt_tokens=len(prompt_ids),
                output_tokens=result.output_tokens,
            )
        )

    wall = time.perf_counter() - t0
    summary = summarize(traces, slo=slo, wall_time=wall)
    summary["request_rate"] = request_rate
    summary["max_new_tokens"] = max_new_tokens

    per_request = [
        {
            "request_id": t.request_id,
            "arrival_s": round(t.arrival_time - t0, 4),
            "queue_ms": round(t.queue_delay * 1000, 3),
            "ttft_ms": round(t.ttft * 1000, 3) if t.token_times else None,
            "e2e_ms": round(t.e2e * 1000, 3),
            "prompt_tokens": t.prompt_tokens,
            "output_tokens": t.output_tokens,
            "itl_p95_ms": round(t.itl_p95 * 1000, 3),
            "met_slo": t.meets_slo(slo),
        }
        for t in traces
    ]
    return summary, (per_request if record_requests else [])


def build_backend(args) -> Backend:
    if args.backend == "hf":
        backend = HFGenerateBackend(
            model_id=args.model,
            device=args.device,
            dtype=args.dtype,
            attn_implementation=args.attn,
        )
        backend.name = f"hf-generate:{args.model}"
        return backend
    raise ValueError(f"unknown backend: {args.backend}")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="nanoserve benchmark driver")
    p.add_argument("--backend", default="hf", choices=["hf"])
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--attn", default=None, help="attn_implementation, e.g. sdpa, flash_attention_2")
    p.add_argument("--rates", default="0.5,1,2", help="comma-separated offered request rates (req/s)")
    p.add_argument("--num-requests", type=int, default=16)
    p.add_argument("--max-new-tokens", type=int, default=256, help="cap on generated tokens per request")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--slo-ttft-ms", type=float, default=500.0)
    p.add_argument("--slo-itl-p95-ms", type=float, default=50.0)
    p.add_argument("--out", default="bench/results")
    p.add_argument("--record-requests", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    slo = SLO(ttft_ms=args.slo_ttft_ms, itl_p95_ms=args.slo_itl_p95_ms)

    print(f"backend: {args.backend}  model: {args.model}  device: {args.device}  dtype: {args.dtype}")
    backend = build_backend(args)
    if args.warmup > 0:
        print(f"warming up ({args.warmup} request(s))...")
        backend.warmup(args.warmup)

    rng = np.random.default_rng(args.seed)
    sampler = LengthSampler.lognormal(rng)

    rows: list[dict] = []
    all_requests: list[dict] = []
    for rate in rates:
        print(f"running rate={rate} req/s, n={args.num_requests}...")
        summary, per_request = run_point(
            backend,
            backend.tokenizer,  # type: ignore[attr-defined]
            sampler,
            request_rate=rate,
            num_requests=args.num_requests,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed + int(rate * 1000),
            slo=slo,
            record_requests=args.record_requests,
        )
        rows.append(summary)
        for rec in per_request:
            rec["request_rate"] = rate
        all_requests.extend(per_request)

    columns = [
        "request_rate",
        "requests_per_s",
        "output_tok_per_s",
        "goodput_tok_per_s",
        "slo_met_frac",
        "ttft_ms_p50",
        "ttft_ms_p95",
        "itl_ms_p95",
        "e2e_ms_p95",
        "queue_ms_p95",
    ]
    print()
    print(format_table(rows, columns))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    tag = f"{args.backend}-{args.model.replace('/', '_')}-{stamp}"
    (out_dir / f"{tag}.summary.json").write_text(
        json.dumps({"args": vars(args), "slo": vars(slo), "points": rows}, indent=2)
    )
    if all_requests:
        with (out_dir / f"{tag}.requests.jsonl").open("w") as fh:
            for rec in all_requests:
                fh.write(json.dumps(rec) + "\n")
    print(f"\nwrote {out_dir / (tag + '.summary.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
