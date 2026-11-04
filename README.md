# nanoserve

A paged-KV, continuously-batched LLM inference server, built from scratch — to
understand, from first principles, why an inference server's throughput-latency
curve looks the way it does.

Target: serve Qwen2.5-0.5B / 1.5B behind an OpenAI-compatible streaming API,
with a fixed-size block KV allocator, continuous batching with preemption under
memory pressure, and a hand-written CUDA paged-attention decode kernel.

## Status

Early — the benchmark harness is in place; the server is not yet. Every stage is
measured on the same workload and the same metrics, against five baselines:

1. HF `generate`, sequential — the floor
2. static batching — padding and reservation waste
3. continuous batching, contiguous KV — scheduling gains
4. continuous batching, paged KV — what paging buys
5. real vLLM — the ceiling

Stages 1–4 live in this repo; 5 is upstream vLLM.

## Layout

```
nanoserve/          the package (model, memory, engine, kernels, sampling, api)
bench/              workload, metrics, backends, driver
  workload.py       Poisson arrivals + lognormal / ShareGPT length sampler
  metrics.py        TTFT, ITL, E2E, throughput, goodput-under-SLO
  backends.py       backend interface + HF generate baseline
  run.py            rate-sweep driver -> bench/results/*.json
tests/              pytest
```

## Quickstart

Requires Python 3.11, a CUDA GPU, and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
.venv/bin/python -m pytest

# sweep offered request rate against the HF baseline
.venv/bin/python -m bench.run \
  --model Qwen/Qwen2.5-0.5B \
  --rates 0.5,1,2 \
  --num-requests 16 \
  --max-new-tokens 128
```

## Metrics

- TTFT — time to first token, measured from arrival (includes queueing)
- ITL — inter-token latency, p50 / p95 / p99
- E2E request latency
- Output throughput, tokens/sec
- Goodput under an SLO (TTFT < 500 ms, p95 ITL < 50 ms)

Workloads are Poisson arrivals with lengths sampled from a distribution, never
fixed — fixed lengths hide the ragged-completion behaviour continuous batching
exists to handle.
