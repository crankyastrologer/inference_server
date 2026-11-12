"""Single-request generation loop: prefill, then greedy/sampled decode.

M1 is "one request served end to end": prompt -> tokens -> streamed text. This
is deliberately the simplest possible loop (one sequence, contiguous KV). The
scheduler and batching arrive in weeks 8-9 and will reuse the same forward.

The cache is sized ``prompt_len + max_new_tokens`` up front — contiguous
reservation, the waste the paged allocator later removes.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import torch

from nanoserve.detokenizer import IncrementalDetokenizer
from nanoserve.engine.sequence import GenerationResult, SamplingParams
from nanoserve.memory import ContiguousKVCache
from nanoserve.sampling import sample


def generate(
    model,
    tokenizer,
    prompt: str,
    params: Optional[SamplingParams] = None,
    stream: Optional[Callable[[str], None]] = None,
) -> GenerationResult:
    params = params or SamplingParams()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    cfg = model.cfg

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = int(input_ids.shape[1])
    max_len = prompt_len + params.max_new_tokens

    cache = ContiguousKVCache(
        cfg.num_hidden_layers,
        cfg.num_key_value_heads,
        cfg.resolved_head_dim(),
        max_len=max_len,
        dtype=dtype,
        device=device,
    )

    generator = None
    if params.seed is not None:
        generator = torch.Generator(device=device).manual_seed(int(params.seed))

    eos_ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        eos_ids.add(int(tokenizer.eos_token_id))

    detok = IncrementalDetokenizer(tokenizer)
    produced: list[int] = []
    text = ""
    finish_reason = "length"
    ttft: Optional[float] = None

    t0 = time.perf_counter()
    with torch.inference_mode():
        logits, _ = model(input_ids, cache=cache)
        next_logits = logits[0, -1]

        for _ in range(params.max_new_tokens):
            token = sample(
                next_logits,
                temperature=params.temperature,
                top_k=params.top_k,
                top_p=params.top_p,
                generator=generator,
            )

            if token in eos_ids:
                finish_reason = "eos"
                break

            produced.append(token)
            if ttft is None:
                ttft = time.perf_counter() - t0

            delta = detok.push(token)
            if delta:
                text += delta
                if stream is not None:
                    stream(delta)

            if params.stop_strings and any(s in text for s in params.stop_strings):
                finish_reason = "stop"
                break

            if len(produced) >= params.max_new_tokens:
                break

            logits, _ = model(torch.tensor([[token]], device=device), cache=cache)
            next_logits = logits[0, -1]

    tail = detok.finish()
    if tail:
        text += tail
        if stream is not None:
            stream(tail)

    return GenerationResult(
        token_ids=produced,
        text=text,
        prompt_tokens=prompt_len,
        output_tokens=len(produced),
        finish_reason=finish_reason,
        ttft_s=ttft,
        e2e_s=time.perf_counter() - t0,
    )


def main(argv=None) -> int:
    import argparse
    import sys

    from transformers import AutoTokenizer

    from nanoserve.model.loader import load_model

    p = argparse.ArgumentParser(description="nanoserve single-request generation (M1)")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--prompt", default="The KV cache is")
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--stop", nargs="*", default=[])
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-stream", action="store_true")
    args = p.parse_args(argv)

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model, _cfg, _info = load_model(args.model, dtype=dtype, device=args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    params = SamplingParams(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        stop_strings=args.stop,
        seed=args.seed,
    )
    sink = None if args.no_stream else (lambda d: print(d, end="", flush=True))
    result = generate(model, tokenizer, args.prompt, params, stream=sink)
    if args.no_stream:
        print(result.text)
    else:
        print()

    speed = result.output_tokens / result.e2e_s if result.e2e_s else float("nan")
    print(
        f"[{result.output_tokens} tokens | {result.finish_reason} | "
        f"ttft={result.ttft_s:.3f}s | e2e={result.e2e_s:.3f}s | {speed:.1f} tok/s]",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
