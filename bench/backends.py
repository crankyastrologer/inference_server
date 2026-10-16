"""Backends the benchmark can drive.

A backend is anything that, given a fixed prompt and a token budget, produces
output token timestamps. Every stage of nanoserve implements this same
interface, so curves are comparable across stages:

    1. HF ``generate``, sequential          (this file)   -- the floor
    2. static batching                                      -- padding waste
    3. continuous batching, contiguous KV                   -- scheduling
    4. continuous batching, paged KV                        -- paging
    5. real vLLM                                            -- the ceiling

Timestamps are seconds since the call to ``generate`` started. The driver turns
those into arrival-relative metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Sequence
import time


@dataclass
class GenerationResult:
    """Output of one ``generate`` call.

    ``token_times`` is one timestamp per *streamed chunk*, which for
    ``TextIteratorStreamer`` is approximately one per token. ``output_tokens``
    is the exact number of ids the model produced, read from the returned
    sequence, and is what throughput is computed from.
    """

    token_times: list[float] = field(default_factory=list)
    output_tokens: int = 0


class Backend(Protocol):
    name: str

    def generate(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int,
    ) -> GenerationResult: ...


class HFGenerateBackend:
    """``model.generate`` on one request at a time -- baseline #1.

    Streams output so we get a first-token timestamp; without that TTFT is
    unmeasurable and the whole curve is blind.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-0.5B",
        device: str = "cuda",
        dtype: Optional[str] = "bfloat16",
        attn_implementation: Optional[str] = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"hf-generate:{model_id}"
        self.device = device
        self._torch = torch

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        kwargs = {}
        if dtype:
            kwargs["dtype"] = getattr(torch, dtype)
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        self.model.to(device)
        self.model.eval()

    def generate(
        self,
        input_ids: Sequence[int],
        max_new_tokens: int,
    ) -> GenerationResult:
        import torch
        from threading import Thread
        from transformers import TextIteratorStreamer

        ids = torch.tensor([list(input_ids)], dtype=torch.long, device=self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            input_ids=ids,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            num_beams=1,
            use_cache=True,
            streamer=streamer,
        )

        out: dict = {}

        def _run() -> None:
            with torch.inference_mode():
                out["ids"] = self.model.generate(**gen_kwargs)

        t0 = time.perf_counter()
        thread = Thread(target=_run)
        thread.start()

        token_times: list[float] = []
        for _ in streamer:
            token_times.append(time.perf_counter() - t0)

        thread.join()
        produced = out["ids"]
        output_tokens = int(produced.shape[1] - ids.shape[1])
        # The streamer can yield before the first real token (prompt echo prefill
        # flush); keep timestamps monotone and no longer than the token count.
        token_times = sorted(token_times)[: max(output_tokens, 1)]
        return GenerationResult(token_times=token_times, output_tokens=output_tokens)

    def warmup(self, n: int = 2) -> None:
        for _ in range(n):
            self.generate(list(range(1, 33)), max_new_tokens=8)
