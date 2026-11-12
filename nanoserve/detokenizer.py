"""Incremental detokenization for streaming.

Byte-level BPE can emit a token that is only part of a multi-byte UTF-8
character. Decoding the prefix at that point yields a trailing U+FFFD
replacement character; emitting it would corrupt the stream. So we hold back
trailing replacement characters until either the character completes (the next
token turns it into a real glyph) or generation finishes (flush as-is).

The algorithm is: keep all ids, decode the whole prefix each step, drop any
trailing replacement, and emit the delta beyond what we already emitted. Whole-
prefix decoding is O(n) per token but exact; a token-level fast path can come
later when the decode loop is the bottleneck.
"""

from __future__ import annotations

REPLACEMENT = "\ufffd"


class IncrementalDetokenizer:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self.token_ids: list[int] = []
        self.emitted: str = ""

    def _decode(self, ids: list[int]) -> str:
        try:
            return self.tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        except TypeError:
            return self.tokenizer.decode(ids, skip_special_tokens=True)

    def push(self, token_id: int) -> str:
        self.token_ids.append(int(token_id))
        return self._emit(self._decode(self.token_ids), final=False)

    def finish(self) -> str:
        return self._emit(self._decode(self.token_ids), final=True)

    @property
    def text(self) -> str:
        return self._decode(self.token_ids)

    def _emit(self, decoded: str, final: bool) -> str:
        if not final:
            decoded = decoded.rstrip(REPLACEMENT)
        if decoded.startswith(self.emitted):
            delta = decoded[len(self.emitted) :]
            self.emitted = decoded
            return delta
        return ""
