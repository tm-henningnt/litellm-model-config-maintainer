"""Read a streamed Server-Sent Events response.

Both live callers stream, and both read the result here. The Prober
calls providers directly; the smoke check calls through the proxy. They
must agree on what a streamed answer looks like, because a disagreement
between them is how a false failure survives.

Why they stream at all: litellm's ChatGPT provider answers a streamed
request and fails a non-streaming one, returning an empty `output`
array. A health check that calls the broken way invents failures on
working Offerings. See `.scratch/maintainer-v1/spec-corrections.md`,
corrections 15 and 16.

This module holds no provider knowledge and makes no call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

_DATA_PREFIX = "data:"
_DONE_SENTINEL = "[DONE]"


@dataclass(frozen=True)
class StreamedRead:
    """What one streamed response carried.

    `chunks_seen` counts well-formed chat completion chunks. It decides
    whether the call answered, because a health check tests WIRING, not
    an answer: a reasoning model on a small token budget spends it on
    `reasoning_content` and emits `content: ""`. That is a working
    route. `content` is the assistant text, for the report only.
    """

    content: str
    chunks_seen: int


def read_stream(raw_text: str) -> StreamedRead:
    """Read a streamed SSE body. Never raise.

    Ignore a blank line, a keep-alive comment line starting with `:`,
    a line carrying no `data:` field, and the `[DONE]` sentinel. Ignore
    a chunk that does not parse as JSON: a parse failure here is not a
    provider failure, and reading it as one would be the same mistake
    streaming exists to fix.

    OpenRouter emits `: OPENROUTER PROCESSING` keep-alives, and Gemini's
    first chunk can carry only `extra_content`. Both are well-formed
    traffic and neither is an answer on its own.
    """
    chunks: list[str] = []
    seen = 0
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith(_DATA_PREFIX):
            continue
        data = line[len(_DATA_PREFIX) :].strip()
        if not data or data == _DONE_SENTINEL:
            continue
        try:
            parsed = json.loads(data)
        except ValueError:
            continue
        if not isinstance(parsed, dict):
            continue
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        first = choices[0]
        if not isinstance(first, dict):
            continue
        delta = first.get("delta")
        if not isinstance(delta, dict):
            continue
        seen += 1
        content = delta.get("content")
        if content:
            chunks.append(str(content))
    return StreamedRead(content="".join(chunks), chunks_seen=seen)
