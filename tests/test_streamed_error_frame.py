"""Tests for a failure stated inside a 2xx stream.

Measured 2026-08-21: Cline answers HTTP 200, sends one frame carrying
an `error`, then `[DONE]`. The frame holds no `choices`. Reading the
chunk count alone reports an empty body, and `classify` calls that
`malformed_response` -- `needs_operator` for a five-second rate limit.

No network call runs.
"""

from __future__ import annotations

from datetime import datetime, timezone

from litellm_maintainer.classify import (
    INCONCLUSIVE,
    NEEDS_OPERATOR,
    REASON_MALFORMED_RESPONSE,
    REASON_RATE_LIMITED,
    SELF_HEALING,
    classify,
)
from litellm_maintainer.prober import _streamed_body
from litellm_maintainer.sse import read_stream

NOW = datetime(2026, 8, 21, 12, 0, tzinfo=timezone.utc)

RATE_LIMIT_STREAM = (
    'data: {"error":{"code":"stream_initialization_failed","message":'
    '"failed to invoke model with streaming: request failed with status 429: '
    'z-ai/glm-5.2:free is temporarily rate-limited upstream. retry in 5s"}}\n'
    "data: [DONE]\n"
)
ANSWERING_STREAM = 'data: {"choices":[{"delta":{"content":"OK"}}]}\ndata: [DONE]\n'
SILENT_STREAM = "data: [DONE]\n"


def test_read_stream_keeps_the_first_error_frame():
    read = read_stream(RATE_LIMIT_STREAM)
    assert read.chunks_seen == 0
    assert read.error is not None


def test_a_rate_limit_stated_in_a_stream_reads_as_self_healing():
    body = _streamed_body(read_stream(RATE_LIMIT_STREAM))
    outcome = classify(
        http_status=200, body=body, transport=None, provider="cline", now=NOW
    )
    assert outcome.bucket == SELF_HEALING
    assert outcome.reason == REASON_RATE_LIMITED
    # The message states its own delay, so the reset time is known.
    assert outcome.reset_at is not None


def test_a_hyphen_does_not_change_the_bucket():
    for wording in ("rate-limited upstream", "rate limited upstream"):
        outcome = classify(
            http_status=200,
            body={"error": {"message": f"temporarily {wording}"}},
            transport=None,
            provider="cline",
            now=NOW,
        )
        assert outcome.reason == REASON_RATE_LIMITED, wording
        assert outcome.bucket == INCONCLUSIVE, wording


def test_a_chunk_still_answers():
    body = _streamed_body(read_stream(ANSWERING_STREAM))
    assert body["choices"][0]["message"]["content"] == "OK"


def test_an_error_frame_beside_a_chunk_still_answers():
    stream = ANSWERING_STREAM.replace(
        "data: [DONE]", 'data: {"error":"late failure"}\ndata: [DONE]'
    )
    read = read_stream(stream)
    assert read.error is not None
    # A chunk proves the route works, so the error frame decides nothing.
    assert "choices" in _streamed_body(read)


def test_a_silent_stream_still_reads_as_malformed():
    body = _streamed_body(read_stream(SILENT_STREAM))
    outcome = classify(
        http_status=200, body=body, transport=None, provider="cline", now=NOW
    )
    assert outcome.bucket == NEEDS_OPERATOR
    assert outcome.reason == REASON_MALFORMED_RESPONSE
