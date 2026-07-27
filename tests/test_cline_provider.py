"""Tests for the Cline response handler.

These tests call `unwrap_or_raise` directly with the frozen fixture
bodies. No network call runs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "providers"))

from cline_provider import (  # noqa: E402
    _build_payload,
    _iter_sse_payloads,
    _stream_chunk,
    unwrap_or_raise,
    unwrap_text_or_raise,
)
from litellm.llms.custom_llm import CustomLLMError  # noqa: E402

# The audit saw this page from Cloudflare in front of the provider.
HTML_GATEWAY_PAGE = "<html><head><title>502 Bad Gateway</title></head></html>"


def test_wrapped_success_unwraps_and_returns_assistant_content(load_fixture):
    fixture = load_fixture("classify/cline-envelope.json")

    result = unwrap_or_raise(fixture["http_status"], fixture["body"])

    assert result["choices"][0]["message"]["content"] == (
        "Hello! How can I help you today?"
    )


def test_string_error_body_raises_a_readable_message(load_fixture):
    fixture = load_fixture("classify/cline-string-error.json")

    with pytest.raises(CustomLLMError) as excinfo:
        unwrap_or_raise(fixture["http_status"], fixture["body"])

    assert excinfo.value.status_code == fixture["http_status"]
    assert "invalid model format" in excinfo.value.message


def test_string_error_body_raises_even_under_http_200():
    body = {"error": "invalid model format", "success": False}

    with pytest.raises(CustomLLMError) as excinfo:
        unwrap_or_raise(200, body)

    assert excinfo.value.status_code == 502
    assert "invalid model format" in excinfo.value.message


def test_false_success_flag_raises_even_under_http_200():
    body = {"success": False}

    with pytest.raises(CustomLLMError) as excinfo:
        unwrap_or_raise(200, body)

    assert excinfo.value.status_code == 502


def test_false_success_flag_raises_under_a_reported_error_status():
    body = {"success": False}

    with pytest.raises(CustomLLMError) as excinfo:
        unwrap_or_raise(400, body)

    assert excinfo.value.status_code == 400


def test_object_error_value_uses_its_message_field():
    body = {"error": {"message": "model not found", "type": "invalid_request"}}

    with pytest.raises(CustomLLMError) as excinfo:
        unwrap_or_raise(404, body)

    assert excinfo.value.message == "model not found"


def test_object_error_value_without_message_does_not_crash():
    body = {"error": {"type": "invalid_request"}}

    with pytest.raises(CustomLLMError) as excinfo:
        unwrap_or_raise(400, body)

    assert "invalid_request" in excinfo.value.message


def test_an_html_gateway_page_at_a_5xx_status_raises_a_provider_error():
    """A failure body that is not JSON raises a provider error, not a parse error."""
    with pytest.raises(CustomLLMError) as excinfo:
        unwrap_text_or_raise(502, HTML_GATEWAY_PAGE)

    assert excinfo.value.status_code == 502
    assert "502 Bad Gateway" in excinfo.value.message


def test_an_html_page_at_http_200_raises_a_provider_error():
    """A non-JSON body under HTTP 200 raises a provider error reported as 502."""
    with pytest.raises(CustomLLMError) as excinfo:
        unwrap_text_or_raise(200, HTML_GATEWAY_PAGE)

    assert excinfo.value.status_code == 502
    assert "502 Bad Gateway" in excinfo.value.message


def test_a_wrapped_success_read_from_raw_text_still_unwraps(load_fixture):
    """The raw-text entry point unwraps the same envelope the handler receives."""
    fixture = load_fixture("classify/cline-envelope.json")

    result = unwrap_text_or_raise(fixture["http_status"], json.dumps(fixture["body"]))

    assert result["choices"][0]["message"]["content"] == (
        "Hello! How can I help you today?"
    )


@pytest.mark.parametrize(
    "body",
    [None, "plain text", [1, 2, 3], 7],
    ids=["null_body", "json_string_body", "json_array_body", "json_number_body"],
)
def test_a_body_that_is_not_an_object_raises_instead_of_reaching_model_response(body):
    """A body that is not an object cannot build a ModelResponse, so it raises."""
    with pytest.raises(CustomLLMError) as excinfo:
        unwrap_or_raise(200, body)

    assert excinfo.value.status_code == 502


def test_a_success_body_with_no_choices_raises_instead_of_an_empty_response():
    """A success body carrying no `choices` raises rather than build an empty response."""
    with pytest.raises(CustomLLMError) as excinfo:
        unwrap_or_raise(200, {"data": {"id": "gen-1"}, "success": True})

    assert "no 'choices'" in excinfo.value.message


# --- Streaming ---------------------------------------------------------
#
# Cline wraps a non-streaming body but not a streaming one: with
# `stream: true` it answers with ordinary OpenAI chunks. These tests
# assert the translation to litellm's streaming chunk, and the guards
# that keep a stream terminating.


def test_a_content_delta_becomes_text():
    chunk = _stream_chunk(
        {"choices": [{"index": 0, "delta": {"content": "ok"}, "finish_reason": None}]}
    )
    assert chunk["text"] == "ok"
    assert chunk["is_finished"] is False


def test_a_finish_reason_marks_the_chunk_finished():
    chunk = _stream_chunk(
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    )
    assert chunk["is_finished"] is True
    assert chunk["finish_reason"] == "stop"


def test_a_usage_only_chunk_with_no_choices_does_not_raise():
    """A bare choices[0] ends the stream with no final chunk.

    litellm's own Anthropic adapter carried this defect: the IndexError
    killed the SSE generator, and the client waited for a message_stop
    that never arrived.
    """
    chunk = _stream_chunk(
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}
    )
    assert chunk["text"] == ""
    assert chunk["is_finished"] is False
    assert chunk["usage"] == {"prompt_tokens": 5, "completion_tokens": 2}


def test_reasoning_content_is_not_emitted_as_text():
    """Reasoning is not the answer; emitting it would corrupt the reply."""
    chunk = _stream_chunk(
        {"choices": [{"index": 0, "delta": {"reasoning_content": "thinking"}}]}
    )
    assert chunk["text"] == ""


def test_sse_payloads_stop_at_done():
    lines = [
        'data: {"choices": [{"delta": {"content": "a"}}]}',
        'data: {"choices": [{"delta": {"content": "b"}}]}',
        "data: [DONE]",
        'data: {"choices": [{"delta": {"content": "never"}}]}',
    ]
    texts = [_stream_chunk(p)["text"] for p in _iter_sse_payloads(lines)]
    assert texts == ["a", "b"]


def test_sse_skips_blank_lines_and_keepalive_comments():
    lines = ["", ": keep-alive", 'data: {"choices": [{"delta": {"content": "a"}}]}', ""]
    assert [_stream_chunk(p)["text"] for p in _iter_sse_payloads(lines)] == ["a"]


def test_sse_skips_a_data_line_that_is_not_json():
    lines = ["data: not json at all", 'data: {"choices": [{"delta": {"content": "a"}}]}']
    assert [_stream_chunk(p)["text"] for p in _iter_sse_payloads(lines)] == ["a"]


def test_the_payload_requests_streaming_only_when_asked():
    assert _build_payload("m", [], {})["stream"] is False
    assert _build_payload("m", [], {}, stream=True)["stream"] is True


def test_a_caller_supplied_stream_flag_never_overrides_the_payload():
    """`optional_params` can carry litellm's own stream flag; ignore it."""
    assert _build_payload("m", [], {"stream": True})["stream"] is False
