"""
Custom litellm provider for the Cline API (https://api.cline.bot/api/v1).

Cline wraps successful chat completions in a non-standard envelope
({"data": {...}, "success": true}) instead of returning the OpenAI
completion object at the top level. litellm's openai-compatible client
does not know to unwrap this, so every call fails with
"provider returned a response with no 'choices'" even though the
underlying request succeeded.

Failure responses are also non-standard, and in a different way. The
provider returns a top-level `error` holding a plain **string**, beside
`"success": false`:

    {"error": "model not found", "success": false}

That is not the OpenAI shape, which puts an object under `error`. So a
plain OpenAI-compatible client breaks on failure as well as on success.
Treat a body carrying an `error`, or a false `success`, as a failure
whatever the HTTP status reports.

A failure does not always answer with JSON at all. The audit saw a
Cloudflare 502 HTML page from this endpoint. So the handler guards the
body before it parses it, and reports the status and the body text.

This handler makes the request directly and unwraps the envelope
before handing the response back to litellm.
"""

import json
from typing import Any

import httpx

from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.utils import ModelResponse

DEFAULT_API_BASE = "https://api.cline.bot/api/v1"


def _build_payload(model: str, messages: list, optional_params: dict, *, stream: bool = False) -> dict:
    payload: dict[str, Any] = {"model": model, "messages": messages, "stream": stream}
    for key, value in optional_params.items():
        if value is not None and key != "stream":
            payload[key] = value
    return payload


def _stream_chunk(parsed: dict) -> dict[str, Any]:
    """Translate one upstream SSE chunk into a litellm streaming chunk.

    Cline wraps a non-streaming body but not a streaming one: with
    `stream: true` it answers with ordinary OpenAI chunks. So this reads
    the standard shape and never unwraps.

    A chunk may carry usage and no choices. Guard the index: a bare
    `choices[0]` raises IndexError and ends the stream with no final
    chunk, which leaves a client waiting.
    """
    choices = parsed.get("choices") or []
    delta = (choices[0].get("delta") or {}) if choices else {}
    finish_reason = choices[0].get("finish_reason") if choices else None
    usage = parsed.get("usage")
    return {
        # `reasoning_content` is dropped on purpose. It is not the answer,
        # and litellm's streaming chunk has no field for it.
        "text": delta.get("content") or "",
        "tool_use": None,
        "is_finished": bool(finish_reason),
        "finish_reason": finish_reason or "",
        "usage": usage,
    }


def _iter_sse_payloads(lines: Any) -> Any:
    """Yield each parsed `data:` payload from an SSE body, in order.

    Stops at `[DONE]`. Skips a line that is not JSON: a keep-alive
    comment is legal in SSE and carries no chunk.
    """
    for line in lines:
        if not line or not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            return
        try:
            yield json.loads(body)
        except json.JSONDecodeError:
            continue


def _error_message(error: Any) -> str:
    """Turn a Cline `error` value into a readable message.

    The value is a plain string on some responses and an object with a
    `message` field on others. Read both shapes without a crash.
    """
    if isinstance(error, dict):
        message = error.get("message")
        if message is not None:
            return str(message)
        return str(error)
    return str(error)


_MAX_BODY_CHARS = 500


def _failure_status(status_code: int) -> int:
    """Choose the status to report on a raised error.

    Use the HTTP status when it already reports a failure (>= 400).
    Otherwise use 502: a broken body under HTTP 200 is the provider's
    fault, not the caller's, and 502 (Bad Gateway) names that an
    upstream server sent a broken response.
    """
    return status_code if status_code >= 400 else 502


def _shorten(text: str) -> str:
    """Cut a body down to a length a log line can carry."""
    stripped = " ".join(text.split())
    if len(stripped) <= _MAX_BODY_CHARS:
        return stripped
    return stripped[:_MAX_BODY_CHARS] + "…"


def unwrap_text_or_raise(status_code: int, text: str) -> dict:
    """Return the completion data from a raw response body.

    Warning: never parse the body before the guard. A failed call does
    not always answer with JSON. The audit saw Cloudflare answer with
    an HTML 502 page at this endpoint, and `json.loads` on that page
    raises `JSONDecodeError` instead of a provider error.

    Parse `text` as JSON. A body that is not JSON raises a
    `CustomLLMError` that carries the HTTP status and the body text.
    Otherwise hand the parsed body to `unwrap_or_raise`.
    """
    try:
        body = json.loads(text)
    except ValueError:
        raise CustomLLMError(
            status_code=_failure_status(status_code),
            message=(
                f"Cline returned a body that is not JSON "
                f"(HTTP {status_code}): {_shorten(text)}"
            ),
        ) from None
    return unwrap_or_raise(status_code, body)


def unwrap_or_raise(status_code: int, body: Any) -> dict:
    """Return the completion data, or raise on a Cline failure body.

    Cline signals failure two ways: a top-level `error` value, or a
    false `success` flag. Both can arrive under HTTP 200, so check the
    body before trusting the HTTP status.

    Raise on a body that is not an object, and on an unwrapped body
    that carries no `choices`. Neither can build a `ModelResponse`.
    """
    if not isinstance(body, dict):
        raise CustomLLMError(
            status_code=_failure_status(status_code),
            message=(
                f"Cline returned a body that is not an object "
                f"(HTTP {status_code}): {_shorten(repr(body))}"
            ),
        )

    error = body.get("error")
    success = body.get("success", True)
    if error is not None or success is False:
        if error is not None:
            message = _error_message(error)
        else:
            message = "Cline reported success: false"
        raise CustomLLMError(status_code=_failure_status(status_code), message=message)

    data = _unwrap(body)
    if not isinstance(data, dict) or "choices" not in data:
        raise CustomLLMError(
            status_code=_failure_status(status_code),
            message=(
                f"Cline returned a response with no 'choices' "
                f"(HTTP {status_code}): {_shorten(repr(data))}"
            ),
        )
    return data


def _unwrap(body: dict) -> Any:
    if "data" in body and "choices" not in body:
        return body["data"]
    return body


class ClineLLM(CustomLLM):
    def completion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout=None,
        client=None,
    ):
        base = (api_base or DEFAULT_API_BASE).rstrip("/")
        payload = _build_payload(model, messages, optional_params)
        response = httpx.post(
            f"{base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                **(headers or {}),
            },
            json=payload,
            timeout=timeout or 60,
        )
        data = unwrap_text_or_raise(response.status_code, response.text)
        return ModelResponse(**data)

    async def acompletion(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout=None,
        client=None,
    ):
        base = (api_base or DEFAULT_API_BASE).rstrip("/")
        payload = _build_payload(model, messages, optional_params)
        async with httpx.AsyncClient() as async_client:
            response = await async_client.post(
                f"{base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    **(headers or {}),
                },
                json=payload,
                timeout=timeout or 60,
            )
        data = unwrap_text_or_raise(response.status_code, response.text)
        return ModelResponse(**data)

    def streaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout=None,
        client=None,
    ):
        base = (api_base or DEFAULT_API_BASE).rstrip("/")
        payload = _build_payload(model, messages, optional_params, stream=True)
        with httpx.stream(
            "POST",
            f"{base}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                **(headers or {}),
            },
            json=payload,
            timeout=timeout or 60,
        ) as response:
            if response.status_code >= 400:
                response.read()
                raise CustomLLMError(
                    status_code=_failure_status(response.status_code),
                    message=_shorten(response.text),
                )
            for parsed in _iter_sse_payloads(response.iter_lines()):
                yield _stream_chunk(parsed)

    async def astreaming(
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers=None,
        timeout=None,
        client=None,
    ):
        base = (api_base or DEFAULT_API_BASE).rstrip("/")
        payload = _build_payload(model, messages, optional_params, stream=True)
        async with httpx.AsyncClient() as async_client:
            async with async_client.stream(
                "POST",
                f"{base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    **(headers or {}),
                },
                json=payload,
                timeout=timeout or 60,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    raise CustomLLMError(
                        status_code=_failure_status(response.status_code),
                        message=_shorten(response.text),
                    )
                async for line in response.aiter_lines():
                    for parsed in _iter_sse_payloads([line]):
                        yield _stream_chunk(parsed)


cline_llm = ClineLLM()
