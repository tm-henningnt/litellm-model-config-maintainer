"""Tests for `litellm_maintainer.classify`.

The main body of tests drives every fixture in `tests/fixtures/classify/`.
Each fixture is a real provider payload captured live on 2026-07-25, or
states plainly that it is reconstructed or synthesised. See
`tests/fixtures/classify/CAPTURE.md`.

`classify` is pure: no network, no filesystem, no clock read. `now` is
always a parameter.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from litellm_maintainer.classify import (
    BUCKETS,
    NEEDS_OPERATOR,
    REASON_PLAN_ENTITLEMENT_REFUSED,
    REASONS,
    Outcome,
    classify,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "classify"

# 2026-07-25, four days before the reset time both Qwen fixtures state
# in prose. Matches the capture date recorded in CAPTURE.md.
CAPTURE_NOW = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)

FIXTURE_FILES = sorted(
    p for p in FIXTURES_DIR.glob("*.json")
)
FIXTURE_IDS = [p.stem for p in FIXTURE_FILES]

# The expected `reason` for each fixture. `reason` names the condition
# classify read; `expected_outcome` (the fixture's own field) names the
# bucket, the consequence. Two fixtures share a reason with a
# different bucket: the two quota fixtures below (`self_healing`) and
# `gemini-quota` (`needs_operator`, a zero limit) all read as
# `quota_exhausted`, per CAPTURE.md's "A quota error is not always
# self-healing".
EXPECTED_REASON: dict[str, str] = {
    "cline-envelope": "malformed_response",
    "cline-string-error": "plan_entitlement_refused",
    "gemini-deprecated": "identifier_gone",
    "gemini-quota": "quota_exhausted",
    "opencode-go-gateway": "gateway_error",
    "opencode-go-plan": "plan_entitlement_refused",
    "opencode-go-rate-limit": "rate_limited",
    "opencode-go-usage-limit": "quota_exhausted",
    "openrouter-gone": "identifier_gone",
    "qwen-quota-anthropic": "quota_exhausted",
    "qwen-quota-openai": "quota_exhausted",
    "transport-timeout": "timeout",
}


def _parse_expected_reset_at(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


@pytest.mark.parametrize("fixture_path", FIXTURE_FILES, ids=FIXTURE_IDS)
def test_classify_matches_fixture_expectation(fixture_path, load_fixture):
    """classify reproduces the outcome, reset time and reason each fixture records."""
    fixture = load_fixture(f"classify/{fixture_path.name}")

    outcome = classify(
        provider=fixture["provider"],
        http_status=fixture["http_status"],
        body=fixture["body"],
        transport=fixture.get("transport"),
        now=CAPTURE_NOW,
    )

    assert outcome.bucket == fixture["expected_outcome"]
    assert outcome.reset_at == _parse_expected_reset_at(fixture["expected_reset_at"])
    assert outcome.reason == EXPECTED_REASON[fixture_path.stem]
    assert outcome.reason in REASONS


def test_the_same_quota_condition_in_two_json_shapes_agrees(load_fixture):
    """The openai-shaped and anthropic-shaped Qwen quota bodies agree.

    One plan states the same reset time in prose through two different
    JSON envelopes. classify must read the same outcome and the same
    reset time from both.
    """
    openai_shape = load_fixture("classify/qwen-quota-openai.json")
    anthropic_shape = load_fixture("classify/qwen-quota-anthropic.json")

    openai_outcome = classify(
        provider=openai_shape["provider"],
        http_status=openai_shape["http_status"],
        body=openai_shape["body"],
        now=CAPTURE_NOW,
    )
    anthropic_outcome = classify(
        provider=anthropic_shape["provider"],
        http_status=anthropic_shape["http_status"],
        body=anthropic_shape["body"],
        now=CAPTURE_NOW,
    )

    assert openai_outcome == anthropic_outcome
    assert openai_outcome.bucket == "self_healing"
    assert openai_outcome.reset_at == datetime(2026, 7, 29, 21, 45, 0, tzinfo=timezone.utc)


def test_an_unparseable_reset_time_yields_no_reset_time_and_does_not_raise():
    """A reset time classify cannot parse becomes no reset time, not an error."""
    outcome = classify(
        provider="qwencloud-token-plan",
        http_status=429,
        body={
            "code": "Throttling.AllocationQuota",
            "message": "Your token-plan 1-week quota has been exhausted. "
            "The quota will reset whenever it feels like it.",
            "request_id": "test-request-id",
        },
        now=CAPTURE_NOW,
    )

    assert outcome.reset_at is None
    assert outcome.bucket == "self_healing"


def test_a_body_carrying_an_error_under_http_200_is_a_failure():
    """A body carrying an error is a failure whatever the HTTP status reports."""
    outcome = classify(
        provider="cline",
        http_status=200,
        body={"error": "invalid model format", "success": False},
        now=CAPTURE_NOW,
    )

    assert outcome.bucket == "needs_operator"


def test_a_false_success_flag_under_http_200_is_a_failure():
    """A false success flag is a failure whatever the HTTP status reports."""
    outcome = classify(
        provider="cline",
        http_status=200,
        body={"success": False},
        now=CAPTURE_NOW,
    )

    assert outcome.bucket == "needs_operator"


def test_a_prose_reset_date_already_past_this_year_rolls_to_the_next_year():
    """classify resolves a year-less reset date forward, never into the past.

    A quota resets forward. When the stated month and day, combined
    with the current year, land before `now`, the real reset date is
    next year.
    """
    now = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)
    outcome = classify(
        provider="qwencloud-token-plan",
        http_status=429,
        body={
            "code": "Throttling.AllocationQuota",
            "message": "Your token-plan 1-week quota has been exhausted. "
            "The quota will reset at 01-15 09:00:00 UTC.",
            "request_id": "test-request-id",
        },
        now=now,
    )

    assert outcome.reset_at == datetime(2027, 1, 15, 9, 0, 0, tzinfo=timezone.utc)


def test_a_relative_delay_produces_now_plus_that_delay():
    """A relative delay ("retry in Ns") resolves to `now` plus that delay."""
    now = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)
    outcome = classify(
        provider="opencode-go",
        http_status=429,
        body={
            "error": {
                "message": "Worker local total request limit reached. Please retry in 10.5s.",
                "type": "rate_limit_error",
            }
        },
        now=now,
    )

    assert outcome.bucket == "self_healing"
    assert outcome.reset_at == datetime(2026, 7, 25, 12, 0, 10, 500000, tzinfo=timezone.utc)


def test_a_rate_limit_with_no_reset_time_measures_nothing():
    """A rate limit with no reset time is inconclusive, not self-healing.

    The failure is attributable to our own request rate, so it must
    never change Health State.
    """
    outcome = classify(
        provider="opencode-go",
        http_status=429,
        body={
            "error": {
                "message": "Worker local total request limit reached (180/32)",
                "type": "rate_limit_error",
            }
        },
        now=CAPTURE_NOW,
    )

    assert outcome.bucket == "inconclusive"
    assert outcome.reset_at is None


@pytest.mark.parametrize(
    "body",
    [{}, None, "plain text response"],
    ids=["empty_body", "none_body", "bare_string_body"],
)
def test_an_unusual_body_shape_classifies_without_raising(body):
    """An empty body, a `None` body, and a bare-string body never raise."""
    outcome = classify(
        provider="opencode-go",
        http_status=500,
        body=body,
        now=CAPTURE_NOW,
    )

    assert isinstance(outcome, Outcome)
    assert outcome.bucket in BUCKETS
    assert outcome.reason in REASONS


def test_a_readable_completion_is_an_answer_and_not_inconclusive():
    """A response carrying a completion answers, so a Probe can clear an exclusion.

    An Inconclusive outcome leaves Health State untouched. If a
    success read as Inconclusive, an Excluded Offering that works
    again could never clear its exclusion by Probe.
    """
    outcome = classify(
        provider="opencode-go",
        http_status=200,
        body={"choices": [{"message": {"role": "assistant", "content": "hi"}}]},
        now=CAPTURE_NOW,
    )

    assert outcome.bucket == "answered"
    assert outcome.reset_at is None
    assert outcome.reason == "answered"


def test_a_success_with_no_readable_completion_needs_the_operator(load_fixture):
    """A success whose body carries no completion is malformed, not an answer.

    This is the Cline envelope: HTTP 200, `success: true`, and the
    completion hidden under `data`.
    """
    fixture = load_fixture("classify/cline-envelope.json")

    outcome = classify(
        provider=fixture["provider"],
        http_status=fixture["http_status"],
        body=fixture["body"],
        now=CAPTURE_NOW,
    )

    assert outcome.bucket == "needs_operator"


def test_a_transient_unavailable_message_is_not_gone():
    """"Temporarily unavailable" heals itself, so the report must not advise removal.

    `gone` makes the report advise removal from Policy. A word that
    also fits a transient condition would remove a working Offering.
    """
    outcome = classify(
        provider="opencode-go",
        http_status=503,
        body={
            "error": {
                "message": "The model is temporarily unavailable, try again shortly.",
                "type": "server_error",
            }
        },
        now=CAPTURE_NOW,
    )

    assert outcome.bucket == "self_healing"


def test_a_gateway_error_with_an_empty_body_still_heals_itself():
    """A 502 that states nothing in its body is still a gateway error."""
    outcome = classify(
        provider="cline",
        http_status=502,
        body=None,
        now=CAPTURE_NOW,
    )

    assert outcome.bucket == "self_healing"
    assert outcome.reset_at is None


def test_a_service_unavailable_with_an_empty_body_still_heals_itself():
    """A 503 that states nothing in its body is still a gateway error."""
    outcome = classify(
        provider="cline",
        http_status=503,
        body={},
        now=CAPTURE_NOW,
    )

    assert outcome.bucket == "self_healing"


def test_a_rate_limit_status_with_an_empty_body_measures_nothing():
    """A 429 that states no reset time is Inconclusive, not a failure."""
    outcome = classify(
        provider="cline",
        http_status=429,
        body={},
        now=CAPTURE_NOW,
    )

    assert outcome.bucket == "inconclusive"
    assert outcome.reset_at is None


def test_an_authentication_failure_with_an_empty_body_needs_the_operator():
    """A 401 that states nothing in its body still needs the operator."""
    outcome = classify(
        provider="cline",
        http_status=401,
        body={},
        now=CAPTURE_NOW,
    )

    assert outcome.bucket == "needs_operator"


def test_a_plan_that_does_not_include_the_model_is_an_entitlement_refusal():
    """Measured on the Qwen Token Plan, 2026-07-26.

    It returns HTTP 403 with "Access to model denied. Please make sure
    you are eligible for using the model." Without the wording, the
    status rule reads 403 as `authentication_failed`, and the report
    sends the operator to check a credential that is correct. The
    bucket is `needs_operator` either way; the reason must name the
    real condition.
    """
    body = {
        "error": {
            "message": (
                "Access to model denied. Please make sure you are "
                "eligible for using the model."
            )
        }
    }
    outcome = classify(
        provider="qwencloud-token-plan", http_status=403, body=body, now=CAPTURE_NOW
    )
    assert outcome.bucket == NEEDS_OPERATOR
    assert outcome.reason == REASON_PLAN_ENTITLEMENT_REFUSED


def test_a_quota_exhaustion_survives_two_litellm_hops_to_a_worker_proxy():
    """A ChatGPT seat is reached through a worker proxy, so the message
    the main proxy's failure callback sees is wrapped twice.

    Measured 2026-07-27 by driving `litellm.acompletion` against a local
    server returning a worker-shaped 429. litellm's `openai` handler
    prefixes its own text but keeps the inner provider message intact.
    Without this rule the seat would read as a bare rate limit --
    `inconclusive` -- and never be Excluded.
    """
    wrapped = (
        "litellm.RateLimitError: RateLimitError: OpenAIException - "
        "litellm.RateLimitError: ChatgptException - "
        '{"detail":{"message":"You have hit your usage limit for GPT-5.6.",'
        '"quota":{"limit":50,"used":50}}}'
    )

    outcome = classify(
        provider="openai",
        http_status=429,
        body={"error": {"message": wrapped}},
        now=CAPTURE_NOW,
    )

    assert outcome.reason == "quota_exhausted"
    assert outcome.bucket == "self_healing"
