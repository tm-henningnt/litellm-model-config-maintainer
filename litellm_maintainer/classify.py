"""Turn a provider response into an Outcome.

`classify` is pure. It reads a provider response and states what it
means: whether the Offering answered, heals itself, needs the
operator, is gone, or the attempt measured nothing. It performs no
input or output: no network, no filesystem, no clock read, no
environment read. The current time is always the `now` parameter.

See `.scratch/maintainer-v1/spec.md`, section "Failure classification".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

Bucket = str  # see BUCKETS
Reason = str  # see REASONS

ANSWERED = "answered"
SELF_HEALING = "self_healing"
NEEDS_OPERATOR = "needs_operator"
GONE = "gone"
INCONCLUSIVE = "inconclusive"

BUCKETS = (ANSWERED, SELF_HEALING, NEEDS_OPERATOR, GONE, INCONCLUSIVE)

# `reason` names the CONDITION classify read. `bucket` names the
# CONSEQUENCE (what the Prober or reduce must do about it). One
# condition can produce more than one bucket: a quota exhaustion is
# `self_healing` when the message states a non-zero limit and
# `needs_operator` when the limit is zero, but both are the same
# condition, so both carry `QUOTA_EXHAUSTED`. A rate limit is
# `self_healing` when the message states a reset time and
# `inconclusive` when it does not, for the same reason.
REASON_ANSWERED = "answered"
REASON_QUOTA_EXHAUSTED = "quota_exhausted"
REASON_PLAN_ENTITLEMENT_REFUSED = "plan_entitlement_refused"
REASON_AUTHENTICATION_FAILED = "authentication_failed"
REASON_RATE_LIMITED = "rate_limited"
REASON_GATEWAY_ERROR = "gateway_error"
REASON_TIMEOUT = "timeout"
REASON_IDENTIFIER_GONE = "identifier_gone"
REASON_MALFORMED_RESPONSE = "malformed_response"
# The proxy does not serve the Alias asked for. A fault in our own
# pipeline, never the Offering's: the Generated Config on disk is older
# than what this tool believes. Paired with `INCONCLUSIVE`, so it
# Excludes nothing — removing a working Offering because our config is
# stale would be the wrong repair entirely.
#
# It is the only measurement of the proxy's own view that exists. The
# Prober calls providers DIRECTLY, bypassing the proxy, so no Probe can
# ever produce this.
REASON_ALIAS_NOT_SERVED = "alias_not_served"
# Two different conditions, never one. `REASON_UNMEASURED` names an
# attempt that measured nothing -- always paired with `INCONCLUSIVE`
# (CONTEXT.md, "Inconclusive": avoid the word "unknown" for this
# concept). `REASON_UNRECOGNIZED_FAILURE` names a genuine failure whose
# specific condition this module does not recognise. Conflating the two
# would call a measurement that never happened a failure, or a real
# failure a non-event.
#
# `classify` always pairs `REASON_UNRECOGNIZED_FAILURE` with
# `NEEDS_OPERATOR`, because a Probe sends a known-good request and so
# every failure it sees is the Offering's fault. `reduce.journal_outcome`
# re-buckets it to `INCONCLUSIVE` for an observation from real traffic,
# where a client can cause the failure. It keeps the reason: the
# condition is unchanged, only the consequence differs. See ADR 0008.
REASON_UNMEASURED = "unmeasured"
REASON_UNRECOGNIZED_FAILURE = "unrecognized_failure"

REASONS = (
    REASON_ANSWERED,
    REASON_QUOTA_EXHAUSTED,
    REASON_PLAN_ENTITLEMENT_REFUSED,
    REASON_AUTHENTICATION_FAILED,
    REASON_RATE_LIMITED,
    REASON_GATEWAY_ERROR,
    REASON_TIMEOUT,
    REASON_IDENTIFIER_GONE,
    REASON_MALFORMED_RESPONSE,
    REASON_ALIAS_NOT_SERVED,
    REASON_UNMEASURED,
    REASON_UNRECOGNIZED_FAILURE,
)


@dataclass(frozen=True)
class Outcome:
    """The meaning of one provider response.

    `bucket` is one of the values in `BUCKETS`; it names the
    consequence. `reason` is one of the values in `REASONS`; it names
    the condition that produced the bucket. `reset_at` is a
    timezone-aware `datetime`, or `None` when the response states no
    reset time or classify could not parse one.
    """

    bucket: Bucket
    reset_at: datetime | None
    reason: Reason = REASON_UNMEASURED


# A relative delay: "Please retry in 32.368329668s."
_RELATIVE_DELAY_RE = re.compile(r"retry in\s+([0-9]+(?:\.[0-9]+)?)\s*s", re.IGNORECASE)

# A relative reset in hours and optional minutes: "Resets in 16hr 32min".
# Measured 2026-07-27 from an exhausted OpenCode Go plan. Without this the
# reset time is lost, and a quota exhaustion with no reset time cannot
# recover by the clock -- it waits for a Probe against a plan that cannot
# answer for another sixteen hours.
_RELATIVE_HOURS_RE = re.compile(
    r"in\s+([0-9]+)\s*(?:hr|hrs|hour|hours|h)\b(?:\s*([0-9]+)\s*(?:min|mins|minute|minutes|m)\b)?",
    re.IGNORECASE,
)

# A relative reset in minutes alone: "Resets in 45min".
_RELATIVE_MINUTES_RE = re.compile(
    r"in\s+([0-9]+)\s*(?:min|mins|minute|minutes)\b", re.IGNORECASE
)

# An absolute prose time with no year: "07-29 21:45:00 UTC"
_ABSOLUTE_TIME_RE = re.compile(
    r"reset at\s+(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})\s*UTC", re.IGNORECASE
)

# A quota metric line reporting its own limit, e.g. "limit: 0".
_QUOTA_LIMIT_RE = re.compile(r"limit:\s*(\d+)")

# Each word below must name a permanent condition. A word that also
# fits a transient condition sends a working Offering to `gone`, and
# `gone` makes the report advise removal from Policy. The bare word
# "unavailable" was such a word: "temporarily unavailable" matched it.
# Use the longer phrase that carries the permanence.
#
# "deprecated" and "retired" appear here as literal words a real
# provider message uses. CONTEXT.md reserves those two words for the
# Feed's own catalogue status, never for a statement about whether a
# call succeeds -- this list names the call-level outcome
# `REASON_IDENTIFIER_GONE`, not a catalogue status, even though it
# matches a message that happens to use the Feed's words.
_CALL_PERMANENTLY_GONE_WORDS = ("no longer available", "deprecated", "retired")
_REMOVAL_WORDS = ("unavailable for free", "use this slug instead", "model not found")
# "access to model denied" and "eligible for using" are the Qwen Token
# Plan's wording for a model the account's plan does not include.
# Measured 2026-07-26: it returns HTTP 403, so without these words the
# status rule reads it as `authentication_failed`. Both are
# `needs_operator`, so behaviour does not change, but the REPORT would
# send the operator to check a credential that is correct. The bucket
# names the consequence; the reason must still name the real condition.
# The PROXY's own refusal when it does not serve the Alias asked for.
# Not a provider condition at all: the Offering is fine and the Generated
# Config on disk is older than what this tool believes.
#
# Matched narrowly, on the proxy's own sentence. A provider saying
# "invalid model" about ITS model id is a different condition, and
# reading that as not-served would blame the proxy for a vendor refusal.
# litellm's wording, measured 2026-07-31:
#
#   400: {'error': 'anthropic_messages: Invalid model name passed in
#   model=<alias>. Call `/v1/models` to view available models for your
#   key.'}
_NOT_SERVED_WORDS = ("invalid model name passed in",)

_ENTITLEMENT_WORDS = (
    "not supported on",
    "model_not_supported",
    "invalid model format",
    "access to model denied",
    "eligible for using",
)
# A quota exhaustion does not always say "quota". Measured 2026-07-27:
# an exhausted OpenCode Go plan answered "Monthly usage limit reached.
# Resets in 16hr 32min." with HTTP 429. No rule matched, so it fell
# through to the bare-429 rule and read as `rate_limited` with no reset
# time -- `inconclusive`, which changes nothing. An Offering that could
# not answer for 16 hours stayed in the Generated Config.
#
# Each phrase must name a SPENT ALLOWANCE, never a request rate. "usage
# limit" and "monthly limit" describe what the plan includes; "rate
# limit" and "request limit" describe how fast we are calling, and those
# belong in `_RATE_LIMIT_WORDS`, where an unstated reset reads as our own
# doing.
_QUOTA_WORDS = ("quota", "usage limit", "monthly limit", "credit balance")
_GATEWAY_WORDS = (
    "internal server error",
    "bad gateway",
    "gateway error",
    "gateway time-out",
    "gateway timeout",
    "service unavailable",
    "temporarily unavailable",
)
# Match the hyphen and the underscore as well as the space. Measured
# 2026-08-21: Cline relays OpenRouter's "is temporarily rate-limited
# upstream", and "rate limit" does not match "rate-limited". One hyphen
# sent a five-second rate limit to `needs_operator` as
# `unrecognized_failure`, which Excluded the Offering and asked a human
# to read a message that states its own retry delay.
_RATE_LIMIT_WORDS = (
    "rate limit",
    "rate-limit",
    "rate_limit",
    "request limit",
    "too many requests",
)

# HTTP statuses that decide the bucket when the body states no message.
# A 502 with an HTML body, which the audit saw from Cloudflare, reads
# as an empty message and must not fall through to `needs_operator`.
_RATE_LIMIT_STATUS = 429
_OPERATOR_STATUSES = (401, 402, 403)


def classify(
    *,
    provider: str,
    http_status: int | None,
    body: object,
    transport: str | None = None,
    now: datetime,
) -> Outcome:
    """State what a provider response means.

    Read `body` and `http_status` for one provider call. `transport`
    carries a transport-level condition such as `"timeout"` when no
    response body exists. `now` resolves a reset time stated with no
    year. Return an `Outcome`. Never raise.

    classify accepts a success as well as a failure, because a success
    is not always readable. A Probe hands its response here whatever
    happened, and reads the bucket:

    - `answered`: the response carries a completion. The Probe
      succeeded, and the Offering clears its exclusion.
    - `self_healing`, `needs_operator`, `gone`: the Probe failed.
    - `inconclusive`: the attempt measured nothing. Health State stays
      as it is.

    A response with a success HTTP status but no top-level `choices`
    is a malformed success, not an answer. It returns `needs_operator`.
    """
    if transport:
        # A transport-level condition produces no body. A timeout, a
        # reset connection and a DNS failure all heal themselves.
        return Outcome(bucket=SELF_HEALING, reset_at=None, reason=REASON_TIMEOUT)

    is_http_failure = http_status is not None and not (200 <= http_status < 300)
    is_body_failure = _is_body_failure(body)

    if not is_http_failure and not is_body_failure:
        if http_status is None:
            # No status, no body failure, no transport condition. The
            # attempt states nothing.
            return Outcome(bucket=INCONCLUSIVE, reset_at=None, reason=REASON_UNMEASURED)
        if _is_malformed_success(http_status, body):
            return Outcome(
                bucket=NEEDS_OPERATOR, reset_at=None, reason=REASON_MALFORMED_RESPONSE
            )
        return Outcome(bucket=ANSWERED, reset_at=None, reason=REASON_ANSWERED)

    message = _extract_message(body)
    lower = message.lower()

    if any(word in lower for word in _CALL_PERMANENTLY_GONE_WORDS) or any(
        word in lower for word in _REMOVAL_WORDS
    ):
        return Outcome(bucket=GONE, reset_at=None, reason=REASON_IDENTIFIER_GONE)

    # Checked before every provider rule. This is the proxy talking
    # about itself, so no provider condition applies, and the Offering
    # must not be Excluded for a fault in our own pipeline.
    if any(word in lower for word in _NOT_SERVED_WORDS):
        return Outcome(bucket=INCONCLUSIVE, reset_at=None, reason=REASON_ALIAS_NOT_SERVED)

    if any(word in lower for word in _ENTITLEMENT_WORDS):
        return Outcome(
            bucket=NEEDS_OPERATOR,
            reset_at=None,
            reason=REASON_PLAN_ENTITLEMENT_REFUSED,
        )

    if any(word in lower for word in _QUOTA_WORDS):
        # A limit of zero means the plan does not include the model,
        # so no wait clears it. A transient quota states a non-zero
        # limit and is self-healing. Both are the same condition, a
        # quota exhaustion, so both carry the same reason.
        quota_limit = _lowest_quota_limit(message)
        if quota_limit == 0:
            return Outcome(
                bucket=NEEDS_OPERATOR, reset_at=None, reason=REASON_QUOTA_EXHAUSTED
            )
        reset_at = _parse_reset_time(message, now=now)
        return Outcome(
            bucket=SELF_HEALING, reset_at=reset_at, reason=REASON_QUOTA_EXHAUSTED
        )

    if any(word in lower for word in _GATEWAY_WORDS):
        return Outcome(bucket=SELF_HEALING, reset_at=None, reason=REASON_GATEWAY_ERROR)

    if any(word in lower for word in _RATE_LIMIT_WORDS):
        return _rate_limit_outcome(message, now=now)

    by_status = _classify_by_status(http_status, message, now=now)
    if by_status is not None:
        return by_status

    # A failure we do not recognise. The operator must look.
    return Outcome(bucket=NEEDS_OPERATOR, reset_at=None, reason=REASON_UNRECOGNIZED_FAILURE)


def _rate_limit_outcome(message: str, *, now: datetime) -> Outcome:
    """Build the Outcome for a rate-limit-shaped condition.

    A stated reset time makes this `self_healing`. No stated reset time
    is attributable to our own request rate; the attempt measured
    nothing, so it is `inconclusive`. Shared by the message-text rule
    and the bare-429-status rule in `_classify_by_status`, so the two
    never drift apart.
    """
    reset_at = _parse_reset_time(message, now=now)
    if reset_at is None:
        return Outcome(bucket=INCONCLUSIVE, reset_at=None, reason=REASON_RATE_LIMITED)
    return Outcome(bucket=SELF_HEALING, reset_at=reset_at, reason=REASON_RATE_LIMITED)


def _classify_by_status(
    http_status: int | None, message: str, *, now: datetime
) -> Outcome | None:
    """Choose a bucket from the HTTP status alone.

    Call this only after the message text decided nothing. A gateway
    returns 502 with an HTML page, and a provider returns 429 with an
    empty body. Both state the condition in the status and nowhere
    else. Return `None` when the status states nothing either.
    """
    if http_status is None:
        return None
    if http_status == _RATE_LIMIT_STATUS:
        return _rate_limit_outcome(message, now=now)
    if 500 <= http_status < 600:
        return Outcome(bucket=SELF_HEALING, reset_at=None, reason=REASON_GATEWAY_ERROR)
    if http_status in _OPERATOR_STATUSES:
        return Outcome(
            bucket=NEEDS_OPERATOR, reset_at=None, reason=REASON_AUTHENTICATION_FAILED
        )
    return None


def _is_body_failure(body: object) -> bool:
    """State whether the body itself signals a failure.

    A body that carries an error, or a success flag that is false, is
    a failure whatever the HTTP status reports. The error value can be
    an object with a message, or a plain string.
    """
    if not isinstance(body, dict):
        return False
    if body.get("error") is not None:
        return True
    return body.get("success") is False


def _extract_message(body: object) -> str:
    """Read the failure message text out of a response body.

    Accept an `error` value that is an object with a `message`, or a
    plain string. Fall back to a top-level `message` key for a body
    with no `error` key, one shape the same plan returns. Return an
    empty string when no message is found. Never raise.
    """
    if not isinstance(body, dict):
        return ""
    error = body.get("error")
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        message = error.get("message")
        return "" if message is None else str(message)
    message = body.get("message")
    return "" if message is None else str(message)


def _lowest_quota_limit(message: str) -> int | None:
    """Return the lowest `limit:` value quoted in a quota message."""
    limits = [int(m) for m in _QUOTA_LIMIT_RE.findall(message)]
    if not limits:
        return None
    return min(limits)


def _is_malformed_success(http_status: int | None, body: object) -> bool:
    """State whether a non-failing response is unreadable.

    A response with a success-range HTTP status but no top-level
    `choices` carries no completion a plain client can read. This is
    the shape one provider's envelope produces.
    """
    if http_status is None or not (200 <= http_status < 300):
        return False
    return not isinstance(body, dict) or "choices" not in body


def _parse_reset_time(message: str, *, now: datetime) -> datetime | None:
    """Parse a reset time out of a provider's prose message.

    Handle a relative delay ("retry in 32.368329668s") and an absolute
    time with no year ("07-29 21:45:00 UTC"). Resolve a missing year
    from `now`, choosing the year that puts the reset time in the
    future, because a quota resets forward. Return `None` when the
    message states no reset time or it cannot be parsed. Never raise.
    """
    relative = _RELATIVE_DELAY_RE.search(message)
    if relative:
        try:
            seconds = float(relative.group(1))
        except ValueError:
            return None
        return now + timedelta(seconds=seconds)

    # Try hours before minutes. "in 16hr 32min" matches both patterns,
    # and the minutes-only one would read it as 32 minutes away -- a
    # reset time sixteen hours too early, which invites a call against
    # a plan that still cannot answer.
    hours = _RELATIVE_HOURS_RE.search(message)
    if hours:
        try:
            delta = timedelta(
                hours=int(hours.group(1)), minutes=int(hours.group(2) or 0)
            )
        except ValueError:
            return None
        return now + delta

    minutes = _RELATIVE_MINUTES_RE.search(message)
    if minutes:
        try:
            return now + timedelta(minutes=int(minutes.group(1)))
        except ValueError:
            return None

    absolute = _ABSOLUTE_TIME_RE.search(message)
    if absolute:
        month, day, hour, minute, second = (int(g) for g in absolute.groups())
        try:
            candidate = datetime(
                now.year, month, day, hour, minute, second, tzinfo=timezone.utc
            )
        except ValueError:
            return None
        if candidate <= now:
            try:
                candidate = candidate.replace(year=now.year + 1)
            except ValueError:
                return None
        return candidate

    return None
