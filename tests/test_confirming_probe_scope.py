"""A journal-triggered run must confirm what one event cannot attribute.

A run the Observation Journal triggered probes almost nothing. That is
right for a self-identifying failure: `classify` already read the
condition from the provider's own message, so a Probe would spend a call
to re-learn a fact. It is wrong where one event cannot say whether the
Offering is at fault.

Three conditions qualify, on two separate arguments.

A timeout states nothing. No response arrived, so there is no message to
read. `classify` gives it `self_healing` with the reason `timeout`, which
shares a bucket with a quota exhaustion, a gateway error and a rate limit
-- three conditions the provider DID state. Reading the bucket alone
treated the least self-identifying failure there is as self-identifying.

An authentication failure states plenty, and still not enough. A 401 says
the provider refused one request; it does not say the credential is
invalid. `reduce._PASSTHROUGH_EXEMPT_REASONS` already holds this reason
for the case where the caller owns the credential.

Measured 2026-07-31, both on this machine. Two timeouts on
`claude-chatgpt1-gpt-5.6-sol` Excluded it with no Probe, and it answered a
Probe both before and after. Then
`qwencloud-token-plan:qwen3.8-max-preview` Excluded on
`authentication_failed` while the same key answered ten Probes out of ten
and its five pool siblings never stopped answering.

Neither reason carries a reset time, so `reduce._apply_reset_expiry`
cannot clear either exclusion by the clock. Only a Probe can, and the run
that Excluded the timeout also postponed the next sweep by 55 minutes.

These tests pin the scope of the confirming worklist, and the precedence
that makes a confirming Probe worth running at all.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import litellm_maintainer.cli as cli_module
from litellm_maintainer.classify import (
    ANSWERED,
    INCONCLUSIVE,
    NEEDS_OPERATOR,
    REASON_ANSWERED,
    REASON_AUTHENTICATION_FAILED,
    REASON_GATEWAY_ERROR,
    REASON_QUOTA_EXHAUSTED,
    REASON_TIMEOUT,
    REASON_UNMEASURED,
    SELF_HEALING,
    Outcome,
)
from litellm_maintainer.feed import parse_feed
from litellm_maintainer.policy import parse_policy
from litellm_maintainer.prober import build_worklist
from litellm_maintainer.reduce import HealthState, Observation, reduce

NOW = datetime(2026, 7, 31, 14, 0, 0, tzinfo=timezone.utc)
EARLIER = NOW - timedelta(minutes=2)

SEAT1_SOL = "claude-chatgpt1-gpt-5.6-sol"
SEAT1_TERRA = "claude-chatgpt1-gpt-5.6-terra"
DIRECT_CLAUDE = "claude-opus-5"

TIMEOUT = Outcome(bucket=SELF_HEALING, reset_at=None, reason=REASON_TIMEOUT)
GATEWAY_ERROR = Outcome(bucket=SELF_HEALING, reset_at=None, reason=REASON_GATEWAY_ERROR)
QUOTA_WITH_RESET = Outcome(
    bucket=SELF_HEALING, reset_at=NOW + timedelta(hours=4), reason=REASON_QUOTA_EXHAUSTED
)
QUOTA_AT_ZERO = Outcome(
    bucket=NEEDS_OPERATOR, reset_at=None, reason=REASON_QUOTA_EXHAUSTED
)
# What `classify` returns for HTTP 401, 402 and 403 (`_OPERATOR_STATUSES`).
AUTH_FAILED = Outcome(
    bucket=NEEDS_OPERATOR, reset_at=None, reason=REASON_AUTHENTICATION_FAILED
)
UNMEASURED = Outcome(bucket=INCONCLUSIVE, reset_at=None, reason=REASON_UNMEASURED)
PROBE_ANSWERED = Outcome(bucket=ANSWERED, reset_at=None, reason=REASON_ANSWERED)


def _feed():
    return parse_feed({"schema_version": "1", "providers": [], "models": []})


def _policy():
    """Two seat Aliases on a local worker, and one Passthrough Auth Alias.

    The two seats mirror the shape the timeout was measured on: one
    worker serving several models, where the slowest one times out and
    its siblings do not.
    """
    return parse_policy(
        {
            "providers": {},
            "quality": {"minimum_coding_score": 18},
            "approved_candidates": [],
            "naming": {
                "alias_prefix": "claude-",
                "provider_labels": {},
                "alias_overrides": {},
            },
            "withheld": {},
            "declared": [
                {
                    "alias": SEAT1_SOL,
                    "litellm_params": {
                        "model": "openai/claude-gpt-5.6-sol",
                        "api_base": "http://127.0.0.1:4011/v1",
                        "api_key": "os.environ/SEAT1_WORKER_KEY",
                    },
                },
                {
                    "alias": SEAT1_TERRA,
                    "litellm_params": {
                        "model": "openai/claude-gpt-5.6-terra",
                        "api_base": "http://127.0.0.1:4011/v1",
                        "api_key": "os.environ/SEAT1_WORKER_KEY",
                    },
                },
                {
                    "alias": DIRECT_CLAUDE,
                    "passthrough_auth": True,
                    "litellm_params": {"model": "anthropic/claude-opus-5"},
                },
            ],
            "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
            "schedule": {
                "enabled": True,
                "interval_minutes": 60,
                "require_proxy": True,
                "maximum_staleness_hours": 24,
            },
            "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
        }
    )


def _narrow(*observations: Observation, health: HealthState | None = None):
    feed = _feed()
    policy = _policy()
    prior = health or HealthState(offerings={})
    worklist = build_worklist(feed=feed, policy=policy, health=prior, now=NOW)
    return cli_module._confirming_worklist(
        worklist,
        feed=feed,
        policy=policy,
        health=prior,
        observations=list(observations),
        now=NOW,
    )


def _observed(alias: str, outcome: Outcome) -> Observation:
    return Observation(offering_id=alias, observed_at=EARLIER, outcome=outcome)


# --- which conditions need confirming ------------------------------------


def test_a_timeout_from_real_traffic_gets_one_confirming_probe():
    narrowed = _narrow(_observed(SEAT1_SOL, TIMEOUT))

    assert [target.key for target in narrowed.targets] == [SEAT1_SOL]


def test_an_inconclusive_observation_still_gets_one_confirming_probe():
    narrowed = _narrow(_observed(SEAT1_SOL, UNMEASURED))

    assert [target.key for target in narrowed.targets] == [SEAT1_SOL]


def test_a_gateway_error_needs_no_probe_though_it_shares_the_timeouts_bucket():
    """Read the reason, never the bucket.

    `self_healing` carries both. A gateway error quotes the gateway's
    own message, so probing it re-learns a stated fact.
    """
    narrowed = _narrow(_observed(SEAT1_SOL, GATEWAY_ERROR))

    assert narrowed.targets == ()


def test_a_quota_exhaustion_needs_no_probe_whichever_bucket_it_carries():
    with_reset = _narrow(_observed(SEAT1_SOL, QUOTA_WITH_RESET))
    at_zero = _narrow(_observed(SEAT1_SOL, QUOTA_AT_ZERO))

    assert with_reset.targets == ()
    assert at_zero.targets == ()


def test_an_authentication_failure_from_real_traffic_gets_one_confirming_probe():
    """A 401 states that the provider refused one request. It does not
    state that the credential is invalid, and those two readings call for
    opposite actions.

    Measured 2026-07-31 on `qwencloud-token-plan:qwen3.8-max-preview`:
    Excluded on `authentication_failed`, while the same credential
    answered ten Probes out of ten and its five pool siblings never
    stopped answering.
    """
    narrowed = _narrow(_observed(SEAT1_SOL, AUTH_FAILED))

    assert [target.key for target in narrowed.targets] == [SEAT1_SOL]


def test_an_authentication_failure_probes_that_offering_alone():
    """A credential is shared, so a real revocation would take every
    sibling down and the ordinary sweep would find them. Probing all of
    them here would spend the window to confirm one event."""
    narrowed = _narrow(_observed(SEAT1_SOL, AUTH_FAILED))

    assert SEAT1_TERRA not in {target.key for target in narrowed.targets}


def test_an_authentication_failure_on_a_passthrough_offering_reaches_no_probe():
    """`reduce._PASSTHROUGH_EXEMPT_REASONS` already exempts this reason,
    so such an observation Excludes nothing and needs no Probe. A Probe
    would also carry the proxy's credential and measure the wrong
    thing."""
    narrowed = _narrow(_observed(DIRECT_CLAUDE, AUTH_FAILED))

    assert narrowed.targets == ()


def test_a_timeout_probes_that_offering_alone_and_not_its_siblings():
    """The sibling on the same worker was not measured, so it is not
    confirmed. A worker serving six models times out on the slowest one
    only, and probing all six would spend the seat's window to learn
    nothing about five of them."""
    narrowed = _narrow(_observed(SEAT1_SOL, TIMEOUT))

    assert SEAT1_TERRA not in {target.key for target in narrowed.targets}


def test_a_timeout_on_a_passthrough_offering_reaches_no_probe():
    """A Passthrough Auth Offering is never probed: a Probe would carry
    the proxy's credential and measure the wrong thing. So it cannot be
    confirmed, and asking for it must still not fabricate a target."""
    narrowed = _narrow(_observed(DIRECT_CLAUDE, TIMEOUT))

    assert narrowed.targets == ()


# --- what narrowing may not touch ----------------------------------------


def test_narrowing_keeps_the_admitted_set_and_the_passthrough_set_whole():
    """`reduce` prunes a record for an Offering `admitted` no longer
    holds, and applies the Passthrough Auth exemption from
    `skipped_passthrough`. Narrowing either would drop health for every
    Offering outside this one trigger."""
    feed = _feed()
    policy = _policy()
    prior = HealthState(offerings={})
    whole = build_worklist(feed=feed, policy=policy, health=prior, now=NOW)

    narrowed = _narrow(_observed(SEAT1_SOL, TIMEOUT))

    assert narrowed.admitted == whole.admitted
    assert narrowed.skipped_passthrough == whole.skipped_passthrough
    assert DIRECT_CLAUDE in narrowed.skipped_passthrough


# --- why the Probe is worth running --------------------------------------


def test_a_probe_that_answers_overrules_the_timeout_that_asked_for_it():
    """The whole point of the confirming Probe.

    `reduce` applies a Probe outcome last, because it carries `now` and
    a Journal entry reports something already past. So the Offering
    keeps its place in the Generated Config, and this is the case that
    Excluded `claude-chatgpt1-gpt-5.6-sol` on 2026-07-31.
    """
    next_health = reduce(
        prior=HealthState(offerings={}),
        outcomes={SEAT1_SOL: PROBE_ANSWERED},
        observations=[_observed(SEAT1_SOL, TIMEOUT)],
        admitted={SEAT1_SOL, SEAT1_TERRA, DIRECT_CLAUDE},
        passthrough_auth=frozenset({DIRECT_CLAUDE}),
        now=NOW,
    )

    record = next_health.offerings[SEAT1_SOL]
    assert record.excluded is False
    assert record.failure_count == 0
    assert record.last_success_at == NOW


def test_a_probe_that_times_out_too_excludes_on_two_measurements():
    """A confirming Probe sends a known-good request of eight tokens. A
    timeout on that as well is the Offering's own condition, so the
    exclusion stands and the count records both events."""
    next_health = reduce(
        prior=HealthState(offerings={}),
        outcomes={SEAT1_SOL: TIMEOUT},
        observations=[_observed(SEAT1_SOL, TIMEOUT)],
        admitted={SEAT1_SOL, SEAT1_TERRA, DIRECT_CLAUDE},
        passthrough_auth=frozenset({DIRECT_CLAUDE}),
        now=NOW,
    )

    record = next_health.offerings[SEAT1_SOL]
    assert record.excluded is True
    assert record.reason == REASON_TIMEOUT
    assert record.failure_count == 2


def test_a_probe_that_answers_overrules_the_auth_failure_that_asked_for_it():
    """The qwen3.8 case of 2026-07-31. The credential was valid, so the
    Offering keeps its place rather than waiting for an operator."""
    next_health = reduce(
        prior=HealthState(offerings={}),
        outcomes={SEAT1_SOL: PROBE_ANSWERED},
        observations=[_observed(SEAT1_SOL, AUTH_FAILED)],
        admitted={SEAT1_SOL, SEAT1_TERRA, DIRECT_CLAUDE},
        passthrough_auth=frozenset({DIRECT_CLAUDE}),
        now=NOW,
    )

    record = next_health.offerings[SEAT1_SOL]
    assert record.excluded is False
    assert record.failure_count == 0
    assert record.last_success_at == NOW


def test_a_probe_that_fails_auth_too_excludes_on_two_measurements():
    """A genuinely revoked credential still Excludes the Offering. The
    Probe carries the proxy's own credential, so a second 401 attributes
    the failure to the Offering rather than to one caller.

    `needs_operator` carries no reset time, which is correct here: a
    revoked credential does not refill, and only the operator clears it.
    """
    next_health = reduce(
        prior=HealthState(offerings={}),
        outcomes={SEAT1_SOL: AUTH_FAILED},
        observations=[_observed(SEAT1_SOL, AUTH_FAILED)],
        admitted={SEAT1_SOL, SEAT1_TERRA, DIRECT_CLAUDE},
        passthrough_auth=frozenset({DIRECT_CLAUDE}),
        now=NOW,
    )

    record = next_health.offerings[SEAT1_SOL]
    assert record.excluded is True
    assert record.reason == REASON_AUTHENTICATION_FAILED
    assert record.reset_at is None
    assert record.failure_count == 2
