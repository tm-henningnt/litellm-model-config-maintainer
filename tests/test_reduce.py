"""Tests for `litellm_maintainer.reduce`.

Each test name states a rule an operator would recognise, in the
glossary's vocabulary (CONTEXT.md): Excluded, Withheld, Candidate and
Inconclusive are four different things.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

from litellm_maintainer.classify import Outcome
from litellm_maintainer.reduce import (
    HealthState,
    Observation,
    OfferingHealth,
    reduce,
)

NOW = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)
PAST = NOW - timedelta(hours=1)
FUTURE = NOW + timedelta(hours=1)

OFFERING = "opencode-go:glm-5.2"


def _state(**offerings: OfferingHealth) -> HealthState:
    return HealthState(offerings=dict(offerings))


def _excluded(
    bucket: str = "self_healing", reset_at: datetime | None = None, failure_count: int = 1
) -> OfferingHealth:
    return OfferingHealth(
        excluded=True,
        reason="prior failure",
        bucket=bucket,
        reset_at=reset_at,
        last_success_at=PAST - timedelta(days=1),
        last_attempt_at=PAST,
        failure_count=failure_count,
    )


def test_an_inconclusive_probe_does_not_evict_a_healthy_offering():
    healthy = OfferingHealth(
        excluded=False,
        last_success_at=PAST,
        last_attempt_at=PAST,
        failure_count=0,
    )
    prior = _state(**{OFFERING: healthy})
    outcomes = {OFFERING: Outcome(bucket="inconclusive", reset_at=None)}

    result = reduce(
        prior=prior,
        outcomes=outcomes,
        observations=[],
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    # Health is untouched. `inconclusive_count` is the one deliberate
    # exception, and it exists to make a silent misclassification
    # visible (see `OfferingHealth`).
    assert replace(result.offerings[OFFERING], inconclusive_count=0) == healthy
    assert result.offerings[OFFERING].inconclusive_count == 1


def test_a_successful_probe_clears_an_exclusion():
    prior = _state(**{OFFERING: _excluded()})
    outcomes = {OFFERING: Outcome(bucket="answered", reset_at=None)}

    result = reduce(
        prior=prior,
        outcomes=outcomes,
        observations=[],
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    record = result.offerings[OFFERING]
    assert record.excluded is False
    assert record.reason is None
    assert record.bucket is None
    assert record.last_success_at == NOW
    assert record.failure_count == 0


def test_an_answered_outcome_for_an_offering_that_was_never_excluded_leaves_it_not_excluded_and_updates_last_success_at():
    healthy = OfferingHealth(excluded=False, last_success_at=PAST, last_attempt_at=PAST)
    prior = _state(**{OFFERING: healthy})
    outcomes = {OFFERING: Outcome(bucket="answered", reset_at=None)}

    result = reduce(
        prior=prior,
        outcomes=outcomes,
        observations=[],
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    record = result.offerings[OFFERING]
    assert record.excluded is False
    assert record.last_success_at == NOW


def test_a_reset_time_that_has_passed_clears_an_exclusion_with_no_probe():
    prior = _state(**{OFFERING: _excluded(reset_at=PAST)})

    result = reduce(
        prior=prior,
        outcomes={},
        observations=[],
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    record = result.offerings[OFFERING]
    assert record.excluded is False
    assert record.reset_at is None


def test_a_reset_time_exactly_equal_to_now_clears_the_exclusion():
    prior = _state(**{OFFERING: _excluded(reset_at=NOW)})

    result = reduce(
        prior=prior,
        outcomes={},
        observations=[],
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    assert result.offerings[OFFERING].excluded is False


def test_a_reset_time_still_in_the_future_keeps_the_exclusion():
    prior = _state(**{OFFERING: _excluded(reset_at=FUTURE)})

    result = reduce(
        prior=prior,
        outcomes={},
        observations=[],
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    record = result.offerings[OFFERING]
    assert record.excluded is True
    assert record.reset_at == FUTURE


def test_a_transient_quota_failure_on_a_passthrough_auth_offering_does_not_exclude_it_but_is_recorded():
    # This is the case the old code got wrong: a transient quota states
    # a non-zero limit and a reset time, so classify gives it
    # `self_healing`, not `needs_operator`. The old exemption only
    # checked the bucket, so it wrongly Excluded this Offering. Per
    # tests/fixtures/classify/qwen-quota-openai.json and CAPTURE.md.
    prior = _state(**{OFFERING: OfferingHealth()})
    outcomes = {
        OFFERING: Outcome(bucket="self_healing", reset_at=FUTURE, reason="quota_exhausted")
    }

    result = reduce(
        prior=prior,
        outcomes=outcomes,
        observations=[],
        admitted={OFFERING},
        passthrough_auth={OFFERING},
        now=NOW,
    )

    record = result.offerings[OFFERING]
    assert record.excluded is False
    assert record.bucket == "self_healing"
    assert record.reason == "quota_exhausted"
    assert record.reset_at == FUTURE
    assert record.failure_count == 1


def test_a_zero_limit_quota_refusal_on_a_passthrough_auth_offering_does_not_exclude_it():
    # A zero-quota entitlement refusal classifies as `needs_operator`,
    # per tests/fixtures/classify/gemini-quota.json and CAPTURE.md. It
    # carries the same reason as the transient case above, whatever
    # the bucket.
    prior = _state(**{OFFERING: OfferingHealth()})
    outcomes = {
        OFFERING: Outcome(bucket="needs_operator", reset_at=None, reason="quota_exhausted")
    }

    result = reduce(
        prior=prior,
        outcomes=outcomes,
        observations=[],
        admitted={OFFERING},
        passthrough_auth={OFFERING},
        now=NOW,
    )

    record = result.offerings[OFFERING]
    assert record.excluded is False
    assert record.bucket == "needs_operator"
    assert record.reason == "quota_exhausted"
    assert record.failure_count == 1


def test_an_authentication_failure_on_a_passthrough_auth_offering_does_not_exclude_it():
    # An HTTP 401/403 with no matching message text classifies as
    # `needs_operator` with reason `authentication_failed`. See
    # classify.py, `_OPERATOR_STATUSES`.
    prior = _state(**{OFFERING: OfferingHealth()})
    outcomes = {
        OFFERING: Outcome(
            bucket="needs_operator", reset_at=None, reason="authentication_failed"
        )
    }

    result = reduce(
        prior=prior,
        outcomes=outcomes,
        observations=[],
        admitted={OFFERING},
        passthrough_auth={OFFERING},
        now=NOW,
    )

    record = result.offerings[OFFERING]
    assert record.excluded is False
    assert record.reason == "authentication_failed"


def test_a_gateway_failure_on_a_passthrough_auth_offering_does_exclude_it():
    prior = _state(**{OFFERING: OfferingHealth()})
    outcomes = {
        OFFERING: Outcome(bucket="self_healing", reset_at=None, reason="gateway_error")
    }

    result = reduce(
        prior=prior,
        outcomes=outcomes,
        observations=[],
        admitted={OFFERING},
        passthrough_auth={OFFERING},
        now=NOW,
    )

    record = result.offerings[OFFERING]
    assert record.excluded is True
    assert record.bucket == "self_healing"
    assert record.reason == "gateway_error"


def test_a_timeout_on_a_passthrough_auth_offering_does_exclude_it():
    prior = _state(**{OFFERING: OfferingHealth()})
    outcomes = {OFFERING: Outcome(bucket="self_healing", reset_at=None, reason="timeout")}

    result = reduce(
        prior=prior,
        outcomes=outcomes,
        observations=[],
        admitted={OFFERING},
        passthrough_auth={OFFERING},
        now=NOW,
    )

    record = result.offerings[OFFERING]
    assert record.excluded is True
    assert record.reason == "timeout"


def test_a_rate_limit_on_a_passthrough_auth_offering_does_exclude_it():
    prior = _state(**{OFFERING: OfferingHealth()})
    outcomes = {
        OFFERING: Outcome(bucket="self_healing", reset_at=FUTURE, reason="rate_limited")
    }

    result = reduce(
        prior=prior,
        outcomes=outcomes,
        observations=[],
        admitted={OFFERING},
        passthrough_auth={OFFERING},
        now=NOW,
    )

    record = result.offerings[OFFERING]
    assert record.excluded is True
    assert record.reason == "rate_limited"


def test_a_needs_operator_failure_on_a_non_passthrough_offering_excludes_it():
    prior = _state(**{OFFERING: OfferingHealth()})
    outcomes = {
        OFFERING: Outcome(
            bucket="needs_operator", reset_at=None, reason="plan_entitlement_refused"
        )
    }

    result = reduce(
        prior=prior,
        outcomes=outcomes,
        observations=[],
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    assert result.offerings[OFFERING].excluded is True


def test_a_quota_exhaustion_on_an_ordinary_offering_still_excludes_it():
    # The exemption is Passthrough Auth only. An ordinary Offering
    # (one whose credentials come from the proxy) is still Excluded by
    # a quota exhaustion.
    prior = _state(**{OFFERING: OfferingHealth()})
    outcomes = {
        OFFERING: Outcome(bucket="self_healing", reset_at=FUTURE, reason="quota_exhausted")
    }

    result = reduce(
        prior=prior,
        outcomes=outcomes,
        observations=[],
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    assert result.offerings[OFFERING].excluded is True


def test_an_authentication_failure_on_an_ordinary_offering_still_excludes_it():
    prior = _state(**{OFFERING: OfferingHealth()})
    outcomes = {
        OFFERING: Outcome(
            bucket="needs_operator", reset_at=None, reason="authentication_failed"
        )
    }

    result = reduce(
        prior=prior,
        outcomes=outcomes,
        observations=[],
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    assert result.offerings[OFFERING].excluded is True


def test_a_gone_outcome_excludes_the_offering_and_names_the_bucket():
    prior = _state(**{OFFERING: OfferingHealth()})
    outcomes = {OFFERING: Outcome(bucket="gone", reset_at=None, reason="identifier_gone")}

    result = reduce(
        prior=prior,
        outcomes=outcomes,
        observations=[],
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    record = result.offerings[OFFERING]
    assert record.excluded is True
    assert record.bucket == "gone"
    assert record.reason == "identifier_gone"


def test_the_health_state_record_keeps_the_reason_alongside_the_bucket():
    # Ticket 13 must report "Excluded with the reason and expected
    # return", so the record needs the classify reason, not just the
    # bucket.
    prior = _state(**{OFFERING: OfferingHealth()})
    outcomes = {
        OFFERING: Outcome(bucket="self_healing", reset_at=FUTURE, reason="gateway_error")
    }

    result = reduce(
        prior=prior,
        outcomes=outcomes,
        observations=[],
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    record = result.offerings[OFFERING]
    assert record.reason == "gateway_error"
    assert record.reset_at == FUTURE


def test_a_journal_observation_and_a_probe_outcome_for_the_same_offering_resolve_to_one_record():
    # Precedence: apply events in chronological order, oldest first,
    # the Probe outcome last (it always carries `now`, the latest
    # timestamp). The last event wins. Here a Journal entry reports an
    # older failure and the Probe, run just now, confirms recovery.
    prior = _state(**{OFFERING: OfferingHealth()})
    observations = [
        Observation(
            offering_id=OFFERING,
            observed_at=PAST,
            outcome=Outcome(bucket="self_healing", reset_at=None),
        )
    ]
    outcomes = {OFFERING: Outcome(bucket="answered", reset_at=None)}

    result = reduce(
        prior=prior,
        outcomes=outcomes,
        observations=observations,
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    assert len(result.offerings) == 1
    record = result.offerings[OFFERING]
    assert record.excluded is False
    assert record.last_success_at == NOW
    # The Journal event still counted as an attempt on its way through.
    assert record.failure_count == 0


def test_a_record_for_an_offering_policy_no_longer_admits_is_discarded():
    prior = _state(**{OFFERING: _excluded()})

    result = reduce(
        prior=prior,
        outcomes={},
        observations=[],
        admitted=set(),
        passthrough_auth=set(),
        now=NOW,
    )

    assert OFFERING not in result.offerings


def test_a_first_ever_observation_of_an_offering_with_no_prior_record():
    prior = _state()
    observations = [
        Observation(
            offering_id=OFFERING,
            observed_at=PAST,
            outcome=Outcome(bucket="self_healing", reset_at=FUTURE),
        )
    ]

    result = reduce(
        prior=prior,
        outcomes={},
        observations=observations,
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    record = result.offerings[OFFERING]
    assert record.excluded is True
    assert record.reset_at == FUTURE
    assert record.failure_count == 1


def test_an_offering_that_is_excluded_then_answers_then_fails_again():
    prior = _state(**{OFFERING: _excluded()})

    first = reduce(
        prior=prior,
        outcomes={OFFERING: Outcome(bucket="answered", reset_at=None)},
        observations=[],
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )
    assert first.offerings[OFFERING].excluded is False

    second = reduce(
        prior=first,
        outcomes={OFFERING: Outcome(bucket="self_healing", reset_at=FUTURE)},
        observations=[],
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW + timedelta(minutes=5),
    )

    record = second.offerings[OFFERING]
    assert record.excluded is True
    assert record.reset_at == FUTURE
    assert record.failure_count == 1


def test_two_journal_observations_for_the_same_offering_in_one_run():
    prior = _state(**{OFFERING: OfferingHealth()})
    observations = [
        Observation(
            offering_id=OFFERING,
            observed_at=PAST,
            outcome=Outcome(bucket="self_healing", reset_at=None),
        ),
        Observation(
            offering_id=OFFERING,
            observed_at=PAST + timedelta(minutes=1),
            outcome=Outcome(bucket="self_healing", reset_at=None),
        ),
    ]

    result = reduce(
        prior=prior,
        outcomes={},
        observations=observations,
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    record = result.offerings[OFFERING]
    assert record.excluded is True
    assert record.failure_count == 2
    assert record.last_attempt_at == PAST + timedelta(minutes=1)


def test_reduce_returns_the_same_result_for_the_same_inputs_and_mutates_nothing():
    prior = _state(**{OFFERING: _excluded(reset_at=PAST)})
    outcomes = {OFFERING: Outcome(bucket="answered", reset_at=None)}
    observations: list[Observation] = []
    admitted = {OFFERING}
    passthrough_auth: set[str] = set()

    before = prior.offerings[OFFERING]

    first = reduce(
        prior=prior,
        outcomes=outcomes,
        observations=observations,
        admitted=admitted,
        passthrough_auth=passthrough_auth,
        now=NOW,
    )
    second = reduce(
        prior=prior,
        outcomes=outcomes,
        observations=observations,
        admitted=admitted,
        passthrough_auth=passthrough_auth,
        now=NOW,
    )

    assert first == second
    assert prior.offerings[OFFERING] == before
    assert outcomes == {OFFERING: Outcome(bucket="answered", reset_at=None)}
    assert observations == []
    assert admitted == {OFFERING}
    assert passthrough_auth == set()


# --- ADR 0008: the Journal path fails open --------------------------------


def test_an_unrecognized_failure_from_real_traffic_does_not_exclude_an_offering():
    """A client can cause a failure `classify` does not recognise.

    An over-long prompt returns HTTP 400 with wording no rule matches.
    Fail-closed would let one oversized request remove a healthy model
    from the Generated Config. See `reduce.journal_outcome`, ADR 0008.
    """
    healthy = OfferingHealth(excluded=False, last_success_at=PAST, last_attempt_at=PAST)
    observation = Observation(
        offering_id=OFFERING,
        observed_at=NOW,
        outcome=Outcome(bucket="needs_operator", reset_at=None, reason="unrecognized_failure"),
        message="prompt is too long: 312000 tokens > 200000 maximum",
    )

    result = reduce(
        prior=_state(**{OFFERING: healthy}),
        outcomes={},
        observations=[observation],
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    assert result.offerings[OFFERING].excluded is False
    # Inconclusive touches nothing, not even the attempt time.
    assert result.offerings[OFFERING].last_attempt_at == PAST
    assert result.offerings[OFFERING].failure_count == 0


def test_a_probe_that_cannot_be_classified_still_excludes_an_offering():
    """The Prober keeps `classify`'s fail-closed default.

    A Probe sends a known-good synthetic request, so a failure it
    cannot classify really is the Offering's fault. Only the Journal
    path fails open.
    """
    healthy = OfferingHealth(excluded=False, last_success_at=PAST, last_attempt_at=PAST)

    result = reduce(
        prior=_state(**{OFFERING: healthy}),
        outcomes={
            OFFERING: Outcome(
                bucket="needs_operator", reset_at=None, reason="unrecognized_failure"
            )
        },
        observations=[],
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    assert result.offerings[OFFERING].excluded is True


def test_a_recognised_failure_from_real_traffic_still_excludes_an_offering():
    """Fail-open applies to the unrecognised case alone.

    A quota exhaustion identifies itself, so it acts with no probe --
    which is the whole point of watching the Journal.
    """
    healthy = OfferingHealth(excluded=False, last_success_at=PAST, last_attempt_at=PAST)
    observation = Observation(
        offering_id=OFFERING,
        observed_at=NOW,
        outcome=Outcome(bucket="needs_operator", reset_at=None, reason="quota_exhausted"),
    )

    result = reduce(
        prior=_state(**{OFFERING: healthy}),
        outcomes={},
        observations=[observation],
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    assert result.offerings[OFFERING].excluded is True
    assert result.offerings[OFFERING].reason == "quota_exhausted"


def test_re_bucketing_keeps_the_reason_so_a_real_failure_is_not_called_a_non_event():
    """`classify.py` forbids conflating the two inconclusive reasons.

    `journal_outcome` changes the bucket, which names the consequence.
    It must not rewrite the reason, which names the condition.
    """
    from litellm_maintainer.reduce import journal_outcome

    result = journal_outcome(
        Outcome(bucket="needs_operator", reset_at=None, reason="unrecognized_failure")
    )

    assert result.bucket == "inconclusive"
    assert result.reason == "unrecognized_failure"


# --- ADR 0004: a shared pool propagates attention, never a verdict ---------

SIBLING = "opencode-go:deepseek-v4-flash"
_POOL = {
    OFFERING: frozenset({OFFERING, SIBLING}),
    SIBLING: frozenset({OFFERING, SIBLING}),
}


def _quota(reset_at: datetime | None = None) -> Outcome:
    return Outcome(bucket="self_healing", reset_at=reset_at, reason="quota_exhausted")


def _healthy() -> OfferingHealth:
    return OfferingHealth(excluded=False, last_success_at=PAST, last_attempt_at=PAST)


def test_a_shared_pool_quota_failure_never_excludes_a_sibling():
    """ADR 0004's two recorded cases: Gemini refused Pro and served
    Flash; ClinePass paid credits ran dry while three free Offerings
    from the same provider id still answered. Excluding on that guess
    removes the only routes still working."""
    result = reduce(
        prior=_state(**{OFFERING: _healthy(), SIBLING: _healthy()}),
        outcomes={OFFERING: _quota()},
        observations=[],
        admitted={OFFERING, SIBLING},
        passthrough_auth=set(),
        now=NOW,
        pool_siblings=_POOL,
    )

    assert result.offerings[OFFERING].excluded is True
    assert result.offerings[SIBLING].excluded is False
    assert result.offerings[SIBLING].reason is None


def test_a_shared_pool_quota_failure_marks_a_sibling_due_for_a_probe():
    result = reduce(
        prior=_state(**{OFFERING: _healthy(), SIBLING: _healthy()}),
        outcomes={OFFERING: _quota()},
        observations=[],
        admitted={OFFERING, SIBLING},
        passthrough_auth=set(),
        now=NOW,
        pool_siblings=_POOL,
    )

    assert result.offerings[SIBLING].probe_due is True


def test_a_sibling_this_run_already_measured_is_not_marked_due():
    """A mark asking to measure it again would be stale on arrival."""
    result = reduce(
        prior=_state(**{OFFERING: _healthy(), SIBLING: _healthy()}),
        outcomes={
            OFFERING: _quota(),
            SIBLING: Outcome(bucket="answered", reset_at=None, reason="answered"),
        },
        observations=[],
        admitted={OFFERING, SIBLING},
        passthrough_auth=set(),
        now=NOW,
        pool_siblings=_POOL,
    )

    assert result.offerings[SIBLING].probe_due is False


def test_measuring_an_offering_clears_its_probe_due_mark():
    marked = OfferingHealth(
        excluded=False, last_success_at=PAST, last_attempt_at=PAST, probe_due=True
    )

    result = reduce(
        prior=_state(**{SIBLING: marked}),
        outcomes={SIBLING: Outcome(bucket="answered", reset_at=None, reason="answered")},
        observations=[],
        admitted={SIBLING},
        passthrough_auth=set(),
        now=NOW,
    )

    assert result.offerings[SIBLING].probe_due is False


def test_an_inconclusive_probe_leaves_a_probe_due_mark_standing():
    """Inconclusive measured nothing, so the Offering is still due."""
    marked = OfferingHealth(
        excluded=False, last_success_at=PAST, last_attempt_at=PAST, probe_due=True
    )

    result = reduce(
        prior=_state(**{SIBLING: marked}),
        outcomes={SIBLING: Outcome(bucket="inconclusive", reset_at=None, reason="rate_limited")},
        observations=[],
        admitted={SIBLING},
        passthrough_auth=set(),
        now=NOW,
    )

    assert result.offerings[SIBLING].probe_due is True


def test_only_a_quota_exhaustion_propagates_attention():
    """Every other reason describes one Offering, not a pool."""
    result = reduce(
        prior=_state(**{OFFERING: _healthy(), SIBLING: _healthy()}),
        outcomes={
            OFFERING: Outcome(bucket="self_healing", reset_at=None, reason="gateway_error")
        },
        observations=[],
        admitted={OFFERING, SIBLING},
        passthrough_auth=set(),
        now=NOW,
        pool_siblings=_POOL,
    )

    assert result.offerings[SIBLING].probe_due is False


def test_a_passthrough_auth_quota_failure_propagates_nothing():
    """Its quota belongs to the calling client, not to our Entitlement,
    so it says nothing at all about the pool."""
    result = reduce(
        prior=_state(**{OFFERING: _healthy(), SIBLING: _healthy()}),
        outcomes={OFFERING: _quota()},
        observations=[],
        admitted={OFFERING, SIBLING},
        passthrough_auth={OFFERING},
        now=NOW,
        pool_siblings=_POOL,
    )

    assert result.offerings[SIBLING].probe_due is False


def test_a_passthrough_auth_sibling_is_never_marked_due():
    """The Prober cannot probe one, so the mark would never clear."""
    result = reduce(
        prior=_state(**{OFFERING: _healthy(), SIBLING: _healthy()}),
        outcomes={OFFERING: _quota()},
        observations=[],
        admitted={OFFERING, SIBLING},
        passthrough_auth={SIBLING},
        now=NOW,
        pool_siblings=_POOL,
    )

    assert result.offerings[SIBLING].probe_due is False


def test_a_quota_failure_from_real_traffic_also_marks_the_pool():
    """This is the whole point: a failure the operator hit while
    working makes the pool worth measuring, with no synthetic probe."""
    result = reduce(
        prior=_state(**{OFFERING: _healthy(), SIBLING: _healthy()}),
        outcomes={},
        observations=[
            Observation(offering_id=OFFERING, observed_at=NOW, outcome=_quota())
        ],
        admitted={OFFERING, SIBLING},
        passthrough_auth=set(),
        now=NOW,
        pool_siblings=_POOL,
    )

    assert result.offerings[SIBLING].probe_due is True
    assert result.offerings[SIBLING].excluded is False



# --- A sub-allowance is contained one way ----------------------------------

_POOL_KEY = "claude-sonnet-5"
_FABLE = "claude-fable-5"
_CLAUDE_POOL = {
    _POOL_KEY: frozenset({_POOL_KEY, _FABLE, "claude-opus-5"}),
    _FABLE: frozenset({_POOL_KEY, _FABLE, "claude-opus-5"}),
    "claude-opus-5": frozenset({_POOL_KEY, _FABLE, "claude-opus-5"}),
}


def _claude_state() -> HealthState:
    return _state(
        **{
            _POOL_KEY: _healthy(),
            _FABLE: _healthy(),
            "claude-opus-5": _healthy(),
        }
    )


def test_a_sub_allowance_exhausting_says_nothing_about_its_pool():
    """At most half the Claude weekly quota may go to fable, so fable
    running out leaves the rest with room."""
    result = reduce(
        prior=_claude_state(),
        outcomes={_FABLE: _quota()},
        observations=[],
        admitted={_POOL_KEY, _FABLE, "claude-opus-5"},
        passthrough_auth=set(),
        now=NOW,
        pool_siblings=_CLAUDE_POOL,
        sub_allowances={_FABLE},
    )

    assert result.offerings[_POOL_KEY].probe_due is False
    assert result.offerings["claude-opus-5"].probe_due is False


def test_the_pools_exhaustion_still_reaches_a_sub_allowance():
    """The whole weekly quota running out takes fable with it. This is
    the case a plain "leave fable out of the group" rule gets wrong."""
    result = reduce(
        prior=_claude_state(),
        outcomes={_POOL_KEY: _quota()},
        observations=[],
        admitted={_POOL_KEY, _FABLE, "claude-opus-5"},
        passthrough_auth=set(),
        now=NOW,
        pool_siblings=_CLAUDE_POOL,
        sub_allowances={_FABLE},
    )

    assert result.offerings[_FABLE].probe_due is True
    assert result.offerings["claude-opus-5"].probe_due is True


def test_a_sub_allowance_is_still_recorded_on_its_own_record():
    """Containment governs propagation, never the observation itself."""
    result = reduce(
        prior=_claude_state(),
        outcomes={_FABLE: _quota()},
        observations=[],
        admitted={_POOL_KEY, _FABLE},
        passthrough_auth=set(),
        now=NOW,
        pool_siblings=_CLAUDE_POOL,
        sub_allowances={_FABLE},
    )

    assert result.offerings[_FABLE].reason == "quota_exhausted"
    assert result.offerings[_FABLE].excluded is True


# --- A silent misclassification must become visible ------------------------


def test_an_inconclusive_observation_counts_even_though_health_does_not_move():
    """The exhausted OpenCode Go plan wrote 90 entries that all read as
    `rate_limited`, changed nothing, and were noticed only because the
    operator said so. The count is what makes that shape visible."""
    healthy = OfferingHealth(excluded=False, last_success_at=PAST, last_attempt_at=PAST)
    observations = [
        Observation(
            offering_id=OFFERING,
            observed_at=NOW,
            outcome=Outcome(bucket="inconclusive", reset_at=None, reason="rate_limited"),
        )
        for _ in range(3)
    ]

    result = reduce(
        prior=_state(**{OFFERING: healthy}),
        outcomes={},
        observations=observations,
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    record = result.offerings[OFFERING]
    assert record.inconclusive_count == 3
    # Everything else is still untouched -- that rule is intact.
    assert record.excluded is False
    assert record.last_attempt_at == PAST
    assert record.failure_count == 0


def test_a_success_resets_the_unproductive_count():
    """The Offering demonstrably works, so the evidence of a misread
    condition is gone with it."""
    prior = OfferingHealth(excluded=False, last_attempt_at=PAST, inconclusive_count=40)

    result = reduce(
        prior=_state(**{OFFERING: prior}),
        outcomes={OFFERING: Outcome(bucket="answered", reset_at=None, reason="answered")},
        observations=[],
        admitted={OFFERING},
        passthrough_auth=set(),
        now=NOW,
    )

    assert result.offerings[OFFERING].inconclusive_count == 0
