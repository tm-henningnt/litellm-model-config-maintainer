"""Tests for ticket 07: the quality gate, Candidates, and Sunsetting.

Assert external behaviour: which Offering ids `plan` puts in
`report.admitted`, `report.candidates`, `report.sunsetting` and
`report.excluded`, and which Aliases end up in the Generated Config. A
test name states a rule an operator would recognise (spec's "What
makes a good test here").

Two fixture sources feed these tests. Most build a small synthetic
Feed document in memory, so the exact boundary under test is visible
in the test itself. The tests about the real four Sunsetting Offerings
read `tests/fixtures/feed-current.json` and the operator's own Policy
at `tests/fixtures/policy-pinned.yaml`, both
read-only.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from litellm_maintainer.feed import Feed, load_feed, parse_feed
from litellm_maintainer.plan import plan
from litellm_maintainer.policy import load_policy, parse_policy
from litellm_maintainer.reduce import OfferingHealth

FIXTURES = Path(__file__).parent / "fixtures"
FEED_CURRENT_PATH = FIXTURES / "feed-current.json"
# Synthetic and committed. Never the operator's own Policy.
PINNED_POLICY_PATH = Path(__file__).parent / "fixtures" / "policy-pinned.yaml"

NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)

# The four OpenCode Go Offerings the operator runs today, marked
# `retired` and `hidden` in `feed-current.json`. Verified directly
# against the fixture: none of the four carries the `coding`
# capability flag. Each is Sunsetting only when OUR Health State
# records it as working (spec-corrections.md, correction 6).
SUNSETTING_OFFERING_IDS = {
    "opencode-go:glm-5",
    "opencode-go:kimi-k2.5",
    "opencode-go:minimax-m2.5",
    "opencode-go:qwen3.5-plus",
}


@pytest.fixture(scope="module")
def feed_current() -> Feed:
    return load_feed(FEED_CURRENT_PATH)


@pytest.fixture(scope="module")
def operator_policy():
    return load_policy(PINNED_POLICY_PATH)


# --- Synthetic Feed and Policy builders ---------------------------------
#
# `opencode-go` is the provider id for every synthetic Offering below.
# It already has a registered translation rule (the generic
# OpenAI-compatible one) and a credential fallback
# (`translate.CREDENTIAL_FALLBACKS`), so a synthetic Offering
# translates without a Feed-published `authentication.credential_hint`.


def _offering_raw(
    *,
    id: str,
    provider_model_id: str,
    coding_score: float | None,
    status: str = "available",
    visibility: str = "listed",
    capabilities: tuple[str, ...] = ("tool_use",),
    last_success_at: str | None = None,
) -> dict[str, Any]:
    return {
        "id": id,
        "provider": {"id": "opencode-go"},
        "provider_model_id": provider_model_id,
        "capabilities": list(capabilities),
        "endpoint": {
            "base_url": "https://opencode-go.example/v1",
            "model": provider_model_id,
        },
        "pricing": {"kind": "subscription_included"},
        "availability": {
            "status": status,
            "last_checked_at": "2026-07-25T00:00:00Z",
            "last_success_at": last_success_at,
            "stale_after_seconds": 86400,
        },
        "quality": {"coding_score": coding_score},
        "policy": {"visibility": visibility, "tags": []},
    }


def _feed_with(*offerings: dict[str, Any]) -> Feed:
    raw = {
        "schema_version": "test",
        "providers": [
            {
                "id": "opencode-go",
                "name": "OpenCode Go",
                "default_base_url": "https://opencode-go.example/v1",
                "authentication": {},
            }
        ],
        "models": [copy.deepcopy(o) for o in offerings],
    }
    return parse_feed(raw)


def _policy_raw(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "providers": {"opencode-go": {"mode": "all"}},
        "quality": {"minimum_coding_score": 18},
        "approved_candidates": [],
        "naming": {
            "alias_prefix": "claude-",
            "provider_labels": {"opencode-go": "opencode-go"},
            "alias_overrides": {},
        },
        "withheld": {},
        "declared": [],
        "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
        "schedule": {
            "enabled": True,
            "interval_minutes": 60,
            "require_proxy": True,
            "maximum_staleness_hours": 24,
        },
        "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
    }
    raw.update(overrides)
    return raw


def _policy(**overrides: Any):
    return parse_policy(_policy_raw(**overrides))


# --- The quality gate and the threshold boundary ------------------------


def test_an_offering_scoring_at_the_threshold_is_admitted():
    feed = _feed_with(
        _offering_raw(id="opencode-go:at-threshold", provider_model_id="at-threshold", coding_score=18)
    )
    result = plan(feed=feed, policy=_policy(), health={}, now=NOW)
    assert "opencode-go:at-threshold" in result.report.admitted
    aliases = {e["model_name"] for e in result.config["model_list"]}
    assert "claude-opencode-go-at-threshold" in aliases


def test_an_offering_scoring_below_the_threshold_does_not_appear():
    feed = _feed_with(
        _offering_raw(
            id="opencode-go:below-threshold", provider_model_id="below-threshold", coding_score=17.99
        )
    )
    result = plan(feed=feed, policy=_policy(), health={}, now=NOW)
    assert "opencode-go:below-threshold" not in result.report.admitted
    assert result.config["model_list"] == []


# --- Candidates -----------------------------------------------------------


def test_an_unscored_offering_becomes_a_candidate_reported_never_added_never_dropped():
    feed = _feed_with(
        _offering_raw(id="opencode-go:unscored", provider_model_id="unscored", coding_score=None)
    )
    policy = _policy()

    # Run twice: `plan` is pure, so a Candidate that Policy has not
    # approved is reported on every run, never added and never
    # silently dropped from one run to the next.
    first = plan(feed=feed, policy=policy, health={}, now=NOW)
    second = plan(feed=feed, policy=policy, health={}, now=NOW)

    for result in (first, second):
        assert "opencode-go:unscored" in result.report.candidates
        assert "opencode-go:unscored" not in result.report.admitted


def test_an_approved_candidate_appears_in_the_generated_config():
    feed = _feed_with(
        _offering_raw(
            id="opencode-go:approved-candidate",
            provider_model_id="approved-candidate",
            coding_score=None,
        )
    )
    policy = _policy(approved_candidates=["opencode-go:approved-candidate"])
    result = plan(feed=feed, policy=policy, health={}, now=NOW)

    assert "opencode-go:approved-candidate" in result.report.admitted
    aliases = {e["model_name"] for e in result.config["model_list"]}
    assert "claude-opencode-go-approved-candidate" in aliases


# --- The `coding` capability flag neither gates nor grants ----------------


def test_an_unscored_offering_carrying_the_coding_flag_is_still_a_candidate_not_auto_admitted():
    feed = _feed_with(
        _offering_raw(
            id="opencode-go:unscored-with-flag",
            provider_model_id="unscored-with-flag",
            coding_score=None,
            capabilities=("tool_use", "coding"),
        )
    )
    result = plan(feed=feed, policy=_policy(), health={}, now=NOW)

    assert "opencode-go:unscored-with-flag" in result.report.candidates
    assert "opencode-go:unscored-with-flag" not in result.report.admitted


def _health_recorded_working(offering_ids) -> dict[str, OfferingHealth]:
    """Health State that records each Offering as working.

    The audit called all four OpenCode Go Offerings directly on
    2026-07-25 and each returned a completion (spec, "Availability is a
    warning, not a verdict"). This states that measurement as our own
    Health State, which is the only evidence the Sunsetting rule
    accepts (spec-corrections.md, correction 6).
    """
    return {
        offering_id: OfferingHealth(
            excluded=False,
            last_success_at=datetime(2026, 7, 25, 4, tzinfo=timezone.utc),
            last_attempt_at=datetime(2026, 7, 25, 4, tzinfo=timezone.utc),
        )
        for offering_id in offering_ids
    }


def test_the_four_sunsetting_offerings_the_operator_runs_lack_the_coding_flag_and_still_appear(
    feed_current, operator_policy
):
    health = _health_recorded_working(SUNSETTING_OFFERING_IDS)
    result = plan(feed=feed_current, policy=operator_policy, health=health, now=NOW)

    for offering_id in SUNSETTING_OFFERING_IDS:
        offering = feed_current.offering(offering_id)
        assert offering is not None, f"fixture offering {offering_id!r} not found"
        assert "coding" not in offering.capabilities, offering_id
        assert offering_id in result.report.admitted, offering_id
        assert offering_id in result.report.sunsetting, offering_id


def test_requiring_the_coding_capability_is_not_applied(feed_current, operator_policy):
    """An admitted Offering that lacks the `coding` flag exists. If the
    flag ever gated admission, this would be empty.

    `result.report.admitted` also names each admitted Declared Offering,
    by its Alias (correction 10: a Declared Offering is offered too).
    The Feed does not publish Declared Offerings, so `feed.offering`
    returns `None` for one; skip those here, since this test is about
    Discovered Offerings' `coding` flag only.
    """
    result = plan(feed=feed_current, policy=operator_policy, health={}, now=NOW)

    admitted_without_flag = [
        offering_id
        for offering_id in result.report.admitted
        if feed_current.offering(offering_id) is not None
        and "coding" not in feed_current.offering(offering_id).capabilities
    ]
    assert admitted_without_flag, "expected at least one admitted Offering with no 'coding' flag"


# --- Selection reads listed Offerings -------------------------------------


def test_a_hidden_offering_that_is_not_sunsetting_does_not_appear():
    feed = _feed_with(
        _offering_raw(
            id="opencode-go:hidden-not-sunsetting",
            provider_model_id="hidden-not-sunsetting",
            coding_score=20,
            status="available",
            visibility="hidden",
        )
    )
    result = plan(feed=feed, policy=_policy(), health={}, now=NOW)
    assert "opencode-go:hidden-not-sunsetting" not in result.report.admitted


# --- Sunsetting -------------------------------------------------------------


def test_an_offering_recorded_as_working_which_the_feed_now_reports_leaving_is_kept_and_reported_sunsetting():
    feed = _feed_with(
        _offering_raw(
            id="opencode-go:leaving-but-worked",
            provider_model_id="leaving-but-worked",
            coding_score=20,
            status="retired",
            visibility="hidden",
        )
    )
    health = {
        "opencode-go:leaving-but-worked": OfferingHealth(
            excluded=False,
            last_success_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
            last_attempt_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
        )
    }
    result = plan(feed=feed, policy=_policy(), health=health, now=NOW)

    assert "opencode-go:leaving-but-worked" in result.report.admitted
    assert "opencode-go:leaving-but-worked" in result.report.sunsetting


def test_an_offering_that_has_never_worked_does_not_become_sunsetting():
    """`status=retired` alone is not enough. No Health State record
    means no evidence WE ever called it, so it stays an ordinary
    (non-Sunsetting) Offering. It is
    `listed` here, so it is still admitted through the ordinary path —
    Sunsetting is not why it appears."""
    feed = _feed_with(
        _offering_raw(
            id="opencode-go:leaving-never-worked",
            provider_model_id="leaving-never-worked",
            coding_score=20,
            status="retired",
            visibility="listed",
            last_success_at=None,
        )
    )
    result = plan(feed=feed, policy=_policy(), health={}, now=NOW)

    assert "opencode-go:leaving-never-worked" in result.report.admitted
    assert "opencode-go:leaving-never-worked" not in result.report.sunsetting


def test_a_sunsetting_offering_that_stops_answering_is_excluded_by_the_ordinary_path():
    feed = _feed_with(
        _offering_raw(
            id="opencode-go:sunsetting-then-fails",
            provider_model_id="sunsetting-then-fails",
            coding_score=20,
            status="retired",
            visibility="hidden",
        )
    )
    health = {
        "opencode-go:sunsetting-then-fails": OfferingHealth(
            excluded=True,
            reason="gone",
            bucket="gone",
            last_success_at=datetime(2026, 7, 20, tzinfo=timezone.utc),
            last_attempt_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
            failure_count=1,
        )
    }
    result = plan(feed=feed, policy=_policy(), health=health, now=NOW)

    assert "opencode-go:sunsetting-then-fails" not in result.report.admitted
    assert "opencode-go:sunsetting-then-fails" in result.report.unlisted
    assert "opencode-go:sunsetting-then-fails" not in result.report.sunsetting


def test_the_feeds_own_success_record_does_not_make_an_offering_sunsetting(feed_current):
    """Correction 6. The Feed observes availability with its owner's
    credentials, so its `last_success_at` never shows that WE can call
    the Offering. With empty Health State no Offering is Sunsetting,
    and a `hidden` one does not bypass the visibility filter.

    The field also discriminates nothing: every Offering in both pinned
    fixtures carries a non-null value. A rule that read it would reduce
    to "the Feed says the Offering leaves".
    """
    for offering_id in SUNSETTING_OFFERING_IDS:
        offering = feed_current.offering(offering_id)
        assert offering.availability["last_success_at"] is not None, offering_id
        assert offering.visibility == "hidden", offering_id

    result = plan(feed=feed_current, policy=_operator_like_policy(), health={}, now=NOW)

    assert result.report.sunsetting == ()
    for offering_id in SUNSETTING_OFFERING_IDS:
        assert offering_id not in result.report.admitted, offering_id


def test_every_offering_in_both_pinned_feeds_carries_a_feed_success_timestamp(feed_current):
    """Pin the fact correction 6 rests on. If a future Feed revision
    starts publishing a null `availability.last_success_at`, this test
    fails and the reasoning in correction 6 must be re-read.
    """
    null_count = sum(
        1 for o in feed_current.offerings if o.availability.get("last_success_at") is None
    )
    assert null_count == 0


def _operator_like_policy():
    """A Policy that takes every OpenCode Go Offering the operator runs.

    Built here rather than read from the operator's file, so this test
    states its own selection and cannot drift with a Policy edit.
    """
    return parse_policy(
        _policy_raw(
            providers={"opencode-go": {"mode": "named", "models": sorted(SUNSETTING_OFFERING_IDS)}},
            approved_candidates=sorted(SUNSETTING_OFFERING_IDS),
        )
    )


def test_a_sunsetting_offering_still_passes_every_other_filter(feed_current):
    """The bypass skips the visibility check and nothing else.

    Withhold each of the four, seed Health State as working, and none
    appears. A bypass that skipped more than visibility would show up
    here.
    """
    policy = parse_policy(
        _policy_raw(
            providers={"opencode-go": {"mode": "named", "models": sorted(SUNSETTING_OFFERING_IDS)}},
            approved_candidates=sorted(SUNSETTING_OFFERING_IDS),
            withheld={offering_id: "under review" for offering_id in SUNSETTING_OFFERING_IDS},
        )
    )
    health = _health_recorded_working(SUNSETTING_OFFERING_IDS)
    result = plan(feed=feed_current, policy=policy, health=health, now=NOW)

    assert result.report.admitted == ()
    assert result.report.sunsetting == ()


def test_a_sunsetting_offering_below_the_quality_threshold_does_not_appear():
    """The quality gate still applies to a Sunsetting Offering."""
    feed = _feed_with(
        _offering_raw(
            id="opencode-go:leaving-and-weak",
            provider_model_id="leaving-and-weak",
            coding_score=2,
            status="retired",
            visibility="hidden",
        )
    )
    health = _health_recorded_working(["opencode-go:leaving-and-weak"])
    result = plan(feed=feed, policy=_policy(), health=health, now=NOW)

    assert result.report.admitted == ()
    assert result.report.sunsetting == ()


def test_a_sunsetting_offering_without_tool_use_does_not_appear():
    """The baseline capability filter still applies to a Sunsetting
    Offering."""
    feed = _feed_with(
        _offering_raw(
            id="opencode-go:leaving-no-tool-use",
            provider_model_id="leaving-no-tool-use",
            coding_score=90,
            status="retired",
            visibility="hidden",
            capabilities=("chat",),
        )
    )
    health = _health_recorded_working(["opencode-go:leaving-no-tool-use"])
    result = plan(feed=feed, policy=_policy(), health=health, now=NOW)

    assert result.report.admitted == ()


# --- What the report says about a Candidate -------------------------------


def test_an_approved_candidate_is_not_reported_as_awaiting_approval():
    """`report.candidates` is the "awaiting approval" list. An approved
    Candidate waits for nothing, so it must not appear there. The CLI
    prints this list under that heading."""
    feed = _feed_with(
        _offering_raw(id="opencode-go:approved", provider_model_id="approved", coding_score=None),
        _offering_raw(id="opencode-go:waiting", provider_model_id="waiting", coding_score=None),
    )
    policy = _policy(approved_candidates=["opencode-go:approved"])
    result = plan(feed=feed, policy=policy, health={}, now=NOW)

    assert result.report.candidates == ("opencode-go:waiting",)
    assert "opencode-go:approved" in result.report.admitted


def test_a_withheld_unscored_offering_is_not_reported_as_awaiting_approval():
    """Withheld and Candidate are two different states. The operator
    already declined a Withheld Offering, so asking them to approve it
    would be wrong. Withheld is therefore checked before the quality
    gate."""
    feed = _feed_with(
        _offering_raw(
            id="opencode-go:withheld-unscored",
            provider_model_id="withheld-unscored",
            coding_score=None,
        )
    )
    policy = _policy(withheld={"opencode-go:withheld-unscored": "billing unclear"})
    result = plan(feed=feed, policy=policy, health={}, now=NOW)

    assert result.report.candidates == ()
    assert result.report.admitted == ()


def test_an_unlisted_unscored_offering_is_reported_as_unlisted_not_as_a_candidate():
    feed = _feed_with(
        _offering_raw(
            id="opencode-go:excluded-unscored",
            provider_model_id="excluded-unscored",
            coding_score=None,
        )
    )
    health = {
        "opencode-go:excluded-unscored": OfferingHealth(
            excluded=True,
            reason="gone",
            bucket="gone",
            last_attempt_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
            failure_count=1,
        )
    }
    result = plan(feed=feed, policy=_policy(), health=health, now=NOW)

    assert result.report.unlisted == ("opencode-go:excluded-unscored",)
    assert result.report.candidates == ()


def test_staleness_is_judged_on_last_success_at_never_on_last_attempt_at():
    """A record with a fresh `last_attempt_at` but no `last_success_at`
    must not read as Sunsetting-eligible. Reading `last_attempt_at`
    instead would wrongly treat a recent failing attempt as evidence
    the Offering still works."""
    feed = _feed_with(
        _offering_raw(
            id="opencode-go:fresh-attempt-no-success",
            provider_model_id="fresh-attempt-no-success",
            coding_score=20,
            status="retired",
            visibility="listed",
            last_success_at=None,
        )
    )
    health = {
        "opencode-go:fresh-attempt-no-success": OfferingHealth(
            excluded=False,
            last_success_at=None,
            last_attempt_at=datetime(2026, 7, 24, tzinfo=timezone.utc),
            failure_count=2,
        )
    }
    result = plan(feed=feed, policy=_policy(), health=health, now=NOW)

    assert "opencode-go:fresh-attempt-no-success" not in result.report.sunsetting


def test_an_unscored_excluded_offering_is_a_candidate_and_is_not_called_served():
    """`excluded` states "in the config and not recommended". An unscored
    Offering stops at the Candidate gate, so it never reaches the config,
    and naming it Excluded would report a served Offering that is not
    served. It is still a Candidate: the operator's decision is live, and
    its health is a separate axis that will change."""
    feed = _feed_with(
        _offering_raw(
            id="opencode-go:unscored-and-failing",
            provider_model_id="unscored-and-failing",
            coding_score=None,
        )
    )
    health = {
        "opencode-go:unscored-and-failing": OfferingHealth(
            excluded=True, reason="gateway_error", bucket="self_healing"
        )
    }

    result = plan(feed=feed, policy=_policy(), health=health, now=NOW)

    assert result.report.candidates == ("opencode-go:unscored-and-failing",)
    assert result.report.excluded == ()
    assert result.report.admitted == ()
