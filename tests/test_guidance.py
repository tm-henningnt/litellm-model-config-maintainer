"""Tests for `guidance.py`: ranked picks for a calling agent.

Assert external behaviour: what `derive` puts in a Guidance Row, how it
orders Routes within a row and rows against each other, and what the
Client Advisory names. A test name states a rule an operator or a
calling agent would recognise, following tests/test_pricing.py and
tests/test_report.py.

ADR 0005 is the module's central rule: guidance reports what Health
State measured, never a balance nobody can read. "Two orderings, never
blended" is pinned directly: rows descend by score, Routes within a row
ascend by cost, and the two are never combined into one number.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from litellm_maintainer.entitlements import FLAT_RATE, FREE, METERED
from litellm_maintainer.feed import Feed, parse_feed
from litellm_maintainer.guidance import (
    AXES,
    ClientAdvisory,
    Guidance,
    GuidanceError,
    Route,
    build_advisory,
    derive,
    render_markdown,
    render_text,
)
from litellm_maintainer.notify import PreviousRunState
from litellm_maintainer.plan import PlanReport
from litellm_maintainer.policy import parse_policy
from litellm_maintainer.reduce import OfferingHealth

NOW = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)


# --- Fixture builders --------------------------------------------------


def _offering_raw(
    *,
    id: str,
    provider_id: str,
    canonical_model_id: str,
    pricing_kind: str = "free",
    coding_score: float | None = 50.0,
    reasoning_score: float | None = None,
    display_name: str | None = None,
) -> dict[str, Any]:
    model_id = id.split(":", 1)[1]
    quality: dict[str, Any] = {}
    if coding_score is not None:
        quality["coding_score"] = coding_score
    if reasoning_score is not None:
        quality["reasoning_score"] = reasoning_score
    return {
        "id": id,
        "provider": {"id": provider_id},
        "provider_model_id": model_id,
        "capabilities": ["tool_use"],
        "endpoint": {"base_url": f"https://{provider_id}.example/v1", "model": model_id},
        "pricing": {"kind": pricing_kind, "metering": "tokens"},
        "availability": {"status": "available"},
        "quality": quality,
        "policy": {"visibility": "listed", "tags": []},
        "canonical_model": {"id": canonical_model_id},
        "display_name": display_name or canonical_model_id,
    }


def _feed_with(*offerings: dict[str, Any]) -> Feed:
    provider_ids = sorted({o["provider"]["id"] for o in offerings})
    return parse_feed(
        {
            "schema_version": "test",
            "providers": [
                {
                    "id": provider_id,
                    "name": provider_id,
                    "default_base_url": f"https://{provider_id}.example/v1",
                    "authentication": {},
                }
                for provider_id in provider_ids
            ],
            "models": list(offerings),
        }
    )


def _policy_raw(**overrides: Any) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "providers": {},
        "quality": {"minimum_coding_score": 18},
        "approved_candidates": [],
        "naming": {
            "alias_prefix": "claude-",
            "provider_labels": {},
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


# --- One row per Canonical Model, never one per Alias -----------------------


def test_one_canonical_model_reachable_by_three_offerings_produces_one_row_with_three_routes():
    feed = _feed_with(
        _offering_raw(id="provA:m", provider_id="provA", canonical_model_id="model-x"),
        _offering_raw(id="provB:m", provider_id="provB", canonical_model_id="model-x"),
        _offering_raw(id="provC:m", provider_id="provC", canonical_model_id="model-x"),
    )
    policy = _policy()
    report = PlanReport(
        admitted=("provA:m", "provB:m", "provC:m"),
        aliases={"provA:m": "claude-a", "provB:m": "claude-b", "provC:m": "claude-c"},
    )

    guidance = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)

    assert len(guidance.rows) == 1
    assert guidance.rows[0].canonical_model_id == "model-x"
    assert len(guidance.rows[0].routes) == 3


# --- Route ordering within a row: available beats cheap ---------------------


def test_routes_sort_available_before_unavailable_even_when_unavailable_is_cheaper():
    feed = _feed_with(
        _offering_raw(
            id="provA:cheap", provider_id="provA", canonical_model_id="model-y",
            pricing_kind="free",
        ),
        _offering_raw(
            id="provB:pricey", provider_id="provB", canonical_model_id="model-y",
            pricing_kind="paid",
        ),
    )
    policy = _policy()
    report = PlanReport(
        admitted=("provB:pricey",),
        excluded=("provA:cheap",),
        aliases={"provA:cheap": "claude-cheap", "provB:pricey": "claude-pricey"},
    )
    health = {"provA:cheap": OfferingHealth(excluded=True, reason="quota_exhausted")}

    guidance = derive(feed=feed, policy=policy, health=health, report=report, now=NOW)

    routes = guidance.rows[0].routes
    assert routes[0].alias == "claude-pricey"
    assert routes[0].available is True
    assert routes[1].alias == "claude-cheap"
    assert routes[1].available is False


def test_among_available_routes_the_cheaper_cost_basis_sorts_first():
    feed = _feed_with(
        _offering_raw(
            id="provA:m", provider_id="provA", canonical_model_id="model-z",
            pricing_kind="paid",
        ),
        _offering_raw(
            id="provB:m", provider_id="provB", canonical_model_id="model-z",
            pricing_kind="subscription_included",
        ),
        _offering_raw(
            id="provC:m", provider_id="provC", canonical_model_id="model-z",
            pricing_kind="free",
        ),
    )
    policy = _policy()
    report = PlanReport(
        admitted=("provA:m", "provB:m", "provC:m"),
        aliases={"provA:m": "claude-metered", "provB:m": "claude-flat", "provC:m": "claude-free"},
    )

    guidance = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)

    bases = [route.cost_basis for route in guidance.rows[0].routes]
    assert bases == [FREE, FLAT_RATE, METERED]


def test_an_unavailable_route_is_present_in_the_row_with_its_reason_and_refills_at():
    reset_at = datetime(2026, 7, 27, 0, 0, 0, tzinfo=timezone.utc)
    feed = _feed_with(
        _offering_raw(id="provA:m", provider_id="provA", canonical_model_id="model-w"),
    )
    policy = _policy()
    report = PlanReport(excluded=("provA:m",), aliases={"provA:m": "claude-a"})
    health = {
        "provA:m": OfferingHealth(
            excluded=True, reason="quota_exhausted", reset_at=reset_at
        )
    }

    guidance = derive(feed=feed, policy=policy, health=health, report=report, now=NOW)

    route = guidance.rows[0].routes[0]
    assert route.available is False
    assert route.reason == "quota_exhausted"
    assert route.refills_at == reset_at


# --- Row ordering: score descending ----------------------------------------


def test_rows_sort_by_the_requested_axis_score_descending():
    feed = _feed_with(
        _offering_raw(
            id="prov:low", provider_id="prov", canonical_model_id="model-low",
            coding_score=10.0,
        ),
        _offering_raw(
            id="prov:high", provider_id="prov", canonical_model_id="model-high",
            coding_score=90.0,
        ),
    )
    policy = _policy()
    report = PlanReport(
        admitted=("prov:low", "prov:high"),
        aliases={"prov:low": "claude-low", "prov:high": "claude-high"},
    )

    guidance = derive(feed=feed, policy=policy, health={}, report=report, now=NOW, axis="coding")

    assert [row.canonical_model_id for row in guidance.rows] == ["model-high", "model-low"]


def test_an_admitted_row_with_no_score_on_the_requested_axis_sorts_last_but_is_present():
    feed = _feed_with(
        _offering_raw(
            id="prov:scored", provider_id="prov", canonical_model_id="model-scored",
            coding_score=40.0,
        ),
        _offering_raw(
            id="prov:unscored", provider_id="prov", canonical_model_id="model-unscored",
            coding_score=None,
        ),
    )
    policy = _policy()
    report = PlanReport(
        admitted=("prov:scored", "prov:unscored"),
        aliases={"prov:scored": "claude-scored", "prov:unscored": "claude-unscored"},
    )

    guidance = derive(feed=feed, policy=policy, health={}, report=report, now=NOW, axis="coding")

    assert [row.canonical_model_id for row in guidance.rows] == ["model-scored", "model-unscored"]
    assert guidance.rows[1].score is None


# --- Errors: unknown axis, invalid prefer ------------------------------------


def test_an_unknown_axis_raises_guidanceerror_naming_the_valid_axes():
    feed = _feed_with(
        _offering_raw(id="prov:m", provider_id="prov", canonical_model_id="model-a")
    )
    policy = _policy()
    report = PlanReport(admitted=("prov:m",), aliases={"prov:m": "claude-a"})

    with pytest.raises(GuidanceError) as excinfo:
        derive(feed=feed, policy=policy, health={}, report=report, now=NOW, axis="charisma")

    for axis_name in AXES:
        assert axis_name in str(excinfo.value)


def test_an_invalid_prefer_value_raises_guidanceerror():
    feed = _feed_with(
        _offering_raw(id="prov:m", provider_id="prov", canonical_model_id="model-a")
    )
    policy = _policy()
    report = PlanReport(admitted=("prov:m",), aliases={"prov:m": "claude-a"})

    with pytest.raises(GuidanceError):
        derive(feed=feed, policy=policy, health={}, report=report, now=NOW, prefer=METERED)


# --- prefer re-sorts into cost tiers -----------------------------------------


def test_prefer_free_puts_a_lower_scored_free_model_above_a_higher_scored_metered_one():
    feed = _feed_with(
        _offering_raw(
            id="prov:metered", provider_id="prov", canonical_model_id="model-metered",
            pricing_kind="paid", coding_score=90.0,
        ),
        _offering_raw(
            id="prov:free", provider_id="prov", canonical_model_id="model-free",
            pricing_kind="free", coding_score=10.0,
        ),
    )
    policy = _policy()
    report = PlanReport(
        admitted=("prov:metered", "prov:free"),
        aliases={"prov:metered": "claude-metered", "prov:free": "claude-free"},
    )

    guidance = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW,
        axis="coding", prefer=FREE,
    )

    assert [row.canonical_model_id for row in guidance.rows] == ["model-free", "model-metered"]


def test_asking_for_a_different_axis_reorders_the_rows():
    feed = _feed_with(
        _offering_raw(
            id="prov:coder", provider_id="prov", canonical_model_id="model-coder",
            coding_score=90.0, reasoning_score=10.0,
        ),
        _offering_raw(
            id="prov:reasoner", provider_id="prov", canonical_model_id="model-reasoner",
            coding_score=10.0, reasoning_score=90.0,
        ),
    )
    policy = _policy()
    report = PlanReport(
        admitted=("prov:coder", "prov:reasoner"),
        aliases={"prov:coder": "claude-coder", "prov:reasoner": "claude-reasoner"},
    )

    by_coding = derive(feed=feed, policy=policy, health={}, report=report, now=NOW, axis="coding")
    by_reasoning = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, axis="reasoning"
    )

    assert [row.canonical_model_id for row in by_coding.rows] == ["model-coder", "model-reasoner"]
    assert [row.canonical_model_id for row in by_reasoning.rows] == ["model-reasoner", "model-coder"]


def test_limit_caps_the_number_of_rows_returned():
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-a", coding_score=90.0),
        _offering_raw(id="prov:b", provider_id="prov", canonical_model_id="model-b", coding_score=80.0),
        _offering_raw(id="prov:c", provider_id="prov", canonical_model_id="model-c", coding_score=70.0),
    )
    policy = _policy()
    report = PlanReport(
        admitted=("prov:a", "prov:b", "prov:c"),
        aliases={"prov:a": "claude-a", "prov:b": "claude-b", "prov:c": "claude-c"},
    )

    guidance = derive(feed=feed, policy=policy, health={}, report=report, now=NOW, limit=2)

    assert len(guidance.rows) == 2
    assert [row.canonical_model_id for row in guidance.rows] == ["model-a", "model-b"]


# --- why -----------------------------------------------------------------


def test_why_mentions_the_flat_rate_window_for_a_flat_rate_best_route():
    feed = _feed_with(
        _offering_raw(
            id="prov:m", provider_id="prov", canonical_model_id="model-a",
            pricing_kind="subscription_included",
        )
    )
    policy = _policy()
    report = PlanReport(admitted=("prov:m",), aliases={"prov:m": "claude-a"})

    guidance = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)

    assert "flat-rate window" in guidance.rows[0].why


def test_why_mentions_billing_for_a_metered_best_route():
    feed = _feed_with(
        _offering_raw(
            id="prov:m", provider_id="prov", canonical_model_id="model-a",
            pricing_kind="paid",
        )
    )
    policy = _policy()
    report = PlanReport(admitted=("prov:m",), aliases={"prov:m": "claude-a"})

    guidance = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)

    assert "bills" in guidance.rows[0].why


# --- Route.rate_is_list_price ------------------------------------------------


def test_route_rate_is_list_price_is_true_for_free_and_flat_rate_false_for_metered():
    free_route = Route(
        alias="claude-free", offering_id="prov:free", provider_id="prov",
        cost_basis=FREE, available=True, entitlement="per_model",
    )
    flat_rate_route = Route(
        alias="claude-flat", offering_id="prov:flat", provider_id="prov",
        cost_basis=FLAT_RATE, available=True, entitlement="per_model",
    )
    metered_route = Route(
        alias="claude-metered", offering_id="prov:metered", provider_id="prov",
        cost_basis=METERED, available=True, entitlement="per_model",
    )

    assert free_route.rate_is_list_price is True
    assert flat_rate_route.rate_is_list_price is True
    assert metered_route.rate_is_list_price is False


# --- Client Advisory ---------------------------------------------------------


def test_the_client_advisory_names_an_alias_added_since_the_previous_run():
    policy = _policy()
    report = PlanReport(admitted=("prov:new",), aliases={"prov:new": "claude-new"})
    previous = PreviousRunState(admitted=frozenset())

    advisory = build_advisory(policy=policy, report=report, health={}, previous=previous)

    assert advisory.added_last_run == ("claude-new",)


def test_the_client_advisory_names_an_alias_removed_since_the_previous_run_with_its_reason_and_refills_at():
    reset_at = datetime(2026, 7, 27, 0, 0, 0, tzinfo=timezone.utc)
    policy = _policy()
    report = PlanReport(admitted=(), aliases={"prov:gone": "claude-gone"})
    previous = PreviousRunState(admitted=frozenset({"prov:gone"}))
    health = {
        "prov:gone": OfferingHealth(
            excluded=True, reason="identifier_gone", reset_at=reset_at
        )
    }

    advisory = build_advisory(policy=policy, report=report, health=health, previous=previous)

    assert len(advisory.removed_last_run) == 1
    removed = advisory.removed_last_run[0]
    assert removed.alias == "claude-gone"
    assert removed.reason == "identifier_gone"
    assert removed.refills_at == reset_at


def test_with_previous_none_the_advisorys_two_sets_are_empty_and_the_note_is_still_present():
    policy = _policy()
    report = PlanReport(admitted=("prov:a",), aliases={"prov:a": "claude-a"})

    advisory = build_advisory(policy=policy, report=report, health={}, previous=None)

    assert advisory.added_last_run == ()
    assert advisory.removed_last_run == ()
    assert advisory.note  # the note stands on its own with no previous record


# --- as_dict -----------------------------------------------------------------


def test_as_dict_carries_schema_version_and_the_client_advisory_block():
    guidance = Guidance(axis="coding", advisory=ClientAdvisory(added_last_run=("claude-a",)))

    document = guidance.as_dict()

    # Raised to "2" on 2026-07-27 when a Route gained `wide_alias`. Pinned
    # to a literal rather than to the constant, so a bump has to be
    # deliberate: a consumer parses against this value.
    assert document["schema_version"] == "2"
    assert document["client_advisory"]["added_last_run"] == ["claude-a"]


# --- rendering on an empty guidance answer -----------------------------------


def test_render_text_and_render_markdown_run_on_a_guidance_answer_with_zero_rows():
    guidance = Guidance(axis="coding")

    text = render_text(guidance)
    markdown = render_markdown(guidance)

    assert "nothing to rank" in text.lower() or "nothing is offered" in text.lower()
    assert markdown.startswith("# Model guidance")


# --- ADR 0010: an exhausted Route is available but not recommendable -------


def _exhausted_record(*, reset_at, last_attempt_at=None):
    """A Passthrough Auth quota exhaustion: recorded, never Excluded."""
    return OfferingHealth(
        excluded=False,
        reason="quota_exhausted",
        bucket="self_healing",
        reset_at=reset_at,
        last_attempt_at=last_attempt_at or (NOW - timedelta(minutes=5)),
    )


_CLAUDE_DECLARED = [
    {
        "alias": "claude-opus-5",
        "passthrough_auth": True,
        "litellm_params": {"model": "anthropic/claude-opus-5"},
    }
]


def _claude_guidance(record):
    return derive(
        feed=_feed_with(),
        policy=_policy(declared=_CLAUDE_DECLARED),
        health={"claude-opus-5": record} if record is not None else {},
        report=PlanReport(admitted=("claude-opus-5",), aliases={}),
        now=NOW,
    )


def test_an_exhausted_passthrough_route_is_available_but_not_callable_now():
    """The harm this fixes. A Passthrough Auth Offering is never
    Excluded on a quota exhaustion, so `available` stayed True and
    `callable_now` reported the exhausted Claude subscription as fine --
    the one field the model-routing skill tells agents to trust."""
    guidance = _claude_guidance(_exhausted_record(reset_at=NOW + timedelta(hours=6)))

    row = guidance.rows[0]
    route = row.routes[0]
    assert route.available is True  # still in the Generated Config
    assert route.exhausted is True
    assert route.recommendable is False
    assert row.callable_now is False


def test_a_passed_reset_time_makes_the_route_recommendable_again():
    guidance = _claude_guidance(_exhausted_record(reset_at=NOW - timedelta(minutes=1)))

    assert guidance.rows[0].routes[0].exhausted is False
    assert guidance.rows[0].callable_now is True


def test_an_exhaustion_with_no_reset_time_expires_after_the_staleness_window():
    """Nothing can clear it on its own: the Journal records only
    failures, and a Passthrough Auth Offering cannot be probed. Left
    unbounded it would hide a working Offering forever."""
    hours = _policy().schedule.maximum_staleness_hours

    fresh = _claude_guidance(
        _exhausted_record(reset_at=None, last_attempt_at=NOW - timedelta(hours=hours / 2))
    )
    stale = _claude_guidance(
        _exhausted_record(reset_at=None, last_attempt_at=NOW - timedelta(hours=hours + 1))
    )

    assert fresh.rows[0].routes[0].exhausted is True
    assert stale.rows[0].routes[0].exhausted is False


def test_a_row_fails_over_to_a_healthy_route_when_one_is_exhausted():
    """Demotion is per Route. The row itself stays callable."""
    feed = _feed_with(
        _offering_raw(id="provA:m", provider_id="provA", canonical_model_id="model-x"),
        _offering_raw(id="provB:m", provider_id="provB", canonical_model_id="model-x"),
    )
    report = PlanReport(
        admitted=("provA:m", "provB:m"),
        aliases={"provA:m": "claude-a", "provB:m": "claude-b"},
    )

    guidance = derive(
        feed=feed,
        policy=_policy(providers={"provA": {"mode": "all"}, "provB": {"mode": "all"}}),
        health={"provA:m": _exhausted_record(reset_at=NOW + timedelta(hours=6))},
        report=report,
        now=NOW,
    )

    row = guidance.rows[0]
    assert any(r.exhausted for r in row.routes)
    assert row.callable_now is True
    assert row.best_route.offering_id == "provB:m"


def test_an_exhausted_route_sorts_behind_a_recommendable_one():
    """The route list is a failover order and the skill tells agents to
    call the first route. Sorting on `available` put an exhausted
    Passthrough Auth Alias first, which is ADR 0010's harm one level
    down."""
    feed = _feed_with(
        _offering_raw(
            id="cheap:m", provider_id="cheap", canonical_model_id="model-x",
            pricing_kind="free",
        ),
        _offering_raw(
            id="dear:m", provider_id="dear", canonical_model_id="model-x",
            pricing_kind="paid",
        ),
    )
    report = PlanReport(
        admitted=("cheap:m", "dear:m"),
        aliases={"cheap:m": "claude-cheap", "dear:m": "claude-dear"},
    )

    guidance = derive(
        feed=feed,
        policy=_policy(providers={"cheap": {"mode": "all"}, "dear": {"mode": "all"}}),
        # The CHEAPER route is exhausted, so cost alone would still put
        # it first.
        health={"cheap:m": _exhausted_record(reset_at=NOW + timedelta(hours=6))},
        report=report,
        now=NOW,
    )

    order = [r.offering_id for r in guidance.rows[0].routes]
    assert order == ["dear:m", "cheap:m"]
    assert guidance.rows[0].routes[0].recommendable is True
