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

from litellm_maintainer.codexbar import CodexbarIdentity, CodexbarReading, CodexbarWindow
from litellm_maintainer.entitlements import FLAT_RATE, FREE, METERED
from litellm_maintainer.feed import Feed, parse_feed
from litellm_maintainer.guidance import (
    AXES,
    NOT_RECOMMENDED_EXHAUSTED,
    NOT_RECOMMENDED_HEADROOM,
    NOT_RECOMMENDED_HEALTH,
    ClientAdvisory,
    Guidance,
    GuidanceError,
    Route,
    build_advisory,
    derive,
    render_markdown,
    render_text,
)
from litellm_maintainer.headroom import HeadroomRecord, HeadroomState
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


# --- Route.tier (ticket 12) ---------------------------------------------------


def test_a_route_with_no_policy_allowances_entry_publishes_no_tier():
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov", canonical_model_id="m"))
    policy = _policy(providers={"prov": {"mode": "all"}})
    report = PlanReport(admitted=("prov:a",))

    guidance = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)

    route = guidance.rows[0].routes[0]
    assert route.tier is None
    assert route.as_dict()["tier"] is None


def test_a_route_publishes_its_allowances_tier_verbatim():
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov", canonical_model_id="m"))
    policy = _policy(
        providers={"prov": {"mode": "all"}},
        allowances={"provider:prov": {"tier": "claude-max-5x"}},
    )
    report = PlanReport(admitted=("prov:a",))

    guidance = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)

    route = guidance.rows[0].routes[0]
    assert route.tier == "claude-max-5x"
    assert route.as_dict()["tier"] == "claude-max-5x"


def test_a_declared_routes_tier_comes_from_its_own_allowance():
    policy = _policy(
        declared=[
            {
                "alias": "claude-direct-1",
                "litellm_params": {"model": "anthropic/claude-x"},
                "entitlement_pool": "claude-subscription",
            }
        ],
        allowances={"pool:claude-subscription": {"tier": "claude-max-5x"}},
    )
    report = PlanReport(admitted=("claude-direct-1",))

    guidance = derive(feed=_feed_with(), policy=policy, health={}, report=report, now=NOW)

    route = guidance.rows[0].routes[0]
    assert route.tier == "claude-max-5x"


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

    # Raised to "2" on 2026-07-27 when a Route gained `wide_alias`, and to
    # "3" on 2026-07-28 when a row gained `score_source` and a Route gained
    # `rate_source`. Pinned to a literal rather than to the constant, so a
    # bump has to be deliberate: a consumer parses against this value.
    assert document["schema_version"] == "3"
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


def test_an_excluded_route_is_available_but_not_recommendable():
    """ADR 0014. An Excluded Offering stays in the Generated Config, so
    `available` is True and a caller can still reach it. It must not be
    recommended: the maintainer called it and was told no."""
    guidance = _claude_guidance(
        OfferingHealth(
            excluded=True,
            reason="gateway_error",
            bucket="self_healing",
            last_attempt_at=NOW - timedelta(minutes=5),
        )
    )

    row = guidance.rows[0]
    route = row.routes[0]
    assert route.available is True  # ADR 0014: still in the Generated Config
    assert route.recommendable is False
    assert route.not_recommended_because == NOT_RECOMMENDED_HEALTH
    assert row.callable_now is False


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


# --- Headroom on a Route (ticket 05) ----------------------------------------
#
# `derive`'s `headroom_state` is optional and defaults to `None`, so every
# test above this line still describes routes with `headroom` reading
# `None` throughout, exactly as before this ticket. These tests cover what
# changes once a `HeadroomState` is supplied. Figures are taken from the
# headroom spec's measured facts (codexbar-sample.json, 2026-07-28):
# `claude` binds at 82% on its 10080-minute window, resetting
# 2026-07-30T19:00:00Z; `clinepass` binds at 100% on its 43200-minute
# window with two 0%-reading siblings; `openrouter` and `deepseek` answer
# with no windows at all.


def _headroom_record(
    allowance_id: str,
    *,
    primary: CodexbarWindow | None = None,
    secondary: CodexbarWindow | None = None,
    tertiary: CodexbarWindow | None = None,
    updated_at: str | None = "2026-07-26T11:55:00Z",
    read_at: str = "2026-07-26T11:59:00Z",
    source: str | None = None,
) -> HeadroomRecord:
    return HeadroomRecord(
        allowance_id=allowance_id,
        source=source or f"codexbar:{allowance_id}/",
        reading=CodexbarReading(
            provider="prov",
            identity=CodexbarIdentity(provider_id="prov", account_email="operator@example.com"),
            primary=primary,
            secondary=secondary,
            tertiary=tertiary,
            extra_windows=(),
            updated_at=updated_at,
            error=None,
        ),
        read_at=read_at,
    )


# `derive` now publishes a stored record only when Policy still declares its
# Allowance AND the record's own `source` still matches (defect 2). Every
# test below that expects a populated `headroom` passes this alongside
# `_headroom_record`'s own default `source`, `f"codexbar:{allowance_id}/"`.
MAPPED_HEADROOM_SOURCES = {"provider:prov": "codexbar:provider:prov/"}


def _single_route_guidance(
    *, headroom_state: HeadroomState | None, headroom_sources: dict[str, str] | None = None
) -> Guidance:
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
    )
    policy_kwargs: dict[str, Any] = {"providers": {"prov": {"mode": "all"}}}
    if headroom_sources is not None:
        policy_kwargs["headroom"] = {"sources": headroom_sources}
    policy = _policy(**policy_kwargs)
    report = PlanReport(admitted=("prov:a",), aliases={"prov:a": "claude-a"})
    return derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=headroom_state
    )


def test_a_route_with_no_headroom_source_publishes_an_explicit_null():
    guidance = _single_route_guidance(headroom_state=HeadroomState())

    route = guidance.rows[0].routes[0]
    assert route.headroom is None
    assert route.as_dict()["headroom"] is None


def test_a_route_with_no_headroom_state_read_at_all_publishes_null_too():
    guidance = _single_route_guidance(headroom_state=None)

    assert guidance.rows[0].routes[0].headroom is None


def test_a_live_binding_window_is_published_on_its_route():
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(
                    used_percent=82, window_minutes=10080, resets_at="2026-07-30T19:00:00Z"
                ),
            )
        }
    )

    guidance = _single_route_guidance(
        headroom_state=state, headroom_sources=MAPPED_HEADROOM_SOURCES
    )

    headroom = guidance.rows[0].routes[0].headroom
    assert headroom is not None
    assert headroom.used_percent == 82
    assert headroom.window_minutes == 10080
    assert headroom.resets_at == "2026-07-30T19:00:00Z"
    assert headroom.age_seconds == 300.0


def test_a_route_publishes_the_binding_figure_only_never_the_window_set():
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=0, window_minutes=300, resets_at=None),
                secondary=CodexbarWindow(used_percent=0, window_minutes=10080, resets_at=None),
                tertiary=CodexbarWindow(
                    used_percent=100, window_minutes=43200, resets_at="2026-08-03T10:19:00Z"
                ),
            )
        }
    )

    guidance = _single_route_guidance(
        headroom_state=state, headroom_sources=MAPPED_HEADROOM_SOURCES
    )

    route = guidance.rows[0].routes[0]
    # ClinePass's own shape: two idle windows beside a fully-drawn one. The
    # Route must bind on the WORST live window, 100%, never on a named
    # window that happens to read free.
    assert route.headroom.used_percent == 100
    as_dict = route.as_dict()
    headroom_dict = as_dict["headroom"]
    assert set(headroom_dict) == {"used_percent", "window_minutes", "resets_at", "age_seconds"}
    assert "primary" not in headroom_dict
    assert "secondary" not in headroom_dict
    assert "tertiary" not in headroom_dict
    assert "windows" not in as_dict


def test_a_reading_with_no_windows_at_all_publishes_no_headroom():
    # Measured 2026-07-28: openrouter and deepseek both answer this way.
    state = HeadroomState(records={"provider:prov": _headroom_record("provider:prov")})

    guidance = _single_route_guidance(
        headroom_state=state, headroom_sources=MAPPED_HEADROOM_SOURCES
    )

    assert guidance.rows[0].routes[0].headroom is None


def test_a_void_window_publishes_no_headroom_rather_than_a_healthy_zero():
    # A window whose own resets_at has already passed describes a period
    # that ended: NOW is 2026-07-26T12:00:00Z, the reset was 11:30:00Z.
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(
                    used_percent=100, window_minutes=300, resets_at="2026-07-26T11:30:00Z"
                ),
                updated_at="2026-07-26T11:00:00Z",
            )
        }
    )

    guidance = _single_route_guidance(
        headroom_state=state, headroom_sources=MAPPED_HEADROOM_SOURCES
    )

    assert guidance.rows[0].routes[0].headroom is None


def test_two_routes_sharing_one_allowance_both_carry_its_headroom():
    # Six ChatGPT aliases share one ceiling in the real system; two routes
    # on one provider stand in for that here.
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-a"),
        _offering_raw(id="prov:b", provider_id="prov", canonical_model_id="model-b"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}}, headroom={"sources": MAPPED_HEADROOM_SOURCES}
    )
    report = PlanReport(
        admitted=("prov:a", "prov:b"),
        aliases={"prov:a": "claude-a", "prov:b": "claude-b"},
    )
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=82, window_minutes=10080, resets_at=None),
            )
        }
    )

    guidance = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    headrooms = {row.routes[0].headroom.used_percent for row in guidance.rows}
    assert headrooms == {82}


def test_a_headroom_reading_changes_no_ordering_or_recommendability():
    """Stage 1 publishes; it does not demote (headroom spec, decision 15).

    The same Feed, Policy and Health, run once with a fully-drawn
    Headroom and once with none at all, must produce identical
    `recommendable`, `callable_now` and row/Route ordering.
    """
    feed = _feed_with(
        _offering_raw(
            id="cheap:m", provider_id="cheap", canonical_model_id="model-x",
            pricing_kind="free", coding_score=10.0,
        ),
        _offering_raw(
            id="dear:m", provider_id="dear", canonical_model_id="model-y",
            pricing_kind="paid", coding_score=90.0,
        ),
    )
    policy = _policy(
        providers={"cheap": {"mode": "all"}, "dear": {"mode": "all"}},
        headroom={
            "sources": {
                "provider:cheap": "codexbar:provider:cheap/",
                "provider:dear": "codexbar:provider:dear/",
            }
        },
    )
    report = PlanReport(
        admitted=("cheap:m", "dear:m"),
        aliases={"cheap:m": "claude-cheap", "dear:m": "claude-dear"},
    )

    without_headroom = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)
    drained_state = HeadroomState(
        records={
            "provider:cheap": _headroom_record(
                "provider:cheap",
                primary=CodexbarWindow(used_percent=100, window_minutes=10080, resets_at=None),
            ),
            "provider:dear": _headroom_record(
                "provider:dear",
                primary=CodexbarWindow(used_percent=100, window_minutes=10080, resets_at=None),
            ),
        }
    )
    with_headroom = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=drained_state
    )

    assert [r.canonical_model_id for r in without_headroom.rows] == [
        r.canonical_model_id for r in with_headroom.rows
    ]
    for before, after in zip(without_headroom.rows, with_headroom.rows):
        assert before.callable_now == after.callable_now
        assert [r.alias for r in before.routes] == [r.alias for r in after.routes]
        assert [r.recommendable for r in before.routes] == [r.recommendable for r in after.routes]
        # The Headroom did land, so this pins that a 100%-drawn Route was
        # published without being demoted -- not that nothing changed.
        assert all(r.headroom is None for r in before.routes)
        assert any(r.headroom is not None for r in after.routes)
    assert all(r.recommendable for row in with_headroom.rows for r in row.routes)


def test_render_text_shows_the_binding_figure_and_omits_it_when_absent():
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=82, window_minutes=10080, resets_at=None),
            )
        }
    )

    with_headroom = render_text(
        _single_route_guidance(headroom_state=state, headroom_sources=MAPPED_HEADROOM_SOURCES)
    )
    without_headroom = render_text(_single_route_guidance(headroom_state=HeadroomState()))

    assert "headroom 82%" in with_headroom
    assert "headroom" not in without_headroom


def test_render_markdown_shows_the_binding_figure_and_a_dash_when_absent():
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=82, window_minutes=10080, resets_at=None),
            )
        }
    )

    with_headroom = render_markdown(
        _single_route_guidance(headroom_state=state, headroom_sources=MAPPED_HEADROOM_SOURCES)
    )
    without_headroom = render_markdown(_single_route_guidance(headroom_state=HeadroomState()))

    assert "82%" in with_headroom
    assert "| Headroom |" in with_headroom
    assert "—" in without_headroom


# --- Demotion at 100%, behind `headroom.demote_at_full` (ticket 08) --------
#
# The flag defaults to `False`. Every test in this section that leaves it
# unset, or sets it explicitly `False`, must produce EXACTLY the answer
# `derive` gave before this ticket -- that is the pin. The tests that set
# it `True` describe the capability the flag exists to gate.


def _fully_drawn_state(allowance_id: str = "provider:prov") -> HeadroomState:
    return HeadroomState(
        records={
            allowance_id: _headroom_record(
                allowance_id,
                primary=CodexbarWindow(used_percent=100, window_minutes=10080, resets_at=None),
            )
        }
    )


def test_the_flag_absent_is_byte_identical_to_no_headroom_at_all():
    """The pin. Policy states no `headroom.demote_at_full` at all, so a
    100%-drawn Reading changes nothing: not the Route's `as_dict()`, not
    the row's `callable_now`, not any ordering.
    """
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}}, headroom={"sources": MAPPED_HEADROOM_SOURCES}
    )
    assert policy.headroom.demote_at_full is False
    report = PlanReport(admitted=("prov:a",), aliases={"prov:a": "claude-a"})

    without_headroom = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)
    with_full_reading = derive(
        feed=feed,
        policy=policy,
        health={},
        report=report,
        now=NOW,
        headroom_state=_fully_drawn_state(),
    )

    before = without_headroom.rows[0].routes[0]
    after = with_full_reading.rows[0].routes[0]
    assert before.recommendable is True
    assert after.recommendable is True
    assert after.headroom is not None and after.headroom.used_percent == 100
    assert after.not_recommended_because is None
    before_dict = dict(before.as_dict(), headroom=None)
    after_dict = dict(after.as_dict(), headroom=None)
    assert before_dict == after_dict
    assert without_headroom.rows[0].callable_now == with_full_reading.rows[0].callable_now


def test_the_flag_explicitly_false_also_changes_nothing():
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}},
        headroom={"sources": MAPPED_HEADROOM_SOURCES, "demote_at_full": False},
    )
    report = PlanReport(admitted=("prov:a",), aliases={"prov:a": "claude-a"})

    guidance = derive(
        feed=feed,
        policy=policy,
        health={},
        report=report,
        now=NOW,
        headroom_state=_fully_drawn_state(),
    )

    route = guidance.rows[0].routes[0]
    assert route.recommendable is True
    assert route.not_recommended_because is None
    assert route.demoted_by_headroom is False


def test_the_flag_on_and_a_full_reading_demotes_the_route():
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}},
        headroom={"sources": MAPPED_HEADROOM_SOURCES, "demote_at_full": True},
    )
    report = PlanReport(admitted=("prov:a",), aliases={"prov:a": "claude-a"})

    guidance = derive(
        feed=feed,
        policy=policy,
        health={},
        report=report,
        now=NOW,
        headroom_state=_fully_drawn_state(),
    )

    route = guidance.rows[0].routes[0]
    assert route.recommendable is False
    assert route.not_recommended_because == NOT_RECOMMENDED_HEADROOM
    assert route.as_dict()["not_recommended_because"] == "headroom"


def test_the_flag_on_and_ninety_nine_percent_demotes_nothing():
    """No warn band. The line sits only where the provider draws it."""
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}},
        headroom={"sources": MAPPED_HEADROOM_SOURCES, "demote_at_full": True},
    )
    report = PlanReport(admitted=("prov:a",), aliases={"prov:a": "claude-a"})
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=99, window_minutes=10080, resets_at=None),
            )
        }
    )

    guidance = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    route = guidance.rows[0].routes[0]
    assert route.recommendable is True
    assert route.not_recommended_because is None


def test_the_flag_on_and_a_void_reading_demotes_nothing():
    """A void window (past its own reset) publishes no Headroom at all
    (`_route_headroom` already returns `None`), so it must not demote
    even with the flag on."""
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}},
        headroom={"sources": MAPPED_HEADROOM_SOURCES, "demote_at_full": True},
    )
    report = PlanReport(admitted=("prov:a",), aliases={"prov:a": "claude-a"})
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(
                    used_percent=100, window_minutes=300, resets_at="2026-07-26T11:30:00Z"
                ),
                updated_at="2026-07-26T11:00:00Z",
            )
        }
    )

    guidance = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    route = guidance.rows[0].routes[0]
    assert route.headroom is None
    assert route.recommendable is True
    assert route.not_recommended_because is None


def test_the_flag_on_and_an_absent_reading_demotes_nothing():
    """No Headroom source declared for this Allowance at all: `headroom`
    reads `None`, and `None` must never demote (headroom spec, decision
    7 in ticket 08's issue), whatever the flag says."""
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}}, headroom={"demote_at_full": True}
    )
    report = PlanReport(admitted=("prov:a",), aliases={"prov:a": "claude-a"})

    guidance = derive(
        feed=feed,
        policy=policy,
        health={},
        report=report,
        now=NOW,
        headroom_state=HeadroomState(),
    )

    route = guidance.rows[0].routes[0]
    assert route.headroom is None
    assert route.recommendable is True
    assert route.not_recommended_because is None


def test_a_row_of_all_demoted_routes_reports_not_callable_now():
    feed = _feed_with(
        _offering_raw(id="provA:m", provider_id="provA", canonical_model_id="model-x"),
        _offering_raw(id="provB:m", provider_id="provB", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"provA": {"mode": "all"}, "provB": {"mode": "all"}},
        headroom={
            "sources": {
                "provider:provA": "codexbar:provider:provA/",
                "provider:provB": "codexbar:provider:provB/",
            },
            "demote_at_full": True,
        },
    )
    report = PlanReport(
        admitted=("provA:m", "provB:m"),
        aliases={"provA:m": "claude-a", "provB:m": "claude-b"},
    )
    state = HeadroomState(
        records={
            "provider:provA": _headroom_record(
                "provider:provA",
                primary=CodexbarWindow(used_percent=100, window_minutes=10080, resets_at=None),
            ),
            "provider:provB": _headroom_record(
                "provider:provB",
                primary=CodexbarWindow(used_percent=100, window_minutes=10080, resets_at=None),
            ),
        }
    )

    guidance = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    row = guidance.rows[0]
    assert all(not r.recommendable for r in row.routes)
    assert row.callable_now is False


def test_an_observed_exhaustion_is_never_cleared_by_a_healthy_reading():
    """Decision 6 / 8: a Headroom may demote a Route, it may never
    un-demote one. An exhausted Route stays not recommendable no matter
    what a later Reading says -- even a Reading reporting 0% used, and
    even with the flag on."""
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}},
        headroom={"sources": MAPPED_HEADROOM_SOURCES, "demote_at_full": True},
    )
    report = PlanReport(admitted=("prov:a",), aliases={"prov:a": "claude-a"})
    health = {"prov:a": _exhausted_record(reset_at=NOW + timedelta(hours=6))}
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=0, window_minutes=10080, resets_at=None),
            )
        }
    )

    guidance = derive(
        feed=feed, policy=policy, health=health, report=report, now=NOW, headroom_state=state
    )

    route = guidance.rows[0].routes[0]
    assert route.exhausted is True
    assert route.recommendable is False
    assert route.not_recommended_because == NOT_RECOMMENDED_EXHAUSTED


def test_cause_names_health_when_the_route_is_simply_excluded():
    feed = _feed_with(
        _offering_raw(id="provA:m", provider_id="provA", canonical_model_id="model-w"),
    )
    policy = _policy(providers={"provA": {"mode": "all"}})
    report = PlanReport(excluded=("provA:m",), aliases={"provA:m": "claude-a"})
    health = {
        "provA:m": OfferingHealth(
            excluded=True, reason="rate_limited", reset_at=NOW + timedelta(hours=1)
        )
    }

    guidance = derive(feed=feed, policy=policy, health=health, report=report, now=NOW)

    route = guidance.rows[0].routes[0]
    assert route.available is False
    assert route.recommendable is False
    assert route.not_recommended_because == NOT_RECOMMENDED_HEALTH


def test_a_health_exclusion_outranks_a_demoting_headroom_in_the_cause_field():
    """A fact always outranks a report: an Excluded Route names its own
    cause even when a Headroom also reads 100% for it."""
    feed = _feed_with(
        _offering_raw(id="provA:m", provider_id="provA", canonical_model_id="model-w"),
    )
    policy = _policy(
        providers={"provA": {"mode": "all"}},
        headroom={
            "sources": {"provider:provA": "codexbar:provider:provA/"},
            "demote_at_full": True,
        },
    )
    report = PlanReport(excluded=("provA:m",), aliases={"provA:m": "claude-a"})
    health = {
        "provA:m": OfferingHealth(
            excluded=True, reason="rate_limited", reset_at=NOW + timedelta(hours=1)
        )
    }
    state = _fully_drawn_state("provider:provA")

    guidance = derive(
        feed=feed, policy=policy, health=health, report=report, now=NOW, headroom_state=state
    )

    route = guidance.rows[0].routes[0]
    assert route.available is False
    assert route.recommendable is False
    assert route.not_recommended_because == NOT_RECOMMENDED_HEALTH


def test_demotion_writes_nothing_to_health_state_or_generated_config():
    """`derive` is a pure transform (module docstring): it reads Feed,
    Policy, Health and Headroom State and returns a value. Ticket 08
    adds a demotion RULE, never a write path. Pin that no writer of
    either file is even reachable from this module."""
    import litellm_maintainer.guidance as guidance_module

    for name in ("write_health", "write_config", "write_headroom"):
        assert not hasattr(guidance_module, name), (
            f"guidance.py must not import {name!r}; a Reading may only be "
            "read here, never written, and the Generated Config is never "
            "rewritten on one (headroom spec, decision 7)"
        )

    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}},
        headroom={"sources": MAPPED_HEADROOM_SOURCES, "demote_at_full": True},
    )
    report = PlanReport(admitted=("prov:a",), aliases={"prov:a": "claude-a"})

    # Calling derive() with a fully-drawn Reading must not raise even
    # though no health/config file exists on disk anywhere near this
    # test -- there is no path here that could touch either file.
    guidance = derive(
        feed=feed,
        policy=policy,
        health={},
        report=report,
        now=NOW,
        headroom_state=_fully_drawn_state(),
    )
    assert guidance.rows[0].routes[0].recommendable is False


# --- Defect 2: a stored Reading must not survive its source leaving Policy -


def test_removing_an_allowances_headroom_source_publishes_null_though_a_record_remains():
    """The operator's real case: Gemini was mapped 2026-07-28 and unmapped
    2026-07-29 because its figure's meaning was unknown. Pruning the file
    is `refresh_headroom`'s own job, run out of band; `derive` must not
    trust a stale record for an Allowance Policy no longer names."""
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=100, window_minutes=1440, resets_at=None),
            )
        }
    )

    guidance = _single_route_guidance(headroom_state=state)  # headroom_sources=None -> {}

    assert guidance.rows[0].routes[0].headroom is None


def test_a_remapped_source_publishes_null_until_a_fresh_reading_matches():
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=100, window_minutes=1440, resets_at=None),
            )
        }
    )
    # Policy now points this Allowance at a DIFFERENT codexbar identity
    # than the one the stored record's own `source` names.
    guidance = _single_route_guidance(
        headroom_state=state,
        headroom_sources={"provider:prov": "codexbar:provider:prov/new-account@example.com"},
    )

    assert guidance.rows[0].routes[0].headroom is None


def test_an_unmapped_allowances_stale_record_never_demotes_even_with_the_flag_on():
    """The worst case the defect names: Policy declares NO headroom
    source, `demote_at_full` is on, and a stale, fully-drawn record for
    this Allowance still sits in Headroom State. The Route must publish
    `headroom: null` and must stay recommendable."""
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
    )
    policy = _policy(providers={"prov": {"mode": "all"}}, headroom={"demote_at_full": True})
    report = PlanReport(admitted=("prov:a",), aliases={"prov:a": "claude-a"})

    guidance = derive(
        feed=feed,
        policy=policy,
        health={},
        report=report,
        now=NOW,
        headroom_state=_fully_drawn_state(),
    )

    route = guidance.rows[0].routes[0]
    assert route.headroom is None
    assert route.recommendable is True
    assert route.not_recommended_because is None


# --- Defect 7(b): a headroom-demoted Route must read demoted, not "available"


def test_render_text_states_demoted_not_plain_available():
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}},
        headroom={"sources": MAPPED_HEADROOM_SOURCES, "demote_at_full": True},
    )
    report = PlanReport(admitted=("prov:a",), aliases={"prov:a": "claude-a"})

    guidance = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW,
        headroom_state=_fully_drawn_state(),
    )
    route = guidance.rows[0].routes[0]
    assert route.available is True
    assert route.recommendable is False

    text = render_text(guidance)

    # Before this fix, the line read plain "available": `render_text` read
    # `route.available` alone, which stays True for a headroom demotion.
    # The demotion was then visible only in JSON's `recommendable` field.
    line = next(line for line in text.splitlines() if "claude-a" in line)
    assert "not recommended" in line
    assert "headroom" in line


def test_render_markdown_states_demoted_not_plain_available():
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}},
        headroom={"sources": MAPPED_HEADROOM_SOURCES, "demote_at_full": True},
    )
    report = PlanReport(admitted=("prov:a",), aliases={"prov:a": "claude-a"})

    guidance = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW,
        headroom_state=_fully_drawn_state(),
    )

    markdown = render_markdown(guidance)

    line = next(line for line in markdown.splitlines() if "claude-a" in line)
    assert "not recommended" in line


def test_render_text_still_says_plain_available_when_not_demoted():
    # The pin: an ordinary recommendable Route's line is unchanged.
    guidance = _single_route_guidance(headroom_state=None)

    text = render_text(guidance)

    line = next(line for line in text.splitlines() if "claude-a" in line)
    assert line.endswith("available")
    assert "not recommended" not in line


# --- The row-level consequence of a demotion, asked 2026-07-29 ------------
#
# A downstream consumer keeps a Route whose only reason is `"headroom"`,
# because a Reading is a report and never a measured refusal. It then asked
# what `callable_now` does when EVERY Route in a row is demoted, and could
# not test it with `demote_at_full` off. These two answer it.


def test_a_row_whose_every_route_is_headroom_demoted_reports_callable_now_false():
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}},
        headroom={"sources": MAPPED_HEADROOM_SOURCES, "demote_at_full": True},
    )
    report = PlanReport(admitted=("prov:a",), aliases={"prov:a": "claude-a"})
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=100, window_minutes=10080, resets_at=None),
            )
        }
    )

    guidance = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )
    row = guidance.rows[0]

    # `callable_now` is `best_route is not None`, and `best_route` is the
    # first recommendable Route. A demoted Route is not recommendable, so a
    # row of only demoted Routes reports false -- the same value an
    # all-exhausted row reports.
    assert row.routes[0].not_recommended_because == "headroom"
    assert row.callable_now is False


def test_one_undemoted_route_keeps_its_row_callable():
    # The row stays callable while any Route in it is recommendable. So a
    # consumer that drops a whole row on `callable_now: false` loses a
    # demoted Route only where EVERY Route in that row is demoted.
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
        _offering_raw(id="other:a", provider_id="other", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}, "other": {"mode": "all"}},
        headroom={"sources": MAPPED_HEADROOM_SOURCES, "demote_at_full": True},
    )
    report = PlanReport(
        admitted=("prov:a", "other:a"),
        aliases={"prov:a": "claude-a", "other:a": "claude-other-a"},
    )
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=100, window_minutes=10080, resets_at=None),
            )
        }
    )

    guidance = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )
    row = guidance.rows[0]

    assert row.callable_now is True
    assert any(r.not_recommended_because == "headroom" for r in row.routes)


def test_a_row_demoted_only_by_readings_names_the_report_as_its_cause():
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}},
        headroom={"sources": MAPPED_HEADROOM_SOURCES, "demote_at_full": True},
    )
    report = PlanReport(admitted=("prov:a",), aliases={"prov:a": "claude-a"})
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=100, window_minutes=10080, resets_at=None),
            )
        }
    )

    row = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    ).rows[0]

    assert row.not_callable_because == "headroom"
    # And `why` no longer states something false about it: a demoted Route
    # is available and not exhausted, so "every Route is excluded" was wrong.
    assert "excluded" not in row.why
    assert "report and not a measured refusal" in row.why


def test_a_callable_row_names_no_cause():
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
    )
    policy = _policy(providers={"prov": {"mode": "all"}})
    report = PlanReport(admitted=("prov:a",), aliases={"prov:a": "claude-a"})

    row = derive(feed=feed, policy=policy, health={}, report=report, now=NOW).rows[0]

    assert row.callable_now is True
    assert row.not_callable_because is None


def test_a_row_with_one_excluded_route_names_the_measured_cause_not_the_report():
    # A fact outranks a report at the row level too. A mixed row names the
    # measured cause; each Route's own field says which are merely demoted.
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
        _offering_raw(id="other:a", provider_id="other", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}, "other": {"mode": "all"}},
        headroom={"sources": MAPPED_HEADROOM_SOURCES, "demote_at_full": True},
    )
    report = PlanReport(
        admitted=("prov:a",),
        excluded=("other:a",),
        aliases={"prov:a": "claude-a", "other:a": "claude-other-a"},
    )
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=100, window_minutes=10080, resets_at=None),
            )
        }
    )

    row = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    ).rows[0]

    assert row.callable_now is False
    assert row.not_callable_because == "health"
    assert any(r.not_recommended_because == "headroom" for r in row.routes)


# --- Two guarantees a downstream gate is built on, asked 2026-07-29 ------


def test_a_callable_row_reads_null_even_while_it_holds_a_demoted_route():
    # Answers question 1. The row keeps a recommendable Route, so it stays
    # callable and states no cause; each Route carries its own.
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
        _offering_raw(id="other:a", provider_id="other", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}, "other": {"mode": "all"}},
        headroom={"sources": MAPPED_HEADROOM_SOURCES, "demote_at_full": True},
    )
    report = PlanReport(
        admitted=("prov:a", "other:a"),
        aliases={"prov:a": "claude-a", "other:a": "claude-other-a"},
    )
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=100, window_minutes=10080, resets_at=None),
            )
        }
    )

    row = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    ).rows[0]

    assert row.callable_now is True
    assert row.not_callable_because is None
    assert any(r.not_recommended_because == "headroom" for r in row.routes)
    assert any(r.not_recommended_because is None for r in row.routes)


def test_callable_now_is_false_exactly_when_every_route_states_a_cause():
    """Answers question 2, and pins the equivalence a row gate rests on.

    `callable_now` is `best_route is not None`, and `best_route` is the
    first recommendable Route. So the row is uncallable exactly when every
    Route carries a non-null `not_recommended_because`, and a consumer
    that filters Routes by their own reason needs no row gate for
    correctness. The row's own field is a summary, never a fact the Routes
    lack.

    Stated as a test because a downstream gate is built on it. Deriving it
    from two other properties by reading the code is exactly the coupling
    that breaks silently.
    """
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
        _offering_raw(id="other:a", provider_id="other", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}, "other": {"mode": "all"}},
        headroom={"sources": MAPPED_HEADROOM_SOURCES, "demote_at_full": True},
    )
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=100, window_minutes=10080, resets_at=None),
            )
        }
    )

    for report in (
        PlanReport(admitted=("prov:a", "other:a"), aliases={"prov:a": "claude-a"}),
        PlanReport(admitted=("prov:a",), excluded=("other:a",), aliases={"prov:a": "claude-a"}),
        PlanReport(admitted=(), excluded=("prov:a", "other:a")),
    ):
        for row in derive(
            feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
        ).rows:
            every_route_states_a_cause = all(
                route.not_recommended_because is not None for route in row.routes
            )
            assert row.callable_now is not every_route_states_a_cause


def test_a_row_can_hold_an_exhausted_route_beside_one_only_demoted():
    """The mixed shape a downstream gate asked about. It OCCURS.

    A Passthrough Auth Route exhausted (recorded, never Excluded -- ADR
    0010) and a Feed Route demoted by a 100% Reading, joined into one row
    by a Reference Model. Neither is recommendable, so the row is
    uncallable, and a fact outranks a report, so the row states
    `"exhausted"`. The demoted Route beside it is still `available: true`
    and `exhausted: false`: nothing measured it.

    A consumer that drops the whole row on that reason therefore refuses
    one Route on the strength of a Reading. Reading each Route's own
    `not_recommended_because` is the intended behaviour, and
    `test_callable_now_is_false_exactly_when_every_route_states_a_cause`
    is what makes it safe: the row's reason is a summary, never a fact the
    Routes lack.
    """
    declared = [
        {
            "alias": "claude-seat",
            "passthrough_auth": True,
            "reference_model": "model-x",
            "entitlement_pool": "seatpool",
            "litellm_params": {"model": "anthropic/x"},
        }
    ]
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", canonical_model_id="model-x"),
    )
    policy = _policy(
        providers={"prov": {"mode": "all"}},
        declared=declared,
        headroom={"sources": MAPPED_HEADROOM_SOURCES, "demote_at_full": True},
    )
    report = PlanReport(admitted=("prov:a", "claude-seat"), aliases={"prov:a": "claude-a"})
    health = {"claude-seat": _exhausted_record(reset_at=NOW + timedelta(hours=2))}
    state = HeadroomState(
        records={
            "provider:prov": _headroom_record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=100, window_minutes=10080, resets_at=None),
            )
        }
    )

    row = derive(
        feed=feed, policy=policy, health=health, report=report, now=NOW, headroom_state=state
    ).rows[0]

    assert {r.not_recommended_because for r in row.routes} == {"exhausted", "headroom"}
    assert row.callable_now is False
    assert row.not_callable_because == "exhausted"
    demoted = next(r for r in row.routes if r.not_recommended_because == "headroom")
    assert demoted.available is True
    assert demoted.exhausted is False
