"""Tests for `entitlements.py`: the Entitlement view.

Assert external behaviour: what `derive` reports for a provider's
`state`, its `earliest_refill_at`, its cost bases, and what `render_text`
says about a shared pool. A test name states a rule an operator would
recognise (spec's "What makes a good test here"), following
tests/test_pricing.py and tests/test_report.py.

ADR 0004 is the module's central rule: an Entitlement explains a
measured split, it never infers one Offering's fate from a sibling's.
Several tests pin that directly — see the "never propagates" tests
below.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from litellm_maintainer.entitlements import (
    FLAT_RATE,
    FREE,
    METERED,
    UNKNOWN_BASIS,
    EntitlementView,
    cost_basis_for_pricing_kind,
    derive,
    render_markdown,
    render_text,
)
from litellm_maintainer.feed import Feed, parse_feed
from litellm_maintainer.plan import PlanReport
from litellm_maintainer.policy import parse_policy
from litellm_maintainer.reduce import OfferingHealth

NOW = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)


# --- Fixture builders, matching tests/test_pricing.py's style --------------


def _offering_raw(
    *,
    id: str,
    provider_id: str,
    pricing_kind: str = "free",
) -> dict[str, Any]:
    model_id = id.split(":", 1)[1]
    return {
        "id": id,
        "provider": {"id": provider_id},
        "provider_model_id": model_id,
        "capabilities": ["tool_use"],
        "endpoint": {"base_url": f"https://{provider_id}.example/v1", "model": model_id},
        "pricing": {"kind": pricing_kind, "metering": "tokens"},
        "availability": {"status": "available"},
        "quality": {"coding_score": 50.0},
        "policy": {"visibility": "listed", "tags": []},
    }


def _feed_with(*offerings: dict[str, Any], provider_ids: tuple[str, ...] = ()) -> Feed:
    ids = provider_ids or tuple(sorted({o["provider"]["id"] for o in offerings}))
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
                for provider_id in ids
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


def _provider_rule(entitlement: str = "per_model") -> dict[str, Any]:
    return {"mode": "all", "entitlement": entitlement}


# --- state: healthy, dry, degraded, empty ----------------------------------


def test_a_provider_whose_every_admitted_offering_answers_reads_state_healthy():
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov"),
        _offering_raw(id="prov:b", provider_id="prov"),
    )
    policy = _policy(providers={"prov": _provider_rule()})
    report = PlanReport(admitted=("prov:a", "prov:b"))

    view = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)

    assert view.entitlements[0].state == "healthy"


def test_a_provider_whose_every_admitted_offering_is_excluded_reads_state_dry():
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov"),
        _offering_raw(id="prov:b", provider_id="prov"),
    )
    policy = _policy(providers={"prov": _provider_rule()})
    report = PlanReport(excluded=("prov:a", "prov:b"))

    view = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)

    assert view.entitlements[0].state == "dry"


def test_a_partly_excluded_provider_reads_degraded_and_names_the_unavailable_offerings():
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov"),
        _offering_raw(id="prov:b", provider_id="prov"),
    )
    policy = _policy(providers={"prov": _provider_rule()})
    report = PlanReport(admitted=("prov:a",), excluded=("prov:b",))

    view = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)

    entitlement = view.entitlements[0]
    assert entitlement.state == "degraded"
    assert [o.offering_id for o in entitlement.unavailable_offerings] == ["prov:b"]


def test_a_provider_policy_names_but_which_admits_nothing_reads_state_empty():
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(providers={"prov": _provider_rule()})
    report = PlanReport()  # Feed reaches the Offering, but Selection admitted nothing

    view = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)

    assert view.entitlements[0].state == "empty"
    assert view.entitlements[0].in_scope == 0


# --- earliest_refill_at -----------------------------------------------------


def test_earliest_refill_at_is_the_minimum_reset_at_across_unavailable_offerings():
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov"),
        _offering_raw(id="prov:b", provider_id="prov"),
    )
    policy = _policy(providers={"prov": _provider_rule()})
    report = PlanReport(excluded=("prov:a", "prov:b"))
    later = datetime(2026, 7, 28, 0, 0, 0, tzinfo=timezone.utc)
    earlier = datetime(2026, 7, 27, 0, 0, 0, tzinfo=timezone.utc)
    health = {
        "prov:a": OfferingHealth(excluded=True, reason="quota_exhausted", reset_at=later),
        "prov:b": OfferingHealth(excluded=True, reason="quota_exhausted", reset_at=earlier),
    }

    view = derive(feed=feed, policy=policy, health=health, report=report, now=NOW)

    assert view.entitlements[0].earliest_refill_at == earlier


def test_earliest_refill_at_is_none_when_no_unavailable_offering_recorded_a_reset_time():
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(providers={"prov": _provider_rule()})
    report = PlanReport(excluded=("prov:a",))
    health = {"prov:a": OfferingHealth(excluded=True, reason="gateway_error", reset_at=None)}

    view = derive(feed=feed, policy=policy, health=health, report=report, now=NOW)

    assert view.entitlements[0].earliest_refill_at is None


# --- UnavailableOffering carries every field --------------------------------


def test_an_unavailable_offering_carries_its_alias_reason_bucket_and_refills_at():
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(providers={"prov": _provider_rule()})
    reset_at = datetime(2026, 7, 27, 0, 0, 0, tzinfo=timezone.utc)
    report = PlanReport(excluded=("prov:a",), aliases={"prov:a": "claude-prov-a"})
    health = {
        "prov:a": OfferingHealth(
            excluded=True, reason="quota_exhausted", bucket="self_healing", reset_at=reset_at
        )
    }

    view = derive(feed=feed, policy=policy, health=health, report=report, now=NOW)

    offering = view.entitlements[0].unavailable_offerings[0]
    assert offering.alias == "claude-prov-a"
    assert offering.reason == "quota_exhausted"
    assert offering.bucket == "self_healing"
    assert offering.refills_at == reset_at


# --- shared_pool never writes health, only renders a note ------------------


def test_a_shared_pool_providers_rendered_text_carries_the_pool_note():
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov"),
        _offering_raw(id="prov:b", provider_id="prov"),
    )
    policy = _policy(providers={"prov": _provider_rule("shared_pool")})
    report = PlanReport(admitted=("prov:a",), excluded=("prov:b",))
    health = {"prov:b": OfferingHealth(excluded=True, reason="quota_exhausted")}

    view = derive(feed=feed, policy=policy, health=health, report=report, now=NOW)
    text = render_text(view)

    assert "shared pool" in text


def test_a_per_model_providers_rendered_text_carries_no_pool_note():
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov"),
        _offering_raw(id="prov:b", provider_id="prov"),
    )
    policy = _policy(providers={"prov": _provider_rule("per_model")})
    report = PlanReport(admitted=("prov:a",), excluded=("prov:b",))
    health = {"prov:b": OfferingHealth(excluded=True, reason="quota_exhausted")}

    view = derive(feed=feed, policy=policy, health=health, report=report, now=NOW)
    text = render_text(view)

    assert "shared pool" not in text


def test_the_pool_note_never_changes_any_offeringhealth_record():
    # ADR 0004's central rule: a `shared_pool` declaration changes how a
    # report reads. It never writes Health State. Prove it directly: the
    # `health` mapping passed to `derive` must come back unchanged.
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov"),
        _offering_raw(id="prov:b", provider_id="prov"),
    )
    policy = _policy(providers={"prov": _provider_rule("shared_pool")})
    report = PlanReport(admitted=("prov:a",), excluded=("prov:b",))
    health = {"prov:b": OfferingHealth(excluded=True, reason="quota_exhausted")}
    health_before = dict(health)

    view = derive(feed=feed, policy=policy, health=health, report=report, now=NOW)
    render_text(view)  # renders the pool note, must not mutate `health`

    assert health == health_before


# --- mixed cost bases --------------------------------------------------------


def test_a_provider_whose_offerings_mix_pricing_kinds_reports_every_cost_basis():
    feed = _feed_with(
        _offering_raw(id="prov:a", provider_id="prov", pricing_kind="free"),
        _offering_raw(id="prov:b", provider_id="prov", pricing_kind="paid"),
    )
    policy = _policy(providers={"prov": _provider_rule()})
    report = PlanReport(admitted=("prov:a", "prov:b"))

    view = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)

    entitlement = view.entitlements[0]
    assert set(entitlement.cost_bases) == {FREE, METERED}
    assert entitlement.cost_basis is None


def test_cost_basis_for_pricing_kind_maps_every_kind():
    assert cost_basis_for_pricing_kind("free") == FREE
    assert cost_basis_for_pricing_kind("free_tier") == FREE
    assert cost_basis_for_pricing_kind("subscription_included") == FLAT_RATE
    assert cost_basis_for_pricing_kind("paid") == METERED
    assert cost_basis_for_pricing_kind("unknown") == UNKNOWN_BASIS
    assert cost_basis_for_pricing_kind("a-kind-nobody-has-seen") == UNKNOWN_BASIS


# --- rendering on an empty view ---------------------------------------------


def test_render_text_and_render_markdown_run_on_a_view_with_zero_providers():
    view = EntitlementView()

    text = render_text(view)
    markdown = render_markdown(view)

    assert "no Entitlement" in text or "no provider" in text
    assert "no provider" in markdown or markdown.startswith("# Entitlements")


def test_as_dict_carries_schema_version():
    view = EntitlementView()

    document = view.as_dict()

    assert document["schema_version"] == "1"
