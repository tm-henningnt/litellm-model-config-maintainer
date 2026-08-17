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

from litellm_maintainer.codexbar import (
    CodexbarExtraWindow,
    CodexbarIdentity,
    CodexbarReading,
    CodexbarWindow,
)
from litellm_maintainer.entitlements import (
    FLAT_RATE,
    FREE,
    METERED,
    UNKNOWN_BASIS,
    AllowanceHeadroom,
    EntitlementView,
    cost_basis_for_pricing_kind,
    derive,
    render_markdown,
    render_text,
)
from litellm_maintainer.feed import Feed, parse_feed
from litellm_maintainer.headroom import HeadroomRecord, HeadroomState
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


# --- Tier (ticket 12) --------------------------------------------------------


def test_an_allowance_with_no_policy_entry_publishes_no_tier():
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(providers={"prov": _provider_rule()})
    report = PlanReport(admitted=("prov:a",))

    view = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)

    assert view.entitlements[0].tier is None
    assert view.entitlements[0].as_dict()["tier"] is None


def test_an_allowance_with_a_stated_tier_publishes_it_verbatim():
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()},
        allowances={"provider:prov": {"tier": "claude-max-5x"}},
    )
    report = PlanReport(admitted=("prov:a",))

    view = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)

    entitlement = view.entitlements[0]
    assert entitlement.tier == "claude-max-5x"
    assert entitlement.as_dict()["tier"] == "claude-max-5x"


def test_a_declared_allowances_tier_is_published_too():
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

    view = derive(feed=_feed_with(), policy=policy, health={}, report=report, now=NOW)

    declared_entitlement = next(
        e for e in view.entitlements if e.allowance_id == "pool:claude-subscription"
    )
    assert declared_entitlement.tier == "claude-max-5x"


def test_guidance_and_entitlements_agree_on_the_same_allowances_tier():
    from litellm_maintainer.guidance import derive as guidance_derive

    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()},
        allowances={"provider:prov": {"tier": "claude-max-5x"}},
    )
    report = PlanReport(admitted=("prov:a",))

    entitlement_view = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)
    guidance = guidance_derive(feed=feed, policy=policy, health={}, report=report, now=NOW)

    entitlement_tier = entitlement_view.entitlements[0].tier
    route_tier = guidance.rows[0].routes[0].tier
    assert entitlement_tier == route_tier == "claude-max-5x"


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

    # Raised to "2" on 2026-07-28: `entitlements` gained one entry per
    # Declared Allowance, so iterating the list yields Offerings it never
    # yielded before and `declared` now duplicates them. Pinned to a literal
    # rather than to the constant, so a bump has to be deliberate.
    assert document["schema_version"] == "2"


# --- Headroom (ticket 04) ----------------------------------------------------
#
# `derive`'s `headroom_state` is optional and defaults to `None`, so every
# test above this line still describes an Entitlement view with no Headroom
# read at all: `headroom` is `None` for every entry, exactly as before this
# ticket. These tests cover what changes once a `HeadroomState` is supplied.

# `derive` now publishes a stored record only when Policy still declares its
# Allowance AND the record's own `source` still matches (defect 2). Every
# test below that expects a populated `headroom` passes this alongside
# `_record`'s own default `source`, `f"codexbar:{allowance_id}/"`.
MAPPED_HEADROOM_SOURCES = {"provider:prov": "codexbar:provider:prov/"}


def _record(
    allowance_id: str,
    *,
    primary: CodexbarWindow | None = None,
    secondary: CodexbarWindow | None = None,
    tertiary: CodexbarWindow | None = None,
    extra_windows: tuple[CodexbarExtraWindow, ...] = (),
    updated_at: str | None = "2026-07-26T11:00:00Z",
    read_at: str = "2026-07-26T11:59:00Z",
) -> HeadroomRecord:
    return HeadroomRecord(
        allowance_id=allowance_id,
        source=f"codexbar:{allowance_id}/",
        reading=CodexbarReading(
            provider="prov",
            identity=CodexbarIdentity(provider_id="prov", account_email="operator@example.com"),
            primary=primary,
            secondary=secondary,
            tertiary=tertiary,
            extra_windows=extra_windows,
            updated_at=updated_at,
            error=None,
        ),
        read_at=read_at,
    )


def test_an_allowance_with_no_headroom_record_publishes_no_headroom():
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(providers={"prov": _provider_rule()})
    report = PlanReport(admitted=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=HeadroomState()
    )

    assert view.entitlements[0].headroom is None
    assert view.entitlements[0].as_dict()["headroom"] is None


def test_a_record_with_no_windows_at_all_publishes_no_headroom():
    # Measured 2026-07-28: OpenRouter and DeepSeek both answer with primary,
    # secondary and tertiary all null. This must never read as 0% used.
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(providers={"prov": _provider_rule()})
    report = PlanReport(admitted=("prov:a",))
    state = HeadroomState(records={"provider:prov": _record("provider:prov")})

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    assert view.entitlements[0].headroom is None


def test_a_record_where_every_window_is_void_publishes_no_headroom():
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(
                    used_percent=100, window_minutes=300, resets_at="2026-07-26T11:30:00Z"
                ),
                updated_at="2026-07-26T11:00:00Z",
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(providers={"prov": _provider_rule()})
    report = PlanReport(admitted=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    assert view.entitlements[0].headroom is None


# --- Ticket 09: a source whose every slot is a declared Sub-allowance -------
#
# Gemini's shape: `primary`, `secondary` and `tertiary` hold one quota per
# MODEL, not a nested time window. Naming a slot in
# `headroom.sources.<id>.windows` turns it from a parent window into a
# Sub-allowance, so it leaves this Allowance's own binding computation.


def test_an_allowance_whose_every_slot_is_named_publishes_no_headroom_of_its_own():
    # Where every slot is a declared Sub-allowance, nothing is left to cap
    # the Allowance as a whole -- correctly. Each model's own figure lives
    # on its Route instead (`guidance`, not `entitlements`).
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=100, window_minutes=1440, resets_at=None),
                secondary=CodexbarWindow(used_percent=0, window_minutes=1440, resets_at=None),
                tertiary=CodexbarWindow(used_percent=0, window_minutes=1440, resets_at=None),
                updated_at="2026-07-26T11:00:00Z",
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()},
        headroom={
            "sources": {
                "provider:prov": {
                    "source": "codexbar:provider:prov/",
                    "windows": {
                        "primary": "prov-a",
                        "secondary": "prov-b",
                        "tertiary": "prov-c",
                    },
                }
            }
        },
    )
    report = PlanReport(admitted=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    # The object is published; only `binding` is absent. Returning `None`
    # here reported a measured Allowance as unmeasured, and the per-model
    # figures then reached `guidance` routes and nowhere else (measured
    # 2026-07-29 on the operator's Gemini free plan).
    headroom = view.entitlements[0].headroom
    assert headroom is not None
    assert headroom.binding is None
    assert headroom.primary.used_percent == 100
    assert headroom.secondary.used_percent == 0
    assert headroom.tertiary.used_percent == 0


def test_an_allowance_whose_every_window_is_void_publishes_no_headroom_at_all():
    # The other reason `binding` cannot be computed, and it means the
    # opposite: nothing was measured, so the whole object stays absent.
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(
                    used_percent=100, window_minutes=1440, resets_at="2026-07-26T11:30:00Z"
                ),
                updated_at="2026-07-26T11:00:00Z",
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()},
        headroom={
            "sources": {
                "provider:prov": {
                    "source": "codexbar:provider:prov/",
                    "windows": {"primary": "prov-a"},
                }
            }
        },
    )
    report = PlanReport(admitted=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    assert view.entitlements[0].headroom is None


def test_an_allowance_with_one_named_slot_still_binds_on_the_rest():
    # 'primary' (100%) is named and excluded from the parent computation.
    # The worst of the two REMAINING slots, 'secondary' at 20%, binds --
    # not the 100% a reader that ignored the named slot would report, and
    # not absent either: an unnamed slot still binds the Allowance.
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=100, window_minutes=1440, resets_at=None),
                secondary=CodexbarWindow(used_percent=20, window_minutes=1440, resets_at=None),
                tertiary=CodexbarWindow(used_percent=0, window_minutes=1440, resets_at=None),
                updated_at="2026-07-26T11:00:00Z",
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()},
        headroom={
            "sources": {
                "provider:prov": {
                    "source": "codexbar:provider:prov/",
                    "windows": {"primary": "prov-a"},
                }
            }
        },
    )
    report = PlanReport(admitted=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    assert view.entitlements[0].headroom is not None
    assert view.entitlements[0].headroom.binding.used_percent == 20


def test_a_live_binding_window_is_published_on_its_allowance():
    # ClinePass's own shape (measured 2026-07-28): two 0% windows and one
    # fully-drawn one. The Allowance must bind on the worst one, 100%, and
    # never on a named window that happens to read free.
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=0, window_minutes=300, resets_at=None),
                secondary=CodexbarWindow(used_percent=0, window_minutes=10080, resets_at=None),
                tertiary=CodexbarWindow(
                    used_percent=100, window_minutes=43200, resets_at="2026-08-03T10:19:34Z"
                ),
                updated_at="2026-07-26T11:00:00Z",
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()}, headroom={"sources": MAPPED_HEADROOM_SOURCES}
    )
    report = PlanReport(admitted=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    headroom = view.entitlements[0].headroom
    assert headroom is not None
    assert headroom.binding.used_percent == 100
    assert headroom.binding.resets_at == "2026-08-03T10:19:34Z"
    # The void-free tertiary window is live in the full window set too.
    assert headroom.tertiary is not None
    assert headroom.tertiary.used_percent == 100
    assert headroom.tertiary.void is False


def test_a_void_window_reports_no_used_share_in_the_full_window_set():
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(
                    used_percent=100, window_minutes=300, resets_at="2026-07-26T11:30:00Z"
                ),
                secondary=CodexbarWindow(used_percent=10, window_minutes=10080, resets_at=None),
                updated_at="2026-07-26T11:00:00Z",
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()}, headroom={"sources": MAPPED_HEADROOM_SOURCES}
    )
    report = PlanReport(admitted=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    headroom = view.entitlements[0].headroom
    assert headroom is not None
    assert headroom.primary is not None
    assert headroom.primary.void is True
    assert headroom.primary.used_percent is None
    # The Binding Window is the live one: secondary at 10%.
    assert headroom.binding.used_percent == 10


def test_age_is_computed_from_codexbars_own_timestamp_never_ours():
    # `read_at` (ours) is far newer than `updated_at` (codexbar's). Age
    # must track codexbar's clock, not ours, per the headroom spec.
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=10, window_minutes=300, resets_at=None),
                updated_at="2026-07-26T09:00:00Z",
                read_at="2026-07-26T11:59:00Z",
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()}, headroom={"sources": MAPPED_HEADROOM_SOURCES}
    )
    report = PlanReport(admitted=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    headroom = view.entitlements[0].headroom
    assert headroom is not None
    # NOW is 2026-07-26T12:00:00Z; updated_at is 09:00:00Z: 3 hours.
    assert headroom.age_seconds == 3 * 3600.0
    assert headroom.updated_at == "2026-07-26T09:00:00Z"
    assert headroom.read_at == "2026-07-26T11:59:00Z"


def test_render_text_shows_the_binding_figure():
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=82, window_minutes=10080, resets_at=None),
                updated_at="2026-07-26T09:00:00Z",
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()}, headroom={"sources": MAPPED_HEADROOM_SOURCES}
    )
    report = PlanReport(admitted=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    text = render_text(view)

    assert "headroom: 82%" in text


def test_render_markdown_shows_the_binding_figure():
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=82, window_minutes=10080, resets_at=None),
                updated_at="2026-07-26T09:00:00Z",
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()}, headroom={"sources": MAPPED_HEADROOM_SOURCES}
    )
    report = PlanReport(admitted=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    markdown = render_markdown(view)

    assert "82%" in markdown


def test_render_text_omits_the_headroom_line_when_there_is_none():
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(providers={"prov": _provider_rule()})
    report = PlanReport(admitted=("prov:a",))

    view = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)

    assert "headroom:" not in render_text(view)


# --- Defect 2: a stored Reading must not survive its source leaving Policy -


def test_a_record_for_an_allowance_no_longer_declared_publishes_no_headroom():
    """The operator's real case: Gemini was mapped 2026-07-28 and unmapped
    2026-07-29 because its figure's meaning was unknown. `refresh_headroom`
    prunes on its own schedule; `derive` must not trust a stale record in
    the meantime, whatever `refresh_headroom` has or has not gotten to."""
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=100, window_minutes=1440, resets_at=None),
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    # No `headroom` block at all: Policy no longer declares this Allowance.
    policy = _policy(providers={"prov": _provider_rule()})
    report = PlanReport(admitted=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    assert view.entitlements[0].headroom is None


def test_a_remapped_source_publishes_no_headroom_until_a_fresh_reading_matches():
    """The stored record's `source` still names the OLD codexbar identity.
    Policy now points this Allowance at a different one -- a remap, not a
    removal. The old Reading must not keep publishing under the new name."""
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=100, window_minutes=1440, resets_at=None),
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()},
        headroom={
            "sources": {
                "provider:prov": "codexbar:provider:prov/new-account@example.com"
            }
        },
    )
    report = PlanReport(admitted=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    assert view.entitlements[0].headroom is None


def test_a_headroom_with_no_binding_renders_without_crashing():
    # Both renderers read `binding` directly. A `None` there is a measured
    # Allowance with no whole-Allowance window, so neither may print "—"
    # (which this output already uses for unmeasured) and neither may raise.
    from litellm_maintainer.entitlements import _headroom_cell, _headroom_line

    headroom = AllowanceHeadroom(
        allowance_id="provider:prov",
        source="codexbar:prov/",
        provider="prov",
        account_email=None,
        updated_at="2026-07-26T11:00:00Z",
        read_at="2026-07-26T11:00:00Z",
        age_seconds=60.0,
        binding=None,
        primary=None,
        secondary=None,
        tertiary=None,
    )

    line = _headroom_line(headroom)
    cell = _headroom_cell(headroom)

    assert line is not None
    assert "no single window caps this Allowance" in line
    assert cell != "—"
    assert "0%" not in cell


def test_a_window_stating_no_reset_publishes_null_not_the_placeholder():
    # Measured 2026-07-29: Gemini's Pro slot carries
    # `resetsAt: "1970-01-01T00:00:00Z"` beside "Resets soon". A reset at
    # or before the Reading's own timestamp states no reset, so publishing
    # it raw leaks a placeholder date to every caller.
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(
                    used_percent=100, window_minutes=1440, resets_at="1970-01-01T00:00:00Z"
                ),
                secondary=CodexbarWindow(used_percent=0, window_minutes=1440, resets_at=None),
                updated_at="2026-07-26T11:00:00Z",
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()},
        headroom={"sources": {"provider:prov": "codexbar:provider:prov/"}},
    )
    report = PlanReport(admitted=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    assert view.entitlements[0].headroom.primary.resets_at is None


# --- Reported 2026-07-29 by an agent consumer ----------------------------


def test_the_binding_figure_reads_flat_the_way_a_guidance_route_reads_it():
    # One field name carried two shapes: `.headroom.used_percent` answered
    # on a guidance route and returned `null` on every entitlement, mapped
    # or not, so a consumer could not tell that `null` from an unmapped
    # allowance.
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=42, window_minutes=300, resets_at=None),
                updated_at="2026-07-26T11:00:00Z",
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()},
        headroom={"sources": {"provider:prov": "codexbar:provider:prov/"}},
    )
    report = PlanReport(admitted=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )
    published = view.entitlements[0].headroom.as_dict()

    assert published["used_percent"] == 42
    assert published["window_minutes"] == 300
    assert published["binding"]["used_percent"] == 42


def test_a_window_states_which_admitted_offerings_draw_on_it():
    # A window can be permanently full and govern nothing spendable. The
    # measured case: a free Gemini plan reads 100% on its Pro slot, whose
    # Offerings are all Withheld, while admitted Flash routes read 0%.
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=100, window_minutes=1440, resets_at=None),
                secondary=CodexbarWindow(used_percent=0, window_minutes=1440, resets_at=None),
                updated_at="2026-07-26T11:00:00Z",
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()},
        headroom={
            "sources": {
                "provider:prov": {
                    "source": "codexbar:provider:prov/",
                    "windows": {"primary": "prov-pro", "secondary": "prov-flash"},
                    "members": {"prov-pro": [], "prov-flash": ["prov:a"]},
                }
            }
        },
    )
    report = PlanReport(admitted=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )
    headroom = view.entitlements[0].headroom

    assert headroom.primary.used_percent == 100
    assert headroom.primary.sub_allowance_id == "prov-pro"
    assert headroom.primary.admitted_members == ()
    assert headroom.secondary.admitted_members == ("prov:a",)


def test_an_excluded_offering_still_counts_as_an_admitted_member():
    # Reading `report.admitted` alone would empty a window's member list
    # the moment its one model failed a Probe, which reads as "this window
    # governs nothing" when Policy still admits it.
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=10, window_minutes=1440, resets_at=None),
                secondary=CodexbarWindow(used_percent=0, window_minutes=1440, resets_at=None),
                updated_at="2026-07-26T11:00:00Z",
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()},
        headroom={
            "sources": {
                "provider:prov": {
                    "source": "codexbar:provider:prov/",
                    "windows": {"primary": "prov-pro", "secondary": "prov-flash"},
                    "members": {"prov-pro": ["prov:a"], "prov-flash": []},
                }
            }
        },
    )
    report = PlanReport(admitted=(), excluded=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    assert view.entitlements[0].headroom.primary.admitted_members == ("prov:a",)


def test_a_parent_window_declares_no_membership_rather_than_an_empty_one():
    # `null` and `[]` are different claims. A parent window on a plain-string
    # mapping governs every Offering on the Allowance and declares no
    # members, so `[]` there would mark every ordinary Allowance as idle
    # capacity. Caught by the very query written to find the Gemini case.
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=90, window_minutes=300, resets_at=None),
                updated_at="2026-07-26T11:00:00Z",
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()},
        headroom={"sources": {"provider:prov": "codexbar:provider:prov/"}},
    )
    report = PlanReport(admitted=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )

    assert view.entitlements[0].headroom.primary.admitted_members is None


def test_an_extra_window_reports_an_empty_member_list_when_nothing_admitted_draws_on_it():
    # Reported 2026-07-29: the published detector query read the three slots
    # only, so a pessimistic window in `extra_windows` was invisible to it.
    # A named Sub-allowance lives there, and it carries both fields, so the
    # state the query must catch is reachable.
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=10, window_minutes=300, resets_at=None),
                extra_windows=(
                    CodexbarExtraWindow(
                        id="prov-scoped",
                        title="Scoped only",
                        window=CodexbarWindow(
                            used_percent=100, window_minutes=10080, resets_at=None
                        ),
                    ),
                ),
                updated_at="2026-07-26T11:00:00Z",
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()},
        headroom={
            "sources": {
                "provider:prov": {
                    "source": "codexbar:provider:prov/",
                    # Policy declares the membership; nothing admitted matches it.
                    "members": {"prov-scoped": ["prov:gone"]},
                }
            }
        },
    )
    report = PlanReport(admitted=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )
    extra = view.entitlements[0].headroom.extra_windows[0]

    assert extra.window.used_percent == 100
    assert extra.window.sub_allowance_id == "prov-scoped"
    assert extra.window.admitted_members == ()


def test_an_allowance_admitting_nothing_still_publishes_its_reading():
    # `provider:cline-pass` reads 100% with every Offering Withheld. The
    # Reading is real and stays published: `headroom: null` means UNMEASURED,
    # and hiding a true figure to spare a reader one join would make that
    # word mean two things. `in_scope` is the cross-check.
    state = HeadroomState(
        records={
            "provider:prov": _record(
                "provider:prov",
                primary=CodexbarWindow(used_percent=100, window_minutes=43200, resets_at=None),
                updated_at="2026-07-26T11:00:00Z",
            )
        }
    )
    feed = _feed_with(_offering_raw(id="prov:a", provider_id="prov"))
    policy = _policy(
        providers={"prov": _provider_rule()},
        withheld={"prov:a": "subscription ending, renewal unconfirmed"},
        headroom={"sources": {"provider:prov": "codexbar:provider:prov/"}},
    )
    report = PlanReport(withheld=("prov:a",))

    view = derive(
        feed=feed, policy=policy, health={}, report=report, now=NOW, headroom_state=state
    )
    entitlement = view.entitlements[0]

    assert entitlement.in_scope == 0
    assert entitlement.headroom is not None
    assert entitlement.headroom.as_dict()["used_percent"] == 100
