"""A Declared Offering appears in guidance, unscored but present.

The Feed does not publish a Declared Offering, so it carries no quality
score and no Canonical Model. An earlier version of `guidance` therefore
dropped every one of them. On the operator's real Policy that hid 22
Aliases, including the strongest models the proxy serves, because a
direct vendor entry is exactly the kind of Offering the Feed does not
cover.

The rule these tests pin: an admitted Offering is always reachable in the
answer. Having no score decides where it ranks, never whether it appears.
"""

from __future__ import annotations

from datetime import datetime, timezone

from litellm_maintainer import guidance
from litellm_maintainer.feed import parse_feed
from litellm_maintainer.plan import PlanReport
from litellm_maintainer.policy import parse_policy
from litellm_maintainer.reduce import OfferingHealth

NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def _policy(declared):
    return parse_policy(
        {
            "providers": {"openrouter": {"mode": "all"}},
            "quality": {"minimum_coding_score": 10},
            "approved_candidates": [],
            "naming": {
                "alias_prefix": "claude-",
                "provider_labels": {"openrouter": "or"},
                "alias_overrides": {},
            },
            "withheld": {},
            "declared": declared,
            "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
            "schedule": {
                "enabled": True,
                "interval_minutes": 60,
                "require_proxy": False,
                "maximum_staleness_hours": 24,
            },
            "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
        }
    )


def _feed():
    return parse_feed(
        {
            "schema_version": "1.0.0",
            "feed": {"generated_at": "2026-07-26T11:00:00Z"},
            "providers": [{"id": "openrouter", "name": "OpenRouter"}],
            "models": [
                {
                    "id": "openrouter:vendor/scored",
                    "provider": {"id": "openrouter"},
                    "provider_model_id": "vendor/scored",
                    "canonical_model": {"id": "vendor/scored"},
                    "capabilities": ["chat", "coding", "tool_use"],
                    "pricing": {"kind": "free"},
                    "availability": {"status": "available"},
                    "quality": {"coding_score": 60.0},
                    "policy": {"visibility": "listed"},
                    "endpoint": {},
                }
            ],
        }
    )


DIRECT = {"alias": "claude-direct", "litellm_params": {"model": "anthropic/direct"}}
PASSTHROUGH = {
    "alias": "claude-caller-billed",
    "passthrough_auth": True,
    "litellm_params": {"model": "chatgpt/caller-billed"},
}


def _derive(*, declared, admitted, excluded=(), health=None, **kwargs):
    return guidance.derive(
        feed=_feed(),
        policy=_policy(declared),
        health=health or {},
        report=PlanReport(
            admitted=admitted,
            excluded=excluded,
            aliases={"openrouter:vendor/scored": "claude-or-scored"},
        ),
        now=NOW,
        **kwargs,
    )


def test_a_declared_offering_gets_its_own_row():
    answer = _derive(
        declared=[DIRECT], admitted=("openrouter:vendor/scored", "claude-direct")
    )

    ids = [row.canonical_model_id for row in answer.rows]

    assert "claude-direct" in ids
    assert "vendor/scored" in ids


def test_a_declared_row_carries_one_route_naming_its_alias():
    answer = _derive(declared=[DIRECT], admitted=("claude-direct",))
    row = next(r for r in answer.rows if r.canonical_model_id == "claude-direct")

    assert len(row.routes) == 1
    assert row.routes[0].alias == "claude-direct"
    assert row.routes[0].provider_id == guidance.DECLARED_PROVIDER
    assert row.callable_now is True


def test_a_declared_row_has_no_score_on_any_axis():
    """The Feed does not cover it, so we state no score rather than guess one."""
    answer = _derive(declared=[DIRECT], admitted=("claude-direct",))
    row = next(r for r in answer.rows if r.canonical_model_id == "claude-direct")

    assert row.score is None
    assert set(row.scores) == set(guidance.AXES)
    assert all(value is None for value in row.scores.values())


def test_a_declared_row_ranks_below_a_scored_one():
    answer = _derive(
        declared=[DIRECT], admitted=("openrouter:vendor/scored", "claude-direct")
    )

    ids = [row.canonical_model_id for row in answer.rows]

    assert ids.index("vendor/scored") < ids.index("claude-direct")


def test_a_declared_row_explains_that_the_feed_does_not_score_it():
    """A caller must not read 'unscored' as 'weak'."""
    answer = _derive(declared=[DIRECT], admitted=("claude-direct",))
    row = next(r for r in answer.rows if r.canonical_model_id == "claude-direct")

    assert "declared by the operator" in row.why
    assert "rank it yourself" in row.why


def test_passthrough_auth_reads_as_billed_to_the_caller():
    answer = _derive(declared=[PASSTHROUGH], admitted=("claude-caller-billed",))
    row = next(
        r for r in answer.rows if r.canonical_model_id == "claude-caller-billed"
    )

    assert row.routes[0].cost_basis == guidance.PASSTHROUGH
    assert "calling client" in row.why


def test_a_declared_offering_with_proxy_credentials_reads_as_unpriced():
    """No `passthrough_auth` means the proxy pays, and the Feed states no rate."""
    answer = _derive(declared=[DIRECT], admitted=("claude-direct",))
    row = next(r for r in answer.rows if r.canonical_model_id == "claude-direct")

    assert row.routes[0].cost_basis == guidance.UNKNOWN_BASIS


def test_an_excluded_declared_offering_is_listed_with_its_reason():
    answer = _derive(
        declared=[DIRECT],
        admitted=("openrouter:vendor/scored",),
        excluded=("claude-direct",),
        health={
            "claude-direct": OfferingHealth(
                excluded=True, reason="needs_operator", bucket="needs_operator"
            )
        },
    )
    row = next(r for r in answer.rows if r.canonical_model_id == "claude-direct")

    assert row.callable_now is False
    assert row.routes[0].available is False
    assert row.routes[0].reason == "needs_operator"


def test_a_declared_offering_neither_admitted_nor_excluded_is_absent():
    """A Policy line for an Offering this run did not offer states nothing useful."""
    answer = _derive(declared=[DIRECT], admitted=("openrouter:vendor/scored",))

    assert all(row.canonical_model_id != "claude-direct" for row in answer.rows)


# --- The same Offerings in the Entitlement view --------------------------


def _entitlement_view(*, declared, admitted, excluded=(), health=None):
    from litellm_maintainer import entitlements

    return entitlements.derive(
        feed=_feed(),
        policy=_policy(declared),
        health=health or {},
        report=PlanReport(admitted=admitted, excluded=excluded),
        now=NOW,
    )


def test_a_declared_offering_now_gets_its_own_entitlement_entry():
    """Reversed on 2026-07-28, deliberately.

    This test used to assert `[e.provider_id for e in view.entitlements] ==
    ["openrouter"]` — that a Declared Offering reached no entry at all. The
    reasoning was that an Entitlement is a relationship with a PROVIDER and
    a Declared Offering has none. That held, and the result was still wrong:
    a whole private host reported one aggregate count with no `state`, no
    `reason` and no refill time, so the only way to find its ceiling was to
    hit it.

    An Entitlement is keyed by Allowance now, and a Declared Offering has
    one. See ADR 0012.
    """
    view = _entitlement_view(declared=[DIRECT, PASSTHROUGH], admitted=("claude-direct",))

    entry = next(e for e in view.entitlements if e.provider_id == "declared")
    assert entry.allowance_id == "alias:claude-direct"
    assert entry.answering == 1
    assert entry.state == "healthy"
    # The Feed provider keeps its position, so a consumer indexing the list
    # reads what it always read.
    assert view.entitlements[0].provider_id == "openrouter"


def test_the_declared_aggregate_survives_beside_the_new_entries():
    """`declared` is kept so an existing consumer does not break. It reports
    the same Offerings, so summing both would double-count."""
    view = _entitlement_view(declared=[DIRECT, PASSTHROUGH], admitted=("claude-direct",))

    assert view.declared_answering == 1
    declared_entries = [e for e in view.entitlements if e.provider_id == "declared"]
    assert sum(e.answering for e in declared_entries) == view.declared_answering


def test_an_excluded_declared_offering_is_reported_with_its_reason():
    """Otherwise 'why is this Alias missing' is answered nowhere."""
    view = _entitlement_view(
        declared=[DIRECT],
        admitted=(),
        excluded=("claude-direct",),
        health={
            "claude-direct": OfferingHealth(
                excluded=True, reason="authentication_failed", bucket="needs_operator"
            )
        },
    )

    assert view.declared_in_scope == 1
    assert view.declared_answering == 0
    assert view.declared_unavailable[0].alias == "claude-direct"
    assert view.declared_unavailable[0].reason == "authentication_failed"


def test_the_declared_block_reaches_every_rendering():
    from litellm_maintainer import entitlements

    view = _entitlement_view(declared=[DIRECT], admitted=("claude-direct",))

    assert "declared" in entitlements.render_text(view)
    assert "Declared Offerings" in entitlements.render_markdown(view)
    assert view.as_dict()["declared"]["answering"] == 1


def test_a_policy_with_no_declared_offerings_reports_none():
    view = _entitlement_view(declared=[], admitted=("openrouter:vendor/scored",))

    assert view.declared_in_scope == 0
    assert view.declared_unavailable == ()


def test_health_is_read_by_alias_for_a_declared_offering():
    """A Declared Offering's Health Key is its Alias, per CONTEXT.md."""
    answer = _derive(
        declared=[DIRECT],
        admitted=("claude-direct",),
        health={"claude-direct": OfferingHealth(last_success_at=NOW)},
    )
    row = next(r for r in answer.rows if r.canonical_model_id == "claude-direct")

    assert row.routes[0].last_success_at == NOW
