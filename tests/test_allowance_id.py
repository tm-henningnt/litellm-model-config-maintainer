"""An Allowance is published, never guessed from a name.

`guidance` said which model to call and never whose allowance paid for it.
Every Declared Route reported `provider_id: "declared"` and nothing else, so
a client could not tell one subscription seat from another, and could not
refuse a fair-use host without refusing every Declared Route with it.
Measured 2026-07-28 downstream: excluding the whole `declared` bucket took an
ordinary Role from 58 Routes to 42.

Two rules these tests pin:

1. **The credential names the Allowance.** Not `group`, which is a heading
   the Generated Config prints and which `policy.py` says "names nothing the
   code acts on". Not the Alias, which does encode the seat and is exactly
   the guess the field exists to prevent.
2. **`fair_use` is not a Cost Basis.** A Cost Basis answers who bills, and a
   fair-use host bills flat rate. Load tolerance is a second question, so it
   is a second field, and it changes no ranking.

See ADR 0012.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from litellm_maintainer import entitlements, guidance
from litellm_maintainer.feed import parse_feed
from litellm_maintainer.plan import PlanReport
from litellm_maintainer.policy import load_policy, parse_policy
from litellm_maintainer.reduce import OfferingHealth

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)
# Synthetic and committed. Never the operator's own Policy.
PINNED_POLICY_PATH = Path(__file__).parent / "fixtures" / "policy-pinned.yaml"

FEED_RAW = {
    "schema_version": "1.0.0",
    "feed": {"generated_at": "2026-07-28T11:00:00Z"},
    "providers": [{"id": "openrouter", "name": "OpenRouter"}],
    "models": [
        {
            "id": "openrouter:vendor/scored",
            "provider": {"id": "openrouter"},
            "provider_model_id": "vendor/scored",
            "canonical_model": {"id": "vendor/scored"},
            "capabilities": ["chat", "coding", "tool_use"],
            "limits": {"context_tokens": 200000},
            "pricing": {"kind": "free", "metering": "tokens"},
            "availability": {"status": "available"},
            "quality": {"coding_score": 60.0},
            "policy": {"visibility": "listed"},
            "endpoint": {},
        }
    ],
}

# Two seats behind one provider prefix, separated only by their credential.
# This is the case the whole design turns on.
SEAT1 = {
    "alias": "claude-seat1-model",
    "group": "Seat 1 — worker on 127.0.0.1:4011",
    "litellm_params": {
        "model": "openai/model",
        "api_base": "http://127.0.0.1:4011/v1",
        "api_key": "os.environ/SEAT1_WORKER_KEY",
    },
}
SEAT2 = {
    "alias": "claude-seat2-model",
    "group": "Seat 2 — worker on 127.0.0.1:4012",
    "litellm_params": {
        "model": "openai/model",
        "api_base": "http://127.0.0.1:4012/v1",
        "api_key": "os.environ/SEAT2_WORKER_KEY",
    },
}
FAIR_USE_HOST = {
    "alias": "claude-host-model",
    "cost_basis": "flat_rate",
    "fair_use": True,
    "litellm_params": {
        "model": "openai/host-model",
        "api_base": "https://host.invalid/v1",
        "api_key": "os.environ/HOST_API_KEY",
    },
}


def _policy(declared, providers=None):
    return parse_policy(
        {
            "providers": providers if providers is not None else {"openrouter": {"mode": "all"}},
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


def _guidance(*, declared, admitted, providers=None, excluded=()):
    return guidance.derive(
        feed=parse_feed(FEED_RAW),
        policy=_policy(declared, providers),
        health={},
        report=PlanReport(
            admitted=admitted,
            excluded=excluded,
            aliases={"openrouter:vendor/scored": "claude-or-scored"},
        ),
        now=NOW,
    )


def _view(*, declared, admitted, excluded=(), health=None, providers=None):
    return entitlements.derive(
        feed=parse_feed(FEED_RAW),
        policy=_policy(declared, providers),
        health=health or {},
        report=PlanReport(admitted=admitted, excluded=excluded),
        now=NOW,
    )


def _route(answer, alias):
    for row in answer.rows:
        for route in row.routes:
            if route.alias == alias:
                return route
    raise AssertionError(f"no Route for {alias!r}")


# --- The credential names the Allowance ---------------------------------


def test_two_seats_get_different_allowance_ids_from_their_credentials_alone():
    """The point of the whole field. Nothing in Policy groups these two, and
    nothing needs to: the credential is what gets billed (ADR 0009)."""
    answer = _guidance(
        declared=[SEAT1, SEAT2],
        admitted=("claude-seat1-model", "claude-seat2-model"),
    )

    first = _route(answer, "claude-seat1-model").allowance_id
    second = _route(answer, "claude-seat2-model").allowance_id

    assert first == "credential:SEAT1_WORKER_KEY"
    assert second == "credential:SEAT2_WORKER_KEY"
    assert first != second


def test_offerings_on_one_credential_share_one_allowance_id():
    second_model = dict(SEAT1, alias="claude-seat1-other")
    answer = _guidance(
        declared=[SEAT1, second_model],
        admitted=("claude-seat1-model", "claude-seat1-other"),
    )

    assert (
        _route(answer, "claude-seat1-model").allowance_id
        == _route(answer, "claude-seat1-other").allowance_id
    )


def test_a_named_entitlement_pool_overrides_the_credential():
    """The escape hatch for the two cases the credential rule gets wrong:
    two keys billed to one account, and one key spanning two plans."""
    pooled = dict(SEAT1, entitlement_pool="one-subscription")
    answer = _guidance(declared=[pooled], admitted=("claude-seat1-model",))

    assert _route(answer, "claude-seat1-model").allowance_id == "pool:one-subscription"


def test_an_offering_with_neither_pool_nor_credential_is_its_own_allowance():
    """`declared_pool_id` answers `None` here, because such an Offering
    propagates a Probe to nobody. This answers who is billed, and the answer
    is itself. A `null` would read as one shared allowance for every
    unpooled Offering, which is the opposite of the truth."""
    passthrough = {
        "alias": "claude-caller-billed",
        "passthrough_auth": True,
        "litellm_params": {"model": "anthropic/caller-billed"},
    }
    answer = _guidance(declared=[passthrough], admitted=("claude-caller-billed",))

    assert _route(answer, "claude-caller-billed").allowance_id == "alias:claude-caller-billed"


def test_a_feed_route_reports_its_provider_as_the_allowance():
    """The Feed states one credential hint per provider, so a provider IS an
    allowance for a Discovered Offering. One field answers everywhere."""
    answer = _guidance(declared=[], admitted=("openrouter:vendor/scored",))

    assert _route(answer, "claude-or-scored").allowance_id == "provider:openrouter"


def test_the_allowance_id_names_a_variable_and_never_a_value():
    """`os.environ/` is stripped. A variable name is not a secret; a value
    is, and this field must never be able to carry one."""
    answer = _guidance(declared=[SEAT1], admitted=("claude-seat1-model",))
    allowance = _route(answer, "claude-seat1-model").allowance_id

    assert allowance == "credential:SEAT1_WORKER_KEY"
    assert "os.environ" not in allowance


def test_renaming_group_does_not_change_any_allowance_id():
    """The regression this design exists to prevent.

    The downstream asked for a slug derived from `group`. Slugifying a
    sentence does not make it stable — renaming the sentence still moves the
    key, and a client's cap would move with it, silently.
    """
    renamed = dict(SEAT1, group="Seat one (renamed by the operator today)")

    before = _guidance(declared=[SEAT1], admitted=("claude-seat1-model",))
    after = _guidance(declared=[renamed], admitted=("claude-seat1-model",))

    assert (
        _route(before, "claude-seat1-model").allowance_id
        == _route(after, "claude-seat1-model").allowance_id
    )


def test_the_allowance_id_is_never_derived_from_the_alias():
    """Change only the Alias and the Allowance must not move: the two seats
    differ by credential, not by name."""
    renamed = dict(SEAT1, alias="claude-totally-different-name")

    answer = _guidance(declared=[renamed], admitted=("claude-totally-different-name",))

    assert (
        _route(answer, "claude-totally-different-name").allowance_id
        == "credential:SEAT1_WORKER_KEY"
    )


# --- fair_use is a risk, not a cost -------------------------------------


def test_fair_use_reaches_the_route_and_leaves_the_cost_basis_alone():
    answer = _guidance(declared=[FAIR_USE_HOST], admitted=("claude-host-model",))
    route = _route(answer, "claude-host-model")

    assert route.fair_use is True
    assert route.cost_basis == entitlements.FLAT_RATE
    assert route.rate_is_list_price is True


def test_fair_use_defaults_to_false_and_never_none():
    """Absence is a claim here: a Policy that says nothing claims the
    allowance takes load normally. A `None` would make every older Policy
    read as unknown risk."""
    answer = _guidance(declared=[SEAT1], admitted=("claude-seat1-model",))

    assert _route(answer, "claude-seat1-model").fair_use is False


def test_fair_use_does_not_change_route_order():
    """It is a risk, not a cost. A caller filters on it; the ranking stays
    ordered by what the Route costs, so a fair-use Route does not quietly
    demote itself and hide behind a worse one."""
    ordinary = dict(SEAT1, cost_basis="flat_rate")
    answer = _guidance(
        declared=[FAIR_USE_HOST, ordinary],
        admitted=("claude-host-model", "claude-seat1-model"),
    )

    ranks = {
        route.alias: guidance._basis_rank(route.cost_basis)
        for row in answer.rows
        for route in row.routes
    }
    assert ranks["claude-host-model"] == ranks["claude-seat1-model"]


def test_the_why_line_names_a_fair_use_allowance():
    """The cost basis beside it reads as safe, and this is the part that is
    not, so an operator reading the text output must see it."""
    answer = _guidance(declared=[FAIR_USE_HOST], admitted=("claude-host-model",))
    row = next(r for r in answer.rows if r.canonical_model_id == "claude-host-model")

    assert "fair-use allowance" in row.why


def test_a_feed_route_never_claims_fair_use():
    """The Feed publishes no such clause, so it cannot be read from one."""
    answer = _guidance(declared=[], admitted=("openrouter:vendor/scored",))

    assert _route(answer, "claude-or-scored").fair_use is False


# --- The guidance schema does not move ----------------------------------


def test_adding_these_fields_did_not_bump_the_guidance_schema():
    """A downstream client pins major 3 and fails loudly on anything else, so
    a bump it was not told about takes the whole proxy away from it. Both
    fields are additive: a consumer ignoring them parses what it always
    parsed."""
    assert guidance.SCHEMA_VERSION == "3"


# --- One entitlements entry per Allowance -------------------------------


def test_each_declared_allowance_gets_its_own_entitlements_entry():
    view = _view(
        declared=[SEAT1, SEAT2, FAIR_USE_HOST],
        admitted=("claude-seat1-model", "claude-seat2-model", "claude-host-model"),
    )

    allowances = [e.allowance_id for e in view.entitlements if e.provider_id == "declared"]

    assert allowances == [
        "credential:HOST_API_KEY",
        "credential:SEAT1_WORKER_KEY",
        "credential:SEAT2_WORKER_KEY",
    ]


def test_one_drained_allowance_leaves_another_healthy():
    """The gap this closes. Before, a whole host reported one aggregate count
    with no state, so a caller could not tell a drained seat from a healthy
    one and the only way to find a ceiling was to hit it."""
    view = _view(
        declared=[SEAT1, SEAT2],
        admitted=("claude-seat2-model",),
        excluded=("claude-seat1-model",),
        health={
            "claude-seat1-model": OfferingHealth(
                excluded=True,
                reason="quota_exhausted",
                bucket="quota",
                reset_at=NOW.replace(hour=21),
            )
        },
    )
    by_allowance = {e.allowance_id: e for e in view.entitlements}

    assert by_allowance["credential:SEAT1_WORKER_KEY"].state == "dry"
    assert by_allowance["credential:SEAT1_WORKER_KEY"].earliest_refill_at is not None
    assert by_allowance["credential:SEAT2_WORKER_KEY"].state == "healthy"


def test_a_declared_allowance_carries_its_cost_basis_and_fair_use():
    view = _view(declared=[FAIR_USE_HOST], admitted=("claude-host-model",))
    entry = next(e for e in view.entitlements if e.provider_id == "declared")

    assert entry.cost_bases == (entitlements.FLAT_RATE,)
    assert entry.fair_use is True


def test_a_feed_provider_keeps_its_position_and_gains_an_allowance_id():
    """Declared entries are appended, so a consumer indexing the list reads
    what it always read."""
    view = _view(declared=[SEAT1], admitted=("openrouter:vendor/scored", "claude-seat1-model"))

    assert view.entitlements[0].provider_id == "openrouter"
    assert view.entitlements[0].allowance_id == "provider:openrouter"


def test_a_client_facing_variant_does_not_double_count_an_allowance():
    """A variant is the same Offering under a second name and shares its
    sibling's health record, so counting it would report a host as twice its
    real size."""
    variant = {
        "alias": "claude-seat1-model[1m]",
        "variant_of": "claude-seat1-model",
        "litellm_params": dict(SEAT1["litellm_params"]),
    }
    view = _view(
        declared=[SEAT1, variant],
        admitted=("claude-seat1-model", "claude-seat1-model[1m]"),
    )
    entry = next(e for e in view.entitlements if e.provider_id == "declared")

    assert entry.answering == 1
    assert entry.in_scope == 1


def test_both_renderings_name_the_allowance_not_the_word_declared_four_times():
    view = _view(
        declared=[SEAT1, SEAT2],
        admitted=("claude-seat1-model", "claude-seat2-model"),
    )

    text = entitlements.render_text(view)
    markdown = entitlements.render_markdown(view)

    for rendering in (text, markdown):
        assert "credential:SEAT1_WORKER_KEY" in rendering
        assert "credential:SEAT2_WORKER_KEY" in rendering


# --- A provider's cost basis may be stated by Policy --------------------


def test_policy_may_state_a_providers_cost_basis_over_the_feeds_pricing_kind():
    """Measured 2026-07-28: the Feed marks Groq `paid` or `unknown` on an
    account where every call is free, because it cannot see the plan. Without
    this the provider reads as spend and a caller avoids capacity already
    paid for."""
    paid_feed = dict(
        FEED_RAW,
        models=[dict(FEED_RAW["models"][0], pricing={"kind": "paid", "metering": "tokens"})],
    )
    answer = guidance.derive(
        feed=parse_feed(paid_feed),
        policy=_policy([], providers={"openrouter": {"mode": "all", "cost_basis": "free"}}),
        health={},
        report=PlanReport(
            admitted=("openrouter:vendor/scored",),
            aliases={"openrouter:vendor/scored": "claude-or-scored"},
        ),
        now=NOW,
    )

    assert _route(answer, "claude-or-scored").cost_basis == entitlements.FREE


def test_a_stated_provider_cost_basis_reaches_the_entitlement_view():
    view = _view(
        declared=[],
        admitted=("openrouter:vendor/scored",),
        providers={"openrouter": {"mode": "all", "cost_basis": "flat_rate"}},
    )

    assert view.entitlements[0].cost_bases == (entitlements.FLAT_RATE,)


def test_a_provider_cost_basis_the_code_does_not_know_is_refused():
    from litellm_maintainer.policy import PolicyError

    with pytest.raises(PolicyError, match="cost_basis"):
        _policy([], providers={"openrouter": {"mode": "all", "cost_basis": "cheap"}})


# --- Pinned against the operator's real Policy --------------------------


@pytest.mark.skipif(
    not PINNED_POLICY_PATH.exists(), reason="the operator's Policy is not on this host"
)
def test_two_seats_of_one_provider_resolve_to_two_allowances_not_one():
    """The credential decides the Allowance. Any provider-level field
    would call two seats of the same provider one Allowance, and their
    quotas are separate.

    The fixture Policy declares two seats that differ only by
    `api_key`. Both must resolve, and to different Allowances.

    Do not assert a total count here. The count follows from how many
    Declared Offerings the fixture holds, so it measures the fixture and
    not the rule.
    """
    policy = load_policy(PINNED_POLICY_PATH)

    allowances = {
        entitlements.allowance_id_for_declared(d)
        for d in policy.declared
        if d.variant_of is None
    }

    seat1 = "credential:EXAMPLE_CHATGPT_SEAT1_WORKER_KEY"
    seat2 = "credential:EXAMPLE_CHATGPT_SEAT2_WORKER_KEY"
    assert seat1 in allowances
    assert seat2 in allowances
    assert seat1 != seat2
    # An Allowance id names the variable, never the `os.environ/` reference.
    assert not any("os.environ" in a for a in allowances)


def test_the_declared_aggregate_no_longer_counts_a_variant_as_an_offering():
    """Corrected on 2026-07-28, with SCHEMA_VERSION.

    `declared` counted each Client-Facing Variant as an Offering of its own,
    so the operator's 20 Declared Offerings read 24 while the per-Allowance
    entries — which never counted one — read 20. A variant shares its
    sibling's Health Key (ADR 0007), so it was never an Offering.
    """
    variant = {
        "alias": "claude-seat1-model[1m]",
        "variant_of": "claude-seat1-model",
        "litellm_params": dict(SEAT1["litellm_params"]),
    }
    view = _view(
        declared=[SEAT1, variant],
        admitted=("claude-seat1-model", "claude-seat1-model[1m]"),
    )

    assert view.declared_answering == 1
    declared_entries = [e for e in view.entitlements if e.provider_id == "declared"]
    assert sum(e.answering for e in declared_entries) == view.declared_answering
