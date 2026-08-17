"""A Declared Offering borrows the Feed's numbers for the same model.

The Feed cannot publish a private host or a subscription seat, so a
Declared Offering carried no score, no rate and cost basis `unknown`. An
agent is told to treat `unknown` as spend, so the strongest models the
proxy serves sorted last and read as billable. Both statements were
wrong, and both were about the FEED's coverage rather than about the
model.

A Reference Model closes it. The operator names the Canonical Model id
the Feed does publish for the same model, and the Route joins that
model's Guidance Row. See ADR 0011.

Three rules these tests pin:

1. The score transfers. It describes the model, and the model is the same.
2. A limit NEVER transfers. It describes the endpoint, and the endpoint is
   not the same: the ChatGPT seats accept about 350,000 tokens where the
   API mirror states 1,050,000. See ADR 0006.
3. Every number states its source. A caller weighs an operator's figure
   differently from the Feed's, so `score_source` and `rate_source` say
   which it is reading.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from litellm_maintainer import entitlements, guidance
from litellm_maintainer.feed import load_feed, parse_feed
from litellm_maintainer.plan import PlanReport
from litellm_maintainer.policy import (
    PolicyError,
    VALID_COST_BASES,
    load_policy,
    parse_policy,
)

NOW = datetime(2026, 7, 28, 12, 0, tzinfo=timezone.utc)

FIXTURES = Path(__file__).parent / "fixtures"
FEED_CURRENT_PATH = FIXTURES / "feed-current.json"
# Synthetic and committed. Never the operator's own Policy.
PINNED_POLICY_PATH = FIXTURES / "policy-pinned.yaml"

# One Canonical Model, `vendor/wide`, reached three ways in the Feed:
#
# - a PAID mirror at 2.00/8.00, which the proxy does not serve,
# - a cheaper PAID mirror at 1.00/4.00, also not served,
# - a FREE mirror at 0.00/0.00, which the proxy DOES serve.
#
# The three differences matter to three separate rules: which rate a
# Reference Model yields (the cheapest paid one), which rate it must never
# yield (the free mirror's zero) and where the score comes from when the
# proxy serves the model itself.
FEED_RAW = {
    "schema_version": "1.0.0",
    "feed": {"generated_at": "2026-07-28T11:00:00Z"},
    "providers": [
        {"id": "mirror-a", "name": "Mirror A"},
        {"id": "mirror-b", "name": "Mirror B"},
        {"id": "freehost", "name": "Free Host"},
    ],
    "models": [
        {
            "id": "mirror-a:vendor/wide",
            "provider": {"id": "mirror-a"},
            "provider_model_id": "vendor/wide",
            "canonical_model": {"id": "vendor/wide"},
            "display_name": "Vendor: Wide",
            "capabilities": ["chat", "coding", "tool_use", "vision"],
            "limits": {"context_tokens": 1_050_000, "max_output_tokens": 128_000},
            "pricing": {
                "kind": "paid",
                "metering": "tokens",
                "input_usd_per_1m_tokens": 2.0,
                "output_usd_per_1m_tokens": 8.0,
            },
            "availability": {"status": "available"},
            "quality": {"coding_score": 70.0, "reasoning_score": 50.0},
            "policy": {"visibility": "listed"},
            "endpoint": {},
        },
        {
            "id": "mirror-b:vendor/wide",
            "provider": {"id": "mirror-b"},
            "provider_model_id": "vendor/wide",
            "canonical_model": {"id": "vendor/wide"},
            "display_name": "Vendor: Wide",
            "capabilities": ["chat", "coding", "tool_use", "vision"],
            "limits": {"context_tokens": 1_050_000, "max_output_tokens": 128_000},
            "pricing": {
                "kind": "paid",
                "metering": "tokens",
                "input_usd_per_1m_tokens": 1.0,
                "output_usd_per_1m_tokens": 4.0,
            },
            "availability": {"status": "available"},
            "quality": {"coding_score": 70.0, "reasoning_score": 50.0},
            "policy": {"visibility": "listed"},
            "endpoint": {},
        },
        {
            "id": "freehost:vendor/wide",
            "provider": {"id": "freehost"},
            "provider_model_id": "vendor/wide",
            "canonical_model": {"id": "vendor/wide"},
            "display_name": "Vendor: Wide",
            "capabilities": ["chat", "coding"],
            "limits": {"context_tokens": 262_144},
            "pricing": {
                "kind": "free",
                "metering": "tokens",
                "input_usd_per_1m_tokens": 0.0,
                "output_usd_per_1m_tokens": 0.0,
            },
            "availability": {"status": "available"},
            "quality": {"coding_score": 70.0, "reasoning_score": 50.0},
            "policy": {"visibility": "listed"},
            "endpoint": {},
        },
    ],
}

# The declared entry under test: a private host serving `vendor/wide`,
# with a window of its own that is far narrower than the mirror's.
SEAT = {
    "alias": "claude-seat-wide",
    "reference_model": "vendor/wide",
    "cost_basis": "flat_rate",
    "model_info": {"max_input_tokens": 350_000},
    "litellm_params": {
        "model": "openai/wide",
        "api_base": "http://127.0.0.1:4011/v1",
        "api_key": "os.environ/SEAT_KEY",
    },
}


def _policy(declared, providers=None):
    return parse_policy(
        {
            "providers": providers if providers is not None else {},
            "quality": {"minimum_coding_score": 10},
            "approved_candidates": [],
            "naming": {
                "alias_prefix": "claude-",
                "provider_labels": {"freehost": "free"},
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


def _derive(*, declared, admitted, providers=None, excluded=(), **kwargs):
    return guidance.derive(
        feed=parse_feed(FEED_RAW),
        policy=_policy(declared, providers),
        health={},
        report=PlanReport(
            admitted=admitted,
            excluded=excluded,
            aliases={"freehost:vendor/wide": "claude-free-wide"},
        ),
        now=NOW,
        **kwargs,
    )


def _row(answer, canonical_model_id):
    return next(r for r in answer.rows if r.canonical_model_id == canonical_model_id)


def _route(answer, alias):
    for row in answer.rows:
        for route in row.routes:
            if route.alias == alias:
                return route
    raise AssertionError(f"no Route for {alias!r}")


# --- The score transfers -------------------------------------------------


def test_a_declared_offering_naming_a_reference_model_joins_that_models_row():
    answer = _derive(declared=[SEAT], admitted=("claude-seat-wide",))

    ids = [row.canonical_model_id for row in answer.rows]

    assert ids == ["vendor/wide"]
    assert "claude-seat-wide" not in ids


def test_the_row_carries_the_feeds_score_for_the_reference_model():
    answer = _derive(declared=[SEAT], admitted=("claude-seat-wide",))
    row = _row(answer, "vendor/wide")

    assert row.score == 70.0
    assert row.scores["reasoning"] == 50.0
    assert row.display_name == "Vendor: Wide"


def test_the_score_source_says_reference_when_only_declared_routes_reach_the_model():
    """The proxy serves no Offering of this model, so nothing here was
    measured against a Route the Feed scored. The row says so."""
    answer = _derive(declared=[SEAT], admitted=("claude-seat-wide",))
    row = _row(answer, "vendor/wide")

    assert row.score_source == guidance.SOURCE_REFERENCE
    assert "reference" in row.why


def test_the_score_source_says_feed_when_the_proxy_serves_the_model_itself():
    """An Offering the proxy serves outranks a Reference Model as a source:
    it is the one the Feed scored AND the one that answers."""
    answer = _derive(
        declared=[SEAT],
        providers={"freehost": {"mode": "all"}},
        admitted=("freehost:vendor/wide", "claude-seat-wide"),
    )
    row = _row(answer, "vendor/wide")

    assert row.score_source == guidance.SOURCE_FEED
    assert row.score == 70.0
    assert {r.alias for r in row.routes} == {"claude-free-wide", "claude-seat-wide"}


def test_two_declared_offerings_naming_one_reference_model_become_one_row():
    """Both ChatGPT seats reach `openai/gpt-5.6-sol`. A flat list of
    Aliases named that model twice before reaching the second model."""
    seat_two = dict(SEAT, alias="claude-seat2-wide")
    answer = _derive(
        declared=[SEAT, seat_two],
        admitted=("claude-seat-wide", "claude-seat2-wide"),
    )

    assert len(answer.rows) == 1
    assert {r.alias for r in answer.rows[0].routes} == {
        "claude-seat-wide",
        "claude-seat2-wide",
    }


# --- A limit never transfers ---------------------------------------------


def test_the_reference_models_window_never_reaches_the_declared_route():
    """ADR 0006: a Stated Limit comes from a source, and the source for
    this endpoint is the operator's own measurement. Taking the mirror's
    1,050,000 would trade an early compaction for a hard refusal."""
    answer = _derive(declared=[SEAT], admitted=("claude-seat-wide",))
    route = _route(answer, "claude-seat-wide")

    assert route.context_tokens == 350_000
    assert route.max_output_tokens is None


def test_a_declared_route_with_no_model_info_states_no_window_at_all():
    """Absence reads as unknown, never as the mirror's figure."""
    bare = {k: v for k, v in SEAT.items() if k != "model_info"}
    answer = _derive(declared=[bare], admitted=("claude-seat-wide",))

    assert _route(answer, "claude-seat-wide").context_tokens is None


# --- The cost basis ------------------------------------------------------


def test_policy_states_the_cost_basis_and_it_wins():
    answer = _derive(declared=[SEAT], admitted=("claude-seat-wide",))
    route = _route(answer, "claude-seat-wide")

    assert route.cost_basis == entitlements.FLAT_RATE
    assert route.rate_is_list_price is True


def test_without_a_stated_basis_the_earlier_rule_still_stands():
    """`passthrough` when the caller supplies the credential, `unknown`
    otherwise. Both are what this reported before Policy could say."""
    plain = {k: v for k, v in SEAT.items() if k != "cost_basis"}
    caller_billed = dict(plain, alias="claude-caller-billed", passthrough_auth=True)
    answer = _derive(
        declared=[plain, caller_billed],
        admitted=("claude-seat-wide", "claude-caller-billed"),
    )

    assert _route(answer, "claude-seat-wide").cost_basis == entitlements.UNKNOWN_BASIS
    assert _route(answer, "claude-caller-billed").cost_basis == entitlements.PASSTHROUGH


# --- The rate and its source --------------------------------------------


def test_the_cheapest_paid_mirror_supplies_the_reference_rate():
    """Mirrors of one model disagree. Measured 2026-07-28:
    `openai/gpt-5.6-terra` read 1.25/7.50 through one provider and
    2.50/15.00 through another, and the lower pair matched the vendor's
    own published price."""
    answer = _derive(declared=[SEAT], admitted=("claude-seat-wide",))
    route = _route(answer, "claude-seat-wide")

    assert route.input_usd_per_1m_tokens == 1.0
    assert route.output_usd_per_1m_tokens == 4.0
    assert route.rate_source == guidance.SOURCE_REFERENCE


def test_a_free_mirrors_zero_is_never_read_as_the_reference_rate():
    """A free mirror's 0.00 describes that mirror's promotion. Reading it
    would report every model with a free tier as costless everywhere,
    which is the one error that cannot be noticed from the output."""
    feed_free_only = dict(
        FEED_RAW,
        models=[m for m in FEED_RAW["models"] if m["pricing"]["kind"] == "free"],
    )
    answer = guidance.derive(
        feed=parse_feed(feed_free_only),
        policy=_policy([SEAT]),
        health={},
        report=PlanReport(admitted=("claude-seat-wide",)),
        now=NOW,
    )
    route = _route(answer, "claude-seat-wide")

    assert route.input_usd_per_1m_tokens is None
    assert route.rate_source is None


def test_the_operators_own_rate_wins_over_the_reference_rate():
    """The operator's figure describes the endpoint the proxy dials. A
    private host's fixed-rate plan is not the mirror's list price."""
    priced = dict(
        SEAT,
        pricing={
            "input_usd_per_1m_tokens": 0.25,
            "output_usd_per_1m_tokens": 0.5,
        },
    )
    answer = _derive(declared=[priced], admitted=("claude-seat-wide",))
    route = _route(answer, "claude-seat-wide")

    assert route.input_usd_per_1m_tokens == 0.25
    assert route.output_usd_per_1m_tokens == 0.5
    assert route.rate_source == guidance.SOURCE_OPERATOR


def test_an_operator_rate_needs_no_reference_model_at_all():
    """A model the Feed does not publish anywhere still states a rate, so
    a caller can weigh its burn even with no score to rank it by."""
    unlisted = {
        "alias": "claude-unlisted",
        "cost_basis": "flat_rate",
        "pricing": {
            "input_usd_per_1m_tokens": 0.25,
            "output_usd_per_1m_tokens": 0.8,
        },
        "litellm_params": {"model": "openai/unlisted", "api_base": "https://x/v1"},
    }
    answer = _derive(declared=[unlisted], admitted=("claude-unlisted",))
    row = _row(answer, "claude-unlisted")

    assert row.score is None
    assert row.score_source is None
    assert row.routes[0].output_usd_per_1m_tokens == 0.8
    assert row.routes[0].rate_source == guidance.SOURCE_OPERATOR


def test_a_feed_offerings_rate_reports_the_feed_as_its_source():
    answer = _derive(
        declared=[],
        providers={"freehost": {"mode": "all"}},
        admitted=("freehost:vendor/wide",),
    )

    assert _route(answer, "claude-free-wide").rate_source == guidance.SOURCE_FEED


# --- Capabilities --------------------------------------------------------


def test_the_operators_capability_list_wins_on_a_reference_row():
    """A host can serve one model with tool use disabled. The operator
    states what THIS endpoint serves; the mirror states what the mirror
    serves."""
    narrowed = dict(SEAT, capabilities=["chat", "coding"])
    answer = _derive(declared=[narrowed], admitted=("claude-seat-wide",))
    row = _row(answer, "vendor/wide")

    assert row.capabilities == ("chat", "coding")
    assert row.capabilities_are_operator_stated is True
    assert "capabilities stated by the operator" in row.why


def test_a_reference_row_falls_back_to_the_feeds_capabilities():
    answer = _derive(declared=[SEAT], admitted=("claude-seat-wide",))
    row = _row(answer, "vendor/wide")

    assert "vision" in row.capabilities
    assert row.capabilities_are_operator_stated is False


# --- A Reference Model the Feed does not publish -------------------------


def test_a_reference_row_unscored_on_one_axis_keeps_its_source_and_its_wording():
    """`score_source` describes the SOURCE of a figure, never its presence.

    The Feed scores most of the catalogue on `coding` and almost none of it
    on `speed`. A seat row asked for `speed` must not claim the Feed knows
    nothing about the model: it knows the model and not that axis."""
    answer = _derive(declared=[SEAT], admitted=("claude-seat-wide",), axis="speed")
    row = _row(answer, "vendor/wide")

    assert row.score is None
    assert row.score_source == guidance.SOURCE_REFERENCE
    assert "carries no score on the requested axis" in row.why
    assert "the Feed does not score it" not in row.why


def test_a_declared_row_with_no_reference_model_still_says_the_feed_covers_it_not():
    unlisted = {
        "alias": "claude-unlisted",
        "litellm_params": {"model": "anthropic/unlisted"},
    }
    answer = _derive(declared=[unlisted], admitted=("claude-unlisted",))

    assert "the Feed does not score it" in _row(answer, "claude-unlisted").why


def test_an_unknown_reference_model_warns_and_keeps_the_aliass_own_row():
    """A typo must not quietly remove the score it was written to add."""
    typo = dict(SEAT, reference_model="vendor/wdie")
    answer = _derive(declared=[typo], admitted=("claude-seat-wide",))

    assert [r.canonical_model_id for r in answer.rows] == ["claude-seat-wide"]
    assert _row(answer, "claude-seat-wide").score is None
    assert any("vendor/wdie" in w for w in answer.warnings)
    assert any("claude-seat-wide" in w for w in answer.warnings)


# --- The schema a consumer parses ---------------------------------------


def test_the_json_shape_carries_both_sources():
    answer = _derive(declared=[SEAT], admitted=("claude-seat-wide",))
    row = _row(answer, "vendor/wide").as_dict()

    assert row["score_source"] == "reference"
    assert row["routes"][0]["rate_source"] == "reference"
    assert row["routes"][0]["cost_basis"] == "flat_rate"


# --- Policy validation ---------------------------------------------------


def test_a_cost_basis_the_code_does_not_know_is_refused():
    with pytest.raises(PolicyError, match="cost_basis"):
        _policy([dict(SEAT, cost_basis="cheap")])


def test_one_rate_without_the_other_is_refused():
    """A caller comparing two models reads a missing rate as zero, so a
    half-stated pair is a wrong answer rather than a missing one."""
    with pytest.raises(PolicyError, match="output_usd_per_1m_tokens"):
        _policy([dict(SEAT, pricing={"input_usd_per_1m_tokens": 0.25})])


def test_a_negative_rate_is_refused():
    with pytest.raises(PolicyError, match="negative"):
        _policy(
            [
                dict(
                    SEAT,
                    pricing={
                        "input_usd_per_1m_tokens": -1.0,
                        "output_usd_per_1m_tokens": 1.0,
                    },
                )
            ]
        )


def test_the_cost_basis_vocabulary_is_defined_once():
    """`policy` owns the five names and `entitlements` re-exports them. Two
    definitions would drift, and a basis Policy accepts but `guidance`
    cannot rank is a silent mis-sort."""
    assert VALID_COST_BASES == {
        entitlements.FREE,
        entitlements.FLAT_RATE,
        entitlements.METERED,
        entitlements.PASSTHROUGH,
        entitlements.UNKNOWN_BASIS,
    }
    assert set(guidance._BASIS_ORDER) == VALID_COST_BASES


# --- Pinned against the operator's real Policy and Feed ------------------


@pytest.mark.skipif(
    not PINNED_POLICY_PATH.exists(), reason="the operator's Policy is not on this host"
)
def test_every_reference_model_in_the_pinned_policy_exists_in_the_feed():
    """A Reference Model naming a Canonical Model the Feed dropped yields
    no score, and `guidance` only warns. This fails instead."""
    policy = load_policy(PINNED_POLICY_PATH)
    feed = load_feed(FEED_CURRENT_PATH)

    stated = {
        d.alias: d.reference_model
        for d in policy.declared
        if d.reference_model is not None
    }
    assert stated, "the fixture Policy names no Reference Model at all"

    missing = {
        alias: model
        for alias, model in stated.items()
        if not feed.offerings_for_canonical_model(model)
    }
    assert missing == {}


@pytest.mark.skipif(
    not PINNED_POLICY_PATH.exists(), reason="the operator's Policy is not on this host"
)
def test_the_four_private_host_aliases_state_a_flat_rate_and_a_rate_of_their_own():
    """One fixed-rate plan bills all four, practically unlimited under a
    fair-use policy. `free` would give a caller no reason to moderate
    anything; the rates state the relative burn."""
    policy = load_policy(PINNED_POLICY_PATH)
    # A Client-Facing Variant states none of these: it is the same Offering
    # under a second name, and `guidance` folds it onto its sibling's Route
    # rather than giving it a Route of its own.
    private_host = [
        d
        for d in policy.declared
        if d.alias.startswith("claude-private-host-") and d.variant_of is None
    ]

    assert len(private_host) == 4
    for declared in private_host:
        assert declared.cost_basis == entitlements.FLAT_RATE, declared.alias
        assert declared.pricing is not None, declared.alias
        assert declared.reference_model is not None, declared.alias
        # One credential bills all four, so one 429 makes the rest worth
        # MEASURING (ADR 0004, ADR 0009).
        assert declared.entitlement == "shared_pool", declared.alias


# --- The Client Advisory names a Declared Alias verbatim ------------------


def test_the_advisory_names_a_declared_alias_verbatim():
    """A Declared Offering's id IS its Alias, so nothing derives one.

    Measured 2026-07-28: running the naming rule over a finished Alias
    produced `claude-claude-private-host-minimax-m3-`, which the proxy does not
    serve. The Advisory exists to say which Alias a caller may now call.
    """
    from litellm_maintainer.notify import PreviousRunState

    answer = guidance.derive(
        feed=parse_feed(FEED_RAW),
        policy=_policy([SEAT]),
        health={},
        report=PlanReport(admitted=("claude-seat-wide",)),
        previous=PreviousRunState(admitted=()),
        now=NOW,
    )

    assert answer.advisory.added_last_run == ("claude-seat-wide",)
