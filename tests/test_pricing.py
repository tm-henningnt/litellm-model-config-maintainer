"""Tests for ticket 09: cost metadata and Feed-shape validation.

Assert external behaviour: what `model_info` `pricing.cost_model_info`
adds to a translated entry, and what `plan` puts in `report.
pricing_contradictions`, `report.duplicate_provider_model_ids` and
`report.feed_notices`. A test name states a rule an operator would
recognise (spec's "What makes a good test here").

Three tests read a real Offering from the frozen
`tests/fixtures/feed-audited.json`, translated through the real
`translate_offering`, so the exact numbers under test are pinned
against real Feed data. The contradiction, collision and
unfamiliar-notice-shape tests construct their own Feed document in
memory: no real Offering in either frozen fixture carries a `free`
kind with a non-zero rate, and both frozen fixtures carry zero
duplicate provider+model ids (see the ticket report for how that was
checked).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from litellm_maintainer.feed import Feed, load_feed, parse_feed
from litellm_maintainer.plan import plan
from litellm_maintainer.policy import parse_policy
from litellm_maintainer.pricing import (
    SUBSCRIPTION_LIST_PRICE_KEY,
    cost_model_info,
    find_duplicate_provider_model_ids,
    is_native_litellm_prefix,
    summarize_feed_notices,
)
from litellm_maintainer.translate import translate_offering

FIXTURES = Path(__file__).parent / "fixtures"
FEED_AUDITED_PATH = FIXTURES / "feed-audited.json"


# --- Real Offerings, translated through the real rule -------------------


def _translated(feed: Feed, offering_id: str):
    offering = feed.offering(offering_id)
    assert offering is not None, f"fixture no longer carries {offering_id!r}"
    provider = feed.providers.get(offering.provider_id)
    litellm_params = translate_offering(offering, provider)
    return offering, litellm_params


def test_an_offering_metered_in_tokens_carries_input_and_output_cost_metadata():
    # cline:anthropic/claude-opus-5-fast, tokens metering, kind "paid",
    # input=10, output=50 (USD per 1M tokens) in feed-audited.json.
    # Translated through `generic_openai_compatible` (an explicit
    # `api_base`, so not a native prefix). Hand-computed conversion:
    # 10 / 1_000_000 = 0.00001, 50 / 1_000_000 = 0.00005.
    feed = load_feed(FEED_AUDITED_PATH)
    offering, litellm_params = _translated(feed, "cline:anthropic/claude-opus-5-fast")
    assert litellm_params["model"] == "openai/anthropic/claude-opus-5-fast"

    info, contradiction = cost_model_info(offering, litellm_params)

    assert info["input_cost_per_token"] == 0.00001
    assert info["output_cost_per_token"] == 0.00005
    assert contradiction is None


def test_a_subscription_pool_offering_is_marked_in_the_generated_entry():
    # opencode-go:minimax-m3, tokens metering, kind
    # "subscription_included", input=0.3, output=1.2 in
    # feed-audited.json.
    feed = load_feed(FEED_AUDITED_PATH)
    offering, litellm_params = _translated(feed, "opencode-go:minimax-m3")
    assert offering.pricing_kind == "subscription_included"

    info, contradiction = cost_model_info(offering, litellm_params)

    assert info[SUBSCRIPTION_LIST_PRICE_KEY] is True
    assert info["input_cost_per_token"] == 0.3 / 1_000_000
    assert info["output_cost_per_token"] == 1.2 / 1_000_000
    assert contradiction is None


def test_an_offering_with_a_native_litellm_prefix_receives_no_cost_metadata():
    # groq:openai/gpt-oss-120b, tokens metering, kind "paid", input=0.15,
    # output=0.6 in feed-audited.json — a real Offering that WOULD get
    # cost metadata by every other rule here, translated through
    # `native_prefix("groq")`, which sets no `api_base`. litellm prices
    # it from its own map; no metadata is added.
    feed = load_feed(FEED_AUDITED_PATH)
    offering, litellm_params = _translated(feed, "groq:openai/gpt-oss-120b")
    assert "api_base" not in litellm_params
    assert is_native_litellm_prefix(litellm_params) is True

    info, contradiction = cost_model_info(offering, litellm_params)

    assert info == {}
    assert contradiction is None


def test_an_offering_metered_in_anything_other_than_tokens_receives_no_cost_metadata():
    # No real Offering among the translatable providers in
    # feed-audited.json meters in anything but tokens (checked
    # directly: every opencode-go/opencode-zen/cline/cline-pass
    # Offering there is `metering: tokens`). Constructed: a synthetic
    # opencode-go Offering metered in "credits", translated through the
    # same generic rule as the real minimax-m3 Offering above, so only
    # the metering unit differs from a case that DOES get metadata.
    offering_raw = {
        "id": "opencode-go:credits-metered-example",
        "provider": {"id": "opencode-go"},
        "provider_model_id": "credits-metered-example",
        "capabilities": ["tool_use"],
        "endpoint": {"base_url": "https://opencode-go.example/v1"},
        "pricing": {
            "kind": "paid",
            "input_usd_per_1m_tokens": None,
            "output_usd_per_1m_tokens": None,
            "metering": "credits",
        },
        "availability": {"status": "available"},
        "quality": {"coding_score": 50.0},
        "policy": {"visibility": "listed", "tags": []},
    }
    feed = parse_feed(
        {
            "schema_version": "test",
            "providers": [
                {
                    "id": "opencode-go",
                    "name": "OpenCode Go",
                    "default_base_url": "https://opencode-go.example/v1",
                    "authentication": {},
                }
            ],
            "models": [offering_raw],
        }
    )
    offering, litellm_params = _translated(feed, "opencode-go:credits-metered-example")
    assert "api_base" in litellm_params  # not a native prefix

    info, contradiction = cost_model_info(offering, litellm_params)

    assert info == {}
    assert contradiction is None


def test_an_offering_priced_free_with_a_non_zero_token_rate_is_treated_as_paid():
    # Constructed: no real Offering in either frozen fixture carries a
    # `free` kind alongside a non-zero token rate (checked directly
    # against both). This is the exact shape spec's "Safety" section
    # warns about: a mirror provider once listed the same model twice
    # with conflicting prices, and whichever copy survived the merge
    # depended on array order.
    offering_raw = {
        "id": "opencode-go:contradictory-example",
        "provider": {"id": "opencode-go"},
        "provider_model_id": "contradictory-example",
        "capabilities": ["tool_use"],
        "endpoint": {"base_url": "https://opencode-go.example/v1"},
        "pricing": {
            "kind": "free",
            "input_usd_per_1m_tokens": 3.0,
            "output_usd_per_1m_tokens": 12.0,
            "metering": "tokens",
        },
        "availability": {"status": "available"},
        "quality": {"coding_score": 50.0},
        "policy": {"visibility": "listed", "tags": []},
    }
    feed = parse_feed(
        {
            "schema_version": "test",
            "providers": [
                {
                    "id": "opencode-go",
                    "name": "OpenCode Go",
                    "default_base_url": "https://opencode-go.example/v1",
                    "authentication": {},
                }
            ],
            "models": [offering_raw],
        }
    )
    offering, litellm_params = _translated(feed, "opencode-go:contradictory-example")

    info, contradiction = cost_model_info(offering, litellm_params)

    # Treated as paid: the cost metadata is still written.
    assert info["input_cost_per_token"] == 3.0 / 1_000_000
    assert info["output_cost_per_token"] == 12.0 / 1_000_000
    assert contradiction is not None
    assert contradiction.offering_id == "opencode-go:contradictory-example"
    assert "free" in contradiction.message
    assert "paid" in contradiction.message


# --- Feed-shape validation, exercised through `plan` ---------------------


def _offering_raw(
    *,
    id: str,
    provider_id: str = "opencode-go",
    provider_model_id: str | None = None,
    coding_score: float = 50.0,
) -> dict[str, Any]:
    model_id = provider_model_id if provider_model_id is not None else id.split(":", 1)[1]
    return {
        "id": id,
        "provider": {"id": provider_id},
        "provider_model_id": model_id,
        "capabilities": ["tool_use"],
        "endpoint": {"base_url": "https://opencode-go.example/v1", "model": model_id},
        "pricing": {"kind": "subscription_included", "metering": "tokens"},
        "availability": {"status": "available"},
        "quality": {"coding_score": coding_score},
        "policy": {"visibility": "listed", "tags": []},
    }


def _feed_with(*offerings: dict[str, Any], notices: list[Any] | None = None) -> Feed:
    raw: dict[str, Any] = {
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
    if notices is not None:
        raw["notices"] = notices
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


def test_two_offerings_sharing_a_provider_and_model_id_are_reported_as_a_collision():
    # Constructed: both frozen fixtures carry zero duplicate
    # provider+model ids (verified directly: see the ticket report).
    # Two distinct Offering ids (`id` is `<provider>:<discriminator>`)
    # can still share the same `(provider_id, provider_model_id)` pair
    # when a mirror publishes the same model under two ids — the exact
    # hazard the spec names. `naming.alias_overrides` on the second
    # keeps the two from also colliding on Alias, so the Alias-collision
    # refusal (ticket 08) does not mask this report.
    first = _offering_raw(id="opencode-go:dup-one", provider_model_id="shared-model")
    second = _offering_raw(id="opencode-go:dup-two", provider_model_id="shared-model")
    feed = _feed_with(first, second)
    policy = _policy(
        naming={
            "alias_prefix": "claude-",
            "provider_labels": {"opencode-go": "opencode-go"},
            "alias_overrides": {"opencode-go:dup-two": "claude-dup-two-renamed"},
        }
    )

    result = plan(feed=feed, policy=policy, health={}, now=None)  # type: ignore[arg-type]

    assert result.refusal is None
    duplicates = find_duplicate_provider_model_ids(feed)
    assert len(duplicates) == 1
    assert duplicates[0].provider_id == "opencode-go"
    assert duplicates[0].provider_model_id == "shared-model"
    assert set(duplicates[0].offering_ids) == {"opencode-go:dup-one", "opencode-go:dup-two"}
    assert result.report.duplicate_provider_model_ids == duplicates
    # The run still proceeds: both Offerings are admitted.
    assert "opencode-go:dup-one" in result.report.admitted
    assert "opencode-go:dup-two" in result.report.admitted


def test_offerings_with_distinct_provider_model_ids_are_not_reported_as_a_collision():
    one = _offering_raw(id="opencode-go:one", provider_model_id="model-one")
    two = _offering_raw(id="opencode-go:two", provider_model_id="model-two")
    feed = _feed_with(one, two)
    policy = _policy()

    result = plan(feed=feed, policy=policy, health={}, now=None)  # type: ignore[arg-type]

    assert result.report.duplicate_provider_model_ids == ()


def test_neither_frozen_fixture_carries_a_duplicate_provider_model_id():
    # Reproduces the acceptance command's check directly, against both
    # pinned Feed snapshots. Both are empty: 0 duplicates in
    # feed-audited.json and 0 in feed-current.json.
    for name in ("feed-audited.json", "feed-current.json"):
        feed = load_feed(FIXTURES / name)
        assert find_duplicate_provider_model_ids(feed) == ()


# --- The Feed's own notices -----------------------------------------------


def test_the_feeds_notices_appear_in_the_report():
    notices = [
        {"message": "alias targets not found", "collector": "canonicalize"},
        {"message": "dropped stale offerings", "collector": "retire-opencode-models"},
    ]
    offering = _offering_raw(id="opencode-go:one")
    feed = _feed_with(offering, notices=notices)
    policy = _policy()

    result = plan(feed=feed, policy=policy, health={}, now=None)  # type: ignore[arg-type]

    assert result.report.feed_notices == (
        "canonicalize: alias targets not found",
        "retire-opencode-models: dropped stale offerings",
    )


def test_a_notice_with_an_unfamiliar_shape_reduces_the_report_not_the_run():
    # Constructed: no shape a real collector has emitted lacks
    # `message` or is a bare string, but the spec requires this to
    # reduce the report, never fail the run, so it must be tested even
    # without a real example.
    notices = [
        {"message": "a normal notice", "collector": "example-collector"},
        {"offering_ids": ["opencode-go:one"]},  # no "message" key at all
        "a bare string, not a mapping at all",
        42,
    ]
    offering = _offering_raw(id="opencode-go:one")
    feed = _feed_with(offering, notices=notices)
    policy = _policy()

    result = plan(feed=feed, policy=policy, health={}, now=None)  # type: ignore[arg-type]

    # Only the one readable notice appears. The run still produced a
    # config; it was not failed by the three unreadable notices.
    assert result.report.feed_notices == ("example-collector: a normal notice",)
    assert result.refusal is None
    assert "opencode-go:one" in result.report.admitted


def test_an_empty_notices_list_produces_an_empty_report_and_no_error():
    offering = _offering_raw(id="opencode-go:one")
    feed = _feed_with(offering, notices=[])
    policy = _policy()

    result = plan(feed=feed, policy=policy, health={}, now=None)  # type: ignore[arg-type]

    assert result.report.feed_notices == ()
    assert result.refusal is None


def test_a_feed_document_with_no_notices_key_produces_an_empty_report_and_no_error():
    offering = _offering_raw(id="opencode-go:one")
    feed = _feed_with(offering)  # no `notices=` passed at all
    assert feed.notices == ()
    policy = _policy()

    result = plan(feed=feed, policy=policy, health={}, now=None)  # type: ignore[arg-type]

    assert result.report.feed_notices == ()
    assert result.refusal is None


def test_summarize_feed_notices_is_tolerant_of_a_non_list_free_form_object():
    # Direct unit test of the helper, independent of `plan`, covering
    # the same "reduce, never fail" rule at the smallest scope.
    assert summarize_feed_notices([]) == ()
    assert summarize_feed_notices(
        [{"message": "x", "collector": "c"}, {"no_message": True}]
    ) == ("c: x",)
