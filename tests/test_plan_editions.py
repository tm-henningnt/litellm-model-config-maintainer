"""Tests for the `plan_edition` Selection filter.

A provider can sell one roster per subscription edition. The Feed lists
each Offering's editions at `pricing.subscription.plan_editions`;
`providers.<id>.plan_edition` in Policy names the edition the operator
holds, and only an Offering on that roster is admitted.

Synthetic Feeds throughout, so no test depends on which edition the
operator happens to hold today. One test reads the operator's real
Policy and Feed, to pin the roster the Personal edition produces.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from litellm_maintainer.feed import load_feed, parse_feed
from litellm_maintainer.plan import plan
from litellm_maintainer.policy import PolicyError, load_policy, parse_policy

NOW = datetime(2026, 7, 26, tzinfo=timezone.utc)
FIXTURES = Path(__file__).parent / "fixtures"
OPERATOR_POLICY_PATH = Path("/Users/hentol/.config/litellm-maintainer/policy.yaml")
OPERATOR_FEED_PATH = Path.home() / ".config/litellm-maintainer/feed.json"


def _offering(model_id: str, editions: list[str] | None):
    """One subscription Offering, on `editions`, or on no roster at all."""
    subscription: dict = {"billing": "flat_monthly"}
    if editions is not None:
        subscription["plan_editions"] = editions
    return {
        "id": f"opencode-go:{model_id}",
        "provider": {"id": "opencode-go", "name": "OpenCode Go"},
        "provider_model_id": model_id,
        "capabilities": ["chat", "tool_use"],
        "endpoint": {"protocol": "openai_chat_completions", "base_url": "https://pool.invalid/v1"},
        "pricing": {"kind": "subscription_included", "subscription": subscription},
        "availability": {"status": "available"},
        "quality": {"coding_score": 60},
        "policy": {"visibility": "listed"},
        "limits": {"context_tokens": 200000},
    }


def _feed(models):
    return parse_feed(
        {
            "schema_version": "test",
            "providers": [
                {
                    "id": "opencode-go",
                    "name": "OpenCode Go",
                    "default_base_url": "https://pool.invalid/v1",
                    "authentication": {"credential_hint": "OPENCODE_API_KEY"},
                }
            ],
            "models": models,
        }
    )


def _policy(plan_edition=None):
    rule: dict = {"mode": "all"}
    if plan_edition is not None:
        rule["plan_edition"] = plan_edition
    return parse_policy(
        {
            "providers": {"opencode-go": rule},
            "quality": {"minimum_coding_score": 20},
            "approved_candidates": [],
            "naming": {
                "alias_prefix": "claude-",
                "provider_labels": {"opencode-go": "pool"},
                "alias_overrides": {},
            },
            "withheld": {},
            "declared": [],
            "pacing": {"default": {"concurrency": 2, "minimum_interval_seconds": 5}},
            "schedule": {
                "enabled": True,
                "interval_minutes": 60,
                "require_proxy": True,
                "maximum_staleness_hours": 24,
            },
            "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
        }
    )


def _admitted(feed, policy):
    result = plan(feed=feed, policy=policy, health={}, now=NOW)
    return {entry["model_name"] for entry in result.config["model_list"]}


BOTH = _offering("on-both", ["personal", "team"])
TEAM_ONLY = _offering("team-only", ["team"])
NO_ROSTER = _offering("pay-as-you-go", None)


def test_the_named_edition_admits_only_its_own_roster():
    feed = _feed([BOTH, TEAM_ONLY])
    assert _admitted(feed, _policy("personal")) == {"claude-pool-on-both"}


def test_a_wider_edition_admits_the_narrower_editions_offerings_too():
    """Personal is a subset of Team, so Team admits both."""
    feed = _feed([BOTH, TEAM_ONLY])
    assert _admitted(feed, _policy("team")) == {
        "claude-pool-on-both",
        "claude-pool-team-only",
    }


def test_no_edition_filter_admits_the_union_of_every_edition():
    """Ignoring the field keeps the old behaviour: the union."""
    feed = _feed([BOTH, TEAM_ONLY])
    assert _admitted(feed, _policy(None)) == {
        "claude-pool-on-both",
        "claude-pool-team-only",
    }


def test_an_offering_on_no_roster_is_excluded_by_any_edition_filter():
    """The Feed contract: an Offering no plan covers publishes no
    `plan_editions`, so a `plan_edition` filter excludes it. This is why
    the filter must never be set for a provider mixing subscription and
    pay-as-you-go Offerings.
    """
    feed = _feed([BOTH, NO_ROSTER])
    assert _admitted(feed, _policy("personal")) == {"claude-pool-on-both"}
    assert _admitted(feed, _policy(None)) == {
        "claude-pool-on-both",
        "claude-pool-pay-as-you-go",
    }


def test_an_unknown_edition_admits_nothing_rather_than_everything():
    """A typo must fail closed, not silently widen the roster."""
    feed = _feed([BOTH, TEAM_ONLY])
    assert _admitted(feed, _policy("personel")) == set()


@pytest.mark.parametrize("editions", [None, [], "personal", {"personal": True}])
def test_a_malformed_plan_editions_value_never_admits(editions):
    """A Feed field of the wrong shape must not crash or admit."""
    raw = _offering("odd", ["personal"])
    raw["pricing"]["subscription"]["plan_editions"] = editions
    feed = _feed([raw])
    assert _admitted(feed, _policy("personal")) == set()


def test_a_missing_subscription_block_never_admits():
    raw = _offering("no-subscription", ["personal"])
    del raw["pricing"]["subscription"]
    feed = _feed([raw])
    assert _admitted(feed, _policy("personal")) == set()
    assert _admitted(feed, _policy(None)) == {"claude-pool-no-subscription"}


def test_policy_parses_and_defaults_the_plan_edition():
    assert _policy("personal").providers["opencode-go"].plan_edition == "personal"
    assert _policy(None).providers["opencode-go"].plan_edition is None


def test_an_empty_plan_edition_is_rejected():
    with pytest.raises(PolicyError, match="plan_edition"):
        _policy("")


def test_a_non_string_plan_edition_is_rejected():
    with pytest.raises(PolicyError, match="plan_edition"):
        _policy(["personal"])


@pytest.mark.skipif(
    not OPERATOR_POLICY_PATH.exists() or not OPERATOR_FEED_PATH.exists(),
    reason="the operator's instance directory is not on this machine",
)
def test_the_operator_holds_the_personal_edition_and_gets_its_roster():
    """Pin the operator's own roster, so an edition change is visible.

    Read against empty Health State on purpose: every Personal Offering
    is quota-exhausted until 2026-07-29, and an Excluded Offering would
    hide what this filter admits.
    """
    policy = load_policy(OPERATOR_POLICY_PATH)
    feed = load_feed(OPERATOR_FEED_PATH)
    assert policy.providers["qwencloud-token-plan"].plan_edition == "personal"

    result = plan(feed=feed, policy=policy, health={}, now=NOW)
    qwen = {a for a in result.report.aliases.values() if a.startswith("claude-qwen-token-plan-")}
    assert qwen == {
        "claude-qwen-token-plan-deepseek-v4-pro",
        "claude-qwen-token-plan-glm-5.2",
        "claude-qwen-token-plan-qwen3.6-flash",
        "claude-qwen-token-plan-qwen3.7-max",
        "claude-qwen-token-plan-qwen3.7-plus",
        "claude-qwen-token-plan-qwen3.8-max-preview",
    }


@pytest.mark.skipif(
    not OPERATOR_POLICY_PATH.exists() or not OPERATOR_FEED_PATH.exists(),
    reason="the operator's instance directory is not on this machine",
)
def test_no_qwen_offering_is_withheld_by_hand_any_more():
    """The nine Team-only ids were Withheld lines until 2026-07-26.

    The filter reads the Feed's roster instead, so a hand-written line
    would now be redundant — and would go stale silently when the
    provider moves a model between editions.
    """
    policy = load_policy(OPERATOR_POLICY_PATH)
    assert not [k for k in policy.withheld if k.startswith("qwencloud-token-plan:")]
