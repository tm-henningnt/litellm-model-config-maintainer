"""What a Declared Offering's Health State does to the Generated Config.

An Excluded Declared Offering stays in the Generated Config and stops
being recommended. Only Gone removes one, because only Gone is terminal.
See ADR 0014 and CONTEXT.md, "Unlisted".

CONTEXT.md, "Passthrough Auth": a quota or authentication failure on a
Passthrough Auth Offering never Excludes it. Other failure kinds still
Exclude it, and an exclusion no longer removes it either.
"""

from __future__ import annotations

from datetime import datetime, timezone

from litellm_maintainer.feed import parse_feed
from litellm_maintainer.plan import plan
from litellm_maintainer.policy import parse_policy
from litellm_maintainer.reduce import OfferingHealth

NOW = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)

EMPTY_FEED = parse_feed({"schema_version": "1", "providers": [], "models": []})


def _policy(declared: list[dict]):
    return parse_policy(
        {
            "providers": {},
            "quality": {"minimum_coding_score": 18},
            "approved_candidates": [],
            "naming": {"alias_prefix": "claude-", "provider_labels": {}, "alias_overrides": {}},
            "withheld": {},
            "declared": declared,
            "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
            "schedule": {
                "enabled": True,
                "interval_minutes": 60,
                "require_proxy": True,
                "maximum_staleness_hours": 24,
            },
            "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
        }
    )


def test_an_excluded_declared_offering_stays_in_the_generated_config():
    """A write to the Generated Config restarts the proxy, so a
    measurement that can change its mind must never drive one. The
    Offering is reported Excluded and a caller can still reach it."""
    policy = _policy(
        [
            {"alias": "claude-direct-a", "litellm_params": {"model": "anthropic/a"}},
            {"alias": "claude-direct-b", "litellm_params": {"model": "anthropic/b"}},
        ]
    )
    health = {
        "claude-direct-a": OfferingHealth(
            excluded=True, reason="gateway_error", bucket="self_healing"
        )
    }

    result = plan(feed=EMPTY_FEED, policy=policy, health=health, now=NOW)

    names = [e["model_name"] for e in result.config["model_list"]]
    assert names == ["claude-direct-a", "claude-direct-b"]
    assert "claude-direct-a" in result.report.excluded
    assert "claude-direct-a" not in result.report.unlisted


def test_a_health_exclusion_changes_no_byte_of_the_generated_config():
    """The reason ADR 0014 exists. An identical config is not written,
    so the proxy does not restart and no session dies for a measurement
    that may reverse itself on the next Probe."""
    policy = _policy(
        [
            {"alias": "claude-direct-a", "litellm_params": {"model": "anthropic/a"}},
            {"alias": "claude-direct-b", "litellm_params": {"model": "anthropic/b"}},
        ]
    )
    healthy = plan(feed=EMPTY_FEED, policy=policy, health={}, now=NOW)
    excluded = plan(
        feed=EMPTY_FEED,
        policy=policy,
        health={
            "claude-direct-a": OfferingHealth(
                excluded=True, reason="gateway_error", bucket="self_healing"
            )
        },
        now=NOW,
    )

    assert excluded.config == healthy.config


def test_a_gone_declared_offering_leaves_the_generated_config():
    """Gone is terminal and never flaps, so it Unlists. The identifier
    no longer answers for this account, and listing it helps nobody."""
    policy = _policy(
        [
            {"alias": "claude-direct-a", "litellm_params": {"model": "anthropic/a"}},
            {"alias": "claude-direct-b", "litellm_params": {"model": "anthropic/b"}},
        ]
    )
    health = {
        "claude-direct-a": OfferingHealth(
            excluded=True, reason="identifier_gone", bucket="gone"
        )
    }

    result = plan(feed=EMPTY_FEED, policy=policy, health=health, now=NOW)

    names = [e["model_name"] for e in result.config["model_list"]]
    assert names == ["claude-direct-b"]
    assert "claude-direct-a" in result.report.unlisted
    assert "claude-direct-a" not in result.report.admitted


def test_a_passthrough_auth_quota_failure_never_removes_the_declared_offering():
    """`reduce` leaves `excluded` False for a quota failure on a
    Passthrough Auth Offering, so the Offering stays offered and the
    failure is reported."""
    policy = _policy(
        [
            {
                "alias": "claude-gpt-direct",
                "litellm_params": {"model": "chatgpt/gpt"},
                "passthrough_auth": True,
            }
        ]
    )
    health = {
        "claude-gpt-direct": OfferingHealth(
            excluded=False, reason="quota_exhausted", bucket="self_healing", failure_count=1
        )
    }

    result = plan(feed=EMPTY_FEED, policy=policy, health=health, now=NOW)

    names = [e["model_name"] for e in result.config["model_list"]]
    assert names == ["claude-gpt-direct"]
    assert result.report.passthrough_auth_failures == ("claude-gpt-direct",)
    assert result.report.excluded == ()


def test_a_declared_offering_recovers_by_the_ordinary_path():
    """A Health State record whose exclusion has cleared (a later Probe
    success, or a passed reset time — both `reduce`'s job) reports the
    Declared Offering as neither Excluded nor Unlisted."""
    policy = _policy(
        [{"alias": "claude-direct-a", "litellm_params": {"model": "anthropic/a"}}]
    )
    health = {
        "claude-direct-a": OfferingHealth(excluded=False, last_success_at=NOW)
    }

    result = plan(feed=EMPTY_FEED, policy=policy, health=health, now=NOW)

    names = [e["model_name"] for e in result.config["model_list"]]
    assert names == ["claude-direct-a"]
    assert result.report.excluded == ()
    assert result.report.unlisted == ()
