"""An Excluded Declared Offering leaves the Generated Config.

CONTEXT.md, "Passthrough Auth": a quota or authentication failure on a
Passthrough Auth Offering never Excludes it, but "other failure kinds
still Exclude it". `reduce` already records that exclusion; before this
rule, `plan` passed every Declared Offering through whatever Health
State said, so "Excluded" had no effect on a Declared Offering at all —
story 19's removal never happened for the operator's ten direct vendor
entries, and the exclusion appeared in no report section either.
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


def test_an_excluded_declared_offering_leaves_the_generated_config_and_is_reported():
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
    assert names == ["claude-direct-b"]
    assert "claude-direct-a" in result.report.excluded
    assert "claude-direct-a" not in result.report.admitted


def test_a_passthrough_auth_quota_failure_never_removes_the_declared_offering():
    """`reduce` leaves `excluded` False for a quota failure on a
    Passthrough Auth Offering, so the Offering stays offered and the
    failure is reported, exactly as stories 32 and 33 ask."""
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
    success, or a passed reset time — both `reduce`'s job) puts the
    Declared Offering back in the Generated Config with no special
    case."""
    policy = _policy(
        [{"alias": "claude-direct-a", "litellm_params": {"model": "anthropic/a"}}]
    )
    health = {
        "claude-direct-a": OfferingHealth(excluded=False, last_success_at=NOW)
    }

    result = plan(feed=EMPTY_FEED, policy=policy, health=health, now=NOW)

    names = [e["model_name"] for e in result.config["model_list"]]
    assert names == ["claude-direct-a"]
