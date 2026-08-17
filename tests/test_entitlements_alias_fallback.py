"""An unavailable Offering reports its Alias, on the shape `plan` really returns.

`PlanReport.aliases` maps ADMITTED Offerings only. An Excluded Offering is
absent from that map by construction, so reading the map alone reported no
Alias for exactly the Offerings the Entitlement view exists to explain. On
the operator's own instance that was all seven of them.

An existing unit test appeared to cover this, but it hand-built a report
carrying an Alias for an Excluded Offering — an input `plan` never
produces. So the test passed while the real command printed nothing.

These tests therefore drive the real `plan`, and assert on the report
shape it actually returns. A test whose fixture cannot occur in
production proves nothing about production.
"""

from __future__ import annotations

from datetime import datetime, timezone

from litellm_maintainer import entitlements
from litellm_maintainer.feed import parse_feed
from litellm_maintainer.plan import plan
from litellm_maintainer.policy import parse_policy
from litellm_maintainer.reduce import OfferingHealth

NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)

# `openrouter` is used because Selection only reaches a provider that has
# a registered translation rule. A fictional provider id contributes
# nothing, and the view would be empty for a reason unrelated to Aliases.
OFFERING_ID = "openrouter:vendor/coder-large"


def _feed():
    return parse_feed(
        {
            "schema_version": "1.0.0",
            "feed": {"generated_at": "2026-07-26T11:00:00Z"},
            # The provider record needs its base URL and authentication, or
            # translation cannot build litellm_params and Selection admits
            # nothing. A fixture missing them tests an empty view by mistake.
            "providers": [
                {
                    "id": "openrouter",
                    "name": "OpenRouter",
                    "default_base_url": "https://openrouter.ai/api/v1",
                    "authentication": {
                        "type": "api_key",
                        "credential_hint": "OPENROUTER_API_KEY",
                    },
                }
            ],
            "models": [
                {
                    "id": OFFERING_ID,
                    "provider": {"id": "openrouter"},
                    "provider_model_id": "vendor/coder-large",
                    "canonical_model": {"id": "vendor/coder-large"},
                    "capabilities": ["chat", "coding", "tool_use"],
                    "pricing": {"kind": "free"},
                    "availability": {"status": "available"},
                    "quality": {"coding_score": 60.0},
                    "policy": {"visibility": "listed"},
                    "endpoint": {"protocol": "openai_chat_completions"},
                },
                {
                    "id": "openrouter:vendor/other",
                    "provider": {"id": "openrouter"},
                    "provider_model_id": "vendor/other",
                    "canonical_model": {"id": "vendor/other"},
                    "capabilities": ["chat", "coding", "tool_use"],
                    "pricing": {"kind": "free"},
                    "availability": {"status": "available"},
                    "quality": {"coding_score": 50.0},
                    "policy": {"visibility": "listed"},
                    "endpoint": {"protocol": "openai_chat_completions"},
                },
            ],
        }
    )


def _policy():
    return parse_policy(
        {
            "providers": {"openrouter": {"mode": "all", "entitlement": "shared_pool"}},
            "quality": {"minimum_coding_score": 10},
            "approved_candidates": [],
            "naming": {
                "alias_prefix": "claude-",
                "provider_labels": {"openrouter": "or"},
                "alias_overrides": {},
            },
            "withheld": {},
            "declared": [],
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


def _derive_through_real_plan(health):
    feed, policy = _feed(), _policy()
    report = plan(feed=feed, policy=policy, health=health, now=NOW).report
    view = entitlements.derive(
        feed=feed, policy=policy, health=health, report=report, now=NOW
    )
    return report, view


EXCLUDED = {
    OFFERING_ID: OfferingHealth(
        excluded=True,
        reason="quota_exhausted",
        bucket="self_healing",
        reset_at=datetime(2026, 7, 29, 21, 45, tzinfo=timezone.utc),
    )
}


GONE = {
    OFFERING_ID: OfferingHealth(
        excluded=True,
        reason="identifier_gone",
        bucket="gone",
    )
}


def test_the_real_report_carries_an_alias_for_an_excluded_offering():
    """An Excluded Offering stays in the Generated Config (ADR 0014), so
    it is admitted and the report derives its Alias. The fallback is not
    what supplies the Alias here."""
    report, _ = _derive_through_real_plan(EXCLUDED)

    assert OFFERING_ID in report.excluded
    assert report.aliases[OFFERING_ID] == "claude-or-coder-large"


def test_the_real_report_carries_no_alias_for_an_unlisted_offering():
    """Pin the premise the fallback rests on. An Unlisted Offering never
    reaches admission, so the report holds no Alias for it. If this ever
    changes, the fallback can go."""
    report, _ = _derive_through_real_plan(GONE)

    assert OFFERING_ID in report.unlisted
    assert OFFERING_ID not in report.aliases


def test_an_unavailable_offering_still_reports_its_alias():
    _, view = _derive_through_real_plan(EXCLUDED)
    unavailable = view.entitlements[0].unavailable_offerings[0]

    assert unavailable.offering_id == OFFERING_ID
    assert unavailable.alias == "claude-or-coder-large"


def test_an_unavailable_offering_reports_its_reason_and_refill():
    _, view = _derive_through_real_plan(EXCLUDED)
    unavailable = view.entitlements[0].unavailable_offerings[0]

    assert unavailable.reason == "quota_exhausted"
    assert unavailable.refills_at == datetime(2026, 7, 29, 21, 45, tzinfo=timezone.utc)


def test_an_alias_override_wins_for_an_unavailable_offering():
    """The derived Alias must not contradict the Alias the proxy serves."""
    feed = _feed()
    raw = {
        "providers": {"openrouter": {"mode": "all"}},
        "quality": {"minimum_coding_score": 10},
        "approved_candidates": [],
        "naming": {
            "alias_prefix": "claude-",
            "provider_labels": {"openrouter": "or"},
            "alias_overrides": {OFFERING_ID: "claude-pinned-name"},
        },
        "withheld": {},
        "declared": [],
        "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
        "schedule": {
            "enabled": True,
            "interval_minutes": 60,
            "require_proxy": False,
            "maximum_staleness_hours": 24,
        },
        "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
    }
    policy = parse_policy(raw)
    report = plan(feed=feed, policy=policy, health=EXCLUDED, now=NOW).report

    view = entitlements.derive(
        feed=feed, policy=policy, health=EXCLUDED, report=report, now=NOW
    )

    assert view.entitlements[0].unavailable_offerings[0].alias == "claude-pinned-name"


def test_the_rendered_text_names_the_alias():
    """The operator reads the text form, so the fix has to reach it."""
    _, view = _derive_through_real_plan(EXCLUDED)

    assert "claude-or-coder-large" in entitlements.render_text(view)


def test_a_provider_that_is_healthy_reports_no_unavailable_offering():
    _, view = _derive_through_real_plan({})

    assert view.entitlements[0].state == "healthy"
    assert view.entitlements[0].unavailable_offerings == ()
