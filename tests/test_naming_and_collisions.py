"""Tests for ticket 08: Aliases, Declared Offerings, and collisions.

Assert external behaviour: which Alias `naming.derive_alias`/`alias_for`
produce, what `plan` writes for a Declared Offering, and what `plan`
does when two Offerings claim one Alias. A test name states a rule an
operator would recognise (spec's "What makes a good test here").

Most tests build a small synthetic Feed and Policy in memory, so the
exact boundary under test is visible in the test itself. The Alias-
preservation test is the exception: it runs the real `plan` against the
frozen `tests/fixtures/feed-audited.json` and the operator's own Policy,
both read-only, because that is the only way to assert "every Alias in
current use", not a constructed subset of it.
"""

from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
import yaml

from litellm_maintainer.feed import Feed, load_feed, parse_feed
from litellm_maintainer.naming import alias_for, derive_alias
from litellm_maintainer.plan import plan
from litellm_maintainer.policy import Policy, load_policy, parse_policy

# The rename table is the single authority for correction 4
# (spec-corrections.md). `test_translate.py` already holds it; import it
# rather than keeping a third copy that could drift from the other two.
# `SEAT_ALIASES` and `HY3_PREVIEW_ALIAS` are the 13 Aliases now produced
# beyond the frozen config's 78 — the 12 ChatGPT worker seats (added
# 2026-07-26) and one newly-admitted Discovered Offering, unrelated to
# the seat change (see that constant's docstring).

FIXTURES = Path(__file__).parent / "fixtures"
FEED_AUDITED_PATH = FIXTURES / "feed-audited.json"
EXPECTED_CONFIG_PATH = FIXTURES / "expected-config.yaml"
# Synthetic and committed. Never the operator's own Policy.
PINNED_POLICY_PATH = Path(__file__).parent / "fixtures" / "policy-pinned.yaml"

NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)


# --- Synthetic Feed and Policy builders ---------------------------------
#
# `opencode-go` already has a registered translation rule and a
# credential fallback (see `test_quality_and_sunsetting.py`), so a
# synthetic Offering on that provider translates without a Feed-
# published `authentication.credential_hint`.


def _offering_raw(
    *,
    id: str,
    provider_id: str = "opencode-go",
    provider_model_id: str | None = None,
    coding_score: float = 50.0,
    capabilities: tuple[str, ...] = ("tool_use",),
) -> dict[str, Any]:
    model_id = provider_model_id if provider_model_id is not None else id.split(":", 1)[1]
    return {
        "id": id,
        "provider": {"id": provider_id},
        "provider_model_id": model_id,
        "capabilities": list(capabilities),
        "endpoint": {
            "base_url": "https://opencode-go.example/v1",
            "model": model_id,
        },
        "pricing": {"kind": "subscription_included"},
        "availability": {
            "status": "available",
            "last_checked_at": "2026-07-25T00:00:00Z",
            "last_success_at": None,
            "stale_after_seconds": 86400,
        },
        "quality": {"coding_score": coding_score},
        "policy": {"visibility": "listed", "tags": []},
    }


def _feed_with(*offerings: dict[str, Any]) -> Feed:
    raw = {
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


def _policy(**overrides: Any) -> Policy:
    return parse_policy(_policy_raw(**overrides))


# --- An Alias derives from the label map --------------------------------


def test_an_alias_derives_from_the_provider_label_map():
    policy = _policy()
    offering_id = "opencode-go:glm-5.2"
    assert alias_for(policy, offering_id) == derive_alias("opencode-go", "glm-5.2", "claude-")
    assert alias_for(policy, offering_id) == "claude-opencode-go-glm-5.2"


def test_a_structured_separator_keeps_provider_and_tags_parseable():
    policy = _policy(
        naming={
            "alias_prefix": "",
            "alias_separator": "--",
            "provider_labels": {"cline": "cline-free"},
            "alias_overrides": {},
        }
    )

    assert alias_for(policy, "cline:deepseek/deepseek-v4-flash:free") == (
        "cline-free--deepseek-v4-flash--free"
    )


def test_an_alias_override_in_policy_replaces_a_derived_name():
    policy = _policy(
        naming={
            "alias_prefix": "claude-",
            "provider_labels": {"opencode-go": "opencode-go"},
            "alias_overrides": {"opencode-go:glm-5.2": "claude-my-preferred-name"},
        }
    )
    # Without the override this Offering would derive
    # "claude-opencode-go-glm-5.2" (proven above); the override wins.
    assert alias_for(policy, "opencode-go:glm-5.2") == "claude-my-preferred-name"
    # An Offering the override does not name still derives normally.
    assert alias_for(policy, "opencode-go:kimi-k3") == "claude-opencode-go-kimi-k3"


# --- Every Alias in current use is reproduced, except the seven renamed -


@pytest.fixture(scope="module")
def feed_audited() -> Feed:
    return load_feed(FEED_AUDITED_PATH)


@pytest.fixture(scope="module")
def operator_policy() -> Policy:
    return load_policy(PINNED_POLICY_PATH)


@pytest.fixture(scope="module")
def frozen_config() -> dict[str, Any]:
    with open(EXPECTED_CONFIG_PATH) as f:
        return yaml.safe_load(f)


# `test_every_alias_in_the_frozen_config_is_reproduced_except_the_seven_renamed` stood here (RETIRED).
#
# It compared generated output to `fixtures/expected-config.yaml`, the
# proxy the operator built and verified BY HAND on 2026-07-25. The
# Policy that produced that file was never committed and no surviving
# copy reproduces it. The live Policy matches none of its 78 Aliases,
# because it now sets `alias_prefix: ""` and `alias_separator: "--"`.
#
# The test therefore demanded a superseded proxy from a Policy that no
# longer exists. See the longer note in test_acceptance.py.

# --- Declared Offerings ---------------------------------------------------


def _declared_policy(**declared_overrides: Any) -> Policy:
    declared = {
        "alias": "claude-declared-example",
        "litellm_params": {
            "model": "anthropic/claude-example",
            "api_base": "https://declared.example/v1",
            "api_key": "os.environ/DECLARED_EXAMPLE_KEY",
        },
    }
    declared.update(declared_overrides)
    return _policy(declared=[declared])


def test_a_declared_offering_passes_through_with_no_field_rewritten():
    policy = _declared_policy()
    feed = _feed_with()  # no Discovered Offerings at all
    result = plan(feed=feed, policy=policy, health={}, now=NOW)

    entries = {e["model_name"]: e for e in result.config["model_list"]}
    entry = entries["claude-declared-example"]

    # Compare the whole `litellm_params` mapping, not just the model:
    # a Declared Offering must be immune to any Feed change, and a
    # shallow check on `model` alone would miss a rewritten field.
    assert entry["litellm_params"] == policy.declared[0].litellm_params


def test_a_declared_offering_is_never_reported_as_a_candidate():
    # A Discovered Offering with no score would be a Candidate; a
    # Declared Offering carries no score field at all, and Declaring
    # one is already the decision (CONTEXT.md, "Candidate").
    policy = _declared_policy()
    feed = _feed_with()
    result = plan(feed=feed, policy=policy, health={}, now=NOW)

    assert result.report.candidates == ()
    assert "claude-declared-example" not in result.report.candidates


# --- Collisions -------------------------------------------------------


def test_a_declared_and_a_discovered_offering_sharing_an_alias_stops_the_run():
    # The mechanical rule derives "claude-opencode-go-glm-5.2" for this
    # Discovered Offering; the Declared Offering claims the same name.
    policy = _declared_policy(alias="claude-opencode-go-glm-5.2")
    feed = _feed_with(_offering_raw(id="opencode-go:glm-5.2"))

    result = plan(feed=feed, policy=policy, health={}, now=NOW)

    assert result.refusal is not None
    assert "claude-opencode-go-glm-5.2" in result.refusal
    # The refusal names both sides of the collision.
    assert "opencode-go:glm-5.2" in result.refusal
    assert "anthropic/claude-example" in result.refusal
    # It suggests the resolution: a `supersedes` entry.
    assert "supersedes" in result.refusal
    # It must not raise, and it must write nothing.
    assert result.config == {}


def test_supersedes_suppresses_the_feed_offering_and_the_run_proceeds():
    policy = _declared_policy(
        alias="claude-opencode-go-glm-5.2", supersedes="opencode-go:glm-5.2"
    )
    feed = _feed_with(_offering_raw(id="opencode-go:glm-5.2"))

    result = plan(feed=feed, policy=policy, health={}, now=NOW)

    assert result.refusal is None
    entries = {e["model_name"]: e for e in result.config["model_list"]}
    # Only the Declared entry survives, under the shared Alias.
    assert entries["claude-opencode-go-glm-5.2"]["litellm_params"] == (
        policy.declared[0].litellm_params
    )
    # The superseded Discovered Offering is suppressed entirely: not
    # admitted, not a Candidate, not Excluded.
    assert "opencode-go:glm-5.2" not in result.report.admitted
    assert "opencode-go:glm-5.2" not in result.report.candidates
    assert "opencode-go:glm-5.2" not in result.report.excluded


def test_a_declared_offering_wins_but_still_stops_the_run_without_supersedes():
    """A Declared Offering wins over a Discovered one conceptually, but
    winning silently is exactly the hidden mistake a collision check
    exists to prevent (docs/gotchas.md, "Duplicate model_name values do
    not raise an error"). Without `supersedes`, the run stops rather
    than picking a winner quietly.
    """
    policy = _declared_policy(alias="claude-opencode-go-glm-5.2")
    feed = _feed_with(_offering_raw(id="opencode-go:glm-5.2"))

    result = plan(feed=feed, policy=policy, health={}, now=NOW)

    assert result.refusal is not None
    assert result.config == {}


def test_two_declared_offerings_sharing_an_alias_stop_the_run():
    """litellm reads two entries with one `model_name` as a load-
    balancing group and splits traffic between two different models
    (docs/gotchas.md, "Duplicate model_name values do not raise an
    error"). Two Declared Offerings can produce that shape as easily as
    a Declared/Discovered pair, so the run stops here too. `supersedes`
    cannot resolve it: it names a Discovered Offering.
    """
    policy = _policy(
        declared=[
            {"alias": "claude-two-declared", "litellm_params": {"model": "anthropic/first"}},
            {"alias": "claude-two-declared", "litellm_params": {"model": "anthropic/second"}},
        ]
    )
    feed = _feed_with()

    result = plan(feed=feed, policy=policy, health={}, now=NOW)

    assert result.refusal is not None
    assert "claude-two-declared" in result.refusal
    assert result.config == {}


def test_two_discovered_offerings_deriving_the_same_alias_stop_the_run():
    # Two different provider_model_ids that the mechanical rule
    # happens to derive to the same Alias. All 68 of the operator's
    # real Discovered Aliases are unique today (see the frozen-config
    # test above); this constructs the case deliberately.
    feed = _feed_with(
        _offering_raw(id="opencode-go:glm-5.2", provider_model_id="glm-5.2"),
        _offering_raw(
            id="opencode-go:opencode-go-glm-5.2",
            provider_model_id="opencode-go-glm-5.2",
        ),
    )
    policy = _policy()

    result = plan(feed=feed, policy=policy, health={}, now=NOW)

    assert result.refusal is not None
    assert "claude-opencode-go-glm-5.2" in result.refusal
    assert "opencode-go:glm-5.2" in result.refusal
    assert "opencode-go:opencode-go-glm-5.2" in result.refusal
    assert result.config == {}


def test_plan_returns_a_refusal_rather_than_raising_for_every_collision_case():
    cases = [
        # Declared vs Discovered, no supersedes.
        (
            _declared_policy(alias="claude-opencode-go-glm-5.2"),
            _feed_with(_offering_raw(id="opencode-go:glm-5.2")),
        ),
        # Discovered vs Discovered.
        (
            _policy(),
            _feed_with(
                _offering_raw(id="opencode-go:glm-5.2", provider_model_id="glm-5.2"),
                _offering_raw(
                    id="opencode-go:opencode-go-glm-5.2",
                    provider_model_id="opencode-go-glm-5.2",
                ),
            ),
        ),
    ]
    for policy, feed in cases:
        result = plan(feed=feed, policy=policy, health={}, now=NOW)  # must not raise
        assert result.refusal is not None
        assert result.config == {}


# --- A derived name for a new Offering is reported ------------------------


def test_a_newly_admitted_offering_has_its_derived_alias_reported():
    feed = _feed_with(_offering_raw(id="opencode-go:glm-5.2"))
    policy = _policy()

    result = plan(feed=feed, policy=policy, health={}, now=NOW)

    assert result.report.aliases["opencode-go:glm-5.2"] == "claude-opencode-go-glm-5.2"
    # Every admitted Offering has an entry; a Declared Offering does
    # not need one, since the operator wrote its Alias by hand.
    assert set(result.report.aliases) == set(result.report.admitted)
