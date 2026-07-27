"""Tests for `proxy_settings`: the Generated Config is a complete file.

The operator overrode the spec's "Out of Scope" line ("the tool does
not manage other proxy settings"): a Generated Config that needs a
manual merge before the proxy can load it defeats the point of the
tool. `Policy.proxy_settings` carries the settings the Generator does
not derive from the Feed — `general_settings` and `litellm_settings`,
both passed through verbatim — with one exception:
`litellm_settings.custom_provider_map` always stays DERIVED from the
Feed's envelope routing (spec-corrections.md, correction 5).

Assert external behaviour: what `plan` writes into `config`, what
`PolicyError` a bad `proxy_settings` value raises, and what
`validate_config_before_write` refuses. Fixtures are inline Policy
dicts, or the frozen `tests/fixtures/feed-current.json` and the
operator's real Policy and live proxy config, read directly rather
than guessed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from litellm_maintainer.feed import Feed, load_feed
from litellm_maintainer.plan import plan
from litellm_maintainer.policy import PolicyError, load_policy, parse_policy
from litellm_maintainer.safety import validate_config_before_write
from litellm_maintainer.translate import ENVELOPE_HANDLER_PREFIX

FIXTURES = Path(__file__).parent / "fixtures"
FEED_CURRENT_PATH = FIXTURES / "feed-current.json"
OPERATOR_POLICY_PATH = Path("/Users/hentol/.config/litellm-maintainer/policy.yaml")
LIVE_PROXY_CONFIG_PATH = Path("/Users/hentol/.config/litellm/config.yaml")

NOW = datetime(2026, 7, 26, tzinfo=timezone.utc)


def _minimal_policy_dict(**overrides) -> dict:
    """A minimal, valid Policy dict naming every REQUIRED top-level key.

    `proxy_settings` is deliberately absent unless a test overrides it —
    that absence is exactly what the no-regression test checks.
    """
    base = {
        "providers": {},
        "quality": {"minimum_coding_score": 20},
        "approved_candidates": [],
        "naming": {"alias_prefix": "claude-", "provider_labels": {}, "alias_overrides": {}},
        "withheld": {},
        "declared": [],
        "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
        "schedule": {
            "enabled": True,
            "interval_minutes": 60,
            "require_proxy": True,
            "maximum_staleness_hours": 24,
        },
        "safety": {"maximum_removal_share": 0.25, "snapshot_count": 5},
    }
    base.update(overrides)
    return base


def _empty_feed() -> Feed:
    return Feed(schema_version="test", offerings=(), providers={}, profiles=(), notices=(), raw={})


@pytest.fixture(scope="module")
def feed_current() -> Feed:
    return load_feed(FEED_CURRENT_PATH)


@pytest.fixture(scope="module")
def operator_policy():
    return load_policy(OPERATOR_POLICY_PATH)


# --- No-regression: the baseline every other test measures against -----


def test_a_policy_with_no_proxy_settings_produces_todays_output_exactly():
    """A Policy naming no `proxy_settings` key produces exactly the
    Generated Config this tool always produced: `model_list` only.

    Mutation-tested: hard-coding `config["general_settings"] = {}` (an
    empty mapping, always present) makes this test fail, because
    `set(result.config)` would then be `{"model_list",
    "general_settings"}`, not `{"model_list"}`.
    """
    policy = parse_policy(_minimal_policy_dict())
    result = plan(feed=_empty_feed(), policy=policy, health={}, now=NOW)
    assert result.refusal is None
    assert set(result.config) == {"model_list"}
    assert result.config["model_list"] == []


# --- general_settings and litellm_settings pass through verbatim -------


def test_general_settings_is_emitted_verbatim():
    """Mutation-tested: dropping the `general_settings` merge in `plan`
    makes this test fail (`KeyError` on `result.config["general_settings"]`).
    """
    policy = parse_policy(
        _minimal_policy_dict(
            proxy_settings={
                "general_settings": {"forward_client_headers_to_llm_api": True}
            }
        )
    )
    result = plan(feed=_empty_feed(), policy=policy, health={}, now=NOW)
    assert result.config["general_settings"] == {"forward_client_headers_to_llm_api": True}


def test_litellm_settings_is_emitted_verbatim_alongside_the_derived_custom_provider_map(
    feed_current, operator_policy
):
    """The operator's real Policy carries `proxy_settings.litellm_settings`
    (`master_key`, `drop_params`,
    `use_chat_completions_url_for_anthropic_messages`) and its `cline`
    provider rule admits envelope-routed Offerings, so both the
    hand-written keys and the derived `custom_provider_map` must appear
    together.

    Mutation-tested: merging Policy's `litellm_settings` only when no
    envelope handler is used (an `elif` instead of always merging first)
    makes the `master_key` assertion fail here, though it would pass the
    no-regression test above on its own.
    """
    result = plan(feed=feed_current, policy=operator_policy, health={}, now=NOW)
    litellm_settings = result.config["litellm_settings"]
    assert litellm_settings["master_key"] == "os.environ/LITELLM_MASTER_KEY"
    assert litellm_settings["drop_params"] is True
    assert litellm_settings["use_chat_completions_url_for_anthropic_messages"] is True
    # `callbacks` carries the Observation Journal hook and nothing else.
    # The main proxy is the only one that records: a worker knows an
    # Offering under a name that carries no seat identity (ADR 0008,
    # CONTEXT.md "Observation Journal").
    assert litellm_settings["callbacks"] == [
        "journal_failure_callback.observation_journal_callback"
    ]
    # `chatgpt_role_fix` stays absent: it served the direct `chatgpt/`
    # entries, retired on 2026-07-26. Only the worker proxies load it
    # now, from their own config.
    assert not any("chatgpt_role_fix" in c for c in litellm_settings["callbacks"])
    assert litellm_settings["custom_provider_map"] == [
        {"provider": ENVELOPE_HANDLER_PREFIX, "custom_handler": "cline_provider.cline_llm"}
    ]
    assert result.report.custom_provider_map_conflict is None


# --- custom_provider_map stays derived; a hand-written one conflicts ---


def test_policy_setting_custom_provider_map_reports_a_conflict_and_the_derived_map_wins(
    feed_current,
):
    """Correction 5: envelope routing is data-driven. A Policy that also
    writes `litellm_settings.custom_provider_map` by hand must not win a
    silent pick — `plan` reports the conflict and the derived map wins
    regardless.

    Mutation-tested: letting Policy's `custom_provider_map` value
    survive the merge (skipping the `del`) makes the final assertion
    fail, since the config would then hold the hand-written,
    wrong-looking entry instead of the derived one. Returning `None` for
    the conflict unconditionally makes the first assertion fail.
    """
    raw = yaml.safe_load(OPERATOR_POLICY_PATH.read_text())
    raw["proxy_settings"]["litellm_settings"]["custom_provider_map"] = [
        {"provider": "should-be-ignored", "custom_handler": "not.the.real.handler"}
    ]
    policy = parse_policy(raw)
    result = plan(feed=feed_current, policy=policy, health={}, now=NOW)

    assert result.report.custom_provider_map_conflict is not None
    assert "derived" in result.report.custom_provider_map_conflict
    assert result.config["litellm_settings"]["custom_provider_map"] == [
        {"provider": ENVELOPE_HANDLER_PREFIX, "custom_handler": "cline_provider.cline_llm"}
    ]


def test_a_hand_written_custom_provider_map_is_dropped_when_no_offering_needs_the_handler():
    """The derived map can also be ABSENT — no admitted Offering uses the
    envelope handler. The hand-written map must still not survive: it is
    ignored, not merged in as a fallback.

    Mutation-tested: skipping the `del` on Policy's `custom_provider_map`
    makes this test fail, because the hand-written value would then
    reach the config with nothing to overwrite it (there is no envelope
    Offering here to trigger the derived assignment that would mask the
    bug, unlike the mixed scenario above).
    """
    policy = parse_policy(
        _minimal_policy_dict(
            proxy_settings={
                "litellm_settings": {
                    "custom_provider_map": [
                        {"provider": "should-be-ignored", "custom_handler": "not.the.real.handler"}
                    ]
                }
            }
        )
    )
    result = plan(feed=_empty_feed(), policy=policy, health={}, now=NOW)
    assert result.report.custom_provider_map_conflict is not None
    assert "custom_provider_map" not in result.config.get("litellm_settings", {})


# --- Invalid proxy_settings: the message names the offending key -------


def test_a_non_mapping_general_settings_raises_policyerror_naming_the_key():
    """Mutation-tested: removing the `_require_dict` call on
    `general_settings` makes this test fail (no `PolicyError` raised at
    all — a list would reach `dict(...)` and blow up with a different,
    unmatched exception, or worse, silently misbehave).
    """
    raw = _minimal_policy_dict(
        proxy_settings={"general_settings": ["not", "a", "mapping"]}
    )
    with pytest.raises(PolicyError, match="proxy_settings.general_settings"):
        parse_policy(raw)


def test_a_non_mapping_litellm_settings_raises_policyerror_naming_the_key():
    raw = _minimal_policy_dict(proxy_settings={"litellm_settings": "not-a-mapping"})
    with pytest.raises(PolicyError, match="proxy_settings.litellm_settings"):
        parse_policy(raw)


# --- Safety: an unresolvable master_key must refuse the write -----------


def test_an_unresolvable_master_key_fails_validation_before_the_write():
    """A Generated Config whose `master_key` does not resolve would lock
    the operator out of their own proxy.

    Mutation-tested: checking `litellm_params` credential variables but
    never `litellm_settings.master_key` makes this test fail (empty
    tuple returned).
    """
    config = {
        "model_list": [],
        "litellm_settings": {"master_key": "os.environ/MISSING_MASTER_KEY"},
    }
    problems = validate_config_before_write(config, credential_resolver=lambda name: None)
    assert any("MISSING_MASTER_KEY" in p and "master_key" in p for p in problems)


def test_a_resolvable_master_key_passes_validation():
    config = {
        "model_list": [],
        "litellm_settings": {"master_key": "os.environ/PRESENT_MASTER_KEY"},
    }
    resolver = {"PRESENT_MASTER_KEY": "fake-value-not-a-real-credential"}.get
    problems = validate_config_before_write(config, credential_resolver=resolver)
    assert problems == ()


# --- Acceptance: the operator's real Policy produces a deployable file -


def test_the_generated_configs_top_level_keys_match_the_live_configs_set(
    feed_current, operator_policy
):
    """Measured, not guessed: read the real live proxy config at
    `~/.config/litellm/config.yaml` and compare its top-level key set to
    what `plan` produces for the operator's real Policy against the
    current Feed snapshot. A Generated Config missing a key here is one
    the operator would have had to merge in by hand before deploying.
    """
    result = plan(feed=feed_current, policy=operator_policy, health={}, now=NOW)
    live_config = yaml.safe_load(LIVE_PROXY_CONFIG_PATH.read_text())
    assert set(result.config) == set(live_config)
