"""Tests for `litellm_maintainer.policy`.

Assert external behaviour: what `load_policy` reports, and what makes it
raise `PolicyError` naming the offending key. Fixtures are inline
dictionaries or files under `tmp_path`, never under `tests/fixtures/`,
which is frozen.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from litellm_maintainer import cli, paths
from litellm_maintainer.policy import (
    PolicyError,
    alias_for,
    derive_alias,
    describe_policy,
    load_policy,
    parse_policy,
)

# `HY3_PREVIEW_ALIAS` is a newly-admitted Discovered Offering, unrelated
# to the ChatGPT-seat change — see its docstring in test_translate.py.
from test_translate import HY3_PREVIEW_ALIAS, personal_plan_denied_aliases

EXAMPLE_POLICY = Path(__file__).parent.parent / "policy.example.yaml"
FIXTURES = Path(__file__).parent / "fixtures"
EXPECTED_CONFIG = FIXTURES / "expected-config.yaml"


def _valid_policy_dict() -> dict:
    """A minimal, valid Policy covering every top-level key."""
    return {
        "providers": {
            "acme": {"mode": "all", "pricing": ["free"]},
            "widgets": {"mode": "named", "models": ["widgets:big-coder"]},
        },
        "quality": {"minimum_coding_score": 30},
        "approved_candidates": ["widgets:small-coder"],
        "naming": {
            "alias_prefix": "claude-",
            "provider_labels": {"acme": "acme", "widgets": "widget-co"},
            "alias_overrides": {"widgets:big-coder": "claude-widget-big"},
        },
        "withheld": {"acme:flaky-model": "quota unclear, re-enable once confirmed"},
        "declared": [
            {
                "alias": "claude-direct-model",
                "litellm_params": {"model": "anthropic/direct-model"},
            },
            {
                "alias": "claude-caller-model",
                "passthrough_auth": True,
                "litellm_params": {"model": "chatgpt/caller-model"},
            },
        ],
        "pacing": {
            "default": {"concurrency": 2, "minimum_interval_seconds": 5},
            "acme": {"concurrency": 1, "minimum_interval_seconds": 10},
        },
        "schedule": {
            "enabled": True,
            "interval_minutes": 60,
            "require_proxy": True,
            "maximum_staleness_hours": 24,
        },
        "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
    }


def test_a_valid_policy_loads_and_reports_providers_and_modes():
    policy = parse_policy(_valid_policy_dict())
    assert policy.providers["acme"].mode == "all"
    assert policy.providers["acme"].pricing == ("free",)
    assert policy.providers["widgets"].mode == "named"
    assert policy.providers["widgets"].models == ("widgets:big-coder",)


def test_a_valid_policy_reports_the_quality_threshold_and_approved_candidates():
    policy = parse_policy(_valid_policy_dict())
    assert policy.quality.minimum_coding_score == 30
    assert policy.approved_candidates == ("widgets:small-coder",)


def test_a_valid_policy_reports_naming_rules():
    policy = parse_policy(_valid_policy_dict())
    assert policy.naming.alias_prefix == "claude-"
    assert policy.naming.provider_labels["acme"] == "acme"
    assert policy.naming.alias_overrides["widgets:big-coder"] == "claude-widget-big"


def test_a_valid_policy_reports_withheld_entries_with_reasons():
    policy = parse_policy(_valid_policy_dict())
    assert policy.withheld["acme:flaky-model"] == "quota unclear, re-enable once confirmed"


def test_a_valid_policy_reports_declared_offerings():
    policy = parse_policy(_valid_policy_dict())
    assert len(policy.declared) == 2
    direct, caller = policy.declared
    assert direct.alias == "claude-direct-model"
    assert direct.passthrough_auth is False
    assert caller.passthrough_auth is True


def test_a_valid_policy_reports_the_schedule():
    policy = parse_policy(_valid_policy_dict())
    assert policy.schedule.enabled is True
    assert policy.schedule.interval_minutes == 60
    assert policy.schedule.require_proxy is True
    assert policy.schedule.maximum_staleness_hours == 24


def test_describe_policy_names_every_section():
    policy = parse_policy(_valid_policy_dict())
    text = describe_policy(policy)
    assert "Providers:" in text
    assert "acme" in text
    assert "Quality: minimum_coding_score=30" in text
    assert "Naming:" in text
    assert "Withheld: 1" in text
    assert "Declared Offerings: 2" in text
    assert "Schedule:" in text


def test_load_policy_reads_a_file_from_disk(tmp_path):
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(yaml.safe_dump(_valid_policy_dict()))
    policy = load_policy(policy_file)
    assert policy.providers["acme"].mode == "all"


# --- Invalid Policy: the message names the offending key ---


def test_missing_top_level_key_names_that_key():
    raw = _valid_policy_dict()
    del raw["schedule"]
    with pytest.raises(PolicyError, match="schedule"):
        parse_policy(raw)


def test_unknown_top_level_key_is_rejected_naming_that_key():
    raw = _valid_policy_dict()
    raw["not_a_real_section"] = {}
    with pytest.raises(PolicyError, match="not_a_real_section"):
        parse_policy(raw)


def test_a_provider_rule_with_mode_named_and_no_models_is_rejected():
    raw = _valid_policy_dict()
    raw["providers"]["widgets"] = {"mode": "named"}
    with pytest.raises(PolicyError, match="widgets.models"):
        parse_policy(raw)


def test_an_unrecognised_provider_mode_names_that_key():
    raw = _valid_policy_dict()
    raw["providers"]["acme"] = {"mode": "everything"}
    with pytest.raises(PolicyError, match="acme.mode"):
        parse_policy(raw)


def test_an_unrecognised_pricing_kind_is_rejected():
    raw = _valid_policy_dict()
    raw["providers"]["acme"] = {"mode": "all", "pricing": ["cheap"]}
    with pytest.raises(PolicyError, match="pricing"):
        parse_policy(raw)


def test_a_non_numeric_quality_threshold_names_that_key():
    raw = _valid_policy_dict()
    raw["quality"] = {"minimum_coding_score": "very good"}
    with pytest.raises(PolicyError, match="quality.minimum_coding_score"):
        parse_policy(raw)


def test_a_declared_offering_without_litellm_params_names_that_key():
    raw = _valid_policy_dict()
    raw["declared"] = [{"alias": "claude-broken"}]
    with pytest.raises(PolicyError, match=r"declared\[0\].litellm_params"):
        parse_policy(raw)


def test_a_declared_offering_without_a_model_names_that_key():
    raw = _valid_policy_dict()
    raw["declared"] = [{"alias": "claude-broken", "litellm_params": {}}]
    with pytest.raises(PolicyError, match=r"declared\[0\].litellm_params.model"):
        parse_policy(raw)


def test_a_naming_section_missing_alias_prefix_names_that_key():
    raw = _valid_policy_dict()
    del raw["naming"]["alias_prefix"]
    with pytest.raises(PolicyError, match="naming.alias_prefix"):
        parse_policy(raw)


def test_a_pacing_table_without_a_default_entry_is_rejected():
    raw = _valid_policy_dict()
    del raw["pacing"]["default"]
    with pytest.raises(PolicyError, match="pacing.default"):
        parse_policy(raw)


def test_a_negative_maximum_removal_share_names_that_key():
    raw = _valid_policy_dict()
    raw["safety"]["maximum_removal_share"] = -0.1
    with pytest.raises(PolicyError, match="safety.maximum_removal_share"):
        parse_policy(raw)


def test_a_schedule_missing_interval_minutes_names_that_key():
    raw = _valid_policy_dict()
    del raw["schedule"]["interval_minutes"]
    with pytest.raises(PolicyError, match="schedule.interval_minutes"):
        parse_policy(raw)


def test_a_non_mapping_policy_is_rejected():
    with pytest.raises(PolicyError):
        parse_policy(["not", "a", "mapping"])


def test_an_empty_policy_file_is_rejected(tmp_path):
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text("")
    with pytest.raises(PolicyError):
        load_policy(policy_file)


# --- Declared Offerings: credentials from the caller ---


def test_a_declared_offering_may_state_its_credentials_come_from_the_caller():
    raw = _valid_policy_dict()
    policy = parse_policy(raw)
    caller_offering = policy.declared[1]
    assert caller_offering.passthrough_auth is True
    assert "api_key" not in caller_offering.litellm_params


def test_a_declared_offering_may_state_which_offering_it_supersedes():
    raw = _valid_policy_dict()
    raw["declared"].append(
        {
            "alias": "claude-superseding-model",
            "supersedes": "widgets:big-coder",
            "litellm_params": {"model": "anthropic/superseding-model"},
        }
    )
    policy = parse_policy(raw)
    superseding = policy.declared[-1]
    assert superseding.supersedes == "widgets:big-coder"


# --- The published example guards against drift ---


def test_the_published_example_policy_loads_and_validates():
    policy = load_policy(EXAMPLE_POLICY)
    assert policy.providers
    assert policy.declared
    # The ticket requires the example to show a Declared Offering whose
    # credentials come from the caller.
    assert any(d.passthrough_auth for d in policy.declared)


def test_the_published_example_policy_holds_no_real_alias_or_host():
    text = EXAMPLE_POLICY.read_text()
    real_aliases = _expected_aliases()
    for alias in real_aliases:
        assert alias not in text, f"the example names a real Alias: {alias}"
    for host in ("api.cline.bot", "opencode.ai", "openrouter.ai", "ai.gdm.se"):
        assert host not in text, f"the example names a real host: {host}"


# --- The validate command ---


def test_the_validate_command_exits_zero_on_a_valid_policy(tmp_path, capsys):
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(yaml.safe_dump(_valid_policy_dict()))
    assert cli.main(["validate", "--policy", str(policy_file)]) == 0
    assert "Providers:" in capsys.readouterr().out


def test_the_validate_command_exits_non_zero_and_names_the_offending_key(tmp_path, capsys):
    raw = _valid_policy_dict()
    del raw["schedule"]["interval_minutes"]
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(yaml.safe_dump(raw))
    assert cli.main(["validate", "--policy", str(policy_file)]) == 1
    assert "schedule.interval_minutes" in capsys.readouterr().err


def test_the_validate_command_redacts_a_credential_from_its_output(tmp_path, capsys):
    env_file = tmp_path / ".env.local"
    env_file.write_text("EXAMPLE_API_KEY=super-secret-credential-value\n")
    raw = _valid_policy_dict()
    raw["declared"].append(
        {
            "alias": "claude-leaky",
            "litellm_params": {
                "model": "anthropic/leaky",
                "api_base": "https://super-secret-credential-value.example",
            },
        }
    )
    policy_file = tmp_path / "policy.yaml"
    policy_file.write_text(yaml.safe_dump(raw))
    assert cli.main(["validate", "--policy", str(policy_file), "--env", str(env_file)]) == 0
    out = capsys.readouterr().out
    assert "super-secret-credential-value" not in out


# --- Alias derivation ---
#
# Ticket 08 owns `naming.py` and will import or move `derive_alias`.
# These tests state the rule the operator's Aliases actually follow.


@pytest.mark.parametrize(
    "label,provider_model_id,expected",
    [
        # A plain identifier passes through under the label.
        ("opencode-go", "glm-5.2", "claude-opencode-go-glm-5.2"),
        # A vendor path segment disappears.
        ("groq", "openai/gpt-oss-120b", "claude-groq-gpt-oss-120b"),
        # A `:free` marker becomes an ordinary trailing token.
        (
            "openrouter",
            "google/gemma-4-31b-it:free",
            "claude-openrouter-gemma-4-31b-it-free",
        ),
        # A trailing token the label already holds is not repeated.
        (
            "cline-free",
            "google/gemma-4-31b-it:free",
            "claude-cline-free-gemma-4-31b-it",
        ),
        # A leading token the label already holds is not repeated.
        ("gemini", "gemini-3.5-flash", "claude-gemini-3.5-flash"),
        # A token in the middle survives, even when the label holds it.
        ("opencode-zen", "deepseek-v4-flash-free", "claude-opencode-zen-deepseek-v4-flash-free"),
        # Upper case is folded.
        ("qwen-token-plan", "MiniMax-M2.5", "claude-qwen-token-plan-minimax-m2.5"),
    ],
)
def test_derive_alias_follows_the_recorded_rule(label, provider_model_id, expected):
    assert derive_alias(label, provider_model_id, "claude-") == expected


def _expected_aliases() -> list[str]:
    """Return the 78 active Aliases in the frozen expected config."""
    prefix = "  - model_name:"
    return [
        line.split(":", 1)[1].strip()
        for line in EXPECTED_CONFIG.read_text().splitlines()
        if line.startswith(prefix)
    ]


def test_the_frozen_expected_config_holds_seventy_eight_aliases():
    assert len(_expected_aliases()) == 78


def _operator_policy():
    path = paths.policy_path()
    if not path.exists():
        pytest.skip(f"the operator's Policy is not present at {path}")
    return load_policy(path)


def test_the_operator_policy_reproduces_every_alias_in_the_frozen_config():
    """Derivation plus overrides reproduces all 78 recorded Aliases.

    Selection does not exist yet, so this test does not run the
    Generator. It applies the baseline capability filter, the
    per-provider rule, the pricing filter and the quality gate here,
    then names every surviving Offering with `alias_for`.
    """
    policy = _operator_policy()
    with open(FIXTURES / "feed-audited.json") as f:
        feed = json.load(f)

    excluded_capabilities = {
        "image_generation",
        "embeddings",
        "text_to_speech",
        "speech_to_text",
        "reranking",
    }
    admitted = []
    for offering in feed["models"]:
        provider_id = offering["provider"]["id"]
        rule = policy.providers.get(provider_id)
        if rule is None:
            continue
        capabilities = set(offering["capabilities"])
        if "tool_use" not in capabilities or capabilities & excluded_capabilities:
            continue
        if offering["policy"]["visibility"] != "listed":
            continue
        if rule.mode == "named" and offering["id"] not in (rule.models or ()):
            continue
        if rule.pricing and offering["pricing"]["kind"] not in rule.pricing:
            continue
        if offering["id"] in policy.withheld:
            continue
        # The same subscription-edition filter `plan` applies: only an
        # Offering whose Feed roster names the operator's edition.
        if rule.plan_edition is not None:
            subscription = (offering.get("pricing") or {}).get("subscription") or {}
            editions = subscription.get("plan_editions")
            if not isinstance(editions, list) or rule.plan_edition not in editions:
                continue
        score = offering["quality"]["coding_score"]
        if score is None:
            if offering["id"] not in policy.approved_candidates:
                continue
        elif score < policy.quality.minimum_coding_score:
            continue
        admitted.append(offering["id"])

    declared_aliases = {d.alias for d in policy.declared}
    # The 6 retired direct `chatgpt/` Aliases are in the frozen config
    # and are no longer declared. They were never Discovered either: the
    # Feed publishes no ChatGPT provider, so no rule can derive them.
    # Drop them here or they read as Aliases the derivation should have
    # produced.
    retired_direct_chatgpt = {
        "claude-gpt-5.6-sol",
        "claude-gpt-5.6-terra",
        "claude-gpt-5.6-luna",
        "claude-gpt-5.5",
        "claude-gpt-5.4",
        "claude-gpt-5.4-mini",
    }
    discovered_aliases = [
        a
        for a in _expected_aliases()
        if a not in declared_aliases and a not in retired_direct_chatgpt
    ]
    # 4 direct-Claude entries, 3 Client-Facing Variants of them (added
    # 2026-07-26: one `[1m]` Alias per 1M model, so a client budgets the
    # whole window — CONTEXT.md, "Client-Facing Variant"), plus the 12
    # ChatGPT worker seats. The 6 direct `chatgpt/` entries were retired
    # on 2026-07-26; they are in the frozen config, so
    # `discovered_aliases` (drawn from the frozen file) must not expect
    # them either.
    assert len(declared_aliases) == 4 + 3 + 12
    assert len(discovered_aliases) == 68

    # The operator renamed seven Aliases on 2026-07-25, to get one
    # consistent rule instead of the hand-built names. So the produced
    # set differs from the frozen config by exactly these seven.
    # `policy.yaml` records the same list for the operator's clients.
    RENAMED = {
        "claude-qwen-token-plan-3.8-max-preview":
            "claude-qwen-token-plan-qwen3.8-max-preview",
        "claude-qwen-token-plan-3.6-flash":
            "claude-qwen-token-plan-qwen3.6-flash",
        "claude-openrouter-nemotron-3-ultra-free":
            "claude-openrouter-nemotron-3-ultra-550b-a55b-free",
        "claude-openrouter-nemotron-3-super-120b-free":
            "claude-openrouter-nemotron-3-super-120b-a12b-free",
        "claude-cline-free-nemotron-3-ultra":
            "claude-cline-free-nemotron-3-ultra-550b-a55b",
        "claude-cline-free-nemotron-3-super-120b":
            "claude-cline-free-nemotron-3-super-120b-a12b",
        "claude-openrouter-free-router": "claude-openrouter-free",
    }
    expected = {RENAMED.get(alias, alias) for alias in discovered_aliases}

    # The nine Aliases of the Qwen Token Plan Offerings the operator's
    # Policy now Withholds (personal-tier denial, HTTP 403 on every
    # call). This loop's own `if offering["id"] in policy.withheld:
    # continue` line already drops them from `admitted`; remove them
    # from `expected` too, by name, so the guarantee stays closed rather
    # than an open-ended "and the count moved".
    # The nine Team-only Offerings the Personal edition denies. They are
    # in the frozen config (they once worked) and the edition filter now
    # excludes them, so the expected set must not hold them.
    denied_aliases = personal_plan_denied_aliases(policy)
    assert len(denied_aliases) == 9
    assert denied_aliases <= expected
    expected = expected - denied_aliases

    # `opencode-go:hy3-preview` is newly admitted here too: it scores
    # 58.8, above the quality threshold, `policy.yaml` sets
    # `opencode-go: mode: all`, and this loop applies the same gates
    # `plan` does. It is unrelated to the ChatGPT-seat change — see its
    # docstring (`HY3_PREVIEW_ALIAS`) in test_translate.py.
    assert "opencode-go:hy3-preview" in admitted
    # 69 - 9 denied by the Personal edition, + 1 re-admitted when its
    # stale Withheld line was pruned (openrouter:qwen/qwen3-coder:free,
    # dropped by the Feed on 2026-07-26 but still in this fixture).
    assert len(admitted) == 69 - 9 + 1
    expected = expected | {HY3_PREVIEW_ALIAS, "claude-openrouter-qwen3-coder-free"}

    produced = {alias_for(policy, offering_id) for offering_id in admitted}
    assert produced == expected
    assert denied_aliases.isdisjoint(produced), (
        "a Withheld Offering must never be admitted"
    )

    # Every Alias derives. The override map is empty on purpose.
    assert policy.naming.alias_overrides == {}
    assert len(admitted) == 69 - 9 + 1
    assert len(produced) == 69 - 9 + 1  # no two Offerings share an Alias
