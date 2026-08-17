"""Tests for `litellm_maintainer.initialize`.

Each test name states a rule an operator would recognise. The Feed
documents used here are built in memory with `feed.parse_feed`; no test
reads a fixture file or the network.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from litellm_maintainer.feed import parse_feed
from litellm_maintainer.initialize import build_starter_policy, write_starter_policy
from litellm_maintainer.policy import PER_MODEL, parse_policy


def _feed_raw():
    return {
        "schema_version": "1",
        "feed": {"generated_at": "2026-07-25T00:00:00Z"},
        "providers": [
            {
                "id": "openrouter",
                "name": "OpenRouter",
                "default_base_url": "https://openrouter.ai/api/v1",
                "authentication": {"credential_hint": "OPENROUTER_API_KEY"},
            },
            {
                "id": "acct-entitlement-provider",
                "name": "Account Entitlement Provider",
                "default_base_url": None,
                "authentication": {},
            },
        ],
        "models": [
            {
                "id": "openrouter:vendor/coder-large",
                "provider": {"id": "openrouter"},
                "provider_model_id": "vendor/coder-large",
                "capabilities": ["tool_use"],
                "endpoint": {},
                "pricing": {"kind": "free"},
                "availability": {"status": "available"},
                "quality": {"coding_score": 40},
                "policy": {"visibility": "public"},
            },
            {
                "id": "acct-entitlement-provider:model-a",
                "provider": {"id": "acct-entitlement-provider"},
                "provider_model_id": "model-a",
                "capabilities": ["tool_use"],
                "endpoint": {},
                "pricing": {"kind": "unknown"},
                "availability": {"status": "available"},
                "quality": {"coding_score": 30},
                "policy": {"visibility": "public"},
            },
        ],
    }


@pytest.fixture
def feed():
    return parse_feed(_feed_raw())


def test_generated_policy_parses(feed):
    starter = build_starter_policy(feed)
    parse_policy_result = parse_policy(_yaml_load(starter.text))
    assert parse_policy_result.providers


def _yaml_load(text: str):
    import yaml

    return yaml.safe_load(text)


def test_it_names_every_provider_the_feed_publishes(feed):
    starter = build_starter_policy(feed)
    policy = parse_policy(_yaml_load(starter.text))
    assert set(policy.providers) == {"openrouter", "acct-entitlement-provider"}
    assert starter.provider_count == 2


def test_every_provider_entry_has_mode_all_and_entitlement_per_model(feed):
    starter = build_starter_policy(feed)
    policy = parse_policy(_yaml_load(starter.text))
    for rule in policy.providers.values():
        assert rule.mode == "all"
        assert rule.entitlement == PER_MODEL


def test_each_providers_credential_hint_appears_in_a_comment(feed):
    starter = build_starter_policy(feed)
    assert "OPENROUTER_API_KEY" in starter.text


def test_a_provider_with_no_credential_hint_still_gets_a_comment(feed):
    starter = build_starter_policy(feed)
    assert "the Feed states no credential_hint for" in starter.text
    assert "acct-entitlement-provider" in starter.text


def test_no_provider_entry_carries_a_pricing_key(feed):
    starter = build_starter_policy(feed)
    policy = parse_policy(_yaml_load(starter.text))
    for rule in policy.providers.values():
        assert rule.pricing is None


def test_generated_text_holds_no_credential_value(feed):
    starter = build_starter_policy(feed)
    # Any Policy field carrying a credential reference uses the
    # os.environ/NAME form, never a bare value.
    for match in re.finditer(r'os\.environ/(\S+)', starter.text):
        assert re.fullmatch(r"[A-Z0-9_]+\"?", match.group(1))
    # Nothing that looks like a secret literal: no long unbroken run of
    # word characters inside a quoted YAML string value (a real API
    # key or token would appear there, never as a bare identifier or
    # comment word such as `minimum_interval_seconds`).
    suspicious = re.findall(r'"[A-Za-z0-9_]{24,}"', starter.text)
    assert not suspicious


def test_generating_twice_from_the_same_feed_gives_identical_text(feed):
    first = build_starter_policy(feed)
    second = build_starter_policy(feed)
    assert first.text == second.text


def test_write_starter_policy_refuses_an_existing_file_without_force(tmp_path, feed):
    starter = build_starter_policy(feed)
    path = tmp_path / "policy.yaml"
    path.write_text("original contents\n")

    with pytest.raises(FileExistsError):
        write_starter_policy(starter, path)

    assert path.read_text() == "original contents\n"


def test_write_starter_policy_with_force_replaces_the_file(tmp_path, feed):
    starter = build_starter_policy(feed)
    path = tmp_path / "policy.yaml"
    path.write_text("original contents\n")

    write_starter_policy(starter, path, force=True)

    assert path.read_text() == starter.text
    policy = parse_policy(_yaml_load(path.read_text()))
    assert policy.providers


def test_no_temporary_file_remains_after_a_write(tmp_path, feed):
    starter = build_starter_policy(feed)
    path = tmp_path / "policy.yaml"

    write_starter_policy(starter, path)

    remaining = list(Path(tmp_path).iterdir())
    assert remaining == [path]


def test_generated_text_carries_a_commented_out_feed_block(feed):
    starter = build_starter_policy(feed)
    lines = starter.text.splitlines()
    assert any(line.strip() == "# feed:" for line in lines)
    assert any(line.strip().startswith("#   url:") for line in lines)
