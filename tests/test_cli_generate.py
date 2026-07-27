"""Tests for the `generate` command's refusal gate.

`plan` can refuse. It then returns an empty config alongside the
refusal text. The command must not write that empty config over the
Generated Config.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from litellm_maintainer.cli import main

FIXTURES = Path(__file__).parent / "fixtures"


def _policy_raw(declared):
    return {
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


def _write(path: Path, data) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def _empty_feed(path: Path) -> Path:
    path.write_text(json.dumps({"schema_version": "test", "providers": [], "models": []}))
    return path


def test_generate_writes_a_config_when_plan_does_not_refuse(tmp_path):
    policy_path = _write(
        tmp_path / "policy.yaml",
        _policy_raw([{"alias": "claude-one", "litellm_params": {"model": "anthropic/one"}}]),
    )
    feed_path = _empty_feed(tmp_path / "feed.json")
    out_path = tmp_path / "out.yaml"

    exit_code = main(
        ["generate", "--feed", str(feed_path), "--policy", str(policy_path), "--out", str(out_path)]
    )

    assert exit_code == 0
    written = yaml.safe_load(out_path.read_text())
    assert [e["model_name"] for e in written["model_list"]] == ["claude-one"]


def test_generate_writes_nothing_when_plan_refuses(tmp_path):
    """A refused result must never reach the Generated Config. Before
    this gate the command wrote `plan`'s empty config, so a detected
    Alias collision replaced every Alias with none.
    """
    policy_path = _write(
        tmp_path / "policy.yaml",
        _policy_raw(
            [
                {"alias": "claude-clash", "litellm_params": {"model": "anthropic/first"}},
                {"alias": "claude-clash", "litellm_params": {"model": "anthropic/second"}},
            ]
        ),
    )
    feed_path = _empty_feed(tmp_path / "feed.json")
    out_path = tmp_path / "out.yaml"
    out_path.write_text("model_list: [{model_name: claude-existing, litellm_params: {}}]\n")

    exit_code = main(
        ["generate", "--feed", str(feed_path), "--policy", str(policy_path), "--out", str(out_path)]
    )

    assert exit_code == 1
    # The previous file survives untouched.
    survived = yaml.safe_load(out_path.read_text())
    assert [e["model_name"] for e in survived["model_list"]] == ["claude-existing"]


def test_generate_warns_about_a_stated_limit_collision_and_still_writes(capsys, tmp_path):
    """A collision is reported, never refused.

    litellm holds one cost-map entry per model string, so the operator has
    to be told which Alias defines the other. The run still writes: the
    condition is litellm handling something badly, not a Policy error.
    """
    policy_path = _write(
        tmp_path / "policy.yaml",
        _policy_raw(
            [
                {
                    "alias": "claude-sized",
                    "litellm_params": {"model": "anthropic/shared"},
                    "model_info": {"max_input_tokens": 1000000},
                },
                {
                    "alias": "claude-silent",
                    "litellm_params": {"model": "anthropic/shared"},
                },
            ]
        ),
    )
    feed_path = _empty_feed(tmp_path / "feed.json")
    out_path = tmp_path / "out.yaml"

    exit_code = main(
        ["generate", "--feed", str(feed_path), "--policy", str(policy_path), "--out", str(out_path)]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "WARNING" in out
    assert "anthropic/shared" in out
    assert "claude-sized" in out
    assert "claude-silent" in out
    written = yaml.safe_load(out_path.read_text())
    assert {e["model_name"] for e in written["model_list"]} == {
        "claude-sized",
        "claude-silent",
    }


def test_generate_is_silent_when_siblings_agree_on_their_stated_limit(capsys, tmp_path):
    """Agreement is the normal case: a seat pair, or a variant and its sibling."""
    sized = {"max_input_tokens": 1000000, "max_output_tokens": 128000}
    policy_path = _write(
        tmp_path / "policy.yaml",
        _policy_raw(
            [
                {
                    "alias": "claude-plain",
                    "litellm_params": {"model": "anthropic/shared"},
                    "model_info": dict(sized),
                },
                {
                    "alias": "claude-plain[1m]",
                    "litellm_params": {"model": "anthropic/shared"},
                    "model_info": dict(sized),
                },
            ]
        ),
    )
    feed_path = _empty_feed(tmp_path / "feed.json")
    out_path = tmp_path / "out.yaml"

    exit_code = main(
        ["generate", "--feed", str(feed_path), "--policy", str(policy_path), "--out", str(out_path)]
    )

    assert exit_code == 0
    assert "Stated Limit collision" not in capsys.readouterr().out
    written = yaml.safe_load(out_path.read_text())
    assert "claude-plain[1m]" in {e["model_name"] for e in written["model_list"]}
