"""The two read commands print an answer and change nothing.

`entitlements` and `guidance` are the surface an orchestrating agent
calls before it dispatches work. Two properties matter more than their
formatting, and both are pinned here: neither command writes a file, and
`--json` output stays parseable with a stated `schema_version`, because
a caller parses it.

These are command-level tests. The ranking and derivation rules
themselves are pinned in `test_guidance.py` and `test_entitlements.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from litellm_maintainer.cli import main


def _policy_raw(**overrides):
    raw = {
        "providers": {
            "openrouter": {"mode": "all", "entitlement": "shared_pool"},
        },
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
            "require_proxy": True,
            "maximum_staleness_hours": 24,
        },
        "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
    }
    raw.update(overrides)
    return raw


def _offering(model_id: str, *, coding: float, kind: str = "free"):
    return {
        "id": f"openrouter:{model_id}",
        "provider": {"id": "openrouter"},
        "provider_model_id": model_id,
        "display_name": model_id,
        "canonical_model": {"id": f"vendor/{model_id}"},
        "capabilities": ["chat", "coding", "tool_use"],
        "limits": {"context_tokens": 128000, "max_output_tokens": 8192},
        "pricing": {"kind": kind, "input_usd_per_1m_tokens": 0, "metering": "tokens"},
        "availability": {"status": "available"},
        "quality": {"coding_score": coding, "reasoning_score": coding - 5},
        # Selection admits a `listed` Offering only. An Offering with no
        # stated visibility never reaches the Generated Config, so a
        # fixture that omits this field tests an empty answer by mistake.
        "policy": {"visibility": "listed"},
        "endpoint": {"protocol": "openai_chat_completions"},
    }


def _feed_raw(generated_at: str = "2999-01-01T00:00:00Z"):
    """A Feed the staleness check treats as fresh.

    The build time is far in the future on purpose: these tests pin the
    commands, not the clock, and a fixed past date would make them start
    failing on their own one day.
    """
    return {
        "schema_version": "1.0.0",
        "feed": {"id": "test-feed", "generated_at": generated_at},
        "providers": [
            {
                "id": "openrouter",
                "name": "OpenRouter",
                "authentication": {"credential_hint": "OPENROUTER_API_KEY"},
            }
        ],
        "models": [
            _offering("fast-coder", coding=70.0),
            _offering("slow-coder", coding=40.0),
        ],
    }


def _inputs(tmp_path: Path, *, policy=None, feed=None):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(policy or _policy_raw()))
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps(feed or _feed_raw()))
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    return policy_path, feed_path, home


def _argv(command: str, tmp_path: Path, *extra: str):
    policy_path, feed_path, home = _inputs(tmp_path)
    return [
        command,
        "--policy",
        str(policy_path),
        "--feed",
        str(feed_path),
        "--home",
        str(home),
        *extra,
    ]


# --- Neither command writes anything -------------------------------------


def test_entitlements_writes_no_file(tmp_path, capsys):
    argv = _argv("entitlements", tmp_path)
    before = sorted(p.name for p in tmp_path.rglob("*"))

    assert main(argv) == 0
    capsys.readouterr()

    assert sorted(p.name for p in tmp_path.rglob("*")) == before


def test_guidance_writes_no_file(tmp_path, capsys):
    argv = _argv("guidance", tmp_path, "--for", "coding")
    before = sorted(p.name for p in tmp_path.rglob("*"))

    assert main(argv) == 0
    capsys.readouterr()

    assert sorted(p.name for p in tmp_path.rglob("*")) == before


# --- The JSON contract ---------------------------------------------------


def test_guidance_json_parses_and_states_its_schema_version(tmp_path, capsys):
    assert main(_argv("guidance", tmp_path, "--for", "coding", "--json")) == 0

    answer = json.loads(capsys.readouterr().out)

    assert answer["schema_version"]
    assert answer["axis"] == "coding"
    assert answer["rows"][0]["canonical_model_id"] == "vendor/fast-coder"
    assert answer["rows"][0]["routes"][0]["alias"].startswith("claude-")
    assert "client_advisory" in answer


def test_entitlements_json_parses_and_states_its_schema_version(tmp_path, capsys):
    assert main(_argv("entitlements", tmp_path, "--json")) == 0

    view = json.loads(capsys.readouterr().out)

    assert view["schema_version"]
    assert view["entitlements"][0]["provider_id"] == "openrouter"
    assert view["entitlements"][0]["entitlement"] == "shared_pool"


def test_the_json_flag_and_the_format_flag_agree(tmp_path, capsys):
    assert main(_argv("entitlements", tmp_path, "--json")) == 0
    from_flag = capsys.readouterr().out
    assert main(_argv("entitlements", tmp_path, "--format", "json")) == 0
    from_format = capsys.readouterr().out

    assert json.loads(from_flag)["entitlements"] == json.loads(from_format)["entitlements"]


# --- Formats -------------------------------------------------------------


def test_markdown_output_is_markdown(tmp_path, capsys):
    assert main(_argv("guidance", tmp_path, "--format", "markdown")) == 0

    out = capsys.readouterr().out

    assert out.startswith("# Model guidance")
    assert "| Alias |" in out


def test_text_output_names_the_axis_and_the_advisory(tmp_path, capsys):
    assert main(_argv("guidance", tmp_path, "--for", "coding")) == 0

    out = capsys.readouterr().out

    assert "Ranked by coding" in out
    assert "Client advisory" in out


# --- Limits and refusals -------------------------------------------------


def test_limit_caps_the_rows_and_says_so(tmp_path, capsys):
    """A silent cap reads as 'this is everything'. It must announce itself."""
    assert main(_argv("guidance", tmp_path, "--limit", "1", "--json")) == 0

    answer = json.loads(capsys.readouterr().out)

    assert len(answer["rows"]) == 1
    assert any("at most 1" in w for w in answer["warnings"])


def test_an_unreadable_feed_exits_one_and_names_the_path(tmp_path, capsys):
    policy_path, _, home = _inputs(tmp_path)
    missing = tmp_path / "absent.json"

    exit_code = main(
        [
            "guidance",
            "--policy",
            str(policy_path),
            "--feed",
            str(missing),
            "--home",
            str(home),
        ]
    )

    assert exit_code == 1
    assert str(missing) in capsys.readouterr().err


def test_an_invalid_policy_exits_one(tmp_path, capsys):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump({"providers": {}}))
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps(_feed_raw()))

    exit_code = main(
        ["entitlements", "--policy", str(policy_path), "--feed", str(feed_path)]
    )

    assert exit_code == 1
    assert "Policy is invalid" in capsys.readouterr().err


# --- Staleness travels with the answer -----------------------------------


def test_a_stale_feed_document_warns_inside_the_answer(tmp_path, capsys):
    """A stale catalogue produces confident, wrong guidance.

    The warning therefore rides with the answer, rather than waiting for
    the operator to run `doctor`.
    """
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(_policy_raw()))
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps(_feed_raw(generated_at="2020-01-01T00:00:00Z")))

    assert (
        main(
            [
                "guidance",
                "--policy",
                str(policy_path),
                "--feed",
                str(feed_path),
                "--home",
                str(tmp_path),
                "--json",
            ]
        )
        == 0
    )

    answer = json.loads(capsys.readouterr().out)

    assert any("stale" in w for w in answer["warnings"])


def test_guidance_min_context_reaches_the_transform(tmp_path, capsys):
    """The flag narrows the answer and the output explains what it dropped."""
    import json

    from litellm_maintainer.cli import main

    policy_path = Path(__file__).parent / "fixtures" / "policy-pinned.yaml"
    feed_path = Path(__file__).parent / "fixtures" / "feed-current.json"

    exit_code = main(
        [
            "guidance", "--feed", str(feed_path), "--policy", str(policy_path),
            "--for", "coding", "--json", "--min-context", "999999999",
        ]
    )

    assert exit_code == 0
    document = json.loads(capsys.readouterr().out)
    assert document["rows"] == []
    assert any("999999999" in w for w in document["warnings"]), (
        "an empty answer must say why it is empty"
    )


def test_guidance_refuses_a_non_positive_min_context(tmp_path, capsys):
    from litellm_maintainer.cli import main

    policy_path = Path(__file__).parent / "fixtures" / "policy-pinned.yaml"
    feed_path = Path(__file__).parent / "fixtures" / "feed-current.json"

    exit_code = main(
        [
            "guidance", "--feed", str(feed_path), "--policy", str(policy_path),
            "--for", "coding", "--min-context", "0",
        ]
    )

    assert exit_code == 1
    assert "min_context" in capsys.readouterr().err


# --- `status` takes the same defaults as its siblings ---------------------
#
# Reported 2026-07-29: `guidance`, `entitlements` and `headroom` all default
# `--feed` and `--policy` to the instance directory. `status` required both,
# so one command refused the invocation the other three accepted.


def test_status_defaults_its_feed_and_policy_to_the_instance_directory(
    tmp_path, capsys, monkeypatch
):
    policy_path, feed_path, home = _inputs(tmp_path)
    (home / "policy.yaml").write_text(policy_path.read_text())
    (home / "feed.json").write_text(feed_path.read_text())
    monkeypatch.setenv("LITELLM_MAINTAINER_HOME", str(home))

    assert main(["status"]) == 0

    assert "Offered:" in capsys.readouterr().out


def test_status_json_parses_and_states_its_schema_version(tmp_path, capsys):
    assert main(_argv("status", tmp_path, "--json")) == 0

    answer = json.loads(capsys.readouterr().out)

    assert answer["schema_version"]
    assert "offered" in answer
    assert "withheld" in answer
    assert "excluded" in answer
    assert "warnings" in answer


def test_status_writes_no_file(tmp_path, capsys):
    argv = _argv("status", tmp_path)
    before = sorted(p.name for p in tmp_path.rglob("*"))

    assert main(argv) == 0
    capsys.readouterr()

    assert sorted(p.name for p in tmp_path.rglob("*")) == before
