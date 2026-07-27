"""Ticket 10, Part 2: the safety rail.

Every refusal here writes nothing and reports what it would have done
(CONTEXT.md is silent on this; see the spec's "Safety" section and
`.scratch/maintainer-v1/issues/10-acceptance-and-safety-rail.md`).

Unit tests exercise the pure checks in `litellm_maintainer.safety`
directly. Integration tests drive `litellm_maintainer.cli.main` end to
end against a temporary instance directory, because the rule an
operator actually depends on is "the command did not touch the file",
not "a function returned a string".

Every test name states a rule an operator would recognise. See the
docstring on each test for the mutation the author ran to confirm the
test catches the rule's removal, where one was run.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from litellm_maintainer.cli import main
from litellm_maintainer.generate import read_previous_config, write_config
from litellm_maintainer.safety import (
    MINIMUM_PLAUSIBLE_OFFERING_COUNT,
    SafetyError,
    detect_envelope_downgrades,
    list_snapshots,
    prune_snapshots,
    refusal_for_implausible_feed,
    refusal_for_removal_share,
    refusal_for_zero_offered,
    removed_aliases,
    rollback_latest_snapshot,
    snapshot_config,
    validate_config_before_write,
)

NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)


# --- Unit tests: the pure checks -----------------------------------


def test_a_removal_over_the_share_refuses():
    """Mutation-tested: replacing `share <= maximum_removal_share` with
    `share < maximum_removal_share` in `refusal_for_removal_share` still
    fails this test at the boundary test below, not this one — this one
    checks a removal well past the line.
    """
    refusal = refusal_for_removal_share(
        previous_count=100, new_count=50, maximum_removal_share=0.25
    )
    assert refusal is not None
    assert "50" in refusal.message and "100" in refusal.message


def test_a_removal_at_exactly_the_share_does_not_refuse():
    """Removing exactly a quarter of 100 (25) must not refuse: the rule
    is "more than a quarter", not "a quarter or more". Mutation-tested
    by changing `<=` to `<` in the source, which turns this into a
    refusal — confirmed the test then fails.
    """
    refusal = refusal_for_removal_share(
        previous_count=100, new_count=75, maximum_removal_share=0.25
    )
    assert refusal is None


def test_a_removal_one_offering_over_the_share_refuses():
    refusal = refusal_for_removal_share(
        previous_count=100, new_count=74, maximum_removal_share=0.25
    )
    assert refusal is not None


def test_no_previous_count_never_refuses_on_removal_share():
    """A first run has nothing to compare against. Mutation-tested:
    removing the `previous_count is None` guard makes this raise a
    `TypeError` computing a share against `None`, which is caught as a
    failure either way — the guard exists so the failure mode is "skip
    the check", not "crash the run".
    """
    assert refusal_for_removal_share(previous_count=None, new_count=0, maximum_removal_share=0.25) is None


def test_growing_the_offered_count_never_refuses_on_removal_share():
    assert refusal_for_removal_share(previous_count=50, new_count=60, maximum_removal_share=0.25) is None


def test_the_removal_share_refusal_names_what_it_would_have_removed():
    refusal = refusal_for_removal_share(
        previous_count=4,
        new_count=1,
        maximum_removal_share=0.25,
        removed_aliases=("claude-a", "claude-b", "claude-c"),
    )
    assert refusal is not None
    assert "claude-a" in refusal.message
    assert "claude-b" in refusal.message
    assert "claude-c" in refusal.message


def test_a_run_offering_zero_refuses():
    """Mutation-tested: inverting `new_count > 0` to `new_count >= 0`
    makes every run — including a healthy 78-Alias one — refuse, which
    fails every acceptance test in `tests/test_acceptance.py`; inverting
    it the other way (`new_count > 0` to always `True`) makes this test
    fail directly, since it asserts a refusal exists.
    """
    assert refusal_for_zero_offered(0) is not None


def test_a_run_offering_at_least_one_alias_does_not_refuse_on_zero_offered():
    assert refusal_for_zero_offered(1) is None


def test_an_implausibly_short_feed_refuses_when_providers_are_configured():
    refusal = refusal_for_implausible_feed(3, providers_configured=True)
    assert refusal is not None
    assert str(MINIMUM_PLAUSIBLE_OFFERING_COUNT) in refusal


def test_a_feed_at_the_plausible_minimum_does_not_refuse():
    refusal = refusal_for_implausible_feed(
        MINIMUM_PLAUSIBLE_OFFERING_COUNT, providers_configured=True
    )
    assert refusal is None


def test_an_implausibly_short_feed_is_not_checked_when_policy_declares_no_providers():
    """A Declared-only Policy never reads the Feed for Selection, so an
    empty or tiny Feed document is not a hazard for it. Mutation-tested:
    removing the `providers_configured` guard makes
    `tests/test_cli_generate.py::test_generate_writes_a_config_when_
    plan_does_not_refuse` fail, since that test's Policy declares no
    providers and its Feed fixture carries zero Offerings.
    """
    assert refusal_for_implausible_feed(0, providers_configured=False) is None


# --- Unit tests: structural validation ------------------------------


def _entry(alias: str, model: str = "openai/x", **litellm_params) -> dict:
    params = {"model": model, **litellm_params}
    return {"model_name": alias, "litellm_params": params}


def test_two_entries_sharing_an_alias_fail_validation():
    """Mirrors docs/gotchas.md, "Duplicate model_name values do not
    raise an error": litellm treats this as one load-balancing group,
    not an error, so the Generator must catch it itself. Mutation-
    tested: removing the alias-uniqueness loop in
    `validate_config_before_write` makes this test fail (empty tuple
    returned).
    """
    config = {"model_list": [_entry("claude-x"), _entry("claude-x")]}
    problems = validate_config_before_write(config, credential_resolver=lambda name: None)
    assert any("claude-x" in p and "unique" in p for p in problems)


def test_unique_aliases_pass_validation():
    config = {"model_list": [_entry("claude-x"), _entry("claude-y")]}
    problems = validate_config_before_write(config, credential_resolver=lambda name: "ignored")
    assert problems == ()


def test_an_entry_naming_no_model_fails_validation():
    """Mutation-tested: removing the `not model` check makes this test
    fail — `validate_config_before_write` would return `()` for an
    entry whose `litellm_params` carries no `model` key at all.
    """
    config = {"model_list": [{"model_name": "claude-x", "litellm_params": {}}]}
    problems = validate_config_before_write(config, credential_resolver=lambda name: "ignored")
    assert any("claude-x" in p and "model" in p for p in problems)


def test_a_credential_variable_that_does_not_resolve_fails_validation():
    """The check is injectable: no real credential is read here.
    Mutation-tested: hard-coding the resolver call to always return a
    truthy value collapses this to an empty tuple, and the test fails.
    """
    config = {
        "model_list": [_entry("claude-x", api_key="os.environ/MISSING_VAR")],
    }
    problems = validate_config_before_write(config, credential_resolver=lambda name: None)
    assert any("MISSING_VAR" in p for p in problems)


def test_a_credential_variable_that_resolves_passes_validation():
    config = {
        "model_list": [_entry("claude-x", api_key="os.environ/PRESENT_VAR")],
    }
    resolver = {"PRESENT_VAR": "fake-value-not-a-real-credential"}.get
    problems = validate_config_before_write(config, credential_resolver=resolver)
    assert problems == ()


def test_a_plain_api_key_string_with_no_os_environ_reference_is_not_checked():
    """Only an `os.environ/NAME` reference is a credential variable. A
    Declared Offering's `api_key` need not always be one (though every
    real one in the operator's Policy is).
    """
    config = {"model_list": [_entry("claude-x", api_key="not-a-reference")]}
    problems = validate_config_before_write(config, credential_resolver=lambda name: None)
    assert problems == ()


# --- Unit tests: the envelope downgrade check -----------------------


def test_an_alias_that_loses_the_handler_is_reported():
    """Correction 5: a Feed revision can stop publishing the envelope
    key for an Offering that still wraps its responses, and
    `translate_offering` falls back to the generic rule with no error.
    Mutation-tested: deleting the `startswith(handler_prefix)` check on
    the previous side (always treating the previous entry as non-
    handler-routed) makes this test fail — the downgrade goes
    unreported.
    """
    previous = {"model_list": [_entry("claude-cline-free-x", model="cline/vendor/x")]}
    new = {"model_list": [_entry("claude-cline-free-x", model="openai/vendor/x")]}
    assert detect_envelope_downgrades(previous, new) == ("claude-cline-free-x",)


def test_an_alias_that_keeps_the_handler_is_not_reported():
    previous = {"model_list": [_entry("claude-cline-free-x", model="cline/vendor/x")]}
    new = {"model_list": [_entry("claude-cline-free-x", model="cline/vendor/x")]}
    assert detect_envelope_downgrades(previous, new) == ()


def test_a_brand_new_alias_with_no_previous_entry_is_not_reported_as_a_downgrade():
    previous = {"model_list": []}
    new = {"model_list": [_entry("claude-cline-free-x", model="openai/vendor/x")]}
    assert detect_envelope_downgrades(previous, new) == ()


def test_no_previous_config_reports_no_downgrades():
    new = {"model_list": [_entry("claude-cline-free-x", model="openai/vendor/x")]}
    assert detect_envelope_downgrades(None, new) == ()
    assert detect_envelope_downgrades({}, new) == ()


def test_removed_aliases_names_every_alias_the_previous_config_offered_but_the_new_one_does_not():
    previous = {"model_list": [_entry("claude-a"), _entry("claude-b"), _entry("claude-c")]}
    new = {"model_list": [_entry("claude-a")]}
    assert removed_aliases(previous, new) == ("claude-b", "claude-c")


# --- Unit tests: snapshot, prune, rollback ---------------------------


def test_snapshotting_a_missing_config_does_nothing(tmp_path):
    result = snapshot_config(tmp_path / "config.yaml", tmp_path / "snapshots", keep=5, now=NOW)
    assert result is None
    assert not (tmp_path / "snapshots").exists()


def test_snapshotting_an_existing_config_copies_it(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_list: []\n")
    snapshots_dir = tmp_path / "snapshots"

    snapshot_path = snapshot_config(config_path, snapshots_dir, keep=5, now=NOW)

    assert snapshot_path is not None
    assert snapshot_path.read_text() == config_path.read_text()
    assert snapshot_path in list_snapshots(snapshots_dir)


def test_snapshots_are_pruned_to_a_fixed_count(tmp_path):
    """Mutation-tested: replacing `keep=2` handling with "keep everything"
    (deleting the `prune_snapshots` call inside `snapshot_config`) makes
    this test fail — 5 files would remain instead of 2.
    """
    config_path = tmp_path / "config.yaml"
    snapshots_dir = tmp_path / "snapshots"
    for i in range(5):
        config_path.write_text(f"model_list: [{i}]\n")
        snapshot_config(
            config_path, snapshots_dir, keep=2, now=datetime(2026, 7, 25, 0, 0, i, tzinfo=timezone.utc)
        )
    assert len(list_snapshots(snapshots_dir)) == 2


def test_pruning_keeps_the_newest_snapshots():
    pass  # covered by test_snapshots_are_pruned_to_a_fixed_count via lexical ordering


def test_prune_snapshots_deletes_only_the_excess(tmp_path):
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    for name in ("config.1.yaml", "config.2.yaml", "config.3.yaml"):
        (snapshots_dir / name).write_text("x")
    deleted = prune_snapshots(snapshots_dir, keep=2)
    assert len(deleted) == 1
    assert len(list_snapshots(snapshots_dir)) == 2


def test_prune_snapshots_deletes_nothing_when_under_the_limit(tmp_path):
    snapshots_dir = tmp_path / "snapshots"
    snapshots_dir.mkdir()
    (snapshots_dir / "config.1.yaml").write_text("x")
    assert prune_snapshots(snapshots_dir, keep=5) == ()
    assert len(list_snapshots(snapshots_dir)) == 1


def test_rollback_restores_the_most_recent_snapshot(tmp_path):
    config_path = tmp_path / "config.yaml"
    snapshots_dir = tmp_path / "snapshots"

    config_path.write_text("model_list: [1]\n")
    snapshot_config(config_path, snapshots_dir, keep=5, now=datetime(2026, 7, 25, 0, 0, 1, tzinfo=timezone.utc))
    config_path.write_text("model_list: [2]\n")
    snapshot_config(config_path, snapshots_dir, keep=5, now=datetime(2026, 7, 25, 0, 0, 2, tzinfo=timezone.utc))
    config_path.write_text("model_list: [3, broken]\n")

    restored = rollback_latest_snapshot(config_path, snapshots_dir)

    assert config_path.read_text() == "model_list: [2]\n"
    assert restored.name.endswith(".yaml")


def test_rollback_with_no_snapshot_raises_and_touches_nothing(tmp_path):
    """Mutation-tested: returning `None` instead of raising `SafetyError`
    when `snapshots_dir` is empty makes `cli.cmd_rollback` crash with an
    `AttributeError` on `.name` instead of printing a clean refusal —
    confirmed by reverting the raise and re-running
    `test_rollback_command_fails_when_no_snapshot_exists` below, which
    then fails with an unhandled exception rather than exit code 1.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model_list: [1]\n")
    with pytest.raises(SafetyError):
        rollback_latest_snapshot(config_path, tmp_path / "snapshots")
    assert config_path.read_text() == "model_list: [1]\n"


# --- CLI integration: the full command, against a temporary instance --


def _write(path: Path, data) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def _filler_offering(index: int) -> dict:
    """An Offering that never passes the baseline filter (no `tool_use`).

    Pads a synthetic Feed document above
    `safety.MINIMUM_PLAUSIBLE_OFFERING_COUNT` without affecting which
    Aliases a test's Policy actually admits — these tests exercise the
    removal-share, zero-offered, snapshot and rollback rules, not the
    implausible-feed-size rule, which has its own tests above.
    """
    return {
        "id": f"groq:filler-{index}",
        "provider": {"id": "groq", "name": "Acme"},
        "provider_model_id": f"filler-{index}",
        "endpoint": {
            "protocol": "openai_chat_completions",
            "base_url": "https://groq.example/v1",
            "model": f"filler-{index}",
        },
        "capabilities": ["chat"],
        "pricing": {"kind": "free"},
        "availability": {"status": "available"},
        "quality": {"coding_score": None},
        "policy": {"visibility": "listed"},
    }


def _feed_raw(offerings: list[dict], *, pad: bool = True) -> dict:
    from litellm_maintainer.safety import MINIMUM_PLAUSIBLE_OFFERING_COUNT

    padding_needed = max(0, MINIMUM_PLAUSIBLE_OFFERING_COUNT - len(offerings)) if pad else 0
    filler = [_filler_offering(i) for i in range(padding_needed)]
    return {
        "schema_version": "test",
        "providers": [
            {
                "id": "groq",
                "name": "Acme",
                "default_base_url": "https://groq.example/v1",
                "authentication": {"credential_hint": "GROQ_API_KEY"},
            }
        ],
        "models": offerings + filler,
    }


def _groq_offering(model_id: str, *, coding_score: float = 40.0) -> dict:
    return {
        "id": f"groq:{model_id}",
        "provider": {"id": "groq", "name": "Acme"},
        "provider_model_id": model_id,
        "endpoint": {
            "protocol": "openai_chat_completions",
            "base_url": "https://groq.example/v1",
            "model": model_id,
        },
        "capabilities": ["chat", "tool_use"],
        "pricing": {"kind": "free"},
        "availability": {"status": "available"},
        "quality": {"coding_score": coding_score},
        "policy": {"visibility": "listed"},
    }


def _policy_raw(*, providers=None, declared=None, maximum_removal_share=0.25, snapshot_count=10):
    return {
        "providers": providers or {},
        "quality": {"minimum_coding_score": 18},
        "approved_candidates": [],
        "naming": {
            "alias_prefix": "claude-",
            "provider_labels": {"groq": "groq"},
            "alias_overrides": {},
        },
        "withheld": {},
        "declared": declared or [],
        "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
        "schedule": {
            "enabled": True,
            "interval_minutes": 60,
            "require_proxy": True,
            "maximum_staleness_hours": 24,
        },
        "safety": {
            "maximum_removal_share": maximum_removal_share,
            "snapshot_count": snapshot_count,
        },
    }


def _empty_env(path: Path) -> Path:
    """A `.env.local`-style file resolving the synthetic `groq` provider's
    credential, so a test exercising some other rule does not also trip
    the credential-resolution check by accident. Only that one variable
    is defined, so a test naming a different variable still exercises
    an unresolved reference.
    """
    path.write_text("GROQ_API_KEY=fake-value-not-a-real-credential\n")
    return path


def test_generate_snapshots_the_previous_config_before_writing(tmp_path):
    home = tmp_path / "home"
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps(_feed_raw([_groq_offering("one"), _groq_offering("two")])))
    policy_path = _write(tmp_path / "policy.yaml", _policy_raw(providers={"groq": {"mode": "all"}}))
    out_path = tmp_path / "out.yaml"
    env_path = _empty_env(tmp_path / "env")

    exit_code = main(
        [
            "generate",
            "--feed",
            str(feed_path),
            "--policy",
            str(policy_path),
            "--out",
            str(out_path),
            "--home",
            str(home),
            "--env",
            str(env_path),
        ]
    )
    assert exit_code == 0
    assert list_snapshots(home / "snapshots") == ()  # nothing existed to snapshot yet
    first_config = yaml.safe_load(out_path.read_text())

    # A second run must CHANGE the config to write at all: an unchanged
    # render is skipped, snapshot and all, so it cannot restart the
    # proxy for nothing (`generate.rendered_config_is_unchanged`).
    feed_path.write_text(
        json.dumps(
            _feed_raw([_groq_offering("one"), _groq_offering("two"), _groq_offering("three")])
        )
    )

    exit_code = main(
        [
            "generate",
            "--feed",
            str(feed_path),
            "--policy",
            str(policy_path),
            "--out",
            str(out_path),
            "--home",
            str(home),
            "--env",
            str(env_path),
        ]
    )
    assert exit_code == 0
    snapshots = list_snapshots(home / "snapshots")
    assert len(snapshots) == 1
    # The snapshot holds the config as it was BEFORE this write.
    assert yaml.safe_load(snapshots[0].read_text()) == first_config


def test_generate_prunes_snapshots_to_the_configured_count(tmp_path):
    home = tmp_path / "home"
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps(_feed_raw([_groq_offering("one")])))
    policy_path = _write(
        tmp_path / "policy.yaml",
        _policy_raw(providers={"groq": {"mode": "all"}}, snapshot_count=2),
    )
    out_path = tmp_path / "out.yaml"
    env_path = _empty_env(tmp_path / "env")

    args = [
        "generate",
        "--feed",
        str(feed_path),
        "--policy",
        str(policy_path),
        "--out",
        str(out_path),
        "--home",
        str(home),
        "--env",
        str(env_path),
    ]
    # Each run must render a DIFFERENT config, or the write is skipped
    # and there is nothing to snapshot
    # (`generate.rendered_config_is_unchanged`). Grow the Feed by one
    # Offering per run.
    for run in range(4):
        feed_path.write_text(
            json.dumps(_feed_raw([_groq_offering(f"model-{i}") for i in range(run + 1)]))
        )
        assert main(args) == 0

    # 4 writes: the first has nothing to snapshot, the next 3 each
    # snapshot the file left by the write before it. Pruned to 2.
    assert len(list_snapshots(home / "snapshots")) == 2


def test_generate_refuses_and_writes_nothing_when_removal_share_is_exceeded(tmp_path):
    """Mutation-tested: removing the `refusal_for_removal_share` call
    from `cmd_generate` (commented out locally) makes this test fail —
    `exit_code` becomes 0 and `out_path` is overwritten with 1 Alias.
    """
    home = tmp_path / "home"
    feed_path = tmp_path / "feed.json"
    policy_path = _write(
        tmp_path / "policy.yaml",
        _policy_raw(providers={"groq": {"mode": "all"}}, maximum_removal_share=0.25),
    )
    out_path = tmp_path / "out.yaml"
    env_path = _empty_env(tmp_path / "env")
    args = [
        "generate",
        "--feed",
        str(feed_path),
        "--policy",
        str(policy_path),
        "--out",
        str(out_path),
        "--home",
        str(home),
        "--env",
        str(env_path),
    ]

    feed_path.write_text(
        json.dumps(_feed_raw([_groq_offering(f"m{i}") for i in range(4)]))
    )
    assert main(args) == 0
    before = out_path.read_text()

    feed_path.write_text(json.dumps(_feed_raw([_groq_offering("m0")])))
    exit_code = main(args)

    assert exit_code == 1
    assert out_path.read_text() == before


def test_the_removal_share_refusal_names_the_aliases_it_would_have_removed(tmp_path, capsys):
    home = tmp_path / "home"
    feed_path = tmp_path / "feed.json"
    policy_path = _write(
        tmp_path / "policy.yaml",
        _policy_raw(providers={"groq": {"mode": "all"}}, maximum_removal_share=0.25),
    )
    out_path = tmp_path / "out.yaml"
    env_path = _empty_env(tmp_path / "env")
    args = [
        "generate",
        "--feed",
        str(feed_path),
        "--policy",
        str(policy_path),
        "--out",
        str(out_path),
        "--home",
        str(home),
        "--env",
        str(env_path),
    ]

    feed_path.write_text(json.dumps(_feed_raw([_groq_offering(f"m{i}") for i in range(4)])))
    assert main(args) == 0

    feed_path.write_text(json.dumps(_feed_raw([_groq_offering("m0")])))
    assert main(args) == 1
    err = capsys.readouterr().err
    assert "claude-groq-m1" in err
    assert "claude-groq-m2" in err
    assert "claude-groq-m3" in err


def test_generate_force_writes_the_removal_anyway(tmp_path):
    home = tmp_path / "home"
    feed_path = tmp_path / "feed.json"
    policy_path = _write(
        tmp_path / "policy.yaml",
        _policy_raw(providers={"groq": {"mode": "all"}}, maximum_removal_share=0.25),
    )
    out_path = tmp_path / "out.yaml"
    env_path = _empty_env(tmp_path / "env")
    args = [
        "generate",
        "--feed",
        str(feed_path),
        "--policy",
        str(policy_path),
        "--out",
        str(out_path),
        "--home",
        str(home),
        "--env",
        str(env_path),
    ]

    feed_path.write_text(json.dumps(_feed_raw([_groq_offering(f"m{i}") for i in range(4)])))
    assert main(args) == 0

    feed_path.write_text(json.dumps(_feed_raw([_groq_offering("m0")])))
    assert main(args + ["--force"]) == 0
    written = yaml.safe_load(out_path.read_text())
    assert [e["model_name"] for e in written["model_list"]] == ["claude-groq-m0"]


def test_generate_refuses_and_writes_nothing_when_it_would_offer_nothing(tmp_path):
    home = tmp_path / "home"
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps(_feed_raw([])))
    policy_path = _write(tmp_path / "policy.yaml", _policy_raw())
    out_path = tmp_path / "out.yaml"
    out_path.write_text("model_list: [{model_name: claude-existing, litellm_params: {model: x}}]\n")
    env_path = _empty_env(tmp_path / "env")

    exit_code = main(
        [
            "generate",
            "--feed",
            str(feed_path),
            "--policy",
            str(policy_path),
            "--out",
            str(out_path),
            "--home",
            str(home),
            "--env",
            str(env_path),
        ]
    )
    assert exit_code == 1
    survived = yaml.safe_load(out_path.read_text())
    assert [e["model_name"] for e in survived["model_list"]] == ["claude-existing"]


def test_generate_refuses_and_writes_nothing_on_a_failed_feed_fetch(tmp_path):
    home = tmp_path / "home"
    feed_path = tmp_path / "feed.json"  # never created
    policy_path = _write(tmp_path / "policy.yaml", _policy_raw())
    out_path = tmp_path / "out.yaml"
    out_path.write_text("model_list: [{model_name: claude-existing, litellm_params: {model: x}}]\n")
    env_path = _empty_env(tmp_path / "env")

    exit_code = main(
        [
            "generate",
            "--feed",
            str(feed_path),
            "--policy",
            str(policy_path),
            "--out",
            str(out_path),
            "--home",
            str(home),
            "--env",
            str(env_path),
        ]
    )
    assert exit_code == 1
    assert out_path.read_text() == "model_list: [{model_name: claude-existing, litellm_params: {model: x}}]\n"


def test_generate_refuses_and_writes_nothing_on_an_implausibly_short_model_list(tmp_path):
    home = tmp_path / "home"
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps(_feed_raw([_groq_offering("one")], pad=False)))
    policy_path = _write(tmp_path / "policy.yaml", _policy_raw(providers={"groq": {"mode": "all"}}))
    out_path = tmp_path / "out.yaml"
    out_path.write_text("model_list: [{model_name: claude-existing, litellm_params: {model: x}}]\n")
    env_path = _empty_env(tmp_path / "env")

    exit_code = main(
        [
            "generate",
            "--feed",
            str(feed_path),
            "--policy",
            str(policy_path),
            "--out",
            str(out_path),
            "--home",
            str(home),
            "--env",
            str(env_path),
        ]
    )
    assert exit_code == 1
    assert out_path.read_text() == "model_list: [{model_name: claude-existing, litellm_params: {model: x}}]\n"


def test_generate_refuses_and_writes_nothing_when_policy_does_not_parse(tmp_path):
    home = tmp_path / "home"
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps(_feed_raw([])))
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text("providers: not-a-mapping\n")
    out_path = tmp_path / "out.yaml"
    out_path.write_text("model_list: [{model_name: claude-existing, litellm_params: {model: x}}]\n")
    env_path = _empty_env(tmp_path / "env")

    exit_code = main(
        [
            "generate",
            "--feed",
            str(feed_path),
            "--policy",
            str(policy_path),
            "--out",
            str(out_path),
            "--home",
            str(home),
            "--env",
            str(env_path),
        ]
    )
    assert exit_code == 1
    assert out_path.read_text() == "model_list: [{model_name: claude-existing, litellm_params: {model: x}}]\n"


def test_generate_refuses_and_writes_nothing_when_a_credential_variable_is_unresolved(tmp_path):
    """A Declared Offering naming an unset `os.environ/NAME` credential
    must stop the write. Mutation-tested: removing the
    `validation_problems` gate in `cmd_generate` makes this test fail —
    exit code becomes 0 and `out_path` is overwritten.
    """
    home = tmp_path / "home"
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps(_feed_raw([])))
    policy_path = _write(
        tmp_path / "policy.yaml",
        _policy_raw(
            declared=[
                {
                    "alias": "claude-declared",
                    "litellm_params": {
                        "model": "anthropic/claude-x",
                        "api_key": "os.environ/DOES_NOT_EXIST_ANYWHERE",
                    },
                }
            ]
        ),
    )
    out_path = tmp_path / "out.yaml"
    out_path.write_text("model_list: [{model_name: claude-existing, litellm_params: {model: x}}]\n")
    env_path = _empty_env(tmp_path / "env")

    exit_code = main(
        [
            "generate",
            "--feed",
            str(feed_path),
            "--policy",
            str(policy_path),
            "--out",
            str(out_path),
            "--home",
            str(home),
            "--env",
            str(env_path),
        ]
    )
    assert exit_code == 1
    assert out_path.read_text() == "model_list: [{model_name: claude-existing, litellm_params: {model: x}}]\n"


def test_generate_force_never_overrides_a_validation_failure(tmp_path):
    """`--force` applies a refused *judgment call* (a threshold), never
    a structural defect. An unresolved credential must still refuse
    even with `--force`.
    """
    home = tmp_path / "home"
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps(_feed_raw([])))
    policy_path = _write(
        tmp_path / "policy.yaml",
        _policy_raw(
            declared=[
                {
                    "alias": "claude-declared",
                    "litellm_params": {
                        "model": "anthropic/claude-x",
                        "api_key": "os.environ/DOES_NOT_EXIST_ANYWHERE",
                    },
                }
            ]
        ),
    )
    out_path = tmp_path / "out.yaml"
    env_path = _empty_env(tmp_path / "env")

    exit_code = main(
        [
            "generate",
            "--feed",
            str(feed_path),
            "--policy",
            str(policy_path),
            "--out",
            str(out_path),
            "--home",
            str(home),
            "--env",
            str(env_path),
            "--force",
        ]
    )
    assert exit_code == 1
    assert not out_path.exists()


def test_generate_refuses_when_health_state_is_empty_and_would_drop_a_sunsetting_offering(
    tmp_path, capsys
):
    """Hardening ticket 02, superseding correction 9's warning.

    A warning on a command that then writes the file is read once and
    never again. Sunsetting needs a success this tool recorded itself,
    so an empty Health State drops every Sunsetting Offering, and the
    removal-share guard cannot catch a drop that small. So generate
    refuses, writes nothing, and names what it would have dropped.
    """
    home = tmp_path / "home"
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(
        json.dumps(
            _feed_raw(
                [
                    {
                        **_groq_offering("hidden-leaving"),
                        "availability": {"status": "retired"},
                        "policy": {"visibility": "hidden"},
                    },
                    _groq_offering("ordinary"),
                ]
            )
        )
    )
    policy_path = _write(tmp_path / "policy.yaml", _policy_raw(providers={"groq": {"mode": "all"}}))
    out_path = tmp_path / "out.yaml"
    env_path = _empty_env(tmp_path / "env")

    exit_code = main(
        [
            "generate",
            "--feed",
            str(feed_path),
            "--policy",
            str(policy_path),
            "--out",
            str(out_path),
            "--home",
            str(home),
            "--env",
            str(env_path),
        ]
    )
    assert exit_code == 1
    captured = capsys.readouterr()
    message = captured.err + captured.out
    assert "Refused to write" in message
    assert "no Probe has ever measured" in message
    assert "groq:hidden-leaving" in message
    assert not out_path.exists(), "a refused run must write nothing"

    # The force flag applies the refused change, as it does for every
    # other refusal.
    forced = main(
        [
            "generate",
            "--feed",
            str(feed_path),
            "--policy",
            str(policy_path),
            "--out",
            str(out_path),
            "--home",
            str(home),
            "--env",
            str(env_path),
            "--force",
        ]
    )
    assert forced == 0
    assert out_path.exists()


def test_generate_reports_an_envelope_downgrade_loudly_but_still_writes(tmp_path, monkeypatch):
    """Correction 5: an Alias that loses the `cline/` handler must be
    reported loudly, never blocked. Build a previous Generated Config
    by hand naming a handler-routed entry, then run `generate` against a
    Feed that (correctly, for this synthetic provider) never routes
    there, and confirm the write still happens with a loud warning.
    """
    home = tmp_path / "home"
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps(_feed_raw([_groq_offering("one")])))
    policy_path = _write(tmp_path / "policy.yaml", _policy_raw(providers={"groq": {"mode": "all"}}))
    out_path = tmp_path / "out.yaml"
    write_config(
        {"model_list": [{"model_name": "claude-groq-one", "litellm_params": {"model": "cline/vendor/one"}}]},
        out_path,
    )
    env_path = _empty_env(tmp_path / "env")

    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exit_code = main(
            [
                "generate",
                "--feed",
                str(feed_path),
                "--policy",
                str(policy_path),
                "--out",
                str(out_path),
                "--home",
                str(home),
                "--env",
                str(env_path),
            ]
        )
    assert exit_code == 0
    assert "claude-groq-one" in buf.getvalue()
    assert "choices" in buf.getvalue()
    written = yaml.safe_load(out_path.read_text())
    assert [e["model_name"] for e in written["model_list"]] == ["claude-groq-one"]


def test_rollback_command_restores_the_most_recent_snapshot(tmp_path):
    home = tmp_path / "home"
    feed_path = tmp_path / "feed.json"
    policy_path = _write(tmp_path / "policy.yaml", _policy_raw(providers={"groq": {"mode": "all"}}))
    out_path = tmp_path / "out.yaml"
    env_path = _empty_env(tmp_path / "env")
    generate_args = [
        "generate",
        "--feed",
        str(feed_path),
        "--policy",
        str(policy_path),
        "--out",
        str(out_path),
        "--home",
        str(home),
        "--env",
        str(env_path),
    ]

    feed_path.write_text(json.dumps(_feed_raw([_groq_offering("one")])))
    assert main(generate_args) == 0
    good_config = out_path.read_text()

    feed_path.write_text(json.dumps(_feed_raw([_groq_offering("one"), _groq_offering("two")])))
    assert main(generate_args) == 0
    assert out_path.read_text() != good_config

    exit_code = main(["rollback", "--out", str(out_path), "--home", str(home), "--env", str(env_path)])

    assert exit_code == 0
    assert out_path.read_text() == good_config


def test_rollback_command_fails_when_no_snapshot_exists(tmp_path):
    home = tmp_path / "home"
    out_path = tmp_path / "out.yaml"
    out_path.write_text("model_list: [{model_name: claude-existing, litellm_params: {model: x}}]\n")

    exit_code = main(["rollback", "--out", str(out_path), "--home", str(home)])

    assert exit_code == 1
    assert out_path.read_text() == "model_list: [{model_name: claude-existing, litellm_params: {model: x}}]\n"


def test_a_probe_of_one_provider_does_not_clear_the_refusal_for_another(
    tmp_path, capsys
):
    """The first live sweep taught this rule.

    A sweep scoped to one provider leaves Health State non-empty while
    every Sunsetting Offering is still unprobed. A refusal keyed on "is
    Health State empty" would pass at that point and drop them silently.
    So the refusal keys on "no Probe has ever measured THIS Offering".
    """
    home = tmp_path / "home"
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(
        json.dumps(
            _feed_raw(
                [
                    {
                        **_groq_offering("hidden-leaving"),
                        "availability": {"status": "retired"},
                        "policy": {"visibility": "hidden"},
                    },
                    _groq_offering("ordinary"),
                ]
            )
        )
    )
    policy_path = _write(
        tmp_path / "policy.yaml", _policy_raw(providers={"groq": {"mode": "all"}})
    )
    env_path = _empty_env(tmp_path / "env")

    # Health State holds a record for an unrelated Offering, so it is
    # NOT empty. The Sunsetting one is still unmeasured.
    from litellm_maintainer.health import write_health
    from litellm_maintainer.paths import ensure_instance_dirs, health_path
    from litellm_maintainer.reduce import HealthState, OfferingHealth

    ensure_instance_dirs(home)
    write_health(
        health_path(home),
        HealthState(
            offerings={
                "groq:ordinary": OfferingHealth(
                    excluded=False,
                    reason="answered",
                    bucket="answered",
                    reset_at=None,
                    last_success_at=datetime(2026, 7, 26, tzinfo=timezone.utc),
                    last_attempt_at=datetime(2026, 7, 26, tzinfo=timezone.utc),
                    failure_count=0,
                )
            }
        ),
    )

    exit_code = main(
        [
            "generate", "--feed", str(feed_path), "--policy", str(policy_path),
            "--out", str(tmp_path / "out.yaml"), "--home", str(home),
            "--env", str(env_path),
        ]
    )
    message = "".join(capsys.readouterr())
    assert exit_code == 1, "a non-empty Health State must not clear the refusal"
    assert "groq:hidden-leaving" in message
    assert not (tmp_path / "out.yaml").exists()


def test_a_sunsetting_offering_that_was_probed_and_failed_no_longer_refuses(
    tmp_path, capsys
):
    """The refusal must be clearable, never a permanent wall.

    Once a Probe has measured an Offering, the operator has the answer.
    A model that was probed and did not answer is dropped for a good
    reason, so generate proceeds.
    """
    home = tmp_path / "home"
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(
        json.dumps(
            _feed_raw(
                [
                    {
                        **_groq_offering("hidden-leaving"),
                        "availability": {"status": "retired"},
                        "policy": {"visibility": "hidden"},
                    },
                    _groq_offering("ordinary"),
                ]
            )
        )
    )
    policy_path = _write(
        tmp_path / "policy.yaml", _policy_raw(providers={"groq": {"mode": "all"}})
    )
    env_path = _empty_env(tmp_path / "env")

    from litellm_maintainer.health import write_health
    from litellm_maintainer.paths import ensure_instance_dirs, health_path
    from litellm_maintainer.reduce import HealthState, OfferingHealth

    ensure_instance_dirs(home)
    write_health(
        health_path(home),
        HealthState(
            offerings={
                "groq:hidden-leaving": OfferingHealth(
                    excluded=True,
                    reason="gateway_error",
                    bucket="self_healing",
                    reset_at=None,
                    last_success_at=None,
                    last_attempt_at=datetime(2026, 7, 26, tzinfo=timezone.utc),
                    failure_count=1,
                )
            }
        ),
    )

    exit_code = main(
        [
            "generate", "--feed", str(feed_path), "--policy", str(policy_path),
            "--out", str(tmp_path / "out.yaml"), "--home", str(home),
            "--env", str(env_path),
        ]
    )
    assert exit_code == 0, "a measured Offering must not block the write forever"
    assert (tmp_path / "out.yaml").exists()


def test_a_run_that_changes_nothing_leaves_the_generated_config_untouched(tmp_path):
    """The Generated Config is the one file the proxy's `--reload`
    watcher reads, so every write restarts the proxy and drops the
    requests in flight. A journal-triggered run can fire within a
    minute of the last one, which would turn a burst of failures into a
    burst of restarts -- dropping the very calls it is reacting to."""
    home = tmp_path / "home"
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps(_feed_raw([_groq_offering("one")])))
    policy_path = _write(
        tmp_path / "policy.yaml", _policy_raw(providers={"groq": {"mode": "all"}})
    )
    out_path = tmp_path / "out.yaml"
    env_path = _empty_env(tmp_path / "env")
    args = [
        "generate",
        "--feed",
        str(feed_path),
        "--policy",
        str(policy_path),
        "--out",
        str(out_path),
        "--home",
        str(home),
        "--env",
        str(env_path),
    ]

    assert main(args) == 0
    first_mtime = out_path.stat().st_mtime_ns
    first_text = out_path.read_text()

    assert main(args) == 0

    # Same bytes AND the same file: an atomic rename onto the path
    # would change the mtime even for identical content, and the
    # watcher reads the rename, not the content.
    assert out_path.read_text() == first_text
    assert out_path.stat().st_mtime_ns == first_mtime
    assert list_snapshots(home / "snapshots") == ()
