"""A tick that cannot take the lock skips quietly. It never crashes.

ADR 0002 gives Health State a lock because three processes act in the
maintainer role at once: the launchd tick, the Journal watcher, and the
operator running `probe` by hand. Contention is therefore normal, not
exceptional, and the tick fires more often than the interval on purpose.

`cmd_run` already had a `LockBusy` handler for this. The handler itself
raised `TypeError`, because it passed three positional arguments to a
keyword-only function and passed the instance directory where the run log
belonged. So the one path built to keep an unattended tick quiet was the
one path that killed it, with no run-log line to show it had happened.
No test covered the handler. This file covers it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import litellm_maintainer.cli as cli_module
from litellm_maintainer.cli import build_parser
from litellm_maintainer.lock import LockBusy


def _policy_raw() -> dict:
    return {
        "providers": {},
        "quality": {"minimum_coding_score": 18},
        "approved_candidates": [],
        "naming": {"alias_prefix": "claude-", "provider_labels": {}, "alias_overrides": {}},
        "withheld": {},
        "declared": [
            {"alias": "claude-one", "litellm_params": {"model": "anthropic/one"}}
        ],
        "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
        "schedule": {
            "enabled": True,
            "interval_minutes": 60,
            "require_proxy": False,
            "maximum_staleness_hours": 24,
        },
        "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
    }


@pytest.fixture
def args(tmp_path):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(_policy_raw()))
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(
        json.dumps({"schema_version": "test", "providers": [], "models": []})
    )
    return build_parser().parse_args(
        [
            "run",
            "--policy",
            str(policy_path),
            "--feed",
            str(feed_path),
            "--home",
            str(tmp_path),
            "--out",
            str(tmp_path / "config.yaml"),
        ]
    )


@pytest.fixture(autouse=True)
def _busy_lock(monkeypatch):
    """Make the Health State lock unavailable, as a second maintainer would."""

    def refuse(*call_args, **call_kwargs):
        raise LockBusy("another maintainer holds the lock")

    monkeypatch.setattr(cli_module, "maintainer_lock", refuse)
    # Probe nothing: the Prober runs before the lock is taken, and a
    # Probe result would only add noise to a test about contention.
    monkeypatch.setattr(cli_module, "probe_offerings", lambda *a, **k: {})


def _run(args):
    return cli_module.cmd_run(
        args,
        proxy_checker=lambda: True,
        probe_transport=lambda target: None,
        notifier=lambda message: None,
        smoke_transport=lambda *a, **k: None,
    )


def test_a_busy_lock_skips_the_tick_and_exits_zero(args, capsys):
    """A skipped tick is not a failure. launchd must not see one."""
    assert _run(args) == 0
    assert "Skipped" in capsys.readouterr().out


def test_a_busy_lock_writes_the_skip_to_the_run_log(args, tmp_path):
    """The operator needs to see that the tick ran and why it did nothing."""
    _run(args)

    log = (tmp_path / "state" / "runs.log").read_text()

    assert "another maintainer is running" in log


def test_a_busy_lock_writes_no_config(args, tmp_path):
    """The other maintainer owns this run. This one must not act."""
    _run(args)

    assert not (tmp_path / "config.yaml").exists()


def test_the_skip_line_helper_rejects_positional_arguments():
    """Pin the signature the broken call site got wrong.

    The handler passed three positional arguments. The helper takes one
    positional argument and three keyword-only ones, so the call raised
    before it could write anything.
    """
    import inspect

    signature = inspect.signature(cli_module._append_tick_skip_line)

    with pytest.raises(TypeError):
        signature.bind(Path("/tmp/x"), "a reason", "a time")

    signature.bind(Path("/tmp/x"), now="t", reason="r", mapping={})
