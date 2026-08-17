"""`litellm-maintainer headroom refresh`, wired through the CLI.

Points Policy's `headroom.command` at `tests/fixtures/codexbar_stub.py`,
never at the real binary, so these tests stay offline.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from litellm_maintainer.cli import main
from litellm_maintainer.headroom import read_headroom
from litellm_maintainer.paths import headroom_path

FIXTURES = Path(__file__).parent / "fixtures"
STUB = str(FIXTURES / "codexbar_stub.py")


def _minimal_policy(home: Path, *, headroom: dict | None = None) -> Path:
    raw = {
        "providers": {},
        "quality": {"minimum_coding_score": 18},
        "approved_candidates": [],
        "naming": {"alias_prefix": "claude-", "provider_labels": {}, "alias_overrides": {}},
        "withheld": {},
        "declared": [],
        "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
        "schedule": {
            "enabled": True,
            "interval_minutes": 60,
            "require_proxy": False,
            "maximum_staleness_hours": 24,
        },
        "safety": {"maximum_removal_share": 0.5, "snapshot_count": 3},
    }
    if headroom is not None:
        raw["headroom"] = headroom
    path = home / "policy.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_headroom_refresh_writes_state_from_the_stub_script(tmp_path, capsys):
    policy_file = _minimal_policy(
        tmp_path,
        headroom={
            "command": STUB,
            "sources": {"pool:claude-subscription": "codexbar:claude/"},
        },
    )

    code = main(
        ["headroom", "refresh", "--policy", str(policy_file), "--home", str(tmp_path)]
    )

    assert code == 0
    state = read_headroom(headroom_path(tmp_path))
    assert "pool:claude-subscription" in state.records
    out = capsys.readouterr().out
    assert "updated 1 of 1" in out


def test_headroom_refresh_passes_policys_timeout_seconds_to_the_runner(tmp_path, monkeypatch):
    """Defect 6: the codexbar timeout was fixed at 40s in code and not
    stated in Policy. Measured 2026-07-28: 24s for four providers, 21-31s
    for all -- a fifth or sixth mapped provider plausibly crosses 40s.
    `headroom.timeout_seconds` must reach the runner `cmd_headroom_refresh`
    builds."""
    captured = {}

    def fake_real_codexbar_runner(command, *, timeout=None):
        captured["command"] = command
        captured["timeout"] = timeout

        def runner(providers, all_accounts_provider=None):
            return "[]"

        return runner

    monkeypatch.setattr(
        "litellm_maintainer.headroom.real_codexbar_runner", fake_real_codexbar_runner
    )

    policy_file = _minimal_policy(
        tmp_path,
        headroom={
            "command": STUB,
            "sources": {"pool:claude-subscription": "codexbar:claude/"},
            "timeout_seconds": 12,
        },
    )

    code = main(
        ["headroom", "refresh", "--policy", str(policy_file), "--home", str(tmp_path)]
    )

    assert code == 0
    assert captured["command"] == STUB
    assert captured["timeout"] == 12


def test_headroom_refresh_with_no_sources_exits_zero_and_says_so(tmp_path, capsys):
    policy_file = _minimal_policy(tmp_path)

    code = main(
        ["headroom", "refresh", "--policy", str(policy_file), "--home", str(tmp_path)]
    )

    assert code == 0
    assert "nothing" in capsys.readouterr().out.lower()
    assert not headroom_path(tmp_path).exists()


def test_headroom_refresh_reports_a_missing_policy(tmp_path, capsys):
    code = main(
        [
            "headroom",
            "refresh",
            "--policy",
            str(tmp_path / "does-not-exist.yaml"),
            "--home",
            str(tmp_path),
        ]
    )

    assert code == 1
    assert "No Policy" in capsys.readouterr().err
