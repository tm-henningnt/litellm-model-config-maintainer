"""Failure paths an operator hits: a missing Feed file, a dying write.

A missing or unreadable Feed must produce one redacted line and exit 1,
never a traceback. `generate` and `status` already did this; `probe`
and `smoke` crashed instead. The Generated Config write must be atomic:
the proxy's `--reload` watcher reads that exact file, so a write that
dies half-way must leave the previous config in place, not a truncated
document.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from litellm_maintainer.cli import main
from litellm_maintainer.generate import write_config


def _policy_path(tmp_path: Path) -> Path:
    path = tmp_path / "policy.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "providers": {},
                "quality": {"minimum_coding_score": 18},
                "approved_candidates": [],
                "naming": {
                    "alias_prefix": "claude-",
                    "provider_labels": {},
                    "alias_overrides": {},
                },
                "withheld": {},
                "declared": [
                    {"alias": "claude-one", "litellm_params": {"model": "anthropic/one"}}
                ],
                "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
                "schedule": {
                    "enabled": True,
                    "interval_minutes": 60,
                    "require_proxy": True,
                    "maximum_staleness_hours": 24,
                },
                "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
            }
        )
    )
    return path


def test_probe_reports_a_missing_feed_file_and_exits_one(tmp_path: Path, capsys):
    exit_code = main(
        [
            "probe",
            "--feed",
            str(tmp_path / "no-such-feed.json"),
            "--policy",
            str(_policy_path(tmp_path)),
            "--home",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert exit_code == 1
    assert "Could not read the Feed" in capsys.readouterr().err


def test_smoke_reports_a_missing_feed_file_and_exits_one(tmp_path: Path, capsys):
    exit_code = main(
        [
            "smoke",
            "--feed",
            str(tmp_path / "no-such-feed.json"),
            "--policy",
            str(_policy_path(tmp_path)),
            "--home",
            str(tmp_path),
            "--dry-run",
        ]
    )

    assert exit_code == 1
    assert "Could not read the Feed" in capsys.readouterr().err


def test_a_config_write_that_dies_leaves_the_previous_config_in_place(
    tmp_path: Path, monkeypatch
):
    out_path = tmp_path / "config.yaml"
    write_config({"model_list": [{"model_name": "claude-old", "litellm_params": {}}]}, out_path)
    before = out_path.read_text()

    def _dying_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", _dying_replace)
    with pytest.raises(OSError):
        write_config(
            {"model_list": [{"model_name": "claude-new", "litellm_params": {}}]}, out_path
        )
    monkeypatch.undo()

    # The previous config survives byte for byte, and no temporary
    # file litters the directory.
    assert out_path.read_text() == before
    assert [p.name for p in tmp_path.iterdir()] == ["config.yaml"]


def test_health_state_writes_into_a_fresh_instance_directory(tmp_path: Path):
    """`probe` and the watcher's confirming Probe call `write_health`
    without `ensure_instance_dirs`. A fresh instance directory must not
    crash the write — the probes have already run by then, and a crash
    here loses their results."""
    from litellm_maintainer.health import read_health, write_health
    from litellm_maintainer.reduce import HealthState, OfferingHealth

    path = tmp_path / "fresh-home" / "state" / "health.json"
    write_health(path, HealthState(offerings={"acme:widget": OfferingHealth()}))

    assert "acme:widget" in read_health(path).offerings


def test_a_config_write_lands_whole(tmp_path: Path):
    out_path = tmp_path / "config.yaml"
    write_config({"model_list": [{"model_name": "claude-one", "litellm_params": {}}]}, out_path)

    document = yaml.safe_load(out_path.read_text())
    assert [e["model_name"] for e in document["model_list"]] == ["claude-one"]
    assert [p.name for p in tmp_path.iterdir()] == ["config.yaml"]
