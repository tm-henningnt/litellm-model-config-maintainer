"""Tests for instance directory resolution."""

from __future__ import annotations


from litellm_maintainer import paths


def test_environment_variable_wins(tmp_path, monkeypatch):
    custom = tmp_path / "custom-home"
    monkeypatch.setenv(paths.ENV_VAR, str(custom))
    assert paths.instance_home() == custom


def test_default_applies_when_unset(monkeypatch):
    monkeypatch.delenv(paths.ENV_VAR, raising=False)
    assert paths.instance_home() == paths.DEFAULT_HOME


def test_default_applies_when_empty(monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, "")
    assert paths.instance_home() == paths.DEFAULT_HOME


def test_explicit_home_argument_overrides_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(paths.ENV_VAR, str(tmp_path / "ignored"))
    explicit = tmp_path / "explicit-home"
    assert paths.instance_home(explicit) == explicit


def test_helper_paths_are_under_the_instance_home(tmp_path):
    home = tmp_path / "home"
    assert paths.policy_path(home) == home / "policy.yaml"
    assert paths.health_path(home) == home / "state" / "health.json"
    assert paths.journal_path(home) == home / "state" / "observations.jsonl"
    assert paths.snapshots_dir(home) == home / "snapshots"
    assert paths.generated_config_path(home) == home / "config.yaml"
    assert paths.run_log_path(home) == home / "state" / "runs.log"
    assert paths.headroom_path(home) == home / "state" / "headroom.json"
    assert paths.headroom_lock_path(home) == home / "state" / "headroom.lock"


def test_headroom_lock_is_never_the_maintainer_lock(tmp_path):
    """ADR 0002's lock protects Health State. `headroom refresh` must
    never take it: codexbar takes 21-31 seconds to answer, and holding
    the maintainer lock that long would queue the Observation Journal
    watcher behind a codexbar sweep."""
    home = tmp_path / "home"
    assert paths.headroom_lock_path(home) != paths.lock_path(home)


def test_a_helper_creates_nothing(tmp_path):
    home = tmp_path / "untouched-home"
    paths.policy_path(home)
    paths.health_path(home)
    paths.journal_path(home)
    paths.snapshots_dir(home)
    paths.generated_config_path(home)
    paths.run_log_path(home)
    paths.headroom_path(home)
    paths.headroom_lock_path(home)
    assert not home.exists()


def test_ensure_instance_dirs_creates_state_and_snapshots(tmp_path):
    home = tmp_path / "new-home"
    paths.ensure_instance_dirs(home)
    assert (home / "state").is_dir()
    assert (home / "snapshots").is_dir()
