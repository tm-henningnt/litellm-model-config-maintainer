"""Tests for `litellm_maintainer.tick_entry`.

The tick's entry point exists so that a failure to IMPORT the CLI still
reaches `runs.log`. Measured 2026-07-30: a syntax error in `policy.py`
stopped the CLI before any code ran, `runs.log` went silent, and launchd
stopped respawning the job on `EX_CONFIG` -- so fixing the code did not
restore the schedule and nobody knew why.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from litellm_maintainer import tick_entry


def test_home_is_read_from_argv_without_an_argument_parser():
    # `argparse` lives in the module that may not import.
    assert tick_entry._home_from_argv(["run", "--home", "/tmp/x"]) == Path("/tmp/x")
    assert tick_entry._home_from_argv(["run", "--home=/tmp/y"]) == Path("/tmp/y")


def test_no_home_flag_defers_to_the_usual_lookup():
    # `None` is what `paths.instance_home` already reads as "use the
    # environment variable, else the default".
    assert tick_entry._home_from_argv(["run", "--policy", "p"]) is None


def test_a_startup_failure_writes_one_named_line_to_the_run_log(tmp_path):
    tick_entry._record_startup_failure(
        ["run", "--home", str(tmp_path)], SyntaxError("bad token")
    )

    line = (tmp_path / "state" / "runs.log").read_text()

    assert "did not start" in line
    assert "SyntaxError" in line
    assert "bad token" in line


def test_the_line_says_to_reload_the_job_not_only_to_fix_the_code(tmp_path):
    # The half that cost three hours: launchd stops respawning a job that
    # exits EX_CONFIG, so correcting the fault leaves the schedule parked.
    tick_entry._record_startup_failure(["--home", str(tmp_path)], ImportError("x"))

    line = (tmp_path / "state" / "runs.log").read_text()

    assert "RELOAD" in line
    assert "EX_CONFIG" in line


def test_the_run_log_directory_is_created_when_absent(tmp_path):
    home = tmp_path / "never-made"

    tick_entry._record_startup_failure(["--home", str(home)], ImportError("x"))

    assert (home / "state" / "runs.log").exists()


def test_a_reporter_that_cannot_write_never_hides_the_real_fault(tmp_path):
    # It runs while the program is already failing. An exception here
    # would replace a useful traceback with a useless one.
    unwritable = tmp_path / "file-not-a-dir"
    unwritable.write_text("x")

    tick_entry._record_startup_failure(["--home", str(unwritable)], ImportError("x"))


def test_an_import_failure_is_recorded_and_then_re_raised(tmp_path, monkeypatch):
    # `tick.err.log` must still receive the traceback: the log line says
    # WHERE to look, the traceback says what broke.
    recorded: list[BaseException] = []
    monkeypatch.setattr(
        tick_entry,
        "_record_startup_failure",
        lambda argv, error: recorded.append(error),
    )

    def explode(name, *args, **kwargs):
        if name == "litellm_maintainer.cli":
            raise ImportError("boom")
        return original(name, *args, **kwargs)

    import builtins

    original = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", explode)

    with pytest.raises(ImportError, match="boom"):
        tick_entry.main(["run"])

    assert len(recorded) == 1


def test_a_working_import_returns_the_cli_exit_code(monkeypatch):
    import litellm_maintainer.cli as cli_module

    monkeypatch.setattr(cli_module, "main", lambda argv: 3)

    assert tick_entry.main(["run"]) == 3
