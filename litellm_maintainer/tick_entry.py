"""The scheduled tick's entry point, which survives its own import failing.

`cli.py` imports most of this package at module level, so a syntax error
or a bad import anywhere in it stops the CLI before any code runs. The
tick then writes NOTHING: no line in `runs.log`, no message a reader
would find. The log simply stops.

Measured 2026-07-30. A syntax error in `policy.py` did exactly that, and
two things followed. `runs.log` went silent, so the fault looked like a
scheduler that stopped firing rather than a program that would not start.
And launchd read the failure as `EX_CONFIG` and STOPPED RESPAWNING the
job, so fixing the code did not restore the schedule -- the job stayed
parked until someone unloaded and loaded it. Three hours passed before
anyone looked at `launchctl print`.

So this module runs first, and it imports almost nothing:

- `os`, `sys`, `datetime` and `Path` from the standard library.
- `litellm_maintainer.paths`, which imports `os` and `pathlib` and
  nothing else. It is the one project module safe to import here, and it
  is where the log's own location is already defined.

Everything that can fail is imported INSIDE `main`, under `try`. A
failure appends one line to `runs.log` naming what broke, then exits
non-zero, so the same reader who runs `tail runs.log` after a silent
schedule finds the cause in the file they already opened.

It never redacts, because it never has a credential to redact: the line
it writes carries an exception type and message from an import, not from
a request. `report.append_run_log` stays the only writer of a real run
line and keeps its own redaction (`report.py`, "Redaction is built into
the write path").
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _home_from_argv(argv: list[str]) -> Path | None:
    """Read `--home` out of `argv` with no argument parser.

    `argparse` lives in `cli.py`, which is the module that may not
    import. Reading the one flag this needs by hand keeps the failure
    path free of the code it is reporting on.

    Returns `None` when `--home` is absent, which `paths.instance_home`
    already reads as "use `$LITELLM_MAINTAINER_HOME`, else the default".
    """
    for index, argument in enumerate(argv):
        if argument == "--home" and index + 1 < len(argv):
            return Path(argv[index + 1])
        if argument.startswith("--home="):
            return Path(argument.split("=", 1)[1])
    return None


def _record_startup_failure(argv: list[str], error: BaseException) -> None:
    """Append one line to `runs.log` naming why the tick could not start.

    Best effort, and silent about its own faults: this runs while the
    program is already failing, and an exception here would replace a
    useful traceback on stderr with a useless one. The traceback still
    reaches `tick.err.log` either way, because the caller re-raises.
    """
    try:
        from litellm_maintainer.paths import run_log_path

        path = run_log_path(_home_from_argv(argv))
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).isoformat()
        line = (
            f"{stamp} did not start: {type(error).__name__}: {error}. "
            "The tick could not import its own code, so no run happened. "
            "Fix the fault, then RELOAD the scheduled job: launchd stops "
            "respawning a job that exits EX_CONFIG, so correcting the code "
            "alone does not restore the schedule.\n"
        )
        with open(path, "a") as handle:
            handle.write(line)
    except Exception:  # noqa: BLE001 - a failing reporter must not hide the fault
        pass


def main(argv: list[str] | None = None) -> int:
    """Run the CLI, recording a startup failure before it propagates.

    Returns the CLI's own exit code on success. On an import failure it
    writes the log line and re-raises, so `tick.err.log` still receives
    the full traceback: the line in `runs.log` says WHERE to look, and
    the traceback says what broke.
    """
    argv = list(sys.argv[1:]) if argv is None else argv
    try:
        from litellm_maintainer.cli import main as cli_main
    except BaseException as error:  # noqa: BLE001 - reported, then re-raised
        _record_startup_failure(argv, error)
        raise
    return cli_main(argv)


if __name__ == "__main__":  # pragma: no cover - exercised through the plist
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    raise SystemExit(main())
