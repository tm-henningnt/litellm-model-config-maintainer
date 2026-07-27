"""Poll the Observation Journal in the foreground, for debugging only.

Warning: this is NOT the production path. The launchd tick is. Do not
install this as a service.

Real traffic teaches the tool. A failure the operator hits while
working reaches Health State promptly, without waiting out the whole
schedule interval and without a synthetic Probe (CONTEXT.md,
"Observation Journal"; `.scratch/maintainer-v1/spec.md`, "Learning
from real traffic").

## The tick does this now

`cli.cmd_run` reads the Journal before its own due gate. An
unprocessed entry elapses `schedule.interval_minutes` on its own
(`schedule.due`, `journal_pending`), and the run it triggers probes
only what needs confirming. The launchd tick already fires every
`schedule.DEFAULT_TICK_SECONDS` seconds, so a recorded failure reaches
Health State within that long, under a process launchd already
supervises.

A KeepAlive daemon was the alternative. It reaches Health State a few
seconds sooner and costs a second process that can die silently, a
second contender for the maintainer lock (ADR 0002), and a second way
for this feature to stop working with nothing to show for it. The
whole reason this feature sat unused was silence; a supervised tick
has none.

`run_watch_command` and `cmd_watch` remain so an operator can watch the
Journal live while diagnosing one. They daemonise nothing.

`JournalWatcher` polls the Journal's size on a short interval, using
only the standard library — a filesystem-event dependency is not worth
it for a file that changes at most a few times a minute. The interval
and the clock are both injected, so a test drives the watcher with no
real waiting.

## What a new entry triggers

`classify` already ran inside the failure callback that wrote the
entry (see `providers/journal_failure_callback.py`), so every
Observation already carries a bucket:

- `self_healing`, `needs_operator`, `gone`: the failure identifies
  itself. A quota exhaustion is the clearest case. The watcher runs
  the maintainer at once; `reduce` and `plan` decide the rest,
  including the Passthrough Auth exemption (CONTEXT.md, "Passthrough
  Auth"; `reduce.py`, `_PASSTHROUGH_EXEMPT_REASONS`). This module does
  not re-implement that rule and must not.
- `inconclusive`: the failure is ambiguous — attributable to the
  operator's own request rate rather than to the Offering (see
  `classify.py`, "A known hazard"). The watcher calls the injected
  `confirm` callable for that Offering before running the maintainer,
  so one ambiguous call cannot change Health State on its own.

## The seam to the Prober

`JournalWatcher` never imports or calls the Prober. It takes a
`confirm: Callable[[str], None]` argument instead — one Offering key
in, and the call settles that Offering's health as a side effect
(a real Probe, folded into Health State) before returning. The Prober
must provide a function of that shape; `cli.build_confirm_probe` is
that function.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from litellm_maintainer.classify import INCONCLUSIVE
from litellm_maintainer.journal import read_observations

Clock = Callable[[], datetime]
Sleep = Callable[[float], None]
ConfirmProbe = Callable[[str], None]
RunMaintainer = Callable[[], None]

#: How often `run_forever` polls the Journal, in seconds, when the
#: caller sets no interval of its own.
DEFAULT_INTERVAL_SECONDS = 5.0


@dataclass(frozen=True)
class WatchResult:
    """What one `poll_once` call did, for a test or a log line to read.

    `new_observations` counts entries the Journal gained since the
    last poll. `confirmed` lists the Offering keys `poll_once` sent to
    `confirm`, in the order it sent them. `ran_maintainer` states
    whether `run_maintainer` was called this poll.
    """

    new_observations: int = 0
    confirmed: list[str] = field(default_factory=list)
    ran_maintainer: bool = False


class JournalWatcher:
    """Poll one Journal file and run the maintainer on a new entry.

    Track the file's size and modification time between polls. A poll
    that finds both unchanged does nothing — no read, no confirm, no
    run. Otherwise it compares the file's current entries against the
    entries it saw last time (see `journal.truncate_processed`): when
    the current entries still start with everything seen before, only
    the entries after that point are new; when they do not — a
    rotation replaced the file — every entry now present counts as
    new, since a rotation only ever removes entries `reduce` has
    already folded in.
    """

    def __init__(
        self,
        *,
        path: Path,
        confirm: ConfirmProbe,
        run_maintainer: RunMaintainer,
        clock: Clock = lambda: datetime.now(timezone.utc),
        sleep: Sleep = time.sleep,
        interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
    ) -> None:
        self._path = path
        self._confirm = confirm
        self._run_maintainer = run_maintainer
        self._clock = clock
        self._sleep = sleep
        self._interval_seconds = interval_seconds
        self._last_signature: tuple[int, int] | None = None
        self._last_observations: list = []

    def poll_once(self) -> WatchResult:
        """Check the Journal once and act on whatever is new.

        Return a `WatchResult` describing what happened, even when
        nothing did.

        Skip the read entirely when both the file's size and its
        modification time match the last poll — the cheap check the
        ticket asks for. Either one changing is enough to read the
        file and compare its entries against what this watcher saw
        last time.
        """
        try:
            stat = self._path.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
        except FileNotFoundError:
            signature = (0, 0)

        if signature == self._last_signature:
            return WatchResult()
        self._last_signature = signature

        read = read_observations(self._path)
        current = read.observations
        previous = self._last_observations

        if current[: len(previous)] == previous:
            new_observations = current[len(previous) :]
        else:
            # The earlier entries no longer match: a rotation replaced
            # the file. Everything present now is unprocessed (see the
            # class docstring).
            new_observations = current
        self._last_observations = current

        if not new_observations:
            return WatchResult()

        confirmed: list[str] = []
        for observation in new_observations:
            if observation.outcome.bucket == INCONCLUSIVE:
                self._confirm(observation.offering_id)
                confirmed.append(observation.offering_id)

        self._run_maintainer()

        return WatchResult(
            new_observations=len(new_observations),
            confirmed=confirmed,
            ran_maintainer=True,
        )

    def run_forever(self, *, stop: Callable[[], bool] = lambda: False) -> None:
        """Poll on the configured interval until `stop()` returns `True`.

        Sleep with the injected `sleep` function between polls, so a
        test can run this loop with no real waiting: pass a `sleep`
        that advances a fake clock instead of blocking, and a `stop`
        that returns `True` once the fake clock has advanced far
        enough.
        """
        while not stop():
            self.poll_once()
            self._sleep(self._interval_seconds)


def run_watch_command(
    *,
    path: Path,
    confirm: ConfirmProbe,
    run_maintainer: RunMaintainer,
    interval_seconds: float = DEFAULT_INTERVAL_SECONDS,
) -> None:
    """Run a `JournalWatcher` in the foreground until interrupted.

    This is the body of the `watch` subcommand, a debugging tool. It
    does not daemonise and it installs no service; the operator's
    terminal is the only thing keeping it running.

    Warning: do not supervise this as a service. The launchd tick is
    the production path -- see this module's docstring.
    """
    watcher = JournalWatcher(
        path=path,
        confirm=confirm,
        run_maintainer=run_maintainer,
        interval_seconds=interval_seconds,
    )
    watcher.run_forever()


# --- CLI wiring -----------------------------------------------------
#
# The pieces below are self-contained, so `cli.py` can wire the `watch`
# subcommand with a two-line change:
#
#   watch_parser = subparsers.add_parser("watch", help="Watch the Observation Journal")
#   litellm_maintainer.watcher.add_watch_arguments(watch_parser)
#   watch_parser.set_defaults(func=litellm_maintainer.watcher.cmd_watch)
#
# Nothing here imports `cli.py`, and `cli.py` imports this module only
# at that one call site.


def add_watch_arguments(parser: object) -> None:
    """Add the `watch` subcommand's arguments to an argparse parser."""
    parser.add_argument(  # type: ignore[attr-defined]
        "--interval",
        type=float,
        default=DEFAULT_INTERVAL_SECONDS,
        help="Poll interval in seconds (default: %(default)s)",
    )


def cmd_watch(args: object) -> int:
    """Run the `watch` subcommand: watch the Journal in the foreground.

    Build the `confirm` and `run_maintainer` callables and hand them to
    `run_watch_command`:

    - `confirm` runs one real, confirming Probe for the single Offering
      key and folds it into Health State, via
      `litellm_maintainer.cli.build_confirm_probe` -- this is the seam
      `prober.py` wires into; `JournalWatcher` still never imports the
      Prober itself.
    - `run_maintainer` calls `litellm_maintainer.cli.cmd_run`, never
      `cmd_generate`. `cmd_generate` plans from Health State the
      observation this poll just saw never reached (`cmd_generate`
      passes an empty or stale Health State on its own); `cmd_run`
      chains probe, reduce, then plan, so the observation is folded in
      before planning (spec-corrections.md, correction 9).
    """
    import argparse
    import sys

    from litellm_maintainer.feed import load_feed
    from litellm_maintainer.paths import journal_path
    from litellm_maintainer.policy import PolicyError, load_policy

    assert isinstance(args, argparse.Namespace)

    from litellm_maintainer import cli as cli_module

    env_path = Path(args.env) if getattr(args, "env", None) else cli_module._default_env_path()
    mapping = cli_module.build_redaction_map(env_path)
    home = Path(args.home) if getattr(args, "home", None) else None

    try:
        policy = load_policy(Path(args.policy))
    except PolicyError as exc:
        print(
            cli_module.redact(f"Policy is invalid: {exc}", mapping), file=sys.stderr
        )
        return 1
    try:
        feed = load_feed(Path(args.feed))
    except Exception as exc:  # noqa: BLE001 - a read failure reports, it never crashes
        print(
            cli_module.redact(f"Could not read the Feed: {exc}", mapping), file=sys.stderr
        )
        return 1

    confirm = cli_module.build_confirm_probe(policy=policy, feed=feed, home=home, mapping=mapping)

    def run_maintainer() -> None:
        cli_module.cmd_run(args)

    try:
        run_watch_command(
            path=journal_path(home),
            confirm=confirm,
            run_maintainer=run_maintainer,
            interval_seconds=getattr(args, "interval", DEFAULT_INTERVAL_SECONDS),
        )
    except KeyboardInterrupt:
        print("watch: stopped", file=sys.stderr)
    return 0
