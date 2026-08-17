"""The scheduled tick: a pure gate, plus the launchd plist it needs.

See CONTEXT.md, the spec's "Schedule" section, and
`.scratch/maintainer-v1/spec-corrections.md`, correction 9: the
scheduled tick chains probe, reduce, then plan. It never plans alone.
That chaining lives in `litellm_maintainer.cli.cmd_run`; this module
holds only the pure decision (`due`) and the plist the launchd job
runs.

`due` is PURE. It takes the schedule from Policy, the last run time,
whether the proxy answered, how stale Health State is, and the current
time — all as parameters. It performs no network call, no filesystem
read, no clock read and no environment read. A launchd job ticks more
often than `schedule.interval_minutes`, so `due` must be cheap and
return `run=False` on most calls.

The plist functions below (`build_plist_spec`, `render_plist`,
`install`, `uninstall`) write and remove one file. They never call
`launchctl`: this module only ever prints the `launchctl` command a
human (or the orchestrator) would run — see `launchctl_load_command`
and `launchctl_unload_command`. Turning the schedule on or off, and
changing the interval, are edits to Policy, read fresh on every tick;
the plist encodes neither, so no service reload follows either edit.

`build_headroom_plist_spec` builds a SECOND, separate launchd job:
`headroom refresh` on its own interval (Policy's
`headroom.interval_minutes`, default 15 minutes). It runs under
`HEADROOM_LABEL`, a distinct label from the tick's `DEFAULT_LABEL`, so
the two jobs install, run and uninstall independently.

This module MUST NOT import or call `refresh_headroom`
(`litellm_maintainer.headroom`). The tick's pipeline writes the
Generated Config; a refresh that cannot reach that pipeline cannot
rewrite the config, whatever a Reading says. Keeping this module blind
to `refresh_headroom` is what makes that guarantee hold — see the
headroom spec, decision 9, and `tests/test_schedule.py`'s import-guard
tests.
"""

from __future__ import annotations

import plistlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from litellm_maintainer.policy import Schedule
from litellm_maintainer.reduce import OfferingHealth

# A catch-up run fires only when the gap since the last run is at least
# this many intervals. An ordinary on-time tick's gap is close to one
# interval; a gap this much larger means the tick was blocked for a
# long stretch — the proxy-down case the spec names — not merely a
# slightly late tick.
CATCH_UP_MULTIPLIER = 2

DEFAULT_LABEL = "no.tallmaker.litellm-maintainer.tick"
# Shorter than any sane `schedule.interval_minutes` (the spec's own
# example ticks "more often than the interval"). This is a launchd
# constant, independent of Policy; it never changes when the operator
# edits `schedule.interval_minutes`.
DEFAULT_TICK_SECONDS = 60

# A distinct label for the headroom-refresh job. It is a SEPARATE
# launchd job from the tick (`DEFAULT_LABEL`), with its own plist file
# and its own interval, so installing, removing or changing one never
# touches the other.
HEADROOM_LABEL = "no.tallmaker.litellm-maintainer.headroom-refresh"

#: The shortest gap between two runs a Journal entry may trigger.
#:
#: A Journal entry elapses `schedule.interval_minutes`, which is what
#: makes a real failure act promptly. On its own that rule assumes
#: failures are occasional. They are not: one client retrying a
#: rate-limited Alias produced 90 entries in four minutes, and with the
#: tick firing every 60 seconds every one of those ticks ran the
#: pipeline (measured 2026-07-27).
#:
#: This floor bounds that. A burst of failures still reaches Health
#: State within five minutes, which is far inside the hour the interval
#: would otherwise impose, and a client stuck in a retry loop cannot
#: turn the tick into a continuous run.
JOURNAL_FLOOR = timedelta(minutes=5)


@dataclass(frozen=True)
class Decision:
    """Whether a scheduled tick should run now, and why.

    `run` states whether the caller's pipeline (probe, reduce, plan,
    write) should proceed this tick. `reason` is one short sentence for
    the run log — the operator reads it to learn why a tick fired or
    stayed quiet. `catch_up` is `True` only on the one tick that runs
    because the proxy returned after a long absence, never on an
    ordinary on-time run and never on the tick after a catch-up (see
    `due`'s docstring for why one is exactly enough).

    `journal_triggered` is `True` only on a run the Observation Journal
    caused before the interval elapsed. `cli.cmd_run` reads it to scope
    the sweep: such a run confirms the ambiguous entries and probes
    nothing else. A run whose interval had elapsed anyway is an
    ordinary run, even when the Journal also holds entries.
    """

    run: bool
    reason: str
    catch_up: bool = False
    journal_triggered: bool = False


def health_state_age(
    offerings: dict[str, OfferingHealth], *, now: datetime
) -> timedelta | None:
    """How long since ANY Offering in Health State last answered.

    Read `last_success_at`, the same field `reduce` and `prober`
    already trust for freshness. Returns `None` when no Offering has
    ever answered — Health State is then unbounded, treated by `due`
    the same as staleness beyond the configured maximum.
    """
    successes = [
        record.last_success_at
        for record in offerings.values()
        if record.last_success_at is not None
    ]
    if not successes:
        return None
    return now - max(successes)


def due(
    *,
    schedule: Schedule,
    last_run_at: datetime | None,
    proxy_up: bool,
    health_age: timedelta | None,
    now: datetime,
    journal_pending: bool = False,
) -> Decision:
    """Decide whether a scheduled tick should run now, and why.

    PURE: reads no clock, no filesystem, no environment, no network.
    `now` is the caller's own reading of the clock, passed in.

    Rules, in the order applied:

    1. A disabled schedule never runs. Nothing overrides this.
    2. An interval that has not elapsed since `last_run_at` does not
       run (`last_run_at` of `None` counts as elapsed: the first run
       ever has nothing to wait on). This check applies before the
       staleness rule, whatever Health State's age is. A launchd tick
       fires far more often than `schedule.interval_minutes`. Staleness
       is a reason to run despite a down proxy. It is never a reason to
       run before the interval itself has elapsed. Skipping this check
       would let a stale Health State turn every single tick into a
       full probe sweep. Health State starts stale on every fresh
       install, since "never recorded" counts as stale.
       `docs/gotchas.md` records the exact tick storm this would cause:
       a provider punishing it with `Worker local total request limit
       reached`.

       `journal_pending` is the one thing that elapses the interval
       early. It states that the proxy recorded a failure while
       serving real traffic and no run has folded it in yet. Without
       this, a quota exhaustion recorded one minute into a 60-minute
       interval waits 59 minutes to reach Health State -- slower than
       the tick it was built to beat, which defeats the Observation
       Journal entirely.

       This does not reopen the tick storm. The storm came from a
       CLOCK: every tick found Health State stale and swept every
       provider. A Journal entry is a real failure the operator just
       hit, so no traffic means no runs. The run it triggers is also
       narrowly scoped: it probes only what needs confirming, never a
       full sweep (`cli.cmd_run`).
    3. Health State older than `schedule.maximum_staleness_hours` (or
       never recorded at all) forces a run when Policy requires a
       running proxy and the proxy is down. This is the one case the
       spec names as the staleness rule overriding the proxy
       requirement (spec-corrections.md, correction 9's neighbour in
       the "Gate: due" list). Staleness overrides the PROXY requirement
       only. It never overrides the interval check above.
    4. A down proxy does not run, when `schedule.require_proxy` is
       `True`. It has no effect on the decision when `require_proxy` is
       `False`.
    5. Otherwise the tick runs. It is marked `catch_up` only when
       `schedule.require_proxy` is `True` and the gap since
       `last_run_at` is at least `CATCH_UP_MULTIPLIER` intervals. A gap
       that size means the proxy blocked several ticks in a row. It
       does not mean this one tick merely arrived on time.

    **Why "exactly one" catch-up needs no state of its own.** `due` is
    called fresh on every tick with no memory between calls. The catch-up
    tick's own pipeline, once it runs, records this tick's time as the
    new `last_run_at` (see `cli.cmd_run`). The very next tick therefore
    sees a small gap since `last_run_at`, so the interval check in step 3
    stops it before the catch-up condition in step 5 is ever evaluated
    again. One tick produces the large gap; running it is what erases
    the large gap for every tick after.
    """
    if not schedule.enabled:
        return Decision(run=False, reason="the schedule is disabled in Policy")

    interval = timedelta(minutes=schedule.interval_minutes)
    maximum_staleness = timedelta(hours=schedule.maximum_staleness_hours)
    is_stale = health_age is None or health_age >= maximum_staleness
    proxy_blocks = schedule.require_proxy and not proxy_up

    interval_elapsed = last_run_at is None or (now - last_run_at) >= interval
    journal_floor_elapsed = last_run_at is None or (now - last_run_at) >= JOURNAL_FLOOR
    if not interval_elapsed and not (journal_pending and journal_floor_elapsed):
        if journal_pending:
            return Decision(
                run=False,
                reason=(
                    "a recorded failure is waiting, but the last run was less "
                    f"than {int(JOURNAL_FLOOR.total_seconds() // 60)} minutes ago"
                ),
            )
        return Decision(run=False, reason="the interval has not elapsed")

    if is_stale and proxy_blocks:
        return Decision(
            run=True,
            reason=(
                "Health State is older than the configured maximum staleness "
                "(or has never been recorded); forcing a run despite the "
                "proxy being down"
            ),
        )

    if proxy_blocks:
        return Decision(
            run=False,
            reason="the proxy is down and Policy requires a running proxy",
        )

    is_catch_up = (
        schedule.require_proxy
        and last_run_at is not None
        and (now - last_run_at) >= interval * CATCH_UP_MULTIPLIER
    )
    if is_catch_up:
        return Decision(
            run=True,
            reason="the proxy returned after a long absence; running one catch-up",
            catch_up=True,
        )
    if not interval_elapsed:
        # Only `journal_pending` can reach here with the interval
        # unelapsed. Name that in the run log: an operator reading
        # "the interval has elapsed" against a 60-minute interval and
        # a run one minute after the last would rightly distrust it.
        return Decision(
            run=True,
            reason="the proxy recorded a failure that no run has folded in yet",
            journal_triggered=True,
        )
    return Decision(run=True, reason="the interval has elapsed")


# --- The launchd plist -------------------------------------------------


@dataclass(frozen=True)
class PlistSpec:
    """What the launchd job runs, and how often it ticks.

    Carries no `interval_minutes` and no `enabled` flag on purpose:
    both live in Policy, read fresh by `run` on every tick. Encoding
    either here would need a service reload on every Policy edit — the
    exact cost the spec's "Schedule" section rules out.
    """

    label: str
    program_arguments: tuple[str, ...]
    tick_seconds: int
    standard_out_path: str | None = None
    standard_error_path: str | None = None
    # What launchd must put in the job's environment. Empty for the tick,
    # which reads only files this package resolves by absolute path.
    #
    # A job inherits almost nothing. Measured 2026-07-30: `codexbar`
    # returned `usage: null` for the Claude provider in 10 of 10 runs
    # without `USER`, and a figure in 10 of 10 with it, so the refresh job
    # updated 5 of 6 Allowances and never said which one it missed.
    environment: tuple[tuple[str, str], ...] = ()


def build_plist_spec(
    *,
    python_executable: str,
    policy_path: str,
    feed_path: str,
    home: str | None = None,
    out_path: str | None = None,
    env_path: str | None = None,
    provider_modules_source: str | None = None,
    provider_modules_target: str | None = None,
    label: str = DEFAULT_LABEL,
    tick_seconds: int = DEFAULT_TICK_SECONDS,
    standard_out_path: str | None = None,
    standard_error_path: str | None = None,
) -> PlistSpec:
    """Build the launchd job spec that invokes `litellm_maintainer run`.

    `policy_path` and `feed_path` are passed through as literal
    `--policy` / `--feed` argument strings, never read here. Neither
    this function nor `render_plist` opens `policy_path`, so a Policy
    edit — including the schedule's own `enabled` and
    `interval_minutes` — never changes the bytes this produces.

    Warning: pass `out_path` whenever the proxy serves a config
    somewhere other than the instance directory. `run` defaults `--out`
    to the instance directory's own copy
    (`cli._default_out_path`), which the proxy never reads. A plist
    without this argument computes the right config every tick and
    writes it where nothing serves it, so the whole loop runs and
    changes nothing an operator can see.

    Warning: pass `env_path` as an ABSOLUTE path. `cli._default_env_path`
    looks for `.env.local` relative to the working directory, and
    launchd runs a job from `/`. Without it the tick resolves no
    credential, so `validate_config_before_write` refuses every write
    with "references credential variable ... which is not set". The
    safety rail makes that refusal loud rather than harmful, but the
    tick still never writes anything.

    `provider_modules_source` and `provider_modules_target` keep the
    proxy's copy of the failure callback current. Without them a change
    to `providers/*.py` reaches the proxy only when the operator copies
    it by hand.
    """
    # `tick_entry`, never `cli` directly. `cli` imports most of this
    # package at module level, so one syntax error anywhere stops it
    # before any code runs and the tick writes NOTHING -- no run line, no
    # error line, just a log that stops. `tick_entry` imports the standard
    # library and `paths`, catches the import failure, and records it.
    # Measured 2026-07-30; see that module's docstring.
    arguments: list[str] = [
        python_executable,
        "-m",
        "litellm_maintainer.tick_entry",
        "run",
        "--policy",
        policy_path,
        "--feed",
        feed_path,
    ]
    if home is not None:
        arguments += ["--home", home]
    if out_path is not None:
        arguments += ["--out", out_path]
    if env_path is not None:
        arguments += ["--env", env_path]
    if provider_modules_source is not None:
        arguments += ["--provider-modules-source", provider_modules_source]
    if provider_modules_target is not None:
        arguments += ["--provider-modules-target", provider_modules_target]
    return PlistSpec(
        label=label,
        program_arguments=tuple(arguments),
        tick_seconds=tick_seconds,
        standard_out_path=standard_out_path,
        standard_error_path=standard_error_path,
    )


def default_log_paths(home: str) -> tuple[str, str]:
    """Return the tick's stdout and stderr paths under `home`.

    A launchd job with neither writes its output nowhere, so an
    unattended tick that refuses, warns, or names an unclassified
    failure says it to no one. These sit beside the run log, in
    `state/`, where nothing the proxy's `--reload` watcher reads can
    see them.
    """
    return (f"{home}/state/tick.out.log", f"{home}/state/tick.err.log")


def default_headroom_log_paths(home: str) -> tuple[str, str]:
    """Return the headroom-refresh job's stdout and stderr paths under `home`.

    Named separately from `default_log_paths` so a codexbar failure
    never lands in the tick's own log, where an operator reading it for
    the tick would misread it as a tick defect.
    """
    return (
        f"{home}/state/headroom-refresh.out.log",
        f"{home}/state/headroom-refresh.err.log",
    )


def build_headroom_plist_spec(
    *,
    python_executable: str,
    policy_path: str,
    home: str | None = None,
    env_path: str | None = None,
    interval_minutes: int,
    label: str = HEADROOM_LABEL,
    standard_out_path: str | None = None,
    standard_error_path: str | None = None,
    user: str | None = None,
    path: str | None = None,
) -> PlistSpec:
    """Build the launchd job spec that invokes `litellm_maintainer headroom refresh`.

    A SEPARATE job from the tick's `build_plist_spec`, on its own label
    and its own `StartInterval`. `interval_minutes` comes from Policy's
    `headroom.interval_minutes` (default 15) — read it at install time,
    the way `build_plist_spec` reads `schedule.interval_minutes` for the
    tick. It is baked into `StartInterval` here, not read fresh per
    tick, because `headroom refresh` runs unconditionally on its own
    schedule: unlike the tick it has no Policy-read gate to apply the
    interval against, so there is nothing left for a fresher read to
    change.

    WHY 15 MINUTES BY DEFAULT: hourly is too coarse. The Claude
    300-minute window moves about 8% an hour, and the operator would
    watch it blind between reads. Under 5 minutes buys nothing: codexbar
    polls the providers on its own schedule underneath, so a shorter
    interval only re-copies a figure that did not move.

    This job runs `headroom refresh`, never `run`. `headroom refresh`
    takes Headroom State's own lock
    (`litellm_maintainer.paths.headroom_lock_path`) and never the
    maintainer lock, so this job never queues behind, or ahead of, the
    tick or the Observation Journal watcher (ADR 0002; headroom spec,
    decision 14).

    `policy_path` is passed through as a literal `--policy` argument
    string, never read here, the same rule `build_plist_spec` follows
    for the tick.

    Warning: pass `env_path` as an ABSOLUTE path. launchd runs a job
    from `/`, and `cli._default_env_path` resolves `.env.local` relative
    to the working directory, so a relative path here resolves nothing.

    `user` and `path` go into the job's own environment, because the
    source this job runs is a third-party binary and inherits nothing.
    Both were measured on 2026-07-30, and each failed differently:

    - Without `PATH` holding the binary's directory, a bare
      `headroom.command` resolves nothing. The job ran every 15 minutes
      for 17 hours, failed every time with "[Errno 2] No such file or
      directory", and Headroom State never moved. State the command as an
      absolute path in Policy as well; this is the belt, that is the
      braces.
    - Without `USER`, `codexbar` returned `usage: null` for the Claude
      provider in 10 of 10 runs, and a figure in 10 of 10 with it. The
      refresh then updated 5 of 6 Allowances and reported no error, since
      a Reading that cannot be read correctly keeps the previous one.

    The second is the worse failure: it is silent, partial, and it looks
    exactly like a provider the source happens not to know.
    """
    arguments: list[str] = [
        python_executable,
        "-m",
        "litellm_maintainer.tick_entry",
        "headroom",
        "refresh",
        "--policy",
        policy_path,
    ]
    if home is not None:
        arguments += ["--home", home]
    if env_path is not None:
        arguments += ["--env", env_path]
    environment: list[tuple[str, str]] = []
    if user:
        environment.append(("USER", user))
    if path:
        environment.append(("PATH", path))
    return PlistSpec(
        label=label,
        program_arguments=tuple(arguments),
        tick_seconds=interval_minutes * 60,
        standard_out_path=standard_out_path,
        standard_error_path=standard_error_path,
        environment=tuple(environment),
    )


def render_plist(spec: PlistSpec) -> bytes:
    """Render `spec` as a launchd property list, in XML plist form."""
    document: dict[str, Any] = {
        "Label": spec.label,
        "ProgramArguments": list(spec.program_arguments),
        "StartInterval": spec.tick_seconds,
        "RunAtLoad": False,
    }
    if spec.standard_out_path:
        document["StandardOutPath"] = spec.standard_out_path
    if spec.standard_error_path:
        document["StandardErrorPath"] = spec.standard_error_path
    if spec.environment:
        document["EnvironmentVariables"] = dict(spec.environment)
    return plistlib.dumps(document)


def plist_path(target_dir: Path, label: str = DEFAULT_LABEL) -> Path:
    """Return where `label`'s plist lives under `target_dir`.

    The real target is `~/Library/LaunchAgents/`; a test passes
    `tmp_path` instead. The file name is fixed per label, which is what
    makes `install` idempotent: two calls write the same path.
    """
    return target_dir / f"{label}.plist"


def install(target_dir: Path, spec: PlistSpec) -> Path:
    """Write `spec`'s plist into `target_dir`. Never calls `launchctl`.

    Idempotent: the file name is fixed by `spec.label`, so a second
    call overwrites the same file rather than adding a second one.
    Creates `target_dir` when it does not exist. Print
    `launchctl_load_command`'s result to tell the operator how to
    register the job; this function never runs it.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    path = plist_path(target_dir, spec.label)
    path.write_bytes(render_plist(spec))
    return path


def uninstall(target_dir: Path, label: str = DEFAULT_LABEL) -> Path | None:
    """Remove `label`'s plist from `target_dir`. Never calls `launchctl`.

    Returns the removed path, or `None` when nothing was installed —
    never raises for that case. Print `launchctl_unload_command`'s
    result before removing the file in a real uninstall; this function
    only removes the file.
    """
    path = plist_path(target_dir, label)
    if not path.exists():
        return None
    path.unlink()
    return path


def launchctl_load_command(path: Path) -> str:
    """The command that registers `path`'s job with launchd.

    This module never runs it. `cli.cmd_install` prints it so a human,
    or the orchestrator, decides whether the job actually starts.
    """
    return f"launchctl load -w {path}"


def launchctl_unload_command(path: Path) -> str:
    """The command that unregisters `path`'s job from launchd.

    This module never runs it. `cli.cmd_uninstall` prints it before it
    removes the plist file, so a human, or the orchestrator, decides
    whether the running job actually stops first.
    """
    return f"launchctl unload {path}"
