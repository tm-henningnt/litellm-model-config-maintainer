"""Headroom State read, write and refresh.

Headroom State holds one Reading per mapped Allowance, captured out of
band by `headroom refresh`. Nothing else writes it, and nothing else
reads it yet — see CONTEXT.md, "Headroom State", and the headroom spec.

`refresh_headroom` is the read-modify-write this file's own lock
protects (`litellm_maintainer.paths.headroom_lock_path`). It NEVER takes
the maintainer lock at `paths.lock_path`: codexbar takes 21-31 seconds
to answer, and holding the maintainer lock that long would queue the
Observation Journal watcher behind a codexbar sweep. See ADR 0002.

A provider codexbar cannot answer for, or whose entry fails the shape
check in `litellm_maintainer.codexbar`, keeps its previous Reading.
Every other mapped Allowance still updates, the same read-modify-write
discipline `fetch.py` applies to the whole Feed Document, applied here
per Allowance instead of per file.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Mapping

from litellm_maintainer.codexbar import (
    CodexbarDocument,
    CodexbarEntryFailure,
    CodexbarError,
    CodexbarExtraWindow,
    CodexbarIdentity,
    CodexbarReading,
    CodexbarWindow,
    parse_codexbar_document,
)
from litellm_maintainer.lock import maintainer_lock
from litellm_maintainer.policy import Headroom
from litellm_maintainer.thresholds import Crossing, crossings

_SCHEMA_VERSION = 1

# Used only to decide whether a window is still live when detecting a
# threshold crossing. The caller passes the Policy figure; this default
# exists so a caller that reports no crossings needs no schedule.
_DEFAULT_STALENESS_HOURS = 24.0

DEFAULT_TIMEOUT_SECONDS = 40.0

# Takes the codexbar provider ids to ask for in one batched call, and the
# single provider id (or `None`) to ask for with `--all-accounts` instead.
# Exactly one of the two forms is used per call: `(providers, None)` asks
# the batched way, `([], provider_id)` asks for one multi-account
# provider's every account (ticket 11). Returns codexbar's stdout as
# text. Raises on any failure to run the tool at all: a missing binary, a
# non-zero exit, a timeout.
CodexbarRunner = Callable[[list[str], str | None], str]


@dataclass(frozen=True)
class HeadroomRecord:
    """One Allowance's stored Reading.

    `reading` is the Reading itself. `read_at` is OUR copy time, never
    codexbar's: `reading.updated_at` is codexbar's own timestamp, and a
    Reading ages from that, not from when we last read it (headroom
    spec, decision 8; ticket 04 owns the computation, this file only
    stores both).
    """

    allowance_id: str
    source: str
    reading: CodexbarReading
    read_at: str


@dataclass(frozen=True)
class HeadroomState:
    """Headroom State's full contents: one record per mapped Allowance."""

    records: dict[str, HeadroomRecord] = field(default_factory=dict)


# --- The Binding Window derivation ----------------------------------------
#
# `entitlements` (ticket 04) and `guidance` (ticket 05) both need the same
# answer to "which window binds, and is it still live". It lives here, next
# to the Reading it reads, so both commands import the same function rather
# than growing two readings of one rule. See CONTEXT.md, "Binding Window",
# and the headroom spec, decisions 2 and 8.


@dataclass(frozen=True)
class BindingWindow:
    """The window inside a Reading with the highest used share (CONTEXT.md).

    Built by `binding_window`, which considers only `primary`, `secondary`
    and `tertiary` and skips a void one. Carries only what a Route or an
    Allowance publishes about it: the used share, the window's length, and
    when it resets.
    """

    used_percent: float
    window_minutes: float | None
    resets_at: str | None


def _parse_timestamp(raw: str | None) -> datetime | None:
    """Parse an ISO-8601 timestamp codexbar publishes, or `None`.

    Codexbar publishes no contract for its own timestamp format. A
    timestamp that fails to parse reads the same as one that is absent:
    unknown, so `window_is_void` falls back to its staleness rule instead
    of trusting a value that did not parse.

    A NAIVE result (no `Z`, no offset) is also unparsable, not "assume
    UTC". Confirmed 2026-07-29: a naive value reaches `window_is_void`'s
    comparison against the timezone-AWARE `now` and raises `TypeError:
    can't compare offset-naive and offset-aware datetimes`, which took
    down both `guidance` and `entitlements` on every run until the state
    file was deleted by hand. This docstring already states the rule for
    an unreadable timestamp -- "reads the same as one that is absent:
    unknown" -- so a naive value falls under it too, rather than being
    guessed into UTC.
    """
    if raw is None:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed


def window_is_void(
    window: CodexbarWindow,
    *,
    reading_updated_at: str | None,
    now: datetime,
    maximum_staleness_hours: float,
) -> bool:
    """Whether `window`'s figure describes a period that has already ended.

    A window past its own `resets_at` has refilled, so its stored
    `used_percent` is VOID, not merely old (headroom spec, decision 8):
    it must not report a used share, and it must not bind.

    Where a window states no `resets_at`, it expires
    `maximum_staleness_hours` after the READING's own timestamp — never
    ours. `reading_updated_at` is codexbar's clock; `now` is compared
    against that. Measured 2026-07-28: codexbar's `updatedAt` advanced 52
    minutes between two of our reads with no call from us, so our own
    copy time understates how old a figure really is.

    A window with neither a readable `resets_at` nor a readable Reading
    timestamp has no way to bound its age, so it reads void: an unknown
    age must never be mistaken for a fresh one.

    A `resets_at` at or before the Reading's own timestamp states no
    reset. The source measured this window at `reading_updated_at`, so a
    reset it already passed by then cannot be the reset this figure is
    counting down to. Read such a value as absent, and fall back to the
    staleness rule.

    Measured 2026-07-28: Gemini stated `usedPercent: 100`,
    `resetsAt: "1970-01-01T00:00:00Z"` and, in the same window,
    `resetDescription: "Resets soon"`. The epoch is an unset sentinel and
    not a reset in the past. Read literally, that window went void and
    its two idle siblings of the same length bound instead, so the
    highest figure in the Reading was discarded and a lower one
    published in its place.

    That is the fault this rule fixes: a sentinel must not silence a
    figure.

    Gemini is not mapped, for a separate reason. Its three slots hold one
    quota per model, not nested time windows, so the worst-of rule in
    `binding_window` does not apply to it. See `docs/gotchas.md`,
    "codexbar's three window slots do not mean one thing".
    """
    updated_at = _parse_timestamp(reading_updated_at)
    resets_at = _parse_timestamp(window.resets_at)
    if resets_at is not None and (updated_at is None or resets_at > updated_at):
        return resets_at <= now
    if updated_at is None:
        return True
    return now - updated_at >= timedelta(hours=maximum_staleness_hours)


# codexbar's three named slots, in the order `binding_window` considers
# them. A provider whose `headroom.sources` entry names one of these under
# `windows` turns that slot from a parent window into a Sub-allowance — see
# ticket 09 and `docs/gotchas.md`, "codexbar's three window slots do not
# mean one thing".
SLOT_NAMES = ("primary", "secondary", "tertiary")


def slot_id_for_health_key(
    members: Mapping[str, tuple[str, ...]] | None, health_key: str
) -> str | None:
    """The declared slot id `health_key` draws on, or `None`.

    `members` is Policy's `headroom.sources.<id>.members` mapping for ONE
    Allowance (`Headroom.source_members`): each declared slot id to the
    Health Keys that draw on it (ticket 10). `health_key` is a Route's own
    Health Key — a Feed Offering's id, or a Declared Offering's Alias
    (`DeclaredOffering.health_key`).

    The caller passes the resolved slot id straight to
    `route_binding_window`'s `sub_allowance_window_id`, which matches it
    against `slot_windows` exactly as it always has: this function only
    answers "which slot, if any", never how the figure is read.

    Returns `None` when `members` is empty or absent (no Sub-allowance
    declared, or none assigned yet) or when no slot lists `health_key`: an
    ordinary Route, or an admitted Offering nobody has assigned to a slot
    (`doctor` reports the gap). A Health Key belongs in one slot; where
    Policy lists it under two by mistake, the first match in `members`'
    own iteration order wins.
    """
    if not members:
        return None
    for slot_id, health_keys in members.items():
        if health_key in health_keys:
            return slot_id
    return None


def binding_window(
    reading: CodexbarReading,
    *,
    now: datetime,
    maximum_staleness_hours: float,
    named_slots: frozenset[str] = frozenset(),
) -> BindingWindow | None:
    """The Binding Window: the live window with the highest used share.

    Considers `primary`, `secondary` and `tertiary`, MINUS whichever of
    those `named_slots` lists (headroom spec, decision 2; ticket 09).
    `extra_windows` measure Sub-allowances and bind only for the Routes
    inside them — `route_binding_window` owns that join (ticket 06), and
    THIS function must never let one affect the figure it returns: it is
    the parent Allowance's figure, read by every Route on the Allowance,
    sub-allowance or not.

    `named_slots` is empty for every provider mapped through a plain
    `headroom.sources` string, so this reads exactly as it always has —
    nested time windows, where the worst live one binds the whole
    Allowance. A provider whose slots hold one quota per MODEL instead
    (Gemini, measured 2026-07-29) names each such slot in Policy, and this
    function then excludes it: a named slot is a Sub-allowance, and a
    Sub-allowance's own draw says nothing about its parent (CONTEXT.md,
    "Sub-allowance"). Where every slot is named, no window is left to
    bind on, and this returns `None` — correct, because nothing then caps
    the Allowance as a whole.

    A reader that instead picked one named window gets ClinePass wrong:
    measured 2026-07-28, its `primary` and `secondary` both read 0% while
    `tertiary` read 100% fully drawn. This picks the WORST live window
    among the ones still eligible, so ClinePass binds at 100%.

    Returns `None` when there is no live window to bind on: every
    eligible window is absent, or every one that exists is void
    (`window_is_void`). Either state yields no Headroom — never a 0%
    that would read as healthy.
    """
    live = [
        window
        for slot in SLOT_NAMES
        if slot not in named_slots
        and (window := getattr(reading, slot)) is not None
        and not window_is_void(
            window,
            reading_updated_at=reading.updated_at,
            now=now,
            maximum_staleness_hours=maximum_staleness_hours,
        )
    ]
    if not live:
        return None
    best = max(live, key=lambda window: window.used_percent)
    return BindingWindow(
        used_percent=best.used_percent,
        window_minutes=best.window_minutes,
        resets_at=stated_reset(best, reading_updated_at=reading.updated_at),
    )


def route_binding_window(
    reading: CodexbarReading,
    *,
    sub_allowance_window_id: str | None,
    now: datetime,
    maximum_staleness_hours: float,
    slot_windows: Mapping[str, str] | None = None,
) -> BindingWindow | None:
    """The Binding Window for ONE Route: the worse of the parent's figure
    and its own Sub-allowance's figure.

    Containment runs one way (CONTEXT.md, "Sub-allowance"): the parent
    Allowance's exhaustion reaches every Sub-allowance inside it, so the
    parent's `binding_window` always takes part — computed here with
    every slot `slot_windows` names EXCLUDED, because a named slot left
    the parent computation (ticket 09). The Sub-allowance's own
    exhaustion says nothing about the parent, so its window binds ONLY
    the Route that names it in `sub_allowance_window_id` — never a
    sibling Route on the same Allowance.

    `sub_allowance_window_id` is `None` for an ordinary Route, and for a
    Sub-allowance that names no window: both read the parent's figure
    alone, exactly as `binding_window` always has.

    `slot_windows` is Policy's own `headroom.sources.<id>.windows`
    mapping for this Allowance (`Headroom.source_windows`), slot name to
    the operator's Sub-allowance id — `None` or empty for every provider
    mapped through a plain string, which keeps this function reading
    exactly as it always has. `sub_allowance_window_id` resolves first
    against `reading.extra_windows`, as it always did; where that finds
    nothing, it resolves against `slot_windows` instead, reading the
    NAMED slot (`primary`, `secondary` or `tertiary`) directly off
    `reading` rather than off `extra_windows`. Gemini is the measured
    case: `gemini-pro` names the `primary` slot, so a Route naming it
    binds on `reading.primary` alone.

    Measured 2026-07-28: Claude's parent window (`secondary`) read 82%
    while its `claude-weekly-scoped-fable` extra window read 59%. The
    parent is worse here, so it wins either way. The case this function
    exists for is the reverse — the spec's example, and the reason the
    Sub-allowance was declared at all: fable can run dry while the rest
    of the Allowance has room. Reading the parent alone in that case would
    report a Route that is about to refuse as though it had headroom to
    spare, which is a permissive lie.

    Reuses `binding_window`, `window_is_void` and `stated_reset` — this
    is the one join ticket 05 left for ticket 06, not a second
    derivation.
    """
    named_slots = frozenset(slot_windows) if slot_windows else frozenset()
    parent = binding_window(
        reading,
        now=now,
        maximum_staleness_hours=maximum_staleness_hours,
        named_slots=named_slots,
    )
    if sub_allowance_window_id is None:
        return parent

    extra = next(
        (w for w in reading.extra_windows if w.id == sub_allowance_window_id), None
    )
    sub_window: CodexbarWindow | None = extra.window if extra is not None else None
    if sub_window is None and slot_windows:
        # No `extraRateWindows` entry matched. Try Policy's own slot
        # mapping instead: `sub_allowance_window_id` may name an operator-
        # chosen id for `primary`, `secondary` or `tertiary`, never one of
        # codexbar's own extra-window ids.
        slot = next(
            (name for name, sub_id in slot_windows.items() if sub_id == sub_allowance_window_id),
            None,
        )
        if slot is not None:
            sub_window = getattr(reading, slot)

    sub: BindingWindow | None = None
    if sub_window is not None and not window_is_void(
        sub_window,
        reading_updated_at=reading.updated_at,
        now=now,
        maximum_staleness_hours=maximum_staleness_hours,
    ):
        sub = BindingWindow(
            used_percent=sub_window.used_percent,
            window_minutes=sub_window.window_minutes,
            resets_at=stated_reset(sub_window, reading_updated_at=reading.updated_at),
        )

    if parent is None:
        return sub
    if sub is None:
        return parent
    return sub if sub.used_percent > parent.used_percent else parent


def stated_reset(window: CodexbarWindow, *, reading_updated_at: str | None) -> str | None:
    """`window`'s reset time, or `None` where it states none.

    `window_is_void` reads a `resets_at` at or before the Reading's own
    timestamp as no reset at all. Publish that decision instead of
    echoing the value back.

    Measured 2026-07-28: one provider stated `resetsAt:
    "1970-01-01T00:00:00Z"` beside `resetDescription: "Resets soon"`.
    Republishing the epoch tells a caller the window already refilled,
    which is the opposite of what we just decided about it. A caller that
    reads a reset time uses it to choose when to try again, so a wrong
    one is worse than none.
    """
    resets_at = _parse_timestamp(window.resets_at)
    if resets_at is None:
        return None
    updated_at = _parse_timestamp(reading_updated_at)
    if updated_at is not None and resets_at <= updated_at:
        return None
    return window.resets_at


# How many missed refresh cycles turn a record's OWN copy time
# (`HeadroomRecord.read_at`) into a warning. `refresh_headroom` only
# advances `read_at` when a run finds a fresh match; a provider that keeps
# erroring, or a job that stopped running, leaves it exactly where it was
# (see `refresh_headroom`'s docstring). Four intervals give the job room
# for an ordinary missed tick or two before the operator is told anything.
HEADROOM_STALE_MULTIPLIER = 4


def headroom_source_warnings(
    *, headroom_policy: Headroom, headroom_state: HeadroomState, now: datetime
) -> tuple[str, ...]:
    """One warning per mapped Allowance whose Headroom stopped refreshing.

    This is a JOB-HEALTH signal, not a window-validity one: it reads
    `HeadroomRecord.read_at`, OUR OWN copy time, never a window's
    `resets_at`. The two are independent on purpose. A window that went
    VOID because its own reset passed is normal and self-correcting
    (`window_is_void`); it needs no warning, whatever `read_at` says. A
    record whose `read_at` has not moved in `HEADROOM_STALE_MULTIPLIER`
    refresh intervals means the job did not write it recently — either it
    stopped running, or codexbar has been erroring for this provider on
    every run since (`refresh_headroom` keeps the previous Reading on
    either fault). Both read the same from here: a fault, not a refill.

    A source with no record at all warns too: Policy declares it, but
    `headroom refresh` has never once produced a match for it, which is
    the same "no Headroom" symptom this whole ticket exists to name.

    An unparsable `read_at` warns as well. An unreadable age must never
    be mistaken for a fresh one — the same rule `window_is_void` applies
    to an unparsable `resets_at`.

    Returns `()` when Policy declares no source at all: the capability is
    off, and silence is the correct output for it.
    """
    if not headroom_policy.sources:
        return ()

    threshold = timedelta(minutes=headroom_policy.interval_minutes * HEADROOM_STALE_MULTIPLIER)
    warnings: list[str] = []
    for allowance_id in sorted(headroom_policy.sources):
        record = headroom_state.records.get(allowance_id)
        if record is None:
            warnings.append(
                f"{allowance_id!r} declares a headroom_source but has never been "
                "refreshed; run 'headroom refresh', or check that its refresh "
                "job is installed and running"
            )
            continue
        read_at = _parse_timestamp(record.read_at)
        if read_at is None:
            warnings.append(
                f"{allowance_id!r}'s Headroom carries an unreadable 'read_at'; "
                "treat its figure as unknown, not fresh"
            )
            continue
        age = now - read_at
        if age >= threshold:
            warnings.append(
                f"{allowance_id!r}'s Headroom was last refreshed "
                f"{_format_warning_age(age)} ago, past "
                f"{HEADROOM_STALE_MULTIPLIER} refresh intervals "
                f"({headroom_policy.interval_minutes} min each); the refresh "
                "job may have stopped, or codexbar may be erroring for it "
                "on every run"
            )
    return tuple(warnings)


def _format_warning_age(age: timedelta) -> str:
    """Render a `timedelta` for a warning line, in whole minutes or hours."""
    minutes = age.total_seconds() / 60
    if minutes < 60:
        return f"{minutes:.0f} min"
    return f"{minutes / 60:.1f} h"


def format_age(age_seconds: float | None) -> str:
    """Render a Headroom's age for an operator, from CODEXBAR's own clock.

    `None` reads `age unknown`, never `0 min`. An unreadable timestamp must
    not read as a fresh one.

    Lives here, beside the Binding Window derivation, for the reason stated
    above it: `entitlements` and `guidance` both render this figure, so one
    function serves both rather than two readings of one rule drifting
    apart.
    """
    if age_seconds is None:
        return "age unknown"
    minutes = age_seconds / 60
    if minutes < 60:
        return f"age {minutes:.0f} min"
    return f"age {minutes / 60:.1f} h"


def format_used_percent(used_percent: float) -> str:
    """Render a Binding Window's used share, never claiming 100% early.

    Lives beside `format_age` for the same reason: `entitlements` and
    `guidance` both render this figure, so one function serves both
    rather than two readings of one rounding rule drifting apart.

    `f"{value:.0f}%"` rounds 99.5 up to "100%" (Python's round-half-to-
    even picks 100, the nearest even integer), which reads as fully
    drawn while `_demoted_by_headroom` only fires at the raw value
    reaching 100 exactly. A caller could then see "100%" and a
    `recommendable` Route side by side, disagreeing about the same
    number. Round normally below 99.5; cap the display at 99% for
    anything under a true 100, so the two never state opposite things.
    """
    if used_percent >= 100:
        return "100%"
    rounded = round(used_percent)
    if rounded >= 100:
        rounded = 99
    return f"{rounded}%"


def reading_age_seconds(reading: CodexbarReading, *, now: datetime) -> float | None:
    """How old `reading` is, measured from CODEXBAR's own timestamp.

    Never from `HeadroomRecord.read_at`: that is our copy time, and
    codexbar polls on its own schedule underneath it (headroom spec,
    decision 8). `None` when `updated_at` is absent or unparsable — an
    unknown age, never a zero one.

    Clamped at zero. Codexbar's clock can run ahead of ours, and a
    Reading is never taken in our own future: a negative result is clock
    skew, not evidence about the Reading's freshness or staleness. Left
    unclamped it rendered "age -120 min", which is not a real age and
    invites the opposite misreading — that a stale figure is somehow
    fresher than "just now". Zero is the closest true statement: this
    Reading is not older than we can currently tell.
    """
    updated_at = _parse_timestamp(reading.updated_at)
    if updated_at is None:
        return None
    age = (now - updated_at).total_seconds()
    return max(age, 0.0)


def read_headroom(path: Path) -> HeadroomState:
    """Read Headroom State from `path`.

    A missing file reads as empty, not an error: the first run has
    none. A file that is not valid JSON, or that has no `records`
    mapping, also reads as empty — the same rule `read_health` applies
    to Health State, because a file broken this badly teaches us
    nothing a restart from empty does not.

    A single bad record is different. Skip it, and keep every record
    that does parse: one bad record must never discard every good one.
    """
    try:
        raw_text = path.read_text()
    except FileNotFoundError:
        return HeadroomState()

    try:
        raw = json.loads(raw_text)
        raw_records = raw["records"]
        if not isinstance(raw_records, dict):
            raise TypeError("'records' is not a mapping")
    except Exception:
        return HeadroomState()

    records: dict[str, HeadroomRecord] = {}
    for allowance_id, value in raw_records.items():
        try:
            records[allowance_id] = _record_from_json(allowance_id, value)
        except Exception:
            continue
    return HeadroomState(records=records)


def write_headroom(path: Path, state: HeadroomState) -> None:
    """Write Headroom State to `path`, atomically.

    Write a temporary file beside `path`, then rename it into place, the
    same pattern `write_health` uses for Health State: a reader never
    observes a partial file.
    """
    document = {
        "schema_version": _SCHEMA_VERSION,
        "records": {
            allowance_id: _record_to_json(record)
            for allowance_id, record in state.records.items()
        },
    }

    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(document, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise


def real_codexbar_runner(
    command: str, *, timeout: float = DEFAULT_TIMEOUT_SECONDS
) -> CodexbarRunner:
    """Build the real codexbar runner: one subprocess call per invocation.

    `command` is Policy's `headroom.command`, defaulting to `codexbar`.
    Ask only for the given providers; never `--provider all`. Measured
    2026-07-28: `--provider all` took 21-31 seconds, and four mapped
    providers took 24 seconds.

    `all_accounts_provider`, when given, asks for that ONE provider with
    `--all-accounts` instead of the batched `providers` list — codexbar's
    own `--help` states "Account selection requires a single provider", so
    the two forms never combine in one call (ticket 11).

    READ THE OUTPUT, NEVER THE EXIT CODE. Measured 2026-07-29 on the
    operator's machine: `codexbar --format json --provider
    claude,clinepass,gemini,opencodego` exited 1 on three runs of three,
    wrote `[codex notify] remoteControl/status/changed` to stderr, and
    wrote a complete, valid JSON array of nine Readings to stdout. The
    exit code carried no information about the answer.

    Trusting it discarded every batched Reading on every run, so the
    capability reported nothing and kept reporting nothing. An answer that
    parses is an answer. A non-zero exit beside one is reported to the
    caller, never a reason to throw the document away.
    """

    def runner(providers: list[str], all_accounts_provider: str | None = None) -> str:
        args = [command, "--format", "json"]
        if all_accounts_provider is not None:
            args += ["--provider", all_accounts_provider, "--all-accounts"]
        elif providers:
            args += ["--provider", ",".join(providers)]
        completed = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, check=False
        )
        if _parses_as_readings(completed.stdout):
            return completed.stdout
        raise RuntimeError(
            f"{command} exited {completed.returncode} and wrote no readable document: "
            f"{completed.stderr.strip()}"
        )

    return runner


def _parses_as_readings(raw_output: str) -> bool:
    """Whether `raw_output` holds a document `parse_codexbar_document` reads.

    The shape check stays in `codexbar.py`. This asks only the narrower
    question the runner needs: did the tool answer at all, whatever it
    then said about its own exit status.
    """
    try:
        return isinstance(json.loads(raw_output), list)
    except (ValueError, TypeError):
        return False


def query_codexbar_readings(
    *,
    provider_ids: list[str],
    all_accounts_providers: frozenset[str],
    runner: CodexbarRunner,
) -> tuple[CodexbarDocument, frozenset[str], tuple[str, ...]]:
    """Ask codexbar for `provider_ids`, in as many calls as the mapping needs.

    Every provider named in `all_accounts_providers` leaves the batched
    call and gets its own `--all-accounts` call (ticket 11): codexbar's
    own `--help` states "Account selection requires a single provider", so
    a multi-account provider cannot ride the one call every other
    provider shares. Its Readings merge into the same `CodexbarDocument`
    the batched call produced — the caller's own source-key match then
    attaches each one to its Allowance exactly as it always has (ADR
    0009's join key already discriminates two accounts sharing one
    `providerID`, with no change to it at all).

    Never probes: a provider not named in `all_accounts_providers` never
    sees `--all-accounts`, whatever its `sources` entries look like.

    Returns `(document, failed_providers, call_errors)`. Each call —
    the one batched call, and one per multi-account provider — is
    isolated from the others: a runner exception or a
    `CodexbarShapeError` on ONE call is caught here and named in
    `failed_providers` (the provider ids that call covered) and
    `call_errors` (one message per failed call), while every OTHER
    call's Readings still reach `document`. A failed batched call
    therefore never discards an extra call's Readings, and a failed
    extra call never discards the batched providers' Readings — the
    same per-provider discipline `refresh_headroom` already applies
    inside one document, extended to the call boundary itself.
    """
    batched_ids = [p for p in provider_ids if p not in all_accounts_providers]
    extra_ids = sorted(p for p in provider_ids if p in all_accounts_providers)

    readings: list[CodexbarReading] = []
    entry_failures: list[CodexbarEntryFailure] = []
    failed_providers: set[str] = set()
    call_errors: list[str] = []

    if batched_ids:
        try:
            document = parse_codexbar_document(runner(batched_ids, None))
        except Exception as exc:  # noqa: BLE001 - isolated to these providers only
            failed_providers.update(batched_ids)
            call_errors.append(
                f"codexbar did not answer for {', '.join(batched_ids)}: {exc}"
            )
        else:
            # Drop any Reading for a provider that gets its own call, even
            # though this call did not ask for one.
            #
            # Measured 2026-07-29: `--provider claude,clinepass,gemini,
            # opencodego` returned nine providers, `codex` among them.
            # codexbar's own `--help` states it "Honors your in-app
            # toggles", so `--provider` widens the answer rather than
            # narrowing it.
            #
            # Keeping those entries made one account arrive twice, once
            # from each call. The ambiguity guard then read two Readings
            # under one source key and kept the previous one. That guard
            # is right — two accounts must never share a key — so the
            # duplicate must not reach it. A provider asked about
            # separately is answered by that call alone.
            readings.extend(
                reading
                for reading in document.readings
                if reading.identity.provider_id not in all_accounts_providers
            )
            entry_failures.extend(document.failures)

    for provider_id in extra_ids:
        try:
            document = parse_codexbar_document(runner([], provider_id))
        except Exception as exc:  # noqa: BLE001 - isolated to this provider only
            failed_providers.add(provider_id)
            call_errors.append(
                f"codexbar did not answer for {provider_id!r} (--all-accounts): {exc}"
            )
        else:
            # Keep only the provider this call asked about. The guard runs
            # both ways: the batched call drops a provider answered
            # separately, and this call drops every provider it did not
            # ask for.
            #
            # Measured 2026-07-29: `--provider` WIDENS codexbar's answer
            # rather than narrowing it, because it "Honors your in-app
            # toggles" — four providers asked, nine returned. An
            # `--all-accounts` call narrows correctly today, but relying
            # on that is the same assumption that already cost one
            # Allowance its figure. A call made to answer about one
            # provider answers about that provider alone.
            readings.extend(
                reading
                for reading in document.readings
                if reading.identity.provider_id == provider_id
            )
            entry_failures.extend(document.failures)

    merged = CodexbarDocument(readings=tuple(readings), failures=tuple(entry_failures))
    return merged, frozenset(failed_providers), tuple(call_errors)


@dataclass(frozen=True)
class RefreshOutcome:
    """What one `headroom refresh` did.

    `ran` is `False` only when Policy declared no source at all, so the
    command did nothing. `updated` and `kept_previous` name Allowance
    ids; `failures` names the codexbar entries that failed their shape
    check, each attributed to its provider where that survived.
    """

    ran: bool
    message: str
    updated: tuple[str, ...] = ()
    kept_previous: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    # Thresholds a Binding Window crossed since the previous Reading.
    # Reported, never acted on here: this module writes Headroom State
    # and delivers no notification.
    crossings: tuple[Crossing, ...] = ()


def refresh_headroom(
    *,
    headroom_policy: Headroom,
    path: Path,
    lock_path: Path,
    runner: CodexbarRunner,
    now: datetime,
    maximum_staleness_hours: float = _DEFAULT_STALENESS_HOURS,
) -> RefreshOutcome:
    """Run one `headroom refresh`: ask codexbar, merge, and write.

    Holds `lock_path` for the whole read-modify-write, NEVER the
    maintainer lock at `paths.lock_path` — see the module docstring and
    ADR 0002.

    A provider codexbar could not answer for, or whose entry failed the
    shape check, keeps its previous Reading; every other mapped
    Allowance still updates. A total failure to run codexbar at all, or
    a document that is not a JSON list, keeps every previous Reading
    unchanged.

    Two codexbar entries matching one declared source is treated the
    same way: the previous Reading is kept and the ambiguity is named in
    `failures`, never resolved by picking whichever entry iteration
    happened to reach last (headroom spec, decision 4: "a reading that
    cannot name its credential attaches to nothing").

    A provider named in `headroom_policy.all_accounts_providers` leaves
    the batched call entirely and gets its own `--all-accounts` call
    instead (ticket 11): codexbar's own `--help` states "Account
    selection requires a single provider", so a multi-account provider
    cannot ride the one call every other provider shares. Its Readings
    merge into the same set the batched call produced, and the ordinary
    per-source match above then attaches each one to its own Allowance —
    no second join. `query_codexbar_readings` isolates the two calls from
    each other: a failed extra call keeps the previous Reading for every
    Allowance mapped to THAT provider and never touches an Allowance the
    batched call updated, and a failed batched call is the same rule run
    the other way.

    An Allowance Policy no longer declares is pruned from the stored
    records here, not merely left unread. Reading alone cannot fix this:
    `entitlements` and `guidance` both key their own lookup by Allowance
    id, so a record surviving under a removed id would still be found by
    a caller that does not also check `record.source` against Policy
    (headroom spec, decision 2b).
    """
    if not headroom_policy.sources:
        return RefreshOutcome(ran=False, message="Policy declares no headroom sources; nothing to do")

    provider_ids = sorted({provider_id_from_source(source) for source in headroom_policy.sources.values()})
    all_accounts_providers = frozenset(headroom_policy.all_accounts_providers)

    with maintainer_lock(lock_path):
        previous = read_headroom(path)
        # Drop every record whose Allowance Policy no longer maps, so the
        # file never accumulates a Reading for an Allowance the operator
        # removed (headroom spec, decision 2b).
        records = {
            allowance_id: record
            for allowance_id, record in previous.records.items()
            if allowance_id in headroom_policy.sources
        }

        document, failed_providers, call_errors = query_codexbar_readings(
            provider_ids=provider_ids,
            all_accounts_providers=all_accounts_providers,
            runner=runner,
        )

        readings_by_source_key: dict[str, list[CodexbarReading]] = {}
        for reading in document.readings:
            readings_by_source_key.setdefault(reading.source_key, []).append(reading)

        read_at = now.isoformat()
        updated: list[str] = []
        kept: list[str] = []
        ambiguous_failures: list[str] = []
        crossed: list[Crossing] = []
        for allowance_id, source in headroom_policy.sources.items():
            if provider_id_from_source(source) in failed_providers:
                # The ONE call that would have covered this Allowance's
                # provider failed outright. The OTHER call's Readings
                # never substitute for it: keep whatever this Allowance
                # held before (ticket 11).
                kept.append(allowance_id)
                continue
            matches = readings_by_source_key.get(source, [])
            if len(matches) > 1:
                # Two codexbar entries name the same credential. Neither
                # is more right than the other, so keep whatever this
                # Allowance held before rather than let iteration order
                # pick a winner (headroom spec, decision 4).
                kept.append(allowance_id)
                ambiguous_failures.append(
                    f"{allowance_id}: {len(matches)} codexbar entries matched "
                    f"{source!r}; kept the previous Reading"
                )
                continue
            reading = matches[0] if matches else None
            if reading is None or reading.error is not None:
                # Named in Policy, but this run gave nothing to store: no
                # entry matched the source string, its entry failed the
                # shape check, or codexbar itself reports an error for
                # it. Every case keeps whatever this Allowance held
                # before, the same rule ADR 0002 states for Health
                # State: a provider that errors keeps its previous
                # Reading, and every other mapped Allowance still
                # updates.
                kept.append(allowance_id)
                continue
            crossed.extend(
                _binding_window_crossings(
                    allowance_id=allowance_id,
                    previous=previous.records.get(allowance_id),
                    reading=reading,
                    now=now,
                    maximum_staleness_hours=maximum_staleness_hours,
                    named_slots=frozenset(
                        headroom_policy.source_windows.get(allowance_id, {})
                    ),
                )
            )
            records[allowance_id] = HeadroomRecord(
                allowance_id=allowance_id, source=source, reading=reading, read_at=read_at
            )
            updated.append(allowance_id)

        write_headroom(path, HeadroomState(records=records))

    message = f"updated {len(updated)} of {len(headroom_policy.sources)} mapped Allowances"
    if document.failures:
        message += f"; {len(document.failures)} codexbar entries failed their shape check"
    if ambiguous_failures:
        message += f"; {len(ambiguous_failures)} declared sources matched more than one entry"
    if call_errors:
        message += f"; {len(call_errors)} codexbar calls failed outright"
    return RefreshOutcome(
        ran=True,
        message=message,
        updated=tuple(updated),
        kept_previous=tuple(kept),
        failures=(
            tuple(f"{failure.provider or '?'}: {failure.message}" for failure in document.failures)
            + tuple(ambiguous_failures)
            + call_errors
        ),
        crossings=tuple(crossed),
    )


def _binding_window_crossings(
    *,
    allowance_id: str,
    previous: HeadroomRecord | None,
    reading: CodexbarReading,
    now: datetime,
    maximum_staleness_hours: float,
    named_slots: frozenset[str],
) -> tuple[Crossing, ...]:
    """Thresholds this Allowance's Binding Window crossed this refresh.

    Compares the Binding Window of the Reading just taken against the
    Binding Window of the one stored before it. Both sides go through
    `binding_window`, so a void window reads as absent on either side and
    fires nothing.

    `named_slots` carries Policy's declared Sub-allowance slots, exactly
    as `entitlements` passes them. A named slot leaves the parent's
    worst-of computation (ADR 0013), so a window that governs a declared
    Sub-allowance cannot become the figure a crossing fires on. This is
    what keeps a slot describing Offerings the operator may not call from
    paging about capacity that was never theirs.

    An Allowance with no stored Reading yet crosses nothing: there is no
    previous share to have moved from.
    """
    if previous is None:
        return ()
    before = binding_window(
        previous.reading,
        now=now,
        maximum_staleness_hours=maximum_staleness_hours,
        named_slots=named_slots,
    )
    after = binding_window(
        reading,
        now=now,
        maximum_staleness_hours=maximum_staleness_hours,
        named_slots=named_slots,
    )
    if before is None or after is None:
        return ()
    return crossings(
        allowance_id=allowance_id,
        window=_binding_window_name(reading, after, named_slots=named_slots),
        previous_percent=before.used_percent,
        used_percent=after.used_percent,
        updated_at=reading.updated_at,
    )


def _binding_window_name(
    reading: CodexbarReading, window: BindingWindow, *, named_slots: frozenset[str]
) -> str:
    """Which slot the Binding Window came from.

    The Binding Window is the worst of several, and which one is worst
    can change between two Readings, so a crossing that does not name its
    slot cannot be acted on.

    `binding_window` builds a fresh object and does not say which slot it
    chose, so the slot is recovered by matching. Named only when the
    match is UNAMBIGUOUS: two slots can carry the same share, and naming
    whichever is listed first would name the wrong window, which is worse
    than naming none. Ambiguity returns `"binding"`.

    A named slot is skipped, because `binding_window` skipped it too.
    Naming a slot the figure did not come from is the same error.
    """
    matches = [
        name
        for name in SLOT_NAMES
        if name not in named_slots
        and (candidate := getattr(reading, name, None)) is not None
        and candidate.used_percent == window.used_percent
        and candidate.window_minutes == window.window_minutes
    ]
    return matches[0] if len(matches) == 1 else "binding"


def provider_id_from_source(source: str) -> str:
    """The codexbar provider id embedded in a `headroom.sources` value.

    Used only to build the `--provider` list this command asks codexbar
    for. The join itself still matches the whole `source_key` string —
    this never substitutes for that check.
    """
    _, _, rest = source.partition(":")
    provider_id, _, _ = rest.partition("/")
    return provider_id


def _window_to_json(window: CodexbarWindow | None) -> dict[str, Any] | None:
    if window is None:
        return None
    return {
        "used_percent": window.used_percent,
        "window_minutes": window.window_minutes,
        "resets_at": window.resets_at,
    }


def _window_from_json(raw: dict[str, Any] | None) -> CodexbarWindow | None:
    if raw is None:
        return None
    window_minutes = raw.get("window_minutes")
    return CodexbarWindow(
        used_percent=float(raw["used_percent"]),
        window_minutes=None if window_minutes is None else float(window_minutes),
        resets_at=raw.get("resets_at"),
    )


def _extra_window_to_json(extra: CodexbarExtraWindow) -> dict[str, Any]:
    return {"id": extra.id, "title": extra.title, "window": _window_to_json(extra.window)}


def _extra_window_from_json(raw: dict[str, Any]) -> CodexbarExtraWindow:
    return CodexbarExtraWindow(
        id=raw["id"], title=raw["title"], window=_window_from_json(raw["window"])
    )


def _error_to_json(error: CodexbarError | None) -> dict[str, Any] | None:
    if error is None:
        return None
    return {"kind": error.kind, "code": error.code, "message": error.message}


def _error_from_json(raw: dict[str, Any] | None) -> CodexbarError | None:
    if raw is None:
        return None
    return CodexbarError(kind=raw["kind"], code=raw.get("code"), message=raw["message"])


def _reading_to_json(reading: CodexbarReading) -> dict[str, Any]:
    return {
        "provider": reading.provider,
        "identity": {
            "provider_id": reading.identity.provider_id,
            "account_email": reading.identity.account_email,
        },
        "primary": _window_to_json(reading.primary),
        "secondary": _window_to_json(reading.secondary),
        "tertiary": _window_to_json(reading.tertiary),
        "extra_windows": [_extra_window_to_json(w) for w in reading.extra_windows],
        "updated_at": reading.updated_at,
        "error": _error_to_json(reading.error),
    }


def _reading_from_json(raw: dict[str, Any]) -> CodexbarReading:
    identity_raw = raw["identity"]
    return CodexbarReading(
        provider=raw["provider"],
        identity=CodexbarIdentity(
            provider_id=identity_raw["provider_id"],
            account_email=identity_raw.get("account_email"),
        ),
        primary=_window_from_json(raw.get("primary")),
        secondary=_window_from_json(raw.get("secondary")),
        tertiary=_window_from_json(raw.get("tertiary")),
        extra_windows=tuple(
            _extra_window_from_json(w) for w in raw.get("extra_windows", [])
        ),
        updated_at=raw.get("updated_at"),
        error=_error_from_json(raw.get("error")),
    )


def _record_to_json(record: HeadroomRecord) -> dict[str, Any]:
    return {
        "allowance_id": record.allowance_id,
        "source": record.source,
        "reading": _reading_to_json(record.reading),
        "read_at": record.read_at,
    }


def _record_from_json(allowance_id: str, raw: dict[str, Any]) -> HeadroomRecord:
    return HeadroomRecord(
        allowance_id=raw.get("allowance_id", allowance_id),
        source=raw["source"],
        reading=_reading_from_json(raw["reading"]),
        read_at=raw["read_at"],
    )
