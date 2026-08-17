"""Run report, run log, `status`, and the Feed's profile picks.

This module covers three things: a run log line for every run, a
`status` view of the operator's four Offering states, and a read of
the Feed's own profile picks. See CONTEXT.md for Withheld, Excluded,
Candidate and Sunsetting. These are four different states. This module
keeps them apart.

**Redaction is built into the write path.** `append_run_log` and
`print_status` are the only functions here that touch a file or a
stream. Both call `litellm_maintainer.redact.redact` on every line
before it leaves the function. A caller cannot forget it, because a
caller never gets a chance to write an unredacted line: the plain-text
line builders (`status_lines`, `run_log_line`) return a value, they
never write one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

from litellm_maintainer.classify import GONE
from litellm_maintainer.feed import Feed
from litellm_maintainer.plan import PlanReport
from litellm_maintainer.policy import Policy
from litellm_maintainer.redact import redact
from litellm_maintainer.reduce import OfferingHealth


# --- The run log ---------------------------------------------------------


def run_log_line(
    *,
    now: datetime,
    report: PlanReport,
    notification_count: int,
    note: str | None = None,
) -> str:
    """Build one run-log line. Pure: takes values, returns text.

    A quiet run — no admitted change, no new Candidate, nothing
    excluded or gone — still produces a line here. The operator needs
    to see the tool ran at all times. `notification_count` is the
    number of notifications this run fired. A reader can then tell a
    quiet run (0) from a loud one at a glance, with no need to re-read
    the whole report.

    `note` appends one condition the run survived but should record — a
    failed Feed fetch, for example. It lands in the run log rather than
    only on the terminal, so a reader can see that three consecutive
    ticks planned on the same stale Feed Document. A run with nothing to
    note produces exactly the line it always produced.
    """
    line = (
        f"{now.isoformat()} run: offered={len(report.admitted)} "
        f"candidates={len(report.candidates)} "
        f"sunsetting={len(report.sunsetting)} "
        f"excluded={len(report.excluded)} "
        f"unlisted={len(report.unlisted)} "
        f"withheld={len(report.withheld)} "
        f"notifications={notification_count}"
    )
    if note:
        line += f" note={note}"
    return line


def append_run_log(
    path: Path,
    *,
    now: datetime,
    report: PlanReport,
    notification_count: int,
    mapping: dict[str, str],
    note: str | None = None,
) -> None:
    """Append one redacted line to the run log at `path`.

    Create the parent directory when it does not exist yet. This is
    the only function in this project that appends to the run log; it
    redacts the line itself, so a caller cannot append an unredacted
    one by mistake.
    """
    line = redact(
        run_log_line(
            now=now,
            report=report,
            notification_count=notification_count,
            note=note,
        ),
        mapping,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(line + "\n")


# --- The Feed's profile picks ---------------------------------------------


@dataclass(frozen=True)
class ProfilePick:
    """One profile pick the Feed publishes, and whether it is offered.

    Profile picks are opinions. They never affect Selection; `plan`
    never reads them. Read `profile_id`, never assume it is one of a
    fixed set: the spec calls the profile collection unstable, so a
    future Feed revision may publish an id this tool has never seen.
    """

    profile_id: str
    display_name: str
    offering_id: str
    offered: bool


def profile_picks(feed: Feed, *, admitted: frozenset[str], now: datetime) -> tuple[ProfilePick, ...]:
    """Read the Feed's profile picks, tolerant of every way they can go wrong.

    Pure: no filesystem, no clock read (`now` is a parameter), no
    network.

    The Feed owner states the profile collection exists mainly to
    support their own feed browser, and that the picks and their
    criteria are subject to change (spec, "Reporting the Feed's own
    picks"). So this reads every entry defensively, one at a time. A
    missing collection reads as no picks (`feed.profiles` already
    defaults to `()` — see `feed.parse_feed`). An unfamiliar profile
    id is still reported: nothing here assumes a fixed catalogue of
    ids, so a Feed revision that adds one is not "unfamiliar" to this
    function. A changed field shape — a missing `selection`, a
    `model_offering_id` that is not a string, an `expires_at` that
    will not parse — drops that one entry and keeps the rest; it never
    raises. An expired pick (`expires_at` at or before `now`) is
    dropped too, and dropped the same way a malformed one is: as
    absent, not flagged stale.

    Nothing downstream may depend on a profile being present, so a
    Feed carrying no readable pick at all returns an empty tuple, never
    an error.
    """
    picks: list[ProfilePick] = []
    for raw in feed.profiles:
        pick = _parse_profile_pick(raw, admitted=admitted, now=now)
        if pick is not None:
            picks.append(pick)
    return tuple(picks)


def _parse_profile_pick(
    raw: object, *, admitted: frozenset[str], now: datetime
) -> ProfilePick | None:
    if not isinstance(raw, dict):
        return None
    profile_id = raw.get("id")
    if not isinstance(profile_id, str) or not profile_id:
        return None
    display_name = raw.get("display_name")
    if not isinstance(display_name, str) or not display_name:
        display_name = profile_id
    selection = raw.get("selection")
    if not isinstance(selection, dict):
        return None
    offering_id = selection.get("model_offering_id")
    if not isinstance(offering_id, str) or not offering_id:
        return None
    expires_at = selection.get("expires_at")
    if not isinstance(expires_at, str):
        return None
    expiry = _parse_feed_timestamp(expires_at)
    if expiry is None:
        return None
    if expiry <= now:
        # Past its one-day expiry. Treated as absent, not as stale.
        return None
    return ProfilePick(
        profile_id=profile_id,
        display_name=display_name,
        offering_id=offering_id,
        offered=offering_id in admitted,
    )


def _parse_feed_timestamp(value: str) -> datetime | None:
    """Parse a Feed timestamp such as `2026-07-26T16:30:02.833Z`.

    Return `None` on anything that will not parse, rather than raise.
    """
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


# --- status ----------------------------------------------------------------


def status_lines(
    *,
    policy: Policy,
    health: dict[str, OfferingHealth],
    report: PlanReport,
    feed: Feed | None = None,
    now: datetime,
) -> tuple[str, ...]:
    """Build the plain-text lines of the `status` view. Pure.

    Prints, in order: what is offered (a Discovered Offering or a
    Declared Offering, marked "(Declared)" alike), what is Excluded
    (with the reason and the expected return), what is Withheld (with
    the reason), the Withheld Policy lines the Feed no longer publishes
    ("Stale Withheld" — worth pruning), what is Sunsetting, what awaits
    approval as a Candidate, and the Feed's own profile picks when
    `feed` is given.

    Withheld, Excluded, Candidate and Sunsetting are four different
    states (CONTEXT.md). Each Offering id appears in at most one of
    these sections: `plan` already keeps them apart in `PlanReport`,
    this function only prints what it is given. "Withheld" and "Stale
    Withheld" together name every id in `policy.withheld`
    (spec-corrections.md, correction 10): `PlanReport.withheld` and
    `PlanReport.withheld_stale` are computed straight from Policy
    against the Feed, not from the Selection pipeline, so this line
    count can never drift from `len(policy.withheld)`.
    """
    lines: list[str] = []

    lines.append(f"Offered: {len(report.admitted)}")
    for offering_id in sorted(report.admitted):
        alias = report.aliases.get(offering_id, "(Declared)")
        lines.append(f"  {offering_id} -> {alias}")

    lines.append(f"Excluded (still served, not recommended): {len(report.excluded)}")
    for offering_id in sorted(report.excluded):
        record = health.get(offering_id)
        lines.append(f"  {offering_id}: {_excluded_reason_line(record)}")

    # The proxy's own view, and the only measurement of it there is. An
    # Offering here is admitted and healthy, and the proxy still refused
    # its Alias — the config it serves is older than this answer.
    not_served = sorted(
        (offering_id, record.alias_not_served_at)
        for offering_id, record in health.items()
        if record.alias_not_served_at is not None
    )
    lines.append(f"Refused by the proxy as not served: {len(not_served)}")
    for offering_id, when in not_served:
        lines.append(f"  {offering_id}: last refused {when.isoformat()}")

    lines.append(f"Unlisted (absent from the config): {len(report.unlisted)}")
    for offering_id in sorted(report.unlisted):
        record = health.get(offering_id)
        lines.append(f"  {offering_id}: {_excluded_reason_line(record)}")

    lines.append(
        f"Passthrough Auth failures (recorded, not Excluded): "
        f"{len(report.passthrough_auth_failures)}"
    )
    for alias in sorted(report.passthrough_auth_failures):
        record = health.get(alias)
        lines.append(f"  {alias}: {_excluded_reason_line(record)}")

    lines.append(f"Withheld: {len(report.withheld)}")
    for offering_id in sorted(report.withheld):
        reason = policy.withheld.get(offering_id, "(no reason recorded)")
        lines.append(f"  {offering_id}: {reason}")

    lines.append(f"Stale Withheld (Feed does not publish this Offering): {len(report.withheld_stale)}")
    for entry in sorted(report.withheld_stale, key=lambda e: e.offering_id):
        reason = policy.withheld.get(entry.offering_id, "(no reason recorded)")
        shade = (
            "provider unknown to the Feed"
            if entry.unknown_provider
            else "provider known, Offering not published"
        )
        lines.append(f"  {entry.offering_id}: {reason} ({shade})")

    lines.append(f"Sunsetting: {len(report.sunsetting)}")
    for offering_id in sorted(report.sunsetting):
        alias = report.aliases.get(offering_id, "(Declared)")
        lines.append(f"  {offering_id} -> {alias}")

    lines.append(f"Awaiting approval (Candidates): {len(report.candidates)}")
    for offering_id in sorted(report.candidates):
        lines.append(f"  {offering_id}")

    if feed is not None:
        admitted = frozenset(report.admitted)
        picks = profile_picks(feed, admitted=admitted, now=now)
        lines.append(f"Feed's own profile picks: {len(picks)}")
        for pick in picks:
            offered_word = "offered" if pick.offered else "not offered"
            lines.append(
                f"  {pick.display_name} ({pick.profile_id}): "
                f"{pick.offering_id} ({offered_word})"
            )

    return tuple(lines)


def _excluded_reason_line(record: OfferingHealth | None) -> str:
    """Describe why an Excluded Offering is excluded, and when it may return.

    Say plainly when there is no reset time, rather than printing an
    empty field: "no expected return" reads as a fact, a blank does
    not.

    A `gone` bucket (a deprecated or removed identifier) never clears by
    itself, so this appends a plain instruction to remove the Offering
    from Policy instead of a return estimate (spec, "Failure
    classification": "The report advises removal from Policy";
    classify.py's own comment names this as the intended, and until now
    unbuilt, behaviour). Every caller of this function already prefixes
    its return value with the exact id or Alias Policy uses to admit
    this Offering (`status_lines`'s "Excluded" section), so "remove it"
    always reads next to the exact Policy key to remove.
    """
    if record is None:
        return "no reason recorded (no Health State record)"
    reason = record.reason or "no reason recorded"
    if record.bucket == GONE:
        return f"{reason}, gone -- remove it from Policy"
    if record.reset_at is None:
        return f"{reason}, no expected return"
    return f"{reason}, expected to return at {record.reset_at.isoformat()}"


# --- The same view as JSON ------------------------------------------------
#
# `guidance` and `entitlements` both answer JSON. `status` answered text
# only, so a consumer had to grep it. The Withheld and Excluded reasons are
# the data that explains a Headroom window governing nothing spendable, and
# no JSON command exposed them. Reported 2026-07-29 by an agent consumer.
#
# Same source as `status_lines`: both read one `PlanReport`, so the two
# renderings cannot disagree about what is offered.

STATUS_SCHEMA_VERSION = "1"


def status_document(
    *,
    policy: Policy,
    health: dict[str, OfferingHealth],
    report: PlanReport,
    feed: Feed | None = None,
    now: datetime,
) -> dict[str, Any]:
    """Build the `status` view as a JSON-ready mapping. Pure.

    Every section `status_lines` prints appears here under its own key.
    An Excluded entry carries `reason`, `bucket` and `reset_at` as
    separate fields rather than one prose line, so a consumer never
    parses the sentence.

    `excluded` and `unlisted` answer different questions. An Excluded
    Offering is in the Generated Config and a caller can reach it; it is
    not recommended. An Unlisted Offering is not in the file at all. See
    ADR 0014.
    """

    def _health_entry(offering_id: str) -> dict[str, Any]:
        record = health.get(offering_id)
        return {
            "offering_id": offering_id,
            "reason": record.reason if record else None,
            "bucket": record.bucket if record else None,
            "reset_at": (
                record.reset_at.isoformat() if record and record.reset_at else None
            ),
            "gone": bool(record and record.bucket == GONE),
        }

    excluded = [_health_entry(offering_id) for offering_id in sorted(report.excluded)]
    unlisted = [_health_entry(offering_id) for offering_id in sorted(report.unlisted)]

    passthrough = []
    for alias in sorted(report.passthrough_auth_failures):
        record = health.get(alias)
        passthrough.append(
            {
                "alias": alias,
                "reason": record.reason if record else None,
                "bucket": record.bucket if record else None,
                "reset_at": (
                    record.reset_at.isoformat() if record and record.reset_at else None
                ),
            }
        )

    document: dict[str, Any] = {
        "schema_version": STATUS_SCHEMA_VERSION,
        "derived_at": now.isoformat(),
        "feed_generated_at": feed.generated_at if feed else None,
        "offered": [
            {"offering_id": oid, "alias": report.aliases.get(oid)}
            for oid in sorted(report.admitted)
        ],
        "excluded": excluded,
        "unlisted": unlisted,
        "passthrough_auth_failures": passthrough,
        "withheld": [
            {"offering_id": oid, "reason": policy.withheld.get(oid)}
            for oid in sorted(report.withheld)
        ],
        "withheld_stale": [
            {
                "offering_id": entry.offering_id,
                "reason": policy.withheld.get(entry.offering_id),
                "unknown_provider": entry.unknown_provider,
            }
            for entry in sorted(report.withheld_stale, key=lambda e: e.offering_id)
        ],
        "sunsetting": [
            {"offering_id": oid, "alias": report.aliases.get(oid)}
            for oid in sorted(report.sunsetting)
        ],
        "candidates": sorted(report.candidates),
    }

    if feed is not None:
        picks = profile_picks(feed, admitted=frozenset(report.admitted), now=now)
        document["feed_profile_picks"] = [
            {
                "profile_id": pick.profile_id,
                "display_name": pick.display_name,
                "offering_id": pick.offering_id,
                "offered": pick.offered,
            }
            for pick in picks
        ]

    return document


def print_status(
    *,
    policy: Policy,
    health: dict[str, OfferingHealth],
    report: PlanReport,
    feed: Feed | None,
    now: datetime,
    mapping: dict[str, str],
    out: TextIO,
) -> None:
    """Print the `status` view to `out`, redacted line by line.

    This is the only function here that prints a status line. Every
    line passes through `redact` before it reaches `out`, so a caller
    cannot forget the redaction step — there is no other way to print
    a status line from this module.
    """
    for line in status_lines(policy=policy, health=health, report=report, feed=feed, now=now):
        print(redact(line, mapping), file=out)
