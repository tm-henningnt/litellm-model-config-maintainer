"""Notification: tell the operator only what is news.

A notification fires when an Offering is added or removed, when one
becomes `needs_operator` or `gone`, when a new Candidate appears, or
when the proxy check fails. It does NOT fire for a routine run, an
Inconclusive result, or a recovery a recorded reset time already
predicted — see `detect_events` for the reasoning on that last, subtle
rule.

`detect_events` is pure: no filesystem, no clock read (`now` is a
parameter), no network. Everything else here — `read_previous_run_state`,
`write_previous_run_state`, `notify_all` — is a thin adapter.

**Where the previous run's state lives, and its one-writer implication.**
The notifier needs to know what the previous run offered and what it
reported as Candidates. That is how it tells "added" and "new" apart
from "unchanged". Neither fact lives in Health State: Health State
records whether an Offering answers, not whether Policy admits it
(CONTEXT.md). Neither fact lives in the Generated Config alone either.
A Declared Offering's Alias never changes, so diffing Aliases cannot
see a Discovered Offering swap places behind one.

This module writes its own small file, `state/last_report.json`. It
holds only the previous `admitted` and `candidates` id sets. **This
file has exactly one writer.** That writer is the code that calls
`write_previous_run_state` once, at the end of a run that has already
computed its `PlanReport`. Nothing else may write it, matching ADR
0001's one-writer rule for every state file this project keeps.

The maintainer writes Health State itself, through
`litellm_maintainer.health.write_health` (ADR 0001). This module only
reads Health State records a caller already has in hand. It never
writes them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from litellm_maintainer.classify import GONE, NEEDS_OPERATOR
from litellm_maintainer.redact import redact
from litellm_maintainer.reduce import OfferingHealth

# A notification is a callable that takes the message text and returns
# nothing. Swap it in a test for a list-appending fake; swap it in
# production for something louder than stdout, with no change to
# `detect_events`.
Notifier = Callable[[str], None]


def default_notifier(message: str) -> None:
    """Print `message` to stdout. The default, injectable notifier."""
    print(message)


@dataclass(frozen=True)
class PreviousRunState:
    """What the previous run offered and reported as awaiting approval.

    `admitted` and `candidates` are Offering id sets, matching
    `PlanReport.admitted` and `PlanReport.candidates`. An empty
    `PreviousRunState` (the default) represents "no previous run" — the
    first run ever, which reports every admitted Offering and every
    Candidate as new. That is the correct result: the operator has
    seen nothing yet.
    """

    admitted: frozenset[str] = frozenset()
    candidates: frozenset[str] = frozenset()


def detect_events(
    *,
    previous: PreviousRunState,
    admitted: frozenset[str],
    candidates: frozenset[str],
    previous_health: dict[str, OfferingHealth],
    health: dict[str, OfferingHealth],
    proxy_ok: bool,
    now: datetime,
) -> tuple[str, ...]:
    """Return the notification messages this run's changes call for.

    Pure. Compares `previous` (the previous run's admitted and
    Candidate sets) and `previous_health` (Health State as it stood
    before this run's Probe or reduce) against this run's `admitted`,
    `candidates` and `health`, plus `proxy_ok` (the result of the
    proxy liveliness check; pass `True` when this run made none).

    Fires one message per:

    - an Offering added: present in `admitted` now, absent before, and
      not a predicted recovery (see below).
    - an Offering removed: present in `previous.admitted`, absent from
      `admitted` now.
    - an Offering that becomes `needs_operator`: its Health State
      bucket is `needs_operator` now and was not before.
    - an Offering that becomes `gone`: the same rule for the `gone`
      bucket.
    - a new Candidate: present in `candidates` now, absent from
      `previous.candidates`.
    - the proxy check failing: `proxy_ok` is `False`.

    Fires NOTHING for a routine run (no set changed, no bucket
    changed, no new Candidate, the proxy answered): the returned tuple
    is then empty, and the caller must still write a run-log line —
    `report.append_run_log`, not this function, is where that always
    happens.

    Fires NOTHING for an Inconclusive result, with no special case
    written for it: `reduce` leaves an Inconclusive Offering's record
    completely untouched (bucket, `excluded`, `reset_at` all
    unchanged), so it produces no event here on its own terms, the
    same way a truly unchanged Offering does.

    **The subtle rule.** An Offering that was Excluded and now answers
    again is, mechanically, the same event as "added" — it moves from
    absent to present in `admitted`. Whether that is news depends on
    whether Health State already predicted it: if the previous record
    was Excluded with a `reset_at`, and `reset_at <= now`, the tool
    already told the operator this Offering would return by now, so
    its return is not news and no message fires for it. When the
    previous record carried no `reset_at`, or the Offering came back
    before its recorded `reset_at`, the recovery was NOT predicted, and
    it fires exactly like an ordinary "added" event.
    """
    events: list[str] = []

    for offering_id in sorted(admitted - previous.admitted):
        prior_record = previous_health.get(offering_id)
        if _recovery_was_predicted(prior_record, now=now):
            continue
        events.append(f"Offering added: {offering_id}")

    for offering_id in sorted(previous.admitted - admitted):
        events.append(f"Offering removed: {offering_id}")

    for offering_id in sorted(set(health) | set(previous_health)):
        current_bucket = health[offering_id].bucket if offering_id in health else None
        prior_bucket = (
            previous_health[offering_id].bucket if offering_id in previous_health else None
        )
        if current_bucket == prior_bucket:
            continue
        if current_bucket == NEEDS_OPERATOR:
            events.append(f"Offering needs the operator: {offering_id}")
        elif current_bucket == GONE:
            events.append(f"Offering gone: {offering_id}")

    for offering_id in sorted(candidates - previous.candidates):
        events.append(f"New Candidate: {offering_id}")

    if not proxy_ok:
        events.append("Proxy check failed")

    return tuple(events)


def _recovery_was_predicted(prior_record: OfferingHealth | None, *, now: datetime) -> bool:
    """Whether Health State already told us this Offering would return by now.

    `True` only when the prior record was Excluded and carried a
    `reset_at` at or before `now`. A prior record with no `reset_at`,
    or one whose `reset_at` is still in the future (an early,
    unpredicted recovery — the Offering answered a real Probe before
    its own recorded clock ran out), returns `False`: that recovery is
    news.
    """
    if prior_record is None or not prior_record.excluded:
        return False
    if prior_record.reset_at is None:
        return False
    return prior_record.reset_at <= now


def notify_all(
    events: tuple[str, ...], *, mapping: dict[str, str], notifier: Notifier = default_notifier
) -> None:
    """Send every message in `events` through `notifier`, redacted first.

    This is the only function here that calls a notifier. It redacts
    each message before delivery, so a caller cannot forget to: there
    is no other path from an event to a delivered notification.
    """
    for message in events:
        notifier(redact(message, mapping))


# --- Previous-run state: the small file this module owns -----------------

_STATE_FILE_NAME = "last_report.json"


def previous_run_state_path(home: Path) -> Path:
    """Return the path to the previous-run state file under `home`.

    `home` is the instance directory (`litellm_maintainer.paths.instance_home`'s
    result), passed explicitly so this stays a plain path computation,
    not a second `$LITELLM_MAINTAINER_HOME` reader.
    """
    return home / "state" / _STATE_FILE_NAME


def read_previous_run_state(path: Path) -> PreviousRunState:
    """Read the previous run's admitted and Candidate id sets from `path`.

    A missing or unreadable file reads as an empty `PreviousRunState`
    (the "first run ever" case), never an error: this file is a small
    convenience record for notification, not Policy or Health State, and
    losing it only costs one run's worth of "what's new" precision.
    """
    try:
        raw_text = path.read_text()
    except FileNotFoundError:
        return PreviousRunState()
    try:
        raw = json.loads(raw_text)
        return PreviousRunState(
            admitted=frozenset(raw.get("admitted", [])),
            candidates=frozenset(raw.get("candidates", [])),
        )
    except (json.JSONDecodeError, TypeError, AttributeError):
        return PreviousRunState()


def write_previous_run_state(
    path: Path, *, admitted: frozenset[str], candidates: frozenset[str]
) -> None:
    """Write this run's admitted and Candidate id sets to `path`.

    Call this once, at the end of a run, after `detect_events` has
    already compared this run against the state the file currently
    holds. Creates the parent directory when it does not exist yet.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "admitted": sorted(admitted),
        "candidates": sorted(candidates),
    }
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
