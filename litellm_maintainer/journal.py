"""Append to, and read back, the Observation Journal.

The Observation Journal is an append-only JSON Lines file. The proxy's
failure callback (`providers/journal_failure_callback.py`) appends one
line per failed request. The maintainer reads the file. It folds the
file's entries into Health State through `reduce` (see `reduce.py`).
It may then shorten the file, so the file does not grow without bound.

**Why `.jsonl` outside the config directory is load-bearing.** The
proxy runs with `--reload`, which watches `*.py`, `.env`, and the
config file's basename for a change. It does NOT watch `*.json` or
`*.jsonl`. `litellm_maintainer.paths.journal_path()` resolves to
`$LITELLM_MAINTAINER_HOME/state/observations.jsonl`, a path the
reloader never sees. A Journal placed inside the config directory, or
written with a watched extension, would restart the proxy on every
recorded failure. See ADR 0001 and CONTEXT.md, "Observation Journal".

See ADR 0001 for why the proxy only appends and the maintainer is the
only process that ever shortens the file.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from litellm_maintainer.classify import Bucket, Outcome, Reason
from litellm_maintainer.reduce import Observation

if TYPE_CHECKING:
    from litellm_maintainer.feed import Feed
    from litellm_maintainer.policy import Policy


def append_observation(path: Path, observation: Observation) -> None:
    """Append exactly one Observation to the Journal at `path`.

    Create the parent directory when it does not exist. Encode the
    Observation as one line of JSON, then make exactly one `write(2)`
    system call of that line, complete and newline-terminated.

    One write call is enough because the file is opened with
    `O_APPEND`. The kernel places an `O_APPEND` write at the current
    end of the file and advances the file position as one operation,
    so two processes appending at the same instant cannot interleave
    their bytes, and a line this small (well under a filesystem block)
    lands whole or not at all.

    That size argument is why the failure callback truncates an
    `unrecognized_failure` message to `MAXIMUM_MESSAGE_CHARACTERS`
    (`providers/journal_failure_callback.py`). An untruncated provider
    message -- an HTML error page, say -- could push one line past a
    block and reintroduce the interleaving this function avoids.

    A multi-call write, or a long-lived
    open handle shared across appends, could interleave with another
    writer's line; this function avoids both by opening, writing, and
    closing the file descriptor on every call.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(_encode(observation), separators=(",", ":")) + "\n").encode("utf-8")
    fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(fd, line)
    finally:
        os.close(fd)


@dataclass(frozen=True)
class JournalRead:
    """The result of reading the Journal.

    `observations` holds every record the file parsed, oldest first.
    `skipped` counts lines that did not parse. The proxy writes this
    file under load; one bad line must not stop the maintainer, so a
    malformed line is dropped, not raised, and its count is reported
    here rather than hidden.
    """

    observations: list[Observation] = field(default_factory=list)
    skipped: int = 0


def read_observations(path: Path) -> JournalRead:
    """Read every Observation from the Journal at `path`.

    Return an empty `JournalRead` when the file does not exist; a
    proxy that has not failed yet writes no Journal at all. Skip a
    line that is not valid JSON, or that does not decode to an
    Observation, and count it in `skipped`. A truncated final line —
    one cut short by a crash mid-write — is just another line that
    fails to parse, so the earlier, complete lines still come back.
    """
    if not path.exists():
        return JournalRead(observations=[], skipped=0)

    observations: list[Observation] = []
    skipped = 0
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                observations.append(_decode(line))
            except Exception:
                skipped += 1
    return JournalRead(observations=observations, skipped=skipped)


def observation_key_map(*, feed: "Feed", policy: "Policy") -> dict[str, str]:
    """Map each Alias the proxy can serve to its Health State key.

    The proxy's failure callback records the ALIAS, because the Alias
    is the only name litellm's Router exposes to it (see
    `providers/journal_failure_callback.py`, `model_group`). Health
    State keys a Discovered Offering by its Offering id
    (`<provider>:<provider_model_id>`) and a Declared Offering by its
    Alias (`litellm_maintainer.reduce`, `OfferingKey`). Without this
    map, a Journal entry for a Discovered Offering lands under a key
    `reduce` discards and `plan` never reads, so a real failure the
    proxy served never Excludes anything.

    Map over the Offerings Policy currently ADMITS
    (`prober._discovered_admitted`, the Prober's own worklist source),
    never over the whole Feed. Within the admitted set every Alias is
    unique — `plan` refuses the run otherwise — but across the whole
    Feed it is not: the operator's own data holds a pair, where a paid
    Offering and its `:free` sibling derive one Alias because the
    provider label swallows the `free` token. Only the admitted one can
    be in the Generated Config, so only it can appear in a Journal
    entry.

    A Declared Offering's Alias maps to its `health_key`, and it wins
    over a Discovered Offering that derives the same Alias — the same
    precedence `plan` enforces with its collision refusal.

    A Client-Facing Variant maps to the Alias it widens, not to itself.
    The pair is one Offering under two names, so an observation on
    either belongs on one record. Mapping a variant to itself created a
    second record that `plan` then read separately, leaving the variant
    offered while its twin was Excluded.
    """
    from litellm_maintainer.naming import alias_for
    from litellm_maintainer.prober import _discovered_admitted

    mapping: dict[str, str] = {}
    for offering_id in _discovered_admitted(feed, policy):
        mapping[alias_for(policy, offering_id)] = offering_id
    for declared in policy.declared:
        mapping[declared.alias] = declared.health_key
    return mapping


def resolve_observation_keys(
    observations: list[Observation], key_map: dict[str, str]
) -> list[Observation]:
    """Rewrite each Observation's Alias to its Health State key.

    Apply `observation_key_map`'s result to what `read_observations`
    returned, before the list reaches `reduce`. An `offering_id` absent
    from `key_map` passes through unchanged: it names an Offering
    Policy no longer admits, and `reduce` discards such a record on its
    own. Returns a new list; never mutates `observations`.
    """
    resolved: list[Observation] = []
    for observation in observations:
        key = key_map.get(observation.offering_id, observation.offering_id)
        if key != observation.offering_id:
            observation = replace(observation, offering_id=key)
        resolved.append(observation)
    return resolved


def truncate_first(path: Path, count: int) -> int:
    """Remove the first `count` entries from the Journal at `path`.

    Prefer this to `truncate_processed`. Rotation must not depend on a
    clock the writer controls.

    The Journal is append-only, so the entries a caller read are always
    the FIRST ones in the file. Dropping that many by position removes
    exactly what was folded in, and keeps anything the proxy appended
    since, whatever timestamp it carries.

    Warning: `truncate_processed` compares `observed_at` to `now`, so a
    writer whose clock is ahead makes it a no-op. That happened: the
    failure callback relabelled a naive local time as UTC, putting
    every entry two hours in the future on a UTC+2 host. Nothing was
    ever removed, the Journal grew without bound, `journal_pending`
    stayed true, and the tick ran a full pipeline every 60 seconds
    (measured 2026-07-27). The timestamp bug is fixed at the source,
    but rotation should not have been able to fail that way at all.

    A line that does not parse is dropped with the rest of its block.
    `read_observations` already excluded it from whatever `reduce`
    folded in, so this loses nothing `reduce` used.

    Return the number of entries removed.
    """
    if count <= 0:
        return 0

    read = read_observations(path)
    survivors = read.observations[count:]
    removed = len(read.observations) - len(survivors)
    _rewrite(path, survivors)
    return removed


def truncate_processed(path: Path, upto: datetime) -> int:
    """Remove every Journal entry observed at or before `upto`.

    Warning: a narrow race can lose one Journal entry. The proxy can
    append a line at the exact instant this function reads the file. If
    that write lands after this function's read but before its rename,
    this function loses that one entry. ADR 0001 ("Considered options")
    accepts this rare, small window as the price of never taking a lock
    in the request path.

    Call this only after `reduce` has folded those entries into Health
    State. Pass the same `now` value `reduce` received as `upto`.
    `reduce` treats an entry timestamped at or before `now` as already
    applied (see `reduce.py`), so this cutoff agrees with that fold.

    Read the file. Keep only entries newer than `upto`. Write the
    survivors back through a temporary file and an atomic rename. A
    line that fails to parse carries no timestamp to compare, so this
    drops it too. `read_observations` already excluded that line from
    whatever `reduce` folded in, so this loses nothing `reduce` used.

    Return the number of entries removed.

    This is the one write the maintainer makes to the Journal, and the
    only one: it never appends. `append_observation` never keeps a file
    handle open between calls. A rename here is therefore safe against
    the proxy's next append: that append opens the path fresh and lands
    in whatever file the rename left behind.
    """
    read = read_observations(path)
    survivors = [obs for obs in read.observations if obs.observed_at > upto]
    removed = len(read.observations) - len(survivors)
    _rewrite(path, survivors)
    return removed


def _rewrite(path: Path, survivors: list[Observation]) -> None:
    """Replace the Journal with `survivors`, through an atomic rename.

    Shared by `truncate_first` and `truncate_processed`. A rename is
    safe against the proxy's next append: `append_observation` opens
    the path fresh every call and keeps no handle between them, so it
    lands in whatever file the rename left behind.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for obs in survivors:
            handle.write(json.dumps(_encode(obs), separators=(",", ":")) + "\n")
    os.replace(tmp_path, path)


def _encode(observation: Observation) -> dict:
    """Encode one Observation as the dict written to one Journal line.

    Write `message` only when the Observation carries one. Omitting the
    key on the common case keeps a classified failure's line as small
    as it was before the field existed.
    """
    outcome = observation.outcome
    payload = {
        "offering_id": observation.offering_id,
        "observed_at": observation.observed_at.isoformat(),
        "outcome": {
            "bucket": outcome.bucket,
            "reason": outcome.reason,
            "reset_at": outcome.reset_at.isoformat() if outcome.reset_at else None,
        },
    }
    if observation.message:
        payload["message"] = observation.message
    return payload


def _decode(line: str) -> Observation:
    """Decode one Journal line into an Observation.

    Warning: read `message` with `.get`, never `[...]`. A line the
    proxy wrote before that field existed holds no such key, and
    `read_observations` counts anything this function raises on as a
    skipped line. A required key here would silently discard every
    entry already on disk.
    """
    payload = json.loads(line)
    offering_id = payload["offering_id"]
    observed_at = datetime.fromisoformat(payload["observed_at"])
    raw_outcome = payload["outcome"]
    bucket: Bucket = raw_outcome["bucket"]
    reason: Reason = raw_outcome["reason"]
    reset_at_raw = raw_outcome.get("reset_at")
    reset_at = datetime.fromisoformat(reset_at_raw) if reset_at_raw else None
    outcome = Outcome(bucket=bucket, reset_at=reset_at, reason=reason)
    if not isinstance(offering_id, str):
        raise ValueError("offering_id must be a string")
    raw_message = payload.get("message")
    message = raw_message if isinstance(raw_message, str) and raw_message else None
    return Observation(
        offering_id=offering_id,
        observed_at=observed_at,
        outcome=outcome,
        message=message,
    )
