"""Health State read and write adapter.

Health State is written only by this path. See ADR 0001 and CONTEXT.md.
`read_health` and `write_health` are the only functions in this project
that touch the Health State file. Everything else that needs Health
State calls `read_health` and passes the result to
`litellm_maintainer.reduce.reduce`.

Health State is JSON, not YAML, on purpose. The proxy's `--reload`
watcher watches `*.py`, `.env` and the config basename, not `*.json`.
See ADR 0001, "State files must not live where the proxy's `--reload`
watcher can see them."
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from litellm_maintainer.reduce import HealthState, OfferingHealth

_SCHEMA_VERSION = 1


def read_health(path: Path) -> HealthState:
    """Read Health State from `path`.

    A missing file reads as an empty Health State, not an error. The
    first run has none.

    A file that is not valid JSON at all, or that has no `offerings`
    mapping, also reads as an empty Health State: `reduce` carries every
    prior record forward each run, so a file broken this badly by a
    killed process or a bad write teaches us nothing a restart from
    empty does not.

    A single bad RECORD inside an otherwise good file is different. Skip
    it, count it, and keep every record that does parse. One bad record
    must never discard every good one: it would drop every recorded
    success and every reset time, force a full Probe sweep, and (since
    Sunsetting now needs our own success record) drop every Sunsetting
    Offering too. Never raise here.
    """
    try:
        raw_text = path.read_text()
    except FileNotFoundError:
        return HealthState(offerings={})

    try:
        raw = json.loads(raw_text)
        raw_offerings = raw["offerings"]
        if not isinstance(raw_offerings, dict):
            raise TypeError("'offerings' is not a mapping")
    except Exception:
        return HealthState(offerings={})

    offerings: dict[str, OfferingHealth] = {}
    skipped = 0
    for key, value in raw_offerings.items():
        try:
            offerings[key] = _record_from_json(value)
        except Exception:
            skipped += 1

    return HealthState(offerings=offerings, skipped_records=skipped)


def write_health(path: Path, state: HealthState) -> None:
    """Write Health State to `path`, atomically.

    Write a temporary file in the same directory as `path`, then
    rename it onto `path`. A rename within one directory is atomic on
    the file systems this project targets, so a reader never observes
    a partial file. A partial Health State file would be worse than
    none, because `read_health` cannot tell "no data" from "half the
    data, and the rest lost".

    Create the parent directory when it does not exist, the same way
    `journal.append_observation` does. `probe` and the watcher's
    confirming Probe write here without `ensure_instance_dirs`, so a
    fresh instance directory crashed the write AFTER the probes ran,
    and their results were lost.
    """
    document = {
        "schema_version": _SCHEMA_VERSION,
        "offerings": {key: _record_to_json(record) for key, record in state.offerings.items()},
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


def _record_to_json(record: OfferingHealth) -> dict[str, Any]:
    return {
        "excluded": record.excluded,
        "reason": record.reason,
        "bucket": record.bucket,
        "reset_at": _dt_to_json(record.reset_at),
        "last_success_at": _dt_to_json(record.last_success_at),
        "last_attempt_at": _dt_to_json(record.last_attempt_at),
        "failure_count": record.failure_count,
        "probe_due": record.probe_due,
        "inconclusive_count": record.inconclusive_count,
    }


def _record_from_json(raw: dict[str, Any]) -> OfferingHealth:
    return OfferingHealth(
        excluded=bool(raw["excluded"]),
        reason=raw.get("reason"),
        bucket=raw.get("bucket"),
        reset_at=_dt_from_json(raw.get("reset_at")),
        last_success_at=_dt_from_json(raw.get("last_success_at")),
        last_attempt_at=_dt_from_json(raw.get("last_attempt_at")),
        failure_count=int(raw["failure_count"]),
        # Read with `.get`. A file written before this field existed
        # holds no such key, and a record this function raises on is
        # counted as skipped and lost.
        probe_due=bool(raw.get("probe_due", False)),
        inconclusive_count=int(raw.get("inconclusive_count", 0)),
    )


def _dt_to_json(value: datetime | None) -> str | None:
    return None if value is None else value.isoformat()


def _dt_from_json(value: str | None) -> datetime | None:
    return None if value is None else datetime.fromisoformat(value)
