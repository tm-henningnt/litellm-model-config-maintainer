"""The maintainer's lock on Health State.

ADR 0001 gives each state file one writing role. ADR 0002 narrows that
rule for Health State, because three processes act in the maintainer
role: the scheduled tick, the Journal watcher, and the operator running
a probe by hand. Each reads Health State, folds in what it measured, and
writes the result back. Without a lock a concurrent update is lost.

The lock uses `flock`. The kernel releases it when the holder dies, so
no lock goes stale and no code has to guess whether a holder is alive.

The holder writes its pid into the file. The pid is for REPORTING only.
No code reads it to decide whether the lock is held. `flock` answers
that question, and a pid can be reused.
"""

from __future__ import annotations

import fcntl
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DEFAULT_TIMEOUT_SECONDS = 30.0
_POLL_SECONDS = 0.05


class LockBusy(Exception):
    """Another maintainer process holds the lock.

    The caller decides what to do. A scheduled tick reports the holder
    and exits without an error, because another maintainer is already
    doing the work. A command the operator typed reports and stops.
    """


@contextmanager
def maintainer_lock(
    path: Path,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    sleep=time.sleep,
    monotonic=time.monotonic,
) -> Iterator[None]:
    """Hold the maintainer's lock for a Health State read, fold and write.

    Wait up to `timeout` seconds for the lock. Raise `LockBusy` when the
    wait runs out. Do not wait forever: a scheduled tick that blocks
    never reports, and never runs again.

    `sleep` and `monotonic` are parameters so a test paces the wait
    without real waiting.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if monotonic() >= deadline:
                    holder = _read_pid(handle)
                    raise LockBusy(
                        f"another maintainer process holds {path}"
                        + (f" (pid {holder})" if holder else "")
                    ) from None
                sleep(_POLL_SECONDS)
        _write_pid(handle)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        os.close(handle)


def lock_holder(path: Path) -> int | None:
    """Return the pid holding the lock, or `None` when it is free.

    Test the lock rather than the file's contents. A pid in a file the
    last holder left behind proves nothing.
    """
    if not path.exists():
        return None
    handle = os.open(path, os.O_RDWR)
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return _read_pid(handle)
        fcntl.flock(handle, fcntl.LOCK_UN)
        return None
    finally:
        os.close(handle)


def _write_pid(handle: int) -> None:
    os.ftruncate(handle, 0)
    os.lseek(handle, 0, os.SEEK_SET)
    os.write(handle, f"{os.getpid()}\n".encode())
    os.fsync(handle)


def _read_pid(handle: int) -> int | None:
    os.lseek(handle, 0, os.SEEK_SET)
    raw = os.read(handle, 32).decode(errors="replace").strip()
    try:
        return int(raw)
    except ValueError:
        return None
