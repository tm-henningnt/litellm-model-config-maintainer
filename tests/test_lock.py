"""The maintainer's lock on Health State. See ADR 0002.

Three processes act in the maintainer role, so the Health State read,
fold and write needs a lock. These tests state the rules an operator
would recognise, not the mechanism.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from litellm_maintainer.classify import Outcome
from litellm_maintainer.health import read_health, write_health
from litellm_maintainer.lock import LockBusy, lock_holder, maintainer_lock
from litellm_maintainer.paths import lock_path
from litellm_maintainer.reduce import reduce

NOW = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)


def _answered(key: str) -> dict[str, Outcome]:
    return {key: Outcome(bucket="answered", reset_at=None, reason="answered")}


def _quota(key: str, reset_at: datetime) -> dict[str, Outcome]:
    return {
        key: Outcome(
            bucket="self_healing", reset_at=reset_at, reason="quota_exhausted"
        )
    }


def _fold(path, outcomes, admitted, now=NOW):
    """Fold outcomes into the Health State at `path`, under the lock."""
    with maintainer_lock(lock_path(path.parent.parent)):
        prior = read_health(path)
        nxt = reduce(
            prior=prior,
            outcomes=outcomes,
            observations=[],
            admitted=frozenset(admitted),
            passthrough_auth=frozenset(),
            now=now,
        )
        write_health(path, nxt)
    return nxt


def test_the_lock_is_free_when_no_maintainer_holds_it(tmp_path):
    assert lock_holder(lock_path(tmp_path)) is None


def test_the_lock_names_its_holder_while_it_is_held(tmp_path):
    with maintainer_lock(lock_path(tmp_path)):
        assert lock_holder(lock_path(tmp_path)) == os.getpid()
    assert lock_holder(lock_path(tmp_path)) is None


def test_a_second_maintainer_waits_rather_than_writing_over_the_first(tmp_path):
    held = threading.Event()
    release = threading.Event()

    def hold():
        with maintainer_lock(lock_path(tmp_path)):
            held.set()
            release.wait(timeout=5)

    worker = threading.Thread(target=hold)
    worker.start()
    held.wait(timeout=5)
    try:
        with pytest.raises(LockBusy):
            with maintainer_lock(lock_path(tmp_path), timeout=0.2):
                pass
    finally:
        release.set()
        worker.join(timeout=5)


def test_a_second_maintainer_proceeds_once_the_first_releases(tmp_path):
    with maintainer_lock(lock_path(tmp_path)):
        pass
    with maintainer_lock(lock_path(tmp_path), timeout=0.2):
        pass


def test_a_concurrent_update_is_never_lost(tmp_path):
    """A maintainer folds onto the freshest Health State, not its own stale copy.

    The Prober reads Health State, spends minutes probing, then folds.
    Another maintainer can write in that gap. The fold must keep what
    the other one wrote.
    """
    (tmp_path / "state").mkdir()
    path = tmp_path / "state" / "health.json"

    _fold(path, _answered("a:1"), {"a:1"})
    # A second maintainer records b:2 while the first is still probing.
    _fold(path, _answered("b:2"), {"a:1", "b:2"})
    # The first maintainer now folds its own result.
    result = _fold(path, _answered("c:3"), {"a:1", "b:2", "c:3"})

    assert set(result.offerings) == {"a:1", "b:2", "c:3"}
    assert set(read_health(path).offerings) == {"a:1", "b:2", "c:3"}


def test_a_recorded_reset_time_survives_a_concurrent_run(tmp_path):
    """The reset time is the value the lock exists to protect.

    A quota failure records when the plan refills. Lose it and the
    Offering can never recover on the clock, which is the recovery that
    needs no further call.
    """
    (tmp_path / "state").mkdir()
    path = tmp_path / "state" / "health.json"
    reset_at = NOW + timedelta(hours=6)

    _fold(path, _quota("a:1", reset_at), {"a:1"})
    _fold(path, _answered("b:2"), {"a:1", "b:2"})

    kept = read_health(path).offerings["a:1"]
    assert kept.reset_at == reset_at
    assert kept.excluded is True


def test_a_holder_that_dies_leaves_no_lock_the_next_run_must_clear(tmp_path):
    """The kernel releases the lock, so no run has to break a stale one."""
    target = lock_path(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    script = (
        "import sys, time\n"
        "sys.path.insert(0, %r)\n"
        "from pathlib import Path\n"
        "from litellm_maintainer.lock import maintainer_lock\n"
        "with maintainer_lock(Path(%r)):\n"
        "    print('held', flush=True)\n"
        "    time.sleep(30)\n" % (os.getcwd(), str(target))
    )
    child = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    try:
        assert child.stdout.readline().strip() == "held"
        assert lock_holder(target) == child.pid
    finally:
        child.kill()
        child.wait(timeout=5)

    deadline = time.monotonic() + 5
    while lock_holder(target) is not None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert lock_holder(target) is None
    with maintainer_lock(target, timeout=0.5):
        pass


def test_the_maintainer_fold_keeps_what_another_maintainer_wrote(tmp_path):
    """Pins the real fold path, not just the pattern the tests above use.

    `cli._fold_into_health_state` re-reads Health State inside the lock.
    A caller that passed its own stale copy in would lose whatever
    another maintainer wrote while it was probing.
    """
    from litellm_maintainer import cli
    from litellm_maintainer.paths import ensure_instance_dirs, health_path

    ensure_instance_dirs(tmp_path)
    path = health_path(tmp_path)

    cli._fold_into_health_state(
        home=tmp_path,
        outcomes=_answered("a:1"),
        admitted=frozenset({"a:1"}),
        passthrough_auth=frozenset(),
        now=NOW,
        key_map=None,
    )
    # Another maintainer records b:2 in the gap.
    _fold(path, _answered("b:2"), {"a:1", "b:2"})

    result = cli._fold_into_health_state(
        home=tmp_path,
        outcomes=_answered("c:3"),
        admitted=frozenset({"a:1", "b:2", "c:3"}),
        passthrough_auth=frozenset(),
        now=NOW,
        key_map=None,
    )
    assert set(result.offerings) == {"a:1", "b:2", "c:3"}


def test_the_maintainer_fold_refuses_while_another_maintainer_holds_the_lock(tmp_path):
    from litellm_maintainer import cli
    from litellm_maintainer.paths import ensure_instance_dirs

    ensure_instance_dirs(tmp_path)
    with maintainer_lock(lock_path(tmp_path)):
        with pytest.raises(LockBusy):
            cli._fold_into_health_state(
                home=tmp_path,
                outcomes=_answered("a:1"),
                admitted=frozenset({"a:1"}),
                passthrough_auth=frozenset(),
                now=NOW,
                key_map=None,
                timeout=0.2,
            )


def test_the_lock_lives_in_the_instance_directory(tmp_path):
    target = lock_path(tmp_path)
    assert target.parent == tmp_path / "state"
    assert target.suffix == ".lock"
    # Not a watched extension: the proxy reloads on *.py, .env and the
    # config file, so a lock here can never restart it.
    assert target.suffix not in {".py", ".yaml", ".env"}
