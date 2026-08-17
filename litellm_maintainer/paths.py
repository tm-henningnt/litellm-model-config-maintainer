"""Instance directory resolution.

The operator's instance directory holds Policy, Health State, the
Observation Journal, snapshots and the Generated Config. It is never
inside this repository. See CONTEXT.md and ADR 0001.

Every helper below takes an optional `home` argument. A test passes a
temporary directory there instead of the operator's real one. No helper
creates a directory. Call `ensure_instance_dirs` for that.
"""

from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "LITELLM_MAINTAINER_HOME"
DEFAULT_HOME = Path.home() / ".config" / "litellm-maintainer"


def instance_home(home: Path | None = None) -> Path:
    """Return the instance directory.

    Use `home` when given. Otherwise read `$LITELLM_MAINTAINER_HOME`.
    Fall back to `~/.config/litellm-maintainer` when that variable is
    unset or empty.
    """
    if home is not None:
        return home
    env_value = os.environ.get(ENV_VAR, "")
    if env_value:
        return Path(env_value)
    return DEFAULT_HOME


def policy_path(home: Path | None = None) -> Path:
    """Return the path to the operator's Policy file."""
    return instance_home(home) / "policy.yaml"


def feed_document_path(home: Path | None = None) -> Path:
    """Return the path to the Feed Document.

    Written only by `fetch`. It lives beside Policy rather than under
    `state/`, because it is not our state: it is the Feed's own
    document, held locally.
    """
    return instance_home(home) / "feed.json"


def health_path(home: Path | None = None) -> Path:
    """Return the path to the Health State file."""
    return instance_home(home) / "state" / "health.json"


def journal_path(home: Path | None = None) -> Path:
    """Return the path to the Observation Journal."""
    return instance_home(home) / "state" / "observations.jsonl"


def snapshots_dir(home: Path | None = None) -> Path:
    """Return the path to the snapshots directory."""
    return instance_home(home) / "snapshots"


def generated_config_path(home: Path | None = None) -> Path:
    """Return the path to the Generated Config."""
    return instance_home(home) / "config.yaml"


def run_log_path(home: Path | None = None) -> Path:
    """Return the path to the run log."""
    return instance_home(home) / "state" / "runs.log"


def lock_path(home: Path | None = None) -> Path:
    """Return the path to the maintainer's lock on Health State.

    Three processes act in the maintainer role, so the Health State
    read, fold and write needs a lock. See ADR 0002.
    """
    return instance_home(home) / "state" / "maintainer.lock"


def headroom_path(home: Path | None = None) -> Path:
    """Return the path to Headroom State.

    Written only by `headroom refresh`. See CONTEXT.md, "Headroom
    State".
    """
    return instance_home(home) / "state" / "headroom.json"


def headroom_lock_path(home: Path | None = None) -> Path:
    """Return the path to Headroom State's own lock.

    `headroom refresh` reads, merges and writes Headroom State, so
    ADR 0002's read-modify-write rule applies here too. This lock is
    never the maintainer lock at `lock_path`: codexbar takes 21-31
    seconds to answer, and holding the maintainer lock that long would
    queue the Observation Journal watcher behind a codexbar sweep. See
    ADR 0002.
    """
    return instance_home(home) / "state" / "headroom.lock"


def ensure_instance_dirs(home: Path | None = None) -> None:
    """Create the instance directory tree.

    Create `state/` and `snapshots/` under the instance home. Create the
    instance home itself when it does not exist.
    """
    base = instance_home(home)
    (base / "state").mkdir(parents=True, exist_ok=True)
    (base / "snapshots").mkdir(parents=True, exist_ok=True)
