"""The Generator's safety rail.

The Generator refuses to write a Generated Config that looks wrong, and
it always says what it would have done instead. See CONTEXT.md and the
spec's "Safety" section.

Every refusal here writes nothing: no config, no snapshot. A refusal is
returned as plain text, the same shape `plan` already uses for its own
refusals, so `cli.cmd_generate` handles both the same way.

This module mixes pure checks (`refusal_for_*`, `validate_config_before_write`,
`detect_envelope_downgrades`) with small filesystem adapters
(`snapshot_config`, `prune_snapshots`, `rollback_latest_snapshot`). The
checks take plain values and return a value; the adapters read or write
one file each and hold no selection logic of their own.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from litellm_maintainer.translate import ENVELOPE_HANDLER_PREFIX

# The Feed's real revisions carry 877 and 876 credentialed-provider
# Offerings (spec, "Amendment: the Feed's 2026-07-25 revision"), and the
# two pinned fixtures carry 1164 and 1163 Offerings overall. A failed or
# truncated fetch commonly returns an empty body, a JSON error page, or
# the first page of a paginated response, all far short of that. 20 sits
# far below any real revision and far above what a legitimate hand-built
# test feed needs, so it catches a truncated fetch without tripping on a
# small fixture. The check only runs when Policy names at least one
# provider — a Policy holding only Declared Offerings does not depend on
# the Feed at all, so an empty or tiny Feed is not a hazard for it.
MINIMUM_PLAUSIBLE_OFFERING_COUNT = 20

_CREDENTIAL_REFERENCE = re.compile(r"^os\.environ/(?P<name>.+)$")

# Snapshot file names sort lexicographically in write order because the
# timestamp is zero-padded and microsecond-resolved. `now` must differ
# between two calls in the same run for this to hold; a real clock read
# always advances, and a test that snapshots twice must pass two
# distinct `now` values.
_SNAPSHOT_GLOB = "config.*.yaml"
_SNAPSHOT_NAME_FORMAT = "config.%Y%m%dT%H%M%S%f.yaml"


class SafetyError(ValueError):
    """A safety operation could not complete, other than a refusal."""


@dataclass(frozen=True)
class RemovalShareRefusal:
    """A run that would drop more Aliases than Policy allows.

    `removed_aliases` names every Alias the previous Generated Config
    offered that the new run would not. Report the names, not only the
    count, so the operator can tell a real outage from a deliberate
    Policy edit at a glance.
    """

    previous_count: int
    new_count: int
    maximum_removal_share: float
    removed_aliases: tuple[str, ...]

    @property
    def message(self) -> str:
        removed = self.previous_count - self.new_count
        share = removed / self.previous_count if self.previous_count else 0.0
        names = ", ".join(self.removed_aliases) or "(none named)"
        return (
            f"Refused to write: the offered count would drop from "
            f"{self.previous_count} to {self.new_count} ({removed} Aliases, "
            f"{share:.0%}), over the configured maximum_removal_share of "
            f"{self.maximum_removal_share:.0%}. Removed: {names}."
        )


def refusal_for_removal_share(
    *,
    previous_count: int | None,
    new_count: int,
    maximum_removal_share: float,
    removed_aliases: tuple[str, ...] = (),
) -> RemovalShareRefusal | None:
    """Whether the drop from `previous_count` to `new_count` is too large.

    Returns `None` when there is no previous count to compare against
    (the first run ever), when the count did not drop, or when the drop
    is at or below `maximum_removal_share`. A `maximum_removal_share` of
    `0.25` means a run may remove up to a quarter of the previous
    Aliases; removing more refuses.
    """
    if previous_count is None or previous_count <= 0:
        return None
    removed = previous_count - new_count
    if removed <= 0:
        return None
    share = removed / previous_count
    if share <= maximum_removal_share:
        return None
    return RemovalShareRefusal(
        previous_count=previous_count,
        new_count=new_count,
        maximum_removal_share=maximum_removal_share,
        removed_aliases=removed_aliases,
    )


def refusal_for_zero_offered(new_count: int) -> str | None:
    """Whether a run that offers zero Aliases must refuse.

    Independent of `refusal_for_removal_share`: a Policy with no
    previous Generated Config to compare against (a first run) still
    must not write an empty one.
    """
    if new_count > 0:
        return None
    return "Refused to write: this run would offer zero Aliases."


def refusal_for_implausible_feed(
    offering_count: int, *, providers_configured: bool, minimum: int = MINIMUM_PLAUSIBLE_OFFERING_COUNT
) -> str | None:
    """Whether the Feed's own Offering count is implausibly short.

    Skipped when Policy names no provider (`providers_configured` is
    `False`): a Declared-only Policy never reads the Feed for
    Selection, so a small or empty Feed document is not a hazard for
    it. See `MINIMUM_PLAUSIBLE_OFFERING_COUNT` for why 20 is the line.
    """
    if not providers_configured:
        return None
    if offering_count >= minimum:
        return None
    return (
        f"Refused to write: the Feed document carries only {offering_count} "
        f"Offerings, below the plausible minimum of {minimum}. Treating "
        "this as a failed or truncated fetch."
    )


def refusal_for_failed_feed_fetch(error: Exception) -> str:
    """The message for a Feed document that could not be read at all."""
    return f"Refused to write: the Feed could not be read ({error})."


def validate_config_before_write(
    config: dict[str, Any], *, credential_resolver: Callable[[str], str | None]
) -> tuple[str, ...]:
    """Structural checks the Generator runs on `config` before it writes.

    Returns one message per problem found, or an empty tuple when
    `config` is safe to write. Checks three things (spec, "Safety"):

    1. Every `model_name` (Alias) is unique.
    2. Every entry names a model: `litellm_params.model` is a non-empty
       string.
    3. Every credential variable resolves. A credential variable is an
       `os.environ/NAME` reference in any string value under
       `litellm_params`, OR the top-level `litellm_settings.master_key`
       (an operator setting, passed through verbatim from Policy's
       `proxy_settings.litellm_settings` — see
       `litellm_maintainer.policy.ProxySettings`). `credential_resolver`
       looks `NAME` up; pass `{}.get` or a fake mapping's `.get` in a
       test, so no test needs a real credential in the environment.

       A Generated Config whose `master_key` does not resolve would
       lock the operator out of their own proxy: litellm refuses every
       call, including the operator's own, until the credential is
       fixed. This check catches it before the write, not after the
       proxy has already reloaded a config nobody can call.

    These never force through: a broken config is a defect to fix, not
    a judgment call to override.
    """
    problems: list[str] = []
    entries = config.get("model_list", [])

    seen_aliases: dict[str, int] = {}
    for entry in entries:
        alias = entry.get("model_name")
        seen_aliases[alias] = seen_aliases.get(alias, 0) + 1
    for alias, count in seen_aliases.items():
        if count > 1:
            problems.append(f"Alias {alias!r} appears {count} times; every Alias must be unique.")

    for entry in entries:
        alias = entry.get("model_name", "<unnamed>")
        model = entry.get("litellm_params", {}).get("model")
        if not model or not isinstance(model, str):
            problems.append(f"Entry {alias!r} names no model in litellm_params.model.")

    for entry in entries:
        alias = entry.get("model_name", "<unnamed>")
        litellm_params = entry.get("litellm_params", {})
        for key, value in litellm_params.items():
            if not isinstance(value, str):
                continue
            match = _CREDENTIAL_REFERENCE.match(value)
            if match is None:
                continue
            name = match.group("name")
            if credential_resolver(name) is None:
                problems.append(
                    f"Entry {alias!r} references credential variable {name!r} "
                    f"(litellm_params.{key}), which is not set."
                )

    master_key = config.get("litellm_settings", {}).get("master_key")
    if isinstance(master_key, str):
        match = _CREDENTIAL_REFERENCE.match(master_key)
        if match is not None:
            name = match.group("name")
            if credential_resolver(name) is None:
                problems.append(
                    f"litellm_settings.master_key references credential variable "
                    f"{name!r}, which is not set. The proxy would refuse every "
                    "call, including the operator's own, until this resolves."
                )

    return tuple(problems)


def detect_envelope_downgrades(
    previous_config: dict[str, Any] | None, new_config: dict[str, Any]
) -> tuple[str, ...]:
    """Aliases that lost the envelope-unwrapping handler since the last write.

    Correction 5 (spec-corrections.md): a Feed revision can simply stop
    publishing `response_envelope_key` for an Offering that still wraps
    its successful responses. `translate_offering` then falls back to
    the generic OpenAI-compatible rule on its own. That fallback is
    correct when the provider truly dropped the envelope. It is
    dangerous when the Feed merely stopped saying so: every error still
    parses, but every SUCCESS starts failing with "provider returned a
    response with no 'choices'" (`docs/gotchas.md`, "Some providers wrap
    successful responses").

    Compares the previous Generated Config (read from disk by the
    caller, `None` on a first run) against the config `plan` just
    produced, by Alias. Reports every Alias whose `litellm_params.model`
    started with `cline/` before and does not now. Never blocks the
    write: this is a loud report, not a refusal, because the downgrade
    can also be the intended, safe case.
    """
    if not previous_config:
        return ()
    handler_prefix = f"{ENVELOPE_HANDLER_PREFIX}/"
    previous_by_alias = {
        entry.get("model_name"): entry for entry in previous_config.get("model_list", [])
    }
    downgraded: list[str] = []
    for entry in new_config.get("model_list", []):
        alias = entry.get("model_name")
        previous_entry = previous_by_alias.get(alias)
        if previous_entry is None:
            continue
        previous_model = previous_entry.get("litellm_params", {}).get("model", "")
        new_model = entry.get("litellm_params", {}).get("model", "")
        if previous_model.startswith(handler_prefix) and not new_model.startswith(handler_prefix):
            downgraded.append(alias)
    return tuple(downgraded)


def removed_aliases(previous_config: dict[str, Any] | None, new_config: dict[str, Any]) -> tuple[str, ...]:
    """Aliases the previous Generated Config offered that `new_config` drops."""
    if not previous_config:
        return ()
    previous_names = [e.get("model_name") for e in previous_config.get("model_list", [])]
    new_names = {e.get("model_name") for e in new_config.get("model_list", [])}
    return tuple(name for name in previous_names if name not in new_names)


def snapshot_config(config_path: Path, snapshots_dir: Path, *, keep: int, now) -> Path | None:
    """Copy the current Generated Config into `snapshots_dir` before a write.

    Returns the snapshot path, or `None` when `config_path` does not
    exist yet (nothing to snapshot on a first run). Prunes old
    snapshots to `keep` afterwards. `now` names the snapshot; pass a
    distinct value on every call in the same run.
    """
    if not config_path.exists():
        return None
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    destination = snapshots_dir / now.strftime(_SNAPSHOT_NAME_FORMAT)
    shutil.copy2(config_path, destination)
    prune_snapshots(snapshots_dir, keep=keep)
    return destination


def list_snapshots(snapshots_dir: Path) -> tuple[Path, ...]:
    """Every snapshot in `snapshots_dir`, oldest first."""
    if not snapshots_dir.exists():
        return ()
    return tuple(sorted(snapshots_dir.glob(_SNAPSHOT_GLOB)))


def prune_snapshots(snapshots_dir: Path, *, keep: int) -> tuple[Path, ...]:
    """Delete every snapshot beyond the newest `keep`. Returns what was deleted."""
    snapshots = list_snapshots(snapshots_dir)
    excess = len(snapshots) - keep
    if excess <= 0:
        return ()
    to_delete = snapshots[:excess]
    for snapshot in to_delete:
        snapshot.unlink()
    return to_delete


def rollback_latest_snapshot(config_path: Path, snapshots_dir: Path) -> Path:
    """Restore the most recent snapshot onto `config_path`.

    Raises `SafetyError` when `snapshots_dir` holds no snapshot. Returns
    the snapshot restored.
    """
    snapshots = list_snapshots(snapshots_dir)
    if not snapshots:
        raise SafetyError(f"no snapshot found in {snapshots_dir}")
    latest = snapshots[-1]
    shutil.copy2(latest, config_path)
    return latest
