"""The Operator Surface: the only writer of Policy other than an editor.

ADR 0001 gave Policy exactly one writer, "a human". ADR 0003 narrows
that to "the operator, acting either through an editor or through the
Operator Surface" -- this module. See CONTEXT.md, "Operator Surface",
and ADR 0003 for the reasoning; only the rules that reasoning demands
are restated here.

No part of the run path calls this module. It exists only for a
command the operator typed, or an agent acting as the operator's
instrument (CONTEXT.md, "Operator Surface"). Each public function is
one verb: approve a Candidate, Withhold an Offering, clear a Withheld
entry, set an Alias, or set a provider's Entitlement.

## Why targeted text edits, not parse-and-redump

A Policy file is hand-written and heavily commented -- see
`policy.example.yaml`. `yaml.safe_load` followed by `yaml.dump` keeps
the data and destroys every comment and the operator's own key order.
This module never does that. It parses with `yaml.safe_load` only to
read the current state and to validate the result. The text it writes
is always the operator's original text plus one inserted, replaced, or
removed line. A verb that finds its target block missing appends a new
block at the end of the file instead of inventing one in the middle,
so an edit never guesses where a hand-maintained section should live.

## What a write must survive

ADR 0003 identifies one race worth guarding: the Surface can race the
operator's own editor. Every write here therefore:

1. Takes the lock ADR 0002 defined for Health State
   (`litellm_maintainer.lock.maintainer_lock`, `paths.lock_path`), so
   two Operator Surface calls cannot interleave.
2. Records a hash of the file's text when it reads it, and refuses the
   write if the file's text on disk differs from that hash right
   before the write happens. This is the guard against the editor
   race; the lock only guards Surface against Surface.
3. Validates the edited text with `policy.parse_policy` before
   promoting it. A write that would produce an invalid Policy is
   refused; nothing is written.
4. Writes through a temporary file in Policy's own directory, then
   `os.replace`, matching the pattern in `litellm_maintainer.health` and
   `litellm_maintainer.fetch`. No partial file can ever survive.

A verb whose requested change is already present returns
`changed=False` and writes nothing. This is success, not an error: the
operator asked for a state, and that state already holds.

`dry_run=True` computes and returns the diff without taking the lock
and without touching the file at all.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from litellm_maintainer import lock, paths
from litellm_maintainer.policy import PER_MODEL, VALID_ENTITLEMENTS, PolicyError, parse_policy


class OperatorSurfaceError(ValueError):
    """A refusal the caller must see: an invalid edit, a lost update, or
    an id the Policy does not name."""


@dataclass(frozen=True)
class PolicyEdit:
    """The result of one Operator Surface verb.

    `changed` is `False` when the requested state already held; `diff`
    is then empty and nothing was written. `diff` is a unified diff of
    the Policy file's text. `message` is one line fit to print.
    """

    changed: bool
    diff: str
    message: str


# ---------------------------------------------------------------------------
# The read-modify-write engine shared by every verb.
# ---------------------------------------------------------------------------


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _simulate_race_point() -> None:
    """A seam with no effect in production.

    Called once, after the edit is built and validated, right before
    the final on-disk check that guards against the operator's editor
    (see the module docstring, point 2). A test monkeypatches this to
    mutate the Policy file at exactly that moment, proving the guard
    catches a real race rather than a fabricated assertion.
    """
    return None


def _unified_diff(path: Path, old_text: str, new_text: str) -> str:
    old_lines = old_text.splitlines(keepends=True)
    new_lines = new_text.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=str(path), tofile=str(path))
    return "".join(diff)


def _write_atomically(path: Path, text: str) -> None:
    """Write `text` to `path` through a temporary file and a rename.

    Matches `litellm_maintainer.health.write_health` and
    `litellm_maintainer.fetch._write_atomically`: the temporary file
    sits in `path`'s own directory, so the rename is atomic, and a
    failure between open and rename removes the temporary file rather
    than leaving it behind.
    """
    directory = path.parent
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise


def _validate_or_refuse(new_text: str) -> None:
    try:
        parse_policy(yaml.safe_load(new_text))
    except PolicyError as exc:
        raise OperatorSurfaceError(
            f"Refused to write: the edit would make Policy invalid ({exc})."
        ) from None


BuildResult = tuple[str, bool, str]
Builder = Callable[[str], BuildResult]


def _edit(path: Path, home: Path | None, dry_run: bool, build: Builder) -> PolicyEdit:
    """Run one verb's `build` against `path`, applying every safety rule
    described in the module docstring.

    `build` takes the current Policy text and returns
    `(new_text, changed, message)`. It may raise `OperatorSurfaceError`
    for a refusal specific to what it edits (an id Policy does not
    name, an invalid Entitlement value).
    """
    if dry_run:
        text = path.read_text()
        new_text, changed, message = build(text)
        if not changed:
            return PolicyEdit(changed=False, diff="", message=message)
        _validate_or_refuse(new_text)
        return PolicyEdit(changed=True, diff=_unified_diff(path, text, new_text), message=message)

    with lock.maintainer_lock(paths.lock_path(home)):
        text = path.read_text()
        initial_hash = _hash(text)

        new_text, changed, message = build(text)
        if not changed:
            return PolicyEdit(changed=False, diff="", message=message)

        _validate_or_refuse(new_text)

        _simulate_race_point()

        current_hash = _hash(path.read_text())
        if current_hash != initial_hash:
            raise OperatorSurfaceError(
                f"Refused to write: {path} changed on disk since it was read. "
                "No write happened. Re-read Policy and re-run the command."
            )

        diff = _unified_diff(path, text, new_text)
        _write_atomically(path, new_text)
        return PolicyEdit(changed=True, diff=diff, message=message)


# ---------------------------------------------------------------------------
# Text-level primitives: locate a block by indentation, never by parsing.
# ---------------------------------------------------------------------------


def _lines(text: str) -> list[str]:
    return text.splitlines(keepends=True)


def _join(lines: list[str]) -> str:
    return "".join(lines)


def _leading_spaces(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _quoted(value: str) -> str:
    """Render `value` as a double-quoted YAML scalar.

    Matches the quoting style every hand-written id and Alias uses in
    `policy.example.yaml`. `json.dumps` produces the same escaping YAML
    double-quoted scalars use for the characters a Policy id or Alias
    ever carries.
    """
    return json.dumps(value)


def _find_key(lines: list[str], start: int, end: int, key: str, indent: int) -> int | None:
    """Find a `key:` line at exactly `indent` spaces, within `lines[start:end]`.

    Matches the key bare, single-quoted or double-quoted. YAML treats
    `openrouter:`, `'openrouter':` and `"openrouter":` as the same key.
    Matching only the bare form refused every write against a Policy that
    quoted its provider keys, which is legal and which a hand-written
    Policy does. `init` and `policy.example.yaml` emit bare keys, so this
    only ever mattered for a file written by hand.
    """
    escaped = re.escape(key)
    pattern = re.compile(
        rf"^{' ' * indent}(?:{escaped}|'{escaped}'|\"{escaped}\"):\s*(#.*)?\s*$"
    )
    for i in range(start, end):
        if pattern.match(lines[i]):
            return i
    return None


def _block_end(lines: list[str], start: int, key_indent: int) -> int:
    """The exclusive end of the block whose key line sits at `start`.

    The block's content is every line indented deeper than
    `key_indent`, plus any blank line between two such lines. The scan
    stops at the first non-blank line indented at or shallower than
    `key_indent` -- either a sibling key or a comment introducing the
    next section.
    """
    i = start + 1
    n = len(lines)
    while i < n:
        line = lines[i]
        if line.strip() == "":
            i += 1
            continue
        if _leading_spaces(line) <= key_indent:
            break
        i += 1
    return i


def _insertion_point(lines: list[str], start: int, key_indent: int) -> int:
    """Where to insert a new entry as the last child of the block at `start`.

    This sits right after the block's last non-blank content line, so
    a new entry lands beside its siblings and not after the blank line
    and comment that introduce the next section.
    """
    end = _block_end(lines, start, key_indent)
    i = end - 1
    while i > start and lines[i].strip() == "":
        i -= 1
    return i + 1


def _collapse_if_empty(lines: list[str], start: int, key: str) -> None:
    """Rewrite `key:` as `key: {}` when removing an entry left it empty.

    `withheld:` with no indented line under it parses as `None`, not
    `{}`, and `policy.parse_policy` rejects `None` for a mapping key.
    Removing the last entry must therefore rewrite the key line itself,
    not just delete the entry line.
    """
    end = _block_end(lines, start, 0)
    for i in range(start + 1, end):
        stripped = lines[i].strip()
        if stripped and not stripped.startswith("#"):
            return
    lines[start] = f"{key}: {{}}\n"


def _line_mapping_key(line: str) -> str | None:
    """The single key a mapping-entry line defines, or `None`.

    Parses the stripped line alone as YAML. A line such as
    `  "foo:bar": "reason"` is a complete one-pair mapping on its own,
    so this needs no awareness of the surrounding block's indentation.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    try:
        parsed = yaml.safe_load(stripped)
    except yaml.YAMLError:
        return None
    if isinstance(parsed, dict) and len(parsed) == 1:
        return next(iter(parsed))
    return None


# ---------------------------------------------------------------------------
# approve_candidate
# ---------------------------------------------------------------------------


def _approve_candidate_text(text: str, offering_id: str) -> BuildResult:
    raw = yaml.safe_load(text) or {}
    current = raw.get("approved_candidates") or []
    if offering_id in current:
        return text, False, f"{offering_id!r} is already an approved Candidate."

    lines = _lines(text)
    start = _find_key(lines, 0, len(lines), "approved_candidates", 0)
    if start is None:
        addition = f"\n# Added by the Operator Surface.\napproved_candidates:\n  - {_quoted(offering_id)}\n"
        return (
            text.rstrip("\n") + "\n" + addition.lstrip("\n"),
            True,
            f"approved Candidate {offering_id!r} (new approved_candidates block).",
        )

    insert_at = _insertion_point(lines, start, 0)
    lines.insert(insert_at, f"  - {_quoted(offering_id)}\n")
    return _join(lines), True, f"approved Candidate {offering_id!r}."


def approve_candidate(
    path: Path, offering_id: str, *, home: Path | None = None, dry_run: bool = False
) -> PolicyEdit:
    """Add `offering_id` to `approved_candidates`.

    A Candidate carries no quality score, so it waits for the operator
    to admit it (CONTEXT.md, "Candidate"). This is that admission.
    Idempotent: an already-approved Candidate returns `changed=False`.
    """
    return _edit(path, home, dry_run, lambda text: _approve_candidate_text(text, offering_id))


# ---------------------------------------------------------------------------
# withhold / unwithhold
# ---------------------------------------------------------------------------


def _withhold_text(text: str, offering_id: str, reason: str) -> BuildResult:
    raw = yaml.safe_load(text) or {}
    current = raw.get("withheld") or {}
    if current.get(offering_id) == reason:
        return text, False, f"{offering_id!r} is already Withheld for reason {reason!r}."

    lines = _lines(text)
    start = _find_key(lines, 0, len(lines), "withheld", 0)
    new_line = f"  {_quoted(offering_id)}: {_quoted(reason)}\n"
    if start is None:
        addition = f"\n# Added by the Operator Surface.\nwithheld:\n{new_line}"
        return (
            text.rstrip("\n") + "\n" + addition.lstrip("\n"),
            True,
            f"Withheld {offering_id!r} (new withheld block): {reason}",
        )

    end = _block_end(lines, start, 0)
    existing_idx = None
    for i in range(start + 1, end):
        if _line_mapping_key(lines[i]) == offering_id:
            existing_idx = i
            break

    if existing_idx is not None:
        lines[existing_idx] = new_line
        message = f"updated the Withheld reason for {offering_id!r}: {reason}"
    else:
        lines.insert(_insertion_point(lines, start, 0), new_line)
        message = f"Withheld {offering_id!r}: {reason}"
    return _join(lines), True, message


def withhold(
    path: Path, offering_id: str, reason: str, *, home: Path | None = None, dry_run: bool = False
) -> PolicyEdit:
    """Record `offering_id` as Withheld, for `reason`.

    Withheld is a reason the Feed cannot know -- billing unclear, a
    subscription ending (CONTEXT.md, "Withheld"). Only a human clears
    it, which is `unwithhold`. Calling this again with the same reason
    is idempotent; calling it with a different reason replaces the
    reason.
    """
    return _edit(path, home, dry_run, lambda text: _withhold_text(text, offering_id, reason))


def _unwithhold_text(text: str, offering_id: str) -> BuildResult:
    raw = yaml.safe_load(text) or {}
    current = raw.get("withheld") or {}
    if offering_id not in current:
        return text, False, f"{offering_id!r} is not Withheld."

    lines = _lines(text)
    start = _find_key(lines, 0, len(lines), "withheld", 0)
    if start is None:
        # Parsed Policy disagrees with the text we just read -- refuse
        # rather than guess, since that means the offering_id was found
        # in `raw` but `withheld:` itself is missing from the text.
        raise OperatorSurfaceError(
            "Refused to write: Policy's parsed 'withheld' entry has no matching "
            "'withheld:' block in the file text."
        )
    end = _block_end(lines, start, 0)
    for i in range(start + 1, end):
        if _line_mapping_key(lines[i]) == offering_id:
            del lines[i]
            _collapse_if_empty(lines, start, "withheld")
            return _join(lines), True, f"cleared the Withheld entry for {offering_id!r}."

    raise OperatorSurfaceError(
        f"Refused to write: {offering_id!r} is Withheld per the parsed Policy, but "
        "its line could not be located in the file text."
    )


def unwithhold(
    path: Path, offering_id: str, *, home: Path | None = None, dry_run: bool = False
) -> PolicyEdit:
    """Remove `offering_id`'s Withheld entry, clearing it for use again.

    Idempotent: an Offering that is not Withheld returns `changed=False`.
    """
    return _edit(path, home, dry_run, lambda text: _unwithhold_text(text, offering_id))


# ---------------------------------------------------------------------------
# set_alias
# ---------------------------------------------------------------------------


def _set_alias_text(text: str, offering_id: str, alias: str) -> BuildResult:
    raw = yaml.safe_load(text) or {}
    naming = raw.get("naming") or {}
    current = naming.get("alias_overrides") or {}
    if current.get(offering_id) == alias:
        return text, False, f"{offering_id!r} already has the Alias {alias!r}."

    lines = _lines(text)
    naming_start = _find_key(lines, 0, len(lines), "naming", 0)
    new_entry = f"    {_quoted(offering_id)}: {_quoted(alias)}\n"

    if naming_start is None:
        addition = (
            "\n# Added by the Operator Surface.\n"
            "naming:\n"
            "  provider_labels: {}\n"
            '  alias_prefix: ""\n'
            "  alias_overrides:\n" + new_entry
        )
        return (
            text.rstrip("\n") + "\n" + addition.lstrip("\n"),
            True,
            f"set Alias {alias!r} for {offering_id!r} (new naming block).",
        )

    naming_end = _block_end(lines, naming_start, 0)
    overrides_start = _find_key(lines, naming_start + 1, naming_end, "alias_overrides", 2)

    if overrides_start is None:
        insert_at = _insertion_point(lines, naming_start, 0)
        lines[insert_at:insert_at] = ["  alias_overrides:\n", new_entry]
        return (
            _join(lines),
            True,
            f"set Alias {alias!r} for {offering_id!r} (added naming.alias_overrides).",
        )

    overrides_end = _block_end(lines, overrides_start, 2)
    existing_idx = None
    for i in range(overrides_start + 1, overrides_end):
        if _line_mapping_key(lines[i]) == offering_id:
            existing_idx = i
            break

    if existing_idx is not None:
        lines[existing_idx] = new_entry
        message = f"changed the Alias for {offering_id!r} to {alias!r}."
    else:
        lines.insert(_insertion_point(lines, overrides_start, 2), new_entry)
        message = f"set Alias {alias!r} for {offering_id!r}."
    return _join(lines), True, message


def set_alias(
    path: Path, offering_id: str, alias: str, *, home: Path | None = None, dry_run: bool = False
) -> PolicyEdit:
    """Set `naming.alias_overrides[offering_id]` to `alias`.

    Overrides the mechanical Alias derivation for one Offering
    (CONTEXT.md, "Alias"; `policy.Naming`). Idempotent when the same
    Alias is already set.
    """
    return _edit(path, home, dry_run, lambda text: _set_alias_text(text, offering_id, alias))


# ---------------------------------------------------------------------------
# set_entitlement
# ---------------------------------------------------------------------------


def _set_entitlement_text(text: str, provider_id: str, entitlement: str) -> BuildResult:
    raw = yaml.safe_load(text) or {}
    providers = raw.get("providers") or {}
    if provider_id not in providers:
        raise OperatorSurfaceError(
            f"Refused to write: {provider_id!r} is not a provider in Policy's "
            "'providers' block."
        )

    provider_rule = providers[provider_id] or {}
    current = provider_rule.get("entitlement", PER_MODEL) if isinstance(provider_rule, dict) else PER_MODEL
    if current == entitlement:
        return text, False, f"{provider_id!r} already has entitlement {entitlement!r}."

    lines = _lines(text)
    providers_start = _find_key(lines, 0, len(lines), "providers", 0)
    if providers_start is None:
        raise OperatorSurfaceError(
            "Refused to write: Policy names providers, but no 'providers:' block "
            "was found in the file text."
        )
    providers_end = _block_end(lines, providers_start, 0)
    provider_start = _find_key(lines, providers_start + 1, providers_end, provider_id, 2)
    if provider_start is None:
        raise OperatorSurfaceError(
            f"Refused to write: {provider_id!r} is a Policy provider, but its line "
            "could not be located in the file text."
        )

    provider_end = _block_end(lines, provider_start, 2)
    existing_idx = _find_key(lines, provider_start + 1, provider_end, "entitlement", 4)
    new_line = f"    entitlement: {entitlement}\n"

    if existing_idx is not None:
        lines[existing_idx] = new_line
        # Not `{provider_id!r}'s`: the repr's own closing quote then `'s`
        # reads as `'openrouter''s`.
        message = f"changed the entitlement of {provider_id!r} to {entitlement!r}."
    else:
        lines.insert(_insertion_point(lines, provider_start, 2), new_line)
        message = f"set the entitlement of {provider_id!r} to {entitlement!r}."
    return _join(lines), True, message


def set_entitlement(
    path: Path, provider_id: str, entitlement: str, *, home: Path | None = None, dry_run: bool = False
) -> PolicyEdit:
    """Set `providers.<provider_id>.entitlement` to `entitlement`.

    `entitlement` must be `shared_pool` or `per_model`
    (`policy.VALID_ENTITLEMENTS`; CONTEXT.md, "Entitlement"). Raises
    `OperatorSurfaceError` for any other value, and for a
    `provider_id` Policy does not name -- an Entitlement describes a
    provider already in Policy, it does not create one.
    """
    if entitlement not in VALID_ENTITLEMENTS:
        raise OperatorSurfaceError(
            f"Refused to write: entitlement must be one of "
            f"{sorted(VALID_ENTITLEMENTS)}, got {entitlement!r}."
        )
    return _edit(
        path, home, dry_run, lambda text: _set_entitlement_text(text, provider_id, entitlement)
    )
