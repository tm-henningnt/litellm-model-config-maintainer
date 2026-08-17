"""Parses codexbar's own JSON shape.

`codexbar --format json` reports one entry per provider it knows: a
Reading (CONTEXT.md, "Reading") or an error in place of one. This
module reads that shape and nothing else. It runs no subprocess; see
`litellm_maintainer.headroom` for the command that invokes codexbar and
merges what this module returns into Headroom State.

Build no plugin interface and no generic headroom source here. The
per-provider burden this capability depends on belongs to codexbar's
own authors — see the headroom spec, decision 10.

Codexbar publishes no contract. Its shape moved during one session:
measured 2026-07-28, the Claude Reading held an `all-model` and a
`fable` extra window at 18:48Z, and only the `fable` window at 20:52Z,
with the all-model figure folded into `secondary`. A missing or renamed
field must never crash the whole refresh and must never be guessed at.
`parse_codexbar_document` isolates a bad entry to that one provider:
`CodexbarDocument.readings` holds every entry that parsed, and
`CodexbarDocument.failures` names every one that did not, so the caller
keeps that provider's previous Reading and still updates the rest.

Only a document that is not a JSON list at all fails as a whole,
because there is then no provider to isolate the failure to.

Store Readings only. `pace` is dropped everywhere, read or not: it is
codexbar's own projection from a past burn rate, not a measurement. See
the headroom spec, decision 2.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class CodexbarShapeError(Exception):
    """The whole codexbar document is not a JSON list of entries.

    Raised only for a failure `parse_codexbar_document` cannot isolate
    to one provider: the output is not valid JSON, or its top level is
    not a list. The caller keeps the entire previous Headroom State.
    """


class CodexbarEntryError(Exception):
    """One entry's shape does not match what this parser expects.

    Caught inside `parse_codexbar_document` and turned into a
    `CodexbarEntryFailure`. Never escapes this module.
    """


@dataclass(frozen=True)
class CodexbarWindow:
    """One window inside a Reading: how much of it is spent, and when it resets.

    `used_percent` is the only field a window must carry to say
    anything at all. `window_minutes` and `resets_at` are optional,
    because codexbar states both only for a window that states them.
    Measured 2026-07-28: ClinePass's `primary` and `secondary` windows
    both carry `windowMinutes` and no `resetsAt`, while its `tertiary`
    window carries both.
    """

    used_percent: float
    window_minutes: float | None
    resets_at: str | None


@dataclass(frozen=True)
class CodexbarExtraWindow:
    """One of codexbar's `extraRateWindows`: a named window beside the three.

    Measured 2026-07-28: Claude's Sub-allowance window, `Fable only`,
    arrives here. It appears and leaves across capture sessions; see the
    module docstring.
    """

    id: str
    title: str
    window: CodexbarWindow


@dataclass(frozen=True)
class CodexbarError:
    """The error codexbar reports in place of a Reading, for one provider."""

    kind: str
    code: int | None
    message: str


@dataclass(frozen=True)
class CodexbarIdentity:
    """Who a Reading measures: the provider id and, where stated, the account.

    `account_email` is `None` when codexbar publishes no such field for
    this provider. Measured 2026-07-28: OpenCode Go's entry carries no
    `identity` object at all, so `provider_id` then falls back to the
    entry's own top-level `provider` field. This is an explicit
    fallback, not a guess at a renamed field: codexbar's `identity`
    object is optional per provider, and Policy's example `headroom`
    sources name the same fallback (`"codexbar:opencodego/"`).
    """

    provider_id: str
    account_email: str | None


@dataclass(frozen=True)
class CodexbarReading:
    """One codexbar entry: a Reading, or the error in place of one.

    An entry carrying `error` has no windows: `primary`, `secondary`
    and `tertiary` are `None`, the same as a Reading that states no
    such window at all.

    A Reading with every window `None` and no `error` yields no
    Headroom. Measured 2026-07-28: OpenRouter and DeepSeek both answer
    this way. OpenRouter carries a dollar balance in its own
    `openRouterUsage` object instead; this parser ignores it, because it
    stores windows codexbar states as windows, and nothing
    provider-specific.
    """

    provider: str
    identity: CodexbarIdentity
    primary: CodexbarWindow | None
    secondary: CodexbarWindow | None
    tertiary: CodexbarWindow | None
    extra_windows: tuple[CodexbarExtraWindow, ...]
    updated_at: str | None
    error: CodexbarError | None

    @property
    def source_key(self) -> str:
        """The join key Policy's `headroom.sources` matches against.

        `codexbar:<providerID>/<accountEmail>`. Match the whole string;
        never parse inside it. See ADR 0012 and the headroom spec,
        decision 5.
        """
        return f"codexbar:{self.identity.provider_id}/{self.identity.account_email or ''}"


@dataclass(frozen=True)
class CodexbarEntryFailure:
    """One entry that did not survive the shape check.

    `provider` names the entry it came from, when the entry's own
    top-level `provider` field itself parsed. `None` when even that
    failed.
    """

    provider: str | None
    message: str


@dataclass(frozen=True)
class CodexbarDocument:
    """The result of parsing one codexbar run.

    `readings` holds every entry that parsed. `failures` names every
    entry that did not. A failure here never raises out of
    `parse_codexbar_document`: the caller keeps that one provider's
    previous Reading and still updates every other.
    """

    readings: tuple[CodexbarReading, ...] = ()
    failures: tuple[CodexbarEntryFailure, ...] = ()


def parse_codexbar_document(raw_text: str) -> CodexbarDocument:
    """Parse `codexbar --format json`'s stdout.

    Raise `CodexbarShapeError` only when the whole document cannot be
    read as a list of entries: the text is not valid JSON, or the
    parsed value is not a list. Every other shape problem is isolated
    to one entry, reported in `CodexbarDocument.failures`.
    """
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise CodexbarShapeError(f"codexbar output is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise CodexbarShapeError(
            f"codexbar output is not a JSON list of entries, got {type(data).__name__}"
        )

    readings: list[CodexbarReading] = []
    failures: list[CodexbarEntryFailure] = []
    for entry in data:
        provider = entry.get("provider") if isinstance(entry, dict) else None
        try:
            readings.append(_parse_entry(entry))
        except CodexbarEntryError as exc:
            failures.append(CodexbarEntryFailure(provider=provider, message=str(exc)))
    return CodexbarDocument(readings=tuple(readings), failures=tuple(failures))


def _parse_entry(entry: Any) -> CodexbarReading:
    if not isinstance(entry, dict):
        raise CodexbarEntryError(f"entry is not a JSON object, got {type(entry).__name__}")

    provider = entry.get("provider")
    if not isinstance(provider, str) or not provider:
        raise CodexbarEntryError("entry has no 'provider' string")

    error_raw = entry.get("error")
    error = _parse_error(error_raw, provider) if error_raw is not None else None

    usage_raw = entry.get("usage")
    if error is not None and usage_raw is None:
        # An error entry carries no 'usage' at all. Measured 2026-07-28:
        # openai, azureopenai and cursor all answer this way. No
        # windows, and no identity beyond the entry's own provider.
        return CodexbarReading(
            provider=provider,
            identity=CodexbarIdentity(provider_id=provider, account_email=None),
            primary=None,
            secondary=None,
            tertiary=None,
            extra_windows=(),
            updated_at=None,
            error=error,
        )

    if not isinstance(usage_raw, dict):
        raise CodexbarEntryError(f"'{provider}.usage' is missing or not an object")

    for required in ("primary", "secondary", "tertiary", "updatedAt"):
        if required not in usage_raw:
            raise CodexbarEntryError(f"'{provider}.usage.{required}' is missing")

    identity = _parse_identity(usage_raw.get("identity"), provider)
    primary = _parse_window(usage_raw.get("primary"), provider, "primary")
    secondary = _parse_window(usage_raw.get("secondary"), provider, "secondary")
    tertiary = _parse_window(usage_raw.get("tertiary"), provider, "tertiary")
    extra_windows = _parse_extra_windows(usage_raw.get("extraRateWindows"), provider)

    updated_at = usage_raw.get("updatedAt")
    if updated_at is not None and not isinstance(updated_at, str):
        raise CodexbarEntryError(f"'{provider}.usage.updatedAt' is not a string")

    return CodexbarReading(
        provider=provider,
        identity=identity,
        primary=primary,
        secondary=secondary,
        tertiary=tertiary,
        extra_windows=extra_windows,
        updated_at=updated_at,
        error=error,
    )


def _parse_identity(raw: Any, provider: str) -> CodexbarIdentity:
    if raw is None:
        # Measured 2026-07-28: OpenCode Go's entry carries no 'identity'
        # object. Fall back to the entry's own 'provider' field, which
        # is the id Policy's example sources name for it.
        return CodexbarIdentity(provider_id=provider, account_email=None)
    if not isinstance(raw, dict):
        raise CodexbarEntryError(f"'{provider}.usage.identity' is not an object")
    provider_id = raw.get("providerID")
    if not isinstance(provider_id, str) or not provider_id:
        raise CodexbarEntryError(f"'{provider}.usage.identity.providerID' is missing")
    account_email = raw.get("accountEmail")
    if account_email is not None and not isinstance(account_email, str):
        raise CodexbarEntryError(f"'{provider}.usage.identity.accountEmail' is not a string")
    return CodexbarIdentity(provider_id=provider_id, account_email=account_email or None)


def _parse_window(raw: Any, provider: str, field_name: str) -> CodexbarWindow | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise CodexbarEntryError(f"'{provider}.usage.{field_name}' is not an object")
    used_percent = raw.get("usedPercent")
    if isinstance(used_percent, bool) or not isinstance(used_percent, (int, float)):
        raise CodexbarEntryError(f"'{provider}.usage.{field_name}.usedPercent' is missing")
    window_minutes = raw.get("windowMinutes")
    if window_minutes is not None and (
        isinstance(window_minutes, bool) or not isinstance(window_minutes, (int, float))
    ):
        raise CodexbarEntryError(
            f"'{provider}.usage.{field_name}.windowMinutes' is not a number"
        )
    resets_at = raw.get("resetsAt")
    if resets_at is not None and not isinstance(resets_at, str):
        raise CodexbarEntryError(f"'{provider}.usage.{field_name}.resetsAt' is not a string")
    return CodexbarWindow(
        used_percent=float(used_percent),
        window_minutes=None if window_minutes is None else float(window_minutes),
        resets_at=resets_at,
    )


def _parse_extra_windows(raw: Any, provider: str) -> tuple[CodexbarExtraWindow, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise CodexbarEntryError(f"'{provider}.usage.extraRateWindows' is not a list")
    result = []
    for index, item in enumerate(raw):
        label = f"extraRateWindows[{index}]"
        if not isinstance(item, dict):
            raise CodexbarEntryError(f"'{provider}.usage.{label}' is not an object")
        window_id = item.get("id")
        if not isinstance(window_id, str) or not window_id:
            raise CodexbarEntryError(f"'{provider}.usage.{label}.id' is missing")
        title = item.get("title")
        if not isinstance(title, str):
            raise CodexbarEntryError(f"'{provider}.usage.{label}.title' is missing")
        window = _parse_window(item.get("window"), provider, f"{label}.window")
        if window is None:
            raise CodexbarEntryError(f"'{provider}.usage.{label}.window' is missing")
        result.append(CodexbarExtraWindow(id=window_id, title=title, window=window))
    return tuple(result)


def _parse_error(raw: Any, provider: str) -> CodexbarError:
    if not isinstance(raw, dict):
        raise CodexbarEntryError(f"'{provider}.error' is not an object")
    kind = raw.get("kind")
    if not isinstance(kind, str) or not kind:
        raise CodexbarEntryError(f"'{provider}.error.kind' is missing")
    message = raw.get("message")
    if not isinstance(message, str) or not message:
        raise CodexbarEntryError(f"'{provider}.error.message' is missing")
    code = raw.get("code")
    if code is not None and (isinstance(code, bool) or not isinstance(code, int)):
        raise CodexbarEntryError(f"'{provider}.error.code' is not an integer")
    return CodexbarError(kind=kind, code=code, message=message)
