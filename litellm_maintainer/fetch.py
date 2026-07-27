"""Downloads the Feed and writes the Feed Document.

The only writer of that file. See CONTEXT.md, "Feed Document" and
"Fetch", and ADR 0001 for the one-writer rule this obeys.

## Promote only a document that survives every check

A half-written or truncated Feed Document is worse than an old one: it
would silently shrink Selection, and the Generator would drop Aliases
the operator never withdrew. `fetch_feed_document` therefore writes to a
temporary file beside the destination and renames it into place only
after the bytes parse as JSON, parse as a Feed, and carry a plausible
number of Offerings. A failure at any step leaves the previous Feed
Document exactly as it was.

`safety.MINIMUM_PLAUSIBLE_OFFERING_COUNT` sets the plausibility line,
and `safety.refusal_for_implausible_feed` applies it. The same rule
already guards `generate`, so a short Feed is refused at both the point
it arrives and the point it would be acted on.

## The transport is injected

`fetch_feed_document` takes a `transport` callable. Nothing in this
module imports an HTTP client, so every test here runs offline, which
the spec requires of the whole test suite. `http_transport` builds the
real one, and only the CLI calls it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from litellm_maintainer.feed import parse_feed
from litellm_maintainer.policy import FeedSource
from litellm_maintainer.safety import (
    refusal_for_failed_feed_fetch,
    refusal_for_implausible_feed,
)

# A transport takes the URL and an optional bearer token, and returns
# the response body as text. It raises on any transport-level failure.
Transport = Callable[[str, "str | None"], str]

DEFAULT_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class FetchOutcome:
    """What one `fetch` attempt did.

    `promoted` is `True` only when a new Feed Document is now on disk.
    When it is `False`, `message` says why and the previous Feed
    Document is untouched. `offering_count` is the count in the
    downloaded document, or `None` when nothing parsed.
    """

    promoted: bool
    message: str
    offering_count: int | None = None
    generated_at: str | None = None

    @property
    def kept_previous(self) -> bool:
        """Whether this attempt left an earlier Feed Document in place."""
        return not self.promoted


def http_transport(*, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Transport:
    """Build the real HTTP transport. The only network code in this module."""

    def transport(url: str, token: str | None) -> str:
        import httpx

        headers = {"Authorization": f"Bearer {token}"} if token else {}
        response = httpx.get(url, headers=headers, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        return response.text

    return transport


def resolve_credential(source: FeedSource, environ: dict[str, str] | None = None) -> str | None:
    """Read the Feed's bearer token from the environment, or return `None`.

    A Policy names the variable, never the token. An unset variable is
    not an error here: a Feed that needs no credential is the common
    case, and one that does will fail the request with the provider's
    own message, which says more than a guess would.
    """
    if source.credential_env is None:
        return None
    values = os.environ if environ is None else environ
    token = values.get(source.credential_env, "")
    return token or None


def fetch_feed_document(
    *,
    source: FeedSource,
    destination: Path,
    transport: Transport,
    providers_configured: bool,
    token: str | None = None,
) -> FetchOutcome:
    """Download the Feed and promote it to `destination` if it survives.

    Never raises. Every failure becomes a `FetchOutcome` with
    `promoted` set to `False`, because `fetch` runs inside an
    unattended tick and a network problem must not stop the run. See
    ADR 0005's consequences and `cli.cmd_run`.
    """
    try:
        body = transport(source.url, token)
    except Exception as exc:  # noqa: BLE001 - a transport failure reports, never raises
        return FetchOutcome(promoted=False, message=refusal_for_failed_feed_fetch(exc))

    try:
        raw = json.loads(body)
    except Exception as exc:  # noqa: BLE001 - malformed bytes report the same way
        return FetchOutcome(promoted=False, message=refusal_for_failed_feed_fetch(exc))

    try:
        feed = parse_feed(raw)
    except Exception as exc:  # noqa: BLE001 - a document we cannot read is a failed fetch
        return FetchOutcome(promoted=False, message=refusal_for_failed_feed_fetch(exc))

    count = len(feed.offerings)
    refusal = refusal_for_implausible_feed(count, providers_configured=providers_configured)
    if refusal is not None:
        return FetchOutcome(
            promoted=False,
            message=refusal,
            offering_count=count,
            generated_at=feed.generated_at,
        )

    try:
        _write_atomically(destination, body)
    except OSError as exc:
        # A full disk or a read-only directory must not escape. This
        # function promises never to raise, because the unattended tick
        # calls it and a write failure there would kill the whole run
        # over a document the tick did not even need.
        return FetchOutcome(
            promoted=False,
            message=refusal_for_failed_feed_fetch(exc),
            offering_count=count,
            generated_at=feed.generated_at,
        )

    return FetchOutcome(
        promoted=True,
        message=f"promoted a Feed Document with {count} Offerings",
        offering_count=count,
        generated_at=feed.generated_at,
    )


def _write_atomically(destination: Path, body: str) -> None:
    """Write `body` to `destination` through a rename.

    The temporary file sits in the destination's own directory, so the
    rename stays within one filesystem and is therefore atomic. A reader
    sees either the old document or the new one, never a partial file.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".fetching")
    try:
        temporary.write_text(body)
        temporary.replace(destination)
    except OSError:
        # Leave no partial file behind. A surviving `feed.json.fetching`
        # is confusing at best, and a second run would overwrite it
        # silently rather than report the disk problem that made it.
        temporary.unlink(missing_ok=True)
        raise


def age_hours(generated_at: str | None, *, now: datetime) -> float | None:
    """How many hours old the Feed Document says it is.

    Returns `None` when the document states no `generated_at`, which is
    reported as unknown rather than assumed fresh.
    """
    if not generated_at:
        return None
    try:
        stamped = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError:
        return None
    if stamped.tzinfo is None:
        stamped = stamped.replace(tzinfo=timezone.utc)
    return (now - stamped).total_seconds() / 3600.0


def staleness_warning(
    *, generated_at: str | None, maximum_age_hours: float, now: datetime
) -> str | None:
    """The warning for a Feed Document past the operator's age threshold.

    Returns `None` while the document is fresh enough. A document with
    no stated build time warns too: an unknown age is not a fresh one,
    and selection running on an unknown catalogue is the failure this
    check exists to surface.
    """
    hours = age_hours(generated_at, now=now)
    if hours is None:
        return (
            "the Feed Document states no generated_at, so its age is unknown; "
            "run fetch to replace it"
        )
    if hours <= maximum_age_hours:
        return None
    return (
        f"the Feed Document was generated {hours:.1f}h ago, past the "
        f"{maximum_age_hours:.0f}h threshold; selection is running on a stale "
        "catalogue, so run fetch"
    )


def read_feed_document_metadata(path: Path) -> dict[str, Any]:
    """Read only the Feed Document's self-description.

    Used by `doctor` and `status`, which need the build time without
    parsing 1000-plus Offerings.
    """
    try:
        with open(path) as f:
            raw = json.load(f)
    except Exception:  # noqa: BLE001 - an unreadable document reports as absent
        return {}
    return dict((raw.get("feed") or {}))
