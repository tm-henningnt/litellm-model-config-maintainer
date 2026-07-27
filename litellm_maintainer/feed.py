"""Feed adapter.

Loads a Feed document from disk and gives typed access to its parts.
This module is an adapter. It holds no decision — read `plan.py` for
selection and translation.

See CONTEXT.md, "Feed", and the spec's "Selection" section for the
Offering shape this module exposes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Offering:
    """One Discovered Offering, as the Feed publishes it.

    `id` has the form `<provider_id>:<provider_model_id>`. `raw` holds
    the full Feed record, for a field this wrapper does not name yet.
    """

    id: str
    provider_id: str
    provider_model_id: str
    capabilities: tuple[str, ...]
    endpoint: dict[str, Any]
    limits: dict[str, Any]
    pricing: dict[str, Any]
    availability: dict[str, Any]
    quality: dict[str, Any]
    policy: dict[str, Any]
    raw: dict[str, Any]

    @property
    def coding_score(self) -> float | None:
        return self.quality.get("coding_score")

    @property
    def context_tokens(self) -> int | None:
        """The context window the Feed states, or `None` when it states none."""
        return self.limits.get("context_tokens")

    @property
    def max_output_tokens(self) -> int | None:
        """The maximum output the Feed states, or `None` when it states none."""
        return self.limits.get("max_output_tokens")

    @property
    def visibility(self) -> str | None:
        return self.policy.get("visibility")

    @property
    def pricing_kind(self) -> str | None:
        return self.pricing.get("kind")

    @property
    def availability_status(self) -> str | None:
        return self.availability.get("status")


@dataclass(frozen=True)
class Provider:
    """One provider record, as the Feed publishes it."""

    id: str
    name: str
    default_base_url: str | None
    authentication: dict[str, Any]
    raw: dict[str, Any]

    @property
    def credential_hint(self) -> str | None:
        return self.authentication.get("credential_hint")


@dataclass(frozen=True)
class Feed:
    """Typed access to a Feed document.

    `profiles` and `notices` are tolerated when absent: the spec calls
    the profile collection unstable, with no contract behind it, and a
    notices list is not guaranteed either. Both default to an empty
    tuple when the document omits them.
    """

    schema_version: str
    offerings: tuple[Offering, ...]
    providers: dict[str, Provider]
    profiles: tuple[dict[str, Any], ...]
    notices: tuple[dict[str, Any], ...]
    raw: dict[str, Any]

    @property
    def generated_at(self) -> str | None:
        """When the Feed built this document, as it states itself.

        Read from the document's own `feed.generated_at`. Staleness is
        judged against this value and the Policy's
        `feed.maximum_age_hours`, never against the file's mtime, which
        records when we downloaded it rather than when it was built.

        The document also carries `expires_at` and
        `default_stale_after_seconds`, and the two have been seen to
        disagree by a factor of 24 in one audited snapshot. Neither is
        read here for that reason: the operator's own threshold decides.
        """
        return (self.raw.get("feed") or {}).get("generated_at")

    def offerings_for(self, provider_id: str) -> tuple[Offering, ...]:
        return tuple(o for o in self.offerings if o.provider_id == provider_id)

    def offering(self, offering_id: str) -> Offering | None:
        for o in self.offerings:
            if o.id == offering_id:
                return o
        return None


def _parse_offering(raw: dict[str, Any]) -> Offering:
    offering_id = raw["id"]
    provider_id, _, provider_model_id = offering_id.partition(":")
    return Offering(
        id=offering_id,
        provider_id=raw.get("provider", {}).get("id", provider_id),
        provider_model_id=raw.get("provider_model_id", provider_model_id),
        capabilities=tuple(raw.get("capabilities", [])),
        endpoint=dict(raw.get("endpoint", {})),
        limits=dict(raw.get("limits") or {}),
        pricing=dict(raw.get("pricing", {})),
        availability=dict(raw.get("availability", {})),
        quality=dict(raw.get("quality", {})),
        policy=dict(raw.get("policy", {})),
        raw=raw,
    )


def _parse_provider(raw: dict[str, Any]) -> Provider:
    return Provider(
        id=raw["id"],
        name=raw.get("name", raw["id"]),
        default_base_url=raw.get("default_base_url"),
        authentication=dict(raw.get("authentication", {})),
        raw=raw,
    )


def parse_feed(raw: dict[str, Any]) -> Feed:
    """Parse an already-loaded Feed document into typed access."""
    providers = {p["id"]: _parse_provider(p) for p in raw.get("providers", [])}
    offerings = tuple(_parse_offering(m) for m in raw.get("models", []))
    profiles = tuple(raw.get("profiles", []) or [])
    notices = tuple(raw.get("notices", []) or [])
    return Feed(
        schema_version=raw.get("schema_version", ""),
        offerings=offerings,
        providers=providers,
        profiles=profiles,
        notices=notices,
        raw=raw,
    )


def load_feed(path: Path) -> Feed:
    """Read the Feed document at `path` and return typed access to it."""
    with open(path) as f:
        raw = json.load(f)
    return parse_feed(raw)
