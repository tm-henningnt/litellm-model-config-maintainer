"""Ranked picks for a calling agent: which model, then which route.

`guidance` answers "what should I use for this kind of work". A Guidance
Row is one Canonical Model, ranked by one of the Feed's own quality
scores, carrying every Route that reaches it in cost order (CONTEXT.md,
"Guidance Row", "Route"). ADR 0005 records the shape and its limits.

**A row is a model, not an Alias.** In the audited Feed, 345 of 618
Canonical Models have more than one Route; `glm-5.2` has seven. A ranked
list of Aliases would name one model seven times before reaching the
second.

**No balance is ever claimed.** Nothing we can read knows how much
credit is left, so this module reports what was measured: which Routes
answer, which refused, why, and when a refusal said it clears. Cost is
the Feed's own token rate plus a cost basis. See ADR 0005.

**Two orderings, never blended.** Rows descend by the requested score.
Routes within a row ascend by what they cost, so the Route order doubles
as a failover order. `prefer` re-sorts the rows into cost tiers for bulk
work. A single weighted composite was rejected: the weights would be
arbitrary and the result unexplainable.

Every function here is a pure transform. Reading files and printing is
the CLI's job.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any

from litellm_maintainer import naming
from litellm_maintainer.entitlements import (
    FLAT_RATE,
    FREE,
    METERED,
    PASSTHROUGH,
    UNKNOWN_BASIS,
    cost_basis_for_pricing_kind,
)
from litellm_maintainer.classify import REASON_QUOTA_EXHAUSTED
from litellm_maintainer.feed import Feed, Offering
from litellm_maintainer.notify import PreviousRunState
from litellm_maintainer.plan import PlanReport
from litellm_maintainer.policy import DeclaredOffering, Policy
from litellm_maintainer.reduce import OfferingHealth

# Consumers parse this output, so it carries its own version. Raise it
# when a field changes meaning or leaves.
# Raised to "2" on 2026-07-27: a Route gained `wide_alias`, and a
# hand-declared Client-Facing Variant folds into its sibling's row rather
# than forming one of its own. Both change the shape a consumer parses.
SCHEMA_VERSION = "2"

# The axes are the Feed's own score names, so this module invents no
# taxonomy of its own. A caller asking for an axis the Feed does not
# score gets an error naming these, never a silent fallback to another
# axis.
AXES = {
    "coding": "coding_score",
    "reasoning": "reasoning_score",
    "agentic": "agentic_score",
    "speed": "speed_score",
}

# Cheapest first. This is the Route order within a row, and the tier
# order under `prefer`. `unknown` sorts last: an unpriced Route may bill,
# so it is never presented as cheap.
_BASIS_ORDER = (FREE, FLAT_RATE, PASSTHROUGH, METERED, UNKNOWN_BASIS)

PREFERABLE_BASES = (FREE, FLAT_RATE)

# The provider id a Declared Offering reports. It has no Feed provider,
# because the Feed does not publish it at all. This value marks that
# fact instead of leaving the field empty.
DECLARED_PROVIDER = "declared"


class GuidanceError(ValueError):
    """A caller asked for something this module cannot answer."""


@dataclass(frozen=True)
class Route:
    """One Alias through which a row's Canonical Model can be reached."""

    alias: str
    offering_id: str
    provider_id: str
    cost_basis: str
    available: bool
    entitlement: str
    input_usd_per_1m_tokens: float | None = None
    output_usd_per_1m_tokens: float | None = None
    context_tokens: int | None = None
    max_output_tokens: int | None = None
    # The Alias to dispatch to when the caller wants the full
    # `context_tokens`. `None` when no Client-Facing Variant exists, which
    # means the plain Alias is all there is. Read from the run report's
    # derived pairs, or from a Declared Offering's own statement — never by
    # appending or stripping a suffix, because the suffix is an operator
    # setting. See CONTEXT.md, "Client-Facing Variant", and ADR 0007.
    wide_alias: str | None = None
    reason: str | None = None
    refills_at: datetime | None = None
    last_success_at: datetime | None = None
    # Whether a recorded quota exhaustion has not yet cleared. Such a
    # Route is still `available` -- it stays in the Generated Config and
    # a caller may still reach it -- but it is not RECOMMENDED, so it
    # cannot be a `best_route`.
    #
    # The two are different for one reason: a Passthrough Auth Offering
    # is never Excluded on a quota exhaustion, because the quota belongs
    # to the calling client (CONTEXT.md). Before this field, an
    # exhausted Claude subscription still reported `callable_now: true`,
    # and the `model-routing` skill tells an agent to trust exactly that
    # field. See ADR 0010.
    exhausted: bool = False

    @property
    def recommendable(self) -> bool:
        """Whether this Route may be handed to a caller as the answer.

        Available AND not exhausted. Excluding an exhausted Route from
        the config instead would give the caller "model not found" in
        place of the provider's own "your quota resets at 09:00", and
        for a Passthrough Auth Offering nothing could ever clear it: no
        Probe is possible, so only the clock can.
        """
        return self.available and not self.exhausted

    @property
    def rate_is_list_price(self) -> bool:
        """Whether the stated rate is a list price rather than a bill.

        True for a flat-rate Route: `pricing.py` writes the same
        distinction into the Generated Config, because summing a
        subscription's token rate into an invoice would be wrong.
        """
        return self.cost_basis in (FREE, FLAT_RATE)

    def as_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "offering_id": self.offering_id,
            "provider_id": self.provider_id,
            "cost_basis": self.cost_basis,
            "entitlement": self.entitlement,
            "available": self.available,
            "rate_is_list_price": self.rate_is_list_price,
            "input_usd_per_1m_tokens": self.input_usd_per_1m_tokens,
            "output_usd_per_1m_tokens": self.output_usd_per_1m_tokens,
            "context_tokens": self.context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "wide_alias": self.wide_alias,
            "reason": self.reason,
            "refills_at": self.refills_at.isoformat() if self.refills_at else None,
            "exhausted": self.exhausted,
            "recommendable": self.recommendable,
        }


@dataclass(frozen=True)
class GuidanceRow:
    """One Canonical Model and every Route that reaches it."""

    canonical_model_id: str
    display_name: str
    score: float | None
    scores: dict[str, float | None]
    routes: tuple[Route, ...]
    capabilities: tuple[str, ...] = ()
    # Whether `capabilities` came from the operator's Policy rather than
    # from the Feed. Only a Declared Offering sets it. `why` states the
    # source, so a caller cannot mistake one for the other.
    capabilities_are_operator_stated: bool = False

    @property
    def best_route(self) -> Route | None:
        """The cheapest Route that answers AND is not exhausted.

        Read `recommendable`, not `available`. An exhausted Route is
        still callable and still in the Generated Config; it just must
        not be the answer this row hands back. See ADR 0010.
        """
        for route in self.routes:
            if route.recommendable:
                return route
        return None

    @property
    def callable_now(self) -> bool:
        return self.best_route is not None

    @property
    def why(self) -> str:
        """One line stating why this row is where it is, from its own inputs."""
        route = self.best_route
        if route is None:
            return "no Route answers right now; every Route is excluded"
        parts = []
        if self.score is not None:
            parts.append(f"scores {self.score:g} on the requested axis")
        elif route.entitlement == "declared":
            # Not a gap in the Feed: the operator declared this Offering
            # because the Feed does not cover it. Ranking it last is a
            # consequence of having no score, not a judgement about it.
            parts.append(
                "declared by the operator, so the Feed does not score it; "
                "rank it yourself"
            )
        else:
            parts.append("carries no score on the requested axis")
        if route.cost_basis == FREE:
            parts.append("reachable free")
        elif route.cost_basis == FLAT_RATE:
            parts.append("no marginal cost, drains a flat-rate window")
        elif route.cost_basis == METERED:
            parts.append("metered, so it bills per token")
        elif route.cost_basis == PASSTHROUGH:
            parts.append("billed to the calling client's own credential")
        else:
            parts.append("unpriced, so treat it as billable")
        if self.capabilities and self.capabilities_are_operator_stated:
            parts.append("capabilities stated by the operator, not the Feed")
        alternatives = sum(1 for r in self.routes if r.available) - 1
        if alternatives > 0:
            parts.append(f"{alternatives} further Route(s) if this one refuses")
        return "; ".join(parts)

    def as_dict(self) -> dict[str, Any]:
        return {
            "canonical_model_id": self.canonical_model_id,
            "display_name": self.display_name,
            "score": self.score,
            "scores": dict(self.scores),
            "capabilities": list(self.capabilities),
            "callable_now": self.callable_now,
            "why": self.why,
            "routes": [r.as_dict() for r in self.routes],
        }


@dataclass(frozen=True)
class RemovedAlias:
    """An Alias a client may hold that the proxy no longer serves."""

    alias: str
    offering_id: str
    reason: str | None
    refills_at: datetime | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "offering_id": self.offering_id,
            "reason": self.reason,
            "refills_at": self.refills_at.isoformat() if self.refills_at else None,
        }


ADVISORY_NOTE = (
    "An Alias listed here is callable by exact id even when it is absent "
    "from a model list your client cached earlier. The proxy resolves a "
    "call by Alias, not by what your client last fetched. An Alias in "
    "removed_last_run is no longer served: read its reason and refills_at "
    "rather than retrying it."
)


@dataclass(frozen=True)
class ClientAdvisory:
    """The drift between the Generated Config and a client's cached list.

    A client fetches the model list once and caches it; the config
    changes underneath. Both directions matter, and only one is
    recoverable by the caller on its own. See CONTEXT.md, "Client
    Advisory".
    """

    added_last_run: tuple[str, ...] = ()
    removed_last_run: tuple[RemovedAlias, ...] = ()
    note: str = ADVISORY_NOTE

    def as_dict(self) -> dict[str, Any]:
        return {
            "note": self.note,
            "added_last_run": list(self.added_last_run),
            "removed_last_run": [r.as_dict() for r in self.removed_last_run],
        }


@dataclass(frozen=True)
class Guidance:
    """A whole guidance answer, as a caller receives it."""

    axis: str
    rows: tuple[GuidanceRow, ...] = ()
    advisory: ClientAdvisory = field(default_factory=ClientAdvisory)
    feed_generated_at: str | None = None
    derived_at: datetime | None = None
    prefer: str | None = None
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "axis": self.axis,
            "prefer": self.prefer,
            "derived_at": self.derived_at.isoformat() if self.derived_at else None,
            "feed_generated_at": self.feed_generated_at,
            "warnings": list(self.warnings),
            "client_advisory": self.advisory.as_dict(),
            "rows": [r.as_dict() for r in self.rows],
        }


def route_is_exhausted(
    record: OfferingHealth | None,
    *,
    now: datetime,
    maximum_staleness_hours: float,
) -> bool:
    """Whether a recorded quota exhaustion has not yet cleared.

    Read a quota exhaustion only. Every other reason either Excludes the
    Offering already, which `available` covers, or says nothing about
    whether the next call can succeed.

    A stated reset time governs when one is recorded: the Route is
    exhausted until it passes.

    A quota exhaustion with NO reset time is the hard case. Nothing can
    clear it on its own: the Journal records only failures, so no
    success is ever observed, and a Passthrough Auth Offering cannot be
    probed at all. Left unbounded it would hide a working Offering
    forever. So it expires after `maximum_staleness_hours` from the
    attempt that recorded it — wrong in the safe direction, since it
    recommends a possibly-exhausted model a day later rather than
    hiding a working one indefinitely.
    """
    if record is None or record.reason != REASON_QUOTA_EXHAUSTED:
        return False
    if record.reset_at is not None:
        return record.reset_at > now
    if record.last_attempt_at is None:
        return False
    return (now - record.last_attempt_at) < timedelta(hours=maximum_staleness_hours)


def _score(offering: Offering, axis_field: str) -> float | None:
    value = offering.quality.get(axis_field)
    return float(value) if isinstance(value, (int, float)) else None


def _basis_rank(basis: str) -> int:
    try:
        return _BASIS_ORDER.index(basis)
    except ValueError:
        return len(_BASIS_ORDER)


def _route_sort_key(route: Route) -> tuple:
    """Recommendable first, then cheapest, then most recently answered.

    Usability outranks cost on purpose: a failover order whose first
    entry cannot be called is not a failover order.

    Sort on `recommendable`, not `available`. An exhausted Route is
    still available — it stays in the Generated Config, and a
    Passthrough Auth Offering is never Excluded on a quota exhaustion
    (ADR 0010). Sorting on `available` put an exhausted Claude Alias
    first in its own row's failover order, which is the same harm ADR
    0010 fixes at the row level, one level down.
    """
    last_success = route.last_success_at.timestamp() if route.last_success_at else 0.0
    return (not route.recommendable, _basis_rank(route.cost_basis), -last_success)


def derive(
    *,
    feed: Feed,
    policy: Policy,
    health: dict[str, OfferingHealth],
    report: PlanReport,
    now: datetime,
    axis: str = "coding",
    prefer: str | None = None,
    previous: PreviousRunState | None = None,
    limit: int | None = None,
    min_context: int | None = None,
    warnings: tuple[str, ...] = (),
) -> Guidance:
    """Build the ranked answer. Pure: reads values, returns a value.

    Raise `GuidanceError` when `axis` or `prefer` names something this
    module cannot answer, rather than quietly answering a different
    question.
    """
    if axis not in AXES:
        raise GuidanceError(
            f"'{axis}' is not a scored axis. The Feed scores: {sorted(AXES)}"
        )
    if prefer is not None and prefer not in PREFERABLE_BASES:
        raise GuidanceError(
            f"'{prefer}' is not a preferable cost basis. Use one of "
            f"{sorted(PREFERABLE_BASES)}"
        )
    if min_context is not None and (
        isinstance(min_context, bool)
        or not isinstance(min_context, int)
        or min_context <= 0
    ):
        raise GuidanceError(
            f"'min_context' must be a positive integer, got {min_context!r}"
        )
    axis_field = AXES[axis]

    # Alias -> the Client-Facing Variant that yields its full window. Read
    # from what the Generator reported, so the Generated Config and this
    # answer cannot disagree about which Aliases exist.
    wide_by_alias = dict(report.client_facing_variants)
    # A hand-declared Client-Facing Variant states which Alias it widens.
    # Fold the pair the same way: the named Alias keeps the row, and the
    # variant becomes that row's `wide_alias`. Nothing is inferred from a
    # name. A variant whose sibling this run did not admit grants nothing,
    # so it is reported rather than shown as a model of its own.
    declared_variant_of: dict[str, str] = {}
    orphan_variants: list[str] = []
    admitted_or_excluded = set(report.admitted) | set(report.excluded)
    for _declared in policy.declared:
        if _declared.variant_of is None:
            continue
        if _declared.variant_of in admitted_or_excluded:
            wide_by_alias[_declared.variant_of] = _declared.alias
            declared_variant_of[_declared.alias] = _declared.variant_of
        elif _declared.alias in admitted_or_excluded:
            orphan_variants.append(_declared.alias)
            declared_variant_of[_declared.alias] = _declared.variant_of
    if orphan_variants:
        warnings = warnings + tuple(
            f"{alias!r} states it is a Client-Facing Variant of "
            f"{declared_variant_of[alias]!r}, which this run did not offer, so "
            "it reaches no Guidance Row."
            for alias in sorted(orphan_variants)
        )

    # The window that bounds a quota exhaustion stating no reset time
    # (`route_is_exhausted`). Policy's own staleness setting, so the
    # operator tunes one number, not two.
    staleness_hours = policy.schedule.maximum_staleness_hours
    rows_by_model: dict[str, list[Route]] = {}
    offering_by_model: dict[str, list[Offering]] = {}

    for offering_id in list(report.admitted) + list(report.excluded):
        offering = feed.offering(offering_id)
        if offering is None:
            continue
        available = offering_id in report.admitted
        alias = report.aliases.get(offering_id)
        if alias is None:
            try:
                alias = naming.alias_for(policy, offering_id)
            except Exception:  # noqa: BLE001 - an Alias we cannot derive is not reportable
                continue
        record = health.get(offering_id)
        rule = policy.providers.get(offering.provider_id)
        model_key = (offering.raw.get("canonical_model") or {}).get("id") or offering_id

        rows_by_model.setdefault(model_key, []).append(
            Route(
                alias=alias,
                offering_id=offering_id,
                provider_id=offering.provider_id,
                cost_basis=cost_basis_for_pricing_kind(offering.pricing_kind),
                available=available,
                entitlement=rule.entitlement if rule is not None else "per_model",
                input_usd_per_1m_tokens=offering.pricing.get("input_usd_per_1m_tokens"),
                output_usd_per_1m_tokens=offering.pricing.get("output_usd_per_1m_tokens"),
                context_tokens=offering.context_tokens,
                max_output_tokens=offering.max_output_tokens,
                wide_alias=wide_by_alias.get(alias),
                reason=record.reason if record is not None else None,
                refills_at=record.reset_at if record is not None else None,
                last_success_at=record.last_success_at if record is not None else None,
                exhausted=route_is_exhausted(
                    record, now=now, maximum_staleness_hours=staleness_hours
                ),
            )
        )
        offering_by_model.setdefault(model_key, []).append(offering)

    # Declared Offerings reach the proxy too, and the Feed knows nothing
    # about them. Leaving them out made the strongest models on the proxy
    # invisible to the caller: a direct vendor entry is often the best
    # model offered, and it carries no Feed record at all. Each becomes
    # its own row with no score, so the score ordering puts it last while
    # the answer still admits it exists.
    declared_by_alias: dict[str, DeclaredOffering] = {}
    for declared in policy.declared:
        if declared.alias not in report.admitted and declared.alias not in report.excluded:
            continue
        # A Client-Facing Variant contributes no row and no Route of its
        # own: it is the same Offering under a second name, already folded
        # onto its sibling's Route above.
        if declared.alias in declared_variant_of:
            continue
        record = health.get(declared.alias)
        declared_by_alias[declared.alias] = declared
        # The Stated Limit of a Declared Offering is whatever the operator
        # wrote in `model_info`, which reaches Generated Config verbatim.
        # Reading it back here keeps the guidance answer and the config
        # stating one figure, not two.
        stated = declared.model_info or {}
        rows_by_model.setdefault(declared.alias, []).append(
            Route(
                alias=declared.alias,
                offering_id=declared.alias,
                provider_id=DECLARED_PROVIDER,
                cost_basis=PASSTHROUGH if declared.passthrough_auth else UNKNOWN_BASIS,
                available=declared.alias in report.admitted,
                entitlement="declared",
                context_tokens=stated.get("max_input_tokens"),
                max_output_tokens=stated.get("max_output_tokens"),
                wide_alias=wide_by_alias.get(declared.alias),
                reason=record.reason if record is not None else None,
                refills_at=record.reset_at if record is not None else None,
                last_success_at=record.last_success_at if record is not None else None,
                exhausted=route_is_exhausted(
                    record, now=now, maximum_staleness_hours=staleness_hours
                ),
            )
        )

    rows: list[GuidanceRow] = []
    for model_key, routes in rows_by_model.items():
        if model_key not in offering_by_model:
            # A Declared Offering. It has no Feed record, so it has no
            # score. Its capabilities are the operator's own statement, or
            # empty when the operator stated none: guess neither.
            declared = declared_by_alias.get(model_key)
            capabilities = declared.capabilities if declared is not None else ()
            rows.append(
                GuidanceRow(
                    canonical_model_id=model_key,
                    display_name=model_key,
                    score=None,
                    scores={name: None for name in AXES},
                    routes=tuple(routes),
                    capabilities=capabilities,
                    capabilities_are_operator_stated=bool(capabilities),
                )
            )
            continue
        offerings = offering_by_model[model_key]
        scores = {
            name: _best_score(offerings, field_name) for name, field_name in AXES.items()
        }
        first = offerings[0]
        rows.append(
            GuidanceRow(
                canonical_model_id=model_key,
                display_name=first.raw.get("display_name") or model_key,
                score=_best_score(offerings, axis_field),
                scores=scores,
                routes=tuple(sorted(routes, key=_route_sort_key)),
                capabilities=first.capabilities,
            )
        )

    if min_context is not None:
        rows, filter_warning = _narrow_to_window(rows, min_context)
        if filter_warning is not None:
            warnings = warnings + (filter_warning,)

    rows.sort(key=lambda row: _row_sort_key(row, prefer=prefer))
    if limit is not None:
        rows = rows[:limit]

    return Guidance(
        axis=axis,
        rows=tuple(rows),
        advisory=build_advisory(
            policy=policy, report=report, health=health, previous=previous
        ),
        feed_generated_at=feed.generated_at,
        derived_at=now,
        prefer=prefer,
        warnings=warnings,
    )


def _narrow_to_window(
    rows: list[GuidanceRow], min_context: int
) -> tuple[list[GuidanceRow], str | None]:
    """Keep only the Routes that hold `min_context` tokens.

    Filters ROUTES and then drops a row with none left, rather than
    filtering rows and keeping their narrow Routes. A Route order doubles as
    a failover order, so a surviving narrow Route would invite a caller to
    fail over into something too small for the work the filter asked about.

    A Route stating no window does not qualify. ADR 0006 says absence reads
    as unknown rather than small, and that still holds — but a filter has to
    decide, and handing back an unmeasured Route as though it qualified is
    the more expensive error. Counted apart from the too-narrow ones so the
    warning can say "unstated" rather than implying "too small".

    Returns the surviving rows and one warning line, or `None` when the
    filter removed nothing.
    """
    kept: list[GuidanceRow] = []
    too_narrow = 0
    unstated = 0
    dropped_rows = 0
    for row in rows:
        survivors = []
        for route in row.routes:
            if route.context_tokens is None:
                unstated += 1
            elif route.context_tokens < min_context:
                too_narrow += 1
            else:
                survivors.append(route)
        if survivors:
            kept.append(replace(row, routes=tuple(survivors)))
        else:
            dropped_rows += 1

    if not (too_narrow or unstated or dropped_rows):
        return kept, None
    return kept, (
        f"minimum context {min_context}: {len(kept)} of {len(rows)} row(s) "
        f"remain. Dropped {dropped_rows} row(s), and {too_narrow} Route(s) "
        f"stating a smaller window plus {unstated} Route(s) stating none. A "
        "Route stating no window is excluded rather than assumed small."
    )


def _best_score(offerings: list[Offering], axis_field: str) -> float | None:
    """The highest score any Route's record states for this model.

    Routes to one Canonical Model can carry different scores, because
    each Offering carries its own record. The score describes the model,
    so the highest stated value is taken and the disagreement is not
    treated as two models.
    """
    values = [s for s in (_score(o, axis_field) for o in offerings) if s is not None]
    return max(values) if values else None


def _row_sort_key(row: GuidanceRow, *, prefer: str | None) -> tuple:
    """Score descending, callable rows first. `prefer` adds a cost tier in front.

    A row with no score on the requested axis sorts last rather than
    being dropped: it is admitted, so it is callable, and hiding it would
    make the answer look shorter than the truth.
    """
    score = row.score if row.score is not None else float("-inf")
    tier = 0
    if prefer is not None:
        route = row.best_route
        tier = 0 if route is not None and route.cost_basis == prefer else 1
    return (tier, not row.callable_now, -score, row.canonical_model_id)


def build_advisory(
    *,
    policy: Policy,
    report: PlanReport,
    health: dict[str, OfferingHealth],
    previous: PreviousRunState | None,
) -> ClientAdvisory:
    """Name the Aliases the last run added and removed.

    Both sets come from the Previous-run record, which already holds the
    previous `admitted` id set (CONTEXT.md, "Previous-run record"), so
    this needs no new file. With no previous record — a first run — the
    sets are empty and the note still stands on its own.
    """
    if previous is None:
        return ClientAdvisory()

    current = set(report.admitted)
    added = sorted(current - set(previous.admitted))
    removed = sorted(set(previous.admitted) - current)

    added_aliases = tuple(
        alias for alias in (_alias_for(policy, report, oid) for oid in added) if alias
    )
    removed_aliases = []
    for offering_id in removed:
        alias = _alias_for(policy, report, offering_id)
        if alias is None:
            continue
        record = health.get(offering_id)
        removed_aliases.append(
            RemovedAlias(
                alias=alias,
                offering_id=offering_id,
                reason=record.reason if record is not None else None,
                refills_at=record.reset_at if record is not None else None,
            )
        )

    return ClientAdvisory(
        added_last_run=added_aliases, removed_last_run=tuple(removed_aliases)
    )


def _alias_for(policy: Policy, report: PlanReport, offering_id: str) -> str | None:
    alias = report.aliases.get(offering_id)
    if alias is not None:
        return alias
    try:
        return naming.alias_for(policy, offering_id)
    except Exception:  # noqa: BLE001 - an Alias we cannot derive is not reportable
        return None


# --- Rendering -----------------------------------------------------------


def render_text(guidance: Guidance) -> str:
    """Render for an operator to read."""
    lines: list[str] = []
    for warning in guidance.warnings:
        lines.append(f"warning: {warning}")
    if guidance.warnings:
        lines.append("")

    header = f"Ranked by {guidance.axis}"
    if guidance.prefer:
        header += f", preferring {guidance.prefer}"
    lines.append(header)
    lines.append(f"Feed generated at: {guidance.feed_generated_at or 'unstated'}")
    lines.append("")

    if not guidance.rows:
        lines.append("Nothing is offered, so there is nothing to rank.")
        return "\n".join(lines) + "\n"

    for index, row in enumerate(guidance.rows, start=1):
        score = f"{row.score:g}" if row.score is not None else "unscored"
        lines.append(f"{index}. {row.canonical_model_id}  {guidance.axis}={score}")
        lines.append(f"   why: {row.why}")
        for position, route in enumerate(row.routes, start=1):
            state = "available" if route.available else (route.reason or "excluded")
            detail = (
                f"   {position}. {route.alias}  {route.cost_basis}  "
                f"{route.provider_id}  {state}"
            )
            if route.refills_at is not None:
                detail += f"  refills {route.refills_at.isoformat()}"
            lines.append(detail)
        lines.append("")

    lines.extend(_advisory_lines(guidance.advisory))
    return "\n".join(lines).rstrip() + "\n"


def _advisory_lines(advisory: ClientAdvisory) -> list[str]:
    lines = ["Client advisory", f"  {advisory.note}"]
    if advisory.added_last_run:
        lines.append(f"  added last run: {', '.join(advisory.added_last_run)}")
    if advisory.removed_last_run:
        for removed in advisory.removed_last_run:
            detail = f"  removed last run: {removed.alias} — {removed.reason or 'reason unrecorded'}"
            if removed.refills_at is not None:
                detail += f", refills {removed.refills_at.isoformat()}"
            lines.append(detail)
    if not advisory.added_last_run and not advisory.removed_last_run:
        lines.append("  the last run added and removed nothing")
    return lines


def render_markdown(guidance: Guidance) -> str:
    """Render as Markdown, for a scheduled task to redirect into a project."""
    lines: list[str] = [f"# Model guidance — {guidance.axis}", ""]
    lines.append(f"Feed generated at: `{guidance.feed_generated_at or 'unstated'}`.")
    if guidance.derived_at is not None:
        lines.append(f"Derived at: `{guidance.derived_at.isoformat()}`.")
    lines.append("")
    for warning in guidance.warnings:
        lines.append(f"> Warning: {warning}")
    if guidance.warnings:
        lines.append("")

    if not guidance.rows:
        lines.append("Nothing is offered.")
        return "\n".join(lines) + "\n"

    for index, row in enumerate(guidance.rows, start=1):
        score = f"{row.score:g}" if row.score is not None else "unscored"
        lines.append(f"## {index}. `{row.canonical_model_id}` — {guidance.axis} {score}")
        lines.append("")
        # Not `str.capitalize()`: it lowercases every other character, so
        # the term "Feed" came out as "feed". CONTEXT.md's terms keep
        # their case.
        lines.append(row.why[:1].upper() + row.why[1:] + ".")
        lines.append("")
        lines.append("| # | Alias | Cost | Provider | State |")
        lines.append("| --- | --- | --- | --- | --- |")
        for position, route in enumerate(row.routes, start=1):
            state = "available" if route.available else (route.reason or "excluded")
            if route.refills_at is not None:
                state += f" (refills `{route.refills_at.isoformat()}`)"
            lines.append(
                f"| {position} | `{route.alias}` | {route.cost_basis} | "
                f"`{route.provider_id}` | {state} |"
            )
        lines.append("")

    lines.append("## Client advisory")
    lines.append("")
    lines.append(guidance.advisory.note)
    lines.append("")
    if guidance.advisory.added_last_run:
        lines.append("Added on the last run:")
        lines.append("")
        for alias in guidance.advisory.added_last_run:
            lines.append(f"- `{alias}`")
        lines.append("")
    if guidance.advisory.removed_last_run:
        lines.append("Removed on the last run, so no longer callable:")
        lines.append("")
        for removed in guidance.advisory.removed_last_run:
            refill = (
                f", refills `{removed.refills_at.isoformat()}`"
                if removed.refills_at
                else ""
            )
            lines.append(
                f"- `{removed.alias}` — {removed.reason or 'reason unrecorded'}{refill}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
