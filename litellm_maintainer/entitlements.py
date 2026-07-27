"""The Entitlement view: what spending through each provider costs us now.

An Entitlement is the operator's spending relationship with one provider
(CONTEXT.md, "Entitlement"). This module derives that view on read, from
the Feed Document, Policy and Health State. It is not a file, has no
writer, and cannot go stale or disagree with `status`.

**Nothing here propagates a failure.** A provider declared
`shared_pool` reads its Offerings' failures as one pool draining, and
says so in words. It never marks a sibling Excluded and never removes an
Offering from the Generated Config. ADR 0004 records why: a provider can
refuse one tier and serve another, and a pool can run dry while its free
tier keeps answering. Every count below was measured, never inferred.

`pool_siblings` is the one function here the run path reads. It still
propagates no conclusion. It names which Offerings share a pool, so
`reduce` can mark them due for a Probe and MEASURE them. Attention, not
a verdict.

Every function here is a pure transform. It takes a Feed, a Policy, a
Health State mapping and a clock reading, and returns a value. It makes
no network call and writes no file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from litellm_maintainer import naming
from litellm_maintainer.feed import Feed
from litellm_maintainer.plan import PlanReport
from litellm_maintainer.policy import PER_MODEL, SHARED_POOL, Policy
from litellm_maintainer.reduce import OfferingHealth

# What using an Offering costs us, derived from the Feed's own pricing
# kind. These are the terms an orchestrator reasons in: a flat-rate call
# costs no marginal money but drains a window, a metered call bills.
FREE = "free"
FLAT_RATE = "flat_rate"
METERED = "metered"
PASSTHROUGH = "passthrough"
UNKNOWN_BASIS = "unknown"

# The Feed's pricing kind to our cost basis. `subscription_included` is
# flat rate: `pricing.py` already marks its token rate a list price and
# never an amount billed, which is the same distinction in a different
# place.
_BASIS_BY_PRICING_KIND = {
    "free": FREE,
    "free_tier": FREE,
    "subscription_included": FLAT_RATE,
    "paid": METERED,
    "unknown": UNKNOWN_BASIS,
}


@dataclass(frozen=True)
class UnavailableOffering:
    """One admitted Offering that is not being offered right now.

    `reason` is the classify reason that produced the exclusion, so the
    answer to "why is this model missing" sits in the same output as the
    picks. `refills_at` is the reset time the provider's own refusal
    stated, or `None` when it stated none — in which case only a Probe
    can restore it.
    """

    offering_id: str
    alias: str | None
    reason: str | None
    bucket: str | None
    refills_at: datetime | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "offering_id": self.offering_id,
            "alias": self.alias,
            "reason": self.reason,
            "bucket": self.bucket,
            "refills_at": self.refills_at.isoformat() if self.refills_at else None,
        }


@dataclass(frozen=True)
class Entitlement:
    """One provider's spending relationship, as we currently measure it.

    `answering` and `unavailable` are counts of Offerings this Policy
    admits: `answering` are in the Generated Config now, `unavailable`
    are Excluded. `cost_bases` holds every distinct basis among the
    provider's Offerings, because a provider can mix them; `cost_basis`
    is the single one when there is exactly one, and `None` otherwise.

    `earliest_refill_at` is the soonest reset time among the unavailable
    Offerings. It answers "when is it worth trying this provider again"
    without a call, which is the recovery path ADR 0002 protects.
    """

    provider_id: str
    kind: str
    cost_bases: tuple[str, ...]
    answering: int
    unavailable_offerings: tuple[UnavailableOffering, ...] = ()
    withheld: int = 0
    candidates: int = 0

    @property
    def unavailable(self) -> int:
        return len(self.unavailable_offerings)

    @property
    def in_scope(self) -> int:
        """Offerings this Policy admits, whether they answer or not."""
        return self.answering + self.unavailable

    @property
    def cost_basis(self) -> str | None:
        return self.cost_bases[0] if len(self.cost_bases) == 1 else None

    @property
    def earliest_refill_at(self) -> datetime | None:
        times = [o.refills_at for o in self.unavailable_offerings if o.refills_at]
        return min(times) if times else None

    @property
    def state(self) -> str:
        """A one-word reading of the counts. Never an inference.

        `healthy` when everything admitted answers, `dry` when nothing
        does, `degraded` in between, `empty` when Policy admits nothing
        from this provider at all.
        """
        if self.in_scope == 0:
            return "empty"
        if self.unavailable == 0:
            return "healthy"
        if self.answering == 0:
            return "dry"
        return "degraded"

    def as_dict(self) -> dict[str, Any]:
        refill = self.earliest_refill_at
        return {
            "provider_id": self.provider_id,
            "entitlement": self.kind,
            "state": self.state,
            "cost_basis": self.cost_basis,
            "cost_bases": list(self.cost_bases),
            "answering": self.answering,
            "unavailable": self.unavailable,
            "in_scope": self.in_scope,
            "withheld": self.withheld,
            "candidates": self.candidates,
            "earliest_refill_at": refill.isoformat() if refill else None,
            "unavailable_offerings": [o.as_dict() for o in self.unavailable_offerings],
        }


@dataclass(frozen=True)
class EntitlementView:
    """Every Entitlement, plus what the whole picture was derived from.

    Declared Offerings sit outside the Entitlement list on purpose. An
    Entitlement is a relationship with one provider, and a Declared
    Offering has no Feed provider at all. They still need reporting, or
    the answer to "why is claude-opus-5 missing" appears nowhere, so they
    get their own counts and their own unavailable list.
    """

    entitlements: tuple[Entitlement, ...] = ()
    declared_answering: int = 0
    declared_unavailable: tuple[UnavailableOffering, ...] = ()
    feed_generated_at: str | None = None
    derived_at: datetime | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def declared_in_scope(self) -> int:
        return self.declared_answering + len(self.declared_unavailable)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "derived_at": self.derived_at.isoformat() if self.derived_at else None,
            "feed_generated_at": self.feed_generated_at,
            "warnings": list(self.warnings),
            "entitlements": [e.as_dict() for e in self.entitlements],
            "declared": {
                "answering": self.declared_answering,
                "in_scope": self.declared_in_scope,
                "unavailable": [o.as_dict() for o in self.declared_unavailable],
            },
        }


# Consumers parse this output, so it carries its own version. Raise it
# when a field changes meaning or leaves; adding a field does not need a
# raise.
SCHEMA_VERSION = "1"


def cost_basis_for_pricing_kind(kind: str | None) -> str:
    """Map the Feed's pricing kind to what it costs us. Unknown stays unknown."""
    if kind is None:
        return UNKNOWN_BASIS
    return _BASIS_BY_PRICING_KIND.get(kind, UNKNOWN_BASIS)


def declared_pool_id(declared) -> str | None:
    """Name the pool a Declared Offering is billed from, or `None`.

    The CREDENTIAL identifies the pool, because the credential is what
    gets billed. Two Offerings billed to one key share an allowance
    whatever else differs about them.

    That rule is not a convenience. The operator runs two ChatGPT seats
    behind `openai/`, six Aliases each, and they are two separate
    subscriptions. Any provider-level field would call them one pool.
    Their credentials — `LITELLM_CHATGPT_SEAT1_WORKER_KEY` and
    `...SEAT2...` — separate them with nothing to configure, and a third
    seat would appear on its own.

    `entitlement_pool` overrides it, for the two cases the credential
    gets wrong: two keys billed to one account (the rule under-groups),
    and one key spanning a subscription plus pay-as-you-go (it
    over-groups). Both are survivable — this propagates attention, never
    a verdict — but neither is imaginary.

    Return `None` for an Offering with neither. A Passthrough Auth
    Offering carries no credential at all, so it joins no pool unless
    `entitlement_pool` names one.
    """
    if declared.entitlement_pool:
        return f"named:{declared.entitlement_pool}"
    credential = declared.litellm_params.get("api_key")
    if isinstance(credential, str) and credential:
        return f"credential:{credential}"
    return None


def pool_siblings(*, feed: Feed, policy: Policy) -> dict[str, frozenset[str]]:
    """Map each Offering on a shared pool to its pool mates.

    `reduce` reads this to mark a sibling `probe_due` when the pool
    reports a quota exhaustion. It marks nothing Excluded: ADR 0004
    forbids propagating the conclusion, and this propagates only the
    decision to MEASURE.

    Group over the Offerings Policy currently ADMITS
    (`prober._discovered_admitted`, the Prober's own worklist source),
    never the whole Feed. An Offering Policy does not admit cannot be
    probed and cannot be in the Generated Config, so marking it due
    would ask for a Probe that never runs and a mark that never clears.

    Include a `per_model` Offering not at all. It is billed separately,
    so its quota says nothing about another's.

    A Discovered Offering's pool is its provider, which is where its
    credential comes from: the Feed states one `credential_hint` per
    provider. A Declared Offering's pool comes from `declared_pool_id`.
    Both are the same rule — the credential identifies the pool — read
    from the two places the two kinds of Offering keep it.

    A Client-Facing Variant contributes its `health_key`, so a pair
    counts once.
    """
    from litellm_maintainer.prober import _discovered_admitted

    pools: dict[str, set[str]] = {}

    for offering_id in _discovered_admitted(feed, policy):
        provider_id = offering_id.partition(":")[0]
        rule = policy.providers.get(provider_id)
        if rule is None or (rule.entitlement or PER_MODEL) != SHARED_POOL:
            continue
        pools.setdefault(f"provider:{provider_id}", set()).add(offering_id)

    for declared in policy.declared:
        if (declared.entitlement or PER_MODEL) != SHARED_POOL:
            continue
        pool_id = declared_pool_id(declared)
        if pool_id is None:
            continue
        pools.setdefault(pool_id, set()).add(declared.health_key)

    mapping: dict[str, frozenset[str]] = {}
    for members in pools.values():
        if len(members) < 2:
            continue
        frozen = frozenset(members)
        for key in members:
            mapping[key] = frozen
    return mapping


def sub_allowance_keys(policy: Policy) -> frozenset[str]:
    """Health Keys that are capped INSIDE their pool.

    A sub-allowance's own exhaustion says nothing about the pool, so it
    propagates nothing outward. The pool's exhaustion still reaches it,
    so it is an ordinary recipient. One-way containment.

    See `DeclaredOffering.sub_allowance` for the measured case.
    """
    return frozenset(d.health_key for d in policy.declared if d.sub_allowance)


def derive(
    *,
    feed: Feed,
    policy: Policy,
    health: dict[str, OfferingHealth],
    report: PlanReport,
    now: datetime,
    warnings: tuple[str, ...] = (),
) -> EntitlementView:
    """Derive the Entitlement view. Pure: reads values, returns a value.

    `report` comes from `plan` over the same Feed, Policy and Health
    State, so this view and `status` can never disagree about what is
    offered.
    """
    admitted_by_provider: dict[str, list[str]] = {}
    for offering_id in report.admitted:
        provider_id = offering_id.partition(":")[0]
        admitted_by_provider.setdefault(provider_id, []).append(offering_id)

    excluded_by_provider: dict[str, list[str]] = {}
    for offering_id in report.excluded:
        provider_id = offering_id.partition(":")[0]
        excluded_by_provider.setdefault(provider_id, []).append(offering_id)

    withheld_counts: dict[str, int] = {}
    for offering_id in report.withheld:
        provider_id = offering_id.partition(":")[0]
        withheld_counts[provider_id] = withheld_counts.get(provider_id, 0) + 1

    candidate_counts: dict[str, int] = {}
    for offering_id in report.candidates:
        provider_id = offering_id.partition(":")[0]
        candidate_counts[provider_id] = candidate_counts.get(provider_id, 0) + 1

    entitlements: list[Entitlement] = []
    for provider_id in sorted(policy.providers):
        rule = policy.providers[provider_id]
        admitted = admitted_by_provider.get(provider_id, [])
        excluded = excluded_by_provider.get(provider_id, [])

        bases = set()
        for offering_id in list(admitted) + list(excluded):
            offering = feed.offering(offering_id)
            if offering is not None:
                bases.add(cost_basis_for_pricing_kind(offering.pricing_kind))

        unavailable = tuple(
            UnavailableOffering(
                offering_id=offering_id,
                alias=_alias_for(policy, report, offering_id),
                reason=_health_field(health, offering_id, "reason"),
                bucket=_health_field(health, offering_id, "bucket"),
                refills_at=_health_field(health, offering_id, "reset_at"),
            )
            for offering_id in sorted(excluded)
        )

        entitlements.append(
            Entitlement(
                provider_id=provider_id,
                kind=rule.entitlement or PER_MODEL,
                cost_bases=tuple(sorted(bases)),
                answering=len(admitted),
                unavailable_offerings=unavailable,
                withheld=withheld_counts.get(provider_id, 0),
                candidates=candidate_counts.get(provider_id, 0),
            )
        )

    # A Declared Offering's Health Key is its Alias, because it has no
    # Feed id (CONTEXT.md, "Health Key"). Read health by Alias here.
    declared_answering = 0
    declared_unavailable: list[UnavailableOffering] = []
    for declared in policy.declared:
        if declared.alias in report.admitted:
            declared_answering += 1
        elif declared.alias in report.excluded:
            declared_unavailable.append(
                UnavailableOffering(
                    offering_id=declared.alias,
                    alias=declared.alias,
                    reason=_health_field(health, declared.alias, "reason"),
                    bucket=_health_field(health, declared.alias, "bucket"),
                    refills_at=_health_field(health, declared.alias, "reset_at"),
                )
            )

    return EntitlementView(
        entitlements=tuple(entitlements),
        declared_answering=declared_answering,
        declared_unavailable=tuple(
            sorted(declared_unavailable, key=lambda o: o.offering_id)
        ),
        feed_generated_at=feed.generated_at,
        derived_at=now,
        warnings=warnings,
    )


def _health_field(health: dict[str, OfferingHealth], offering_id: str, name: str):
    record = health.get(offering_id)
    return getattr(record, name) if record is not None else None


def _alias_for(policy: Policy, report: PlanReport, offering_id: str) -> str | None:
    """The Alias for an Offering, admitted or not.

    `PlanReport.aliases` holds admitted Offerings only, so reading it
    alone reported no Alias for every Excluded Offering — the exact set
    this view exists to explain. On the operator's own instance that was
    all seven unavailable Offerings. Derive the Alias instead when the
    report does not carry it: the operator knows the Alias, not the Feed
    id, so an unavailable Offering with no Alias answers nothing.
    """
    alias = report.aliases.get(offering_id)
    if alias is not None:
        return alias
    try:
        return naming.alias_for(policy, offering_id)
    except Exception:  # noqa: BLE001 - an Alias we cannot derive is not reportable
        return None


# --- Rendering -----------------------------------------------------------
#
# Three formats, one derivation. `json` is what an agent parses, `text`
# is what an operator reads, and `markdown` exists so a scheduled task
# can redirect this command into a project's own documentation. None of
# them writes a file: each returns text for the caller to print or
# redirect.


def _pool_note(entitlement: Entitlement) -> str:
    """Why several Offerings failed together, in the operator's own terms.

    This is the whole value of the `entitlement` declaration. It explains
    a measured pattern. It never adds a claim about an Offering nobody
    probed. See ADR 0004.
    """
    if entitlement.kind != "shared_pool":
        return ""
    if entitlement.state == "dry":
        return "one shared pool, and every admitted Offering has refused"
    if entitlement.state == "degraded":
        return (
            "one shared pool: these refusals are probably the same pool draining, "
            "and the Offerings still answering may follow"
        )
    return "one shared pool"


def render_text(view: EntitlementView) -> str:
    """Render the Entitlement view for an operator to read."""
    lines: list[str] = []
    for warning in view.warnings:
        lines.append(f"warning: {warning}")
    if view.warnings:
        lines.append("")

    lines.append(f"Feed generated at: {view.feed_generated_at or 'unstated'}")
    lines.append("")

    if not view.entitlements:
        lines.append("Policy names no provider, so there is no Entitlement to report.")
        return "\n".join(lines) + "\n"

    for entitlement in view.entitlements:
        basis = entitlement.cost_basis or "/".join(entitlement.cost_bases) or "unknown"
        lines.append(
            f"{entitlement.provider_id}  {entitlement.state}  "
            f"{entitlement.kind}  {basis}"
        )
        lines.append(
            f"  {entitlement.answering} of {entitlement.in_scope} answering"
            + (f", {entitlement.withheld} withheld" if entitlement.withheld else "")
            + (f", {entitlement.candidates} awaiting approval" if entitlement.candidates else "")
        )
        note = _pool_note(entitlement)
        if note:
            lines.append(f"  {note}")
        refill = entitlement.earliest_refill_at
        if refill is not None:
            lines.append(f"  earliest refill: {refill.isoformat()}")
        for offering in entitlement.unavailable_offerings:
            detail = f"  unavailable: {offering.offering_id}"
            if offering.alias:
                detail += f" ({offering.alias})"
            detail += f" — {offering.reason or 'reason unrecorded'}"
            if offering.refills_at is not None:
                detail += f", refills {offering.refills_at.isoformat()}"
            lines.append(detail)
        lines.append("")

    if view.declared_in_scope:
        lines.append(
            f"declared  {view.declared_answering} of {view.declared_in_scope} answering"
        )
        lines.append("  Offerings you declared. The Feed does not publish them.")
        for offering in view.declared_unavailable:
            lines.append(
                f"  unavailable: {offering.alias} — "
                f"{offering.reason or 'reason unrecorded'}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def render_markdown(view: EntitlementView) -> str:
    """Render the Entitlement view as Markdown, for a scheduled task to redirect."""
    lines: list[str] = ["# Entitlements", ""]
    lines.append(f"Feed generated at: `{view.feed_generated_at or 'unstated'}`.")
    if view.derived_at is not None:
        lines.append(f"Derived at: `{view.derived_at.isoformat()}`.")
    lines.append("")
    for warning in view.warnings:
        lines.append(f"> Warning: {warning}")
    if view.warnings:
        lines.append("")

    if not view.entitlements:
        lines.append("Policy names no provider.")
        return "\n".join(lines) + "\n"

    lines.append("| Provider | State | Pool | Cost | Answering | Earliest refill |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for e in view.entitlements:
        basis = e.cost_basis or "/".join(e.cost_bases) or "unknown"
        refill = e.earliest_refill_at
        lines.append(
            f"| `{e.provider_id}` | {e.state} | {e.kind} | {basis} | "
            f"{e.answering}/{e.in_scope} | {refill.isoformat() if refill else '—'} |"
        )
    lines.append("")

    if view.declared_in_scope:
        lines.append(
            f"Declared Offerings: {view.declared_answering} of "
            f"{view.declared_in_scope} answering. The Feed does not publish "
            "these; you declared them."
        )
        lines.append("")

    unavailable = [(e, o) for e in view.entitlements for o in e.unavailable_offerings]
    unavailable += [(None, o) for o in view.declared_unavailable]
    if unavailable:
        lines.append("## Unavailable now")
        lines.append("")
        for entitlement, offering in unavailable:
            alias = f"`{offering.alias}`" if offering.alias else "no Alias"
            refill = (
                f", refills `{offering.refills_at.isoformat()}`"
                if offering.refills_at
                else ""
            )
            lines.append(
                f"- `{offering.offering_id}` ({alias}) — "
                f"{offering.reason or 'reason unrecorded'}{refill}"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
