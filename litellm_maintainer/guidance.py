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
    DECLARED_PROVIDER_ID,
    FLAT_RATE,
    FREE,
    METERED,
    PASSTHROUGH,
    UNKNOWN_BASIS,
    _scale_note_for_allowance,
    _tier_for_allowance,
    allowance_id_for_declared,
    allowance_id_for_provider,
    cost_basis_for_pricing_kind,
)
from litellm_maintainer.classify import REASON_QUOTA_EXHAUSTED
from litellm_maintainer.feed import Feed, Offering
from litellm_maintainer.headroom import (
    format_age,
    format_used_percent,
    HeadroomState,
    reading_age_seconds,
    route_binding_window,
    slot_id_for_health_key,
)
from litellm_maintainer.notify import PreviousRunState
from litellm_maintainer.plan import PlanReport
from litellm_maintainer.policy import DeclaredOffering, Policy
from litellm_maintainer.reduce import OfferingHealth

# Consumers parse this output, so it carries its own version. Raise it
# when a field changes meaning or leaves.
# Raised to "2" on 2026-07-27: a Route gained `wide_alias`, and a
# hand-declared Client-Facing Variant folds into its sibling's row rather
# than forming one of its own. Both change the shape a consumer parses.
# Raised to "3" on 2026-07-28: a row gained `score_source` and a Route
# gained `rate_source`, and a Declared Offering naming a Reference Model
# folds onto that model's row instead of forming an unscored row of its
# own. See ADR 0011.
#
# READ THIS BEFORE BUMPING. Both bumps above accompanied a new field, so
# the pattern looks like "a new field bumps". It is not. Each bump was for
# the ROW FOLDING that shipped with the field — a variant folding onto its
# sibling, then a Reference Model folding onto a model's row — because a
# consumer counting rows got a different answer afterwards. The fields rode
# along.
#
# A purely additive field does NOT bump. `allowance_id` and `fair_use`
# arrived on 2026-07-28 and this stayed "3", because a consumer that
# ignores them parses exactly what it parsed before. `headroom` arrived
# the same way, on the same day (ticket 05): `null` for a Route whose
# Allowance has no source, exactly the value every Route published before
# it existed.
#
# The rule matters in one direction especially: a downstream client pins
# major 3 and fails loudly on anything else, so a bump it was not told
# about takes the whole proxy away from it. Coordinate a real bump; never
# reach for one out of caution.
SCHEMA_VERSION = "3"

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
#
# Defined in `entitlements`, which reports the same string on a Declared
# Allowance entry. Two definitions of one wire value drift, and this one is
# read by a client.
DECLARED_PROVIDER = DECLARED_PROVIDER_ID

# Where a number came from. A caller weighs a Feed figure against an
# operator's figure differently, so the answer never presents one as the
# other. See ADR 0011.
#
# `feed`      — an Offering this proxy serves, as the Feed states it.
# `reference` — a Reference Model: the same model served elsewhere, whose
#               numbers the Feed does state. The rate is another vendor's.
# `operator`  — the operator wrote it in Policy.
SOURCE_FEED = "feed"
SOURCE_REFERENCE = "reference"
SOURCE_OPERATOR = "operator"

# Why a Route is not `recommendable`. A caller reads this to tell a fact
# from a report: an observed refusal names `NOT_RECOMMENDED_EXHAUSTED`,
# a Reading through a mapping that can rot names
# `NOT_RECOMMENDED_HEADROOM`, and an Excluded Offering names
# `NOT_RECOMMENDED_HEALTH`. See ADR 0010, the headroom spec (decision 7
# in ticket 08's issue), and CONTEXT.md, "Exhausted".
NOT_RECOMMENDED_EXHAUSTED = "exhausted"
NOT_RECOMMENDED_HEADROOM = "headroom"
NOT_RECOMMENDED_HEALTH = "health"

# The pricing kind a Reference Model's rate may be read from. A free
# mirror states 0.00, which describes that mirror's promotion rather than
# the model's rate, so reading it would report every model with a free
# tier as costless everywhere. Only a `paid` Offering states a rate.
_REFERENCE_RATE_PRICING_KIND = "paid"


class GuidanceError(ValueError):
    """A caller asked for something this module cannot answer."""


@dataclass(frozen=True)
class RouteHeadroom:
    """A Route's Binding Window figure, and nothing beneath it.

    Built by `_route_headroom` from `HeadroomState` and the shared
    derivation in `litellm_maintainer.headroom` (`route_binding_window`,
    `reading_age_seconds`) — never a second copy of either. `entitlements`
    publishes the full window set for the same Allowance; a Route
    publishes only the number that binds, because an agent reads the Route
    it is about to dispatch to and nothing else (headroom spec, decision
    11; CONTEXT.md, "Binding Window").

    `used_percent` is the WORSE of the parent Allowance's own figure and
    this Route's Sub-allowance's figure, where one is named (headroom
    spec, decision 12). Two Routes sharing one Allowance can therefore
    carry different figures: a fable Route reads its own drained window
    while a sibling Route on the same Allowance still reads the parent's.

    `age_seconds` is computed from the Reading's OWN timestamp, never from
    when we last read Headroom State — the same rule `entitlements`
    follows, for the same reason: codexbar polls on its own schedule.
    """

    used_percent: float
    window_minutes: float | None
    resets_at: str | None
    age_seconds: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "used_percent": self.used_percent,
            "window_minutes": self.window_minutes,
            "resets_at": self.resets_at,
            "age_seconds": self.age_seconds,
        }


@dataclass(frozen=True)
class Route:
    """One Alias through which a row's Canonical Model can be reached."""

    alias: str
    offering_id: str
    provider_id: str
    cost_basis: str
    available: bool
    entitlement: str
    # Which Allowance this Route draws on: the thing that gets billed
    # (CONTEXT.md, "Allowance"). Two Routes sharing this string share one
    # ceiling, and draining one drains the other.
    #
    # `provider_id` cannot answer this. Every Declared Route reports
    # `declared`, so before this field a client could not tell one
    # subscription seat from another, and could not refuse a fair-use host
    # without refusing every Declared Route with it.
    #
    # Never derived from an Alias. The Aliases here do encode the seat, and
    # guessing from them is exactly what this field exists to prevent: the
    # naming rule is an operator setting, so a guess breaks the day it
    # changes. See ADR 0012.
    allowance_id: str | None = None
    # Whether this Route's Allowance tolerates load badly — an unmetered
    # plan under a "fair use" clause. Operator-stated; the Feed has no such
    # concept, so a Discovered Route is always `False`.
    #
    # It does NOT change this Route's rank. It is a risk, not a cost, so the
    # Route still sorts by its cost basis and a caller filters on this
    # field. See ADR 0012.
    fair_use: bool = False
    # The subscription level this Route's Allowance bills under, as the
    # operator states it in `policy.allowances.<id>.tier` (CONTEXT.md,
    # "Tier"). `None` when Policy names no entry. Published verbatim: a
    # label, never parsed, ranked or derived from, and it does NOT change
    # this Route's rank. `entitlements` publishes the same string for the
    # same Allowance — one read answers the question either way.
    tier: str | None = None
    # How big this Allowance is, where the vendor states a size but sells
    # no Tier. Prose, verbatim, ranked by nothing. Read it beside
    # `headroom`, exactly as `tier` is read: both answer "a share of
    # WHAT". See `policy.AllowanceInfo.scale_note`.
    scale_note: str | None = None
    # How fast THIS Offering draws on its Allowance, as the operator
    # states it in `policy.draw_notes`. Prose, verbatim, ranked by nothing.
    #
    # A different question from `scale_note`, which sizes the whole
    # Allowance: a pool can hold six Offerings that empty it at six rates,
    # and for a subscription Offering the Feed publishes no rate at all.
    draw_note: str | None = None
    # This Route's Binding Window figure, or `None`. `None` covers every
    # case that must never read as free capacity: Policy names no
    # `headroom_source` for this Allowance, no Reading has been captured
    # yet, or every window in the Reading is void. See ADR 0013 and the
    # headroom spec, decisions 2, 8 and 11. It does NOT change this
    # Route's rank or `recommendable` — demotion is a later ticket.
    #
    # For a Route on a declared Sub-allowance, this is the WORSE of the
    # parent Allowance's figure and the Sub-allowance's own window — never
    # the parent's figure alone. A sibling Route on the same Allowance that
    # names no Sub-allowance is unaffected (headroom spec, decision 12).
    headroom: RouteHeadroom | None = None
    # Whether this Route's Binding Window reads 100% AND Policy's
    # `headroom.demote_at_full` is `True`. `False` by default, and it
    # stays `False` whenever the flag is off or `headroom` is `None` --
    # a void or absent Reading never demotes (headroom spec, decision
    # 7). Computed once in `derive`, from `_route_headroom`'s own
    # output, never from a second read of Headroom State.
    #
    # This field can only ever turn `recommendable` from `True` to
    # `False`; nothing here can turn it back. That is how "a Headroom
    # may demote a Route, it may never un-demote one" (headroom spec,
    # decision 8) holds without a second rule: an observed exhaustion
    # keeps demoting a Route regardless of what a later Reading says,
    # because `recommendable` is a plain AND of every demoting fact.
    demoted_by_headroom: bool = False
    input_usd_per_1m_tokens: float | None = None
    output_usd_per_1m_tokens: float | None = None
    # Where the two rates above came from: `SOURCE_FEED`,
    # `SOURCE_REFERENCE`, `SOURCE_OPERATOR`, or `None` when no rate is
    # stated. A `reference` rate is what ANOTHER vendor charges for the
    # same model, so it states the relative burn and never this Route's
    # bill. See ADR 0011.
    rate_source: str | None = None
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
    # Whether Health State records this Offering Excluded. Such a Route
    # stays `available`, because an exclusion no longer removes it from
    # the Generated Config (ADR 0014). It must not be recommended: the
    # maintainer called this Offering and was told no.
    #
    # Read this beside `available`, never instead of it. `available`
    # answers "is the Alias in the file"; this answers "did it work".
    excluded: bool = False
    # When the PROXY last refused this Alias as one it does not serve.
    # A measurement of our own pipeline, not of the Offering.
    #
    # There is no `served: true` beside it, and there will not be. The
    # proxy states "not served" on a failed call and states nothing on a
    # successful one, and the Prober bypasses the proxy, so nothing
    # measures the positive. Judge this timestamp's age as you would a
    # Reading's.
    not_served_at: datetime | None = None

    @property
    def recommendable(self) -> bool:
        """Whether this Route may be handed to a caller as the answer.

        Available AND not Excluded AND not exhausted AND not demoted by
        a Headroom. Removing a failing Route from the config instead
        would give the caller "model not found" in place of the
        provider's own "your quota resets at 09:00", and for a
        Passthrough Auth Offering nothing could ever clear it: no Probe
        is possible, so only the clock can.

        `demoted_by_headroom` is `False` by default and stays `False`
        with `headroom.demote_at_full` off.
        """
        return (
            self.available
            and not self.excluded
            and not self.exhausted
            and not self.demoted_by_headroom
        )

    @property
    def not_recommended_because(self) -> str | None:
        """Which fact makes this Route not `recommendable`, or `None`.

        `None` when the Route is recommendable. Otherwise the first of
        three causes that applies, in this order:

        `NOT_RECOMMENDED_HEALTH` -- the Offering is Excluded, or it is
        Unlisted and therefore not `available`. This is a fact about the
        Offering itself, not about the calling client's own allowance.
        An Excluded Offering is still `available`, because an exclusion
        no longer removes it from the Generated Config (ADR 0014), so
        both facts map to this one cause.

        `NOT_RECOMMENDED_EXHAUSTED` -- a recorded quota exhaustion has
        not cleared (ADR 0010). This is a measured refusal: the
        maintainer called this Offering and was told no.

        `NOT_RECOMMENDED_HEADROOM` -- `headroom.demote_at_full` is on
        and this Route's Binding Window reads 100%. This is a REPORT,
        not a measurement: it travels through a hand-written Policy
        mapping and a source that documents no contract, and either can
        rot. An operator debugging why a Route stopped being
        recommended reads this field to tell the two apart without
        opening Health State by hand.

        Checked in this order because a fact always outranks a report:
        an Excluded or exhausted Route names its own cause even when a
        Headroom also reads 100% for it.
        """
        if self.recommendable:
            return None
        if not self.available or self.excluded:
            return NOT_RECOMMENDED_HEALTH
        if self.exhausted:
            return NOT_RECOMMENDED_EXHAUSTED
        return NOT_RECOMMENDED_HEADROOM

    @property
    def rate_is_list_price(self) -> bool:
        """Whether this rate is a burn to RANK, never a bill to sum.

        Read what this does and does not claim, because a caller
        classifying spend rests everything on it.

        It is derived from `cost_basis` alone and reads nothing about the
        rate field. It licenses one thing: rank Routes by this figure,
        and never add it to an invoice. Summing a subscription's token
        rate would bill a night that cost nothing.

        It does NOT claim the Offering is subscription-included. A `free`
        Route is `true` here too, and its rate of 0 is its actual bill —
        over-broad, and harmless, because summing 0 harms nobody.

        `true` beside a `null` rate carries no information: there is no
        figure to rank. That is not a contradiction, because this field
        restates `cost_basis`, and `cost_basis` is the field to read for
        a claim about the Offering itself.
        """
        return self.cost_basis in (FREE, FLAT_RATE)

    def as_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "offering_id": self.offering_id,
            "provider_id": self.provider_id,
            "cost_basis": self.cost_basis,
            "allowance_id": self.allowance_id,
            "fair_use": self.fair_use,
            "tier": self.tier,
            "scale_note": self.scale_note,
            "draw_note": self.draw_note,
            "headroom": self.headroom.as_dict() if self.headroom else None,
            "entitlement": self.entitlement,
            "available": self.available,
            "rate_is_list_price": self.rate_is_list_price,
            "input_usd_per_1m_tokens": self.input_usd_per_1m_tokens,
            "output_usd_per_1m_tokens": self.output_usd_per_1m_tokens,
            "rate_source": self.rate_source,
            "context_tokens": self.context_tokens,
            "max_output_tokens": self.max_output_tokens,
            "wide_alias": self.wide_alias,
            "reason": self.reason,
            "refills_at": self.refills_at.isoformat() if self.refills_at else None,
            "exhausted": self.exhausted,
            "excluded": self.excluded,
            "not_served_at": (
                self.not_served_at.isoformat() if self.not_served_at else None
            ),
            "recommendable": self.recommendable,
            "not_recommended_because": self.not_recommended_because,
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
    # Where this row's `scores` block came from: `SOURCE_FEED` when the
    # Feed publishes an Offering this proxy serves, `SOURCE_REFERENCE` when
    # every Route here is Declared and the numbers come from a Reference
    # Model the proxy does NOT serve, and `None` when the Feed states
    # nothing about this model at all. See ADR 0011.
    #
    # It describes the SOURCE, never the presence, of a figure. A row can
    # read `reference` and still carry `score: None` on the requested
    # axis: the Feed scores most of the catalogue on `coding` and almost
    # none of it on `speed`.
    score_source: str | None = None

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
    def not_callable_because(self) -> str | None:
        """Which kind of fact makes this row not `callable_now`, or `None`.

        The row-level counterpart of `Route.not_recommended_because`, and
        it exists for the same reason one level up: a caller could tell
        why a ROUTE was withheld and not why a ROW was, so a row held back
        purely by Readings read exactly like one whose every Route the
        maintainer had called and seen refuse.

        `NOT_RECOMMENDED_HEADROOM` means EVERY Route here is demoted only
        by a Reading. Nothing was measured, `headroom.demote_at_full` is
        on, and each Binding Window reads 100%. A caller that declines to
        refuse work on the strength of a Reading may keep this row; the
        Routes inside it are still `available` and not `exhausted`.

        Any other value is measured, and follows `not_recommended_because`
        precedence over the Routes that are not merely demoted:
        `NOT_RECOMMENDED_HEALTH` first, then
        `NOT_RECOMMENDED_EXHAUSTED`. A mixed row therefore names the
        measured cause, never the report — a fact outranks a report here
        exactly as it does on a Route. Read each Route's own field to find
        which of them are merely demoted.

        Asked 2026-07-29 by a consumer that keeps a Route whose only cause
        is a Reading, and then found the row gate dropping it anyway.
        """
        if self.callable_now:
            return None
        reasons = [route.not_recommended_because for route in self.routes]
        if reasons and all(reason == NOT_RECOMMENDED_HEADROOM for reason in reasons):
            return NOT_RECOMMENDED_HEADROOM
        if NOT_RECOMMENDED_HEALTH in reasons:
            return NOT_RECOMMENDED_HEALTH
        if NOT_RECOMMENDED_EXHAUSTED in reasons:
            return NOT_RECOMMENDED_EXHAUSTED
        return NOT_RECOMMENDED_HEALTH

    @property
    def why(self) -> str:
        """One line stating why this row is where it is, from its own inputs."""
        route = self.best_route
        if route is None:
            # Never "every Route is excluded" unconditionally. A demoted
            # Route is `available` and not `exhausted`, so that sentence
            # stated something false about it (measured 2026-07-29).
            if self.not_callable_because == NOT_RECOMMENDED_HEADROOM:
                return (
                    "no Route is recommended right now; every Route is demoted on a "
                    "Headroom reading, which is a report and not a measured refusal"
                )
            if self.not_callable_because == NOT_RECOMMENDED_EXHAUSTED:
                return "no Route answers right now; every Route is exhausted or excluded"
            return "no Route answers right now; every Route is excluded"
        parts = []
        if self.score is not None and self.score_source == SOURCE_REFERENCE:
            # The score describes the model, and the model is the same one.
            # The Route is not the one the Feed scored, so say so: nothing
            # here was measured against THIS endpoint.
            parts.append(
                f"scores {self.score:g} on the requested axis, read from the "
                "same model served elsewhere, so treat it as a reference"
            )
        elif self.score is not None:
            parts.append(f"scores {self.score:g} on the requested axis")
        elif route.entitlement == "declared" and self.score_source is None:
            # Not a gap in the Feed: the operator declared this Offering
            # because the Feed does not cover it. Ranking it last is a
            # consequence of having no score, not a judgement about it.
            #
            # `score_source is None` is the test, not the Route's
            # entitlement alone. A Declared Offering naming a Reference
            # Model IS scored by the Feed, and can still carry no score on
            # ONE axis — `speed` is unscored on most of the catalogue. That
            # row must not claim the Feed covers nothing about it.
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
        if route.fair_use:
            # Said in words as well as in a field, because the cost basis
            # beside it reads as safe and this is the part that is not.
            parts.append("fair-use allowance, so pace bulk work or name it first")
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
            "score_source": self.score_source,
            "scores": dict(self.scores),
            "capabilities": list(self.capabilities),
            "callable_now": self.callable_now,
            "not_callable_because": self.not_callable_because,
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
class Routeless:
    """One Offering the Feed publishes that reached no Route.

    A caller resolving a named Alias and finding nothing can otherwise
    say only "not offered", and several different situations produce
    that. `stage` names the one that blocked it, as `plan` recorded it.

    `alias` is the Alias this Offering WOULD carry, where naming can
    derive one, and `None` where it cannot. `refills_at` is a recorded
    quota exhaustion's own reset time; `None` means no reset time was
    stated, never that it returns immediately.
    """

    offering_id: str
    stage: str
    alias: str | None = None
    refills_at: datetime | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "offering_id": self.offering_id,
            "stage": self.stage,
            "alias": self.alias,
            "refills_at": self.refills_at.isoformat() if self.refills_at else None,
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
    routeless: tuple[Routeless, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "axis": self.axis,
            "prefer": self.prefer,
            "derived_at": self.derived_at.isoformat() if self.derived_at else None,
            "feed_generated_at": self.feed_generated_at,
            "warnings": list(self.warnings),
            "client_advisory": self.advisory.as_dict(),
            "routeless": [r.as_dict() for r in self.routeless],
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


def _route_headroom(
    headroom_state: HeadroomState | None,
    allowance_id: str | None,
    *,
    headroom_sources: dict[str, str],
    now: datetime,
    maximum_staleness_hours: float,
    sub_allowance_window_id: str | None = None,
    slot_windows: dict[str, str] | None = None,
) -> RouteHeadroom | None:
    """The Binding Window figure for one Route, or `None`.

    Reuses `route_binding_window` and `reading_age_seconds` from
    `litellm_maintainer.headroom` — the same derivation `entitlements`
    reads for the parent figure, never a second copy of it.

    `sub_allowance_window_id` names the id that measures THIS Route's own
    Sub-allowance, or `None` for an ordinary Route. The caller resolves it
    by Health Key against Policy's `members` map
    (`headroom.slot_id_for_health_key`, ticket 10), so it always names a
    slot id from `slot_windows` when it names anything at all. Passing it
    makes this Route bind on the WORSE of its Allowance's own windows and
    that window (headroom spec, decision 12; CONTEXT.md, "Sub-allowance"):
    a sibling Route on the same Allowance that names no Sub-allowance is
    unaffected, because `route_binding_window` reads it for this call
    alone.

    `slot_windows` is Policy's `headroom.sources.<id>.windows` mapping
    for THIS Route's Allowance (`Headroom.source_windows`), passed
    whether or not this Route names a Sub-allowance: a named slot leaves
    the PARENT computation for every Route on the Allowance, not only the
    one that names it, so an ordinary sibling Route needs it too (ticket
    09's Gemini case: `Flash` and `Flash Lite` never name a Sub-allowance
    of their own, but they must still see `Pro` excluded from the parent
    figure they would otherwise borrow).

    `headroom_sources` is Policy's own `headroom.sources` map, read
    fresh here rather than trusted from the stored record: a record
    survives on disk until the next `headroom refresh` prunes it, so a
    Reading whose Allowance Policy already unmapped -- or remapped to a
    different codexbar identity -- would otherwise still publish and
    still demote. `None` unless BOTH the Allowance is still declared AND
    the stored record's own `source` still equals what Policy currently
    states for it (headroom spec, decision 2).

    `None` covers every other case that must read as unknown, never as
    free capacity (headroom spec, decision 2): no Headroom State was
    read at all, this Route names no Allowance, or the stored Reading
    has no live window left.
    """
    if headroom_state is None or allowance_id is None:
        return None
    declared_source = headroom_sources.get(allowance_id)
    if declared_source is None:
        return None
    record = headroom_state.records.get(allowance_id)
    if record is None or record.source != declared_source:
        return None
    binding = route_binding_window(
        record.reading,
        sub_allowance_window_id=sub_allowance_window_id,
        now=now,
        maximum_staleness_hours=maximum_staleness_hours,
        slot_windows=slot_windows,
    )
    if binding is None:
        return None
    return RouteHeadroom(
        used_percent=binding.used_percent,
        window_minutes=binding.window_minutes,
        resets_at=binding.resets_at,
        age_seconds=reading_age_seconds(record.reading, now=now),
    )


# The figure a Binding Window reads at full draw. There is no warn band:
# the line is drawn only where the provider draws it (headroom spec,
# decision 7 in ticket 08's issue). A caller reads the used share below
# 100% and sets its own bar.
_FULLY_DRAWN_PERCENT = 100


def _demoted_by_headroom(headroom: RouteHeadroom | None, *, demote_at_full: bool) -> bool:
    """Whether a Route's Binding Window demotes it, under ticket 08's flag.

    `False` whenever `demote_at_full` is off -- the default -- so a
    Policy that has not opted in gets exactly the answer it always got.

    `False` too when `headroom` is `None`: a void or absent Reading
    demotes nothing (headroom spec, decision 7). Only a live Binding
    Window reading exactly `_FULLY_DRAWN_PERCENT` demotes, because there
    is no warn band -- the line sits only where the provider draws it.
    """
    if not demote_at_full or headroom is None:
        return False
    return headroom.used_percent >= _FULLY_DRAWN_PERCENT


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
    headroom_state: HeadroomState | None = None,
) -> Guidance:
    """Build the ranked answer. Pure: reads values, returns a value.

    Raise `GuidanceError` when `axis` or `prefer` names something this
    module cannot answer, rather than quietly answering a different
    question.

    `headroom_state` is `None` by default, so a caller that has not read
    Headroom State yet gets exactly the answer it always got: every
    Route's `headroom` reads `None`. Each Route's figure is built once
    here, from that Route's own Allowance id, `now`, and Policy's
    `schedule.maximum_staleness_hours` — see `_route_headroom`.
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
        allowance_id = allowance_id_for_provider(offering.provider_id)
        route_headroom = _route_headroom(
            headroom_state,
            allowance_id,
            headroom_sources=policy.headroom.sources,
            now=now,
            maximum_staleness_hours=staleness_hours,
            # `offering_id` is this Discovered Offering's own Health Key
            # (CONTEXT.md, "Health Key"), so a Feed provider running
            # `mode: all` can now claim a slot exactly the way a Declared
            # Offering always could — the gap ticket 10 closes.
            sub_allowance_window_id=slot_id_for_health_key(
                policy.headroom.source_members.get(allowance_id), offering_id
            ),
            slot_windows=policy.headroom.source_windows.get(allowance_id),
        )

        rows_by_model.setdefault(model_key, []).append(
            Route(
                alias=alias,
                offering_id=offering_id,
                provider_id=offering.provider_id,
                # Policy's own statement wins over the Feed's pricing kind.
                # The Feed cannot see an account-level plan: it marks Groq
                # `paid` or `unknown` on an account where every call is free.
                cost_basis=(
                    rule.cost_basis
                    if rule is not None and rule.cost_basis is not None
                    else cost_basis_for_pricing_kind(offering.pricing_kind)
                ),
                allowance_id=allowance_id,
                # The Feed states no fair-use clause for any Offering, and
                # absence must not read as a claim either way. `False` is the
                # operator's default too, so both kinds agree.
                fair_use=False,
                tier=_tier_for_allowance(policy, allowance_id),
                scale_note=_scale_note_for_allowance(policy, allowance_id),
                draw_note=policy.draw_notes.get(offering_id),
                headroom=route_headroom,
                demoted_by_headroom=_demoted_by_headroom(
                    route_headroom, demote_at_full=policy.headroom.demote_at_full
                ),
                available=available,
                entitlement=rule.entitlement if rule is not None else "per_model",
                input_usd_per_1m_tokens=offering.pricing.get("input_usd_per_1m_tokens"),
                output_usd_per_1m_tokens=offering.pricing.get("output_usd_per_1m_tokens"),
                rate_source=SOURCE_FEED,
                context_tokens=offering.context_tokens,
                max_output_tokens=offering.max_output_tokens,
                wide_alias=wide_by_alias.get(alias),
                reason=record.reason if record is not None else None,
                refills_at=record.reset_at if record is not None else None,
                last_success_at=record.last_success_at if record is not None else None,
                exhausted=route_is_exhausted(
                    record, now=now, maximum_staleness_hours=staleness_hours
                ),
                excluded=bool(record is not None and record.excluded),
                not_served_at=record.alias_not_served_at if record is not None else None,
            )
        )
        offering_by_model.setdefault(model_key, []).append(offering)

    # Declared Offerings reach the proxy too, and the Feed knows nothing
    # about them. Leaving them out made the strongest models on the proxy
    # invisible to the caller: a direct vendor entry is often the best
    # model offered, and it carries no Feed record at all. Each becomes
    # its own row with no score, so the score ordering puts it last while
    # the answer still admits it exists.
    #
    # A Declared Offering naming a Reference Model is keyed under that
    # Canonical Model instead, so its Route joins the model's row. Both
    # ChatGPT seats reach `openai/gpt-5.6-sol`, so they become two Routes
    # of one row rather than two unscored rows. See ADR 0011.
    declared_by_model: dict[str, DeclaredOffering] = {}
    reference_offerings: dict[str, tuple[Offering, ...]] = {}
    unknown_references: dict[str, str] = {}
    for declared in policy.declared:
        if declared.alias not in report.admitted and declared.alias not in report.excluded:
            continue
        # A Client-Facing Variant contributes no row and no Route of its
        # own: it is the same Offering under a second name, already folded
        # onto its sibling's Route above.
        if declared.alias in declared_variant_of:
            continue
        record = health.get(declared.alias)
        model_key = declared.alias
        mirrors: tuple[Offering, ...] = ()
        if declared.reference_model is not None:
            mirrors = feed.offerings_for_canonical_model(declared.reference_model)
            if mirrors:
                model_key = declared.reference_model
                reference_offerings.setdefault(model_key, mirrors)
            else:
                # The Feed publishes no such Canonical Model. Report it and
                # keep the Alias's own row: a typo here would otherwise
                # remove the score it was written to add, with no symptom.
                unknown_references[declared.alias] = declared.reference_model
        declared_by_model.setdefault(model_key, declared)
        # The Stated Limit of a Declared Offering is whatever the operator
        # wrote in `model_info`, which reaches Generated Config verbatim.
        # Reading it back here keeps the guidance answer and the config
        # stating one figure, not two. A Reference Model never supplies it:
        # the mirror's window belongs to the mirror's endpoint (ADR 0006).
        stated = declared.model_info or {}
        input_rate, output_rate, rate_source = _declared_rates(declared, mirrors)
        declared_allowance_id = allowance_id_for_declared(declared)
        declared_route_headroom = _route_headroom(
            headroom_state,
            declared_allowance_id,
            headroom_sources=policy.headroom.sources,
            now=now,
            maximum_staleness_hours=staleness_hours,
            sub_allowance_window_id=slot_id_for_health_key(
                policy.headroom.source_members.get(declared_allowance_id),
                declared.health_key,
            ),
            slot_windows=policy.headroom.source_windows.get(declared_allowance_id),
        )
        rows_by_model.setdefault(model_key, []).append(
            Route(
                alias=declared.alias,
                offering_id=declared.alias,
                provider_id=DECLARED_PROVIDER,
                cost_basis=_declared_cost_basis(declared),
                allowance_id=declared_allowance_id,
                fair_use=declared.fair_use,
                tier=_tier_for_allowance(policy, declared_allowance_id),
                scale_note=_scale_note_for_allowance(policy, declared_allowance_id),
                draw_note=policy.draw_notes.get(declared.health_key),
                headroom=declared_route_headroom,
                demoted_by_headroom=_demoted_by_headroom(
                    declared_route_headroom, demote_at_full=policy.headroom.demote_at_full
                ),
                available=declared.alias in report.admitted,
                entitlement="declared",
                input_usd_per_1m_tokens=input_rate,
                output_usd_per_1m_tokens=output_rate,
                rate_source=rate_source,
                context_tokens=stated.get("max_input_tokens"),
                max_output_tokens=stated.get("max_output_tokens"),
                wide_alias=wide_by_alias.get(declared.alias),
                reason=record.reason if record is not None else None,
                refills_at=record.reset_at if record is not None else None,
                last_success_at=record.last_success_at if record is not None else None,
                exhausted=route_is_exhausted(
                    record, now=now, maximum_staleness_hours=staleness_hours
                ),
                excluded=bool(record is not None and record.excluded),
                not_served_at=record.alias_not_served_at if record is not None else None,
            )
        )

    if unknown_references:
        warnings = warnings + tuple(
            f"{alias!r} names Reference Model "
            f"{unknown_references[alias]!r}, which this Feed Document does not "
            "publish, so the row carries no score. Check the Canonical Model "
            "id against the Feed."
            for alias in sorted(unknown_references)
        )

    rows: list[GuidanceRow] = []
    for model_key, routes in rows_by_model.items():
        declared = declared_by_model.get(model_key)
        # Three sources of a score, in order. An Offering this proxy serves
        # wins: it is the one the Feed scored AND the one that answers. A
        # Reference Model is second: the same model, scored on another
        # provider's Route. Neither exists for a Declared Offering that
        # names no Reference Model, and its row stays unscored.
        offerings = offering_by_model.get(model_key)
        score_source: str | None = SOURCE_FEED
        if offerings is None:
            offerings = list(reference_offerings.get(model_key, ()))
            score_source = SOURCE_REFERENCE if offerings else None

        if not offerings:
            # Its capabilities are the operator's own statement, or empty
            # when the operator stated none: guess neither.
            capabilities = declared.capabilities if declared is not None else ()
            rows.append(
                GuidanceRow(
                    canonical_model_id=model_key,
                    display_name=model_key,
                    score=None,
                    scores={name: None for name in AXES},
                    routes=tuple(sorted(routes, key=_route_sort_key)),
                    capabilities=capabilities,
                    capabilities_are_operator_stated=bool(capabilities),
                    score_source=None,
                )
            )
            continue
        scores = {
            name: _best_score(offerings, field_name) for name, field_name in AXES.items()
        }
        first = offerings[0]
        # The operator's own capability list wins over the Feed's on a
        # Reference Model row: the operator states what THIS endpoint
        # serves, and a host can serve one model with tool use disabled.
        operator_stated = bool(
            declared is not None
            and declared.capabilities
            and score_source == SOURCE_REFERENCE
        )
        capabilities = (
            declared.capabilities if operator_stated else first.capabilities  # type: ignore[union-attr]
        )
        rows.append(
            GuidanceRow(
                canonical_model_id=model_key,
                display_name=first.raw.get("display_name") or model_key,
                score=_best_score(offerings, axis_field),
                scores=scores,
                routes=tuple(sorted(routes, key=_route_sort_key)),
                capabilities=capabilities,
                capabilities_are_operator_stated=operator_stated,
                score_source=score_source,
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
        routeless=_routeless(policy=policy, report=report, health=health),
    )


# Stages that drop the Feed in bulk rather than one Offering on its own
# merits. Every provider filter rejects hundreds, so listing them buries
# the answer `routeless` exists to give.
_BULK_STAGES = frozenset(
    {
        "feed_visibility",
        "feed_baseline",
        "provider_models",
        "provider_pricing",
    }
)


def _routeless(*, policy, report, health) -> tuple[Routeless, ...]:
    """Offerings the Feed publishes that reached no Route.

    Carries the ones Policy could plausibly have admitted. A bulk Feed or
    provider filter rejects hundreds at a time, and a list holding those
    answers nothing — see `_BULK_STAGES`. `explain` still names the stage
    for any Offering by id, bulk-dropped or not, so nothing is hidden;
    this list is the subset worth reading without asking.
    """
    entries: list[Routeless] = []
    for offering_id, stage in sorted(report.dropped.items()):
        if stage in _BULK_STAGES:
            continue
        record = health.get(offering_id)
        alias = report.aliases.get(offering_id)
        if alias is None:
            try:
                alias = naming.alias_for(policy, offering_id)
            except Exception:  # noqa: BLE001 - an Alias we cannot derive is reported as None
                alias = None
        entries.append(
            Routeless(
                offering_id=offering_id,
                stage=stage,
                alias=alias,
                refills_at=record.reset_at if record is not None else None,
            )
        )
    return tuple(entries)


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


def _declared_cost_basis(declared: DeclaredOffering) -> str:
    """What calling a Declared Offering costs.

    The operator's own statement wins. Without one, the earlier rule
    stands: `passthrough` when the caller supplies the credential,
    `unknown` otherwise. `unknown` is the honest default — an agent is
    told to treat it as spend — but it is wrong about a fixed-rate host,
    which is why Policy can now say so.
    """
    if declared.cost_basis is not None:
        return declared.cost_basis
    return PASSTHROUGH if declared.passthrough_auth else UNKNOWN_BASIS


def _declared_rates(
    declared: DeclaredOffering, mirrors: tuple[Offering, ...]
) -> tuple[float | None, float | None, str | None]:
    """One Declared Offering's token rates, and where they came from.

    The operator's `pricing` block wins: it describes the endpoint the
    proxy actually dials. A Reference Model's rate is second, and it is
    another vendor's price for the same model — useful for ranking the
    relative burn, never a bill.

    Reads the CHEAPEST paid mirror. Mirrors of one model disagree: on
    2026-07-28 `openai/gpt-5.6-terra` read 1.25/7.50 through one provider
    and 2.50/15.00 through another, and the lower pair matched the
    vendor's own published price. A free mirror is skipped entirely
    (`_REFERENCE_RATE_PRICING_KIND`): its 0.00 describes a promotion, not
    the model.
    """
    if declared.pricing is not None:
        return (
            declared.pricing["input_usd_per_1m_tokens"],
            declared.pricing["output_usd_per_1m_tokens"],
            SOURCE_OPERATOR,
        )
    priced = [
        (
            offering.pricing.get("input_usd_per_1m_tokens"),
            offering.pricing.get("output_usd_per_1m_tokens"),
        )
        for offering in mirrors
        if offering.pricing_kind == _REFERENCE_RATE_PRICING_KIND
    ]
    candidates = [
        (in_rate, out_rate)
        for in_rate, out_rate in priced
        if in_rate is not None and out_rate is not None
    ]
    if not candidates:
        return None, None, None
    input_rate, output_rate = min(candidates, key=lambda pair: (pair[1], pair[0]))
    return input_rate, output_rate, SOURCE_REFERENCE


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
    """The Alias for an id the Previous-run record holds.

    A Declared Offering's id IS its Alias: it has no Feed id, so `plan`
    records it under the Alias itself (CONTEXT.md, "Health Key"). Return it
    verbatim.

    Deriving one instead ran the naming rule over a finished Alias and
    produced a name that does not exist. Measured 2026-07-28, when two GDM
    Aliases were added: the Client Advisory named
    `claude-claude-private-host-minimax-m3-`. The Advisory's whole purpose is to
    tell a caller which Alias it may now call, so a mangled name there is
    worse than no name.
    """
    alias = report.aliases.get(offering_id)
    if alias is not None:
        return alias
    if any(d.alias == offering_id for d in policy.declared):
        return offering_id
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
        if row.score_source == SOURCE_REFERENCE:
            score += " (reference)"
        lines.append(f"{index}. {row.canonical_model_id}  {guidance.axis}={score}")
        lines.append(f"   why: {row.why}")
        for position, route in enumerate(row.routes, start=1):
            state = _route_state(route)
            basis = route.cost_basis
            if route.fair_use:
                basis += " (fair use)"
            allowance_label = route.allowance_id or route.provider_id
            if route.tier:
                allowance_label += f" ({route.tier})"
            detail = f"   {position}. {route.alias}  {basis}  {allowance_label}  {state}"
            rate = _rate_note(route)
            if rate:
                detail += f"  {rate}"
            if route.refills_at is not None:
                detail += f"  refills {route.refills_at.isoformat()}"
            detail += _route_headroom_note(route.headroom)
            lines.append(detail)
        lines.append("")

    lines.extend(_advisory_lines(guidance.advisory))
    return "\n".join(lines).rstrip() + "\n"



def _route_state(route: Route) -> str:
    """The State column for one Route: available, excluded, or demoted.

    Reads `route.available` first, then `route.recommendable` — never
    `available` alone. A Route `demoted_by_headroom` stays `available`:
    it is still in the Generated Config, only not recommended. Reporting
    `available` for it made a headroom demotion visible in JSON's
    `recommendable` field and invisible everywhere a human actually
    reads: `render_text` and `render_markdown` both said "available" for
    a Route `guidance` had already stopped recommending.
    """
    if not route.available:
        return route.reason or "excluded"
    if not route.recommendable:
        return f"available, not recommended ({route.not_recommended_because})"
    return "available"


def _route_headroom_note(headroom: RouteHeadroom | None) -> str:
    """One Route's Binding Window figure, or "" when there is none.

    Absence — no Headroom source, no Reading, or every window void — must
    never render as a healthy figure, so this prints nothing at all
    rather than "0%" or a dash a reader could mistake for "plenty left"
    (headroom spec, decision 2).
    """
    if headroom is None:
        return ""
    window = f" ({headroom.window_minutes:.0f} min)" if headroom.window_minutes else ""
    resets = f", resets {headroom.resets_at}" if headroom.resets_at else ""
    percent = format_used_percent(headroom.used_percent)
    return f"  headroom {percent}{window}{resets} — {format_age(headroom.age_seconds)}"


def _route_headroom_cell(headroom: RouteHeadroom | None) -> str:
    """The Markdown table cell for one Route's Binding Window figure."""
    if headroom is None:
        return "—"
    percent = format_used_percent(headroom.used_percent)
    return f"{percent} ({format_age(headroom.age_seconds)})"


def _rate_note(route: Route) -> str:
    """One Route's token rate, in USD per 1M tokens in/out, or "".

    States the source too. A `flat_rate` Route draws on a window rather
    than a wallet, and its rate says how FAST — 15.00 per 1M output tokens
    and 0.28 per 1M drain the same subscription at 54 times the pace. An
    operator reading a ranked list needs that beside the score, or the
    strongest model looks free.
    """
    if route.input_usd_per_1m_tokens is None or route.output_usd_per_1m_tokens is None:
        return ""
    note = (
        f"{route.input_usd_per_1m_tokens:g}/{route.output_usd_per_1m_tokens:g} per 1M"
    )
    if route.rate_is_list_price:
        note += " (list)"
    if route.rate_source and route.rate_source != SOURCE_FEED:
        note += f" [{route.rate_source}]"
    return note


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
        if row.score_source == SOURCE_REFERENCE:
            score += " (reference)"
        lines.append(f"## {index}. `{row.canonical_model_id}` — {guidance.axis} {score}")
        lines.append("")
        # Not `str.capitalize()`: it lowercases every other character, so
        # the term "Feed" came out as "feed". CONTEXT.md's terms keep
        # their case.
        lines.append(row.why[:1].upper() + row.why[1:] + ".")
        lines.append("")
        lines.append(
            "| # | Alias | Cost | Rate per 1M | Allowance | Tier | State | Headroom |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for position, route in enumerate(row.routes, start=1):
            state = _route_state(route)
            if route.refills_at is not None:
                state += f" (refills `{route.refills_at.isoformat()}`)"
            basis = route.cost_basis
            if route.fair_use:
                basis += " (fair use)"
            lines.append(
                f"| {position} | `{route.alias}` | {basis} | "
                f"{_rate_note(route) or 'unstated'} | "
                f"`{route.allowance_id or route.provider_id}` | {route.tier or '—'} | "
                f"{state} | {_route_headroom_cell(route.headroom)} |"
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
