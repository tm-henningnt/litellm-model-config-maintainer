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
from litellm_maintainer.codexbar import CodexbarReading, CodexbarWindow
from litellm_maintainer.feed import Feed
from litellm_maintainer.headroom import (
    format_age,
    format_used_percent,
    BindingWindow,
    HeadroomRecord,
    HeadroomState,
    binding_window,
    reading_age_seconds,
    stated_reset,
    window_is_void,
)
from litellm_maintainer.plan import PlanReport
from litellm_maintainer.policy import (  # noqa: F401 - re-exported, see below
    FLAT_RATE,
    FREE,
    METERED,
    PASSTHROUGH,
    PER_MODEL,
    SHARED_POOL,
    UNKNOWN_BASIS,
    VALID_COST_BASES,
    Policy,
)
from litellm_maintainer.reduce import OfferingHealth

# The five cost bases are defined in `litellm_maintainer.policy`, because
# a Declared Offering states its own basis there. They are re-exported
# here, where the Entitlement view reads them, so every existing
# `from litellm_maintainer.entitlements import FLAT_RATE` keeps working.

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
class HeadroomWindowView:
    """One named window (`primary`, `secondary`, `tertiary` or an extra),
    as `entitlements` reports it.

    `used_percent` is `None` when `void` is `True`: the window's own
    reset has passed, so the stored figure describes a period that
    already ended, and it must not read as a used share (headroom spec,
    decision 8).
    """

    used_percent: float | None
    window_minutes: float | None
    resets_at: str | None
    void: bool
    # What this window is a quota OF, and which admitted Offerings draw on
    # it. Both are `None` / `()` unless Policy names the slot in
    # `headroom.sources.<id>.windows`.
    #
    # A window can be permanently full and govern nothing you can spend.
    # Measured 2026-07-29: the operator's free Gemini plan reads
    # `primary.used_percent: 100.0`, because the plan includes no Pro at
    # all. The two Pro Offerings it describes are Withheld, and three
    # admitted free Routes on the same Allowance read 0%. Reading the 100%
    # as the Allowance's state discards working capacity.
    #
    # `binding: null` already said so, and a reader beside a non-null
    # `primary` missed it. `admitted_members: []` says it on the window
    # itself.
    sub_allowance_id: str | None = None
    # `None` where Policy declares no membership for this window. An empty
    # LIST is a different claim: Policy declares members for this window
    # and none of them is admitted.
    #
    # The two must not share a value. A parent window on a plain-string
    # mapping governs every Offering on the Allowance and declares no
    # members, so `[]` there would read as "governs nothing" and mark
    # every ordinary Allowance as idle capacity.
    admitted_members: tuple[str, ...] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "used_percent": self.used_percent,
            "window_minutes": self.window_minutes,
            "resets_at": self.resets_at,
            "void": self.void,
            "sub_allowance_id": self.sub_allowance_id,
            "admitted_members": (
                list(self.admitted_members) if self.admitted_members is not None else None
            ),
        }


@dataclass(frozen=True)
class HeadroomExtraWindowView:
    """One of codexbar's `extraRateWindows`: a Sub-allowance's own window.

    Published for the record; it never binds here. `entitlements` is
    keyed by ALLOWANCE, and containment runs one way — the Sub-allowance's
    exhaustion says nothing about its parent — so this view must not let
    an extra window raise the Allowance-level figure. `guidance` is the
    one place a Sub-allowance's window binds, and only for the Routes
    that name it (`litellm_maintainer.headroom.route_binding_window`,
    headroom spec, decision 12).
    """

    id: str
    title: str
    window: HeadroomWindowView

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "window": self.window.as_dict()}


@dataclass(frozen=True)
class AllowanceHeadroom:
    """One Allowance's Headroom, as `entitlements` reports it (ticket 04).

    Built by `describe_headroom` from a `HeadroomRecord`, the clock and
    Policy's `schedule.maximum_staleness_hours`. `describe_headroom`
    returns `None` outright — no object at all — when nothing was
    measured: every window is void, or the Reading carries none.

    The flat `used_percent` / `window_minutes` / `resets_at` that a consumer
    reads exist in `as_dict` alone, derived from `binding`. This object keeps
    one field for one fact; only the published shape repeats it.

    `binding` is `None` for the one other reason, and it is not absence:
    Policy names every slot as a Sub-allowance, so nothing caps the
    Allowance as a whole and each Route reads its own window. The object
    is still published, carrying those windows.

    `updated_at` is codexbar's own timestamp; `read_at` is ours.
    `age_seconds` is computed from `updated_at`, never from `read_at`
    (headroom spec, decision 8).
    """

    allowance_id: str
    source: str
    provider: str
    account_email: str | None
    updated_at: str | None
    read_at: str
    age_seconds: float | None
    # `None` means the Allowance was measured and nothing caps it as a
    # whole: every slot is a declared Sub-allowance, so each Route reads
    # its own window instead. It never means unmeasured -- a whole
    # `headroom` of `null` means that.
    binding: BindingWindow | None
    primary: HeadroomWindowView | None
    secondary: HeadroomWindowView | None
    tertiary: HeadroomWindowView | None
    extra_windows: tuple[HeadroomExtraWindowView, ...] = ()
    error: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "allowance_id": self.allowance_id,
            "source": self.source,
            "provider": self.provider,
            "account_email": self.account_email,
            "updated_at": self.updated_at,
            "read_at": self.read_at,
            "age_seconds": self.age_seconds,
            # The binding figure appears TWICE, and deliberately.
            #
            # A `guidance` route publishes its binding figure flat, as
            # `headroom.used_percent`. This object publishes the full
            # window set, so the same fact sat one level down under
            # `binding` and nowhere else. One field name then carried two
            # shapes, and `.headroom.used_percent` answered on a route and
            # returned `null` on every entitlement, mapped or not. A
            # consumer could not tell that `null` from an unmapped
            # Allowance. Reported 2026-07-29 by an agent whose runbook
            # shipped exactly that expression.
            #
            # So the three binding fields are repeated at the top, where a
            # route carries them. `binding` stays, because it names what
            # they ARE. Both read from `self.binding`, so they cannot
            # disagree.
            "used_percent": self.binding.used_percent if self.binding else None,
            "window_minutes": self.binding.window_minutes if self.binding else None,
            "resets_at": self.binding.resets_at if self.binding else None,
            "binding": (
                {
                    "used_percent": self.binding.used_percent,
                    "window_minutes": self.binding.window_minutes,
                    "resets_at": self.binding.resets_at,
                }
                if self.binding is not None
                else None
            ),
            "primary": self.primary.as_dict() if self.primary else None,
            "secondary": self.secondary.as_dict() if self.secondary else None,
            "tertiary": self.tertiary.as_dict() if self.tertiary else None,
            "extra_windows": [w.as_dict() for w in self.extra_windows],
            "error": self.error,
        }


def _window_view(
    window: CodexbarWindow | None,
    *,
    reading_updated_at: str | None,
    now: datetime,
    maximum_staleness_hours: float,
    sub_allowance_id: str | None = None,
    admitted_members: tuple[str, ...] | None = None,
) -> HeadroomWindowView | None:
    """One window as `entitlements` reports it, or `None` when absent.

    `resets_at` comes from `stated_reset`, never from the raw window. A
    reset at or before the Reading's own timestamp states no reset, and
    publishing it raw leaks a placeholder to callers. Measured 2026-07-29:
    Gemini's Pro slot carries `resetsAt: "1970-01-01T00:00:00Z"` beside
    `resetDescription: "Resets soon"`, and `entitlements` published the
    epoch verbatim.
    """
    if window is None:
        return None
    void = window_is_void(
        window,
        reading_updated_at=reading_updated_at,
        now=now,
        maximum_staleness_hours=maximum_staleness_hours,
    )
    return HeadroomWindowView(
        used_percent=None if void else window.used_percent,
        window_minutes=window.window_minutes,
        resets_at=stated_reset(window, reading_updated_at=reading_updated_at),
        void=void,
        sub_allowance_id=sub_allowance_id,
        admitted_members=admitted_members,
    )


def _codexbar_error_view(reading: CodexbarReading) -> dict[str, Any] | None:
    """The per-provider error carried on this Reading, from the last refresh.

    A stored Reading with `error` set never comes from `refresh_headroom`
    today — a provider that errors keeps its previous Reading (see
    `litellm_maintainer.headroom.refresh_headroom`) — so this reads
    `None` for every mapped Allowance now. It is read from the Reading
    itself, and not hardcoded to `None`, so a future refresh that starts
    storing an error alongside a stale Reading is visible here without
    another change.
    """
    error = reading.error
    if error is None:
        return None
    return {"kind": error.kind, "code": error.code, "message": error.message}


def describe_headroom(
    record: HeadroomRecord,
    *,
    now: datetime,
    maximum_staleness_hours: float,
    named_slots: frozenset[str] = frozenset(),
    slot_windows: dict[str, str] | None = None,
    admitted_members_by_slot: dict[str, tuple[str, ...]] | None = None,
) -> AllowanceHeadroom | None:
    """Build the Headroom `entitlements` publishes for one Allowance.

    `None` when nothing was measured: the Reading carries no window at all
    (measured 2026-07-28: `openrouter` and `deepseek` both answer this
    way), or every `primary`/`secondary`/`tertiary` window is void.
    Absence must never read as a healthy 0% (headroom spec, decisions 2
    and 8).

    A `binding` of `None` inside a PRESENT object means something else:
    the Allowance was measured, and `named_slots` (Policy's
    `headroom.sources.<id>.windows` keys) covers every slot the Reading
    states, so nothing caps the Allowance as a whole. That is ticket 09's
    case, and publishing no whole-Allowance figure for it is correct — a
    figure borrowed from whichever slot reads worst would be a lie.

    Returning `None` for that case as well was wrong. Measured 2026-07-29
    on the operator's Gemini free plan, whose three slots hold Pro, Flash
    and Flash Lite: `entitlements` published nothing at all for the
    Allowance, so the one command documented for reading Allowance state
    reported a measured Allowance as unmeasured. The per-model figures
    reached `guidance` routes and nowhere else. The object now carries
    them, with `binding: null` stating that no single figure caps the
    whole.
    """
    reading = record.reading
    binding = binding_window(
        reading,
        now=now,
        maximum_staleness_hours=maximum_staleness_hours,
        named_slots=named_slots,
    )
    if binding is None:
        # Tell "nothing was measured" apart from "every slot is a declared
        # Sub-allowance". Recompute with no slot named: a figure here means
        # the Reading does hold live windows, and only Policy's own slot
        # mapping removed them from the whole-Allowance computation.
        if named_slots and binding_window(
            reading,
            now=now,
            maximum_staleness_hours=maximum_staleness_hours,
        ) is None:
            return None
        if not named_slots:
            return None

    slots = slot_windows or {}
    members = admitted_members_by_slot or {}

    def view(
        window: CodexbarWindow | None, *, slot: str | None = None
    ) -> HeadroomWindowView | None:
        sub_id = slots.get(slot) if slot is not None else None
        return _window_view(
            window,
            reading_updated_at=reading.updated_at,
            now=now,
            maximum_staleness_hours=maximum_staleness_hours,
            sub_allowance_id=sub_id,
            admitted_members=members.get(sub_id) if sub_id is not None else None,
        )

    return AllowanceHeadroom(
        allowance_id=record.allowance_id,
        source=record.source,
        provider=reading.provider,
        account_email=reading.identity.account_email,
        updated_at=reading.updated_at,
        read_at=record.read_at,
        age_seconds=reading_age_seconds(reading, now=now),
        binding=binding,
        primary=view(reading.primary, slot="primary"),
        secondary=view(reading.secondary, slot="secondary"),
        tertiary=view(reading.tertiary, slot="tertiary"),
        extra_windows=tuple(
            HeadroomExtraWindowView(
                id=extra.id,
                title=extra.title,
                window=_window_view(
                    extra.window,
                    reading_updated_at=reading.updated_at,
                    now=now,
                    maximum_staleness_hours=maximum_staleness_hours,
                    sub_allowance_id=extra.id,
                    admitted_members=members.get(extra.id),
                ),
            )
            for extra in reading.extra_windows
        ),
        error=_codexbar_error_view(reading),
    )


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
    # The Allowance this entry describes: the same key `guidance` puts on
    # every Route (CONTEXT.md, "Allowance"). It is what lets a caller join
    # the two answers — pick a Route in `guidance`, then read that
    # allowance's ceiling here.
    #
    # A Feed provider's is `provider:<id>`. A Declared group's names the
    # credential or the pool, so two subscription seats behind one provider
    # prefix are two entries and not one.
    allowance_id: str | None = None
    # Whether this Allowance tolerates load badly. True when any Offering in
    # it declares `fair_use`. See ADR 0012.
    fair_use: bool = False
    # The subscription level this Allowance bills under, as the operator
    # states it in `policy.allowances.<id>.tier` (CONTEXT.md, "Tier"). `None`
    # when Policy names no entry for this Allowance. Published verbatim: a
    # label, never parsed, ranked or derived from. A Headroom states a SHARE
    # of this Tier's own ceiling, so it means nothing without the Tier
    # beside it.
    tier: str | None = None
    # How big the Allowance is, as the operator states it, where the
    # vendor states a size but sells no Tier. Prose, verbatim, ranked by
    # nothing. See `policy.AllowanceInfo.scale_note`.
    scale_note: str | None = None
    # How much of this Allowance a source has measured, or `None`. `None`
    # covers three cases alike: Policy declares no `headroom_source`, no
    # Reading has been captured yet, or every window in the Reading is void.
    # Absence must never read as free capacity (headroom spec, decision 2).
    headroom: AllowanceHeadroom | None = None

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
            "allowance_id": self.allowance_id,
            "fair_use": self.fair_use,
            "tier": self.tier,
            "scale_note": self.scale_note,
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
            "headroom": self.headroom.as_dict() if self.headroom else None,
        }


@dataclass(frozen=True)
class EntitlementView:
    """Every Entitlement, plus what the whole picture was derived from.

    Declared Offerings used to sit outside `entitlements` entirely, on the
    reasoning that an Entitlement is a relationship with one PROVIDER and a
    Declared Offering has no Feed provider. That reasoning held and the
    result was still wrong: a whole private host reported one aggregate
    count, with no `state`, no `reason` and no `earliest_refill_at`, so a
    caller could not tell a drained seat from a healthy one and the only way
    to find a ceiling was to hit it.

    An Entitlement is now keyed by ALLOWANCE, not by provider, and a
    Declared Offering has one of those. So each Declared Allowance gets a
    full entry, ordered after the Feed providers.

    WARNING: `declared` remains, reporting the same Offerings in aggregate.
    Two readings of one fact, kept so an existing consumer does not break.
    Never sum `answering` across `entitlements` AND `declared`; that
    double-counts every Declared Offering.

    The two agree exactly, and did not before. `declared` counted each
    Client-Facing Variant as an Offering of its own, so the operator's 20
    Declared Offerings read 24. A variant shares its sibling's Health Key
    (ADR 0007), so it was never an Offering; the count is corrected here.
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
#
# Raised to "2" on 2026-07-28: `entitlements` gained one entry per Declared
# Allowance. That is not an added field — iterating the list yields
# Offerings it never yielded before, and `declared` still reports the same
# ones, so a consumer summing both now double-counts. See ADR 0012.
SCHEMA_VERSION = "2"


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
    Their credentials — `EXAMPLE_CHATGPT_SEAT1_WORKER_KEY` and
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


# The prefix litellm uses to read a credential out of the process
# environment. Stripped from an Allowance id, so the id names a variable
# and never looks like a value.
_ENVIRONMENT_PREFIX = "os.environ/"

# The Allowance id namespaces. A caller uses the whole string as a key, so
# the prefix keeps two kinds of allowance from colliding: a provider named
# `gdm` and a credential named `gdm` are not the same allowance.
ALLOWANCE_PROVIDER = "provider"
ALLOWANCE_POOL = "pool"
ALLOWANCE_CREDENTIAL = "credential"
ALLOWANCE_ALIAS = "alias"

# The `provider_id` a Declared Offering reports: it has no Feed provider,
# because the Feed does not publish it at all. Defined here rather than in
# `guidance`, which imports this module, so one string serves both answers.
# `guidance.DECLARED_PROVIDER` is an alias of it.
DECLARED_PROVIDER_ID = "declared"


def allowance_id_for_provider(provider_id: str) -> str:
    """The Allowance id of a Discovered Offering's provider.

    The Feed states one `credential_hint` per provider, so a provider IS an
    allowance for a Discovered Offering. Same shape as `pool_siblings`
    builds for its own grouping.
    """
    return f"{ALLOWANCE_PROVIDER}:{provider_id}"


def allowance_id_for_declared(declared) -> str:
    """The Allowance id of a Declared Offering. Never `None`.

    An **Allowance** is what gets billed (CONTEXT.md). This is the key a
    caller uses to refuse or cap one allowance without touching another, so
    three properties matter more than beauty:

    1. **It is derived from the credential, never from `group`.** `group` is
       a heading the Generated Config prints; `policy.py` says it "names
       nothing the code acts on". A key derived from a display sentence
       changes when somebody rewrites the sentence, and nothing would show
       that the client's cap had silently moved. ADR 0009 and ADR 0012.
    2. **It carries no credential VALUE.** `os.environ/EXAMPLE_PRIVATE_HOST_API_KEY` reduces
       to `credential:EXAMPLE_PRIVATE_HOST_API_KEY`: a variable name, which is not a secret,
       and which cannot be mistaken for one.
    3. **It is never `None`.** `declared_pool_id` returns `None` for an
       Offering with neither a named pool nor a credential, because it
       answers "which pool propagates a Probe" and such an Offering
       propagates to nobody. This answers "who is billed", and the answer
       there is "itself" — so it falls back to the Alias. A `null` would
       make every unpooled Offering read as one shared allowance, which is
       the opposite of the truth.
    """
    pool = declared_pool_id(declared)
    if pool is None:
        return f"{ALLOWANCE_ALIAS}:{declared.alias}"
    kind, _, value = pool.partition(":")
    if kind == "named":
        return f"{ALLOWANCE_POOL}:{value}"
    return f"{ALLOWANCE_CREDENTIAL}:{value.removeprefix(_ENVIRONMENT_PREFIX)}"


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
    headroom_state: HeadroomState | None = None,
) -> EntitlementView:
    """Derive the Entitlement view. Pure: reads values, returns a value.

    `report` comes from `plan` over the same Feed, Policy and Health
    State, so this view and `status` can never disagree about what is
    offered.

    `headroom_state` is `None` by default, so a caller that has not read
    Headroom State yet (or a test that predates ticket 04) gets exactly
    the view it always got, every `headroom` field `None`. Each
    Allowance's Headroom is built once here, from that Allowance's own
    `HeadroomRecord`, `now` and Policy's `schedule.maximum_staleness_hours`
    — see `describe_headroom`.

    A stored record publishes only when Policy STILL declares its
    Allowance in `headroom.sources`, AND the record's own `source` still
    equals what Policy currently states for that id. Neither check is
    redundant with `refresh_headroom` pruning the file: an Allowance
    removed from Policy between one refresh and the next still sits on
    disk until the next refresh runs, and a REMAPPED source (the same
    Allowance id, pointed at a different codexbar identity) is never
    pruned at all, because `refresh_headroom` only drops ids Policy no
    longer names. Reading a Reading whose Allowance mapping already
    changed is the fault this guards: Gemini was mapped 2026-07-28 and
    unmapped 2026-07-29 precisely because its figure's meaning is
    unknown, and a stale record must not keep publishing it (headroom
    spec, decision 2).
    """
    maximum_staleness_hours = policy.schedule.maximum_staleness_hours
    headroom_by_allowance: dict[str, AllowanceHeadroom] = {}
    if headroom_state is not None:
        # Which admitted Health Keys reach each Allowance. A window states
        # the ones that draw on it, so a reader can see a window that is
        # full and governs nothing spendable -- see `HeadroomWindowView`.
        #
        # Admitted means what Policy admits, so `report.excluded` counts
        # too: an Excluded Offering is admitted and temporarily not
        # answering. Reading `report.admitted` alone would empty a window's
        # list the moment its one model failed a Probe, which reads as
        # "this window governs nothing" when Policy still admits it.
        # Withheld and Candidate Offerings are correctly absent.
        admitted_keys_by_allowance: dict[str, set[str]] = {}
        for offering_id in (*report.admitted, *report.excluded):
            allowance = allowance_id_for_provider(offering_id.partition(":")[0])
            admitted_keys_by_allowance.setdefault(allowance, set()).add(offering_id)
        for declared in policy.declared:
            if declared.variant_of is not None:
                # Shares its sibling's Health Key (ADR 0007).
                continue
            allowance = allowance_id_for_declared(declared)
            admitted_keys_by_allowance.setdefault(allowance, set()).add(declared.health_key)

        for allowance_id, record in headroom_state.records.items():
            declared_source = policy.headroom.sources.get(allowance_id)
            if declared_source is None or record.source != declared_source:
                continue
            slot_windows = policy.headroom.source_windows.get(allowance_id, {})
            admitted_here = admitted_keys_by_allowance.get(allowance_id, set())
            admitted_members_by_slot = {
                sub_id: tuple(sorted(k for k in health_keys if k in admitted_here))
                for sub_id, health_keys in policy.headroom.source_members.get(
                    allowance_id, {}
                ).items()
            }
            described = describe_headroom(
                record,
                now=now,
                maximum_staleness_hours=maximum_staleness_hours,
                named_slots=frozenset(slot_windows),
                slot_windows=dict(slot_windows),
                admitted_members_by_slot=admitted_members_by_slot,
            )
            if described is not None:
                headroom_by_allowance[allowance_id] = described

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
        if rule.cost_basis is not None:
            # Policy states what this provider costs THIS account, so the
            # Feed's pricing kind is not consulted at all. It cannot see an
            # account plan, and a provider that is free here reads `paid`
            # there. One basis, because Policy states one.
            bases.add(rule.cost_basis)
        else:
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

        allowance_id = allowance_id_for_provider(provider_id)
        entitlements.append(
            Entitlement(
                provider_id=provider_id,
                allowance_id=allowance_id,
                kind=rule.entitlement or PER_MODEL,
                cost_bases=tuple(sorted(bases)),
                answering=len(admitted),
                unavailable_offerings=unavailable,
                withheld=withheld_counts.get(provider_id, 0),
                candidates=candidate_counts.get(provider_id, 0),
                tier=_tier_for_allowance(policy, allowance_id),
                scale_note=_scale_note_for_allowance(policy, allowance_id),
                headroom=headroom_by_allowance.get(allowance_id),
            )
        )

    # A Declared Offering's Health Key is its Alias, because it has no
    # Feed id (CONTEXT.md, "Health Key"). Read health by Alias here.
    declared_answering = 0
    declared_unavailable: list[UnavailableOffering] = []
    for declared in policy.declared:
        # A Client-Facing Variant is the same Offering under a second name
        # and shares its sibling's Health Key (ADR 0007), so it is not an
        # Offering of its own. Counting it reported the operator's 20
        # Declared Offerings as 24, and the per-Allowance entries below —
        # which never counted it — disagreed with this total by exactly the
        # 4 variants. Corrected on 2026-07-28, with SCHEMA_VERSION.
        if declared.variant_of is not None:
            continue
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

    # One entry per Declared Allowance, AFTER the sorted Feed providers.
    # Order matters to a consumer indexing the list, so the Feed providers
    # keep the positions they have always had.
    entitlements.extend(
        _declared_entitlements(
            policy=policy,
            report=report,
            health=health,
            headroom_by_allowance=headroom_by_allowance,
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


def _declared_entitlements(
    *,
    policy: Policy,
    report: PlanReport,
    health: dict[str, OfferingHealth],
    headroom_by_allowance: dict[str, AllowanceHeadroom],
) -> list[Entitlement]:
    """One Entitlement per Declared Allowance, ordered by Allowance id.

    Grouped by `allowance_id_for_declared`, so the credential decides — two
    ChatGPT seats behind one `openai/` prefix are two allowances, and one
    running dry says nothing about the other (ADR 0009).

    A Client-Facing Variant contributes nothing. It is the same Offering
    under a second name and shares its sibling's `health_key`, so counting
    it would report a host as twice its real size.

    `withheld` and `candidates` are always 0. A Declared Offering is never a
    Candidate — declaring one is already the decision (CONTEXT.md) — and
    Policy's `withheld` map keys on Feed Offering ids.
    """
    from litellm_maintainer.guidance import _declared_cost_basis

    members: dict[str, list[Any]] = {}
    for declared in policy.declared:
        if declared.variant_of is not None:
            continue
        if declared.alias not in report.admitted and declared.alias not in report.excluded:
            continue
        members.setdefault(allowance_id_for_declared(declared), []).append(declared)

    entitlements: list[Entitlement] = []
    for allowance in sorted(members):
        group = members[allowance]
        unavailable = tuple(
            UnavailableOffering(
                offering_id=declared.alias,
                alias=declared.alias,
                reason=_health_field(health, declared.alias, "reason"),
                bucket=_health_field(health, declared.alias, "bucket"),
                refills_at=_health_field(health, declared.alias, "reset_at"),
            )
            for declared in sorted(group, key=lambda d: d.alias)
            if declared.alias in report.excluded
        )
        # Every member states one Entitlement kind. They should agree, and
        # `shared_pool` wins if they do not: one member sharing the pool
        # means the pool is shared, whatever a sibling forgot to say.
        kind = (
            SHARED_POOL
            if any((d.entitlement or PER_MODEL) == SHARED_POOL for d in group)
            else PER_MODEL
        )
        entitlements.append(
            Entitlement(
                provider_id=DECLARED_PROVIDER_ID,
                allowance_id=allowance,
                kind=kind,
                cost_bases=tuple(sorted({_declared_cost_basis(d) for d in group})),
                answering=sum(1 for d in group if d.alias in report.admitted),
                unavailable_offerings=unavailable,
                fair_use=any(d.fair_use for d in group),
                tier=_tier_for_allowance(policy, allowance),
                scale_note=_scale_note_for_allowance(policy, allowance),
                headroom=headroom_by_allowance.get(allowance),
            )
        )
    return entitlements


def _tier_for_allowance(policy: Policy, allowance_id: str) -> str | None:
    """The Tier the operator states for `allowance_id`, or `None`.

    Reads `policy.allowances`, the block keyed on the same Allowance id
    `headroom.sources` and this module's own `allowance_id` use. `None`
    when the operator names no entry, or names one that states no `tier`
    — both read as "the operator has not said". See CONTEXT.md, "Tier".
    """
    entry = policy.allowances.get(allowance_id)
    return entry.tier if entry is not None else None


def _scale_note_for_allowance(policy: Policy, allowance_id: str) -> str | None:
    """The scale note the operator states for `allowance_id`, or `None`.

    Reads the same block `_tier_for_allowance` does. Prose, published
    verbatim; see `policy.AllowanceInfo.scale_note` for why it is not a
    number. It answers the question `tier` cannot for a vendor that sells
    one fixed price with one quota and no levels.
    """
    entry = policy.allowances.get(allowance_id)
    return entry.scale_note if entry is not None else None


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



def _headroom_line(headroom: AllowanceHeadroom | None) -> str | None:
    """One line naming the Binding Window's figure, or `None` when absent.

    Absence — Policy names no source, no Reading exists yet, or every
    window in it is void — prints nothing here. It must never print a
    reassuring 0%.

    A present Headroom with no `binding` prints a different line: the
    Allowance WAS measured, and Policy names every slot a Sub-allowance,
    so no single window caps it. Read the per-Route figures in `guidance`.
    """
    if headroom is None:
        return None
    binding = headroom.binding
    if binding is None:
        # Name any window that is full and governs nothing admitted. A
        # reader who sees only "100%" beside the Allowance discards working
        # capacity -- the measured Gemini Pro case, see
        # `HeadroomWindowView`.
        idle = [
            w.sub_allowance_id or "?"
            for w in (headroom.primary, headroom.secondary, headroom.tertiary)
            if w is not None and w.admitted_members == () and w.used_percent
        ]
        note = (
            f" ({', '.join(idle)} reads full and governs no admitted Offering)" if idle else ""
        )
        return (
            "  headroom: no single window caps this Allowance; every slot is a "
            f"Sub-allowance — read each Route in guidance{note} — "
            f"{format_age(headroom.age_seconds)} ({headroom.source})"
        )
    window = f" ({binding.window_minutes:.0f} min)" if binding.window_minutes else ""
    resets = f", resets {binding.resets_at}" if binding.resets_at else ""
    percent = format_used_percent(binding.used_percent)
    return (
        f"  headroom: {percent} of its Binding Window{window}"
        f"{resets} — {format_age(headroom.age_seconds)} ({headroom.source})"
    )


def _headroom_cell(headroom: AllowanceHeadroom | None) -> str:
    """The Markdown table cell for one Allowance's Headroom."""
    if headroom is None:
        return "—"
    binding = headroom.binding
    if binding is None:
        # Measured, and nothing caps the whole Allowance. Never "—", which
        # this table already uses for unmeasured.
        return f"per-Route ({format_age(headroom.age_seconds)})"
    percent = format_used_percent(binding.used_percent)
    return f"{percent} ({format_age(headroom.age_seconds)})"


def _entitlement_label(entitlement: Entitlement) -> str:
    """What to call this entry in a rendering.

    A Feed provider is its own name. Every Declared entry reports
    `provider_id: "declared"`, so four allowances would print one word four
    times; each shows its Allowance id instead, which is what distinguishes
    one subscription seat from the next.
    """
    if entitlement.provider_id != DECLARED_PROVIDER_ID:
        return entitlement.provider_id
    return entitlement.allowance_id or DECLARED_PROVIDER_ID


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
        if entitlement.fair_use:
            basis += " (fair use)"
        label = _entitlement_label(entitlement)
        if entitlement.tier:
            label += f" ({entitlement.tier})"
        lines.append(f"{label}  {entitlement.state}  {entitlement.kind}  {basis}")
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
        headroom_line = _headroom_line(entitlement.headroom)
        if headroom_line is not None:
            lines.append(headroom_line)
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
            f"declared (total)  {view.declared_answering} of "
            f"{view.declared_in_scope} answering"
        )
        lines.append(
            "  Every Offering you declared, summed. Each Allowance is also "
            "listed above; do not add the two."
        )
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

    lines.append(
        "| Allowance | Tier | State | Pool | Cost | Answering | Earliest refill | Headroom |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for e in view.entitlements:
        basis = e.cost_basis or "/".join(e.cost_bases) or "unknown"
        if e.fair_use:
            basis += " (fair use)"
        refill = e.earliest_refill_at
        lines.append(
            f"| `{_entitlement_label(e)}` | {e.tier or '—'} | {e.state} | {e.kind} | {basis} | "
            f"{e.answering}/{e.in_scope} | {refill.isoformat() if refill else '—'} | "
            f"{_headroom_cell(e.headroom)} |"
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
