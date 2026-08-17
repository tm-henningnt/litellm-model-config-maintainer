"""Fold Probe outcomes and Journal observations into the next Health State.

`reduce` is pure. It takes the prior Health State, this run's Probe
outcomes, the new Observation Journal entries, and the set of Offerings
Policy currently admits. It returns the next Health State. It performs
no input or output: no network, no filesystem, no clock read, no
environment read. The current time is always the `now` parameter.

See CONTEXT.md for Health State, Excluded, Withheld and Passthrough
Auth. See `.scratch/maintainer-v1/spec.md`, sections "Failure
classification", "Recovery does not need a probe" and "Seam 2: reduce".
Health State is written only by the maintainer (ADR 0001), through
`litellm_maintainer.health.write_health`, the only function that ever
calls `reduce` and persists its result. Nothing else may write that
file.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime

from litellm_maintainer.classify import (
    ANSWERED,
    INCONCLUSIVE,
    REASON_ALIAS_NOT_SERVED,
    REASON_AUTHENTICATION_FAILED,
    REASON_QUOTA_EXHAUSTED,
    REASON_UNRECOGNIZED_FAILURE,
    Bucket,
    Outcome,
    Reason,
)

# A per-Offering key is the Discovered Offering id (`<provider>:<id>`),
# or, for a Declared Offering, its Alias. Whichever id Policy uses to
# admit the Offering is the id `reduce` reads back from `outcomes`,
# `observations` and `admitted`. Use the same id everywhere.
OfferingKey = str

# A Passthrough Auth Offering's credentials come from the calling
# client, not the proxy (CONTEXT.md, "Passthrough Auth"). A quota
# exhaustion or an authentication failure belongs to that one caller,
# not to the Offering, so neither Excludes it, whatever the bucket a
# quota exhaustion carries (`self_healing` with a reset time, or
# `needs_operator` with a zero limit). Every other reason still
# Excludes it, including a gateway error, a timeout and a rate limit.
_PASSTHROUGH_EXEMPT_REASONS: frozenset[Reason] = frozenset(
    {REASON_QUOTA_EXHAUSTED, REASON_AUTHENTICATION_FAILED}
)


def journal_outcome(outcome: Outcome) -> Outcome:
    """Return the Outcome a JOURNAL observation may act on.

    `classify` fails closed: a failure it does not recognise returns
    `needs_operator`, which Excludes the Offering until a human clears
    it. That is right for a Probe, which sends a known-good synthetic
    request, so every failure it sees is the Offering's fault.

    It is wrong for real traffic, where the CLIENT causes failures too.
    An over-long prompt returns HTTP 400 with wording no rule matches
    (`classify.py`, `_OPERATOR_STATUSES` holds 401, 402 and 403 only).
    Fail-closed would let one oversized request Exclude a healthy
    Offering, and hand every client the power to remove any model.

    So the Journal path fails OPEN. An `unrecognized_failure` keeps its
    reason and takes the `inconclusive` bucket, which `reduce` applies
    to nothing. The observation is still recorded and reported,
    carrying the message that says which rule is missing (see
    `Observation.message`). Every failure `classify` DOES recognise
    passes through unchanged.

    Change the BUCKET, never the reason. `bucket` names the
    consequence and `reason` names the condition (`OfferingHealth`).
    The condition really is a failure this project does not recognise,
    and rewriting the reason to `unmeasured` would call a real failure
    a non-event -- the exact conflation `classify.py` forbids beside
    those two names.

    The cost, stated plainly: a genuine Offering failure whose wording
    `classify` does not know goes unnoticed by the Journal path until a
    Probe finds it. See ADR 0008.
    """
    if outcome.reason != REASON_UNRECOGNIZED_FAILURE:
        return outcome
    return Outcome(bucket=INCONCLUSIVE, reset_at=None, reason=outcome.reason)


@dataclass(frozen=True)
class OfferingHealth:
    """What we know now about one Offering.

    `excluded` states whether the Generator must leave this Offering
    out. `reason` is the classify `Reason` that produced this record
    (see `classify.REASONS`), or `None` when the Offering has never
    failed. It names the condition: a quota exhaustion, a gateway
    error, and so on. `bucket` is the classify bucket `reason`
    produced, or `None` when the Offering has never failed. `bucket`
    names the consequence; `reason` names the condition. `reset_at` is
    the time a recorded failure states it clears by itself, or `None`.

    Read `last_success_at` for staleness, never `last_checked_at` (this
    record has no such field, on purpose). `last_success_at` advances
    only on an `answered` outcome. `last_attempt_at` advances on every
    Probe or Journal entry that reaches this record, success or
    failure, but not on an Inconclusive one, which touches nothing.

    `failure_count` counts consecutive failures. An `answered` outcome
    resets it to zero.

    `probe_due` asks the Prober to measure this Offering on its next
    sweep, whatever its freshness would otherwise say. It is NOT a
    health verdict: the Offering stays offered and keeps serving. A
    sibling on a `shared_pool` Entitlement sets it when the pool
    reports a quota exhaustion (see `_pool_siblings_to_mark`). Any
    event applied to this record clears it, because the record was
    then measured.

    `inconclusive_count` counts observations that reached this record
    and changed nothing. It is the ONE field an Inconclusive outcome is
    allowed to move, and that is a deliberate exception to a rule this
    module otherwise keeps absolutely: an Inconclusive attempt measured
    nothing, so it must not touch health.

    The exception exists because a wrong classification is silent. ADR
    0008 makes an UNRECOGNISED failure visible by storing its message,
    but a confidently misread one carries no message and looks like
    ordinary operation. An exhausted OpenCode Go plan wrote 90 entries
    that all read as `rate_limited`, changed nothing, and were noticed
    only because the operator said so. A rising count on one key is the
    signal that `classify` is reading something wrong. An `answered`
    outcome resets it, because the Offering demonstrably works.

    `alias_not_served_at` is when the PROXY last refused this Offering's
    Alias as one it does not serve. It says nothing about the Offering
    and everything about our own pipeline: the Generated Config on disk
    was older than what this tool believed. It never Excludes.

    Stored as a time, not a flag, because nothing observes the opposite.
    The proxy states "not served" on a failed call and states nothing at
    all on a successful one, and the Prober bypasses the proxy entirely,
    so no measurement can ever set a flag back to False. A reader judges
    the timestamp's age the way it judges a Reading's.
    """

    excluded: bool = False
    reason: Reason | None = None
    bucket: Bucket | None = None
    reset_at: datetime | None = None
    last_success_at: datetime | None = None
    last_attempt_at: datetime | None = None
    failure_count: int = 0
    probe_due: bool = False
    inconclusive_count: int = 0
    alias_not_served_at: datetime | None = None


_EMPTY_RECORD = OfferingHealth()


@dataclass(frozen=True)
class HealthState:
    """The machine's record of every Offering it has ever heard from.

    `offerings` maps an Offering key to its `OfferingHealth`. An
    Offering with no entry has never been probed or observed.

    `skipped_records` counts a record `read_health` could not parse.
    `read_health` (`litellm_maintainer.health`) keeps every record that
    does parse and only skips the bad one; it never discards the whole
    file for one bad record. A caller reports a non-zero count to the
    operator. `reduce`'s own result always carries `0` here: this field
    is `read_health`'s to set, never `reduce`'s.
    """

    offerings: dict[OfferingKey, OfferingHealth] = field(default_factory=dict)
    skipped_records: int = 0


@dataclass(frozen=True)
class Observation:
    """One entry from the Observation Journal.

    The proxy appends one `Observation` per failure it serves to a real
    caller. `offering_id` is the Offering key (see `OfferingKey`).
    `observed_at` is when the proxy served the request. `outcome` is
    the `classify.Outcome` the proxy's failure callback computed.

    `message` carries the provider's own text, redacted and truncated,
    and ONLY when `outcome.reason` is `unrecognized_failure`. Every
    classified failure leaves it `None`. The text is what tells the
    operator which `classify` rule is missing; a failure `classify`
    already understands needs no text, and provider text is the one
    place a credential can leak (see `litellm_maintainer.redact`). So
    the exposure is bounded to the cases that teach us something, and
    it shrinks as rules are added.

    `providers/journal_failure_callback.py` writes these. This is where
    it finds the type.
    """

    offering_id: OfferingKey
    observed_at: datetime
    outcome: Outcome
    message: str | None = None


def _pool_siblings_to_mark(
    *,
    key: OfferingKey,
    outcome: Outcome,
    pool_siblings: dict[OfferingKey, frozenset[OfferingKey]],
    passthrough_auth: set[OfferingKey],
    sub_allowances: set[OfferingKey] = frozenset(),
) -> frozenset[OfferingKey]:
    """Return the siblings a quota exhaustion makes due for a Probe.

    ADR 0004 forbids propagating a CONCLUSION across an Entitlement. A
    shared pool that refuses one Offering does not mean the rest are
    dry: Gemini refused every Pro-tier model while every Flash, Gemma
    and Lite model kept answering, and ClinePass paid credits ran out
    while three free Offerings from the same provider id still served.
    Excluding a sibling on that guess would have removed the only
    routes still working.

    So propagate ATTENTION instead. A sibling is marked `probe_due`,
    never Excluded. It stays in the Generated Config and keeps serving.
    The next sweep measures it and learns the truth. This is ADR 0004's
    own stated price -- "learning a sibling's state costs a Probe" --
    paid promptly rather than whenever freshness happens to expire.

    Propagate from a quota exhaustion only. Every other reason
    describes one Offering, not a pool.

    Never propagate from a Passthrough Auth Offering. Its quota belongs
    to the calling client, not to our Entitlement (CONTEXT.md,
    "Passthrough Auth"), so it says nothing about the pool. Never mark
    a Passthrough Auth sibling either: the Prober cannot probe one, so
    the mark would never clear.
    """
    if outcome.reason != REASON_QUOTA_EXHAUSTED:
        return frozenset()
    if key in passthrough_auth:
        return frozenset()
    if key in sub_allowances:
        # One-way containment. A sub-allowance is capped inside the
        # pool, so its own exhaustion says nothing about the pool: at
        # most half the operator's Claude weekly quota may go to
        # `claude-fable-5`, and fable running out leaves the rest with
        # room. The pool's exhaustion still reaches a sub-allowance,
        # which is why this guard is here and not in the recipient
        # filter below.
        return frozenset()
    siblings = pool_siblings.get(key)
    if not siblings:
        return frozenset()
    return frozenset(s for s in siblings if s != key and s not in passthrough_auth)


def reduce(
    *,
    prior: HealthState,
    outcomes: dict[OfferingKey, Outcome],
    observations: list[Observation],
    admitted: set[OfferingKey],
    passthrough_auth: set[OfferingKey],
    now: datetime,
    pool_siblings: dict[OfferingKey, frozenset[OfferingKey]] | None = None,
    sub_allowances: set[OfferingKey] | None = None,
) -> HealthState:
    """Return the next Health State.

    Precedence rule, for an Offering that has both a Probe outcome and
    one or more Journal observations this run: apply every event in
    chronological order, oldest first, with the Probe outcome always
    last. A Probe outcome carries the timestamp `now`. That timestamp is
    never earlier than a Journal observation's `observed_at`, because a
    Journal entry reports something that already happened. The last
    event applied wins. A Probe is a deliberate, current measurement.
    This order lets a Probe confirm or override the very Journal entries
    that made the Prober decide to measure in the first place. Every
    event still updates `failure_count` and `last_attempt_at` on its way
    through.

    An Inconclusive outcome leaves the prior record untouched, not even
    `last_attempt_at`. It states that the attempt measured nothing.

    A Journal observation passes through `journal_outcome` first, so an
    `unrecognized_failure` from real traffic becomes Inconclusive and
    changes nothing. A Probe outcome never does: the Prober keeps
    `classify`'s fail-closed default. See `journal_outcome` and ADR
    0008.

    An `answered` outcome clears an exclusion, sets `last_success_at`
    and `last_attempt_at` to the event time, and resets `failure_count`
    to zero.

    A failing outcome (`self_healing`, `needs_operator`, `gone`)
    Excludes the Offering. The one exception: the Offering is
    Passthrough Auth and `outcome.reason` is a quota exhaustion or an
    authentication failure, whatever the bucket. A quota exhaustion
    carries `self_healing` when the message states a non-zero limit,
    and `needs_operator` when the limit is zero. An authentication
    failure always carries `needs_operator`. Either belongs to the
    caller whose credentials failed, not to the Offering. So it is
    recorded (`reason`, `bucket`, `failure_count`, `last_attempt_at` all
    update), but `excluded` does not become `True`. Every other reason
    still Excludes a Passthrough Auth Offering, including a gateway
    error, a timeout and a rate limit. Those conditions describe the
    Offering itself, not one caller's credentials.

    An Offering with no event this run, whose prior record is Excluded
    and whose `reset_at` has passed (`reset_at <= now`), clears its
    exclusion with no Probe. A `reset_at` still in the future keeps the
    exclusion. A `reset_at` exactly equal to `now` counts as passed:
    the reset already names the instant recovery begins.

    `sub_allowances` names keys capped INSIDE their pool. Such a key
    propagates nothing outward when it exhausts, but still receives the
    pool's exhaustion — one-way containment. See
    `entitlements.sub_allowance_keys`.

    `pool_siblings` maps an Offering key to the other keys billed from
    the same pool. A quota exhaustion marks those siblings `probe_due`,
    which asks the Prober to measure them next sweep. It never Excludes
    one and never removes one from the Generated Config. See
    `_pool_siblings_to_mark` and ADR 0004. Omit the argument and
    nothing propagates at all.

    A record for an Offering `admitted` no longer contains is discarded.

    `reduce` mutates neither `prior` nor any argument, and returns the
    same result for the same inputs.
    """
    events: dict[OfferingKey, list[tuple[datetime, Outcome]]] = {}
    for observation in observations:
        events.setdefault(observation.offering_id, []).append(
            (observation.observed_at, journal_outcome(observation.outcome))
        )
    for key, outcome in outcomes.items():
        events.setdefault(key, []).append((now, outcome))

    keys = set(prior.offerings) | set(events)

    siblings_map = pool_siblings or {}
    sub_allowance_keys = sub_allowances or frozenset()

    next_offerings: dict[OfferingKey, OfferingHealth] = {}
    due: set[OfferingKey] = set()
    for key in keys:
        record = prior.offerings.get(key, _EMPTY_RECORD)
        key_events = events.get(key)
        if key_events is None:
            next_offerings[key] = _apply_reset_expiry(record, now=now)
            continue
        key_events.sort(key=lambda item: item[0])
        is_passthrough = key in passthrough_auth
        for at, outcome in key_events:
            record = _apply_outcome(record, outcome, at=at, is_passthrough=is_passthrough)
            due |= _pool_siblings_to_mark(
                key=key,
                outcome=outcome,
                pool_siblings=siblings_map,
                passthrough_auth=passthrough_auth,
                sub_allowances=sub_allowance_keys,
            )
        next_offerings[key] = record

    # Mark the pool siblings last, and only those this run did not
    # measure. An Offering with its own event was just measured, so a
    # mark asking to measure it again would be stale the moment it was
    # written.
    for key in due - set(events):
        record = next_offerings.get(key)
        if record is not None and not record.probe_due:
            next_offerings[key] = replace(record, probe_due=True)

    admitted_offerings = {key: record for key, record in next_offerings.items() if key in admitted}
    return HealthState(offerings=admitted_offerings)


def _apply_reset_expiry(record: OfferingHealth, *, now: datetime) -> OfferingHealth:
    """Clear an exclusion whose reset time has passed, with no Probe.

    Leave every other record untouched. `reset_at <= now` counts as
    passed.
    """
    if not record.excluded or record.reset_at is None:
        return record
    if record.reset_at > now:
        return record
    return replace(record, excluded=False, reason=None, bucket=None, reset_at=None)


def _apply_outcome(
    record: OfferingHealth, outcome: Outcome, *, at: datetime, is_passthrough: bool
) -> OfferingHealth:
    """Fold one Outcome, at one time, onto one record."""
    if outcome.bucket == INCONCLUSIVE:
        # Health is untouched: an Inconclusive attempt measured nothing
        # ABOUT THE OFFERING. `inconclusive_count` is one exception, and
        # it exists to make a silent misclassification visible.
        #
        # `alias_not_served` is the other. It measured nothing about the
        # Offering and a great deal about our own pipeline: the proxy
        # refused an Alias the Generated Config was supposed to carry.
        # Recorded as a timestamp so a caller can judge its age, and
        # never as an exclusion — the Offering is fine and removing it
        # would be the wrong repair. See `OfferingHealth`.
        if outcome.reason == REASON_ALIAS_NOT_SERVED:
            return replace(
                record,
                inconclusive_count=record.inconclusive_count + 1,
                alias_not_served_at=at,
            )
        return replace(record, inconclusive_count=record.inconclusive_count + 1)

    # Any applied event measured this Offering, so it is no longer due
    # for one. An Inconclusive outcome returned above without clearing
    # the flag, which is right: it measured nothing.
    if outcome.bucket == ANSWERED:
        return replace(
            record,
            excluded=False,
            reason=None,
            bucket=None,
            reset_at=None,
            last_success_at=at,
            last_attempt_at=at,
            failure_count=0,
            probe_due=False,
            inconclusive_count=0,
        )

    exempt = is_passthrough and outcome.reason in _PASSTHROUGH_EXEMPT_REASONS
    return replace(
        record,
        excluded=record.excluded if exempt else True,
        reason=outcome.reason,
        bucket=outcome.bucket,
        reset_at=outcome.reset_at,
        last_attempt_at=at,
        failure_count=record.failure_count + 1,
        probe_due=False,
    )
