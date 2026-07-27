"""`plan`: the pure transform from Feed, Policy and Health State to a
Generated Config.

`plan` is PURE. It performs no network call, no filesystem read, no
clock read and no environment read. `now` is a parameter, not a call to
the clock. See the spec's "Three pure transforms and thin adapters".

`plan` implements the baseline filter, Selection per the per-provider
rule, the quality gate and Candidates, Alias naming, the generic
translation rule, the ordinary Excluded check, and Sunsetting. A future
change may extend this same function; it must not change the shape of
`PlanResult`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from litellm_maintainer.feed import Feed, Offering
from litellm_maintainer.limits import (
    LimitCollision,
    find_limit_collisions,
    limits_model_info,
)
from litellm_maintainer.naming import alias_for
from litellm_maintainer.policy import Policy
from litellm_maintainer.pricing import (
    DuplicateProviderModelId,
    PricingContradiction,
    cost_model_info,
    find_duplicate_provider_model_ids,
    summarize_feed_notices,
)
from litellm_maintainer.reduce import OfferingHealth
from litellm_maintainer.translate import (
    ENVELOPE_HANDLER_PREFIX,
    UnknownProviderError,
    translate_offering,
)

# The Feed's own words for "leaving its provider's catalogue" (spec,
# "Availability is a warning, not a verdict"). Verified against
# `tests/fixtures/feed-current.json`: it carries three availability
# values (`available`, `deprecated`, `retired`); `feed-audited.json`
# carries only `available`. `available` is never a leaving status.
_LEAVING_AVAILABILITY_STATUSES = frozenset({"deprecated", "retired"})

# A baseline capability filter, applied regardless of provider mode.
# An Offering must carry `tool_use`. An Offering carrying any of these
# excluded capabilities does not appear, whatever else it carries. This
# removes transcription, speech and moderation models without naming
# them. Do not add `coding` to either list — see the spec's "Selection".
#
# The spec names the categories image, audio, video, embedding and
# safety. The Feed's own capability vocabulary spells these
# `image_generation`, `text_to_speech` / `speech_to_text`,
# `video_generation`, `embeddings`, and `moderation` / `safety`. Note
# `vision` (a chat model that accepts image input) is NOT excluded: it
# is a different capability from `image_generation` (a model that
# produces images), and several admitted OpenCode Go Offerings carry it.
_REQUIRED_CAPABILITY = "tool_use"
_EXCLUDED_CAPABILITIES = frozenset(
    {
        "image_generation",
        "text_to_speech",
        "speech_to_text",
        "video_generation",
        "embeddings",
        "moderation",
        "safety",
    }
)


@dataclass(frozen=True)
class PlanReport:
    """What `plan` found, for the operator to read.

    `admitted` lists the Offering ids written into the Generated
    Config. `candidates` lists Offering ids that clear every structural
    filter but carry no quality score and are not in
    `approved_candidates` — reported, never added. `sunsetting` lists
    admitted Offering ids the Feed reports as leaving their provider's
    catalogue, kept because they still answer (CONTEXT.md,
    "Sunsetting"); report these on every run so the warning cannot be
    missed once. `excluded` lists Offering ids Health State marks
    excluded, left out by the ordinary path — a Sunsetting Offering
    that stops answering ends up here, with no special case.

    `aliases` maps each admitted Discovered Offering id to the Alias
    `naming.alias_for` chose for it. A Declared Offering is not in this
    map — the operator wrote its Alias by hand, so it needs no
    derivation to report. This is what makes the Alias of a newly
    admitted Offering visible in the report, so the operator can pin it
    with an `alias_overrides` entry if they care.

    `pricing_contradictions` lists an Offering the Feed prices `free`
    while stating a non-zero token rate. Each is treated as
    paid in the Generated Config; the entry here is the report of the
    contradiction, not a refusal. `limit_collisions` lists a
    `litellm_params.model` that two or more entries share while stating
    different Stated Limits — litellm holds one cost-map entry per model
    string, so the last entry registered defines every sibling. Reported,
    never refused, and silent when the siblings agree (ADR 0006).
    `duplicate_provider_model_ids` lists
    a provider id and model identifier that two or more Offerings in
    the Feed both claim — a Feed-shape hazard, checked over the whole
    Feed regardless of Selection. `feed_notices` carries one line per
    notice the Feed's own collector published, tolerant of a notice
    shape this tool does not recognise (see
    `litellm_maintainer.pricing.summarize_feed_notices`).

    `client_facing_variants` pairs each primary Alias with the variant
    Alias added beside it, so the operator can see the Alias count grow
    without diffing the file. `client_facing_variants_unknown` names an
    `operator_stated` Offering id this run did not admit, so a stale line
    cannot sit in Policy granting nothing.

    `restorable_by_probe` lists a Discovered Offering id that is
    `hidden`, leaving its catalogue (spec-corrections.md, correction
    9), and would clear every other Selection filter, but is absent
    from `health` (or holds no success record there). It stays out of
    `admitted` on this run for one reason only: no Probe has recorded a
    success for it yet. Once one does, the ordinary Sunsetting rule
    admits it with no Policy change. `cli.cmd_generate` reads the
    length of this tuple to warn the operator when it runs with empty
    Health State: 4 of the operator's 78 Aliases take exactly this
    path (correction 9), and 4 of 78 is under the default
    `maximum_removal_share`, so the removal-share safety check does not
    catch the loss on its own.

    `passthrough_auth_failures` lists a Passthrough Auth Declared
    Offering's Alias whose quota or authentication failure Health State
    has recorded, but which is NOT Excluded, because such a failure
    belongs to one caller, not the Offering (CONTEXT.md, "Passthrough
    Auth"). Story 33 found this state reached no report section at all:
    `reduce` correctly leaves `excluded=False`, so the Alias appears in
    no `status` section, and `notify.detect_events` fires only on
    `needs_operator` and `gone`. A quota error stating a non-zero limit
    classifies `self_healing`, so it was completely silent. Read `bucket`
    and `reason` on the same `health[alias]` record for the detail.

    `withheld` lists a Discovered Offering id Policy's `withheld` map
    names, reached in the Feed this run. Without this field, a Withheld
    Offering was reported nowhere; the operator could not tell it apart
    from a Candidate or an Exclusion. `report.py` reads this field,
    alongside `policy.withheld[offering_id]` for the reason text, to
    print the "Withheld" section of `status`.

    Correction 10 (spec-corrections.md) found that `withheld` must not
    come from the Selection pipeline: Withheld is an operator decision,
    read straight from Policy against the Feed, so it names every
    Withheld Offering id the Feed still publishes — whatever else would
    also have excluded it. `withheld_stale` names a Withheld Policy line
    for an Offering the Feed does not publish at all, so the operator
    can prune it. See `_compute_withheld`.

    `custom_provider_map_conflict`, when not `None`, states that
    Policy's `proxy_settings.litellm_settings` set `custom_provider_map`
    itself. The Generator derives this map from the Feed's envelope
    routing (correction 5) and the derived map always wins; this field
    is the loud report that a hand-written one was ignored, not a
    refusal. See `litellm_maintainer.policy.ProxySettings`.
    """

    admitted: tuple[str, ...] = ()
    candidates: tuple[str, ...] = ()
    sunsetting: tuple[str, ...] = ()
    excluded: tuple[str, ...] = ()
    passthrough_auth_failures: tuple[str, ...] = ()
    withheld: tuple[str, ...] = ()
    withheld_stale: tuple["WithheldStaleEntry", ...] = ()
    aliases: dict[str, str] = field(default_factory=dict)
    pricing_contradictions: tuple[PricingContradiction, ...] = ()
    duplicate_provider_model_ids: tuple[DuplicateProviderModelId, ...] = ()
    limit_collisions: tuple[LimitCollision, ...] = ()
    # (primary Alias, variant Alias) for each Client-Facing Variant added.
    client_facing_variants: tuple[tuple[str, str], ...] = ()
    # An `operator_stated` Offering id the Feed does not publish, or which
    # this run did not admit. A stale line here grants nothing and would
    # otherwise go stale in silence.
    client_facing_variants_unknown: tuple[str, ...] = ()
    feed_notices: tuple[str, ...] = ()
    restorable_by_probe: tuple[str, ...] = ()
    custom_provider_map_conflict: str | None = None


@dataclass(frozen=True)
class WithheldStaleEntry:
    """A Withheld Policy line naming an Offering the Feed does not publish.

    Worth pruning from Policy either way. `unknown_provider` tells the
    operator which shade of stale this is: `True` when the id's
    provider is not one the Feed lists at all (Policy names a provider
    the Feed never covered); `False` when the provider is a Feed
    provider, but this Offering id is not one of its current
    Offerings (the Feed retired or renamed it).
    """

    offering_id: str
    unknown_provider: bool


def _compute_withheld(
    policy: Policy, feed: Feed
) -> tuple[tuple[str, ...], tuple[WithheldStaleEntry, ...]]:
    """Read the Withheld report straight from Policy against the Feed.

    Withheld is an operator decision (CONTEXT.md, "Withheld"), not a
    consequence of Selection. Correction 10 found that recording a
    Withheld Offering only when the Withheld check is the gate that
    stops it under-reports: an earlier gate (visibility, the baseline
    capability filter, the named list, or the pricing filter) can drop
    the same Offering first, so it never reaches the Withheld check at
    all and the operator never sees it. This function does not walk
    the Selection loop; it checks each `policy.withheld` id against the
    Feed directly, so no other gate can pre-empt it.
    """
    present: list[str] = []
    stale: list[WithheldStaleEntry] = []
    for offering_id in policy.withheld:
        if feed.offering(offering_id) is not None:
            present.append(offering_id)
        else:
            provider_id, _, _ = offering_id.partition(":")
            stale.append(
                WithheldStaleEntry(
                    offering_id=offering_id,
                    unknown_provider=provider_id not in feed.providers,
                )
            )
    stale.sort(key=lambda entry: entry.offering_id)
    return tuple(sorted(present)), tuple(stale)


@dataclass(frozen=True)
class AliasAnnotation:
    """What the Generated Config prints above and beside one Alias.

    `group` is the heading this Alias sits under; consecutive Aliases
    sharing one are printed under a single heading. `note` is the short
    line beside the Alias, holding what a human scrolling the file wants
    and YAML does not carry: quality scores, context size, and how the
    call authenticates.

    Comments carry no meaning to litellm. Nothing reads these back.
    """

    group: str
    note: str | None = None


@dataclass(frozen=True)
class PlanResult:
    """The result of `plan`: a config document, a report, and a refusal.

    `refusal`, when not `None`, states why `plan` declined to produce a
    config at all: an Alias collision between two Offerings, for
    example. `litellm_maintainer.safety` applies a separate set of
    safety-gate refusals (a removal share too large, zero Aliases
    offered); `plan` itself never raises those.
    """

    config: dict[str, Any]
    report: PlanReport
    refusal: str | None = None
    # Per-Alias heading and note for the writer. Comments only: nothing
    # reads them back, and `render_config` works without them.
    annotations: dict[str, "AliasAnnotation"] = field(default_factory=dict)


def _plan_editions(offering: Offering) -> tuple[str, ...]:
    """The subscription editions the Feed says include this Offering.

    Read from `pricing.subscription.plan_editions`. An Offering no plan
    covers publishes none, so this returns an empty tuple and a
    `plan_edition` filter excludes it: a pay-as-you-go Offering is not on
    any subscription roster.
    """
    subscription = (offering.pricing or {}).get("subscription") or {}
    editions = subscription.get("plan_editions")
    if not isinstance(editions, list):
        return ()
    return tuple(str(edition) for edition in editions)


def _passes_plan_edition(offering: Offering, edition: str | None) -> bool:
    """Whether `offering` is on the roster of the edition the operator holds."""
    if edition is None:
        return True
    return edition in _plan_editions(offering)


def _passes_baseline(offering: Offering) -> bool:
    capabilities = set(offering.capabilities)
    if _REQUIRED_CAPABILITY not in capabilities:
        return False
    if capabilities & _EXCLUDED_CAPABILITIES:
        return False
    return True


def _admit_offering(offering, policy: Policy, candidates: list[str]) -> bool:
    """Apply the quality gate. Return whether the Offering is admitted.

    An Offering with no score is a Candidate. Policy's
    `approved_candidates` admits it. An Offering the Policy does not
    name is recorded in `candidates`, which the report prints as
    "awaiting approval".

    Warning: do not record an approved Candidate in `candidates`. It
    waits for nothing. `PlanReport.candidates` states this contract,
    and `cli.cmd_generate` prints the list under "Candidates awaiting
    approval". An earlier version appended every unscored Offering, so
    the operator read 19 already-approved Offerings as still awaiting a
    decision.

    This reads `offering.coding_score` only, never `capabilities`. The
    spec's "Selection" section explains why the `coding` capability
    flag must not gate or grant admission, at length, on purpose: the
    flag needs a high-confidence Canonical Model join that five
    single-provider Offerings can never complete, it propagates at
    medium confidence so gating on it would make a silent drop the
    failure mode Candidates exist to prevent, and requiring it would
    drop the four Sunsetting Offerings the Sunsetting rule exists to
    keep. Do not reverse this on intuition; read that section first.
    """
    score = offering.coding_score
    if score is None:
        if offering.id in policy.approved_candidates:
            return True
        candidates.append(offering.id)
        return False
    return score >= policy.quality.minimum_coding_score


def _would_pass_selection_ignoring_visibility(
    offering: Offering,
    *,
    allowed_ids: set[str] | None,
    allowed_pricing: set[str] | None,
    plan_edition: str | None,
    policy: Policy,
) -> bool:
    """Whether `offering` would clear every Selection filter but visibility.

    Used only to compute `PlanReport.restorable_by_probe`: it repeats
    the baseline, named-model, pricing, Withheld and quality checks the
    main loop in `plan` already applies, skipping only the visibility
    check (the whole point here) and the Excluded check (health has no
    record yet by construction — see the caller). A scratch list
    absorbs any Candidate `_admit_offering` would otherwise record, so
    counting a restorable Offering never pollutes `PlanReport.candidates`.
    """
    if not _passes_baseline(offering):
        return False
    if allowed_ids is not None and offering.id not in allowed_ids:
        return False
    if allowed_pricing is not None and offering.pricing_kind not in allowed_pricing:
        return False
    if not _passes_plan_edition(offering, plan_edition):
        return False
    if offering.id in policy.withheld:
        return False
    return _admit_offering(offering, policy, [])


def _is_sunsetting(offering: Offering, record: OfferingHealth | None) -> bool:
    """Whether `offering` is Sunsetting: leaving its catalogue, still working.

    CONTEXT.md, "Sunsetting": a Discovered Offering the Feed reports as
    leaving its provider's catalogue, which still answers a Probe. The
    Feed gives the warning (`availability.status`); the Probe decides
    whether the Offering still works.

    "Still works" is read from OUR Health State only. The Offering is
    Sunsetting when Health State holds a record for it and
    `record.last_success_at is not None`, which means our own Probe
    answered at least once. Staleness is judged on
    `record.last_success_at`, never `record.last_attempt_at`: the
    latter advances on a failing attempt too, so reading it instead
    could mistake a live-but-failing Offering for a working one. An
    Offering with no record, or with a record that has never answered,
    is not Sunsetting — the rule keeps working capacity, it is not a
    route in for a model that the Feed disowns.

    Warning: do not fall back to the Feed's own
    `availability.last_success_at`. An earlier version did, and
    correction 6 in `.scratch/maintainer-v1/spec-corrections.md`
    settled it as wrong. Two reasons. First, the spec states that every
    availability fact in the Feed is observed with the Feed owner's
    credentials, so it never shows that this operator can call the
    Offering. Second, the field does not discriminate: all 1164
    Offerings in `tests/fixtures/feed-audited.json` and all 1163 in
    `tests/fixtures/feed-current.json` carry a non-null value. The
    fallback therefore reduced to "the Feed says the Offering leaves",
    and it let a `hidden` Offering bypass the visibility filter on the
    Feed's word alone.

    This function does not read `record.excluded`. The ordinary
    Excluded check runs separately in `plan`, after Selection, and
    applies to every Offering alike — Sunsetting or not. That is what
    keeps "a Sunsetting Offering that stops answering is Excluded by
    the ordinary path" free of a special case: this function still
    says "Sunsetting" (the Feed still reports it leaving, and it once
    worked), and the ordinary check, not this one, is what removes it
    once `record.excluded` is `True`.

    Health State starts empty on the first run, so no Offering is
    Sunsetting on that run. That is the correct result. A Sunsetting
    Offering that is also `hidden` therefore appears only after our own
    Prober records a success for it. Seed Health State to test the rule
    (see `tests/test_quality_and_sunsetting.py`).
    """
    if offering.availability_status not in _LEAVING_AVAILABILITY_STATUSES:
        return False
    if record is None:
        return False
    return record.last_success_at is not None


def plan(
    *, feed: Feed, policy: Policy, health: dict[str, OfferingHealth], now: datetime
) -> PlanResult:
    """Produce a Generated Config from a Feed, a Policy and Health State.

    `health` maps a Discovered Offering id to its `OfferingHealth`
    record (`litellm_maintainer.reduce.HealthState.offerings`). An
    Offering absent from `health` has never been probed or observed.

    `now` is accepted so the signature stays stable for later tickets;
    this slice does not read it. Every timestamp comparison this ticket
    needs — Excluded, and Sunsetting's staleness rule — reads and
    compares recorded fields (`last_success_at` against
    `last_attempt_at`, or the Feed's own recorded fields), never the
    clock.
    """
    del now  # not read by this slice; kept for a stable signature

    entries: list[dict[str, Any]] = []
    annotations: dict[str, AliasAnnotation] = {}
    admitted: list[str] = []
    candidates: list[str] = []
    sunsetting: list[str] = []
    excluded: list[str] = []
    pricing_contradictions: list[PricingContradiction] = []
    restorable_by_probe: list[str] = []
    # (primary Alias, variant Alias) for each Client-Facing Variant added.
    client_facing_variants: list[tuple[str, str]] = []

    # Both checks read the whole Feed document, not the admitted set:
    # the hazard is in the Feed's own shape, regardless of Selection.
    duplicate_provider_model_ids = find_duplicate_provider_model_ids(feed)
    feed_notices = summarize_feed_notices(feed.notices)
    # Alias -> the Discovered Offering id that already claimed it.
    # Checked before an Alias is granted, so two Discovered Offerings
    # deriving the same Alias are caught in the order the Feed lists
    # them.
    alias_owner: dict[str, str] = {}
    # Offering id -> the Alias `naming.alias_for` chose for it. Reported
    # so a newly admitted Offering's Alias is visible to the operator.
    aliases_by_id: dict[str, str] = {}

    # A Declared Offering may name the Discovered Offering it supersedes
    # (CONTEXT.md, "Declared Offering"; spec, "Declared Offerings"). That
    # Discovered Offering is suppressed below: it never reaches
    # Selection, so it cannot collide with the Declared entry that
    # replaces it and never appears as admitted, Candidate or Excluded.
    superseded_ids = {d.supersedes for d in policy.declared if d.supersedes}

    # Alias -> the Declared Offering claiming it. Used to detect a
    # Declared/Discovered Alias collision below, and to name the
    # Declared side of the report when one is found.
    declared_by_alias = {d.alias: d for d in policy.declared}

    # Two Declared Offerings that claim one Alias collide too. litellm
    # reads the two entries as one load-balancing group and splits
    # traffic between two different models (docs/gotchas.md, "Duplicate
    # model_name values do not raise an error"). `supersedes` cannot
    # resolve this pair, because it names a Discovered Offering, so the
    # refusal tells the operator to rename one side.
    if len(declared_by_alias) != len(policy.declared):
        seen: set[str] = set()
        for declared in policy.declared:
            if declared.alias in seen:
                refusal = (
                    f"Alias collision on {declared.alias!r}: two Declared "
                    "Offerings in Policy both claim it. Give one of the two "
                    "a different Alias."
                )
                return PlanResult(config={}, report=PlanReport(), refusal=refusal)
            seen.add(declared.alias)

    # Declared Offerings pass through verbatim. The Feed does not
    # publish them, so no selection, translation or naming rule applies.
    # The ordinary Excluded check still does: `reduce` Excludes a
    # Declared Offering on a non-exempt failure (a gateway error, a
    # timeout — CONTEXT.md, "Passthrough Auth": "Other failure kinds
    # still Exclude it"), and an Excluded Offering must leave the
    # Generated Config (story 19) whether it is Declared or Discovered.
    # A Passthrough Auth Offering's quota and authentication failures
    # never set `excluded` (`reduce._PASSTHROUGH_EXEMPT_REASONS`), so
    # this check cannot drop one for a failure that belongs to a
    # caller. Recovery is the ordinary path too: a later Probe success
    # or a passed reset time clears `excluded` in Health State.
    for declared in policy.declared:
        # Read the pair's shared record. A Client-Facing Variant is the
        # same wire request under a second name, so it leaves the
        # Generated Config exactly when the Alias it widens does.
        # Reading `declared.alias` here left a variant offered while its
        # twin was Excluded -- an Alias certain to fail.
        record = health.get(declared.health_key)
        if record is not None and record.excluded:
            excluded.append(declared.alias)
            continue
        entry: dict[str, Any] = {
            "model_name": declared.alias,
            "litellm_params": dict(declared.litellm_params),
        }
        if declared.model_info is not None:
            entry["model_info"] = dict(declared.model_info)
        entries.append(entry)
        annotations[declared.alias] = AliasAnnotation(
            group=declared.group or "Declared",
            note=_declared_note(declared),
        )
        # A Declared Offering IS offered: it reaches the Generated
        # Config verbatim (correction 10). `report.py` prints
        # "(Declared)" for an admitted id absent from `aliases` — see
        # `PlanReport.aliases` — so the id recorded here is the Alias
        # itself, the only id a Declared Offering has.
        admitted.append(declared.alias)

    # Story 33: a Passthrough Auth Declared Offering's quota or
    # authentication failure is recorded (`reason`, `bucket` set) but
    # never Excludes it (`reduce._PASSTHROUGH_EXEMPT_REASONS`). That
    # combination -- a record with `reason` set and `excluded` still
    # `False` -- exists only for exactly this case, so it is read
    # straight from Health State rather than re-deriving which reasons
    # are exempt (`reduce.py` already owns that rule).
    passthrough_auth_failures = tuple(
        sorted(
            declared.alias
            for declared in policy.declared
            if declared.passthrough_auth
            and (record := health.get(declared.alias)) is not None
            and record.reason is not None
            and not record.excluded
        )
    )

    for provider_id, rule in sorted(policy.providers.items()):
        if provider_id not in _translatable_providers():
            # No translation rule registered for this provider. It
            # contributes nothing rather than raising; see
            # `litellm_maintainer.translate.TRANSLATION_RULES`.
            continue

        provider = feed.providers.get(provider_id)
        allowed_ids = set(rule.models or ()) if rule.mode == "named" else None
        allowed_pricing = set(rule.pricing) if rule.pricing else None

        for offering in feed.offerings_for(provider_id):
            record = health.get(offering.id)
            sunsetting_offering = _is_sunsetting(offering, record)

            # Selection reads listed Offerings (spec, "Availability is a
            # warning, not a verdict"). A Sunsetting Offering is the one
            # deliberate exception: the four OpenCode Go Offerings this
            # rule exists to keep are `hidden`, not `listed`, because
            # the Feed's own visibility follows its availability
            # status. Excluding them here would make the "listed"
            # check, not the quality gate, Policy or Health State, the
            # reason a kept Offering disappears — the opposite of what
            # Sunsetting is for. This also lets a Sunsetting Offering
            # that later stops answering reach the ordinary Excluded
            # check below, rather than vanishing here unreported: see
            # `_is_sunsetting`'s docstring on why it never reads
            # `record.excluded`. Every other filter below still applies
            # unchanged.
            if offering.visibility != "listed" and not sunsetting_offering:
                # This Offering would be Sunsetting, not merely hidden,
                # the moment our own Prober records one success for it
                # (see `_is_sunsetting`). Count it as restorable when it
                # is leaving its catalogue, holds no such record yet,
                # is not superseded by a Declared Offering, and clears
                # every other Selection filter.
                if (
                    offering.availability_status in _LEAVING_AVAILABILITY_STATUSES
                    and (record is None or record.last_success_at is None)
                    and offering.id not in superseded_ids
                    and _would_pass_selection_ignoring_visibility(
                        offering,
                        allowed_ids=allowed_ids,
                        allowed_pricing=allowed_pricing,
                        plan_edition=rule.plan_edition,
                        policy=policy,
                    )
                ):
                    restorable_by_probe.append(offering.id)
                continue
            if not _passes_baseline(offering):
                continue
            if allowed_ids is not None and offering.id not in allowed_ids:
                continue
            if allowed_pricing is not None and offering.pricing_kind not in allowed_pricing:
                continue
            if not _passes_plan_edition(offering, rule.plan_edition):
                # The operator's subscription edition does not include
                # this Offering, so a call on it cannot succeed. Filtered
                # by Selection, from the Feed's own
                # `pricing.subscription.plan_editions`, rather than by a
                # hand-written Withheld line that goes stale when the
                # roster changes.
                continue
            if offering.id in policy.withheld:
                # Reported separately by `_compute_withheld`, read
                # straight from Policy against the Feed (correction 10)
                # so no earlier gate can pre-empt it. This `continue`
                # still keeps a Withheld Offering out of admission.
                continue
            if offering.id in superseded_ids:
                # A Declared Offering names this one as the Discovered
                # Offering it supersedes. Suppressed: it does not reach
                # admission, so it cannot collide with the Declared
                # entry, and it is neither reported nor excluded.
                continue

            # The ordinary Excluded path. It applies uniformly, whether
            # or not the Offering is Sunsetting: a Sunsetting Offering
            # that stops answering is Excluded here, with no special
            # case written for its death (spec, "Availability is a
            # warning, not a verdict").
            if record is not None and record.excluded:
                excluded.append(offering.id)
                continue

            if not _admit_offering(offering, policy, candidates):
                continue

            alias = alias_for(policy, offering.id)

            # A collision stops the run rather than corrupting routing.
            # litellm treats two entries sharing a `model_name` as one
            # load-balancing group, not an error, so an undetected
            # collision would silently split traffic between two
            # different models (docs/gotchas.md, "Duplicate model_name
            # values do not raise an error"). A Declared Offering always
            # wins conceptually, but "wins silently" is exactly the
            # hidden mistake this check exists to prevent, so it still
            # stops the run unless Policy names the resolution with
            # `supersedes`.
            declared_clash = declared_by_alias.get(alias)
            if declared_clash is not None:
                refusal = (
                    f"Alias collision on {alias!r}: a Declared Offering "
                    f"(model={declared_clash.litellm_params.get('model')!r}) "
                    f"and the Discovered Offering {offering.id!r} both "
                    f"claim it. Add 'supersedes: {offering.id!r}' to the "
                    "Declared Offering in Policy to resolve it, or give "
                    "one of the two a different Alias."
                )
                return PlanResult(config={}, report=PlanReport(), refusal=refusal)
            other_owner = alias_owner.get(alias)
            if other_owner is not None:
                refusal = (
                    f"Alias collision on {alias!r}: the Discovered "
                    f"Offerings {other_owner!r} and {offering.id!r} both "
                    "derive it. Add a 'naming.alias_overrides' entry in "
                    "Policy for one of them to resolve it."
                )
                return PlanResult(config={}, report=PlanReport(), refusal=refusal)

            override = dict(rule.translation or {})
            override.update(policy.translation_overrides.get(offering.id, {}))
            try:
                litellm_params = translate_offering(
                    offering,
                    provider,
                    override=override or None,
                    policy_envelope_key=rule.response_envelope_key,
                )
            except UnknownProviderError:
                continue

            model_info, contradiction = cost_model_info(offering, litellm_params)
            if contradiction is not None:
                pricing_contradictions.append(contradiction)
            # Stated Limits go underneath, so `cost_model_info` stays the
            # sole author of every cost key. The two key sets do not
            # overlap; the order states which module owns which key.
            model_info = {**limits_model_info(offering), **model_info}

            entry: dict[str, Any] = {"model_name": alias, "litellm_params": litellm_params}
            if model_info:
                entry["model_info"] = model_info
            entries.append(entry)
            annotations[alias] = AliasAnnotation(
                group=_provider_heading(offering.provider_id, provider),
                note=_offering_note(offering),
            )
            admitted.append(offering.id)
            alias_owner[alias] = offering.id
            aliases_by_id[offering.id] = alias
            if sunsetting_offering:
                sunsetting.append(offering.id)

            # A Client-Facing Variant: the same entry under a second Alias,
            # because a calling client reads its own context budget out of
            # the name (CONTEXT.md, ADR 0007). The Offering is admitted
            # once; only the Alias count grows.
            if _qualifies_for_variant(offering, policy.client_facing_variants):
                variant_alias = f"{alias}{policy.client_facing_variants.suffix}"  # type: ignore[union-attr]
                owner = alias_owner.get(variant_alias)
                declared_aliases = {d.alias for d in policy.declared}
                if owner is not None or variant_alias in declared_aliases:
                    held_by = owner or "a Declared Offering"
                    refusal = (
                        f"Alias collision on {variant_alias!r}: the "
                        f"Client-Facing Variant of {alias!r} would take a "
                        f"name {held_by!r} already holds. Rename one, or "
                        "change 'client_facing_variants.suffix'."
                    )
                    return PlanResult(config={}, report=PlanReport(), refusal=refusal)
                variant = {"model_name": variant_alias, "litellm_params": litellm_params}
                if model_info:
                    variant["model_info"] = model_info
                entries.append(variant)
                annotations[variant_alias] = AliasAnnotation(
                    group=_provider_heading(offering.provider_id, provider),
                    note=_variant_note(alias),
                )
                alias_owner[variant_alias] = offering.id
                client_facing_variants.append((alias, variant_alias))

    # Every entry is built now, Declared and Discovered alike, so this
    # reads the same facts that reach the file rather than re-deriving
    # which Offerings share a model string.
    limit_collisions = find_limit_collisions(entries)

    # `model_list` first, then `general_settings`, then `litellm_settings`
    # — the order the operator's own hand-built config already used, so
    # the Generated Config reads the same way.
    config: dict[str, Any] = {"model_list": entries}

    if policy.proxy_settings.general_settings:
        config["general_settings"] = dict(policy.proxy_settings.general_settings)

    # `litellm_settings.custom_provider_map` stays DERIVED. Start from
    # Policy's own `litellm_settings` (an arbitrary mapping, passed
    # through verbatim), but a `custom_provider_map` key there is never
    # emitted: it is reported as a conflict instead, and the derived
    # value below always wins. See `PlanReport.custom_provider_map_conflict`
    # and `litellm_maintainer.policy.ProxySettings`.
    litellm_settings: dict[str, Any] = dict(policy.proxy_settings.litellm_settings)
    custom_provider_map_conflict: str | None = None
    if "custom_provider_map" in litellm_settings:
        custom_provider_map_conflict = (
            "Policy's litellm_settings.custom_provider_map is ignored. The "
            "Generator derives this map from the Feed's envelope routing "
            "(spec-corrections.md, correction 5), so a hand-written map "
            "would go stale the moment the Feed's routing changes, with no "
            "symptom to notice by. The derived map wins."
        )
        del litellm_settings["custom_provider_map"]
    if _any_entry_uses_envelope_handler(entries):
        litellm_settings["custom_provider_map"] = [
            {
                "provider": ENVELOPE_HANDLER_PREFIX,
                "custom_handler": "cline_provider.cline_llm",
            }
        ]
    if litellm_settings:
        config["litellm_settings"] = litellm_settings

    withheld, withheld_stale = _compute_withheld(policy, feed)

    report = PlanReport(
        admitted=tuple(admitted),
        candidates=tuple(sorted(set(candidates))),
        sunsetting=tuple(sunsetting),
        excluded=tuple(excluded),
        passthrough_auth_failures=passthrough_auth_failures,
        withheld=withheld,
        withheld_stale=withheld_stale,
        aliases=dict(aliases_by_id),
        pricing_contradictions=tuple(pricing_contradictions),
        duplicate_provider_model_ids=duplicate_provider_model_ids,
        limit_collisions=limit_collisions,
        client_facing_variants=tuple(client_facing_variants),
        client_facing_variants_unknown=_unknown_variant_statements(
            policy, tuple(admitted)
        ),
        feed_notices=feed_notices,
        restorable_by_probe=tuple(restorable_by_probe),
        custom_provider_map_conflict=custom_provider_map_conflict,
    )
    return PlanResult(config=config, report=report, refusal=None, annotations=annotations)



def _provider_heading(provider_id: str, provider: Any) -> str:
    """The heading a Discovered Offering sits under.

    The Feed's own provider name when it publishes one, because that is
    what a human recognises ("OpenCode Go", not "opencode-go"). Falls
    back to the provider id, which every Offering has.
    """
    name = getattr(provider, "name", None) if provider is not None else None
    return name or provider_id


def _format_context(tokens: Any) -> str | None:
    """A context window as a human reads it: 1M, 262K, 8192."""
    if not isinstance(tokens, int) or tokens <= 0:
        return None
    if tokens >= 1_000_000 and tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}M ctx"
    if tokens >= 1000:
        return f"{round(tokens / 1000)}K ctx"
    return f"{tokens} ctx"


def _offering_note(offering: Offering) -> str | None:
    """The note beside a Discovered Offering.

    Holds what YAML does not carry and a human scanning the file wants:
    the three quality scores, then the context window. Returns `None`
    when the Feed states none of them, so no empty comment is written.
    """
    quality = offering.quality or {}
    scores = [
        quality.get("coding_score"),
        quality.get("reasoning_score"),
        quality.get("agentic_score"),
    ]
    parts: list[str] = []
    if any(isinstance(s, (int, float)) for s in scores):
        rendered = " / ".join(
            f"{s:g}" if isinstance(s, (int, float)) else "-" for s in scores
        )
        parts.append(f"{rendered} coding/reasoning/agentic")
    context = _format_context(offering.context_tokens)
    if context:
        parts.append(context)
    return " — ".join(parts) or None


def _qualifies_for_variant(offering: Offering, rule: Any) -> bool:
    """Whether this Offering earns a Client-Facing Variant.

    Two ways to qualify, and no third. The Feed states a context window at
    or above the threshold, so the Feed decides and the set cannot go
    stale. Or the operator names the Offering in `operator_stated`, for a
    model the Feed has not sized yet.

    An Offering the Feed does not size qualifies through neither. Nothing
    is derived from a model name — see ADR 0006.
    """
    if rule is None:
        return False
    if offering.id in rule.operator_stated:
        return True
    stated = offering.context_tokens
    return isinstance(stated, int) and stated >= rule.minimum_context_tokens


def _variant_note(primary_alias: str) -> str:
    """The note beside a Client-Facing Variant.

    States the one thing a reader cannot see from the entry: that this
    Alias is not a second Offering. Both Aliases send the same request.
    """
    return (
        f"same request as {primary_alias}; the suffix widens the calling "
        "client's own context budget, and the provider never sees it"
    )


def _declared_note(declared: Any) -> str | None:
    """The note beside a Declared Offering.

    Notes the exception, never the norm. Auth mode is uniform within a
    group, so the operator's `group` heading carries it once; repeating
    it per entry only adds noise. `proxy_authenticated` is the unusual
    case — a Passthrough Auth Offering the proxy authenticates itself —
    and that is worth naming on the entry it applies to.
    """
    if declared.proxy_authenticated:
        return "the proxy holds this credential itself, not the caller"
    return None

def _any_entry_uses_envelope_handler(entries: list[dict[str, Any]]) -> bool:
    """Whether any entry's `model` was routed through the envelope handler.

    Read from the generated `litellm_params`, not from any provider id,
    matching the same data-driven basis `translate_offering` uses.
    """
    handler_prefix = f"{ENVELOPE_HANDLER_PREFIX}/"
    for entry in entries:
        model = entry.get("litellm_params", {}).get("model", "")
        if model.startswith(handler_prefix):
            return True
    return False


def _translatable_providers() -> frozenset[str]:
    from litellm_maintainer.translate import TRANSLATION_RULES

    return frozenset(TRANSLATION_RULES)


def _unknown_variant_statements(policy: Policy, admitted: tuple[str, ...]) -> tuple[str, ...]:
    """`operator_stated` Offering ids this run did not admit.

    Such a line grants nothing: the Offering reached no entry, so no
    variant was added for it. Reported rather than refused, because a Feed
    revision can drop an Offering the operator still expects back.
    """
    rule = policy.client_facing_variants
    if rule is None:
        return ()
    return tuple(sorted(set(rule.operator_stated) - set(admitted)))
