"""`explain`: where one Offering left the path to a callable Alias.

The path runs Feed -> Policy -> Health State -> Generated Config ->
proxy. An Offering can leave it at any stage, and every stage is
reported by a different command. This module walks the whole path for
one Offering and names the stage that stopped it.

`explain` is PURE. It reads no clock, no filesystem and no network.
The live proxy stage arrives as a parameter, so a caller that cannot
reach the proxy passes `None` and the stage reports UNKNOWN.

## A stop is typed, and the type decides what the operator does

**Decision.** Nothing is broken. Policy, the Feed or a pending operator
choice stopped the Offering, and the system did what it was told. The
report names the construct responsible, so the operator can change it or
agree with it.

**Fault.** Something is stale or broken, and the operator repairs it.

The two demand opposite responses, and one word for both teaches the
operator to read their own deliberate filters as bugs. CONTEXT.md keeps
Withheld (the operator's choice) apart from Excluded (an observation)
for the same reason.

A Decision stop reached before any Route is minted is the "routeless"
case. It needs no separate concept.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from litellm_maintainer.feed import Feed
from litellm_maintainer.plan import (
    DROPPED_BASELINE,
    DROPPED_GONE,
    DROPPED_NO_TRANSLATION_RULE,
    DROPPED_PLAN_EDITION,
    DROPPED_PROVIDER_MODELS,
    DROPPED_PROVIDER_NOT_CONFIGURED,
    DROPPED_PROVIDER_PRICING,
    DROPPED_SUPERSEDED,
    DROPPED_UNSCORED,
    DROPPED_UNTRANSLATABLE,
    DROPPED_VISIBILITY,
    DROPPED_WITHHELD,
    PlanReport,
)
from litellm_maintainer.policy import Policy
from litellm_maintainer.reduce import OfferingHealth

#: A stop the system chose. Nothing is broken.
DECISION = "decision"
#: A stop that needs a repair.
FAULT = "fault"

#: A stage's verdict.
PASSED = "passed"
STOPPED = "stopped"
UNKNOWN = "unknown"

STAGE_FEED = "feed"
STAGE_POLICY = "policy"
STAGE_HEALTH = "health"
STAGE_CONFIG = "generated_config"
STAGE_PROXY = "proxy"

STAGES = (STAGE_FEED, STAGE_POLICY, STAGE_HEALTH, STAGE_CONFIG, STAGE_PROXY)

# Each Selection token, as the kind of stop it is and the construct a
# reader should go and look at. The construct is a Policy path where
# Policy decided, and a Feed field where the Feed did.
_DROPPED_DETAIL: dict[str, tuple[str, str]] = {
    DROPPED_PROVIDER_NOT_CONFIGURED: (
        DECISION,
        "Policy configures no `providers.{provider}` rule, so Selection never "
        "reads this provider's Offerings.",
    ),
    DROPPED_NO_TRANSLATION_RULE: (
        FAULT,
        "No translation rule is registered for provider `{provider}`, so "
        "every one of its Offerings is dropped. See "
        "`litellm_maintainer.translate.TRANSLATION_RULES`.",
    ),
    DROPPED_UNTRANSLATABLE: (
        FAULT,
        "The Feed describes provider `{provider}` too thinly to build a call "
        "from, so no litellm entry could be written for this Offering.",
    ),
    DROPPED_VISIBILITY: (
        DECISION,
        "The Feed does not list this Offering (`visibility`), and it is not "
        "Sunsetting, so Selection skipped it.",
    ),
    DROPPED_BASELINE: (
        DECISION,
        "The Offering fails the baseline filter: the Feed states it is not "
        "usable as a chat model.",
    ),
    DROPPED_PROVIDER_MODELS: (
        DECISION,
        "`providers.{provider}.models` does not name this Offering, and the "
        "rule's mode is `named`.",
    ),
    DROPPED_PROVIDER_PRICING: (
        DECISION,
        "`providers.{provider}.pricing` does not admit this Offering's "
        "pricing kind.",
    ),
    DROPPED_PLAN_EDITION: (
        DECISION,
        "`providers.{provider}.plan_edition` does not include this Offering, "
        "per the Feed's own `pricing.subscription.plan_editions`.",
    ),
    DROPPED_WITHHELD: (
        DECISION,
        "`withheld` holds this Offering. Only the operator clears a Withheld "
        "line.",
    ),
    DROPPED_SUPERSEDED: (
        DECISION,
        "A Declared Offering names this one as the Discovered Offering it "
        "supersedes, so it is suppressed in favour of that entry.",
    ),
    DROPPED_GONE: (
        FAULT,
        "Health State records the identifier Gone: it no longer answers for "
        "this account. Only the operator clears it.",
    ),
    DROPPED_UNSCORED: (
        DECISION,
        "The Offering carries no quality score and `approved_candidates` does "
        "not name it, so it is a Candidate awaiting the operator.",
    ),
}


@dataclass(frozen=True)
class Stage:
    """One stage of the path, and what it said about this Offering."""

    name: str
    verdict: str
    detail: str = ""


@dataclass(frozen=True)
class Explanation:
    """Where one Offering left the path, and why.

    `stopped_at` names the stage that stopped it, or `None` when the
    Offering reaches a client. `stop_kind` is `DECISION` or `FAULT`, and
    `None` when nothing stopped it.

    An Offering can pass every stage and still not be recommended: an
    Excluded Offering stays in the Generated Config and the proxy serves
    it (ADR 0014). `recommended` states that separately, so a reader
    never mistakes "reachable" for "a good idea".
    """

    query: str
    health_key: str | None = None
    offering_id: str | None = None
    alias: str | None = None
    stages: tuple[Stage, ...] = ()
    stopped_at: str | None = None
    stop_kind: str | None = None
    stop_detail: str = ""
    recommended: bool | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def reaches_a_client(self) -> bool:
        """Whether a client can call this Offering now."""
        return self.stopped_at is None

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "health_key": self.health_key,
            "offering_id": self.offering_id,
            "alias": self.alias,
            "stages": [
                {"stage": s.name, "verdict": s.verdict, "detail": s.detail}
                for s in self.stages
            ],
            "stopped_at": self.stopped_at,
            "stop_kind": self.stop_kind,
            "stop_detail": self.stop_detail,
            "reaches_a_client": self.reaches_a_client,
            "recommended": self.recommended,
            "notes": list(self.notes),
        }


def _stopped(
    *,
    query: str,
    stages: list[Stage],
    stage: str,
    kind: str,
    detail: str,
    health_key: str | None = None,
    offering_id: str | None = None,
    alias: str | None = None,
) -> Explanation:
    """Build the answer for a walk that stopped, padding what follows.

    Every stage appears, so a reader sees the whole path and where it
    ended. A stage after the stop reports UNKNOWN, never PASSED and never
    STOPPED: the walk never reached it, and an unreached stage is
    unmeasured rather than good or bad.
    """
    reached = STAGES.index(stage)
    for later in STAGES[reached + 1 :]:
        stages.append(Stage(later, UNKNOWN, "not reached: the walk stopped earlier."))
    return Explanation(
        query=query,
        health_key=health_key,
        offering_id=offering_id,
        alias=alias,
        stages=tuple(stages),
        stopped_at=stage,
        stop_kind=kind,
        stop_detail=detail,
    )


def _resolve(
    query: str, *, feed: Feed, policy: Policy, report: PlanReport
) -> tuple[str | None, str | None, str | None]:
    """Return `(health_key, offering_id, alias)` for one query string.

    A query is an Offering id or an Alias, and a Declared Offering's
    Health Key IS its Alias (CONTEXT.md, "Health Key"), so one string can
    legitimately be both. A Declared Offering wins: the operator wrote
    that Alias by hand, so it is the more specific match.
    """
    for declared in policy.declared:
        if query in (declared.alias, declared.health_key):
            return declared.health_key, None, declared.alias

    if feed.offering(query) is not None:
        return query, query, report.aliases.get(query)

    for offering_id, alias in report.aliases.items():
        if alias == query:
            return offering_id, offering_id, alias

    return None, None, None


def explain(
    *,
    query: str,
    feed: Feed,
    policy: Policy,
    health: dict[str, OfferingHealth],
    report: PlanReport,
    served_aliases: frozenset[str] | None = None,
    proxy_note: str = "",
) -> Explanation:
    """Walk one Offering's path and name the stage that stopped it.

    `served_aliases` is the Alias set the running proxy reports, or
    `None` when the proxy could not be asked. `None` reports the proxy
    stage UNKNOWN and never STOPPED: an absent answer is not a negative
    answer, and treating it as one is the exact error this verb exists
    to catch.
    """
    health_key, offering_id, alias = _resolve(
        query, feed=feed, policy=policy, report=report
    )
    stages: list[Stage] = []

    if health_key is None:
        stages.append(
            Stage(
                STAGE_FEED,
                STOPPED,
                "No Feed Offering, Declared Offering or Alias matches this name.",
            )
        )
        return _stopped(
            query=query,
            stages=stages,
            stage=STAGE_FEED,
            kind=DECISION,
            detail=(
                "Nothing by this name is known. Check the id against "
                "`status`, or the Feed Document."
            ),
        )

    is_declared = offering_id is None
    stages.append(
        Stage(
            STAGE_FEED,
            PASSED,
            "Declared in Policy, so the Feed does not describe it."
            if is_declared
            else "Published by the Feed.",
        )
    )

    # --- Policy -----------------------------------------------------
    if offering_id is not None and offering_id in report.dropped:
        token = report.dropped[offering_id]
        kind, detail = _DROPPED_DETAIL.get(
            token, (FAULT, f"Selection dropped it: {token}.")
        )
        provider = offering_id.split(":", 1)[0]
        detail = detail.format(provider=provider)
        stage = STAGE_HEALTH if token == DROPPED_GONE else STAGE_POLICY
        if stage == STAGE_HEALTH:
            stages.append(Stage(STAGE_POLICY, PASSED, "Policy admits it."))
            stages.append(Stage(STAGE_HEALTH, STOPPED, detail))
        else:
            stages.append(Stage(STAGE_POLICY, STOPPED, detail))
        return _stopped(
            query=query,
            stages=stages,
            stage=stage,
            kind=kind,
            detail=detail,
            health_key=health_key,
            offering_id=offering_id,
            alias=alias,
        )

    if (
        offering_id is not None
        and offering_id.split(":", 1)[0] not in policy.providers
        and feed.offering(offering_id) is not None
    ):
        provider = offering_id.split(":", 1)[0]
        kind, detail = _DROPPED_DETAIL[DROPPED_PROVIDER_NOT_CONFIGURED]
        detail = detail.format(provider=provider)
        stages.append(Stage(STAGE_POLICY, STOPPED, detail))
        return _stopped(
            query=query,
            stages=stages,
            stage=STAGE_POLICY,
            kind=DECISION,
            detail=detail,
            health_key=health_key,
            offering_id=offering_id,
            alias=alias,
        )

    stages.append(Stage(STAGE_POLICY, PASSED, "Policy admits it."))

    # --- Health State -----------------------------------------------
    record = health.get(health_key)
    key = alias if is_declared else offering_id
    if key is not None and key in report.unlisted:
        detail = _DROPPED_DETAIL[DROPPED_GONE][1]
        stages.append(Stage(STAGE_HEALTH, STOPPED, detail))
        return _stopped(
            query=query,
            stages=stages,
            stage=STAGE_HEALTH,
            kind=FAULT,
            detail=detail,
            health_key=health_key,
            offering_id=offering_id,
            alias=alias,
        )

    excluded = bool(record is not None and record.excluded)
    stages.append(
        Stage(
            STAGE_HEALTH,
            PASSED,
            (
                f"Excluded ({record.reason}), and still written to the config: "
                "an exclusion does not remove an Offering (ADR 0014)."
            )
            if excluded and record is not None
            else "Health State records no exclusion.",
        )
    )

    # --- Generated Config -------------------------------------------
    if key is None or key not in report.admitted:
        detail = (
            "Policy admits it and Health State does not Unlist it, yet it is "
            "absent from the Generated Config. The config on disk is older "
            "than this answer: run `deploy`."
        )
        stages.append(Stage(STAGE_CONFIG, STOPPED, detail))
        return _stopped(
            query=query,
            stages=stages,
            stage=STAGE_CONFIG,
            kind=FAULT,
            detail=detail,
            health_key=health_key,
            offering_id=offering_id,
            alias=alias,
        )

    stages.append(Stage(STAGE_CONFIG, PASSED, f"Written as `{alias}`."))

    # --- The running proxy -------------------------------------------
    if served_aliases is None:
        stages.append(
            Stage(
                STAGE_PROXY,
                UNKNOWN,
                proxy_note or "The proxy could not be asked, so this is unmeasured.",
            )
        )
        return Explanation(
            query=query,
            health_key=health_key,
            offering_id=offering_id,
            alias=alias,
            stages=tuple(stages),
            stopped_at=None,
            stop_kind=None,
            recommended=not excluded,
            notes=("The proxy stage is unknown, never absent.",),
        )

    if alias is not None and alias not in served_aliases:
        detail = (
            f"The Generated Config holds `{alias}` and the running proxy does "
            "not serve it. The proxy is serving an older generation: it "
            "reloads on a write, so a write may be pending."
        )
        stages.append(Stage(STAGE_PROXY, STOPPED, detail))
        return _stopped(
            query=query,
            stages=stages,
            stage=STAGE_PROXY,
            kind=FAULT,
            detail=detail,
            health_key=health_key,
            offering_id=offering_id,
            alias=alias,
        )

    stages.append(Stage(STAGE_PROXY, PASSED, "The proxy serves this Alias."))
    return Explanation(
        query=query,
        health_key=health_key,
        offering_id=offering_id,
        alias=alias,
        stages=tuple(stages),
        stopped_at=None,
        stop_kind=None,
        recommended=not excluded,
        notes=(
            ("Reachable, and not recommended: Health State records it Excluded.",)
            if excluded
            else ()
        ),
    )
