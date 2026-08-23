"""Build the Role entries of the Generated Config.

A Role is a Model Group a client calls by name — `role-coder` — whose
members are Aliases this tool already offers, ordered best-first. It
answers "give me the best coder you can reach right now" without the
client naming a model, and without an agent being trusted to remember
which model is currently drained.

The ranking is `guidance.derive`'s. This module calls it and reads the
result; it holds no scoring rule, no axis table and no headroom rule of
its own. A Role ladder and `guidance --for <axis>` therefore cannot
disagree, because there is one ranking and both read it.

Three properties follow from building the ladder HERE rather than in a
proxy hook, and each is the reason this module exists:

- A Role can only name an Alias that Selection admitted and Health
  permits. It cannot reach a withheld, Excluded or unadmitted Offering,
  so it can never make something callable that was not callable before.
- The ladder is rebuilt by the same run that writes the Generated
  Config, from the same Feed, Health State and Headroom Reading. No
  sidecar file, no second cron, and no window where the ladder describes
  a catalogue the config no longer holds.
- Each member carries the `litellm_params` of the Alias it names,
  copied, so a Role never states a credential or a base URL of its own.

WARNING: a Role reacts at the pace of the run that writes the config,
not per request. A subscription that drains between two ticks is caught
by litellm's own cooldown and retry down the ladder, never by this
module. Read `docs/` before adding a runtime hook on top: the ladder
below is what such a hook would have to agree with.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from litellm_maintainer.feed import Feed
from litellm_maintainer.guidance import Guidance, GuidanceError, derive
from litellm_maintainer.headroom import HeadroomState
from litellm_maintainer.plan import AliasAnnotation, PlanReport
from litellm_maintainer.policy import Policy, RoleRule
from litellm_maintainer.reduce import OfferingHealth


@dataclass(frozen=True)
class RoleLadder:
    """One Role's ordered Aliases, and what was left out of it."""

    role_name: str
    axis: str
    aliases: tuple[str, ...]
    # Why the ladder is short, or empty, in the operator's words. Empty
    # when every qualifying Route entered it.
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class RoleResult:
    """Every Role's `model_list` entries, plus what to report."""

    entries: tuple[dict[str, Any], ...] = ()
    annotations: dict[str, AliasAnnotation] | None = None
    ladders: tuple[RoleLadder, ...] = ()
    refusal: str | None = None


def _row_qualifies(row: Any, rule: RoleRule) -> bool:
    """State whether one ranked row may enter this Role's ladder.

    A row with no score on the axis is admitted only when the rule names
    no other rows — see `build_role_entries`. Capability filtering reads
    the Feed's own `capabilities`, so a model that gains or loses one
    joins or leaves the ladder on the next run with no edit here.
    """
    required = set(rule.require_capabilities)
    if required and not required.issubset(set(row.capabilities)):
        return False
    return True


def _ladder_for(
    guidance: Guidance, rule: RoleRule, *, alias_owner: set[str]
) -> RoleLadder:
    """Read one ranked answer into one Role's ordered Alias list.

    Rows arrive best-first on the axis and each row's Routes arrive
    cheapest-first, so walking both in order yields the ladder directly.

    A Route is admitted only when it is recommendable: `guidance` has
    already applied Health, exhaustion and Headroom to that flag, so
    this module never re-decides any of them. An Alias appears once, at
    its best position: a Canonical Model reachable two ways contributes
    its cheapest Route, and the second Route would only re-rank the same
    model against itself.
    """
    aliases: list[str] = []
    notes: list[str] = []
    skipped_capability = 0
    skipped_not_recommended = 0

    for row in guidance.rows:
        if not _row_qualifies(row, rule):
            skipped_capability += 1
            continue
        for route in row.routes:
            if route.alias not in alias_owner:
                # The Alias is not in this run's Generated Config. A Role
                # naming it would produce a Model Group litellm cannot
                # resolve, so it is left out rather than written.
                continue
            if not route.recommendable:
                skipped_not_recommended += 1
                continue
            if route.alias in aliases:
                continue
            aliases.append(route.alias)
            break
        if rule.limit is not None and len(aliases) >= rule.limit:
            break

    if skipped_capability:
        notes.append(
            f"{skipped_capability} model(s) left out: the Feed does not state "
            f"{', '.join(sorted(rule.require_capabilities))}"
        )
    if skipped_not_recommended:
        notes.append(
            f"{skipped_not_recommended} Route(s) left out as not recommended "
            "(Excluded, exhausted, or drained)"
        )
    if not aliases:
        notes.append(
            "no Route qualified, so this Role names nothing and is not written"
        )
    return RoleLadder(
        role_name="", axis=guidance.axis, aliases=tuple(aliases), notes=tuple(notes)
    )


def build_role_entries(
    *,
    feed: Feed,
    policy: Policy,
    health: dict[str, OfferingHealth],
    report: PlanReport,
    now: datetime,
    entries: list[dict[str, Any]],
    headroom_state: HeadroomState | None = None,
) -> RoleResult:
    """Build every Role's `model_list` entries from the ranking.

    `entries` is the Generated Config's `model_list` as `plan` built it.
    Each Role member copies the `litellm_params` of the Alias it names,
    so a Role states no credential of its own and cannot drift from the
    Alias it stands for.

    Refuse when a Role name collides with an Alias. litellm holds one
    Model Group per `model_name`, so a Role sharing a name with an Alias
    would silently make that Alias one member of a ladder — the calls a
    client meant for one model would land on several. `plan` refuses on
    an Alias collision for the same reason.

    Return an empty result when Policy names no Role, so a Policy
    without a `roles` block produces exactly the Generated Config this
    tool always produced.
    """
    if not policy.roles:
        return RoleResult()

    params_by_alias: dict[str, dict[str, Any]] = {}
    for entry in entries:
        alias = entry.get("model_name")
        if isinstance(alias, str) and "litellm_params" in entry:
            params_by_alias.setdefault(alias, entry["litellm_params"])

    collisions = sorted(set(policy.roles) & set(params_by_alias))
    if collisions:
        return RoleResult(
            refusal=(
                f"Role name collision on {collisions[0]!r}: an Alias already "
                "holds that name. litellm holds one Model Group per "
                "`model_name`, so the Role would absorb the Alias and a call "
                "meant for one model would reach several. Rename the Role, or "
                "the Alias through `naming`."
            )
        )

    role_entries: list[dict[str, Any]] = []
    annotations: dict[str, AliasAnnotation] = {}
    ladders: list[RoleLadder] = []

    for role_name in sorted(policy.roles):
        rule = policy.roles[role_name]
        try:
            guidance = derive(
                feed=feed,
                policy=policy,
                health=health,
                report=report,
                now=now,
                axis=rule.axis,
                prefer=rule.prefer,
                min_context=rule.min_context,
                headroom_state=headroom_state,
            )
        except GuidanceError as exc:
            # Policy validated `axis` and `prefer` on load, so reaching
            # here means the ranking itself refused. Report it rather
            # than writing a Role whose ladder nobody ranked.
            return RoleResult(refusal=f"Role {role_name!r}: {exc}")

        ladder = _ladder_for(guidance, rule, alias_owner=set(params_by_alias))
        ladder = RoleLadder(
            role_name=role_name,
            axis=ladder.axis,
            aliases=ladder.aliases,
            notes=ladder.notes,
        )
        ladders.append(ladder)
        if not ladder.aliases:
            continue

        for position, alias in enumerate(ladder.aliases, start=1):
            member: dict[str, Any] = {
                "model_name": role_name,
                # `order` is litellm's own ladder field
                # (`LiteLLMParamsTypedDict.order`): the Router tries a
                # lower `order` first and falls to the next on failure.
                # Copy the Alias's params so the Role never states a
                # credential, a base URL or a model string of its own.
                "litellm_params": {**params_by_alias[alias], "order": position},
                "model_info": {"id": f"{role_name}--{alias}", "role_member_of": alias},
            }
            role_entries.append(member)
        annotations[role_name] = AliasAnnotation(
            group="Roles",
            note=(
                f"ranked by {ladder.axis}; "
                f"{len(ladder.aliases)} rung(s), best first: "
                f"{', '.join(ladder.aliases)}"
            ),
        )

    return RoleResult(
        entries=tuple(role_entries),
        annotations=annotations,
        ladders=tuple(ladders),
    )
