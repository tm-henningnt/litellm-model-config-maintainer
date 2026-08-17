"""Reports every reason the system is not working.

`diagnose` is a pure function. It performs no file read, no network
call and no clock read: the caller reads Policy, the Feed, Health
State, the Feed Document's metadata and the environment, probes the
proxy, and reads the clock, then passes every value in. This is the
same seam `plan.py` and `pricing.py` use for their own pure checks, and
it is what makes `diagnose` testable without a filesystem or a running
proxy. Nothing in this module writes a file or calls a provider.

See CONTEXT.md for "Policy", "Feed", "Feed Document", "Health State"
and "Withheld". A `Check` names one thing that can go wrong; a
`Diagnosis` is every `Check` this run produced. `render_text` turns a
`Diagnosis` into the report an operator reads on a terminal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from litellm_maintainer.codexbar import CodexbarReading
from litellm_maintainer.headroom import HEADROOM_STALE_MULTIPLIER
from litellm_maintainer.entitlements import allowance_id_for_declared, allowance_id_for_provider
from litellm_maintainer.feed import Feed
from litellm_maintainer.fetch import staleness_warning
from litellm_maintainer.litellm_patches import PatchStatus
from litellm_maintainer.policy import Policy
from litellm_maintainer.reduce import OfferingHealth

_DEFAULT_MAXIMUM_AGE_HOURS = 24.0


@dataclass(frozen=True)
class Check:
    """One diagnostic result.

    `ok` states whether this check passed. `detail` names the
    condition found, in either case. `remedy` names the exact command
    or edit the operator makes to fix a failed check; it is `None` on
    a passing check.
    """

    name: str
    ok: bool
    detail: str
    remedy: str | None = None


@dataclass(frozen=True)
class Diagnosis:
    """Every `Check` one `diagnose` call produced."""

    checks: tuple[Check, ...]

    @property
    def ok(self) -> bool:
        """`True` only when every check passed."""
        return all(check.ok for check in self.checks)

    def as_dict(self) -> dict[str, Any]:
        """A JSON-serialisable rendering of this `Diagnosis`."""
        return {
            "ok": self.ok,
            "checks": [
                {
                    "name": check.name,
                    "ok": check.ok,
                    "detail": check.detail,
                    "remedy": check.remedy,
                }
                for check in self.checks
            ],
        }


def _policy_parses_check() -> Check:
    # The caller only reaches `diagnose` after `policy.parse_policy`
    # (or `load_policy`) already returned a `Policy` without raising
    # `PolicyError`. This check reports that fact; it never re-parses.
    return Check(
        name="policy.parses",
        ok=True,
        detail="Policy parsed without error.",
    )


def _credential_checks(policy: Policy, feed: Feed, environ: dict[str, str]) -> list[Check]:
    checks: list[Check] = []
    for provider_id in sorted(policy.providers):
        name = f"credential.{provider_id}"
        provider = feed.providers.get(provider_id)
        if provider is None:
            checks.append(
                Check(
                    name=name,
                    ok=False,
                    detail=f"policy.providers names {provider_id!r}, which the Feed does not publish.",
                    remedy="run fetch to refresh the Feed Document, or remove this provider from policy.providers",
                )
            )
            continue

        hint = provider.credential_hint
        if not hint:
            checks.append(
                Check(
                    name=name,
                    ok=True,
                    detail=f"the Feed states no credential_hint for {provider_id!r}.",
                )
            )
            continue

        if environ.get(hint):
            checks.append(
                Check(
                    name=name,
                    ok=True,
                    detail=f"{hint} is set in the environment.",
                )
            )
        else:
            checks.append(
                Check(
                    name=name,
                    ok=False,
                    detail=f"{hint} is not set in the environment.",
                    remedy=f"export {hint}, the credential variable the Feed names for {provider_id!r}",
                )
            )
    return checks


def _feed_document_age_check(policy: Policy, feed_document_metadata: dict[str, Any], now: datetime) -> Check:
    maximum_age_hours = (
        policy.feed.maximum_age_hours if policy.feed is not None else _DEFAULT_MAXIMUM_AGE_HOURS
    )
    generated_at = feed_document_metadata.get("generated_at")
    warning = staleness_warning(
        generated_at=generated_at, maximum_age_hours=maximum_age_hours, now=now
    )
    if warning is None:
        return Check(
            name="feed_document.age",
            ok=True,
            detail=f"the Feed Document is within the {maximum_age_hours:.0f}h threshold.",
        )
    return Check(
        name="feed_document.age",
        ok=False,
        detail=warning,
        remedy="run the fetch command to refresh the Feed Document",
    )


def _proxy_check(proxy_ok: bool) -> Check:
    if proxy_ok:
        return Check(name="proxy.reachable", ok=True, detail="the proxy answered.")
    return Check(
        name="proxy.reachable",
        ok=False,
        detail="the proxy did not answer.",
        remedy="start the litellm proxy, then rerun doctor",
    )


def _health_state_populated_check(health: dict[str, OfferingHealth]) -> Check:
    if health:
        return Check(
            name="health_state.populated",
            ok=True,
            detail=f"Health State holds {len(health)} record(s).",
        )
    return Check(
        name="health_state.populated",
        ok=False,
        detail="Health State is empty; no Probe has ever run.",
        remedy="run the probe command to populate Health State",
    )


def _probed_checks(policy: Policy, feed: Feed, health: dict[str, OfferingHealth]) -> list[Check]:
    """One check per provider whose Offerings a Probe can actually reach.

    Read the Prober's OWN worklist source, not every Offering the Feed
    publishes for the provider. A Withheld Offering is never probed —
    "only a human clears those" (CONTEXT.md, "Prober") — so demanding a
    Health State record for one asks for a Probe that never runs, and the
    remedy it prints ("run the probe command") cannot work.

    Measured 2026-07-28: `cline-pass` publishes 11 Offerings and Policy
    Withholds all 11, so this check failed with no action able to clear
    it. One permanent false failure hides the real ones behind it, which
    is the whole cost.
    """
    from litellm_maintainer.prober import _discovered_admitted

    probeable = set(_discovered_admitted(feed, policy))
    checks: list[Check] = []
    for provider_id in sorted(policy.providers):
        offering_ids = tuple(
            o.id for o in feed.offerings_for(provider_id) if o.id in probeable
        )
        if not offering_ids:
            # Nothing a Probe can reach. Either the Feed does not cover
            # this provider — the credential check above already names
            # that — or Policy admits none of its Offerings.
            continue
        name = f"health_state.probed.{provider_id}"
        if any(offering_id in health for offering_id in offering_ids):
            checks.append(
                Check(
                    name=name,
                    ok=True,
                    detail=f"at least one Offering of {provider_id!r} has a Health State record.",
                )
            )
        else:
            checks.append(
                Check(
                    name=name,
                    ok=False,
                    detail=f"no Offering of {provider_id!r} has a Health State record; no Probe has ever reached it.",
                    remedy="run the probe command to reach this provider's Offerings",
                )
            )
    return checks


def _litellm_patch_checks(patches: Sequence[PatchStatus]) -> list[Check]:
    """One check per local litellm patch.

    A patch whose marker was not read at all (`present is None`) passes:
    an operator who runs the proxy on another host has nothing here to
    read, and a check that cannot measure must not fail. The detail still
    states that nothing was read.
    """
    checks: list[Check] = []
    for patch in patches:
        name = f"litellm_patch.{patch.name}"
        if patch.present is False:
            checks.append(
                Check(name=name, ok=False, detail=patch.detail, remedy=patch.remedy)
            )
        else:
            checks.append(Check(name=name, ok=True, detail=patch.detail))
    return checks


def _withheld_checks(policy: Policy, feed: Feed) -> list[Check]:
    checks: list[Check] = []
    for offering_id in sorted(policy.withheld):
        name = f"withheld.{offering_id}"
        if feed.offering(offering_id) is not None:
            checks.append(
                Check(
                    name=name,
                    ok=True,
                    detail=f"{offering_id!r} is still published by the Feed.",
                )
            )
        else:
            checks.append(
                Check(
                    name=name,
                    ok=False,
                    detail=f"{offering_id!r} is Withheld, but the Feed no longer publishes it.",
                    remedy=f"remove {offering_id!r} from policy.withheld",
                )
            )
    return checks


def _reference_model_checks(policy: Policy, feed: Feed) -> list[Check]:
    """One check per Declared Offering naming a Reference Model.

    A Reference Model the Feed no longer publishes yields no score, and
    `guidance` only warns about it. The failure is silent in the place it
    matters: the row still appears, still answers, and simply sorts last
    as though the model were unrated. Report it here, where a stale Policy
    line is what the reader is looking for.
    """
    checks: list[Check] = []
    for declared in policy.declared:
        if declared.reference_model is None:
            continue
        name = f"declared.{declared.alias}.reference_model"
        if feed.offerings_for_canonical_model(declared.reference_model):
            checks.append(
                Check(
                    name=name,
                    ok=True,
                    detail=(
                        f"{declared.reference_model!r} is still published by the "
                        "Feed, so the row carries a score."
                    ),
                )
            )
        else:
            checks.append(
                Check(
                    name=name,
                    ok=False,
                    detail=(
                        f"{declared.alias!r} names Reference Model "
                        f"{declared.reference_model!r}, which the Feed does not "
                        "publish, so its row carries no score."
                    ),
                    remedy=(
                        f"check the Canonical Model id against the Feed, then "
                        f"correct or remove reference_model on {declared.alias!r}"
                    ),
                )
            )
    return checks


#: The callback a proxy must register for the Observation Journal to
#: receive anything at all.
JOURNAL_CALLBACK = "journal_failure_callback.observation_journal_callback"


def _journal_callback_checks(served_configs: dict[str, bool]) -> list[Check]:
    """One check per served proxy config, for the Journal callback.

    Warning: an unregistered callback is SILENT. The proxy serves every
    request normally, the maintainer runs normally, and the Observation
    Journal simply stays empty forever. Nothing else in the system can
    tell "no failures happened" from "no failures were recorded".

    `docs/gotchas.md` draws this lesson for `chatgpt_role_fix`: "a
    registered hook that matches nothing is a misconfiguration, never a
    valid state. That warning is what turns this class of fault from
    silent into obvious." The same holds for a hook that is not
    registered at all.

    Only the MAIN proxy must register it. A worker proxy is an
    implementation detail of one Offering, and its own `model_group`
    carries no seat identity: `claude-chatgpt1-gpt-5.6-sol` and
    `claude-chatgpt2-gpt-5.6-sol` both reach a worker that knows itself
    only as `claude-gpt-5.6-sol`. An entry written there names a key
    Health State does not hold, so `reduce` discards it, and it could
    not tell an exhausted seat from a healthy one anyway. The main
    proxy is an ordinary client of the worker: when the worker fails,
    the main proxy's own call fails and its hook records the
    seat-qualified Alias, which is the correct Health Key.

    `served_configs` maps a config path to `(registered, generated)`.
    `generated` says the file carries the Generator's own header, which
    is what identifies the main proxy's config: the maintainer writes
    that file and nothing else in the directory. A hand-written config
    is reported, never failed.

    The caller reads the files; this stays pure. An empty mapping
    produces no check, because the caller found no served config to
    judge -- a missing proxy directory is not a failed registration.
    """
    checks: list[Check] = []
    for path in sorted(served_configs):
        registered, generated = served_configs[path]
        name = f"journal.callback_registered[{path}]"

        if not generated:
            detail = (
                "hand-written config; it registers the Journal callback, which "
                "the main proxy already does."
                if registered
                else (
                    "hand-written config, so it records nothing. That is "
                    "correct: only the main proxy records, because only it "
                    "knows the Alias Health State is keyed by."
                )
            )
            checks.append(Check(name=name, ok=True, detail=detail))
            continue

        if registered:
            checks.append(
                Check(name=name, ok=True, detail="the proxy records failures to the Journal.")
            )
            continue

        checks.append(
            Check(
                name=name,
                ok=False,
                detail=(
                    "the main proxy's config registers no Observation Journal "
                    "callback, so every failure it serves goes unrecorded."
                ),
                remedy=(
                    f"Add '{JOURNAL_CALLBACK}' to Policy's "
                    "proxy_settings.litellm_settings.callbacks, then run "
                    "generate and deploy. Editing this file directly is lost "
                    "on the next run: the Generator overwrites it."
                ),
            )
        )
    return checks


def _tick_installed_check(tick_installed: bool | None, plist_path: str) -> Check:
    """Whether the launchd tick that runs the maintainer is installed.

    Nothing in this project runs on its own until this plist exists.
    An operator can hold a fully configured instance, a registered
    callback and a growing Journal, and still have no process that ever
    reads any of it.

    `tick_installed` of `None` means the caller could not look.
    """
    if tick_installed is None:
        return Check(
            name="schedule.tick_installed",
            ok=True,
            detail="not checked: the LaunchAgents directory could not be read.",
        )
    if tick_installed:
        return Check(
            name="schedule.tick_installed",
            ok=True,
            detail=f"the launchd tick is installed at {plist_path}.",
        )
    return Check(
        name="schedule.tick_installed",
        ok=False,
        detail=(
            "no launchd tick is installed, so nothing runs the maintainer on "
            "its own. Health State only changes when you run a command by hand."
        ),
        remedy=(
            "Run 'litellm-maintainer install', then the launchctl command it "
            "prints. Registering the job is a separate step; writing the plist "
            "alone starts nothing."
        ),
    )


# --- Headroom: a rotted mapping is a named finding -------------------------
#
# Every part of the Headroom capability degrades to the same symptom: no
# Headroom. That is indistinguishable from "this Allowance was never
# mapped", the normal state of most Allowances here. The four checks below
# turn a broken mapping into a named finding instead of a quiet return to
# silence. See CONTEXT.md, "Headroom", "Reading" and "Sub-allowance", and
# the headroom spec, decision 9.
#
# `headroom_readings` is a LIVE codexbar document -- the caller's own
# invocation of `policy.headroom.command` for exactly the mapped
# providers, made fresh for this diagnosis. Headroom State on disk cannot
# answer these checks: `refresh_headroom` keeps a stale Reading under its
# Allowance forever once one match ever succeeded, so a mapping that
# worked once and then silently rotted still shows a record on disk with
# nothing to say it stopped matching. Only asking codexbar again can tell.
#
# `None` covers every reason this run could not measure: Policy declares
# no source, the binary is not on the PATH, or the run failed. Each of
# those already produces its own Check (`_headroom_binary_check` for a
# missing binary, `_headroom_run_check` for a binary that ran and failed),
# and a check that cannot measure must not fail -- the same rule
# `_litellm_patch_checks` applies.


def _headroom_mapping_checks(
    policy: Policy,
    headroom_readings: tuple[CodexbarReading, ...] | None,
    stored_extra_window_ids: dict[str, frozenset[str]] | None = None,
) -> list[Check]:
    """Checks 1, 2 and 4: a declared source that matches no Reading, one
    that matches several, and a declared slot
    (`headroom.sources.<id>.windows`) the Reading no longer publishes.

    One `headroom.mapped.<allowance_id>` Check per declared source, plus
    one `headroom.window.<allowance_id>.<slot>` Check per declared slot
    (ticket 09). `_headroom_membership_checks` covers `members` (ticket
    10) separately, since those three checks need no live Reading at all.
    """
    sources = policy.headroom.sources
    if not sources or headroom_readings is None:
        return []

    by_source_key: dict[str, list[CodexbarReading]] = {}
    for reading in headroom_readings:
        by_source_key.setdefault(reading.source_key, []).append(reading)

    checks: list[Check] = []
    matched: dict[str, CodexbarReading] = {}
    for allowance_id in sorted(sources):
        source = sources[allowance_id]
        name = f"headroom.mapped.{allowance_id}"
        matches = by_source_key.get(source, [])
        if not matches:
            checks.append(
                Check(
                    name=name,
                    ok=False,
                    detail=(
                        f"{allowance_id!r} declares headroom_source {source!r}, "
                        "which matches no Reading codexbar published just now. "
                        "The tool may have renamed the provider, or the account "
                        "logged out."
                    ),
                    remedy=(
                        f"run '{policy.headroom.command} --format json' by hand, "
                        "read the identity it now states for this provider, and "
                        f"correct 'headroom.sources.{allowance_id}' in Policy"
                    ),
                )
            )
            continue
        if len(matches) > 1:
            checks.append(
                Check(
                    name=name,
                    ok=False,
                    detail=(
                        f"{allowance_id!r} declares headroom_source {source!r}, "
                        f"which matches {len(matches)} Readings codexbar "
                        "published just now. The key no longer discriminates "
                        "one account from another."
                    ),
                    remedy=(
                        f"name the account explicitly in "
                        f"'headroom.sources.{allowance_id}': "
                        "'codexbar:<providerID>/<accountEmail>'"
                    ),
                )
            )
            continue
        checks.append(
            Check(
                name=name,
                ok=True,
                detail=f"{allowance_id!r} matches exactly one Reading.",
            )
        )
        matched[allowance_id] = matches[0]

    # Check 4a: a declared slot (`headroom.sources.<id>.windows`) the
    # Reading no longer publishes at all. A rename or a dropped field
    # reads identically here: `getattr(reading, slot)` is `None` either
    # way, and both mean the operator's mapping now measures nothing.
    for allowance_id in sorted(policy.headroom.source_windows):
        windows = policy.headroom.source_windows[allowance_id]
        reading = matched.get(allowance_id)
        if reading is None:
            # Either this Allowance names no headroom_source, or its own
            # mapping already failed above; that Check names the root
            # cause, and adding a second failure here would only repeat it.
            continue
        for slot in sorted(windows):
            sub_id = windows[slot]
            name = f"headroom.window.{allowance_id}.{slot}"
            if getattr(reading, slot) is not None:
                checks.append(
                    Check(
                        name=name,
                        ok=True,
                        detail=f"{allowance_id!r}'s {slot!r} slot ({sub_id!r}) is still published.",
                    )
                )
            else:
                checks.append(
                    Check(
                        name=name,
                        ok=False,
                        detail=(
                            f"{allowance_id!r} declares 'windows.{slot}' as {sub_id!r}, "
                            f"which codexbar's Reading for {allowance_id!r} no longer "
                            "publishes."
                        ),
                        remedy=(
                            f"run '{policy.headroom.command} --provider <id> --format json' "
                            f"by hand, confirm the '{slot}' slot still exists, and correct "
                            f"'headroom.sources.{allowance_id}.windows.{slot}' in Policy"
                        ),
                    )
                )

    # Check 4b: a `members` key that reaches no window at all. A key names
    # either a slot id `windows` declares, or a codexbar `extraRateWindows`
    # id -- `route_binding_window` resolves an extra window first and a
    # declared slot second, so both forms are legal in Policy.
    #
    # Policy cannot reject the second form, because only a Reading knows
    # which extra windows exist. Measured 2026-07-28: codexbar published
    # `claude-weekly-scoped-all-model` at 18:48Z and had dropped it by
    # 20:52Z. A parse failure there would stop the Generator for every
    # provider over one vendor release, so the gap is reported here instead.
    for allowance_id in sorted(policy.headroom.source_members):
        reading = matched.get(allowance_id)
        if reading is None:
            continue
        declared_slot_ids = set(policy.headroom.source_windows.get(allowance_id, {}).values())
        extra_window_ids = {window.id for window in reading.extra_windows}
        stored_ids = (stored_extra_window_ids or {}).get(allowance_id, frozenset())
        for member_key in sorted(policy.headroom.source_members[allowance_id]):
            if member_key in declared_slot_ids or member_key in extra_window_ids:
                continue
            if member_key in stored_ids:
                # The live Reading omits this window, and a Reading already
                # on disk publishes it. codexbar drops an extra window and
                # restores it between consecutive calls -- measured
                # 2026-07-29, `claude-weekly-scoped-fable` was present,
                # absent and present again across three calls one minute
                # apart, with no Policy change and no vendor release.
                #
                # A mapping this check failed on that flap would send the
                # operator to correct a line that is already correct. So
                # report the flap and pass: the window is reachable, and a
                # Reading that catches it fills the Headroom in.
                checks.append(
                    Check(
                        name=f"headroom.member.unreachable.{allowance_id}.{member_key}",
                        ok=True,
                        detail=(
                            f"{member_key!r} is missing from the Reading codexbar "
                            "published just now, and Headroom State holds one that "
                            "publishes it. The mapping is correct; codexbar drops "
                            "this window intermittently."
                        ),
                    )
                )
                continue
            checks.append(
                Check(
                    name=f"headroom.member.unreachable.{allowance_id}.{member_key}",
                    ok=False,
                    detail=(
                        f"{allowance_id!r} names members under {member_key!r}, which is "
                        f"neither a slot 'headroom.sources.{allowance_id}.windows' "
                        "declares nor an extra window codexbar's Reading publishes. "
                        "Every Offering listed under it reads no Headroom."
                    ),
                    remedy=(
                        f"run '{policy.headroom.command} --provider <id> --format json' "
                        "by hand and read the ids under 'extraRateWindows', or declare "
                        f"{member_key!r} under "
                        f"'headroom.sources.{allowance_id}.windows'"
                    ),
                )
            )

    return checks


# --- Allowances: a Tier stated for an Allowance nothing reaches -----------
#
# `policy.allowances` states operator facts about an Allowance itself (today,
# only `tier` — CONTEXT.md, "Tier"). The one shape this can rot into that
# `parse_policy` cannot refuse: an `allowance_id` well-formed enough to parse,
# but naming no Allowance any Offering, Discovered or Declared, actually
# reaches. A typo or a removed provider produces exactly this, silently,
# since a Tier is never read back against anything live.
#
# Static, like `_headroom_membership_checks`: Policy and the Feed alone, no
# codexbar Reading needed.


def _allowances_checks(policy: Policy, feed: Feed) -> list[Check]:
    """Check: an `allowances` entry naming an `allowance_id` no Offering
    reaches.

    Reachable Allowance ids come from every Offering the Feed publishes
    (whether admitted, withheld or excluded — a Tier describes the
    Allowance, not today's Selection) plus every Declared Offering. Fires
    per unreachable entry; passes per entry that is reached.

    Also checks the reverse: an Allowance that publishes a Headroom and
    that `allowances` names nowhere. So a Policy with headroom sources and
    NO `allowances` block at all still runs -- that is the loudest case of
    the gap, not a reason to skip.
    """
    if not policy.allowances and not policy.headroom.sources:
        return []

    reachable = {allowance_id_for_provider(o.provider_id) for o in feed.offerings}
    reachable |= {allowance_id_for_declared(d) for d in policy.declared}

    checks: list[Check] = []
    for allowance_id in sorted(policy.allowances):
        name = f"allowances.{allowance_id}"
        if allowance_id in reachable:
            checks.append(
                Check(
                    name=name,
                    ok=True,
                    detail=f"{allowance_id!r} is reached by at least one Offering.",
                )
            )
        else:
            checks.append(
                Check(
                    name=name,
                    ok=False,
                    detail=(
                        f"'allowances' names {allowance_id!r}, which no Offering — "
                        "Discovered or Declared — reaches."
                    ),
                    remedy=(
                        f"check {allowance_id!r} against the Feed and against "
                        "'declared', then correct or remove it from 'allowances'"
                    ),
                )
            )

    # Check: an Allowance that publishes a Headroom and states no Tier.
    #
    # A Headroom states a SHARE, and a share is scale-free. A caller is
    # told to read `tier` beside it, and cannot where Policy states none.
    # Reported 2026-07-29: 24 of the 40 Routes carrying a Reading carried
    # no `tier`, and 20 of those 24 were one Allowance.
    #
    # A PRESENT `allowances` entry silences this, whether or not it states
    # `tier`. That is the distinction the report asked for, and neither a
    # Route nor `entitlements` can express it: an entry with no `tier` says
    # the operator looked and found no Tier to state, and no entry at all
    # says nobody has looked. The same rule an empty `members` list follows
    # -- silence stated is not silence by default.
    for allowance_id in sorted(policy.headroom.sources):
        if allowance_id in policy.allowances:
            continue
        if allowance_id not in reachable:
            # No Offering reaches it, so no Route can report its Tier and
            # stating one would describe nothing. The mapping itself is
            # what is wrong here, and `headroom.mapped.<id>` says so.
            continue
        checks.append(
            Check(
                name=f"allowances.tier_unstated.{allowance_id}",
                ok=False,
                detail=(
                    f"{allowance_id!r} publishes a Headroom and 'allowances' names "
                    "it nowhere, so every Route on it reports 'tier: null'. A share "
                    "is scale-free: 90% of one subscription level is not 90% of "
                    "another, and a caller cannot tell which it is reading."
                ),
                remedy=(
                    f"add an 'allowances.{allowance_id}' entry stating 'tier'. "
                    "Where the source has no subscription level to state, add the "
                    "entry with no 'tier' key: a present entry states that you "
                    "looked, and silences this check"
                ),
            )
        )
    return checks


# --- Headroom: `all_accounts_providers` is a static Policy fact too --------
#
# ticket 11 lets one codexbar provider id hold two accounts. Both ways this
# can rot show up in Policy alone, with no live Reading needed: a marker
# naming a provider no `sources` entry reaches, and two `sources` entries
# sharing a `providerID` with no marker naming it (the shape ADR 0009
# refuses -- one of the two keys then matches nothing, for a reason
# invisible from outside).


def _headroom_multi_account_checks(policy: Policy) -> list[Check]:
    """Checks for ticket 11: `all_accounts_providers` against `sources`.

    Both checks read Policy alone. Static, like `_headroom_membership_checks`
    below, and for the same reason: the failure shape is visible without
    asking codexbar anything.
    """
    from litellm_maintainer.headroom import provider_id_from_source

    sources = policy.headroom.sources
    all_accounts = policy.headroom.all_accounts_providers
    if not sources and not all_accounts:
        return []

    mapped_provider_ids = {provider_id_from_source(source) for source in sources.values()}
    checks: list[Check] = []

    # Check: a provider named in `all_accounts_providers` that no source
    # entry reaches at all -- the marker names a provider nothing maps.
    for provider_id in sorted(all_accounts):
        name = f"headroom.all_accounts.unreachable.{provider_id}"
        if provider_id in mapped_provider_ids:
            checks.append(
                Check(
                    name=name,
                    ok=True,
                    detail=f"{provider_id!r} is reached by at least one 'headroom.sources' entry.",
                )
            )
        else:
            checks.append(
                Check(
                    name=name,
                    ok=False,
                    detail=(
                        f"'headroom.all_accounts_providers' names {provider_id!r}, which no "
                        "'headroom.sources' entry reaches."
                    ),
                    remedy=(
                        f"map a source to {provider_id!r} under 'headroom.sources', or remove "
                        f"{provider_id!r} from 'headroom.all_accounts_providers'"
                    ),
                )
            )

    # Check: two source entries share a providerID and that provider is NOT
    # named in `all_accounts_providers`. Without the marker, a plain call
    # returns whichever account codexbar treats as default, so one of the
    # two keys matches no Reading for a reason invisible from here.
    allowance_ids_by_provider: dict[str, list[str]] = {}
    for allowance_id, source in sources.items():
        allowance_ids_by_provider.setdefault(provider_id_from_source(source), []).append(
            allowance_id
        )

    for provider_id in sorted(allowance_ids_by_provider):
        allowance_ids = sorted(allowance_ids_by_provider[provider_id])
        if len(allowance_ids) <= 1:
            continue
        name = f"headroom.all_accounts.unmarked.{provider_id}"
        if provider_id in all_accounts:
            checks.append(
                Check(
                    name=name,
                    ok=True,
                    detail=(
                        f"{provider_id!r} maps {len(allowance_ids)} Allowances "
                        f"({', '.join(allowance_ids)}) and is named in "
                        "'headroom.all_accounts_providers'."
                    ),
                )
            )
        else:
            checks.append(
                Check(
                    name=name,
                    ok=False,
                    detail=(
                        f"{', '.join(allowance_ids)} both declare a 'headroom.sources' entry for "
                        f"{provider_id!r}, but 'headroom.all_accounts_providers' does not name "
                        "it. A plain call returns one account, and one of these two keys "
                        "matches nothing for a reason invisible from here."
                    ),
                    remedy=f"add {provider_id!r} to 'headroom.all_accounts_providers'",
                )
            )

    return checks


# --- Headroom: a member is a static Policy+Feed fact, no codexbar needed ---
#
# ticket 10 lets `headroom.sources.<id>.members` name which Health Keys draw
# on each declared slot. Every way this can rot is visible from Policy and
# the Feed alone -- an admitted Offering nobody assigned, a slot with no
# members, or a member naming no known Health Key -- so these three checks
# take no live codexbar Reading at all, unlike `_headroom_mapping_checks`
# above. See CONTEXT.md, "Sub-allowance" and "Health Key".


def _headroom_membership_checks(policy: Policy, feed: Feed) -> list[Check]:
    """Checks for ticket 10: an unclaimed admitted Health Key, a declared
    Sub-allowance with no members, and a member naming no known Health Key.

    Only an Allowance naming at least one slot in
    `headroom.sources.<id>.windows` gets any of these three: an Allowance
    mapped through a plain string, or through a mapping naming no
    `windows`, has no Sub-allowance to misconfigure.
    """
    from litellm_maintainer.prober import _discovered_admitted

    windows_by_allowance = policy.headroom.source_windows
    members_by_allowance = policy.headroom.source_members
    if not windows_by_allowance:
        return []

    admitted_discovered = set(_discovered_admitted(feed, policy))
    admitted_by_allowance: dict[str, set[str]] = {}
    for offering_id in admitted_discovered:
        offering = feed.offering(offering_id)
        if offering is None:
            continue
        allowance_id = allowance_id_for_provider(offering.provider_id)
        admitted_by_allowance.setdefault(allowance_id, set()).add(offering_id)
    for declared in policy.declared:
        if declared.variant_of is not None:
            # Shares its sibling's Health Key (ADR 0007); it names no
            # separate one of its own.
            continue
        allowance_id = allowance_id_for_declared(declared)
        admitted_by_allowance.setdefault(allowance_id, set()).add(declared.health_key)

    checks: list[Check] = []

    # Check: an admitted Health Key no Sub-allowance claims. Fires even
    # when `members` is absent entirely -- silence must not read as "every
    # Health Key is already assigned".
    for allowance_id in sorted(windows_by_allowance):
        claimed: set[str] = set()
        for health_keys in members_by_allowance.get(allowance_id, {}).values():
            claimed.update(health_keys)
        # A key the operator states draws on no published window counts as
        # accounted for. Stating it is the honest alternative to listing it
        # under a slot that does not measure it.
        claimed.update(policy.headroom.source_unmeasured.get(allowance_id, ()))
        unclaimed = sorted(admitted_by_allowance.get(allowance_id, set()) - claimed)
        name = f"headroom.member.unclaimed.{allowance_id}"
        if unclaimed:
            checks.append(
                Check(
                    name=name,
                    ok=False,
                    detail=(
                        f"{allowance_id!r} declares one or more Sub-allowance slots, "
                        f"but 'members' claims none of: {', '.join(unclaimed)}. Nobody "
                        "knows what these draw on."
                    ),
                    remedy=(
                        f"run '{policy.headroom.command} --provider <id>', read the "
                        f"labels, and add each Health Key to "
                        f"'headroom.sources.{allowance_id}.members' under the slot it "
                        "measures"
                    ),
                )
            )
        else:
            checks.append(
                Check(
                    name=name,
                    ok=True,
                    detail=f"every admitted Health Key on {allowance_id!r} is claimed.",
                )
            )

    # Check: a declared Sub-allowance (a slot named in `windows`) with no
    # members at all.
    #
    # An ABSENT slot key fails: nobody has assigned this window yet. A slot
    # key present with an EMPTY list passes: the operator states that
    # nothing draws on it. The two look alike and mean opposite things.
    #
    # Measured 2026-07-29 on the operator's Gemini free plan: its `primary`
    # slot measures Pro, the free plan includes no Pro, and no Pro model is
    # admitted. Dropping the slot from `windows` was the only other way to
    # pass, and it is the wrong one -- an undeclared slot rejoins the
    # parent's worst-of computation, and Pro reads 100% used, so the whole
    # Allowance would report exhausted while Flash and Flash Lite sit
    # untouched. Declaring the slot empty keeps Pro out of that computation
    # and keeps a truthful home for a Pro model admitted later.
    for allowance_id in sorted(windows_by_allowance):
        declared_slot_ids = set(windows_by_allowance[allowance_id].values())
        members = members_by_allowance.get(allowance_id, {})
        for slot_id in sorted(declared_slot_ids):
            name = f"headroom.member.empty.{allowance_id}.{slot_id}"
            if slot_id in members and not members[slot_id]:
                checks.append(
                    Check(
                        name=name,
                        ok=True,
                        detail=(
                            f"{allowance_id!r} declares the Sub-allowance {slot_id!r} "
                            "with an empty member list: nothing admitted draws on it."
                        ),
                    )
                )
            elif members.get(slot_id):
                checks.append(
                    Check(
                        name=name,
                        ok=True,
                        detail=f"{slot_id!r} on {allowance_id!r} names at least one member.",
                    )
                )
            else:
                checks.append(
                    Check(
                        name=name,
                        ok=False,
                        detail=(
                            f"{allowance_id!r} declares the Sub-allowance {slot_id!r}, "
                            "but 'members' names no Health Key for it."
                        ),
                        remedy=(
                            f"run '{policy.headroom.command} --provider <id>', read the "
                            f"labels, and add each Health Key that draws on {slot_id!r} "
                            f"to 'headroom.sources.{allowance_id}.members.{slot_id}'"
                        ),
                    )
                )

    # Check: a member naming no known Health Key -- a typo, or a model the
    # Feed dropped.
    known_health_keys = {o.id for o in feed.offerings} | {
        d.health_key for d in policy.declared
    }
    for allowance_id in sorted(members_by_allowance):
        for slot_id, health_keys in sorted(members_by_allowance[allowance_id].items()):
            for health_key in health_keys:
                name = f"headroom.member.unknown.{allowance_id}.{slot_id}.{health_key}"
                if health_key in known_health_keys:
                    checks.append(
                        Check(
                            name=name,
                            ok=True,
                            detail=f"{health_key!r} is a known Health Key.",
                        )
                    )
                else:
                    checks.append(
                        Check(
                            name=name,
                            ok=False,
                            detail=(
                                f"'headroom.sources.{allowance_id}.members.{slot_id}' "
                                f"names {health_key!r}, which matches no Offering the "
                                "Feed publishes and no Declared Offering's Alias. It "
                                "may be a typo, or a model the Feed dropped."
                            ),
                            remedy=(
                                f"check {health_key!r} against the Feed and against "
                                "'declared', then correct or remove it from "
                                f"'headroom.sources.{allowance_id}.members.{slot_id}'"
                            ),
                        )
                    )

    # Check: an `unmeasured` entry naming no known Health Key. It rots the
    # same way a `members` entry does, and it rots more quietly: the key it
    # names publishes nothing either way, so a typo here reads exactly like
    # a correct line until the real key fires `member.unclaimed`.
    for allowance_id in sorted(policy.headroom.source_unmeasured):
        for health_key in policy.headroom.source_unmeasured[allowance_id]:
            name = f"headroom.unmeasured.unknown.{allowance_id}.{health_key}"
            if health_key in known_health_keys:
                checks.append(
                    Check(
                        name=name,
                        ok=True,
                        detail=f"{health_key!r} is a known Health Key.",
                    )
                )
            else:
                checks.append(
                    Check(
                        name=name,
                        ok=False,
                        detail=(
                            f"'headroom.sources.{allowance_id}.unmeasured' names "
                            f"{health_key!r}, which matches no Offering the Feed "
                            "publishes and no Declared Offering's Alias. It may be a "
                            "typo, or a model the Feed dropped."
                        ),
                        remedy=(
                            f"check {health_key!r} against the Feed and against "
                            "'declared', then correct or remove it from "
                            f"'headroom.sources.{allowance_id}.unmeasured'"
                        ),
                    )
                )
    return checks


def _headroom_staleness_checks(policy: Policy, warnings: tuple[str, ...]) -> list[Check]:
    """One Check per mapped Allowance whose Headroom stopped refreshing.

    `warnings` comes from `headroom.headroom_source_warnings`, the same
    function `guidance` and `entitlements` already publish. This is a
    second RENDERING of that answer, never a second derivation of it: a
    `doctor` check that could disagree with the warning printed beside a
    figure would be worse than no check at all.

    It exists because the two surfaces answer different questions. A
    warning travels beside an answer, so a caller reading `guidance` sees
    it. `doctor` is what an operator runs to ask "is this instance
    healthy", and a refresh job that stopped is exactly that question.
    Measured 2026-07-29: the job was never registered, Headroom State sat
    4.9 hours stale, every figure kept publishing, and `doctor` exited 0.

    Reads no clock and no file of its own, so it stays pure.
    """
    if not policy.headroom.sources:
        return []
    if not warnings:
        return [
            Check(
                name="headroom.refresh_current",
                ok=True,
                detail=(
                    f"every one of the {len(policy.headroom.sources)} mapped "
                    "Allowances has a Headroom refreshed within "
                    f"{HEADROOM_STALE_MULTIPLIER} refresh intervals."
                ),
            )
        ]
    return [
        Check(
            name="headroom.refresh_current",
            ok=False,
            detail="; ".join(warnings),
            remedy=(
                "check the refresh job is loaded with "
                "'launchctl list | grep headroom-refresh'. Install it with "
                "'litellm-maintainer headroom install', then run the "
                "'launchctl load' command it prints. Read "
                "'state/headroom-refresh.err.log' when the job is loaded and "
                "the Reading is still stale"
            ),
        )
    ]


def _draw_notes_checks(policy: Policy, feed: Feed) -> list[Check]:
    """Check: a `draw_notes` key naming no known Health Key.

    It rots exactly the way a `members` key does, and it rots more
    quietly: a note on a key nothing matches publishes nothing, and the
    Route it was written for goes on publishing `draw_note: null`. So the
    operator sees the line in Policy, believes a caller reads it, and no
    caller ever does.

    Static: Policy and the Feed alone, no live Reading needed.
    """
    if not policy.draw_notes:
        return []
    known = {o.id for o in feed.offerings} | {d.health_key for d in policy.declared}
    checks: list[Check] = []
    for health_key in sorted(policy.draw_notes):
        name = f"draw_notes.unknown.{health_key}"
        if health_key in known:
            checks.append(
                Check(name=name, ok=True, detail=f"{health_key!r} is a known Health Key.")
            )
        else:
            checks.append(
                Check(
                    name=name,
                    ok=False,
                    detail=(
                        f"'draw_notes' names {health_key!r}, which matches no Offering "
                        "the Feed publishes and no Declared Offering's Alias. Every "
                        "Route it was written for publishes 'draw_note: null'."
                    ),
                    remedy=(
                        f"check {health_key!r} against the Feed and against 'declared', "
                        "then correct or remove it from 'draw_notes'"
                    ),
                )
            )
    return checks


def _headroom_run_check(policy: Policy, run_error: str | None) -> Check | None:
    """Check: whether the LIVE codexbar run made for the checks above
    actually answered.

    `None` when Policy declares no headroom source (the capability is
    off, and there is nothing to check) or the run succeeded (`run_error`
    is `None`, and `_headroom_mapping_checks` already reports what that
    run found). A missing binary is `_headroom_binary_check`'s Check, not
    this one -- reporting the same absence under two names would be
    noise, not a second finding.

    Before this check existed, a non-zero exit, a timeout, or output
    codexbar's own parser could not read all made `cli._headroom_readings`
    return `None` with no `run_error` of its own, which
    `_headroom_mapping_checks` read exactly like "Policy declares no
    headroom source": no Check, no failure, `doctor` exits 0. Codexbar
    could then fail on every real invocation with nothing here to say so,
    though this module's own note above claims "each of those already
    produces its own Check" -- true for a missing binary, false for a
    binary that runs and fails.
    """
    if not policy.headroom.sources or run_error is None:
        return None
    return Check(
        name="headroom.readings",
        ok=False,
        detail=f"running {policy.headroom.command!r} to check the mapping failed: {run_error}",
        remedy=(
            f"run '{policy.headroom.command} --format json' by hand and confirm it answers "
            "within 'headroom.timeout_seconds'"
        ),
    )


def _headroom_binary_check(policy: Policy, binary_present: bool) -> Check | None:
    """Check 3: whether Policy's `headroom.command` binary is on the PATH.

    `None` when Policy declares no headroom source at all: the capability
    is off, and silence is the correct output for it.
    """
    if not policy.headroom.sources:
        return None
    command = policy.headroom.command
    if binary_present:
        return Check(
            name="headroom.binary",
            ok=True,
            detail=f"{command!r} is on the PATH.",
        )
    return Check(
        name="headroom.binary",
        ok=False,
        detail=f"{command!r} is not on the PATH, so 'headroom refresh' cannot run.",
        remedy=f"install {command!r}, or correct 'headroom.command' in Policy",
    )


def _headroom_interval_check(
    policy: Policy,
    installed_interval_seconds: int | None,
    plist_path: str,
) -> Check | None:
    """Check 5: whether the installed refresh job's interval matches Policy.

    `headroom refresh`'s launchd job bakes `headroom.interval_minutes`
    into `StartInterval` at install time (`schedule.build_headroom_plist_spec`),
    because the job has no gate of its own to re-read Policy against on
    every tick. An operator who edits the interval and does not run
    `headroom install` again keeps the old cadence with no symptom.

    `None` when Policy declares no source at all, or the caller could not
    read a `StartInterval` -- no job is installed, or the LaunchAgents
    directory could not be read. Both read the same as `_tick_installed_check`
    reads a `None` `tick_installed`: not checked, not failed.
    """
    if not policy.headroom.sources:
        return None
    if installed_interval_seconds is None:
        return Check(
            name="headroom.refresh_interval",
            ok=True,
            detail=(
                "not checked: no headroom-refresh job is installed, or the "
                "LaunchAgents directory could not be read."
            ),
        )
    expected_seconds = policy.headroom.interval_minutes * 60
    if installed_interval_seconds == expected_seconds:
        return Check(
            name="headroom.refresh_interval",
            ok=True,
            detail=(
                f"the installed job at {plist_path} ticks every "
                f"{policy.headroom.interval_minutes} minutes, matching Policy."
            ),
        )
    installed_minutes = installed_interval_seconds / 60
    return Check(
        name="headroom.refresh_interval",
        ok=False,
        detail=(
            f"the installed job at {plist_path} ticks every "
            f"{installed_minutes:g} minutes, but Policy's "
            f"'headroom.interval_minutes' now states "
            f"{policy.headroom.interval_minutes}. The job bakes its interval in "
            "at install time and does not re-read Policy on its own."
        ),
        remedy="run 'litellm-maintainer headroom install' again, then reload the job",
    )


def diagnose(
    *,
    policy: Policy,
    feed: Feed,
    health: dict[str, OfferingHealth],
    feed_document_metadata: dict[str, Any],
    environ: dict[str, str],
    proxy_ok: bool,
    now: datetime,
    litellm_patches: Sequence[PatchStatus] = (),
    served_configs: dict[str, tuple[bool, bool]] | None = None,
    tick_installed: bool | None = None,
    tick_plist_path: str = "",
    headroom_readings: tuple[CodexbarReading, ...] | None = None,
    headroom_run_error: str | None = None,
    headroom_binary_present: bool = False,
    headroom_installed_interval_seconds: int | None = None,
    headroom_plist_path: str = "",
    headroom_stored_extra_window_ids: dict[str, frozenset[str]] | None = None,
    headroom_staleness_warnings: tuple[str, ...] = (),
) -> Diagnosis:
    """Run every check and return the `Diagnosis`.

    Pure: every value this needs is a parameter. `policy` is an already
    parsed `Policy`; `feed` an already parsed `Feed`; `health` the
    `offerings` mapping of an already read Health State
    (`health.read_health(...).offerings`); `feed_document_metadata` the
    result of `fetch.read_feed_document_metadata`; `environ` the
    process environment (or a fake mapping in a test); `proxy_ok`
    whether a caller-run smoke probe reached the proxy; `now` the
    current time.

    Checks, in order:

    1. Policy parses (the caller already parsed it).
    2. One check per provider `policy.providers` names, for whether its
       Feed `credential_hint` variable is set in `environ`.
    3. The Feed Document's age against `policy.feed.maximum_age_hours`.
    4. Whether the proxy answered.
    5. Whether Health State holds any record at all.
    6. One check per provider `policy.providers` names that has no
       Health State record for any of its Offerings.
    7. One check per `policy.withheld` entry naming an Offering the
       Feed no longer publishes.
    8. One check per local litellm patch in `litellm_patches`, as
       `litellm_patches.inspect_patches` reports it. Pass `()` to skip
       them; a patch the caller could not read passes with a detail
       saying so.
    9. One check per served proxy config in `served_configs`, for
       whether it registers the Observation Journal callback. Pass an
       empty mapping to skip them.
    10. Whether the launchd tick is installed. Pass `None` for
        `tick_installed` when the caller could not look.
    11. One check per declared `headroom.sources` entry, for whether it
        matches exactly one Reading in `headroom_readings` -- a LIVE
        codexbar document the caller reads fresh for this diagnosis, never
        Headroom State on disk (see the module note above
        `_headroom_mapping_checks`). Pass `None` to skip these: Policy
        declares no source, the binary is missing, or the run failed.
    12. One check per slot named in a `headroom.sources.<id>.windows`
        mapping (ticket 09), for whether `headroom_readings` still
        publishes that slot at all.
    13. Whether the live codexbar run made for checks 11 and 12 succeeded
        at all (`headroom_run_error`; `None` on success). A Policy
        declaring a source whose run raised -- a non-zero exit, a timeout,
        unparsable output -- gets its own failed Check here,
        `headroom.readings`, distinct from a missing binary (check 15)
        and from "the capability is off".
    14. Whether Policy's `headroom.command` binary is on the PATH
        (`headroom_binary_present`).
    15. Whether the installed headroom-refresh job's `StartInterval`
        matches Policy's `headroom.interval_minutes`
        (`headroom_installed_interval_seconds`). Pass `None` when no job
        is installed, or the caller could not read the LaunchAgents
        directory.
    16. Three checks per Allowance naming a Sub-allowance slot (ticket
        10): an admitted Health Key no `members` entry claims, a declared
        slot with no members at all, and a member naming no known Health
        Key. Static -- Policy and the Feed alone, no live codexbar
        Reading -- so these run even when `headroom_readings` is `None`.
    17. Two checks for `headroom.all_accounts_providers` (ticket 11): a
        marker naming a provider no `sources` entry reaches, and two
        `sources` entries sharing a `providerID` with no marker naming
        it. Static, like check 16.
    18. One check per `allowances` entry, for whether its `allowance_id`
        is reached by at least one Offering, Discovered or Declared.
        Static, like checks 16 and 17.

    Checks 9 and 10 cover the two ways this system can be fully
    configured and still do nothing: no writer for the Journal, and no
    process that reads it. Checks 11 through 16 cover the ways the
    Headroom mapping can rot silently: every part of that capability
    degrades to the same symptom, no Headroom, which reads exactly like
    "this Allowance was never mapped" (headroom spec, and CONTEXT.md,
    "Headroom"). A Policy naming no `headroom` source produces none of
    checks 11-16, at all: silence is the correct output for a capability
    that is switched off.

    Every failed check carries a non-`None` `remedy`.
    """
    checks: list[Check] = [_policy_parses_check()]
    checks.extend(_credential_checks(policy, feed, environ))
    checks.append(_feed_document_age_check(policy, feed_document_metadata, now))
    checks.append(_proxy_check(proxy_ok))
    checks.append(_health_state_populated_check(health))
    checks.extend(_probed_checks(policy, feed, health))
    checks.extend(_withheld_checks(policy, feed))
    checks.extend(_reference_model_checks(policy, feed))
    checks.extend(_litellm_patch_checks(litellm_patches))
    checks.extend(_journal_callback_checks(served_configs or {}))
    checks.append(_tick_installed_check(tick_installed, tick_plist_path))
    checks.extend(
        _headroom_mapping_checks(policy, headroom_readings, headroom_stored_extra_window_ids)
    )
    checks.extend(_headroom_staleness_checks(policy, headroom_staleness_warnings))
    checks.extend(_draw_notes_checks(policy, feed))
    run_check = _headroom_run_check(policy, headroom_run_error)
    if run_check is not None:
        checks.append(run_check)
    binary_check = _headroom_binary_check(policy, headroom_binary_present)
    if binary_check is not None:
        checks.append(binary_check)
    interval_check = _headroom_interval_check(
        policy, headroom_installed_interval_seconds, headroom_plist_path
    )
    if interval_check is not None:
        checks.append(interval_check)
    checks.extend(_headroom_membership_checks(policy, feed))
    checks.extend(_headroom_multi_account_checks(policy))
    checks.extend(_allowances_checks(policy, feed))
    return Diagnosis(checks=tuple(checks))


def render_text(diagnosis: Diagnosis) -> str:
    """Render `diagnosis` as a plain-text report, one line per check.

    Each line starts with a marker: `[OK]` or `[FAIL]`. A failed line
    ends with its remedy. Runs on an empty `Diagnosis` (no checks at
    all) and returns a one-line report rather than raising.
    """
    if not diagnosis.checks:
        return "No checks ran.\n"

    lines: list[str] = []
    overall = "OK" if diagnosis.ok else "FAIL"
    lines.append(f"Overall: [{overall}]")
    for check in diagnosis.checks:
        marker = "OK" if check.ok else "FAIL"
        line = f"[{marker}] {check.name}: {check.detail}"
        if not check.ok and check.remedy:
            line += f" Remedy: {check.remedy}"
        lines.append(line)
    return "\n".join(lines) + "\n"
