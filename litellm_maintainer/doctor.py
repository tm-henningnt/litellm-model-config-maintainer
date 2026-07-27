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
    checks: list[Check] = []
    for provider_id in sorted(policy.providers):
        offering_ids = tuple(o.id for o in feed.offerings_for(provider_id))
        if not offering_ids:
            # No Offering to probe: the credential check above already
            # names a provider the Feed does not cover.
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

    Checks 9 and 10 cover the two ways this system can be fully
    configured and still do nothing: no writer for the Journal, and no
    process that reads it.

    Every failed check carries a non-`None` `remedy`.
    """
    checks: list[Check] = [_policy_parses_check()]
    checks.extend(_credential_checks(policy, feed, environ))
    checks.append(_feed_document_age_check(policy, feed_document_metadata, now))
    checks.append(_proxy_check(proxy_ok))
    checks.append(_health_state_populated_check(health))
    checks.extend(_probed_checks(policy, feed, health))
    checks.extend(_withheld_checks(policy, feed))
    checks.extend(_litellm_patch_checks(litellm_patches))
    checks.extend(_journal_callback_checks(served_configs or {}))
    checks.append(_tick_installed_check(tick_installed, tick_plist_path))
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
