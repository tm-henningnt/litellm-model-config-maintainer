"""Command-line entry points.

Keep this module thin. The work belongs in the module a command wraps —
`validate` wraps `litellm_maintainer.policy`.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import os
import plistlib
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

import litellm_maintainer.watcher
from litellm_maintainer.deploy import deploy_provider_modules
from litellm_maintainer.feed import load_feed
from litellm_maintainer.entitlements import (
    pool_siblings as entitlement_pool_siblings,
)
from litellm_maintainer.entitlements import sub_allowance_keys
from litellm_maintainer.generate import (
    read_previous_config,
    rendered_config_is_unchanged,
    write_config,
)
from litellm_maintainer.health import read_health, write_health
from litellm_maintainer.litellm_patches import inspect_patches, litellm_source_root
from litellm_maintainer.lock import LockBusy, maintainer_lock
from litellm_maintainer.journal import (
    observation_key_map,
    read_observations,
    resolve_observation_keys,
    truncate_first,
)
from litellm_maintainer.notify import (
    default_notifier,
    detect_events,
    notify_all,
    previous_run_state_path,
    read_previous_run_state,
    write_previous_run_state,
)
from litellm_maintainer.plan import plan
from litellm_maintainer.policy import PolicyError, describe_policy, load_policy
from litellm_maintainer.prober import (
    ProbeTarget,
    UnknownProviderError,
    _declared_provider_id,
    build_worklist,
    format_summary_line,
    live_transport,
    probe_credential,
    probe_offering,
    probe_offerings,
)
from litellm_maintainer.redact import build_redaction_map, parse_dotenv_file, redact
from litellm_maintainer.reduce import reduce
from litellm_maintainer.report import append_run_log, print_status
from litellm_maintainer.schedule import (
    DEFAULT_TICK_SECONDS,
    build_plist_spec,
    due,
    health_state_age,
    install as install_plist,
    launchctl_load_command,
    launchctl_unload_command,
    uninstall as uninstall_plist,
)
from litellm_maintainer.smoke import (
    build_smoke_entries,
    format_smoke_line,
    group_by_rule,
    live_smoke_transport,
    pick_healthiest,
    run_smoke_check,
)
from litellm_maintainer.safety import (
    SafetyError,
    detect_envelope_downgrades,
    refusal_for_implausible_feed,
    refusal_for_removal_share,
    refusal_for_zero_offered,
    removed_aliases,
    rollback_latest_snapshot,
    snapshot_config,
    validate_config_before_write,
)


def _default_env_path() -> Path | None:
    candidate = Path(".env.local")
    return candidate if candidate.exists() else None


def _credential_resolver(env_path: Path | None):
    """Return a `NAME -> value or None` resolver for the credential check.

    Checks `os.environ` first, matching a real deployment where the
    proxy's own process environment carries the credential. Falls back
    to `env_path` (the `.env.local`-style file also used for
    redaction), since the operator's shell does not export every
    variable the file defines. `validate_config_before_write` only
    checks whether the result is `None`, never the value itself.
    """
    file_values: dict[str, str] = {}
    if env_path is not None and env_path.exists():
        file_values = parse_dotenv_file(env_path)

    def resolver(name: str) -> str | None:
        return os.environ.get(name) or file_values.get(name) or None

    return resolver


def _credential_environment(env_path: Path | None) -> dict[str, str]:
    """Every credential name this instance can resolve, as one mapping.

    `doctor` needs the same answer `probe` gets, so it must read both
    sources `_credential_resolver` reads: the process environment and the
    `.env.local`-style file. The file's values come first in the merge
    and the process environment wins, matching the resolver's own order.
    """
    values: dict[str, str] = {}
    if env_path is not None and env_path.exists():
        values.update(parse_dotenv_file(env_path))
    values.update({k: v for k, v in os.environ.items() if v})
    return values


def _probe_live_transport(feed, env_path: Path | None):
    """Return a live probe transport that authenticates per provider.

    The Prober calls providers DIRECTLY, so each call must carry that
    provider's own credential, named by the Feed provider's
    `authentication.credential_hint` (see `prober.probe_credential`).
    Warning: never pass `LITELLM_MASTER_KEY` here. That is the proxy's
    own inbound credential; no provider accepts it, so every probe
    would classify `needs_operator` and Exclude a working Offering.
    Only `smoke.py`, which calls THROUGH the proxy, sends the master
    key.
    """
    resolver = _credential_resolver(env_path)

    def transport(target):
        return live_transport(
            target, credential=probe_credential(target, feed=feed, resolver=resolver)
        )

    return transport


def _fold_into_health_state(
    *,
    home: Path | None,
    outcomes: dict,
    admitted,
    passthrough_auth,
    now,
    key_map: dict[str, str] | None = None,
    rotate_journal: bool = True,
    timeout: float | None = None,
    pool_siblings: dict[str, frozenset[str]] | None = None,
    sub_allowances: frozenset[str] | None = None,
):
    """Fold this run's outcomes into Health State, under the lock.

    Take the maintainer's lock, read Health State again, reduce, and
    write. Read again inside the lock because the copy the caller read
    before probing is stale by the length of the sweep.

    Warning: never hold this lock across a Probe. A paced sweep takes
    minutes, and a lock held that long stops the tick and the watcher
    from recording anything. See ADR 0002.

    Rotate the Observation Journal only after the write succeeds.
    Rotating first would lose an entry that no Health State ever
    recorded.

    `pool_siblings` comes from `entitlements.pool_siblings`. It lets a
    quota exhaustion mark its pool mates due for a Probe. It never
    Excludes one -- see ADR 0004.

    Raise `LockBusy` when another maintainer process holds the lock. The
    caller decides what that means.
    """
    from litellm_maintainer.paths import health_path, journal_path, lock_path

    h_path = health_path(home)
    j_path = journal_path(home)

    lock_kwargs = {} if timeout is None else {"timeout": timeout}
    with maintainer_lock(lock_path(home), **lock_kwargs):
        fresh_health = read_health(h_path)
        observations = []
        journal_read = None
        if key_map is not None:
            journal_read = read_observations(j_path)
            observations = resolve_observation_keys(journal_read.observations, key_map)
        next_health = reduce(
            prior=fresh_health,
            outcomes=outcomes,
            observations=observations,
            admitted=admitted,
            passthrough_auth=passthrough_auth,
            now=now,
            pool_siblings=pool_siblings,
            sub_allowances=sub_allowances,
        )
        write_health(h_path, next_health)
        if rotate_journal and journal_read is not None and journal_read.observations:
            # Remove exactly the entries this fold consumed, by
            # position. `truncate_processed` compared timestamps to
            # `now`, so a writer whose clock ran ahead made rotation a
            # silent no-op and the Journal never emptied. See
            # `journal.truncate_first`.
            truncate_first(j_path, len(journal_read.observations))
        return next_health


def _needs_confirming(outcome) -> bool:
    """State whether one Journal observation needs a Probe to confirm it.

    Three conditions qualify. None of them attributes the failure to the
    Offering on one real-traffic event.

    `inconclusive`: the attempt measured nothing. One ambiguous call
    must never change Health State on its own.

    A timeout: no response arrived, so there is no provider message to
    read. `classify` maps every transport condition to `self_healing`
    with the reason `timeout`, which is the right CONDITION, and it is
    the one failing condition that carries no provider statement at
    all. The client's own deadline, a slow model and a busy worker are
    indistinguishable from here, and on real traffic the client chooses
    the prompt, the tools and the token count -- the same argument ADR
    0008 makes for an unrecognised failure.

    Measured 2026-07-31: two timeouts on `claude-chatgpt1-gpt-5.6-sol`,
    the slowest model on that seat, Excluded it with no Probe. The
    Offering answered a Probe 23 hours earlier and answered one again
    afterwards. It was removed from the Generated Config under a
    `claude -p` run that was pinned to that very Alias, so a slow answer
    took the model out from under a live job.

    An authentication failure: the provider DID state this one, so it
    joins the list on a different argument. A 401 states that the
    provider refused one request. It does not state that the credential
    is invalid. Those two readings call for opposite actions, and one
    event cannot tell them apart.

    This project already accepts that doubt. `reduce`'s
    `_PASSTHROUGH_EXEMPT_REASONS` holds this exact reason, because a
    Passthrough Auth Offering carries the caller's credential and the
    failure belongs to that caller. Here the proxy owns the credential,
    so the doubt is smaller -- and it is not zero. So confirm the
    failure; do not exempt it. A Probe sends a known-good synthetic
    request under the proxy's own credential, which is the measurement
    that separates the two readings.

    Measured 2026-07-31: `qwencloud-token-plan:qwen3.8-max-preview`
    Excluded on `authentication_failed` with `failure_count: 7`. The
    same credential then answered ten Probes out of ten, and the five
    pool siblings that share it never stopped answering. A revoked
    credential cannot refuse one Offering and serve five.

    Recovery corrects none of the three by itself. None carries a reset
    time, so `reduce._apply_reset_expiry` cannot clear the exclusion by
    the clock. Only a Probe can, and the journal-triggered run that
    Excluded the Offering also reset the interval, which postponed the
    next sweep by 55 minutes on the timeout above.

    Read the reason, never the bucket. `self_healing` also carries a
    quota exhaustion, a gateway error and a rate limit, and each of
    those the provider DID state. `needs_operator` also carries a quota
    exhaustion at a zero limit, which states its own condition exactly.
    """
    from litellm_maintainer.classify import (
        INCONCLUSIVE,
        REASON_ALIAS_NOT_SERVED,
        REASON_AUTHENTICATION_FAILED,
        REASON_TIMEOUT,
    )

    # `alias_not_served` is Inconclusive and must NOT be confirmed. A
    # Probe calls the provider directly, bypassing the proxy, so it would
    # answer "the Offering is healthy" — true, and about the wrong
    # subject. The condition is that the proxy does not serve the Alias,
    # and no Probe can see that. Confirming it would spend a call to
    # learn nothing and would mask the one signal that names it.
    if outcome.reason == REASON_ALIAS_NOT_SERVED:
        return False

    return outcome.bucket == INCONCLUSIVE or outcome.reason in (
        REASON_TIMEOUT,
        REASON_AUTHENTICATION_FAILED,
    )


def _confirming_worklist(
    worklist,
    *,
    feed,
    policy,
    health,
    observations,
    now,
):
    """Narrow a journal-triggered run's sweep to the confirming Probes.

    A run the Observation Journal triggered probes almost nothing, and
    that is deliberate:

    - A self-identifying failure needs no Probe. `classify` already read
      the condition from the provider's own message inside the failure
      callback. A quota exhaustion is the clearest case: probing it
      spends a call to re-learn a fact, against a provider that is
      already refusing us.
    - A failure this run cannot attribute to the Offering gets exactly
      one Probe, for that one Offering. `_needs_confirming` names those
      three conditions: an `inconclusive` outcome, a timeout, and an
      authentication failure.

    Everything else is left alone. The ordinary tick still sweeps what
    is stale on its own schedule; this run is not that run.

    A confirming Probe overrules the observation that asked for it.
    `reduce` applies a Probe outcome last, because it carries the
    timestamp `now` and a Journal entry reports something already past.
    So an Offering that answers its Probe keeps its place in the
    Generated Config, and one that fails the Probe too is Excluded on
    two independent measurements rather than one.

    Narrow `targets` ONLY. `admitted` and `skipped_passthrough` stay
    whole, because `reduce` uses them to prune records and to apply the
    Passthrough Auth exemption. Narrowing those would silently drop
    health for every Offering outside this trigger.

    Build the confirming targets with `force=True`, not from the
    worklist the caller passed. That worklist already dropped anything
    whose health looked fresh, and an Offering that just failed in
    production can easily still look fresh.
    """
    from dataclasses import replace as _replace

    key_map = observation_key_map(feed=feed, policy=policy)
    resolved = resolve_observation_keys(list(observations), key_map)
    confirm_keys = {
        observation.offering_id
        for observation in resolved
        if _needs_confirming(observation.outcome)
    }
    if not confirm_keys:
        return _replace(worklist, targets=())

    reachable = build_worklist(feed=feed, policy=policy, health=health, now=now, force=True)
    targets = tuple(target for target in reachable.targets if target.key in confirm_keys)
    return _replace(worklist, targets=targets)


def _report_unclassified_observations(observations, mapping: dict[str, str]) -> None:
    """Name every failure `classify` could not read, and its message.

    These changed no Health State: the Journal path fails open, so an
    `unrecognized_failure` from real traffic is re-bucketed to
    `inconclusive` (`reduce.journal_outcome`, ADR 0008). That is the
    safe behaviour, and it is also silent. This print is what stops it
    being invisible.

    Each line is the evidence for a missing `classify` rule. Add the
    rule because ten of these arrived, never because a rule was
    guessed at.
    """
    from litellm_maintainer.classify import REASON_UNRECOGNIZED_FAILURE

    unclassified = [
        observation
        for observation in observations
        if observation.outcome.reason == REASON_UNRECOGNIZED_FAILURE
    ]
    if not unclassified:
        return

    counts: dict[str, int] = {}
    messages_by_key: dict[str, set[str]] = {}
    for observation in unclassified:
        key = observation.offering_id
        counts[key] = counts.get(key, 0) + 1
        if observation.message:
            messages_by_key.setdefault(key, set()).add(observation.message)

    print(
        redact(
            f"{len(unclassified)} failure(s) real traffic hit that classify does "
            "not recognise. These changed no Health State (ADR 0008). Each one "
            "names a rule classify is missing:",
            mapping,
        )
    )
    for key in sorted(counts):
        print(redact(f"  {key}: {counts[key]} failure(s)", mapping))
        for message in sorted(messages_by_key.get(key, ())):
            print(redact(f"    {message}", mapping))


#: How many observations may change nothing before `run` says so. Low
#: enough to catch a misread condition in minutes; high enough that an
#: ordinary rate limit, which is inconclusive by design, stays quiet.
UNPRODUCTIVE_OBSERVATION_THRESHOLD = 10


def _report_unproductive_offerings(health_state, mapping: dict[str, str]) -> None:
    """Name every Offering whose observations keep changing nothing.

    A wrong classification is SILENT. ADR 0008 makes an unrecognised
    failure visible by storing its message, but a confidently misread
    one carries no message and looks like ordinary operation.

    Measured 2026-07-27: an exhausted OpenCode Go plan answered
    "Monthly usage limit reached. Resets in 16hr 32min." with HTTP 429.
    No rule matched the wording, so it fell through to the bare-429
    rule and read as `rate_limited` with no reset time -- inconclusive,
    which changes nothing. Ninety entries accumulated and the Offering
    stayed in the Generated Config. Nothing reported it; the operator
    noticed.

    A high `inconclusive_count` is the shape of that fault: real
    traffic keeps failing on one Alias and Health State keeps not
    moving. Print it with the reason, so the next step is to capture
    the provider's own message and write the missing `classify` rule.
    """
    offenders = sorted(
        (
            (key, record)
            for key, record in health_state.offerings.items()
            if record.inconclusive_count >= UNPRODUCTIVE_OBSERVATION_THRESHOLD
        ),
        key=lambda item: -item[1].inconclusive_count,
    )
    if not offenders:
        return

    print(
        redact(
            f"{len(offenders)} Offering(s) keep failing without changing Health "
            "State. classify is probably reading the condition wrong. Capture "
            "the provider's own message and add a rule (see "
            "tests/fixtures/classify/CAPTURE.md):",
            mapping,
        )
    )
    for key, record in offenders:
        print(
            redact(
                f"  {key}: {record.inconclusive_count} observations changed "
                f"nothing, last read as {record.reason or 'no reason recorded'}",
                mapping,
            )
        )


def _report_skipped_health_records(health_state, mapping: dict[str, str]) -> None:
    """Warn when `read_health` skipped a malformed Health State record.

    Print nothing when `skipped_records` is zero. `read_health` already
    keeps every good record; this only tells the operator that one
    record did not parse, so a bad write is worth a look.
    """
    if health_state.skipped_records:
        print(
            redact(
                f"Warning: skipped {health_state.skipped_records} malformed "
                "Health State record(s). Every other record was kept.",
                mapping,
            )
        )


def _default_out_path() -> Path:
    """Return the default Generated Config path.

    This is the instance directory's own copy
    (`~/.config/litellm-maintainer/config.yaml`), never the live proxy
    config at `~/.config/litellm/config.yaml`. A write to the live path
    restarts the proxy; this command must never default there.
    """
    from litellm_maintainer.paths import generated_config_path

    return generated_config_path()


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a Policy file and print what it understood.

    Exit 0 on a valid Policy. Exit 1 on an invalid one, printing a
    message that names the offending key.
    """
    env_path = Path(args.env) if args.env else _default_env_path()
    mapping = build_redaction_map(env_path)
    try:
        policy = load_policy(Path(args.policy))
    except PolicyError as exc:
        print(redact(f"Policy is invalid: {exc}", mapping), file=sys.stderr)
        return 1
    print(redact(describe_policy(policy), mapping))
    return 0


def _implausible_feed_refusal(*, feed, policy, force: bool) -> str | None:
    """Return the implausible-Feed refusal message, or `None` to proceed.

    Shared by `cmd_generate` and `cmd_run` (defect 5): before this fix,
    only `cmd_generate` called `refusal_for_implausible_feed`. `cmd_run`
    is the SCHEDULED path -- the one that runs unattended -- so it must
    apply every safety refusal `cmd_generate` applies, never fewer.
    Honours `--force` the same way both callers already did.
    """
    refusal = refusal_for_implausible_feed(
        len(feed.offerings), providers_configured=bool(policy.providers)
    )
    if refusal is not None and not force:
        return refusal
    return None


def _apply_safety_rail_and_write(
    *,
    result,
    out_path: Path,
    home: Path | None,
    env_path: Path | None,
    mapping: dict[str, str],
    now: datetime,
    policy,
    force: bool,
) -> tuple[int, int, frozenset[str], bool]:
    """Validate a `PlanResult`, apply the safety rail, and write it.

    Shared by `cmd_generate` and `cmd_run` (defect 5). Before this fix
    the two commands each carried their own copy of this sequence, and
    `cmd_run` -- the scheduled, unattended path -- had already drifted:
    it never applied the safety refusals `cmd_generate` did. Factoring
    it once here is what stops that drift from happening again.

    Returns `(exit_code, new_count, dropped_aliases, written)`.
    `exit_code` is `0` when this function got as far as deciding about
    the write, including a forced write past a threshold refusal; any
    other value is what the caller must return at once, having already
    printed why.

    `written` states whether the file actually changed. It is `False`
    on a skipped, unchanged write, which is a success and not a
    refusal. A caller reports what happened; it must not claim it wrote
    a file it did not write.
    """
    from litellm_maintainer.paths import snapshots_dir

    if result.refusal is not None:
        print(redact(f"Refused to write: {result.refusal}", mapping), file=sys.stderr)
        return 1, 0, frozenset(), False

    previous_config = read_previous_config(out_path)
    previous_count = (
        len(previous_config.get("model_list", [])) if previous_config is not None else None
    )
    new_count = len(result.config.get("model_list", []))
    dropped_aliases = removed_aliases(previous_config, result.config)

    validation_problems = validate_config_before_write(
        result.config, credential_resolver=_credential_resolver(env_path)
    )
    if validation_problems:
        print(redact("Refused to write: the config failed validation.", mapping), file=sys.stderr)
        for problem in validation_problems:
            print(redact(f"  {problem}", mapping), file=sys.stderr)
        return 1, new_count, dropped_aliases, False

    zero_refusal = refusal_for_zero_offered(new_count)
    removal_refusal = refusal_for_removal_share(
        previous_count=previous_count,
        new_count=new_count,
        maximum_removal_share=policy.safety.maximum_removal_share,
        removed_aliases=dropped_aliases,
    )
    threshold_refusal_message = zero_refusal or (
        removal_refusal.message if removal_refusal is not None else None
    )
    if threshold_refusal_message is not None:
        if not force:
            print(redact(threshold_refusal_message, mapping), file=sys.stderr)
            return 1, new_count, dropped_aliases, False
        print(redact(f"Forced past refusal: {threshold_refusal_message}", mapping))

    if result.report.custom_provider_map_conflict is not None:
        print(redact(f"WARNING: {result.report.custom_provider_map_conflict}", mapping))

    for collision in result.report.limit_collisions:
        print(redact(f"WARNING: {collision.message}", mapping))

    if result.report.client_facing_variants:
        print(
            f"Client-Facing Variants: {len(result.report.client_facing_variants)} "
            "(a second Alias for a wide Offering, so a client budgets its "
            "whole window)"
        )
    for offering_id in result.report.client_facing_variants_unknown:
        print(
            redact(
                "WARNING: client_facing_variants.operator_stated names "
                f"{offering_id!r}, which this run did not admit, so it grants "
                "no variant. Prune the line or find out why the Offering left.",
                mapping,
            )
        )

    envelope_downgrades = detect_envelope_downgrades(previous_config, result.config)
    if envelope_downgrades:
        print(
            redact(
                "WARNING: the following Aliases lose the envelope-unwrapping "
                "handler this run — every SUCCESS on them may fail with "
                "\"provider returned a response with no 'choices'\" (see "
                "docs/gotchas.md, 'Some providers wrap successful responses'; "
                "spec-corrections.md, correction 5): "
                f"{', '.join(sorted(envelope_downgrades))}.",
                mapping,
            )
        )

    # Every write to the Generated Config restarts the proxy, because
    # it is the one file the `--reload` watcher reads. Skip a write that
    # would change nothing, and skip its snapshot with it: a snapshot of
    # an identical config spends one of the operator's `snapshot_count`
    # rollback slots on no change at all.
    #
    # The safety rail above still ran. It reads the config already on
    # disk and refuses on a removal share or a zero-offered plan, and it
    # must reach those refusals whether or not this write happens.
    if rendered_config_is_unchanged(result.config, out_path, result.annotations):
        return 0, new_count, dropped_aliases, False

    snapshot_config(out_path, snapshots_dir(home), keep=policy.safety.snapshot_count, now=now)
    write_config(result.config, out_path, result.annotations)

    return 0, new_count, dropped_aliases, True


def cmd_generate(args: argparse.Namespace) -> int:
    """Read the Feed, Policy and Health State, and write a Generated Config.

    Exit 0 on success, having written a config. Exit 1 on any refusal,
    having written nothing: an invalid Policy, a Feed that failed to
    load or looks implausibly short, a `plan` refusal (an Alias
    collision), a structural validation failure (a duplicate Alias, an
    entry with no model, an unresolved credential variable), or a
    safety-rail refusal (the offered count would drop too far, or reach
    zero) — unless `--force` applies it anyway. `--force` never
    overrides a structural validation failure: those are defects to
    fix, not judgment calls.

    Warning: `generate` on its own is a convenience command. Health
    State starts empty until something probes. Empty Health State means
    no Offering can be Sunsetting yet (spec-corrections.md, correction
    9), so this command warns and names how many Aliases a probe run
    would restore. The real pipeline is probe, then reduce, then plan
    (`run`, below); a scheduled run must never plan alone.
    """
    from litellm_maintainer.paths import health_path

    env_path = Path(args.env) if args.env else _default_env_path()
    mapping = build_redaction_map(env_path)
    home = Path(args.home) if args.home else None

    try:
        policy = load_policy(Path(args.policy))
    except PolicyError as exc:
        print(redact(f"Policy is invalid: {exc}", mapping), file=sys.stderr)
        return 1

    feed_path = Path(args.feed)
    try:
        feed = load_feed(feed_path)
    except Exception as exc:  # noqa: BLE001 - any load failure refuses, none crashes
        print(redact(f"Refused to write: the Feed could not be read ({exc}).", mapping), file=sys.stderr)
        return 1

    implausible_feed_refusal = _implausible_feed_refusal(feed=feed, policy=policy, force=args.force)
    if implausible_feed_refusal is not None:
        print(redact(implausible_feed_refusal, mapping), file=sys.stderr)
        return 1

    health_state = read_health(health_path(home))
    _report_skipped_health_records(health_state, mapping)

    now = datetime.now(timezone.utc)
    result = plan(feed=feed, policy=policy, health=health_state.offerings, now=now)

    # Refuse rather than warn. Sunsetting needs OUR OWN recorded success,
    # so an empty Health State drops every Sunsetting Offering the
    # operator runs. The removal-share rail does not catch it: the
    # operator's own case drops 4 of 78, far under any sane share. A
    # warning on a command that then writes the file is read once and
    # never again.
    #
    # Refuse only when the run would actually lose something, and only
    # while a Probe could still settle it.
    #
    # Key on "no record at all", not on an empty Health State and not on
    # `restorable_by_probe` alone. An empty Health State is the wrong
    # test: a sweep scoped to one provider leaves Health State non-empty
    # while every Sunsetting Offering is still unprobed. And
    # `restorable_by_probe` also lists an Offering that WAS probed and
    # failed, which is dropped for a good reason; refusing on that would
    # be a wall no probe could ever clear.
    never_probed = tuple(
        sorted(
            offering_id
            for offering_id in result.report.restorable_by_probe
            if offering_id not in health_state.offerings
        )
    )
    if never_probed and not args.force:
        dropped = ", ".join(never_probed)
        print(
            redact(
                f"Refused to write: this run would drop {len(never_probed)} "
                f"Sunsetting Offering(s) no Probe has ever measured: {dropped}.\n"
                "Sunsetting needs a success this tool recorded itself. The "
                "removal-share guard does not catch this, because the count "
                "is small.\n"
                "Run 'run' to probe, reduce and generate in one step, or "
                "'probe' and then 'generate'. Probe them and this refusal "
                "clears, whether they answer or not. Use --force to write "
                "anyway.",
                mapping,
            ),
            file=sys.stderr,
        )
        return 1

    out_path = Path(args.out)
    exit_code, new_count, _dropped_aliases, written = _apply_safety_rail_and_write(
        result=result,
        out_path=out_path,
        home=home,
        env_path=env_path,
        mapping=mapping,
        now=now,
        policy=policy,
        force=args.force,
    )
    if exit_code != 0:
        return exit_code

    if written:
        print(redact(f"Wrote {new_count} Aliases to {out_path}", mapping))
    else:
        print(
            redact(
                f"{out_path} already holds these {new_count} Aliases. Nothing "
                "written, so the proxy did not reload.",
                mapping,
            )
        )
    if result.report.candidates:
        print(redact(f"Candidates awaiting approval: {len(result.report.candidates)}", mapping))
        for candidate_id in result.report.candidates:
            print(redact(f"  {candidate_id}", mapping))
    return 0


def cmd_rollback(args: argparse.Namespace) -> int:
    """Restore the most recent snapshot onto the Generated Config.

    Exit 0 and name the restored snapshot on success. Exit 1 when no
    snapshot exists yet, printing why.
    """
    from litellm_maintainer.paths import snapshots_dir

    env_path = Path(args.env) if args.env else _default_env_path()
    mapping = build_redaction_map(env_path)
    home = Path(args.home) if args.home else None
    out_path = Path(args.out)

    try:
        restored = rollback_latest_snapshot(out_path, snapshots_dir(home))
    except SafetyError as exc:
        print(redact(f"Rollback failed: {exc}", mapping), file=sys.stderr)
        return 1

    print(redact(f"Restored {restored.name} onto {out_path}", mapping))
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    """Build the Prober's worklist and, unless `--dry-run`, run it.

    The worklist comes from Policy, not the Generated Config (see
    CONTEXT.md, "Prober"), so an Excluded Offering can still be reached
    and can recover. `--dry-run` prints the worklist and makes no
    network call: it never constructs a transport, never calls
    `probe_offerings`, and never writes Health State. Exit 0 on success.
    Exit 1 when the Policy is invalid, printing a message that names the
    offending key.
    """
    env_path = Path(args.env) if args.env else _default_env_path()
    mapping = build_redaction_map(env_path)
    try:
        policy = load_policy(Path(args.policy))
    except PolicyError as exc:
        print(redact(f"Policy is invalid: {exc}", mapping), file=sys.stderr)
        return 1

    try:
        feed = load_feed(Path(args.feed))
    except Exception as exc:  # noqa: BLE001 - a read failure reports, it never crashes
        print(redact(f"Could not read the Feed: {exc}", mapping), file=sys.stderr)
        return 1
    now = datetime.now(timezone.utc)

    from litellm_maintainer.paths import health_path

    home = Path(args.home) if args.home else None
    h_path = health_path(home)
    prior_health = read_health(h_path)
    _report_skipped_health_records(prior_health, mapping)

    try:
        worklist = build_worklist(
            feed=feed,
            policy=policy,
            health=prior_health,
            now=now,
            provider=getattr(args, "provider", None),
            force=getattr(args, "force", False),
        )
    except UnknownProviderError as exc:
        print(redact(str(exc), mapping), file=sys.stderr)
        return 1

    if getattr(args, "provider", None):
        print(redact(f"Scoped to provider: {args.provider}", mapping))

    print(
        redact(
            f"Worklist: {len(worklist.targets)} to probe, "
            f"{len(worklist.skipped_fresh)} skipped (fresh), "
            f"{len(worklist.skipped_withheld)} withheld, "
            f"{len(worklist.skipped_passthrough)} passthrough auth",
            mapping,
        )
    )
    for target in worklist.targets:
        print(redact(f"  probe: {target.key} ({target.provider_id})", mapping))
    for key in worklist.skipped_fresh:
        print(redact(f"  skip (fresh): {key}", mapping))
    for key in worklist.skipped_withheld:
        print(redact(f"  skip (withheld): {key}: {policy.withheld[key]}", mapping))
    for key in worklist.skipped_passthrough:
        print(redact(f"  skip (passthrough auth): {key}", mapping))

    if args.dry_run:
        return 0

    # Live path. The orchestrator decides whether this ever runs; it
    # makes a real network call per target in `worklist.targets`. Each
    # call carries the target provider's own credential
    # (`_probe_live_transport`), never the proxy's master key.
    outcomes = probe_offerings(
        worklist.targets,
        pacing=policy.pacing,
        transport=_probe_live_transport(feed, env_path),
        now=lambda: datetime.now(timezone.utc),
    )
    # The Observation Journal never reaches Health State unless `reduce`
    # sees it (ADR 0001: the proxy appends, the maintainer reads and
    # rotates). Read it fresh every run, fold
    # it in here, and only then rotate the file -- rotating before
    # `write_health` succeeds would lose an entry no Health State ever
    # recorded. The proxy records the ALIAS; Health State keys a
    # Discovered Offering by its Offering id, so resolve the keys first
    # (`journal.observation_key_map`) or the entry silently changes
    # nothing.
    try:
        _fold_into_health_state(
            home=home,
            outcomes=outcomes,
            admitted=worklist.admitted,
            passthrough_auth=frozenset(worklist.skipped_passthrough),
            now=now,
            key_map=observation_key_map(feed=feed, policy=policy),
            pool_siblings=entitlement_pool_siblings(feed=feed, policy=policy),
            sub_allowances=sub_allowance_keys(policy),
        )
    except LockBusy as busy:
        print(
            redact(f"Health State not written: {busy}", mapping),
            file=sys.stderr,
        )
        return 1

    print("Results:")
    for target in worklist.targets:
        print(redact(format_summary_line(target.key, outcomes[target.key]), mapping))
    return 0


def build_confirm_probe(*, policy, feed, home: Path | None, mapping: dict[str, str]):
    """Build the `confirm` callable `watcher.JournalWatcher` needs.

    This is the watcher's "seam to the Prober": `watcher.JournalWatcher`
    takes an injected `Callable[[str], None]` and never imports the
    Prober itself. This is that wiring (defect 4) -- one real, confirming Probe for a single
    Offering key, folded into Health State before returning, so an
    ambiguous (Inconclusive) Journal entry never changes Health State on
    a Journal entry alone.

    A Journal entry records the ALIAS (see
    `providers/journal_failure_callback.py`), so `offering_id` arrives
    as an Alias. Resolve it to its Health State key first
    (`journal.observation_key_map`): a Discovered Offering's key is its
    Offering id, a Declared Offering's key is the Alias itself. Prints
    a warning and does nothing when the resolved key names no Offering
    in the Feed or Policy -- the Offering left Policy or the Feed
    between the failure and the confirming call, which is worth a log
    line, never a crash.

    The confirming Probe authenticates with the target provider's own
    credential (`_probe_live_transport`), never the proxy's master key.
    """
    from litellm_maintainer.paths import health_path

    key_map = observation_key_map(feed=feed, policy=policy)
    transport = _probe_live_transport(feed, env_path=_default_env_path())

    def confirm(offering_id: str) -> None:
        h_path = health_path(home)
        prior_health = read_health(h_path)
        now = datetime.now(timezone.utc)

        key = key_map.get(offering_id, offering_id)
        offering = feed.offering(key)
        declared = next((d for d in policy.declared if d.alias == key), None)

        if declared is not None:
            target = ProbeTarget(
                key=key,
                provider_id=_declared_provider_id(declared),
                declared=declared,
            )
            is_passthrough = declared.passthrough_auth
        elif offering is not None:
            target = ProbeTarget(
                key=key, provider_id=offering.provider_id, offering=offering
            )
            is_passthrough = False
        else:
            print(
                redact(
                    f"watch: cannot confirm {offering_id}: it names no Offering "
                    "in the Feed or Policy",
                    mapping,
                )
            )
            return

        outcome = probe_offering(
            target,
            transport=transport,
            now=lambda: datetime.now(timezone.utc),
        )
        try:
            _fold_into_health_state(
                home=home,
                outcomes={key: outcome},
                admitted=frozenset(prior_health.offerings) | {key},
                passthrough_auth=frozenset({key}) if is_passthrough else frozenset(),
                now=now,
                key_map=None,
            )
        except LockBusy as busy:
            print(redact(f"watch: {busy}", mapping))
            return
        print(redact(f"watch: confirming Probe for {key}: {outcome.bucket}", mapping))

    return confirm


def cmd_fetch(args: argparse.Namespace, *, transport=None) -> int:
    """Download the Feed and write the Feed Document.

    The only writer of that file (CONTEXT.md, "Feed Document"). Exit 0
    when a new document was promoted, 1 when nothing was: a refused
    fetch is a failure the operator asked for directly, so it is worth a
    non-zero exit here. Inside `run` the same failure is not fatal, and
    the tick carries on with the previous document.

    `transport` is the test seam. It defaults to the real HTTP transport,
    which is the only network code this command reaches.
    """
    from litellm_maintainer.fetch import (
        fetch_feed_document,
        http_transport,
        resolve_credential,
    )
    from litellm_maintainer.paths import ensure_instance_dirs, feed_document_path
    from litellm_maintainer.policy import FeedSource

    env_path = Path(args.env) if args.env else _default_env_path()
    mapping = build_redaction_map(env_path)
    home = Path(args.home) if args.home else None

    # `--url` exists to break a bootstrap cycle. `init` needs a Feed
    # Document to write a Policy, and a Policy is the ordinary place the
    # Feed URL lives — so a first fetch had no valid Policy to read. With
    # `--url`, the very first command an operator runs needs no Policy at
    # all. Afterwards the Policy `feed` block is the better home for it,
    # because the scheduled tick reads it from there.
    if args.url:
        source = FeedSource(url=args.url, credential_env=args.credential_env)
        providers_configured = False
    else:
        if not args.policy:
            print(
                "Pass --policy, or --url for a first fetch before any Policy "
                "exists.",
                file=sys.stderr,
            )
            return 1
        try:
            policy = load_policy(Path(args.policy))
        except PolicyError as exc:
            print(redact(f"Policy is invalid: {exc}", mapping), file=sys.stderr)
            return 1

        if policy.feed is None:
            print(
                "Policy names no 'feed' block, so there is nothing to fetch. "
                "Add feed.url to Policy, pass --url, or keep pointing --feed "
                "at a document you refresh yourself.",
                file=sys.stderr,
            )
            return 1
        source = policy.feed
        providers_configured = bool(policy.providers)

    ensure_instance_dirs(home)
    destination = Path(args.out) if args.out else feed_document_path(home)

    outcome = fetch_feed_document(
        source=source,
        destination=destination,
        transport=transport or http_transport(),
        providers_configured=providers_configured,
        # Read the `--env` file too, not just `os.environ`. The operator's
        # shell does not export every variable the file defines, and the
        # scheduled tick has no shell at all. See `_fetch_for_tick`.
        token=resolve_credential(source, _credential_environment(env_path)),
    )

    stream = sys.stdout if outcome.promoted else sys.stderr
    print(redact(f"{destination}: {outcome.message}", mapping), file=stream)
    if outcome.kept_previous and destination.exists():
        print(
            "The previous Feed Document is unchanged. Selection will run on it.",
            file=sys.stderr,
        )
    return 0 if outcome.promoted else 1


def cmd_headroom(args: argparse.Namespace) -> int:
    """Dispatch a `headroom` verb."""
    if args.verb == "refresh":
        return cmd_headroom_refresh(args)
    if args.verb == "install":
        return cmd_headroom_install(args)
    if args.verb == "uninstall":
        return cmd_headroom_uninstall(args)
    raise ValueError(f"unknown headroom verb: {args.verb!r}")  # pragma: no cover


def cmd_headroom_refresh(args: argparse.Namespace) -> int:
    """Run `headroom refresh`: ask codexbar and write Headroom State.

    Reads only the providers Policy's `headroom.sources` names — never
    every provider codexbar knows. This command holds Headroom State's
    OWN lock and never the maintainer lock, because codexbar takes 21-31
    seconds to answer and the maintainer lock protects the Observation
    Journal watcher's reset times (ADR 0002).

    A provider that errors, or whose entry fails codexbar's shape
    check, keeps its previous Reading; every other mapped Allowance
    still updates. Exit 0 whenever the refresh ran, including when
    Policy names no source at all. Exit 1 only when Policy itself
    cannot be read.
    """
    from litellm_maintainer.headroom import real_codexbar_runner, refresh_headroom
    from litellm_maintainer.paths import (
        ensure_instance_dirs,
        headroom_lock_path,
        headroom_path,
        policy_path,
    )

    env_path = Path(args.env) if args.env else _default_env_path()
    mapping = build_redaction_map(env_path)
    home = Path(args.home) if args.home else None

    resolved_policy = Path(args.policy) if args.policy else policy_path(home)
    try:
        policy = load_policy(resolved_policy)
    except FileNotFoundError:
        print(
            redact(
                f"No Policy at {resolved_policy}. Run init to write one, or pass --policy.",
                mapping,
            ),
            file=sys.stderr,
        )
        return 1
    except PolicyError as exc:
        print(redact(f"Policy is invalid: {exc}", mapping), file=sys.stderr)
        return 1

    ensure_instance_dirs(home)

    try:
        outcome = refresh_headroom(
            headroom_policy=policy.headroom,
            path=headroom_path(home),
            lock_path=headroom_lock_path(home),
            runner=real_codexbar_runner(
                policy.headroom.command, timeout=policy.headroom.timeout_seconds
            ),
            now=datetime.now(timezone.utc),
            maximum_staleness_hours=policy.schedule.maximum_staleness_hours,
        )
    except LockBusy as busy:
        print(redact(f"Headroom State not written: {busy}", mapping), file=sys.stderr)
        return 1

    print(redact(outcome.message, mapping))
    for failure in outcome.failures:
        print(redact(f"  {failure}", mapping), file=sys.stderr)

    # A crossing is the whole point of refreshing on a schedule: it says
    # the figure MOVED, which no poll of the current level can. Delivered
    # through `notify_all`, the one path that redacts before it sends.
    if outcome.crossings:
        from litellm_maintainer.notify import notify_all
        from litellm_maintainer.thresholds import crossing_messages

        notify_all(crossing_messages(outcome.crossings), mapping=mapping)
    return 0


def cmd_headroom_install(args: argparse.Namespace) -> int:
    """Write the launchd plist that ticks `headroom refresh`.

    A job SEPARATE from the tick's (`cmd_install`): its own label
    (`schedule.HEADROOM_LABEL`), its own plist file, its own interval.
    Never calls `launchctl`; prints the command that registers the job,
    the same rule `cmd_install` follows.

    When Policy declares no `headroom.sources`, the capability is off:
    write no plist. When one is already installed from an earlier
    Policy, say so plainly instead of leaving a stale job the operator
    never asked about again.
    """
    from litellm_maintainer.paths import instance_home
    from litellm_maintainer.schedule import (
        HEADROOM_LABEL,
        build_headroom_plist_spec,
        default_headroom_log_paths,
        plist_path,
    )

    resolved_policy = Path(args.policy)
    try:
        policy = load_policy(resolved_policy)
    except FileNotFoundError:
        print(f"No Policy at {resolved_policy}. Run init to write one, or pass --policy.")
        return 1
    except PolicyError as exc:
        print(f"Policy is invalid: {exc}")
        return 1

    target_dir = Path(args.target_dir)
    existing = plist_path(target_dir, HEADROOM_LABEL)

    if not policy.headroom.sources:
        message = "Policy declares no headroom sources; installing no job."
        if existing.exists():
            message += (
                f" A job is already installed at {existing}. Remove it with "
                f"'litellm-maintainer headroom uninstall --target-dir {target_dir}', "
                "since Policy no longer maps any Allowance to it."
            )
        print(message)
        return 0

    resolved_home = args.home or str(instance_home(None))
    standard_out_path, standard_error_path = default_headroom_log_paths(resolved_home)

    # Resolve every path to absolute before it enters the plist. launchd
    # runs a job from '/', so a relative '--policy' or '--home' resolves
    # against the wrong directory and the job fails on every tick into a
    # log nobody reads. `--env` already gets this treatment below; a
    # relative `--policy` was missed when this command was added.
    spec = build_headroom_plist_spec(
        python_executable=args.python,
        policy_path=str(Path(args.policy).resolve()),
        home=str(Path(args.home).resolve()) if args.home else None,
        env_path=str(Path(args.env).resolve()) if args.env else None,
        interval_minutes=policy.headroom.interval_minutes,
        standard_out_path=standard_out_path,
        standard_error_path=standard_error_path,
        # The source this job runs is a third-party binary and inherits
        # nothing from a login shell. Both were measured 2026-07-30; see
        # `build_headroom_plist_spec`.
        user=os.environ.get("USER"),
        path=os.environ.get("PATH"),
    )
    path = install_plist(target_dir, spec)
    print(f"Wrote {path}")
    print(f"Refresh interval: {policy.headroom.interval_minutes} minutes")
    if args.env is None:
        print(
            "WARNING: no --env given. launchd runs a job from '/', and the "
            "default '.env.local' lookup is relative to the working "
            "directory, so codexbar may resolve no credential."
        )
    print(f"Logs: {standard_out_path} and {standard_error_path}")
    print(f"Run this to register the job: {launchctl_load_command(path)}")
    return 0


def cmd_headroom_uninstall(args: argparse.Namespace) -> int:
    """Remove the headroom-refresh launchd plist. Safe when nothing is installed.

    Mirrors `cmd_uninstall`, on `schedule.HEADROOM_LABEL` instead of the
    tick's label, so removing one job never touches the other.
    """
    from litellm_maintainer.schedule import HEADROOM_LABEL, plist_path

    target_dir = Path(args.target_dir)
    path = plist_path(target_dir, HEADROOM_LABEL)
    print(f"Run this first, to unregister the job: {launchctl_unload_command(path)}")

    removed = uninstall_plist(target_dir, HEADROOM_LABEL)
    if removed is None:
        print("Nothing was installed.")
    else:
        print(f"Removed {removed}")
    return 0


def _load_read_inputs(args: argparse.Namespace, mapping: dict[str, str]):
    """Read Policy, the Feed Document and Health State, then plan over them.

    Shared by `entitlements` and `guidance`. Both derive their whole
    answer from `plan`'s own report, so neither can disagree with
    `status` about what is offered.

    Returns `(policy, feed, health, report, warnings, now)`, or `None`
    when a read failed, having already printed why. `now` is returned
    rather than read again by the caller, so every part of one answer
    describes the same instant.

    `warnings` carries the Feed staleness line when the Feed Document is
    past Policy's threshold: a stale catalogue produces confident, wrong
    guidance, so the warning travels with the answer rather than only
    appearing in `doctor`.
    """
    from litellm_maintainer.fetch import staleness_warning
    from litellm_maintainer.paths import feed_document_path, health_path, policy_path

    home = Path(args.home) if args.home else None

    resolved_policy = Path(args.policy) if args.policy else policy_path(home)
    try:
        policy = load_policy(resolved_policy)
    except FileNotFoundError:
        print(
            redact(
                f"No Policy at {resolved_policy}. Run init to write one, or "
                "pass --policy.",
                mapping,
            ),
            file=sys.stderr,
        )
        return None
    except PolicyError as exc:
        print(redact(f"Policy is invalid: {exc}", mapping), file=sys.stderr)
        return None

    feed_path = Path(args.feed) if args.feed else feed_document_path(home)
    try:
        feed = load_feed(feed_path)
    except Exception as exc:  # noqa: BLE001 - a read failure reports, it never crashes
        print(
            redact(f"Could not read the Feed Document at {feed_path}: {exc}", mapping),
            file=sys.stderr,
        )
        return None

    health_state = read_health(health_path(home))
    _report_skipped_health_records(health_state, mapping)
    now = datetime.now(timezone.utc)
    result = plan(feed=feed, policy=policy, health=health_state.offerings, now=now)

    warnings: list[str] = []
    maximum_age = policy.feed.maximum_age_hours if policy.feed else 24.0
    stale = staleness_warning(
        generated_at=feed.generated_at, maximum_age_hours=maximum_age, now=now
    )
    if stale is not None:
        warnings.append(stale)
    if not health_state.offerings:
        warnings.append(
            "Health State is empty, so nothing here has been measured yet; run probe"
        )

    return policy, feed, health_state.offerings, result.report, tuple(warnings), now


def _print_rendered(text: str, mapping: dict[str, str]) -> None:
    """Print a rendered answer, redacted. Every output path goes through here."""
    print(redact(text, mapping), end="")


def cmd_entitlements(args: argparse.Namespace) -> int:
    """Print the Entitlement view: what spending through each provider costs now.

    Read-only. Makes no network call and writes no file. See
    `litellm_maintainer.entitlements` and ADR 0004: a `shared_pool`
    declaration changes how this output reads and never changes Health
    State.
    """
    import json as _json

    from litellm_maintainer import entitlements as entitlements_module
    from litellm_maintainer.headroom import headroom_source_warnings, read_headroom
    from litellm_maintainer.paths import headroom_path

    env_path = Path(args.env) if args.env else _default_env_path()
    mapping = build_redaction_map(env_path)

    loaded = _load_read_inputs(args, mapping)
    if loaded is None:
        return 1
    policy, feed, health, report, warnings, now = loaded

    home = Path(args.home) if args.home else None
    headroom_state = read_headroom(headroom_path(home))
    warnings = warnings + headroom_source_warnings(
        headroom_policy=policy.headroom, headroom_state=headroom_state, now=now
    )

    view = entitlements_module.derive(
        feed=feed,
        policy=policy,
        health=health,
        report=report,
        now=now,
        warnings=warnings,
        headroom_state=headroom_state,
    )

    output_format = "json" if getattr(args, "json", False) else args.format
    if output_format == "json":
        _print_rendered(_json.dumps(view.as_dict(), indent=2) + "\n", mapping)
    elif output_format == "markdown":
        _print_rendered(entitlements_module.render_markdown(view), mapping)
    else:
        _print_rendered(entitlements_module.render_text(view), mapping)
    return 0


def cmd_guidance(args: argparse.Namespace) -> int:
    """Print ranked Canonical Models for one task axis, each with its Routes.

    Read-only. Makes no network call and writes no file. A caller that
    wants this in a file redirects it; see ADR 0005. Exit 1 when the axis
    or `--prefer` names something the Feed cannot answer, naming what it
    can.
    """
    import json as _json

    from litellm_maintainer import guidance as guidance_module
    from litellm_maintainer.headroom import headroom_source_warnings, read_headroom
    from litellm_maintainer.notify import previous_run_state_path, read_previous_run_state
    from litellm_maintainer.paths import headroom_path, instance_home

    env_path = Path(args.env) if args.env else _default_env_path()
    mapping = build_redaction_map(env_path)

    loaded = _load_read_inputs(args, mapping)
    if loaded is None:
        return 1
    policy, feed, health, report, warnings, now = loaded

    home = Path(args.home) if args.home else None
    previous = read_previous_run_state(previous_run_state_path(instance_home(home)))
    headroom_state = read_headroom(headroom_path(home))
    warnings = warnings + headroom_source_warnings(
        headroom_policy=policy.headroom, headroom_state=headroom_state, now=now
    )

    try:
        answer = guidance_module.derive(
            feed=feed,
            policy=policy,
            health=health,
            report=report,
            now=now,
            axis=args.axis,
            prefer=args.prefer,
            previous=previous,
            min_context=args.min_context,
            warnings=tuple(warnings),
            headroom_state=headroom_state,
        )
    except guidance_module.GuidanceError as exc:
        print(redact(str(exc), mapping), file=sys.stderr)
        return 1

    # Truncate here, not in `derive`, so the warning can state what was
    # actually dropped. Deriving with a limit cannot tell "you asked for
    # 5 and there are 40" from "you asked for 5 and there are 5", and a
    # cap that claims to have hidden rows it did not hide is a false
    # claim. A silent cap is worse still: it reads as "this is
    # everything".
    if args.limit is not None and len(answer.rows) > args.limit:
        dropped = len(answer.rows) - args.limit
        answer = dataclasses.replace(
            answer,
            rows=answer.rows[: args.limit],
            warnings=answer.warnings
            + (
                f"showing at most {args.limit} of {len(answer.rows)} Canonical "
                f"Models; {dropped} more are offered",
            ),
        )

    output_format = "json" if getattr(args, "json", False) else args.format
    if output_format == "json":
        _print_rendered(_json.dumps(answer.as_dict(), indent=2) + "\n", mapping)
    elif output_format == "markdown":
        _print_rendered(guidance_module.render_markdown(answer), mapping)
    else:
        _print_rendered(guidance_module.render_text(answer), mapping)
    return 0


def cmd_policy(args: argparse.Namespace) -> int:
    """Record one operator decision into Policy.

    The Operator Surface: the only writer of Policy other than an editor
    (ADR 0003). Every verb takes the lock, refuses when Policy changed on
    disk since it was read, validates the result before promoting it, and
    prints the diff it applied. The run path never reaches this code.

    Exit 0 when the decision was recorded, and also when it was already
    recorded — a decision already in force is not an error. Exit 1 on a
    refusal.
    """
    from litellm_maintainer import operator_surface

    env_path = Path(args.env) if args.env else _default_env_path()
    mapping = build_redaction_map(env_path)
    home = Path(args.home) if args.home else None
    policy_file = Path(args.policy)

    verbs = {
        "approve-candidate": lambda: operator_surface.approve_candidate(
            policy_file, args.offering_id, home=home, dry_run=args.dry_run
        ),
        "withhold": lambda: operator_surface.withhold(
            policy_file, args.offering_id, args.reason, home=home, dry_run=args.dry_run
        ),
        "unwithhold": lambda: operator_surface.unwithhold(
            policy_file, args.offering_id, home=home, dry_run=args.dry_run
        ),
        "set-alias": lambda: operator_surface.set_alias(
            policy_file, args.offering_id, args.alias, home=home, dry_run=args.dry_run
        ),
        "set-entitlement": lambda: operator_surface.set_entitlement(
            policy_file,
            args.provider_id,
            args.entitlement,
            home=home,
            dry_run=args.dry_run,
        ),
    }

    try:
        edit = verbs[args.verb]()
    except (operator_surface.OperatorSurfaceError, PolicyError) as exc:
        print(redact(str(exc), mapping), file=sys.stderr)
        return 1
    except FileNotFoundError:
        print(
            redact(f"No Policy at {policy_file}. Run init first.", mapping),
            file=sys.stderr,
        )
        return 1

    print(redact(edit.message, mapping))
    if edit.diff:
        print(redact(edit.diff, mapping), end="" if edit.diff.endswith("\n") else "\n")
    return 0


def cmd_init(args: argparse.Namespace) -> int:
    """Write a starter Policy derived from the Feed's own providers.

    The Feed states each provider's id, base URL and credential hint, so
    the starter Policy names real providers and the exact environment
    variables to set. It never writes a credential. Exit 1 when a Policy
    already exists and `--force` was not given: overwriting the
    operator's own decisions must be deliberate.
    """
    from litellm_maintainer.initialize import build_starter_policy, write_starter_policy
    from litellm_maintainer.paths import policy_path

    env_path = Path(args.env) if args.env else _default_env_path()
    mapping = build_redaction_map(env_path)
    home = Path(args.home) if args.home else None

    try:
        feed = load_feed(Path(args.feed))
    except Exception as exc:  # noqa: BLE001 - a read failure reports, it never crashes
        print(redact(f"Could not read the Feed: {exc}", mapping), file=sys.stderr)
        return 1

    destination = Path(args.out) if args.out else policy_path(home)
    starter = build_starter_policy(feed, alias_prefix=args.alias_prefix)

    try:
        write_starter_policy(starter, destination, force=args.force)
    except FileExistsError:
        print(
            redact(
                f"{destination} already exists. Nothing was written. "
                "Pass --force to replace it, but read it first: it holds your "
                "own decisions.",
                mapping,
            ),
            file=sys.stderr,
        )
        return 1

    print(
        redact(
            f"Wrote {destination}: {starter.provider_count} providers, "
            "every one commented.",
            mapping,
        )
    )
    for note in starter.notes:
        print(redact(f"  next: {note}", mapping))
    return 0


#: Where a litellm proxy's configs live when the operator names no
#: other directory. `doctor` reads every `*.yaml` here to find out
#: whether the Observation Journal callback is registered.
DEFAULT_SERVED_CONFIG_DIR = Path.home() / ".config" / "litellm"


def _read_served_configs(directory: Path | None) -> dict[str, tuple[bool, bool]]:
    """Map each served proxy config to `(registered, generated)`.

    `registered` says the file registers the Observation Journal
    callback. `generated` says it carries the Generator's own header,
    which is what identifies the MAIN proxy's config: the maintainer
    writes that file and nothing else in this directory. A worker
    proxy serves a hand-written file (`docs/gotchas.md`, "a worker
    proxy does not serve config.yaml"), so the header separates the two
    without a hand-maintained list of names.

    Read every `*.yaml` in `directory`. Return an empty mapping when
    the directory does not exist: an operator whose proxy lives
    somewhere else has nothing to answer for here, and a check that
    cannot measure must not fail (the same rule
    `doctor._litellm_patch_checks` applies).

    Skip a file that does not parse. A config the proxy itself cannot
    load is a different fault, and `doctor` must not report it as a
    missing callback.
    """
    from litellm_maintainer.doctor import JOURNAL_CALLBACK
    from litellm_maintainer.generate import HEADER

    generated_marker = HEADER.splitlines()[0]

    base = directory if directory is not None else DEFAULT_SERVED_CONFIG_DIR
    if not base.is_dir():
        return {}

    found: dict[str, tuple[bool, bool]] = {}
    for path in sorted(base.glob("*.yaml")):
        try:
            text = path.read_text()
            document = yaml.safe_load(text)
        except Exception:
            continue
        if not isinstance(document, dict):
            continue
        settings = document.get("litellm_settings")
        callbacks = settings.get("callbacks") if isinstance(settings, dict) else None
        registered = isinstance(callbacks, list) and JOURNAL_CALLBACK in callbacks
        found[str(path)] = (registered, text.startswith(generated_marker))
    return found


def _tick_plist_path(args: argparse.Namespace) -> Path | None:
    """Return where the launchd tick's plist would live, or `None`.

    `None` means the LaunchAgents directory is not there to look in, so
    the check reports "not checked" rather than "not installed".
    """
    from litellm_maintainer.schedule import DEFAULT_LABEL, plist_path

    target = Path(getattr(args, "target_dir", None) or Path.home() / "Library" / "LaunchAgents")
    if not target.is_dir():
        return None
    return plist_path(target, DEFAULT_LABEL)


def _headroom_installed_interval_seconds(args: argparse.Namespace) -> tuple[int | None, str]:
    """The installed headroom-refresh job's baked `StartInterval`, and its path.

    Mirrors `_tick_plist_path`: `None` means either no job is installed or
    the LaunchAgents directory could not be read, and `doctor` must not
    fail a check (`_headroom_interval_check`) it cannot measure.
    """
    from litellm_maintainer.schedule import HEADROOM_LABEL, plist_path

    target = Path(getattr(args, "target_dir", None) or Path.home() / "Library" / "LaunchAgents")
    if not target.is_dir():
        return None, ""
    path = plist_path(target, HEADROOM_LABEL)
    if not path.exists():
        return None, str(path)
    try:
        document = plistlib.loads(path.read_bytes())
    except Exception:  # noqa: BLE001 - an unreadable plist is "not checked", not a failure
        return None, str(path)
    interval = document.get("StartInterval")
    if isinstance(interval, bool) or not isinstance(interval, int):
        return None, str(path)
    return interval, str(path)


def _headroom_stored_extra_window_ids(home: Path) -> dict[str, frozenset[str]]:
    """Which extra windows each Allowance's stored Reading publishes.

    Read from Headroom State, never from a live call. `doctor` uses it for
    exactly one purpose: to pass a `members` key that the live Reading
    omits and a stored Reading holds. codexbar drops an extra window and
    restores it between consecutive calls, and a check that failed on that
    flap would send the operator to correct a correct line.

    Returns an empty mapping when no state file exists. Never raises: an
    unreadable state file leaves the check exactly as strict as it was.
    """
    from litellm_maintainer.headroom import read_headroom
    from litellm_maintainer.paths import headroom_path

    try:
        state = read_headroom(headroom_path(home))
    except Exception:  # noqa: BLE001 - an unreadable state file weakens no check
        return {}
    return {
        allowance_id: frozenset(window.id for window in record.reading.extra_windows)
        for allowance_id, record in state.records.items()
    }


def _headroom_staleness_warnings(policy, home: Path | None) -> tuple[str, ...]:
    """Which mapped Allowances stopped refreshing, for `doctor`.

    Delegates to `headroom.headroom_source_warnings`, so `doctor` cannot
    disagree with the warning `guidance` prints beside the same figure.

    Returns `()` when Policy declares no source, and on an unreadable
    state file: a check that cannot measure must not fail.
    """
    from litellm_maintainer.headroom import headroom_source_warnings, read_headroom
    from litellm_maintainer.paths import headroom_path

    if not policy.headroom.sources:
        return ()
    try:
        state = read_headroom(headroom_path(home))
    except Exception:  # noqa: BLE001 - an unreadable state file fails no check
        return ()
    return headroom_source_warnings(
        headroom_policy=policy.headroom,
        headroom_state=state,
        now=datetime.now(timezone.utc),
    )


def _headroom_readings(policy) -> tuple[tuple | None, str | None]:
    """Ask codexbar for every mapped provider's CURRENT identity.

    This is `doctor`'s one extra network call, made only when Policy
    declares a `headroom` source at all -- a machine with the capability
    off makes it never. It exists because Headroom State on disk cannot
    answer whether a mapping still matches: `refresh_headroom` keeps a
    stale Reading under its Allowance forever once one match ever
    succeeded, so only asking codexbar again can say a mapping stopped
    working (see `doctor._headroom_mapping_checks`).

    Returns `(readings, run_error)`. `readings` is `None` -- never raises
    -- when Policy declares no source, the binary is missing, or the run
    fails; `run_error` is `None` in the first two cases (nothing to check,
    or `doctor._headroom_binary_check` already names the missing binary)
    and states why in the third, so `doctor._headroom_run_check` can turn
    a failed run into its own Check instead of the silent "no check at
    all" a machine with the capability off correctly gets. An empty tuple
    would read as "matched nothing", which is a real finding; `None` is
    the only value that means "could not measure".

    A provider named in `policy.headroom.all_accounts_providers` gets its
    own `--all-accounts` call, exactly as `refresh_headroom` gives it
    (ticket 11): `query_codexbar_readings` isolates the two calls, but
    this diagnosis reports the SAME "could not measure" for either one
    failing, rather than a partial merge -- a live check that could not
    fully answer must not let `_headroom_mapping_checks` read the gap as
    "this Allowance's mapping rotted".
    """
    from litellm_maintainer.headroom import (
        provider_id_from_source,
        query_codexbar_readings,
        real_codexbar_runner,
    )

    if not policy.headroom.sources:
        return None, None
    if shutil.which(policy.headroom.command) is None:
        return None, None
    provider_ids = sorted(
        {provider_id_from_source(source) for source in policy.headroom.sources.values()}
    )
    runner = real_codexbar_runner(policy.headroom.command, timeout=policy.headroom.timeout_seconds)
    document, failed_providers, call_errors = query_codexbar_readings(
        provider_ids=provider_ids,
        all_accounts_providers=frozenset(policy.headroom.all_accounts_providers),
        runner=runner,
    )
    if failed_providers:
        return None, "; ".join(call_errors)
    return document.readings, None


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report every reason this instance is not working.

    Read-only: writes no file. Two network calls, both read-only: the
    proxy liveliness check `run` already makes, and — only when Policy
    declares a `headroom` source — one call to `headroom.command` to
    check whether the mapping still matches (`_headroom_readings`).
    Headroom State on disk cannot answer that: it keeps a stale Reading
    forever once one match ever succeeded, so only asking codexbar again
    can say a mapping stopped working. A machine that declares no
    `headroom` source makes this call never. Exit 0 when every check
    passed, 1 when any failed, so a script can gate on it.
    """
    import json as _json

    from litellm_maintainer import doctor as doctor_module
    from litellm_maintainer.fetch import read_feed_document_metadata
    from litellm_maintainer.paths import feed_document_path, health_path, policy_path

    env_path = Path(args.env) if args.env else _default_env_path()
    mapping = build_redaction_map(env_path)
    home = Path(args.home) if args.home else None

    resolved_policy = Path(args.policy) if args.policy else policy_path(home)
    try:
        policy = load_policy(resolved_policy)
    except FileNotFoundError:
        print(
            redact(
                f"No Policy at {resolved_policy}. Run init to write one, or "
                "pass --policy.",
                mapping,
            ),
            file=sys.stderr,
        )
        return 1
    except PolicyError as exc:
        print(redact(f"Policy is invalid: {exc}", mapping), file=sys.stderr)
        return 1

    feed_path = Path(args.feed) if args.feed else feed_document_path(home)
    try:
        feed = load_feed(feed_path)
    except Exception as exc:  # noqa: BLE001 - an unreadable Feed is a diagnosis, not a crash
        print(
            redact(
                f"Could not read the Feed Document at {feed_path}: {exc}. "
                "Run fetch, or pass --feed.",
                mapping,
            ),
            file=sys.stderr,
        )
        return 1

    tick_plist = _tick_plist_path(args)
    headroom_interval_seconds, headroom_plist_path = _headroom_installed_interval_seconds(args)
    headroom_readings, headroom_run_error = _headroom_readings(policy)

    checker = functools.partial(_live_proxy_check, args.proxy_base)
    diagnosis = doctor_module.diagnose(
        policy=policy,
        feed=feed,
        health=read_health(health_path(home)).offerings,
        feed_document_metadata=read_feed_document_metadata(feed_path),
        # Not `os.environ` alone. This project resolves a credential from
        # the process environment OR the `.env.local`-style file, because
        # the operator's shell does not export every variable that file
        # defines (see `_credential_resolver`). Checking only the process
        # environment made `doctor` report every provider as missing a
        # credential while `probe` reached all of them.
        environ=_credential_environment(env_path),
        proxy_ok=checker(),
        now=datetime.now(timezone.utc),
        # The proxy usually runs a different litellm from this package's
        # own, so locate the tree the proxy's own interpreter imports
        # rather than reading the imported module. `--litellm-path` names
        # it directly when the automatic lookup cannot.
        litellm_patches=inspect_patches(
            Path(args.litellm_path) if args.litellm_path else litellm_source_root()
        ),
        served_configs=_read_served_configs(
            Path(args.served_config_dir) if args.served_config_dir else None
        ),
        tick_installed=tick_plist.exists() if tick_plist is not None else None,
        tick_plist_path=str(tick_plist) if tick_plist is not None else "",
        # The one call to codexbar this command makes, only when Policy
        # declares a headroom source at all; see `_headroom_readings`.
        headroom_readings=headroom_readings,
        headroom_run_error=headroom_run_error,
        headroom_binary_present=(
            bool(policy.headroom.sources) and shutil.which(policy.headroom.command) is not None
        ),
        headroom_installed_interval_seconds=headroom_interval_seconds,
        headroom_plist_path=headroom_plist_path,
        # Headroom State's own view of which extra windows exist. Read
        # only to keep a `members` check from failing on a window codexbar
        # drops between calls; see `_headroom_mapping_checks`.
        headroom_stored_extra_window_ids=_headroom_stored_extra_window_ids(home),
        # The same answer `guidance` and `entitlements` publish as warnings,
        # rendered here as a Check. One derivation, two surfaces.
        headroom_staleness_warnings=_headroom_staleness_warnings(policy, home),
    )

    if getattr(args, "json", False):
        print(redact(_json.dumps(diagnosis.as_dict(), indent=2), mapping))
    else:
        print(redact(doctor_module.render_text(diagnosis), mapping), end="")
    return 0 if diagnosis.ok else 1


def cmd_status(args: argparse.Namespace) -> int:
    """Print what is offered, Excluded, Withheld, Sunsetting, and awaiting approval.

    Read-only: makes no network call, writes no file. See `report.py`.
    Exit 0 on success. Exit 1 when the Policy is invalid or the Feed
    could not be read, printing why.

    `status` runs `plan` itself, over the Feed and Policy given and the
    Health State already on disk, so the picture it prints is the same
    one `generate` would act on right now. It never writes Health
    State (see `litellm_maintainer.health`, "written only by this
    path") and never writes the Generated Config.
    """
    import json as _json

    from litellm_maintainer.paths import health_path
    from litellm_maintainer.report import status_document

    env_path = Path(args.env) if args.env else _default_env_path()
    mapping = build_redaction_map(env_path)
    home = Path(args.home) if args.home else None

    # The same loader `guidance` and `entitlements` use, so all three
    # resolve a missing `--feed` or `--policy` the same way and report a
    # read failure in the same words.
    loaded = _load_read_inputs(args, mapping)
    if loaded is None:
        return 1
    policy, feed, _health, report, warnings, now = loaded

    health_state = read_health(health_path(home))

    if args.json:
        document = status_document(
            policy=policy,
            health=health_state.offerings,
            report=report,
            feed=feed,
            now=now,
        )
        document["warnings"] = list(warnings)
        print(redact(_json.dumps(document, indent=2), mapping))
        return 0

    # A stale Feed Document makes every section below describe a
    # catalogue that has moved on. Warn before the picture, never after
    # it: a warning under 80 lines of status is a warning nobody reads.
    for warning in warnings:
        print(redact(f"warning: {warning}", mapping))

    print_status(
        policy=policy,
        health=health_state.offerings,
        report=report,
        feed=feed,
        now=now,
        mapping=mapping,
        out=sys.stdout,
    )
    return 0


def fetch_served_aliases(
    base_url: str, *, credential: str | None = None, timeout: float = 5.0
) -> tuple[frozenset[str] | None, str]:
    """Return the Alias set the running proxy serves, and a note.

    `(None, why)` when the proxy cannot be asked. An absent answer is
    NOT a negative answer: reading a refused call as "the Alias is not
    served" is the error `explain` exists to catch, so this never
    returns an empty set for a failure.

    `credential` is the proxy's own inbound master key, which
    `/v1/models` requires. The note carries a status code and never a
    header value, so no key can reach the transcript.
    """
    import json as _json
    import urllib.error
    import urllib.request

    url = f"{base_url.rstrip('/')}/v1/models"
    request = urllib.request.Request(url)  # noqa: S310
    if credential:
        request.add_header("Authorization", f"Bearer {credential}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            payload = _json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return None, f"the proxy answered HTTP {exc.code} for /v1/models"
    except Exception as exc:  # noqa: BLE001 - any transport failure is UNKNOWN, never absent
        return None, f"the proxy could not be reached ({type(exc).__name__})"

    entries = payload.get("data")
    if not isinstance(entries, list):
        return None, "the proxy's /v1/models answer carried no 'data' list"
    aliases = {
        entry.get("id")
        for entry in entries
        if isinstance(entry, dict) and isinstance(entry.get("id"), str)
    }
    return frozenset(aliases), ""


def installed_out_path(target_dir: Path | None = None) -> tuple[Path | None, str]:
    """Return the config path the installed tick writes, and its source.

    The tick's plist carries `--out`, and it is the only record on this
    machine of which file the proxy actually reads. `deploy` resolves
    the path from there so the operator cannot write the right config to
    the wrong file.

    `(None, why)` when no job is installed or the plist cannot be read.
    """
    from litellm_maintainer.schedule import DEFAULT_LABEL, plist_path

    target = Path(target_dir or Path.home() / "Library" / "LaunchAgents")
    if not target.is_dir():
        return None, "no LaunchAgents directory to read"
    path = plist_path(target, DEFAULT_LABEL)
    if not path.exists():
        return None, "no tick job is installed"
    try:
        document = plistlib.loads(path.read_bytes())
    except Exception:  # noqa: BLE001 - an unreadable plist resolves nothing
        return None, f"{path} could not be read"
    arguments = document.get("ProgramArguments")
    if not isinstance(arguments, list):
        return None, f"{path} states no ProgramArguments"
    for index, value in enumerate(arguments):
        if value == "--out" and index + 1 < len(arguments):
            return Path(str(arguments[index + 1])), str(path)
    return None, f"{path} passes no --out"


def cmd_deploy(args: argparse.Namespace) -> int:
    """Apply pending changes to the proxy now, whatever the interval says.

    Warning: this restarts the proxy. Writing the Generated Config is
    what deploys it, because that file is the one the litellm `--reload`
    watcher reads, and a restart ends every session in flight. Under ADR
    0014 a write is a rare, deliberate act, so the operator picks the
    moment. The command states this before it writes.

    Exit 0 when the config was written, and when it was already current.
    Exit 1 on any refusal, having written nothing.

    This runs the safety rail and does not imply `--force`. It takes the
    maintainer lock (ADR 0002), so it can never run beside a tick.
    """
    from litellm_maintainer.paths import feed_document_path, policy_path

    home = Path(args.home) if args.home else None
    # `generate` requires both paths; `deploy` defaults them to the
    # instance directory, as `status`, `guidance` and `entitlements` do.
    if not args.policy:
        args.policy = str(policy_path(home))
    if not args.feed:
        args.feed = str(feed_document_path(home))

    resolved_out = Path(args.out) if args.out else None
    source = "--out"
    if resolved_out is None:
        resolved_out, source = installed_out_path(getattr(args, "target_dir", None))
        if resolved_out is None:
            print(
                f"Refused to deploy: no --out given and {source}, so the file "
                "the proxy reads is unknown. Pass --out with that path.",
                file=sys.stderr,
            )
            return 1
        print(f"Writing the config the installed tick writes: {resolved_out}")

    # Warning first, then the action. An unchanged config is not
    # written at all, so this states the consequence of a CHANGE rather
    # than promising a restart the next line may then deny.
    print(
        "Warning: a change to the Generated Config restarts the proxy and "
        "ends every session in flight."
    )

    args.out = str(resolved_out)

    # `generate` holds no lock, because it is a convenience command a
    # human runs. `deploy` writes the file the proxy reads, so a tick
    # writing the same file at the same moment must be impossible.
    from litellm_maintainer.paths import lock_path

    try:
        with maintainer_lock(lock_path(home)):
            return cmd_generate(args)
    except LockBusy:
        print(
            "Refused to deploy: another maintainer process holds the lock. "
            "A tick is running; try again when it finishes.",
            file=sys.stderr,
        )
        return 1


def cmd_explain(args: argparse.Namespace) -> int:
    """Name the stage that stopped one Offering reaching a client.

    Read-only. It writes no file and changes no Health State. It makes
    one call to the running proxy unless `--no-proxy` is passed, and a
    failed call reports that stage UNKNOWN rather than absent.

    Exit 0 whenever the walk completed, including when the walk found a
    stop. A stop is an answer, not a failure of the command. Exit 1 only
    when the Policy or the Feed could not be read.
    """
    import json as _json

    from litellm_maintainer.explain import DECISION, PASSED, STOPPED, UNKNOWN, explain
    from litellm_maintainer.paths import health_path

    env_path = Path(args.env) if args.env else _default_env_path()
    mapping = build_redaction_map(env_path)
    home = Path(args.home) if args.home else None

    loaded = _load_read_inputs(args, mapping)
    if loaded is None:
        return 1
    policy, feed, _health, report, warnings, _now = loaded

    health_state = read_health(health_path(home))

    served: frozenset[str] | None = None
    proxy_note = "not asked (--no-proxy)"
    if not args.no_proxy:
        # The proxy's own inbound credential, exactly as `smoke` sends
        # it. No provider key belongs here: this call reaches the proxy
        # and never a provider.
        master_key = _credential_resolver(env_path)("LITELLM_MASTER_KEY")
        served, proxy_note = fetch_served_aliases(
            args.proxy_base, credential=master_key
        )

    result = explain(
        query=args.target,
        feed=feed,
        policy=policy,
        health=health_state.offerings,
        report=report,
        served_aliases=served,
        proxy_note=proxy_note,
    )

    if args.json:
        document = result.as_dict()
        document["warnings"] = list(warnings)
        print(redact(_json.dumps(document, indent=2), mapping))
        return 0

    for warning in warnings:
        print(redact(f"warning: {warning}", mapping))

    marks = {PASSED: "ok  ", STOPPED: "STOP", UNKNOWN: "????"}
    print(redact(f"explain {result.query}", mapping))
    if result.alias:
        print(redact(f"  alias: {result.alias}", mapping))
    print("")
    for stage in result.stages:
        print(redact(f"  [{marks.get(stage.verdict, stage.verdict)}] {stage.name}", mapping))
        if stage.detail:
            print(redact(f"         {stage.detail}", mapping))
    print("")

    if result.stopped_at is None:
        if result.recommended:
            print("Reaches a client, and recommended.")
        else:
            print("Reaches a client, and NOT recommended.")
        for note in result.notes:
            print(redact(f"  {note}", mapping))
        return 0

    kind = "Decision" if result.stop_kind == DECISION else "Fault"
    print(redact(f"Stopped at {result.stopped_at} — {kind}.", mapping))
    print(redact(f"  {result.stop_detail}", mapping))
    if result.stop_kind == DECISION:
        print("  Nothing is broken. Change the construct above, or agree with it.")
    else:
        print("  This needs a repair.")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Check one Offering per distinct translation rule, through the proxy.

    Grouping and Offering choice come from `litellm_maintainer.smoke`;
    see its module docstring. `--dry-run` prints which rule would be
    checked with which Alias and calls nothing: it never builds a
    transport and never calls `run_smoke_check`. Otherwise it makes one
    real HTTP call per distinct rule to the running proxy, using
    `LITELLM_MASTER_KEY` from the environment.

    Exit 0 whether a rule reports FAILED, UNVERIFIED or INCONCLUSIVE:
    the smoke check reports loudly, it never refuses a write and never
    changes Health State. Exit 1 only when the Policy is invalid,
    printing a message that names the offending key.
    """
    env_path = Path(args.env) if args.env else _default_env_path()
    mapping = build_redaction_map(env_path)
    try:
        policy = load_policy(Path(args.policy))
    except PolicyError as exc:
        print(redact(f"Policy is invalid: {exc}", mapping), file=sys.stderr)
        return 1

    try:
        feed = load_feed(Path(args.feed))
    except Exception as exc:  # noqa: BLE001 - a read failure reports, it never crashes
        print(redact(f"Could not read the Feed: {exc}", mapping), file=sys.stderr)
        return 1
    entries = build_smoke_entries(feed=feed, policy=policy)
    grouped = group_by_rule(entries)

    from litellm_maintainer.paths import health_path

    home = Path(args.home) if args.home else None
    health_state = read_health(health_path(home))
    _report_skipped_health_records(health_state, mapping)

    print(redact(f"{len(grouped)} distinct translation rule(s) to check", mapping))
    for rule in sorted(grouped):
        chosen = pick_healthiest(grouped[rule], health=health_state.offerings)
        if chosen is None:
            if not any(entry.callable_by_proxy for entry in grouped[rule]):
                why = "every Offering is Passthrough Auth, the caller holds the credential"
            else:
                why = "no healthy Offering"
            print(redact(f"  {rule}: UNVERIFIED ({why})", mapping))
        else:
            print(redact(f"  {rule}: would call {chosen.alias}", mapping))

    if args.dry_run:
        return 0

    # Live path. The orchestrator decides whether this ever runs; it
    # makes one real HTTP call to the running proxy per distinct rule,
    # never a call to a provider directly. `LITELLM_MASTER_KEY` is read
    # from the environment here, never hardcoded, and never printed:
    # every line below goes through `redact` first.
    # Resolve through the same seam every other credential uses, NOT
    # `os.environ` alone. An unexported key sends every smoke call
    # unauthenticated, and the proxy answers "No api key passed in." --
    # which reads as a translation-rule failure rather than as the
    # missing credential it is.
    master_key = _credential_resolver(env_path)("LITELLM_MASTER_KEY")
    base_url = f"{args.proxy_base.rstrip('/')}/v1/chat/completions"
    transport = functools.partial(
        live_smoke_transport, base_url=base_url, credential=master_key
    )

    result = run_smoke_check(
        entries,
        health=health_state.offerings,
        transport=transport,
        now=lambda: datetime.now(timezone.utc),
    )

    print("Results:")
    for check in result.checks:
        print(redact(format_smoke_line(check), mapping))

    if result.failed:
        print(
            redact(
                f"WARNING: {len(result.failed)} translation rule(s) FAILED the smoke check.",
                mapping,
            ),
            file=sys.stderr,
        )
    return 0


# --- run: the scheduled tick ------------------------------------------------
#
# See `litellm_maintainer.schedule` for the pure `due` gate and
# `.scratch/maintainer-v1/spec-corrections.md`, correction 9: this command
# chains probe, reduce, then plan, in that order, and never plans alone.
# `generate` (above) is a convenience command only; a scheduled run must
# use this one.


def _read_last_run_at(run_log_path: Path) -> "datetime | None":
    """Read when this command last actually RAN, from the run log.

    Warning: count only a line whose marker is in `RUN_MARKERS`. A
    `skip:` line records a tick the gate turned away before any work,
    and it carries the timestamp of that tick. Counting one as a run
    means every 60-second tick refreshes `last_run_at`, so
    `now - last_run_at` never reaches `interval_minutes` and the
    pipeline never runs again. One skip used to wedge the tick
    permanently; the defect hid because no tick was installed to
    produce a second one.

    A `refused:` line DOES count. It records a tick that probed and
    then refused to write, so the work it must not repeat for another
    interval has already happened.

    A missing file, an empty file, or a file with no run line at all
    reads as `None` — "no previous run" — never an error: this is a
    convenience read of a log file, not Health State or Policy.
    """
    try:
        lines = run_log_path.read_text().splitlines()
    except FileNotFoundError:
        return None
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ", 2)
        if len(parts) < 2 or parts[1] not in RUN_MARKERS:
            continue
        try:
            return datetime.fromisoformat(parts[0])
        except ValueError:
            continue
    return None


#: Marks a run-log line that `_read_last_run_at` counts as a run.
#: `report.run_log_line` writes `run:`; a tick that did real work and
#: then refused writes `refused:`. A tick the gate turned away before
#: any work writes `skip:` and is NOT counted.
RUN_MARKERS = ("run:", "refused:")


def _append_tick_skip_line(
    run_log_path: Path,
    *,
    now: datetime,
    reason: str,
    mapping: dict[str, str],
    did_work: bool = False,
) -> None:
    """Append one run-log line stating why a tick did not finish.

    A real run instead calls `report.append_run_log`, which carries the
    full `PlanReport`; a tick with none to carry writes a shorter line
    in the same file, redacted the same way.

    Warning: `did_work` decides the marker, and the marker decides
    whether the NEXT tick sees this as a run. Get it wrong in either
    direction and the tick misbehaves at 60-second resolution:

    - Marking a gate refusal as work (`did_work=True`) stamps a fresh
      timestamp every tick, so `now - last_run_at` never reaches
      `interval_minutes` and the pipeline never runs again. This was a
      real defect: every skip line advanced `last_run_at`, so one skip
      wedged the tick permanently. It hid because no tick was installed.
    - Marking real work as a skip (`did_work=False`) lets the next tick
      probe again 60 seconds later, and again, which is the tick storm
      `schedule.due`'s interval rule exists to prevent
      (`docs/gotchas.md`: `Worker local total request limit reached`).

    Pass `did_work=True` from any point after the Prober has run, or
    after a refusal that already spent a provider call.
    """
    marker = "refused:" if did_work else "skip:"
    line = redact(f"{now.isoformat()} {marker} {reason}", mapping)
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(run_log_path, "a") as f:
        f.write(line + "\n")


def _live_proxy_check(base_url: str, *, timeout: float = 5.0) -> bool:
    """Whether the proxy answers `/health/liveliness`. The orchestrator
    decides when this ever runs: `cmd_run` calls it only when a test has
    not supplied its own `proxy_checker`, and never at all on
    `--dry-run`."""
    import httpx

    try:
        response = httpx.get(f"{base_url.rstrip('/')}/health/liveliness", timeout=timeout)
    except httpx.HTTPError:
        return False
    return response.status_code < 500


def _fetch_for_tick(
    args: argparse.Namespace,
    *,
    policy,
    mapping: dict[str, str],
    env_path: Path | None = None,
    transport=None,
) -> str | None:
    """Refresh the Feed Document for an unattended tick. Never fatal.

    Returns a note for the run log when the fetch failed or when Policy
    named no Feed to fetch, and `None` when a new document was promoted.
    A dry run fetches nothing and returns `None`.

    The download goes to the same path the tick then reads (`--feed`), so
    the run plans from the document it just refreshed.

    Warning: a Policy with no `feed` block leaves the tick planning on
    whatever document is on disk, forever. That is a supported mode — the
    operator refreshes the Feed themselves — but it is indistinguishable
    from a broken loop unless the tick says so. Measured 2026-07-27: a
    Policy carried no `feed` block for a full day of hourly ticks, every
    tick planned on one hand-fetched document, and no line in `runs.log`
    or `tick.out.log` named the reason. Hence the note.

    `env_path` is the `--env` file. The credential comes from that file
    MERGED with the process environment (`_credential_environment`), not
    from the process environment alone. launchd starts the tick with no
    shell profile, so a token the operator keeps in `.env.local` is not
    exported and `os.environ` alone resolves nothing. A Feed behind a
    bearer token then answers 401 on every tick, the fetch keeps the
    previous document, and the loop looks like it never fetches.
    """
    if getattr(args, "dry_run", False):
        return None
    if policy.feed is None:
        message = (
            "Policy names no 'feed' block, so this tick fetched nothing and "
            "planned on the Feed Document already on disk. Add feed.url to "
            "Policy to let the tick refresh it."
        )
        print(redact(message, mapping), file=sys.stderr)
        return "feed_not_configured: Policy names no 'feed' block"

    from litellm_maintainer.fetch import (
        fetch_feed_document,
        http_transport,
        resolve_credential,
    )

    outcome = fetch_feed_document(
        source=policy.feed,
        destination=Path(args.feed),
        transport=transport or http_transport(),
        providers_configured=bool(policy.providers),
        token=resolve_credential(policy.feed, _credential_environment(env_path)),
    )
    if outcome.promoted:
        print(redact(f"Fetched the Feed: {outcome.message}", mapping))
        return None

    print(
        redact(
            f"Fetch failed, planning on the previous Feed Document: {outcome.message}",
            mapping,
        ),
        file=sys.stderr,
    )
    return f"fetch_failed: {outcome.message}"


def cmd_run(
    args: argparse.Namespace,
    *,
    proxy_checker=None,
    probe_transport=None,
    notifier=None,
    clock=None,
    smoke_transport=None,
    fetch_transport=None,
) -> int:
    """Run one scheduled tick: the due gate, then the real pipeline.

    `due` (`litellm_maintainer.schedule`) decides whether this tick
    runs. When it does not, this command appends one run-log line
    naming why and exits 0: a quiet tick is not a failure.

    When it does run, the pipeline chains, in order: refuse an
    implausible Feed, probe what is stale, reduce the outcomes (and the
    Observation Journal) into Health State and write it, plan with that
    real Health State (never an empty one — correction 9), the safety
    rail, snapshot and write, the proxy smoke check, deploy the
    provider modules (content-compare only), append the run log, and
    notify only when `notify.detect_events` finds news.

    Defect 5: this command used to skip two safety steps `cmd_generate`
    applied -- the implausible-Feed refusal and the post-write smoke
    check -- because it is the SCHEDULED path, the one that runs
    unattended, drift here is the most dangerous kind. Both now run the
    same shared sequence (`_implausible_feed_refusal`,
    `_apply_safety_rail_and_write`), so `generate` and `run` cannot
    drift apart again. The smoke check reports loudly and never blocks
    the write and never changes Health State; a smoke
    failure still surfaces as a notification, through the same
    `proxy_ok` gate `detect_events` already fires on for a failed proxy
    check.

    `--dry-run` reports the schedule's own inputs (last run time,
    Health State's age) and returns 0. It calls `proxy_checker`,
    `probe_transport`, `smoke_transport` and every network or write step
    below not at all: dry-run reads Policy, the Feed and Health State
    from disk and prints what it read, nothing else.

    `proxy_checker`, `probe_transport`, `smoke_transport`, `notifier`
    and `clock` are dependency-injection seams for tests, matching the
    shape `litellm_maintainer.prober` already uses (a transport and a
    clock). Each defaults to the real thing — an HTTP check, the real
    provider transport, the real smoke transport,
    `notify.default_notifier`, and the real clock — used only when the
    operator runs `run` for real, never in a test.
    """
    from litellm_maintainer.paths import (
        ensure_instance_dirs,
        health_path,
        journal_path,
        run_log_path,
    )

    env_path = Path(args.env) if args.env else _default_env_path()
    mapping = build_redaction_map(env_path)
    home = Path(args.home) if args.home else None

    now_fn = clock or (lambda: datetime.now(timezone.utc))
    now = now_fn()

    try:
        policy = load_policy(Path(args.policy))
    except PolicyError as exc:
        print(redact(f"Policy is invalid: {exc}", mapping), file=sys.stderr)
        return 1

    ensure_instance_dirs(home)
    h_path = health_path(home)
    prior_health = read_health(h_path)
    _report_skipped_health_records(prior_health, mapping)
    log_path = run_log_path(home)
    last_run_at = _read_last_run_at(log_path)
    age = health_state_age(prior_health.offerings, now=now)

    # Read the Journal before the gate, not after. A failure the proxy
    # recorded while serving real traffic elapses the interval on its
    # own (`schedule.due`, `journal_pending`); without this read the
    # gate cannot see it, and an entry recorded one minute into a
    # 60-minute interval would wait 59 minutes to reach Health State.
    pending_observations = read_observations(journal_path(home)).observations
    journal_pending = bool(pending_observations)

    if args.dry_run:
        print(redact(f"Schedule: {policy.schedule}", mapping))
        print(redact(f"Last run: {last_run_at.isoformat() if last_run_at else 'never'}", mapping))
        print(
            redact(
                f"Health State age: {age if age is not None else 'never recorded'}",
                mapping,
            )
        )
        print(
            "Dry run: would check the proxy, evaluate the due gate, then chain "
            "probe, reduce, plan, the safety rail and deploy. No network call "
            "made, no file written."
        )
        return 0

    # A launchd tick fires far more often than most Policy would ever run.
    # Check the cheapest, optimistic case (proxy up) first: when even that
    # would not run — a disabled schedule, or an interval not yet elapsed —
    # no real proxy state changes the answer, so skip the network call
    # entirely. `due` only ever turns a run into a skip when the proxy is
    # required and down, never the reverse, so this pre-check is sound.
    optimistic = due(
        schedule=policy.schedule,
        last_run_at=last_run_at,
        proxy_up=True,
        health_age=age,
        now=now,
        journal_pending=journal_pending,
    )
    if not optimistic.run:
        _append_tick_skip_line(log_path, now=now, reason=optimistic.reason, mapping=mapping)
        print(redact(f"Skipped: {optimistic.reason}", mapping))
        return 0

    checker = proxy_checker or functools.partial(_live_proxy_check, args.proxy_base)
    proxy_up = checker()

    decision = due(
        schedule=policy.schedule,
        last_run_at=last_run_at,
        proxy_up=proxy_up,
        health_age=age,
        now=now,
        journal_pending=journal_pending,
    )
    if not decision.run:
        _append_tick_skip_line(log_path, now=now, reason=decision.reason, mapping=mapping)
        print(redact(f"Skipped: {decision.reason}", mapping))
        return 0

    catch_up_note = " (catch-up)" if decision.catch_up else ""
    print(redact(f"Running: {decision.reason}{catch_up_note}", mapping))

    # Refresh the Feed Document before planning, when Policy names a
    # feed source. A failed fetch is NOT fatal here: the tick runs
    # unattended, and a network problem must not be able to shrink the
    # Generated Config. `fetch_feed_document` promotes nothing unless the
    # download parses and is plausible, so the previous document is
    # always a valid one to plan from. See ADR 0005 and ticket 08.
    #
    # A journal-triggered run skips the fetch. It reacts to a failure
    # the proxy just served, which says nothing about the Feed, and it
    # can fire minutes after the last run. The Feed is republished once
    # a day, so a fetch here would download an unchanged document at the
    # rate the operator happens to hit failures. `policy.feed`'s own
    # staleness rules still govern the ordinary tick, which is the run
    # that exists to notice a new Feed.
    if decision.journal_triggered:
        fetch_note = None
    else:
        fetch_note = _fetch_for_tick(
            args,
            policy=policy,
            mapping=mapping,
            env_path=env_path,
            transport=fetch_transport,
        )

    try:
        feed = load_feed(Path(args.feed))
    except Exception as exc:  # noqa: BLE001 - a read failure refuses, it never crashes
        print(redact(f"Refused to run: the Feed could not be read ({exc}).", mapping), file=sys.stderr)
        return 1

    implausible_feed_refusal = _implausible_feed_refusal(feed=feed, policy=policy, force=args.force)
    if implausible_feed_refusal is not None:
        print(redact(implausible_feed_refusal, mapping), file=sys.stderr)
        _append_tick_skip_line(
            log_path,
            now=now,
            reason=f"refused: {implausible_feed_refusal}",
            mapping=mapping,
            # The Feed was already downloaded above. Repeating that
            # every 60 seconds while this refusal stands is the waste
            # the interval exists to prevent.
            did_work=True,
        )
        return 1

    # 1. Probe what is stale. The live transport authenticates per
    # provider (`_probe_live_transport`); the proxy's master key would
    # fail at every provider.
    worklist = build_worklist(feed=feed, policy=policy, health=prior_health, now=now)
    if decision.journal_triggered:
        worklist = _confirming_worklist(
            worklist,
            feed=feed,
            policy=policy,
            health=prior_health,
            observations=pending_observations,
            now=now,
        )
        print(
            redact(
                f"Journal-triggered run: {len(worklist.targets)} confirming "
                f"Probe(s), no sweep.",
                mapping,
            )
        )
    transport = probe_transport or _probe_live_transport(feed, env_path)
    outcomes = probe_offerings(worklist.targets, pacing=policy.pacing, transport=transport, now=now_fn)

    # 2. Reduce the outcomes into Health State, and write it. Fold in
    # the Observation Journal too: a failure the proxy recorded must
    # reach Health State, not just what this tick itself probed. Rotate
    # the Journal only after `reduce`'s result is safely written, per
    # ADR 0001.
    # The proxy records the ALIAS; Health State keys a Discovered
    # Offering by its Offering id, so resolve the keys first
    # (`journal.observation_key_map`) or the entry silently changes
    # nothing.
    try:
        next_health = _fold_into_health_state(
            home=home,
            outcomes=outcomes,
            admitted=worklist.admitted,
            passthrough_auth=frozenset(worklist.skipped_passthrough),
            now=now,
            key_map=observation_key_map(feed=feed, policy=policy),
            pool_siblings=entitlement_pool_siblings(feed=feed, policy=policy),
            sub_allowances=sub_allowance_keys(policy),
        )
        _report_unclassified_observations(pending_observations, mapping)
        _report_unproductive_offerings(next_health, mapping)
    except LockBusy as busy:
        # Another maintainer is already doing this work. That is not an
        # error: the tick fires more often than the interval on purpose.
        # Three positional arguments used to be passed here, to a
        # keyword-only signature, and the first was the instance
        # directory rather than the run log. So the unattended tick
        # raised TypeError on the one path this handler exists to make
        # quiet. See `tests/test_tick_lock_contention.py`.
        _append_tick_skip_line(
            log_path,
            now=now,
            reason=f"another maintainer is running: {busy}",
            mapping=mapping,
            # The Prober already ran above; those provider calls must
            # not repeat on the next tick.
            did_work=True,
        )
        print(redact(f"Skipped: {busy}", mapping))
        return 0

    # 3. Plan with the REAL Health State this run just wrote, never a
    # stale one: `cmd_generate` also reads real Health State from disk,
    # but reading a run BEFORE probing means Sunsetting cannot yet see
    # a success this run's own Probe just recorded (correction 9). This
    # command chains probe, reduce, then plan, so `plan` always sees
    # this run's Probe results.
    result = plan(feed=feed, policy=policy, health=next_health.offerings, now=now)

    # 4. The safety rail, then write, snapshot and prune -- the exact
    # sequence `cmd_generate` applies, factored once so the two commands
    # cannot drift apart again (defect 5).
    out_path = Path(args.out)
    exit_code, new_count, _dropped_aliases, written = _apply_safety_rail_and_write(
        result=result,
        out_path=out_path,
        home=home,
        env_path=env_path,
        mapping=mapping,
        now=now,
        policy=policy,
        force=args.force,
    )
    if exit_code != 0:
        reason = result.refusal if result.refusal is not None else "refused: safety rail"
        _append_tick_skip_line(
            log_path,
            now=now,
            reason=f"refused: {reason}",
            mapping=mapping,
            # The Prober already ran above.
            did_work=True,
        )
        return exit_code

    # 5. The proxy smoke check: one call per distinct translation rule,
    # through the running proxy, using the Health State this run just
    # wrote to pick the healthiest Offering per rule. Reports loudly,
    # changes nothing, blocks nothing (defect 5 -- this was the second
    # safety step `cmd_run` skipped that `cmd_smoke` alone applied).
    #
    # A journal-triggered run runs NO smoke check. This is what stops a
    # runaway loop, and the loop is not hypothetical: measured
    # 2026-07-27, the tick ran 7 times in 7 minutes. The smoke check
    # calls the proxy, the proxy's failure callback records those calls
    # in the Journal, an unprocessed Journal entry makes the next tick
    # due at once (`schedule.due`, `journal_pending`), and that run
    # smoke-checks again. The maintainer was observing its own traffic
    # and re-triggering on it.
    #
    # Skipping here breaks the cycle structurally rather than throttling
    # it. An ordinary tick still smoke-checks; if one of its calls
    # fails, the next tick is journal-triggered, folds that entry in,
    # generates no proxy traffic of its own, and the chain stops after
    # exactly one extra run.
    #
    # It also fits what a journal-triggered run already is: narrow. It
    # skips the Feed fetch and the probe sweep for the same reason.
    if decision.journal_triggered:
        smoke_result = None
    else:
        smoke_entries = build_smoke_entries(feed=feed, policy=policy)
        smoke_transport_fn = smoke_transport or functools.partial(
            live_smoke_transport,
            base_url=f"{args.proxy_base.rstrip('/')}/v1/chat/completions",
            # Resolve through the same seam every other credential uses,
            # NOT `os.environ` alone. A launchd job exports nothing, so
            # reading the environment directly sent every smoke call
            # unauthenticated and the proxy answered "No api key passed
            # in." -- 36 Journal entries in 7 minutes, all of them the
            # maintainer failing to authenticate to itself.
            credential=_credential_resolver(env_path)("LITELLM_MASTER_KEY"),
        )
        smoke_result = run_smoke_check(
            smoke_entries,
            health=next_health.offerings,
            transport=smoke_transport_fn,
            now=now_fn,
        )
    if smoke_result is None:
        print("Journal-triggered run: no smoke check, so this run adds no proxy traffic.")
    else:
        for check in smoke_result.checks:
            print(redact(format_smoke_line(check), mapping))
        if smoke_result.failed:
            print(
                redact(
                    f"WARNING: {len(smoke_result.failed)} translation rule(s) FAILED "
                    "the smoke check.",
                    mapping,
                ),
                file=sys.stderr,
            )

    # 6. Deploy the provider modules, content-compare only.
    written_modules: list[Path] = []
    if args.provider_modules_source:
        target_dir = Path(args.provider_modules_target)
        written_modules = deploy_provider_modules(Path(args.provider_modules_source), target_dir)

    # 7. Append the run log, and notify only when there is news. A
    # smoke-check failure surfaces through the same `proxy_ok` gate
    # `detect_events` already fires "Proxy check failed" on: the smoke
    # check exists to catch exactly the fault a live proxy check cannot
    # (a stale proxy environment, an unregistered handler), so it is the
    # same news to the operator, through the one gate that already
    # exists rather than a second one invented here.
    from litellm_maintainer.paths import instance_home

    state_path = previous_run_state_path(instance_home(home))
    previous_run_state = read_previous_run_state(state_path)
    admitted = frozenset(result.report.admitted)
    candidates = frozenset(result.report.candidates)
    events = detect_events(
        previous=previous_run_state,
        admitted=admitted,
        candidates=candidates,
        previous_health=prior_health.offerings,
        health=next_health.offerings,
        # A run with no smoke check measured nothing about the proxy
        # beyond the liveliness probe, so it must not report a smoke
        # failure it never looked for.
        proxy_ok=proxy_up and not (smoke_result.failed if smoke_result else False),
        now=now,
    )
    notify_all(events, mapping=mapping, notifier=notifier or default_notifier)
    write_previous_run_state(state_path, admitted=admitted, candidates=candidates)

    append_run_log(
        log_path,
        now=now,
        report=result.report,
        notification_count=len(events),
        mapping=mapping,
        note=fetch_note,
    )

    config_note = (
        f"Wrote {new_count} Aliases to {out_path}."
        if written
        else (
            f"{out_path} already holds these {new_count} Aliases; nothing "
            "written, so the proxy did not reload."
        )
    )
    print(
        redact(
            f"{config_note} "
            f"{len(written_modules)} provider module(s) deployed. "
            f"{len(events)} notification(s).",
            mapping,
        )
    )
    return 0


# --- install / uninstall: the launchd tick job ------------------------------


def cmd_install(args: argparse.Namespace) -> int:
    """Write the launchd plist that ticks `run`. Never calls `launchctl`.

    Idempotent: running this twice writes the same file (fixed by
    label), leaving exactly one job. Prints the `launchctl` command
    that registers the job; never runs it — the orchestrator decides
    whether the job actually starts.
    """
    from litellm_maintainer.paths import instance_home
    from litellm_maintainer.schedule import default_log_paths

    resolved_home = args.home or str(instance_home(None))
    standard_out_path, standard_error_path = default_log_paths(resolved_home)

    spec = build_plist_spec(
        python_executable=args.python,
        policy_path=args.policy,
        feed_path=args.feed,
        home=args.home,
        out_path=args.out,
        env_path=str(Path(args.env).resolve()) if args.env else None,
        provider_modules_source=args.provider_modules_source,
        provider_modules_target=args.provider_modules_target,
        tick_seconds=args.tick_seconds,
        standard_out_path=standard_out_path,
        standard_error_path=standard_error_path,
    )
    path = install_plist(Path(args.target_dir), spec)
    print(f"Wrote {path}")
    if args.out is None:
        print(
            "WARNING: no --out given, so the tick writes the instance "
            f"directory's own copy ({_default_out_path()}). Pass --out with "
            "the config your proxy actually serves, or the tick will compute "
            "the right config every 60 seconds and write it where nothing "
            "reads it."
        )
    if args.env is None:
        print(
            "WARNING: no --env given. launchd runs a job from '/', and the "
            "default '.env.local' lookup is relative to the working "
            "directory, so the tick will resolve no credential and refuse "
            "every write. Pass --env with the absolute path to your "
            "credential file."
        )
    print(f"Logs: {standard_out_path} and {standard_error_path}")
    print(f"Run this to register the job: {launchctl_load_command(path)}")
    return 0


def cmd_uninstall(args: argparse.Namespace) -> int:
    """Remove the launchd plist. Safe when nothing is installed.

    Prints the `launchctl` command that unregisters the job first,
    since removing the file alone does not stop an already-running
    job; never runs the command itself.
    """
    from litellm_maintainer.schedule import DEFAULT_LABEL, plist_path

    target_dir = Path(args.target_dir)
    path = plist_path(target_dir, DEFAULT_LABEL)
    print(f"Run this first, to unregister the job: {launchctl_unload_command(path)}")

    removed = uninstall_plist(target_dir)
    if removed is None:
        print("Nothing was installed.")
    else:
        print(f"Removed {removed}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    from litellm_maintainer.guidance import AXES as GUIDANCE_AXES

    parser = argparse.ArgumentParser(prog="litellm-maintainer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="Validate a Policy file")
    validate_parser.add_argument("--policy", required=True, help="Path to the Policy file")
    validate_parser.add_argument(
        "--env", default=None, help="Path to a .env-style file to redact from output"
    )
    validate_parser.set_defaults(func=cmd_validate)

    generate_parser = subparsers.add_parser(
        "generate", help="Generate a config from a Feed and a Policy"
    )
    generate_parser.add_argument("--feed", required=True, help="Path to the Feed document")
    generate_parser.add_argument("--policy", required=True, help="Path to the Policy file")
    generate_parser.add_argument(
        "--out",
        default=str(_default_out_path()),
        help="Path to write the Generated Config to (default: %(default)s)",
    )
    generate_parser.add_argument(
        "--env", default=None, help="Path to a .env-style file to redact from output"
    )
    generate_parser.add_argument(
        "--home", default=None, help="Instance directory (default: $LITELLM_MAINTAINER_HOME)"
    )
    generate_parser.add_argument(
        "--force",
        action="store_true",
        help="Write anyway when only a safety-rail threshold refused (never a validation failure)",
    )
    generate_parser.set_defaults(func=cmd_generate)

    probe_parser = subparsers.add_parser(
        "probe", help="Probe Offerings and update Health State"
    )
    probe_parser.add_argument("--feed", required=True, help="Path to the Feed document")
    probe_parser.add_argument("--policy", required=True, help="Path to the Policy file")
    probe_parser.add_argument(
        "--home", default=None, help="Instance directory (default: $LITELLM_MAINTAINER_HOME)"
    )
    probe_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the worklist and make no network call",
    )
    probe_parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Probe every target in scope, ignoring freshness and any "
            "recorded reset time. Use it when a provider refills early."
        ),
    )
    probe_parser.add_argument(
        "--provider",
        default=None,
        help="Probe only this provider. Use it to keep a first live sweep cheap.",
    )
    probe_parser.add_argument(
        "--env", default=None, help="Path to a .env-style file to redact from output"
    )
    probe_parser.set_defaults(func=cmd_probe)

    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Download the Feed and write the Feed Document",
        description=(
            "Downloads the Feed named by Policy's 'feed.url' and promotes it "
            "to the Feed Document, but only after it parses and carries a "
            "plausible number of Offerings. A failed or truncated download "
            "leaves the previous document in place and exits 1.\n\n"
            "Use --url for your first fetch, before any Policy exists. After "
            "that, put the address in Policy's 'feed' block, because the "
            "scheduled run reads it from there.\n\n"
            "Examples:\n"
            "  litellm-maintainer fetch --url https://example.invalid/feed.json\n"
            "  litellm-maintainer fetch --policy ~/.config/litellm-maintainer/policy.yaml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    fetch_parser.add_argument(
        "--policy",
        default=None,
        help="Path to the Policy file, which names the Feed in its 'feed' block",
    )
    fetch_parser.add_argument(
        "--url",
        default=None,
        help="Fetch this address instead of the one in Policy. Use it for a first fetch.",
    )
    fetch_parser.add_argument(
        "--credential-env",
        default=None,
        help="With --url: the environment variable holding the Feed's bearer token",
    )
    fetch_parser.add_argument(
        "--out",
        default=None,
        help="Path to write the Feed Document to (default: <home>/feed.json)",
    )
    fetch_parser.add_argument(
        "--home", default=None, help="Instance directory (default: $LITELLM_MAINTAINER_HOME)"
    )
    fetch_parser.add_argument(
        "--env", default=None, help="Path to a .env-style file to redact from output"
    )
    fetch_parser.set_defaults(func=cmd_fetch)

    headroom_parser = subparsers.add_parser(
        "headroom",
        help="Capture Readings of each mapped Allowance",
        description=(
            "Reads codexbar and writes Headroom State: one Reading per "
            "Allowance Policy's 'headroom.sources' names. Nothing else "
            "reads that file yet.\n\n"
            "Example:\n"
            "  litellm-maintainer headroom refresh --policy policy.yaml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    headroom_verbs = headroom_parser.add_subparsers(dest="verb", required=True)

    headroom_refresh_verb = headroom_verbs.add_parser(
        "refresh", help="Ask codexbar for the mapped providers and update Headroom State"
    )
    headroom_refresh_verb.add_argument(
        "--policy",
        default=None,
        help="Path to the Policy file (default: <home>/policy.yaml)",
    )
    headroom_refresh_verb.add_argument(
        "--home",
        default=None,
        help="Instance directory (default: $LITELLM_MAINTAINER_HOME)",
    )
    headroom_refresh_verb.add_argument(
        "--env", default=None, help="Path to a .env-style file to redact from output"
    )
    headroom_refresh_verb.set_defaults(func=cmd_headroom)

    headroom_install_verb = headroom_verbs.add_parser(
        "install",
        help=(
            "Write the launchd plist that ticks 'headroom refresh'. A "
            "SEPARATE job from the tick's own 'install'. Never calls "
            "launchctl."
        ),
    )
    headroom_install_verb.add_argument(
        "--policy", required=True, help="Path the installed job passes to 'headroom refresh --policy'"
    )
    headroom_install_verb.add_argument(
        "--home", default=None, help="Path the installed job passes to 'headroom refresh --home'"
    )
    headroom_install_verb.add_argument(
        "--env",
        default=None,
        help=(
            "Path the installed job passes to 'headroom refresh --env', "
            "stored absolute. launchd runs from '/', so without this "
            "codexbar may resolve no credential."
        ),
    )
    headroom_install_verb.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable the plist invokes (default: %(default)s)",
    )
    headroom_install_verb.add_argument(
        "--target-dir",
        default=str(Path.home() / "Library" / "LaunchAgents"),
        help="Directory to write the plist into (default: %(default)s)",
    )
    headroom_install_verb.set_defaults(func=cmd_headroom)

    headroom_uninstall_verb = headroom_verbs.add_parser(
        "uninstall",
        help="Remove the headroom-refresh launchd plist. Safe when nothing is installed.",
    )
    headroom_uninstall_verb.add_argument(
        "--target-dir",
        default=str(Path.home() / "Library" / "LaunchAgents"),
        help="Directory the plist was written into (default: %(default)s)",
    )
    headroom_uninstall_verb.set_defaults(func=cmd_headroom)

    def add_read_arguments(target: argparse.ArgumentParser) -> None:
        """The arguments both read commands share.

        `--feed` is optional here, unlike on the older commands: `fetch`
        writes the Feed Document to a known path, so the common case
        needs no path at all.
        """
        # Both paths default to the instance directory. A read command is
        # the one an agent runs most often, and making it name two paths
        # it could derive is friction that invites a wrong path.
        target.add_argument(
            "--policy",
            default=None,
            help="Path to the Policy file (default: <home>/policy.yaml)",
        )
        target.add_argument(
            "--feed",
            default=None,
            help="Path to the Feed Document (default: <home>/feed.json)",
        )
        target.add_argument(
            "--home",
            default=None,
            help="Instance directory (default: $LITELLM_MAINTAINER_HOME)",
        )
        target.add_argument(
            "--format",
            choices=("text", "json", "markdown"),
            default="text",
            help="Output format (default: %(default)s)",
        )
        target.add_argument(
            "--json",
            action="store_true",
            help="Shorthand for --format json",
        )
        target.add_argument(
            "--env", default=None, help="Path to a .env-style file to redact from output"
        )

    entitlements_parser = subparsers.add_parser(
        "entitlements",
        help="Print what spending through each provider costs, and what answers now",
        description=(
            "Prints one entry per provider Policy names: whether it bills from "
            "one shared pool or per model, what it costs, how many Offerings "
            "answer right now, and the earliest time an exhausted one refills. "
            "Every Offering that is admitted but not currently offered is "
            "listed with its reason.\n\n"
            "Read-only: writes no file and makes no provider call.\n\n"
            "Examples:\n"
            "  litellm-maintainer entitlements --policy ~/.config/litellm-maintainer/policy.yaml\n"
            "  litellm-maintainer entitlements --policy policy.yaml --json\n"
            "  litellm-maintainer entitlements --policy policy.yaml --format markdown > docs/entitlements.md"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_read_arguments(entitlements_parser)
    entitlements_parser.set_defaults(func=cmd_entitlements)

    guidance_parser = subparsers.add_parser(
        "guidance",
        help="Print ranked models for one task axis, each with its routes",
        description=(
            "Ranks the Canonical Models this proxy currently offers by one of "
            "the model feed's own quality scores, and lists every route to each "
            "one, cheapest first, so the route order doubles as a failover "
            "order. Each row states why it ranks where it does.\n\n"
            "An alias printed here is callable by its exact id even when a "
            "model list your client cached earlier does not hold it. The proxy "
            "resolves a call by alias, not by what the client last fetched. "
            "Read client_advisory for what the last run added and removed.\n\n"
            "This command never reports how much credit or quota is left. "
            "Nothing it can read knows that. It reports what was measured: "
            "which routes answer, which refused, why, and when a refusal said "
            "it clears.\n\n"
            "Read-only: writes no file and makes no provider call.\n\n"
            "Examples:\n"
            "  litellm-maintainer guidance --policy policy.yaml --for coding\n"
            "  litellm-maintainer guidance --policy policy.yaml --for reasoning --json\n"
            "  litellm-maintainer guidance --policy policy.yaml --for coding --prefer free --limit 5\n"
            "  litellm-maintainer guidance --policy policy.yaml --for agentic --format markdown > docs/models.md"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_read_arguments(guidance_parser)
    # No `choices=` on either of the next two. argparse rejects a bad
    # choice with exit code 2, and every other refusal this tool makes
    # exits 1, so a script gating on 1 read a bad axis as success.
    # `guidance.derive` raises `GuidanceError` instead, and `cmd_guidance`
    # turns that into exit 1 with a message naming the valid values.
    guidance_parser.add_argument(
        "--for",
        dest="axis",
        default="coding",
        metavar="{" + ",".join(sorted(GUIDANCE_AXES)) + "}",
        help="The scored axis to rank by (default: %(default)s)",
    )
    guidance_parser.add_argument(
        "--prefer",
        default=None,
        metavar="{free,flat_rate}",
        help=(
            "Sort models into cost tiers before score, for bulk work. "
            "Omit it to rank by score alone."
        ),
    )
    guidance_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Show at most this many models. The cap is stated in the output.",
    )
    guidance_parser.add_argument(
        "--min-context",
        type=int,
        default=None,
        dest="min_context",
        help=(
            "Show only Routes that hold at least this many input tokens. A "
            "Route stating no window is excluded rather than assumed small, "
            "and the output states how many were dropped and why."
        ),
    )
    guidance_parser.set_defaults(func=cmd_guidance)

    policy_parser = subparsers.add_parser(
        "policy",
        help="Record an operator decision into Policy, safely",
        description=(
            "The operator surface. Each verb records one decision into your "
            "Policy file: it takes the lock, refuses when the file changed on "
            "disk since it was read, validates the result before it writes, "
            "and prints the diff it applied. Your comments and key order "
            "survive.\n\n"
            "Editing policy.yaml by hand stays first-class. These verbs exist "
            "so a one-line decision does not need an editor.\n\n"
            "Nothing in the scheduled run path writes Policy.\n\n"
            "Examples:\n"
            "  litellm-maintainer policy approve-candidate openrouter:vendor/new-coder:free --policy policy.yaml\n"
            "  litellm-maintainer policy withhold cline-pass:cline-pass/glm-5.2 --reason 'credits exhausted' --policy policy.yaml\n"
            "  litellm-maintainer policy set-entitlement qwencloud-token-plan shared_pool --policy policy.yaml\n"
            "  litellm-maintainer policy set-alias openrouter:vendor/coder-large claude-coder-xl --policy policy.yaml --dry-run"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    policy_verbs = policy_parser.add_subparsers(dest="verb", required=True)

    def add_policy_arguments(target: argparse.ArgumentParser) -> None:
        target.add_argument("--policy", required=True, help="Path to the Policy file")
        target.add_argument(
            "--home",
            default=None,
            help="Instance directory, which holds the lock (default: $LITELLM_MAINTAINER_HOME)",
        )
        target.add_argument(
            "--dry-run",
            action="store_true",
            help="Print the diff and write nothing",
        )
        target.add_argument(
            "--env", default=None, help="Path to a .env-style file to redact from output"
        )
        target.set_defaults(func=cmd_policy)

    approve = policy_verbs.add_parser(
        "approve-candidate",
        help="Admit a Candidate that carries no quality score",
    )
    approve.add_argument("offering_id", help="Offering id, as <provider>:<model>")
    add_policy_arguments(approve)

    withhold_verb = policy_verbs.add_parser(
        "withhold", help="Withhold an Offering, with a reason"
    )
    withhold_verb.add_argument("offering_id", help="Offering id, as <provider>:<model>")
    withhold_verb.add_argument(
        "--reason", required=True, help="Why you withhold it. Required: a bare id ages badly."
    )
    add_policy_arguments(withhold_verb)

    unwithhold_verb = policy_verbs.add_parser(
        "unwithhold", help="Stop withholding an Offering"
    )
    unwithhold_verb.add_argument("offering_id", help="Offering id, as <provider>:<model>")
    add_policy_arguments(unwithhold_verb)

    set_alias_verb = policy_verbs.add_parser(
        "set-alias", help="Pin one Offering's Alias, overriding the derived name"
    )
    set_alias_verb.add_argument("offering_id", help="Offering id, as <provider>:<model>")
    set_alias_verb.add_argument("alias", help="The exact Alias clients will ask for")
    add_policy_arguments(set_alias_verb)

    set_entitlement_verb = policy_verbs.add_parser(
        "set-entitlement",
        help="State whether a provider bills from one shared pool or per model",
    )
    set_entitlement_verb.add_argument("provider_id", help="Provider id")
    set_entitlement_verb.add_argument(
        "entitlement", choices=("shared_pool", "per_model"), help="The entitlement kind"
    )
    add_policy_arguments(set_entitlement_verb)

    init_parser = subparsers.add_parser(
        "init",
        help="Write a starter Policy derived from the Feed's own providers",
        description=(
            "Reads the Feed and writes a commented Policy naming every provider "
            "it publishes. Each entry names the environment variable that "
            "provider's credential comes from. No credential is ever written "
            "into Policy.\n\n"
            "Refuses to overwrite an existing Policy unless --force is given.\n\n"
            "Example:\n"
            "  litellm-maintainer init --feed feed.json\n"
            "  litellm-maintainer doctor --policy policy.yaml   # then see what is missing"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    init_parser.add_argument("--feed", required=True, help="Path to the Feed document")
    init_parser.add_argument(
        "--out",
        default=None,
        help="Path to write the Policy to (default: <home>/policy.yaml)",
    )
    init_parser.add_argument(
        "--alias-prefix",
        default="claude-",
        help="Prefix for every derived Alias (default: %(default)s)",
    )
    init_parser.add_argument(
        "--home", default=None, help="Instance directory (default: $LITELLM_MAINTAINER_HOME)"
    )
    init_parser.add_argument(
        "--force", action="store_true", help="Replace an existing Policy"
    )
    init_parser.add_argument(
        "--env", default=None, help="Path to a .env-style file to redact from output"
    )
    init_parser.set_defaults(func=cmd_init)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Report every reason this instance is not working",
        description=(
            "Checks the whole install: every provider's credential variable, "
            "the Feed Document's age, whether the proxy answers, whether "
            "Health State holds anything, which providers no probe has ever "
            "reached, and which withheld entries name an offering the feed "
            "dropped. Each failed check names the command that fixes it.\n\n"
            "Exits 0 when every check passed and 1 when any failed, so a "
            "script can gate on it. Writes no file.\n\n"
            "Example:\n"
            "  litellm-maintainer doctor --policy ~/.config/litellm-maintainer/policy.yaml"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    doctor_parser.add_argument(
        "--policy",
        default=None,
        help="Path to the Policy file (default: <home>/policy.yaml)",
    )
    doctor_parser.add_argument(
        "--served-config-dir",
        default=None,
        help=(
            "Directory holding the proxy configs actually served, checked for "
            "the Observation Journal callback (default: %(default)s)"
        ),
    )
    doctor_parser.add_argument(
        "--target-dir",
        default=None,
        help=(
            "LaunchAgents directory to check for the installed tick "
            "(default: ~/Library/LaunchAgents)"
        ),
    )
    doctor_parser.add_argument(
        "--feed",
        default=None,
        help="Path to the Feed Document (default: <home>/feed.json)",
    )
    doctor_parser.add_argument(
        "--home", default=None, help="Instance directory (default: $LITELLM_MAINTAINER_HOME)"
    )
    doctor_parser.add_argument(
        "--litellm-path",
        default=None,
        help=(
            "Path to the litellm package directory the proxy runs, checked for "
            "the local patches. Found from the 'litellm' executable by default. "
            "The patch checks pass with a note when it cannot be read."
        ),
    )
    doctor_parser.add_argument(
        "--proxy-base",
        default="http://localhost:4000",
        help="The proxy's base URL, checked for liveliness (default: %(default)s)",
    )
    doctor_parser.add_argument(
        "--json", action="store_true", help="Emit the diagnosis as JSON"
    )
    doctor_parser.add_argument(
        "--env", default=None, help="Path to a .env-style file to redact from output"
    )
    doctor_parser.set_defaults(func=cmd_doctor)

    rollback_parser = subparsers.add_parser(
        "rollback", help="Restore the most recent Generated Config snapshot"
    )
    rollback_parser.add_argument(
        "--out",
        default=str(_default_out_path()),
        help="Path to the Generated Config to restore (default: %(default)s)",
    )
    rollback_parser.add_argument(
        "--home", default=None, help="Instance directory (default: $LITELLM_MAINTAINER_HOME)"
    )
    rollback_parser.add_argument(
        "--env", default=None, help="Path to a .env-style file to redact from output"
    )
    rollback_parser.set_defaults(func=cmd_rollback)

    deploy_parser = subparsers.add_parser(
        "deploy",
        help="Apply pending changes to the proxy now, ignoring the tick's interval",
    )
    deploy_parser.add_argument(
        "--feed", default=None, help="Path to the Feed document (default: the instance directory)"
    )
    deploy_parser.add_argument(
        "--policy", default=None, help="Path to the Policy file (default: the instance directory)"
    )
    deploy_parser.add_argument(
        "--out",
        default=None,
        help="Where to write (default: the --out the installed tick job passes)",
    )
    deploy_parser.add_argument(
        "--home", default=None, help="Instance directory (default: $LITELLM_MAINTAINER_HOME)"
    )
    deploy_parser.add_argument(
        "--env", default=None, help="Path to a .env-style file to redact from output"
    )
    deploy_parser.add_argument(
        "--force",
        action="store_true",
        help="Write anyway when only a safety-rail threshold refused",
    )
    deploy_parser.set_defaults(func=cmd_deploy)

    explain_parser = subparsers.add_parser(
        "explain", help="Name the stage that stopped one Offering reaching a client"
    )
    explain_parser.add_argument("target", help="An Offering id or an Alias")
    explain_parser.add_argument(
        "--feed", default=None, help="Path to the Feed document (default: the instance directory)"
    )
    explain_parser.add_argument(
        "--policy", default=None, help="Path to the Policy file (default: the instance directory)"
    )
    explain_parser.add_argument(
        "--home", default=None, help="Instance directory (default: $LITELLM_MAINTAINER_HOME)"
    )
    explain_parser.add_argument(
        "--proxy-base",
        default="http://localhost:4000",
        help="The running proxy's base URL, asked for its model list (default: %(default)s)",
    )
    explain_parser.add_argument(
        "--no-proxy",
        action="store_true",
        help="Skip the live proxy check. That stage then reports unknown.",
    )
    explain_parser.add_argument(
        "--env", default=None, help="Path to a .env-style file to redact from output"
    )
    explain_parser.add_argument(
        "--json", action="store_true", help="Print the walk as JSON"
    )
    explain_parser.set_defaults(func=cmd_explain)

    status_parser = subparsers.add_parser(
        "status", help="Print what is offered, Excluded, Withheld, Sunsetting, and awaiting approval"
    )
    # `--feed` and `--policy` default to the instance directory, exactly as
    # `guidance`, `entitlements` and `headroom` do. They were required
    # here, and a consumer reading all four hit one command that refused
    # the same invocation the other three accepted (reported 2026-07-29).
    status_parser.add_argument(
        "--feed", default=None, help="Path to the Feed document (default: the instance directory)"
    )
    status_parser.add_argument(
        "--policy", default=None, help="Path to the Policy file (default: the instance directory)"
    )
    status_parser.add_argument(
        "--home", default=None, help="Instance directory (default: $LITELLM_MAINTAINER_HOME)"
    )
    status_parser.add_argument(
        "--env", default=None, help="Path to a .env-style file to redact from output"
    )
    status_parser.add_argument(
        "--json", action="store_true", help="Print the same view as JSON"
    )
    status_parser.set_defaults(func=cmd_status)

    smoke_parser = subparsers.add_parser(
        "smoke",
        help="Check one Offering per distinct translation rule, through the running proxy",
    )
    smoke_parser.add_argument("--feed", required=True, help="Path to the Feed document")
    smoke_parser.add_argument("--policy", required=True, help="Path to the Policy file")
    smoke_parser.add_argument(
        "--home", default=None, help="Instance directory (default: $LITELLM_MAINTAINER_HOME)"
    )
    smoke_parser.add_argument(
        "--proxy-base",
        default="http://localhost:4000",
        help="The running proxy's base URL (default: %(default)s)",
    )
    smoke_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print which rule would be checked with which Alias, and call nothing",
    )
    smoke_parser.add_argument(
        "--env", default=None, help="Path to a .env-style file to redact from output"
    )
    smoke_parser.set_defaults(func=cmd_smoke)

    run_parser = subparsers.add_parser(
        "run",
        help="Run one scheduled tick: the due gate, then probe, reduce, plan, and deploy",
    )
    run_parser.add_argument("--feed", required=True, help="Path to the Feed document")
    run_parser.add_argument("--policy", required=True, help="Path to the Policy file")
    run_parser.add_argument(
        "--out",
        default=str(_default_out_path()),
        help="Path to write the Generated Config to (default: %(default)s)",
    )
    run_parser.add_argument(
        "--home", default=None, help="Instance directory (default: $LITELLM_MAINTAINER_HOME)"
    )
    run_parser.add_argument(
        "--env", default=None, help="Path to a .env-style file to redact from output"
    )
    run_parser.add_argument(
        "--proxy-base",
        default="http://localhost:4000",
        help="The running proxy's base URL, checked for liveliness (default: %(default)s)",
    )
    run_parser.add_argument(
        "--provider-modules-source",
        default=None,
        help="Directory of provider handler modules to deploy (default: none deployed)",
    )
    run_parser.add_argument(
        "--provider-modules-target",
        default=None,
        help="Directory to deploy provider handler modules into, content-compare only",
    )
    run_parser.add_argument(
        "--force",
        action="store_true",
        help="Write anyway when only a safety-rail threshold refused (never a validation failure)",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the schedule's own inputs and call nothing: no network, no write",
    )
    run_parser.set_defaults(func=cmd_run)

    # Defect 3: `watch` had no subcommand at all, so `watcher.cmd_watch`
    # was unreachable. Wired per `watcher.py`'s own footer: it needs the
    # same paths `run` does, since `watch`'s `run_maintainer` calls
    # `cmd_run` (defect 4), plus its own `--interval`.
    watch_parser = subparsers.add_parser(
        "watch",
        help=(
            "Watch the Observation Journal in the foreground (debugging). "
            "The installed tick already does this; do not run both."
        ),
    )
    watch_parser.add_argument("--feed", required=True, help="Path to the Feed document")
    watch_parser.add_argument("--policy", required=True, help="Path to the Policy file")
    watch_parser.add_argument(
        "--out",
        default=str(_default_out_path()),
        help="Path to write the Generated Config to (default: %(default)s)",
    )
    watch_parser.add_argument(
        "--home", default=None, help="Instance directory (default: $LITELLM_MAINTAINER_HOME)"
    )
    watch_parser.add_argument(
        "--env", default=None, help="Path to a .env-style file to redact from output"
    )
    watch_parser.add_argument(
        "--proxy-base",
        default="http://localhost:4000",
        help="The running proxy's base URL, checked for liveliness (default: %(default)s)",
    )
    watch_parser.add_argument(
        "--provider-modules-source",
        default=None,
        help="Directory of provider handler modules to deploy (default: none deployed)",
    )
    watch_parser.add_argument(
        "--provider-modules-target",
        default=None,
        help="Directory to deploy provider handler modules into, content-compare only",
    )
    watch_parser.add_argument(
        "--force",
        action="store_true",
        help="Write anyway when only a safety-rail threshold refused (never a validation failure)",
    )
    litellm_maintainer.watcher.add_watch_arguments(watch_parser)
    watch_parser.set_defaults(func=litellm_maintainer.watcher.cmd_watch, dry_run=False)

    install_parser = subparsers.add_parser(
        "install", help="Write the launchd plist that ticks 'run'. Never calls launchctl."
    )
    install_parser.add_argument(
        "--policy", required=True, help="Path the installed job passes to 'run --policy'"
    )
    install_parser.add_argument(
        "--feed", required=True, help="Path the installed job passes to 'run --feed'"
    )
    install_parser.add_argument(
        "--home", default=None, help="Path the installed job passes to 'run --home'"
    )
    install_parser.add_argument(
        "--out",
        default=None,
        help=(
            "Path the installed job passes to 'run --out': the config your "
            "proxy actually serves. Without it the tick writes the instance "
            "directory's own copy, which the proxy never reads."
        ),
    )
    install_parser.add_argument(
        "--env",
        default=None,
        help=(
            "Path the installed job passes to 'run --env', stored absolute. "
            "launchd runs from '/', so without this the tick resolves no "
            "credential and refuses every write."
        ),
    )
    install_parser.add_argument(
        "--provider-modules-source",
        default=None,
        help="Path the installed job passes to 'run --provider-modules-source'",
    )
    install_parser.add_argument(
        "--provider-modules-target",
        default=None,
        help="Path the installed job passes to 'run --provider-modules-target'",
    )
    install_parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable the plist invokes (default: %(default)s)",
    )
    install_parser.add_argument(
        "--tick-seconds",
        type=int,
        default=DEFAULT_TICK_SECONDS,
        help="How often launchd ticks 'run' (default: %(default)s)",
    )
    install_parser.add_argument(
        "--target-dir",
        default=str(Path.home() / "Library" / "LaunchAgents"),
        help="Directory to write the plist into (default: %(default)s)",
    )
    install_parser.set_defaults(func=cmd_install)

    uninstall_parser = subparsers.add_parser(
        "uninstall", help="Remove the launchd plist. Safe when nothing is installed."
    )
    uninstall_parser.add_argument(
        "--target-dir",
        default=str(Path.home() / "Library" / "LaunchAgents"),
        help="Directory the plist was written into (default: %(default)s)",
    )
    uninstall_parser.set_defaults(func=cmd_uninstall)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
