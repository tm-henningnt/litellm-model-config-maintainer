"""Tests for `litellm_maintainer.doctor`.

`diagnose` is pure: every test builds its own Policy, Feed and Health
State in memory (`policy.parse_policy`, `feed.parse_feed`, a plain dict
of `reduce.OfferingHealth`) and passes them straight in. No test reads
a file or the network.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from litellm_maintainer.codexbar import CodexbarExtraWindow, CodexbarIdentity, CodexbarReading, CodexbarWindow
from litellm_maintainer.doctor import Diagnosis, diagnose, render_text
from litellm_maintainer.feed import parse_feed
from litellm_maintainer.litellm_patches import REQUIRED_PATCHES, inspect_patches
from litellm_maintainer.policy import parse_policy
from litellm_maintainer.reduce import OfferingHealth

NOW = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)


def _feed_raw():
    return {
        "schema_version": "1",
        "feed": {"generated_at": "2026-07-26T00:00:00Z"},
        "providers": [
            {
                "id": "openrouter",
                "name": "OpenRouter",
                "default_base_url": "https://openrouter.ai/api/v1",
                "authentication": {"credential_hint": "OPENROUTER_API_KEY"},
            },
        ],
        "models": [
            {
                "id": "openrouter:vendor/coder-large",
                "provider": {"id": "openrouter"},
                "provider_model_id": "vendor/coder-large",
                "capabilities": ["tool_use"],
                "endpoint": {},
                "pricing": {"kind": "free"},
                "availability": {"status": "available"},
                "quality": {"coding_score": 40},
                "policy": {"visibility": "public"},
            },
        ],
    }


def _policy_raw(**overrides):
    raw = {
        "providers": {"openrouter": {"mode": "all"}},
        "quality": {"minimum_coding_score": 20},
        "approved_candidates": [],
        "naming": {
            "provider_labels": {"openrouter": "openrouter"},
            "alias_overrides": {},
            "alias_prefix": "claude-",
        },
        "withheld": {},
        "declared": [],
        "pacing": {"default": {"concurrency": 2, "minimum_interval_seconds": 5}},
        "schedule": {
            "enabled": True,
            "interval_minutes": 60,
            "require_proxy": True,
            "maximum_staleness_hours": 24,
        },
        "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
        "feed": {"url": "https://example.invalid/feed.json", "maximum_age_hours": 24},
    }
    raw.update(overrides)
    return raw


@pytest.fixture
def feed():
    return parse_feed(_feed_raw())


@pytest.fixture
def policy():
    return parse_policy(_policy_raw())


def _healthy_record() -> OfferingHealth:
    return OfferingHealth(
        excluded=False,
        last_success_at=NOW - timedelta(hours=1),
        last_attempt_at=NOW - timedelta(hours=1),
        failure_count=0,
    )


def _all_ok_kwargs(policy, feed):
    return dict(
        policy=policy,
        feed=feed,
        health={"openrouter:vendor/coder-large": _healthy_record()},
        feed_document_metadata={"generated_at": "2026-07-26T00:00:00Z"},
        environ={"OPENROUTER_API_KEY": "set"},
        proxy_ok=True,
        now=NOW,
    )


def test_every_check_passing_gives_diagnosis_ok_true(policy, feed):
    diagnosis = diagnose(**_all_ok_kwargs(policy, feed))
    assert diagnosis.ok is True


def test_one_failing_check_gives_diagnosis_ok_false(policy, feed):
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["proxy_ok"] = False
    diagnosis = diagnose(**kwargs)
    assert diagnosis.ok is False


def test_missing_credential_variable_produces_a_failed_check_naming_it(policy, feed):
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["environ"] = {}
    diagnosis = diagnose(**kwargs)
    check = next(c for c in diagnosis.checks if c.name == "credential.openrouter")
    assert check.ok is False
    assert "OPENROUTER_API_KEY" in check.detail


def test_set_credential_variable_produces_a_passing_check(policy, feed):
    diagnosis = diagnose(**_all_ok_kwargs(policy, feed))
    check = next(c for c in diagnosis.checks if c.name == "credential.openrouter")
    assert check.ok is True


def test_stale_feed_document_produces_a_failed_check_naming_its_age(policy, feed):
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["feed_document_metadata"] = {"generated_at": "2026-07-01T00:00:00Z"}
    diagnosis = diagnose(**kwargs)
    check = next(c for c in diagnosis.checks if c.name == "feed_document.age")
    assert check.ok is False
    assert "h ago" in check.detail or "generated_at" in check.detail


def test_fresh_feed_document_produces_a_passing_check(policy, feed):
    diagnosis = diagnose(**_all_ok_kwargs(policy, feed))
    check = next(c for c in diagnosis.checks if c.name == "feed_document.age")
    assert check.ok is True


def test_proxy_ok_false_produces_a_failed_check_and_does_not_raise(policy, feed):
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["proxy_ok"] = False
    diagnosis = diagnose(**kwargs)
    check = next(c for c in diagnosis.checks if c.name == "proxy.reachable")
    assert check.ok is False


def test_empty_health_state_produces_a_failed_check_naming_probe(policy, feed):
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["health"] = {}
    diagnosis = diagnose(**kwargs)
    check = next(c for c in diagnosis.checks if c.name == "health_state.populated")
    assert check.ok is False
    assert check.remedy is not None
    assert "probe" in check.remedy


def test_provider_with_no_health_state_record_produces_a_failed_check_naming_it(policy, feed):
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["health"] = {}
    diagnosis = diagnose(**kwargs)
    check = next(c for c in diagnosis.checks if c.name == "health_state.probed.openrouter")
    assert check.ok is False
    assert "openrouter" in check.detail


def test_a_provider_whose_every_offering_is_withheld_produces_no_probed_check(feed):
    """A Withheld Offering is never probed, so demanding a Health State
    record for one asks for a Probe that never runs.

    Measured 2026-07-28: `cline-pass` publishes 11 Offerings and Policy
    Withholds all 11, so this check failed permanently and its remedy
    could not clear it. One false failure hides the real ones behind it.
    """
    policy = parse_policy(
        _policy_raw(withheld={"openrouter:vendor/coder-large": "renewal unconfirmed"})
    )
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["health"] = {}
    diagnosis = diagnose(**kwargs)

    assert not [
        c for c in diagnosis.checks if c.name == "health_state.probed.openrouter"
    ]


def test_withheld_entry_for_offering_absent_from_feed_produces_a_failed_check(feed):
    policy = parse_policy(
        _policy_raw(withheld={"openrouter:vendor/no-longer-published": "reason"})
    )
    kwargs = _all_ok_kwargs(policy, feed)
    diagnosis = diagnose(**kwargs)
    check = next(
        c for c in diagnosis.checks if c.name == "withheld.openrouter:vendor/no-longer-published"
    )
    assert check.ok is False


def _declared_with_reference(reference_model):
    return [
        {
            "alias": "claude-seat",
            "reference_model": reference_model,
            "litellm_params": {
                "model": "openai/seat",
                "api_base": "http://127.0.0.1:4011/v1",
            },
        }
    ]


def _feed_with_canonical_model(canonical_model_id):
    raw = _feed_raw()
    raw["models"][0]["canonical_model"] = {"id": canonical_model_id}
    return parse_feed(raw)


def test_a_reference_model_the_feed_still_publishes_produces_a_passing_check():
    policy = parse_policy(
        _policy_raw(declared=_declared_with_reference("vendor/coder-large"))
    )
    feed = _feed_with_canonical_model("vendor/coder-large")
    diagnosis = diagnose(**_all_ok_kwargs(policy, feed))

    check = next(
        c for c in diagnosis.checks if c.name == "declared.claude-seat.reference_model"
    )
    assert check.ok is True


def test_a_reference_model_the_feed_dropped_produces_a_failed_check_naming_it():
    """`guidance` only warns about this one, and the row still answers, so
    the lost score is invisible where it matters. This is where a stale
    Policy line is meant to be read."""
    policy = parse_policy(
        _policy_raw(declared=_declared_with_reference("vendor/renamed-upstream"))
    )
    feed = _feed_with_canonical_model("vendor/coder-large")
    diagnosis = diagnose(**_all_ok_kwargs(policy, feed))

    check = next(
        c for c in diagnosis.checks if c.name == "declared.claude-seat.reference_model"
    )
    assert check.ok is False
    assert "vendor/renamed-upstream" in check.detail
    assert check.remedy is not None
    assert "reference_model" in check.remedy


def test_a_declared_offering_naming_no_reference_model_produces_no_such_check(feed):
    policy = parse_policy(
        _policy_raw(
            declared=[
                {"alias": "claude-seat", "litellm_params": {"model": "anthropic/seat"}}
            ]
        )
    )
    diagnosis = diagnose(**_all_ok_kwargs(policy, feed))

    assert not [c for c in diagnosis.checks if c.name.endswith(".reference_model")]


def test_every_failed_check_carries_a_non_empty_remedy(policy, feed):
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["environ"] = {}
    kwargs["proxy_ok"] = False
    kwargs["health"] = {}
    kwargs["feed_document_metadata"] = {}
    diagnosis = diagnose(**kwargs)
    for check in diagnosis.checks:
        if not check.ok:
            assert check.remedy


def test_render_text_runs_on_an_empty_and_a_full_diagnosis(policy, feed):
    empty = Diagnosis(checks=())
    assert render_text(empty)

    full = diagnose(**_all_ok_kwargs(policy, feed))
    assert render_text(full)

    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["proxy_ok"] = False
    failing = diagnose(**kwargs)
    text = render_text(failing)
    assert "FAIL" in text


def test_as_dict_output_is_json_serialisable(policy, feed):
    diagnosis = diagnose(**_all_ok_kwargs(policy, feed))
    json.dumps(diagnosis.as_dict())


# --- Local litellm patches ---------------------------------------------
#
# Two litellm defects have a local patch, applied to the installed
# litellm rather than to this repository. `uv tool upgrade litellm`
# removes both with no other symptom, so `doctor` reads a marker per
# patch. See `litellm_maintainer.litellm_patches` and `docs/gotchas.md`.


def _patched_tree(tmp_path, *, present=(True, True)):
    """Write a litellm tree carrying each marker, or not."""
    for (name, relative_path, marker, _remedy), carry in zip(REQUIRED_PATCHES, present):
        target = tmp_path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# stock line\n{marker if carry else '# stock line only'}\n")
    return tmp_path


def test_a_patched_tree_reports_every_patch_present(tmp_path):
    results = inspect_patches(_patched_tree(tmp_path))
    assert [r.present for r in results] == [True, True]
    assert all("carries the patch" in r.detail for r in results)


def test_a_stock_tree_reports_every_patch_absent(tmp_path):
    results = inspect_patches(_patched_tree(tmp_path, present=(False, False)))
    assert [r.present for r in results] == [False, False]


def test_a_tree_missing_the_files_reports_unknown_not_absent(tmp_path):
    """Unknown must not read as absent; only a read file can say absent."""
    results = inspect_patches(tmp_path)
    assert [r.present for r in results] == [None, None]


def test_no_located_tree_reports_unknown_for_every_patch():
    results = inspect_patches(None)
    assert [r.present for r in results] == [None, None]
    assert all("not located" in r.detail for r in results)


def test_an_absent_patch_fails_the_diagnosis_and_names_a_remedy(policy, feed, tmp_path):
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["litellm_patches"] = inspect_patches(
        _patched_tree(tmp_path, present=(False, True))
    )
    diagnosis = diagnose(**kwargs)

    failed = next(c for c in diagnosis.checks if c.name == "litellm_patch.chatgpt_stream")
    assert failed.ok is False
    assert failed.remedy and "gotchas" in failed.remedy
    assert diagnosis.ok is False

    passed = next(c for c in diagnosis.checks if c.name == "litellm_patch.usage_only_chunk")
    assert passed.ok is True


def test_an_unreadable_tree_does_not_fail_the_diagnosis(policy, feed, tmp_path):
    """An operator running the proxy elsewhere must not see a failure."""
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["litellm_patches"] = inspect_patches(tmp_path)
    diagnosis = diagnose(**kwargs)
    assert diagnosis.ok is True
    names = {c.name for c in diagnosis.checks}
    assert "litellm_patch.chatgpt_stream" in names


def test_patch_checks_are_absent_when_the_caller_passes_none(policy, feed):
    """The parameter defaults to (), so existing callers gain no checks."""
    diagnosis = diagnose(**_all_ok_kwargs(policy, feed))
    assert not [c for c in diagnosis.checks if c.name.startswith("litellm_patch.")]


# --- The two ways a configured instance can still do nothing ---------------

_CALLBACK = "journal_failure_callback.observation_journal_callback"


def _check(diagnosis, prefix):
    return [c for c in diagnosis.checks if c.name.startswith(prefix)]


def test_a_main_proxy_that_registers_no_journal_callback_fails():
    """An unregistered callback is silent: the proxy serves normally and
    the Journal simply stays empty forever. Nothing else can tell
    "no failures happened" from "no failures were recorded"."""
    policy = parse_policy(_policy_raw())
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["served_configs"] = {"/etc/litellm/config.yaml": (False, True)}

    checks = _check(diagnose(**kwargs), "journal.callback_registered")

    assert len(checks) == 1
    assert checks[0].ok is False
    assert checks[0].remedy is not None
    # The remedy must send the operator to Policy, not to the generated
    # file, which the next run overwrites.
    assert "proxy_settings" in checks[0].remedy


def test_a_main_proxy_that_registers_the_callback_passes():
    policy = parse_policy(_policy_raw())
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["served_configs"] = {"/etc/litellm/config.yaml": (True, True)}

    checks = _check(diagnose(**kwargs), "journal.callback_registered")

    assert [c.ok for c in checks] == [True]


def test_a_worker_proxy_that_records_nothing_is_not_a_failure():
    """Decision: the main proxy records, workers do not. A worker's own
    `model_group` carries no seat identity, so an entry written there
    names a key Health State does not hold."""
    policy = parse_policy(_policy_raw())
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["served_configs"] = {"/etc/litellm/chatgpt-worker.yaml": (False, False)}

    checks = _check(diagnose(**kwargs), "journal.callback_registered")

    assert [c.ok for c in checks] == [True]


def test_no_served_config_directory_produces_no_callback_check():
    """A check that cannot measure must not fail."""
    policy = parse_policy(_policy_raw())
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["served_configs"] = {}

    assert _check(diagnose(**kwargs), "journal.callback_registered") == []


def test_a_missing_launchd_tick_fails():
    """Nothing runs on its own until the plist exists. An operator can
    hold a configured instance, a registered callback and a growing
    Journal, and still have no process that reads any of it."""
    policy = parse_policy(_policy_raw())
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["tick_installed"] = False

    check = _check(diagnose(**kwargs), "schedule.tick_installed")[0]

    assert check.ok is False
    assert "launchctl" in (check.remedy or "")


def test_an_installed_launchd_tick_passes():
    policy = parse_policy(_policy_raw())
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["tick_installed"] = True

    assert _check(diagnose(**kwargs), "schedule.tick_installed")[0].ok is True


def test_a_launchagents_directory_that_cannot_be_read_does_not_fail_the_tick_check():
    policy = parse_policy(_policy_raw())
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["tick_installed"] = None

    check = _check(diagnose(**kwargs), "schedule.tick_installed")[0]

    assert check.ok is True
    assert "not checked" in check.detail


# --- Ticket 07: a rotted Headroom mapping is a named finding ----------------
#
# Every part of the Headroom capability degrades to the same symptom: no
# Headroom. These tests cover the five conditions the ticket names, plus
# the two states doctor must never confuse: a healthy machine that declares
# no source at all (silence, no finding, no warning), and a healthy machine
# that declares one (every check passes, `diagnose` still exits 0).


def _reading(
    *,
    provider_id: str = "claude",
    account_email: str | None = None,
    extra_windows: tuple[CodexbarExtraWindow, ...] = (),
) -> CodexbarReading:
    return CodexbarReading(
        provider=provider_id,
        identity=CodexbarIdentity(provider_id=provider_id, account_email=account_email),
        primary=CodexbarWindow(used_percent=10, window_minutes=300, resets_at=None),
        secondary=None,
        tertiary=None,
        extra_windows=extra_windows,
        updated_at="2026-07-28T20:52:00Z",
        error=None,
    )


def _policy_with_headroom(**headroom_overrides):
    headroom = {"sources": {"provider:claude": "codexbar:claude/"}}
    headroom.update(headroom_overrides)
    return parse_policy(_policy_raw(headroom=headroom))


def test_a_declared_source_matching_no_reading_fails():
    policy = _policy_with_headroom()
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_readings"] = (_reading(provider_id="claude2"),)

    check = _check(diagnose(**kwargs), "headroom.mapped.provider:claude")[0]

    assert check.ok is False
    assert "matches no Reading" in check.detail
    assert check.remedy is not None


def test_a_declared_source_matching_several_readings_fails():
    policy = _policy_with_headroom()
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_readings"] = (
        _reading(provider_id="claude", account_email=None),
        _reading(provider_id="claude", account_email=None),
    )

    check = _check(diagnose(**kwargs), "headroom.mapped.provider:claude")[0]

    assert check.ok is False
    assert "matches 2 Readings" in check.detail


def test_a_declared_source_matching_exactly_one_reading_passes():
    policy = _policy_with_headroom()
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_readings"] = (_reading(),)

    check = _check(diagnose(**kwargs), "headroom.mapped.provider:claude")[0]

    assert check.ok is True


def test_no_headroom_readings_produces_no_mapping_checks():
    """`headroom_readings=None` means the caller could not measure: no
    binary, no source, or a failed run. A check that cannot measure must
    not fail."""
    policy = _policy_with_headroom()
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_readings"] = None

    assert _check(diagnose(**kwargs), "headroom.mapped.") == []


# --- Defect 4: a failed codexbar run is its own failed Check ----------------
#
# Before this fix, `_headroom_readings` returned `None` for a binary that
# ran and failed (non-zero exit, a timeout, output its parser could not
# read) exactly the same as for "Policy declares no source" or "the binary
# is missing". `_headroom_mapping_checks` then emitted nothing, and
# `doctor` exited 0 while codexbar failed on every real invocation --
# though this module's own header comment claimed every such case "already
# produces its own Check".


def test_a_failed_codexbar_run_produces_its_own_failed_check():
    policy = _policy_with_headroom()
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_readings"] = None
    kwargs["headroom_run_error"] = "codexbar exited 1: rate limited"

    check = _check(diagnose(**kwargs), "headroom.readings")[0]

    assert check.ok is False
    assert "rate limited" in check.detail
    assert check.remedy is not None
    assert diagnose(**kwargs).ok is False


def test_a_successful_run_raises_no_readings_check():
    policy = _policy_with_headroom()
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_readings"] = (_reading(),)
    kwargs["headroom_run_error"] = None

    assert _check(diagnose(**kwargs), "headroom.readings") == []


def test_a_machine_declaring_no_source_raises_no_readings_check_even_given_an_error():
    """The capability-off rule wins regardless: a machine declaring no
    `headroom.sources` produces NO check at all, even if a caller passed
    an error string through by mistake."""
    policy = parse_policy(_policy_raw())
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_readings"] = None
    kwargs["headroom_run_error"] = "codexbar exited 1"

    assert _check(diagnose(**kwargs), "headroom.readings") == []


def test_the_binary_missing_from_the_path_fails():
    policy = _policy_with_headroom()
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_binary_present"] = False

    check = _check(diagnose(**kwargs), "headroom.binary")[0]

    assert check.ok is False
    assert "codexbar" in check.detail
    assert check.remedy is not None


def test_the_binary_on_the_path_passes():
    policy = _policy_with_headroom()
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_binary_present"] = True

    check = _check(diagnose(**kwargs), "headroom.binary")[0]

    assert check.ok is True


# --- Ticket 09: a declared slot the Reading no longer publishes ------------
#
# `headroom.sources.<id>.windows` names what codexbar's three slots
# measure, for a provider like Gemini whose slots hold one quota per
# model. Losing a slot -- a rename, or codexbar dropping the field --
# must read as a named finding, the same as losing an `extraRateWindows`
# id already does.


def _gemini_reading(*, primary: bool = True, secondary: bool = True, tertiary: bool = True):
    return CodexbarReading(
        provider="gemini",
        identity=CodexbarIdentity(provider_id="gemini", account_email="operator@example.com"),
        primary=CodexbarWindow(used_percent=100, window_minutes=1440, resets_at=None)
        if primary
        else None,
        secondary=CodexbarWindow(used_percent=0, window_minutes=1440, resets_at=None)
        if secondary
        else None,
        tertiary=CodexbarWindow(used_percent=0, window_minutes=1440, resets_at=None)
        if tertiary
        else None,
        extra_windows=(),
        updated_at="2026-07-28T20:52:30Z",
        error=None,
    )


def _policy_with_gemini_windows(
    *,
    members: dict[str, list[str]] | None = None,
    unmeasured: list[str] | None = None,
):
    source: dict = {
        "source": "codexbar:gemini/operator@example.com",
        "windows": {
            "primary": "gemini-pro",
            "secondary": "gemini-flash",
            "tertiary": "gemini-flash-lite",
        },
    }
    if members is not None:
        source["members"] = members
    if unmeasured is not None:
        source["unmeasured"] = unmeasured
    return parse_policy(
        _policy_raw(
            headroom={"sources": {"pool:gemini": source}},
            declared=[
                {
                    "alias": "claude-gemini-pro",
                    "litellm_params": {"model": "gemini/gemini-pro"},
                    "entitlement_pool": "gemini",
                    "sub_allowance": True,
                }
            ],
        )
    )


def test_a_declared_slot_still_published_passes():
    policy = _policy_with_gemini_windows()
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_readings"] = (_gemini_reading(),)

    checks = _check(diagnose(**kwargs), "headroom.window.pool:gemini.")
    assert len(checks) == 3
    assert all(check.ok for check in checks)


def test_a_declared_slot_the_reading_no_longer_publishes_fails():
    policy = _policy_with_gemini_windows()
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    # Codexbar dropped 'secondary' from this Reading entirely.
    kwargs["headroom_readings"] = (_gemini_reading(secondary=False),)

    check = _check(diagnose(**kwargs), "headroom.window.pool:gemini.secondary")[0]

    assert check.ok is False
    assert "no longer" in check.detail
    assert check.remedy is not None

    # The other two slots still publish, so they still pass.
    for slot in ("primary", "tertiary"):
        assert _check(diagnose(**kwargs), f"headroom.window.pool:gemini.{slot}")[0].ok is True


def test_no_windows_declared_produces_no_slot_checks():
    policy = _policy_with_headroom()
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_readings"] = (_reading(),)

    assert _check(diagnose(**kwargs), "headroom.window.") == []


# --- Ticket 10: `members` says which Health Key draws on each slot --------
#
# Every one of these three checks is static: Policy and the Feed alone,
# no live codexbar Reading. `_all_ok_kwargs` sets no `headroom_readings`
# at all here, on purpose -- these checks must still fire without one.


def test_an_admitted_health_key_no_member_claims_fails():
    # `members` is absent entirely: silence must not read as "every Health
    # Key is already assigned".
    policy = _policy_with_gemini_windows(members=None)
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(diagnose(**kwargs), "headroom.member.unclaimed.pool:gemini")[0]

    assert check.ok is False
    assert "claude-gemini-pro" in check.detail
    assert check.remedy is not None


def test_every_admitted_health_key_claimed_passes():
    policy = _policy_with_gemini_windows(members={"gemini-pro": ["claude-gemini-pro"]})
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(diagnose(**kwargs), "headroom.member.unclaimed.pool:gemini")[0]

    assert check.ok is True


def test_a_declared_sub_allowance_with_no_members_fails():
    policy = _policy_with_gemini_windows(members={"gemini-pro": ["claude-gemini-pro"]})
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(diagnose(**kwargs), "headroom.member.empty.pool:gemini.gemini-flash")[0]

    assert check.ok is False
    assert check.remedy is not None
    # The slot that does name a member still passes.
    assert _check(diagnose(**kwargs), "headroom.member.empty.pool:gemini.gemini-pro")[0].ok is True


def test_a_member_naming_no_known_health_key_fails():
    policy = _policy_with_gemini_windows(
        members={"gemini-pro": ["claude-gemini-pro", "claude-a-typo"]}
    )
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(
        diagnose(**kwargs), "headroom.member.unknown.pool:gemini.gemini-pro.claude-a-typo"
    )[0]

    assert check.ok is False
    assert check.remedy is not None


def test_a_member_naming_a_known_health_key_passes():
    policy = _policy_with_gemini_windows(members={"gemini-pro": ["claude-gemini-pro"]})
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(
        diagnose(**kwargs), "headroom.member.unknown.pool:gemini.gemini-pro.claude-gemini-pro"
    )[0]

    assert check.ok is True


def test_no_windows_declared_produces_no_membership_checks():
    policy = _policy_with_headroom()
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    assert _check(diagnose(**kwargs), "headroom.member.") == []


# --- Ticket 10 fix: a `members` key names a slot OR an extra window --------
#
# `_parse_headroom_members` used to require every `members` key to be a slot
# id `windows` declares. That broke Claude's real Sub-allowance case:
# `claude-weekly-scoped-fable` is a codexbar `extraRateWindows` id, not one
# of the three named slots. The parser now accepts both forms, and this
# Check (`headroom.member.unreachable.<allowance_id>.<member_key>`) is what
# tells the operator when a key reaches neither -- a live Reading is
# required to judge that, unlike the three static Checks above.


def test_a_member_key_naming_a_declared_slot_raises_no_unreachable_check():
    policy = _policy_with_gemini_windows(members={"gemini-pro": ["claude-gemini-pro"]})
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_readings"] = (_gemini_reading(),)

    assert _check(diagnose(**kwargs), "headroom.member.unreachable.") == []


def test_a_member_key_naming_an_extra_window_raises_no_unreachable_check():
    # 'gemini-extra-window' names no declared slot at all -- it reaches a
    # figure only because the Reading publishes it under `extraRateWindows`.
    policy = _policy_with_gemini_windows(members={"gemini-extra-window": ["claude-gemini-pro"]})
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_readings"] = (
        CodexbarReading(
            provider="gemini",
            identity=CodexbarIdentity(provider_id="gemini", account_email="operator@example.com"),
            primary=CodexbarWindow(used_percent=100, window_minutes=1440, resets_at=None),
            secondary=CodexbarWindow(used_percent=0, window_minutes=1440, resets_at=None),
            tertiary=CodexbarWindow(used_percent=0, window_minutes=1440, resets_at=None),
            extra_windows=(
                CodexbarExtraWindow(
                    id="gemini-extra-window",
                    title="Extra",
                    window=CodexbarWindow(used_percent=10, window_minutes=10080, resets_at=None),
                ),
            ),
            updated_at="2026-07-28T20:52:30Z",
            error=None,
        ),
    )

    assert _check(diagnose(**kwargs), "headroom.member.unreachable.") == []


def test_a_member_key_naming_neither_a_slot_nor_an_extra_window_fails():
    policy = _policy_with_gemini_windows(members={"gemini-typo": ["claude-gemini-pro"]})
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_readings"] = (_gemini_reading(),)

    check = _check(
        diagnose(**kwargs), "headroom.member.unreachable.pool:gemini.gemini-typo"
    )[0]

    assert check.ok is False
    assert "gemini-typo" in check.detail
    assert check.remedy is not None


def test_a_member_key_a_stored_reading_publishes_passes_when_the_live_one_omits_it():
    # codexbar drops an extra window and restores it between consecutive
    # calls -- measured 2026-07-29 on `claude-weekly-scoped-fable`, present,
    # absent and present again across three calls one minute apart. Failing
    # on that flap sends the operator to correct a correct line.
    policy = _policy_with_gemini_windows(members={"gemini-extra-window": ["claude-gemini-pro"]})
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_readings"] = (_gemini_reading(),)  # publishes no extra window
    kwargs["headroom_stored_extra_window_ids"] = {"pool:gemini": frozenset({"gemini-extra-window"})}

    check = _check(
        diagnose(**kwargs), "headroom.member.unreachable.pool:gemini.gemini-extra-window"
    )[0]

    assert check.ok is True
    assert "intermittently" in check.detail


def test_a_member_key_no_stored_reading_publishes_still_fails():
    policy = _policy_with_gemini_windows(members={"gemini-typo": ["claude-gemini-pro"]})
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_readings"] = (_gemini_reading(),)
    kwargs["headroom_stored_extra_window_ids"] = {"pool:gemini": frozenset({"some-other-window"})}

    check = _check(diagnose(**kwargs), "headroom.member.unreachable.pool:gemini.gemini-typo")[0]

    assert check.ok is False


def test_a_member_key_with_no_live_reading_raises_no_unreachable_check():
    # No Reading at all means the key cannot be judged -- a Check that
    # cannot measure must not fail (same rule as `_headroom_mapping_checks`
    # above).
    policy = _policy_with_gemini_windows(members={"gemini-typo": ["claude-gemini-pro"]})
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_readings"] = None

    assert _check(diagnose(**kwargs), "headroom.member.unreachable.") == []


# --- Ticket 11: `all_accounts_providers` against `sources`, Policy only ---
#
# `codexbar --provider codex` alone returns one Reading; two accounts on
# one provider need `all_accounts_providers` to name it. Both findings
# below read Policy alone -- no live Reading needed, unlike the mapping
# checks above.


def test_an_all_accounts_provider_no_source_reaches_fails():
    policy = _policy_with_headroom(all_accounts_providers=["codex"])
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(diagnose(**kwargs), "headroom.all_accounts.unreachable.codex")[0]

    assert check.ok is False
    assert "no 'headroom.sources' entry reaches" in check.detail
    assert check.remedy is not None


def test_an_all_accounts_provider_a_source_reaches_passes():
    policy = _policy_with_headroom(
        sources={
            "credential:SEAT1_KEY": "codexbar:codex/one@example.com",
            "credential:SEAT2_KEY": "codexbar:codex/two@example.com",
        },
        all_accounts_providers=["codex"],
    )
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(diagnose(**kwargs), "headroom.all_accounts.unreachable.codex")[0]

    assert check.ok is True


def test_two_sources_sharing_a_provider_id_with_no_marker_fails():
    policy = _policy_with_headroom(
        sources={
            "credential:SEAT1_KEY": "codexbar:codex/one@example.com",
            "credential:SEAT2_KEY": "codexbar:codex/two@example.com",
        }
    )
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(diagnose(**kwargs), "headroom.all_accounts.unmarked.codex")[0]

    assert check.ok is False
    assert "credential:SEAT1_KEY" in check.detail
    assert "credential:SEAT2_KEY" in check.detail
    assert check.remedy is not None
    assert "all_accounts_providers" in check.remedy


def test_two_sources_sharing_a_provider_id_with_the_marker_passes():
    policy = _policy_with_headroom(
        sources={
            "credential:SEAT1_KEY": "codexbar:codex/one@example.com",
            "credential:SEAT2_KEY": "codexbar:codex/two@example.com",
        },
        all_accounts_providers=["codex"],
    )
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(diagnose(**kwargs), "headroom.all_accounts.unmarked.codex")[0]

    assert check.ok is True


def test_one_source_per_provider_raises_no_unmarked_finding():
    # A single account needs no marker at all: nothing to discriminate.
    policy = _policy_with_headroom()
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    assert _check(diagnose(**kwargs), "headroom.all_accounts.unmarked.") == []


def test_no_headroom_and_no_all_accounts_providers_raises_no_finding():
    policy = parse_policy(_policy_raw())
    feed = parse_feed(_feed_raw())
    diagnosis = diagnose(**_all_ok_kwargs(policy, feed))

    assert not [c for c in diagnosis.checks if c.name.startswith("headroom.all_accounts.")]


def test_an_installed_refresh_job_with_a_mismatched_interval_fails():
    policy = _policy_with_headroom(interval_minutes=15)
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_installed_interval_seconds"] = 3600  # baked at 60 minutes
    kwargs["headroom_plist_path"] = "/tmp/headroom.plist"

    check = _check(diagnose(**kwargs), "headroom.refresh_interval")[0]

    assert check.ok is False
    assert "15" in check.detail
    assert check.remedy is not None
    assert "install" in check.remedy


def test_an_installed_refresh_job_with_a_matching_interval_passes():
    policy = _policy_with_headroom(interval_minutes=15)
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_installed_interval_seconds"] = 900
    kwargs["headroom_plist_path"] = "/tmp/headroom.plist"

    check = _check(diagnose(**kwargs), "headroom.refresh_interval")[0]

    assert check.ok is True


def test_no_installed_interval_reads_as_not_checked_not_failed():
    policy = _policy_with_headroom()
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_installed_interval_seconds"] = None

    check = _check(diagnose(**kwargs), "headroom.refresh_interval")[0]

    assert check.ok is True
    assert "not checked" in check.detail


def test_a_healthy_machine_with_no_headroom_sources_raises_no_finding_and_no_warning():
    """Silence is the correct output for a capability that is switched off."""
    policy = parse_policy(_policy_raw())
    feed = parse_feed(_feed_raw())
    diagnosis = diagnose(**_all_ok_kwargs(policy, feed))

    assert not [c for c in diagnosis.checks if c.name.startswith("headroom.")]
    assert diagnosis.ok is True


def test_a_healthy_machine_with_headroom_mapped_still_exits_ok():
    policy = _policy_with_headroom(interval_minutes=15)
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_readings"] = (_reading(),)
    kwargs["headroom_binary_present"] = True
    kwargs["headroom_installed_interval_seconds"] = 900
    kwargs["headroom_plist_path"] = "/tmp/headroom.plist"

    diagnosis = diagnose(**kwargs)

    assert diagnosis.ok is True
    headroom_checks = [c for c in diagnosis.checks if c.name.startswith("headroom.")]
    assert headroom_checks
    assert all(c.ok for c in headroom_checks)


# --- Ticket 12: an `allowances` entry naming an unreachable Allowance ------


def test_an_allowances_entry_a_discovered_offering_reaches_passes():
    policy = parse_policy(
        _policy_raw(allowances={"provider:openrouter": {"tier": "openrouter-free"}})
    )
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(diagnose(**kwargs), "allowances.provider:openrouter")[0]

    assert check.ok is True


def test_an_allowances_entry_no_offering_reaches_fails():
    policy = parse_policy(
        _policy_raw(allowances={"provider:not-a-real-provider": {"tier": "some-tier"}})
    )
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(diagnose(**kwargs), "allowances.provider:not-a-real-provider")[0]

    assert check.ok is False
    assert "no Offering" in check.detail
    assert check.remedy is not None


def test_an_allowances_entry_a_declared_offering_reaches_passes():
    policy = parse_policy(
        _policy_raw(
            declared=[
                {
                    "alias": "claude-direct-1",
                    "litellm_params": {"model": "anthropic/claude-x"},
                    "entitlement_pool": "claude-subscription",
                }
            ],
            allowances={"pool:claude-subscription": {"tier": "claude-max-5x"}},
        )
    )
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(diagnose(**kwargs), "allowances.pool:claude-subscription")[0]

    assert check.ok is True


def test_no_allowances_block_raises_no_finding():
    policy = parse_policy(_policy_raw())
    feed = parse_feed(_feed_raw())
    diagnosis = diagnose(**_all_ok_kwargs(policy, feed))

    assert not [c for c in diagnosis.checks if c.name.startswith("allowances.")]


# --- `unmeasured`: a Health Key that draws on no published window --------


def test_an_unmeasured_health_key_counts_as_claimed():
    # Gemini's three slots hold Pro, Flash and Flash Lite; the same account
    # serves Gemma, which none of them measures. Stating that is the honest
    # alternative to filing Gemma under a slot that does not measure it.
    policy = _policy_with_gemini_windows(
        members={"gemini-pro": []}, unmeasured=["claude-gemini-pro"]
    )
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(diagnose(**kwargs), "headroom.member.unclaimed.pool:gemini")[0]

    assert check.ok is True


def test_an_unmeasured_entry_naming_no_known_health_key_fails():
    policy = _policy_with_gemini_windows(
        members={"gemini-pro": ["claude-gemini-pro"]}, unmeasured=["gemini:typo"]
    )
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(
        diagnose(**kwargs), "headroom.unmeasured.unknown.pool:gemini.gemini:typo"
    )[0]

    assert check.ok is False
    assert check.remedy is not None


def test_an_unmeasured_entry_naming_a_known_health_key_passes():
    policy = _policy_with_gemini_windows(
        members={"gemini-pro": []}, unmeasured=["claude-gemini-pro"]
    )
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(
        diagnose(**kwargs), "headroom.unmeasured.unknown.pool:gemini.claude-gemini-pro"
    )[0]

    assert check.ok is True


def test_a_declared_slot_with_an_explicitly_empty_member_list_passes():
    # Absent means "nobody assigned it yet"; an empty list means "nothing
    # admitted draws on it". Gemini's Pro slot is the measured case: the
    # free plan includes no Pro, and dropping the slot would put Pro's 100%
    # back into the parent's worst-of computation.
    policy = _policy_with_gemini_windows(
        members={"gemini-pro": []}, unmeasured=["claude-gemini-pro"]
    )
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(diagnose(**kwargs), "headroom.member.empty.pool:gemini.gemini-pro")[0]

    assert check.ok is True


def test_a_declared_slot_absent_from_members_still_fails():
    policy = _policy_with_gemini_windows(members={"gemini-flash": ["claude-gemini-pro"]})
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(diagnose(**kwargs), "headroom.member.empty.pool:gemini.gemini-pro")[0]

    assert check.ok is False
    assert check.remedy is not None


# --- A mapped Allowance that states no Tier, reported 2026-07-29 ---------


def _policy_with_headroom_and_allowances(allowances=None):
    raw = _policy_raw(
        headroom={"sources": {"provider:openrouter": "codexbar:openrouter/"}},
    )
    if allowances is not None:
        raw["allowances"] = allowances
    return parse_policy(raw)


def test_a_mapped_allowance_naming_no_tier_entry_at_all_fails():
    policy = _policy_with_headroom_and_allowances()
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(
        diagnose(**kwargs), "allowances.tier_unstated.provider:openrouter"
    )[0]

    assert check.ok is False
    assert check.remedy is not None


def test_an_allowance_entry_stating_a_tier_passes():
    policy = _policy_with_headroom_and_allowances(
        {"provider:openrouter": {"tier": "openrouter-paid"}}
    )
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    assert _check(diagnose(**kwargs), "allowances.tier_unstated.") == []


def test_an_allowance_entry_with_no_tier_key_still_silences_the_check():
    # A present entry says the operator looked and found no subscription
    # level to state. No entry says nobody looked. The two must not share
    # a value -- the same rule an empty `members` list follows.
    policy = _policy_with_headroom_and_allowances({"provider:openrouter": {}})
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    assert _check(diagnose(**kwargs), "allowances.tier_unstated.") == []


def test_an_unmapped_allowance_raises_no_tier_check():
    # Only an Allowance that publishes a Headroom needs a Tier: a share is
    # what makes the scale matter, and an unmapped Allowance states none.
    policy = parse_policy(_policy_raw())
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    assert _check(diagnose(**kwargs), "allowances.tier_unstated.") == []


def test_a_mapped_allowance_no_offering_reaches_raises_no_tier_check():
    # A Tier for an Allowance nothing reaches describes nothing, and the
    # mapping itself is what is wrong. One finding, not two.
    policy = parse_policy(
        _policy_raw(headroom={"sources": {"provider:nobody": "codexbar:nobody/"}})
    )
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    assert _check(diagnose(**kwargs), "allowances.tier_unstated.") == []


# --- A refresh job that stopped, added 2026-07-29 ------------------------
#
# Measured that day: the job was never registered, Headroom State sat 4.9
# hours stale, every figure kept publishing, and `doctor` exited 0.
# `guidance` and `entitlements` had warned all along; `doctor` had not.


def test_a_stale_headroom_fails_a_check_naming_the_job():
    policy = _policy_with_headroom()
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_staleness_warnings"] = (
        "'provider:claude''s Headroom was last refreshed 4.9 h ago",
    )

    check = _check(diagnose(**kwargs), "headroom.refresh_current")[0]

    assert check.ok is False
    assert "4.9 h ago" in check.detail
    assert check.remedy is not None
    assert "headroom-refresh" in check.remedy


def test_a_current_headroom_passes_the_check():
    policy = _policy_with_headroom()
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)
    kwargs["headroom_staleness_warnings"] = ()

    check = _check(diagnose(**kwargs), "headroom.refresh_current")[0]

    assert check.ok is True


def test_a_machine_with_no_headroom_source_raises_no_staleness_check():
    # The capability is off. Silence is the correct output.
    policy = parse_policy(_policy_raw())
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    assert _check(diagnose(**kwargs), "headroom.refresh_current") == []


def test_a_draw_note_naming_no_known_health_key_fails():
    policy = parse_policy(_policy_raw(draw_notes={"openrouter:gone": "10% of normal"}))
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(diagnose(**kwargs), "draw_notes.unknown.openrouter:gone")[0]

    assert check.ok is False
    assert check.remedy is not None


def test_a_draw_note_naming_a_known_health_key_passes():
    policy = parse_policy(
        _policy_raw(draw_notes={"openrouter:vendor/coder-large": "10% of normal"})
    )
    feed = parse_feed(_feed_raw())
    kwargs = _all_ok_kwargs(policy, feed)

    check = _check(
        diagnose(**kwargs), "draw_notes.unknown.openrouter:vendor/coder-large"
    )[0]

    assert check.ok is True
