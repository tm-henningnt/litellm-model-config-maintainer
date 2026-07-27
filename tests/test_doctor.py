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

from litellm_maintainer.doctor import Check, Diagnosis, diagnose, render_text
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
