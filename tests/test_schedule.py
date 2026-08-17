"""Tests for the schedule gate (`due`), the plist, and the `run` tick.

See CONTEXT.md, the spec's "Schedule" section, and
`.scratch/maintainer-v1/spec-corrections.md`, correction 9: the
scheduled tick chains probe, reduce, then plan. It never plans alone.

Every test name states a rule an operator would recognise, per the
spec's "What makes a good test here".
"""

from __future__ import annotations

import inspect
import json
import plistlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

import litellm_maintainer.cli as cli_module
import litellm_maintainer.schedule as schedule_module
from litellm_maintainer.cli import build_parser, main
from litellm_maintainer.plan import PlanReport, PlanResult
from litellm_maintainer.policy import Schedule, load_policy
from litellm_maintainer.reduce import HealthState, OfferingHealth
from litellm_maintainer.schedule import (
    build_headroom_plist_spec,
    build_plist_spec,
    due,
    health_state_age,
    install,
    launchctl_load_command,
    launchctl_unload_command,
    render_plist,
    uninstall,
)

NOW = datetime(2026, 7, 25, 12, 0, tzinfo=timezone.utc)
FRESH = timedelta(hours=1)


def _schedule(
    *,
    enabled: bool = True,
    interval_minutes: int = 60,
    require_proxy: bool = True,
    maximum_staleness_hours: float = 24.0,
) -> Schedule:
    return Schedule(
        enabled=enabled,
        interval_minutes=interval_minutes,
        require_proxy=require_proxy,
        maximum_staleness_hours=maximum_staleness_hours,
    )


# --- due: the gate rules -----------------------------------------------


def test_a_disabled_schedule_does_not_run():
    decision = due(
        schedule=_schedule(enabled=False),
        last_run_at=NOW - timedelta(hours=5),
        proxy_up=True,
        health_age=FRESH,
        now=NOW,
    )
    assert decision.run is False
    assert "disabled" in decision.reason


def test_an_interval_that_has_not_elapsed_does_not_run():
    decision = due(
        schedule=_schedule(interval_minutes=60),
        last_run_at=NOW - timedelta(minutes=10),
        proxy_up=True,
        health_age=FRESH,
        now=NOW,
    )
    assert decision.run is False
    assert "interval" in decision.reason


def test_an_interval_that_has_elapsed_runs():
    decision = due(
        schedule=_schedule(interval_minutes=60),
        last_run_at=NOW - timedelta(minutes=60),
        proxy_up=True,
        health_age=FRESH,
        now=NOW,
    )
    assert decision.run is True
    assert decision.catch_up is False


def test_a_first_ever_run_has_no_last_run_time_and_still_runs():
    decision = due(
        schedule=_schedule(interval_minutes=60),
        last_run_at=None,
        proxy_up=True,
        health_age=FRESH,
        now=NOW,
    )
    assert decision.run is True
    assert decision.catch_up is False


def test_a_down_proxy_does_not_run_when_policy_requires_a_proxy():
    decision = due(
        schedule=_schedule(require_proxy=True, interval_minutes=60),
        last_run_at=NOW - timedelta(minutes=60),
        proxy_up=False,
        health_age=FRESH,
        now=NOW,
    )
    assert decision.run is False
    assert "proxy" in decision.reason


def test_a_down_proxy_does_run_when_policy_does_not_require_one():
    decision = due(
        schedule=_schedule(require_proxy=False, interval_minutes=60),
        last_run_at=NOW - timedelta(minutes=60),
        proxy_up=False,
        health_age=FRESH,
        now=NOW,
    )
    assert decision.run is True


def test_health_older_than_maximum_staleness_forces_a_run_despite_a_down_proxy():
    decision = due(
        schedule=_schedule(
            require_proxy=True, interval_minutes=60, maximum_staleness_hours=24.0
        ),
        # The interval HAS elapsed here: staleness overrides the proxy
        # requirement, never the interval (defect 1).
        last_run_at=NOW - timedelta(minutes=60),
        proxy_up=False,
        health_age=timedelta(hours=25),
        now=NOW,
    )
    assert decision.run is True
    assert "stale" in decision.reason.lower()


def test_stale_health_does_not_run_before_the_interval_has_elapsed():
    """Defect 1: staleness overrides the PROXY requirement, not the
    interval. Without this rule, Health State that has never been
    recorded (the default state on a fresh install) counts as stale on
    every tick, so a launchd tick firing every 60 seconds would run the
    full probe sweep on every single tick while the proxy stays down --
    the tick storm that made a provider report
    `Worker local total request limit reached` (docs/gotchas.md)."""
    decision = due(
        schedule=_schedule(
            require_proxy=True, interval_minutes=60, maximum_staleness_hours=24.0
        ),
        # The interval has NOT elapsed: one minute since the last run.
        last_run_at=NOW - timedelta(minutes=1),
        proxy_up=False,
        health_age=timedelta(hours=99),
        now=NOW,
    )
    assert decision.run is False
    assert "interval" in decision.reason.lower()


def test_two_ticks_a_minute_apart_produce_at_most_one_run():
    """A launchd tick fires far more often than the interval. Even with
    Health State maximally stale and the proxy down throughout, two
    ticks one minute apart must not both run."""
    schedule = _schedule(require_proxy=True, interval_minutes=60, maximum_staleness_hours=24.0)

    first = due(
        schedule=schedule,
        last_run_at=NOW - timedelta(minutes=1),
        proxy_up=False,
        health_age=timedelta(hours=99),
        now=NOW,
    )
    second = due(
        schedule=schedule,
        last_run_at=NOW - timedelta(minutes=1),
        proxy_up=False,
        health_age=timedelta(hours=99),
        now=NOW + timedelta(minutes=1),
    )
    assert sum(1 for decision in (first, second) if decision.run) <= 1


def test_health_never_recorded_counts_as_stale_and_forces_a_run_despite_a_down_proxy():
    decision = due(
        schedule=_schedule(require_proxy=True, interval_minutes=60),
        last_run_at=NOW - timedelta(minutes=60),
        proxy_up=False,
        health_age=None,
        now=NOW,
    )
    assert decision.run is True


def test_fresh_health_does_not_force_a_run_past_a_down_proxy():
    decision = due(
        schedule=_schedule(
            require_proxy=True, interval_minutes=60, maximum_staleness_hours=24.0
        ),
        last_run_at=NOW - timedelta(minutes=60),
        proxy_up=False,
        health_age=timedelta(hours=1),
        now=NOW,
    )
    assert decision.run is False


def test_a_proxy_returning_after_a_long_absence_produces_exactly_one_catch_up_run():
    """Two consecutive ticks. The first, after a long gap, is a catch-up.
    The second, moments after the first ran, is not — because the run
    that just happened advances `last_run_at`, as `cli.cmd_run` does."""
    schedule = _schedule(require_proxy=True, interval_minutes=60)

    first = due(
        schedule=schedule,
        last_run_at=NOW - timedelta(hours=10),
        proxy_up=True,
        health_age=FRESH,
        now=NOW,
    )
    assert first.run is True
    assert first.catch_up is True

    second = due(
        schedule=schedule,
        last_run_at=NOW,  # the pipeline recorded the catch-up run at NOW
        proxy_up=True,
        health_age=FRESH,
        now=NOW + timedelta(minutes=1),
    )
    assert second.run is False
    assert second.catch_up is False


def test_an_ordinary_on_time_run_is_not_a_catch_up():
    decision = due(
        schedule=_schedule(require_proxy=True, interval_minutes=60),
        last_run_at=NOW - timedelta(minutes=61),
        proxy_up=True,
        health_age=FRESH,
        now=NOW,
    )
    assert decision.run is True
    assert decision.catch_up is False


def test_schedule_module_reads_no_clock_environment_or_network():
    source = inspect.getsource(schedule_module)
    for forbidden in ("datetime.now", "utcnow", "os.environ", "httpx"):
        assert forbidden not in source, f"{forbidden!r} must not appear in schedule.py"


# --- health_state_age ----------------------------------------------------


def test_health_state_age_is_none_when_nothing_has_ever_answered():
    assert health_state_age({}, now=NOW) is None


def test_health_state_age_is_the_gap_since_the_most_recent_success():
    offerings = {
        "a": OfferingHealth(last_success_at=NOW - timedelta(hours=5)),
        "b": OfferingHealth(last_success_at=NOW - timedelta(hours=1)),
    }
    assert health_state_age(offerings, now=NOW) == timedelta(hours=1)


# --- install / uninstall / the plist -------------------------------------


def _spec(*, home: str | None = None) -> schedule_module.PlistSpec:
    return build_plist_spec(
        python_executable="/usr/bin/python3",
        policy_path="/instance/policy.yaml",
        feed_path="/instance/feed.json",
        home=home,
    )


def test_install_twice_leaves_one_job(tmp_path: Path):
    target = tmp_path / "LaunchAgents"
    spec = _spec()

    install(target, spec)
    install(target, spec)

    matches = list(target.glob("*.plist"))
    assert len(matches) == 1


def test_uninstall_with_nothing_installed_does_not_fail(tmp_path: Path):
    target = tmp_path / "LaunchAgents"
    assert uninstall(target) is None


def test_uninstall_removes_the_installed_job(tmp_path: Path):
    target = tmp_path / "LaunchAgents"
    spec = _spec()
    path = install(target, spec)
    assert path.exists()

    removed = uninstall(target, spec.label)

    assert removed == path
    assert not path.exists()


def _policy_raw(interval_minutes: int) -> dict:
    return {
        "providers": {},
        "quality": {"minimum_coding_score": 18},
        "approved_candidates": [],
        "naming": {"alias_prefix": "claude-", "provider_labels": {}, "alias_overrides": {}},
        "withheld": {},
        "declared": [],
        "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
        "schedule": {
            "enabled": True,
            "interval_minutes": interval_minutes,
            "require_proxy": True,
            "maximum_staleness_hours": 24,
        },
        "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
    }


def test_changing_the_interval_in_policy_does_not_change_the_plist(tmp_path: Path):
    policy_path = tmp_path / "policy.yaml"

    policy_path.write_text(yaml.safe_dump(_policy_raw(5)))
    policy_short = load_policy(policy_path)
    plist_short = render_plist(
        build_plist_spec(
            python_executable="/usr/bin/python3",
            policy_path=str(policy_path),
            feed_path="/instance/feed.json",
        )
    )

    policy_path.write_text(yaml.safe_dump(_policy_raw(120)))
    policy_long = load_policy(policy_path)
    plist_long = render_plist(
        build_plist_spec(
            python_executable="/usr/bin/python3",
            policy_path=str(policy_path),
            feed_path="/instance/feed.json",
        )
    )

    assert policy_short.schedule.interval_minutes != policy_long.schedule.interval_minutes
    assert plist_short == plist_long


def test_disabling_the_schedule_in_policy_does_not_change_the_plist(tmp_path: Path):
    policy_path = tmp_path / "policy.yaml"

    enabled_raw = _policy_raw(60)
    policy_path.write_text(yaml.safe_dump(enabled_raw))
    plist_enabled = render_plist(
        build_plist_spec(
            python_executable="/usr/bin/python3",
            policy_path=str(policy_path),
            feed_path="/instance/feed.json",
        )
    )

    disabled_raw = _policy_raw(60)
    disabled_raw["schedule"]["enabled"] = False
    policy_path.write_text(yaml.safe_dump(disabled_raw))
    plist_disabled = render_plist(
        build_plist_spec(
            python_executable="/usr/bin/python3",
            policy_path=str(policy_path),
            feed_path="/instance/feed.json",
        )
    )

    assert plist_enabled == plist_disabled


def test_install_never_calls_launchctl_it_only_prints_the_command(tmp_path: Path):
    source = inspect.getsource(schedule_module)
    assert "subprocess" not in source

    target = tmp_path / "LaunchAgents"
    path = install(target, _spec())

    assert launchctl_load_command(path) == f"launchctl load -w {path}"
    assert launchctl_unload_command(path) == f"launchctl unload {path}"


# --- the `run` command: chaining, --dry-run, and CLI-level install ------


def _run_policy_raw(*, require_proxy: bool = False) -> dict:
    return {
        "providers": {},
        "quality": {"minimum_coding_score": 18},
        "approved_candidates": [],
        "naming": {"alias_prefix": "claude-", "provider_labels": {}, "alias_overrides": {}},
        "withheld": {},
        "declared": [],
        "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
        "schedule": {
            "enabled": True,
            "interval_minutes": 60,
            "require_proxy": require_proxy,
            "maximum_staleness_hours": 24,
        },
        "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
    }


def _write_run_fixtures(tmp_path: Path, *, require_proxy: bool = False) -> tuple[Path, Path]:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(_run_policy_raw(require_proxy=require_proxy)))
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps({"schema_version": "test", "providers": [], "models": []}))
    return policy_path, feed_path


def _run_args(policy_path: Path, feed_path: Path, tmp_path: Path, *, dry_run: bool = False):
    parser = build_parser()
    argv = [
        "run",
        "--policy",
        str(policy_path),
        "--feed",
        str(feed_path),
        "--home",
        str(tmp_path),
        "--out",
        str(tmp_path / "config.yaml"),
    ]
    if dry_run:
        argv.append("--dry-run")
    return parser.parse_args(argv)


def test_run_dry_run_calls_the_proxy_checker_the_transport_and_the_notifier_never():
    """`run --dry-run` reports what it would do and calls nothing."""

    def _boom(*a, **k):
        raise AssertionError("dry-run must call nothing")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        policy_path, feed_path = _write_run_fixtures(tmp_path)
        args = _run_args(policy_path, feed_path, tmp_path, dry_run=True)

        exit_code = cli_module.cmd_run(
            args,
            proxy_checker=_boom,
            probe_transport=_boom,
            notifier=_boom,
        )

        assert exit_code == 0
        # No Health State, no Generated Config, no run log: nothing written.
        assert not (tmp_path / "state" / "health.json").exists()
        assert not (tmp_path / "config.yaml").exists()


def test_run_chains_probe_then_reduce_then_plan_never_plans_alone(tmp_path: Path, monkeypatch):
    """Correction 9: the scheduled tick probes, reduces, then plans, in
    that order, and `plan` sees the Health State `reduce` just produced
    -- never the empty dict `generate` uses on its own."""
    policy_path, feed_path = _write_run_fixtures(tmp_path, require_proxy=False)
    args = _run_args(policy_path, feed_path, tmp_path)

    call_order: list[str] = []
    post_probe_health = {"marker:offering": OfferingHealth(bucket="answered")}

    def fake_probe_offerings(targets, *, pacing, transport, now):
        call_order.append("probe_offerings")
        return {}

    def fake_reduce(
        *, prior, outcomes, observations, admitted, passthrough_auth, now, **kwargs
    ):
        call_order.append("reduce")
        assert prior.offerings == {}
        return HealthState(offerings=post_probe_health)

    def fake_plan(*, feed, policy, health, now):
        call_order.append("plan")
        assert health == post_probe_health, (
            "plan must see the Health State reduce just wrote, not the "
            "empty prior Health State"
        )
        return PlanResult(
            config={
                "model_list": [
                    {"model_name": "claude-test", "litellm_params": {"model": "anthropic/test"}}
                ]
            },
            report=PlanReport(admitted=("declared:test",), aliases={}),
        )

    monkeypatch.setattr(cli_module, "probe_offerings", fake_probe_offerings)
    monkeypatch.setattr(cli_module, "reduce", fake_reduce)
    monkeypatch.setattr(cli_module, "plan", fake_plan)

    exit_code = cli_module.cmd_run(
        args,
        proxy_checker=lambda: True,
        probe_transport=lambda target: None,
        notifier=lambda message: None,
    )

    assert exit_code == 0
    assert call_order == ["probe_offerings", "reduce", "plan"]
    written = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert [e["model_name"] for e in written["model_list"]] == ["claude-test"]


def test_run_folds_the_observation_journal_into_health_state(tmp_path: Path, monkeypatch):
    """Defect 2: `cmd_probe` and `cmd_run` used to pass `observations=[]`
    to `reduce`, so a failure the proxy recorded in the Observation
    Journal never reached Health State -- ticket 15's whole purpose,
    stories 36, 37 and 39. A real Journal entry for an Offering Policy
    currently declares must change the Health State this run writes."""
    from litellm_maintainer.classify import Outcome
    from litellm_maintainer.health import read_health
    from litellm_maintainer.journal import append_observation
    from litellm_maintainer.paths import health_path, journal_path
    from litellm_maintainer.reduce import Observation

    raw = _run_policy_raw(require_proxy=False)
    raw["declared"] = [
        {"alias": "claude-declared-test", "litellm_params": {"model": "anthropic/claude-test"}},
        # A healthy sibling: the gateway failure below Excludes
        # `claude-declared-test`, and an Excluded Offering leaves the
        # Generated Config, so a second Alias keeps the run from the
        # zero-offered refusal.
        {"alias": "claude-declared-healthy", "litellm_params": {"model": "anthropic/claude-two"}},
    ]
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(raw))
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps({"schema_version": "test", "providers": [], "models": []}))
    args = _run_args(policy_path, feed_path, tmp_path)

    append_observation(
        journal_path(tmp_path),
        Observation(
            offering_id="claude-declared-test",
            observed_at=NOW - timedelta(minutes=1),
            outcome=Outcome(bucket="self_healing", reset_at=None, reason="gateway_error"),
        ),
    )

    from litellm_maintainer.prober import TransportResponse

    def fake_probe_offerings(targets, *, pacing, transport, now):
        return {}

    monkeypatch.setattr(cli_module, "probe_offerings", fake_probe_offerings)

    exit_code = cli_module.cmd_run(
        args,
        proxy_checker=lambda: True,
        probe_transport=lambda target: None,
        notifier=lambda message: None,
        # Never let the post-write smoke check reach a real proxy: the
        # test's only interest is Health State, not the smoke check.
        smoke_transport=lambda entry: TransportResponse(
            http_status=None, body=None, transport="not-called-in-this-test"
        ),
    )

    assert exit_code == 0
    written = read_health(health_path(tmp_path))
    record = written.offerings["claude-declared-test"]
    assert record.excluded is True
    assert record.reason == "gateway_error"
    # The Excluded Declared Offering keeps its entry. A gateway error can
    # clear on the next Probe, and a write here would restart the proxy
    # for it (ADR 0014). Both Aliases stay; only the recommendation
    # changes.
    config = yaml.safe_load((tmp_path / "config.yaml").read_text())
    names = [e["model_name"] for e in config["model_list"]]
    assert sorted(names) == ["claude-declared-healthy", "claude-declared-test"]


def test_run_refuses_an_implausibly_short_feed_the_same_way_generate_does(tmp_path: Path):
    """Defect 5: `cmd_run` -- the SCHEDULED path, the one that runs
    unattended -- used to load the Feed and go straight to probe and
    plan, skipping the implausible-Feed refusal `cmd_generate` applies
    (story 45, spec line 433). A Feed this short must refuse the run,
    exactly as `generate` would, and must probe nothing first."""
    # A Declared Offering passes through unconditionally, whatever the
    # Feed says (CONTEXT.md, "Declared Offering"), so it would make
    # `plan` admit one Alias regardless of the implausible-Feed check --
    # this isolates the refusal itself from the separate "zero offered"
    # safety rail, which would otherwise also return exit code 1 here
    # and mask whether the implausible-Feed check ran at all.
    raw = _run_policy_raw(require_proxy=False)
    raw["providers"] = {"acme": {"mode": "all"}}
    raw["declared"] = [
        {"alias": "claude-declared-test", "litellm_params": {"model": "anthropic/claude-test"}}
    ]
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(raw))
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps({"schema_version": "test", "providers": [], "models": []}))
    args = _run_args(policy_path, feed_path, tmp_path)

    def _boom(*a, **k):
        raise AssertionError("a refused run must probe nothing")

    exit_code = cli_module.cmd_run(
        args, proxy_checker=lambda: True, probe_transport=_boom, notifier=_boom
    )

    assert exit_code == 1
    assert not (tmp_path / "config.yaml").exists()


def test_run_performs_the_smoke_check_after_a_successful_write_and_does_not_block_it(
    tmp_path: Path, monkeypatch, capsys
):
    """Defect 5's second half: `cmd_run` never called `run_smoke_check`
    (stories 48 and 49, spec line 377: "one call per distinct
    translation rule goes through the running proxy"), so a stale proxy
    environment after a deploy was caught by nobody on the scheduled
    path. The smoke check must run after a successful write, report a
    failure loudly, and never block or revert the write."""
    from litellm_maintainer.prober import TransportResponse

    raw = _run_policy_raw(require_proxy=False)
    raw["declared"] = [
        {"alias": "claude-declared-smoke", "litellm_params": {"model": "anthropic/claude-test"}}
    ]
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(raw))
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps({"schema_version": "test", "providers": [], "models": []}))
    args = _run_args(policy_path, feed_path, tmp_path)

    def fake_probe_offerings(targets, *, pacing, transport, now):
        return {}

    monkeypatch.setattr(cli_module, "probe_offerings", fake_probe_offerings)

    def failing_smoke_transport(entry):
        return TransportResponse(http_status=500, body={"error": "boom"}, transport=None)

    exit_code = cli_module.cmd_run(
        args,
        proxy_checker=lambda: True,
        probe_transport=lambda target: None,
        notifier=lambda message: None,
        smoke_transport=failing_smoke_transport,
    )

    assert exit_code == 0, "a smoke failure must never block the write"
    written = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert [e["model_name"] for e in written["model_list"]] == ["claude-declared-smoke"]

    captured = capsys.readouterr()
    assert "FAILED" in captured.out or "FAILED" in captured.err


def test_run_skips_and_logs_why_when_the_schedule_is_disabled(tmp_path: Path):
    policy_path = tmp_path / "policy.yaml"
    raw = _run_policy_raw()
    raw["schedule"]["enabled"] = False
    policy_path.write_text(yaml.safe_dump(raw))
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps({"schema_version": "test", "providers": [], "models": []}))
    args = _run_args(policy_path, feed_path, tmp_path)

    def _boom(*a, **k):
        raise AssertionError("a disabled schedule must call no transport or checker")

    exit_code = cli_module.cmd_run(
        args, proxy_checker=_boom, probe_transport=_boom, notifier=_boom
    )

    assert exit_code == 0
    log_path = tmp_path / "state" / "runs.log"
    assert "disabled" in log_path.read_text()
    assert not (tmp_path / "config.yaml").exists()


def test_install_command_writes_a_plist_and_uninstall_removes_it(tmp_path: Path):
    target_dir = tmp_path / "LaunchAgents"

    install_exit = main(
        [
            "install",
            "--policy",
            "/instance/policy.yaml",
            "--feed",
            "/instance/feed.json",
            "--target-dir",
            str(target_dir),
        ]
    )
    assert install_exit == 0
    plists = list(target_dir.glob("*.plist"))
    assert len(plists) == 1

    # Idempotent through the CLI too.
    main(
        [
            "install",
            "--policy",
            "/instance/policy.yaml",
            "--feed",
            "/instance/feed.json",
            "--target-dir",
            str(target_dir),
        ]
    )
    assert len(list(target_dir.glob("*.plist"))) == 1

    uninstall_exit = main(["uninstall", "--target-dir", str(target_dir)])
    assert uninstall_exit == 0
    assert list(target_dir.glob("*.plist")) == []


def test_uninstall_command_with_nothing_installed_does_not_fail(tmp_path: Path):
    target_dir = tmp_path / "LaunchAgents"
    exit_code = main(["uninstall", "--target-dir", str(target_dir)])
    assert exit_code == 0


# --- A Journal entry elapses the interval ----------------------------------


def test_a_recorded_failure_runs_before_the_interval_has_elapsed():
    """The Observation Journal exists so a failure reaches Health State
    within seconds. Without this rule a quota exhaustion recorded one
    minute into a 60-minute interval waits 59 minutes -- slower than
    the tick it was built to beat."""
    decision = due(
        schedule=_schedule(interval_minutes=60, require_proxy=False),
        last_run_at=NOW - timedelta(minutes=10),
        proxy_up=True,
        health_age=FRESH,
        now=NOW,
        journal_pending=True,
    )
    assert decision.run is True
    assert "recorded a failure" in decision.reason


def test_an_unelapsed_interval_with_an_empty_journal_still_does_not_run():
    """The clock alone never reopens the tick storm."""
    decision = due(
        schedule=_schedule(interval_minutes=60, require_proxy=False),
        last_run_at=NOW - timedelta(minutes=1),
        proxy_up=True,
        health_age=timedelta(hours=99),
        now=NOW,
        journal_pending=False,
    )
    assert decision.run is False
    assert decision.reason == "the interval has not elapsed"


def test_a_disabled_schedule_ignores_a_recorded_failure():
    """Rule 1 is absolute. Nothing overrides a disabled schedule."""
    decision = due(
        schedule=_schedule(enabled=False),
        last_run_at=None,
        proxy_up=True,
        health_age=None,
        now=NOW,
        journal_pending=True,
    )
    assert decision.run is False
    assert "disabled" in decision.reason


def test_a_recorded_failure_does_not_override_a_down_proxy():
    """A Journal entry elapses the INTERVAL. It is not a reason to run
    a pipeline that needs a proxy Policy says must be up."""
    decision = due(
        schedule=_schedule(interval_minutes=60, require_proxy=True),
        last_run_at=NOW - timedelta(minutes=10),
        proxy_up=False,
        health_age=FRESH,
        now=NOW,
        journal_pending=True,
    )
    assert decision.run is False
    assert "proxy" in decision.reason


def test_a_journal_entry_reaches_health_state_inside_an_unelapsed_interval(
    tmp_path: Path, monkeypatch
):
    """The defect this whole feature turned on.

    `cmd_run` applies the due gate first, and the interval is 60
    minutes. A quota exhaustion recorded one minute after the last run
    used to be answered with "the interval has not elapsed", so it
    reached Health State an hour late -- slower than the plain tick.
    """
    from litellm_maintainer.classify import Outcome
    from litellm_maintainer.health import read_health
    from litellm_maintainer.journal import append_observation
    from litellm_maintainer.paths import health_path, journal_path
    from litellm_maintainer.prober import TransportResponse
    from litellm_maintainer.reduce import Observation

    raw = _run_policy_raw(require_proxy=False)
    raw["schedule"]["interval_minutes"] = 60
    raw["declared"] = [
        {"alias": "claude-declared-test", "litellm_params": {"model": "anthropic/claude-test"}},
        {"alias": "claude-declared-healthy", "litellm_params": {"model": "anthropic/claude-two"}},
    ]
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(raw))
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps({"schema_version": "test", "providers": [], "models": []}))
    args = _run_args(policy_path, feed_path, tmp_path)

    # A run happened one minute ago, so the 60-minute interval has not
    # elapsed. Nothing but the Journal can make this tick run.
    run_log = tmp_path / "state" / "runs.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    recent = datetime.now(timezone.utc) - timedelta(minutes=10)
    run_log.write_text(f"{recent.isoformat()} run: offered=2\n")

    append_observation(
        journal_path(tmp_path),
        Observation(
            offering_id="claude-declared-test",
            observed_at=datetime.now(timezone.utc),
            outcome=Outcome(bucket="needs_operator", reset_at=None, reason="quota_exhausted"),
        ),
    )

    probed: list = []

    def fake_probe_offerings(targets, *, pacing, transport, now):
        probed.extend(targets)
        return {}

    monkeypatch.setattr(cli_module, "probe_offerings", fake_probe_offerings)

    exit_code = cli_module.cmd_run(
        args,
        proxy_checker=lambda: True,
        probe_transport=lambda target: None,
        notifier=lambda message: None,
        smoke_transport=lambda entry: TransportResponse(
            http_status=None, body=None, transport="not-called-in-this-test"
        ),
    )

    assert exit_code == 0
    record = read_health(health_path(tmp_path)).offerings["claude-declared-test"]
    assert record.excluded is True
    assert record.reason == "quota_exhausted"
    # A self-identifying failure acts without a confirming Probe. The
    # journal-triggered run must sweep nothing at all.
    assert probed == []


def test_an_ambiguous_journal_entry_is_confirmed_by_exactly_one_probe(
    tmp_path: Path, monkeypatch
):
    """One ambiguous call must never change Health State on its own."""
    from litellm_maintainer.classify import Outcome
    from litellm_maintainer.journal import append_observation
    from litellm_maintainer.paths import journal_path
    from litellm_maintainer.prober import TransportResponse
    from litellm_maintainer.reduce import Observation

    raw = _run_policy_raw(require_proxy=False)
    raw["schedule"]["interval_minutes"] = 60
    raw["declared"] = [
        {"alias": "claude-declared-test", "litellm_params": {"model": "anthropic/claude-test"}},
        {"alias": "claude-declared-healthy", "litellm_params": {"model": "anthropic/claude-two"}},
    ]
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(raw))
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps({"schema_version": "test", "providers": [], "models": []}))
    args = _run_args(policy_path, feed_path, tmp_path)

    run_log = tmp_path / "state" / "runs.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    recent = datetime.now(timezone.utc) - timedelta(minutes=10)
    run_log.write_text(f"{recent.isoformat()} run: offered=2\n")

    append_observation(
        journal_path(tmp_path),
        Observation(
            offering_id="claude-declared-test",
            observed_at=datetime.now(timezone.utc),
            outcome=Outcome(bucket="inconclusive", reset_at=None, reason="rate_limited"),
        ),
    )

    probed: list = []

    def fake_probe_offerings(targets, *, pacing, transport, now):
        probed.extend(targets)
        return {}

    monkeypatch.setattr(cli_module, "probe_offerings", fake_probe_offerings)

    cli_module.cmd_run(
        args,
        proxy_checker=lambda: True,
        probe_transport=lambda target: None,
        notifier=lambda message: None,
        smoke_transport=lambda entry: TransportResponse(
            http_status=None, body=None, transport="not-called-in-this-test"
        ),
    )

    # Exactly the ambiguous Offering, and nothing else -- not the
    # healthy sibling, and not a full sweep.
    assert [target.key for target in probed] == ["claude-declared-test"]


# --- The installed plist must name the config the proxy serves -------------


def test_the_installed_plist_passes_the_out_path_the_proxy_serves():
    """`run` defaults `--out` to the instance directory's own copy, which
    the proxy never reads. A plist without this argument computes the
    right config every tick and writes it where nothing serves it, so
    the whole loop runs and changes nothing an operator can see."""
    spec = build_plist_spec(
        python_executable="/venv/bin/python",
        policy_path="/instance/policy.yaml",
        feed_path="/instance/feed.json",
        home="/instance",
        out_path="/served/config.yaml",
    )

    arguments = list(spec.program_arguments)
    assert "--out" in arguments
    assert arguments[arguments.index("--out") + 1] == "/served/config.yaml"


def test_the_installed_plist_omits_out_when_none_is_given():
    """Absent, not empty: `run`'s own default must still apply."""
    spec = build_plist_spec(
        python_executable="/venv/bin/python",
        policy_path="/instance/policy.yaml",
        feed_path="/instance/feed.json",
    )

    assert "--out" not in spec.program_arguments


def test_the_installed_plist_can_keep_the_proxys_provider_modules_current():
    spec = build_plist_spec(
        python_executable="/venv/bin/python",
        policy_path="/instance/policy.yaml",
        feed_path="/instance/feed.json",
        provider_modules_source="/repo/providers",
        provider_modules_target="/served",
    )

    arguments = list(spec.program_arguments)
    assert arguments[arguments.index("--provider-modules-source") + 1] == "/repo/providers"
    assert arguments[arguments.index("--provider-modules-target") + 1] == "/served"


def test_the_tick_writes_its_output_somewhere_an_operator_can_read():
    """An unattended tick that refuses, warns, or names an unclassified
    failure must not say it to nobody."""
    from litellm_maintainer.schedule import default_log_paths

    out_path, err_path = default_log_paths("/instance")

    assert out_path == "/instance/state/tick.out.log"
    assert err_path == "/instance/state/tick.err.log"
    # `state/` is not a path the proxy's --reload watcher reads.
    assert "/state/" in out_path and out_path.endswith(".log")


def test_a_journal_triggered_run_does_not_refetch_the_feed(tmp_path, monkeypatch):
    """The Feed is republished once a day. A journal-triggered run
    reacts to a failure the proxy just served, which says nothing about
    the Feed, and it can fire minutes after the last run."""
    from litellm_maintainer.classify import Outcome
    from litellm_maintainer.journal import append_observation
    from litellm_maintainer.paths import journal_path
    from litellm_maintainer.prober import TransportResponse
    from litellm_maintainer.reduce import Observation

    raw = _run_policy_raw(require_proxy=False)
    raw["schedule"]["interval_minutes"] = 60
    raw["declared"] = [
        {"alias": "claude-declared-test", "litellm_params": {"model": "anthropic/claude-test"}},
        {"alias": "claude-declared-healthy", "litellm_params": {"model": "anthropic/claude-two"}},
    ]
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(raw))
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps({"schema_version": "test", "providers": [], "models": []}))
    args = _run_args(policy_path, feed_path, tmp_path)

    run_log = tmp_path / "state" / "runs.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    recent = datetime.now(timezone.utc) - timedelta(minutes=10)
    run_log.write_text(f"{recent.isoformat()} run: offered=2\n")

    append_observation(
        journal_path(tmp_path),
        Observation(
            offering_id="claude-declared-test",
            observed_at=datetime.now(timezone.utc),
            outcome=Outcome(bucket="needs_operator", reset_at=None, reason="quota_exhausted"),
        ),
    )

    fetched: list = []

    def spy_fetch_for_tick(args, *, policy, mapping, transport):
        fetched.append(True)
        return None

    monkeypatch.setattr(cli_module, "_fetch_for_tick", spy_fetch_for_tick)
    monkeypatch.setattr(
        cli_module, "probe_offerings", lambda targets, *, pacing, transport, now: {}
    )

    cli_module.cmd_run(
        args,
        proxy_checker=lambda: True,
        probe_transport=lambda target: None,
        notifier=lambda message: None,
        smoke_transport=lambda entry: TransportResponse(
            http_status=None, body=None, transport="not-called-in-this-test"
        ),
    )

    assert fetched == []


def test_the_installed_plist_names_the_credential_file_by_absolute_path():
    """launchd runs a job from '/', and `cli._default_env_path` looks for
    `.env.local` relative to the working directory. Without an absolute
    `--env` the tick resolves no credential and
    `validate_config_before_write` refuses every write."""
    spec = build_plist_spec(
        python_executable="/venv/bin/python",
        policy_path="/instance/policy.yaml",
        feed_path="/instance/feed.json",
        env_path="/repo/.env.local",
    )

    arguments = list(spec.program_arguments)
    assert arguments[arguments.index("--env") + 1] == "/repo/.env.local"


# --- A skipped tick must not count as a run --------------------------------


def test_a_skipped_tick_does_not_advance_the_last_run_time(tmp_path: Path):
    """The defect a 60-second tick exposes.

    Every skip line used to advance `last_run_at`, so `now -
    last_run_at` never reached `interval_minutes` and the pipeline never
    ran again. One skip wedged the tick permanently. It hid because no
    tick was installed to produce a second skip.
    """
    log_path = tmp_path / "runs.log"
    log_path.write_text(
        "2026-07-26T19:25:09+00:00 run: offered=69\n"
        "2026-07-27T13:36:30+00:00 skip: the interval has not elapsed\n"
        "2026-07-27T13:38:05+00:00 skip: the interval has not elapsed\n"
    )

    last_run_at = cli_module._read_last_run_at(log_path)

    assert last_run_at == datetime(2026, 7, 26, 19, 25, 9, tzinfo=timezone.utc)


def test_a_refused_tick_does_count_as_a_run(tmp_path: Path):
    """A refusal after the Prober ran must hold the interval. Otherwise
    the next tick probes again 60 seconds later, and again -- the tick
    storm the interval rule exists to prevent."""
    log_path = tmp_path / "runs.log"
    log_path.write_text(
        "2026-07-26T19:25:09+00:00 run: offered=69\n"
        "2026-07-27T13:33:08+00:00 refused: safety rail\n"
    )

    last_run_at = cli_module._read_last_run_at(log_path)

    assert last_run_at == datetime(2026, 7, 27, 13, 33, 8, tzinfo=timezone.utc)


def test_a_run_log_holding_only_skips_reads_as_no_previous_run(tmp_path: Path):
    log_path = tmp_path / "runs.log"
    log_path.write_text("2026-07-27T13:36:30+00:00 skip: the proxy is down\n")

    assert cli_module._read_last_run_at(log_path) is None


def test_a_gate_refusal_writes_a_skip_line_and_a_post_work_refusal_does_not(
    tmp_path: Path,
):
    """The marker is what the next tick reads. Both directions are
    wrong in their own way, so both are pinned here."""
    log_path = tmp_path / "runs.log"

    cli_module._append_tick_skip_line(
        log_path, now=NOW, reason="the interval has not elapsed", mapping={}
    )
    cli_module._append_tick_skip_line(
        log_path, now=NOW, reason="safety rail", mapping={}, did_work=True
    )

    lines = log_path.read_text().splitlines()
    assert lines[0].split(" ")[1] == "skip:"
    assert lines[1].split(" ")[1] == "refused:"


def test_a_journal_triggered_run_makes_no_proxy_traffic_of_its_own(
    tmp_path: Path, monkeypatch
):
    """The runaway loop, measured 2026-07-27: 7 full runs in 7 minutes.

    The smoke check calls the proxy, the proxy's failure callback records
    those calls in the Journal, an unprocessed entry makes the next tick
    due at once, and that run smoke-checks again. The maintainer was
    observing its own traffic and re-triggering on it. A
    journal-triggered run must generate no proxy traffic, so the chain
    stops after exactly one extra run.
    """
    from litellm_maintainer.classify import Outcome
    from litellm_maintainer.journal import append_observation
    from litellm_maintainer.paths import journal_path
    from litellm_maintainer.reduce import Observation

    raw = _run_policy_raw(require_proxy=False)
    raw["schedule"]["interval_minutes"] = 60
    raw["declared"] = [
        {"alias": "claude-declared-test", "litellm_params": {"model": "anthropic/claude-test"}},
        {"alias": "claude-declared-healthy", "litellm_params": {"model": "anthropic/claude-two"}},
    ]
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(raw))
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(json.dumps({"schema_version": "test", "providers": [], "models": []}))
    args = _run_args(policy_path, feed_path, tmp_path)

    run_log = tmp_path / "state" / "runs.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)
    recent = datetime.now(timezone.utc) - timedelta(minutes=10)
    run_log.write_text(f"{recent.isoformat()} run: offered=2\n")

    append_observation(
        journal_path(tmp_path),
        Observation(
            offering_id="claude-declared-test",
            observed_at=datetime.now(timezone.utc),
            outcome=Outcome(bucket="needs_operator", reset_at=None, reason="quota_exhausted"),
        ),
    )

    smoke_calls: list = []

    def spy_smoke_transport(entry):
        smoke_calls.append(entry)
        raise AssertionError("a journal-triggered run must not call the proxy")

    monkeypatch.setattr(
        cli_module, "probe_offerings", lambda targets, *, pacing, transport, now: {}
    )

    exit_code = cli_module.cmd_run(
        args,
        proxy_checker=lambda: True,
        probe_transport=lambda target: None,
        notifier=lambda message: None,
        smoke_transport=spy_smoke_transport,
    )

    assert exit_code == 0
    assert smoke_calls == []


# --- The headroom-refresh job: a second job, on its own label --------------


def test_the_headroom_job_has_a_distinct_label_from_the_tick():
    from litellm_maintainer.schedule import DEFAULT_LABEL, HEADROOM_LABEL

    assert HEADROOM_LABEL != DEFAULT_LABEL


def test_the_headroom_plist_ticks_at_policys_interval_in_seconds():
    from litellm_maintainer.schedule import build_headroom_plist_spec

    spec = build_headroom_plist_spec(
        python_executable="/venv/bin/python",
        policy_path="/instance/policy.yaml",
        interval_minutes=15,
    )
    assert spec.tick_seconds == 15 * 60

    spec_5 = build_headroom_plist_spec(
        python_executable="/venv/bin/python",
        policy_path="/instance/policy.yaml",
        interval_minutes=5,
    )
    assert spec_5.tick_seconds == 5 * 60


def test_the_headroom_plist_invokes_headroom_refresh_never_run():
    """The job-level guarantee behind decision 9: this plist can only
    ever invoke `headroom refresh`, whose own lock is Headroom State's
    lock (see `tests/test_headroom.py`,
    `test_holding_the_maintainer_lock_does_not_block_a_refresh`), never
    `run`, whose pipeline takes the maintainer lock and writes the
    Generated Config."""
    from litellm_maintainer.schedule import build_headroom_plist_spec

    spec = build_headroom_plist_spec(
        python_executable="/venv/bin/python",
        policy_path="/instance/policy.yaml",
        home="/instance",
        env_path="/repo/.env.local",
        interval_minutes=15,
    )

    arguments = list(spec.program_arguments)
    assert "headroom" in arguments
    assert arguments[arguments.index("headroom") + 1] == "refresh"
    assert "run" not in arguments
    assert arguments[arguments.index("--policy") + 1] == "/instance/policy.yaml"
    assert arguments[arguments.index("--home") + 1] == "/instance"
    assert arguments[arguments.index("--env") + 1] == "/repo/.env.local"


def test_the_headroom_job_installs_and_uninstalls_independently_of_the_tick(
    tmp_path: Path,
):
    from litellm_maintainer.schedule import (
        DEFAULT_LABEL,
        HEADROOM_LABEL,
        build_headroom_plist_spec,
    )

    target = tmp_path / "LaunchAgents"
    tick_path = install(target, _spec())
    headroom_spec = build_headroom_plist_spec(
        python_executable="/usr/bin/python3",
        policy_path="/instance/policy.yaml",
        interval_minutes=15,
    )
    headroom_path = install(target, headroom_spec)

    assert tick_path != headroom_path
    assert tick_path.exists() and headroom_path.exists()

    removed = uninstall(target, HEADROOM_LABEL)
    assert removed == headroom_path
    assert not headroom_path.exists()
    assert tick_path.exists(), "removing the headroom job must not touch the tick's"
    assert uninstall(target, DEFAULT_LABEL) == tick_path


def test_schedule_module_never_imports_or_calls_refresh_headroom():
    """The point of the whole ticket: the guarantee that the tick can
    never rewrite the Generated Config from a Headroom Reading must come
    from what `schedule.py` cannot reach, not from a rule a future edit
    has to remember. The module's own docstring names `refresh_headroom`
    in prose, to explain the rule this test enforces -- so check for an
    IMPORT or a CALL, never the bare word."""
    source = inspect.getsource(schedule_module)
    assert "refresh_headroom(" not in source
    assert "import refresh_headroom" not in source
    assert "import litellm_maintainer.headroom" not in source
    assert "from litellm_maintainer.headroom" not in source


def test_the_ticks_run_path_never_imports_or_calls_refresh_headroom():
    """Same guarantee, checked at `cli.cmd_run` itself: the scheduled
    tick's own pipeline must neither import nor invoke the headroom
    refresh, even indirectly through a module-level import."""
    cmd_run_source = inspect.getsource(cli_module.cmd_run)
    assert "refresh_headroom" not in cmd_run_source

    module_source = inspect.getsource(cli_module)
    top_level_source = module_source.split("\ndef cmd_headroom", 1)[0]
    assert "refresh_headroom" not in top_level_source, (
        "refresh_headroom must not be imported at module scope, where the "
        "tick's own imports would pull it in too"
    )


def test_headroom_install_writes_no_job_when_policy_declares_no_sources(tmp_path: Path):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(_policy_raw(60)))
    target_dir = tmp_path / "LaunchAgents"

    exit_code = main(
        ["headroom", "install", "--policy", str(policy_path), "--target-dir", str(target_dir)]
    )

    assert exit_code == 0
    assert list(target_dir.glob("*.plist")) == []


def test_headroom_install_names_an_existing_orphaned_job_plainly(tmp_path: Path, capsys):
    """Policy once mapped a source, a job was installed, and the operator
    then removed every `headroom.sources` entry. The next `install` must
    say the job is now orphaned, not leave it unmentioned."""
    from litellm_maintainer.schedule import HEADROOM_LABEL, plist_path

    raw = _policy_raw(60)
    raw["headroom"] = {"sources": {"pool:claude-subscription": "codexbar:claude/"}}
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(raw))
    target_dir = tmp_path / "LaunchAgents"

    main(["headroom", "install", "--policy", str(policy_path), "--target-dir", str(target_dir)])
    assert plist_path(target_dir, HEADROOM_LABEL).exists()

    raw["headroom"] = {"sources": {}}
    policy_path.write_text(yaml.safe_dump(raw))

    exit_code = main(
        ["headroom", "install", "--policy", str(policy_path), "--target-dir", str(target_dir)]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "already installed" in captured.out
    # `install` never removes a job silently: it names it and leaves the
    # decision to the operator.
    assert plist_path(target_dir, HEADROOM_LABEL).exists()


def test_headroom_install_writes_a_plist_at_policys_interval_and_uninstall_removes_it(
    tmp_path: Path,
):
    from litellm_maintainer.schedule import HEADROOM_LABEL, plist_path

    raw = _policy_raw(60)
    raw["headroom"] = {
        "interval_minutes": 5,
        "sources": {"pool:claude-subscription": "codexbar:claude/"},
    }
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(raw))
    target_dir = tmp_path / "LaunchAgents"

    install_exit = main(
        ["headroom", "install", "--policy", str(policy_path), "--target-dir", str(target_dir)]
    )
    assert install_exit == 0
    path = plist_path(target_dir, HEADROOM_LABEL)
    assert path.exists()

    import plistlib

    document = plistlib.loads(path.read_bytes())
    assert document["StartInterval"] == 5 * 60
    assert document["Label"] == HEADROOM_LABEL

    uninstall_exit = main(["headroom", "uninstall", "--target-dir", str(target_dir)])
    assert uninstall_exit == 0
    assert not path.exists()


def test_headroom_install_resolves_relative_policy_and_home_to_absolute_paths(
    tmp_path: Path, monkeypatch
):
    """Defect 5: launchd runs a job from '/'. A relative '--policy' or
    '--home' baked into the plist then resolves against the wrong
    directory, and the job fails on every tick into a log nobody reads.
    '--env' already got this treatment (`str(Path(args.env).resolve())`);
    '--policy' and '--home' were passed through verbatim."""
    from litellm_maintainer.schedule import HEADROOM_LABEL, plist_path

    raw = _policy_raw(60)
    raw["headroom"] = {"sources": {"pool:claude-subscription": "codexbar:claude/"}}
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(raw))
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    target_dir = tmp_path / "LaunchAgents"

    monkeypatch.chdir(tmp_path)
    exit_code = main(
        [
            "headroom",
            "install",
            "--policy",
            "policy.yaml",
            "--home",
            "home",
            "--target-dir",
            str(target_dir),
        ]
    )
    assert exit_code == 0

    import plistlib

    document = plistlib.loads(plist_path(target_dir, HEADROOM_LABEL).read_bytes())
    arguments = document["ProgramArguments"]
    policy_arg = arguments[arguments.index("--policy") + 1]
    home_arg = arguments[arguments.index("--home") + 1]

    assert Path(policy_arg).is_absolute()
    assert Path(home_arg).is_absolute()
    assert Path(policy_arg) == policy_path.resolve()
    assert Path(home_arg) == home_dir.resolve()


def test_headroom_uninstall_with_nothing_installed_does_not_fail(tmp_path: Path):
    target_dir = tmp_path / "LaunchAgents"
    exit_code = main(["headroom", "uninstall", "--target-dir", str(target_dir)])
    assert exit_code == 0


def test_installing_the_headroom_job_never_touches_the_ticks_own_plist(tmp_path: Path):
    raw = _policy_raw(60)
    raw["headroom"] = {"sources": {"pool:claude-subscription": "codexbar:claude/"}}
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(raw))
    target_dir = tmp_path / "LaunchAgents"

    main(
        [
            "install",
            "--policy",
            str(policy_path),
            "--feed",
            "/instance/feed.json",
            "--target-dir",
            str(target_dir),
        ]
    )
    main(["headroom", "install", "--policy", str(policy_path), "--target-dir", str(target_dir)])

    assert len(list(target_dir.glob("*.plist"))) == 2


def test_a_burst_of_failures_cannot_turn_every_tick_into_a_run():
    """One client retrying a rate-limited Alias produced 90 entries in
    four minutes, and every 60-second tick ran the pipeline. The floor
    bounds that without giving up prompt reaction."""
    from litellm_maintainer.schedule import JOURNAL_FLOOR

    just_ran = due(
        schedule=_schedule(interval_minutes=60, require_proxy=False),
        last_run_at=NOW - timedelta(seconds=90),
        proxy_up=True,
        health_age=FRESH,
        now=NOW,
        journal_pending=True,
    )
    assert just_ran.run is False
    assert "less than" in just_ran.reason

    past_floor = due(
        schedule=_schedule(interval_minutes=60, require_proxy=False),
        last_run_at=NOW - JOURNAL_FLOOR,
        proxy_up=True,
        health_age=FRESH,
        now=NOW,
        journal_pending=True,
    )
    assert past_floor.run is True
    # Still far inside the hour the interval alone would have imposed.
    assert JOURNAL_FLOOR < timedelta(minutes=60)


# --- The refresh job's own environment, measured 2026-07-30 --------------
#
# A launchd job inherits almost nothing, and the source this one runs is a
# third-party binary. Two failures, each different:
#
#   no PATH holding the binary -> every run fails loudly, state never moves
#   no USER                    -> codexbar returns `usage: null` for the
#                                 Claude provider, the run updates 5 of 6
#                                 and reports no error at all
#
# The second is the worse one: silent, partial, and indistinguishable from
# a provider the source does not know.


def test_the_headroom_job_carries_user_and_path_into_its_environment():
    spec = build_headroom_plist_spec(
        python_executable="/venv/bin/python",
        policy_path="/home/policy.yaml",
        interval_minutes=15,
        user="operator",
        path="/opt/homebrew/bin:/usr/bin",
    )

    assert dict(spec.environment) == {
        "USER": "operator",
        "PATH": "/opt/homebrew/bin:/usr/bin",
    }

    document = plistlib.loads(render_plist(spec))

    assert document["EnvironmentVariables"]["USER"] == "operator"
    assert document["EnvironmentVariables"]["PATH"] == "/opt/homebrew/bin:/usr/bin"


def test_a_job_with_no_environment_writes_no_environment_key():
    # The tick reads only files this package resolves by absolute path, so
    # it needs none. An empty dict in the plist would be noise.
    spec = build_headroom_plist_spec(
        python_executable="/venv/bin/python",
        policy_path="/home/policy.yaml",
        interval_minutes=15,
    )

    assert spec.environment == ()
    assert "EnvironmentVariables" not in plistlib.loads(render_plist(spec))


# --- Both jobs go through the resilient entry point ----------------------


def test_the_tick_plist_invokes_tick_entry_not_the_cli_directly():
    # `cli` imports most of the package at module level, so one syntax
    # error stops it before any code runs and the tick writes NOTHING.
    # `tick_entry` catches that and records it. Measured 2026-07-30.
    spec = build_plist_spec(
        python_executable="/venv/bin/python",
        policy_path="/home/policy.yaml",
        feed_path="/home/feed.json",
    )

    assert "litellm_maintainer.tick_entry" in spec.program_arguments
    assert "litellm_maintainer.cli" not in spec.program_arguments


def test_the_headroom_plist_invokes_tick_entry_too():
    spec = build_headroom_plist_spec(
        python_executable="/venv/bin/python",
        policy_path="/home/policy.yaml",
        interval_minutes=15,
    )

    assert "litellm_maintainer.tick_entry" in spec.program_arguments
    assert "litellm_maintainer.cli" not in spec.program_arguments
