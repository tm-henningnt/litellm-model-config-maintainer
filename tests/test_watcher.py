"""Tests for `litellm_maintainer.watcher`.

Each test name states a rule an operator would recognise. No test
sleeps in real time: the clock and the sleep function are both fakes
the test drives by hand.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from litellm_maintainer.classify import Outcome
from litellm_maintainer.journal import append_observation
from litellm_maintainer.reduce import Observation
from litellm_maintainer.watcher import JournalWatcher

NOW = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)
OFFERING = "opencode-go:glm-5.2"
PASSTHROUGH_ALIAS = "claude-cline-glm-5.2"


class _FakeClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now


class _Recorder:
    def __init__(self) -> None:
        self.confirmed: list[str] = []
        self.run_count = 0

    def confirm(self, offering_id: str) -> None:
        self.confirmed.append(offering_id)

    def run_maintainer(self) -> None:
        self.run_count += 1


def _observation(
    offering_id: str, bucket: str, reason: str = "quota_exhausted", offset_seconds: int = 0
) -> Observation:
    return Observation(
        offering_id=offering_id,
        observed_at=NOW + timedelta(seconds=offset_seconds),
        outcome=Outcome(bucket=bucket, reset_at=None, reason=reason),
    )


def _watcher(tmp_path: Path, recorder: _Recorder, clock: _FakeClock) -> JournalWatcher:
    # Dispatch through the recorder on every call, rather than binding
    # `recorder.confirm` once, so a test can swap `recorder.confirm` or
    # `recorder.run_maintainer` after the watcher is built.
    return JournalWatcher(
        path=tmp_path / "observations.jsonl",
        confirm=lambda offering_id: recorder.confirm(offering_id),
        run_maintainer=lambda: recorder.run_maintainer(),
        clock=clock,
        sleep=lambda _seconds: None,
        interval_seconds=1.0,
    )


def test_a_new_journal_entry_triggers_a_run_within_the_polling_interval(tmp_path: Path):
    recorder = _Recorder()
    clock = _FakeClock(NOW)
    watcher = _watcher(tmp_path, recorder, clock)
    path = tmp_path / "observations.jsonl"

    # Nothing has been written yet: the first poll finds nothing new.
    result = watcher.poll_once()
    assert result.ran_maintainer is False
    assert recorder.run_count == 0

    ticks = 0

    def stop() -> bool:
        nonlocal ticks
        ticks += 1
        if ticks == 2:
            append_observation(path, _observation(OFFERING, "self_healing"))
        return ticks > 3

    watcher.run_forever(stop=stop)

    assert recorder.run_count == 1


def test_a_quota_failure_acts_with_no_confirming_probe(tmp_path: Path):
    recorder = _Recorder()
    clock = _FakeClock(NOW)
    watcher = _watcher(tmp_path, recorder, clock)
    path = tmp_path / "observations.jsonl"
    append_observation(
        path, _observation(OFFERING, "self_healing", reason="quota_exhausted")
    )

    result = watcher.poll_once()

    assert result.ran_maintainer is True
    assert result.confirmed == []
    assert recorder.confirmed == []
    assert recorder.run_count == 1


def test_an_ambiguous_failure_triggers_a_confirming_probe_before_anything_changes(
    tmp_path: Path,
):
    recorder = _Recorder()
    clock = _FakeClock(NOW)
    watcher = _watcher(tmp_path, recorder, clock)
    path = tmp_path / "observations.jsonl"
    append_observation(
        path, _observation(OFFERING, "inconclusive", reason="rate_limited")
    )

    order: list[str] = []
    recorder.confirm = lambda offering_id: order.append(f"confirm:{offering_id}")  # type: ignore[method-assign]
    recorder.run_maintainer = lambda: order.append("run")  # type: ignore[method-assign]

    result = watcher.poll_once()

    assert order == [f"confirm:{OFFERING}", "run"]
    assert result.confirmed == [OFFERING]
    assert result.ran_maintainer is True


def test_a_failure_on_a_passthrough_auth_offering_is_recorded_and_does_not_exclude_it(
    tmp_path: Path,
):
    # The watcher does not know, and must not need to know, which
    # Offerings are Passthrough Auth. `reduce` alone applies that
    # exemption (CONTEXT.md, "Passthrough Auth"; reduce.py). This test
    # shows the watcher treats such a failure exactly like any other
    # decisive one: it records it and triggers a run, with no
    # Passthrough Auth argument anywhere in its own seam.
    recorder = _Recorder()
    clock = _FakeClock(NOW)
    watcher = _watcher(tmp_path, recorder, clock)
    path = tmp_path / "observations.jsonl"
    append_observation(
        path,
        _observation(PASSTHROUGH_ALIAS, "self_healing", reason="quota_exhausted"),
    )

    result = watcher.poll_once()

    assert result.ran_maintainer is True
    assert recorder.confirmed == []
    assert recorder.run_count == 1


def test_a_second_poll_with_no_new_entry_does_nothing(tmp_path: Path):
    recorder = _Recorder()
    clock = _FakeClock(NOW)
    watcher = _watcher(tmp_path, recorder, clock)
    path = tmp_path / "observations.jsonl"
    append_observation(path, _observation(OFFERING, "self_healing"))

    first = watcher.poll_once()
    second = watcher.poll_once()

    assert first.ran_maintainer is True
    assert second.ran_maintainer is False
    assert recorder.run_count == 1


def test_a_journal_rotation_does_not_replay_already_processed_entries(tmp_path: Path):
    recorder = _Recorder()
    clock = _FakeClock(NOW)
    watcher = _watcher(tmp_path, recorder, clock)
    path = tmp_path / "observations.jsonl"
    append_observation(path, _observation(OFFERING, "self_healing"))

    watcher.poll_once()
    assert recorder.run_count == 1

    # The maintainer rotated the Journal down to nothing, then a fresh
    # failure arrived a minute later.
    path.write_text("")
    append_observation(path, _observation(OFFERING, "self_healing", offset_seconds=60))

    result = watcher.poll_once()

    assert result.new_observations == 1
    assert recorder.run_count == 2


# --- Defect 3: the `watch` subcommand is registered -------------------------


def test_the_watch_subcommand_is_registered_and_reachable():
    """Defect 3: `build_parser()` had no `watch` entry, so
    `watcher.cmd_watch` was unreachable from the CLI."""
    from litellm_maintainer.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(
        ["watch", "--policy", "/instance/policy.yaml", "--feed", "/instance/feed.json"]
    )

    assert args.command == "watch"
    from litellm_maintainer.watcher import cmd_watch

    assert args.func is cmd_watch


# --- Defect 4: the confirming Probe is wired in for real --------------------


def test_build_confirm_probe_runs_a_real_probe_and_folds_it_into_health_state(
    tmp_path: Path, monkeypatch
):
    """Defect 4: `cmd_watch`'s `confirm` used to be a stub that only
    printed a message. Story 27 / ticket 15's checkbox requires an
    ambiguous failure to get a confirming Probe before anything
    changes. `cli.build_confirm_probe` is the real wiring: it must run
    one Probe and fold the result into Health State."""
    import litellm_maintainer.cli as cli_module
    from litellm_maintainer.feed import Feed, Offering
    from litellm_maintainer.health import read_health
    from litellm_maintainer.paths import health_path
    from litellm_maintainer.policy import Naming, Policy, Quality, Safety, Schedule
    from litellm_maintainer.prober import TransportResponse

    offering = Offering(
        id="acme:some-model",
        provider_id="acme",
        provider_model_id="some-model",
        capabilities=("tool_use",),
        endpoint={"model": "acme/some-model"},
        limits={},
        pricing={"kind": "free"},
        availability={"status": "available"},
        quality={"coding_score": 30},
        policy={"visibility": "listed"},
        raw={},
    )
    feed = Feed(schema_version="1", offerings=(offering,), providers={}, profiles=(), notices=(), raw={})
    policy = Policy(
        providers={},
        quality=Quality(minimum_coding_score=18),
        approved_candidates=(),
        naming=Naming(provider_labels={}, alias_overrides={}, alias_prefix="claude-"),
        withheld={},
        declared=(),
        pacing={},
        schedule=Schedule(
            enabled=False, interval_minutes=60, require_proxy=True, maximum_staleness_hours=24
        ),
        safety=Safety(maximum_removal_share=0.25, snapshot_count=5),
    )

    def fake_live_transport(target, *, credential=None):
        return TransportResponse(
            http_status=200, body={"choices": [{"message": {"content": "ok"}}]}, transport=None
        )

    # `build_confirm_probe` reads the module-global `live_transport` at
    # call time, so patching the name in `cli` (not `prober`) is what a
    # real caller's import binding sees too.
    monkeypatch.setattr(cli_module, "live_transport", fake_live_transport)

    from litellm_maintainer.paths import ensure_instance_dirs

    ensure_instance_dirs(tmp_path)
    confirm = cli_module.build_confirm_probe(policy=policy, feed=feed, home=tmp_path, mapping={})
    confirm("acme:some-model")

    written = read_health(health_path(tmp_path))
    record = written.offerings["acme:some-model"]
    assert record.last_success_at is not None
    assert record.excluded is False


def test_cmd_watchs_run_maintainer_uses_the_run_pipeline_not_generate(monkeypatch):
    """Defect 4's second half: `run_maintainer` called `cmd_generate`,
    which plans from Health State the observation never entered.
    `cmd_watch` must point it at `cmd_run` instead, so the chain is
    probe, reduce, plan (correction 9)."""
    import argparse

    import litellm_maintainer.cli as cli_module
    import litellm_maintainer.watcher as watcher_module

    calls: list[str] = []
    monkeypatch.setattr(cli_module, "cmd_run", lambda args: calls.append("run") or 0)

    def _boom_generate(args):
        calls.append("generate")
        return 0

    monkeypatch.setattr(cli_module, "cmd_generate", _boom_generate)
    monkeypatch.setattr(
        cli_module,
        "build_confirm_probe",
        lambda **kwargs: (lambda offering_id: None),
    )
    monkeypatch.setattr(
        watcher_module,
        "run_watch_command",
        lambda *, path, confirm, run_maintainer, interval_seconds: run_maintainer(),
    )

    args = argparse.Namespace(
        policy=None,
        feed=None,
        home=None,
        env=None,
        interval=5.0,
    )
    # A minimal real Feed/Policy on disk keeps `load_feed`/`load_policy`
    # honest without pulling in the operator's real files.
    import json
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        policy_path = Path(tmp) / "policy.yaml"
        feed_path = Path(tmp) / "feed.json"
        import yaml

        policy_path.write_text(
            yaml.safe_dump(
                {
                    "providers": {},
                    "quality": {"minimum_coding_score": 18},
                    "approved_candidates": [],
                    "naming": {
                        "alias_prefix": "claude-",
                        "provider_labels": {},
                        "alias_overrides": {},
                    },
                    "withheld": {},
                    "declared": [],
                    "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
                    "schedule": {
                        "enabled": True,
                        "interval_minutes": 60,
                        "require_proxy": False,
                        "maximum_staleness_hours": 24,
                    },
                    "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
                }
            )
        )
        feed_path.write_text(json.dumps({"schema_version": "test", "providers": [], "models": []}))
        args.policy = str(policy_path)
        args.feed = str(feed_path)

        watcher_module.cmd_watch(args)

    assert calls == ["run"]
