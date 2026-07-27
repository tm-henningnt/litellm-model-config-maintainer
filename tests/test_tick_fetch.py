"""The tick refreshes the Feed Document, and a failed fetch never stops it.

`run` is the unattended path. A network problem must not be able to
shrink the Generated Config, so `fetch` promotes nothing unless the
download parses and is plausible, and the tick then plans on whatever
document is on disk.

The failure also has to be visible later, not only on the terminal of a
run nobody watched. So a failed fetch appends a note to the run log, and
three quiet ticks on the same stale document leave three notes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import litellm_maintainer.cli as cli_module
from litellm_maintainer.cli import build_parser
from litellm_maintainer.report import run_log_line
from litellm_maintainer.plan import PlanReport


def _policy_raw(*, feed_block: dict | None) -> dict:
    raw = {
        "providers": {},
        "quality": {"minimum_coding_score": 18},
        "approved_candidates": [],
        "naming": {"alias_prefix": "claude-", "provider_labels": {}, "alias_overrides": {}},
        "withheld": {},
        "declared": [
            {"alias": "claude-one", "litellm_params": {"model": "anthropic/one"}}
        ],
        "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
        "schedule": {
            "enabled": True,
            "interval_minutes": 60,
            "require_proxy": False,
            "maximum_staleness_hours": 24,
        },
        "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
    }
    if feed_block is not None:
        raw["feed"] = feed_block
    return raw


def _feed_document(offering_count: int = 0) -> str:
    """A Declared-only Policy tolerates a Feed with no Offerings.

    `providers_configured` is False for such a Policy, so the
    plausibility line does not apply. That keeps these tests about the
    fetch, not about Selection.
    """
    return json.dumps(
        {
            "schema_version": "test",
            "feed": {"id": "test", "generated_at": "2999-01-01T00:00:00Z"},
            "providers": [],
            "models": [],
        }
    )


def _fixtures(tmp_path: Path, *, feed_block: dict | None) -> tuple[Path, Path]:
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(_policy_raw(feed_block=feed_block)))
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(_feed_document())
    return policy_path, feed_path


def _args(
    policy_path: Path,
    feed_path: Path,
    tmp_path: Path,
    *,
    dry_run: bool = False,
    env_path: Path | None = None,
):
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
    if env_path is not None:
        argv += ["--env", str(env_path)]
    if dry_run:
        argv.append("--dry-run")
    return build_parser().parse_args(argv)


def _transport(body: str):
    calls: list[str] = []
    tokens: list[str | None] = []

    def transport(url: str, token: str | None) -> str:
        calls.append(url)
        tokens.append(token)
        return body

    transport.calls = calls  # type: ignore[attr-defined]
    transport.tokens = tokens  # type: ignore[attr-defined]
    return transport


def _boom(*args, **kwargs):
    raise AssertionError("this seam must not be called")


def _answering_smoke_transport(*args, **kwargs):
    """A smoke check that answers.

    The post-write smoke check runs on every tick. It needs a real
    `TransportResponse`, not `None`, so give it one that reads as
    Answered and keeps these tests focused on the fetch.
    """
    from litellm_maintainer.prober import TransportResponse

    return TransportResponse(
        http_status=200,
        body={"choices": [{"delta": {"content": "ok"}}]},
    )


FEED_BLOCK = {"url": "https://feed.example/feed.json"}


@pytest.fixture(autouse=True)
def _no_probes(monkeypatch):
    """Probe nothing in this file.

    These tests pin the fetch step and the run log. `probe_offerings`
    returning an empty mapping keeps a Probe result out of the picture,
    the same way `test_schedule.py` does it.
    """
    monkeypatch.setattr(cli_module, "probe_offerings", lambda *a, **k: {})


def _run(args, *, fetch_transport):
    """Drive one tick with every seam stubbed except the fetch.

    The Prober and the smoke check are stubbed to do nothing. These tests
    are about the fetch step and the run log, so a probe result would only
    add noise.
    """
    return cli_module.cmd_run(
        args,
        proxy_checker=lambda: True,
        probe_transport=lambda target: None,
        notifier=lambda message: None,
        smoke_transport=_answering_smoke_transport,
        fetch_transport=fetch_transport,
    )


def test_the_tick_fetches_into_the_document_it_then_reads(tmp_path):
    policy_path, feed_path = _fixtures(tmp_path, feed_block=FEED_BLOCK)
    fresh = json.dumps(
        {
            "schema_version": "fetched",
            "feed": {"id": "test", "generated_at": "2999-01-01T00:00:00Z"},
            "providers": [],
            "models": [],
        }
    )
    transport = _transport(fresh)

    assert _run(_args(policy_path, feed_path, tmp_path), fetch_transport=transport) == 0

    assert transport.calls == ["https://feed.example/feed.json"]
    assert json.loads(feed_path.read_text())["schema_version"] == "fetched"


def test_a_failed_fetch_does_not_stop_the_tick(tmp_path):
    """The rule that matters: a network problem must not empty the config."""
    policy_path, feed_path = _fixtures(tmp_path, feed_block=FEED_BLOCK)
    before = feed_path.read_text()

    def failing(url: str, token: str | None) -> str:
        raise TimeoutError("connection timed out")

    exit_code = _run(_args(policy_path, feed_path, tmp_path), fetch_transport=failing)

    assert exit_code == 0
    assert feed_path.read_text() == before
    assert (tmp_path / "config.yaml").exists()


def test_a_failed_fetch_lands_in_the_run_log(tmp_path):
    policy_path, feed_path = _fixtures(tmp_path, feed_block=FEED_BLOCK)

    def failing(url: str, token: str | None) -> str:
        raise TimeoutError("connection timed out")

    _run(_args(policy_path, feed_path, tmp_path), fetch_transport=failing)

    log = (tmp_path / "state" / "runs.log").read_text()
    assert "fetch_failed" in log
    assert "connection timed out" in log


def test_a_successful_fetch_leaves_no_note(tmp_path):
    policy_path, feed_path = _fixtures(tmp_path, feed_block=FEED_BLOCK)

    _run(
        _args(policy_path, feed_path, tmp_path),
        fetch_transport=_transport(_feed_document()),
    )

    log = (tmp_path / "state" / "runs.log").read_text()
    assert "fetch_failed" not in log


def test_a_policy_with_no_feed_block_fetches_nothing(tmp_path):
    """An operator who refreshes the Feed themselves keeps today's behaviour."""
    policy_path, feed_path = _fixtures(tmp_path, feed_block=None)

    assert _run(_args(policy_path, feed_path, tmp_path), fetch_transport=_boom) == 0


def test_a_policy_with_no_feed_block_says_so_in_the_run_log(tmp_path):
    """A tick that cannot refresh the Feed must not look like one that did.

    Measured 2026-07-27: a Policy carried no `feed` block for a day of
    hourly ticks. Every tick planned on one hand-fetched document and no
    log line named the reason, so the loop read as healthy.
    """
    policy_path, feed_path = _fixtures(tmp_path, feed_block=None)

    _run(_args(policy_path, feed_path, tmp_path), fetch_transport=_boom)

    log = (tmp_path / "state" / "runs.log").read_text()
    assert "feed_not_configured" in log


def test_the_tick_reads_the_feed_credential_from_the_env_file(tmp_path, monkeypatch):
    """launchd exports nothing, so `os.environ` alone resolves no token.

    The tick is given `--env`. Without reading that file the fetch sends
    no Authorization header, a Feed behind a bearer token answers 401 on
    every tick, and the loop keeps the previous document forever.
    """
    monkeypatch.delenv("FEED_TOKEN", raising=False)
    env_path = tmp_path / ".env.local"
    env_path.write_text('FEED_TOKEN="token-from-the-file"\n')
    policy_path, feed_path = _fixtures(
        tmp_path,
        feed_block={"url": "https://feed.example/feed.json", "credential_env": "FEED_TOKEN"},
    )
    transport = _transport(_feed_document())

    _run(
        _args(policy_path, feed_path, tmp_path, env_path=env_path),
        fetch_transport=transport,
    )

    assert transport.tokens == ["token-from-the-file"]


def test_a_dry_run_fetches_nothing(tmp_path):
    policy_path, feed_path = _fixtures(tmp_path, feed_block=FEED_BLOCK)

    args = _args(policy_path, feed_path, tmp_path, dry_run=True)

    assert (
        cli_module.cmd_run(
            args,
            proxy_checker=_boom,
            probe_transport=_boom,
            notifier=_boom,
            fetch_transport=_boom,
        )
        == 0
    )


# --- The bootstrap path --------------------------------------------------


def test_fetch_takes_a_url_when_no_policy_exists_yet(tmp_path, capsys):
    """`init` needs a Feed Document, and the Feed URL normally lives in Policy.

    That is a cycle: the first fetch has no valid Policy to read. `--url`
    breaks it, so the first command an operator runs needs no Policy.
    """
    destination = tmp_path / "feed.json"

    exit_code = cli_module.cmd_fetch(
        build_parser().parse_args(
            [
                "fetch",
                "--url",
                "https://feed.example/feed.json",
                "--out",
                str(destination),
                "--home",
                str(tmp_path),
            ]
        ),
        transport=_transport(_feed_document()),
    )

    assert exit_code == 0
    assert destination.exists()


def test_fetch_without_a_policy_or_a_url_says_which_to_pass(tmp_path, capsys):
    exit_code = cli_module.cmd_fetch(
        build_parser().parse_args(["fetch", "--home", str(tmp_path)]),
        transport=_boom,
    )

    assert exit_code == 1
    assert "--url" in capsys.readouterr().err


# --- The run log line itself ---------------------------------------------


def test_a_run_log_line_without_a_note_is_unchanged():
    """Adding the note must not alter the line every previous run produced."""
    line = run_log_line(now=_fixed_now(), report=PlanReport(), notification_count=0)

    assert line.endswith("notifications=0")
    assert "note=" not in line


def test_a_run_log_line_carries_its_note():
    line = run_log_line(
        now=_fixed_now(),
        report=PlanReport(),
        notification_count=0,
        note="fetch_failed: timed out",
    )

    assert line.endswith("note=fetch_failed: timed out")


def _fixed_now():
    from datetime import datetime, timezone

    return datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)
