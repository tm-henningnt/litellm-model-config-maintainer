"""Fetch writes the Feed Document, and only when it survives every check.

Every test here is offline: `fetch_feed_document` takes its transport as
an argument, so no test imports an HTTP client. See `fetch.py`, "The
transport is injected".
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from litellm_maintainer.fetch import (
    FetchOutcome,
    age_hours,
    fetch_feed_document,
    resolve_credential,
    staleness_warning,
)
from litellm_maintainer.policy import FeedSource

NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)


def _document(offering_count: int, *, generated_at: str = "2026-07-26T11:00:00Z") -> str:
    return json.dumps(
        {
            "schema_version": "1.0.0",
            "feed": {"id": "test-feed", "generated_at": generated_at},
            "providers": [{"id": "example", "name": "Example"}],
            "models": [
                {
                    "id": f"example:model-{index}",
                    "provider": {"id": "example"},
                    "provider_model_id": f"model-{index}",
                    "capabilities": ["chat", "tool_use"],
                    "pricing": {"kind": "free"},
                    "availability": {"status": "available"},
                    "quality": {"coding_score": 50.0},
                    "policy": {},
                    "endpoint": {},
                }
                for index in range(offering_count)
            ],
        }
    )


def _transport(body: str):
    def transport(url: str, token: str | None) -> str:
        return body

    return transport


def _raising_transport(error: Exception):
    def transport(url: str, token: str | None) -> str:
        raise error

    return transport


SOURCE = FeedSource(url="https://feed.example/v1/feed.json")


def test_a_good_document_is_promoted(tmp_path):
    destination = tmp_path / "feed.json"

    outcome = fetch_feed_document(
        source=SOURCE,
        destination=destination,
        transport=_transport(_document(40)),
        providers_configured=True,
    )

    assert outcome.promoted is True
    assert outcome.offering_count == 40
    assert outcome.generated_at == "2026-07-26T11:00:00Z"
    assert json.loads(destination.read_text())["schema_version"] == "1.0.0"


def test_a_transport_failure_keeps_the_previous_document(tmp_path):
    """The point of the whole module: a network problem must not shrink Selection."""
    destination = tmp_path / "feed.json"
    destination.write_text(_document(40))

    outcome = fetch_feed_document(
        source=SOURCE,
        destination=destination,
        transport=_raising_transport(TimeoutError("connection timed out")),
        providers_configured=True,
    )

    assert outcome.promoted is False
    assert outcome.kept_previous is True
    assert "connection timed out" in outcome.message
    assert len(json.loads(destination.read_text())["models"]) == 40


def test_malformed_json_keeps_the_previous_document(tmp_path):
    destination = tmp_path / "feed.json"
    destination.write_text(_document(40))

    outcome = fetch_feed_document(
        source=SOURCE,
        destination=destination,
        transport=_transport('{"schema_version": "1.0.0", "models": ['),
        providers_configured=True,
    )

    assert outcome.promoted is False
    assert len(json.loads(destination.read_text())["models"]) == 40


def test_a_truncated_document_is_refused_not_promoted(tmp_path):
    """A short Feed is treated as a failed fetch, per safety.py's rule."""
    destination = tmp_path / "feed.json"
    destination.write_text(_document(40))

    outcome = fetch_feed_document(
        source=SOURCE,
        destination=destination,
        transport=_transport(_document(3)),
        providers_configured=True,
    )

    assert outcome.promoted is False
    assert outcome.offering_count == 3
    assert "plausible minimum" in outcome.message
    assert len(json.loads(destination.read_text())["models"]) == 40


def test_a_declared_only_policy_accepts_a_short_document(tmp_path):
    """`providers_configured=False` skips the plausibility line, as generate does."""
    destination = tmp_path / "feed.json"

    outcome = fetch_feed_document(
        source=SOURCE,
        destination=destination,
        transport=_transport(_document(2)),
        providers_configured=False,
    )

    assert outcome.promoted is True
    assert outcome.offering_count == 2


def test_no_partial_file_is_left_behind(tmp_path):
    destination = tmp_path / "feed.json"

    fetch_feed_document(
        source=SOURCE,
        destination=destination,
        transport=_transport(_document(40)),
        providers_configured=True,
    )

    assert list(p.name for p in tmp_path.iterdir()) == ["feed.json"]


def test_a_write_failure_reports_and_leaves_no_partial_file(tmp_path, monkeypatch):
    """`fetch_feed_document` promises never to raise. A full disk counts.

    The unattended tick calls it, so an OSError escaping here would kill
    a run over a document the tick could have done without.
    """
    destination = tmp_path / "feed.json"
    destination.write_text(_document(40))

    real_write_text = Path.write_text

    def refuse(self, *args, **kwargs):
        if self.name.endswith(".fetching"):
            real_write_text(self, "partial")
            raise OSError("no space left on device")
        return real_write_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", refuse)

    outcome = fetch_feed_document(
        source=SOURCE,
        destination=destination,
        transport=_transport(_document(40)),
        providers_configured=True,
    )

    monkeypatch.undo()

    assert outcome.promoted is False
    assert "no space left on device" in outcome.message
    assert not (tmp_path / "feed.json.fetching").exists()
    assert len(json.loads(destination.read_text())["models"]) == 40


def test_the_token_reaches_the_transport(tmp_path):
    seen: list[str | None] = []

    def transport(url: str, token: str | None) -> str:
        seen.append(token)
        return _document(40)

    fetch_feed_document(
        source=FeedSource(url=SOURCE.url, credential_env="FEED_TOKEN"),
        destination=tmp_path / "feed.json",
        transport=transport,
        providers_configured=True,
        token="secret-value",
    )

    assert seen == ["secret-value"]


def test_a_credential_comes_from_the_environment_never_from_policy():
    source = FeedSource(url=SOURCE.url, credential_env="FEED_TOKEN")

    assert resolve_credential(source, {"FEED_TOKEN": "abc"}) == "abc"
    assert resolve_credential(source, {}) is None
    assert resolve_credential(source, {"FEED_TOKEN": ""}) is None
    assert resolve_credential(FeedSource(url=SOURCE.url), {"FEED_TOKEN": "abc"}) is None


# --- Staleness -----------------------------------------------------------


def test_age_is_read_from_the_document_not_the_mtime():
    assert age_hours("2026-07-26T06:00:00Z", now=NOW) == pytest.approx(6.0)
    assert age_hours(None, now=NOW) is None
    assert age_hours("not-a-time", now=NOW) is None


def test_a_fresh_document_warns_about_nothing():
    assert (
        staleness_warning(
            generated_at="2026-07-26T06:00:00Z", maximum_age_hours=24.0, now=NOW
        )
        is None
    )


def test_a_stale_document_warns():
    stale = (NOW - timedelta(hours=30)).isoformat().replace("+00:00", "Z")

    warning = staleness_warning(generated_at=stale, maximum_age_hours=24.0, now=NOW)

    assert warning is not None
    assert "stale" in warning
    assert "run fetch" in warning


def test_an_unstated_build_time_warns_rather_than_assuming_fresh():
    warning = staleness_warning(generated_at=None, maximum_age_hours=24.0, now=NOW)

    assert warning is not None
    assert "unknown" in warning


def test_outcome_reports_whether_a_previous_document_survived():
    assert FetchOutcome(promoted=True, message="x").kept_previous is False
    assert FetchOutcome(promoted=False, message="x").kept_previous is True
