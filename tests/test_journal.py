"""Tests for `litellm_maintainer.journal`.

Each test name states a rule an operator would recognise. The
Observation Journal is append-only JSON Lines (CONTEXT.md, "Observation
Journal"; ADR 0001).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from litellm_maintainer.classify import Outcome
from litellm_maintainer.journal import append_observation, read_observations, truncate_processed
from litellm_maintainer.reduce import Observation

NOW = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)
OFFERING = "opencode-go:glm-5.2"


def _observation(offset_seconds: int = 0, bucket: str = "self_healing") -> Observation:
    return Observation(
        offering_id=OFFERING,
        observed_at=NOW + timedelta(seconds=offset_seconds),
        outcome=Outcome(bucket=bucket, reset_at=None, reason="quota_exhausted"),
    )


def test_one_append_writes_exactly_one_line(tmp_path: Path):
    path = tmp_path / "state" / "observations.jsonl"
    append_observation(path, _observation())

    lines = path.read_text().splitlines()

    assert len(lines) == 1


def test_many_appends_write_many_lines_and_the_file_stays_valid(tmp_path: Path):
    path = tmp_path / "observations.jsonl"
    for i in range(25):
        append_observation(path, _observation(offset_seconds=i))

    lines = path.read_text().splitlines()
    read = read_observations(path)

    assert len(lines) == 25
    assert len(read.observations) == 25
    assert read.skipped == 0


def test_a_missing_journal_reads_as_an_empty_list(tmp_path: Path):
    path = tmp_path / "does-not-exist.jsonl"

    read = read_observations(path)

    assert read.observations == []
    assert read.skipped == 0


def test_a_malformed_line_is_skipped_and_counted_and_good_lines_still_read(tmp_path: Path):
    path = tmp_path / "observations.jsonl"
    append_observation(path, _observation(offset_seconds=0))
    with open(path, "a") as handle:
        handle.write("not valid json at all\n")
    append_observation(path, _observation(offset_seconds=1))

    read = read_observations(path)

    assert len(read.observations) == 2
    assert read.skipped == 1


def test_a_truncated_final_line_does_not_lose_the_earlier_records(tmp_path: Path):
    path = tmp_path / "observations.jsonl"
    append_observation(path, _observation(offset_seconds=0))
    append_observation(path, _observation(offset_seconds=1))
    with open(path, "a") as handle:
        handle.write('{"offering_id": "opencode-go:glm-5.2", "observed_at": "2026')

    read = read_observations(path)

    assert len(read.observations) == 2
    assert read.skipped == 1


def test_a_round_trip_preserves_the_alias_the_time_and_the_outcome(tmp_path: Path):
    path = tmp_path / "observations.jsonl"
    reset_at = NOW + timedelta(minutes=5)
    observation = Observation(
        offering_id=OFFERING,
        observed_at=NOW,
        outcome=Outcome(bucket="self_healing", reset_at=reset_at, reason="quota_exhausted"),
    )
    append_observation(path, observation)

    read = read_observations(path)

    assert len(read.observations) == 1
    got = read.observations[0]
    assert got.offering_id == OFFERING
    assert got.observed_at == NOW
    assert got.outcome.bucket == "self_healing"
    assert got.outcome.reason == "quota_exhausted"
    assert got.outcome.reset_at == reset_at


def test_truncate_processed_removes_only_entries_at_or_before_the_cutoff(tmp_path: Path):
    path = tmp_path / "observations.jsonl"
    append_observation(path, _observation(offset_seconds=-10))
    append_observation(path, _observation(offset_seconds=0))
    append_observation(path, _observation(offset_seconds=10))

    removed = truncate_processed(path, upto=NOW)

    read = read_observations(path)
    assert removed == 2
    assert len(read.observations) == 1
    assert read.observations[0].observed_at == NOW + timedelta(seconds=10)


def test_truncate_processed_on_a_missing_file_removes_nothing(tmp_path: Path):
    path = tmp_path / "observations.jsonl"

    removed = truncate_processed(path, upto=NOW)

    assert removed == 0
    assert not path.exists() or read_observations(path).observations == []


# --- The message field (ADR 0008) ------------------------------------------


def test_an_unclassified_message_survives_a_write_and_a_read(tmp_path: Path):
    """The message is what tells the operator which rule is missing."""
    path = tmp_path / "observations.jsonl"
    append_observation(
        path,
        Observation(
            offering_id="opencode-go:glm-5.2",
            observed_at=datetime(2026, 7, 27, 9, 0, tzinfo=timezone.utc),
            outcome=Outcome(
                bucket="inconclusive", reset_at=None, reason="unrecognized_failure"
            ),
            message="prompt is too long: 312000 tokens > 200000 maximum",
        ),
    )

    read = read_observations(path)

    assert read.skipped == 0
    assert read.observations[0].message == (
        "prompt is too long: 312000 tokens > 200000 maximum"
    )


def test_a_classified_failure_carries_no_message(tmp_path: Path):
    """Provider text is stored only where it teaches us something."""
    path = tmp_path / "observations.jsonl"
    append_observation(
        path,
        Observation(
            offering_id="opencode-go:glm-5.2",
            observed_at=datetime(2026, 7, 27, 9, 0, tzinfo=timezone.utc),
            outcome=Outcome(bucket="needs_operator", reset_at=None, reason="quota_exhausted"),
        ),
    )

    assert "message" not in path.read_text()
    assert read_observations(path).observations[0].message is None


def test_a_line_written_before_the_message_field_existed_still_reads(tmp_path: Path):
    """A required key here would silently discard every entry on disk.

    `read_observations` counts anything `_decode` raises on as a
    skipped line, so a missing `message` must read as absent.
    """
    path = tmp_path / "observations.jsonl"
    path.write_text(
        '{"offering_id":"opencode-go:glm-5.2",'
        '"observed_at":"2026-07-27T09:00:00+00:00",'
        '"outcome":{"bucket":"needs_operator","reason":"quota_exhausted","reset_at":null}}\n'
    )

    read = read_observations(path)

    assert read.skipped == 0
    assert len(read.observations) == 1
    assert read.observations[0].message is None


def test_rotation_removes_what_was_read_even_when_the_writer_clock_is_ahead(tmp_path: Path):
    """Rotation must not depend on a clock the writer controls.

    The failure callback relabelled a naive local time as UTC, putting
    every entry two hours in the future on a UTC+2 host.
    `truncate_processed` keeps entries newer than `now`, so it removed
    nothing: the Journal grew without bound, `journal_pending` stayed
    true, and the tick ran a full pipeline every 60 seconds.
    """
    from litellm_maintainer.journal import truncate_first

    path = tmp_path / "observations.jsonl"
    future = datetime(2026, 7, 27, 17, 30, tzinfo=timezone.utc)  # ahead of "now"
    for index in range(3):
        append_observation(
            path,
            Observation(
                offering_id=f"acme:model-{index}",
                observed_at=future,
                outcome=Outcome(bucket="self_healing", reset_at=None, reason="gateway_error"),
            ),
        )

    now = datetime(2026, 7, 27, 15, 30, tzinfo=timezone.utc)
    assert truncate_processed(path, now) == 0  # the old rule: a silent no-op

    assert truncate_first(path, 3) == 3
    assert read_observations(path).observations == []


def test_rotation_keeps_an_entry_appended_after_the_read(tmp_path: Path):
    """The proxy appends while the maintainer folds. Removing by
    position drops exactly what was read and keeps the rest."""
    from litellm_maintainer.journal import truncate_first

    path = tmp_path / "observations.jsonl"
    at = datetime(2026, 7, 27, 9, 0, tzinfo=timezone.utc)
    for index in range(2):
        append_observation(
            path,
            Observation(
                offering_id=f"acme:read-{index}",
                observed_at=at,
                outcome=Outcome(bucket="self_healing", reset_at=None, reason="gateway_error"),
            ),
        )
    read_count = len(read_observations(path).observations)
    append_observation(
        path,
        Observation(
            offering_id="acme:arrived-later",
            observed_at=at,
            outcome=Outcome(bucket="self_healing", reset_at=None, reason="gateway_error"),
        ),
    )

    truncate_first(path, read_count)

    survivors = [o.offering_id for o in read_observations(path).observations]
    assert survivors == ["acme:arrived-later"]
