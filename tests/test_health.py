"""Tests for `litellm_maintainer.health`, the Health State adapter.

Health State is written only by this path (ADR 0001). These tests cover
the read/write round trip, a missing file, a corrupt file, and the
atomic write.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from litellm_maintainer.health import read_health, write_health
from litellm_maintainer.reduce import HealthState, OfferingHealth

OFFERING = "opencode-go:glm-5.2"
WHEN = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)


def test_a_missing_health_state_file_reads_as_empty(tmp_path):
    path = tmp_path / "state" / "health.json"

    state = read_health(path)

    assert state == HealthState(offerings={})


def test_writing_then_reading_health_state_round_trips(tmp_path):
    path = tmp_path / "health.json"
    record = OfferingHealth(
        excluded=True,
        reason="self-healing failure: rate limit, gateway error or timeout",
        bucket="self_healing",
        reset_at=WHEN,
        last_success_at=WHEN,
        last_attempt_at=WHEN,
        failure_count=3,
    )
    state = HealthState(offerings={OFFERING: record})

    write_health(path, state)
    result = read_health(path)

    assert result == state


def test_a_corrupt_health_state_file_reads_as_empty_rather_than_crashing_the_run(tmp_path):
    path = tmp_path / "health.json"
    path.write_text("{not valid json")

    state = read_health(path)

    assert state == HealthState(offerings={})


def test_one_malformed_record_is_skipped_and_counted_the_rest_survive(tmp_path):
    path = tmp_path / "health.json"
    good = OfferingHealth(
        excluded=False,
        last_success_at=WHEN,
        last_attempt_at=WHEN,
        failure_count=0,
    )
    write_health(
        path,
        HealthState(
            offerings={
                "acme:model-a": good,
                "acme:model-b": good,
                "acme:model-c": good,
            }
        ),
    )
    document = json.loads(path.read_text())
    document["offerings"]["acme:model-b"] = {"excluded": False}  # no failure_count: KeyError
    path.write_text(json.dumps(document))

    state = read_health(path)

    assert set(state.offerings) == {"acme:model-a", "acme:model-c"}
    assert state.offerings["acme:model-a"] == good
    assert state.offerings["acme:model-c"] == good
    assert state.skipped_records == 1


def test_a_truncated_health_state_file_reads_as_empty(tmp_path):
    path = tmp_path / "health.json"
    write_health(path, HealthState(offerings={OFFERING: OfferingHealth()}))
    full = path.read_text()
    path.write_text(full[: len(full) // 2])

    state = read_health(path)

    assert state == HealthState(offerings={})


def test_writing_health_state_leaves_no_temporary_file_behind(tmp_path):
    path = tmp_path / "health.json"
    state = HealthState(offerings={OFFERING: OfferingHealth()})

    write_health(path, state)

    names = os.listdir(tmp_path)
    assert names == ["health.json"]
