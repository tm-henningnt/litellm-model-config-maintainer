"""Tests for ticket 13's `notify.py`.

Each test name states a rule an operator would recognise. The subtle
rule — a recovery a recorded reset time already predicted produces no
notification, but an unpredicted one does — gets two tests, one per
side, per the ticket's warning.

A note on mutation-testing the rules: each test below was re-run with
the corresponding branch in `detect_events` deleted or inverted, to
confirm it actually fails when the rule it names is removed. See the
final report for which rules were checked this way and the result.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from litellm_maintainer.notify import (
    PreviousRunState,
    detect_events,
    notify_all,
    read_previous_run_state,
    write_previous_run_state,
)
from litellm_maintainer.reduce import OfferingHealth

NOW = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)


def _detect(
    *,
    previous=PreviousRunState(),
    admitted=frozenset(),
    candidates=frozenset(),
    previous_health=None,
    health=None,
    proxy_ok=True,
    now=NOW,
):
    return detect_events(
        previous=previous,
        admitted=admitted,
        candidates=candidates,
        previous_health=previous_health or {},
        health=health or {},
        proxy_ok=proxy_ok,
        now=now,
    )


# --- Fires ------------------------------------------------------------


def test_a_notification_fires_when_an_offering_is_added():
    events = _detect(
        previous=PreviousRunState(admitted=frozenset({"a"})),
        admitted=frozenset({"a", "b"}),
    )

    assert any("b" in e and "added" in e for e in events)


def test_a_notification_fires_when_an_offering_is_removed():
    events = _detect(
        previous=PreviousRunState(admitted=frozenset({"a", "b"})),
        admitted=frozenset({"a"}),
    )

    assert any("b" in e and "removed" in e for e in events)


def test_a_notification_fires_when_an_offering_becomes_needs_operator():
    previous_health = {"a": OfferingHealth(excluded=True, bucket="self_healing")}
    health = {"a": OfferingHealth(excluded=True, bucket="needs_operator")}

    events = _detect(previous_health=previous_health, health=health)

    assert any("a" in e and "needs the operator" in e for e in events)


def test_a_notification_fires_when_an_offering_becomes_gone():
    previous_health = {"a": OfferingHealth(excluded=True, bucket="self_healing")}
    health = {"a": OfferingHealth(excluded=True, bucket="gone")}

    events = _detect(previous_health=previous_health, health=health)

    assert any("a" in e and "gone" in e for e in events)


def test_a_notification_fires_when_a_new_candidate_appears():
    events = _detect(
        previous=PreviousRunState(candidates=frozenset({"x"})),
        candidates=frozenset({"x", "y"}),
    )

    assert any("y" in e and "Candidate" in e for e in events)


def test_a_notification_fires_when_the_proxy_check_fails():
    events = _detect(proxy_ok=False)

    assert any("Proxy check failed" == e for e in events)


# --- Does not fire ------------------------------------------------------


def test_no_notification_for_a_routine_run():
    previous_health = {"a": OfferingHealth(bucket="answered")}
    health = {"a": OfferingHealth(bucket="answered")}

    events = _detect(
        previous=PreviousRunState(admitted=frozenset({"a"}), candidates=frozenset({"c"})),
        admitted=frozenset({"a"}),
        candidates=frozenset({"c"}),
        previous_health=previous_health,
        health=health,
        proxy_ok=True,
    )

    assert events == ()


def test_no_notification_for_an_inconclusive_result():
    # `reduce` leaves an Inconclusive Offering's record completely
    # untouched: same bucket, same excluded flag, same reset_at as
    # before. From `detect_events`'s point of view this is
    # indistinguishable from "nothing happened" to this Offering, which
    # is the whole point of Inconclusive (CONTEXT.md).
    record = OfferingHealth(excluded=True, bucket="self_healing", reason="rate_limited")

    events = _detect(
        previous=PreviousRunState(admitted=frozenset()),
        admitted=frozenset(),
        previous_health={"a": record},
        health={"a": record},
    )

    assert events == ()


def test_no_notification_for_a_recovery_a_recorded_reset_time_already_predicted():
    reset_at = NOW - timedelta(minutes=1)  # already passed by the time of this run
    previous_health = {
        "a": OfferingHealth(excluded=True, bucket="self_healing", reason="quota_exhausted", reset_at=reset_at)
    }
    health = {"a": OfferingHealth(excluded=False, bucket=None, reason=None)}

    events = _detect(
        previous=PreviousRunState(admitted=frozenset()),
        admitted=frozenset({"a"}),
        previous_health=previous_health,
        health=health,
    )

    assert events == ()


def test_a_notification_fires_for_a_recovery_that_was_not_predicted():
    # No recorded reset time at all: the recovery could not have been
    # predicted, so it is news.
    previous_health = {
        "a": OfferingHealth(excluded=True, bucket="self_healing", reason="gateway_error", reset_at=None)
    }
    health = {"a": OfferingHealth(excluded=False, bucket=None, reason=None)}

    events = _detect(
        previous=PreviousRunState(admitted=frozenset()),
        admitted=frozenset({"a"}),
        previous_health=previous_health,
        health=health,
    )

    assert any("a" in e and "added" in e for e in events)


def test_a_notification_fires_for_a_recovery_before_its_own_predicted_reset_time():
    # The reset time names a future instant; the Offering answered
    # before then. That early recovery was not predicted either.
    reset_at = NOW + timedelta(hours=1)
    previous_health = {
        "a": OfferingHealth(excluded=True, bucket="self_healing", reason="quota_exhausted", reset_at=reset_at)
    }
    health = {"a": OfferingHealth(excluded=False, bucket=None, reason=None)}

    events = _detect(
        previous=PreviousRunState(admitted=frozenset()),
        admitted=frozenset({"a"}),
        previous_health=previous_health,
        health=health,
    )

    assert any("a" in e and "added" in e for e in events)


def test_a_recovery_at_exactly_the_predicted_reset_time_counts_as_predicted():
    reset_at = NOW
    previous_health = {
        "a": OfferingHealth(excluded=True, bucket="self_healing", reason="quota_exhausted", reset_at=reset_at)
    }
    health = {"a": OfferingHealth(excluded=False, bucket=None, reason=None)}

    events = _detect(
        previous=PreviousRunState(admitted=frozenset()),
        admitted=frozenset({"a"}),
        previous_health=previous_health,
        health=health,
        now=NOW,
    )

    assert events == ()


# --- Delivery: injectable, redacted, simple --------------------------------


def test_notify_all_calls_the_injected_notifier_once_per_event():
    delivered = []

    notify_all(("first", "second"), mapping={}, notifier=delivered.append)

    assert delivered == ["first", "second"]


def test_notify_all_redacts_a_credential_before_delivery():
    delivered = []
    mapping = {"sk-super-secret-value": "<REDACTED:KEY>"}

    notify_all(("Offering added: sk-super-secret-value",), mapping=mapping, notifier=delivered.append)

    assert delivered == ["Offering added: <REDACTED:KEY>"]


# --- Previous-run state: the small file this module owns -------------------


def test_a_missing_previous_run_state_file_reads_as_the_first_run_ever(tmp_path):
    state = read_previous_run_state(tmp_path / "last_report.json")

    assert state == PreviousRunState()


def test_writing_then_reading_previous_run_state_round_trips(tmp_path):
    path = tmp_path / "state" / "last_report.json"

    write_previous_run_state(path, admitted=frozenset({"a", "b"}), candidates=frozenset({"c"}))
    state = read_previous_run_state(path)

    assert state == PreviousRunState(admitted=frozenset({"a", "b"}), candidates=frozenset({"c"}))


def test_a_corrupt_previous_run_state_file_reads_as_the_first_run_ever(tmp_path):
    path = tmp_path / "last_report.json"
    path.write_text("{not valid json")

    state = read_previous_run_state(path)

    assert state == PreviousRunState()
