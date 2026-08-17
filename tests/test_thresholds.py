"""A Binding Window crossing 80% or 95% notifies, and nothing else does.

Each test here is a way this goes wrong in production rather than a way
the arithmetic goes wrong.
"""

from __future__ import annotations

from litellm_maintainer.thresholds import (
    DEFAULT_THRESHOLDS,
    crossing_messages,
    crossings,
)


def _cross(previous, used, **kwargs):
    return crossings(
        allowance_id=kwargs.pop("allowance_id", "provider:acme"),
        window=kwargs.pop("window", "weekly"),
        previous_percent=previous,
        used_percent=used,
        **kwargs,
    )


# --- fire on the crossing, not on the level -------------------------------


def test_rising_through_a_threshold_fires_once():
    found = _cross(70.0, 82.0)

    assert [c.threshold for c in found] == [80.0]
    assert found[0].rising is True


def test_a_window_resting_above_a_threshold_fires_nothing():
    """The reason this reads two Readings. A window sitting at 81% would
    otherwise notify on every refresh, and nobody reads that channel."""
    assert _cross(81.0, 82.0) == ()


def test_crossing_both_thresholds_at_once_fires_both():
    found = _cross(10.0, 99.0)

    assert [c.threshold for c in found] == [80.0, 95.0]


def test_falling_back_below_a_threshold_fires_once_in_the_other_direction():
    found = _cross(97.0, 90.0)

    assert [c.threshold for c in found] == [95.0]
    assert found[0].rising is False


def test_landing_exactly_on_a_threshold_counts_as_crossing_it():
    assert [c.threshold for c in _cross(79.0, 80.0)] == [80.0]


# --- an absent Reading is unmeasured, never 0% ----------------------------


def test_an_absent_current_reading_fires_nothing():
    assert _cross(70.0, None) == ()


def test_a_first_reading_fires_nothing_because_it_crossed_nothing():
    assert _cross(None, 99.0) == ()


# --- a window that governs nothing admitted -------------------------------


def test_a_declared_and_empty_membership_fires_nothing():
    """A window can read 100% while describing Offerings this operator
    may not call. Paging about capacity that was never theirs teaches
    them to ignore the channel."""
    assert _cross(10.0, 99.0, admitted_members=()) == ()


def test_undeclared_membership_still_fires_because_that_is_the_ordinary_case():
    """`None` and `()` are different claims. A `len(...) == 0` test would
    silence the ordinary case too."""
    found = _cross(10.0, 99.0, admitted_members=None)

    assert [c.threshold for c in found] == [80.0, 95.0]


def test_a_declared_and_populated_membership_fires():
    found = _cross(10.0, 82.0, admitted_members=("acme:model-a",))

    assert [c.threshold for c in found] == [80.0]


# --- the message names what a reader must act on --------------------------


def test_the_message_names_the_allowance_the_window_and_the_source_time():
    found = _cross(
        10.0,
        82.0,
        allowance_id="provider:qwencloud-token-plan",
        window="weekly",
        updated_at="2026-08-01T09:00:00Z",
    )

    message = found[0].message
    assert "provider:qwencloud-token-plan" in message
    assert "weekly" in message
    assert "2026-08-01T09:00:00Z" in message
    assert "80%" in message


def test_a_reading_with_no_source_timestamp_says_so_rather_than_inventing_one():
    message = _cross(10.0, 82.0, updated_at=None)[0].message

    assert "an unstated time" in message


def test_messages_report_the_worst_threshold_first():
    messages = crossing_messages(_cross(10.0, 99.0))

    assert "95%" in messages[0]
    assert "80%" in messages[1]


def test_the_default_thresholds_are_eighty_and_ninety_five():
    assert DEFAULT_THRESHOLDS == (80.0, 95.0)
