"""Threshold crossings on a Binding Window, for an event-driven pacing.

A Reading arrives every refresh, and a caller that must pace against it
otherwise polls and hopes. This module compares the Reading just taken
against the one before it and reports the thresholds a Binding Window
CROSSED.

Pure: no clock, no filesystem, no network, no notifier. It returns
messages; `notify_all` delivers them.

## Four rules, each from a way this goes wrong

**Fire on the crossing, never on the level.** A window resting at 81%
would notify on every refresh, and a stream of identical warnings is one
nobody reads. A crossing needs the previous Reading, which is why this
takes both.

**An absent Reading fires nothing.** Absence means unmeasured: an
unmapped Allowance, no reading yet, an expired window, or a source that
dropped one window on the call. None of those is 0%, and none is a
crossing.

**A window that governs nothing admitted fires nothing.** A window can
be permanently full while describing Offerings this operator may not
call. Paging someone about capacity that was never theirs teaches them
to ignore the channel.

Two mechanisms enforce that rule, and a caller needs whichever it can
supply. Passing `named_slots` to `binding_window` keeps a declared
Sub-allowance out of the figure entirely, which is how
`headroom.refresh_headroom` does it — it reads Policy and has no
admitted set to intersect with. Passing `admitted_members` here catches
the finer case, a PARENT window whose declared members are all withheld,
and needs a caller that has run Selection. Supply it where you can.

**Name the window.** The Binding Window is the worst of several, and
which one is worst can change between two Readings. A crossing that does
not say which window it crossed cannot be acted on.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The shares a crossing is reported at, low to high.
DEFAULT_THRESHOLDS: tuple[float, ...] = (80.0, 95.0)


@dataclass(frozen=True)
class Crossing:
    """One Binding Window crossing one threshold, in one direction."""

    allowance_id: str
    window: str
    threshold: float
    previous_percent: float
    used_percent: float
    updated_at: str | None
    rising: bool

    @property
    def message(self) -> str:
        direction = "passed" if self.rising else "fell back below"
        measured = self.updated_at or "an unstated time"
        return (
            f"Headroom: {self.allowance_id} ({self.window}) {direction} "
            f"{self.threshold:g}% — now {self.used_percent:g}%, was "
            f"{self.previous_percent:g}%. Measured at {measured}."
        )


def crossings(
    *,
    allowance_id: str,
    window: str,
    previous_percent: float | None,
    used_percent: float | None,
    updated_at: str | None = None,
    admitted_members: tuple[str, ...] | None = None,
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> tuple[Crossing, ...]:
    """Return the thresholds this window crossed between two Readings.

    `previous_percent` or `used_percent` of `None` returns nothing: an
    absent Reading is unmeasured, never 0%, and a first Reading has
    nothing to have crossed FROM.

    `admitted_members` is what Policy admits to this window. `None` means
    Policy declares no membership, which is the ordinary case and fires
    normally. An EMPTY tuple means Policy declares members and none is
    admitted, so the window governs nothing this operator may call and
    fires nothing. The two are different claims: a `len(...) == 0` test
    would silence the ordinary case as well.
    """
    if used_percent is None or previous_percent is None:
        return ()
    if admitted_members is not None and len(admitted_members) == 0:
        return ()

    found: list[Crossing] = []
    for threshold in sorted(thresholds):
        rose = previous_percent < threshold <= used_percent
        fell = used_percent < threshold <= previous_percent
        if rose or fell:
            found.append(
                Crossing(
                    allowance_id=allowance_id,
                    window=window,
                    threshold=threshold,
                    previous_percent=previous_percent,
                    used_percent=used_percent,
                    updated_at=updated_at,
                    rising=rose,
                )
            )
    return tuple(found)


def crossing_messages(found: tuple[Crossing, ...]) -> tuple[str, ...]:
    """Render crossings as notification messages, worst threshold first."""
    return tuple(
        crossing.message
        for crossing in sorted(found, key=lambda c: c.threshold, reverse=True)
    )
