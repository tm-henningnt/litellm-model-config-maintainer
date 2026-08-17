"""Ticket 06: a Sub-allowance binds its own Routes, and only its own.

`binding_window` (ticket 02) reads only `primary`, `secondary` and
`tertiary`: `extra_windows` never raise the Allowance-level figure.
`guidance` (ticket 05) then published one figure per Route, always the
Allowance's own. Both were correct and still left a gap: the operator's
Claude subscription caps `claude-fable-5` inside its own weekly pool
(CONTEXT.md, "Sub-allowance"), and nothing read the window that measures
that cap.

Measured 2026-07-28: Claude's parent window read 82% while its
`claude-weekly-scoped-fable` extra window read 59%. The parent bound
either way, which hid the case that matters -- fable running dry with the
rest of the pool untouched. This file pins the fix: a Route binds on the
WORSE of its Allowance's own windows and its own Sub-allowance's window,
and containment runs one way, out no, in yes.

Ticket 10 retired `DeclaredOffering.sub_allowance_window`. This file's
join now runs through Policy's `headroom.sources.<id>.members`. A
`members` key names either a slot id `windows` declares, or a codexbar
`extraRateWindows` id, and both forms are legal. Fable's own window,
`claude-weekly-scoped-fable`, is a genuine extra window, not one of the
three named slots, so the tests below declare it the honest way: a
`members` entry naming the extra window id directly, with no `windows`
entry inventing a slot for it. `route_binding_window` resolves an extra
window first and a declared slot second, so the id reaches the same
figure either way.
"""

from __future__ import annotations

from datetime import datetime, timezone

from litellm_maintainer import guidance
from litellm_maintainer.codexbar import (
    CodexbarExtraWindow,
    CodexbarIdentity,
    CodexbarReading,
    CodexbarWindow,
)
from litellm_maintainer.feed import parse_feed
from litellm_maintainer.headroom import HeadroomRecord, HeadroomState
from litellm_maintainer.plan import PlanReport
from litellm_maintainer.policy import parse_policy

NOW = datetime(2026, 7, 28, 21, 0, tzinfo=timezone.utc)

FEED_RAW = {
    "schema_version": "1.0.0",
    "feed": {"generated_at": "2026-07-28T20:00:00Z"},
    "providers": [{"id": "openrouter", "name": "OpenRouter"}],
    "models": [
        {
            "id": "openrouter:vendor/scored",
            "provider": {"id": "openrouter"},
            "provider_model_id": "vendor/scored",
            "canonical_model": {"id": "vendor/scored"},
            "capabilities": ["chat", "coding", "tool_use"],
            "pricing": {"kind": "free", "metering": "tokens"},
            "availability": {"status": "available"},
            "quality": {"coding_score": 60.0},
            "policy": {"visibility": "listed"},
            "endpoint": {},
        }
    ],
}

# The extraRateWindows id codexbar publishes for fable's own window.
FABLE_WINDOW_ID = "claude-weekly-scoped-fable"

# Two Aliases sharing one pool. `FABLE` is capped inside the pool
# (`sub_allowance: True`); which window measures its own cap is now named
# in `headroom.sources` (`_policy`'s own `members` argument), never here.
FABLE = {
    "alias": "claude-fable-5",
    "entitlement_pool": "claude-subscription",
    "sub_allowance": True,
    "litellm_params": {"model": "anthropic/fable"},
}
OPUS = {
    "alias": "claude-opus-5",
    "entitlement_pool": "claude-subscription",
    "litellm_params": {"model": "anthropic/opus"},
}


def _policy(declared, *, members: dict[str, list[str]] | None = None):
    if members is None:
        members = {FABLE_WINDOW_ID: ["claude-fable-5"]}
    return parse_policy(
        {
            "providers": {"openrouter": {"mode": "all"}},
            "quality": {"minimum_coding_score": 10},
            "approved_candidates": [],
            "naming": {
                "alias_prefix": "claude-",
                "provider_labels": {"openrouter": "or"},
                "alias_overrides": {},
            },
            "withheld": {},
            "declared": declared,
            "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
            "schedule": {
                "enabled": True,
                "interval_minutes": 60,
                "require_proxy": False,
                "maximum_staleness_hours": 24,
            },
            "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
            # Matches `_headroom_state`'s own `source`, so `guidance.derive`
            # publishes the Reading it is handed: a Route's Headroom now
            # requires Policy to still declare the Allowance AND agree with
            # the stored record's `source` (defect 2), not merely that a
            # record exists under the right key.
            "headroom": {
                "sources": {
                    "pool:claude-subscription": {
                        "source": "codexbar:claude/",
                        # No 'windows' entry: fable's window is a codexbar
                        # extra window, not one of the three named slots.
                        # Naming a slot for it would leave the parent's own
                        # binding computation (ticket 10 measured this on
                        # 2026-07-28) and would state something false about
                        # Claude's real Reading.
                        "members": members,
                    }
                }
            },
        }
    )


def _claude_reading(
    *,
    parent_used_percent: float,
    fable_used_percent: float | None,
    window_id: str = FABLE_WINDOW_ID,
) -> CodexbarReading:
    extra_windows = ()
    if fable_used_percent is not None:
        extra_windows = (
            CodexbarExtraWindow(
                id=window_id,
                title="Fable only",
                window=CodexbarWindow(
                    used_percent=fable_used_percent, window_minutes=10080, resets_at=None
                ),
            ),
        )
    return CodexbarReading(
        provider="claude",
        identity=CodexbarIdentity(provider_id="claude", account_email=None),
        primary=CodexbarWindow(used_percent=4, window_minutes=1440, resets_at=None),
        secondary=CodexbarWindow(used_percent=parent_used_percent, window_minutes=10080, resets_at=None),
        tertiary=None,
        extra_windows=extra_windows,
        updated_at="2026-07-28T20:55:00Z",
        error=None,
    )


def _headroom_state(reading: CodexbarReading) -> HeadroomState:
    return HeadroomState(
        records={
            "pool:claude-subscription": HeadroomRecord(
                allowance_id="pool:claude-subscription",
                source="codexbar:claude/",
                reading=reading,
                read_at="2026-07-28T21:00:00Z",
            )
        }
    )


def _derive(*, declared, admitted, headroom_state, members=None):
    return guidance.derive(
        feed=parse_feed(FEED_RAW),
        policy=_policy(declared, members=members),
        health={},
        report=PlanReport(
            admitted=admitted,
            aliases={"openrouter:vendor/scored": "claude-or-scored"},
        ),
        now=NOW,
        headroom_state=headroom_state,
    )


def _route(answer, alias):
    for row in answer.rows:
        for route in row.routes:
            if route.alias == alias:
                return route
    raise AssertionError(f"no Route for {alias!r}")  # pragma: no cover


# --- The join: a Route binds on the worse of the two ------------------------


def test_a_drained_sub_allowance_binds_only_its_own_route():
    # Fable at 100%, parent at 70%: only claude-fable-5 must read 100%.
    reading = _claude_reading(parent_used_percent=70, fable_used_percent=100)
    answer = _derive(
        declared=[OPUS, FABLE],
        admitted=("claude-opus-5", "claude-fable-5"),
        headroom_state=_headroom_state(reading),
    )

    fable_route = _route(answer, "claude-fable-5")
    opus_route = _route(answer, "claude-opus-5")

    assert fable_route.headroom is not None
    assert fable_route.headroom.used_percent == 100
    assert opus_route.headroom is not None
    assert opus_route.headroom.used_percent == 70


def test_a_drained_parent_reaches_every_route_including_the_sub_allowance():
    # Containment runs one way: the parent at 100% reaches fable even
    # though fable's own window reads only 50%.
    reading = _claude_reading(parent_used_percent=100, fable_used_percent=50)
    answer = _derive(
        declared=[OPUS, FABLE],
        admitted=("claude-opus-5", "claude-fable-5"),
        headroom_state=_headroom_state(reading),
    )

    fable_route = _route(answer, "claude-fable-5")
    opus_route = _route(answer, "claude-opus-5")

    assert fable_route.headroom is not None
    assert fable_route.headroom.used_percent == 100
    assert opus_route.headroom is not None
    assert opus_route.headroom.used_percent == 100


def test_a_sub_allowance_no_member_lists_binds_on_its_parent_alone():
    reading = _claude_reading(parent_used_percent=70, fable_used_percent=100)
    answer = _derive(
        declared=[OPUS, FABLE],
        admitted=("claude-opus-5", "claude-fable-5"),
        headroom_state=_headroom_state(reading),
        members={},  # nobody assigned to the declared slot yet
    )

    fable_route = _route(answer, "claude-fable-5")

    # No member names 'claude-fable-5', so the Route reads the parent's
    # own figure -- the Reading's fable window at 100% must publish no
    # sub-figure here.
    assert fable_route.headroom is not None
    assert fable_route.headroom.used_percent == 70


def test_nothing_infers_a_window_id_from_an_alias():
    # "fable" appears nowhere in codexbar's private id in this Reading, so
    # a reader that pattern-matched the Alias against the id would find
    # nothing and silently fall back -- exactly what must not happen
    # silently. Naming the wrong id here must read as the parent's own
    # figure, never crash and never guess.
    reading = _claude_reading(
        parent_used_percent=70, fable_used_percent=100, window_id="opaque-vendor-string-9f2"
    )
    answer = _derive(
        declared=[OPUS, FABLE],  # members claims FABLE_WINDOW_ID, not the opaque id
        admitted=("claude-opus-5", "claude-fable-5"),
        headroom_state=_headroom_state(reading),
    )

    fable_route = _route(answer, "claude-fable-5")

    assert fable_route.headroom is not None
    assert fable_route.headroom.used_percent == 70


# --- A Client-Facing Variant shares its sibling's Sub-allowance -------------


def test_a_client_facing_variant_binds_exactly_as_the_alias_it_widens():
    variant = {
        "alias": "claude-fable-5[1m]",
        "variant_of": "claude-fable-5",
        "litellm_params": {"model": "anthropic/fable"},
    }
    reading = _claude_reading(parent_used_percent=70, fable_used_percent=100)
    answer = _derive(
        declared=[OPUS, FABLE, variant],
        admitted=("claude-opus-5", "claude-fable-5", "claude-fable-5[1m]"),
        headroom_state=_headroom_state(reading),
    )

    fable_route = _route(answer, "claude-fable-5")

    # The variant contributes no Route of its own (ADR 0007): it folds
    # onto its sibling's Route as `wide_alias`, so its own Sub-allowance
    # figure is exactly the sibling's -- nothing extra was declared for it,
    # and `members` never lists the variant's own Alias.
    assert fable_route.wide_alias == "claude-fable-5[1m]"
    assert fable_route.headroom is not None
    assert fable_route.headroom.used_percent == 100
