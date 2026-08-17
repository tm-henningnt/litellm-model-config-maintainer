"""Ticket 09: a Policy source entry declares what each window slot measures.
Ticket 10: `members` says which Health Key draws on each declared slot.

`codexbar` fills `primary`, `secondary` and `tertiary` with a different
KIND per provider (`docs/gotchas.md`, "codexbar's three window slots do
not mean one thing"). Claude, ClinePass and OpenCode Go use nested time
windows, where the worst one binds the whole Allowance. Gemini uses one
quota per MODEL: its free plan states `Pro` 100% spent while `Flash` and
`Flash Lite` state 0%. Binding on the worst reports the whole Allowance
drained while two of its three models answer.

This file pins the fix at the Route level, mirroring
`test_sub_allowance_binding.py`'s own harness: a `headroom.sources` entry
may name what `primary`, `secondary` and `tertiary` measure, and a named
slot becomes a Sub-allowance that binds only the Route whose Health Key
`members` lists under it (ticket 10; ticket 09 attached the id through a
Declared Offering's `sub_allowance_window`, since retired).

The Gemini Reading below is built from the shape already measured in
`tests/fixtures/codexbar-sample.json`'s own `gemini` entry: `primary`
100% (an unset-sentinel `resetsAt`), `secondary` and `tertiary` both 0%,
every `windowMinutes` 1440.
"""

from __future__ import annotations

from datetime import datetime, timezone


from litellm_maintainer import guidance
from litellm_maintainer.codexbar import CodexbarIdentity, CodexbarReading, CodexbarWindow
from litellm_maintainer.feed import parse_feed
from litellm_maintainer.headroom import HeadroomRecord, HeadroomState
from litellm_maintainer.plan import PlanReport
from litellm_maintainer.policy import parse_policy

NOW = datetime(2026, 7, 28, 21, 0, tzinfo=timezone.utc)

FEED_RAW = {
    "schema_version": "1.0.0",
    "feed": {"generated_at": "2026-07-29T20:00:00Z"},
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

GEMINI_ALLOWANCE = "pool:example-gemini"
GEMINI_SOURCE = "codexbar:gemini/operator@example.com"

GEMINI_PRO = {
    "alias": "claude-gemini-pro",
    "entitlement_pool": "example-gemini",
    "sub_allowance": True,
    "litellm_params": {"model": "gemini/gemini-pro"},
}
GEMINI_FLASH = {
    "alias": "claude-gemini-flash",
    "entitlement_pool": "example-gemini",
    "sub_allowance": True,
    "litellm_params": {"model": "gemini/gemini-flash"},
}
GEMINI_FLASH_LITE = {
    "alias": "claude-gemini-flash-lite",
    "entitlement_pool": "example-gemini",
    "sub_allowance": True,
    "litellm_params": {"model": "gemini/gemini-flash-lite"},
}
# An ordinary sibling on the same pool naming no Sub-allowance at all. Its
# Alias contains "gemini-pro" on purpose: nothing may infer a slot id from
# an Alias, so this Route must never bind as though it named one.
GEMINI_ORDINARY = {
    "alias": "claude-gemini-pro-ordinary",
    "entitlement_pool": "example-gemini",
    "litellm_params": {"model": "gemini/gemini-ordinary"},
}

# Members matching the three Declared Offerings above, one per slot id.
_ALL_MEMBERS = {
    "gemini-pro": ["claude-gemini-pro"],
    "gemini-flash": ["claude-gemini-flash"],
    "gemini-flash-lite": ["claude-gemini-flash-lite"],
}


def _policy(declared, *, windows, members=None):
    source: dict = {"source": GEMINI_SOURCE}
    if windows is not None:
        source["windows"] = windows
    if members is not None:
        source["members"] = members
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
            "headroom": {"sources": {GEMINI_ALLOWANCE: source}},
        }
    )


def _gemini_reading(
    *, primary: float = 100, secondary: float = 0, tertiary: float = 0
) -> CodexbarReading:
    return CodexbarReading(
        provider="gemini",
        identity=CodexbarIdentity(provider_id="gemini", account_email="operator@example.com"),
        primary=CodexbarWindow(used_percent=primary, window_minutes=1440, resets_at=None),
        secondary=CodexbarWindow(used_percent=secondary, window_minutes=1440, resets_at=None),
        tertiary=CodexbarWindow(used_percent=tertiary, window_minutes=1440, resets_at=None),
        extra_windows=(),
        updated_at="2026-07-28T20:52:30Z",
        error=None,
    )


def _headroom_state(reading: CodexbarReading) -> HeadroomState:
    return HeadroomState(
        records={
            GEMINI_ALLOWANCE: HeadroomRecord(
                allowance_id=GEMINI_ALLOWANCE,
                source=GEMINI_SOURCE,
                reading=reading,
                read_at="2026-07-28T21:00:00Z",
            )
        }
    )


def _derive(*, declared, admitted, headroom_state, windows, members=None):
    return guidance.derive(
        feed=parse_feed(FEED_RAW),
        policy=_policy(declared, windows=windows, members=members),
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


_ALL_WINDOWS = {
    "primary": "gemini-pro",
    "secondary": "gemini-flash",
    "tertiary": "gemini-flash-lite",
}


# --- Gemini maps correctly, from one Reading --------------------------------


def test_gemini_pro_binds_100_percent_and_flash_binds_0_percent():
    answer = _derive(
        declared=[GEMINI_PRO, GEMINI_FLASH, GEMINI_FLASH_LITE],
        admitted=("claude-gemini-pro", "claude-gemini-flash", "claude-gemini-flash-lite"),
        headroom_state=_headroom_state(_gemini_reading()),
        windows=_ALL_WINDOWS,
        members=_ALL_MEMBERS,
    )

    pro = _route(answer, "claude-gemini-pro")
    flash = _route(answer, "claude-gemini-flash")
    flash_lite = _route(answer, "claude-gemini-flash-lite")

    assert pro.headroom is not None
    assert pro.headroom.used_percent == 100
    assert flash.headroom is not None
    assert flash.headroom.used_percent == 0
    assert flash_lite.headroom is not None
    assert flash_lite.headroom.used_percent == 0


def test_every_slot_named_leaves_the_allowance_with_no_headroom_of_its_own():
    # An ordinary Route naming no Sub-allowance at all reads `None`, not a
    # figure borrowed from whichever named slot happens to read worst:
    # nothing then caps the Allowance as a whole.
    answer = _derive(
        declared=[GEMINI_PRO, GEMINI_FLASH, GEMINI_FLASH_LITE, GEMINI_ORDINARY],
        admitted=(
            "claude-gemini-pro",
            "claude-gemini-flash",
            "claude-gemini-flash-lite",
            "claude-gemini-pro-ordinary",
        ),
        headroom_state=_headroom_state(_gemini_reading()),
        windows=_ALL_WINDOWS,
        members=_ALL_MEMBERS,
    )

    ordinary = _route(answer, "claude-gemini-pro-ordinary")

    assert ordinary.headroom is None


# --- A named slot leaves the parent, an unnamed one still binds ------------


def test_a_named_slot_does_not_drag_the_parent_figure():
    # Only 'primary' (100%) is named. 'secondary' (20%) and 'tertiary'
    # (0%) stay parent windows, so an ordinary Route reads the worst of
    # THOSE two -- 20% -- never the 100% a reader that ignored the named
    # slot would report.
    answer = _derive(
        declared=[GEMINI_PRO, GEMINI_ORDINARY],
        admitted=("claude-gemini-pro", "claude-gemini-pro-ordinary"),
        headroom_state=_headroom_state(_gemini_reading(secondary=20, tertiary=0)),
        windows={"primary": "gemini-pro"},
        members={"gemini-pro": ["claude-gemini-pro"]},
    )

    ordinary = _route(answer, "claude-gemini-pro-ordinary")

    assert ordinary.headroom is not None
    assert ordinary.headroom.used_percent == 20


def test_an_unnamed_slot_still_binds_every_route():
    # 'secondary' (20%) names no Sub-allowance, so it stays a parent
    # window and reaches every Route on the Allowance -- including the one
    # that DOES name its own Sub-allowance, since containment runs one
    # way (parent in, sub-allowance not out).
    answer = _derive(
        declared=[GEMINI_PRO],
        admitted=("claude-gemini-pro",),
        headroom_state=_headroom_state(_gemini_reading(primary=0, secondary=20, tertiary=0)),
        windows={"primary": "gemini-pro"},
        members={"gemini-pro": ["claude-gemini-pro"]},
    )

    pro = _route(answer, "claude-gemini-pro")

    # gemini-pro's own slot (primary) reads 0%, but the unnamed parent
    # window (secondary, 20%) still reaches it.
    assert pro.headroom is not None
    assert pro.headroom.used_percent == 20


def test_the_gemini_pro_route_binds_on_its_own_slot_when_it_is_worse():
    answer = _derive(
        declared=[GEMINI_PRO, GEMINI_ORDINARY],
        admitted=("claude-gemini-pro", "claude-gemini-pro-ordinary"),
        headroom_state=_headroom_state(_gemini_reading(primary=100, secondary=20, tertiary=0)),
        windows={"primary": "gemini-pro"},
        members={"gemini-pro": ["claude-gemini-pro"]},
    )

    pro = _route(answer, "claude-gemini-pro")
    ordinary = _route(answer, "claude-gemini-pro-ordinary")

    # gemini-pro's own slot (100%) is worse than the parent (20%), so it
    # binds on its own figure -- the ordinary sibling still reads the
    # parent alone.
    assert pro.headroom is not None
    assert pro.headroom.used_percent == 100
    assert ordinary.headroom is not None
    assert ordinary.headroom.used_percent == 20


# --- Nothing is inferred from an Alias --------------------------------------


def test_nothing_infers_a_slot_id_from_an_alias():
    # GEMINI_ORDINARY's Alias contains "gemini-pro", and 'primary' (the
    # slot named "gemini-pro") reads 100%. A reader that pattern-matched
    # the Alias would wrongly bind this Route on it. It must instead read
    # the parent alone (20%), because 'members' never lists its Alias.
    answer = _derive(
        declared=[GEMINI_ORDINARY],
        admitted=("claude-gemini-pro-ordinary",),
        headroom_state=_headroom_state(_gemini_reading(primary=100, secondary=20, tertiary=0)),
        windows={"primary": "gemini-pro"},
        members={"gemini-pro": ["claude-gemini-pro"]},
    )

    ordinary = _route(answer, "claude-gemini-pro-ordinary")

    assert ordinary.headroom is not None
    assert ordinary.headroom.used_percent == 20


def test_an_offering_no_member_list_names_reads_the_parent_alone():
    # GEMINI_PRO is admitted but 'members' claims nothing for this
    # Allowance at all -- the Sub-allowance is declared, nobody is
    # assigned. It must fall back to the parent's own figure rather than
    # crash or silently read as absent.
    answer = _derive(
        declared=[GEMINI_PRO],
        admitted=("claude-gemini-pro",),
        headroom_state=_headroom_state(_gemini_reading(primary=100, secondary=20, tertiary=0)),
        windows={"primary": "gemini-pro"},
        members=None,
    )

    pro = _route(answer, "claude-gemini-pro")

    assert pro.headroom is not None
    assert pro.headroom.used_percent == 20


# --- Policy: a member must name a slot id 'windows' already declares -------


def test_a_member_key_reaching_no_window_still_parses():
    """A key that reaches neither a declared slot nor an extra window is a
    `doctor` finding, never a parse failure: a vendor can retire a window
    id, and one release must not stop the Generator."""
    policy = _policy(
        [GEMINI_PRO],
        windows={"primary": "gemini-pro"},
        members={"gemini-pro-typo": ["claude-gemini-pro"]},
    )

    members = policy.headroom.source_members["pool:example-gemini"]
    assert members["gemini-pro-typo"] == ("claude-gemini-pro",)


def test_a_member_naming_no_known_health_key_still_parses():
    # A typo, or a model the Feed dropped: Policy still parses. `doctor`
    # reports the gap (ticket 10) -- never a parse failure, because a
    # Feed change must never stop config generation for every provider.
    policy = _policy(
        [GEMINI_PRO],
        windows={"primary": "gemini-pro"},
        members={"gemini-pro": ["claude-gemini-pro", "claude-a-model-nobody-declares"]},
    )

    assert policy.headroom.source_members[GEMINI_ALLOWANCE]["gemini-pro"] == (
        "claude-gemini-pro",
        "claude-a-model-nobody-declares",
    )


# --- ticket 10: a Feed provider's own Offerings can claim a slot -----------
#
# Ticket 09 could only attach a slot to a DECLARED Offering, through a
# field only a Declared Offering carries. A per-model provider the Feed
# itself publishes -- Gemini running `mode: all`, the measured case --
# got the safe half (its Allowance stopped reporting drained) and not the
# useful half (no Route reported a figure). `members` reaches a Discovered
# Offering too, keyed by its own Feed Offering id.

DISCOVERED_GEMINI_FEED = {
    "schema_version": "1.0.0",
    "feed": {"generated_at": "2026-07-29T20:00:00Z"},
    "providers": [{"id": "gemini", "name": "Gemini"}],
    "models": [
        {
            "id": "gemini:gemini-3-pro-preview",
            "provider": {"id": "gemini"},
            "provider_model_id": "gemini-3-pro-preview",
            "capabilities": ["chat", "coding", "tool_use"],
            "pricing": {"kind": "free", "metering": "tokens"},
            "availability": {"status": "available"},
            "quality": {"coding_score": 70.0},
            "policy": {"visibility": "listed"},
            "endpoint": {},
        },
        {
            "id": "gemini:gemini-3-flash-preview",
            "provider": {"id": "gemini"},
            "provider_model_id": "gemini-3-flash-preview",
            "capabilities": ["chat", "coding", "tool_use"],
            "pricing": {"kind": "free", "metering": "tokens"},
            "availability": {"status": "available"},
            "quality": {"coding_score": 55.0},
            "policy": {"visibility": "listed"},
            "endpoint": {},
        },
        {
            "id": "gemini:gemini-3.5-flash",
            "provider": {"id": "gemini"},
            "provider_model_id": "gemini-3.5-flash",
            "capabilities": ["chat", "coding", "tool_use"],
            "pricing": {"kind": "free", "metering": "tokens"},
            "availability": {"status": "available"},
            "quality": {"coding_score": 57.0},
            "policy": {"visibility": "listed"},
            "endpoint": {},
        },
        {
            # An admitted model nobody has assigned to a slot. Measured
            # 2026-07-29 against the operator's own 'gemini' provider:
            # 'gemma-4-26b-a4b-it' and 'gemma-4-31b-it' match none of Pro,
            # Flash or Flash Lite.
            "id": "gemini:gemma-4-26b-a4b-it",
            "provider": {"id": "gemini"},
            "provider_model_id": "gemma-4-26b-a4b-it",
            "capabilities": ["chat"],
            "pricing": {"kind": "free", "metering": "tokens"},
            "availability": {"status": "available"},
            "quality": {"coding_score": 30.0},
            "policy": {"visibility": "listed"},
            "endpoint": {},
        },
    ],
}

DISCOVERED_ALLOWANCE = "provider:gemini"


def _discovered_policy(*, windows, members):
    source: dict = {"source": GEMINI_SOURCE}
    if windows is not None:
        source["windows"] = windows
    if members is not None:
        source["members"] = members
    return parse_policy(
        {
            "providers": {"gemini": {"mode": "all"}},
            "quality": {"minimum_coding_score": 10},
            "approved_candidates": [],
            "naming": {
                "alias_prefix": "claude-",
                "provider_labels": {"gemini": "gemini"},
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
            "headroom": {"sources": {DISCOVERED_ALLOWANCE: source}},
        }
    )


def _discovered_derive(*, admitted, headroom_state, windows, members, client_facing_variants=()):
    return guidance.derive(
        feed=parse_feed(DISCOVERED_GEMINI_FEED),
        policy=_discovered_policy(windows=windows, members=members),
        health={},
        report=PlanReport(
            admitted=admitted, aliases={}, client_facing_variants=client_facing_variants
        ),
        now=NOW,
        headroom_state=headroom_state,
    )


def _discovered_headroom_state(reading: CodexbarReading) -> HeadroomState:
    return HeadroomState(
        records={
            DISCOVERED_ALLOWANCE: HeadroomRecord(
                allowance_id=DISCOVERED_ALLOWANCE,
                source=GEMINI_SOURCE,
                reading=reading,
                read_at="2026-07-28T21:00:00Z",
            )
        }
    )


def test_a_discovered_offerings_own_id_can_claim_a_slot():
    # Gemini reports per model from ONE Reading: a listed Flash Route
    # binds 0%, an unlisted Gemma Route reads null.
    answer = _discovered_derive(
        admitted=(
            "gemini:gemini-3-pro-preview",
            "gemini:gemini-3-flash-preview",
            "gemini:gemini-3.5-flash",
            "gemini:gemma-4-26b-a4b-it",
        ),
        headroom_state=_discovered_headroom_state(_gemini_reading()),
        windows=_ALL_WINDOWS,
        members={
            "gemini-pro": ["gemini:gemini-3-pro-preview"],
            "gemini-flash": [
                "gemini:gemini-3-flash-preview",
                "gemini:gemini-3.5-flash",
            ],
        },
    )

    pro = _route(answer, "claude-gemini-3-pro-preview")
    flash_preview = _route(answer, "claude-gemini-3-flash-preview")
    flash_stable = _route(answer, "claude-gemini-3.5-flash")
    gemma = _route(answer, "claude-gemini-gemma-4-26b-a4b-it")

    assert pro.headroom is not None
    assert pro.headroom.used_percent == 100
    assert flash_preview.headroom is not None
    assert flash_preview.headroom.used_percent == 0
    assert flash_stable.headroom is not None
    assert flash_stable.headroom.used_percent == 0
    # Not a member of any slot, and every slot is named: no parent window
    # is left to borrow, so this Route reads null.
    assert gemma.headroom is None


def test_a_client_facing_variant_of_a_discovered_offering_shares_its_headroom():
    # ADR 0007: a Variant reaches the same Offering with the same wire
    # request, so it shares its sibling's Health Key and contributes no
    # Route of its own. Listing the plain Alias's Offering id in `members`
    # must cover the variant with no extra entry -- verified here rather
    # than assumed.
    answer = _discovered_derive(
        admitted=("gemini:gemini-3-flash-preview",),
        headroom_state=_discovered_headroom_state(_gemini_reading()),
        windows=_ALL_WINDOWS,
        members={"gemini-flash": ["gemini:gemini-3-flash-preview"]},
        client_facing_variants=(
            ("claude-gemini-3-flash-preview", "claude-gemini-3-flash-preview[1m]"),
        ),
    )

    flash = _route(answer, "claude-gemini-3-flash-preview")

    assert flash.wide_alias == "claude-gemini-3-flash-preview[1m]"
    assert flash.headroom is not None
    assert flash.headroom.used_percent == 0
