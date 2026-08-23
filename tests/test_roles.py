"""Tests for Role Model Groups in the Generated Config.

A Role is a Model Group whose members are Aliases this tool already
offers, ordered by `guidance`'s ranking. These tests assert the two
properties that make that safe: a Role can only name an Alias the run
actually wrote, and a Role never states a credential of its own.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from litellm_maintainer.plan import plan
from litellm_maintainer.policy import PolicyError, parse_policy
from litellm_maintainer.roles import build_role_entries

from litellm_maintainer.feed import parse_feed
from test_guidance import _offering_raw, _policy_raw


def _feed_with(*offerings):
    """A Feed whose Providers state a credential hint.

    `test_guidance._feed_with` states none, which is fine there because
    `guidance` never translates. `plan` does, and an Offering whose
    provider names no credential variable is dropped as
    `untranslatable_offering`, so these tests would silently plan an
    empty config.
    """
    provider_ids = sorted({o["provider"]["id"] for o in offerings})
    return parse_feed(
        {
            "schema_version": "test",
            "providers": [
                {
                    "id": provider_id,
                    "name": provider_id,
                    "default_base_url": f"https://{provider_id}.example/v1",
                    "authentication": {"credential_hint": "TEST_API_KEY"},
                }
                for provider_id in provider_ids
            ],
            "models": list(offerings),
        }
    )

NOW = datetime(2026, 8, 23, 12, 0, tzinfo=timezone.utc)


def _setup(*, roles: dict[str, Any], offerings, providers=None):
    """Plan a config from `offerings`, then build Roles over it."""
    feed = _feed_with(*offerings)
    provider_ids = sorted({o["provider"]["id"] for o in offerings})
    raw = _policy_raw(
        providers=providers
        or {pid: {"mode": "all"} for pid in provider_ids},
        roles=roles,
    )
    policy = parse_policy(raw)
    result = plan(feed=feed, policy=policy, health={}, now=NOW)
    role_result = build_role_entries(
        feed=feed,
        policy=policy,
        health={},
        report=result.report,
        now=NOW,
        entries=list(result.config.get("model_list") or []),
    )
    return policy, result, role_result


def _members(role_result, role_name: str) -> list[dict]:
    return [e for e in role_result.entries if e["model_name"] == role_name]


# ---------------------------------------------------------------------------
# The default: a Policy with no Role changes nothing.


def test_a_policy_with_no_roles_adds_no_entry():
    _, _, role_result = _setup(
        roles={},
        offerings=[_offering_raw(id="openrouter:a", provider_id="openrouter", canonical_model_id="v/a")],
    )
    assert role_result.entries == ()
    assert role_result.refusal is None


# ---------------------------------------------------------------------------
# The ladder.


def test_a_role_orders_its_members_best_first_on_the_axis():
    _, _, role_result = _setup(
        roles={"role-coder": {"axis": "coding"}},
        offerings=[
            _offering_raw(id="openrouter:low", provider_id="openrouter", canonical_model_id="v/low", coding_score=30.0),
            _offering_raw(id="openrouter:high", provider_id="openrouter", canonical_model_id="v/high", coding_score=90.0),
            _offering_raw(id="openrouter:mid", provider_id="openrouter", canonical_model_id="v/mid", coding_score=60.0),
        ],
    )
    members = _members(role_result, "role-coder")
    assert [m["litellm_params"]["order"] for m in members] == [1, 2, 3]
    assert [m["model_info"]["role_member_of"] for m in members] == [
        "claude-openrouter-high",
        "claude-openrouter-mid",
        "claude-openrouter-low",
    ]


def test_a_member_copies_the_aliass_params_and_states_no_credential_of_its_own():
    """A Role names an Alias. It must never carry a model string, base
    URL or credential the Alias does not already carry, or the two drift.
    """
    _, result, role_result = _setup(
        roles={"role-coder": {"axis": "coding"}},
        offerings=[_offering_raw(id="openrouter:a", provider_id="openrouter", canonical_model_id="v/a")],
    )
    alias_entry = next(
        e for e in result.config["model_list"] if e["model_name"] == "claude-openrouter-a"
    )
    member = _members(role_result, "role-coder")[0]

    expected = {**alias_entry["litellm_params"], "order": 1}
    assert member["litellm_params"] == expected


def test_limit_caps_the_ladder():
    _, _, role_result = _setup(
        roles={"role-coder": {"axis": "coding", "limit": 2}},
        offerings=[
            _offering_raw(id="openrouter:a", provider_id="openrouter", canonical_model_id="v/a", coding_score=90.0),
            _offering_raw(id="openrouter:b", provider_id="openrouter", canonical_model_id="v/b", coding_score=80.0),
            _offering_raw(id="openrouter:c", provider_id="openrouter", canonical_model_id="v/c", coding_score=70.0),
        ],
    )
    assert len(_members(role_result, "role-coder")) == 2


def test_a_role_that_nothing_qualifies_for_is_not_written_at_all():
    """An empty Model Group is a group litellm cannot resolve. Write no
    entry and say so in the ladder's notes instead.
    """
    _, _, role_result = _setup(
        roles={"role-vision": {"axis": "coding", "require_capabilities": ["vision"]}},
        offerings=[_offering_raw(id="openrouter:a", provider_id="openrouter", canonical_model_id="v/a")],
    )
    assert _members(role_result, "role-vision") == []
    ladder = next(x for x in role_result.ladders if x.role_name == "role-vision")
    assert any("no Route qualified" in note for note in ladder.notes)


def test_a_capability_requirement_reads_the_feed_not_a_hand_picked_list():
    """The scaffold this replaces hand-picked vision models at authoring
    time. Reading the Feed means a model that gains or loses vision joins
    or leaves the ladder with no edit.
    """
    seeing = _offering_raw(id="openrouter:seeing", provider_id="openrouter", canonical_model_id="v/seeing")
    seeing["capabilities"] = ["tool_use", "vision"]
    blind = _offering_raw(id="openrouter:blind", provider_id="openrouter", canonical_model_id="v/blind", coding_score=99.0)

    _, _, role_result = _setup(
        roles={"role-vision": {"axis": "coding", "require_capabilities": ["vision"]}},
        offerings=[seeing, blind],
    )
    members = _members(role_result, "role-vision")
    # `blind` scores higher and is still left out.
    assert [m["model_info"]["role_member_of"] for m in members] == ["claude-openrouter-seeing"]


# ---------------------------------------------------------------------------
# The refusal that keeps a Role from absorbing an Alias.


def test_a_role_named_after_an_alias_is_refused():
    """litellm holds one Model Group per `model_name`. A Role sharing a
    name with an Alias would absorb it, and calls meant for one model
    would reach several.
    """
    _, _, role_result = _setup(
        roles={"claude-openrouter-a": {"axis": "coding"}},
        offerings=[_offering_raw(id="openrouter:a", provider_id="openrouter", canonical_model_id="v/a")],
    )
    assert role_result.refusal is not None
    assert "collision" in role_result.refusal
    assert role_result.entries == ()


def test_a_role_never_names_an_alias_the_run_did_not_write():
    """Every member must exist in this run's `model_list`, or the Model
    Group points at nothing.
    """
    _, result, role_result = _setup(
        roles={"role-coder": {"axis": "coding"}},
        offerings=[
            _offering_raw(id="openrouter:a", provider_id="openrouter", canonical_model_id="v/a"),
            _offering_raw(id="openrouter:b", provider_id="openrouter", canonical_model_id="v/b"),
        ],
    )
    written = {e["model_name"] for e in result.config["model_list"]}
    for member in _members(role_result, "role-coder"):
        assert member["model_info"]["role_member_of"] in written


# ---------------------------------------------------------------------------
# Policy validation, against `guidance`'s own vocabularies.


def test_an_axis_the_feed_does_not_score_is_refused_on_load():
    with pytest.raises(PolicyError) as excinfo:
        parse_policy(_policy_raw(roles={"role-x": {"axis": "vibes"}}))
    assert "vibes" in str(excinfo.value)


def test_a_cost_basis_that_is_not_preferable_is_refused_on_load():
    with pytest.raises(PolicyError):
        parse_policy(
            _policy_raw(roles={"role-x": {"axis": "coding", "prefer": "metered"}})
        )


def test_an_unknown_role_key_is_refused_on_load():
    with pytest.raises(PolicyError):
        parse_policy(
            _policy_raw(roles={"role-x": {"axis": "coding", "limt": 3}})
        )


def test_a_role_block_is_optional():
    policy = parse_policy(_policy_raw())
    assert policy.roles == {}


# ---------------------------------------------------------------------------
# `cost_bases` filters. `prefer` does not.


def test_cost_bases_drops_a_route_billed_on_another_basis():
    """A Role meant to spend a free tier and nothing else must not bill
    on the day every free Route is drained. It fails instead.
    """
    free = _offering_raw(
        id="openrouter:free-one", provider_id="openrouter",
        canonical_model_id="v/free-one", pricing_kind="free", coding_score=40.0,
    )
    paid = _offering_raw(
        id="openrouter:paid-one", provider_id="openrouter",
        canonical_model_id="v/paid-one", pricing_kind="paid", coding_score=99.0,
    )

    _, _, role_result = _setup(
        roles={"role-free": {"axis": "coding", "cost_bases": ["free"]}},
        offerings=[free, paid],
    )
    members = _members(role_result, "role-free")
    # `paid-one` scores far higher and is still dropped.
    assert [m["model_info"]["role_member_of"] for m in members] == [
        "claude-openrouter-free-one"
    ]
    ladder = next(x for x in role_result.ladders if x.role_name == "role-free")
    assert any("not billed as free" in note for note in ladder.notes)


def test_an_unknown_cost_basis_is_refused_on_load():
    with pytest.raises(PolicyError) as excinfo:
        parse_policy(
            _policy_raw(roles={"role-x": {"axis": "coding", "cost_bases": ["cheap"]}})
        )
    assert "cheap" in str(excinfo.value)
