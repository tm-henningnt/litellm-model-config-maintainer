"""Policy's `allowances` block: the Tier the operator states an Allowance
bills under.

Assert external behaviour: what `parse_policy` returns, and what makes it
raise `PolicyError` naming the offending key. See CONTEXT.md, "Tier", and
ticket 12.
"""

from __future__ import annotations

import pytest

from litellm_maintainer.policy import AllowanceInfo, PolicyError, parse_policy
from test_policy import _valid_policy_dict


def test_a_policy_naming_no_allowances_block_gets_an_empty_mapping():
    policy = parse_policy(_valid_policy_dict())

    assert policy.allowances == {}


def test_an_allowances_entry_states_a_tier():
    raw = _valid_policy_dict()
    raw["allowances"] = {
        "pool:claude-subscription": {"tier": "claude-max-5x"},
        "credential:EXAMPLE_CHATGPT_SEAT1_WORKER_KEY": {"tier": "chatgpt-plus"},
    }

    policy = parse_policy(raw)

    assert policy.allowances["pool:claude-subscription"] == AllowanceInfo(
        tier="claude-max-5x"
    )
    assert (
        policy.allowances["credential:EXAMPLE_CHATGPT_SEAT1_WORKER_KEY"].tier
        == "chatgpt-plus"
    )


def test_tier_is_published_verbatim_including_odd_characters():
    raw = _valid_policy_dict()
    odd_tier = "Claude Max 5x — \"beta\" (2026-07) 100%"
    raw["allowances"] = {"pool:claude-subscription": {"tier": odd_tier}}

    policy = parse_policy(raw)

    assert policy.allowances["pool:claude-subscription"].tier == odd_tier


def test_an_entry_may_state_no_tier_at_all():
    raw = _valid_policy_dict()
    raw["allowances"] = {"pool:claude-subscription": {}}

    policy = parse_policy(raw)

    assert policy.allowances["pool:claude-subscription"].tier is None


def test_a_malformed_allowance_id_key_is_a_policy_error():
    raw = _valid_policy_dict()
    raw["allowances"] = {"claude-subscription": {"tier": "claude-max-5x"}}

    with pytest.raises(PolicyError, match="well-formed Allowance id"):
        parse_policy(raw)


def test_an_unrecognised_key_in_an_entry_is_a_policy_error():
    raw = _valid_policy_dict()
    raw["allowances"] = {
        "pool:claude-subscription": {"tier": "claude-max-5x", "plan": "x"}
    }

    with pytest.raises(PolicyError, match="not a recognised key"):
        parse_policy(raw)


def test_tier_must_be_a_string():
    raw = _valid_policy_dict()
    raw["allowances"] = {"pool:claude-subscription": {"tier": 5}}

    with pytest.raises(PolicyError, match="tier"):
        parse_policy(raw)


def test_allowances_must_be_a_mapping():
    raw = _valid_policy_dict()
    raw["allowances"] = ["pool:claude-subscription"]

    with pytest.raises(PolicyError, match="allowances"):
        parse_policy(raw)


def test_an_entry_must_be_a_mapping():
    raw = _valid_policy_dict()
    raw["allowances"] = {"pool:claude-subscription": "claude-max-5x"}

    with pytest.raises(PolicyError, match="allowances.pool:claude-subscription"):
        parse_policy(raw)


# --- `scale_note`: a size where the vendor sells no level ----------------


def test_an_allowance_entry_may_state_a_scale_note():
    raw = _valid_policy_dict()
    raw["allowances"] = {
        "provider:fixed": {"scale_note": "10 USD/month, roughly 2x-5x its API cost"}
    }

    policy = parse_policy(raw)

    assert policy.allowances["provider:fixed"].tier is None
    assert "2x-5x" in policy.allowances["provider:fixed"].scale_note


def test_an_allowance_entry_may_state_both_a_tier_and_a_scale_note():
    raw = _valid_policy_dict()
    raw["allowances"] = {
        "provider:fixed": {"tier": "pro", "scale_note": "about 60 USD/month equivalent"}
    }

    policy = parse_policy(raw)

    assert policy.allowances["provider:fixed"].tier == "pro"
    assert policy.allowances["provider:fixed"].scale_note == "about 60 USD/month equivalent"


def test_a_scale_note_that_is_not_a_string_is_rejected():
    # Deliberately prose, never a number: a vendor states a range, and
    # inventing one figure it declined to give is the mistake `metered`
    # already made here.
    raw = _valid_policy_dict()
    raw["allowances"] = {"provider:fixed": {"scale_note": 60}}

    with pytest.raises(PolicyError, match="scale_note"):
        parse_policy(raw)


def test_an_entry_stating_neither_still_parses():
    raw = _valid_policy_dict()
    raw["allowances"] = {"provider:fixed": {}}

    policy = parse_policy(raw)

    assert policy.allowances["provider:fixed"].tier is None
    assert policy.allowances["provider:fixed"].scale_note is None


# --- `draw_notes`: how fast ONE Offering empties its Allowance -----------
#
# `scale_note` sizes the Allowance. This states the rate a single Offering
# draws on it, and the two are different questions: a pool can hold six
# Offerings that empty it at six rates, and for a subscription Offering the
# Feed publishes no rate at all.


def test_draw_notes_parse_keyed_by_health_key():
    raw = _valid_policy_dict()
    raw["draw_notes"] = {"prov:model-a": "10% of the normal rate during preview"}

    policy = parse_policy(raw)

    assert policy.draw_notes["prov:model-a"].startswith("10%")


def test_an_absent_draw_notes_block_states_none():
    policy = parse_policy(_valid_policy_dict())

    assert policy.draw_notes == {}


def test_a_draw_note_that_is_not_a_string_is_rejected():
    # Prose, never a number. A vendor states a multiple, a window and a
    # promotion; collapsing that to one figure states something it did not.
    raw = _valid_policy_dict()
    raw["draw_notes"] = {"prov:model-a": 0.1}

    with pytest.raises(PolicyError, match="draw_notes"):
        parse_policy(raw)
