"""Ticket 04: a Declared Offering states its own limits and capabilities.

The Feed does not publish a Declared Offering, so guidance had nothing to
say about one beyond its Alias. Every direct Claude route on the
operator's real Policy is Declared, so an agent reading a guidance answer
could not tell that the strongest models the proxy serves support
reasoning, and could not size a prompt for them.

The rule these tests pin: guidance reports a Declared Offering's Stated
Limit and its capabilities when the operator stated them, attributes them
to the operator rather than to the Feed, and states nothing when the
operator stated nothing.

Assert external behaviour: what a Route and a Row report, what
`parse_policy` accepts and refuses, and what does NOT reach the Generated
Config. Reuses the fixtures of `test_guidance_declared.py`.
"""

from __future__ import annotations

import pytest

from litellm_maintainer.plan import plan
from litellm_maintainer.policy import PolicyError
from tests.test_guidance_declared import DIRECT, NOW, _derive, _feed, _policy

WITH_LIMITS = {
    "alias": "claude-direct-sized",
    "litellm_params": {"model": "anthropic/direct-sized"},
    "model_info": {"max_input_tokens": 1000000, "max_output_tokens": 128000},
}

WITH_CAPABILITIES = {
    "alias": "claude-direct-capable",
    "litellm_params": {"model": "anthropic/direct-capable"},
    "capabilities": ["chat", "reasoning", "tool_use"],
}


# --- A Declared Route reports the Stated Limit the operator declared ---


def test_a_declared_route_reports_the_stated_limit_the_operator_declared():
    answer = _derive(declared=[WITH_LIMITS], admitted=("claude-direct-sized",))
    row = next(r for r in answer.rows if r.canonical_model_id == "claude-direct-sized")

    assert row.routes[0].context_tokens == 1000000
    assert row.routes[0].max_output_tokens == 128000


def test_a_declared_offering_stating_no_model_info_reports_neither_figure():
    """Absence must read as unknown, never as small."""
    answer = _derive(declared=[DIRECT], admitted=("claude-direct",))
    row = next(r for r in answer.rows if r.canonical_model_id == "claude-direct")

    assert row.routes[0].context_tokens is None
    assert row.routes[0].max_output_tokens is None


def test_a_declared_route_reports_one_figure_when_only_one_is_declared():
    declared = {
        "alias": "claude-direct-partial",
        "litellm_params": {"model": "anthropic/direct-partial"},
        "model_info": {"max_input_tokens": 200000},
    }
    answer = _derive(declared=[declared], admitted=("claude-direct-partial",))
    row = next(r for r in answer.rows if r.canonical_model_id == "claude-direct-partial")

    assert row.routes[0].context_tokens == 200000
    assert row.routes[0].max_output_tokens is None


def test_the_declared_stated_limit_reaches_the_json_shape():
    answer = _derive(declared=[WITH_LIMITS], admitted=("claude-direct-sized",))
    row = next(r for r in answer.rows if r.canonical_model_id == "claude-direct-sized")

    assert row.routes[0].as_dict()["context_tokens"] == 1000000
    assert row.routes[0].as_dict()["max_output_tokens"] == 128000


# --- A Declared Row reports operator-stated capabilities ---


def test_a_declared_row_reports_the_capabilities_the_operator_declared():
    answer = _derive(declared=[WITH_CAPABILITIES], admitted=("claude-direct-capable",))
    row = next(r for r in answer.rows if r.canonical_model_id == "claude-direct-capable")

    assert row.capabilities == ("chat", "reasoning", "tool_use")


def test_a_declared_row_attributes_its_capabilities_to_the_operator():
    """A caller must not mistake an operator statement for a Feed claim."""
    answer = _derive(declared=[WITH_CAPABILITIES], admitted=("claude-direct-capable",))
    row = next(r for r in answer.rows if r.canonical_model_id == "claude-direct-capable")

    assert "operator" in row.why
    assert "capabilit" in row.why


def test_a_declared_row_with_no_stated_capabilities_reports_none():
    answer = _derive(declared=[DIRECT], admitted=("claude-direct",))
    row = next(r for r in answer.rows if r.canonical_model_id == "claude-direct")

    assert row.capabilities == ()


def test_a_declared_row_with_no_stated_capabilities_claims_nothing_about_them():
    answer = _derive(declared=[DIRECT], admitted=("claude-direct",))
    row = next(r for r in answer.rows if r.canonical_model_id == "claude-direct")

    assert "capabilit" not in row.why


def test_a_discovered_row_still_takes_its_capabilities_from_the_feed():
    answer = _derive(declared=[], admitted=("openrouter:vendor/scored",))
    row = next(r for r in answer.rows if r.canonical_model_id == "vendor/scored")

    assert row.capabilities == ("chat", "coding", "tool_use")
    assert "operator" not in row.why


def test_the_declared_capabilities_reach_the_json_shape():
    answer = _derive(declared=[WITH_CAPABILITIES], admitted=("claude-direct-capable",))
    row = next(r for r in answer.rows if r.canonical_model_id == "claude-direct-capable")

    assert row.as_dict()["capabilities"] == ["chat", "reasoning", "tool_use"]


# --- Policy accepts the key, and still refuses an unknown one ---


def test_a_declared_offering_may_state_a_list_of_capabilities():
    policy = _policy([WITH_CAPABILITIES])

    assert policy.declared[0].capabilities == ("chat", "reasoning", "tool_use")


def test_a_declared_offering_that_states_no_capabilities_carries_none():
    policy = _policy([DIRECT])

    assert policy.declared[0].capabilities == ()


def test_capabilities_that_are_not_a_list_are_refused():
    with pytest.raises(PolicyError, match="capabilities"):
        _policy([{**DIRECT, "capabilities": "reasoning"}])


def test_a_capability_that_is_not_a_string_is_refused():
    with pytest.raises(PolicyError, match="capabilities"):
        _policy([{**DIRECT, "capabilities": ["chat", 7]}])


def test_an_unknown_key_under_a_declared_offering_is_still_refused():
    """Accepting one new key must not loosen the whole block."""
    with pytest.raises(PolicyError):
        _policy([{**DIRECT, "no_such_key": True}])


# --- Capabilities never reach the Generated Config ---


def _config_entry(declared: dict) -> dict:
    result = plan(feed=_feed(), policy=_policy([declared]), health={}, now=NOW)
    return next(
        entry
        for entry in result.config["model_list"]
        if entry["model_name"] == declared["alias"]
    )


def test_declared_capabilities_never_reach_the_generated_config():
    """litellm has no such key. Guidance is the only reader."""
    entry = _config_entry(WITH_CAPABILITIES)

    assert "capabilities" not in entry
    assert "capabilities" not in entry.get("model_info", {})
    assert "capabilities" not in entry["litellm_params"]


def test_a_declared_model_info_still_reaches_the_generated_config_verbatim():
    entry = _config_entry(WITH_LIMITS)

    assert entry["model_info"] == {
        "max_input_tokens": 1000000,
        "max_output_tokens": 128000,
    }
