"""Ticket 03: two Aliases sharing a model string must agree on their limits.

Why this exists. litellm holds one cost-map entry per
`litellm_params.model`. Two entries sharing that string therefore share
one Stated Limit, and the last one registered defines both. Measured on
2026-07-26 against the installed litellm: an entry carrying limits
replaced its sibling's correct exact-map figures, and an entry carrying no
limits at all inherited its sibling's.

This is live in the operator's real config. Both ChatGPT seats point at
one model string per model, and a Client-Facing Variant shares a model
string with its plain sibling.

Reported, never refused. The condition is litellm handling something
badly rather than an error in Policy, and a legitimate config reaches it.
Silent when the figures agree, because agreement is the normal case and a
warning there is noise an operator learns to ignore.

Assert external behaviour: what the run report carries, and that the run
still proceeds.
"""

from __future__ import annotations

from typing import Any

from litellm_maintainer.plan import plan
from tests.test_pricing import _feed_with, _policy


def _declared(alias: str, model: str, model_info: dict[str, Any] | None = None) -> dict:
    entry: dict[str, Any] = {"alias": alias, "litellm_params": {"model": model}}
    if model_info is not None:
        entry["model_info"] = model_info
    return entry


def _report(declared: list[dict]):
    result = plan(
        feed=_feed_with(),
        policy=_policy(declared=declared),
        health={},
        now=None,  # type: ignore[arg-type]
    )
    assert result.refusal is None, "a collision must never refuse the run"
    return result


SIZED = {"max_input_tokens": 1000000, "max_output_tokens": 128000}
SMALLER = {"max_input_tokens": 200000, "max_output_tokens": 64000}


# --- Agreement is silent ---


def test_two_aliases_sharing_a_model_string_and_agreeing_are_not_reported():
    """The normal case: a seat pair, and a variant with its plain sibling."""
    result = _report(
        [
            _declared("claude-seat1", "openai/claude-gpt", SIZED),
            _declared("claude-seat2", "openai/claude-gpt", SIZED),
        ]
    )

    assert result.report.limit_collisions == ()


def test_two_aliases_sharing_a_model_string_and_stating_nothing_are_not_reported():
    result = _report(
        [
            _declared("claude-seat1", "openai/claude-gpt"),
            _declared("claude-seat2", "openai/claude-gpt"),
        ]
    )

    assert result.report.limit_collisions == ()


def test_one_alias_alone_on_a_model_string_is_never_a_collision():
    result = _report([_declared("claude-only", "openai/claude-gpt", SIZED)])

    assert result.report.limit_collisions == ()


def test_aliases_on_different_model_strings_are_never_a_collision():
    result = _report(
        [
            _declared("claude-one", "openai/claude-gpt-a", SIZED),
            _declared("claude-two", "openai/claude-gpt-b", SMALLER),
        ]
    )

    assert result.report.limit_collisions == ()


# --- Disagreement is reported ---


def test_two_aliases_stating_different_limits_are_reported():
    result = _report(
        [
            _declared("claude-big", "openai/claude-gpt", SIZED),
            _declared("claude-small", "openai/claude-gpt", SMALLER),
        ]
    )

    collisions = result.report.limit_collisions
    assert len(collisions) == 1
    assert collisions[0].model == "openai/claude-gpt"
    assert [alias for alias, _ in collisions[0].stated_by_alias] == [
        "claude-big",
        "claude-small",
    ]


def test_an_alias_stating_nothing_beside_one_stating_limits_is_reported():
    """The silent one inherits, which is the surprise worth reporting."""
    result = _report(
        [
            _declared("claude-sized", "openai/claude-gpt", SIZED),
            _declared("claude-silent", "openai/claude-gpt"),
        ]
    )

    collisions = result.report.limit_collisions
    assert len(collisions) == 1
    assert dict(collisions[0].stated_by_alias)["claude-silent"] == {}


def test_a_collision_names_the_alias_whose_figures_litellm_keeps():
    result = _report(
        [
            _declared("claude-first", "openai/claude-gpt", SIZED),
            _declared("claude-last", "openai/claude-gpt", SMALLER),
        ]
    )

    assert result.report.limit_collisions[0].winner == "claude-last"


def test_a_collision_on_one_figure_alone_is_reported():
    result = _report(
        [
            _declared("claude-one", "openai/claude-gpt", {"max_input_tokens": 1000000}),
            _declared("claude-two", "openai/claude-gpt", {"max_input_tokens": 200000}),
        ]
    )

    assert len(result.report.limit_collisions) == 1


def test_a_cost_key_difference_alone_is_not_a_limit_collision():
    """This report is about Stated Limits. Cost has its own reporting."""
    result = _report(
        [
            _declared("claude-one", "openai/claude-gpt", {"input_cost_per_token": 1e-6}),
            _declared("claude-two", "openai/claude-gpt", {"input_cost_per_token": 9e-6}),
        ]
    )

    assert result.report.limit_collisions == ()


# --- What the operator reads ---


def test_the_message_names_both_aliases_the_model_and_the_winner():
    result = _report(
        [
            _declared("claude-big", "openai/claude-gpt", SIZED),
            _declared("claude-small", "openai/claude-gpt", SMALLER),
        ]
    )
    message = result.report.limit_collisions[0].message

    assert "openai/claude-gpt" in message
    assert "claude-big" in message
    assert "claude-small" in message
    assert "1000000" in message
    assert "200000" in message


def test_the_run_still_writes_every_colliding_entry():
    result = _report(
        [
            _declared("claude-big", "openai/claude-gpt", SIZED),
            _declared("claude-small", "openai/claude-gpt", SMALLER),
        ]
    )

    aliases = {e["model_name"] for e in result.config["model_list"]}
    assert {"claude-big", "claude-small"} <= aliases
