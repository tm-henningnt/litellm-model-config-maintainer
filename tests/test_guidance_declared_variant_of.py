"""Ticket 02: a hand-declared Client-Facing Variant folds into one row.

Why this exists. Measured 2026-07-27: `claude-opus-5` and
`claude-opus-5[1m]` appeared as two separate unscored rows. That is one
model named twice, which is the repetition a row-per-Canonical-Model exists
to prevent (ADR 0005). Both are Declared Offerings, so the Generator's
derived pairs do not cover them.

The operator states the relationship instead. Nothing is inferred from a
name: the suffix is an operator setting, so a guess would break the moment
they changed it.

Assert external behaviour: which rows appear, what a Route carries, and
what `parse_policy` accepts or refuses.
"""

from __future__ import annotations

import pytest

from litellm_maintainer.policy import PolicyError
from tests.test_guidance_declared import DIRECT, _derive, _policy

PLAIN = {"alias": "claude-plain", "litellm_params": {"model": "anthropic/plain"}}
WIDE = {
    "alias": "claude-plain[1m]",
    "variant_of": "claude-plain",
    "litellm_params": {"model": "anthropic/plain"},
}


def _rows(answer):
    return {row.canonical_model_id for row in answer.rows}


def _route(answer, model_key):
    row = next(r for r in answer.rows if r.canonical_model_id == model_key)
    assert len(row.routes) == 1
    return row.routes[0]


# --- The pair becomes one row ------------------------------------------


def test_a_declared_pair_produces_one_row_not_two():
    answer = _derive(
        declared=[PLAIN, WIDE], admitted=("claude-plain", "claude-plain[1m]")
    )

    assert "claude-plain" in _rows(answer)
    assert "claude-plain[1m]" not in _rows(answer)


def test_the_row_keeps_the_plain_alias_as_its_route():
    """The Alias a caller dispatches to by default must not change."""
    answer = _derive(
        declared=[PLAIN, WIDE], admitted=("claude-plain", "claude-plain[1m]")
    )

    assert _route(answer, "claude-plain").alias == "claude-plain"


def test_the_variant_becomes_the_wide_alias_on_that_route():
    answer = _derive(
        declared=[PLAIN, WIDE], admitted=("claude-plain", "claude-plain[1m]")
    )

    assert _route(answer, "claude-plain").wide_alias == "claude-plain[1m]"


def test_a_declared_offering_stating_no_variant_of_still_gets_its_own_row():
    answer = _derive(declared=[DIRECT], admitted=("claude-direct",))

    assert "claude-direct" in _rows(answer)
    assert _route(answer, "claude-direct").wide_alias is None


def test_a_policy_with_no_variant_of_anywhere_produces_todays_rows():
    answer = _derive(declared=[PLAIN, DIRECT], admitted=("claude-plain", "claude-direct"))

    assert _rows(answer) >= {"claude-plain", "claude-direct"}


# --- A variant whose sibling this run did not admit --------------------


def test_a_variant_whose_named_alias_was_not_admitted_produces_no_row():
    """It grants nothing, so it must not appear as a model of its own."""
    answer = _derive(declared=[PLAIN, WIDE], admitted=("claude-plain[1m]",))

    assert "claude-plain[1m]" not in _rows(answer)


def test_a_variant_whose_named_alias_was_not_admitted_is_reported():
    answer = _derive(declared=[PLAIN, WIDE], admitted=("claude-plain[1m]",))

    assert "claude-plain[1m]" in answer.warnings[-1] or any(
        "claude-plain[1m]" in w for w in answer.warnings
    )


# --- Policy validation -------------------------------------------------


def test_a_declared_offering_may_state_the_alias_it_is_a_variant_of():
    policy = _policy([PLAIN, WIDE])

    variant = next(d for d in policy.declared if d.alias == "claude-plain[1m]")
    assert variant.variant_of == "claude-plain"


def test_the_key_is_optional():
    policy = _policy([PLAIN])

    assert policy.declared[0].variant_of is None


def test_a_variant_of_naming_no_declared_alias_is_refused():
    """A typo would otherwise produce the duplicate row this removes."""
    with pytest.raises(PolicyError, match="variant_of"):
        _policy([PLAIN, {**WIDE, "variant_of": "claude-typo"}])


def test_a_variant_of_that_is_not_a_string_is_refused():
    with pytest.raises(PolicyError, match="variant_of"):
        _policy([PLAIN, {**WIDE, "variant_of": 7}])


def test_a_variant_of_naming_itself_is_refused():
    with pytest.raises(PolicyError, match="variant_of"):
        _policy([{**PLAIN, "variant_of": "claude-plain"}])


def test_an_unknown_key_under_a_declared_offering_is_still_refused():
    with pytest.raises(PolicyError):
        _policy([{**PLAIN, "no_such_key": True}])
