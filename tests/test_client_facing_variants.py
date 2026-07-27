"""A Client-Facing Variant is generated for a wide-window Offering.

Why this exists. A calling client reads its own context budget out of the
Alias name. Measured 2026-07-27: `claude-opus-5[1m]` reports a 1M budget
in Claude Code while `claude-opus-5` reports 200k, from the same Offering
and the same request. See ADR 0007.

Doing that by hand for every wide Offering is the artifact this project
exists to remove, so the Generator derives one from the Feed's own
`limits.context_tokens`. An Offering the Feed does not size gets no
variant, unless the operator states that it qualifies.

Assert external behaviour: which Aliases reach the Generated Config, what
each variant entry carries, and what the run report names.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from litellm_maintainer.plan import plan
from litellm_maintainer.policy import PolicyError, parse_policy
from tests.test_pricing import _feed_with, _offering_raw, _policy_raw

WIDE = 1_000_000
NARROW = 262_144


def _offering(offering_id: str, context_tokens: int | None) -> dict[str, Any]:
    raw = _offering_raw(id=offering_id)
    raw["limits"] = {"context_tokens": context_tokens, "max_output_tokens": 65536}
    return raw


def _run(*offerings: dict[str, Any], variants: Any = None, **overrides: Any):
    raw = _policy_raw(**overrides)
    if variants is not None:
        raw["client_facing_variants"] = variants
    result = plan(
        feed=_feed_with(*[copy.deepcopy(o) for o in offerings]),
        policy=parse_policy(raw),
        health={},
        now=None,  # type: ignore[arg-type]
    )
    assert result.refusal is None, result.refusal
    return result


def _aliases(result) -> set[str]:
    return {e["model_name"] for e in result.config["model_list"]}


def _entry(result, alias: str) -> dict[str, Any]:
    return next(e for e in result.config["model_list"] if e["model_name"] == alias)


# --- Off by default ---


def test_no_variant_is_generated_when_policy_declares_none():
    """A Policy written before this feature behaves exactly as before."""
    result = _run(_offering("opencode-go:wide", WIDE))

    assert not [a for a in _aliases(result) if "[" in a]


# --- Feed-driven generation ---


def test_an_offering_the_feed_sizes_at_the_threshold_gets_a_variant():
    result = _run(_offering("opencode-go:wide", WIDE), variants={})
    aliases = _aliases(result)

    primary = next(a for a in aliases if "[" not in a)
    assert f"{primary}[1m]" in aliases


def test_an_offering_below_the_threshold_gets_no_variant():
    result = _run(_offering("opencode-go:narrow", NARROW), variants={})

    assert not [a for a in _aliases(result) if "[" in a]


def test_an_offering_the_feed_does_not_size_gets_no_variant():
    """Absence reads as unknown, and nothing is derived from a name."""
    result = _run(_offering("opencode-go:unsized", None), variants={})

    assert not [a for a in _aliases(result) if "[" in a]


def test_the_threshold_is_configurable():
    result = _run(
        _offering("opencode-go:narrow", NARROW),
        variants={"minimum_context_tokens": 200_000},
    )

    assert [a for a in _aliases(result) if a.endswith("[1m]")]


def test_the_suffix_is_configurable():
    result = _run(
        _offering("opencode-go:wide", WIDE),
        variants={"suffix": "[big]"},
    )

    assert [a for a in _aliases(result) if a.endswith("[big]")]


# --- The variant is the same Offering under another name ---


def test_a_variant_sends_the_same_request_as_its_sibling():
    """The provider never sees the suffix."""
    result = _run(_offering("opencode-go:wide", WIDE), variants={})
    primary = next(a for a in _aliases(result) if "[" not in a)

    assert _entry(result, f"{primary}[1m]")["litellm_params"] == _entry(
        result, primary
    )["litellm_params"]


def test_a_variant_carries_the_same_model_info_as_its_sibling():
    """They share a model string, so a difference would collide (ADR 0007)."""
    result = _run(_offering("opencode-go:wide", WIDE), variants={})
    primary = next(a for a in _aliases(result) if "[" not in a)

    assert _entry(result, f"{primary}[1m]").get("model_info") == _entry(
        result, primary
    ).get("model_info")


def test_a_variant_never_provokes_a_stated_limit_collision():
    result = _run(_offering("opencode-go:wide", WIDE), variants={})

    assert result.report.limit_collisions == ()


def test_a_variant_is_reported_beside_the_alias_it_widens():
    result = _run(_offering("opencode-go:wide", WIDE), variants={})
    primary = next(a for a in _aliases(result) if "[" not in a)

    assert result.report.client_facing_variants == ((primary, f"{primary}[1m]"),)


def test_a_variant_does_not_add_a_second_admitted_offering():
    """One Offering, two Aliases. The Offering is admitted once."""
    result = _run(_offering("opencode-go:wide", WIDE), variants={})

    assert result.report.admitted == ("opencode-go:wide",)


# --- The operator may state that an Offering qualifies ---


def test_the_operator_may_state_that_an_unsized_offering_qualifies():
    """The Feed states no window for a brand-new model. The operator can."""
    result = _run(
        _offering("opencode-go:unsized", None),
        variants={
            "operator_stated": {
                "opencode-go:unsized": "vendor docs state 1M; the Feed states none"
            }
        },
    )
    primary = next(a for a in _aliases(result) if "[" not in a)

    assert f"{primary}[1m]" in _aliases(result)


def test_an_operator_stated_offering_still_carries_no_invented_stated_limit():
    """A variant needs no Stated Limit, so stating one is not implied."""
    result = _run(
        _offering("opencode-go:unsized", None),
        variants={"operator_stated": {"opencode-go:unsized": "vendor docs state 1M"}},
    )
    primary = next(a for a in _aliases(result) if "[" not in a)
    info = _entry(result, f"{primary}[1m]").get("model_info") or {}

    assert "max_input_tokens" not in info


def test_an_operator_stated_offering_the_feed_does_not_publish_is_reported():
    """A stale line must be visible, or it goes stale in silence."""
    result = _run(
        _offering("opencode-go:wide", WIDE),
        variants={"operator_stated": {"opencode-go:long-gone": "why"}},
    )

    assert result.report.client_facing_variants_unknown == ("opencode-go:long-gone",)


def test_an_operator_stated_entry_needs_a_reason():
    with pytest.raises(PolicyError, match="operator_stated"):
        _run(
            _offering("opencode-go:wide", WIDE),
            variants={"operator_stated": {"opencode-go:wide": ""}},
        )


def test_an_unknown_key_in_the_block_is_refused():
    with pytest.raises(PolicyError, match="client_facing_variants"):
        _run(_offering("opencode-go:wide", WIDE), variants={"no_such_key": 1})


# --- Collisions ---


def test_a_variant_that_would_collide_with_a_declared_alias_refuses_the_run():
    """Silently dropping one would hide a name the operator chose."""
    offering = _offering("opencode-go:wide", WIDE)
    raw = _policy_raw(client_facing_variants={})
    result_first = _run(offering, variants={})
    variant_alias = next(a for a in _aliases(result_first) if a.endswith("[1m]"))

    raw["declared"] = [
        {"alias": variant_alias, "litellm_params": {"model": "anthropic/other"}}
    ]
    outcome = plan(
        feed=_feed_with(copy.deepcopy(offering)),
        policy=parse_policy(raw),
        health={},
        now=None,  # type: ignore[arg-type]
    )

    assert outcome.refusal is not None
    assert variant_alias in outcome.refusal


# --- An excluded Offering takes its variant with it ---


def test_an_excluded_offering_contributes_no_variant():
    from litellm_maintainer.reduce import OfferingHealth

    offering = _offering("opencode-go:wide", WIDE)
    result = plan(
        feed=_feed_with(copy.deepcopy(offering)),
        policy=parse_policy(_policy_raw(client_facing_variants={})),
        health={"opencode-go:wide": OfferingHealth(excluded=True, reason="gone")},
        now=None,  # type: ignore[arg-type]
    )

    assert not [a for a in _aliases(result) if "[" in a]
