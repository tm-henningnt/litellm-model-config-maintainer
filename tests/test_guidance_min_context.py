"""Ticket 03: guidance answers "which models hold at least N tokens".

Why this exists. Picking a planner for a large input was the one question
this tool could not answer. Every Route already carries its Stated Limit,
so a caller filtered client-side — and every such caller re-implemented the
same rule, including the decision about an unstated window, which most
would get wrong by treating unknown as zero.

Two rules the tests pin, because both are judgement calls:

The filter selects ROUTES, then drops a row with none left. Filtering rows
and keeping their narrow Routes would let a caller fail over into a Route
too small for the work, which defeats the query.

A Route stating no window does not qualify. ADR 0006 says absence reads as
unknown rather than small, but a filter must decide, and handing a caller an
unmeasured Route as though it qualified is the more expensive error. So it
is excluded, counted separately, and named in the warning as unstated.

Assert external behaviour: which rows and Routes come back, and what the
warnings say.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from litellm_maintainer import guidance
from litellm_maintainer.feed import parse_feed
from litellm_maintainer.plan import PlanReport
from litellm_maintainer.policy import parse_policy

NOW = datetime(2026, 7, 27, 12, 0, tzinfo=timezone.utc)


def _offering(offering_id: str, context_tokens: int | None, score: float) -> dict:
    provider = offering_id.split(":", 1)[0]
    return {
        "id": offering_id,
        "provider": {"id": provider},
        "provider_model_id": offering_id.split(":", 1)[1],
        "canonical_model": {"id": offering_id.split(":", 1)[1]},
        "capabilities": ["tool_use"],
        "limits": {"context_tokens": context_tokens, "max_output_tokens": 65536},
        "pricing": {"kind": "free", "metering": "tokens"},
        "availability": {"status": "available"},
        "quality": {"coding_score": score},
        "policy": {"visibility": "listed"},
        "endpoint": {},
    }


def _answer(*offerings: dict, min_context: Any = None, **kwargs: Any):
    feed = parse_feed(
        {
            "schema_version": "test",
            "feed": {"generated_at": "2026-07-27T11:00:00Z"},
            "providers": [{"id": "openrouter", "name": "OpenRouter"}],
            "models": list(offerings),
        }
    )
    policy = parse_policy(
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
            "declared": [],
            "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
            "schedule": {
                "enabled": True,
                "interval_minutes": 60,
                "require_proxy": False,
                "maximum_staleness_hours": 24,
            },
            "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
        }
    )
    report = PlanReport(
        admitted=tuple(o["id"] for o in offerings),
        aliases={o["id"]: f"claude-or-{o['id'].split(':', 1)[1]}" for o in offerings},
    )
    return guidance.derive(
        feed=feed,
        policy=policy,
        health={},
        report=report,
        now=NOW,
        min_context=min_context,
        **kwargs,
    )


def _aliases(answer) -> set[str]:
    return {r.alias for row in answer.rows for r in row.routes}


WIDE = _offering("openrouter:wide", 1_000_000, 80.0)
EXACT = _offering("openrouter:exact", 800_000, 70.0)
NARROW = _offering("openrouter:narrow", 200_000, 90.0)
UNSTATED = _offering("openrouter:unstated", None, 60.0)


# --- The threshold ------------------------------------------------------


def test_a_route_at_exactly_the_threshold_is_kept():
    answer = _answer(EXACT, min_context=800_000)

    assert "claude-or-exact" in _aliases(answer)


def test_a_route_below_the_threshold_is_dropped():
    answer = _answer(WIDE, NARROW, min_context=800_000)

    assert "claude-or-wide" in _aliases(answer)
    assert "claude-or-narrow" not in _aliases(answer)


def test_a_row_whose_every_route_is_too_narrow_is_dropped():
    answer = _answer(NARROW, min_context=800_000)

    assert answer.rows == ()


def test_no_filter_returns_every_route():
    answer = _answer(WIDE, NARROW, UNSTATED)

    assert _aliases(answer) == {
        "claude-or-wide",
        "claude-or-narrow",
        "claude-or-unstated",
    }


# --- An unstated window does not qualify --------------------------------


def test_a_route_stating_no_window_is_excluded():
    answer = _answer(UNSTATED, min_context=1)

    assert answer.rows == ()


def test_the_warning_counts_the_unstated_separately_from_the_too_narrow():
    """The rule, not the wording: the two reasons must be named apart.

    One Route was too narrow and one stated no window. A line reporting
    "2 Routes removed" would let a reader conclude the unstated one was
    small, which is the inference ADR 0006 forbids.
    """
    answer = _answer(WIDE, NARROW, UNSTATED, min_context=800_000)
    text = " ".join(answer.warnings).lower()

    assert "smaller window" in text
    assert "stating none" in text
    assert text.count("1 route(s)") == 2


# --- Never silent -------------------------------------------------------


def test_a_filter_that_removes_everything_still_explains_itself():
    answer = _answer(NARROW, UNSTATED, min_context=999_999_999)

    assert answer.rows == ()
    assert answer.warnings, "an empty answer must say why it is empty"
    assert "999999999" in " ".join(answer.warnings).replace(",", "")


def test_a_filter_that_removes_nothing_adds_no_warning():
    answer = _answer(WIDE, min_context=1)

    assert not [w for w in answer.warnings if "context" in w.lower()]


def test_the_warning_reaches_the_json_shape():
    answer = _answer(NARROW, min_context=800_000)

    assert answer.as_dict()["warnings"]


# --- Composition -------------------------------------------------------


def test_the_filter_composes_with_prefer():
    answer = _answer(WIDE, NARROW, min_context=800_000, prefer="free")

    assert "claude-or-narrow" not in _aliases(answer)
    assert answer.prefer == "free"


def test_the_filter_composes_with_limit():
    answer = _answer(WIDE, EXACT, NARROW, min_context=800_000, limit=1)

    assert len(answer.rows) == 1
    assert "claude-or-narrow" not in _aliases(answer)


def test_the_filter_leaves_the_row_ordering_alone():
    """Narrowing must not re-rank what survives."""
    unfiltered = _answer(WIDE, EXACT)
    filtered = _answer(WIDE, EXACT, NARROW, min_context=800_000)

    assert [r.canonical_model_id for r in filtered.rows] == [
        r.canonical_model_id for r in unfiltered.rows
    ]


# --- A bad value is refused at the flag --------------------------------


@pytest.mark.parametrize("value", [0, -1, "wide"])
def test_a_non_positive_or_non_integer_value_is_refused(value):
    with pytest.raises(guidance.GuidanceError, match="min_context"):
        _answer(WIDE, min_context=value)
