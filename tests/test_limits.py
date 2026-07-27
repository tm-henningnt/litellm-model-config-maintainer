"""Ticket 02: the Stated Limits the Feed publishes reach the Generated Config.

Why this exists. No entry carried a token limit, so litellm fell back to
its own cost map. For a model absent from that map it applies a regex
rule, `claude-family-baseline`, asserting 200,000 input and 64,000 output
tokens. Then proxy startup registers every deployment and writes an
all-null cost-map entry for a model that is not already an exact key, and
rules run only after exact lookups miss — so the null shadows the guess.
Measured on 2026-07-26: `openrouter/anthropic/claude-opus-5` resolved to
200000/64000 outside the proxy and to nothing inside it, against the
Feed's 1000000/128000. A client sizing a prompt from that reading loses
most of the window.

Assert external behaviour: the `model_info` a built entry carries. A test
name states a rule an operator would recognise.

Two tests read a real Offering from the frozen `feed-audited.json`,
translated through the real `translate_offering`, so the divergence from
`cost_model_info` is pinned against real Feed data in one place. The
edge cases construct their own Feed document in memory, because no real
Offering in the frozen fixture states a `null` or non-positive figure
beside a stated one.
"""

from __future__ import annotations

import copy
from typing import Any

from litellm_maintainer.feed import load_feed
from litellm_maintainer.plan import plan
from litellm_maintainer.pricing import SUBSCRIPTION_LIST_PRICE_KEY
from tests.conftest import FIXTURES
from tests.test_pricing import _feed_with, _offering_raw, _policy

NATIVE_PREFIX_OFFERING = "openrouter:qwen/qwen3-coder:free"
GENERIC_BASE_OFFERING = "opencode-go:minimax-m3"


def _entry_for(offering_id: str) -> dict[str, Any]:
    """The built entry for one Offering of the frozen audited Feed."""
    feed = load_feed(FIXTURES / "feed-audited.json")
    provider_id = offering_id.split(":", 1)[0]
    policy = _policy(
        providers={provider_id: {"mode": "all"}},
        naming={
            "alias_prefix": "claude-",
            "provider_labels": {provider_id: provider_id},
            "alias_overrides": {offering_id: "claude-under-test"},
        },
    )
    result = plan(feed=feed, policy=policy, health={}, now=None)  # type: ignore[arg-type]
    assert result.refusal is None
    assert offering_id in result.report.admitted
    return next(
        entry
        for entry in result.config["model_list"]
        if entry["model_name"] == "claude-under-test"
    )


def _entry_from(offering: dict[str, Any]) -> dict[str, Any]:
    """The built entry for one constructed Offering."""
    result = plan(feed=_feed_with(offering), policy=_policy(), health={}, now=None)  # type: ignore[arg-type]
    assert result.refusal is None
    return result.config["model_list"][0]


def _with_limits(limits: Any, **kwargs: Any) -> dict[str, Any]:
    raw = _offering_raw(id="opencode-go:under-test", **kwargs)
    if limits is not _ABSENT:
        raw["limits"] = copy.deepcopy(limits)
    return raw


_ABSENT = object()


# --- The ordinary case, pinned against real Feed data ---


def test_an_offering_whose_feed_states_both_figures_carries_both_limits():
    entry = _entry_for(NATIVE_PREFIX_OFFERING)

    assert entry["model_info"]["max_input_tokens"] == 262144
    assert entry["model_info"]["max_output_tokens"] == 65536


def test_a_native_prefix_offering_receives_limit_metadata_unlike_cost_metadata():
    """The divergence from `cost_model_info`, pinned in one place.

    Cost suppresses itself for a native litellm prefix, because litellm
    prices such a model correctly. That reasoning does not transfer:
    litellm resolved a native-prefix OpenRouter route to Claude Opus 5 at
    200000/64000 against the Feed's 1000000/128000. A native prefix does
    not imply litellm knows the window. See ADR 0006.
    """
    entry = _entry_for(NATIVE_PREFIX_OFFERING)

    assert "api_base" not in entry["litellm_params"]  # the native-prefix marker
    assert "input_cost_per_token" not in entry["model_info"]  # cost suppressed
    assert entry["model_info"]["max_input_tokens"] == 262144  # limits are not


def test_cost_metadata_and_limit_metadata_coexist_on_one_entry():
    entry = _entry_for(GENERIC_BASE_OFFERING)

    assert entry["model_info"]["max_input_tokens"] == 1000000
    assert entry["model_info"]["max_output_tokens"] == 131072
    assert "input_cost_per_token" in entry["model_info"]
    assert "output_cost_per_token" in entry["model_info"]
    assert entry["model_info"][SUBSCRIPTION_LIST_PRICE_KEY] is True


# --- A figure is written only when the Feed states a positive int ---


def test_an_offering_whose_feed_states_no_limits_carries_no_limit_keys():
    """Absence must read as unknown, never as small."""
    entry = _entry_from(_with_limits(_ABSENT))

    info = entry.get("model_info", {})
    assert "max_input_tokens" not in info
    assert "max_output_tokens" not in info


def test_a_null_figure_is_treated_as_unstated():
    entry = _entry_from(
        _with_limits({"context_tokens": None, "max_output_tokens": None})
    )

    info = entry.get("model_info", {})
    assert "max_input_tokens" not in info
    assert "max_output_tokens" not in info


def test_a_zero_figure_is_treated_as_unstated():
    entry = _entry_from(_with_limits({"context_tokens": 0, "max_output_tokens": 0}))

    info = entry.get("model_info", {})
    assert "max_input_tokens" not in info
    assert "max_output_tokens" not in info


def test_a_negative_figure_is_treated_as_unstated():
    entry = _entry_from(_with_limits({"context_tokens": -1, "max_output_tokens": -1}))

    info = entry.get("model_info", {})
    assert "max_input_tokens" not in info
    assert "max_output_tokens" not in info


def test_a_non_integer_figure_is_treated_as_unstated():
    entry = _entry_from(
        _with_limits({"context_tokens": "262144", "max_output_tokens": 1.5})
    )

    info = entry.get("model_info", {})
    assert "max_input_tokens" not in info
    assert "max_output_tokens" not in info


def test_one_stated_figure_is_not_withheld_because_the_other_is_missing():
    entry = _entry_from(_with_limits({"context_tokens": 262144}))

    assert entry["model_info"]["max_input_tokens"] == 262144
    assert "max_output_tokens" not in entry["model_info"]


def test_a_stated_output_figure_is_written_without_a_context_figure():
    entry = _entry_from(
        _with_limits({"context_tokens": None, "max_output_tokens": 65536})
    )

    assert entry["model_info"]["max_output_tokens"] == 65536
    assert "max_input_tokens" not in entry["model_info"]


# --- Rules about where a limit may never be written ---


def test_max_tokens_is_never_written():
    """`trim_messages` falls back from the input limit to `max_tokens`.

    Writing both makes the pair ambiguous, so only the two explicit
    figures are ever written.
    """
    entry = _entry_from(
        _with_limits({"context_tokens": 262144, "max_output_tokens": 65536})
    )

    assert "max_tokens" not in entry["model_info"]


def test_no_limit_is_written_to_litellm_params():
    """A `max_tokens` there is sent to the provider and caps every caller.

    An input limit there is read by nothing and travels as an unknown
    completion kwarg.
    """
    entry = _entry_from(
        _with_limits({"context_tokens": 262144, "max_output_tokens": 65536})
    )

    for key in ("max_tokens", "max_input_tokens", "max_output_tokens"):
        assert key not in entry["litellm_params"]


def test_an_entry_with_neither_cost_nor_limits_carries_no_model_info_at_all():
    offering = _with_limits(_ABSENT)
    offering["pricing"] = {"kind": "unknown", "metering": "tokens"}
    entry = _entry_from(offering)

    assert "model_info" not in entry
