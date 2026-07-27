"""Ticket 01: a Route names the Alias that yields its stated window.

Why this exists. A Route reported `context_tokens: 1000000` beside an Alias
that delivers 200,000, because a calling client derives its budget from the
Alias name rather than from the Stated Limit the proxy reports (ADR 0007).
The Alias that yields the window is the Client-Facing Variant the Generator
already writes, and the answer never named it. So an agent following
`guidance` could not reach the window `guidance` had just advertised.

The value comes from the run report's derived pairs. Never from parsing an
Alias: the suffix is an operator setting, so a guess breaks the moment they
change it. The sibling project forbids that workaround in its own ADR 0001.

Assert external behaviour: the fields of a Route, and the JSON a caller
parses.
"""

from __future__ import annotations

from datetime import datetime, timezone

from litellm_maintainer import guidance
from litellm_maintainer.feed import load_feed
from litellm_maintainer.plan import PlanReport, plan
from litellm_maintainer.policy import load_policy, parse_policy
from tests.conftest import FIXTURES
from tests.test_guidance_declared import _feed, _policy

NOW = datetime(2026, 7, 26, 12, 0, tzinfo=timezone.utc)

SCORED_ID = "openrouter:vendor/scored"
SCORED_ALIAS = "claude-or-scored"


def _derive(*, variants=(), aliases=None):
    return guidance.derive(
        feed=_feed(),
        policy=_policy([]),
        health={},
        report=PlanReport(
            admitted=(SCORED_ID,),
            aliases=aliases if aliases is not None else {SCORED_ID: SCORED_ALIAS},
            client_facing_variants=variants,
        ),
        now=NOW,
    )


def _route(answer, alias):
    return next(
        route for row in answer.rows for route in row.routes if route.alias == alias
    )


# --- The field ----------------------------------------------------------


def test_a_route_names_the_wide_alias_the_generator_derived():
    answer = _derive(variants=((SCORED_ALIAS, f"{SCORED_ALIAS}[1m]"),))

    assert _route(answer, SCORED_ALIAS).wide_alias == f"{SCORED_ALIAS}[1m]"


def test_a_route_with_no_variant_names_nothing():
    """Absence means the plain Alias is all there is."""
    answer = _derive(variants=())

    assert _route(answer, SCORED_ALIAS).wide_alias is None


def test_the_wide_alias_reaches_the_json_shape():
    answer = _derive(variants=((SCORED_ALIAS, f"{SCORED_ALIAS}[1m]"),))

    assert _route(answer, SCORED_ALIAS).as_dict()["wide_alias"] == f"{SCORED_ALIAS}[1m]"


def test_the_json_shape_carries_the_key_even_when_there_is_no_variant():
    """A consumer reads a stable key set, so the field is never omitted."""
    answer = _derive(variants=())

    assert _route(answer, SCORED_ALIAS).as_dict()["wide_alias"] is None


def test_a_pair_naming_another_alias_does_not_leak_onto_this_route():
    answer = _derive(variants=(("claude-someone-else", "claude-someone-else[1m]"),))

    assert _route(answer, SCORED_ALIAS).wide_alias is None


def test_the_value_is_the_reported_pair_not_a_derived_suffix():
    """Proves nothing is appended: the report names an unrelated suffix."""
    answer = _derive(variants=((SCORED_ALIAS, f"{SCORED_ALIAS}-WIDE-BY-REPORT"),))

    assert _route(answer, SCORED_ALIAS).wide_alias == f"{SCORED_ALIAS}-WIDE-BY-REPORT"


# --- The schema version -------------------------------------------------


def test_the_schema_version_rose_for_the_new_field():
    answer = _derive()

    assert answer.as_dict()["schema_version"] == guidance.SCHEMA_VERSION
    assert guidance.SCHEMA_VERSION == "2"


# --- Pinned against the real Feed and the real Policy -------------------


def test_a_wide_alias_names_an_alias_the_generated_config_holds():
    """The whole point: the named Alias must be dispatchable.

    Runs the real Generator over the frozen audited Feed and the operator's
    Policy, then checks every Wide Alias `guidance` reports against the
    Aliases the Generated Config actually contains.
    """
    feed = load_feed(FIXTURES / "feed-audited.json")
    policy = load_policy(
        __import__("pathlib").Path("/Users/hentol/.config/litellm-maintainer/policy.yaml")
    )
    result = plan(feed=feed, policy=policy, health={}, now=NOW)
    config_aliases = {e["model_name"] for e in result.config["model_list"]}

    answer = guidance.derive(
        feed=feed, policy=policy, health={}, report=result.report, now=NOW
    )
    named = {
        route.wide_alias
        for row in answer.rows
        for route in row.routes
        if route.wide_alias is not None
    }

    assert named, "expected at least one Route to name a Wide Alias"
    assert named <= config_aliases, (
        "guidance named a Wide Alias the Generated Config does not hold: "
        f"{sorted(named - config_aliases)}"
    )


def test_every_reported_pair_reaches_a_route():
    """No derived variant is silently unreachable through guidance."""
    feed = load_feed(FIXTURES / "feed-audited.json")
    policy = load_policy(
        __import__("pathlib").Path("/Users/hentol/.config/litellm-maintainer/policy.yaml")
    )
    result = plan(feed=feed, policy=policy, health={}, now=NOW)
    answer = guidance.derive(
        feed=feed, policy=policy, health={}, report=result.report, now=NOW
    )

    reported = {variant for _, variant in result.report.client_facing_variants}
    named = {
        route.wide_alias
        for row in answer.rows
        for route in row.routes
        if route.wide_alias is not None
    }

    assert reported <= named, f"unreachable: {sorted(reported - named)}"
