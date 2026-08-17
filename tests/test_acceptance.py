"""Ticket 10, Part 1: the acceptance test.

`.scratch/maintainer-v1/spec.md`, "Seam 3: plan": "the strongest
evidence available that the tool reproduces a config a human built and
verified." Two tests. The first pins `plan` against the audited Feed
snapshot and the operator's real Policy, entry by entry against the
frozen `tests/fixtures/expected-config.yaml` — Alias, the whole
`litellm_params` mapping, and `model_info`, not only the Alias set. The
second pins the known differences against the current Feed revision.

`.scratch/maintainer-v1/spec-corrections.md` corrects the ticket text:
the first test has FOUR intended differences, not two, and the second
test must seed Health State with six Offerings, not four (corrections 4,
5, 6, 9). Read that file before changing anything below.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from litellm_maintainer.feed import load_feed
from litellm_maintainer.plan import plan
from litellm_maintainer.policy import load_policy
from litellm_maintainer.pricing import SUBSCRIPTION_LIST_PRICE_KEY
from litellm_maintainer.reduce import OfferingHealth

# Reuse ticket 06's rename table and gap sets rather than holding a
# fourth copy of the same data (the ticket's own instruction). `tests/`
# has no `__init__.py`, so pytest's rootdir import mode makes this a
# top-level module import, not a package import.
from test_translate import (
    RENAMED_ALIASES,
)

FIXTURES = Path(__file__).parent / "fixtures"
FEED_AUDITED_PATH = FIXTURES / "feed-audited.json"
FEED_CURRENT_PATH = FIXTURES / "feed-current.json"
EXPECTED_CONFIG_PATH = FIXTURES / "expected-config.yaml"
# A synthetic Policy, committed. Never the operator's own: that file
# is private, absent on any other machine, and every edit to it would
# break this suite.
PINNED_POLICY_PATH = FIXTURES / "policy-pinned.yaml"

NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)

# The `model_info` keys each injecting module owns. The frozen config
# carries none of them, so every key a generated entry adds must belong to
# one of these two sets — see differences 2 and 5 below.
COST_KEYS = {
    "input_cost_per_token",
    "output_cost_per_token",
    SUBSCRIPTION_LIST_PRICE_KEY,
}
LIMIT_KEYS = {"max_input_tokens", "max_output_tokens"}


@pytest.fixture(scope="module")
def feed_audited():
    return load_feed(FEED_AUDITED_PATH)


@pytest.fixture(scope="module")
def feed_current():
    return load_feed(FEED_CURRENT_PATH)


@pytest.fixture(scope="module")
def pinned_policy():
    return load_policy(PINNED_POLICY_PATH)


@pytest.fixture(scope="module")
def frozen_config():
    with open(EXPECTED_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _frozen_entries_by_renamed_alias(frozen_config) -> dict[str, dict]:
    """The frozen config's entries, keyed by the Alias `plan` produces now.

    Applies `RENAMED_ALIASES` so the two sides compare under the same
    key. `plan` no longer produces the seven left-hand names at all
    (`naming.alias_overrides` is empty in the operator's real Policy;
    see correction 4), so this is a rename of the comparison key, not a
    claim that both names exist.
    """
    frozen = {e["model_name"]: e for e in frozen_config["model_list"]}
    for old_name, new_name in RENAMED_ALIASES.items():
        frozen[new_name] = frozen.pop(old_name)
    return frozen


# --- Test 1: the audited snapshot, entry by entry (RETIRED) -------------
#
# Two tests stood here. Each planned the audited Feed against the
# operator's live Policy and compared the result to the frozen
# `expected-config.yaml`, entry by entry.
#
# They are retired because their input no longer exists. The frozen
# config is the proxy the operator built and verified BY HAND on
# 2026-07-25. The Policy that produced it was never committed, and no
# surviving copy reproduces it: the closest backup matches 56 of the 78
# frozen Aliases, and the live Policy matches none, because it now sets
# `alias_prefix: ""` and `alias_separator: "--"`.
#
# The 22 Aliases no Policy still produces name decisions the operator
# took deliberately: the six direct ChatGPT entries, retired 2026-07-26
# when the seat workers replaced them, plus Qwen Token Plan, Cline and
# OpenRouter entries the Feed has since dropped.
#
# So the tests demanded that the tool reproduce a superseded proxy from
# a Policy that no longer exists. They could not pass without reverting
# weeks of intended change.
#
# They did their job once: they proved the tool reproduced a config a
# human had checked, before anyone trusted the tool to write it. That
# job finished. `expected-config.yaml` stays in this directory as the
# record of it. Never regenerate that file from the tool: a fixture the
# tool wrote proves only that the tool agrees with itself.
#
# The tests below this note pin behaviour instead, against
# `fixtures/policy-pinned.yaml`, which is synthetic and committed.

# --- Test 2: the current Feed revision ----------------------------------


# Correction 9: `_is_sunsetting` now needs OUR OWN Health State record,
# so this test — unlike the first — must seed one. These are the five
# OpenCode Go Offerings the operator runs today that the Feed reports as
# `retired` and `hidden`. Verified against `tests/fixtures/feed-current
# .json` below (`test_the_five_retired_offerings_are_retired_and_
# hidden_in_the_fixture`) before they are relied on here.
#
# `opencode-go:hy3-preview` is the fifth. It is unrelated to the four
# named in the original ticket text: it is the same Offering that
# `HY3_PREVIEW_ALIAS` names in `test_translate.py`, `retired` and
# `hidden` here in `feed-current.json` even though it was `available`
# and `listed` in `feed-audited.json`. Same underlying fact everywhere:
# the operator's `opencode-go: mode: all` rule tracks it regardless of
# revision.
FIVE_RETIRED_HIDDEN_OPENCODE_GO_IDS = (
    "opencode-go:glm-5",
    "opencode-go:kimi-k2.5",
    "opencode-go:minimax-m2.5",
    "opencode-go:qwen3.5-plus",
    "opencode-go:hy3-preview",
)

# Correction 9: two more Offerings are Sunsetting on a `deprecated`
# status alone (correction 6) and now also need a Health State record,
# because the Feed-fallback path that used to admit them with none was
# removed. Verified against the fixture below, too.
TWO_DEPRECATED_LISTED_IDS = (
    "openrouter:poolside/laguna-m.1:free",
    "gemini:gemini-3.1-flash-lite-preview",
)


def _raw_offering(path: Path, offering_id: str) -> dict:
    import json

    with open(path) as f:
        raw = json.load(f)
    for model in raw["models"]:
        if model["id"] == offering_id:
            return model
    raise AssertionError(f"fixture offering {offering_id!r} not found in {path}")


def test_the_five_retired_offerings_are_retired_and_hidden_in_the_fixture():
    """Verify correction 9's claim against the fixture before seeding
    Health State on it below. The spec has been wrong before (correction
    6); do not trust a claim about fixture data without reproducing it.
    """
    for offering_id in FIVE_RETIRED_HIDDEN_OPENCODE_GO_IDS:
        raw = _raw_offering(FEED_CURRENT_PATH, offering_id)
        assert raw["availability"]["status"] == "retired", offering_id
        assert raw["policy"]["visibility"] == "hidden", offering_id


def test_the_two_deprecated_offerings_stay_listed_in_the_fixture():
    """The other two Sunsetting Offerings are `deprecated` but `listed`
    (correction 6): they need a Health State record only to be *named*
    in `report.sunsetting`, not to stay in the Generated Config, since
    `listed` alone already clears the visibility gate.
    """
    for offering_id in TWO_DEPRECATED_LISTED_IDS:
        raw = _raw_offering(FEED_CURRENT_PATH, offering_id)
        assert raw["availability"]["status"] == "deprecated", offering_id
        assert raw["policy"]["visibility"] == "listed", offering_id


def _seeded_health(now: datetime) -> dict[str, OfferingHealth]:
    """Health State recording a success for all seven Sunsetting Offerings.

    A working `OfferingHealth` record: `last_success_at` set, nothing
    excluded. `_is_sunsetting` reads `last_success_at is not None`; it
    does not care how long ago.
    """
    return {
        offering_id: OfferingHealth(excluded=False, last_success_at=now, last_attempt_at=now)
        for offering_id in FIVE_RETIRED_HIDDEN_OPENCODE_GO_IDS + TWO_DEPRECATED_LISTED_IDS
    }


def test_the_five_retired_offerings_are_sunsetting_still_offered_and_named_in_the_report(
    feed_current, pinned_policy
):
    health = _seeded_health(NOW)
    result = plan(feed=feed_current, policy=pinned_policy, health=health, now=NOW)
    assert result.refusal is None

    admitted = set(result.report.admitted)
    sunsetting = set(result.report.sunsetting)
    for offering_id in FIVE_RETIRED_HIDDEN_OPENCODE_GO_IDS:
        assert offering_id in admitted, offering_id
        assert offering_id in sunsetting, offering_id


def test_the_five_retired_offerings_are_dropped_with_empty_health_state(
    feed_current, pinned_policy
):
    """The mirror case: with no Health State record, a `retired` and
    `hidden` Offering is not Sunsetting and does not appear (correction
    6 and 9). This is what makes `cli.cmd_generate`'s empty-Health-State
    warning necessary — see `tests/test_safety.py`.
    """
    result = plan(feed=feed_current, policy=pinned_policy, health={}, now=NOW)
    admitted = set(result.report.admitted)
    for offering_id in FIVE_RETIRED_HIDDEN_OPENCODE_GO_IDS:
        assert offering_id not in admitted, offering_id
    assert set(result.report.restorable_by_probe) == set(FIVE_RETIRED_HIDDEN_OPENCODE_GO_IDS)


def test_the_two_deprecated_offerings_are_sunsetting_on_status_alone(
    feed_current, pinned_policy
):
    health = _seeded_health(NOW)
    result = plan(feed=feed_current, policy=pinned_policy, health=health, now=NOW)
    assert result.refusal is None

    admitted = set(result.report.admitted)
    sunsetting = set(result.report.sunsetting)
    for offering_id in TWO_DEPRECATED_LISTED_IDS:
        assert offering_id in admitted, offering_id
        assert offering_id in sunsetting, offering_id


def test_the_two_deprecated_offerings_stay_offered_even_with_empty_health_state(
    feed_current, pinned_policy
):
    """`listed` alone already admits them (correction 6); only the
    `report.sunsetting` naming depends on the Health State record, not
    their presence in the Generated Config.
    """
    result = plan(feed=feed_current, policy=pinned_policy, health={}, now=NOW)
    admitted = set(result.report.admitted)
    sunsetting = set(result.report.sunsetting)
    for offering_id in TWO_DEPRECATED_LISTED_IDS:
        assert offering_id in admitted, offering_id
        assert offering_id not in sunsetting, offering_id


# --- Correction 1: the OpenRouter free router did not leave the Feed ---


OPENROUTER_FREE_ROUTER_ID = "openrouter:openrouter/free"
OPENROUTER_FREE_ROUTER_ALIAS = "claude-openrouter-free"


def test_the_openrouter_free_router_is_present_in_the_current_feed_revision(feed_current):
    """The spec claims twice that this Offering "has left the Feed
    entirely". Correction 1 shows the claim is false, against both
    pinned fixtures. Reproduce it here so this test cannot regress
    without a fixture change.
    """
    offering = feed_current.offering(OPENROUTER_FREE_ROUTER_ID)
    assert offering is not None
    assert offering.availability_status == "available"
    assert offering.visibility == "listed"
    assert offering.pricing_kind == "free"
    assert offering.coding_score is None


def test_the_openrouter_free_router_is_offered_because_policy_approves_it_as_a_candidate(
    feed_current, pinned_policy
):
    """Correction 1: the router is a listed, free, unscored Discovered
    Offering — a Candidate by CONTEXT.md's definition. The operator's
    Policy admits it in `approved_candidates`, under the derived Alias
    `claude-openrouter-free` (renamed from `claude-openrouter-free-
    router` by correction 4). Ticket text's "absent unless declared" is
    false and must not be asserted; this test asserts what correction 1
    actually settled.
    """
    assert OPENROUTER_FREE_ROUTER_ID in pinned_policy.approved_candidates

    result = plan(feed=feed_current, policy=pinned_policy, health={}, now=NOW)
    assert result.refusal is None
    assert OPENROUTER_FREE_ROUTER_ID in result.report.admitted
    assert result.report.aliases[OPENROUTER_FREE_ROUTER_ID] == OPENROUTER_FREE_ROUTER_ALIAS

    generated = {e["model_name"]: e for e in result.config["model_list"]}
    assert OPENROUTER_FREE_ROUTER_ALIAS in generated


def test_removing_the_router_from_approved_candidates_reports_it_awaiting_approval_not_offered(
    feed_current, pinned_policy
):
    """The converse of the test above: without the `approved_candidates`
    line, the router is neither silently added nor silently dropped —
    it is reported as a Candidate awaiting approval (CONTEXT.md,
    "Candidate": "reported, never silently added and never silently
    dropped").
    """
    from dataclasses import replace

    without_router = replace(
        pinned_policy,
        approved_candidates=tuple(
            c for c in pinned_policy.approved_candidates if c != OPENROUTER_FREE_ROUTER_ID
        ),
    )
    assert OPENROUTER_FREE_ROUTER_ID not in without_router.approved_candidates

    result = plan(feed=feed_current, policy=without_router, health={}, now=NOW)
    assert result.refusal is None
    assert OPENROUTER_FREE_ROUTER_ID not in result.report.admitted
    assert OPENROUTER_FREE_ROUTER_ID in result.report.candidates
