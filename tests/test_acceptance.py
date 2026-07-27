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
    CLIENT_FACING_VARIANT_ALIASES,
    derived_variant_aliases,
    DIRECT_CHATGPT_ALIASES_RETIRED,
    UNWITHHELD_OPENROUTER_ALIAS,
    HY3_PREVIEW_ALIAS,
    OPENCODE_GO_MOVED_ALIASES,
    RENAMED_ALIASES,
    SEAT_ALIASES,
    personal_plan_denied_aliases,
)

FIXTURES = Path(__file__).parent / "fixtures"
FEED_AUDITED_PATH = FIXTURES / "feed-audited.json"
FEED_CURRENT_PATH = FIXTURES / "feed-current.json"
EXPECTED_CONFIG_PATH = FIXTURES / "expected-config.yaml"
OPERATOR_POLICY_PATH = Path("/Users/hentol/.config/litellm-maintainer/policy.yaml")

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
def operator_policy():
    return load_policy(OPERATOR_POLICY_PATH)


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


# --- Test 1: the audited snapshot, entry by entry -----------------------


def test_the_audited_snapshot_reproduces_the_frozen_config_entry_by_entry(
    feed_audited, operator_policy, frozen_config
):
    """Given the audited Feed, the operator's Policy and empty Health
    State, `plan` reproduces all 78 frozen entries — Alias,
    `litellm_params` and `model_info` alike — except for the nine
    Aliases the operator's Policy now Withholds (Qwen Token Plan,
    personal-tier denial), plus four other documented differences
    (spec-corrections.md, corrections 4 and 5; ticket text names only
    two). Any entry outside those five categories must match the frozen
    config exactly, field for field.

    On top of the 78, exactly 13 new Aliases now appear and are named
    explicitly, never absorbed into an open-ended "anything else is
    fine": the 12 ChatGPT worker-seat Declared Offerings (`SEAT_ALIASES`,
    added 2026-07-26) and one newly-admitted Discovered Offering,
    `HY3_PREVIEW_ALIAS` — unrelated to the seat change, see that
    constant's docstring in `test_translate.py`. Nine Aliases are
    withdrawn: `personal_plan_denied_aliases`, named explicitly in
    `test_translate.py` next to `PERSONAL_PLAN_DENIED_OFFERING_IDS`. Net
    count: 78 + 13 - 9 = 82.
    """
    result = plan(feed=feed_audited, policy=operator_policy, health={}, now=NOW)
    assert result.refusal is None

    generated = {e["model_name"]: e for e in result.config["model_list"]}
    frozen = _frozen_entries_by_renamed_alias(frozen_config)
    new_aliases = (
        SEAT_ALIASES
        | CLIENT_FACING_VARIANT_ALIASES
        | derived_variant_aliases(generated)
        | {HY3_PREVIEW_ALIAS, UNWITHHELD_OPENROUTER_ALIAS}
    )
    denied_aliases = personal_plan_denied_aliases(operator_policy)

    # 14 hand-named new Aliases + 3 hand-declared Claude variants + every
    # variant the Generator DERIVES from the Feed. The derived count follows
    # the Feed, so it is stated as a composition rather than as a number: a
    # Feed revision that widens one model changes it. It must not be zero,
    # or the derivation silently stopped.
    derived_variants = derived_variant_aliases(generated)
    assert derived_variants, "the Generator derived no Client-Facing Variant"
    assert len(new_aliases) == 17 + len(derived_variants)
    assert len(denied_aliases) == 9
    assert denied_aliases <= set(frozen)
    assert denied_aliases.isdisjoint(new_aliases)
    assert denied_aliases.isdisjoint(set(generated)), (
        "a Withheld Alias must never reach the Generated Config"
    )

    # 6 retired (DIRECT_CHATGPT_ALIASES_RETIRED), 1 re-admitted when its
    # stale Withheld line was pruned (UNWITHHELD_OPENROUTER_ALIAS).
    # + the variants the Generator derives from the Feed, counted above.
    assert len(generated) == 78 + 13 + 3 - 9 - 6 + 1 + len(derived_variants)
    assert new_aliases <= set(generated)
    assert DIRECT_CHATGPT_ALIASES_RETIRED.isdisjoint(set(generated)), (
        "a retired direct chatgpt/ Alias must never reach the Generated Config"
    )
    assert (
        set(generated) - new_aliases
        == set(frozen) - denied_aliases - DIRECT_CHATGPT_ALIASES_RETIRED
    ), (
        "the Alias set itself differs — that would hide a wrong base URL "
        "even if this assertion alone passed, which is why the entry-by-"
        "entry comparison below still runs on every shared Alias"
    )

    unexplained: list[tuple[str, dict, dict]] = []
    for alias, generated_entry in generated.items():
        # Difference 5: the 12 ChatGPT worker-seat Aliases. They are not
        # IN the frozen config at all (added 2026-07-26), so there is no
        # frozen entry to diff against; their exact shape is pinned in
        # test_translate.py's dedicated seat test.
        if alias in SEAT_ALIASES:
            continue

        # Difference 7: the three Client-Facing Variants, also absent
        # from the frozen config. Each must reach the SAME model string
        # as its plain sibling, with no extra `litellm_params` — the
        # suffix is for the client, never for the provider.
        if alias in CLIENT_FACING_VARIANT_ALIASES or alias in derived_variants:
            plain = alias.removesuffix("[1m]")
            assert (
                generated_entry["litellm_params"] == generated[plain]["litellm_params"]
            ), "a Client-Facing Variant must send the same request as its sibling"
            continue

        # Difference 6, unrelated to the seat change: a newly-admitted
        # Discovered Offering, also absent from the frozen config. See
        # `HY3_PREVIEW_ALIAS`'s docstring in test_translate.py.
        if alias == HY3_PREVIEW_ALIAS:
            assert generated_entry["litellm_params"]["model"] == "openai/hy3-preview"
            continue

        # Re-admitted when its stale Withheld line was pruned, so it is
        # absent from the frozen config too.
        if alias == UNWITHHELD_OPENROUTER_ALIAS:
            assert generated_entry["litellm_params"]["model"] == (
                "openrouter/qwen/qwen3-coder:free"
            )
            continue

        frozen_entry = frozen[alias]

        # Difference 1: the seven OpenCode Go Aliases move from the
        # anthropic-shaped prefix to the generic openai/ prefix.
        if alias in OPENCODE_GO_MOVED_ALIASES:
            assert generated_entry["litellm_params"]["model"].startswith("openai/")
            continue

        # The nine Cline Aliases were difference 4: they lacked the
        # `cline/` handler prefix, because the audited snapshot carries
        # the envelope key on zero of its 1164 Offerings (correction 5).
        # The operator's `providers.cline.response_envelope_key` supplies
        # what the snapshot omits, so they now reach the frozen prefix and
        # fall through to the comparison below like any other Alias.

        generated_info = generated_entry.get("model_info", {})
        frozen_info = frozen_entry.get("model_info", {})

        # Differences 2 and 5: cost metadata injected wherever the Feed
        # meters in tokens, and limit metadata injected wherever the Feed
        # states a Stated Limit. The frozen config carries neither, so any
        # `model_info` difference here must be exactly one of those two
        # shapes: `litellm_params` still matches, and every extra key is a
        # cost field or a limit field, never a changed
        # `model`/`api_base`/`api_key`.
        #
        # The two are independent. An Offering with a native litellm
        # prefix gets limits and NO cost metadata, because
        # `cost_model_info` suppresses itself there while
        # `limits_model_info` deliberately does not (ADR 0006). So this
        # check must not require the cost pair.
        if generated_info != frozen_info:
            if generated_entry["litellm_params"] != frozen_entry["litellm_params"]:
                unexplained.append((alias, generated_entry, frozen_entry))
                continue
            if not set(generated_info).issubset(COST_KEYS | LIMIT_KEYS):
                unexplained.append((alias, generated_entry, frozen_entry))
            continue

        if generated_entry["litellm_params"] != frozen_entry["litellm_params"]:
            unexplained.append((alias, generated_entry, frozen_entry))

    assert unexplained == [], (
        f"{len(unexplained)} Alias(es) differ from the frozen config for a "
        "reason none of the four documented differences explains: "
        f"{unexplained}"
    )


def test_the_audited_snapshot_has_exactly_five_documented_differences_no_sixth(
    feed_audited, operator_policy, frozen_config
):
    """Pin the difference categories down by name and by count, so a
    future change that widens or narrows any of them fails here first,
    loudly, rather than inside the entry-by-entry test above.

    One category is a closure rather than a gap: nine Cline Aliases fell
    back to the generic prefix because the audited snapshot declares no
    response envelope key. The operator's Policy now declares it, so the
    Generator reproduces the frozen prefix for those nine. Check 4 asserts
    that closure.

    Check 5 is the newest: limit metadata, injected wherever the Feed
    states a Stated Limit. It is deliberately NOT folded into the cost
    check, because the two modules disagree about a native litellm prefix
    on purpose (ADR 0006).
    """
    result = plan(feed=feed_audited, policy=operator_policy, health={}, now=NOW)
    generated = {e["model_name"]: e for e in result.config["model_list"]}
    frozen = _frozen_entries_by_renamed_alias(frozen_config)

    # 1. Seven OpenCode Go Aliases move to the generic prefix.
    assert len(OPENCODE_GO_MOVED_ALIASES) == 7
    for alias in OPENCODE_GO_MOVED_ALIASES:
        assert frozen[alias]["litellm_params"]["model"].startswith("anthropic/")
        assert generated[alias]["litellm_params"]["model"].startswith("openai/")

    # 2. Cost metadata injected wherever the Feed meters in tokens. The
    # frozen config carries no `model_info` at all, so any generated one
    # is new. Cost and limits are injected independently, so select the
    # entries by the keys they actually carry rather than assuming every
    # new `model_info` is a cost injection.
    info_injected = {
        alias
        for alias, entry in generated.items()
        if alias in frozen and entry.get("model_info") and not frozen[alias].get("model_info")
    }
    assert info_injected, "expected at least one metadata injection"
    cost_injected = {
        alias for alias in info_injected if COST_KEYS & set(generated[alias]["model_info"])
    }
    assert cost_injected, "expected at least one cost-metadata injection"
    for alias in cost_injected:
        info = generated[alias]["model_info"]
        assert "input_cost_per_token" in info
        assert "output_cost_per_token" in info

    # 5. Limit metadata injected wherever the Feed states a Stated Limit.
    # This is a separate category from cost, not a widening of it: an
    # Offering with a native litellm prefix receives limits and no cost
    # metadata, because `cost_model_info` suppresses itself for a native
    # prefix while `limits_model_info` deliberately does not (ADR 0006).
    # At least one such entry must exist, or the divergence has silently
    # stopped happening.
    limit_injected = {
        alias for alias in info_injected if LIMIT_KEYS & set(generated[alias]["model_info"])
    }
    assert limit_injected, "expected at least one limit-metadata injection"
    for alias in limit_injected:
        info = generated[alias]["model_info"]
        stated = LIMIT_KEYS & set(info)
        assert all(isinstance(info[key], int) and info[key] > 0 for key in stated)
        # Never written, whatever the Feed states.
        assert "max_tokens" not in info

    # ADR 0006's divergence, asserted in the one direction that states it:
    # an entry whose `litellm_params` carries a NATIVE litellm prefix (the
    # marker is a missing `api_base`) receives limits and no cost. Note
    # the converse is not true — cost is also absent when the Feed states
    # no rate or meters in something other than tokens — so this selects
    # on the prefix, not on the absence of cost.
    native_prefix_with_limits = {
        alias
        for alias in limit_injected
        if "api_base" not in generated[alias]["litellm_params"]
    }
    assert native_prefix_with_limits, (
        "expected at least one native-prefix entry carrying limits; if none "
        "remains, ADR 0006's divergence from cost_model_info is untested"
    )
    for alias in native_prefix_with_limits:
        assert not COST_KEYS & set(generated[alias]["model_info"]), (
            "cost_model_info must still suppress itself for a native prefix"
        )

    # Every injected key belongs to one of the two categories. This is
    # what makes "no sixth difference" true rather than merely untested.
    for alias in info_injected:
        assert set(generated[alias]["model_info"]).issubset(COST_KEYS | LIMIT_KEYS)

    # 3. Exactly the seven Aliases in RENAMED_ALIASES change name.
    assert len(RENAMED_ALIASES) == 7
    for new_name in RENAMED_ALIASES.values():
        assert new_name in generated
    for old_name in RENAMED_ALIASES:
        assert old_name not in generated

    # 4. Every claude-cline-free-* Alias routes to the unwrapping handler.
    # The audited snapshot declares the envelope key on no Offering, so
    # this holds only because the operator's Policy declares it instead
    # (`providers.cline.response_envelope_key`). Nine Aliases carried the
    # generic prefix before that declaration existed; none does now.
    cline_aliases = {
        a
        for a in generated
        if a.startswith("claude-cline-free-") and not a.endswith("[1m]")
    }
    assert len(cline_aliases) == 9
    cline_gap = {
        alias
        for alias in cline_aliases
        if not generated[alias]["litellm_params"]["model"].startswith("cline/")
    }
    assert cline_gap == set()


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
    feed_current, operator_policy
):
    health = _seeded_health(NOW)
    result = plan(feed=feed_current, policy=operator_policy, health=health, now=NOW)
    assert result.refusal is None

    admitted = set(result.report.admitted)
    sunsetting = set(result.report.sunsetting)
    for offering_id in FIVE_RETIRED_HIDDEN_OPENCODE_GO_IDS:
        assert offering_id in admitted, offering_id
        assert offering_id in sunsetting, offering_id


def test_the_five_retired_offerings_are_dropped_with_empty_health_state(
    feed_current, operator_policy
):
    """The mirror case: with no Health State record, a `retired` and
    `hidden` Offering is not Sunsetting and does not appear (correction
    6 and 9). This is what makes `cli.cmd_generate`'s empty-Health-State
    warning necessary — see `tests/test_safety.py`.
    """
    result = plan(feed=feed_current, policy=operator_policy, health={}, now=NOW)
    admitted = set(result.report.admitted)
    for offering_id in FIVE_RETIRED_HIDDEN_OPENCODE_GO_IDS:
        assert offering_id not in admitted, offering_id
    assert set(result.report.restorable_by_probe) == set(FIVE_RETIRED_HIDDEN_OPENCODE_GO_IDS)


def test_the_two_deprecated_offerings_are_sunsetting_on_status_alone(
    feed_current, operator_policy
):
    health = _seeded_health(NOW)
    result = plan(feed=feed_current, policy=operator_policy, health=health, now=NOW)
    assert result.refusal is None

    admitted = set(result.report.admitted)
    sunsetting = set(result.report.sunsetting)
    for offering_id in TWO_DEPRECATED_LISTED_IDS:
        assert offering_id in admitted, offering_id
        assert offering_id in sunsetting, offering_id


def test_the_two_deprecated_offerings_stay_offered_even_with_empty_health_state(
    feed_current, operator_policy
):
    """`listed` alone already admits them (correction 6); only the
    `report.sunsetting` naming depends on the Health State record, not
    their presence in the Generated Config.
    """
    result = plan(feed=feed_current, policy=operator_policy, health={}, now=NOW)
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
    feed_current, operator_policy
):
    """Correction 1: the router is a listed, free, unscored Discovered
    Offering — a Candidate by CONTEXT.md's definition. The operator's
    Policy admits it in `approved_candidates`, under the derived Alias
    `claude-openrouter-free` (renamed from `claude-openrouter-free-
    router` by correction 4). Ticket text's "absent unless declared" is
    false and must not be asserted; this test asserts what correction 1
    actually settled.
    """
    assert OPENROUTER_FREE_ROUTER_ID in operator_policy.approved_candidates

    result = plan(feed=feed_current, policy=operator_policy, health={}, now=NOW)
    assert result.refusal is None
    assert OPENROUTER_FREE_ROUTER_ID in result.report.admitted
    assert result.report.aliases[OPENROUTER_FREE_ROUTER_ID] == OPENROUTER_FREE_ROUTER_ALIAS

    generated = {e["model_name"]: e for e in result.config["model_list"]}
    assert OPENROUTER_FREE_ROUTER_ALIAS in generated


def test_removing_the_router_from_approved_candidates_reports_it_awaiting_approval_not_offered(
    feed_current, operator_policy
):
    """The converse of the test above: without the `approved_candidates`
    line, the router is neither silently added nor silently dropped —
    it is reported as a Candidate awaiting approval (CONTEXT.md,
    "Candidate": "reported, never silently added and never silently
    dropped").
    """
    from dataclasses import replace

    without_router = replace(
        operator_policy,
        approved_candidates=tuple(
            c for c in operator_policy.approved_candidates if c != OPENROUTER_FREE_ROUTER_ID
        ),
    )
    assert OPENROUTER_FREE_ROUTER_ID not in without_router.approved_candidates

    result = plan(feed=feed_current, policy=without_router, health={}, now=NOW)
    assert result.refusal is None
    assert OPENROUTER_FREE_ROUTER_ID not in result.report.admitted
    assert OPENROUTER_FREE_ROUTER_ID in result.report.candidates
