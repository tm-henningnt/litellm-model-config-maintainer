"""Tests for ticket 13's `report.py`: the run log, `status`, and the
Feed's own profile picks.

Assert external behaviour, not internal structure. A test name states a
rule an operator would recognise (spec, "What makes a good test
here"). Withheld, Excluded, Candidate and Sunsetting are four different
states (CONTEXT.md); one test per state proves the others are not
confused with it.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from litellm_maintainer.feed import Feed, load_feed
from litellm_maintainer.plan import PlanReport, plan
from litellm_maintainer.policy import Policy, Quality, Naming, Safety, Schedule, load_policy
from litellm_maintainer.redact import build_redaction_map, redact
from litellm_maintainer.reduce import OfferingHealth
from litellm_maintainer.report import (
    ProfilePick,
    append_run_log,
    print_status,
    profile_picks,
    status_lines,
)
from test_translate import PERSONAL_PLAN_DENIED_OFFERING_IDS

NOW = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)

FIXTURES = Path(__file__).parent / "fixtures"
FEED_CURRENT_PATH = FIXTURES / "feed-current.json"
# Synthetic and committed. Never the operator's own Policy.
PINNED_POLICY_PATH = FIXTURES / "policy-pinned.yaml"


def _empty_policy(**overrides) -> Policy:
    defaults = dict(
        providers={},
        quality=Quality(minimum_coding_score=18),
        approved_candidates=(),
        naming=Naming(provider_labels={}, alias_overrides={}, alias_prefix="claude-"),
        withheld={},
        declared=(),
        pacing={},
        schedule=Schedule(
            enabled=False, interval_minutes=60, require_proxy=True, maximum_staleness_hours=24
        ),
        safety=Safety(maximum_removal_share=0.25, snapshot_count=5),
    )
    defaults.update(overrides)
    return Policy(**defaults)


def _feed(profiles: list[dict]) -> Feed:
    return Feed(
        schema_version="1",
        offerings=(),
        providers={},
        profiles=tuple(profiles),
        notices=(),
        raw={},
    )


# --- The run log -----------------------------------------------------------


def test_a_run_that_did_nothing_still_appends_a_log_line(tmp_path):
    path = tmp_path / "state" / "runs.log"
    report = PlanReport()

    append_run_log(path, now=NOW, report=report, notification_count=0, mapping={})

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    assert "notifications=0" in lines[0]
    assert "offered=0" in lines[0]


def test_the_run_log_appends_rather_than_overwrites(tmp_path):
    path = tmp_path / "runs.log"

    append_run_log(path, now=NOW, report=PlanReport(), notification_count=0, mapping={})
    append_run_log(path, now=NOW, report=PlanReport(), notification_count=1, mapping={})

    lines = path.read_text().splitlines()
    assert len(lines) == 2


def test_a_credential_value_in_a_run_log_line_is_redacted(tmp_path):
    path = tmp_path / "runs.log"
    mapping = {"sk-super-secret-value": "<REDACTED:MY_KEY>"}
    report = PlanReport(admitted=("provider:sk-super-secret-value",))

    append_run_log(path, now=NOW, report=report, notification_count=0, mapping=mapping)

    text = path.read_text()
    assert "sk-super-secret-value" not in text


# --- status: Excluded ------------------------------------------------------


def test_status_prints_an_excluded_offering_with_its_reason_and_expected_return():
    policy = _empty_policy()
    reset_at = datetime(2026, 7, 29, 21, 45, 0, tzinfo=timezone.utc)
    report = PlanReport(excluded=("qwencloud-token-plan:glm-5",))
    health = {
        "qwencloud-token-plan:glm-5": OfferingHealth(
            excluded=True, reason="quota_exhausted", bucket="self_healing", reset_at=reset_at
        )
    }

    lines = status_lines(policy=policy, health=health, report=report, now=NOW)

    excluded_line = next(line for line in lines if line.strip().startswith("qwencloud-token-plan:glm-5"))
    assert "quota_exhausted" in excluded_line
    assert "2026-07-29T21:45:00" in excluded_line


def test_status_says_plainly_when_an_excluded_offering_has_no_expected_return():
    policy = _empty_policy()
    report = PlanReport(excluded=("groq:some-model",))
    health = {
        "groq:some-model": OfferingHealth(excluded=True, reason="gateway_error", bucket="self_healing")
    }

    lines = status_lines(policy=policy, health=health, report=report, now=NOW)

    excluded_line = next(line for line in lines if line.strip().startswith("groq:some-model"))
    assert "no expected return" in excluded_line
    # An empty field is never printed for the missing reset time.
    assert excluded_line.strip().endswith("no expected return")


# --- status: the four states never get confused for each other ------------


def test_status_distinguishes_withheld_from_excluded_candidate_and_sunsetting():
    policy = _empty_policy(withheld={"openrouter:qwen/qwen3-coder:free": "vendor retired the free slug"})
    report = PlanReport(
        excluded=("groq:excluded-model",),
        withheld=("openrouter:qwen/qwen3-coder:free",),
        candidates=("provider:candidate-model",),
        sunsetting=("opencode-go:sunsetting-model",),
        admitted=("opencode-go:sunsetting-model",),
        aliases={"opencode-go:sunsetting-model": "claude-sunsetting-model"},
    )
    health = {"groq:excluded-model": OfferingHealth(excluded=True, reason="gateway_error", bucket="self_healing")}

    lines = "\n".join(status_lines(policy=policy, health=health, report=report, now=NOW))

    assert "openrouter:qwen/qwen3-coder:free" in lines
    assert "vendor retired the free slug" in lines
    withheld_section = lines.split("Withheld:")[1].split("Sunsetting:")[0]
    assert "groq:excluded-model" not in withheld_section
    assert "provider:candidate-model" not in withheld_section


def test_status_distinguishes_excluded_from_withheld_candidate_and_sunsetting():
    policy = _empty_policy(withheld={"other:withheld-model": "billing unclear"})
    report = PlanReport(
        excluded=("groq:excluded-model",),
        withheld=("other:withheld-model",),
        candidates=("provider:candidate-model",),
        sunsetting=(),
    )
    health = {"groq:excluded-model": OfferingHealth(excluded=True, reason="gateway_error", bucket="self_healing")}

    lines = "\n".join(status_lines(policy=policy, health=health, report=report, now=NOW))

    excluded_section = lines.split("Excluded (still served")[1].split("Unlisted (")[0]
    assert "groq:excluded-model" in excluded_section
    assert "other:withheld-model" not in excluded_section
    assert "provider:candidate-model" not in excluded_section


def test_status_separates_an_unlisted_offering_from_an_excluded_one():
    """An Excluded Offering is still served; an Unlisted one is gone
    from the file. Reporting them in one list would call a served model
    removed (ADR 0014)."""
    policy = _empty_policy()
    report = PlanReport(
        excluded=("groq:excluded-model",),
        unlisted=("groq:gone-model",),
    )
    health = {
        "groq:excluded-model": OfferingHealth(
            excluded=True, reason="gateway_error", bucket="self_healing"
        ),
        "groq:gone-model": OfferingHealth(
            excluded=True, reason="identifier_gone", bucket="gone"
        ),
    }

    lines = "\n".join(status_lines(policy=policy, health=health, report=report, now=NOW))

    excluded_section = lines.split("Excluded (still served")[1].split("Unlisted (")[0]
    unlisted_section = lines.split("Unlisted (")[1]
    assert "groq:excluded-model" in excluded_section
    assert "groq:gone-model" not in excluded_section
    assert "groq:gone-model" in unlisted_section


def test_status_distinguishes_candidate_from_withheld_excluded_and_sunsetting():
    policy = _empty_policy(withheld={"other:withheld-model": "billing unclear"})
    report = PlanReport(
        excluded=("groq:excluded-model",),
        withheld=("other:withheld-model",),
        candidates=("provider:candidate-model",),
        sunsetting=(),
    )
    health = {}

    lines = "\n".join(status_lines(policy=policy, health=health, report=report, now=NOW))

    candidates_section = lines.split("Awaiting approval (Candidates):")[1]
    assert "provider:candidate-model" in candidates_section
    assert "groq:excluded-model" not in candidates_section
    assert "other:withheld-model" not in candidates_section


def test_status_distinguishes_sunsetting_from_withheld_excluded_and_candidate():
    policy = _empty_policy(withheld={"other:withheld-model": "billing unclear"})
    report = PlanReport(
        excluded=("groq:excluded-model",),
        withheld=("other:withheld-model",),
        candidates=("provider:candidate-model",),
        sunsetting=("opencode-go:sunsetting-model",),
        admitted=("opencode-go:sunsetting-model",),
        aliases={"opencode-go:sunsetting-model": "claude-sunsetting-model"},
    )
    health = {"groq:excluded-model": OfferingHealth(excluded=True, reason="gateway_error", bucket="self_healing")}

    lines = "\n".join(status_lines(policy=policy, health=health, report=report, now=NOW))

    sunsetting_section = lines.split("Sunsetting:")[1].split("Awaiting approval")[0]
    assert "opencode-go:sunsetting-model" in sunsetting_section
    assert "groq:excluded-model" not in sunsetting_section
    assert "other:withheld-model" not in sunsetting_section
    assert "provider:candidate-model" not in sunsetting_section


# --- The Feed's profile picks -----------------------------------------------


def _pick_raw(*, offering_id="cline:anthropic/claude-opus-5", expires_at="2026-07-26T16:30:02.833Z"):
    return {
        "id": "best-coder",
        "display_name": "Best Coding Model",
        "object": "profile",
        "selection": {
            "model_offering_id": offering_id,
            "selected_at": "2026-07-25T16:30:02.833Z",
            "expires_at": expires_at,
        },
    }


def test_the_report_names_each_profile_pick_and_whether_it_is_offered():
    feed = _feed([_pick_raw(offering_id="cline:anthropic/claude-opus-5")])

    picks = profile_picks(feed, admitted=frozenset({"cline:anthropic/claude-opus-5"}), now=NOW)

    assert picks == (
        ProfilePick(
            profile_id="best-coder",
            display_name="Best Coding Model",
            offering_id="cline:anthropic/claude-opus-5",
            offered=True,
        ),
    )


def test_a_profile_pick_not_yet_offered_is_marked_as_such():
    feed = _feed([_pick_raw(offering_id="cline:some/other-model")])

    picks = profile_picks(feed, admitted=frozenset(), now=NOW)

    assert picks[0].offered is False


def test_an_expired_pick_is_treated_as_absent():
    feed = _feed([_pick_raw(expires_at="2026-07-20T00:00:00.000Z")])  # before NOW

    picks = profile_picks(feed, admitted=frozenset(), now=NOW)

    assert picks == ()


def test_a_pick_at_its_expiry_instant_is_treated_as_absent():
    feed = _feed([_pick_raw(expires_at=NOW.isoformat().replace("+00:00", "Z"))])

    picks = profile_picks(feed, admitted=frozenset(), now=NOW)

    assert picks == ()


def test_a_missing_profile_collection_reduces_the_report_and_does_not_fail_the_run():
    feed = _feed([])

    picks = profile_picks(feed, admitted=frozenset(), now=NOW)

    assert picks == ()


def test_a_profile_with_an_unfamiliar_id_still_reports_rather_than_crashing():
    feed = _feed(
        [
            {
                "id": "brand-new-pick-nobody-has-seen",
                "display_name": "A New Pick",
                "selection": {
                    "model_offering_id": "cline:something",
                    "selected_at": "2026-07-25T16:30:02.833Z",
                    "expires_at": "2026-07-26T16:30:02.833Z",
                },
            }
        ]
    )

    picks = profile_picks(feed, admitted=frozenset(), now=NOW)

    assert len(picks) == 1
    assert picks[0].profile_id == "brand-new-pick-nobody-has-seen"


def test_a_profile_with_an_unexpected_field_shape_reduces_the_report_and_does_not_fail_the_run():
    # `selection` is a string, not an object: a shape this tool does not
    # recognise. Must be dropped, never raise.
    feed = _feed([{"id": "odd-shape", "display_name": "Odd", "selection": "not-an-object"}])

    picks = profile_picks(feed, admitted=frozenset(), now=NOW)

    assert picks == ()


def test_a_profile_missing_its_offering_id_reduces_the_report_and_does_not_fail_the_run():
    feed = _feed(
        [
            {
                "id": "no-offering",
                "selection": {
                    "selected_at": "2026-07-25T16:30:02.833Z",
                    "expires_at": "2026-07-26T16:30:02.833Z",
                },
            }
        ]
    )

    picks = profile_picks(feed, admitted=frozenset(), now=NOW)

    assert picks == ()


def test_one_malformed_pick_does_not_hide_the_others_in_the_same_feed():
    feed = _feed(
        [
            {"id": "broken", "selection": "not-an-object"},
            _pick_raw(offering_id="cline:anthropic/claude-opus-5"),
        ]
    )

    picks = profile_picks(feed, admitted=frozenset({"cline:anthropic/claude-opus-5"}), now=NOW)

    assert len(picks) == 1
    assert picks[0].offering_id == "cline:anthropic/claude-opus-5"


def test_the_four_profiles_in_feed_current_all_parse_without_error(load_fixture):
    from litellm_maintainer.feed import parse_feed

    feed = parse_feed(load_fixture("feed-current.json"))

    # Every expires_at in this fixture is in the past relative to
    # today's real date, so an explicit past `now` shows them present,
    # proving the code path runs; a `now` far in the future shows them
    # all expired.
    past = datetime(2026, 7, 25, 16, 31, 0, tzinfo=timezone.utc)
    picks = profile_picks(feed, admitted=frozenset(), now=past)
    assert len(picks) == 4

    far_future = datetime(2030, 1, 1, tzinfo=timezone.utc)
    assert profile_picks(feed, admitted=frozenset(), now=far_future) == ()


def test_status_names_profile_picks_when_a_feed_is_given():
    policy = _empty_policy()
    report = PlanReport(admitted=("cline:anthropic/claude-opus-5",))
    feed = _feed([_pick_raw(offering_id="cline:anthropic/claude-opus-5")])

    lines = "\n".join(status_lines(policy=policy, health={}, report=report, feed=feed, now=NOW))

    assert "Best Coding Model" in lines
    assert "offered" in lines


# --- Redaction ---------------------------------------------------------


def test_a_credential_value_in_a_status_line_is_redacted(tmp_path, capsys):
    env_path = tmp_path / ".env.local"
    env_path.write_text("MY_KEY=sk-super-secret-value-1234567890\n")
    mapping = build_redaction_map(env_path)
    policy = _empty_policy(withheld={"provider:sk-super-secret-value-1234567890": "kept for the test"})
    report = PlanReport(withheld=("provider:sk-super-secret-value-1234567890",))

    print_status(policy=policy, health={}, report=report, feed=None, now=NOW, mapping=mapping, out=sys.stdout)

    captured = capsys.readouterr()
    assert "sk-super-secret-value-1234567890" not in captured.out
    assert "<REDACTED:MY_KEY>" in captured.out


def _section(lines: tuple[str, ...], start_marker: str, end_marker: str) -> str:
    """Return the lines between the line starting `start_marker` and the
    line starting `end_marker`, exclusive of both.

    Line-index based, unlike a naive `"\n".join(lines).split(marker)`:
    "Stale Withheld:" itself contains "Withheld:" as a substring, which
    makes a plain string split misattribute the boundary. Matching
    whole lines by their start avoids that trap.
    """
    start = next(i for i, line in enumerate(lines) if line.startswith(start_marker))
    end = next(i for i, line in enumerate(lines) if i > start and line.startswith(end_marker))
    return "\n".join(lines[start + 1 : end])


def _empty_feed(offerings=(), providers=None) -> Feed:
    return Feed(
        schema_version="1",
        offerings=tuple(offerings),
        providers=providers or {},
        profiles=(),
        notices=(),
        raw={},
    )


def test_a_base_url_from_a_declared_offering_is_redacted_when_it_matches_the_redaction_map():
    # redact() maps whole credential VALUES read from an env file, and
    # applies a regex net for a bare `sk-...` token or a `Bearer ...`
    # header (litellm_maintainer.redact). It does not hold a general
    # "looks like a URL" rule, so a base URL is redacted only when it
    # is itself one of the mapped values (for example, the operator put
    # a private host in the same .env.local the redaction map is built
    # from) or embeds a `sk-...`/`Bearer ...` credential. Demonstrated
    # here rather than assumed.
    mapping = {"https://private-gateway.example.internal": "<REDACTED:BASE_URL>"}
    text = "model uses base_url https://private-gateway.example.internal/v1"

    result = redact(text, mapping)

    assert "private-gateway.example.internal" not in result
    assert "<REDACTED:BASE_URL>" in result


# --- Correction 10: Offered omits Declared Offerings; Withheld omits an ----
# --- entry an earlier gate would also drop --------------------------------
#
# The first group below uses the operator's own real Policy and the real
# `feed-current.json` fixture (both already used the same way by
# `tests/test_quality_and_sunsetting.py`), so a regression here is a
# regression against the operator's actual numbers, not a made-up shape.


@pytest.fixture(scope="module")
def feed_current() -> Feed:
    return load_feed(FEED_CURRENT_PATH)


@pytest.fixture(scope="module")
def operator_policy() -> Policy:
    return load_policy(PINNED_POLICY_PATH)


def test_offered_counts_and_names_declared_offerings_alongside_discovered(
    feed_current, operator_policy
):
    """`Offered` must equal the entry count `generate` writes, computed
    from the same `plan` result so the two numbers cannot drift apart.

    A Declared Offering IS offered (CONTEXT.md, "Declared Offering"): it
    reaches the Generated Config verbatim. Before correction 10, `status`
    counted Discovered Offerings only, so `Offered` undercounted by
    exactly the operator's 10 Declared Offerings (64 instead of 74).

    74 is now 77: +12 for the ChatGPT worker seats added 2026-07-26, also
    Declared Offerings, and -9 for the Qwen Token Plan Offerings the
    operator's Policy now Withholds (personal-tier denial — see
    `PERSONAL_PLAN_DENIED_OFFERING_IDS` in test_translate.py).
    `opencode-go:hy3-preview` (see `HY3_PREVIEW_ALIAS` in
    test_translate.py) does not move this number — against
    `feed-current.json` it is `retired` and `hidden`, so it is Withheld
    from the Generated Config even though the Prober still reaches it
    (see `test_prober.py`).
    """
    result = plan(feed=feed_current, policy=operator_policy, health={}, now=NOW)
    entry_count = len(result.config["model_list"])

    # One Offering can now reach two Aliases: a Client-Facing Variant is a
    # second entry for the same admitted Offering, so entries outnumber
    # admitted ids by exactly the number of variants.
    assert len(result.report.admitted) + len(
        result.report.client_facing_variants
    ) == entry_count
    # Every Declared Offering the fixture holds reaches the file, and a
    # Declared Offering IS offered. Assert that relation, not a total:
    # a total counts the fixture's roster instead of the rule.
    declared_aliases = {d.alias for d in operator_policy.declared}
    assert declared_aliases <= {e["model_name"] for e in result.config["model_list"]}
    # Discovery contributes as well, so the file holds strictly more
    # than the Declared Offerings alone.
    assert entry_count > len(declared_aliases)

    lines = status_lines(policy=operator_policy, health={}, report=result.report, now=NOW)
    # "Offered" counts OFFERINGS, not Aliases. A Client-Facing Variant adds
    # an Alias to the file without adding an Offering, so this line stays
    # below the entry count once any variant exists.
    assert lines[0] == f"Offered: {len(result.report.admitted)}"


def test_offered_marks_a_declared_offering_as_declared_in_the_listing():
    """A Declared Offering behaves differently from a Discovered one — it
    is never a Candidate, never renamed, never repriced, and no Feed
    change can affect it (CONTEXT.md, "Declared Offering"). The listing
    marks it "(Declared)" so a reader can tell the two apart at a glance,
    rather than reading a bare Alias with no clue why it never moves.
    """
    policy = _empty_policy()
    report = PlanReport(admitted=("claude-sonnet-5",), aliases={})

    lines = status_lines(policy=policy, health={}, report=report, now=NOW)

    assert "  claude-sonnet-5 -> (Declared)" in lines


def test_withheld_names_every_feed_published_entry_even_one_an_earlier_gate_would_also_drop(
    feed_current, operator_policy
):
    """The Policy holds 27 Withheld entries (18 plus the nine Qwen Token
    Plan Offerings the personal plan denies — see
    `PERSONAL_PLAN_DENIED_OFFERING_IDS` in test_translate.py); 24 name
    Offerings the Feed still publishes. Before correction 10, `status`
    named only 13, because a Withheld Offering was recorded only when
    the Withheld check was the gate that stopped it — an earlier gate
    (here, the named-list / pricing / baseline filter) dropped
    `opencode-go:mimo-v2-omni` and `opencode-go:mimo-v2-pro` first, so
    neither ever reached the Withheld check.
    """
    result = plan(feed=feed_current, policy=operator_policy, health={}, now=NOW)

    assert "opencode-go:mimo-v2-omni" in result.report.withheld
    assert "opencode-go:mimo-v2-pro" in result.report.withheld
    # The nine Team-only Qwen ids were Withheld lines until 2026-07-26.
    # `plan_edition` filters them by Selection now, so they are correctly
    # absent from the Withheld report: nothing in Policy withholds them.
    for offering_id in PERSONAL_PLAN_DENIED_OFFERING_IDS:
        assert offering_id not in result.report.withheld
        assert offering_id not in operator_policy.withheld
    # Every Withheld line the Feed still publishes is reported, and no
    # other. A count would measure the fixture's Withheld list instead.
    published = {o.id for o in feed_current.offerings}
    assert set(result.report.withheld) == set(operator_policy.withheld) & published

    withheld_section = _section(
        status_lines(policy=operator_policy, health={}, report=result.report, now=NOW),
        "Withheld:",
        "Stale Withheld",
    )
    assert "opencode-go:mimo-v2-omni" in withheld_section
    assert "opencode-go:mimo-v2-pro" in withheld_section
    # Selection filters the nine Team-only Qwen ids now, so the Withheld
    # section must not claim the operator withheld them.
    for offering_id in PERSONAL_PLAN_DENIED_OFFERING_IDS:
        assert offering_id not in withheld_section


def test_a_withheld_entry_naming_an_offering_absent_from_the_feed_is_reported_separately(
    feed_current, operator_policy
):
    """One Withheld Policy line names an Offering `feed-current.json`
    does not publish at all: `openrouter:qwen/qwen3-coder:free`. It is a
    stale Policy line worth pruning (correction 10); the report names it
    in its own group rather than silently dropping it, and never in the
    ordinary Withheld count.

    The operator's Policy holds no such line any more: the two GDM lines
    moved to `declared` (commented out) and the
    `openrouter:qwen/qwen3-coder:free` line was pruned once the Feed
    dropped that Offering, both on 2026-07-26. So this test adds one, to
    assert the grouping rather than a Policy state that keeps changing.
    """
    from dataclasses import replace

    clean = plan(feed=feed_current, policy=operator_policy, health={}, now=NOW)
    assert clean.report.withheld_stale == (), (
        "the operator's Policy should carry no stale Withheld line"
    )

    policy = replace(
        operator_policy,
        withheld={**operator_policy.withheld, "openrouter:gone/model": "dropped by the Feed"},
    )
    result = plan(feed=feed_current, policy=policy, health={}, now=NOW)

    stale_ids = {entry.offering_id for entry in result.report.withheld_stale}
    assert stale_ids == {"openrouter:gone/model"}
    assert stale_ids.isdisjoint(result.report.withheld)

    all_lines = status_lines(policy=operator_policy, health={}, report=result.report, now=NOW)
    withheld_section = _section(all_lines, "Withheld:", "Stale Withheld")
    stale_section = _section(all_lines, "Stale Withheld", "Sunsetting:")
    assert "openrouter:gone/model" not in withheld_section
    assert "openrouter:gone/model" in stale_section


def test_a_stale_withheld_entry_names_an_unknown_provider_differently_from_a_known_one(
    feed_current, operator_policy
):
    """Two shades of stale, kept apart rather than flattened.

    A Withheld line can name a provider the Feed does not carry at all,
    or a Feed provider that simply no longer publishes that one
    Offering. An unknown provider likely means the whole line is
    obsolete; a known provider missing one model may just need the id
    refreshed.

    The unknown-provider case is synthetic here. The operator's Policy
    held two (`private-host:*`) until 2026-07-26 and now holds none, and the
    distinction is a property of the code either way.
    """
    from dataclasses import replace

    assert "openrouter" in feed_current.providers
    assert "no-such-provider" not in feed_current.providers

    policy = replace(
        operator_policy,
        withheld={
            **operator_policy.withheld,
            "no-such-provider:some-model": "provider is not in the Feed at all",
            "openrouter:gone/model": "a Feed provider that no longer lists this model",
        },
    )
    result = plan(feed=feed_current, policy=policy, health={}, now=NOW)
    by_id = {entry.offering_id: entry for entry in result.report.withheld_stale}

    assert by_id["no-such-provider:some-model"].unknown_provider is True
    assert by_id["openrouter:gone/model"].unknown_provider is False


def test_withheld_and_stale_withheld_together_account_for_every_policy_line(
    feed_current, operator_policy
):
    """`Withheld` plus `Stale Withheld` must always add up to every id
    Policy's `withheld` map names — the operator's full "what have I
    withheld" answer, with no entry silently lost between the two
    groups (correction 10: "Withheld is an operator DECISION"). The
    Policy now holds 15 Withheld lines. On 2026-07-26 the nine Qwen Token
    Plan Offerings the Personal edition denies left (filtered by
    `plan_edition` instead), the two GDM lines moved to `declared`
    commented out, and the `openrouter:qwen/qwen3-coder:free` line was
    pruned once the Feed dropped that Offering.
    """
    result = plan(feed=feed_current, policy=operator_policy, health={}, now=NOW)

    accounted = set(result.report.withheld) | {
        entry.offering_id for entry in result.report.withheld_stale
    }
    assert accounted == set(operator_policy.withheld)
    assert set(PERSONAL_PLAN_DENIED_OFFERING_IDS).isdisjoint(operator_policy.withheld)
    # No count here. The property is that the two lists TOGETHER account
    # for every Policy line, which the assertion above states exactly.
    assert operator_policy.withheld, "the fixture Policy withholds nothing"


def test_withheld_report_is_computed_independent_of_the_selection_pipeline():
    """A synthetic, minimal reproduction of the filter-order fault,
    independent of the real fixtures above. An Offering that fails the
    baseline capability filter (no `tool_use`) AND is named in
    `policy.withheld` must still be reported as Withheld: the baseline
    filter is an earlier gate than the Withheld check in the Selection
    loop, so a version that records Withheld only at that check's own
    `continue` misses it entirely.
    """
    from litellm_maintainer.feed import Offering, Provider
    from litellm_maintainer.policy import ProviderRule

    offering = Offering(
        id="acme:no-tool-use",
        provider_id="acme",
        provider_model_id="no-tool-use",
        capabilities=(),  # no tool_use: fails `_passes_baseline`, an earlier gate
        endpoint={"model": "acme/no-tool-use"},
        limits={},
        pricing={"kind": "free"},
        availability={"status": "available"},
        quality={"coding_score": 25},
        policy={"visibility": "listed"},
        raw={},
    )
    feed = _empty_feed(
        offerings=(offering,),
        providers={
            "acme": Provider(
                id="acme", name="Acme", default_base_url=None, authentication={}, raw={}
            )
        },
    )
    policy = _empty_policy(
        providers={"acme": ProviderRule(mode="all")},
        withheld={"acme:no-tool-use": "kept for the test"},
    )

    result = plan(feed=feed, policy=policy, health={}, now=NOW)

    assert result.report.withheld == ("acme:no-tool-use",)
    assert result.report.admitted == ()


# --- Defect 6: a Passthrough Auth quota failure is reported, never silent -----


def test_a_passthrough_auth_quota_failure_is_reported_though_it_never_excludes():
    """Story 33: `reduce` correctly leaves a Passthrough Auth Offering's
    quota failure `excluded=False` (CONTEXT.md, "Passthrough Auth" --
    the failure belongs to one caller, not the Offering). Before this
    fix that meant the failure reached no report section at all: not
    `excluded` (never Excluded), not `withheld` (not a Policy decision),
    not even a notification (`detect_events` fires only on
    `needs_operator` and `gone`). The spec requires it "recorded and
    reported but never Exclude it" -- this proves the "reported" half.
    """
    from litellm_maintainer.policy import DeclaredOffering

    declared = DeclaredOffering(
        alias="claude-chatgpt-team",
        litellm_params={"model": "chatgpt/gpt-5.6-luna"},
        passthrough_auth=True,
    )
    policy = _empty_policy(declared=(declared,))
    feed = _empty_feed()
    health = {
        "claude-chatgpt-team": OfferingHealth(
            excluded=False, reason="quota_exhausted", bucket="self_healing"
        )
    }

    result = plan(feed=feed, policy=policy, health=health, now=NOW)

    assert result.report.passthrough_auth_failures == ("claude-chatgpt-team",)
    assert "claude-chatgpt-team" not in result.report.excluded

    lines = status_lines(policy=policy, health=health, report=result.report, now=NOW)
    section = _section(lines, "Passthrough Auth failures", "Withheld:")
    assert "claude-chatgpt-team" in section
    assert "quota_exhausted" in section


def test_a_passthrough_auth_offering_with_no_failure_is_not_reported():
    """The negative case: a Passthrough Auth Offering that has never
    failed must not appear in `passthrough_auth_failures` -- an empty
    Health State record must not be misread as a recorded failure."""
    from litellm_maintainer.policy import DeclaredOffering

    declared = DeclaredOffering(
        alias="claude-chatgpt-team",
        litellm_params={"model": "chatgpt/gpt-5.6-luna"},
        passthrough_auth=True,
    )
    policy = _empty_policy(declared=(declared,))
    feed = _empty_feed()

    result = plan(feed=feed, policy=policy, health={}, now=NOW)

    assert result.report.passthrough_auth_failures == ()


def test_a_gone_offering_is_reported_with_advice_to_remove_it_from_policy():
    """Story 24: a `gone` Offering (a deprecated or removed identifier)
    never clears by itself, so the report must advise removal from
    Policy, naming the exact Policy key -- spec, "Failure
    classification": "The report advises removal from Policy"."""
    policy = _empty_policy()
    report = PlanReport(excluded=("acme:retired-model",))
    health = {
        "acme:retired-model": OfferingHealth(
            excluded=True, reason="identifier_gone", bucket="gone"
        )
    }

    lines = status_lines(policy=policy, health=health, report=report, now=NOW)

    gone_line = next(line for line in lines if line.strip().startswith("acme:retired-model"))
    assert "acme:retired-model" in gone_line
    assert "remove" in gone_line.lower()
    assert "policy" in gone_line.lower()


# --- status as JSON, reported 2026-07-29 by an agent consumer -------------
#
# `guidance` and `entitlements` both answer JSON. `status` answered text
# only, and the Withheld and Excluded reasons it prints are the data that
# explains a Headroom window governing nothing spendable.


def test_status_json_carries_the_withheld_reason_as_its_own_field():
    from litellm_maintainer.report import status_document

    policy = _empty_policy(
        withheld={"gemini:gemini-2.5-pro": "429 quota exceeded — Pro tier needs Cloud billing"}
    )
    report = PlanReport(withheld=("gemini:gemini-2.5-pro",))

    document = status_document(policy=policy, health={}, report=report, now=NOW)

    entry = document["withheld"][0]
    assert entry["offering_id"] == "gemini:gemini-2.5-pro"
    assert "Cloud billing" in entry["reason"]


def test_status_json_splits_an_excluded_reason_from_its_reset_time():
    # A consumer must never parse the prose line the text view prints.
    from litellm_maintainer.report import status_document

    policy = _empty_policy()
    reset_at = datetime(2026, 7, 29, 21, 45, 0, tzinfo=timezone.utc)
    report = PlanReport(excluded=("qwencloud-token-plan:glm-5",))
    health = {
        "qwencloud-token-plan:glm-5": OfferingHealth(
            excluded=True, reason="quota_exhausted", bucket="self_healing", reset_at=reset_at
        )
    }

    document = status_document(policy=policy, health=health, report=report, now=NOW)

    entry = document["excluded"][0]
    assert entry["reason"] == "quota_exhausted"
    assert entry["bucket"] == "self_healing"
    assert entry["reset_at"].startswith("2026-07-29T21:45:00")
    assert entry["gone"] is False


def test_status_json_and_status_text_agree_on_what_is_offered():
    from litellm_maintainer.report import status_document

    policy = _empty_policy()
    report = PlanReport(admitted=("groq:a", "groq:b"), aliases={"groq:a": "claude-groq-a"})

    document = status_document(policy=policy, health={}, report=report, now=NOW)
    lines = status_lines(policy=policy, health={}, report=report, now=NOW)

    assert len(document["offered"]) == 2
    assert any("Offered: 2" in line for line in lines)
