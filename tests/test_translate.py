"""Tests for `translate` and the remaining `plan` translation rules,
ticket 06.

Assert external behaviour: what litellm `litellm_params` a rule
produces, and which Offerings a selection filter admits. Fixtures are
the frozen `tests/fixtures/feed-audited.json` and `feed-current.json`,
copied and mutated in memory where a real Offering does not already
exercise the rule under test. Both source files are read-only; this
test writes nothing to either.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from litellm_maintainer.feed import load_feed, parse_feed
from litellm_maintainer.plan import plan
from litellm_maintainer.policy import alias_for, load_policy, parse_policy
from litellm_maintainer.translate import ENVELOPE_HANDLER_PREFIX, translate_offering

FIXTURES = Path(__file__).parent / "fixtures"
FEED_AUDITED_PATH = FIXTURES / "feed-audited.json"
FEED_CURRENT_PATH = FIXTURES / "feed-current.json"
EXPECTED_CONFIG_PATH = FIXTURES / "expected-config.yaml"
# Synthetic and committed. Never the operator's own Policy.
PINNED_POLICY_PATH = FIXTURES / "policy-pinned.yaml"

NOW = datetime(2026, 7, 25, tzinfo=timezone.utc)

# The seven renamed Aliases (spec-corrections.md, correction 4). The
# frozen config still uses the left-hand name; a Policy with an empty
# `alias_overrides` now derives the right-hand one instead.
RENAMED_ALIASES = {
    "claude-qwen-token-plan-3.8-max-preview": "claude-qwen-token-plan-qwen3.8-max-preview",
    "claude-qwen-token-plan-3.6-flash": "claude-qwen-token-plan-qwen3.6-flash",
    "claude-openrouter-nemotron-3-ultra-free": "claude-openrouter-nemotron-3-ultra-550b-a55b-free",
    "claude-openrouter-nemotron-3-super-120b-free": (
        "claude-openrouter-nemotron-3-super-120b-a12b-free"
    ),
    "claude-cline-free-nemotron-3-ultra": "claude-cline-free-nemotron-3-ultra-550b-a55b",
    "claude-cline-free-nemotron-3-super-120b": "claude-cline-free-nemotron-3-super-120b-a12b",
    "claude-openrouter-free-router": "claude-openrouter-free",
}

# The seven OpenCode Go Aliases that move from the anthropic-shaped
# prefix to the generic openai/ prefix (ticket 05's difference,
# unaffected by this ticket).
OPENCODE_GO_MOVED_ALIASES = {
    "claude-opencode-go-minimax-m3",
    "claude-opencode-go-minimax-m2.7",
    "claude-opencode-go-minimax-m2.5",
    "claude-opencode-go-qwen3.7-max",
    "claude-opencode-go-qwen3.7-plus",
    "claude-opencode-go-qwen3.6-plus",
    "claude-opencode-go-qwen3.5-plus",
}

# The six Qwen Token Plan Aliases that move from the anthropic-shaped
# prefix to the openai/ prefix. The proxy forwards a client's own
# `Authorization` header, and an anthropic-shaped route leaves that
# header in place, so the provider reads the client's token and answers
# HTTP 401. See `translate.qwencloud_token_plan_openai`.
QWEN_TOKEN_PLAN_MOVED_ALIASES = {
    "claude-qwen-token-plan-qwen3.8-max-preview",
    "claude-qwen-token-plan-qwen3.7-max",
    "claude-qwen-token-plan-qwen3.7-plus",
    "claude-qwen-token-plan-qwen3.6-flash",
    "claude-qwen-token-plan-glm-5.2",
    "claude-qwen-token-plan-deepseek-v4-pro",
}


# The 12 ChatGPT worker-seat Aliases. The operator added these to
# `policy.yaml` on 2026-07-26, as ordinary Declared Offerings: two local
# litellm instances (`127.0.0.1:4011` and `:4012`), each fronting its own
# ChatGPT OAuth session, six models named per seat. They are NOT
# Passthrough Auth — this proxy holds the worker key itself
# (`os.environ/EXAMPLE_CHATGPT_SEAT{1,2}_WORKER_KEY`), so the Prober
# probes them and the smoke check calls them like any other Declared
# Offering.
SEAT_MODELS = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
)
SEAT_ALIASES = {
    f"claude-chatgpt{seat}-{model}" for seat in (1, 2) for model in SEAT_MODELS
}

# The four private-host Aliases (example-private-host.invalid), re-enabled 2026-07-28 once the billing
# terms were confirmed: one fixed-rate plan, fair use, so `flat_rate`. The
# frozen config holds all four commented out, so each is a new Alias here.
# `minimax-m3` and `qwen35-397b-a17b` are new lines; the host's
# `/v1/models` omits both while both answer a completion.
PRIVATE_HOST_ALIASES = {
    "claude-private-host-gemma4-31b-it",
    "claude-private-host-gpt-oss-120b",
    "claude-private-host-minimax-m3",
    "claude-private-host-qwen35-397b-a17b",
}

# The one hand-declared Client-Facing Variant among them: `minimax-m3` is
# the only GDM model at or above `minimum_context_tokens`. Named here, and
# excluded from `derived_variant_aliases` below, so a variant the Generator
# stops DERIVING cannot hide behind a variant the operator DECLARED.
PRIVATE_HOST_VARIANT_ALIASES = {"claude-private-host-minimax-m3[1m]"}

# The six direct `chatgpt/` Declared Offerings the frozen config held.
# The operator retired them on 2026-07-26: every ChatGPT model is now
# reached through a worker seat, so the Policy declares no `chatgpt/`
# entry at all. They are named here, not folded into "anything else is
# fine", so re-adding one has to change this set.
DIRECT_CHATGPT_ALIASES_RETIRED = {f"claude-{model}" for model in SEAT_MODELS}

# The Client-Facing Variants, added 2026-07-26. One `[1m]` Alias per 1M
# model. Each reaches the same Offering with the same wire request as its
# plain sibling: the provider never sees the suffix, and no beta header is
# involved. It exists because Claude Code reads its own context budget out
# of the Alias name, so the plain Alias made it budget 200000 tokens
# against a model that accepts 1000000. See CONTEXT.md, "Client-Facing
# Variant", and ADR 0007.
#
# `claude-haiku-4-5` has no variant: it accepts 200000 tokens, which is
# what litellm's own map already states.
ONE_MILLION_MODELS = ("claude-sonnet-5", "claude-opus-5", "claude-fable-5")
CLIENT_FACING_VARIANT_ALIASES = {f"{alias}[1m]" for alias in ONE_MILLION_MODELS}

# The Generator now derives a variant for every admitted Discovered
# Offering the Feed sizes at or above `minimum_context_tokens`, so the set
# follows the Feed rather than a list here. Named as a predicate, not as
# names: a Feed revision that widens one model changes the count.
VARIANT_SUFFIX = "[1m]"


def derived_variant_aliases(generated) -> set[str]:
    """Every derived Client-Facing Variant Alias in a generated config."""
    return {
        name
        for name in generated
        if name.endswith(VARIANT_SUFFIX)
        and name not in CLIENT_FACING_VARIANT_ALIASES
        and name not in PRIVATE_HOST_VARIANT_ALIASES
    }

# `openrouter:qwen/qwen3-coder:free` was a Withheld line: the vendor
# retired the free slug and a direct call returns 404. The Feed dropped
# the Offering on 2026-07-26, so `doctor` reported the line as stale and
# the operator pruned it. Both pinned fixtures still publish the
# Offering, so planning against either now admits it.
UNWITHHELD_OPENROUTER_ID = "openrouter:qwen/qwen3-coder:free"
UNWITHHELD_OPENROUTER_ALIAS = "claude-openrouter-qwen3-coder-free"

# A second, unrelated fact, discovered while measuring the seat change
# rather than caused by it: the operator's Policy sets
# `opencode-go: mode: all`. Against `feed-audited.json`,
# `opencode-go:hy3-preview` is `available`, `listed`, and scores 58.8 —
# above the quality threshold of 18 — so `mode: all` admits it, even
# though `expected-config.yaml` never wrote it. The same Offering is
# `retired` and `hidden` in `feed-current.json` (see
# `FIVE_RETIRED_HIDDEN_OPENCODE_GO_IDS` in `test_acceptance.py`). Named
# here on its own so no test folds it silently into "anything else is
# fine".
HY3_PREVIEW_ID = "opencode-go:hy3-preview"
HY3_PREVIEW_ALIAS = "claude-opencode-go-hy3-preview"

# The nine Qwen Token Plan Offerings on the TEAM tier, denied to the
# operator's PERSONAL plan. Measured 2026-07-26: each returns HTTP 403
# "Access to model denied. Please make sure you are eligible for using
# the model." They have never worked and never will on this plan.
#
# The operator Withheld all nine by hand until 2026-07-26. The Feed now
# publishes each Offering's editions at
# `pricing.subscription.plan_editions`, so
# `providers.qwencloud-token-plan.plan_edition: personal` filters them by
# Selection instead, and no Withheld line names one. These nine are the
# Team-only roster: still a CLOSED, named list, never "and anything else
# denied is fine". The six other Qwen Offerings the operator does have
# access to are NOT here: they are only out of quota until
# 2026-07-29T21:45Z and must still be admitted.
PERSONAL_PLAN_DENIED_OFFERING_IDS = (
    "qwencloud-token-plan:MiniMax-M2.5",
    "qwencloud-token-plan:deepseek-v3.2",
    "qwencloud-token-plan:deepseek-v4-flash",
    "qwencloud-token-plan:glm-5",
    "qwencloud-token-plan:glm-5.1",
    "qwencloud-token-plan:kimi-k2.5",
    "qwencloud-token-plan:kimi-k2.6",
    "qwencloud-token-plan:kimi-k2.7-code",
    "qwencloud-token-plan:qwen3.6-plus",
)


def personal_plan_denied_aliases(policy) -> set[str]:
    """The Aliases the nine denied Offerings would have used.

    Derived from `PERSONAL_PLAN_DENIED_OFFERING_IDS` through the same
    `alias_for` the Generator uses, rather than a second hand-written
    list that could drift from the first.
    """
    return {alias_for(policy, offering_id) for offering_id in PERSONAL_PLAN_DENIED_OFFERING_IDS}


@pytest.fixture(scope="module")
def feed_audited():
    return load_feed(FEED_AUDITED_PATH)


@pytest.fixture(scope="module")
def feed_current():
    return load_feed(FEED_CURRENT_PATH)


@pytest.fixture(scope="module")
def operator_policy():
    return load_policy(PINNED_POLICY_PATH)


@pytest.fixture(scope="module")
def frozen_config():
    with open(EXPECTED_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _offering(feed, offering_id: str):
    offering = feed.offering(offering_id)
    assert offering is not None, f"fixture offering {offering_id!r} not found"
    return offering


def _raw_model(path: Path, offering_id: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    for model in raw["models"]:
        if model["id"] == offering_id:
            return copy.deepcopy(model)
    raise AssertionError(f"fixture offering {offering_id!r} not found in {path}")


def _raw_provider(path: Path, provider_id: str) -> dict:
    with open(path) as f:
        raw = json.load(f)
    for provider in raw["providers"]:
        if provider["id"] == provider_id:
            return copy.deepcopy(provider)
    raise AssertionError(f"fixture provider {provider_id!r} not found in {path}")


# --- Native-prefix providers ------------------------------------------


@pytest.mark.parametrize(
    "offering_id, expected_prefix",
    [
        ("gemini:gemini-2.5-flash", "gemini/"),
        ("groq:qwen/qwen3.6-27b", "groq/"),
        ("openrouter:poolside/laguna-m.1:free", "openrouter/"),
    ],
)
def test_a_native_prefix_provider_generates_with_its_prefix_and_no_api_base(
    feed_audited, offering_id, expected_prefix
):
    offering = _offering(feed_audited, offering_id)
    provider = feed_audited.providers[offering.provider_id]
    params = translate_offering(offering, provider)
    assert params["model"].startswith(expected_prefix)
    assert "api_base" not in params


def test_groq_native_keeps_the_vendor_path_inside_the_model_id(feed_audited):
    offering = _offering(feed_audited, "groq:qwen/qwen3.6-27b")
    provider = feed_audited.providers["groq"]
    params = translate_offering(offering, provider)
    assert params["model"] == "groq/qwen/qwen3.6-27b"


def test_openrouter_keeps_the_vendor_path_inside_the_model_id_never_the_bare_vendor(
    feed_audited,
):
    """Read `docs/gotchas.md`, "a wrong provider prefix sends your
    traffic to the wrong vendor". `cohere/north-mini-code:free` must
    become `openrouter/cohere/north-mini-code:free`, never
    `cohere/north-mini-code:free` (which litellm would send straight to
    Cohere's own API under the OpenRouter credential).
    """
    offering = _offering(feed_audited, "openrouter:cohere/north-mini-code:free")
    provider = feed_audited.providers["openrouter"]
    params = translate_offering(offering, provider)
    assert params["model"] == "openrouter/cohere/north-mini-code:free"


def test_openrouters_self_referential_free_router_is_not_double_prefixed(feed_audited):
    """`openrouter:openrouter/free` already carries `openrouter/` as its
    own provider_model_id. The rule must not prepend a second one."""
    offering = _offering(feed_audited, "openrouter:openrouter/free")
    provider = feed_audited.providers["openrouter"]
    params = translate_offering(offering, provider)
    assert params["model"] == "openrouter/free"


# --- Identifier normalisation ------------------------------------------


def test_gemini_generates_with_its_collection_prefix_removed(feed_audited):
    offering = _offering(feed_audited, "gemini:gemini-2.5-flash")
    assert offering.endpoint["model"] == "models/gemini-2.5-flash"

    provider = feed_audited.providers["gemini"]
    params = translate_offering(offering, provider)
    assert params["model"] == "gemini/gemini-2.5-flash"


def test_only_gemini_carries_a_collection_prefix_in_the_audited_snapshot(feed_audited):
    """Verify the claim, rather than assume Gemini is the only one.

    No other provider's `endpoint.model` disagrees with its
    `provider_model_id` in the audited snapshot.
    """
    disagreeing_providers = {
        offering.provider_id
        for offering in feed_audited.offerings
        if offering.endpoint.get("model") not in (None, offering.provider_model_id)
    }
    assert disagreeing_providers == {"gemini"}


# --- The pricing filter --------------------------------------------


def test_the_three_free_only_providers_generate_only_free_and_free_tier_offerings(
    feed_audited, operator_policy
):
    result = plan(feed=feed_audited, policy=operator_policy, health={}, now=NOW)
    admitted_ids = set(result.report.admitted)

    free_only_providers = {"openrouter", "cline", "opencode-zen"}
    for provider_id in free_only_providers:
        rule = operator_policy.providers[provider_id]
        assert rule.pricing == ("free",), provider_id

    for offering_id in admitted_ids:
        provider_id = offering_id.partition(":")[0]
        if provider_id in free_only_providers:
            offering = _offering(feed_audited, offering_id)
            assert offering.pricing_kind in ("free", "free_tier"), offering_id


def test_groq_has_no_pricing_filter_and_admits_unknown_priced_offerings(
    feed_audited, operator_policy
):
    """Groq's free tier is an account entitlement (spec, "Selection"):
    the Feed marks its Offerings paid or unknown, never free, because
    it cannot see this account's plan. Policy sets no pricing filter for
    groq, on purpose.
    """
    rule = operator_policy.providers["groq"]
    assert rule.pricing is None

    result = plan(feed=feed_audited, policy=operator_policy, health={}, now=NOW)
    groq_ids = {oid for oid in result.report.admitted if oid.startswith("groq:")}
    assert groq_ids, "expected at least one admitted groq Offering"
    for offering_id in groq_ids:
        offering = _offering(feed_audited, offering_id)
        assert offering.pricing_kind in ("paid", "unknown"), offering_id


def test_a_pricing_filter_on_groq_would_select_nothing(feed_audited):
    """The Feed publishes 8 paid and 7 unknown Groq Offerings and no
    free ones (spec-corrections.md, the Groq worked example). A pricing
    filter of `[free]` or `[free, free_tier]` on groq would admit
    nothing, which is exactly why Policy sets none.
    """
    groq_offerings = feed_audited.offerings_for("groq")
    assert groq_offerings, "expected groq Offerings in the audited snapshot"
    assert all(o.pricing_kind not in ("free", "free_tier") for o in groq_offerings)


def test_groq_generates_its_unknown_priced_offerings_when_they_clear_every_other_gate(
    feed_audited,
):
    """The real Feed's `unknown`-priced groq Offerings all lack
    `tool_use` and a coding score, so none reach admission on their own
    merits. Build a minimal synthetic groq Offering, `unknown`-priced,
    that clears the baseline filter and the quality gate, and assert
    `plan` admits it under a Policy with no pricing filter for groq —
    proving the pricing gate itself, isolated from those other gates.
    """
    synthetic_feed_doc = {
        "schema_version": "test",
        "providers": [
            _raw_provider(FEED_AUDITED_PATH, "groq"),
        ],
        "models": [
            {
                "id": "groq:synthetic-unknown-priced-coder",
                "provider": {"id": "groq", "name": "Groq"},
                "provider_model_id": "synthetic-unknown-priced-coder",
                "endpoint": {
                    "protocol": "openai_chat_completions",
                    "base_url": "https://api.groq.com/openai/v1",
                    "model": "synthetic-unknown-priced-coder",
                },
                "capabilities": ["chat", "tool_use", "coding"],
                "pricing": {"kind": "unknown"},
                "availability": {"status": "available"},
                "quality": {"coding_score": 40.0},
                "policy": {"visibility": "listed"},
            }
        ],
    }
    feed = parse_feed(synthetic_feed_doc)

    policy = parse_policy(
        {
            "providers": {"groq": {"mode": "all"}},
            "quality": {"minimum_coding_score": 18},
            "approved_candidates": [],
            "naming": {
                "alias_prefix": "claude-",
                "provider_labels": {"groq": "groq"},
                "alias_overrides": {},
            },
            "withheld": {},
            "declared": [],
            "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
            "schedule": {
                "enabled": True,
                "interval_minutes": 60,
                "require_proxy": True,
                "maximum_staleness_hours": 24,
            },
            "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
        }
    )

    result = plan(feed=feed, policy=policy, health={}, now=NOW)
    assert "groq:synthetic-unknown-priced-coder" in result.report.admitted

    # The other half: applying a pricing filter to this same synthetic
    # provider would select nothing, since its one Offering is
    # `unknown`-priced and carries no `free`/`free_tier` Offering at all.
    filtered_policy = parse_policy(
        {
            "providers": {"groq": {"mode": "all", "pricing": ["free", "free_tier"]}},
            "quality": {"minimum_coding_score": 18},
            "approved_candidates": [],
            "naming": {
                "alias_prefix": "claude-",
                "provider_labels": {"groq": "groq"},
                "alias_overrides": {},
            },
            "withheld": {},
            "declared": [],
            "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
            "schedule": {
                "enabled": True,
                "interval_minutes": 60,
                "require_proxy": True,
                "maximum_staleness_hours": 24,
            },
            "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
        }
    )
    filtered_result = plan(feed=feed, policy=filtered_policy, health={}, now=NOW)
    assert filtered_result.report.admitted == ()


# --- Envelope routing: data-driven, never provider-named ---------------


def test_an_offering_declaring_the_envelope_key_routes_to_the_unwrapping_handler(
    feed_current,
):
    offering = _offering(feed_current, "cline:nvidia/nemotron-3-ultra-550b-a55b:free")
    assert offering.endpoint["protocol_options"]["response_envelope_key"] == "data"

    provider = feed_current.providers["cline"]
    params = translate_offering(offering, provider)
    assert params["model"] == f"{ENVELOPE_HANDLER_PREFIX}/nvidia/nemotron-3-ultra-550b-a55b:free"
    assert params["api_base"] == "https://api.cline.bot/api/v1"


def test_a_non_cline_offering_given_an_envelope_key_also_routes_to_the_handler(
    feed_current,
):
    """The rule is chosen from the field, never from the provider id.

    Take a real OpenRouter Offering — a provider that does not normally
    carry the envelope key — and add the key in a copied Feed document.
    It must route to the same handler prefix as a Cline Offering does,
    with no code change keyed to "openrouter".
    """
    raw_offering = _raw_model(
        FEED_CURRENT_PATH, "openrouter:nvidia/nemotron-3-ultra-550b-a55b:free"
    )
    assert "protocol_options" not in raw_offering["endpoint"]
    raw_offering["endpoint"]["protocol_options"] = {"response_envelope_key": "data"}

    synthetic_feed_doc = {
        "schema_version": "test",
        "providers": [_raw_provider(FEED_CURRENT_PATH, "openrouter")],
        "models": [raw_offering],
    }
    feed = parse_feed(synthetic_feed_doc)
    offering = feed.offering("openrouter:nvidia/nemotron-3-ultra-550b-a55b:free")
    provider = feed.providers["openrouter"]

    params = translate_offering(offering, provider)
    assert params["model"] == f"{ENVELOPE_HANDLER_PREFIX}/nvidia/nemotron-3-ultra-550b-a55b:free"
    assert params["api_base"] == "https://openrouter.ai/api/v1"
    assert params["api_key"] == "os.environ/OPENROUTER_API_KEY"


def test_a_cline_offering_with_its_envelope_key_removed_does_not_route_to_the_handler(
    feed_current,
):
    """The converse: the mechanism reacts to the field, not the
    provider. Strip the key from a real Cline Offering in a copied Feed
    document, and it falls back to Cline's ordinary rule — the generic
    OpenAI-compatible one — instead of the handler.
    """
    raw_offering = _raw_model(FEED_CURRENT_PATH, "cline:nvidia/nemotron-3-ultra-550b-a55b:free")
    assert raw_offering["endpoint"]["protocol_options"]["response_envelope_key"] == "data"
    del raw_offering["endpoint"]["protocol_options"]

    synthetic_feed_doc = {
        "schema_version": "test",
        "providers": [_raw_provider(FEED_CURRENT_PATH, "cline")],
        "models": [raw_offering],
    }
    feed = parse_feed(synthetic_feed_doc)
    offering = feed.offering("cline:nvidia/nemotron-3-ultra-550b-a55b:free")
    provider = feed.providers["cline"]

    params = translate_offering(offering, provider)
    assert not params["model"].startswith(f"{ENVELOPE_HANDLER_PREFIX}/")
    assert params["model"] == "openai/nvidia/nemotron-3-ultra-550b-a55b:free"


def test_envelope_routing_against_the_audited_snapshot_has_no_data_to_act_on(feed_audited):
    """The audited snapshot (`feed-audited.json`) predates the Feed's
    envelope-key revision: it carries the key on zero Offerings (see the
    spec's 2026-07-25 amendment table). Cline's Offerings in this
    snapshot therefore translate through the generic OpenAI-compatible
    rule, not the handler — a real, honestly-reported gap in this one
    fixture's data, not a bug in the mechanism. See
    `test_an_offering_declaring_the_envelope_key_routes_to_the_unwrapping_handler`
    and its neighbours above for the mechanism proven against
    `feed-current.json`, which does carry the key.
    """
    envelope_bearing = [
        o for o in feed_audited.offerings if o.endpoint.get("protocol_options")
    ]
    assert envelope_bearing == []

    offering = _offering(feed_audited, "cline:nvidia/nemotron-3-ultra-550b-a55b:free")
    provider = feed_audited.providers["cline"]
    params = translate_offering(offering, provider)
    assert params["model"] == "openai/nvidia/nemotron-3-ultra-550b-a55b:free"


def test_the_custom_provider_map_entry_appears_only_when_an_entry_uses_the_handler(
    feed_current, feed_audited, operator_policy
):
    current_result = plan(feed=feed_current, policy=operator_policy, health={}, now=NOW)
    assert current_result.config["litellm_settings"]["custom_provider_map"] == [
        {"provider": ENVELOPE_HANDLER_PREFIX, "custom_handler": "cline_provider.cline_llm"}
    ]

    # The audited snapshot declares the key on no Offering, but the
    # operator's Policy declares it for Cline, so the handler is still in
    # use and the map is still emitted.
    audited_result = plan(feed=feed_audited, policy=operator_policy, health={}, now=NOW)
    assert audited_result.config["litellm_settings"]["custom_provider_map"] == [
        {"provider": ENVELOPE_HANDLER_PREFIX, "custom_handler": "cline_provider.cline_llm"}
    ]

    # Take the operator's declaration away and nothing uses the handler,
    # so the map disappears. `operator_policy` carries
    # `proxy_settings.litellm_settings` (master_key, drop_params,
    # callbacks), so `litellm_settings` itself is always present: the rule
    # under test is narrower than its presence.
    no_envelope = dataclasses.replace(
        operator_policy,
        providers={
            provider_id: dataclasses.replace(rule, response_envelope_key=None)
            for provider_id, rule in operator_policy.providers.items()
        },
    )
    bare_result = plan(feed=feed_audited, policy=no_envelope, health={}, now=NOW)
    assert "custom_provider_map" not in bare_result.config["litellm_settings"]


# --- A Policy translation override -------------------------------------


def test_a_policy_translation_override_replaces_the_translation_for_one_offering(
    feed_audited,
):
    offering_id = "openrouter:cohere/north-mini-code:free"
    synthetic_policy_raw = {
        "providers": {"openrouter": {"mode": "all", "pricing": ["free"]}},
        "quality": {"minimum_coding_score": 0},
        "approved_candidates": [],
        "naming": {
            "alias_prefix": "claude-",
            "provider_labels": {"openrouter": "openrouter"},
            "alias_overrides": {},
        },
        "withheld": {},
        "declared": [],
        "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
        "schedule": {
            "enabled": True,
            "interval_minutes": 60,
            "require_proxy": True,
            "maximum_staleness_hours": 24,
        },
        "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
        "translation_overrides": {
            offering_id: {"api_base": "https://overridden.example/v1"},
        },
    }
    policy = parse_policy(synthetic_policy_raw)
    assert policy.translation_overrides[offering_id] == {
        "api_base": "https://overridden.example/v1"
    }

    result = plan(feed=feed_audited, policy=policy, health={}, now=NOW)
    entries = {e["model_name"]: e["litellm_params"] for e in result.config["model_list"]}
    overridden = entries["claude-openrouter-north-mini-code-free"]
    assert overridden["model"] == "openrouter/cohere/north-mini-code:free"
    assert overridden["api_base"] == "https://overridden.example/v1"


def test_translation_overrides_defaults_to_empty_when_the_policy_omits_it(operator_policy):
    assert operator_policy.translation_overrides == {}


# --- Declared Offerings --------------------------------------------


def test_declared_offerings_pass_through_verbatim(operator_policy, feed_audited):
    result = plan(feed=feed_audited, policy=operator_policy, health={}, now=NOW)
    entries = {e["model_name"]: e for e in result.config["model_list"]}

    # Do not pin a total here. The count follows from how many Declared
    # Offerings the fixture holds, so it measures the fixture and not
    # the rule this test is about.
    assert operator_policy.declared, "the fixture Policy declares nothing"
    for declared in operator_policy.declared:
        entry = entries[declared.alias]
        # The whole `litellm_params` mapping, not only `model`: a
        # Declared Offering must be immune to every layer, and a
        # shallow check would miss a rewritten `api_base` or `api_key`.
        assert entry["litellm_params"] == declared.litellm_params
        if declared.model_info is not None:
            assert entry["model_info"] == declared.model_info
        else:
            assert "model_info" not in entry


def test_the_twelve_chatgpt_worker_seats_pass_through_with_the_worker_key_unrewritten(
    operator_policy, feed_audited
):
    """The 12 seat Aliases (SEAT_ALIASES) are Declared Offerings, not
    Passthrough Auth: this proxy holds the worker key itself
    (`policy.yaml`'s comment above the seat entries). Every field must
    reach the Generated Config unrewritten: the `openai/` prefix (not
    `chatgpt/`, which would silently misroute to the real ChatGPT
    backend — see `docs/gotchas.md`), the worker's own `api_base`, and
    the unresolved `os.environ/...` key reference.
    """
    result = plan(feed=feed_audited, policy=operator_policy, health={}, now=NOW)
    entries = {e["model_name"]: e for e in result.config["model_list"]}

    seats = {
        d.alias
        for d in operator_policy.declared
        if d.litellm_params.get("api_key", "").endswith("_WORKER_KEY")
    }
    assert len(seats) >= 2, "the fixture must declare at least two seats"
    assert seats <= set(entries)

    declared_by_alias = {d.alias: d for d in operator_policy.declared}
    ports = set()
    for alias in sorted(seats):
        declared = declared_by_alias[alias]
        params = entries[alias]["litellm_params"]

        # The `openai/` prefix, never `chatgpt/`. A `chatgpt/` prefix
        # would route to the real ChatGPT backend and skip the worker
        # entirely. See docs/gotchas.md.
        assert params["model"].startswith("openai/"), alias
        # The worker's own api_base, and the key reference UNRESOLVED.
        assert params["api_base"] == declared.litellm_params["api_base"], alias
        assert params["api_key"] == declared.litellm_params["api_key"], alias
        assert params["api_key"].startswith("os.environ/"), alias
        ports.add(params["api_base"])

    # Each seat is its own worker on its own port. One shared api_base
    # would send both seats' traffic to one subscription.
    assert len(ports) == len(seats)


# --- The whole-config test ---------------------------------------------


# `test_the_operator_policy_reproduces_78_aliases_with_four_differences_and_four_net_additions` stood here (RETIRED).
#
# It compared generated output to `fixtures/expected-config.yaml`, the
# proxy the operator built and verified BY HAND on 2026-07-25. The
# Policy that produced that file was never committed and no surviving
# copy reproduces it. The live Policy matches none of its 78 Aliases,
# because it now sets `alias_prefix: ""` and `alias_separator: "--"`.
#
# The test therefore demanded a superseded proxy from a Policy that no
# longer exists. See the longer note in test_acceptance.py.

# `test_no_alias_translates_to_a_different_model_than_the_frozen_config` stood here (RETIRED).
#
# It compared generated output to `fixtures/expected-config.yaml`, the
# proxy the operator built and verified BY HAND on 2026-07-25. The
# Policy that produced that file was never committed and no surviving
# copy reproduces it. The live Policy matches none of its 78 Aliases,
# because it now sets `alias_prefix: ""` and `alias_separator: "--"`.
#
# The test therefore demanded a superseded proxy from a Policy that no
# longer exists. See the longer note in test_acceptance.py.

# --- Envelope routing: the operator declares what the Feed omits -------


def test_a_policy_declared_envelope_key_routes_an_offering_the_feed_says_nothing_about():
    """A provider can wrap responses without the Feed declaring it.

    The audited snapshot carries `response_envelope_key` on no Offering,
    so a Cline Offering there translates as generic openai-compatible and
    every SUCCESS on it fails with "no 'choices'". The operator's
    `providers.cline.response_envelope_key` states the wrapper, and the
    same Offering then routes to the unwrapping handler.
    """
    raw_offering = _raw_model(FEED_AUDITED_PATH, "cline:nvidia/nemotron-3-ultra-550b-a55b:free")
    assert "protocol_options" not in raw_offering["endpoint"]

    feed = parse_feed(
        {
            "schema_version": "test",
            "providers": [_raw_provider(FEED_AUDITED_PATH, "cline")],
            "models": [raw_offering],
        }
    )
    offering = feed.offering("cline:nvidia/nemotron-3-ultra-550b-a55b:free")
    provider = feed.providers["cline"]

    without = translate_offering(offering, provider)
    assert without["model"].startswith("openai/")

    with_key = translate_offering(offering, provider, policy_envelope_key="data")
    assert with_key["model"] == f"{ENVELOPE_HANDLER_PREFIX}/nvidia/nemotron-3-ultra-550b-a55b:free"
    assert with_key["api_base"] == "https://api.cline.bot/api/v1"


def test_the_feeds_own_envelope_key_wins_over_the_operators(feed_current):
    """The Feed is the authority; Policy states only what it omits."""
    offering = _offering(feed_current, "cline:nvidia/nemotron-3-ultra-550b-a55b:free")
    assert offering.endpoint["protocol_options"]["response_envelope_key"] == "data"

    provider = feed_current.providers["cline"]
    params = translate_offering(offering, provider, policy_envelope_key="somethingelse")
    assert params["model"].startswith(f"{ENVELOPE_HANDLER_PREFIX}/")


def test_a_policy_envelope_key_for_one_provider_does_not_route_another(feed_current):
    """The key is passed per provider, so it cannot leak across them."""
    offering = _offering(feed_current, "groq:qwen/qwen3.6-27b")
    provider = feed_current.providers["groq"]
    params = translate_offering(offering, provider, policy_envelope_key=None)
    assert params["model"].startswith("groq/")


def _policy_with_providers(providers: dict) -> dict:
    """A valid Policy dict naming every required top-level key."""
    return {
        "providers": providers,
        "quality": {"minimum_coding_score": 20},
        "approved_candidates": [],
        "naming": {"alias_prefix": "claude-", "provider_labels": {}, "alias_overrides": {}},
        "withheld": {},
        "declared": [],
        "pacing": {"default": {"concurrency": 2, "minimum_interval_seconds": 5}},
        "schedule": {
            "enabled": True,
            "interval_minutes": 60,
            "require_proxy": True,
            "maximum_staleness_hours": 24,
        },
        "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
    }


def test_policy_parses_and_reports_a_response_envelope_key():
    policy = parse_policy(
        _policy_with_providers({"cline": {"mode": "all", "response_envelope_key": "data"}})
    )
    assert policy.providers["cline"].response_envelope_key == "data"


def test_policy_defaults_the_response_envelope_key_to_none():
    policy = parse_policy(_policy_with_providers({"groq": {"mode": "all"}}))
    assert policy.providers["groq"].response_envelope_key is None


def test_an_empty_response_envelope_key_is_rejected():
    from litellm_maintainer.policy import PolicyError

    with pytest.raises(PolicyError, match="response_envelope_key"):
        parse_policy(
            _policy_with_providers({"cline": {"mode": "all", "response_envelope_key": ""}})
        )
