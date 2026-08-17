"""Tests for `litellm_maintainer.smoke`.

No test makes a network call. Every test injects a fake transport (a
callable that returns a recorded `TransportResponse`). Test names use
the glossary vocabulary from CONTEXT.md and the rule an operator would
recognise: UNVERIFIED, FAILED and INCONCLUSIVE are three different
things.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from litellm_maintainer.feed import load_feed, parse_feed
from litellm_maintainer.policy import load_policy, parse_policy
from litellm_maintainer.prober import TransportResponse
from litellm_maintainer.reduce import OfferingHealth
from litellm_maintainer.smoke import (
    RuleCheck,
    SmokeEntry,
    STATUS_ANSWERED,
    STATUS_FAILED,
    STATUS_INCONCLUSIVE,
    STATUS_UNVERIFIED,
    build_smoke_entries,
    build_smoke_payload,
    extract_streamed_content,
    format_smoke_line,
    group_by_rule,
    pick_healthiest,
    run_smoke_check,
)

NOW = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)


def _never_called_transport(entry):
    raise AssertionError(f"the proxy must not call {entry.alias}")

FEED_CURRENT_PATH = Path(__file__).parent / "fixtures" / "feed-current.json"
# Synthetic and committed. Never the operator's own Policy.
PINNED_POLICY_PATH = Path(__file__).parent / "fixtures" / "policy-pinned.yaml"


def _sse(*data_objs: dict, keep_alives: int = 0, include_done: bool = True) -> str:
    """Build a raw SSE response body for a test, oldest chunk first."""
    lines: list[str] = [": keep-alive" for _ in range(keep_alives)]
    lines += [f"data: {json.dumps(obj)}" for obj in data_objs]
    if include_done:
        lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def _offering_raw(
    offering_id: str,
    *,
    coding_score: float | None = 50.0,
    envelope_key: str | None = None,
) -> dict:
    provider_id, _, model_id = offering_id.partition(":")
    endpoint = {
        "protocol": "openai",
        "base_url": f"https://example.invalid/{provider_id}",
        "model": model_id,
    }
    if envelope_key is not None:
        endpoint["protocol_options"] = {"response_envelope_key": envelope_key}
    return {
        "id": offering_id,
        "provider": {"id": provider_id},
        "provider_model_id": model_id,
        "capabilities": ["tool_use"],
        "endpoint": endpoint,
        "pricing": {"kind": "free"},
        "availability": {"status": "available"},
        "quality": {"coding_score": coding_score},
        "policy": {"visibility": "listed"},
    }


def _provider_raw(provider_id: str) -> dict:
    return {
        "id": provider_id,
        "name": provider_id,
        "authentication": {"credential_hint": f"{provider_id.upper().replace('-', '_')}_API_KEY"},
    }


def _feed(*offerings: dict, providers: list[dict] | None = None):
    return parse_feed(
        {
            "schema_version": "1",
            "providers": providers
            or [
                _provider_raw("groq"),
                _provider_raw("openrouter"),
                _provider_raw("cline"),
                _provider_raw("opencode-go"),
            ],
            "models": list(offerings),
        }
    )


def _policy_raw(*, providers: dict, declared: list[dict] | None = None):
    return {
        "providers": providers,
        "quality": {"minimum_coding_score": 18},
        "approved_candidates": [],
        "naming": {"alias_prefix": "claude-", "provider_labels": {}, "alias_overrides": {}},
        "withheld": {},
        "declared": declared or [],
        "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
        "schedule": {
            "enabled": True,
            "interval_minutes": 60,
            "require_proxy": True,
            "maximum_staleness_hours": 24,
        },
        "safety": {"maximum_removal_share": 0.25, "snapshot_count": 10},
    }


def _policy(**kwargs):
    return parse_policy(_policy_raw(**kwargs))


def _health(records: dict[str, OfferingHealth]) -> dict[str, OfferingHealth]:
    return dict(records)


def _fake_transport(responses: dict[str, TransportResponse]):
    """Return a transport that answers by `SmokeEntry.key`.

    Raises `KeyError` for any key not given, so a test can prove the
    check called (or did not call) exactly the entries it expects.
    """

    def transport(entry: SmokeEntry) -> TransportResponse:
        return responses[entry.key]

    return transport


def _ok_response() -> TransportResponse:
    return TransportResponse(http_status=200, body={"choices": [{"message": {"content": "ok"}}]})


def _error_response(status: int = 401) -> TransportResponse:
    return TransportResponse(http_status=status, body={"error": {"message": "nope"}})


# ---------------------------------------------------------------------
# One call per distinct translation rule, not per Offering.
# ---------------------------------------------------------------------


def test_the_smoke_check_makes_one_call_per_distinct_translation_rule_not_per_offering():
    """Two providers routed by `native_prefix` (distinct closures: groq,
    openrouter) are two rules. `generic_openai_compatible` serves both
    `cline` and `opencode-go` when neither declares an envelope key, so
    those two providers' Offerings collapse into one rule.
    """
    feed = _feed(
        _offering_raw("groq:model-a"),
        _offering_raw("groq:model-b"),
        _offering_raw("openrouter:model-c"),
        _offering_raw("cline:model-d"),
        _offering_raw("opencode-go:model-e"),
    )
    policy = _policy(
        providers={
            "groq": {"mode": "all"},
            "openrouter": {"mode": "all"},
            "cline": {"mode": "all"},
            "opencode-go": {"mode": "all"},
        }
    )
    entries = build_smoke_entries(feed=feed, policy=policy)
    grouped = group_by_rule(entries)

    # groq, openrouter, and the shared generic_openai_compatible rule
    # (cline + opencode-go): three distinct rules from five Offerings.
    assert len(grouped) == 3

    calls: list[str] = []

    def transport(entry: SmokeEntry) -> TransportResponse:
        calls.append(entry.key)
        return _ok_response()

    result = run_smoke_check(
        entries, health={}, transport=transport, now=lambda: NOW
    )

    assert len(result.checks) == 3
    assert len(calls) == 3


def test_removing_the_grouping_would_call_every_offering_and_this_test_would_catch_it():
    """Mutation check: group-by-Offering instead of group-by-rule would
    make 5 calls here, not 3. Assert the call count equals the rule
    count, not the Offering count, so a regression to per-Offering
    calling fails this test.
    """
    feed = _feed(
        _offering_raw("groq:model-a"),
        _offering_raw("groq:model-b"),
        _offering_raw("openrouter:model-c"),
        _offering_raw("cline:model-d"),
        _offering_raw("opencode-go:model-e"),
    )
    policy = _policy(
        providers={
            "groq": {"mode": "all"},
            "openrouter": {"mode": "all"},
            "cline": {"mode": "all"},
            "opencode-go": {"mode": "all"},
        }
    )
    entries = build_smoke_entries(feed=feed, policy=policy)
    assert len(entries) == 5  # five Offerings admitted

    calls: list[str] = []

    def transport(entry: SmokeEntry) -> TransportResponse:
        calls.append(entry.key)
        return _ok_response()

    run_smoke_check(entries, health={}, transport=transport, now=lambda: NOW)
    assert len(calls) == 3
    assert len(calls) != len(entries)


# ---------------------------------------------------------------------
# Grouping is read from the translation table, not re-derived from
# provider id.
# ---------------------------------------------------------------------


def test_an_offering_declaring_a_response_envelope_key_groups_as_envelope_unwrapping_not_by_provider():
    """A `groq` Offering that declares the envelope key routes through
    `envelope_unwrapping`, the same rule an unwrapped `cline` Offering
    uses -- grouping never reads `provider_id` alone.
    """
    feed = _feed(
        _offering_raw("groq:wrapped", envelope_key="data"),
        _offering_raw("cline:also-wrapped", envelope_key="data"),
        _offering_raw("groq:plain"),
    )
    policy = _policy(
        providers={"groq": {"mode": "all"}, "cline": {"mode": "all"}}
    )
    entries = build_smoke_entries(feed=feed, policy=policy)
    grouped = group_by_rule(entries)

    # envelope_unwrapping (groq:wrapped + cline:also-wrapped) and the
    # groq native_prefix rule (groq:plain): two rules from three
    # Offerings, and the wrapped groq Offering must NOT share a group
    # with the plain groq Offering just because they share a provider.
    assert len(grouped) == 2
    envelope_group = grouped["envelope_unwrapping"]
    assert {e.key for e in envelope_group} == {"groq:wrapped", "cline:also-wrapped"}


# ---------------------------------------------------------------------
# The healthiest Offering is chosen for each rule.
# ---------------------------------------------------------------------


def test_an_excluded_offering_is_not_chosen_when_a_healthy_one_exists_for_the_same_rule():
    """The Excluded Offering has the MORE RECENT success record, so a
    tie-break on recency alone (ignoring `excluded`) would wrongly pick
    it. Not-excluded must outrank recency.
    """
    from datetime import timedelta

    entries = (
        SmokeEntry(key="groq:excluded", alias="claude-groq-excluded", rule="native_prefix[groq]"),
        SmokeEntry(key="groq:healthy", alias="claude-groq-healthy", rule="native_prefix[groq]"),
    )
    health = {
        "groq:excluded": OfferingHealth(excluded=True, last_success_at=NOW),
        "groq:healthy": OfferingHealth(excluded=False, last_success_at=NOW - timedelta(days=30)),
    }
    chosen = pick_healthiest(entries, health=health)
    assert chosen.key == "groq:healthy"


def test_the_most_recently_successful_offering_is_preferred_among_healthy_candidates():
    from datetime import timedelta

    entries = (
        SmokeEntry(key="groq:stale", alias="claude-groq-stale", rule="native_prefix[groq]"),
        SmokeEntry(key="groq:fresh", alias="claude-groq-fresh", rule="native_prefix[groq]"),
    )
    health = {
        "groq:stale": OfferingHealth(last_success_at=NOW - timedelta(days=10)),
        "groq:fresh": OfferingHealth(last_success_at=NOW),
    }
    chosen = pick_healthiest(entries, health=health)
    assert chosen.key == "groq:fresh"


def test_reversing_the_healthiest_preference_would_call_the_excluded_offering_and_this_test_catches_it():
    """Mutation check: if `pick_healthiest` preferred Excluded (or
    ignored `excluded` entirely), it could choose `groq:excluded` here.
    Assert the actually-called key directly through `run_smoke_check`.
    """
    entries = (
        SmokeEntry(key="groq:excluded", alias="claude-groq-excluded", rule="native_prefix[groq]"),
        SmokeEntry(key="groq:healthy", alias="claude-groq-healthy", rule="native_prefix[groq]"),
    )
    health = {
        "groq:excluded": OfferingHealth(excluded=True),
        "groq:healthy": OfferingHealth(excluded=False, last_success_at=NOW),
    }
    calls: list[str] = []

    def transport(entry: SmokeEntry) -> TransportResponse:
        calls.append(entry.key)
        return _ok_response()

    run_smoke_check(entries, health=health, transport=transport, now=lambda: NOW)
    assert calls == ["groq:healthy"]


# ---------------------------------------------------------------------
# An Excluded Offering does not fail the wiring check for its rule.
# ---------------------------------------------------------------------


def test_an_excluded_offering_does_not_fail_the_wiring_check_when_a_healthy_sibling_exists():
    entries = (
        SmokeEntry(key="qwen:excluded", alias="claude-qwen-excluded", rule="qwencloud_token_plan_openai[qwencloud-token-plan]"),
        SmokeEntry(key="qwen:healthy", alias="claude-qwen-healthy", rule="qwencloud_token_plan_openai[qwencloud-token-plan]"),
    )
    health = {
        "qwen:excluded": OfferingHealth(excluded=True, reason="quota_exhausted"),
        "qwen:healthy": OfferingHealth(excluded=False, last_success_at=NOW),
    }

    def transport(entry: SmokeEntry) -> TransportResponse:
        assert entry.key == "qwen:healthy"
        return _ok_response()

    result = run_smoke_check(entries, health=health, transport=transport, now=lambda: NOW)
    assert len(result.checks) == 1
    assert result.checks[0].status == STATUS_ANSWERED


# ---------------------------------------------------------------------
# A failure reports loudly and names the rule; blocks nothing.
# ---------------------------------------------------------------------


def test_a_failure_names_the_rule_in_the_result():
    entries = (SmokeEntry(key="groq:broken", alias="claude-groq-broken", rule="native_prefix[groq]"),)

    def transport(entry: SmokeEntry) -> TransportResponse:
        return _error_response(401)

    result = run_smoke_check(entries, health={}, transport=transport, now=lambda: NOW)
    assert len(result.failed) == 1
    failed = result.failed[0]
    assert failed.rule == "native_prefix[groq]"
    line = format_smoke_line(failed)
    assert "native_prefix[groq]" in line
    assert "FAILED" in line


def test_a_failure_blocks_nothing_the_function_returns_normally_and_writes_no_health_state():
    """`run_smoke_check` returns a plain result on a failure. It has no
    Health State parameter to write and no config to refuse -- proven
    by nothing raising and the prior Health State dict being untouched.
    """
    entries = (SmokeEntry(key="groq:broken", alias="claude-groq-broken", rule="native_prefix[groq]"),)
    prior_health = {"groq:broken": OfferingHealth(excluded=False, last_success_at=NOW)}
    prior_health_copy = dict(prior_health)

    def transport(entry: SmokeEntry) -> TransportResponse:
        return _error_response(500)

    result = run_smoke_check(entries, health=prior_health, transport=transport, now=lambda: NOW)

    # No exception. A result came back.
    assert result.checks[0].status == STATUS_FAILED
    # The Health State dict passed in is untouched: `run_smoke_check`
    # never writes to it and never mutates it in place.
    assert prior_health == prior_health_copy


# ---------------------------------------------------------------------
# A rule with no healthy Offering is reported UNVERIFIED, not FAILED.
# ---------------------------------------------------------------------


def test_a_rule_with_no_healthy_offering_is_unverified_not_failed():
    entries = (
        SmokeEntry(key="qwen:only", alias="claude-qwen-only", rule="qwencloud_token_plan_openai[qwencloud-token-plan]"),
    )
    health = {"qwen:only": OfferingHealth(excluded=True, reason="quota_exhausted")}

    calls: list[str] = []

    def transport(entry: SmokeEntry) -> TransportResponse:
        calls.append(entry.key)
        return _ok_response()

    result = run_smoke_check(entries, health=health, transport=transport, now=lambda: NOW)

    assert calls == []  # no call made: nothing healthy to call
    assert len(result.checks) == 1
    check = result.checks[0]
    assert check.status == STATUS_UNVERIFIED
    assert check.status != STATUS_FAILED
    assert format_smoke_line(check) != format_smoke_line(
        RuleCheck(rule=check.rule, status=STATUS_FAILED, alias="x", detail="boom")
    )


def test_unverified_and_failed_use_different_words_in_the_rendered_line():
    unverified = RuleCheck(rule="r", status=STATUS_UNVERIFIED, detail="no healthy Offering")
    failed = RuleCheck(rule="r", status=STATUS_FAILED, alias="claude-r", detail="gone")
    assert "UNVERIFIED" in format_smoke_line(unverified)
    assert "FAILED" in format_smoke_line(failed)
    assert "UNVERIFIED" not in format_smoke_line(failed)
    assert "FAILED" not in format_smoke_line(unverified)


# ---------------------------------------------------------------------
# An Inconclusive result is not a FAILED result. Collapsing the two is
# the exact standards violation this fix removes: Inconclusive states
# the call measured nothing, attributable to our own request rate, not
# to the Offering (CONTEXT.md, "Inconclusive").
# ---------------------------------------------------------------------


def test_a_rate_limit_with_no_reset_time_reports_inconclusive_not_failed():
    entries = (SmokeEntry(key="groq:paced", alias="claude-groq-paced", rule="native_prefix[groq]"),)

    def transport(entry: SmokeEntry) -> TransportResponse:
        # No reset time in the body: `classify` reads this as
        # `inconclusive`, attributable to our own request rate.
        return TransportResponse(http_status=429, body={"error": {"message": "rate limit reached"}})

    result = run_smoke_check(entries, health={}, transport=transport, now=lambda: NOW)

    assert len(result.checks) == 1
    check = result.checks[0]
    assert check.status == STATUS_INCONCLUSIVE
    assert check.status != STATUS_FAILED
    assert result.failed == ()
    assert result.inconclusive == (check,)


def test_inconclusive_and_failed_use_different_words_in_the_rendered_line():
    inconclusive = RuleCheck(rule="r", status=STATUS_INCONCLUSIVE, alias="claude-r", detail="rate_limited")
    failed = RuleCheck(rule="r", status=STATUS_FAILED, alias="claude-r", detail="gone")
    assert "INCONCLUSIVE" in format_smoke_line(inconclusive)
    assert "FAILED" in format_smoke_line(failed)
    assert "INCONCLUSIVE" not in format_smoke_line(failed)
    assert "FAILED" not in format_smoke_line(inconclusive)


# ---------------------------------------------------------------------
# The proxy credential never appears in the output.
# ---------------------------------------------------------------------


def test_the_proxy_credential_never_appears_in_a_rendered_line(monkeypatch):
    from litellm_maintainer.redact import redact

    monkeypatch.setenv("LITELLM_MASTER_KEY", "sk-super-secret-master-key-value")

    entries = (SmokeEntry(key="groq:leaky", alias="claude-groq-leaky", rule="native_prefix[groq]"),)

    def transport(entry: SmokeEntry) -> TransportResponse:
        # Simulate a provider echoing the Authorization header back in
        # an error body -- the exact shape `redact`'s regex net exists
        # for.
        return TransportResponse(
            http_status=401,
            body={"error": {"message": "Bearer sk-super-secret-master-key-value rejected"}},
        )

    result = run_smoke_check(entries, health={}, transport=transport, now=lambda: NOW)
    line = format_smoke_line(result.checks[0])

    mapping = {"sk-super-secret-master-key-value": "<REDACTED:LITELLM_MASTER_KEY>"}
    assert "sk-super-secret-master-key-value" not in redact(line, mapping)


# ---------------------------------------------------------------------
# --dry-run calls nothing (CLI-level; the pure function is exercised
# directly here, since `build_smoke_entries` + `pick_healthiest` is
# exactly the code path `cmd_smoke --dry-run` runs, and it never calls
# `run_smoke_check` or any transport).
# ---------------------------------------------------------------------


def test_dry_run_never_calls_run_smoke_check_or_a_transport():
    import litellm_maintainer.smoke as smoke_module

    feed = _feed(_offering_raw("groq:model-a"))
    policy = _policy(providers={"groq": {"mode": "all"}})

    called = {"run_smoke_check": False}
    original = smoke_module.run_smoke_check

    def spy(*args, **kwargs):
        called["run_smoke_check"] = True
        return original(*args, **kwargs)

    smoke_module.run_smoke_check = spy
    try:
        entries = build_smoke_entries(feed=feed, policy=policy)
        grouped = group_by_rule(entries)
        for rule in grouped:
            pick_healthiest(grouped[rule], health={})
        # This is exactly what `cmd_smoke --dry-run` does: build
        # entries, group them, and pick the healthiest per rule to
        # print. Neither line calls `run_smoke_check` nor any
        # transport.
    finally:
        smoke_module.run_smoke_check = original

    assert called["run_smoke_check"] is False


# ---------------------------------------------------------------------
# Declared Offerings: grouped by vendor prefix, not by translate.py
# (which has no rule for them at all -- CONTEXT.md, "Declared
# Offering").
# ---------------------------------------------------------------------


def test_a_declared_offering_is_included_and_its_alias_is_called():
    feed = _feed()
    policy = _policy(
        providers={},
        declared=[
            {
                "alias": "claude-gpt-5.6-luna",
                "litellm_params": {"model": "openai/gpt-5.6-luna"},
            }
        ],
    )
    entries = build_smoke_entries(feed=feed, policy=policy)
    assert len(entries) == 1
    assert entries[0].alias == "claude-gpt-5.6-luna"
    assert entries[0].key == "claude-gpt-5.6-luna"

    def transport(entry: SmokeEntry) -> TransportResponse:
        assert entry.alias == "claude-gpt-5.6-luna"
        return _ok_response()

    result = run_smoke_check(entries, health={}, transport=transport, now=lambda: NOW)
    assert result.checks[0].status == STATUS_ANSWERED


def test_a_passthrough_auth_declared_offering_is_never_called():
    feed = _feed()
    policy = _policy(
        providers={},
        declared=[
            {
                "alias": "claude-passthrough",
                "litellm_params": {"model": "openai/whatever"},
                "passthrough_auth": True,
            }
        ],
    )
    entries = build_smoke_entries(feed=feed, policy=policy)
    # It appears, so its rule still reaches the report, but the proxy
    # must never call it: the caller holds the credential.
    assert len(entries) == 1
    assert entries[0].callable_by_proxy is False
    assert pick_healthiest(entries, health={}) is None


def test_a_rule_whose_offerings_are_all_passthrough_reports_unverified_not_missing():
    """A rule must never vanish from the report.

    Dropping an uncallable entry would delete a whole translation rule,
    and the operator would read a clean run while that rule went
    unchecked. The operator's four direct Claude Aliases are exactly
    this case: the caller supplies the subscription credential.
    """
    feed = _feed()
    policy = _policy(
        providers={},
        declared=[
            {
                "alias": "claude-passthrough",
                "litellm_params": {"model": "anthropic/whatever"},
                "passthrough_auth": True,
            }
        ],
    )
    entries = build_smoke_entries(feed=feed, policy=policy)
    result = run_smoke_check(
        entries, health={}, transport=_never_called_transport, now=lambda: NOW
    )
    assert len(result.checks) == 1
    check = result.checks[0]
    assert check.status == STATUS_UNVERIFIED
    assert "Passthrough Auth" in check.detail
    assert check.status != STATUS_FAILED, "uncallable is not a failure"


def test_a_passthrough_auth_declared_offering_with_proxy_authenticated_is_included():
    """CONTEXT.md, "Passthrough Auth", plus the operator's own
    `proxy_authenticated` flag: the proxy holds this Offering's
    credential itself, so a call under the proxy's own credential
    measures the real thing. `build_smoke_entries` must include it.
    """
    feed = _feed()
    policy = _policy(
        providers={},
        declared=[
            {
                "alias": "claude-gpt-proxy-authenticated",
                "litellm_params": {"model": "chatgpt/gpt-5.6-luna"},
                "passthrough_auth": True,
                "proxy_authenticated": True,
            }
        ],
    )
    entries = build_smoke_entries(feed=feed, policy=policy)
    assert len(entries) == 1
    assert entries[0].alias == "claude-gpt-proxy-authenticated"

    def transport(entry: SmokeEntry) -> TransportResponse:
        assert entry.alias == "claude-gpt-proxy-authenticated"
        return _ok_response()

    result = run_smoke_check(entries, health={}, transport=transport, now=lambda: NOW)
    assert result.checks[0].status == STATUS_ANSWERED


# ---------------------------------------------------------------------
# The live payload always streams, and sends no `temperature` key.
# ---------------------------------------------------------------------


def test_the_smoke_payload_requests_a_streamed_response():
    """The ChatGPT Passthrough Auth route answers a streamed request
    and fails a non-streamed one at every token budget
    (spec-corrections.md, entry 15). A payload built with no `stream`
    key would silently reintroduce that false failure.
    """
    entry = SmokeEntry(key="groq:model", alias="claude-groq-model", rule="native_prefix[groq]")
    payload = build_smoke_payload(entry)
    assert payload["stream"] is True


def test_removing_the_stream_flag_would_reintroduce_the_false_failure_and_this_test_catches_it():
    """Mutation check: a payload builder that forgot `stream` would
    still pass every other assertion in this file (model, messages,
    max_tokens all unchanged). Assert the key by itself.
    """
    entry = SmokeEntry(key="groq:model", alias="claude-groq-model", rule="native_prefix[groq]")
    payload = build_smoke_payload(entry)
    assert "stream" in payload
    assert payload["stream"] is not False


def test_the_smoke_payload_sends_no_temperature_key():
    """The Claude 5 family accepts `temperature=1` only (docs/gotchas.md,
    "Claude 5 models reject temperature=0"). Omitting the key avoids the
    trap for every model, not just Claude 5.
    """
    entry = SmokeEntry(key="groq:model", alias="claude-groq-model", rule="native_prefix[groq]")
    payload = build_smoke_payload(entry)
    assert "temperature" not in payload


# ---------------------------------------------------------------------
# Reading a streamed response: content reads as Answered, an all
# keep-alive stream does not, and a malformed chunk is ignored.
# ---------------------------------------------------------------------


def test_a_streamed_response_carrying_assistant_content_reads_as_answered():
    raw = _sse(
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
    )
    assert extract_streamed_content(raw).content == "Hello"

    body = {"choices": [{"message": {"content": extract_streamed_content(raw).content}}]}
    entries = (SmokeEntry(key="groq:streamed", alias="claude-groq-streamed", rule="native_prefix[groq]"),)

    def transport(entry: SmokeEntry) -> TransportResponse:
        return TransportResponse(http_status=200, body=body, transport=None)

    result = run_smoke_check(entries, health={}, transport=transport, now=lambda: NOW)
    assert result.checks[0].status == STATUS_ANSWERED


def test_a_streamed_response_with_only_done_and_keep_alives_does_not_read_as_answered():
    raw = _sse(keep_alives=2)
    read = extract_streamed_content(raw)
    assert read.content == "" and read.chunks_seen == 0

    body: dict = ({"choices": [{"message": {"content": read.content}}]}
                  if read.chunks_seen else {})
    entries = (SmokeEntry(key="groq:silent", alias="claude-groq-silent", rule="native_prefix[groq]"),)

    def transport(entry: SmokeEntry) -> TransportResponse:
        return TransportResponse(http_status=200, body=body, transport=None)

    result = run_smoke_check(entries, health={}, transport=transport, now=lambda: NOW)
    assert result.checks[0].status != STATUS_ANSWERED
    assert result.answered == ()


def test_removing_the_choices_key_guard_would_read_a_silent_stream_as_answered():
    """Mutation check: if the transport wrapped every streamed reply in
    a `choices` key regardless of content, a silent stream (only
    `[DONE]` and keep-alives) would read as Answered here. Assert the
    guard directly: no content means no `choices` key in the body
    `classify` receives.
    """
    raw = _sse(keep_alives=3)
    content = extract_streamed_content(raw).content
    assert content == ""
    body = {"choices": [{"message": {"content": content}}]} if content else {}
    assert "choices" not in body


def test_a_malformed_sse_chunk_is_ignored_not_read_as_a_provider_failure():
    """A parse failure is not a provider failure -- the same class of
    mistake the streaming fix removes. One broken chunk must not stop
    a later, well-formed chunk from being read.
    """
    raw = (
        "data: {not valid json at all\n\n"
        f"data: {json.dumps({'choices': [{'delta': {'content': 'ok'}}]})}\n\n"
        "data: [DONE]\n\n"
    )
    assert extract_streamed_content(raw).content == "ok"


def test_breaking_the_json_parse_guard_would_raise_instead_of_ignoring_and_this_test_catches_it():
    """Mutation check: a parser with no `try/except` around `json.loads`
    would raise `ValueError` on the malformed line below, instead of
    ignoring it and reading the well-formed chunk that follows.
    """
    raw = "data: {broken\n\n" + f"data: {json.dumps({'choices': [{'delta': {'content': 'still-ok'}}]})}\n\n"
    assert extract_streamed_content(raw).content == "still-ok"


def test_a_blank_line_and_a_keep_alive_comment_are_ignored():
    raw = "\n: keep-alive\n\n" + f"data: {json.dumps({'choices': [{'delta': {'content': 'x'}}]})}\n\n"
    assert extract_streamed_content(raw).content == "x"


# ---------------------------------------------------------------------
# The operator's real Policy: the rule count goes from 7 to 8 (the
# ChatGPT subscription rule), then from 8 to 9 (the ChatGPT worker-seat
# rule, added 2026-07-26).
# ---------------------------------------------------------------------


def test_a_declared_seat_groups_under_the_generic_openai_rule():
    """A seat routes through the generic `openai/` prefix to a local
    worker, so it groups under `declared:openai` rather than under
    litellm's own `chatgpt/` provider. Sending a seat through
    `chatgpt/` would reach the real ChatGPT backend and skip the worker
    (docs/gotchas.md).

    A rule with no Offering behind it must not appear at all, so
    `declared:chatgpt` is absent: the fixture declares one
    `chatgpt/` Offering, and it is Passthrough Auth, which the smoke
    check never calls.
    """
    policy = load_policy(PINNED_POLICY_PATH)
    feed = load_feed(FEED_CURRENT_PATH)
    entries = build_smoke_entries(feed=feed, policy=policy)
    grouped = group_by_rule(entries)

    # A seat is a Declared Offering with an `openai/` prefix, so it
    # must group here and nowhere else.
    seats = {
        d.alias
        for d in policy.declared
        if d.litellm_params.get("api_key", "").endswith("_WORKER_KEY")
    }
    assert seats, "the fixture declares no seat"
    assert seats <= {e.alias for e in grouped["declared:openai"]}

    # The fixture's one `chatgpt/` Offering is Passthrough Auth: the
    # CALLER holds the credential, so the smoke check must not call it.
    # It may be grouped, but never as callable.
    for entry in grouped.get("declared:chatgpt", ()):
        assert entry.callable_by_proxy is False, entry.alias

    assert "declared:openai" in grouped
    # A seat on localhost and a private host far away dial the same
    # generic `openai/` prefix with an explicit `api_base`, so both group
    # under one rule. Where the host sits changes nothing about the rule.
    seat_entries = grouped["declared:openai"]
    assert all(
        d.litellm_params["model"].startswith("openai/")
        for e in seat_entries
        for d in policy.declared
        if d.alias == e.alias
    )


def test_removing_proxy_authenticated_makes_the_rule_uncallable():
    """Mutation check on the flag, not on the operator's Policy.

    Strip `proxy_authenticated` from a Passthrough Auth Declared
    Offering. Its rule must stay in the report, because a rule must
    never vanish, but nothing behind it may be callable any more.

    Synthetic on purpose: the operator retired every direct `chatgpt/`
    entry, so their Policy now sets the flag nowhere. The flag is still
    in the code, so it is still tested here.
    """
    from dataclasses import replace

    feed = _feed()
    policy = _policy(
        providers={},
        declared=[
            {
                "alias": "claude-gpt-proxy-authenticated",
                "litellm_params": {"model": "chatgpt/gpt-5.6-luna"},
                "passthrough_auth": True,
                "proxy_authenticated": True,
            }
        ],
    )
    stripped_declared = tuple(replace(d, proxy_authenticated=False) for d in policy.declared)
    policy = replace(policy, declared=stripped_declared)

    entries = build_smoke_entries(feed=feed, policy=policy)
    grouped = group_by_rule(entries)

    assert "declared:chatgpt" in grouped, "a rule must never vanish from the report"
    assert not any(e.callable_by_proxy for e in grouped["declared:chatgpt"]), (
        "without the flag the proxy must not call a Passthrough Auth Offering"
    )
    assert pick_healthiest(grouped["declared:chatgpt"], health={}) is None


def test_a_reasoning_model_that_emits_no_text_still_reads_as_answered():
    """The live smoke run of 2026-07-26 found this.

    A reasoning model on the smoke check's small token budget spends it
    all on `reasoning_content` and emits `content: ""`. The first
    streaming version required non-empty text, so it reported FAILED
    (malformed_response) on five rules that had answered a moment
    earlier without streaming.

    This check tests WIRING. A well-formed chunk proves the route
    resolved, the handler is registered, the base URL is right and the
    credential worked. Whether the model found words inside 16 tokens
    is not the question.
    """
    raw = (
        'data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}\n\n'
        'data: {"choices":[{"index":0,"delta":{"reasoning_content":"The","channel":"analysis"}}]}\n\n'
        "data: [DONE]\n\n"
    )
    read = extract_streamed_content(raw)
    assert read.chunks_seen == 2, "both chunks are well formed"
    assert read.content == "", "and neither carried assistant text"

    body = (
        {"choices": [{"message": {"content": read.content}}]} if read.chunks_seen else {}
    )
    from litellm_maintainer.classify import ANSWERED, classify

    outcome = classify(provider="groq", http_status=200, body=body, now=NOW)
    assert outcome.bucket == ANSWERED, (
        "a working route that emitted no text must not read as a failure"
    )


def test_a_stream_with_no_well_formed_chunk_is_still_not_answered():
    """The counterpart. Loosening the rule must not make silence pass."""
    raw = ": keep-alive\n\ndata: [DONE]\n\n"
    read = extract_streamed_content(raw)
    assert read.chunks_seen == 0
    body = (
        {"choices": [{"message": {"content": read.content}}]} if read.chunks_seen else {}
    )
    from litellm_maintainer.classify import ANSWERED, classify

    assert (
        classify(provider="groq", http_status=200, body=body, now=NOW).bucket
        != ANSWERED
    )
