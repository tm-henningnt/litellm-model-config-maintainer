"""Tests for `litellm_maintainer.prober`.

No test makes a network call. Every test injects a fake transport (a
callable that returns a recorded body) and, where pacing matters, a
fake clock (`now` and `sleep`) so a pacing rule is assertable with no
real waiting. Test names use the glossary vocabulary from CONTEXT.md:
Withheld, Excluded, Passthrough Auth, Inconclusive and Candidate are
four different things.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

from litellm_maintainer.classify import ANSWERED, NEEDS_OPERATOR, Outcome, classify
from litellm_maintainer.feed import load_feed, parse_feed
from litellm_maintainer.policy import DeclaredOffering, Pacing, load_policy, parse_policy
from litellm_maintainer.sse import read_stream
from litellm_maintainer.prober import (
    build_probe_payload,
    ProbeTarget,
    TransportResponse,
    build_probe_url,
    build_worklist,
    format_summary_line,
    probe_offering,
    probe_offerings,
)
from litellm_maintainer.redact import redact
from litellm_maintainer.reduce import HealthState, OfferingHealth, reduce
from test_translate import PERSONAL_PLAN_DENIED_OFFERING_IDS

FIXTURES = Path(__file__).parent / "fixtures" / "classify"
FEED_CURRENT_PATH = Path(__file__).parent / "fixtures" / "feed-current.json"
OPERATOR_POLICY_PATH = Path("/Users/hentol/.config/litellm-maintainer/policy.yaml")

NOW = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)


def _offering_raw(
    offering_id: str,
    *,
    coding_score: float | None = 50.0,
    capabilities: list[str] | None = None,
) -> dict:
    provider_id, _, model_id = offering_id.partition(":")
    return {
        "id": offering_id,
        "provider": {"id": provider_id},
        "provider_model_id": model_id,
        "capabilities": ["tool_use"] if capabilities is None else capabilities,
        "endpoint": {
            "protocol": "openai",
            "base_url": f"https://example.invalid/{provider_id}",
            "model": model_id,
        },
        "pricing": {"kind": "free"},
        "availability": {"status": "available"},
        "quality": {"coding_score": coding_score},
        "policy": {"visibility": "listed"},
    }


def _feed(*offerings: dict):
    return parse_feed(
        {
            "schema_version": "1",
            "providers": [{"id": "acme", "name": "Acme"}],
            "models": list(offerings),
        }
    )


def _policy_dict(**overrides) -> dict:
    base = {
        "providers": {"acme": {"mode": "all"}},
        "quality": {"minimum_coding_score": 18},
        "approved_candidates": [],
        "naming": {
            "alias_prefix": "claude-",
            "provider_labels": {"acme": "acme"},
            "alias_overrides": {},
        },
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
    base.update(overrides)
    return base


def _policy(**overrides):
    return parse_policy(_policy_dict(**overrides))


def _excluded(*, reset_at: datetime | None, last_success_at: datetime | None) -> OfferingHealth:
    return OfferingHealth(
        excluded=True,
        reason="prior failure",
        bucket="self_healing",
        reset_at=reset_at,
        last_success_at=last_success_at,
        last_attempt_at=NOW - timedelta(hours=2),
        failure_count=1,
    )


def _healthy(last_success_at: datetime) -> OfferingHealth:
    return OfferingHealth(
        excluded=False,
        last_success_at=last_success_at,
        last_attempt_at=last_success_at,
        failure_count=0,
    )


# ---------------------------------------------------------------------------
# The worklist comes from Policy.


def test_the_worklist_comes_from_policy_so_an_excluded_offering_is_still_reached():
    """An Excluded Offering is absent from the Generated Config, so a
    worklist read from that file could never reach it again.
    Auto-recovery depends on reading Policy instead.

    Uses an Excluded Offering with NO reset time. A recorded reset time
    in the future defers the Probe on purpose (see the reset-time tests
    below), so it would prove the wrong thing here.
    """
    feed = _feed(_offering_raw("acme:model-a"))
    policy = _policy()
    health = HealthState(
        offerings={
            "acme:model-a": _excluded(
                reset_at=None,
                last_success_at=NOW - timedelta(days=10),
            )
        }
    )

    worklist = build_worklist(feed=feed, policy=policy, health=health, now=NOW)

    assert [t.key for t in worklist.targets] == ["acme:model-a"]


def test_a_withheld_offering_is_never_probed():
    feed = _feed(_offering_raw("acme:model-a"))
    policy = _policy(withheld={"acme:model-a": "quota unclear"})
    health = HealthState(offerings={})

    worklist = build_worklist(feed=feed, policy=policy, health=health, now=NOW)

    assert worklist.targets == ()
    assert worklist.skipped_withheld == ("acme:model-a",)


def test_a_passthrough_auth_offering_is_never_probed():
    feed = _feed()
    policy = _policy(
        declared=[
            {
                "alias": "claude-caller-model",
                "passthrough_auth": True,
                "litellm_params": {"model": "chatgpt/caller-model"},
            }
        ]
    )
    health = HealthState(offerings={})

    worklist = build_worklist(feed=feed, policy=policy, health=health, now=NOW)

    assert worklist.targets == ()
    assert worklist.skipped_passthrough == ("claude-caller-model",)
    # Still tracked, so its Health State record is not discarded by `reduce`.
    assert "claude-caller-model" in worklist.admitted


def test_an_offering_whose_health_is_fresh_is_skipped():
    feed = _feed(_offering_raw("acme:model-a"))
    policy = _policy()
    health = HealthState(offerings={"acme:model-a": _healthy(NOW - timedelta(hours=1))})

    worklist = build_worklist(feed=feed, policy=policy, health=health, now=NOW)

    assert worklist.targets == ()
    assert worklist.skipped_fresh == ("acme:model-a",)


def test_an_offering_whose_reset_time_has_passed_is_probed():
    feed = _feed(_offering_raw("acme:model-a"))
    policy = _policy()
    health = HealthState(
        offerings={
            "acme:model-a": _excluded(
                reset_at=NOW - timedelta(minutes=1),
                last_success_at=NOW - timedelta(hours=1),
            )
        }
    )

    worklist = build_worklist(feed=feed, policy=policy, health=health, now=NOW)

    assert [t.key for t in worklist.targets] == ["acme:model-a"]


# ---------------------------------------------------------------------------
# The worklist applies the same baseline capability filter `plan` applies.
#
# Every synthetic Offering below passes the provider mode ("all"), carries no
# pricing filter to fail, scores above the quality threshold (default 50, the
# policy minimum is 18), and is not Withheld. The only thing that can make it
# fail is the baseline itself, so a passing test here is evidence the rule
# runs, not evidence of some other gate.


def test_an_offering_without_tool_use_is_not_probed():
    feed = _feed(_offering_raw("acme:model-a", capabilities=[]))
    policy = _policy()
    health = HealthState(offerings={})

    worklist = build_worklist(feed=feed, policy=policy, health=health, now=NOW)

    assert worklist.targets == ()


def test_an_offering_carrying_an_excluded_capability_is_not_probed():
    feed = _feed(
        _offering_raw("acme:model-a", capabilities=["tool_use", "image_generation"])
    )
    policy = _policy()
    health = HealthState(offerings={})

    worklist = build_worklist(feed=feed, policy=policy, health=health, now=NOW)

    assert worklist.targets == ()


def test_control_the_same_offering_is_probed_once_its_baseline_fault_is_removed():
    # Same builder, same score, same policy as the two tests above, with
    # only the baseline fault removed (plain `tool_use`, no excluded
    # capability). Without this control, the two tests above could pass
    # because the builder is broken, not because the baseline rule works.
    feed = _feed(_offering_raw("acme:model-a", capabilities=["tool_use"]))
    policy = _policy()
    health = HealthState(offerings={})

    worklist = build_worklist(feed=feed, policy=policy, health=health, now=NOW)

    assert [t.key for t in worklist.targets] == ["acme:model-a"]


def test_the_worklist_for_the_operators_real_policy_is_pinned():
    # A future change to the baseline, or to any other filter, must not
    # move these counts silently. All were read from the code, not
    # guessed (`build_worklist` against `feed-current.json` and the
    # operator's real Policy, empty Health State).
    #
    # 72, not the earlier 68, and not the 81 of a moment ago. Three
    # things changed it in total, all measured, not guessed:
    #
    # 1. +12: the ChatGPT worker seats added 2026-07-26. They are
    #    ordinary Declared Offerings, NOT Passthrough Auth — this proxy
    #    holds the worker key itself — so the Prober probes all 12.
    # 2. +1: `opencode-go:hy3-preview`. Unrelated to the seats: it scores
    #    58.8, above the quality threshold, and the operator's Policy
    #    sets `opencode-go: mode: all`, with no dedup for a `hy3` /
    #    `hy3-preview` pair. `build_worklist` reads Policy, not the
    #    Generated Config (module docstring), so it reaches this
    #    Offering even though it is `retired` and `hidden` in
    #    `feed-current.json` and never appears in the generated
    #    entries — the worklist may reach an Excluded Offering so it
    #    can recover.
    # 3. -9: the Qwen Token Plan Offerings in
    #    `PERSONAL_PLAN_DENIED_OFFERING_IDS`, now Withheld (personal-tier
    #    denial, HTTP 403 on every call). The Prober skips a Withheld
    #    Offering (CONTEXT.md, "Prober": "only a human clears those"),
    #    so `build_worklist` must never put one of these nine in
    #    `targets`. 81 - 9 = 72.
    #
    # The earlier 72-before-the-seats already accounted for the four
    # direct Claude Declared Offerings being marked Passthrough Auth on
    # 2026-07-26: the operator routes a Claude subscription from the
    # client through the proxy, so the CALLER holds the credential, and
    # the Prober must not probe them — it would measure litellm's
    # fallback to ANTHROPIC_API_KEY, which is not the credential real
    # traffic uses.
    feed = load_feed(FEED_CURRENT_PATH)
    policy = load_policy(OPERATOR_POLICY_PATH)
    health = HealthState(offerings={})

    worklist = build_worklist(feed=feed, policy=policy, health=health, now=NOW)

    assert len(worklist.targets) == 72
    # The 4 direct-Claude Aliases. Their 3 Client-Facing Variants are
    # NOT counted separately: this list is keyed by `health_key`, and a
    # variant shares one record with the Alias it widens, because it is
    # the same wire request under a second name. Counting 4 + 3 here
    # meant two health records for one Offering, and they drifted: an
    # exhausted quota Excluded `claude-opus-5` and left
    # `claude-opus-5[1m]` in the Generated Config. The 6 direct ChatGPT
    # Aliases were also Passthrough Auth and were retired on
    # 2026-07-26, so they no longer reach this list.
    assert len(worklist.skipped_passthrough) == 4
    assert not any(alias.endswith("[1m]") for alias in worklist.skipped_passthrough)

    # None of the nine personal-plan-denied Offerings is in the
    # worklist, by name — a closed, named check, not "the count moved".
    target_keys = {t.key for t in worklist.targets}
    assert target_keys.isdisjoint(PERSONAL_PLAN_DENIED_OFFERING_IDS)

    # No Declared Offering in the worklist is Passthrough Auth — a
    # Passthrough Auth Declared Offering belongs in
    # `skipped_passthrough`, never in `targets`. The 12 ChatGPT worker
    # seats ARE legitimate non-passthrough Declared targets (`offering`
    # is None because they carry no Feed id; `declared` names them).
    declared_targets = [t for t in worklist.targets if t.offering is None]
    assert len(declared_targets) == 12
    for target in declared_targets:
        assert target.declared is not None
        assert target.declared.passthrough_auth is False
        assert target.declared.alias.startswith(("claude-chatgpt1-", "claude-chatgpt2-"))


# ---------------------------------------------------------------------------
# Pacing: concurrency and minimum interval, per provider, via an injected clock.


def test_each_provider_is_probed_within_its_own_concurrency_limit():
    targets = tuple(
        ProbeTarget(key=f"prov:model-{i}", provider_id="prov") for i in range(4)
    )
    pacing = {
        "default": Pacing(concurrency=1, minimum_interval_seconds=0),
        "prov": Pacing(concurrency=2, minimum_interval_seconds=0),
    }

    lock = threading.Lock()
    active = 0
    peak = 0

    def fake_transport(target: ProbeTarget) -> TransportResponse:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        return TransportResponse(http_status=200, body={"choices": [{"message": {}}]})

    probe_offerings(
        targets,
        pacing=pacing,
        transport=fake_transport,
        now=lambda: NOW,
        sleep=lambda seconds: None,
    )

    assert peak == 2


def test_each_provider_is_probed_after_its_own_minimum_interval():
    targets = tuple(
        ProbeTarget(key=f"prov:model-{i}", provider_id="prov") for i in range(3)
    )
    pacing = {
        "default": Pacing(concurrency=1, minimum_interval_seconds=0),
        "prov": Pacing(concurrency=1, minimum_interval_seconds=10),
    }

    class FakeClock:
        def __init__(self, start: datetime) -> None:
            self.value = start
            self.lock = threading.Lock()

        def now(self) -> datetime:
            with self.lock:
                return self.value

        def sleep(self, seconds: float) -> None:
            with self.lock:
                self.value += timedelta(seconds=seconds)

    clock = FakeClock(NOW)
    call_times: list[datetime] = []
    call_lock = threading.Lock()

    def fake_transport(target: ProbeTarget) -> TransportResponse:
        with call_lock:
            call_times.append(clock.now())
        return TransportResponse(http_status=200, body={"choices": [{"message": {}}]})

    probe_offerings(
        targets,
        pacing=pacing,
        transport=fake_transport,
        now=clock.now,
        sleep=clock.sleep,
    )

    assert len(call_times) == 3
    call_times.sort()
    assert call_times[1] - call_times[0] >= timedelta(seconds=10)
    assert call_times[2] - call_times[1] >= timedelta(seconds=10)


# ---------------------------------------------------------------------------
# Retry once on a rate-limit shape.


def test_a_rate_limit_shaped_failure_is_retried_once_after_a_backoff_before_it_counts(
    load_fixture,
):
    rate_limited = load_fixture("classify/opencode-go-rate-limit.json")
    answered_body = {"choices": [{"message": {"role": "assistant", "content": "pong"}}]}

    responses = [
        TransportResponse(http_status=rate_limited["http_status"], body=rate_limited["body"]),
        TransportResponse(http_status=200, body=answered_body),
    ]
    calls: list[TransportResponse] = []

    def fake_transport(target: ProbeTarget) -> TransportResponse:
        response = responses[len(calls)]
        calls.append(response)
        return response

    sleeps: list[float] = []

    outcome = probe_offering(
        ProbeTarget(key="opencode-go:model-a", provider_id="opencode-go"),
        transport=fake_transport,
        now=lambda: NOW,
        sleep=lambda seconds: sleeps.append(seconds),
    )

    assert len(calls) == 2
    assert sleeps == [5.0]
    assert outcome.bucket == ANSWERED


def test_a_probe_that_remains_ambiguous_records_inconclusive_and_health_state_does_not_change(
    load_fixture,
):
    rate_limited = load_fixture("classify/opencode-go-rate-limit.json")

    def fake_transport(target: ProbeTarget) -> TransportResponse:
        return TransportResponse(http_status=rate_limited["http_status"], body=rate_limited["body"])

    sleeps: list[float] = []

    outcome = probe_offering(
        ProbeTarget(key="opencode-go:model-a", provider_id="opencode-go"),
        transport=fake_transport,
        now=lambda: NOW,
        sleep=lambda seconds: sleeps.append(seconds),
    )

    assert outcome.bucket == "inconclusive"
    assert sleeps == [5.0]  # retried once, and only once

    healthy = _healthy(NOW - timedelta(hours=1))
    prior = HealthState(offerings={"opencode-go:model-a": healthy})
    result = reduce(
        prior=prior,
        outcomes={"opencode-go:model-a": outcome},
        observations=[],
        admitted={"opencode-go:model-a"},
        passthrough_auth=set(),
        now=NOW,
    )

    # Health is untouched. `inconclusive_count` is the one deliberate
    # exception, so that a silent misclassification becomes visible.
    assert replace(result.offerings["opencode-go:model-a"], inconclusive_count=0) == healthy
    assert result.offerings["opencode-go:model-a"].inconclusive_count == 1


# ---------------------------------------------------------------------------
# Output redaction.


def test_the_printed_summary_redacts_a_credential():
    outcome = Outcome(bucket=NEEDS_OPERATOR, reset_at=None, reason="authentication_failed")
    body = {"error": {"message": "Bad key sk-fake1234567890abcdefFAKE, request denied"}}

    line = format_summary_line("acme:model-a", outcome, body)
    printed = redact(line, {})

    assert "sk-fake1234567890abcdefFAKE" not in printed
    assert "<REDACTED:sk-token>" in printed


# ---------------------------------------------------------------------------
# The live transport builds the completions URL, per protocol. No test here
# makes a network call: `build_probe_url` is pure string work, asserted
# directly (defect 1a, spec-corrections.md #12).


def test_an_openai_shaped_offering_probes_the_chat_completions_path():
    url = build_probe_url("https://opencode.ai/zen/go/v1", "opencode-go")

    assert url == "https://opencode.ai/zen/go/v1/chat/completions"


def test_a_base_url_with_a_trailing_slash_is_not_doubled():
    url = build_probe_url("https://opencode.ai/zen/go/v1/", "opencode-go")

    assert url == "https://opencode.ai/zen/go/v1/chat/completions"


def test_the_path_follows_the_feeds_protocol_never_the_provider_name():
    """The first full live sweep found this.

    An earlier version forced `/messages` for the Qwen Token Plan by
    NAME, reasoning that the operator's config routes it
    Anthropic-shaped. That is true of the config route, which lives on a
    different host path, and false of the Feed route. The Feed publishes
    `openai_chat_completions` with a `compatible-mode/v1` base, so the
    override built `compatible-mode/v1/messages` and every one of the 15
    Qwen Offerings came back `unrecognized_failure` on a 404. The real
    endpoint returns a 429 naming its quota reset time.

    Base URL and protocol come from the same `endpoint` object, so they
    cannot disagree. A provider name can disagree with both.
    """
    base = "https://token-plan.example.invalid/compatible-mode/v1"

    assert (
        build_probe_url(base, "qwencloud-token-plan", "openai_chat_completions")
        == base + "/chat/completions"
    )
    # The name alone must not select a path.
    assert (
        build_probe_url(base, "qwencloud-token-plan") == base + "/chat/completions"
    )
    # A Feed that does state an Anthropic protocol still gets /messages.
    assert (
        build_probe_url(base, "anyone", "anthropic_messages") == base + "/messages"
    )


# ---------------------------------------------------------------------------
# The Probe streams. See spec-corrections.md 15 and 16, and the shared
# reader in litellm_maintainer.sse.
# ---------------------------------------------------------------------------


def _sse_body(*objs, keep_alives: int = 0, done: bool = True) -> str:
    lines = [": keep-alive" for _ in range(keep_alives)]
    lines += [f"data: {json.dumps(o)}" for o in objs]
    if done:
        lines.append("data: [DONE]")
    return "\n\n".join(lines) + "\n\n"


def _target() -> ProbeTarget:
    declared = DeclaredOffering(
        alias="claude-seat",
        litellm_params={
            "model": "openai/claude-gpt-5.6-luna",
            "api_base": "http://127.0.0.1:4011/v1",
        },
    )
    return ProbeTarget(key="claude-seat", provider_id="openai", declared=declared)


def test_a_probe_requests_a_streamed_response():
    """litellm's ChatGPT provider answers streamed and fails
    non-streaming. The operator runs two local litellm worker seats, so
    a non-streaming Probe would Exclude twelve working Offerings on its
    first sweep.
    """
    payload = build_probe_payload(_target())
    assert payload["stream"] is True
    assert "temperature" not in payload, "Claude 5 accepts temperature=1 only"


def test_a_streamed_probe_that_emits_no_text_still_answers():
    """A reasoning model on this token budget spends it on reasoning and
    emits `content: ""`. That is a working route, not a failure.
    Measured on Groq and on Gemini.
    """
    raw = _sse_body(
        {"choices": [{"delta": {"role": "assistant", "content": ""}}]},
        {"choices": [{"delta": {"reasoning_content": "The"}}]},
        keep_alives=1,
    )
    read = read_stream(raw)
    assert read.chunks_seen == 2 and read.content == ""
    body = {"choices": [{"message": {"content": read.content}}]} if read.chunks_seen else {}
    assert classify(provider="groq", http_status=200, body=body, now=NOW).bucket == ANSWERED


def test_a_streamed_probe_with_no_well_formed_chunk_does_not_answer():
    read = read_stream(": keep-alive\n\ndata: [DONE]\n\n")
    assert read.chunks_seen == 0
    body = {"choices": [{"message": {"content": read.content}}]} if read.chunks_seen else {}
    assert classify(provider="groq", http_status=200, body=body, now=NOW).bucket != ANSWERED


def test_an_openrouter_keep_alive_comment_is_not_read_as_a_chunk():
    """OpenRouter emits `: OPENROUTER PROCESSING` before real chunks."""
    raw = ": OPENROUTER PROCESSING\n\n" + _sse_body(
        {"choices": [{"delta": {"content": "ok"}}]}
    )
    read = read_stream(raw)
    assert read.chunks_seen == 1 and read.content == "ok"


def test_a_declared_probe_strips_the_litellm_routing_prefix():
    """The first live sweep of the worker seats found this.

    `litellm_params.model` carries a litellm ROUTING prefix. litellm
    strips it before calling the provider. The Prober posts directly, so
    it must strip it too. Leaving it on sent
    `openai/claude-gpt-5.6-luna` to a worker advertising
    `claude-gpt-5.6-luna`, and all twelve seats reported
    needs_operator.
    """
    assert build_probe_payload(_target())["model"] == "claude-gpt-5.6-luna"


def test_a_declared_probe_keeps_a_vendor_path_inside_the_model_id():
    """Strip only the routing prefix, never a vendor path."""
    declared = DeclaredOffering(
        alias="claude-x",
        litellm_params={
            "model": "openrouter/cohere/north-mini-code:free",
            "api_base": "https://example.invalid/v1",
        },
    )
    target = ProbeTarget(key="claude-x", provider_id="openrouter", declared=declared)
    assert build_probe_payload(target)["model"] == "cohere/north-mini-code:free"


def test_a_declared_model_with_no_prefix_is_sent_unchanged():
    declared = DeclaredOffering(
        alias="claude-y",
        litellm_params={"model": "bare-model", "api_base": "https://example.invalid/v1"},
    )
    target = ProbeTarget(key="claude-y", provider_id="bare", declared=declared)
    assert build_probe_payload(target)["model"] == "bare-model"


def test_an_excluded_offering_is_not_probed_before_its_reset_time():
    """The spec's Probing rule: probe when health is stale, when the
    reset time has PASSED, or when the last outcome was Inconclusive. A
    reset time still in the future says none of those.

    Measured 2026-07-26: six Qwen Token Plan Offerings carried a reset
    time four days out, and every sweep probed all six. Calling an
    exhausted plan before it refills costs a call and cannot succeed.
    """
    feed = _feed(_offering_raw("acme:model-a"))
    policy = _policy(providers={"acme": {"mode": "all"}})
    record = OfferingHealth(
        excluded=True,
        reason="quota_exhausted",
        bucket="self_healing",
        reset_at=NOW + timedelta(days=4),
        last_success_at=None,
        last_attempt_at=NOW,
        failure_count=1,
    )
    worklist = build_worklist(
        feed=feed,
        policy=policy,
        health=HealthState(offerings={"acme:model-a": record}),
        now=NOW,
    )
    assert [t.key for t in worklist.targets] == []
    assert "acme:model-a" in worklist.skipped_fresh
    assert "acme:model-a" in worklist.admitted, "its record must survive the run"


def test_an_excluded_offering_is_probed_once_its_reset_time_has_passed():
    feed = _feed(_offering_raw("acme:model-a"))
    policy = _policy(providers={"acme": {"mode": "all"}})
    record = OfferingHealth(
        excluded=True,
        reason="quota_exhausted",
        bucket="self_healing",
        reset_at=NOW - timedelta(minutes=1),
        last_success_at=None,
        last_attempt_at=NOW - timedelta(days=1),
        failure_count=1,
    )
    worklist = build_worklist(
        feed=feed,
        policy=policy,
        health=HealthState(offerings={"acme:model-a": record}),
        now=NOW,
    )
    assert [t.key for t in worklist.targets] == ["acme:model-a"]


def test_an_excluded_offering_with_no_reset_time_is_still_probed():
    """Only a recorded reset time defers a Probe. Without one there is
    nothing to wait for, so the Offering is reached as before.
    """
    feed = _feed(_offering_raw("acme:model-a"))
    policy = _policy(providers={"acme": {"mode": "all"}})
    record = OfferingHealth(
        excluded=True,
        reason="gateway_error",
        bucket="self_healing",
        reset_at=None,
        last_success_at=None,
        last_attempt_at=NOW,
        failure_count=1,
    )
    worklist = build_worklist(
        feed=feed,
        policy=policy,
        health=HealthState(offerings={"acme:model-a": record}),
        now=NOW,
    )
    assert [t.key for t in worklist.targets] == ["acme:model-a"]


def test_force_probes_an_offering_whose_reset_time_has_not_passed():
    """A recorded reset time is a provider's promise, not a fact.

    The operator's ChatGPT plan refilled days before the time its own
    429 stated, on 2026-07-25. Deferring is the right default, because
    it stops an hourly tick calling a plan that cannot answer. `force`
    is how the operator overrules it.
    """
    feed = _feed(_offering_raw("acme:model-a"))
    policy = _policy()
    health = HealthState(
        offerings={
            "acme:model-a": _excluded(
                reset_at=NOW + timedelta(days=4), last_success_at=None
            )
        }
    )

    deferred = build_worklist(feed=feed, policy=policy, health=health, now=NOW)
    assert deferred.targets == ()

    forced = build_worklist(
        feed=feed, policy=policy, health=health, now=NOW, force=True
    )
    assert [t.key for t in forced.targets] == ["acme:model-a"]


def test_force_also_probes_an_offering_whose_health_is_fresh():
    feed = _feed(_offering_raw("acme:model-a"))
    policy = _policy()
    health = HealthState(
        offerings={"acme:model-a": _healthy(NOW - timedelta(minutes=5))}
    )

    assert build_worklist(feed=feed, policy=policy, health=health, now=NOW).targets == ()
    forced = build_worklist(
        feed=feed, policy=policy, health=health, now=NOW, force=True
    )
    assert [t.key for t in forced.targets] == ["acme:model-a"]


def test_force_still_never_probes_a_withheld_or_passthrough_offering():
    """Force overrules Health State, never an operator decision.

    Withheld is cleared only by a human, and a Passthrough Auth Probe
    would carry the wrong credential. Neither is a freshness question.
    """
    feed = _feed(_offering_raw("acme:model-a"))
    policy = _policy(
        withheld={"acme:model-a": "personal plan does not include this model"},
        declared=[
            {
                "alias": "claude-caller",
                "passthrough_auth": True,
                "litellm_params": {"model": "chatgpt/caller"},
            }
        ],
    )

    forced = build_worklist(
        feed=feed, policy=policy, health=HealthState(offerings={}), now=NOW, force=True
    )
    assert forced.targets == ()
    assert forced.skipped_withheld == ("acme:model-a",)
    assert forced.skipped_passthrough == ("claude-caller",)


def test_an_offering_marked_probe_due_is_probed_even_when_its_health_is_fresh():
    """ADR 0004: a shared pool propagates attention, not a verdict.

    A sibling's quota exhaustion sets `probe_due`
    (`reduce._pool_siblings_to_mark`). The Offering stays offered; this
    is what makes the next sweep measure it rather than guess.
    """
    from dataclasses import replace

    feed = _feed(_offering_raw("acme:model-a"))
    policy = _policy()
    fresh = _healthy(NOW - timedelta(hours=1))
    health = HealthState(offerings={"acme:model-a": replace(fresh, probe_due=True)})

    worklist = build_worklist(feed=feed, policy=policy, health=health, now=NOW)

    assert [t.key for t in worklist.targets] == ["acme:model-a"]
    assert worklist.skipped_fresh == ()


# --- A Client-Facing Variant is one Offering, not two ----------------------

_VARIANT_PAIR = [
    {
        "alias": "claude-opus-5",
        "passthrough_auth": True,
        "litellm_params": {"model": "anthropic/claude-opus-5"},
    },
    {
        "alias": "claude-opus-5[1m]",
        "passthrough_auth": True,
        "variant_of": "claude-opus-5",
        "litellm_params": {"model": "anthropic/claude-opus-5"},
    },
]


def test_a_variant_shares_the_health_key_of_the_alias_it_widens():
    """Identity, not inference. A variant reaches the same Offering with
    the same wire request, and "the provider never sees the difference"
    (CONTEXT.md), so the pair cannot legitimately disagree about health.
    """
    from litellm_maintainer.journal import observation_key_map

    feed = _feed(_offering_raw("acme:model-a"))
    policy = _policy(declared=_VARIANT_PAIR)

    mapping = observation_key_map(feed=feed, policy=policy)

    assert mapping["claude-opus-5"] == "claude-opus-5"
    assert mapping["claude-opus-5[1m]"] == "claude-opus-5"


def test_an_excluded_alias_takes_its_variant_out_of_the_generated_config():
    """The failure this prevents: an exhausted Claude quota Excluded
    `claude-opus-5` and left `claude-opus-5[1m]` offered -- the same
    wire request to the same provider, certain to fail."""
    from litellm_maintainer.plan import plan

    feed = _feed(_offering_raw("acme:model-a"))
    policy = _policy(declared=_VARIANT_PAIR)
    health = {
        "claude-opus-5": OfferingHealth(
            excluded=True, reason="quota_exhausted", bucket="self_healing"
        )
    }

    result = plan(feed=feed, policy=policy, health=health, now=NOW)

    offered = [e["model_name"] for e in result.config["model_list"]]
    assert "claude-opus-5" not in offered
    assert "claude-opus-5[1m]" not in offered


def test_a_healthy_alias_keeps_its_variant_offered():
    from litellm_maintainer.plan import plan

    feed = _feed(_offering_raw("acme:model-a"))
    policy = _policy(declared=_VARIANT_PAIR)

    result = plan(feed=feed, policy=policy, health={}, now=NOW)

    offered = [e["model_name"] for e in result.config["model_list"]]
    assert "claude-opus-5" in offered
    assert "claude-opus-5[1m]" in offered


# --- The credential identifies the Entitlement (ADR 0009) ------------------


def _seat(alias: str, key_var: str, **extra) -> dict:
    entry = {
        "alias": alias,
        "entitlement": "shared_pool",
        "litellm_params": {"model": f"openai/{alias}", "api_key": f"os.environ/{key_var}"},
    }
    entry.update(extra)
    return entry


def test_two_seats_behind_one_provider_prefix_are_two_pools():
    """The operator runs two ChatGPT subscriptions behind `openai/`.
    Any provider-level field would call them one pool, and seat 1
    running dry says nothing about seat 2."""
    from litellm_maintainer.entitlements import pool_siblings

    feed = _feed(_offering_raw("acme:model-a"))
    policy = _policy(
        declared=[
            _seat("claude-chatgpt1-a", "SEAT1"),
            _seat("claude-chatgpt1-b", "SEAT1"),
            _seat("claude-chatgpt2-a", "SEAT2"),
            _seat("claude-chatgpt2-b", "SEAT2"),
        ]
    )

    mapping = pool_siblings(feed=feed, policy=policy)

    assert mapping["claude-chatgpt1-a"] == frozenset(
        {"claude-chatgpt1-a", "claude-chatgpt1-b"}
    )
    assert "claude-chatgpt2-a" not in mapping["claude-chatgpt1-a"]


def test_an_explicit_pool_overrides_the_credential():
    """For two keys billed to one account, which the credential rule
    under-groups."""
    from litellm_maintainer.entitlements import pool_siblings

    feed = _feed(_offering_raw("acme:model-a"))
    policy = _policy(
        declared=[
            _seat("claude-a", "KEY_ONE", entitlement_pool="one-account"),
            _seat("claude-b", "KEY_TWO", entitlement_pool="one-account"),
        ]
    )

    mapping = pool_siblings(feed=feed, policy=policy)

    assert mapping["claude-a"] == frozenset({"claude-a", "claude-b"})


def test_a_per_model_declared_offering_joins_no_pool():
    from litellm_maintainer.entitlements import pool_siblings

    feed = _feed(_offering_raw("acme:model-a"))
    policy = _policy(
        declared=[
            {"alias": "claude-a", "litellm_params": {"model": "openai/a", "api_key": "os.environ/K"}},
            {"alias": "claude-b", "litellm_params": {"model": "openai/b", "api_key": "os.environ/K"}},
        ]
    )

    assert pool_siblings(feed=feed, policy=policy) == {}
