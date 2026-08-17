"""`explain` names the stage that stopped an Offering, and types the stop.

Two incidents drove this verb, and each is a test below.

A Policy `pricing` filter dropped a Feed Offering, which then appeared
in no `status` list and no `guidance` row. Nothing said why. That is a
Decision stop: the filter did what it was told.

A Probe cleared an exclusion and wrote no config, so Health State said
healthy while the proxy answered "model not found". That is a Fault
stop.

One word for both would teach the operator to read a deliberate filter
as a bug, which is the mistake this verb exists to avoid.
"""

from __future__ import annotations

from datetime import datetime, timezone

from litellm_maintainer.explain import (
    DECISION,
    FAULT,
    PASSED,
    STAGE_CONFIG,
    STAGE_FEED,
    STAGE_HEALTH,
    STAGE_POLICY,
    STAGE_PROXY,
    STOPPED,
    UNKNOWN,
    explain,
)
from litellm_maintainer.feed import parse_feed
from litellm_maintainer.plan import plan
from litellm_maintainer.policy import parse_policy
from litellm_maintainer.reduce import OfferingHealth

NOW = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)


def _offering(offering_id: str, *, pricing_kind="free", score=60.0):
    provider, model = offering_id.split(":", 1)
    return {
        "id": offering_id,
        "provider": {"id": provider},
        "provider_model_id": model,
        "capabilities": ["tool_use"],
        "endpoint": {"base_url": "https://cline.example/v1", "model": model},
        "pricing": {"kind": pricing_kind},
        "availability": {
            "status": "available",
            "last_checked_at": "2026-08-01T00:00:00Z",
            "stale_after_seconds": 86400,
        },
        "quality": {"coding_score": score},
        "policy": {"visibility": "listed", "tags": []},
    }


def _feed(*offerings):
    return parse_feed(
        {
            "schema_version": "1",
            "providers": [
                {
                    "id": "cline",
                    "object": "provider",
                    "name": "Cline",
                    "api_protocols": ["openai_chat_completions"],
                    "default_base_url": "https://api.cline.bot/api/v1",
                    "authentication": {
                        "type": "api_key",
                        "header": "Authorization",
                        "scheme": "Bearer",
                        "credential_hint": "CLINE_API_KEY",
                    },
                }
            ],
            "models": list(offerings),
        }
    )


def _policy(*, providers=None, declared=None, withheld=None):
    return parse_policy(
        {
            "providers": providers if providers is not None else {"cline": {"mode": "all"}},
            "quality": {"minimum_coding_score": 18},
            "approved_candidates": [],
            "naming": {"alias_prefix": "claude-", "provider_labels": {}, "alias_overrides": {}},
            "withheld": withheld or {},
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
    )


def _walk(query, *, feed, policy, health=None, served=None, proxy_note=""):
    health = health or {}
    report = plan(feed=feed, policy=policy, health=health, now=NOW).report
    return explain(
        query=query,
        feed=feed,
        policy=policy,
        health=health,
        report=report,
        served_aliases=served,
        proxy_note=proxy_note,
    )


def _stage(result, name):
    return next(s for s in result.stages if s.name == name)


# --- the Decision stop ----------------------------------------------------


def test_a_pricing_filter_is_a_decision_stop_that_names_the_policy_line():
    """The incident. A paid Offering under `pricing: [free]` vanished
    from every list with no reason given anywhere."""
    feed = _feed(_offering("cline:deepseek-v4-flash", pricing_kind="paid"))
    policy = _policy(providers={"cline": {"mode": "all", "pricing": ["free"]}})

    result = _walk("cline:deepseek-v4-flash", feed=feed, policy=policy)

    assert result.stopped_at == STAGE_POLICY
    assert result.stop_kind == DECISION
    assert "providers.cline.pricing" in result.stop_detail
    assert result.reaches_a_client is False


def test_a_withheld_offering_is_a_decision_stop():
    feed = _feed(_offering("cline:widget"))
    policy = _policy(withheld={"cline:widget": "subscription ending"})

    result = _walk("cline:widget", feed=feed, policy=policy)

    assert result.stopped_at == STAGE_POLICY
    assert result.stop_kind == DECISION
    assert "withheld" in result.stop_detail


def test_an_unconfigured_provider_is_a_decision_stop():
    feed = _feed(_offering("cline:widget"))
    policy = _policy(providers={})

    result = _walk("cline:widget", feed=feed, policy=policy)

    assert result.stopped_at == STAGE_POLICY
    assert result.stop_kind == DECISION
    assert "providers.cline" in result.stop_detail


def test_an_unknown_name_stops_at_the_feed_without_claiming_a_fault():
    result = _walk("cline:nothing-by-this-name", feed=_feed(), policy=_policy())

    assert result.stopped_at == STAGE_FEED
    assert result.stop_kind == DECISION


# --- the Fault stop -------------------------------------------------------


def test_a_gone_offering_is_a_fault_stop_at_health():
    feed = _feed(_offering("cline:widget"))
    policy = _policy()
    health = {
        "cline:widget": OfferingHealth(
            excluded=True, reason="identifier_gone", bucket="gone"
        )
    }

    result = _walk("cline:widget", feed=feed, policy=policy, health=health)

    assert result.stopped_at == STAGE_HEALTH
    assert result.stop_kind == FAULT


def test_an_alias_the_proxy_does_not_serve_is_a_fault_stop():
    """The qwen incident: admitted, healthy, written, and the proxy
    answers "model not found" because it serves an older generation."""
    feed = _feed(_offering("cline:widget"))
    policy = _policy()

    result = _walk("cline:widget", feed=feed, policy=policy, served=frozenset())

    assert _stage(result, STAGE_CONFIG).verdict == PASSED
    assert result.stopped_at == STAGE_PROXY
    assert result.stop_kind == FAULT


# --- an absent answer is not a negative answer ----------------------------


def test_an_unreachable_proxy_reports_unknown_and_never_stops_the_walk():
    feed = _feed(_offering("cline:widget"))
    policy = _policy()

    result = _walk(
        "cline:widget", feed=feed, policy=policy, served=None, proxy_note="refused"
    )

    assert _stage(result, STAGE_PROXY).verdict == UNKNOWN
    assert result.stopped_at is None
    assert result.stop_kind is None


# --- reachable is not the same as recommended -----------------------------


def test_an_excluded_offering_reaches_a_client_and_is_not_recommended():
    """ADR 0014. The Offering is served, so no stage stops the walk. It
    is still a bad choice, and `recommended` says so separately."""
    feed = _feed(_offering("cline:widget"))
    policy = _policy()
    health = {
        "cline:widget": OfferingHealth(
            excluded=True, reason="gateway_error", bucket="self_healing"
        )
    }

    result = _walk(
        "cline:widget",
        feed=feed,
        policy=policy,
        health=health,
        served=frozenset({"claude-cline-widget"}),
    )

    assert result.reaches_a_client is True
    assert result.recommended is False
    assert _stage(result, STAGE_HEALTH).verdict == PASSED


def test_a_healthy_offering_passes_every_stage():
    feed = _feed(_offering("cline:widget"))
    policy = _policy()

    result = _walk(
        "cline:widget",
        feed=feed,
        policy=policy,
        served=frozenset({"claude-cline-widget"}),
    )

    assert result.reaches_a_client is True
    assert result.recommended is True
    assert [s.verdict for s in result.stages] == [PASSED] * 5


# --- resolving the query --------------------------------------------------


def test_an_alias_resolves_to_the_same_answer_as_its_offering_id():
    feed = _feed(_offering("cline:widget"))
    policy = _policy()

    by_id = _walk("cline:widget", feed=feed, policy=policy)
    by_alias = _walk("claude-cline-widget", feed=feed, policy=policy)

    assert by_alias.offering_id == by_id.offering_id == "cline:widget"
    assert by_alias.alias == by_id.alias == "claude-cline-widget"


def test_a_declared_offering_resolves_by_its_alias():
    feed = _feed()
    policy = _policy(
        declared=[{"alias": "claude-direct", "litellm_params": {"model": "anthropic/x"}}]
    )

    result = _walk(
        "claude-direct", feed=feed, policy=policy, served=frozenset({"claude-direct"})
    )

    assert result.health_key == "claude-direct"
    assert result.offering_id is None
    assert result.reaches_a_client is True


def test_the_walk_serialises_for_a_machine_reader():
    feed = _feed(_offering("cline:widget", pricing_kind="paid"))
    policy = _policy(providers={"cline": {"mode": "all", "pricing": ["free"]}})

    payload = _walk("cline:widget", feed=feed, policy=policy).as_dict()

    assert payload["stopped_at"] == STAGE_POLICY
    assert payload["stop_kind"] == DECISION
    assert payload["stages"][0]["verdict"] == PASSED
    assert payload["stages"][1]["verdict"] == STOPPED


def test_a_stop_still_reports_every_stage_with_the_rest_unreached():
    """A reader sees the whole path and where it ended. A stage after the
    stop is UNKNOWN: the walk never reached it, so it is unmeasured
    rather than good or bad."""
    feed = _feed(_offering("cline:widget", pricing_kind="paid"))
    policy = _policy(providers={"cline": {"mode": "all", "pricing": ["free"]}})

    result = _walk("cline:widget", feed=feed, policy=policy)

    assert [s.name for s in result.stages] == [
        STAGE_FEED,
        STAGE_POLICY,
        STAGE_HEALTH,
        STAGE_CONFIG,
        STAGE_PROXY,
    ]
    assert [s.verdict for s in result.stages] == [
        PASSED,
        STOPPED,
        UNKNOWN,
        UNKNOWN,
        UNKNOWN,
    ]


# --- the live proxy stage, at the CLI seam --------------------------------


def test_a_refused_model_list_call_reports_unknown_and_names_no_header():
    """An absent answer is not a negative answer. The note carries a
    status code, never a header value, so no key reaches a transcript."""
    from litellm_maintainer.cli import fetch_served_aliases

    served, note = fetch_served_aliases(
        "http://127.0.0.1:9", credential="sk-must-not-appear", timeout=0.25
    )

    assert served is None
    assert "sk-must-not-appear" not in note


def test_a_model_list_without_a_data_list_reports_unknown(monkeypatch):
    import contextlib
    import io
    import urllib.request

    from litellm_maintainer.cli import fetch_served_aliases

    @contextlib.contextmanager
    def _fake(request, timeout=None):
        yield io.BytesIO(b'{"object": "list"}')

    monkeypatch.setattr(urllib.request, "urlopen", _fake)
    served, note = fetch_served_aliases("http://proxy.invalid")

    assert served is None
    assert "data" in note


# --- guidance publishes what reached no Route -----------------------------


def test_guidance_lists_a_withheld_offering_as_routeless_with_its_alias():
    """A caller resolving a named Alias and finding nothing can otherwise
    say only "not offered"."""
    from litellm_maintainer.guidance import derive

    feed = _feed(_offering("cline:widget"))
    policy = _policy(withheld={"cline:widget": "subscription ending"})
    report = plan(feed=feed, policy=policy, health={}, now=NOW).report

    guidance = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)

    entry = next(r for r in guidance.routeless if r.offering_id == "cline:widget")
    assert entry.stage == "withheld"
    assert entry.alias == "claude-cline-widget"
    assert entry.refills_at is None


def test_a_bulk_filter_stays_out_of_the_routeless_list():
    """A pricing filter rejects hundreds at a time. Listing them buries
    the answer the list exists to give; `explain` still names the stage
    for any Offering by id."""
    from litellm_maintainer.guidance import derive

    feed = _feed(_offering("cline:widget", pricing_kind="paid"))
    policy = _policy(providers={"cline": {"mode": "all", "pricing": ["free"]}})
    report = plan(feed=feed, policy=policy, health={}, now=NOW).report

    guidance = derive(feed=feed, policy=policy, health={}, report=report, now=NOW)

    assert guidance.routeless == ()
    assert report.dropped["cline:widget"] == "provider_pricing"


def test_routeless_carries_a_recorded_reset_time():
    """`refills_at` says when it returns. `None` means no reset time was
    stated, never that it returns immediately."""
    from datetime import timedelta

    from litellm_maintainer.guidance import derive

    feed = _feed(_offering("cline:widget"))
    policy = _policy(withheld={"cline:widget": "billing unclear"})
    refills = NOW + timedelta(hours=4)
    health = {
        "cline:widget": OfferingHealth(
            excluded=False, reason="quota_exhausted", bucket="self_healing", reset_at=refills
        )
    }
    report = plan(feed=feed, policy=policy, health=health, now=NOW).report

    guidance = derive(feed=feed, policy=policy, health=health, report=report, now=NOW)

    entry = next(r for r in guidance.routeless if r.offering_id == "cline:widget")
    assert entry.refills_at == refills
