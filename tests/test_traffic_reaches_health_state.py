"""Real traffic must reach Health State under the key `plan` reads.

The proxy's failure callback records the ALIAS (`model_group` is the
only name litellm's Router exposes to it). Health State keys a
Discovered Offering by its Offering id and a Declared Offering by its
Alias (`litellm_maintainer.reduce`, `OfferingKey`). These tests pin the
seam between the two key spaces: `journal.observation_key_map` and
`journal.resolve_observation_keys` translate at the read boundary, so a
failure the proxy served Excludes the Offering it names (stories 36,
37 and 39). Before this seam existed, a Journal entry for a Discovered
Offering landed under a key `reduce` discarded as no-longer-admitted,
so real traffic taught the maintainer nothing for 68 of the operator's
78 Aliases.

Also pins the Prober's live-call contract: the credential each direct
Probe sends (`prober.probe_credential`), the URL it posts to
(`prober.build_probe_url`), and that a target with no base URL measures
NOTHING rather than fabricating a failure.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yaml

import litellm_maintainer.cli as cli_module
from litellm_maintainer.classify import INCONCLUSIVE, Outcome, classify
from litellm_maintainer.feed import parse_feed
from litellm_maintainer.journal import (
    append_observation,
    observation_key_map,
    resolve_observation_keys,
)
from litellm_maintainer.plan import plan
from litellm_maintainer.policy import parse_policy
from litellm_maintainer.prober import (
    ProbeTarget,
    build_probe_url,
    live_transport,
    probe_credential,
)
from litellm_maintainer.reduce import HealthState, Observation, OfferingHealth, reduce

NOW = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)


def _offering_raw(offering_id: str, *, coding_score: float | None = 50.0) -> dict:
    provider_id, _, model_id = offering_id.partition(":")
    return {
        "id": offering_id,
        "provider": {"id": provider_id},
        "provider_model_id": model_id,
        "capabilities": ["tool_use"],
        "endpoint": {
            "protocol": "openai_chat_completions",
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
            "providers": [
                {"id": "acme", "name": "Acme", "authentication": {"credential_hint": "ACME_API_KEY"}}
            ],
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
        "pacing": {"default": {"concurrency": 1, "minimum_interval_seconds": 1}},
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


GATEWAY_FAILURE = Outcome(bucket="self_healing", reset_at=None, reason="gateway_error")


# --- the key map ---------------------------------------------------------


def test_a_discovered_offerings_alias_maps_to_its_offering_id():
    feed = _feed(_offering_raw("acme:widget"))
    policy = _policy()

    key_map = observation_key_map(feed=feed, policy=policy)

    assert key_map["claude-acme-widget"] == "acme:widget"


def test_a_declared_offerings_alias_maps_to_itself():
    feed = _feed()
    policy = _policy(
        declared=[{"alias": "claude-direct", "litellm_params": {"model": "anthropic/direct"}}]
    )

    key_map = observation_key_map(feed=feed, policy=policy)

    assert key_map["claude-direct"] == "claude-direct"


def test_an_alias_the_key_map_does_not_know_passes_through_unchanged():
    observation = Observation(
        offering_id="claude-left-policy", observed_at=NOW, outcome=GATEWAY_FAILURE
    )

    resolved = resolve_observation_keys([observation], {})

    assert resolved == [observation]


# --- the seam, end to end (pure) -----------------------------------------


def test_a_proxy_recorded_failure_on_a_discovered_offering_excludes_it_from_the_next_config():
    """The proxy records the Alias; the next `plan` must drop the
    Offering. This is stories 36 and 37 for a Discovered Offering —
    the case the Declared-only test in `test_schedule.py` cannot see,
    because a Declared Offering's Alias IS its Health State key.

    Uses `openrouter`, a provider with a registered translation rule,
    because `plan` skips a provider `translate.TRANSLATION_RULES` does
    not know."""
    feed = parse_feed(
        {
            "schema_version": "1",
            "providers": [
                {
                    "id": "openrouter",
                    "name": "OpenRouter",
                    "authentication": {"credential_hint": "OPENROUTER_API_KEY"},
                }
            ],
            "models": [
                _offering_raw("openrouter:widget"),
                _offering_raw("openrouter:gadget"),
            ],
        }
    )
    policy = parse_policy(
        _policy_dict(
            providers={"openrouter": {"mode": "all"}},
            naming={
                "alias_prefix": "claude-",
                "provider_labels": {"openrouter": "openrouter"},
                "alias_overrides": {},
            },
        )
    )

    proxy_wrote = Observation(
        offering_id="claude-openrouter-widget", observed_at=NOW, outcome=GATEWAY_FAILURE
    )
    observations = resolve_observation_keys(
        [proxy_wrote], observation_key_map(feed=feed, policy=policy)
    )
    next_health = reduce(
        prior=HealthState(offerings={}),
        outcomes={},
        observations=observations,
        admitted={"openrouter:widget", "openrouter:gadget"},
        passthrough_auth=frozenset(),
        now=NOW,
    )

    record = next_health.offerings["openrouter:widget"]
    assert record.excluded is True
    assert record.reason == "gateway_error"
    assert "claude-openrouter-widget" not in next_health.offerings

    result = plan(feed=feed, policy=policy, health=next_health.offerings, now=NOW)
    assert "openrouter:widget" in result.report.excluded
    names = [e["model_name"] for e in result.config["model_list"]]
    assert names == ["claude-openrouter-gadget"]


# --- the seam, through the `probe` command --------------------------------


def test_the_probe_command_folds_a_proxy_recorded_alias_into_health_state(
    tmp_path: Path, monkeypatch
):
    """CLI-level: `cmd_probe` must resolve the Journal's Alias to the
    Offering id before `reduce`, and the Health State it writes must
    key the record by the Offering id."""
    from litellm_maintainer.health import read_health
    from litellm_maintainer.paths import health_path, journal_path

    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(yaml.safe_dump(_policy_dict()))
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(
        json.dumps(
            {
                "schema_version": "1",
                "providers": [{"id": "acme", "name": "Acme"}],
                "models": [_offering_raw("acme:widget")],
            }
        )
    )

    append_observation(
        journal_path(tmp_path),
        Observation(
            offering_id="claude-acme-widget",
            observed_at=NOW - timedelta(minutes=1),
            outcome=GATEWAY_FAILURE,
        ),
    )

    # No network call: the Probe sweep itself measures nothing here.
    # An Inconclusive outcome per target leaves Health State untouched,
    # so the Journal entry is the only health input this run has.
    monkeypatch.setattr(
        cli_module,
        "probe_offerings",
        lambda targets, *, pacing, transport, now: {
            t.key: Outcome(bucket=INCONCLUSIVE, reset_at=None, reason="unmeasured")
            for t in targets
        },
    )

    exit_code = cli_module.main(
        [
            "probe",
            "--feed",
            str(feed_path),
            "--policy",
            str(policy_path),
            "--home",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    written = read_health(health_path(tmp_path))
    assert "claude-acme-widget" not in written.offerings
    record = written.offerings["acme:widget"]
    assert record.excluded is True
    assert record.reason == "gateway_error"


# --- the Prober's live-call contract --------------------------------------


def test_a_discovered_probe_sends_the_providers_own_credential():
    feed = _feed(_offering_raw("acme:widget"))
    target = ProbeTarget(
        key="acme:widget", provider_id="acme", offering=feed.offerings[0]
    )
    resolver = {"ACME_API_KEY": "sk-acme-real"}.get

    assert probe_credential(target, feed=feed, resolver=resolver) == "sk-acme-real"


def test_a_declared_probe_resolves_an_environment_credential_reference():
    feed = _feed()
    policy = _policy(
        declared=[
            {
                "alias": "claude-direct",
                "litellm_params": {
                    "model": "acme/direct",
                    "api_base": "https://direct.example.invalid/v1",
                    "api_key": "os.environ/DIRECT_API_KEY",
                },
            }
        ]
    )
    target = ProbeTarget(
        key="claude-direct", provider_id="acme", declared=policy.declared[0]
    )
    resolver = {"DIRECT_API_KEY": "sk-direct"}.get

    assert probe_credential(target, feed=feed, resolver=resolver) == "sk-direct"


def test_a_probe_with_no_credential_source_sends_none():
    policy = _policy(
        declared=[{"alias": "claude-direct", "litellm_params": {"model": "anthropic/direct"}}]
    )
    target = ProbeTarget(
        key="claude-direct", provider_id="anthropic", declared=policy.declared[0]
    )

    assert probe_credential(target, feed=_feed(), resolver={}.get) is None


def test_the_live_probe_transport_resolves_the_providers_credential(
    tmp_path: Path, monkeypatch
):
    """`cli._probe_live_transport` must hand each Probe the credential
    the Feed provider's `credential_hint` names — never the proxy's
    `LITELLM_MASTER_KEY`, which no provider accepts. An earlier version
    sent the master key on every direct probe, so a live sweep would
    have classified every Discovered Offering `needs_operator`."""
    feed = _feed(_offering_raw("acme:widget"))
    target = ProbeTarget(key="acme:widget", provider_id="acme", offering=feed.offerings[0])

    env_path = tmp_path / ".env.local"
    env_path.write_text("ACME_API_KEY=sk-acme-real\nLITELLM_MASTER_KEY=sk-master\n")
    monkeypatch.delenv("ACME_API_KEY", raising=False)

    sent: dict = {}

    def spy_transport(t, *, credential=None, timeout=15.0):
        sent["credential"] = credential
        return None

    monkeypatch.setattr(cli_module, "live_transport", spy_transport)
    transport = cli_module._probe_live_transport(feed, env_path)
    transport(target)

    assert sent["credential"] == "sk-acme-real"


def test_a_gemini_probe_posts_to_the_openai_compatible_path():
    """The Feed states `gemini_generate_content` with the native base
    URL. The probe payload is OpenAI-shaped, so posting it to the
    native base 404s; the same host serves an OpenAI-compatible
    surface under `/openai`. Asserted on the built URL only — this
    build made no live call."""
    url = build_probe_url(
        "https://generativelanguage.googleapis.com/v1beta",
        "gemini",
        "gemini_generate_content",
    )
    assert url == (
        "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    )


def test_a_declared_offering_with_no_base_url_measures_nothing():
    """A Declared Offering with no `api_base` cannot be called
    directly. The attempt must classify Inconclusive, so Health State
    stays untouched. An earlier version returned `transport="timeout"`,
    which classifies `self_healing` — a fabricated failure that
    Excluded every such Offering without one network packet sent."""
    policy = _policy(
        declared=[{"alias": "claude-direct", "litellm_params": {"model": "anthropic/direct"}}]
    )
    target = ProbeTarget(
        key="claude-direct", provider_id="anthropic", declared=policy.declared[0]
    )

    # No base URL, so `live_transport` returns before any network code.
    response = live_transport(target)
    outcome = classify(
        provider="anthropic",
        http_status=response.http_status,
        body=response.body,
        transport=response.transport,
        now=NOW,
    )
    assert outcome.bucket == INCONCLUSIVE

    prior = HealthState(offerings={})
    next_health = reduce(
        prior=prior,
        outcomes={"claude-direct": outcome},
        observations=[],
        admitted={"claude-direct"},
        passthrough_auth=frozenset(),
        now=NOW,
    )
    # Untouched: no exclusion, no failure count, no attempt recorded.
    # Health is untouched. `inconclusive_count` is the one deliberate
    # exception, so that a silent misclassification becomes visible.
    record = next_health.offerings.get("claude-direct", OfferingHealth())
    assert replace(record, inconclusive_count=0) == OfferingHealth()
