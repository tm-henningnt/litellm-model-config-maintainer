"""The proxy's own not-served refusal reaches Health State and a Route.

The proxy states, on every call it refuses for an Alias it does not
serve, that it does not serve it. That sentence was classified as an
unrecognised failure, folded to Inconclusive, and applied to nothing.

It is the ONLY measurement of the proxy's own view that exists. The
Prober calls providers directly, bypassing the proxy, so no Probe can
produce it and no Probe can confirm it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import litellm_maintainer.cli as cli_module
from litellm_maintainer.classify import (
    INCONCLUSIVE,
    REASON_ALIAS_NOT_SERVED,
    classify,
)
from litellm_maintainer.reduce import HealthState, Observation, OfferingHealth, reduce

NOW = datetime(2026, 8, 1, 12, 0, 0, tzinfo=timezone.utc)

# litellm's own wording, as `lane` recorded it ten times on 2026-07-31.
PROXY_REFUSAL = {
    "error": (
        "anthropic_messages: Invalid model name passed in model=claude-widget. "
        "Call `/v1/models` to view available models for your key."
    )
}


def _classify(body, status=400):
    return classify(provider="acme", http_status=status, body=body, now=NOW)


# --- classify keeps the sentence -----------------------------------------


def test_the_proxys_not_served_refusal_gets_its_own_reason():
    outcome = _classify(PROXY_REFUSAL)

    assert outcome.reason == REASON_ALIAS_NOT_SERVED
    assert outcome.bucket == INCONCLUSIVE


def test_it_excludes_nothing_because_the_offering_is_fine():
    """The Generated Config on disk is older than what this tool
    believes. Removing a working Offering is the wrong repair."""
    outcome = _classify(PROXY_REFUSAL)

    next_health = reduce(
        prior=HealthState(offerings={}),
        outcomes={},
        observations=[
            Observation(offering_id="acme:widget", observed_at=NOW, outcome=outcome)
        ],
        admitted={"acme:widget"},
        passthrough_auth=frozenset(),
        now=NOW,
    )

    record = next_health.offerings["acme:widget"]
    assert record.excluded is False
    assert record.alias_not_served_at == NOW


def test_a_vendor_saying_invalid_model_is_a_different_condition():
    """Matched on the proxy's own sentence. Reading a vendor's refusal
    about ITS model id as not-served would blame the proxy for it."""
    outcome = _classify({"error": {"message": "invalid model format"}}, status=400)

    assert outcome.reason != REASON_ALIAS_NOT_SERVED


# --- no Probe can confirm it ----------------------------------------------


def test_it_asks_for_no_confirming_probe():
    """A Probe bypasses the proxy, so it would answer about the vendor
    and mask the one signal that names this condition."""
    outcome = _classify(PROXY_REFUSAL)

    assert cli_module._needs_confirming(outcome) is False


# --- it reaches a Route ---------------------------------------------------


def test_a_route_publishes_when_the_proxy_last_refused_the_alias():
    from litellm_maintainer.guidance import Route

    route = Route(
        alias="claude-widget",
        offering_id="acme:widget",
        provider_id="acme",
        cost_basis="free",
        available=True,
        entitlement="per_model",
        not_served_at=NOW,
    )

    assert route.as_dict()["not_served_at"] == NOW.isoformat()


def test_a_route_never_claims_the_proxy_does_serve_it():
    """Nothing measures the positive: the proxy says nothing on a
    successful call, and the Prober never asks the proxy at all."""
    from litellm_maintainer.guidance import Route

    payload = Route(
        alias="claude-widget",
        offering_id="acme:widget",
        provider_id="acme",
        cost_basis="free",
        available=True,
        entitlement="per_model",
    ).as_dict()

    assert payload["not_served_at"] is None
    assert "served" not in payload


# --- it survives a round trip through Health State ------------------------


def test_the_timestamp_survives_a_write_and_a_read(tmp_path):
    from litellm_maintainer.health import read_health, write_health

    path = tmp_path / "health.json"
    write_health(
        path,
        HealthState(offerings={"acme:widget": OfferingHealth(alias_not_served_at=NOW)}),
    )

    assert read_health(path).offerings["acme:widget"].alias_not_served_at == NOW


def test_a_record_written_before_the_field_existed_still_reads():
    """`read_health` keeps every record that parses; a missing key is
    not a reason to lose one."""
    from litellm_maintainer.health import _record_from_json

    record = _record_from_json({"excluded": False, "failure_count": 0})

    assert record.alias_not_served_at is None
