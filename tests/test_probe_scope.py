"""Scoping a probe sweep to one provider.

The live probe path had three separate defects, each repaired without a
live call to confirm it. So the first real sweep should cost one cheap
provider, not every provider at once.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from litellm_maintainer.feed import load_feed
from litellm_maintainer.paths import policy_path
from litellm_maintainer.policy import load_policy
from litellm_maintainer.prober import UnknownProviderError, build_worklist
from litellm_maintainer.reduce import HealthState

from conftest import FIXTURES

NOW = datetime(2026, 7, 26, 12, 0, 0, tzinfo=timezone.utc)


def _operator_policy():
    path = policy_path()
    if not path.exists():
        pytest.skip("the operator's Policy lives outside the repository")
    return load_policy(path)


def _feed():
    return load_feed(FIXTURES / "feed-current.json")


def _worklist(provider=None, health=None):
    return build_worklist(
        feed=_feed(),
        policy=_operator_policy(),
        health=health or HealthState(offerings={}),
        now=NOW,
        provider=provider,
    )


def test_a_scoped_sweep_probes_only_that_providers_offerings():
    scoped = _worklist(provider="groq")
    assert scoped.targets, "groq must have something to probe"
    assert {t.provider_id for t in scoped.targets} == {"groq"}


def test_a_scoped_sweep_is_smaller_than_the_whole_worklist():
    assert len(_worklist(provider="groq").targets) < len(_worklist().targets)


def test_an_unscoped_sweep_still_probes_every_provider():
    everything = _worklist()
    assert len({t.provider_id for t in everything.targets}) > 1


def test_an_unknown_provider_fails_and_names_it():
    with pytest.raises(UnknownProviderError) as caught:
        _worklist(provider="no-such-provider")
    message = str(caught.value)
    assert "no-such-provider" in message
    assert "groq" in message, "the message must name the providers Policy knows"


def test_a_scoped_sweep_leaves_health_state_for_every_other_provider_admitted():
    """Scoping must not narrow the admitted set.

    `reduce` discards a record for an Offering Policy no longer admits.
    So a scoped sweep that also narrowed `admitted` would DELETE the
    Health State of every provider outside the scope, including every
    recorded reset time.
    """
    scoped = _worklist(provider="groq")
    everything = _worklist()
    assert scoped.admitted == everything.admitted
    assert any(not key.startswith("groq:") for key in scoped.admitted)


def test_a_scoped_sweep_keeps_every_other_rule_unchanged():
    scoped = _worklist(provider="groq")
    everything = _worklist()
    assert scoped.skipped_withheld == everything.skipped_withheld
    assert scoped.skipped_passthrough == everything.skipped_passthrough


def test_a_scoped_sweep_still_skips_a_fresh_offering():
    everything = _worklist(provider="groq")
    first = everything.targets[0].key
    fresh = HealthState(
        offerings={
            first: __import__(
                "litellm_maintainer.reduce", fromlist=["OfferingHealth"]
            ).OfferingHealth(
                excluded=False,
                reason="answered",
                bucket="answered",
                reset_at=None,
                last_success_at=NOW,
                last_attempt_at=NOW,
                failure_count=0,
            )
        }
    )
    scoped = _worklist(provider="groq", health=fresh)
    assert first not in {t.key for t in scoped.targets}
    assert first in scoped.skipped_fresh


def test_the_scoped_dry_run_reports_the_scope_and_calls_nothing(tmp_path, capsys):
    from litellm_maintainer.cli import main

    path = policy_path()
    if not path.exists():
        pytest.skip("the operator's Policy lives outside the repository")

    exit_code = main(
        [
            "probe",
            "--dry-run",
            "--provider",
            "groq",
            "--policy",
            str(path),
            "--feed",
            str(FIXTURES / "feed-current.json"),
            "--home",
            str(tmp_path),
        ]
    )
    out = capsys.readouterr().out
    assert exit_code == 0
    assert "Scoped to provider: groq" in out
    assert "(groq)" in out
    assert "(openrouter)" not in out
    assert not (tmp_path / "state" / "health.json").exists()


def test_an_unknown_provider_exits_non_zero_from_the_command(tmp_path, capsys):
    from litellm_maintainer.cli import main

    path = policy_path()
    if not path.exists():
        pytest.skip("the operator's Policy lives outside the repository")

    exit_code = main(
        [
            "probe",
            "--dry-run",
            "--provider",
            "no-such-provider",
            "--policy",
            str(path),
            "--feed",
            str(FIXTURES / "feed-current.json"),
            "--home",
            str(tmp_path),
        ]
    )
    assert exit_code == 1
    assert "no-such-provider" in capsys.readouterr().err
