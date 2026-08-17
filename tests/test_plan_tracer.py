"""Tests for `plan`, ticket 05: the tracer slice.

Assert external behaviour: which Offerings the baseline filter admits,
what a generated OpenCode Go entry looks like, and that `plan` is pure.
Fixtures are the frozen `tests/fixtures/feed-audited.json` and the
synthetic `tests/fixtures/policy-pinned.yaml`. Both are committed, both
are read-only inputs, and this test writes to neither.
"""

from __future__ import annotations

import builtins
import copy
import json
import os
import socket
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from litellm_maintainer.feed import load_feed, parse_feed
from litellm_maintainer.generate import HEADER, render_config, write_config
from litellm_maintainer.plan import plan
from litellm_maintainer.policy import load_policy, parse_policy

# `HY3_PREVIEW_ALIAS` is a newly-admitted OpenCode Go Discovered
# Offering, absent from the frozen config, unrelated to ticket 05 or the
# ChatGPT-seat change — see its docstring in test_translate.py.

FIXTURES = Path(__file__).parent / "fixtures"
FEED_PATH = FIXTURES / "feed-audited.json"
EXPECTED_CONFIG_PATH = FIXTURES / "expected-config.yaml"
# Synthetic and committed. Never the operator's own Policy.
PINNED_POLICY_PATH = FIXTURES / "policy-pinned.yaml"

OPENCODE_GO_BASE_URL = "https://opencode.ai/zen/go/v1"

# The seven Aliases the frozen config writes with the anthropic-shaped
# prefix and the `/messages` base URL. Ticket 05 must move every one of
# them to the generic `openai/` prefix and the plain OpenCode Go base
# URL — one of the intended differences ticket 10 expects.
ANTHROPIC_PREFIXED_ALIASES = (
    "claude-opencode-go-minimax-m3",
    "claude-opencode-go-minimax-m2.7",
    "claude-opencode-go-minimax-m2.5",
    "claude-opencode-go-qwen3.7-max",
    "claude-opencode-go-qwen3.7-plus",
    "claude-opencode-go-qwen3.6-plus",
    "claude-opencode-go-qwen3.5-plus",
)


@pytest.fixture(scope="module")
def feed():
    return load_feed(FEED_PATH)


@pytest.fixture(scope="module")
def policy():
    return load_policy(PINNED_POLICY_PATH)


@pytest.fixture(scope="module")
def frozen_config():
    with open(EXPECTED_CONFIG_PATH) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def frozen_opencode_go_entries(frozen_config):
    return {
        entry["model_name"]: entry["litellm_params"]
        for entry in frozen_config["model_list"]
        if entry["model_name"].startswith("claude-opencode-go-")
    }


@pytest.fixture(scope="module")
def plan_result(feed, policy):
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    return plan(feed=feed, policy=policy, health={}, now=now)


@pytest.fixture(scope="module")
def generated_entries(plan_result):
    return {
        entry["model_name"]: entry["litellm_params"]
        for entry in plan_result.config["model_list"]
    }


def _offering(feed, offering_id: str):
    offering = feed.offering(offering_id)
    assert offering is not None, f"fixture offering {offering_id!r} not found"
    return offering


# --- The baseline filter --------------------------------------------
#
# Warning: do not write these against a real Offering in
# `feed-audited.json`. Every Offering there that fails the baseline
# also fails the pricing filter or the quality gate, so such a test
# passes with the baseline filter deleted. Mutation testing on
# 2026-07-25 proved it: removing `_passes_baseline` from `plan` broke
# no test. Each test below uses a synthetic Offering whose only fault
# is its capability list.


def _baseline_feed(*, capabilities: tuple[str, ...]):
    """A one-Offering Feed. The Offering passes every filter but the
    baseline: it is listed, available, scores 90, and sits on a
    provider the Policy below takes whole with no pricing filter.
    """
    raw = {
        "schema_version": "test",
        "providers": [
            {
                "id": "opencode-go",
                "name": "OpenCode Go",
                "default_base_url": "https://opencode-go.example/v1",
                "authentication": {},
            }
        ],
        "models": [
            {
                "id": "opencode-go:baseline-subject",
                "provider": {"id": "opencode-go"},
                "provider_model_id": "baseline-subject",
                "capabilities": list(capabilities),
                "endpoint": {
                    "base_url": "https://opencode-go.example/v1",
                    "model": "baseline-subject",
                },
                "pricing": {"kind": "subscription_included"},
                "availability": {
                    "status": "available",
                    "last_checked_at": "2026-07-25T00:00:00Z",
                    "last_success_at": "2026-07-25T00:00:00Z",
                    "stale_after_seconds": 86400,
                },
                "quality": {"coding_score": 90},
                "policy": {"visibility": "listed", "tags": []},
            }
        ],
    }
    return parse_feed(raw)


def _baseline_policy():
    return parse_policy(
        {
            "providers": {"opencode-go": {"mode": "all"}},
            "quality": {"minimum_coding_score": 18},
            "approved_candidates": [],
            "naming": {
                "alias_prefix": "claude-",
                "provider_labels": {"opencode-go": "opencode-go"},
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


def test_an_offering_that_clears_the_baseline_does_appear():
    """The control. Without this, a test that asserts absence proves
    only that the builder above is broken."""
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    result = plan(
        feed=_baseline_feed(capabilities=("chat", "tool_use")),
        policy=_baseline_policy(),
        health={},
        now=now,
    )
    assert result.report.admitted == ("opencode-go:baseline-subject",)


def test_an_offering_without_tool_use_does_not_appear():
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    result = plan(
        feed=_baseline_feed(capabilities=("chat", "reasoning")),
        policy=_baseline_policy(),
        health={},
        now=now,
    )
    assert result.report.admitted == ()
    assert result.report.candidates == ()


@pytest.mark.parametrize(
    "excluded_capability",
    [
        "image_generation",
        "text_to_speech",
        "speech_to_text",
        "video_generation",
        "embeddings",
        "moderation",
        "safety",
    ],
)
def test_an_offering_with_an_excluded_capability_does_not_appear(excluded_capability):
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    result = plan(
        feed=_baseline_feed(capabilities=("chat", "tool_use", excluded_capability)),
        policy=_baseline_policy(),
        health={},
        now=now,
    )
    assert result.report.admitted == ()


def test_vision_is_not_an_excluded_capability():
    """`vision` accepts an image; `image_generation` produces one. The
    two are different, and several admitted Offerings carry `vision`."""
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    result = plan(
        feed=_baseline_feed(capabilities=("chat", "tool_use", "vision")),
        policy=_baseline_policy(),
        health={},
        now=now,
    )
    assert result.report.admitted == ("opencode-go:baseline-subject",)


def test_the_named_offerings_in_the_audited_snapshot_lack_tool_use(feed):
    """The rule matters against the real Feed too: these Offerings do
    lack `tool_use`. They are kept out of the config above by other
    filters as well, which is why the rule itself needs the synthetic
    tests above."""
    for offering_id in (
        "openrouter:google/gemini-3.1-flash-lite-image",
        "qwencloud:qwen3-tts-vd-2026-01-26",
        "qwencloud:fun-asr-mtl",
    ):
        assert "tool_use" not in _offering(feed, offering_id).capabilities


# --- The generic OpenCode Go translation rule ------------------------


def test_every_generated_opencode_go_entry_uses_the_generic_openai_prefix(generated_entries):
    opencode_go_entries = {
        name: params
        for name, params in generated_entries.items()
        if name.startswith("claude-opencode-go-")
    }
    assert opencode_go_entries, "expected at least one generated OpenCode Go entry"

    for model_name, litellm_params in opencode_go_entries.items():
        assert litellm_params["model"].startswith("openai/"), model_name
        assert litellm_params["api_base"] == OPENCODE_GO_BASE_URL, model_name
        assert litellm_params["api_key"] == "os.environ/OPENCODE_API_KEY", model_name


# --- The generated-by header -----------------------------------------


def test_the_written_file_carries_the_generated_header(tmp_path, plan_result):
    out_path = tmp_path / "generated-config.yaml"
    write_config(plan_result.config, out_path)
    written = out_path.read_text()
    assert written.startswith(HEADER)
    assert "generated" in written.lower()
    assert "do not edit by hand" in written.lower()


def test_render_config_carries_the_generated_header(plan_result):
    rendered = render_config(plan_result.config)
    assert rendered.startswith(HEADER)


# --- The acceptance test: OpenCode Go matches the frozen config ------


# `test_generated_opencode_go_entries_match_the_frozen_config_with_one_difference` stood here (RETIRED).
#
# It compared generated output to `fixtures/expected-config.yaml`, the
# proxy the operator built and verified BY HAND on 2026-07-25. The
# Policy that produced that file was never committed and no surviving
# copy reproduces it. The live Policy matches none of its 78 Aliases,
# because it now sets `alias_prefix: ""` and `alias_separator: "--"`.
#
# The test therefore demanded a superseded proxy from a Policy that no
# longer exists. See the longer note in test_acceptance.py.

def test_the_two_offerings_that_return_gateway_errors_stay_withheld(feed, policy, generated_entries):
    """Both are Withheld in the operator's Policy after consistent 502
    gateway errors.

    Warning: absence alone proves nothing here. Both Offerings also
    carry no coding score, so the Candidate path would keep them out
    even with the Withheld filter deleted — mutation testing on
    2026-07-25 showed this test passing with that filter removed. So
    approve both as Candidates first. Then Withheld is the only rule
    left that can keep them out.
    """
    withheld_ids = ("opencode-go:mimo-v2-omni", "opencode-go:mimo-v2-pro")
    for offering_id in withheld_ids:
        assert offering_id in policy.withheld

    assert "claude-opencode-go-mimo-v2-omni" not in generated_entries
    assert "claude-opencode-go-mimo-v2-pro" not in generated_entries

    approved = replace(
        policy, approved_candidates=tuple(policy.approved_candidates) + withheld_ids
    )
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    result = plan(feed=feed, policy=approved, health={}, now=now)
    for offering_id in withheld_ids:
        assert offering_id not in result.report.admitted, offering_id


# --- Purity -----------------------------------------------------------


def test_plan_is_pure_same_inputs_same_result(feed, policy):
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    first = plan(feed=feed, policy=policy, health={}, now=now)
    second = plan(feed=feed, policy=policy, health={}, now=now)
    assert first.config == second.config
    assert first.report == second.report
    assert first.refusal == second.refusal


def test_plan_writes_nothing(feed, policy, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    plan(feed=feed, policy=policy, health={}, now=now)
    assert list(tmp_path.iterdir()) == []


def test_plan_opens_no_file_reads_no_clock_and_reads_no_environment(feed, policy, monkeypatch):
    """`plan` is pure (the brief, "Package layout"). Replace every
    input-output entry point with a call that fails, then run it.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError(f"plan performed input or output: {args!r}")

    monkeypatch.setattr(builtins, "open", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(time, "time", forbidden)
    monkeypatch.setattr(os, "getenv", forbidden)
    monkeypatch.setattr(os, "environ", _FailingEnviron())

    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    result = plan(feed=feed, policy=policy, health={}, now=now)
    assert result.config["model_list"]


class _FailingEnviron(dict):
    def __getitem__(self, key):
        raise AssertionError(f"plan read the environment: {key!r}")

    def get(self, key, default=None):
        raise AssertionError(f"plan read the environment: {key!r}")


def test_plan_does_not_mutate_the_feed_document_or_the_policy(policy):
    """A shared mutable input is a real hazard: the Generator runs
    `plan` more than once per process. Compare a deep copy of both
    inputs before and after.
    """
    with open(FEED_PATH) as f:
        raw = json.load(f)
    raw_before = copy.deepcopy(raw)
    policy_before = copy.deepcopy(policy)

    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    plan(feed=parse_feed(raw), policy=policy, health={}, now=now)

    assert raw == raw_before
    assert policy == policy_before


def test_editing_one_result_does_not_change_the_next_run(feed, policy):
    """No entry shares a mutable object with Policy or with a later
    run."""
    now = datetime(2026, 7, 25, tzinfo=timezone.utc)
    first = plan(feed=feed, policy=policy, health={}, now=now)
    first.config["model_list"][0]["litellm_params"]["model"] = "poisoned/model"

    second = plan(feed=feed, policy=policy, health={}, now=now)

    assert all(
        entry["litellm_params"].get("model") != "poisoned/model"
        for entry in second.config["model_list"]
    )
    assert all(
        declared.litellm_params.get("model") != "poisoned/model" for declared in policy.declared
    )


# --- The Generated Config reads as a document, not a dump -------------
#
# `render_config` groups `model_list` under a heading per group and puts
# a note beside each Alias. Comments carry no meaning to litellm, so
# every test here also asserts the parsed document is unchanged.


def _annotated_config():
    from litellm_maintainer.plan import AliasAnnotation

    config = {
        "model_list": [
            {"model_name": "a-one", "litellm_params": {"model": "p/one"}},
            {"model_name": "a-two", "litellm_params": {"model": "p/two"}},
            {"model_name": "b-one", "litellm_params": {"model": "q/one"}},
        ],
        "general_settings": {"forward_client_headers_to_llm_api": True},
    }
    annotations = {
        "a-one": AliasAnnotation(group="Provider A", note="70.1 / 50.2 / 37.4 — 1M ctx"),
        "a-two": AliasAnnotation(group="Provider A", note=None),
        "b-one": AliasAnnotation(group="Provider B", note="256K ctx"),
    }
    return config, annotations


def test_an_annotated_render_parses_to_the_same_document():
    config, annotations = _annotated_config()
    assert yaml.safe_load(render_config(config, annotations)) == config


def test_a_heading_is_written_once_per_group_not_once_per_alias():
    config, annotations = _annotated_config()
    text = render_config(config, annotations)
    assert text.count("# Provider A") == 1
    assert text.count("# Provider B") == 1
    assert text.index("# Provider A") < text.index("# Provider B")


def test_a_note_is_written_beside_its_own_alias():
    config, annotations = _annotated_config()
    lines = render_config(config, annotations).splitlines()
    at = lines.index("  - model_name: a-one")
    assert lines[at + 1].strip() == "# 70.1 / 50.2 / 37.4 — 1M ctx"
    # `a-two` carries no note, so its next line is already the params.
    at_two = lines.index("  - model_name: a-two")
    assert lines[at_two + 1].strip() == "litellm_params:"


def test_every_other_top_level_key_still_renders():
    config, annotations = _annotated_config()
    parsed = yaml.safe_load(render_config(config, annotations))
    assert parsed["general_settings"] == {"forward_client_headers_to_llm_api": True}


def test_no_annotations_renders_the_plain_dump():
    config, _ = _annotated_config()
    assert render_config(config) == render_config(config, {})
    assert yaml.safe_load(render_config(config)) == config


def test_an_alias_with_no_annotation_still_renders():
    """A missing annotation must not drop the entry or raise."""
    config, annotations = _annotated_config()
    annotations.pop("a-two")
    parsed = yaml.safe_load(render_config(config, annotations))
    assert parsed == config


def test_write_config_writes_the_annotated_form(tmp_path):
    config, annotations = _annotated_config()
    path = tmp_path / "config.yaml"
    write_config(config, path, annotations)
    text = path.read_text()
    assert text.startswith(HEADER)
    assert "# Provider A" in text
    assert yaml.safe_load(text) == config
