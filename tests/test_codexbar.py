"""`codexbar.py` reads codexbar's own JSON shape.

`tests/fixtures/codexbar-sample.json` is a sanitised copy of a real
`codexbar --format json` capture from 2026-07-28: 12 entries, 9
answering and 3 erroring. Every email in it reads `operator@example.com`
or `CodexBar`; nothing here is a real account. Never re-capture from the
live tool, and never add a real identifier to this file.

A missing or renamed field must be a named failure isolated to one
entry, never a crash and never a guess. Only a document that is not a
JSON list at all fails as a whole, because there is then no entry to
isolate the failure to.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from litellm_maintainer.codexbar import (
    CodexbarShapeError,
    parse_codexbar_document,
)

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = (FIXTURES / "codexbar-sample.json").read_text()


def _reading(document, provider: str):
    return next(r for r in document.readings if r.provider == provider)


# --- The frozen fixture, read whole ---


def test_every_entry_in_the_sample_parses():
    document = parse_codexbar_document(SAMPLE)

    assert len(document.readings) == 12
    assert document.failures == ()


def test_claude_binding_data_survives():
    document = parse_codexbar_document(SAMPLE)
    claude = _reading(document, "claude")

    assert claude.secondary.used_percent == 82
    assert claude.secondary.resets_at == "2026-07-30T19:00:00Z"
    assert claude.primary.used_percent == 4
    assert claude.tertiary is None
    assert claude.updated_at == "2026-07-28T20:52:27Z"
    assert claude.error is None

    fable = next(w for w in claude.extra_windows if w.id == "claude-weekly-scoped-fable")
    assert fable.title == "Fable only"
    assert fable.window.used_percent == 59


def test_clinepass_carries_no_account_email_but_a_provider_id():
    document = parse_codexbar_document(SAMPLE)
    clinepass = _reading(document, "clinepass")

    assert clinepass.identity.provider_id == "clinepass"
    assert clinepass.identity.account_email is None
    assert clinepass.tertiary.used_percent == 100
    assert clinepass.tertiary.resets_at == "2026-08-03T10:19:34Z"
    assert clinepass.secondary.resets_at is None


def test_opencodego_has_no_identity_object_at_all():
    """Measured 2026-07-28: OpenCode Go's entry states no 'identity' key.

    `provider_id` falls back to the entry's own top-level `provider`
    field, and `source_key` reads with an empty account email, matching
    Policy's example `"codexbar:opencodego/"`.
    """
    document = parse_codexbar_document(SAMPLE)
    opencodego = _reading(document, "opencodego")

    assert opencodego.identity.provider_id == "opencodego"
    assert opencodego.identity.account_email is None
    assert opencodego.source_key == "codexbar:opencodego/"
    assert opencodego.primary.used_percent == 0


def test_gemini_source_key_carries_its_account_email():
    document = parse_codexbar_document(SAMPLE)
    gemini = _reading(document, "gemini")

    assert gemini.source_key == "codexbar:gemini/operator@example.com"


def test_a_provider_with_no_windows_yields_no_headroom():
    """Measured 2026-07-28: OpenRouter and DeepSeek both answer with a
    null primary, secondary and tertiary. Neither must read as 0%."""
    document = parse_codexbar_document(SAMPLE)

    for provider in ("openrouter", "deepseek"):
        reading = _reading(document, provider)
        assert reading.primary is None
        assert reading.secondary is None
        assert reading.tertiary is None
        assert reading.error is None


def test_error_entries_carry_no_windows_and_no_account_email():
    document = parse_codexbar_document(SAMPLE)

    for provider in ("openai", "azureopenai", "cursor"):
        reading = _reading(document, provider)
        assert reading.error is not None
        assert reading.error.kind == "provider"
        assert reading.primary is None
        assert reading.secondary is None
        assert reading.tertiary is None
        assert reading.identity.provider_id == provider
        assert reading.identity.account_email is None


# --- Isolating a bad entry ---


def _entry(**overrides):
    base = {
        "provider": "claude",
        "usage": {
            "identity": {"providerID": "claude"},
            "primary": None,
            "secondary": None,
            "tertiary": None,
            "updatedAt": "2026-07-28T20:00:00Z",
        },
    }
    base.update(overrides)
    return base


def test_a_missing_usage_key_is_isolated_to_its_own_entry():
    good = _entry(provider="gemini")
    bad = {"provider": "claude"}  # no 'usage' and no 'error': the shape check must catch this
    document = parse_codexbar_document(json.dumps([good, bad]))

    assert len(document.readings) == 1
    assert document.readings[0].provider == "gemini"
    assert len(document.failures) == 1
    assert document.failures[0].provider == "claude"


def test_a_renamed_window_key_is_a_named_failure_not_a_guess():
    bad = _entry()
    del bad["usage"]["primary"]
    bad["usage"]["primaryWindow"] = {"usedPercent": 10}  # renamed, not read
    document = parse_codexbar_document(json.dumps([_entry(provider="gemini"), bad]))

    assert len(document.readings) == 1
    assert document.failures[0].provider == "claude"
    assert "primary" in document.failures[0].message


def test_a_window_missing_used_percent_is_a_named_failure():
    bad = _entry()
    bad["usage"]["primary"] = {"windowMinutes": 300}  # no usedPercent
    document = parse_codexbar_document(json.dumps([bad]))

    assert document.readings == ()
    assert "usedPercent" in document.failures[0].message


def test_an_entry_that_is_not_an_object_fails_with_no_provider_name():
    document = parse_codexbar_document(json.dumps([_entry(provider="gemini"), "not-an-object"]))

    assert len(document.readings) == 1
    assert document.failures[0].provider is None


def test_an_identity_missing_provider_id_is_a_named_failure():
    bad = _entry()
    bad["usage"]["identity"] = {"loginMethod": "6"}  # no providerID
    document = parse_codexbar_document(json.dumps([bad]))

    assert document.readings == ()
    assert "providerID" in document.failures[0].message


def test_an_extra_window_missing_its_id_is_a_named_failure():
    bad = _entry()
    bad["usage"]["extraRateWindows"] = [{"title": "Fable only", "window": {"usedPercent": 59}}]
    document = parse_codexbar_document(json.dumps([bad]))

    assert document.readings == ()
    assert "extraRateWindows" in document.failures[0].message


def test_extra_windows_survive_appearing_and_leaving():
    """Measured 2026-07-28: the Claude Reading carried two extra windows
    at 18:48Z and only one at 20:52Z. The parser survives either shape."""
    two_windows = _entry()
    two_windows["usage"]["extraRateWindows"] = [
        {"id": "claude-weekly-scoped-all-model", "title": "All models", "window": {"usedPercent": 82}},
        {"id": "claude-weekly-scoped-fable", "title": "Fable only", "window": {"usedPercent": 59}},
    ]
    one_window = _entry()
    one_window["usage"]["extraRateWindows"] = [
        {"id": "claude-weekly-scoped-fable", "title": "Fable only", "window": {"usedPercent": 59}},
    ]
    zero_windows = _entry()

    for entry, expected_count in ((two_windows, 2), (one_window, 1), (zero_windows, 0)):
        document = parse_codexbar_document(json.dumps([entry]))
        assert len(document.readings[0].extra_windows) == expected_count


# --- Whole-document failures ---


def test_invalid_json_raises_a_shape_error():
    with pytest.raises(CodexbarShapeError):
        parse_codexbar_document("not json at all {{{")


def test_a_top_level_object_instead_of_a_list_raises_a_shape_error():
    with pytest.raises(CodexbarShapeError):
        parse_codexbar_document(json.dumps({"providers": []}))
