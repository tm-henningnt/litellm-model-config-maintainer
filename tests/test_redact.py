"""Tests for credential redaction.

Every dotenv file here is written by the test itself, with invented
values. No test reads the operator's real `.env.local`.
"""

from __future__ import annotations

from litellm_maintainer import redact as redact_mod


def write_env(tmp_path, lines):
    path = tmp_path / ".env.test"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_a_mapped_value_is_replaced(tmp_path):
    env_path = write_env(tmp_path, ["FAKE_API_KEY=invented-secret-value-12345"])
    mapping = redact_mod.build_redaction_map(env_path)
    text = "calling with key invented-secret-value-12345 now"
    result = redact_mod.redact(text, mapping)
    assert "invented-secret-value-12345" not in result
    assert "<REDACTED:FAKE_API_KEY>" in result


def test_the_longest_of_two_overlapping_values_wins(tmp_path):
    env_path = write_env(
        tmp_path,
        [
            "SHORT_PART=abcdefgh12",
            "LONG_WHOLE=abcdefgh12-extra-tail-9999",
        ],
    )
    mapping = redact_mod.build_redaction_map(env_path)
    text = "token abcdefgh12-extra-tail-9999 seen"
    result = redact_mod.redact(text, mapping)
    assert "<REDACTED:LONG_WHOLE>" in result
    assert "<REDACTED:SHORT_PART>" not in result
    assert "abcdefgh12-extra-tail-9999" not in result


def test_a_bare_sk_token_with_no_map_entry_is_still_caught(tmp_path):
    env_path = write_env(tmp_path, ["UNRELATED_KEY=totally-different-value-000"])
    mapping = redact_mod.build_redaction_map(env_path)
    text = "auth failed for sk-invented1234567890fakeToken"
    result = redact_mod.redact(text, mapping)
    assert "sk-invented1234567890fakeToken" not in result
    assert "<REDACTED:sk-token>" in result


def test_a_short_value_is_not_mapped(tmp_path):
    env_path = write_env(tmp_path, ["SHORT_KEY=abc123"])
    mapping = redact_mod.build_redaction_map(env_path)
    assert mapping == {}


def test_a_quoted_value_loses_its_quotes(tmp_path):
    """A dotenv file may quote a value. Quotes that survive defeat both callers.

    The bare credential would appear unredacted, because the map holds the
    quoted form. The same value sent as a bearer token would carry the
    quotes and the provider would answer 401.
    """
    env_path = write_env(tmp_path, ['QUOTED_KEY="secret-value-1234567890"'])

    assert redact_mod.parse_dotenv_file(env_path) == {
        "QUOTED_KEY": "secret-value-1234567890"
    }

    result = redact_mod.redact(
        "auth failed for secret-value-1234567890",
        redact_mod.build_redaction_map(env_path),
    )
    assert "secret-value-1234567890" not in result
    assert "<REDACTED:QUOTED_KEY>" in result
