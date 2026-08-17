"""Tests for the ChatGPT system-role-fix hook.

The hook derives its Alias set from the config file the proxy loaded
(`LITELLM_CONFIG_PATH`). These tests write a small config to a temp
directory and assert on behaviour an operator would recognise: an
Alias present in the config gets its system role rewritten, and an
Alias absent from the config does not.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "providers"))


CONFIG_YAML = """
model_list:
  - model_name: claude-gpt-5.6-sol
    litellm_params:
      model: chatgpt/gpt-5.6-sol
  - model_name: some-other-model
    litellm_params:
      model: anthropic/claude-opus-4.8
"""


def _make_hook(config_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LITELLM_CONFIG_PATH", str(config_path))
    import chatgpt_role_fix

    importlib.reload(chatgpt_role_fix)
    return chatgpt_role_fix.ChatGPTSystemRoleFix()


def test_an_alias_present_in_the_config_gets_its_system_role_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_YAML)
    hook = _make_hook(config_path, monkeypatch)

    data = {
        "model": "claude-gpt-5.6-sol",
        "messages": [{"role": "system", "content": "be helpful"}],
    }

    result = asyncio.run(hook.async_pre_call_hook(None, None, data, "completion"))

    assert result["messages"][0]["role"] == "developer"


def test_an_alias_absent_from_the_config_does_not_get_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_YAML)
    hook = _make_hook(config_path, monkeypatch)

    data = {
        "model": "some-other-model",
        "messages": [{"role": "system", "content": "be helpful"}],
    }

    result = asyncio.run(hook.async_pre_call_hook(None, None, data, "completion"))

    assert result["messages"][0]["role"] == "system"


def test_a_new_alias_added_to_the_config_is_picked_up_without_a_code_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_YAML)
    hook = _make_hook(config_path, monkeypatch)

    # Add a seventh alias the hook never held in a hard-coded list.
    config_path.write_text(
        CONFIG_YAML
        + """
  - model_name: claude-gpt-5.7-nova
    litellm_params:
      model: chatgpt/gpt-5.7-nova
"""
    )
    import os
    import time

    new_time = time.time() + 5
    os.utime(config_path, (new_time, new_time))

    data = {
        "model": "claude-gpt-5.7-nova",
        "messages": [{"role": "system", "content": "be helpful"}],
    }

    result = asyncio.run(hook.async_pre_call_hook(None, None, data, "completion"))

    assert result["messages"][0]["role"] == "developer"


def test_the_hook_prefers_the_config_litellm_actually_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A worker serves `chatgpt-worker.yaml`, not the `config.yaml` beside it.

    Measured 2026-07-26: every fallback candidate resolved to the wrong
    file, so a worker read a config it does not serve. The main proxy once
    declared the same Aliases, so the wrong file still held the right
    names and the fault hid. When those were retired every worker request
    began failing with "System messages are not allowed".
    """
    served = tmp_path / "chatgpt-worker.yaml"
    served.write_text(CONFIG_YAML)
    # The decoy sits beside it, holds no chatgpt model, and would win every
    # fallback candidate.
    decoy = tmp_path / "config.yaml"
    decoy.write_text(
        "model_list:\n"
        "  - model_name: claude-seat-1\n"
        "    litellm_params:\n"
        "      model: openai/claude-gpt-5.6-sol\n"
    )
    monkeypatch.delenv("LITELLM_CONFIG_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    import chatgpt_role_fix

    importlib.reload(chatgpt_role_fix)
    monkeypatch.setattr(
        chatgpt_role_fix, "_loaded_config_path", lambda: str(served), raising=True
    )
    hook = chatgpt_role_fix.ChatGPTSystemRoleFix()

    data = {
        "model": "claude-gpt-5.6-sol",
        "messages": [{"role": "system", "content": "be helpful"}],
    }
    result = asyncio.run(hook.async_pre_call_hook(None, None, data, "completion"))

    assert result["messages"][0]["role"] == "developer"


def test_a_config_declaring_no_chatgpt_model_warns_rather_than_matching_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
):
    """The silent failure that cost a release. A registered hook that
    matches nothing is a misconfiguration, so it must say so."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "model_list:\n"
        "  - model_name: claude-seat-1\n"
        "    litellm_params:\n"
        "      model: openai/claude-gpt-5.6-sol\n"
    )
    import logging

    with caplog.at_level(logging.WARNING):
        hook = _make_hook(config_path, monkeypatch)

    assert hook._aliases == set()
    assert "rewrites nothing" in caplog.text


def test_the_top_level_anthropic_system_field_becomes_a_developer_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Claude Code sends the system prompt at the top level, not in messages."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_YAML)
    hook = _make_hook(config_path, monkeypatch)

    data = {
        "model": "claude-gpt-5.6-sol",
        "system": "you are terse",
        "messages": [{"role": "user", "content": "hi"}],
    }
    result = asyncio.run(hook.async_pre_call_hook(None, None, data, "completion"))

    assert "system" not in data
    assert result["messages"][0] == {"role": "developer", "content": "you are terse"}
    assert result["messages"][1]["role"] == "user"


def test_a_list_shaped_system_prompt_is_flattened_to_a_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Measured 2026-07-27: the role alone is not enough.

    litellm's Responses bridge turns a string-content system message into
    the top-level `instructions` field, so no system message reaches the
    backend. A list-content one stays an input message, and the backend
    refuses it with `{"detail":"System messages are not allowed"}` — an
    error that names the role and hides the shape.
    """
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_YAML)
    hook = _make_hook(config_path, monkeypatch)

    data = {
        "model": "claude-gpt-5.6-sol",
        "messages": [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "be terse", "cache_control": {"type": "ephemeral"}},
                    {"type": "text", "text": "and correct"},
                ],
            },
            {"role": "user", "content": "hi"},
        ],
    }
    result = asyncio.run(hook.async_pre_call_hook(None, None, data, "completion"))

    assert result["messages"][0] == {"role": "developer", "content": "be terse\n\nand correct"}


def test_a_list_shaped_top_level_system_is_flattened_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Claude Code sends its system prompt this way on every request."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_YAML)
    hook = _make_hook(config_path, monkeypatch)

    data = {
        "model": "claude-gpt-5.6-sol",
        "system": [{"type": "text", "text": "be terse"}],
        "messages": [{"role": "user", "content": "hi"}],
    }
    result = asyncio.run(hook.async_pre_call_hook(None, None, data, "completion"))

    assert result["messages"][0] == {"role": "developer", "content": "be terse"}


def test_a_user_turn_keeps_its_content_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Flattening a user turn would discard an image. Only an instruction
    role is flattened, and a list-shaped user turn already works."""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_YAML)
    hook = _make_hook(config_path, monkeypatch)

    blocks = [
        {"type": "text", "text": "what is this"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
    ]
    data = {
        "model": "claude-gpt-5.6-sol",
        "messages": [{"role": "user", "content": list(blocks)}],
    }
    result = asyncio.run(hook.async_pre_call_hook(None, None, data, "completion"))

    assert result["messages"][0]["content"] == blocks


def test_an_already_flat_system_prompt_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(CONFIG_YAML)
    hook = _make_hook(config_path, monkeypatch)

    data = {
        "model": "claude-gpt-5.6-sol",
        "messages": [{"role": "system", "content": "be terse"}],
    }
    result = asyncio.run(hook.async_pre_call_hook(None, None, data, "completion"))

    assert result["messages"][0] == {"role": "developer", "content": "be terse"}


def test_a_block_carrying_no_text_is_skipped_rather_than_stringified():
    import chatgpt_role_fix

    flatten = chatgpt_role_fix.flatten_instruction_content
    assert flatten([{"type": "text", "text": "keep"}, {"type": "image_url"}]) == "keep"
    assert flatten([]) == ""
    assert flatten(None) == ""
    assert flatten("plain") == "plain"
    assert flatten({"type": "text", "text": "one"}) == "one"
