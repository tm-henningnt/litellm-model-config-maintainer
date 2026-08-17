"""
Rewrites system prompts for the litellm `chatgpt/` provider.

The ChatGPT subscription backend needs the `developer` role. It does not
accept the `system` role, and litellm does not convert the role for this
provider. Add this hook to convert it.

The hook does three things to a request for a `chatgpt/` model:

1. It moves a top-level Anthropic `system` field into the messages list.
2. It changes every `system` role to `developer`.
3. It flattens the content of that instruction to a plain string.

litellm converts between the Anthropic and OpenAI request shapes on its
own, so no other provider needs this hook.

## Warning: the role is not enough. Flatten the content too

Measured 2026-07-27 against a live worker. A system prompt with
list-shaped content fails, whatever role it carries:

| message | result |
| --- | --- |
| `{"role": "system", "content": "be terse"}` | 200 |
| `{"role": "developer", "content": "be terse"}` | 200 |
| `{"role": "system", "content": [{"type": "text", ...}]}` | 400 |
| `{"role": "developer", "content": [{"type": "text", ...}]}` | 400 |

Both 400s read `{"detail":"System messages are not allowed"}`, which names
the role and hides the real cause.

The cause is in litellm's Responses bridge,
`completion_extras/litellm_responses_transformation/transformation.py`.
A `system` message with string content becomes the top-level
`instructions` field, so no system message reaches the backend at all. A
`system` message with list content takes the other branch and stays an
input message, which this backend refuses.

Claude Code always sends its system prompt as a list of content blocks,
so every Claude Code request failed. A user or assistant turn with
list-shaped content is fine, so this hook flattens an instruction role
only. Flattening a user turn would discard an image.


## Why the hook reads the config file

litellm calls `async_pre_call_hook` before it resolves the alias to a
deployment. The hook therefore sees the alias that the client sent, such
as `claude-gpt-5.6-sol`, and not the resolved `chatgpt/gpt-5.6-sol`. The
hook must know which aliases belong to the `chatgpt/` provider.

An earlier version of this file held that list by hand. Such a list goes
stale without a symptom: a new alias gets no rewrite and the request
fails at the provider. This version reads the same config file that the
proxy loaded, so the two cannot disagree.

## Install

Copy this file next to your `config.yaml`, then register it:

```yaml
litellm_settings:
  callbacks:
    - chatgpt_role_fix.chatgpt_system_role_fix
```

The hook asks litellm for the config path it actually loaded, then falls
back to `LITELLM_CONFIG_PATH`, `CONFIG_FILE_PATH`, `LITELLM_CONFIG_DIR`,
the working directory, and its own directory. If it finds no config file,
it falls back to a prefix test and writes a warning. It never raises an
error while it loads, because an error there stops the proxy from
starting.

## Warning: ask litellm first, or a worker reads the wrong file

Measured 2026-07-26. A proxy started with
`--config chatgpt-worker.yaml` loads that file, but every fallback
candidate above resolves to `config.yaml` in the same directory instead.
The hook then reads a config it does not serve.

That failed silently for one release. The main proxy once declared the
same six Aliases with a `chatgpt/` target. So the hook read the wrong
file and still found the right names. The operator retired those six. The
alias set became empty, and every worker request began to fail with
`{"detail":"System messages are not allowed"}`. Nothing warned, because
reading the file succeeded. It simply held no `chatgpt/` entry.

So the hook now warns when the config it resolved holds no `chatgpt/`
Alias at all. A registered hook that matches nothing is a
misconfiguration, not a valid state.
"""

import os
from pathlib import Path
from typing import Any, Optional, Set, Tuple

import yaml
from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

PROVIDER_PREFIX = "chatgpt/"

# The roles whose content this hook flattens to a plain string. Only an
# instruction role: a user or assistant turn keeps its content blocks,
# because flattening one would discard an image.
_INSTRUCTION_ROLES = ("system", "developer")


def flatten_instruction_content(content: Any) -> str:
    """One plain string from a system prompt in any shape litellm accepts.

    Reads the `text` of each block and joins them with a blank line. Skips
    a block that carries no text, because a system prompt carries text and
    nothing else this backend can use. A `cache_control` marker is
    dropped: it is an Anthropic concept, and this backend does not read it.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        return text if isinstance(text, str) else ""
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n\n".join(part for part in parts if part)
    return str(content)


def _loaded_config_path() -> Optional[str]:
    """The config path litellm itself loaded, when it knows one.

    This is the only candidate that cannot name the wrong file. litellm
    sets it from its own `--config` argument. A worker started with
    `--config chatgpt-worker.yaml` therefore resolves to that file, not to
    the `config.yaml` beside it.

    Imported inside the function, and guarded. At import time this module
    may run before the proxy module is importable, and an error here would
    stop the proxy from starting.
    """
    try:
        from litellm.proxy.proxy_server import user_config_file_path

        return user_config_file_path
    except Exception:  # noqa: BLE001 - a missing attribute must not break a request
        return None


def _candidate_config_paths() -> list[Path]:
    paths: list[Path] = []
    loaded = _loaded_config_path()
    if loaded:
        paths.append(Path(loaded))
    for name in ("LITELLM_CONFIG_PATH", "CONFIG_FILE_PATH"):
        explicit = os.environ.get(name)
        if explicit:
            paths.append(Path(explicit))
    config_dir = os.environ.get("LITELLM_CONFIG_DIR")
    if config_dir:
        paths.append(Path(config_dir) / "config.yaml")
    paths.append(Path.cwd() / "config.yaml")
    paths.append(Path(__file__).resolve().parent / "config.yaml")
    return paths


def _find_config() -> Optional[Path]:
    for path in _candidate_config_paths():
        if path.is_file():
            return path
    return None


def _read_aliases(path: Path) -> Set[str]:
    """Return every model_name whose model targets the chatgpt provider."""
    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    aliases: Set[str] = set()
    for entry in config.get("model_list") or []:
        if not isinstance(entry, dict):
            continue
        params = entry.get("litellm_params") or {}
        target = str(params.get("model") or "")
        name = entry.get("model_name")
        if name and target.startswith(PROVIDER_PREFIX):
            aliases.add(str(name))
    return aliases


class ChatGPTSystemRoleFix(CustomLogger):
    """Converts the system role to the developer role for chatgpt models."""

    def __init__(self) -> None:
        super().__init__()
        self._aliases: Set[str] = set()
        self._signature: Optional[Tuple[str, float]] = None
        self._refresh_aliases()

    def _refresh_aliases(self) -> None:
        """Reload the alias set when the config file changes.

        The proxy restarts on a config change when it runs with --reload,
        so this is usually a no-op. It keeps the hook correct when the
        proxy runs without that flag.
        """
        path = _find_config()
        if path is None:
            if self._signature is not None or not self._aliases:
                verbose_logger.warning(
                    "chatgpt_role_fix: found no config.yaml. The hook now "
                    "matches the %s prefix only. Set LITELLM_CONFIG_PATH.",
                    PROVIDER_PREFIX,
                )
            self._signature = None
            return

        try:
            signature = (str(path), path.stat().st_mtime)
        except OSError:
            return
        if signature == self._signature:
            return

        try:
            self._aliases = _read_aliases(path)
            self._signature = signature
            if self._aliases:
                verbose_logger.debug(
                    "chatgpt_role_fix: matched %d chatgpt aliases from %s",
                    len(self._aliases),
                    path,
                )
            else:
                # A registered hook that matches nothing is a
                # misconfiguration, not a valid state. Reading the wrong
                # config file looks exactly like this, and it used to fail
                # silently: see this module's docstring.
                verbose_logger.warning(
                    "chatgpt_role_fix: %s declares no %s model, so this hook "
                    "rewrites nothing. Every chatgpt request will fail with "
                    "'System messages are not allowed'. Check that this is "
                    "the config this proxy serves; set LITELLM_CONFIG_PATH "
                    "when it is not.",
                    path,
                    PROVIDER_PREFIX,
                )
        except Exception as exc:  # a broken config must not stop the proxy
            verbose_logger.warning("chatgpt_role_fix: cannot read %s: %s", path, exc)

    def _is_chatgpt(self, model: str) -> bool:
        return model.startswith(PROVIDER_PREFIX) or model in self._aliases

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> dict[str, Any]:
        try:
            self._refresh_aliases()
        except Exception:  # never fail a request over the alias cache
            pass

        if not self._is_chatgpt(str(data.get("model", ""))):
            return data

        # The Anthropic Messages API holds the system prompt at the top
        # level. Move it into the messages list so the rename covers it.
        top_level_system = data.pop("system", None)

        messages = data.get("messages")
        if not isinstance(messages, list):
            messages = []

        rewritten: list[Any] = []
        if top_level_system:
            flattened = flatten_instruction_content(top_level_system)
            if flattened:
                rewritten.append({"role": "developer", "content": flattened})

        for message in messages:
            if not isinstance(message, dict):
                rewritten.append(message)
                continue
            updated = dict(message)
            if updated.get("role") in _INSTRUCTION_ROLES:
                updated["role"] = "developer"
                # Flatten the content too. litellm's Responses bridge keeps
                # a list-shaped instruction as an input message, and this
                # backend refuses one. A string reaches it intact.
                updated["content"] = flatten_instruction_content(updated.get("content"))
            rewritten.append(updated)

        data["messages"] = rewritten
        return data


chatgpt_system_role_fix = ChatGPTSystemRoleFix()
