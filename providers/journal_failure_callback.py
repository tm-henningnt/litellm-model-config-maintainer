"""
Records real proxy failures into the Observation Journal.

The maintainer learns from real traffic. When a request through the
proxy fails, this hook classifies the failure and appends one entry to
the Observation Journal (CONTEXT.md, "Observation Journal"). It never
writes Health State. Only the maintainer writes Health State, by
reducing the Journal (ADR 0001, "Four files, one writer each").

## What this hook receives, and what it does not

litellm calls `async_log_failure_event(kwargs, response_obj, start_time,
end_time)` on every failed call when the handler is registered as a
callback. `kwargs` is litellm's internal `model_call_details`. Verified
live against litellm 1.93.0 (installed in `.venv`) by registering a spy
`CustomLogger` and driving a real failing call through `litellm.Router`:

- The Alias is `kwargs["litellm_params"]["metadata"]["model_group"]`.
  This is the `model_name` the client asked the proxy for — the Router
  sets it, not the resolved deployment. `kwargs["model"]` is the
  resolved model instead (e.g. `"gpt-4"`), which is the wrong value
  here.
- The failure is `kwargs["exception"]`, a `litellm.exceptions.APIError`
  subclass. Its `.status_code` and `.message` are set. `isinstance` of
  `litellm.exceptions.Timeout` marks a transport-level condition with
  no provider response at all.
- **The raw provider body is not reachable.** litellm parses the
  provider's HTTP response before this hook ever runs, and does not
  keep the parsed JSON on the exception: `exception.body` and
  `exception.response.text` were both empty in every case tried,
  including a live 429 from a local server returning a JSON error
  object. The only survivor is the flattened message string, on
  `exception.message` and duplicated in
  `kwargs["standard_logging_object"]["error_information"]["error_message"]`.
  This hook therefore builds a synthetic body, `{"error": {"message":
  <that string>}}`, for `classify` to read. `classify` only ever reads
  the message text and the HTTP status (see `classify.py`), so the
  synthetic body carries everything it needs; only a field beyond
  those two would be lost, and no rule in `classify` reads one.

## What reaches the Journal

A classified failure records its Alias, its time and its Outcome. It
records no provider text at all.

An `unrecognized_failure` also records the provider's message, redacted
and truncated to `MAXIMUM_MESSAGE_CHARACTERS`. `reduce.journal_outcome`
re-buckets that case to `inconclusive`, so it changes no Health State
(ADR 0008); the message is the only thing that tells the operator which
`classify` rule is missing. Storing text on the unknown cases alone
keeps the credential exposure bounded, and it shrinks as rules are
added.

## Never raises

A logging hook must never raise into the proxy: the request already
failed, and an exception here would break error handling for a
response the client is waiting on. Every step below runs inside a
`try`/`except Exception`, and a failure to derive any one field simply
narrows what gets recorded, down to recording nothing.

## No hand-maintained list

This hook derives the Alias from the call it just observed. It holds no
list of Aliases, unlike an earlier version of `chatgpt_role_fix.py`
that did and went stale without a symptom.

## Install

Copy this file next to your `config.yaml`, then register it:

```yaml
litellm_settings:
  callbacks:
    - journal_failure_callback.observation_journal_callback
```

The Journal path comes from `litellm_maintainer.paths.journal_path()`,
which reads `$LITELLM_MAINTAINER_HOME` (default
`~/.config/litellm-maintainer`). Set that variable in the proxy's
environment if the instance directory is not the default.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from litellm._logging import verbose_logger
from litellm.integrations.custom_logger import CustomLogger

if TYPE_CHECKING:  # pragma: no cover - types only, never imported at runtime
    from litellm_maintainer.classify import Outcome

#: Names the directory holding the `litellm_maintainer` package, for a
#: proxy whose interpreter cannot already import it. Point it at the
#: repository root.
PACKAGE_ROOT_VARIABLE = "LITELLM_MAINTAINER_PACKAGE_ROOT"

_maintainer = None
_load_failure_reported = False


def _load_maintainer():
    """Import the maintainer package, and cache what this hook uses.

    Warning: nothing here runs at import time, and this function is
    never called from module scope. `docs/gotchas.md` records the rule:
    "Import it inside a function and guard it, because an error at
    import time stops the proxy from starting." A module-level import
    of `litellm_maintainer` broke that rule and took the whole proxy
    down with `ImportError: Could not import
    observation_journal_callback` -- a logging hook stopping the
    service it logs for.

    Resolve the package in two steps:

    1. A plain import. This succeeds when the proxy's own interpreter
       has the package installed, which is the clean arrangement.
    2. Otherwise, add `$LITELLM_MAINTAINER_PACKAGE_ROOT` to `sys.path`
       and try again.

    Never derive the path from `__file__`. An earlier version added
    `Path(__file__).parent.parent`, which is the repository root only
    while the file sits in `providers/`. Deployed beside `config.yaml`
    that resolves to `~/.config`, so the import failed exactly where
    the hook is meant to run and nowhere it was tested.
    """
    global _maintainer
    if _maintainer is not None:
        return _maintainer

    try:
        import litellm_maintainer  # noqa: F401
    except ModuleNotFoundError:
        root = os.environ.get(PACKAGE_ROOT_VARIABLE, "")
        if not root:
            raise ModuleNotFoundError(
                "journal_failure_callback cannot import 'litellm_maintainer'. "
                "Either install the package into the proxy's own interpreter, "
                f"or set {PACKAGE_ROOT_VARIABLE} to the directory that holds "
                "it (the repository root)."
            ) from None
        if root not in sys.path:
            sys.path.insert(0, root)
        import litellm_maintainer  # noqa: F401

    from types import SimpleNamespace

    from litellm_maintainer.classify import REASON_UNRECOGNIZED_FAILURE, classify
    from litellm_maintainer.journal import append_observation
    from litellm_maintainer.paths import journal_path
    from litellm_maintainer.redact import MIN_VALUE_LENGTH, parse_dotenv_file, redact
    from litellm_maintainer.reduce import Observation

    _maintainer = SimpleNamespace(
        REASON_UNRECOGNIZED_FAILURE=REASON_UNRECOGNIZED_FAILURE,
        classify=classify,
        append_observation=append_observation,
        journal_path=journal_path,
        MIN_VALUE_LENGTH=MIN_VALUE_LENGTH,
        parse_dotenv_file=parse_dotenv_file,
        redact=redact,
        Observation=Observation,
    )
    return _maintainer


#: How much of an unclassified provider message reaches the Journal.
#: A provider that answers with an HTML error page would otherwise put
#: kilobytes on every line, and a line past one filesystem block breaks
#: the single-`write(2)` atomicity `journal.append_observation` relies
#: on. Enough text to recognise the condition and write a `classify`
#: rule; not enough to be a log.
MAXIMUM_MESSAGE_CHARACTERS = 500


def _extract_alias(kwargs: dict) -> Optional[str]:
    """Return the Alias the client asked for, or `None` when unreadable.

    Read `model_group`, and ONLY `model_group`. It is the `model_name`
    the client asked the proxy for, which is the Alias
    `journal.observation_key_map` can translate into a Health Key.

    Try three places, because the Router does not fill the first one on
    every path:

    1. `litellm_params.metadata.model_group`
    2. `standard_logging_object.model_group`, a top-level field of
       litellm's `StandardLoggingPayload`
    3. `metadata.model_group` at the top level of `kwargs`

    Warning: never fall back to `kwargs["model"]`. That is the RESOLVED
    deployment, not the Alias. An earlier version did, and it looked
    harmless until real traffic arrived: an exhausted opencode-go plan
    recorded 90 entries under `grok-4.5`, while the proxy serves that
    Offering as `claude-opencode-go-grok-4.5`. `observation_key_map`
    holds no such key, so `reduce` discarded all 90 and the exhaustion
    changed nothing. The entries were not merely useless -- they kept
    the Journal non-empty, which kept every tick due. Measured
    2026-07-27.

    Recording nothing is better than recording a key that cannot map.
    The caller warns, so the gap is visible rather than silent.
    """
    litellm_params = kwargs.get("litellm_params")
    if isinstance(litellm_params, dict):
        metadata = litellm_params.get("metadata")
        if isinstance(metadata, dict):
            model_group = metadata.get("model_group")
            if isinstance(model_group, str) and model_group:
                return model_group

    standard = kwargs.get("standard_logging_object")
    if isinstance(standard, dict):
        model_group = standard.get("model_group")
        if isinstance(model_group, str) and model_group:
            return model_group

    metadata = kwargs.get("metadata")
    if isinstance(metadata, dict):
        model_group = metadata.get("model_group")
        if isinstance(model_group, str) and model_group:
            return model_group

    return None


def _extract_provider(kwargs: dict) -> str:
    """Return the provider name `classify` expects, best-effort.

    `classify`'s `provider` argument names the condition in error
    messages the audit has not needed to disambiguate by provider, so
    an empty string here changes no rule. Read it anyway, for a future
    provider-specific rule and for anyone reading a Journal entry.
    """
    litellm_params = kwargs.get("litellm_params")
    if isinstance(litellm_params, dict):
        provider = litellm_params.get("custom_llm_provider")
        if isinstance(provider, str) and provider:
            return provider
    provider = kwargs.get("custom_llm_provider")
    return provider if isinstance(provider, str) else ""


def _extract_message(exception: Any) -> str:
    message = getattr(exception, "message", None)
    if isinstance(message, str) and message:
        return message
    return str(exception) if exception is not None else ""


#: Substrings that mark an environment variable as holding a
#: credential. Mapping EVERY variable would replace `PATH` and `HOME`
#: inside a message and make it unreadable; mapping none would leak the
#: values this hook exists to keep out of the Journal.
_CREDENTIAL_NAME_MARKERS = (
    "KEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "AUTH",
    "CREDENTIAL",
    "COOKIE",
    "SESSION",
)


def _redaction_map() -> dict[str, str]:
    """Build the credential map used to redact an unclassified message.

    Read `os.environ`, NOT a dotenv file alone. `docs/gotchas.md` ("The
    proxy environment can differ from your .env file") records that
    `load_dotenv()` does not overwrite a variable that already exists,
    so a map built from `~/.config/litellm/.env` can miss the very
    value the proxy actually sent. The environment is what the proxy
    used, so the environment is what this reads.

    Also read the dotenv file beside the served config when
    `$LITELLM_MAINTAINER_ENV` names one. Every value there is a
    credential by construction, whatever its variable is called, so it
    needs no name test.

    Build the map only when a message is about to be recorded, which
    happens only for an `unrecognized_failure`. A classified failure
    stores no text and pays nothing for this.
    """
    maintainer = _load_maintainer()
    mapping: dict[str, str] = {}

    for name, value in os.environ.items():
        if len(value) < maintainer.MIN_VALUE_LENGTH:
            continue
        upper = name.upper()
        if not any(marker in upper for marker in _CREDENTIAL_NAME_MARKERS):
            continue
        mapping[value] = f"<REDACTED:{name}>"

    env_file = os.environ.get("LITELLM_MAINTAINER_ENV", "")
    if env_file:
        try:
            for name, value in maintainer.parse_dotenv_file(Path(env_file)).items():
                if len(value) >= maintainer.MIN_VALUE_LENGTH:
                    mapping[value] = f"<REDACTED:{name}>"
        except Exception:
            # An unreadable env file narrows the map. It must never
            # stop a failure being recorded. The regex net in `redact`
            # still catches a bare `sk-` or `Bearer` token.
            pass

    return mapping


def _message_to_record(outcome: "Outcome", message: str) -> Optional[str]:
    """Return the message to store on the Observation, or `None`.

    Store text only for an `unrecognized_failure`: that is the one case
    where the operator needs to see the wording, because it names the
    `classify` rule that is missing (see `reduce.journal_outcome`, ADR
    0008). Redact it, then truncate it.

    Redact BEFORE truncating. Truncating first could cut a credential
    in half and leave a fragment no map entry matches.
    """
    maintainer = _load_maintainer()
    if outcome.reason != maintainer.REASON_UNRECOGNIZED_FAILURE or not message:
        return None
    redacted = maintainer.redact(message, _redaction_map())
    if len(redacted) > MAXIMUM_MESSAGE_CHARACTERS:
        return redacted[:MAXIMUM_MESSAGE_CHARACTERS] + "…"
    return redacted


def _is_timeout(exception: Any) -> bool:
    try:
        import litellm.exceptions as litellm_exceptions

        return isinstance(exception, litellm_exceptions.Timeout)
    except Exception:
        return False


def _classify_failure(kwargs: dict, *, now: datetime) -> tuple["Outcome", str]:
    """Turn one failed call's `kwargs` into an Outcome and its message.

    Build the synthetic body described in the module docstring: the
    raw provider body is not reachable, so `classify` reads the
    flattened message string instead.

    Return the message alongside the Outcome. `_message_to_record`
    decides whether it reaches the Journal; only an
    `unrecognized_failure` does.
    """
    exception = kwargs.get("exception")
    provider = _extract_provider(kwargs)
    transport = "timeout" if _is_timeout(exception) else None
    status_code = getattr(exception, "status_code", None)
    message = _extract_message(exception)
    body = {"error": {"message": message}} if message else None
    outcome = _load_maintainer().classify(
        provider=provider,
        http_status=status_code if isinstance(status_code, int) else None,
        body=body,
        transport=transport,
        now=now,
    )
    return outcome, message


class ObservationJournalCallback(CustomLogger):
    """Appends one Observation Journal entry per failed proxy call.

    See the module docstring for what `async_log_failure_event`
    receives, why the raw provider body is unreachable, and why this
    hook never raises.
    """

    def __init__(self, *, home: Optional[Path] = None) -> None:
        super().__init__()
        # `home` is set only by a test. The proxy always resolves the
        # instance directory from `$LITELLM_MAINTAINER_HOME`, read
        # fresh on every call so a changed environment variable takes
        # effect with no restart.
        self._home_override = home

    def _journal_path(self) -> Path:
        return _load_maintainer().journal_path(self._home_override)

    async def async_log_failure_event(
        self, kwargs: Any, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        try:
            self._record(kwargs, end_time)
        except Exception as exc:  # never break the request over a logging hook
            try:
                verbose_logger.warning(
                    "journal_failure_callback: could not record a failure: %s", exc
                )
            except Exception:
                pass

    def _record(self, kwargs: Any, end_time: Any) -> None:
        if not isinstance(kwargs, dict):
            return
        alias = _extract_alias(kwargs)
        if alias is None:
            verbose_logger.warning(
                "journal_failure_callback: could not read the Alias; recording nothing"
            )
            return

        observed_at = end_time if isinstance(end_time, datetime) else datetime.now(timezone.utc)
        if observed_at.tzinfo is None:
            # Warning: CONVERT a naive datetime, never relabel it.
            # litellm passes `end_time` as a naive LOCAL datetime.
            # `.replace(tzinfo=utc)` used to stamp 17:32 local as
            # 17:32Z, putting every entry two hours in the future on a
            # UTC+2 host. `journal.truncate_processed` keeps entries
            # newer than `now`, so nothing was ever removed: the
            # Journal grew without bound, `journal_pending` stayed true
            # forever, and the tick ran a full pipeline every 60
            # seconds. Measured 2026-07-27.
            #
            # `astimezone` on a naive datetime reads it as local time
            # and converts, which is what litellm actually gave us.
            observed_at = observed_at.astimezone(timezone.utc)

        maintainer = _load_maintainer()
        outcome, message = _classify_failure(kwargs, now=observed_at)
        observation = maintainer.Observation(
            offering_id=alias,
            observed_at=observed_at,
            outcome=outcome,
            message=_message_to_record(outcome, message),
        )
        maintainer.append_observation(self._journal_path(), observation)


observation_journal_callback = ObservationJournalCallback()
