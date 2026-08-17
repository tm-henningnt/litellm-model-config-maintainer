"""Tests for `providers/journal_failure_callback.py`.

Each test name states a rule an operator would recognise. The proxy's
failure callback appends to the Observation Journal and never writes
Health State (ADR 0001, "Four files, one writer each").
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent / "providers"))

from litellm_maintainer import paths  # noqa: E402
from litellm_maintainer.journal import read_observations  # noqa: E402


class _FakeException(Exception):
    """Stands in for a `litellm.exceptions.APIError`."""

    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


def _kwargs(
    *,
    alias: str = "claude-glm-5.2",
    message: str = "Rate limit reached: retry in 5s.",
    status_code: int = 429,
) -> dict:
    return {
        "model": "opencode-go/glm-5.2",
        "litellm_params": {
            "model": "opencode-go/glm-5.2",
            "metadata": {"model_group": alias},
            "custom_llm_provider": "opencode-go",
        },
        "exception": _FakeException(message, status_code),
    }


def _make_hook(home: Path):
    import journal_failure_callback as hook_module

    return hook_module.ObservationJournalCallback(home=home)


def test_a_failure_appends_exactly_one_journal_entry_naming_the_alias_time_and_outcome(
    tmp_path: Path,
):
    hook = _make_hook(tmp_path)
    end_time = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)

    asyncio.run(
        hook.async_log_failure_event(
            _kwargs(alias="claude-glm-5.2"), None, None, end_time
        )
    )

    read = read_observations(paths.journal_path(tmp_path))
    assert len(read.observations) == 1
    observation = read.observations[0]
    assert observation.offering_id == "claude-glm-5.2"
    assert observation.observed_at == end_time
    assert observation.outcome.bucket in (
        "self_healing",
        "needs_operator",
        "gone",
        "inconclusive",
        "answered",
    )


def test_the_callback_never_raises_even_on_a_broken_input(tmp_path: Path):
    hook = _make_hook(tmp_path)

    # Deliberately broken: `kwargs` is not a dict, `exception` is
    # missing where present, and `end_time` is not a datetime.
    asyncio.run(hook.async_log_failure_event(None, None, None, None))
    asyncio.run(hook.async_log_failure_event({}, None, None, "not-a-time"))
    asyncio.run(
        hook.async_log_failure_event(
            {"litellm_params": {"metadata": {"model_group": 12345}}},
            object(),
            None,
            None,
        )
    )
    asyncio.run(
        hook.async_log_failure_event(
            {"litellm_params": "not-a-dict", "exception": _FakeException("x", 500)},
            None,
            None,
            None,
        )
    )

    # Reaching here at all is the assertion: no call above raised.
    assert True


def test_the_callback_never_writes_health_state(tmp_path: Path):
    hook = _make_hook(tmp_path)
    end_time = datetime(2026, 7, 25, 12, 0, 0, tzinfo=timezone.utc)

    asyncio.run(
        hook.async_log_failure_event(
            _kwargs(alias="claude-glm-5.2", message="quota exceeded, limit: 0"),
            None,
            None,
            end_time,
        )
    )

    health_path = paths.health_path(tmp_path)
    assert not health_path.exists()


def test_an_alias_absent_from_kwargs_records_nothing_rather_than_a_wrong_alias(
    tmp_path: Path,
):
    hook = _make_hook(tmp_path)

    asyncio.run(
        hook.async_log_failure_event(
            {"litellm_params": {}, "exception": _FakeException("boom", 500)},
            None,
            None,
            None,
        )
    )

    read = read_observations(paths.journal_path(tmp_path))
    assert read.observations == []


# --- ADR 0008: an unclassified message, redacted and truncated -------------

# HTTP 400 with wording no `classify` rule matches: the client sent too
# many tokens. `_OPERATOR_STATUSES` holds 401, 402 and 403 only, so this
# reaches `classify`'s fail-closed default.
_OVERSIZED_PROMPT = "prompt is too long: 312000 tokens > 200000 maximum"


def test_an_unclassified_failure_records_the_message_that_names_the_missing_rule(
    tmp_path: Path,
):
    hook = _make_hook(tmp_path)

    asyncio.run(
        hook.async_log_failure_event(
            _kwargs(message=_OVERSIZED_PROMPT, status_code=400), None, None, None
        )
    )

    observation = read_observations(paths.journal_path(tmp_path)).observations[0]
    assert observation.outcome.reason == "unrecognized_failure"
    assert observation.message == _OVERSIZED_PROMPT


def test_a_classified_failure_records_no_provider_text(tmp_path: Path):
    """Text is stored only where it teaches us a missing rule."""
    hook = _make_hook(tmp_path)

    asyncio.run(hook.async_log_failure_event(_kwargs(), None, None, None))

    observation = read_observations(paths.journal_path(tmp_path)).observations[0]
    assert observation.outcome.reason == "rate_limited"
    assert observation.message is None


def test_a_credential_in_the_environment_never_reaches_the_journal(
    tmp_path: Path, monkeypatch
):
    """`docs/gotchas.md`: the proxy environment can differ from .env.

    The map is built from `os.environ`, because that is what the proxy
    actually sent. A map built from a dotenv file alone would miss this.
    """
    monkeypatch.setenv("OPENCODE_API_KEY", "supersecretvalue123456")
    hook = _make_hook(tmp_path)

    asyncio.run(
        hook.async_log_failure_event(
            _kwargs(
                message="unknown failure calling upstream with supersecretvalue123456",
                status_code=418,
            ),
            None,
            None,
            None,
        )
    )

    observation = read_observations(paths.journal_path(tmp_path)).observations[0]
    assert "supersecretvalue123456" not in (observation.message or "")
    assert "<REDACTED:OPENCODE_API_KEY>" in (observation.message or "")


def test_a_non_credential_environment_variable_is_left_alone(tmp_path: Path, monkeypatch):
    """Mapping every variable would replace PATH inside a message."""
    monkeypatch.setenv("LITELLM_LOG_LEVEL", "debugging")
    hook = _make_hook(tmp_path)

    asyncio.run(
        hook.async_log_failure_event(
            _kwargs(message="unknown failure while debugging", status_code=418),
            None,
            None,
            None,
        )
    )

    observation = read_observations(paths.journal_path(tmp_path)).observations[0]
    assert observation.message == "unknown failure while debugging"


def test_a_long_provider_message_is_truncated(tmp_path: Path):
    """An HTML error page must not put kilobytes on one Journal line.

    `journal.append_observation` relies on one line fitting a single
    `write(2)` under a filesystem block.
    """
    import journal_failure_callback as hook_module

    hook = _make_hook(tmp_path)

    asyncio.run(
        hook.async_log_failure_event(
            _kwargs(message="<html>" + "x" * 5000, status_code=418), None, None, None
        )
    )

    observation = read_observations(paths.journal_path(tmp_path)).observations[0]
    assert observation.message is not None
    assert len(observation.message) == hook_module.MAXIMUM_MESSAGE_CHARACTERS + 1


# --- The hook must survive being deployed away from the repository ---------


def test_importing_the_hook_never_raises_when_the_package_is_unreachable(tmp_path):
    """A logging hook must not stop the proxy from starting.

    Measured 2026-07-27: a module-level `from litellm_maintainer...`
    import killed proxy startup outright with `ImportError: Could not
    import observation_journal_callback`. `docs/gotchas.md` already
    recorded the rule this broke: "Import it inside a function and
    guard it, because an error at import time stops the proxy from
    starting."
    """
    import importlib.util

    source = Path(__file__).parent.parent / "providers" / "journal_failure_callback.py"
    deployed = tmp_path / "journal_failure_callback.py"
    deployed.write_text(source.read_text())

    # Import the DEPLOYED copy the way litellm does, with the repository
    # absent from `sys.path` and no package root named.
    spec = importlib.util.spec_from_file_location("deployed_hook", deployed)
    module = importlib.util.module_from_spec(spec)

    original_path = list(sys.path)
    repo_root = str(Path(__file__).parent.parent)
    try:
        sys.path[:] = [p for p in sys.path if p not in (repo_root, "", ".")]
        spec.loader.exec_module(module)  # must not raise
    finally:
        sys.path[:] = original_path

    assert module.observation_journal_callback is not None


def test_the_hook_finds_the_package_through_the_package_root_variable(
    tmp_path, monkeypatch
):
    """The deployed copy sits beside `config.yaml`, so `__file__` cannot
    locate the repository. An earlier version derived the path that way
    and resolved to `~/.config`, failing exactly where the hook runs."""
    import journal_failure_callback as hook_module

    monkeypatch.setattr(hook_module, "_maintainer", None)
    monkeypatch.setenv(
        hook_module.PACKAGE_ROOT_VARIABLE, str(Path(__file__).parent.parent)
    )

    maintainer = hook_module._load_maintainer()

    assert maintainer.classify is not None
    assert maintainer.Observation is not None


def test_a_failure_is_still_recorded_after_a_lazy_load(tmp_path, monkeypatch):
    """The lazy import must not change what the hook records."""
    import journal_failure_callback as hook_module

    monkeypatch.setattr(hook_module, "_maintainer", None)
    hook = _make_hook(tmp_path)

    asyncio.run(hook.async_log_failure_event(_kwargs(), None, None, None))

    read = read_observations(paths.journal_path(tmp_path))
    assert len(read.observations) == 1
    assert read.observations[0].offering_id == "claude-glm-5.2"


# --- The Alias, and the clock ----------------------------------------------


def test_a_naive_end_time_is_converted_from_local_time_not_relabelled(
    tmp_path: Path, monkeypatch
):
    """litellm passes `end_time` as a naive LOCAL datetime.

    `.replace(tzinfo=utc)` stamped 17:32 local as 17:32Z, putting every
    entry two hours in the future on a UTC+2 host. Rotation keeps
    entries newer than `now`, so nothing was ever removed and the tick
    ran a full pipeline every 60 seconds. Measured 2026-07-27.
    """
    from datetime import timedelta

    hook = _make_hook(tmp_path)
    naive_local = datetime.now().replace(microsecond=0)

    asyncio.run(hook.async_log_failure_event(_kwargs(), None, None, naive_local))

    recorded = read_observations(paths.journal_path(tmp_path)).observations[0].observed_at
    assert recorded.tzinfo is not None
    # The same instant the naive local time named, not the same digits.
    assert abs(recorded - naive_local.astimezone()) < timedelta(seconds=1)
    # And never in the future, which is what broke rotation.
    assert recorded <= datetime.now(timezone.utc) + timedelta(seconds=1)


def test_the_alias_is_read_from_the_standard_logging_object_when_metadata_lacks_it(
    tmp_path: Path,
):
    """The Router does not fill `litellm_params.metadata` on every path.
    `StandardLoggingPayload` carries `model_group` at the top level."""
    hook = _make_hook(tmp_path)
    kwargs = _kwargs()
    kwargs["litellm_params"]["metadata"] = {}
    kwargs["standard_logging_object"] = {"model_group": "claude-opencode-go-grok-4.5"}

    asyncio.run(hook.async_log_failure_event(kwargs, None, None, None))

    observation = read_observations(paths.journal_path(tmp_path)).observations[0]
    assert observation.offering_id == "claude-opencode-go-grok-4.5"


def test_a_failure_with_no_model_group_anywhere_records_nothing(tmp_path: Path):
    """Never fall back to the RESOLVED deployment.

    An exhausted opencode-go plan recorded 90 entries under `grok-4.5`
    while the proxy serves that Offering as
    `claude-opencode-go-grok-4.5`. `observation_key_map` holds no such
    key, so `reduce` discarded all 90 and the exhaustion changed
    nothing -- while the entries kept the Journal non-empty, which kept
    every tick due.
    """
    hook = _make_hook(tmp_path)
    kwargs = _kwargs()
    kwargs["litellm_params"]["metadata"] = {}
    kwargs["model"] = "grok-4.5"  # the resolved deployment, not the Alias

    asyncio.run(hook.async_log_failure_event(kwargs, None, None, None))

    assert read_observations(paths.journal_path(tmp_path)).observations == []
