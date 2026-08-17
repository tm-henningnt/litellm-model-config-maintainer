"""`headroom refresh`: capture Readings of each mapped Allowance to disk.

Every test here is offline: `refresh_headroom` takes its `runner` as an
argument, the same seam `fetch_feed_document` takes its `transport`
through, so no test here calls the real `codexbar` binary. The one
exception, `test_real_codexbar_runner_...`, points `real_codexbar_runner`
at `tests/fixtures/codexbar_stub.py`, never at the real tool.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from litellm_maintainer.codexbar import (
    CodexbarExtraWindow,
    CodexbarIdentity,
    CodexbarReading,
    CodexbarWindow,
    parse_codexbar_document,
)
from litellm_maintainer.headroom import (
    HeadroomRecord,
    HeadroomState,
    binding_window,
    format_age,
    format_used_percent,
    headroom_source_warnings,
    read_headroom,
    query_codexbar_readings,
    real_codexbar_runner,
    reading_age_seconds,
    refresh_headroom,
    route_binding_window,
    window_is_void,
    write_headroom,
)
from litellm_maintainer.lock import maintainer_lock
from litellm_maintainer.policy import Headroom

FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE = (FIXTURES / "codexbar-sample.json").read_text()

NOW = datetime(2026, 7, 28, 21, 0, tzinfo=timezone.utc)


def _fake_runner(document_text: str):
    """A `CodexbarRunner` that ignores its arguments and returns fixed text."""

    def runner(providers: list[str], all_accounts_provider: str | None = None) -> str:
        return document_text

    return runner


def _failing_runner(exc: Exception):
    def runner(providers: list[str], all_accounts_provider: str | None = None) -> str:
        raise exc

    return runner


def _entry(provider: str, *, used_percent: float = 0, account_email: str | None = None, error=None):
    identity = {"providerID": provider}
    if account_email is not None:
        identity["accountEmail"] = account_email
    if error is not None:
        return {"provider": provider, "error": error}
    return {
        "provider": provider,
        "usage": {
            "identity": identity,
            "primary": {"usedPercent": used_percent},
            "secondary": None,
            "tertiary": None,
            "updatedAt": "2026-07-28T20:00:00Z",
        },
    }


MAPPED_SOURCES = {
    # `tests/fixtures/codexbar-sample.json`'s Claude and ClinePass entries
    # state no `accountEmail` (only OpenCode Go and Gemini's do), so the
    # join keys here read with an empty account, matching that capture.
    "pool:claude-subscription": "codexbar:claude/",
    "provider:cline": "codexbar:clinepass/",
    "provider:opencode-go": "codexbar:opencodego/",
}


# --- Headroom State read and write ---


def test_a_missing_file_reads_as_empty(tmp_path):
    assert read_headroom(tmp_path / "headroom.json").records == {}


def test_a_written_record_survives_a_round_trip(tmp_path):
    path = tmp_path / "headroom.json"
    outcome = refresh_headroom(
        headroom_policy=Headroom(sources=MAPPED_SOURCES),
        path=path,
        lock_path=tmp_path / "headroom.lock",
        runner=_fake_runner(SAMPLE),
        now=NOW,
    )
    assert outcome.ran is True
    assert set(outcome.updated) == set(MAPPED_SOURCES)

    state = read_headroom(path)
    claude = state.records["pool:claude-subscription"]
    assert claude.reading.secondary.used_percent == 82
    assert claude.reading.extra_windows[0].title == "Fable only"
    assert claude.read_at == NOW.isoformat()

    opencodego = state.records["provider:opencode-go"]
    assert opencodego.reading.identity.provider_id == "opencodego"
    assert opencodego.reading.identity.account_email is None


def test_a_single_bad_record_is_skipped_and_the_rest_kept(tmp_path):
    path = tmp_path / "headroom.json"
    refresh_headroom(
        headroom_policy=Headroom(sources=MAPPED_SOURCES),
        path=path,
        lock_path=tmp_path / "headroom.lock",
        runner=_fake_runner(SAMPLE),
        now=NOW,
    )
    raw = json.loads(path.read_text())
    del raw["records"]["provider:cline"]["reading"]  # corrupt one record
    path.write_text(json.dumps(raw))

    state = read_headroom(path)
    assert set(state.records) == {"pool:claude-subscription", "provider:opencode-go"}


# --- No sources declared ---


def test_no_sources_declared_does_nothing_and_says_so(tmp_path):
    path = tmp_path / "headroom.json"
    outcome = refresh_headroom(
        headroom_policy=Headroom(),
        path=path,
        lock_path=tmp_path / "headroom.lock",
        runner=_fake_runner(SAMPLE),
        now=NOW,
    )

    assert outcome.ran is False
    assert not path.exists()


# --- A provider that errors keeps its previous Reading ---


def test_an_erroring_provider_keeps_its_previous_reading_others_still_update(tmp_path):
    path = tmp_path / "headroom.json"
    lock = tmp_path / "headroom.lock"
    sources = {"provider:a": "codexbar:a/", "provider:b": "codexbar:b/"}

    first = json.dumps([_entry("a", used_percent=10), _entry("b", used_percent=20)])
    refresh_headroom(
        headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
        runner=_fake_runner(first), now=NOW,
    )

    # 'b' now errors; 'a' reports a new figure.
    second = json.dumps(
        [
            _entry("a", used_percent=30),
            _entry("b", error={"kind": "provider", "code": 1, "message": "rate limited"}),
        ]
    )
    outcome = refresh_headroom(
        headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
        runner=_fake_runner(second), now=NOW,
    )

    assert "provider:a" in outcome.updated
    assert "provider:b" in outcome.kept_previous

    state = read_headroom(path)
    assert state.records["provider:a"].reading.primary.used_percent == 30
    assert state.records["provider:b"].reading.primary.used_percent == 20


def test_a_malformed_field_is_a_named_failure_that_keeps_last_good_state(tmp_path):
    path = tmp_path / "headroom.json"
    lock = tmp_path / "headroom.lock"
    sources = {"provider:a": "codexbar:a/"}

    good = json.dumps([_entry("a", used_percent=10)])
    refresh_headroom(
        headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
        runner=_fake_runner(good), now=NOW,
    )

    broken_entry = _entry("a", used_percent=99)
    del broken_entry["usage"]["primary"]
    broken_entry["usage"]["primaryRenamed"] = {"usedPercent": 99}
    outcome = refresh_headroom(
        headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
        runner=_fake_runner(json.dumps([broken_entry])), now=NOW,
    )

    assert "provider:a" in outcome.kept_previous
    assert len(outcome.failures) == 1
    state = read_headroom(path)
    assert state.records["provider:a"].reading.primary.used_percent == 10


def test_a_runner_that_cannot_answer_keeps_every_previous_reading(tmp_path):
    path = tmp_path / "headroom.json"
    lock = tmp_path / "headroom.lock"
    sources = {"provider:a": "codexbar:a/"}

    good = json.dumps([_entry("a", used_percent=10)])
    refresh_headroom(
        headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
        runner=_fake_runner(good), now=NOW,
    )
    before = read_headroom(path)

    outcome = refresh_headroom(
        headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
        runner=_failing_runner(RuntimeError("codexbar: command not found")), now=NOW,
    )

    assert outcome.ran is True
    assert "provider:a" in outcome.kept_previous
    after = read_headroom(path)
    assert after.records["provider:a"].reading.primary.used_percent == (
        before.records["provider:a"].reading.primary.used_percent
    )


def test_a_document_that_is_not_a_json_list_keeps_every_previous_reading(tmp_path):
    path = tmp_path / "headroom.json"
    lock = tmp_path / "headroom.lock"
    sources = {"provider:a": "codexbar:a/"}

    good = json.dumps([_entry("a", used_percent=10)])
    refresh_headroom(
        headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
        runner=_fake_runner(good), now=NOW,
    )

    outcome = refresh_headroom(
        headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
        runner=_fake_runner("not json at all"), now=NOW,
    )

    assert "provider:a" in outcome.kept_previous
    assert read_headroom(path).records["provider:a"].reading.primary.used_percent == 10


# --- Ask only the mapped providers ---


def test_the_runner_is_asked_only_for_the_mapped_providers(tmp_path):
    seen = {}

    def runner(providers: list[str], all_accounts_provider: str | None = None) -> str:
        seen["providers"] = providers
        return json.dumps([_entry("claude", used_percent=1), _entry("clinepass", used_percent=2)])

    refresh_headroom(
        headroom_policy=Headroom(sources=MAPPED_SOURCES),
        path=tmp_path / "headroom.json",
        lock_path=tmp_path / "headroom.lock",
        runner=runner,
        now=NOW,
    )

    assert set(seen["providers"]) == {"claude", "clinepass", "opencodego"}


# --- The lock is Headroom State's own, never the maintainer lock ---


def test_holding_the_maintainer_lock_does_not_block_a_refresh(tmp_path):
    """`refresh_headroom` must never take `paths.lock_path`. Prove it by
    holding that lock throughout a refresh and confirming it still runs."""
    maintainer_lock_path = tmp_path / "state" / "maintainer.lock"
    maintainer_lock_path.parent.mkdir(parents=True)

    with maintainer_lock(maintainer_lock_path):
        outcome = refresh_headroom(
            headroom_policy=Headroom(sources={"provider:a": "codexbar:a/"}),
            path=tmp_path / "headroom.json",
            lock_path=tmp_path / "headroom.lock",
            runner=_fake_runner(json.dumps([_entry("a", used_percent=5)])),
            now=NOW,
        )

    assert outcome.ran is True
    assert "provider:a" in outcome.updated


def test_a_second_refresh_waits_for_the_first_rather_than_losing_its_update(tmp_path):
    path = tmp_path / "headroom.json"
    lock = tmp_path / "headroom.lock"
    sources = {"provider:a": "codexbar:a/", "provider:b": "codexbar:b/"}

    entered = threading.Event()
    release = threading.Event()

    def slow_runner(providers: list[str], all_accounts_provider: str | None = None) -> str:
        entered.set()
        release.wait(timeout=5)
        return json.dumps([_entry("a", used_percent=11)])

    def fast_runner(providers: list[str], all_accounts_provider: str | None = None) -> str:
        return json.dumps([_entry("b", used_percent=22)])

    results: dict[str, object] = {}

    def run_slow():
        results["slow"] = refresh_headroom(
            headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
            runner=slow_runner, now=NOW,
        )

    worker = threading.Thread(target=run_slow)
    worker.start()
    assert entered.wait(timeout=5)

    def run_fast():
        results["fast"] = refresh_headroom(
            headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
            runner=fast_runner, now=NOW,
        )

    second = threading.Thread(target=run_fast)
    second.start()
    release.set()
    worker.join(timeout=5)
    second.join(timeout=5)

    state = read_headroom(path)
    assert set(state.records) == {"provider:a", "provider:b"}


# --- The real subprocess runner, against a fixture script only ---


def test_real_codexbar_runner_reads_a_fixture_script_never_the_real_binary():
    script = FIXTURES / "codexbar_stub.py"
    runner = real_codexbar_runner(str(script))

    output = runner(["claude"])

    document = json.loads(output)
    assert any(entry["provider"] == "claude" for entry in document)


def test_real_codexbar_runner_passes_its_own_timeout_to_subprocess(monkeypatch):
    """Defect 6: the timeout was fixed at `DEFAULT_TIMEOUT_SECONDS` (40)
    with no way to raise it from Policy. Measured 2026-07-28: 24s for four
    mapped providers, 21-31s for every provider codexbar knows -- a fifth
    or sixth mapped provider plausibly crosses 40s. Pin that a caller's
    own `timeout` reaches `subprocess.run`, not the module default."""
    captured = {}

    class _Completed:
        returncode = 0
        stdout = "[]"
        stderr = ""

    def fake_run(args, **kwargs):
        captured["timeout"] = kwargs.get("timeout")
        return _Completed()

    monkeypatch.setattr("litellm_maintainer.headroom.subprocess.run", fake_run)

    runner = real_codexbar_runner("codexbar", timeout=12.5)
    runner(["claude"])

    assert captured["timeout"] == 12.5


def test_real_codexbar_runner_asks_for_all_accounts_of_one_provider(monkeypatch):
    """Ticket 11: `all_accounts_provider` builds `--provider <id>
    --all-accounts`, never combined with a batched `--provider` list --
    codexbar's own `--help` states account selection requires a single
    provider."""
    captured = {}

    class _Completed:
        returncode = 0
        stdout = "[]"
        stderr = ""

    def fake_run(args, **kwargs):
        captured["args"] = args
        return _Completed()

    monkeypatch.setattr("litellm_maintainer.headroom.subprocess.run", fake_run)

    runner = real_codexbar_runner("codexbar")
    runner([], "codex")

    assert captured["args"] == ["codexbar", "--format", "json", "--provider", "codex", "--all-accounts"]


def test_write_headroom_is_atomic_and_reads_back(tmp_path):
    from litellm_maintainer.codexbar import (
        CodexbarIdentity,
        CodexbarReading,
        CodexbarWindow,
    )
    from litellm_maintainer.headroom import HeadroomRecord

    path = tmp_path / "state" / "headroom.json"
    reading = CodexbarReading(
        provider="claude",
        identity=CodexbarIdentity(provider_id="claude", account_email="operator@example.com"),
        primary=CodexbarWindow(used_percent=4, window_minutes=300, resets_at="2026-07-29T00:00:00Z"),
        secondary=None,
        tertiary=None,
        extra_windows=(),
        updated_at="2026-07-28T20:00:00Z",
        error=None,
    )
    record = HeadroomRecord(
        allowance_id="pool:claude-subscription",
        source="codexbar:claude/operator@example.com",
        reading=reading,
        read_at=NOW.isoformat(),
    )
    write_headroom(path, HeadroomState(records={"pool:claude-subscription": record}))

    reread = read_headroom(path)
    assert reread.records["pool:claude-subscription"].reading.primary.used_percent == 4
    assert not any(p.suffix == ".tmp" for p in path.parent.iterdir())


# --- The Binding Window derivation ------------------------------------------
#
# `binding_window`, `window_is_void` and `reading_age_seconds` are the
# functions ticket 05 imports verbatim for its own per-Route figure. Tests
# here pin their behaviour directly against `codexbar-sample.json`, the
# same fixture `entitlements` and `guidance` both read from in their own
# tests.


def _reading(provider: str) -> CodexbarReading:
    document = parse_codexbar_document(SAMPLE)
    for reading in document.readings:
        if reading.provider == provider:
            return reading
    raise AssertionError(f"no {provider!r} entry in the fixture")  # pragma: no cover


def test_binding_window_picks_clinepasss_worst_window_not_a_named_one():
    # Measured 2026-07-28: ClinePass's primary and secondary both read 0%
    # while tertiary reads 100%, fully drawn. A reader that picked one
    # named window (say, primary) would report ClinePass as free.
    reading = _reading("clinepass")

    binding = binding_window(reading, now=NOW, maximum_staleness_hours=24)

    assert binding is not None
    assert binding.used_percent == 100
    assert binding.resets_at == "2026-08-03T10:19:34Z"


def test_binding_window_ignores_an_extra_window_for_claude():
    # Claude's primary reads 4%, secondary 82%, and an extra window
    # (Fable only) reads 59%. Decision 2 forbids the extra window from
    # binding: the figure must be 82%, from secondary, never 59%.
    reading = _reading("claude")

    binding = binding_window(reading, now=NOW, maximum_staleness_hours=24)

    assert binding is not None
    assert binding.used_percent == 82


# --- `route_binding_window`: the per-Route join (ticket 06) ----------------
#
# `binding_window` above never lets `extra_windows` affect its figure --
# that is the Allowance-level answer `entitlements` reads. A Route that
# names its own Sub-allowance's window reads a DIFFERENT figure, from
# `route_binding_window`: the worse of the parent's binding_window and its
# own extra window.


def test_route_binding_window_with_no_sub_allowance_id_matches_binding_window():
    reading = _reading("claude")

    route = route_binding_window(
        reading, sub_allowance_window_id=None, now=NOW, maximum_staleness_hours=24
    )

    assert route == binding_window(reading, now=NOW, maximum_staleness_hours=24)


def test_route_binding_window_picks_the_parent_when_it_is_worse():
    # Measured 2026-07-28: Claude's parent (secondary) reads 82%, its
    # `claude-weekly-scoped-fable` extra window reads 59%. The parent is
    # worse, so a fable Route still binds at 82%.
    reading = _reading("claude")

    binding = route_binding_window(
        reading,
        sub_allowance_window_id="claude-weekly-scoped-fable",
        now=NOW,
        maximum_staleness_hours=24,
    )

    assert binding is not None
    assert binding.used_percent == 82


def test_route_binding_window_picks_the_sub_allowance_when_it_is_worse():
    # The inverse of the measured case, and the reason ticket 06 exists:
    # fable can run dry while the rest of the pool has room. A Route
    # naming the fable window must then bind on fable's own figure, never
    # the parent's, or it reports a Route that is about to refuse as
    # though it still had headroom.
    reading = CodexbarReading(
        provider="claude",
        identity=CodexbarIdentity(provider_id="claude", account_email=None),
        primary=CodexbarWindow(used_percent=4, window_minutes=1440, resets_at=None),
        secondary=CodexbarWindow(used_percent=20, window_minutes=10080, resets_at=None),
        tertiary=None,
        extra_windows=(
            CodexbarExtraWindow(
                id="claude-weekly-scoped-fable",
                title="Fable only",
                window=CodexbarWindow(used_percent=100, window_minutes=10080, resets_at=None),
            ),
        ),
        updated_at="2026-07-28T20:55:00Z",
        error=None,
    )

    binding = route_binding_window(
        reading,
        sub_allowance_window_id="claude-weekly-scoped-fable",
        now=NOW,
        maximum_staleness_hours=24,
    )

    assert binding is not None
    assert binding.used_percent == 100


def test_route_binding_window_never_lets_the_sub_allowance_affect_a_route_without_it():
    # Containment runs one way: a sibling Route that names no Sub-allowance
    # reads the parent's figure alone, whatever fable is doing.
    reading = CodexbarReading(
        provider="claude",
        identity=CodexbarIdentity(provider_id="claude", account_email=None),
        primary=CodexbarWindow(used_percent=4, window_minutes=1440, resets_at=None),
        secondary=CodexbarWindow(used_percent=70, window_minutes=10080, resets_at=None),
        tertiary=None,
        extra_windows=(
            CodexbarExtraWindow(
                id="claude-weekly-scoped-fable",
                title="Fable only",
                window=CodexbarWindow(used_percent=100, window_minutes=10080, resets_at=None),
            ),
        ),
        updated_at="2026-07-28T20:55:00Z",
        error=None,
    )

    sibling = route_binding_window(
        reading, sub_allowance_window_id=None, now=NOW, maximum_staleness_hours=24
    )

    assert sibling is not None
    assert sibling.used_percent == 70


def test_route_binding_window_with_an_unknown_id_reads_the_parent_alone():
    # The id names a window codexbar renamed or never sent. It must fall
    # back to the parent's own figure rather than raising or silently
    # reading 0%.
    reading = _reading("claude")

    binding = route_binding_window(
        reading,
        sub_allowance_window_id="a-window-codexbar-never-sent",
        now=NOW,
        maximum_staleness_hours=24,
    )

    assert binding is not None
    assert binding.used_percent == 82


def test_route_binding_window_treats_a_void_sub_allowance_window_as_absent():
    reading = CodexbarReading(
        provider="claude",
        identity=CodexbarIdentity(provider_id="claude", account_email=None),
        primary=CodexbarWindow(used_percent=4, window_minutes=1440, resets_at=None),
        secondary=CodexbarWindow(used_percent=20, window_minutes=10080, resets_at=None),
        tertiary=None,
        extra_windows=(
            CodexbarExtraWindow(
                id="claude-weekly-scoped-fable",
                title="Fable only",
                # Reset stated AFTER the Reading's own timestamp, but
                # already passed by `now`: void, and must not bind even
                # though its stored figure would otherwise be the worse
                # one.
                window=CodexbarWindow(
                    used_percent=100, window_minutes=10080, resets_at="2026-07-28T20:56:00Z"
                ),
            ),
        ),
        updated_at="2026-07-28T20:55:00Z",
        error=None,
    )

    binding = route_binding_window(
        reading,
        sub_allowance_window_id="claude-weekly-scoped-fable",
        now=NOW,
        maximum_staleness_hours=24,
    )

    assert binding is not None
    assert binding.used_percent == 20


# --- Ticket 09: named slots leave the parent computation --------------------
#
# Gemini fills its three slots with one quota per MODEL, not a nested time
# window: `primary` reads 100% (the free plan has no Pro), `secondary` and
# `tertiary` read 0% (Flash and Flash Lite are free). Built from
# `tests/fixtures/codexbar-sample.json`'s own `gemini` entry.


def test_binding_window_excludes_every_named_slot_and_returns_none():
    # Naming all three slots is Gemini's own case: nothing is left to bind
    # on, so the Allowance publishes no Headroom of its own.
    reading = _reading("gemini")

    binding = binding_window(
        reading,
        now=NOW,
        maximum_staleness_hours=24,
        named_slots=frozenset({"primary", "secondary", "tertiary"}),
    )

    assert binding is None


def test_binding_window_with_one_named_slot_binds_on_the_rest():
    # 'primary' (100%, Gemini's Pro slot) is named and excluded. 'secondary'
    # and 'tertiary' both read 0%, so the worst of THOSE two binds -- not
    # the 100% a reader that ignored named_slots would report.
    reading = _reading("gemini")

    binding = binding_window(
        reading, now=NOW, maximum_staleness_hours=24, named_slots=frozenset({"primary"})
    )

    assert binding is not None
    assert binding.used_percent == 0


def test_binding_window_with_no_named_slots_behaves_exactly_as_before():
    # Pinned: an existing caller passing no 'named_slots' argument at all
    # gets exactly the byte-identical answer it always got.
    reading = _reading("claude")

    assert binding_window(reading, now=NOW, maximum_staleness_hours=24) == binding_window(
        reading, now=NOW, maximum_staleness_hours=24, named_slots=frozenset()
    )


_GEMINI_SLOT_WINDOWS = {
    "primary": "gemini-pro",
    "secondary": "gemini-flash",
    "tertiary": "gemini-flash-lite",
}


def test_route_binding_window_resolves_gemini_pro_to_the_primary_slot():
    reading = _reading("gemini")

    binding = route_binding_window(
        reading,
        sub_allowance_window_id="gemini-pro",
        slot_windows=_GEMINI_SLOT_WINDOWS,
        now=NOW,
        maximum_staleness_hours=24,
    )

    assert binding is not None
    assert binding.used_percent == 100


def test_route_binding_window_resolves_gemini_flash_to_the_secondary_slot():
    reading = _reading("gemini")

    binding = route_binding_window(
        reading,
        sub_allowance_window_id="gemini-flash",
        slot_windows=_GEMINI_SLOT_WINDOWS,
        now=NOW,
        maximum_staleness_hours=24,
    )

    assert binding is not None
    assert binding.used_percent == 0


def test_route_binding_window_with_no_slot_windows_behaves_exactly_as_before():
    # Pinned: passing no 'slot_windows' at all -- every provider mapped
    # through a plain string -- reads exactly as before this ticket.
    reading = _reading("claude")

    with_none = route_binding_window(
        reading, sub_allowance_window_id=None, now=NOW, maximum_staleness_hours=24
    )
    without_the_argument_at_all = route_binding_window(
        reading, sub_allowance_window_id=None, now=NOW, maximum_staleness_hours=24
    )

    assert with_none == without_the_argument_at_all


def test_binding_window_is_none_when_a_reading_has_no_windows_at_all():
    # Measured 2026-07-28: OpenRouter answers with primary, secondary and
    # tertiary all null. This must never read as a healthy 0%.
    reading = _reading("openrouter")

    assert binding_window(reading, now=NOW, maximum_staleness_hours=24) is None


def test_a_window_past_its_own_reset_is_void_and_takes_no_part_in_binding():
    # The reset must be LATER than the Reading's own timestamp, because
    # that is the only shape a real reset takes: the source measures the
    # window, and states when it next refills. The window then goes void
    # while the Reading sits on disk. A reset stamped at or before the
    # measurement states no reset at all -- see the Gemini case below.
    window = CodexbarWindow(used_percent=100, window_minutes=300, resets_at="2026-07-01T00:00:00Z")

    assert window_is_void(
        window, reading_updated_at="2026-06-30T00:00:00Z", now=NOW, maximum_staleness_hours=24
    )


def test_a_live_window_with_a_future_reset_is_not_void():
    window = CodexbarWindow(used_percent=100, window_minutes=300, resets_at="2026-08-01T00:00:00Z")

    assert not window_is_void(
        window, reading_updated_at="2026-07-28T20:00:00Z", now=NOW, maximum_staleness_hours=24
    )


def test_a_window_with_no_reset_expires_at_the_schedules_maximum_staleness():
    window = CodexbarWindow(used_percent=100, window_minutes=300, resets_at=None)

    # The Reading is 30 hours old; the schedule allows 24.
    stale = window_is_void(
        window, reading_updated_at="2026-07-27T15:00:00Z", now=NOW, maximum_staleness_hours=24
    )
    # The Reading is 2 hours old; well inside 24.
    fresh = window_is_void(
        window, reading_updated_at="2026-07-28T19:00:00Z", now=NOW, maximum_staleness_hours=24
    )

    assert stale
    assert not fresh


def test_binding_window_is_none_when_every_window_is_void():
    reading = CodexbarReading(
        provider="fictional",
        identity=CodexbarIdentity(provider_id="fictional", account_email=None),
        primary=CodexbarWindow(used_percent=100, window_minutes=300, resets_at="2026-07-01T00:00:00Z"),
        secondary=CodexbarWindow(used_percent=50, window_minutes=10080, resets_at="2026-07-02T00:00:00Z"),
        tertiary=None,
        extra_windows=(),
        updated_at="2026-06-30T00:00:00Z",
        error=None,
    )

    assert binding_window(reading, now=NOW, maximum_staleness_hours=24) is None


# --- Defect 1: a naive ISO timestamp must never crash a comparison ---------
#
# Confirmed 2026-07-29: `datetime.fromisoformat` parses a timestamp with no
# 'Z' and no offset into a NAIVE datetime. `window_is_void` then compares it
# to the timezone-AWARE `now`, raising `TypeError: can't compare
# offset-naive and offset-aware datetimes` -- and every existing fixture up
# to this point carries 'Z', so nothing exercised the naive case. A naive
# value is unparsable, exactly like a garbled string: it must read as
# absent, never as UTC.


def test_a_naive_resets_at_is_treated_as_unparsable_and_the_window_goes_void():
    naive = "2026-07-29T12:00:00"  # no 'Z', no offset
    window = CodexbarWindow(used_percent=50, window_minutes=300, resets_at=naive)

    # Must not raise. Neither the reset nor the Reading's own timestamp is
    # readable, so the window has no way to bound its age: it reads void.
    assert window_is_void(
        window, reading_updated_at=naive, now=NOW, maximum_staleness_hours=24
    )


def test_a_naive_updated_at_never_crashes_binding_window():
    reading = CodexbarReading(
        provider="claude",
        identity=CodexbarIdentity(provider_id="claude", account_email=None),
        primary=CodexbarWindow(used_percent=90, window_minutes=300, resets_at=None),
        secondary=None,
        tertiary=None,
        extra_windows=(),
        updated_at="2026-07-29T12:00:00",  # no offset: this is the poison
        error=None,
    )

    # Must not raise TypeError. An unreadable Reading timestamp cannot
    # bound the window's age, so it reads void -- no Headroom, never a
    # crash that takes `guidance` and `entitlements` down with it.
    assert binding_window(reading, now=NOW, maximum_staleness_hours=24) is None


def test_a_naive_updated_at_makes_reading_age_seconds_none_not_a_crash():
    reading = CodexbarReading(
        provider="claude",
        identity=CodexbarIdentity(provider_id="claude", account_email=None),
        primary=None,
        secondary=None,
        tertiary=None,
        extra_windows=(),
        updated_at="2026-07-29T12:00:00",
        error=None,
    )

    assert reading_age_seconds(reading, now=NOW) is None


def test_reading_age_seconds_uses_codexbars_own_timestamp():
    # Measured 2026-07-28: codexbar's `updatedAt` advances on its own
    # schedule. Age must come from that field, never from our own
    # `HeadroomRecord.read_at` copy time, which this function never sees.
    reading = CodexbarReading(
        provider="claude",
        identity=CodexbarIdentity(provider_id="claude", account_email=None),
        primary=None,
        secondary=None,
        tertiary=None,
        extra_windows=(),
        updated_at="2026-07-28T20:00:00Z",
        error=None,
    )

    age = reading_age_seconds(reading, now=NOW)

    assert age == 3600.0  # NOW is 21:00Z, updated_at is 20:00Z


def test_a_clock_ahead_of_ours_clamps_age_to_zero_not_negative():
    # Defect 7(a): codexbar's clock can run ahead of ours. `updated_at`
    # here is 30 minutes AFTER `NOW`, so the naive subtraction is
    # negative -- which used to render "age -120 min". A Reading is never
    # taken in our own future, so clamp at zero rather than publish a
    # negative figure.
    reading = CodexbarReading(
        provider="claude",
        identity=CodexbarIdentity(provider_id="claude", account_email=None),
        primary=None,
        secondary=None,
        tertiary=None,
        extra_windows=(),
        updated_at="2026-07-28T21:30:00Z",  # NOW is 21:00Z
        error=None,
    )

    age = reading_age_seconds(reading, now=NOW)

    assert age == 0.0
    assert format_age(age) == "age 0 min"


def test_reading_age_seconds_is_none_when_updated_at_is_absent():
    reading = CodexbarReading(
        provider="claude",
        identity=CodexbarIdentity(provider_id="claude", account_email=None),
        primary=None,
        secondary=None,
        tertiary=None,
        extra_windows=(),
        updated_at=None,
        error=None,
    )

    assert reading_age_seconds(reading, now=NOW) is None


def test_a_reset_at_or_before_the_readings_own_timestamp_states_no_reset():
    """Gemini, measured 2026-07-28 20:52:30Z.

    The Reading stated `usedPercent: 100`, `resetsAt:
    "1970-01-01T00:00:00Z"` and `resetDescription: "Resets soon"` in one
    window. The epoch is an unset sentinel, and the prose beside it says
    the reset is still ahead. Read the epoch literally and that window
    goes void, so its two idle 1440-minute siblings bind instead and the
    highest figure in the Reading is silently discarded.

    This pins the derivation rule, not Gemini's number. What that 100
    counted is unknown, and Gemini is not mapped for exactly that reason.

    A reset at or before the measurement is therefore no reset. The
    window falls back to the staleness rule and keeps its figure.
    """
    drawn = CodexbarWindow(used_percent=100, window_minutes=1440, resets_at="1970-01-01T00:00:00Z")
    idle = CodexbarWindow(
        used_percent=0, window_minutes=1440, resets_at="2026-07-29T20:52:30Z"
    )
    reading = CodexbarReading(
        provider="gemini",
        identity=CodexbarIdentity(provider_id="gemini", account_email=None),
        primary=drawn,
        secondary=idle,
        tertiary=idle,
        extra_windows=(),
        updated_at="2026-07-28T20:52:30Z",
        error=None,
    )

    assert not window_is_void(
        drawn, reading_updated_at=reading.updated_at, now=NOW, maximum_staleness_hours=24
    )

    binding = binding_window(reading, now=NOW, maximum_staleness_hours=24)
    assert binding is not None
    assert binding.used_percent == 100

    # Having read the epoch as "no reset stated", publish that. Echoing it
    # back would tell a caller the window already refilled, which is the
    # opposite of what we just decided about it.
    assert binding.resets_at is None

    # The staleness rule still bounds it: 24 hours after the Reading's own
    # timestamp the figure stops counting, and the Allowance reports none.
    stale = datetime(2026, 7, 30, 3, 0, tzinfo=timezone.utc)
    assert binding_window(reading, now=stale, maximum_staleness_hours=24) is None


# --- `headroom_source_warnings`: distinguishing void from stale -------------
#
# ticket 07. `entitlements` and `guidance` both raise a warning when a
# mapped Allowance's Headroom stopped refreshing. The distinction that
# matters: a window gone VOID because its own reset passed is normal and
# self-correcting (`window_is_void` already covers it); a record whose
# `read_at` has not moved in `HEADROOM_STALE_MULTIPLIER` refresh intervals
# is a FAULT, because the job either stopped or codexbar keeps erroring.


def _warnings_reading(*, resets_at: str | None = None) -> CodexbarReading:
    return CodexbarReading(
        provider="claude",
        identity=CodexbarIdentity(provider_id="claude", account_email=None),
        primary=CodexbarWindow(used_percent=82, window_minutes=10080, resets_at=resets_at),
        secondary=None,
        tertiary=None,
        extra_windows=(),
        updated_at="2026-07-28T20:52:00Z",
        error=None,
    )


def _warnings_record(*, read_at: str, resets_at: str | None = None) -> HeadroomRecord:
    return HeadroomRecord(
        allowance_id="provider:claude",
        source="codexbar:claude/",
        reading=_warnings_reading(resets_at=resets_at),
        read_at=read_at,
    )


def _headroom_policy(interval_minutes: int = 15, sources: dict | None = None) -> Headroom:
    return Headroom(
        command="codexbar",
        interval_minutes=interval_minutes,
        sources=sources if sources is not None else {"provider:claude": "codexbar:claude/"},
    )


def test_no_sources_declared_raises_no_warning():
    warnings = headroom_source_warnings(
        headroom_policy=_headroom_policy(sources={}), headroom_state=HeadroomState(), now=NOW
    )
    assert warnings == ()


def test_a_source_never_refreshed_raises_a_warning():
    warnings = headroom_source_warnings(
        headroom_policy=_headroom_policy(), headroom_state=HeadroomState(), now=NOW
    )
    assert len(warnings) == 1
    assert "provider:claude" in warnings[0]
    assert "never" in warnings[0]


def test_a_recently_refreshed_source_raises_no_warning():
    state = HeadroomState(
        records={"provider:claude": _warnings_record(read_at=(NOW - timedelta(minutes=5)).isoformat())}
    )
    warnings = headroom_source_warnings(
        headroom_policy=_headroom_policy(interval_minutes=15), headroom_state=state, now=NOW
    )
    assert warnings == ()


def test_a_source_stale_past_the_multiplier_raises_a_warning():
    # 15-minute interval, 4x multiplier: 60 minutes is the floor. 90
    # minutes with no refresh means the job likely stopped.
    state = HeadroomState(
        records={
            "provider:claude": _warnings_record(read_at=(NOW - timedelta(minutes=90)).isoformat())
        }
    )
    warnings = headroom_source_warnings(
        headroom_policy=_headroom_policy(interval_minutes=15), headroom_state=state, now=NOW
    )
    assert len(warnings) == 1
    assert "provider:claude" in warnings[0]
    assert "1.5 h" in warnings[0]


def test_a_void_window_with_a_fresh_read_at_raises_no_warning():
    """A window past its own reset is normal and self-correcting. The
    warning tracks job health (`read_at`), never window validity."""
    state = HeadroomState(
        records={
            "provider:claude": _warnings_record(
                read_at=(NOW - timedelta(minutes=1)).isoformat(),
                resets_at="2026-07-01T00:00:00Z",  # long past; the window refilled
            )
        }
    )
    warnings = headroom_source_warnings(
        headroom_policy=_headroom_policy(interval_minutes=15), headroom_state=state, now=NOW
    )
    assert warnings == ()


def test_an_unparsable_read_at_raises_a_warning():
    state = HeadroomState(
        records={"provider:claude": _warnings_record(read_at="not-a-timestamp")}
    )
    warnings = headroom_source_warnings(
        headroom_policy=_headroom_policy(interval_minutes=15), headroom_state=state, now=NOW
    )
    assert len(warnings) == 1
    assert "unreadable" in warnings[0]


def test_only_the_stale_source_is_named_others_stay_quiet():
    fresh = _warnings_record(read_at=(NOW - timedelta(minutes=1)).isoformat())
    stale = _warnings_record(read_at=(NOW - timedelta(hours=5)).isoformat())
    state = HeadroomState(records={"provider:claude": fresh, "provider:cline": stale})
    policy = _headroom_policy(
        interval_minutes=15,
        sources={"provider:claude": "codexbar:claude/", "provider:cline": "codexbar:clinepass/"},
    )
    warnings = headroom_source_warnings(headroom_policy=policy, headroom_state=state, now=NOW)
    assert len(warnings) == 1
    assert "provider:cline" in warnings[0]


# --- Defect 2(b): `refresh_headroom` prunes a removed Allowance -------------
#
# The operator's real case: Gemini was mapped 2026-07-28 and unmapped
# 2026-07-29. Before this fix, `refresh_headroom` started from
# `dict(previous.records)` and never dropped a record whose Allowance
# Policy no longer names, so the stale Reading sat on disk forever.


def test_refresh_prunes_a_record_whose_allowance_is_no_longer_declared(tmp_path):
    path = tmp_path / "headroom.json"
    lock = tmp_path / "headroom.lock"
    sources_before = {"provider:a": "codexbar:a/", "provider:b": "codexbar:b/"}

    refresh_headroom(
        headroom_policy=Headroom(sources=sources_before), path=path, lock_path=lock,
        runner=_fake_runner(
            json.dumps([_entry("a", used_percent=10), _entry("b", used_percent=20)])
        ),
        now=NOW,
    )
    assert set(read_headroom(path).records) == {"provider:a", "provider:b"}

    # The operator removes 'provider:b' from Policy (Gemini's real case).
    sources_after = {"provider:a": "codexbar:a/"}
    outcome = refresh_headroom(
        headroom_policy=Headroom(sources=sources_after), path=path, lock_path=lock,
        runner=_fake_runner(json.dumps([_entry("a", used_percent=15)])),
        now=NOW,
    )

    assert outcome.ran is True
    state = read_headroom(path)
    assert set(state.records) == {"provider:a"}


def test_refresh_prunes_a_removed_allowance_even_when_the_runner_fails(tmp_path):
    # Pruning is a Policy-mapping cleanup, independent of whether codexbar
    # answered this run. A failed run must not leave the stale record
    # behind just because nothing new came back.
    path = tmp_path / "headroom.json"
    lock = tmp_path / "headroom.lock"
    sources_before = {"provider:a": "codexbar:a/", "provider:b": "codexbar:b/"}

    refresh_headroom(
        headroom_policy=Headroom(sources=sources_before), path=path, lock_path=lock,
        runner=_fake_runner(
            json.dumps([_entry("a", used_percent=10), _entry("b", used_percent=20)])
        ),
        now=NOW,
    )

    sources_after = {"provider:a": "codexbar:a/"}
    outcome = refresh_headroom(
        headroom_policy=Headroom(sources=sources_after), path=path, lock_path=lock,
        runner=_failing_runner(RuntimeError("codexbar: command not found")),
        now=NOW,
    )

    assert outcome.ran is True
    assert set(read_headroom(path).records) == {"provider:a"}


# --- Defect 3: two codexbar entries matching one declared source ----------
#
# Spec decision 4: "A reading that cannot name its credential attaches to
# nothing." Before this fix, `by_source_key = {reading.source_key: reading
# for reading in document.readings}` resolved a duplicate by iteration
# order alone, with no `failures` entry and no `kept_previous` -- an
# arbitrary winner silently stored.


def test_two_entries_matching_one_source_keep_the_previous_reading_and_are_named(tmp_path):
    path = tmp_path / "headroom.json"
    lock = tmp_path / "headroom.lock"
    sources = {"provider:a": "codexbar:a/"}

    good = json.dumps([_entry("a", used_percent=10)])
    refresh_headroom(
        headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
        runner=_fake_runner(good), now=NOW,
    )

    # codexbar now reports two entries that both match "codexbar:a/" --
    # the same fault shape as two accounts sharing one providerID.
    ambiguous = json.dumps([_entry("a", used_percent=40), _entry("a", used_percent=60)])
    outcome = refresh_headroom(
        headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
        runner=_fake_runner(ambiguous), now=NOW,
    )

    assert "provider:a" in outcome.kept_previous
    assert any("provider:a" in failure for failure in outcome.failures)
    state = read_headroom(path)
    # The PREVIOUS Reading survives. Neither 40 nor 60 -- an arbitrary
    # winner from the ambiguous pair -- was ever stored.
    assert state.records["provider:a"].reading.primary.used_percent == 10


def test_an_ambiguous_match_never_picks_either_new_entry(tmp_path):
    path = tmp_path / "headroom.json"
    lock = tmp_path / "headroom.lock"
    sources = {"provider:a": "codexbar:a/"}

    ambiguous = json.dumps([_entry("a", used_percent=40), _entry("a", used_percent=60)])
    outcome = refresh_headroom(
        headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
        runner=_fake_runner(ambiguous), now=NOW,
    )

    # No previous Reading existed either: the Allowance stays unstored,
    # never picking 40 or 60 by iteration order.
    assert "provider:a" in outcome.kept_previous
    assert "provider:a" not in read_headroom(path).records


# --- Ticket 11: a multi-account provider gets its own call ------------------
#
# `codexbar --provider codex` alone returns one Reading, and Policy can hold
# two ChatGPT seats with their own credentials. Measured 2026-07-29:
# `codexbar --provider codex --all-accounts --format json` returns BOTH
# accounts, sharing one `providerID` and differing only by `accountEmail`.
# `all_accounts_providers` names `codex` once; the seats never carry real
# account ids in this test file -- `one@example.com` and `two@example.com`
# stand in for them.


def _codex_seat_sources() -> dict[str, str]:
    return {
        "credential:SEAT1_KEY": "codexbar:codex/one@example.com",
        "credential:SEAT2_KEY": "codexbar:codex/two@example.com",
    }


def test_two_readings_sharing_a_provider_id_attach_to_two_different_allowances(tmp_path):
    path = tmp_path / "headroom.json"
    lock = tmp_path / "headroom.lock"
    sources = _codex_seat_sources()

    document = json.dumps(
        [
            _entry("codex", used_percent=10, account_email="one@example.com"),
            _entry("codex", used_percent=70, account_email="two@example.com"),
        ]
    )
    outcome = refresh_headroom(
        headroom_policy=Headroom(sources=sources, all_accounts_providers=("codex",)),
        path=path,
        lock_path=lock,
        runner=_fake_runner(document),
        now=NOW,
    )

    assert set(outcome.updated) == set(sources)
    state = read_headroom(path)
    assert state.records["credential:SEAT1_KEY"].reading.primary.used_percent == 10
    assert state.records["credential:SEAT2_KEY"].reading.primary.used_percent == 70


def test_an_unmarked_provider_keeps_using_the_batched_call(tmp_path):
    seen: dict[str, object] = {}

    def runner(providers: list[str], all_accounts_provider: str | None = None) -> str:
        seen["providers"] = providers
        seen["all_accounts_provider"] = all_accounts_provider
        return json.dumps([_entry("claude", used_percent=5)])

    refresh_headroom(
        headroom_policy=Headroom(sources={"provider:claude": "codexbar:claude/"}),
        path=tmp_path / "headroom.json",
        lock_path=tmp_path / "headroom.lock",
        runner=runner,
        now=NOW,
    )

    assert seen["providers"] == ["claude"]
    assert seen["all_accounts_provider"] is None


def test_a_marked_provider_is_excluded_from_the_batched_calls_provider_list(tmp_path):
    seen: dict[str, object] = {}

    def runner(providers: list[str], all_accounts_provider: str | None = None) -> str:
        if all_accounts_provider is None:
            seen["batched"] = providers
            return json.dumps([_entry("claude", used_percent=5)])
        seen["all_accounts_provider"] = all_accounts_provider
        return json.dumps(
            [
                _entry("codex", used_percent=1, account_email="one@example.com"),
                _entry("codex", used_percent=2, account_email="two@example.com"),
            ]
        )

    sources = {"provider:claude": "codexbar:claude/", **_codex_seat_sources()}
    refresh_headroom(
        headroom_policy=Headroom(sources=sources, all_accounts_providers=("codex",)),
        path=tmp_path / "headroom.json",
        lock_path=tmp_path / "headroom.lock",
        runner=runner,
        now=NOW,
    )

    assert seen["batched"] == ["claude"]
    assert seen["all_accounts_provider"] == "codex"


def test_nothing_probes_all_accounts_for_an_unmarked_provider(tmp_path):
    seen: list[str | None] = []

    def runner(providers: list[str], all_accounts_provider: str | None = None) -> str:
        seen.append(all_accounts_provider)
        return json.dumps([_entry("claude", used_percent=5)])

    refresh_headroom(
        headroom_policy=Headroom(sources={"provider:claude": "codexbar:claude/"}),
        path=tmp_path / "headroom.json",
        lock_path=tmp_path / "headroom.lock",
        runner=runner,
        now=NOW,
    )

    assert seen == [None]


def test_a_failed_extra_call_keeps_its_allowances_and_the_batched_ones_still_update(tmp_path):
    path = tmp_path / "headroom.json"
    lock = tmp_path / "headroom.lock"
    sources = {"provider:claude": "codexbar:claude/", **_codex_seat_sources()}
    policy = Headroom(sources=sources, all_accounts_providers=("codex",))

    def good_runner(providers: list[str], all_accounts_provider: str | None = None) -> str:
        if all_accounts_provider is None:
            return json.dumps([_entry("claude", used_percent=5)])
        return json.dumps(
            [
                _entry("codex", used_percent=10, account_email="one@example.com"),
                _entry("codex", used_percent=20, account_email="two@example.com"),
            ]
        )

    refresh_headroom(
        headroom_policy=policy, path=path, lock_path=lock, runner=good_runner, now=NOW
    )

    def flaky_runner(providers: list[str], all_accounts_provider: str | None = None) -> str:
        if all_accounts_provider is None:
            return json.dumps([_entry("claude", used_percent=55)])
        raise RuntimeError("codexbar: --all-accounts timed out")

    outcome = refresh_headroom(
        headroom_policy=policy, path=path, lock_path=lock, runner=flaky_runner, now=NOW
    )

    assert "provider:claude" in outcome.updated
    assert "credential:SEAT1_KEY" in outcome.kept_previous
    assert "credential:SEAT2_KEY" in outcome.kept_previous

    state = read_headroom(path)
    assert state.records["provider:claude"].reading.primary.used_percent == 55
    assert state.records["credential:SEAT1_KEY"].reading.primary.used_percent == 10
    assert state.records["credential:SEAT2_KEY"].reading.primary.used_percent == 20


def test_a_failed_batched_call_keeps_its_allowances_and_the_extra_call_still_updates(tmp_path):
    path = tmp_path / "headroom.json"
    lock = tmp_path / "headroom.lock"
    sources = {"provider:claude": "codexbar:claude/", **_codex_seat_sources()}
    policy = Headroom(sources=sources, all_accounts_providers=("codex",))

    def good_runner(providers: list[str], all_accounts_provider: str | None = None) -> str:
        if all_accounts_provider is None:
            return json.dumps([_entry("claude", used_percent=5)])
        return json.dumps(
            [
                _entry("codex", used_percent=10, account_email="one@example.com"),
                _entry("codex", used_percent=20, account_email="two@example.com"),
            ]
        )

    refresh_headroom(
        headroom_policy=policy, path=path, lock_path=lock, runner=good_runner, now=NOW
    )

    def flaky_runner(providers: list[str], all_accounts_provider: str | None = None) -> str:
        if all_accounts_provider is None:
            raise RuntimeError("codexbar: command not found")
        return json.dumps(
            [
                _entry("codex", used_percent=33, account_email="one@example.com"),
                _entry("codex", used_percent=44, account_email="two@example.com"),
            ]
        )

    outcome = refresh_headroom(
        headroom_policy=policy, path=path, lock_path=lock, runner=flaky_runner, now=NOW
    )

    assert "provider:claude" in outcome.kept_previous
    assert "credential:SEAT1_KEY" in outcome.updated
    assert "credential:SEAT2_KEY" in outcome.updated

    state = read_headroom(path)
    assert state.records["provider:claude"].reading.primary.used_percent == 5
    assert state.records["credential:SEAT1_KEY"].reading.primary.used_percent == 33
    assert state.records["credential:SEAT2_KEY"].reading.primary.used_percent == 44


# --- Defect 7(b): a Binding Window's used share never claims 100% early ----


def test_format_used_percent_never_rounds_ninety_nine_point_five_up_to_100():
    # `f"{99.5:.0f}%"` rounds to "100%" (round-half-to-even), which reads
    # as fully drawn while demotion only fires on the raw value reaching
    # 100 exactly.
    assert format_used_percent(99.5) == "99%"
    assert format_used_percent(99.9) == "99%"


def test_format_used_percent_states_100_only_when_it_really_is():
    assert format_used_percent(100) == "100%"
    assert format_used_percent(100.0) == "100%"


def test_format_used_percent_rounds_normally_below_the_boundary():
    assert format_used_percent(45.3) == "45%"
    assert format_used_percent(0) == "0%"


# --- codexbar's exit code carries no information -------------------------


def test_a_non_zero_exit_beside_a_readable_document_is_still_an_answer(tmp_path):
    """Measured 2026-07-29 on the operator's machine.

    `codexbar --format json --provider claude,clinepass,gemini,opencodego`
    exited 1 on three runs of three. It wrote `[codex notify]
    remoteControl/status/changed` to stderr, and a complete, valid array of
    nine Readings to stdout.

    Trusting the exit code discarded every batched Reading on every run, so
    the capability reported nothing and kept reporting nothing. An answer
    that parses is an answer.
    """
    stub = tmp_path / "noisy-codexbar"
    stub.write_text(
        "#!/bin/sh\n"
        'echo "[codex notify] remoteControl/status/changed" >&2\n'
        f"cat {FIXTURES / 'codexbar-sample.json'}\n"
        "exit 1\n"
    )
    stub.chmod(0o755)

    runner = real_codexbar_runner(str(stub))
    document = parse_codexbar_document(runner(["claude"], None))

    assert [r.provider for r in document.readings if r.provider == "claude"] == ["claude"]


def test_a_non_zero_exit_with_no_readable_document_still_fails(tmp_path):
    """The exit code is ignored, never the absence of an answer."""
    stub = tmp_path / "broken-codexbar"
    stub.write_text('#!/bin/sh\necho "boom" >&2\nexit 1\n')
    stub.chmod(0o755)

    runner = real_codexbar_runner(str(stub))
    try:
        runner(["claude"], None)
    except RuntimeError as exc:
        assert "no readable document" in str(exc)
    else:  # pragma: no cover - the runner must raise here
        raise AssertionError("a runner with no document must raise")


def test_the_batched_call_never_supplies_a_provider_that_has_its_own_call():
    """Measured 2026-07-29: `--provider claude,clinepass,gemini,opencodego`
    returned nine providers, `codex` among them, because codexbar "Honors
    your in-app toggles" and widens the answer rather than narrowing it.

    One account then arrived twice, once from each call, and the ambiguity
    guard read two Readings under one source key and kept the previous one.
    A provider asked about separately is answered by that call alone.
    """
    batched = json.dumps(
        [_entry("claude", used_percent=10), _entry("codex", used_percent=20, account_email="a@x")]
    )
    extra = json.dumps([_entry("codex", used_percent=30, account_email="a@x")])

    def runner(providers: list[str], all_accounts_provider: str | None = None) -> str:
        return extra if all_accounts_provider == "codex" else batched

    document, failed, errors = query_codexbar_readings(
        provider_ids=["claude", "codex"],
        all_accounts_providers=frozenset({"codex"}),
        runner=runner,
    )

    assert not failed and not errors
    codex_readings = [r for r in document.readings if r.identity.provider_id == "codex"]
    assert len(codex_readings) == 1
    assert codex_readings[0].primary.used_percent == 30


def test_an_all_accounts_call_answers_about_its_own_provider_alone():
    """The duplicate guard runs both ways.

    The batched call already drops a provider answered separately. This
    pins the other direction: an `--all-accounts` call keeps only the
    provider it asked about.

    Measured 2026-07-29: `--provider` WIDENS codexbar's answer rather than
    narrowing it — four providers asked, nine returned, because it "Honors
    your in-app toggles". An `--all-accounts` call narrows correctly today.
    Relying on that is the assumption that already cost one Allowance its
    figure, so the guard does not depend on it: a runner that answers every
    provider on both calls must still yield one Reading per provider.
    """
    every_provider = json.dumps(
        [
            _entry("claude", used_percent=40),
            _entry("codex", used_percent=50, account_email="a@x"),
        ]
    )

    def runner(providers: list[str], all_accounts_provider: str | None = None) -> str:
        return every_provider  # answers everything, whatever was asked

    document, failed, errors = query_codexbar_readings(
        provider_ids=["claude", "codex"],
        all_accounts_providers=frozenset({"codex"}),
        runner=runner,
    )

    assert not failed and not errors
    by_provider = [r.identity.provider_id for r in document.readings]
    assert sorted(by_provider) == ["claude", "codex"]


# --- A Binding Window crossing a threshold ---


def test_a_refresh_reports_the_threshold_a_window_crossed(tmp_path):
    """Pacing becomes event-driven: the refresh says the figure MOVED,
    which no poll of the current level can."""
    path = tmp_path / "headroom.json"
    lock = tmp_path / "headroom.lock"
    sources = {"provider:a": "codexbar:a/"}

    refresh_headroom(
        headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
        runner=_fake_runner(json.dumps([_entry("a", used_percent=10)])), now=NOW,
    )
    outcome = refresh_headroom(
        headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
        runner=_fake_runner(json.dumps([_entry("a", used_percent=97)])), now=NOW,
    )

    assert [c.threshold for c in outcome.crossings] == [80.0, 95.0]
    assert all(c.allowance_id == "provider:a" for c in outcome.crossings)


def test_a_first_reading_crosses_nothing(tmp_path):
    """There is no previous share to have moved from, and an absent
    Reading is unmeasured rather than 0%."""
    path = tmp_path / "headroom.json"
    outcome = refresh_headroom(
        headroom_policy=Headroom(sources={"provider:a": "codexbar:a/"}),
        path=path,
        lock_path=tmp_path / "headroom.lock",
        runner=_fake_runner(json.dumps([_entry("a", used_percent=99)])),
        now=NOW,
    )

    assert outcome.crossings == ()


def test_a_window_resting_above_a_threshold_crosses_nothing_on_refresh(tmp_path):
    path = tmp_path / "headroom.json"
    lock = tmp_path / "headroom.lock"
    sources = {"provider:a": "codexbar:a/"}

    refresh_headroom(
        headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
        runner=_fake_runner(json.dumps([_entry("a", used_percent=81)])), now=NOW,
    )
    outcome = refresh_headroom(
        headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
        runner=_fake_runner(json.dumps([_entry("a", used_percent=84)])), now=NOW,
    )

    assert outcome.crossings == ()


def test_an_allowance_that_kept_its_previous_reading_crosses_nothing(tmp_path):
    """Nothing moved, so nothing crossed. An error must not read as a
    change of share."""
    path = tmp_path / "headroom.json"
    lock = tmp_path / "headroom.lock"
    sources = {"provider:a": "codexbar:a/"}

    refresh_headroom(
        headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
        runner=_fake_runner(json.dumps([_entry("a", used_percent=10)])), now=NOW,
    )
    outcome = refresh_headroom(
        headroom_policy=Headroom(sources=sources), path=path, lock_path=lock,
        runner=_fake_runner(
            json.dumps([_entry("a", error={"kind": "provider", "code": 1, "message": "down"})])
        ),
        now=NOW,
    )

    assert outcome.crossings == ()
    assert "provider:a" in outcome.kept_previous


def test_a_declared_sub_allowance_slot_never_fires_a_crossing(tmp_path):
    """The window this rule exists for. Measured 2026-07-29:
    `provider:gemini` read `primary: 100%` while every admitted Route on
    it read 0%, because that slot described two withheld Pro Offerings on
    a plan the operator does not hold. A named slot leaves the parent's
    worst-of computation (ADR 0013), so it cannot become the figure a
    crossing fires on."""
    path = tmp_path / "headroom.json"
    lock = tmp_path / "headroom.lock"
    sources = {"provider:a": "codexbar:a/"}
    policy = Headroom(sources=sources, source_windows={"provider:a": {"primary": "a-pro"}})

    refresh_headroom(
        headroom_policy=policy, path=path, lock_path=lock,
        runner=_fake_runner(json.dumps([_entry("a", used_percent=10)])), now=NOW,
    )
    outcome = refresh_headroom(
        headroom_policy=policy, path=path, lock_path=lock,
        runner=_fake_runner(json.dumps([_entry("a", used_percent=100)])), now=NOW,
    )

    assert outcome.crossings == ()
    assert "provider:a" in outcome.updated
