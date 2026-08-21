"""The Prober: calls Offerings directly to find out whether they work.

See CONTEXT.md, "Prober", "Probe", "Inconclusive", "Withheld", "Excluded"
and "Passthrough Auth". See `.scratch/maintainer-v1/spec.md`, sections
"Probing", "Failure classification" and "Recovery does not need a
probe". See `docs/gotchas.md`, "Probe concurrency creates false
failures" and "Claude 5 models reject temperature=0".

The worklist comes from Policy, never from the Generated Config. An
Excluded Offering is absent from the Generated Config, so nothing that
read that file could ever probe it again. Reading Policy instead is
what lets an Excluded Offering recover.

This module performs no network call and no clock read on its own. A
call site supplies `transport` (how to reach a provider) and `now` /
`sleep` (the clock), so every rule here is testable with an injected
fake and no real waiting. `litellm_maintainer.cli` is the only caller
that supplies a transport that reaches a real provider, and only when
the operator runs the `probe` command with no `--dry-run`.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable

from litellm_maintainer.classify import (
    ANSWERED,
    INCONCLUSIVE,
    REASON_RATE_LIMITED,
    Outcome,
    classify,
)
from litellm_maintainer.feed import Feed, Offering
from litellm_maintainer.plan import _passes_baseline, _passes_plan_edition
from litellm_maintainer.policy import DeclaredOffering, Pacing, Policy
from litellm_maintainer.sse import StreamedRead, read_stream
from litellm_maintainer.reduce import HealthState, OfferingHealth

OfferingKey = str

# The probe request is deliberately tiny: one short user message and a
# small `max_tokens`. It carries no `temperature` key at all. The
# Claude 5 family accepts `temperature=1` only, so a health check that
# sends `temperature=0` fails on every Claude 5 model (docs/gotchas.md,
# "Claude 5 models reject temperature=0"). Omitting the key avoids the
# trap outright, for every provider, rather than special-casing one
# family.
PROBE_MESSAGES: tuple[dict[str, str], ...] = (
    {"role": "user", "content": "ping"},
)
# Keep this small, and never read a Probe as proof that a NON-streaming
# call works. Measured 2026-08-21: Cline answers HTTP 500 "empty
# response content" whenever a non-streamed completion carries empty
# `content`, and a reasoning model empties it by spending a small budget
# on reasoning. Four ids failed at `max_tokens=8`-scale budgets and
# answered at 400. Every Probe streams, so a Probe never meets the
# condition. Raising the number here would buy no measurement and would
# spend tokens on every sweep; the asymmetry belongs in the docs, where
# a client author reads it (docs/gotchas.md, "One provider fails a
# non-streaming call that streams").
PROBE_MAX_TOKENS = 8

# A rate-limit-shaped failure that measured nothing (classify returns
# `inconclusive` with `reason="rate_limited"`) is retried once, after
# this backoff, before it counts. See docs/gotchas.md, "Probe
# concurrency creates false failures".
RATE_LIMIT_RETRY_BACKOFF_SECONDS = 5.0


@dataclass(frozen=True)
class TransportResponse:
    """What one call to a provider produced.

    `http_status` is `None` when the call never reached the provider.
    `transport` carries a transport-level condition such as `"timeout"`
    in that case; `classify` reads it the same way it reads a real
    provider response. `body` is the parsed JSON body, or `None`.
    """

    http_status: int | None
    body: Any
    transport: str | None = None


Transport = Callable[["ProbeTarget"], TransportResponse]
Clock = Callable[[], datetime]
Sleep = Callable[[float], None]


@dataclass(frozen=True)
class ProbeTarget:
    """One Offering the Prober may call.

    `key` is the Health State key: the Discovered Offering id
    (`<provider>:<provider_model_id>`), or a Declared Offering's Alias
    (see `litellm_maintainer.reduce`, `OfferingKey`). `provider_id`
    selects the Policy `pacing` entry. Exactly one of `offering` and
    `declared` is set.
    """

    key: OfferingKey
    provider_id: str
    offering: Offering | None = None
    declared: DeclaredOffering | None = None

    def request_model(self) -> str:
        """Return the model identifier a transport would call.

        A Declared Offering's `litellm_params.model` carries a litellm
        ROUTING prefix, such as `openai/claude-gpt-5.6-luna`. litellm
        strips that prefix before it calls the provider; it names which
        client litellm should use, and is no part of the provider's own
        model id. The Prober posts directly, so it must strip it too.

        Leaving it on sent `openai/claude-gpt-5.6-luna` to a worker that
        advertises `claude-gpt-5.6-luna`, and every one of the twelve
        worker seats came back `needs_operator` on the first live sweep.
        Only a live run found it: the URL was right and the credential
        was right, so nothing else could have.

        Strip only the FIRST segment. A vendor path inside the id
        survives, so a Declared `openrouter/cohere/north-mini-code:free`
        correctly calls `cohere/north-mini-code:free`.
        """
        if self.offering is not None:
            return self.offering.endpoint.get("model", self.offering.provider_model_id)
        assert self.declared is not None
        model = str(self.declared.litellm_params.get("model", ""))
        _, separator, remainder = model.partition("/")
        return remainder if separator and remainder else model

    def base_url(self) -> str | None:
        """Return the endpoint base URL, when the Offering states one.

        A Declared Offering may state one as `litellm_params.api_base`.
        Most do not (litellm resolves the vendor's URL itself from the
        `model` prefix); such an Offering cannot be probed directly,
        and `live_transport` reports the attempt as measuring nothing.
        """
        if self.offering is not None:
            return self.offering.endpoint.get("base_url")
        assert self.declared is not None
        api_base = self.declared.litellm_params.get("api_base")
        return api_base if isinstance(api_base, str) and api_base else None

    def protocol(self) -> str | None:
        """Return the Feed's stated endpoint protocol, when one exists."""
        if self.offering is not None:
            return self.offering.endpoint.get("protocol")
        return None


@dataclass(frozen=True)
class Worklist:
    """What `build_worklist` found.

    `targets` are the Offerings to probe this run. The other fields are
    reporting only, read by `--dry-run` and the run summary; they name
    what the Prober chose not to call and why.

    `admitted` is every Offering key Policy currently tracks: every
    `targets` key, every key skipped for freshness, and every
    Passthrough Auth Declared Offering. Pass it to `reduce` as-is, so a
    fresh or Passthrough Auth Offering's Health State record survives
    the run instead of being discarded as no-longer-admitted.
    """

    targets: tuple[ProbeTarget, ...]
    admitted: frozenset[OfferingKey]
    skipped_fresh: tuple[OfferingKey, ...]
    skipped_withheld: tuple[OfferingKey, ...]
    skipped_passthrough: tuple[OfferingKey, ...]


def _discovered_admitted(feed: Feed, policy: Policy) -> dict[OfferingKey, Offering]:
    """Return the Discovered Offerings Policy currently admits.

    Mirrors the selection rule in `litellm_maintainer.plan` (the
    baseline capability filter, per-provider mode, pricing and
    subscription edition, the quality gate, `withheld`), reusing
    `plan._passes_baseline` and `plan._passes_plan_edition` rather
    than a second copy of that rule. Otherwise kept independent of
    `plan.py` on purpose: the Prober's worklist deliberately differs
    from `plan`'s Selection, since the Prober must still reach an
    Offering that is currently Excluded, which `plan` leaves out of the
    Generated Config. The baseline filter is a "this is not a coding
    model" rule, not a health rule, so it applies here the same way it
    applies in `plan`: an Excluded Offering that passes the baseline is
    still probed.
    """
    admitted: dict[OfferingKey, Offering] = {}
    for provider_id, rule in policy.providers.items():
        offerings = list(feed.offerings_for(provider_id))
        if rule.mode == "named":
            allowed_ids = set(rule.models or ())
            offerings = [o for o in offerings if o.id in allowed_ids]
        if rule.pricing:
            allowed_pricing = set(rule.pricing)
            offerings = [o for o in offerings if o.pricing_kind in allowed_pricing]
        for offering in offerings:
            if not _passes_baseline(offering):
                continue
            if not _passes_plan_edition(offering, rule.plan_edition):
                # The operator's subscription edition does not include
                # this Offering. Probing it spends a request to be told
                # so, on every sweep, forever.
                continue
            if offering.id in policy.withheld:
                continue
            score = offering.coding_score
            if score is None:
                if offering.id not in policy.approved_candidates:
                    continue
            elif score < policy.quality.minimum_coding_score:
                continue
            admitted[offering.id] = offering
    return admitted


def _declared_admitted(
    policy: Policy,
) -> tuple[dict[OfferingKey, DeclaredOffering], frozenset[OfferingKey]]:
    """Split Declared Offerings into probeable and Passthrough Auth.

    Return `(probeable, passthrough_aliases)`. A Passthrough Auth
    Offering's credentials come from the calling client, so a Probe
    would carry the wrong credentials and measure the proxy's
    credential, not the Offering's. It is never probed. See
    CONTEXT.md, "Passthrough Auth".

    Both mappings are keyed by `health_key`, so a Client-Facing Variant
    collapses onto the Alias it widens. That keeps the Prober from
    calling one Offering twice under two names, and keeps `reduce` from
    holding two records that can never legitimately disagree.
    """
    probeable: dict[OfferingKey, DeclaredOffering] = {}
    passthrough: set[OfferingKey] = set()
    for declared in policy.declared:
        if declared.passthrough_auth:
            passthrough.add(declared.health_key)
        else:
            probeable.setdefault(declared.health_key, declared)
    return probeable, frozenset(passthrough)


def _declared_provider_id(declared: DeclaredOffering) -> str:
    """Return a pacing-table key for a Declared Offering.

    Declared Offerings are not Feed Offerings, so they carry no
    `provider_id` of their own. Use the vendor prefix of
    `litellm_params.model` (`anthropic/claude-sonnet-5` ->
    `anthropic`). A provider absent from Policy's `pacing` table falls
    back to `pacing.default`, so an unrecognised prefix still paces.
    """
    model = str(declared.litellm_params.get("model", ""))
    prefix, _, _ = model.partition("/")
    return prefix or "default"


def _is_fresh(record: OfferingHealth | None, *, now: datetime, maximum_staleness_hours: float) -> bool:
    """State whether an Offering's health is fresh enough to skip.

    Judge freshness on `last_success_at`, never `last_attempt_at`: an
    Inconclusive Probe never advances either field, so an Offering that
    was worth probing before an Inconclusive attempt is still worth
    probing after one, with no special case needed here.

    An Excluded Offering is reached again, because the worklist comes
    from Policy and not from the Generated Config. That is what lets it
    recover. But it is NOT reached before its own recorded reset time.

    The spec states the rule directly: probe when health is stale, when
    the reset time has passed, or when the last outcome was
    Inconclusive. A reset time still in the future says the opposite of
    all three. Read "Recovery does not need a probe": the clock path
    "costs nothing", and calling an exhausted plan before it refills
    both costs a call and cannot succeed.

    Measured 2026-07-26: six Qwen Token Plan Offerings carried a reset
    time four days out and every sweep probed all six anyway.
    """
    if record is None:
        return False
    if record.probe_due:
        # A sibling on the same shared pool reported a quota
        # exhaustion, so this Offering is worth measuring even though
        # its own record still looks fresh (ADR 0004, and
        # `reduce._pool_siblings_to_mark`). The mark clears the moment
        # this sweep applies an outcome to the record.
        return False
    if record.excluded:
        if record.reset_at is not None and record.reset_at > now:
            # Not fresh, but not due either. The clock will clear it.
            return True
        return False
    if record.last_success_at is None:
        return False
    age = now - record.last_success_at
    return age < timedelta(hours=maximum_staleness_hours)


class UnknownProviderError(Exception):
    """A scope named a provider that Policy does not configure."""


def build_worklist(
    *,
    feed: Feed,
    policy: Policy,
    health: HealthState,
    now: datetime,
    provider: str | None = None,
    force: bool = False,
) -> Worklist:
    """Build this run's Prober worklist from Policy, Feed and Health State.

    Read `.scratch/maintainer-v1/spec.md`, "Probing", before changing
    this function. The worklist source is Policy, not the Generated
    Config, so an Offering the Generator currently leaves out (Excluded)
    can still be reached here and recover.

    `force` probes every target in scope, whatever Health State says:
    it skips the freshness check and the reset-time deferral alike.

    Use it when a provider refills early. That is not rare — the
    operator's ChatGPT plan reset days before the time its own 429
    stated, on 2026-07-25. The deferral is the right DEFAULT, because
    it stops an hourly tick calling a plan that cannot answer, but a
    recorded reset time is a provider's promise and not a fact. This
    flag is how the operator overrules it.

    `provider` scopes the sweep to one provider. Use it to spend one
    provider's quota rather than every provider's, which is how a first
    live run stays cheap. It narrows only which targets are probed:
    every other rule is unchanged, and Health State for a provider
    outside the scope is untouched.

    Raise `UnknownProviderError` when `provider` names a provider Policy
    does not configure. A silent empty sweep would read as success.
    """
    if provider is not None:
        # A Declared Offering has no Feed provider. Its provider id comes
        # from its own model prefix, so the worker seats resolve to
        # `openai`, which is not a Policy provider at all. Scoping must
        # reach them: they are the reason `--provider` exists.
        declared_providers = {_declared_provider_id(d) for d in policy.declared}
        if provider not in policy.providers and provider not in declared_providers:
            known = ", ".join(sorted(set(policy.providers) | declared_providers)) or "none"
            raise UnknownProviderError(
                f"Policy configures no provider named {provider!r}. Known: {known}."
            )
    discovered = _discovered_admitted(feed, policy)
    declared_probeable, passthrough_aliases = _declared_admitted(policy)

    admitted = frozenset(set(discovered) | set(declared_probeable) | passthrough_aliases)
    staleness_hours = policy.schedule.maximum_staleness_hours

    targets: list[ProbeTarget] = []
    skipped_fresh: list[OfferingKey] = []

    for offering_id, offering in sorted(discovered.items()):
        if provider is not None and offering.provider_id != provider:
            continue
        record = health.offerings.get(offering_id)
        if not force and _is_fresh(
            record, now=now, maximum_staleness_hours=staleness_hours
        ):
            skipped_fresh.append(offering_id)
            continue
        targets.append(
            ProbeTarget(key=offering_id, provider_id=offering.provider_id, offering=offering)
        )

    for alias, declared in sorted(declared_probeable.items()):
        if provider is not None and _declared_provider_id(declared) != provider:
            continue
        record = health.offerings.get(alias)
        if not force and _is_fresh(
            record, now=now, maximum_staleness_hours=staleness_hours
        ):
            skipped_fresh.append(alias)
            continue
        targets.append(
            ProbeTarget(key=alias, provider_id=_declared_provider_id(declared), declared=declared)
        )

    return Worklist(
        targets=tuple(targets),
        admitted=admitted,
        skipped_fresh=tuple(sorted(skipped_fresh)),
        skipped_withheld=tuple(sorted(policy.withheld)),
        skipped_passthrough=tuple(sorted(passthrough_aliases)),
    )


def probe_offering(
    offering: ProbeTarget,
    *,
    transport: Transport,
    now: Clock,
    sleep: Sleep = time.sleep,
    retry_backoff_seconds: float = RATE_LIMIT_RETRY_BACKOFF_SECONDS,
) -> Outcome:
    """Call one Offering once, and return what it means.

    A rate-limit-shaped failure — `classify` returns `inconclusive` with
    `reason="rate_limited"` — is retried once, after `retry_backoff_seconds`,
    before it counts. Whatever the second attempt classifies to is the
    final `Outcome`, even if it is `inconclusive` again. Every other
    outcome, including a genuine `self_healing` rate limit that states a
    reset time, stands on the first attempt.
    """
    response = transport(offering)
    at = now()
    outcome = classify(
        provider=offering.provider_id,
        http_status=response.http_status,
        body=response.body,
        transport=response.transport,
        now=at,
    )
    if outcome.bucket == INCONCLUSIVE and outcome.reason == REASON_RATE_LIMITED:
        sleep(retry_backoff_seconds)
        response = transport(offering)
        at = now()
        outcome = classify(
            provider=offering.provider_id,
            http_status=response.http_status,
            body=response.body,
            transport=response.transport,
            now=at,
        )
    return outcome


class _ProviderPacer:
    """Enforce one provider's minimum interval between Probe starts.

    `wait_turn` reserves the next start slot under a lock, then sleeps
    outside a caller's own accounting only through the injected `sleep`.
    The reservation is what enforces the interval; the transport call
    itself runs outside the lock, so up to `concurrency` calls (bounded
    by the caller's thread pool size) can still overlap.
    """

    def __init__(self, minimum_interval_seconds: float, *, now: Clock, sleep: Sleep) -> None:
        self._interval = minimum_interval_seconds
        self._now = now
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_start: datetime | None = None

    def wait_turn(self) -> None:
        with self._lock:
            current = self._now()
            if self._next_start is not None and current < self._next_start:
                wait_seconds = (self._next_start - current).total_seconds()
                if wait_seconds > 0:
                    self._sleep(wait_seconds)
                current = self._now()
            self._next_start = current + timedelta(seconds=self._interval)


def _pacing_for(provider_id: str, pacing: dict[str, Pacing]) -> Pacing:
    return pacing.get(provider_id, pacing["default"])


def probe_offerings(
    targets: tuple[ProbeTarget, ...],
    *,
    pacing: dict[str, Pacing],
    transport: Transport,
    now: Clock,
    sleep: Sleep = time.sleep,
) -> dict[OfferingKey, Outcome]:
    """Probe every target, respecting each provider's pacing.

    Each provider carries its own concurrency limit (at most that many
    calls to it in flight at once) and its own minimum interval (at
    least that many seconds between the starts of two calls to it),
    both from Policy's `pacing` table. Different providers proceed
    independently — one provider's pace never delays another's.

    `now` and `sleep` are the injected clock. A test supplies a fake
    pair so pacing is assertable with no real waiting.
    """
    by_provider: dict[str, list[ProbeTarget]] = {}
    for target in targets:
        by_provider.setdefault(target.provider_id, []).append(target)

    results: dict[OfferingKey, Outcome] = {}
    results_lock = threading.Lock()

    def _run_one(target: ProbeTarget, pacer: _ProviderPacer) -> None:
        pacer.wait_turn()
        outcome = probe_offering(target, transport=transport, now=now, sleep=sleep)
        with results_lock:
            results[target.key] = outcome

    executors: list[ThreadPoolExecutor] = []
    try:
        futures = []
        for provider_id, provider_targets in by_provider.items():
            rule = _pacing_for(provider_id, pacing)
            pacer = _ProviderPacer(rule.minimum_interval_seconds, now=now, sleep=sleep)
            executor = ThreadPoolExecutor(max_workers=rule.concurrency)
            executors.append(executor)
            for target in provider_targets:
                futures.append(executor.submit(_run_one, target, pacer))
        for future in futures:
            future.result()
    finally:
        for executor in executors:
            executor.shutdown(wait=True)

    return results


def format_summary_line(key: OfferingKey, outcome: Outcome, body: Any = None) -> str:
    """Render one Offering's Probe result as one report line.

    Pass the line through `litellm_maintainer.redact.redact` before
    printing it. A provider error can echo a request header or a
    credential; this function does not redact anything itself.
    """
    line = f"{key}: {outcome.bucket}"
    if outcome.reason and outcome.bucket != ANSWERED:
        line += f" ({outcome.reason})"
    if outcome.reset_at is not None:
        line += f" reset_at={outcome.reset_at.isoformat()}"
    if outcome.bucket != ANSWERED and body is not None:
        detail = _short_body_repr(body)
        if detail:
            line += f" — {detail}"
    return line


# A Probe pairs the Feed's base URL with the Feed's stated protocol.
# Both come from the same `endpoint` object, so they cannot disagree.
#
# WARNING: do not special-case a provider by NAME here. An earlier
# version forced `/messages` for the Qwen Token Plan, reasoning that the
# operator's config routes it Anthropic-shaped. That is true of the
# CONFIG route, which uses a different host path entirely
# (`/apps/anthropic`), and false of the FEED route. The Feed publishes
# `openai_chat_completions` with the `compatible-mode/v1` base, so the
# override produced `compatible-mode/v1/messages`, which 404s. Measured
# 2026-07-26: the same base with `/chat/completions` returns the
# provider's real 429 and its quota reset time.
#
# The lesson is the one the spec already states for envelope routing:
# read the data, never the provider name.
_OPENAI_COMPLETIONS_PATH = "/chat/completions"
_ANTHROPIC_MESSAGES_PATH = "/messages"

# The Feed states one protocol per Offering (`endpoint.protocol`). The
# probe payload is OpenAI-shaped, so a protocol that is not
# `openai_chat_completions` needs its own path. Gemini publishes
# `gemini_generate_content` with the native base URL, but the same host
# serves an OpenAI-compatible surface under `/openai`, which accepts
# the same bearer credential. Posting the OpenAI payload to the native
# base 404s, so route by protocol here. NOT verified by a live call
# (this build forbade live probes); verified by the URL test only.
_PROTOCOL_PATHS = {
    "openai_chat_completions": _OPENAI_COMPLETIONS_PATH,
    "gemini_generate_content": "/openai" + _OPENAI_COMPLETIONS_PATH,
    "anthropic_messages": _ANTHROPIC_MESSAGES_PATH,
}


def completions_path_for(provider_id: str, protocol: str | None = None) -> str:
    """Return the completions path a Probe appends.

    Route by the Feed's stated `endpoint.protocol` (see
    `_PROTOCOL_PATHS`), never by the provider's name. Return
    `/chat/completions` when the protocol is absent or unrecognised,
    because the probe payload is OpenAI-shaped.

    `provider_id` is kept in the signature for the caller's clarity and
    for a future protocol the Feed does not state. It must not select a
    path on its own; see the warning above `_PROTOCOL_PATHS`.
    """
    if protocol is not None and protocol in _PROTOCOL_PATHS:
        return _PROTOCOL_PATHS[protocol]
    return _OPENAI_COMPLETIONS_PATH


def build_probe_url(base_url: str, provider_id: str, protocol: str | None = None) -> str:
    """Build the full URL a Probe posts to.

    Append the completions path for the provider and protocol (see
    `completions_path_for`) to `base_url`. This is pure string work: a
    test can assert the built URL with no network call.
    """
    return base_url.rstrip("/") + completions_path_for(provider_id, protocol)


_ENV_CREDENTIAL_REFERENCE = "os.environ/"


def probe_credential(
    target: ProbeTarget, *, feed: Feed, resolver: Callable[[str], str | None]
) -> str | None:
    """Resolve the credential a live Probe of `target` must send.

    The Prober calls providers DIRECTLY, so each call carries that
    provider's own credential — never the proxy's `LITELLM_MASTER_KEY`,
    which no provider accepts. For a Discovered Offering, the Feed
    provider's `authentication.credential_hint` names the variable and
    `resolver` looks it up (`cli._credential_resolver` reads the
    environment, then the `.env.local`-style file). For a Declared
    Offering, read `litellm_params.api_key`: an `os.environ/NAME`
    reference resolves through `resolver`, a literal string is the
    credential itself. Return `None` when nothing names one.
    """
    if target.offering is not None:
        provider = feed.providers.get(target.provider_id)
        hint = provider.credential_hint if provider is not None else None
        return resolver(hint) if hint else None
    assert target.declared is not None
    api_key = target.declared.litellm_params.get("api_key")
    if not isinstance(api_key, str) or not api_key:
        return None
    if api_key.startswith(_ENV_CREDENTIAL_REFERENCE):
        return resolver(api_key[len(_ENV_CREDENTIAL_REFERENCE):])
    return api_key


def live_transport(
    target: ProbeTarget, *, credential: str | None = None, timeout: float = 15.0
) -> TransportResponse:
    """Call a real provider. The orchestrator decides when this runs.

    `--dry-run` never calls this function. It builds the tiny probe request (see `PROBE_MESSAGES`,
    `PROBE_MAX_TOKENS`, and the warning at the top of this module about
    `temperature`) and posts it to the Offering's completions URL (see
    `build_probe_url`). `credential` is the bearer token to send; the
    caller resolves it from the Feed provider's credential hint. A
    transport-level failure, including a timeout, returns a
    `TransportResponse` with `transport="timeout"` rather than raising,
    so `classify` always has something to read.
    """
    import httpx

    base_url = target.base_url()
    if not base_url:
        # No URL to call, so no call happened. Return a response
        # `classify` reads as Inconclusive (`unmeasured`): the attempt
        # measured nothing, and Health State must stay untouched. An
        # earlier version returned `transport="timeout"` here, which
        # classifies `self_healing` — a fabricated failure that
        # Excluded every Declared Offering with no `api_base` on the
        # first live run, without one network packet sent.
        return TransportResponse(http_status=None, body=None, transport=None)

    url = build_probe_url(base_url, target.provider_id, target.protocol())

    headers = {"Content-Type": "application/json"}
    if credential:
        headers["Authorization"] = f"Bearer {credential}"

    payload = build_probe_payload(target)

    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=timeout)
    except httpx.HTTPError:
        return TransportResponse(http_status=None, body=None, transport="timeout")

    if response.status_code >= 400:
        # An error is not streamed. Read it as a plain body so classify
        # sees the provider's own message and status.
        try:
            body = response.json()
        except ValueError:
            body = None
        return TransportResponse(
            http_status=response.status_code, body=body, transport=None
        )

    read = read_stream(response.text)
    # Answered when the stream carried at least one well-formed chunk,
    # not when it carried text. A reasoning model on this small token
    # budget spends it on reasoning and emits `content: ""`, and that is
    # a working route. Measured on Groq and Gemini.
    #
    # A stream that carried no chunk but did carry an error frame states
    # its own condition. Hand that frame to `classify` rather than an
    # empty body, which reads as `malformed_response`. See
    # `StreamedRead.error`.
    body = _streamed_body(read)
    return TransportResponse(http_status=response.status_code, body=body, transport=None)



def _streamed_body(read: StreamedRead) -> dict[str, Any]:
    """Build the body `classify` reads from one streamed response.

    A chunk means the route answered. No chunk and an error frame means
    the provider stated a failure inside a 2xx stream, so report that
    error. No chunk and no error frame reports an empty body, which
    `classify` reads as a malformed response.

    The smoke check builds its body the same way, on purpose: the two
    live callers must agree, because a disagreement between them is how
    a false failure survives.
    """
    if read.chunks_seen:
        return {"choices": [{"message": {"content": read.content}}]}
    if read.error is not None:
        return {"error": read.error}
    return {}


def build_probe_payload(target: ProbeTarget) -> dict[str, Any]:
    """Build the request one Probe sends. Pure, so a test can read it.

    Always streams. litellm's ChatGPT provider answers a streamed
    request and fails a non-streaming one, and the operator runs two
    local litellm worker seats behind this proxy, so a non-streaming
    Probe would Exclude twelve working Offerings on its first sweep. The
    smoke check streams for the same reason; the two must agree, because
    a disagreement between them is how a false failure survives.

    Verified before the change that every provider this Prober calls
    streams on the URL `build_probe_url` produces: groq, openrouter,
    cline, opencode-zen, opencode-go and gemini.

    Sends no `temperature`. The Claude 5 family accepts `temperature=1`
    only, so a health check that sends 0 fails every Claude 5 model.
    """
    return {
        "model": target.request_model(),
        "messages": list(PROBE_MESSAGES),
        "max_tokens": PROBE_MAX_TOKENS,
        "stream": True,
    }


def _short_body_repr(body: Any, limit: int = 200) -> str:
    """Return a short, printable form of a provider response body."""
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            text = str(error.get("message", ""))
        elif isinstance(error, str):
            text = error
        else:
            text = str(body.get("message", ""))
    else:
        text = str(body) if body is not None else ""
    return text[:limit]
