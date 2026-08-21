"""The proxy smoke check: one call per distinct translation rule.

See CONTEXT.md and `.scratch/maintainer-v1/spec.md`, "Verifying through
the proxy". After the Generator writes a Generated Config, one call per
distinct translation rule goes through the RUNNING proxy. This catches a
fault a direct Probe cannot see: a stale proxy environment, an
unregistered custom handler, a wrong base URL. See `docs/gotchas.md`,
"The proxy environment can differ from your .env file" -- every model
failing with an auth error through the proxy while a direct call
succeeds is the exact symptom this check exists to catch.

Follows the shape `litellm_maintainer.prober` set: a pure
core plus one injected `transport` and one injected clock, so every
rule is testable with an injected fake and no real network call.
`litellm_maintainer.cli` is the only caller that ever supplies a
transport reaching the real proxy, and only when the operator runs
`smoke` with no `--dry-run`.

A "distinct translation rule" is read from
`litellm_maintainer.translate`, never re-derived from a provider id.
`_rule_label` reads the actual output of `translate_offering` -- the
same function `plan` calls to build the Generated Config -- so the
grouping here cannot disagree with what produced it. Two providers that
share one rule function (`generic_openai_compatible` serves
`opencode-go`, `opencode-zen`, `cline` and `cline-pass` alike, whenever
none of them declares a response envelope key) collapse into one call
here, exactly as `translate_offering` collapses them.

A Declared Offering carries no translation rule at all (CONTEXT.md,
"Declared Offering": passed through verbatim, so no translation rule
ever applies to it). It is grouped instead by the vendor prefix of its
own `litellm_params.model` -- the same grouping
`litellm_maintainer.prober._declared_provider_id` already uses to pace
it, reused here rather than invented again.

The check never writes Health State and never blocks a write. It only
reports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable

from litellm_maintainer.classify import ANSWERED, INCONCLUSIVE, Outcome, classify
from litellm_maintainer.feed import Feed, Offering
from litellm_maintainer.naming import alias_for
from litellm_maintainer.prober import _streamed_body
from litellm_maintainer.sse import read_stream
from litellm_maintainer.policy import Policy
from litellm_maintainer.prober import (
    TransportResponse,
    _declared_admitted,
    _declared_provider_id,
    _discovered_admitted,
)
from litellm_maintainer.reduce import OfferingHealth
from litellm_maintainer.translate import (
    ENVELOPE_HANDLER_PREFIX,
    TRANSLATION_RULES,
    UnknownProviderError,
    translate_offering,
)

OfferingKey = str

# The smoke request is deliberately tiny: one short user message, a
# small `max_tokens`, and no `temperature` key at all. The Claude 5
# family accepts `temperature=1` only, so a check that sends
# `temperature=0` fails on every Claude 5 model (docs/gotchas.md,
# "Claude 5 models reject temperature=0"). Omitting the key avoids the
# trap for every model, not just Claude 5.
SMOKE_MESSAGES: tuple[dict[str, str], ...] = ({"role": "user", "content": "ping"},)
SMOKE_MAX_TOKENS = 8

_DECLARED_RULE_PREFIX = "declared"

_SSE_DATA_PREFIX = "data:"
_SSE_DONE_SENTINEL = "[DONE]"

# A record with no recorded success sorts as though it succeeded at the
# start of time: worse than any real success, better than nothing when
# every candidate is equally unproven. Timezone-aware, so it compares
# against `OfferingHealth.last_success_at` with no naive/aware mismatch.
_NEVER_SUCCEEDED = datetime.min.replace(tzinfo=timezone.utc)

STATUS_ANSWERED = "answered"
STATUS_FAILED = "failed"
STATUS_UNVERIFIED = "unverified"
STATUS_INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class SmokeEntry:
    """One Alias the smoke check could call through the proxy.

    `key` is the Health State key used to judge which Offering behind a
    rule is healthiest: a Discovered Offering id, or a Declared
    Offering's Alias (matching `litellm_maintainer.reduce.OfferingKey`).
    `alias` is the litellm `model_name` sent to the proxy -- never the
    raw provider model id; the proxy only knows Aliases. `rule` names
    the translation rule that produced this entry.
    """

    key: OfferingKey
    alias: str
    rule: str
    # False when the proxy cannot supply this Offering's credential, so
    # a call here would measure the wrong thing. Such an entry still
    # joins its rule's group, so the rule reports UNVERIFIED with the
    # reason instead of vanishing from the report entirely.
    callable_by_proxy: bool = True


@dataclass(frozen=True)
class RuleCheck:
    """One rule's smoke-check result.

    `status` is one of `STATUS_ANSWERED`, `STATUS_FAILED`,
    `STATUS_UNVERIFIED` or `STATUS_INCONCLUSIVE`. These name four
    different things (CONTEXT.md, "Inconclusive"); do not collapse any
    two of them.

    UNVERIFIED states that no healthy Offering existed for this rule, so
    no call was made at all. FAILED states a call was made and
    `classify` read a genuine failure. INCONCLUSIVE states a call was
    made and `classify` could not read it as either a success or a
    genuine failure — attributable to our own request rate, not to the
    Offering. Printing INCONCLUSIVE as FAILED would claim a wiring fault
    that was never measured; that is the exact distinction that stops
    the tool evicting a healthy Offering.

    UNVERIFIED and INCONCLUSIVE both mean the check learned nothing
    useful about the wiring. FAILED means it learned something bad.
    `key` and `alias` name the Offering actually called, or `None` for
    UNVERIFIED. `detail` is a short description of a non-answered
    result; pass the rendered line through
    `litellm_maintainer.redact.redact` before printing it.
    """

    rule: str
    status: str
    key: OfferingKey | None = None
    alias: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class SmokeCheckResult:
    """Every rule's result from one smoke-check run."""

    checks: tuple[RuleCheck, ...]

    @property
    def answered(self) -> tuple[RuleCheck, ...]:
        return tuple(c for c in self.checks if c.status == STATUS_ANSWERED)

    @property
    def failed(self) -> tuple[RuleCheck, ...]:
        return tuple(c for c in self.checks if c.status == STATUS_FAILED)

    @property
    def unverified(self) -> tuple[RuleCheck, ...]:
        return tuple(c for c in self.checks if c.status == STATUS_UNVERIFIED)

    @property
    def inconclusive(self) -> tuple[RuleCheck, ...]:
        return tuple(c for c in self.checks if c.status == STATUS_INCONCLUSIVE)


Transport = Callable[[SmokeEntry], TransportResponse]
Clock = Callable[[], datetime]


def _rule_label(offering: Offering, params: dict[str, Any]) -> str:
    """Name the translation rule that produced `params`.

    Reads the ACTUAL output of `translate_offering` -- the same
    function `plan` calls to build the Generated Config -- rather than
    re-deriving a guess from `offering.provider_id`. That is what keeps
    this label from disagreeing with the table that actually produced
    the entry. `generic_openai_compatible` serves several providers
    under one function object; every one of them reports the same
    label here, so they group as one rule, matching what
    `translate_offering` itself would do.
    """
    model = params.get("model", "")
    if model.startswith(f"{ENVELOPE_HANDLER_PREFIX}/"):
        return "envelope_unwrapping"
    rule_func = TRANSLATION_RULES.get(offering.provider_id)
    if rule_func is None:
        # plan() drops such an Offering silently (a Feed provider with
        # no registered rule yet); mirrored here for the same reason.
        return f"unregistered:{offering.provider_id}"
    sharing = sorted(pid for pid, func in TRANSLATION_RULES.items() if func is rule_func)
    # `native_prefix("groq")` and `native_prefix("openrouter")` are two
    # different closures, both named `rule` at the code level (their
    # `__name__`). Read `__qualname__` instead -- "native_prefix" for
    # both, "envelope_unwrapping" or "gemini_native" for a module-level
    # function with no enclosing factory -- so the printed label names
    # the actual rule, not the closure's generic inner name.
    factory_name = rule_func.__qualname__.split(".")[0]
    return f"{factory_name}[{','.join(sharing)}]"


def build_smoke_entries(*, feed: Feed, policy: Policy) -> tuple[SmokeEntry, ...]:
    """Build one `SmokeEntry` per admitted Offering, tagged with its rule.

    Mirrors `litellm_maintainer.prober`'s own admission logic
    (`_discovered_admitted`, `_declared_admitted`) rather than
    re-deriving it, so the smoke check's worklist cannot silently drift
    from the Prober's. A Passthrough Auth Declared Offering is excluded
    by default: the smoke check authenticates with the proxy's own
    `LITELLM_MASTER_KEY`, and an ordinary Passthrough Auth Offering's
    credentials come from the calling client, not the proxy (CONTEXT.md,
    "Passthrough Auth"), so a call under the proxy's own credential
    would measure the wrong thing, exactly as it would for the Prober.

    A Passthrough Auth Declared Offering that also sets
    `proxy_authenticated: true` is the one exception: the proxy holds
    that Offering's credential itself, so a call under the proxy's own
    credential measures the real thing. This flag governs the smoke
    check only. It changes nothing about the Prober, which still skips
    every Passthrough Auth Offering, and nothing about `reduce`, whose
    Passthrough Auth exemption stands unchanged.
    """
    entries: list[SmokeEntry] = []

    discovered = _discovered_admitted(feed, policy)
    for offering_id, offering in sorted(discovered.items()):
        provider = feed.providers.get(offering.provider_id)
        provider_rule = policy.providers[offering.provider_id]
        override = dict(provider_rule.translation or {})
        override.update(policy.translation_overrides.get(offering.id, {}))
        try:
            params = translate_offering(offering, provider, override=override or None)
        except UnknownProviderError:
            # plan() drops this Offering with no report line (a known
            # gap, spec-corrections.md #8); mirrored here rather than
            # raising, so a smoke run over the same Feed and Policy
            # never fails on an Offering the Generated Config also
            # never wrote.
            continue
        alias = alias_for(policy, offering.id)
        entries.append(
            SmokeEntry(key=offering_id, alias=alias, rule=_rule_label(offering, params))
        )

    declared_probeable, _passthrough = _declared_admitted(policy)
    for alias, declared in sorted(declared_probeable.items()):
        rule = f"{_DECLARED_RULE_PREFIX}:{_declared_provider_id(declared)}"
        entries.append(SmokeEntry(key=alias, alias=alias, rule=rule))

    # A Passthrough Auth Declared Offering. The proxy can call it only
    # when it holds the credential itself (`proxy_authenticated`).
    # Otherwise the caller holds it, and a call under the proxy's own
    # credential measures the wrong thing.
    #
    # Add it EITHER WAY, marking whether the proxy can call it. An entry
    # the proxy cannot call still puts its rule in the report, as
    # UNVERIFIED with the reason. Dropping it instead would delete a
    # whole translation rule from the report, and the operator would
    # read a clean run while that rule went unchecked.
    for declared in sorted(policy.declared, key=lambda d: d.alias):
        if not declared.passthrough_auth:
            continue
        rule = f"{_DECLARED_RULE_PREFIX}:{_declared_provider_id(declared)}"
        entries.append(
            SmokeEntry(
                key=declared.alias,
                alias=declared.alias,
                rule=rule,
                callable_by_proxy=declared.proxy_authenticated,
            )
        )

    return tuple(entries)


def group_by_rule(entries: tuple[SmokeEntry, ...]) -> dict[str, tuple[SmokeEntry, ...]]:
    """Group entries by the translation rule that produced them.

    Preserves entry order within a group. Groups on `entry.rule`,
    already read from the translation table by `_rule_label` -- never
    on `provider_id` again here, which is what keeps this grouping from
    disagreeing with `litellm_maintainer.translate`.
    """
    grouped: dict[str, list[SmokeEntry]] = {}
    for entry in entries:
        grouped.setdefault(entry.rule, []).append(entry)
    return {rule: tuple(items) for rule, items in grouped.items()}


def _is_healthy(record: OfferingHealth | None) -> bool:
    """Whether an Offering counts as healthy enough to try.

    An Offering never probed (`record is None`) is optimistically
    healthy: the smoke check must not need a Prober run first. A
    currently Excluded Offering is not healthy -- routing around
    exactly that Offering, in favour of a healthy sibling, is the whole
    point of picking "the healthiest available".
    """
    return record is None or not record.excluded


def pick_healthiest(
    entries: tuple[SmokeEntry, ...], *, health: dict[OfferingKey, OfferingHealth]
) -> SmokeEntry | None:
    """Choose the healthiest Offering among one rule's candidates.

    "Healthiest": prefer an Offering that is not Excluded; among those,
    prefer the most recent `last_success_at`. An Offering with no
    record, or a record with no recorded success, ranks below one with
    a recorded success, but still counts as healthy (see `_is_healthy`)
    -- it may simply never have been probed. Return `None` when every
    candidate is Excluded: that is what makes the rule UNVERIFIED
    rather than FAILED, so a quota-dead provider never fails a wiring
    check when a healthy sibling exists.
    """
    callable_entries = [entry for entry in entries if entry.callable_by_proxy]
    healthy = [entry for entry in callable_entries if _is_healthy(health.get(entry.key))]
    if not healthy:
        return None

    def _last_success(entry: SmokeEntry) -> datetime:
        record = health.get(entry.key)
        if record is None or record.last_success_at is None:
            return _NEVER_SUCCEEDED
        return record.last_success_at

    return max(healthy, key=_last_success)


def _detail_for(outcome: Outcome) -> str:
    detail = outcome.bucket
    if outcome.reason:
        detail += f" ({outcome.reason})"
    return detail


def run_smoke_check(
    entries: tuple[SmokeEntry, ...],
    *,
    health: dict[OfferingKey, OfferingHealth],
    transport: Transport,
    now: Clock,
) -> SmokeCheckResult:
    """Make one call per distinct translation rule, through `transport`.

    Never writes Health State, never refuses a write and never blocks
    anything: a result only ever appears in the returned
    `SmokeCheckResult`. This check makes one call with no retry, unlike
    the Prober, which retries a rate-limit-shaped Inconclusive once. So
    a rate-limit-shaped `classify` result reports as INCONCLUSIVE here,
    never as FAILED. The call measured nothing -- the same distinction
    that stops the Prober evicting a healthy Offering. Every other
    non-`answered` bucket reports as FAILED. Either FAILED or
    INCONCLUSIVE is worth a human look. Neither may change Health State
    or block a write. That is `litellm_maintainer.reduce`'s and
    `litellm_maintainer.safety`'s job, never this one's.
    """
    checks: list[RuleCheck] = []
    for rule, group in sorted(group_by_rule(entries).items()):
        chosen = pick_healthiest(group, health=health)
        if chosen is None:
            if not any(entry.callable_by_proxy for entry in group):
                detail = (
                    "every Offering behind this rule is Passthrough Auth: the "
                    "caller supplies the credential, so a call under the "
                    "proxy's own credential would measure the wrong thing"
                )
            else:
                detail = "no healthy Offering available for this rule"
            checks.append(
                RuleCheck(rule=rule, status=STATUS_UNVERIFIED, detail=detail)
            )
            continue

        response = transport(chosen)
        at = now()
        outcome = classify(
            provider=chosen.key,
            http_status=response.http_status,
            body=response.body,
            transport=response.transport,
            now=at,
        )
        if outcome.bucket == ANSWERED:
            status = STATUS_ANSWERED
        elif outcome.bucket == INCONCLUSIVE:
            status = STATUS_INCONCLUSIVE
        else:
            status = STATUS_FAILED
        detail = "" if status == STATUS_ANSWERED else _detail_for(outcome)
        checks.append(
            RuleCheck(
                rule=rule, status=status, key=chosen.key, alias=chosen.alias, detail=detail
            )
        )

    return SmokeCheckResult(checks=tuple(checks))


def format_smoke_line(check: RuleCheck) -> str:
    """Render one rule's smoke-check result as one report line.

    Pass the line through `litellm_maintainer.redact.redact` before
    printing it. A provider error can echo a request header or a
    credential; this function does not redact anything itself.
    """
    if check.status == STATUS_UNVERIFIED:
        return f"{check.rule}: UNVERIFIED ({check.detail})"
    if check.status == STATUS_ANSWERED:
        return f"{check.rule}: OK ({check.alias})"
    if check.status == STATUS_INCONCLUSIVE:
        return f"{check.rule}: INCONCLUSIVE ({check.alias}) -- {check.detail}"
    return f"{check.rule}: FAILED ({check.alias}) -- {check.detail}"


def build_smoke_payload(entry: SmokeEntry) -> dict[str, Any]:
    """Build the request body the smoke check sends for one entry.

    Requests a streamed response: `stream` is always `True`. A measured
    fact on 2026-07-26 forces this. The ChatGPT Passthrough Auth route
    answers a streamed request and fails a non-streamed one with
    "Unknown items in responses API response: []" -- a fault in
    litellm's own responses transformation, at every token budget, with
    a valid credential. So a non-streaming check reports a false
    failure on six working Aliases. A health check must not invent
    failures, so streaming is the only mode this check uses, for every
    rule, not only for the Offerings that need it.

    Sends no `temperature` key. The Claude 5 family accepts
    `temperature=1` only (docs/gotchas.md, "Claude 5 models reject
    temperature=0"); omitting the key avoids the trap for every model.
    """
    return {
        "model": entry.alias,
        "messages": list(SMOKE_MESSAGES),
        "max_tokens": SMOKE_MAX_TOKENS,
        "stream": True,
    }


extract_streamed_content = read_stream


def live_smoke_transport(
    entry: SmokeEntry, *, base_url: str, credential: str | None, timeout: float = 15.0
) -> TransportResponse:
    """Call the running proxy for one smoke entry, as a streamed request.

    Never invoked by a test and never invoked by `--dry-run`. Posts to
    `base_url` (the proxy's own `/v1/chat/completions` endpoint) with
    `Authorization: Bearer <credential>` -- the proxy's own
    `LITELLM_MASTER_KEY`. `litellm_maintainer.cli` resolves `credential`
    from the environment and passes it in. This function never reads
    the environment and never hardcodes a credential.

    The request always streams (see `build_smoke_payload` for why). The
    proxy resolves each Offering's own provider credential itself, from
    its own process environment -- for an ordinary Offering that
    credential is the caller's, forwarded through; for a
    `proxy_authenticated` Passthrough Auth Offering, the proxy holds and
    uses the credential itself, so a call under the proxy's own
    `LITELLM_MASTER_KEY` measures the real thing. This function never
    sees the Offering's own credential either way (docs/gotchas.md,
    "The proxy environment can differ from your .env file" -- that gap
    is exactly what this call is for).

    Reads a success body with `extract_streamed_content`, which ignores
    any chunk it cannot parse rather than reading a parse failure as a
    provider failure. `classify` sees a `choices` key when the stream
    carried at least one well-formed chunk, so a stream of only `[DONE]`
    and keep-alives reads as a malformed response, never as Answered. An
    error status is read from the plain JSON body, as before: the proxy
    does not stream an error.
    """
    import httpx

    headers = {"Content-Type": "application/json"}
    if credential:
        headers["Authorization"] = f"Bearer {credential}"

    payload = build_smoke_payload(entry)

    try:
        response = httpx.post(base_url, json=payload, headers=headers, timeout=timeout)
    except httpx.HTTPError:
        return TransportResponse(http_status=None, body=None, transport="timeout")

    if response.status_code >= 400:
        try:
            body = response.json()
        except ValueError:
            body = None
        return TransportResponse(http_status=response.status_code, body=body, transport=None)

    read = extract_streamed_content(response.text)
    # Answered when the stream carried at least one well-formed chunk,
    # not when it carried text. This check tests wiring: a chunk proves
    # the route resolved, the handler is registered, the base URL is
    # right and the credential worked. A reasoning model on a small
    # token budget spends it all on reasoning and emits empty content,
    # and that is a working route.
    #
    # A stream with no chunk but an error frame states its own
    # condition. `_streamed_body` hands that frame to `classify`; the
    # Prober builds its body with the same function, because the two
    # must agree.
    body: dict[str, Any] = _streamed_body(read)
    return TransportResponse(http_status=response.status_code, body=body, transport=None)
