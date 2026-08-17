"""Per-provider translation rules.

The Feed states one calling convention per Offering. litellm needs a
provider prefix it recognises. The mapping from an Offering to litellm
`litellm_params` is per provider and lives here, in code, not in the
Feed.

Read `docs/gotchas.md`, section "Warning: a wrong provider prefix sends
your traffic to the wrong vendor", before you add a rule. A wrong
prefix does not error. It sends the call to a real vendor API under
the wrong credential, silently.

This module holds one rule table, keyed by provider id. Add a rule for
a new provider by adding a table entry. Do not edit a rule already
written here.
"""

from __future__ import annotations

from typing import Any, Callable

from litellm_maintainer.feed import Offering, Provider

TranslationRule = Callable[[Offering, "Provider | None"], dict[str, Any]]


class UnknownProviderError(ValueError):
    """No translation rule, and no per-provider fallback, for a provider id."""


# A provider's credential environment variable, used only when the Feed
# provider record carries no `authentication.credential_hint`. Prefer
# the Feed's own hint; this table is the fallback, not the source of
# truth.
CREDENTIAL_FALLBACKS: dict[str, str] = {
    "opencode-go": "OPENCODE_API_KEY",
}


def _credential_variable(offering: Offering, provider: Provider | None) -> str:
    hint = provider.credential_hint if provider is not None else None
    if hint:
        return hint
    fallback = CREDENTIAL_FALLBACKS.get(offering.provider_id)
    if fallback is None:
        raise UnknownProviderError(
            f"no credential hint from the Feed and no fallback for provider "
            f"{offering.provider_id!r}"
        )
    return fallback


def native_prefix(prefix: str) -> TranslationRule:
    """Build a rule for a provider with a native litellm provider.

    Writes `model: <prefix>/<provider_model_id>` and no `api_base`, so
    litellm's own cost map and parameter handling apply. Read
    `docs/gotchas.md`, section "litellm cannot price a generic openai/
    model", before removing the missing `api_base` — it is deliberate,
    not an omission.

    Never uses the model's vendor name as `prefix`. `prefix` is the
    ROUTING provider litellm must dial, never the vendor the model id
    names inside it. See `docs/gotchas.md`, section "Warning: a wrong
    provider prefix sends your traffic to the wrong vendor".

    When `provider_model_id` already starts with `<prefix>/`, the
    prefix is not repeated. This covers a self-referential id such as
    OpenRouter's own free router, published as `openrouter/free`: adding
    a second `openrouter/` would misstate the model id litellm sends.
    """

    def rule(offering: Offering, provider: Provider | None) -> dict[str, Any]:
        model_id = offering.provider_model_id
        if not model_id.startswith(f"{prefix}/"):
            model_id = f"{prefix}/{model_id}"
        credential_variable = _credential_variable(offering, provider)
        return {
            "model": model_id,
            "api_key": f"os.environ/{credential_variable}",
        }

    return rule


# Gemini is the one vendor, among those this operator uses, whose Feed
# identifier carries a collection prefix litellm rejects: it reports
# `models/gemini-3.5-flash` where litellm needs `gemini/gemini-3.5-flash`.
# See `docs/gotchas.md`, section "Model identifiers need normalisation
# per provider". Checked against the audited Feed snapshot: no other
# provider this operator uses shows the same split between
# `provider_model_id` and `endpoint.model`. If one starts to, add its
# prefix here rather than special-casing the provider by name.
_COLLECTION_PREFIXES = ("models/",)


def _strip_collection_prefix(raw_model_id: str) -> str:
    for collection_prefix in _COLLECTION_PREFIXES:
        if raw_model_id.startswith(collection_prefix):
            return raw_model_id[len(collection_prefix) :]
    return raw_model_id


def gemini_native(offering: Offering, provider: Provider | None) -> dict[str, Any]:
    """The Gemini rule: native `gemini/` prefix, collection prefix stripped.

    Reads `endpoint.model` (the Feed's raw calling identifier, the field
    that actually carries the `models/` prefix) rather than
    `provider_model_id`, so the normalisation holds even if a future Feed
    revision stops pre-cleaning `provider_model_id`.
    """
    raw_model_id = offering.endpoint.get("model") or offering.provider_model_id
    model_id = _strip_collection_prefix(raw_model_id)
    credential_variable = _credential_variable(offering, provider)
    return {
        "model": f"gemini/{model_id}",
        "api_key": f"os.environ/{credential_variable}",
    }


# Warning: never give this provider an `anthropic/` prefix. The proxy
# sets `forward_client_headers_to_llm_api`, so a client's own
# `Authorization` header reaches the provider. An Anthropic-shaped
# route authenticates with `x-api-key` and leaves that header in place.
# The provider then reads the client's token and answers HTTP 401
# `InvalidApiKey`. `extra_headers` does not repair this, because the
# forwarded header wins over it. An OpenAI-shaped route writes
# `Authorization` itself and overwrites the client's token.
#
# The Qwen Token Plan speaks two protocols. Read the Feed's own
# `api_protocols` for `qwencloud-token-plan`: it publishes both
# `openai_chat_completions` and `anthropic_messages`. The two routes
# live at different base URLs. `api_base` is a fixed environment
# variable reference, because the Feed publishes no `base_url` for this
# provider.
QWEN_TOKEN_PLAN_OPENAI_BASE_URL_VAR = "QWEN_TOKEN_PLAN_OPENAI_BASE_URL"


def qwencloud_token_plan_openai(
    offering: Offering, provider: Provider | None
) -> dict[str, Any]:
    """The Qwen Token Plan rule: OpenAI-shaped calling convention."""
    credential_variable = _credential_variable(offering, provider)
    return {
        "model": f"openai/{offering.provider_model_id}",
        "api_base": f"os.environ/{QWEN_TOKEN_PLAN_OPENAI_BASE_URL_VAR}",
        "api_key": f"os.environ/{credential_variable}",
    }


def generic_openai_compatible(offering: Offering, provider: Provider | None) -> dict[str, Any]:
    """The generic OpenAI-compatible rule.

    Writes `model: openai/<provider_model_id>`, an explicit `api_base`,
    and an `api_key` as an `os.environ/NAME` reference. Use this rule
    for a provider that speaks the OpenAI chat-completions protocol
    under its own base URL, and that is not a litellm-native provider.
    """
    base_url = offering.endpoint.get("base_url") or (
        provider.default_base_url if provider is not None else None
    )
    if not base_url:
        raise ValueError(f"Offering {offering.id!r} has no base_url to translate against")
    credential_variable = _credential_variable(offering, provider)
    return {
        "model": f"openai/{offering.provider_model_id}",
        "api_base": base_url,
        "api_key": f"os.environ/{credential_variable}",
    }


# Response envelopes: data-driven, never provider-named
# ---------------------------------------------------------------
#
# An Offering that wraps its successful response declares the wrapper
# key at `endpoint.protocol_options.response_envelope_key` (see
# `tests/fixtures/feed-current.json`, where 356 Offerings carry it; the
# audited snapshot pinned for the acceptance test carries it on none).
# Read `docs/gotchas.md`, section "Some providers wrap successful
# responses".
#
# `_envelope_key` and `envelope_unwrapping` decide and act on that field
# alone. Neither reads `offering.provider_id`. A provider that starts
# declaring the key needs no change here; one that stops declaring it
# falls back to its own rule in `TRANSLATION_RULES` on its own.
#
# `ENVELOPE_HANDLER_PREFIX` is the litellm prefix of the one handler
# registered for this, `cline_provider.cline_llm`, listed under
# `litellm_settings.custom_provider_map` in the Generated Config
# whenever any entry uses it. It happens to share its name with the
# `cline` provider, since that provider is where the handler was first
# needed, but `_envelope_key` never reads `offering.provider_id` to
# decide whether to use it.
ENVELOPE_HANDLER_PREFIX = "cline"


def _envelope_key(offering: Offering, policy_key: str | None = None) -> str | None:
    """The wrapper key for this Offering, from the Feed or the operator.

    The Feed's own declaration wins. `policy_key` is the operator's
    `providers.<id>.response_envelope_key`, which states what the Feed
    omits: a provider can wrap its responses without saying so, and then
    every SUCCESS on it fails with "no 'choices'". This still never reads
    `offering.provider_id` — the operator names the provider in Policy,
    not this module.
    """
    protocol_options = offering.endpoint.get("protocol_options") or {}
    return protocol_options.get("response_envelope_key") or policy_key


def envelope_unwrapping(offering: Offering, provider: Provider | None) -> dict[str, Any]:
    """Route an Offering that declares a response envelope key.

    Chosen by `translate_offering` from `_envelope_key(offering)` alone,
    never from `offering.provider_id`.
    """
    base_url = offering.endpoint.get("base_url") or (
        provider.default_base_url if provider is not None else None
    )
    if not base_url:
        raise ValueError(f"Offering {offering.id!r} has no base_url to translate against")
    credential_variable = _credential_variable(offering, provider)
    return {
        "model": f"{ENVELOPE_HANDLER_PREFIX}/{offering.provider_model_id}",
        "api_base": base_url,
        "api_key": f"os.environ/{credential_variable}",
    }


# The per-provider rule table, used only when an Offering declares no
# response envelope key. `opencode-go` and `opencode-zen` use the
# generic rule: both speak the OpenAI chat-completions protocol at
# their own base URL, have no litellm-native provider, and need no
# response-unwrapping handler. `cline` and `cline-pass` fall back to the
# same generic rule here too: this is their *ordinary* rule for an
# Offering that (not yet, or no longer) declares the envelope key. When
# one of their Offerings does declare it, `translate_offering` routes
# through `envelope_unwrapping` before this table is ever consulted.
TRANSLATION_RULES: dict[str, TranslationRule] = {
    "opencode-go": generic_openai_compatible,
    "opencode-zen": generic_openai_compatible,
    "cline": generic_openai_compatible,
    "cline-pass": generic_openai_compatible,
    "gemini": gemini_native,
    "groq": native_prefix("groq"),
    "openrouter": native_prefix("openrouter"),
    "qwencloud-token-plan": qwencloud_token_plan_openai,
}


def translate_offering(
    offering: Offering,
    provider: Provider | None,
    *,
    override: dict[str, Any] | None = None,
    policy_envelope_key: str | None = None,
) -> dict[str, Any]:
    """Translate one Offering into litellm `litellm_params`.

    Check `_envelope_key(offering, policy_envelope_key)` first. When it is
    set, route through `envelope_unwrapping`, regardless of
    `offering.provider_id`. Otherwise look up the rule for
    `offering.provider_id` in `TRANSLATION_RULES`. Raise
    `UnknownProviderError` when no rule is registered. `override`, when
    given, is Policy's translation override (per-provider, per-Offering,
    or both merged by the caller): its keys replace the rule's output,
    key by key. `policy_envelope_key` is the operator's declaration for a
    provider whose Feed entry omits the wrapper key.
    """
    if _envelope_key(offering, policy_envelope_key):
        params = envelope_unwrapping(offering, provider)
    else:
        rule = TRANSLATION_RULES.get(offering.provider_id)
        if rule is None:
            raise UnknownProviderError(
                f"no translation rule registered for provider {offering.provider_id!r}"
            )
        params = rule(offering, provider)
    if override:
        params = {**params, **override}
    return params
