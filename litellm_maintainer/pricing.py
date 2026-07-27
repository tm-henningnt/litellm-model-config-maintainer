"""Cost metadata injection and Feed-shape validation.

See CONTEXT.md, the spec's "Cost data" and "Safety" sections, and
`docs/gotchas.md`, "litellm cannot price a generic openai/ model" — the
reason this module exists.

Every function here is a pure transform. It takes an Offering, a set of
translated `litellm_params`, or a Feed, and returns a value. It performs
no network call, no filesystem read and no clock read. `plan.py` calls
into this module for every Discovered Offering it admits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from litellm_maintainer.feed import Feed, Offering

# The Feed states a token rate per 1,000,000 tokens. litellm's
# `model_info` wants a rate per single token. Divide by one million.
# Get this factor wrong and every spend report reads a million times
# too high or too low, with no error to catch it. See
# `tests/test_pricing.py`, the test that checks one converted value by
# hand.
_TOKENS_PER_MILLION = 1_000_000

# The Feed's metering unit for an ordinary chat model. An Offering
# metered in anything else (`credits`, `images`, `characters`,
# `video_seconds`, `audio_seconds` all appear in the audited snapshot)
# receives no cost metadata: litellm's per-token fields cannot state
# its rate.
TOKEN_METERING = "tokens"

# The `model_info` key that marks a subscription-pool rate as a list
# price, never an amount billed. A reader who greps `model_info` in the
# Generated Config and finds this key on an entry knows, with no
# comment needed, not to sum its `input_cost_per_token` into an
# invoice.
SUBSCRIPTION_LIST_PRICE_KEY = "litellm_maintainer_subscription_list_price"


@dataclass(frozen=True)
class PricingContradiction:
    """An Offering priced `free` while stating a non-zero token rate.

    Reported, and treated as paid: `cost_model_info` still returns its
    cost metadata, because a mirror provider once published the same
    model twice with conflicting prices, and whichever copy survived
    the merge depended on array order (spec, "Safety"). Treating the
    contradiction as paid is the safe direction; treating it as free
    would risk publishing a paid model as free.
    """

    offering_id: str
    input_usd_per_1m_tokens: float | None
    output_usd_per_1m_tokens: float | None

    @property
    def message(self) -> str:
        return (
            f"{self.offering_id}: priced 'free' but states a non-zero token "
            f"rate (input={self.input_usd_per_1m_tokens}, "
            f"output={self.output_usd_per_1m_tokens} per 1M tokens); "
            "treated as paid"
        )


@dataclass(frozen=True)
class DuplicateProviderModelId:
    """Two or more Offerings sharing a provider id and model identifier.

    Reported rather than let one silently overwrite the other. `plan`
    still writes a Generated Config; this is a report entry, not a
    refusal.
    """

    provider_id: str
    provider_model_id: str
    offering_ids: tuple[str, ...]

    @property
    def message(self) -> str:
        return (
            f"duplicate provider+model id ({self.provider_id!r}, "
            f"{self.provider_model_id!r}): Offerings "
            f"{list(self.offering_ids)!r} all claim it"
        )


def is_native_litellm_prefix(litellm_params: dict[str, Any]) -> bool:
    """Whether `litellm_params` was produced by a native-prefix rule.

    litellm's own price map resolves an entry by provider prefix and
    model name alone when the entry carries no explicit `api_base`
    (`docs/gotchas.md`, "litellm cannot price a generic openai/
    model"). Every translation rule that dials a litellm-native
    provider (`translate.native_prefix`, `translate.gemini_native`)
    omits `api_base` on purpose, for exactly this reason. Every rule
    that dials a generic base URL (`generic_openai_compatible`,
    `envelope_unwrapping`, the Qwen Token Plan rule) always sets one.

    Reading the presence of `api_base` off the already-translated
    `litellm_params` — rather than asking which provider or which rule
    produced it — is the one signal that cannot disagree with
    `translate_offering`: it is the same fact that rule already
    decided, read back rather than re-derived.
    """
    return "api_base" not in litellm_params


def cost_model_info(
    offering: Offering, litellm_params: dict[str, Any]
) -> tuple[dict[str, Any], PricingContradiction | None]:
    """Cost metadata for one Offering, and a contradiction if found.

    Returns `({}, None)` when no cost metadata applies:

    - `litellm_params` carries a native litellm prefix
      (`is_native_litellm_prefix`); litellm prices it already.
    - The Feed meters the Offering in a unit other than `tokens`.
    - The Feed states no rate at all (an `unknown` or `free_tier`
      Offering commonly carries neither figure).

    Otherwise returns a `model_info` dict carrying
    `input_cost_per_token` and `output_cost_per_token`, converted from
    the Feed's rate per 1,000,000 tokens, plus
    `{SUBSCRIPTION_LIST_PRICE_KEY: True}` when `offering.pricing_kind`
    is `subscription_included`.

    A `PricingContradiction` is also returned when `offering.pricing_kind`
    is `free` but the Feed states a non-zero rate. The cost metadata is
    still returned in that case — the contradiction is reported, not
    hidden by suppressing the metadata.
    """
    if is_native_litellm_prefix(litellm_params):
        return {}, None

    pricing = offering.pricing
    if pricing.get("metering") != TOKEN_METERING:
        return {}, None

    input_rate = pricing.get("input_usd_per_1m_tokens")
    output_rate = pricing.get("output_usd_per_1m_tokens")
    if input_rate is None or output_rate is None:
        return {}, None

    contradiction: PricingContradiction | None = None
    if offering.pricing_kind == "free" and (input_rate or output_rate):
        contradiction = PricingContradiction(
            offering_id=offering.id,
            input_usd_per_1m_tokens=input_rate,
            output_usd_per_1m_tokens=output_rate,
        )

    info: dict[str, Any] = {
        "input_cost_per_token": input_rate / _TOKENS_PER_MILLION,
        "output_cost_per_token": output_rate / _TOKENS_PER_MILLION,
    }
    if offering.pricing_kind == "subscription_included":
        info[SUBSCRIPTION_LIST_PRICE_KEY] = True
    return info, contradiction


def find_duplicate_provider_model_ids(feed: Feed) -> tuple[DuplicateProviderModelId, ...]:
    """Offerings in `feed` sharing a provider id and a model identifier.

    Reads every Offering the Feed publishes, regardless of Policy or
    Selection: the hazard is in the Feed document itself. Spec,
    "Safety": "a mirror provider once listed the same model twice with
    conflicting prices, and whichever copy survived the merge depended
    on array order, so a paid model could have been published as
    free."

    Returns one `DuplicateProviderModelId` per colliding key, in Feed
    order. An empty tuple means no collision was found.
    """
    ids_by_key: dict[tuple[str, str], list[str]] = {}
    for offering in feed.offerings:
        key = (offering.provider_id, offering.provider_model_id)
        ids_by_key.setdefault(key, []).append(offering.id)
    return tuple(
        DuplicateProviderModelId(
            provider_id=provider_id,
            provider_model_id=provider_model_id,
            offering_ids=tuple(ids),
        )
        for (provider_id, provider_model_id), ids in ids_by_key.items()
        if len(ids) > 1
    )


def summarize_feed_notices(notices: Iterable[Any]) -> tuple[str, ...]:
    """One short line per Feed notice, tolerant of an unfamiliar shape.

    The Feed's `notices` are objects of varying shape: one carries
    `stale_aliases`, others carry `offering_ids`, others carry pricing
    figures (spec, "Safety"). Every notice emitted by every collector
    seen so far carries a `message` field and, alongside it, a
    `collector` field naming which collector raised it. This reads
    only those two fields.

    A notice that is not a mapping, or carries no `message`, is
    skipped rather than raised on: one unfamiliar notice reduces the
    report by one line and never fails the run (spec, "Safety": "Read
    the Feed's own notices... A notice you cannot read must reduce to
    reporting less, never to a failed run.").
    """
    lines: list[str] = []
    for notice in notices:
        try:
            message = notice.get("message")
            collector = notice.get("collector")
        except AttributeError:
            # Not a mapping at all. Skip it: one line fewer, not a
            # failed run.
            continue
        if not message:
            continue
        if collector:
            lines.append(f"{collector}: {message}")
        else:
            lines.append(str(message))
    return tuple(lines)
