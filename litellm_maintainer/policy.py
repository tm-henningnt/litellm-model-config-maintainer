"""Policy schema, loader and validator.

Policy is the operator's hand-written declaration of which Offerings the
Generator may use, under which Aliases, and which Offerings are
deliberately Withheld and why. See CONTEXT.md and the spec's
"Selection", "Naming", "Declared Offerings", "Probing" and "Schedule"
sections.

A human writes Policy once, by hand, or edits it by hand later. Nothing
in this project writes to Policy at run time. `load_policy` only reads
it.

This module uses hand-written dataclasses and validation. It does not
add a schema library as a dependency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from litellm_maintainer.naming import alias_for, derive_alias  # noqa: F401

# `derive_alias` and `alias_for` live in `litellm_maintainer.naming`.
# Re-exported here so an existing
# `from litellm_maintainer.policy import derive_alias` (or `alias_for`)
# keeps working. Prefer importing from `naming` in new code.

VALID_MODES = {"all", "named"}
# The Feed publishes five pricing kinds. `subscription_included` is one
# of them: 56 Offerings carry it in the audited snapshot. Reject a kind
# the Feed never emits, and accept every kind it does.
VALID_PRICING_KINDS = {"free", "free_tier", "paid", "unknown", "subscription_included"}

TOP_LEVEL_KEYS = {
    "providers",
    "quality",
    "approved_candidates",
    "naming",
    "withheld",
    "declared",
    "pacing",
    "schedule",
    "safety",
    "translation_overrides",
    "proxy_settings",
    "feed",
    "client_facing_variants",
}
CLIENT_FACING_VARIANT_KEYS = {
    "suffix",
    "minimum_context_tokens",
    "operator_stated",
}

# `translation_overrides` and `proxy_settings` are optional. Neither is
# the norm: leave `translation_overrides` out of a Policy that needs no
# per-Offering exception, and leave `proxy_settings` out of a Policy
# that adds no non-derived proxy setting. A Policy naming neither
# produces exactly the Generated Config this tool always produced.
# `feed` is optional too. A Policy without it names no Feed URL, so
# `fetch` has nothing to download and says so; every other command
# still reads the Feed Document from `--feed`.
REQUIRED_TOP_LEVEL_KEYS = TOP_LEVEL_KEYS - {
    "translation_overrides",
    "proxy_settings",
    "feed",
    "client_facing_variants",
}

PROVIDER_RULE_KEYS = {
    "mode",
    "pricing",
    "models",
    "translation",
    "entitlement",
    "response_envelope_key",
    "plan_edition",
}
FEED_KEYS = {"url", "credential_env", "maximum_age_hours"}

# The two Entitlement kinds. `per_model` is the default for a provider
# that states none, because it is the reading that never over-claims:
# it treats each Offering's failure as its own. See CONTEXT.md,
# "Entitlement", and ADR 0004 for why neither kind ever propagates a
# failure to a sibling.
SHARED_POOL = "shared_pool"
PER_MODEL = "per_model"
VALID_ENTITLEMENTS = {SHARED_POOL, PER_MODEL}
NAMING_KEYS = {"provider_labels", "alias_overrides", "alias_prefix"}
PROXY_SETTINGS_KEYS = {"general_settings", "litellm_settings"}
DECLARED_KEYS = {
    "alias",
    "litellm_params",
    "passthrough_auth",
    "proxy_authenticated",
    "supersedes",
    "model_info",
    "group",
    "capabilities",
    "variant_of",
    "entitlement",
    "entitlement_pool",
    "sub_allowance",
}
PACING_ENTRY_KEYS = {"concurrency", "minimum_interval_seconds"}
SCHEDULE_KEYS = {"enabled", "interval_minutes", "require_proxy", "maximum_staleness_hours"}
SAFETY_KEYS = {"maximum_removal_share", "snapshot_count"}
QUALITY_KEYS = {"minimum_coding_score"}


class PolicyError(ValueError):
    """An invalid Policy. The message names the offending key."""


def _require_dict(value: Any, key: str) -> dict:
    if not isinstance(value, dict):
        raise PolicyError(f"'{key}' must be a mapping")
    return value


def _require_list(value: Any, key: str) -> list:
    if not isinstance(value, list):
        raise PolicyError(f"'{key}' must be a list")
    return value


def _require_str(value: Any, key: str) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyError(f"'{key}' must be a non-empty string")
    return value


def _require_number(value: Any, key: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PolicyError(f"'{key}' must be a number")
    return float(value)


def _require_positive_int(value: Any, key: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PolicyError(f"'{key}' must be a positive integer")
    return value


def _require_bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise PolicyError(f"'{key}' must be a boolean")
    return value


def _reject_unknown_keys(raw: dict, allowed: set[str], prefix: str) -> None:
    unknown = set(raw) - allowed
    if unknown:
        bad = sorted(unknown)[0]
        raise PolicyError(f"'{prefix}.{bad}' is not a recognised key")


@dataclass(frozen=True)
class ProviderRule:
    """One provider's selection rule.

    `mode` is `all` (take every Offering after the baseline filter) or
    `named` (take only the Offerings in `models`). `pricing`, when set,
    admits only those pricing kinds. Never set `pricing` for a provider
    whose free tier is an account entitlement — see the spec's
    "Selection" section. `translation` overrides the per-provider
    translation rule for this provider; leave it unset for the ordinary
    case.

    `entitlement` states whether the provider bills from one shared pool
    or from per-model limits. It changes how the Entitlement view reads a
    set of failures, and nothing else: it never writes Health State and
    never removes an Offering from the Generated Config. See ADR 0004.

    `response_envelope_key` names the key this provider wraps a
    successful response under, for a provider whose Feed entry declares
    no `endpoint.protocol_options.response_envelope_key`. The Feed's own
    declaration always wins; this states what the Feed omits. Leave it
    unset unless a Probe measured the wrapper. See `docs/gotchas.md`,
    section "Some providers wrap successful responses".

    `plan_edition` names the subscription edition the operator holds, for
    a provider that sells one roster per edition. The Feed lists each
    Offering's editions at `pricing.subscription.plan_editions`, and only
    an Offering naming this edition is admitted.

    Warning: an Offering no plan covers publishes no `plan_editions`, so
    this filter excludes it. Never set `plan_edition` for a provider
    whose catalogue mixes subscription and pay-as-you-go Offerings, or
    the pay-as-you-go ones disappear. Only the operator knows which
    edition they hold; it can never be inferred from the provider id.
    """

    mode: str
    pricing: tuple[str, ...] | None = None
    models: tuple[str, ...] | None = None
    translation: dict[str, Any] | None = None
    entitlement: str = PER_MODEL
    response_envelope_key: str | None = None
    plan_edition: str | None = None


@dataclass(frozen=True)
class FeedSource:
    """Where `fetch` downloads the Feed Document from.

    `url` is the Feed's own address. `credential_env` names the
    environment variable holding a bearer token, when the Feed needs
    one; it never holds the token itself, exactly as a Declared
    Offering never holds a key. `maximum_age_hours` is the age past
    which `doctor` and `status` call the Feed Document stale.
    """

    url: str
    credential_env: str | None = None
    maximum_age_hours: float = 24.0


@dataclass(frozen=True)
class Quality:
    """The quality gate. An Offering scoring below `minimum_coding_score`
    is not admitted. An Offering with no score is a Candidate."""

    minimum_coding_score: float


@dataclass(frozen=True)
class Naming:
    """Alias derivation rules.

    `provider_labels` maps a provider id to the label used in an Alias.
    `alias_overrides` maps an Offering id to the exact Alias to use
    instead of the mechanical derivation. `alias_prefix` is prepended to
    every derived Alias.
    """

    provider_labels: dict[str, str]
    alias_overrides: dict[str, str]
    alias_prefix: str


@dataclass(frozen=True)
class DeclaredOffering:
    """An Offering the Feed does not publish, declared by the operator.

    Passed through to Generated Config verbatim. `passthrough_auth`
    marks an Offering whose credentials come from the calling client:
    the Prober never probes it, and a quota or authentication failure on
    it is reported but never Excludes it. `supersedes` names the Feed
    Offering id this declaration replaces, resolving a name collision.

    `proxy_authenticated` marks a Passthrough Auth Offering whose
    credential the proxy in fact holds and resolves itself, from its
    own process environment, rather than from the calling client. This
    is unusual: a call carrying only the proxy's own master key still
    returns a completion. It governs the smoke check only
    (`litellm_maintainer.smoke.build_smoke_entries`). It changes
    nothing about the Prober and nothing about `reduce`: the Prober
    still skips every Passthrough Auth Offering, because there is still
    no direct route to construct for a litellm-internal OAuth provider,
    and the Passthrough Auth exemption in `reduce` still stands.
    Defaults to `False`, so a Policy that never sets it behaves exactly
    as before.

    `group` is the heading the Generated Config prints this Offering
    under, so a reader scrolling the file can see where one set of
    Offerings ends and the next begins. It names nothing the code acts
    on. A Discovered Offering takes its heading from the Feed's provider
    name instead; only a Declared Offering needs this, because the Feed
    names no provider for one.

    `capabilities` is what the operator states this Offering can do. The
    Feed publishes a capability list for a Discovered Offering; it
    publishes nothing at all for a Declared one, so a Guidance Row for a
    direct vendor entry claimed no capabilities and an agent could not
    tell that the strongest models the proxy serves support reasoning.
    It is read by `guidance` alone and never reaches Generated Config,
    because litellm has no such key. Empty by default: a Policy that
    states none claims none.

    `model_info` carries this Offering's Stated Limit, among anything
    else litellm accepts there. It reaches Generated Config verbatim.
    """

    alias: str
    litellm_params: dict[str, Any]
    passthrough_auth: bool = False
    proxy_authenticated: bool = False
    supersedes: str | None = None
    model_info: dict[str, Any] | None = None
    group: str | None = None
    capabilities: tuple[str, ...] = ()
    # The Alias this one is a Client-Facing Variant of. Set it on the
    # variant, naming its plain sibling. `guidance` then folds the pair into
    # one Guidance Row instead of reporting one model twice, and the variant
    # becomes that row's Route's `wide_alias`.
    #
    # Stated, never inferred. The suffix that marks a variant is an operator
    # setting (`client_facing_variants.suffix`), so pairing by name would
    # break the moment it changed. `None` for an ordinary Declared Offering.
    variant_of: str | None = None

    # Whether one Offering's quota exhaustion says anything about the
    # others billed to the same credential. `per_model` is the default,
    # because it never over-claims. See CONTEXT.md, "Entitlement".
    entitlement: str = PER_MODEL

    # Names this Offering's pool explicitly, overriding the credential
    # rule. Use it for the two cases the credential gets wrong: two keys
    # billed to one account, and one key spanning a subscription plus
    # pay-as-you-go. A Passthrough Auth Offering carries no credential
    # at all, so a pool it belongs to can only be named here.
    entitlement_pool: str | None = None

    # A sub-allowance is capped INSIDE its pool. Its own exhaustion says
    # nothing about the pool, but the pool's exhaustion still reaches
    # it. One-way containment: out, no; in, yes.
    #
    # The operator's Claude subscription is the measured case. At most
    # half the weekly quota may go to `claude-fable-5`, so fable can run
    # out while the rest has room, and the whole quota can run out while
    # fable's own half is untouched. Flagging fable gets both right
    # without encoding the percentage, which the provider can change
    # without telling us.
    sub_allowance: bool = False

    @property
    def health_key(self) -> str:
        """The Health Key this Offering's observations belong under.

        A Client-Facing Variant reaches the same Offering with the same
        wire request, and "the provider never sees the difference"
        (CONTEXT.md). So the pair shares ONE health record, keyed by the
        Alias the variant widens.

        Two records would be an anomaly, not a nuance: they can never
        legitimately disagree, and when they drifted apart the effect
        was concrete. An exhausted quota Excluded `claude-opus-5` and
        left `claude-opus-5[1m]` in the Generated Config, offering a
        client an Alias certain to fail. ADR 0007 already made the pair
        share one Stated Limit for the neighbouring reason; health is
        the same argument.
        """
        return self.variant_of or self.alias


@dataclass(frozen=True)
class Pacing:
    """A provider's probing pace: how many Probes run at once, and the
    minimum time between two Probes to the same provider."""

    concurrency: int
    minimum_interval_seconds: float


@dataclass(frozen=True)
class Schedule:
    """When the maintainer runs. A Policy edit changes this; no service
    reload is needed."""

    enabled: bool
    interval_minutes: int
    require_proxy: bool
    maximum_staleness_hours: float


@dataclass(frozen=True)
class Safety:
    """Guard rails the Generator checks before it writes."""

    maximum_removal_share: float
    snapshot_count: int


@dataclass(frozen=True)
class ProxySettings:
    """The non-derived parts of the Generated Config, the operator's own.

    Everything the Generator does not derive from the Feed still
    belongs in the Generated Config, so the file the proxy loads never
    needs a manual merge (spec-corrections.md and the scope change it
    records). `general_settings` and `litellm_settings` are each an
    arbitrary mapping, emitted into the Generated Config verbatim.

    `litellm_settings.custom_provider_map` is the one exception: the
    Generator always derives it from the Feed's envelope routing (see
    `plan._any_entry_uses_envelope_handler`), because a hand-written map
    goes stale the moment the Feed's routing changes, with no symptom
    to notice by. A value here under that key is never emitted; `plan`
    reports the conflict instead of picking silently. See
    `PlanReport.custom_provider_map_conflict`.

    Both mappings default to empty, so a Policy naming neither produces
    exactly the Generated Config this tool always produced.
    """

    general_settings: dict[str, Any] = field(default_factory=dict)
    litellm_settings: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClientFacingVariants:
    """When the Generator adds a Client-Facing Variant, and for which Offering.

    A calling client reads its own context budget out of the Alias name, so
    a wide Offering needs a second Alias carrying `suffix` to be used at
    its full size. See CONTEXT.md, "Client-Facing Variant", and ADR 0007.

    `minimum_context_tokens` is the Feed-stated context window at which an
    Offering qualifies. The Feed decides, so the list cannot go stale: a
    model that grows its window earns a variant on the next run, and one
    the Feed does not size earns none.

    `operator_stated` names an Offering that qualifies anyway, mapped to
    the operator's reason. It exists for a model the Feed has not sized
    yet, where the operator holds a figure the Feed does not publish. It
    grants the variant only. It states no Stated Limit, because a variant
    needs none and an unverified figure would add a claim without adding a
    capability (ADR 0006).

    Warning: a wider budget is a claim about the client, never about the
    provider. Naming an Offering here that refuses the wider window trades
    an early compaction for a late refusal. The ChatGPT subscription seats
    are the measured example: they accept about 380,000 tokens, not 1M.
    """

    suffix: str = "[1m]"
    minimum_context_tokens: int = 1_000_000
    operator_stated: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Policy:
    """The operator's full, validated Policy."""

    providers: dict[str, ProviderRule]
    quality: Quality
    approved_candidates: tuple[str, ...]
    naming: Naming
    withheld: dict[str, str]
    declared: tuple[DeclaredOffering, ...]
    pacing: dict[str, Pacing]
    schedule: Schedule
    safety: Safety
    # A per-Offering translation override, keyed by Offering id. The
    # escape hatch for a single named Offering; a per-provider exception
    # belongs in `providers.<id>.translation` instead. Its keys replace
    # the translated `litellm_params`, key by key, after any per-provider
    # override already applied. Defaults to empty: most Policy files need
    # none.
    translation_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    # The operator's non-derived proxy settings, passed through verbatim
    # by `plan` — see `ProxySettings`. Defaults to both mappings empty,
    # so a Policy that never sets `proxy_settings` behaves exactly as
    # before.
    proxy_settings: ProxySettings = field(default_factory=ProxySettings)
    # Where `fetch` downloads the Feed Document from. `None` when the
    # Policy names no `feed` block: `fetch` then has nothing to do and
    # reports so, while every other command still reads the Feed
    # Document from the path it is given.
    feed: FeedSource | None = None
    # When to add a Client-Facing Variant. `None` when the Policy names no
    # `client_facing_variants` block, and the Generator then adds none, so
    # a Policy written before this feature produces exactly the Generated
    # Config it always produced.
    client_facing_variants: ClientFacingVariants | None = None


def _parse_provider_rule(provider_id: str, raw: Any) -> ProviderRule:
    prefix = f"providers.{provider_id}"
    raw = _require_dict(raw, prefix)
    _reject_unknown_keys(raw, PROVIDER_RULE_KEYS, prefix)

    mode = _require_str(raw.get("mode"), f"{prefix}.mode")
    if mode not in VALID_MODES:
        raise PolicyError(f"'{prefix}.mode' must be one of {sorted(VALID_MODES)}, got {mode!r}")

    pricing_raw = raw.get("pricing")
    pricing: tuple[str, ...] | None = None
    if pricing_raw is not None:
        pricing_list = _require_list(pricing_raw, f"{prefix}.pricing")
        for kind in pricing_list:
            if kind not in VALID_PRICING_KINDS:
                raise PolicyError(
                    f"'{prefix}.pricing' names an unrecognised pricing kind: {kind!r}"
                )
        pricing = tuple(pricing_list)

    models_raw = raw.get("models")
    if mode == "named" and not models_raw:
        raise PolicyError(f"'{prefix}.models' is required when mode is 'named'")
    models: tuple[str, ...] | None = None
    if models_raw is not None:
        models = tuple(_require_list(models_raw, f"{prefix}.models"))

    translation = raw.get("translation")
    if translation is not None:
        translation = dict(_require_dict(translation, f"{prefix}.translation"))

    entitlement = raw.get("entitlement", PER_MODEL)
    if entitlement not in VALID_ENTITLEMENTS:
        raise PolicyError(
            f"'{prefix}.entitlement' must be one of "
            f"{sorted(VALID_ENTITLEMENTS)}, got {entitlement!r}"
        )

    plan_edition = raw.get("plan_edition")
    if plan_edition is not None:
        plan_edition = _require_str(plan_edition, f"{prefix}.plan_edition")

    envelope_key = raw.get("response_envelope_key")
    if envelope_key is not None:
        envelope_key = _require_str(envelope_key, f"{prefix}.response_envelope_key")
        if not envelope_key:
            raise PolicyError(f"'{prefix}.response_envelope_key' must not be empty")

    return ProviderRule(
        mode=mode,
        pricing=pricing,
        models=models,
        translation=translation,
        entitlement=entitlement,
        response_envelope_key=envelope_key,
        plan_edition=plan_edition,
    )


def _parse_feed(raw: Any) -> FeedSource | None:
    if raw is None:
        return None
    raw = _require_dict(raw, "feed")
    _reject_unknown_keys(raw, FEED_KEYS, "feed")

    url = _require_str(raw.get("url"), "feed.url")

    credential_env = raw.get("credential_env")
    if credential_env is not None:
        credential_env = _require_str(credential_env, "feed.credential_env")

    maximum_age_hours = 24.0
    if raw.get("maximum_age_hours") is not None:
        maximum_age_hours = _require_number(
            raw["maximum_age_hours"], "feed.maximum_age_hours"
        )
        if maximum_age_hours <= 0:
            raise PolicyError("'feed.maximum_age_hours' must be greater than zero")

    return FeedSource(
        url=url,
        credential_env=credential_env,
        maximum_age_hours=maximum_age_hours,
    )


def _parse_providers(raw: Any) -> dict[str, ProviderRule]:
    raw = _require_dict(raw, "providers")
    return {
        provider_id: _parse_provider_rule(provider_id, rule)
        for provider_id, rule in raw.items()
    }


def _parse_quality(raw: Any) -> Quality:
    raw = _require_dict(raw, "quality")
    _reject_unknown_keys(raw, QUALITY_KEYS, "quality")
    if "minimum_coding_score" not in raw:
        raise PolicyError("'quality.minimum_coding_score' is required")
    score = _require_number(raw["minimum_coding_score"], "quality.minimum_coding_score")
    return Quality(minimum_coding_score=score)


def _parse_approved_candidates(raw: Any) -> tuple[str, ...]:
    raw = _require_list(raw, "approved_candidates")
    for i, item in enumerate(raw):
        _require_str(item, f"approved_candidates[{i}]")
    return tuple(raw)


def _parse_naming(raw: Any) -> Naming:
    raw = _require_dict(raw, "naming")
    _reject_unknown_keys(raw, NAMING_KEYS, "naming")
    for required in ("provider_labels", "alias_overrides", "alias_prefix"):
        if required not in raw:
            raise PolicyError(f"'naming.{required}' is required")
    provider_labels = _require_dict(raw["provider_labels"], "naming.provider_labels")
    for k, v in provider_labels.items():
        _require_str(v, f"naming.provider_labels.{k}")
    alias_overrides = _require_dict(raw["alias_overrides"], "naming.alias_overrides")
    for k, v in alias_overrides.items():
        _require_str(v, f"naming.alias_overrides.{k}")
    alias_prefix = _require_str(raw["alias_prefix"], "naming.alias_prefix")
    return Naming(
        provider_labels=dict(provider_labels),
        alias_overrides=dict(alias_overrides),
        alias_prefix=alias_prefix,
    )


def _parse_withheld(raw: Any) -> dict[str, str]:
    raw = _require_dict(raw, "withheld")
    for k, v in raw.items():
        _require_str(v, f"withheld.{k}")
    return dict(raw)


def _parse_declared_offering(index: int, raw: Any) -> DeclaredOffering:
    prefix = f"declared[{index}]"
    raw = _require_dict(raw, prefix)
    _reject_unknown_keys(raw, DECLARED_KEYS, prefix)
    if "alias" not in raw:
        raise PolicyError(f"'{prefix}.alias' is required")
    alias = _require_str(raw["alias"], f"{prefix}.alias")
    if "litellm_params" not in raw:
        raise PolicyError(f"'{prefix}.litellm_params' is required")
    litellm_params = _require_dict(raw["litellm_params"], f"{prefix}.litellm_params")
    if "model" not in litellm_params:
        raise PolicyError(f"'{prefix}.litellm_params.model' is required")
    passthrough_auth = raw.get("passthrough_auth", False)
    passthrough_auth = _require_bool(passthrough_auth, f"{prefix}.passthrough_auth")
    proxy_authenticated = raw.get("proxy_authenticated", False)
    proxy_authenticated = _require_bool(proxy_authenticated, f"{prefix}.proxy_authenticated")
    supersedes = raw.get("supersedes")
    if supersedes is not None:
        supersedes = _require_str(supersedes, f"{prefix}.supersedes")
    model_info = raw.get("model_info")
    if model_info is not None:
        model_info = dict(_require_dict(model_info, f"{prefix}.model_info"))
    group = raw.get("group")
    if group is not None:
        group = _require_str(group, f"{prefix}.group")
    capabilities = _parse_declared_capabilities(raw.get("capabilities"), prefix)
    variant_of = raw.get("variant_of")
    if variant_of is not None:
        variant_of = _require_str(variant_of, f"{prefix}.variant_of")
        if variant_of == alias:
            raise PolicyError(
                f"'{prefix}.variant_of' names its own Alias {alias!r}. A "
                "Client-Facing Variant names the plain Alias it widens."
            )
    entitlement = raw.get("entitlement", PER_MODEL)
    if entitlement not in VALID_ENTITLEMENTS:
        raise PolicyError(
            f"'{prefix}.entitlement' must be one of "
            f"{sorted(VALID_ENTITLEMENTS)}, got {entitlement!r}"
        )
    entitlement_pool = raw.get("entitlement_pool")
    if entitlement_pool is not None:
        entitlement_pool = _require_str(entitlement_pool, f"{prefix}.entitlement_pool")
    sub_allowance = bool(raw.get("sub_allowance", False))
    if sub_allowance and entitlement_pool is None and entitlement != SHARED_POOL:
        raise PolicyError(
            f"'{prefix}.sub_allowance' is set, but this Offering names no "
            "pool to sit inside. A sub-allowance is capped WITHIN a pool, so "
            "set 'entitlement_pool', or 'entitlement: shared_pool' if the "
            "credential already groups it."
        )
    return DeclaredOffering(
        alias=alias,
        litellm_params=dict(litellm_params),
        passthrough_auth=passthrough_auth,
        proxy_authenticated=proxy_authenticated,
        supersedes=supersedes,
        model_info=model_info,
        group=group,
        capabilities=capabilities,
        variant_of=variant_of,
        entitlement=entitlement,
        entitlement_pool=entitlement_pool,
        sub_allowance=sub_allowance,
    )


def _parse_declared_capabilities(raw: Any, prefix: str) -> tuple[str, ...]:
    """The capabilities a Declared Offering states, as a tuple of strings.

    Returns an empty tuple when the Policy states none, so a Declared
    Offering that claims nothing is indistinguishable from one written
    before this key existed.
    """
    if raw is None:
        return ()
    items = _require_list(raw, f"{prefix}.capabilities")
    return tuple(
        _require_str(item, f"{prefix}.capabilities[{i}]") for i, item in enumerate(items)
    )


def _parse_declared(raw: Any) -> tuple[DeclaredOffering, ...]:
    raw = _require_list(raw, "declared")
    declared = tuple(_parse_declared_offering(i, item) for i, item in enumerate(raw))
    # A `variant_of` naming an Alias no Declared Offering defines is a typo.
    # Refuse it here: left alone it produces exactly the duplicate Guidance
    # Row that this key exists to remove, which is a silent failure.
    aliases = {d.alias for d in declared}
    for index, entry in enumerate(declared):
        if entry.variant_of is not None and entry.variant_of not in aliases:
            raise PolicyError(
                f"'declared[{index}].variant_of' names {entry.variant_of!r}, "
                "which no Declared Offering defines"
            )
    return declared


def _parse_pacing_entry(provider_id: str, raw: Any) -> Pacing:
    prefix = f"pacing.{provider_id}"
    raw = _require_dict(raw, prefix)
    _reject_unknown_keys(raw, PACING_ENTRY_KEYS, prefix)
    if "concurrency" not in raw:
        raise PolicyError(f"'{prefix}.concurrency' is required")
    concurrency = _require_positive_int(raw["concurrency"], f"{prefix}.concurrency")
    if "minimum_interval_seconds" not in raw:
        raise PolicyError(f"'{prefix}.minimum_interval_seconds' is required")
    interval = _require_number(raw["minimum_interval_seconds"], f"{prefix}.minimum_interval_seconds")
    if interval < 0:
        raise PolicyError(f"'{prefix}.minimum_interval_seconds' must not be negative")
    return Pacing(concurrency=concurrency, minimum_interval_seconds=interval)


def _parse_pacing(raw: Any) -> dict[str, Pacing]:
    raw = _require_dict(raw, "pacing")
    if "default" not in raw:
        raise PolicyError("'pacing.default' is required")
    return {
        provider_id: _parse_pacing_entry(provider_id, entry) for provider_id, entry in raw.items()
    }


def _parse_schedule(raw: Any) -> Schedule:
    raw = _require_dict(raw, "schedule")
    _reject_unknown_keys(raw, SCHEDULE_KEYS, "schedule")
    for required in SCHEDULE_KEYS:
        if required not in raw:
            raise PolicyError(f"'schedule.{required}' is required")
    enabled = _require_bool(raw["enabled"], "schedule.enabled")
    interval_minutes = _require_positive_int(raw["interval_minutes"], "schedule.interval_minutes")
    require_proxy = _require_bool(raw["require_proxy"], "schedule.require_proxy")
    maximum_staleness_hours = _require_number(
        raw["maximum_staleness_hours"], "schedule.maximum_staleness_hours"
    )
    if maximum_staleness_hours <= 0:
        raise PolicyError("'schedule.maximum_staleness_hours' must be positive")
    return Schedule(
        enabled=enabled,
        interval_minutes=interval_minutes,
        require_proxy=require_proxy,
        maximum_staleness_hours=maximum_staleness_hours,
    )


def _parse_safety(raw: Any) -> Safety:
    raw = _require_dict(raw, "safety")
    _reject_unknown_keys(raw, SAFETY_KEYS, "safety")
    for required in SAFETY_KEYS:
        if required not in raw:
            raise PolicyError(f"'safety.{required}' is required")
    maximum_removal_share = _require_number(
        raw["maximum_removal_share"], "safety.maximum_removal_share"
    )
    if not (0 < maximum_removal_share <= 1):
        raise PolicyError("'safety.maximum_removal_share' must be between 0 and 1")
    snapshot_count = _require_positive_int(raw["snapshot_count"], "safety.snapshot_count")
    return Safety(maximum_removal_share=maximum_removal_share, snapshot_count=snapshot_count)


def _parse_translation_overrides(raw: Any) -> dict[str, dict[str, Any]]:
    raw = _require_dict(raw, "translation_overrides")
    result: dict[str, dict[str, Any]] = {}
    for offering_id, params in raw.items():
        result[offering_id] = dict(
            _require_dict(params, f"translation_overrides.{offering_id}")
        )
    return result


def _parse_proxy_settings(raw: Any) -> ProxySettings:
    raw = _require_dict(raw, "proxy_settings")
    _reject_unknown_keys(raw, PROXY_SETTINGS_KEYS, "proxy_settings")
    general_settings = raw.get("general_settings", {})
    general_settings = _require_dict(general_settings, "proxy_settings.general_settings")
    litellm_settings = raw.get("litellm_settings", {})
    litellm_settings = _require_dict(litellm_settings, "proxy_settings.litellm_settings")
    return ProxySettings(
        general_settings=dict(general_settings), litellm_settings=dict(litellm_settings)
    )


def parse_policy(raw: Any) -> Policy:
    """Validate a parsed YAML value and return a `Policy`.

    Raise `PolicyError` on any invalid Policy. The message names the
    offending key.
    """
    raw = _require_dict(raw, "<policy>")
    _reject_unknown_keys(raw, TOP_LEVEL_KEYS, "<policy>")
    missing = REQUIRED_TOP_LEVEL_KEYS - set(raw)
    if missing:
        raise PolicyError(f"'{sorted(missing)[0]}' is required")

    return Policy(
        providers=_parse_providers(raw["providers"]),
        quality=_parse_quality(raw["quality"]),
        approved_candidates=_parse_approved_candidates(raw["approved_candidates"]),
        naming=_parse_naming(raw["naming"]),
        withheld=_parse_withheld(raw["withheld"]),
        declared=_parse_declared(raw["declared"]),
        pacing=_parse_pacing(raw["pacing"]),
        schedule=_parse_schedule(raw["schedule"]),
        safety=_parse_safety(raw["safety"]),
        translation_overrides=_parse_translation_overrides(
            raw.get("translation_overrides", {})
        ),
        proxy_settings=_parse_proxy_settings(raw.get("proxy_settings", {})),
        feed=_parse_feed(raw.get("feed")),
        client_facing_variants=_parse_client_facing_variants(
            raw.get("client_facing_variants")
        ),
    )


def _parse_client_facing_variants(raw: Any) -> ClientFacingVariants | None:
    """Parse the `client_facing_variants` block, or `None` when absent.

    An absent block means the Generator adds no variant at all. An empty
    block means the defaults apply: `[1m]` at 1,000,000 tokens.
    """
    if raw is None:
        return None
    raw = _require_dict(raw, "client_facing_variants")
    _reject_unknown_keys(raw, CLIENT_FACING_VARIANT_KEYS, "client_facing_variants")
    suffix = raw.get("suffix", "[1m]")
    suffix = _require_str(suffix, "client_facing_variants.suffix")
    minimum = raw.get("minimum_context_tokens", 1_000_000)
    minimum = _require_positive_int(
        minimum, "client_facing_variants.minimum_context_tokens"
    )
    stated_raw = _require_dict(
        raw.get("operator_stated", {}), "client_facing_variants.operator_stated"
    )
    stated: dict[str, str] = {}
    for offering_id, reason in stated_raw.items():
        key = f"client_facing_variants.operator_stated.{offering_id}"
        # A reason is required, not decorative. This key asserts a window
        # the Feed does not state, so the record of why must travel with it.
        stated[_require_str(offering_id, key)] = _require_str(reason, key)
    return ClientFacingVariants(
        suffix=suffix, minimum_context_tokens=minimum, operator_stated=stated
    )


def load_policy(path: Path) -> Policy:
    """Read, parse and validate the Policy at `path`.

    Raise `PolicyError` on an invalid Policy. The message names the
    offending key.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)
    if raw is None:
        raise PolicyError("the Policy file is empty")
    return parse_policy(raw)


def describe_policy(policy: Policy) -> str:
    """Render a plain-text summary of what `load_policy` understood.

    Used by the `validate` command. Run the result through
    `litellm_maintainer.redact.redact` before printing it, since a
    Declared Offering can hold a base URL.
    """
    lines: list[str] = []
    lines.append("Providers:")
    for provider_id in sorted(policy.providers):
        rule = policy.providers[provider_id]
        detail = f"mode={rule.mode} entitlement={rule.entitlement}"
        if rule.pricing:
            detail += f" pricing={list(rule.pricing)}"
        if rule.models:
            detail += f" models={len(rule.models)}"
        if rule.translation:
            detail += " translation=overridden"
        if rule.response_envelope_key:
            detail += f" response_envelope_key={rule.response_envelope_key}"
        if rule.plan_edition:
            detail += f" plan_edition={rule.plan_edition}"
        lines.append(f"  {provider_id}: {detail}")

    lines.append(f"Quality: minimum_coding_score={policy.quality.minimum_coding_score}")
    lines.append(f"Approved candidates: {len(policy.approved_candidates)}")
    for candidate_id in policy.approved_candidates:
        lines.append(f"  {candidate_id}")

    lines.append("Naming:")
    lines.append(f"  alias_prefix={policy.naming.alias_prefix}")
    lines.append(f"  provider_labels: {len(policy.naming.provider_labels)}")
    for provider_id in sorted(policy.naming.provider_labels):
        lines.append(f"    {provider_id} -> {policy.naming.provider_labels[provider_id]}")
    lines.append(f"  alias_overrides: {len(policy.naming.alias_overrides)}")

    lines.append(f"Withheld: {len(policy.withheld)}")
    for offering_id in sorted(policy.withheld):
        lines.append(f"  {offering_id}: {policy.withheld[offering_id]}")

    lines.append(f"Declared Offerings: {len(policy.declared)}")
    for declared in policy.declared:
        detail = f"model={declared.litellm_params.get('model')}"
        if declared.passthrough_auth:
            detail += " passthrough_auth=true"
        if declared.proxy_authenticated:
            detail += " proxy_authenticated=true"
        if declared.supersedes:
            detail += f" supersedes={declared.supersedes}"
        lines.append(f"  {declared.alias}: {detail}")

    lines.append("Pacing:")
    for provider_id in sorted(policy.pacing):
        pacing = policy.pacing[provider_id]
        lines.append(
            f"  {provider_id}: concurrency={pacing.concurrency} "
            f"minimum_interval_seconds={pacing.minimum_interval_seconds}"
        )

    schedule = policy.schedule
    lines.append(
        "Schedule: enabled={enabled} interval_minutes={interval} "
        "require_proxy={require_proxy} maximum_staleness_hours={staleness}".format(
            enabled=schedule.enabled,
            interval=schedule.interval_minutes,
            require_proxy=schedule.require_proxy,
            staleness=schedule.maximum_staleness_hours,
        )
    )

    safety = policy.safety
    lines.append(
        f"Safety: maximum_removal_share={safety.maximum_removal_share} "
        f"snapshot_count={safety.snapshot_count}"
    )

    lines.append(f"Translation overrides: {len(policy.translation_overrides)}")
    for offering_id in sorted(policy.translation_overrides):
        lines.append(f"  {offering_id}")

    proxy_settings = policy.proxy_settings
    lines.append(
        "Proxy settings: general_settings="
        f"{len(proxy_settings.general_settings)} key(s), litellm_settings="
        f"{len(proxy_settings.litellm_settings)} key(s)"
    )

    return "\n".join(lines)
