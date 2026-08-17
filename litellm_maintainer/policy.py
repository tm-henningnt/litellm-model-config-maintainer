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
    "headroom",
    "allowances",
    "draw_notes",
}
CLIENT_FACING_VARIANT_KEYS = {
    "suffix",
    "minimum_context_tokens",
    "operator_stated",
}
# The one key an `allowances.<allowance_id>` entry may carry. See
# `AllowanceInfo`.
ALLOWANCE_ENTRY_KEYS = {"tier", "scale_note"}

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
    "headroom",
    "allowances",
    "draw_notes",
}

PROVIDER_RULE_KEYS = {
    "mode",
    "pricing",
    "models",
    "translation",
    "entitlement",
    "response_envelope_key",
    "plan_edition",
    "cost_basis",
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

# What using an Offering costs us. These are the terms an orchestrator
# reasons in: a flat-rate call costs no marginal money but drains a
# window, a metered call bills, a passthrough call bills the caller's own
# credential.
#
# The names live here, not in `entitlements`, because a Declared Offering
# states its own cost basis in Policy: the Feed publishes no pricing kind
# for an Offering it does not publish at all. `entitlements` imports them
# from here and re-exports them, so an existing
# `from litellm_maintainer.entitlements import FREE` keeps working.
FREE = "free"
FLAT_RATE = "flat_rate"
METERED = "metered"
PASSTHROUGH = "passthrough"
UNKNOWN_BASIS = "unknown"
VALID_COST_BASES = {FREE, FLAT_RATE, METERED, PASSTHROUGH, UNKNOWN_BASIS}
NAMING_KEYS = {"provider_labels", "alias_overrides", "alias_prefix", "alias_separator"}
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
    "reference_model",
    "cost_basis",
    "pricing",
    "fair_use",
}
DECLARED_PRICING_KEYS = {"input_usd_per_1m_tokens", "output_usd_per_1m_tokens"}
PACING_ENTRY_KEYS = {"concurrency", "minimum_interval_seconds"}
SCHEDULE_KEYS = {"enabled", "interval_minutes", "require_proxy", "maximum_staleness_hours"}
SAFETY_KEYS = {"maximum_removal_share", "snapshot_count"}
HEADROOM_KEYS = {
    "command",
    "interval_minutes",
    "sources",
    "demote_at_full",
    "timeout_seconds",
    "all_accounts_providers",
}
# The three keys a `headroom.sources` entry may carry when it is written as a
# mapping instead of a plain string. `source` is required. `windows` is
# optional and, when present, names what each of codexbar's three slots
# measures. `members` is optional and, when present, names which Health Keys
# draw on each declared slot. See `Headroom`, `_parse_headroom_windows` and
# `_parse_headroom_members`.
HEADROOM_SOURCE_ENTRY_KEYS = {"source", "windows", "members", "unmeasured"}
# The only three slot names a `windows` mapping may use. They are codexbar's
# own `primary`/`secondary`/`tertiary` fields, never an operator's own name
# for one.
HEADROOM_WINDOW_SLOTS = ("primary", "secondary", "tertiary")
DEFAULT_HEADROOM_COMMAND = "codexbar"
DEFAULT_HEADROOM_INTERVAL_MINUTES = 15
# Measured 2026-07-28: codexbar took 21-31 seconds for every provider it
# knows, 24 seconds for four mapped providers. 40 leaves room for a fifth or
# sixth mapped provider before a refresh times out on its own margin.
DEFAULT_HEADROOM_TIMEOUT_SECONDS = 40.0

# The Allowance id namespaces a `headroom.sources` key must start with.
# `litellm_maintainer.entitlements` defines the same four prefixes as
# `ALLOWANCE_PROVIDER`, `ALLOWANCE_POOL`, `ALLOWANCE_CREDENTIAL` and
# `ALLOWANCE_ALIAS`. That module imports this one, so this one cannot
# import it back; the strings are duplicated here rather than shared,
# and a change to either set needs to change both.
_ALLOWANCE_ID_PREFIXES = ("provider:", "pool:", "credential:", "alias:")
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


def _require_allowance_id(allowance_id: Any, key: str) -> str:
    """Validate one Allowance id key, shared by `headroom.sources` and
    `allowances`.

    An Allowance id names one of the four namespaces CONTEXT.md's
    "Allowance" entry states: `provider:`, `pool:`, `credential:` or
    `alias:`. Both blocks key on the same string, so both refuse the same
    malformed key rather than each inventing its own rule.
    """
    allowance_id = _require_str(allowance_id, key)
    if not allowance_id.startswith(_ALLOWANCE_ID_PREFIXES):
        raise PolicyError(
            f"'{key}' is not a well-formed Allowance id; it must start with one of "
            f"{list(_ALLOWANCE_ID_PREFIXES)}"
        )
    return allowance_id


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

    `cost_basis` states what this provider costs THIS account, overriding
    what the Feed's pricing kind implies. Set it only where the Feed
    cannot see the account's plan.

    Two measured cases, both on 2026-07-28. Groq is free to this account
    and the Feed marks its Offerings `paid` or `unknown`, never `free`,
    because it cannot see the plan. Gemini is covered by a Google One AI
    Plus subscription and the Feed prices it per token. Both therefore
    read as spend, so a caller instructed to treat `metered` and
    `unknown` as money avoided capacity already paid for.

    It changes what `guidance` and `entitlements` REPORT and nothing
    else. It never filters an Offering, never reaches the Generated
    Config, and never overrides the Feed's token RATES: a rate is a
    number the Feed measured, and this is a statement about who bills.
    """

    mode: str
    pricing: tuple[str, ...] | None = None
    models: tuple[str, ...] | None = None
    translation: dict[str, Any] | None = None
    entitlement: str = PER_MODEL
    response_envelope_key: str | None = None
    plan_edition: str | None = None
    cost_basis: str | None = None


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
    every derived Alias. `alias_separator` separates the provider label,
    model id, and any model tags.
    """

    provider_labels: dict[str, str]
    alias_overrides: dict[str, str]
    alias_prefix: str
    alias_separator: str = "-"


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

    `reference_model`, `cost_basis` and `pricing` give a caller the
    numbers it needs to weigh this Offering against a Feed one. See each
    field below, and ADR 0011.

    `fair_use` says the allowance tolerates load badly. It is a separate
    field from `cost_basis` on purpose: see ADR 0012.
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

    # The Canonical Model id in the Feed that serves the SAME model as
    # this Offering. `guidance` then folds this Offering's Route onto that
    # model's Guidance Row, so the row carries the Feed's score, display
    # name and capabilities, and this Route is ranked against every other
    # model rather than sorted last for want of a number.
    #
    # Stated, never inferred from the Alias or the model string: the
    # operator names the Alias and the Feed names the Canonical Model, and
    # the two agree by accident at best. `claude-chatgpt1-gpt-5.6-sol`
    # reaches `openai/gpt-5.6-sol`, and no rule derives one from the
    # other.
    #
    # A Reference Model NEVER supplies a limit. This endpoint's window is
    # its own: the ChatGPT seats accept about 350,000 tokens where the
    # Feed's API mirror states 1,050,000. See ADR 0006 and ADR 0011.
    reference_model: str | None = None

    # What calling this Offering costs, in the five terms
    # `VALID_COST_BASES` names. The Feed publishes no pricing kind for an
    # Offering it does not publish, so without this every Declared
    # Offering read `unknown` — which an agent is told to treat as spend.
    # A fixed-rate host is `flat_rate`, and a caller that reads that stops
    # avoiding capacity already paid for.
    #
    # `None` keeps the earlier rule: `passthrough` when
    # `passthrough_auth` is set, `unknown` otherwise.
    cost_basis: str | None = None

    # The token rates to report for this Offering, USD per 1,000,000
    # tokens, as `{"input_usd_per_1m_tokens": …,
    # "output_usd_per_1m_tokens": …}`. It overrides a Reference Model's
    # rates, which belong to another vendor serving the same model.
    #
    # On a `free` or `flat_rate` Offering these state the RELATIVE burn on
    # a pool, never an amount billed. `guidance` marks that with
    # `rate_is_list_price`, the same distinction `pricing.py` writes into
    # the Generated Config.
    pricing: dict[str, float] | None = None

    # Whether this Offering's allowance tolerates load badly: a plan that is
    # unmetered under a "fair use" clause, with no number stating the line.
    #
    # Deliberately NOT a Cost Basis. A Cost Basis answers who bills, and a
    # fair-use host bills flat rate — that is simply true. Load tolerance is
    # a second question, so it gets a second field. One term, one meaning.
    #
    # It exists because `flat_rate` is honest about the billing and wrong
    # about the risk: a downstream Role accepts `flat_rate` by default, so a
    # fair-use host sits in every failover path, and a bulk batch whose free
    # Routes drain walks into it unthrottled. A caller reads this field to
    # require the host be named before it is used. See ADR 0012.
    #
    # `False` by default, and never `None`: a Policy that says nothing
    # claims the allowance takes load normally, which is the safe reading.
    fair_use: bool = False

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
class Headroom:
    """Which Allowance each codexbar Reading belongs to, and how to ask.

    `headroom refresh` reads codexbar and writes one Reading per key in
    `sources`. Nothing infers the join: an Allowance does not always
    carry a Policy block of its own (`pool:claude-subscription` names a
    pool across several Declared Offerings; `credential:EXAMPLE_PRIVATE_HOST_API_KEY`
    names none at all), so `sources` is its own top-level map, keyed on
    the WHOLE Allowance id string, exactly as `entitlements` reports it.

    Measured 2026-07-28: matching by name instead would send the
    ClinePass Reading to `provider:cline-pass`, which holds 0 Offerings
    in scope, while the real Allowance that serves ClinePass models,
    `provider:cline`, holds 8 answering. The inferred join reports a
    fully drawn Allowance as unmeasured and an empty one as drained.
    ADR 0012 already forbids inferring an Allowance; this is that rule
    applied to a second key. See CONTEXT.md, "Headroom", and the
    headroom spec, decision 3.

    Each value reads `codexbar:<providerID>/<accountEmail>`, matched
    against the WHOLE identity string codexbar publishes for one entry
    (`CodexbarReading.source_key`). Never parse inside it. A provider
    that publishes no account email states an empty one — the trailing
    slash stays, so `"codexbar:opencodego/"` never collides with an
    account that starts publishing an email later.

    A `sources` entry may instead be a mapping, `{"source": ..., "windows":
    {...}}`, for a provider whose three slots hold one quota per MODEL
    rather than nested time windows. `source` reads exactly as the plain
    string form does. `windows` names what `primary`, `secondary` and
    `tertiary` measure, mapping each named slot to an operator-chosen
    Sub-allowance id — never codexbar's own id, since codexbar labels these
    three slots only in its text output, never in the JSON.

    A slot NOT named in `windows` stays a parent window and keeps binding
    every Route on the Allowance, exactly as it always has. A slot that IS
    named leaves the parent computation, and binds only the Route whose
    Health Key `members` lists under it. Where every slot is named, the
    Allowance publishes no Headroom of its own — `headroom.binding_window`
    then returns `None`, correctly: nothing caps the Allowance as a whole.
    See `docs/gotchas.md`, "codexbar's three window slots do not mean one
    thing", and ADR 0012.

    The same mapping may also carry `members`: which Health Keys draw on
    each declared slot, keyed by the slot id `windows` names (ticket 10).
    A member is a Health Key — a Feed Offering's own id, or a Declared
    Offering's Alias — matched EXACTLY, never as a glob, a prefix or a
    regular expression. Measured 2026-07-29: the pattern
    `gemini-3*-flash*`, written for Flash, also matches
    `gemini-3.1-flash-lite`, so a Route for Flash Lite would silently
    borrow Flash's own figure.

    Ticket 09 could only attach a slot id to a Declared Offering, through
    the field it named `sub_allowance_window`. That field is retired: it
    resolved no Discovered Offering at all, so a per-model provider the
    Feed itself publishes — Gemini running `mode: all`, the measured case
    — got the safe half of ticket 09 (its Allowance stopped reporting
    drained) and not the useful half (no Route reported a figure). Naming
    a member here reaches BOTH kinds of Offering, because the Health Key is
    already the one name this system records either kind under
    (CONTEXT.md, "Health Key").

    A slot `members` never mentions, or a `members` map absent entirely,
    still leaves that slot named in `windows` and out of the parent
    computation — a Sub-allowance with nobody assigned yet, not a parent
    window again. `doctor` reports the gap either way (ticket 10).

    Gemini is the measured case: its free plan reports its `Pro` slot
    100% spent while `Flash` and `Flash Lite` report 0, because the plan
    excludes Pro rather than because the account is out of headroom. See
    `policy.example.yaml`.

    Codex names two ChatGPT seats through `all_accounts_providers`
    instead of staying unmapped (ticket 11). `codexbar --provider codex`
    alone returns one Reading, and Policy holds two seats with their own
    credentials, so a single call could never name which seat it
    measured (ADR 0009). Measured 2026-07-29: `codexbar --provider codex
    --all-accounts --format json` returns BOTH accounts, and they carry
    the same `providerID` and differ only by `accountEmail` — so the
    ordinary join key here, `codexbar:<providerID>/<accountEmail>`,
    already discriminates them once both reach the same document.

    `command` names the binary `headroom refresh` runs, defaulting to
    `codexbar`. It exists so a test can point it at a fixture script;
    codexbar itself is a Homebrew binary, and this project adds no
    packaging extra for it. `interval_minutes` is read by the scheduled
    refresh job, not by this command, and defaults to 15: an hourly
    interval is too coarse for Claude's 300-minute window, and under 5
    minutes re-copies a figure that has not moved.

    An empty `sources` map turns the whole capability off: `headroom
    refresh` then does nothing and says so.

    `demote_at_full` turns a Reading of 100% into a demotion:
    `guidance` stops calling that Route `recommendable`. It defaults to
    `False`, and stays `False` on this operator's Policy until the
    Readings prove themselves over real weeks (headroom spec, decision
    15; ticket 08).

    Four things stand behind this flag, and none had run in anger when
    it was written: codexbar's own semantics at the 100% boundary, the
    hand-written `sources` mapping above, the expiry rule in
    `headroom.window_is_void`, and the Sub-allowance join in
    `route_binding_window`. Any one of them wrong turns the flag against
    the operator's own main agent: `pool:claude-subscription` is the
    first Route it can demote, because it is the first Allowance mapped
    at all. Flip it only after watching real Readings reach 100%, void
    themselves at reset with no help, and a Sub-allowance behave as
    decision 12 predicts.

    `timeout_seconds` bounds one `headroom.command` invocation, defaulting
    to `DEFAULT_HEADROOM_TIMEOUT_SECONDS` (40). Measured 2026-07-28: 24
    seconds for four mapped providers, 21-31 seconds for every provider
    codexbar knows. A fifth or sixth mapped provider can plausibly cross
    40 seconds, after which every refresh times out and the capability
    goes stale for good, so the operator states this in Policy instead
    of a code constant nobody reading Policy would find.
    """

    command: str = DEFAULT_HEADROOM_COMMAND
    interval_minutes: int = DEFAULT_HEADROOM_INTERVAL_MINUTES
    sources: dict[str, str] = field(default_factory=dict)
    demote_at_full: bool = False
    timeout_seconds: float = DEFAULT_HEADROOM_TIMEOUT_SECONDS
    # The codexbar provider ids that hold more than one account. `headroom
    # refresh` gives each one its own `--all-accounts` call, instead of
    # riding the batched call every other mapped provider shares (ticket
    # 11). codexbar's own `--help` states "Account selection requires a
    # single provider", so a multi-account provider cannot join that
    # batched call at all.
    #
    # Name the provider id ONCE here, never on each `sources` entry: two
    # entries billed to the same provider would otherwise carry the flag
    # twice, with nothing to say which copy is authoritative.
    #
    # STATED, NEVER DETECTED. Two `sources` entries sharing a `providerID`
    # looks like it implies this flag, but the failing case is one account
    # of two mapped: a plain call then returns whichever account codexbar
    # treats as default, which may not be the mapped one, and the key
    # matches no Reading for a reason invisible from outside. `doctor`
    # reports that shape (`headroom.all_accounts.unmarked.<provider_id>`)
    # instead of this module inferring it.
    #
    # Empty by default, so a Policy written before this field existed
    # behaves exactly as before: every provider rides the one batched call.
    all_accounts_providers: tuple[str, ...] = ()
    # Which of an Allowance's three named slots are Sub-allowances rather
    # than parent windows, keyed by Allowance id, each value a mapping of
    # slot name (`primary`, `secondary` or `tertiary`) to the operator's
    # own Sub-allowance id for it. Empty for every Allowance whose
    # `sources` entry stayed a plain string, and empty for a mapping-form
    # entry that named no `windows` at all — both mean "every slot is a
    # parent window", the byte-identical behaviour this project has always
    # had.
    source_windows: dict[str, dict[str, str]] = field(default_factory=dict)
    # Which Health Keys draw on each declared Sub-allowance slot, keyed by
    # Allowance id, each value a mapping of slot id (one of the operator's
    # own values in `source_windows`, never a slot NAME) to the Health Keys
    # that draw on it — a Feed Offering's own id, or a Declared Offering's
    # Alias (ticket 10). Empty for every Allowance whose `sources` entry
    # named no `members` at all, which reads as "nobody is assigned yet",
    # never as "no Sub-allowance exists": `source_windows` still names the
    # slot, and `doctor` reports the gap.
    source_members: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)
    # The Health Keys the operator states draw on NO window this source
    # publishes, keyed by Allowance id. Read only by `doctor`, to keep
    # `headroom.member.unclaimed` from firing on a key that is correctly
    # unassigned — Gemini serves Gemma beside Pro, Flash and Flash Lite,
    # and none of the three slots measures it. Every Route on such a key
    # publishes `headroom: null` with or without this list; the list is
    # how the operator says the silence is deliberate.
    source_unmeasured: dict[str, tuple[str, ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class AllowanceInfo:
    """One fact the operator states about an Allowance itself, not about
    any Reading of it.

    `tier` is the subscription level this Allowance bills under, as the
    operator states it — CONTEXT.md, "Tier". A Headroom states a SHARE of
    that Tier's own ceiling, so two Allowances both reading 50% are not
    the same amount of work unless their Tiers match too. Nothing here
    verifies the string: it is a label, published verbatim, never parsed,
    ranked or derived from. `None` when the operator states none.

    Kept its own dataclass, separate from `Headroom`, because a Tier
    belongs to the Allowance itself: it holds whether or not `codexbar`
    can read that Allowance at all, and more Allowance-level facts may
    join it later.

    `scale_note` is free prose stating how big the Allowance is, where the
    vendor states a size but no Tier. Some vendors sell ONE fixed price
    with ONE quota and no levels at all, so `tier` has nothing to hold,
    and a caller then reads a share against no scale whatever.

    Measured 2026-07-30, on two of one operator's six mapped Allowances:
    ClinePass sells 10 USD per month and calls the quota "generous",
    stating only that it is roughly 2x to 5x its own API cost. OpenCode Go
    sells 10 USD per month and states roughly 12 USD per 5 hours, 30 per
    week and 60 per month of API-equivalent spend. Both are real scale.
    Neither is a subscription level.

    Prose, and deliberately not a number. The vendors state a RANGE, or a
    rough equivalence, and inventing a single figure they declined to give
    would repeat the mistake `metered` already made here: a published list
    price read as this account's own bill. Published verbatim, ranked by
    nothing, parsed by nothing — the same contract `tier` carries.
    """

    tier: str | None = None
    scale_note: str | None = None


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
    # Which Allowance each codexbar Reading belongs to. Empty when the
    # Policy names no `headroom` block, and `headroom refresh` then does
    # nothing, so a Policy written before this feature behaves exactly
    # as before.
    headroom: Headroom = field(default_factory=Headroom)
    # What the operator states about each Allowance itself — currently
    # only `tier` (CONTEXT.md, "Tier"). Keyed on the WHOLE Allowance id,
    # the same key `headroom.sources` and `entitlements` use. Empty when
    # the Policy names no `allowances` block, and every Allowance then
    # publishes `tier: null`, so a Policy written before this field
    # behaves exactly as before.
    # How fast one Offering draws on its Allowance, as the operator states
    # it, keyed by Health Key. Free prose, published verbatim on every
    # Route for that Offering, ranked by nothing and parsed by nothing.
    #
    # `allowances.<id>.scale_note` answers "how big is this Allowance".
    # This answers "how fast does THIS model empty it", and the two are
    # different questions: one belongs to the credential, one to the
    # model. A pool can hold six Offerings that draw at six rates.
    #
    # Measured 2026-07-30 on one pool. `qwen3.8-max-preview` bills at 10%
    # of the normal Credits rate during its preview, and a stacked
    # discount takes a further 80% off inside one daily window, so the two
    # compound to 2%. Fifty times the work per Credit, on one Offering of
    # six, invisible in every rate the Feed publishes -- it publishes none
    # for a subscription Offering at all.
    draw_notes: dict[str, str] = field(default_factory=dict)
    allowances: dict[str, AllowanceInfo] = field(default_factory=dict)


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

    cost_basis = raw.get("cost_basis")
    if cost_basis is not None:
        cost_basis = _require_str(cost_basis, f"{prefix}.cost_basis")
        if cost_basis not in VALID_COST_BASES:
            raise PolicyError(
                f"'{prefix}.cost_basis' must be one of "
                f"{sorted(VALID_COST_BASES)}, got {cost_basis!r}"
            )

    return ProviderRule(
        mode=mode,
        pricing=pricing,
        models=models,
        translation=translation,
        entitlement=entitlement,
        response_envelope_key=envelope_key,
        plan_edition=plan_edition,
        cost_basis=cost_basis,
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
    alias_prefix = raw["alias_prefix"]
    if not isinstance(alias_prefix, str):
        raise PolicyError("'naming.alias_prefix' must be a string")
    alias_separator = _require_str(raw.get("alias_separator", "-"), "naming.alias_separator")
    if not alias_separator:
        raise PolicyError("'naming.alias_separator' must not be empty")
    return Naming(
        provider_labels=dict(provider_labels),
        alias_overrides=dict(alias_overrides),
        alias_prefix=alias_prefix,
        alias_separator=alias_separator,
    )


def _parse_withheld(raw: Any) -> dict[str, str]:
    raw = _require_dict(raw, "withheld")
    for k, v in raw.items():
        _require_str(v, f"withheld.{k}")
    return dict(raw)


def _parse_draw_notes(raw: Any) -> dict[str, str]:
    """Parse the `draw_notes` block: how fast each Offering draws.

    Keyed by Health Key, the one name this system already records either
    kind of Offering under — a Feed Offering's own id, or a Declared
    Offering's Alias. The same key `withheld` and
    `headroom.sources.<id>.members` use, matched exactly.

    Prose, never a number. A vendor states these as multiples, windows and
    promotions ("10% of the normal rate during preview, and 20% of that
    between 22:00 and 08:00 Beijing time"), and collapsing that to one
    figure would state something the vendor did not. `None` — the block is
    absent — reads as "the operator states none", the ordinary case.
    """
    if raw is None:
        return {}
    raw = _require_dict(raw, "draw_notes")
    for health_key, note in raw.items():
        _require_str(note, f"draw_notes.{health_key}")
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
    reference_model = raw.get("reference_model")
    if reference_model is not None:
        reference_model = _require_str(reference_model, f"{prefix}.reference_model")
    cost_basis = raw.get("cost_basis")
    if cost_basis is not None:
        cost_basis = _require_str(cost_basis, f"{prefix}.cost_basis")
        if cost_basis not in VALID_COST_BASES:
            raise PolicyError(
                f"'{prefix}.cost_basis' must be one of "
                f"{sorted(VALID_COST_BASES)}, got {cost_basis!r}"
            )
    pricing = _parse_declared_pricing(raw.get("pricing"), prefix)
    fair_use = _require_bool(raw.get("fair_use", False), f"{prefix}.fair_use")
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
        reference_model=reference_model,
        cost_basis=cost_basis,
        pricing=pricing,
        fair_use=fair_use,
    )


def _parse_declared_pricing(raw: Any, prefix: str) -> dict[str, float] | None:
    """The token rates a Declared Offering states, or `None`.

    Both rates are required together. One rate alone reads as "input is
    free" to a caller comparing two models, which is a wrong answer rather
    than a missing one, so it is refused here.
    """
    if raw is None:
        return None
    raw = _require_dict(raw, f"{prefix}.pricing")
    _reject_unknown_keys(raw, DECLARED_PRICING_KEYS, f"{prefix}.pricing")
    missing = sorted(DECLARED_PRICING_KEYS - set(raw))
    if missing:
        raise PolicyError(
            f"'{prefix}.pricing' states one rate and not the other; "
            f"{missing} is missing. A caller comparing two models reads a "
            "missing rate as zero, so state both or neither."
        )
    rates: dict[str, float] = {}
    for key in sorted(DECLARED_PRICING_KEYS):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PolicyError(
                f"'{prefix}.pricing.{key}' must be a number, got {value!r}"
            )
        if value < 0:
            raise PolicyError(
                f"'{prefix}.pricing.{key}' must not be negative, got {value!r}"
            )
        rates[key] = float(value)
    return rates


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
        draw_notes=_parse_draw_notes(raw.get("draw_notes")),
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
        headroom=_parse_headroom(raw.get("headroom")),
        allowances=_parse_allowances(raw.get("allowances")),
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


def _parse_headroom(raw: Any) -> Headroom:
    """Parse the `headroom` block, or return the all-empty default.

    An absent block means Policy declares no source at all, which turns
    the whole capability off: `headroom refresh` then does nothing and
    reports so.
    """
    if raw is None:
        return Headroom()
    raw = _require_dict(raw, "headroom")
    _reject_unknown_keys(raw, HEADROOM_KEYS, "headroom")
    command = raw.get("command", DEFAULT_HEADROOM_COMMAND)
    command = _require_str(command, "headroom.command")
    interval_minutes = raw.get("interval_minutes", DEFAULT_HEADROOM_INTERVAL_MINUTES)
    interval_minutes = _require_positive_int(interval_minutes, "headroom.interval_minutes")

    sources_raw = _require_dict(raw.get("sources", {}), "headroom.sources")
    sources: dict[str, str] = {}
    source_windows: dict[str, dict[str, str]] = {}
    source_members: dict[str, dict[str, tuple[str, ...]]] = {}
    source_unmeasured: dict[str, tuple[str, ...]] = {}
    for allowance_id, entry in sources_raw.items():
        key = f"headroom.sources.{allowance_id}"
        allowance_id = _require_allowance_id(allowance_id, key)
        source, windows, members, unmeasured = _parse_headroom_source_entry(entry, key)
        rest = source.removeprefix("codexbar:")
        if rest == source or "/" not in rest:
            raise PolicyError(
                f"'{key}' must read 'codexbar:<providerID>/<accountEmail>', got {source!r}"
            )
        sources[allowance_id] = source
        if windows:
            source_windows[allowance_id] = windows
        if members:
            source_members[allowance_id] = members
        if unmeasured:
            source_unmeasured[allowance_id] = unmeasured

    demote_at_full = _require_bool(
        raw.get("demote_at_full", False), "headroom.demote_at_full"
    )

    timeout_seconds = _require_number(
        raw.get("timeout_seconds", DEFAULT_HEADROOM_TIMEOUT_SECONDS), "headroom.timeout_seconds"
    )
    if timeout_seconds <= 0:
        raise PolicyError("'headroom.timeout_seconds' must be a positive number")

    all_accounts_raw = _require_list(
        raw.get("all_accounts_providers", []), "headroom.all_accounts_providers"
    )
    all_accounts_providers = tuple(
        _require_str(item, f"headroom.all_accounts_providers[{i}]")
        for i, item in enumerate(all_accounts_raw)
    )

    return Headroom(
        command=command,
        interval_minutes=interval_minutes,
        sources=sources,
        demote_at_full=demote_at_full,
        timeout_seconds=timeout_seconds,
        source_windows=source_windows,
        source_members=source_members,
        source_unmeasured=source_unmeasured,
        all_accounts_providers=all_accounts_providers,
    )


def _parse_headroom_source_entry(
    raw: Any, key: str
) -> tuple[str, dict[str, str], dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Parse one `headroom.sources` value: a plain string, or a mapping.

    A plain string means "every slot is a parent window", unchanged from
    before this function existed. A mapping names `source` (required, read
    exactly like the plain-string form), `windows` (optional): what each of
    codexbar's three slots measures, for a provider whose slots hold one
    quota per model rather than nested time windows, and `members`
    (optional): which Health Keys draw on each declared slot (ticket 10).

    `unmeasured` (optional) names the Health Keys that draw on NO window
    this source publishes. Every slot a per-model provider fills is one
    quota, and a provider can serve a model outside all of them: Gemini's
    three slots hold Pro, Flash and Flash Lite, and the same account also
    serves Gemma, which none of the three measures.

    Without this key the operator has two choices, and both are worse than
    silence. Leaving such a Health Key out fails
    `doctor`'s `headroom.member.unclaimed`, which is right to fire — an
    unassigned key is normally a gap. Listing it under a slot states
    something FALSE about the Reading, the same defect the `windows`
    comments already warn about. So state it here instead. A key named
    here publishes `headroom: null` — unknown, exactly as it should.

    Returns `(source, windows, members, unmeasured)`. The last three are
    empty for a plain string, and for a mapping that names none of them.
    """
    if isinstance(raw, str):
        return raw, {}, {}, ()
    if not isinstance(raw, dict):
        raise PolicyError(f"'{key}' must be a string or a mapping")
    _reject_unknown_keys(raw, HEADROOM_SOURCE_ENTRY_KEYS, key)
    if "source" not in raw:
        raise PolicyError(f"'{key}.source' is required")
    source = _require_str(raw["source"], f"{key}.source")
    windows = _parse_headroom_windows(raw.get("windows"), f"{key}.windows")
    members = _parse_headroom_members(
        raw.get("members"),
        f"{key}.members",
        declared_slot_ids=set(windows.values()),
        windows_key=f"{key}.windows",
    )
    unmeasured_raw = _require_list(raw.get("unmeasured", []), f"{key}.unmeasured")
    unmeasured = tuple(
        _require_str(item, f"{key}.unmeasured[{i}]") for i, item in enumerate(unmeasured_raw)
    )
    claimed = {health_key for keys in members.values() for health_key in keys}
    for health_key in unmeasured:
        if health_key in claimed:
            raise PolicyError(
                f"'{key}.unmeasured' names {health_key!r}, which '{key}.members' "
                "also claims. A Health Key draws on one window or on none."
            )
    return source, windows, members, unmeasured


def _parse_headroom_windows(raw: Any, key: str) -> dict[str, str]:
    """Parse a `windows` mapping: which of the three slots are Sub-allowances.

    `None` (the key is absent) reads as "names no slot", the same as an
    empty mapping. Only `primary`, `secondary` and `tertiary` are valid
    keys — codexbar's own three slot names, never an operator's label for
    one. A slot may be named without naming the others.
    """
    if raw is None:
        return {}
    raw = _require_dict(raw, key)
    _reject_unknown_keys(raw, set(HEADROOM_WINDOW_SLOTS), key)
    return {slot: _require_str(sub_id, f"{key}.{slot}") for slot, sub_id in raw.items()}


def _parse_headroom_members(
    raw: Any, key: str, *, declared_slot_ids: set[str], windows_key: str
) -> dict[str, tuple[str, ...]]:
    """Parse a `members` mapping: which Health Keys draw on each slot.

    `None` (the key is absent) reads as "names no member", the same as an
    empty mapping — a Sub-allowance nobody has assigned yet, reported by
    `doctor`, never a parse failure (ticket 10).

    Each key names the window its members draw on. It is EITHER a slot id
    `windows_key` declares, or a codexbar `extraRateWindows` id.
    `route_binding_window` resolves an extra window first and a declared
    slot second, so both forms already reach a figure.

    Both stay legal here, and `doctor` reports a key that reaches neither.
    An earlier rule rejected a key `windows_key` did not declare, on the
    reasoning that only a typo could produce one. That reasoning was wrong
    twice.

    It broke the case the Sub-allowance exists for. Claude's
    `claude-weekly-scoped-fable` is an extra window, not one of the three
    named slots, so the honest declaration was refused and the operator had
    to name it under an unused slot instead — stating that Claude's
    `tertiary` IS the fable window. That is false, and a named slot leaves
    the parent's own figure, so a real monthly window arriving later would
    stop binding with no symptom.

    And a Feed change CAN produce an unreachable key. Measured 2026-07-28:
    codexbar published `claude-weekly-scoped-all-model` at 18:48Z and had
    dropped it by 20:52Z, moving the figure into `secondary`. A Policy that
    named it would then fail to parse, so one vendor release would stop the
    Generator for every provider.

    Each value is a list of Health Keys, read verbatim as exact strings.
    Never a glob, a prefix or a regular expression: `gemini-3*-flash*`,
    written for Flash, also matches `gemini-3.1-flash-lite` (measured
    2026-07-29), so a pattern would silently attach one model's figure to
    another's Route.
    """
    if raw is None:
        return {}
    raw = _require_dict(raw, key)
    members: dict[str, tuple[str, ...]] = {}
    for slot_id, health_keys_raw in raw.items():
        slot_id = _require_str(slot_id, key)
        health_keys_list = _require_list(health_keys_raw, f"{key}.{slot_id}")
        members[slot_id] = tuple(
            _require_str(item, f"{key}.{slot_id}[{i}]")
            for i, item in enumerate(health_keys_list)
        )
    return members


def _parse_allowances(raw: Any) -> dict[str, AllowanceInfo]:
    """Parse the `allowances` block, or return an empty mapping.

    An absent block means the operator states no Tier at all: every
    Allowance then publishes `tier: null`. Each key is validated the same
    way `headroom.sources` validates one, through `_require_allowance_id`,
    because both blocks key on the same Allowance id.
    """
    if raw is None:
        return {}
    raw = _require_dict(raw, "allowances")
    allowances: dict[str, AllowanceInfo] = {}
    for allowance_id, entry in raw.items():
        key = f"allowances.{allowance_id}"
        allowance_id = _require_allowance_id(allowance_id, key)
        entry = _require_dict(entry, key)
        _reject_unknown_keys(entry, ALLOWANCE_ENTRY_KEYS, key)
        tier = entry.get("tier")
        if tier is not None:
            tier = _require_str(tier, f"{key}.tier")
        scale_note = entry.get("scale_note")
        if scale_note is not None:
            scale_note = _require_str(scale_note, f"{key}.scale_note")
        allowances[allowance_id] = AllowanceInfo(tier=tier, scale_note=scale_note)
    return allowances


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
    lines.append(f"  alias_separator={policy.naming.alias_separator}")
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
        if declared.reference_model:
            detail += f" reference_model={declared.reference_model}"
        if declared.cost_basis:
            detail += f" cost_basis={declared.cost_basis}"
        if declared.pricing:
            detail += (
                f" pricing={declared.pricing['input_usd_per_1m_tokens']:g}"
                f"/{declared.pricing['output_usd_per_1m_tokens']:g} per 1M"
            )
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

    lines.append(f"Allowances: {len(policy.allowances)}")
    for allowance_id in sorted(policy.allowances):
        tier = policy.allowances[allowance_id].tier
        lines.append(f"  {allowance_id}: tier={tier if tier is not None else 'unstated'}")

    return "\n".join(lines)
