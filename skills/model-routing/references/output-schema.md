# Output schema and flags

Reference for parsing `litellm-maintainer` output programmatically. Read
`SKILL.md` first; this page only adds detail.

## Contents

- [Command surface](#command-surface)
- [guidance JSON](#guidance-json)
- [entitlements JSON](#entitlements-json)
- [doctor](#doctor)
- [Exit codes](#exit-codes)
- [Vocabulary](#vocabulary)

## Command surface

Shared flags on `guidance`, `entitlements` and `doctor`:

| Flag | Default | Notes |
| --- | --- | --- |
| `--policy` | `<home>/policy.yaml` | the operator's declaration |
| `--feed` | `<home>/feed.json` | the model catalogue |
| `--home` | `$LITELLM_MAINTAINER_HOME`, else `~/.config/litellm-maintainer` | instance directory |
| `--format` | `text` | `text`, `json` or `markdown` |
| `--json` | — | shorthand for `--format json` |
| `--env` | `.env.local` when present | file whose values are redacted from output |

`guidance` adds:

| Flag | Default | Notes |
| --- | --- | --- |
| `--for` | `coding` | `coding`, `reasoning`, `agentic`, `speed` |
| `--prefer` | none | `free` or `flat_rate`; sorts into a cost tier before score |
| `--limit` | none | caps rows; the cap is announced in `warnings` |

`doctor` adds `--proxy-base` (default `http://localhost:4000`).

`--format markdown` exists so a scheduled task can write a page into a project:
`litellm-maintainer guidance --for coding --format markdown > docs/models.md`.
The redirect is the caller's; these commands never write a file themselves.

## guidance JSON

```json
{
  "schema_version": "2",
  "axis": "coding",
  "prefer": null,
  "derived_at": "2026-07-26T08:14:25.799242+00:00",
  "feed_generated_at": "2026-07-25T04:24:25.512Z",
  "warnings": [],
  "client_advisory": {
    "note": "An Alias listed here is callable by exact id even when ...",
    "added_last_run": ["claude-zen-laguna-s-2-1-free"],
    "removed_last_run": [
      {"alias": "claude-pass-glm-5-2", "offering_id": "cline-pass:cline-pass/glm-5.2",
       "reason": "quota_exhausted", "refills_at": "2026-07-27T00:00:00+00:00"}
    ]
  },
  "rows": [...]
}
```

Pin `schema_version` if you parse this. It rises when a field changes meaning or
leaves; a new field does not raise it.

### A row

| Field | Notes |
| --- | --- |
| `canonical_model_id` | the model, e.g. `z-ai/glm-5.2`. For a declared offering, its alias. A declared offering that states it is a variant of another gets no row of its own: it appears as that row's route `wide_alias`. |
| `display_name` | human name from the feed |
| `score` | score on the requested axis; `null` for a declared offering |
| `scores` | all four axes, each possibly `null` |
| `capabilities` | capability strings, e.g. `chat`, `coding`, `tool_use`, `reasoning`. From the feed for a discovered row; from the operator's Policy for a declared row, whose `why` says so. Empty on a declared row means "not stated", never "not supported" |
| `callable_now` | true when at least one route is **recommendable** (available AND not exhausted) |
| `why` | one sentence built from this row's own inputs |
| `routes` | every route, sorted recommendable-first then cheapest |

Rows sort by `score` descending. A row with no score on the requested axis sorts
last but is always present — never dropped.

### A route

| Field | Notes |
| --- | --- |
| `alias` | **what you send to the proxy as the model id** |
| `offering_id` | `<provider>:<provider_model_id>`, the feed's own id |
| `provider_id` | feed provider, or `declared` for an operator-declared entry |
| `cost_basis` | `free`, `flat_rate`, `metered`, `passthrough`, `unknown` |
| `entitlement` | `shared_pool`, `per_model`, or `declared` |
| `available` | the proxy still serves this alias. `false` means "model not found" if you call it |
| `exhausted` | a recorded quota exhaustion has not cleared. The alias IS served and IS callable, but will almost certainly refuse until `refills_at` |
| `recommendable` | `available and not exhausted`. **Filter on this, not on `available`** |
| `rate_is_list_price` | true for `free` and `flat_rate`: the rate is a list price, not a bill |
| `input_usd_per_1m_tokens` | may be `null` |
| `output_usd_per_1m_tokens` | may be `null` |
| `context_tokens` | context window, may be `null`. A Stated Limit: the Feed states it for a discovered route, the operator for a declared one. `null` means no source stated it — read it as unknown, never as small |
| `max_output_tokens` | may be `null`, same source rule as `context_tokens` |
| `wide_alias` | **the alias to send when you want the full `context_tokens`**; `null` when there is none, which means `alias` is all there is. Some clients derive their context budget from the model name, so dispatching to `alias` can budget far less than `context_tokens` states. Never build this yourself by appending a suffix: it is an operator setting, and a guess breaks when they change it |
| `reason` | why it is unavailable or exhausted; `null` when neither |
| `refills_at` | ISO time the refusal said it clears; `null` when none stated |

Route order is recommendable-first, then cost basis ascending
(`free`, `flat_rate`, `passthrough`, `metered`, `unknown`), then most recently
answered. Usability outranks cost on purpose: a failover list whose first entry
cannot be called is not a failover list.

#### Why `available` and `recommendable` differ

Some routes bill to the **calling client's own credential** rather than the
proxy's — every direct Claude alias does. One caller exhausting their own
allowance must not remove the model for every other caller, so such a route is
never withdrawn from the config on a quota exhaustion. It stays `available` and
becomes `exhausted`.

Keeping it served is deliberate: a caller whose allowance is intact still
reaches it, and one whose allowance is spent gets the provider's own "your quota
resets at 09:00" instead of a bare "model not found".

An exhaustion that states no reset time expires after the schedule's maximum
staleness (24 hours by default), because nothing else could ever clear it.

### Reason values

From the classifier, on both routes and entitlement entries:

| `reason` | Waiting helps? |
| --- | --- |
| `quota_exhausted` | yes, if `refills_at` is set |
| `rate_limited` | yes, shortly |
| `plan_entitlement_refused` | no — the plan does not include this model |
| `authentication_failed` | no — a human must fix a credential |
| `identifier_gone` | no — the model id no longer exists |
| `gateway_error`, `timeout` | yes, transient |
| `malformed_response` | no — needs an operator |

## entitlements JSON

```json
{
  "schema_version": "2",
  "derived_at": "...",
  "feed_generated_at": "...",
  "warnings": [],
  "entitlements": [
    {"provider_id": "qwencloud-token-plan",
     "entitlement": "shared_pool",
     "state": "dry",
     "cost_basis": "flat_rate",
     "cost_bases": ["flat_rate"],
     "answering": 0, "unavailable": 6, "in_scope": 6,
     "withheld": 9, "candidates": 0,
     "earliest_refill_at": "2026-07-29T21:45:00+00:00",
     "unavailable_offerings": [
       {"offering_id": "qwencloud-token-plan:glm-5.2",
        "alias": "claude-qwen-token-plan-glm-5.2",
        "reason": "quota_exhausted", "bucket": "self_healing",
        "refills_at": "2026-07-29T21:45:00+00:00"}]}
  ],
  "declared": {"answering": 22, "in_scope": 22, "unavailable": []}
}
```

| Field | Notes |
| --- | --- |
| `state` | `healthy` (all answering), `degraded` (some), `dry` (none), `empty` (policy admits none) |
| `cost_basis` | the single basis, or `null` when the provider mixes several |
| `cost_bases` | every basis present |
| `answering` / `unavailable` / `in_scope` | counts of admitted offerings |
| `withheld` | offerings the operator deliberately holds back; never probed |
| `candidates` | offerings awaiting the operator's approval; not offered |
| `earliest_refill_at` | soonest refill among the unavailable; `null` when none stated |
| `declared` | operator-declared offerings, which have no provider |

`state` is arithmetic over measured offerings. It is never a prediction: a
`shared_pool` provider with three models still answering reports `degraded`,
not `dry`, even though one pool feeds them all.

### What an allowance actually is

An allowance is identified by the **credential** it bills to, not by the
provider. Two subscriptions with the same vendor are two allowances, and one
running dry says nothing about the other. Two ChatGPT seats behind one `openai/`
prefix are the worked example: six aliases each, two separate pools.

`shared_pool` also drives measurement. When one member reports a quota
exhaustion, the maintainer marks its pool mates for probing on the next run, so
they are measured rather than assumed. It never marks a sibling unavailable on
the strength of another's failure — a pool has been observed running dry while
three offerings from the same provider kept answering.

An offering can be capped *inside* its pool: `claude-fable-5` may take at most
half the Claude weekly quota. Fable running out says nothing about the pool,
while the pool running out still takes fable with it.

## doctor

Text by default, `--json` for structured. Each check has `name`, `ok`, `detail`
and, when failed, a `remedy` naming the exact command.

Check names: `policy.parses`, `credential.<provider>`, `feed_document.age`,
`proxy.reachable`, `health_state.populated`, `health_state.probed.<provider>`,
`withheld.<offering_id>`, `litellm_patch.<patch>`,
`journal.callback_registered[<config path>]`, `schedule.tick_installed`.

## Exit codes

| Code | Meaning |
| --- | --- |
| 0 | success; for `doctor`, every check passed |
| 1 | a refusal the tool explains on stderr: unreadable feed, invalid or missing policy, unknown axis, unknown `--prefer`, or any failed `doctor` check |
| 2 | argparse rejected the command line itself |

A missing or invalid policy prints the path it looked at and names `init`.

## Vocabulary

The project uses these words precisely; reusing them keeps you legible to the
operator and to the tool's own output.

| Term | Means |
| --- | --- |
| Offering | one provider-specific way to call a model |
| Alias | the name a client asks the proxy for; what you put in the model field |
| Canonical Model | the underlying model an offering exposes; one model, several routes |
| Entitlement | the operator's spending relationship with one provider |
| Route | one alias reaching a row's model, with its cost and health |
| Excluded | measured as unusable, so left out of the config; clears on a probe or a clock |
| Withheld | the operator chose not to use it; only a human clears it |
| Candidate | clears the structural filters but has no quality score; awaits approval |
| Declared Offering | an offering the operator wrote by hand; the feed does not publish it |
