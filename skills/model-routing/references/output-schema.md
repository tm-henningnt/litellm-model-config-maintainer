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

Shared flags on `guidance`, `entitlements`, `status` and `doctor`:

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

`status` answers `--json` too, at its own `schema_version`. Read it for the
data no other JSON command carries: why each offering is Withheld, and why
each is Excluded with its `bucket` and `reset_at` as separate fields. Those
reasons explain a headroom window that reads full and governs nothing
spendable. `status` takes no `--for`; it describes the whole instance.

`--format markdown` exists so a scheduled task can write a page into a project:
`litellm-maintainer guidance --for coding --format markdown > docs/models.md`.
The redirect is the caller's; these commands never write a file themselves.

## guidance JSON

```json
{
  "schema_version": "3",
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
leaves; a new field does not raise it. So pin the major and tolerate fields you do
not know — `allowance_id`, `fair_use` and `headroom` all arrived on 2026-07-28 and
this stayed `3`. `tier` arrived on 2026-07-29, purely additive, and this stayed `3`
too.

`guidance` and `entitlements` carry SEPARATE versions. `entitlements` rose to `2`
on 2026-07-28, when declared offerings gained one entry per allowance; `guidance`
did not move.

### A row

| Field | Notes |
| --- | --- |
| `canonical_model_id` | the model, e.g. `z-ai/glm-5.2`. For a declared offering, its alias — unless it names a **reference model**, in which case its route joins that model's row. A declared offering that states it is a variant of another gets no row of its own: it appears as that row's route `wide_alias`. |
| `display_name` | human name from the feed |
| `score` | score on the requested axis; `null` for a declared offering with no reference model |
| `score_source` | `feed` (an offering this proxy serves), `reference` (the same model served elsewhere, named by the operator), or `null`. Rank a `reference` score exactly as you rank a `feed` score: a score describes the model, and the model is the same |
| `scores` | all four axes, each possibly `null` |
| `capabilities` | capability strings, e.g. `chat`, `coding`, `tool_use`, `reasoning`. From the feed for a discovered row; from the operator's Policy for a declared row, whose `why` says so. Empty on a declared row means "not stated", never "not supported" |
| `callable_now` | true when at least one route is **recommendable** (available AND not exhausted AND not demoted by a reading) |
| `not_callable_because` | `null` when `callable_now` is `true`. Otherwise the row-level counterpart of `not_recommended_because`, same value set. `"headroom"` means EVERY route here is demoted only by a reading — nothing was measured, and the routes are still `available` and not `exhausted`, so a caller that will not refuse work on a report may keep the row. Any other value is measured, and a mixed row names the measured cause: read each route's own field to find which are merely demoted. Added 2026-07-29, after a consumer that keeps a demoted ROUTE found the ROW gate dropping it anyway. **Guaranteed: `callable_now` is `false` exactly when every route in the row carries a non-null `not_recommended_because`.** So filtering routes by their own reason is correct on its own, and this field is a summary, never a fact the routes lack. A row CAN state a measured reason while holding a route nothing measured — one exhausted route beside one merely demoted gives `"exhausted"` — so gate on the routes, not on the row, when that distinction matters |
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
| `cost_basis` | `free`, `flat_rate`, `metered`, `passthrough`, `unknown`. A provider may state its own in Policy, overriding what the feed's pricing kind implies — the feed cannot see an account plan |
| `allowance_id` | **what gets billed.** Two routes sharing this share one ceiling. `provider:<id>` for a feed provider; `pool:<name>`, `credential:<VARIABLE>` or `alias:<alias>` for a declared one. **Filter on this, not `provider_id`**, which reads `declared` for every declared route. Never build one from an alias |
| `fair_use` | the allowance is unmetered under a fair-use clause. `cost_basis` still says who bills; this says it tolerates load badly. Require it be named rather than failing over into it. Changes no ranking |
| `tier` | the subscription level this route's allowance bills under, as the operator stated it in Policy's `allowances` block, or `null` when they named none. A label: published verbatim, never parsed, ranked or derived from. A headroom `used_percent` is a SHARE of this tier's own ceiling, so read the two together |
| `scale_note` | how big this route's allowance is, as the operator states it, or `null`. Prose. For a vendor that sells ONE fixed price and ONE quota with no levels, `tier` has nothing to name and this carries the size instead. Read either beside `headroom`: both answer "a share of WHAT" |
| `draw_note` | how fast THIS offering empties its allowance, as the operator states it, or `null`. Prose — print it, rank nothing by it. Answers a different question from `scale_note`: one sizes the allowance, this states the rate one model draws on it, and a pool can hold several models drawing at several rates. It exists because a subscription offering carries no published rate at all: measured 2026-07-30, one model billed 10% of its pool's normal rate, and 2% inside a daily window, while every rate field on it read `null`. Absent from `entitlements`, which is keyed by allowance and has no offering to hang it on |
| `headroom` | this route's BINDING WINDOW figure, flat, or `null`. The same three fields also read flat on an `entitlements` headroom, so `.headroom.used_percent` answers identically on both. An object with `used_percent`, `window_minutes`, `resets_at` and `age_seconds` (how long ago the SOURCE measured it, never how long ago you asked). `null` means one of: no source is mapped, no reading yet, every window expired, or the source dropped this route's sub-allowance window on the call that produced the reading (it restores it -- one route `null` beside siblings with figures is unknown, never uncapped). `null` NEVER means free — do not print it as `0%`. It is measured and it lags: another caller can spend from the same credential before you dispatch. On most Policy it changes no ranking and no `recommendable` — read it and decide for yourself, unless `headroom.demote_at_full` is on (see `recommendable` below). `entitlements` publishes the full window set for the same allowance; this is the one figure that binds. For a route on a declared sub-allowance (see `headroom.sources.<id>.members` in Policy), this is the WORSE of the allowance's own figure and the sub-allowance's own window, never the allowance's figure alone — a sibling route on the same allowance that names no sub-allowance is unaffected |
| `entitlement` | `shared_pool`, `per_model`, or `declared` |
| `available` | the proxy still serves this alias. `false` means "model not found" if you call it |
| `exhausted` | a recorded quota exhaustion has not cleared. The alias IS served and IS callable, but will almost certainly refuse until `refills_at` |
| `recommendable` | `available and not exhausted`, and — only where Policy's `headroom.demote_at_full` is `true` — not demoted on a `headroom.used_percent` of 100. That flag defaults off. **Filter on this, not on `available`** |
| `not_recommended_because` | `null` when `recommendable` is `true`. Otherwise `"health"` (the offering is excluded), `"exhausted"` (a recorded quota exhaustion; a measured fact), or `"headroom"` (demoted on a 100% reading; a report through a Policy mapping that can rot). Read this before treating a demoted route as one the maintainer actually called and saw refuse |
| `rate_is_list_price` | true for `free` and `flat_rate`: the rate is a list price, not a bill |
| `input_usd_per_1m_tokens` | may be `null` |
| `output_usd_per_1m_tokens` | may be `null` |
| `rate_source` | `feed`, `reference`, `operator`, or `null`. A `reference` rate is what ANOTHER vendor charges for the same model: read it as relative burn, never as your bill |
| `context_tokens` | context window, may be `null`. A Stated Limit: the Feed states it for a discovered route, the operator for a declared one. A reference model NEVER supplies one — this figure always describes the route you will call. `null` means no source stated it — read it as unknown, never as small |
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
| `allowance_id` | the same key a route carries, so the two answers join |
| `fair_use` | true when any offering in this allowance declares it |
| `tier` | the same string a route carries for the same allowance, or `null`. Read it beside `headroom` below: a used share means nothing without the tier it is a share of |
| `headroom` | how drawn this allowance is, or `null`. `null` means UNMEASURED: no source mapped, no reading yet, or every window void. Never free — do not print it as `0%`. When present it carries the binding figure FLAT — `used_percent`, `window_minutes`, `resets_at`, the same three names a guidance route uses — plus `source`, `updated_at`, `read_at`, `age_seconds`, `binding`, the three slots `primary` / `secondary` / `tertiary`, and `extra_windows`. The flat fields and `binding` hold the same figure and cannot disagree |
| a window inside `headroom` | `used_percent`, `window_minutes`, `resets_at`, `void`, `sub_allowance_id` and `admitted_members`. A void window reports `used_percent: null`: its own reset has passed, so the stored figure describes a period that ended. `resets_at` is `null` where the source states no reset. **an empty LIST in `admitted_members` on a full window means it governs nothing you can spend** — a free Gemini plan reads 100% on its Pro slot, whose offerings are all withheld, while admitted Flash routes on the same allowance read 0%. Do not read that 100% as the allowance's state. `admitted_members` is `null` — not `[]` — where policy declares no membership for the window, which is the ordinary case: a parent window governs every offering on the allowance. It carries no signal about an allowance that admits NOTHING; a parent window there reads full with `admitted_members: null`, so cross-check `in_scope: 0` on the entitlement itself. Windows in `extra_windows` carry both fields too. `sub_allowance_id` is the operator's own name for the window, or `null` where policy names no slot |
| `headroom.binding` | the one window that caps the whole allowance, or `null`. **A `null` here is not the same as a `null` `headroom`.** It means the allowance WAS measured and nothing caps it as a whole: policy names every slot a sub-allowance, because the source fills the three slots with one quota per MODEL rather than nested time windows (Gemini: Pro, Flash, Flash Lite). Read the per-route `headroom` in `guidance` for those figures — each route carries its own model's window. The text and Markdown renderings print "per-Route" rather than a dash |
| `cost_bases` | every basis present |
| `answering` / `unavailable` / `in_scope` | counts of admitted offerings |
| `withheld` | offerings the operator deliberately holds back; never probed |
| `candidates` | offerings awaiting the operator's approval; not offered |
| `earliest_refill_at` | soonest refill among the unavailable; `null` when none stated |
| `declared` | **deprecated duplicate.** Every declared offering, summed. Each also appears as its own `entitlements[]` entry since schema 2, so NEVER sum `answering` across both — that counts every declared offering twice. Kept only so an older consumer does not break |

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
