---
name: model-routing
description: Query the local litellm proxy for which models are actually reachable right now, what they cost, and which are exhausted, using the `litellm-maintainer` CLI. Use this before delegating work to a model or subagent, before picking a model for a batch of tasks, and whenever a proxy call fails with a quota, rate-limit or unknown-model error. Also use it whenever the user asks what models are available, which model to use for a task, what a lane costs, why a model stopped working, or when an exhausted plan comes back. Reach for this rather than guessing from a hardcoded model table or a cached model list, because both go stale silently.
---

# Model routing

You are choosing a model to run work on. A hardcoded table of "good models"
and a `/v1/models` list your client fetched at startup both go stale without
telling you: plans run dry, free tiers appear, aliases get withdrawn mid-session.
The `litellm-maintainer` CLI answers from what was actually measured minutes ago.

Three commands. All are read-only: they print and exit, and write nothing.

```
litellm-maintainer guidance --for coding --json    # which model, and how to reach it
litellm-maintainer entitlements --json             # what is exhausted, and when it refills
litellm-maintainer doctor                          # why nothing works
```

No flags are needed. Both the Policy and the Feed default to the instance
directory (`$LITELLM_MAINTAINER_HOME`, or `~/.config/litellm-maintainer`).

## Pick a model

```
litellm-maintainer guidance --for coding --json
```

`--for` takes `coding`, `reasoning`, `agentic` or `speed`. These are the score
names the upstream model feed publishes, so asking for an axis it does not
score exits 1 and names the four.

Read the first row whose `callable_now` is true, then call its **first route**:

```json
{"rows": [{
  "canonical_model_id": "<vendor>/<model>",
  "score": 68.8,
  "callable_now": true,
  "why": "scores 68.8 on the requested axis; no marginal cost, drains a flat-rate window",
  "routes": [
    {"alias": "claude-<provider-a>-<model>", "cost_basis": "flat_rate",
     "available": true, "exhausted": false, "recommendable": true},
    {"alias": "claude-<provider-b>-<model>", "cost_basis": "flat_rate",
     "available": true, "exhausted": true, "recommendable": false,
     "reason": "quota_exhausted", "refills_at": "2026-07-29T21:45:00+00:00"}
  ]}]}
```

The `alias` is what you send to the proxy as the model id. When a route also
carries a `wide_alias`, send that instead to get the model's full context
window — some clients read their context budget from the model name, so the
plain `alias` can budget far less than the model accepts. Never build a wide
name yourself by appending a suffix.

**The route list is a failover order.** Routes are sorted recommendable-first,
then cheapest. If route 1 refuses at call time, call route 2 — the ordering
already accounts for cost, so reading down the list is the correct retry.

**Read `recommendable`, not `available`.** They are different, and the
difference is not a quirk:

- `available: false` — the proxy no longer serves this alias at all. Calling it
  gets you "model not found".
- `available: true, recommendable: false` — the alias IS served and IS callable,
  but a recorded quota exhaustion has not cleared, so it will almost certainly
  refuse. `refills_at` says when that changes.

The second case exists because some routes bill to the *calling client's* own
credential rather than the proxy's. One caller exhausting their allowance must
not remove the model for everyone, so the alias stays served. All the direct
Claude aliases work this way. `callable_now` is false when every route in a row
is unrecommendable, so trusting `callable_now` and then the first route is
correct — but if you filter routes yourself, filter on `recommendable`.

One row is one model, not one alias. Most models are reachable several ways —
one model can have seven routes — so a flat list of aliases would name it
seven times before reaching a second.

### Choose the axis and the cost tier deliberately

Warning: `--prefer` sorts, it does not filter. Neither flag removes a route,
so a failover walk can still reach paid capacity while free routes sit
untried. Measured 2026-07-27: with `--prefer free` the live route order read
free, flat_rate, free, flat_rate. To constrain spend, filter on `cost_basis`
yourself.

For a batch of bulk mechanical work, add `--prefer free` or `--prefer
flat_rate`. That sorts models into a cost tier before the score, which is what
you want when you are spending a rate-limit window across many calls rather
than buying one good answer:

```
litellm-maintainer guidance --for coding --prefer free --limit 5 --json
```

Without `--prefer`, rows rank by score alone. Use that for one hard task.

Read `cost_basis` for money and `entitlement` for nothing about money. Cost is
`free`, `flat_rate`, `metered`, `passthrough` or `unknown`. An `entitlement`
says how an allowance is held: `shared_pool` when one allowance covers several
models, `per_model` when each has its own, `declared` for a hand-declared entry.

An allowance is identified by the CREDENTIAL it bills to, not by the provider.
Two subscriptions with the same vendor are two allowances, and one running dry
says nothing about the other.

`free` and `flat_rate` cost nothing at the margin; they drain capacity already
paid for. `metered` bills per call. `passthrough` is the direct-bill tier, and
it is the most expensive: measured 2026-07-27, one trivial Dispatch to a
passthrough route cost 2.93 USD. `unknown` may be money, so treat it as money.

`--limit` caps the rows; the output states what it dropped, so a capped list
never reads as the whole picture.

### Read `allowance_id` to refuse capacity, never `provider_id`

Each route carries `allowance_id`: what gets billed. Two routes sharing that
string share one ceiling, so draining one drains the other.

Every declared route reports `provider_id: "declared"`, so filtering on that drops
two subscription seats and a private host together. Measured 2026-07-28: excluding
the `declared` bucket cost 20 routes of 70, where refusing one allowance costs 4.

Forms are `provider:<id>` for a feed provider, and `pool:<name>`,
`credential:<VARIABLE>` or `alias:<alias>` for a declared one. Use the whole string
as a key. Never build one yourself from an alias: the aliases do encode the seat,
and the naming rule is an operator setting, so a guess breaks when it changes.

`entitlements` reports one entry per allowance under the same id, so pick a route
in `guidance` and read its ceiling in `entitlements`.

### `fair_use: true` means name it, do not fail over into it

The allowance is unmetered under a fair-use clause, with no number stating the
line. Its `cost_basis` still reads `flat_rate`, because that is who bills — the
flag is the part that is not safe.

So a Role accepting `flat_rate` by default walks into it when its free routes
drain. Require it be named, the way `metered` and `passthrough` already must be,
and pace bulk work when you use it. It changes no ranking.

### `tier` names what a headroom percentage is a share of

Each route carries `tier`: a string, or `null`. It is the subscription level
the route's allowance bills under, as the operator stated it in Policy —
`"claude-max-5x"`, say.

A headroom `used_percent` is a SHARE, and a share is scale-free: 89% used
means far more work on a Claude Max 5x seat than on a Max 1x seat. Read
`tier` beside `headroom` to know which one you are looking at.

It is a label. Nothing ranks by it, parses it, or derives anything from it
— print it verbatim. `null` means the operator named no tier for this
allowance, not that none exists. `entitlements` publishes the same string
for the same allowance, so the two answers agree.

### `headroom` says how drawn a route's allowance is, or nothing at all

Each route carries `headroom`: an object with `used_percent`, `window_minutes`,
`resets_at` and `age_seconds`, or `null`. `age_seconds` is how long ago the
source measured it, not how long ago you asked.

`null` covers four different states alike: no source is mapped to this
allowance, no reading exists yet, every window in the reading has expired, or
`headroom refresh` has not run. Read `null` as unknown. Never read it as free
capacity, and never print it as `0%`.

One route reading `null` while its siblings on the same allowance read a figure
is a fifth state, and it means the same thing: unknown. The source drops a
sub-allowance window between calls and restores it, so a reading taken during a
drop leaves that one route with nothing. Treat the route as unmeasured. Do not
conclude its cap was lifted.

The figure lags: a second caller can spend from the same credential between
the reading and your dispatch, so it is a report, never a reservation. On
most operators' Policy it changes no ranking and no `recommendable` — a
fully-drawn route still sorts and answers exactly as it did before this
field existed. Route around it yourself when the figure is high.

`entitlements` publishes the full window set behind this same allowance,
including the windows that are not binding; `guidance` publishes only the one
that binds, so a route stays one line to read.

`.headroom.used_percent` reads the same on both. A route carries the binding
figure flat; an entitlement carries it flat AND under `binding`, which names
what it is. Neither command needs a different expression.

On `entitlements`, `headroom.binding` can be `null` while `headroom` itself is
present. That is not absence. It means nothing caps the allowance as a whole,
because the source measures one quota per MODEL rather than a nested time
window — a free Gemini plan reports Pro, Flash and Flash Lite that way. Read
the per-route figures in `guidance` for those; each route carries its own
model's window.

### A full window can govern nothing you can spend

Each window inside an `entitlements` headroom states `admitted_members`: the
health keys that draw on it. An empty list beside a high `used_percent` means
policy admits nothing on that window, so the figure is not capacity you lost.

A free Gemini plan is the measured case. Its Pro slot reads 100% used, because
the plan includes no Pro at all, and both Pro offerings are withheld. Three
admitted Flash routes on the same allowance read 0%. Excluding that allowance
on the 100% discards working capacity.

Note the direction. A sub-allowance normally binds TIGHTER than its parent, and
reading the parent alone is then optimistic. This is the reverse: the allowance
rollup is pessimistic. Read the routes, and read `admitted_members` before you
act on any allowance-level figure.

`admitted_members` appears on windows in `extra_windows` too, not only on the
three slots. A sub-allowance named directly lives there.

It answers for a window whose policy DECLARES membership, and nothing else.
An allowance can admit nothing at all while its parent window reads full, and
every window there reports `admitted_members: null`. Cross-check `in_scope`:
`0` means policy admits nothing, so the figure describes capacity you no
longer have rather than capacity you spent.

### `not_recommended_because` names why a route is not recommendable

An operator may turn on `headroom.demote_at_full` in Policy. Where that
flag is on, a route whose `headroom.used_percent` reads 100 stops being
`recommendable`, and a row whose every route reads 100 reports
`callable_now: false` — the same demotion an observed quota exhaustion
already gets. The flag defaults off: a reading travels through a
hand-written Policy mapping and a tool that documents no contract, and
either can rot before the operator notices.

Read `not_recommended_because` on any route where `recommendable` is
`false`. It names one of three causes: `"health"` (the offering is
excluded), `"exhausted"` (a recorded quota exhaustion has not cleared), or
`"headroom"` (demoted on a 100% reading). The first two are facts: the
maintainer called this offering and was told no. The third is a report
through a mapping that can rot. It is `null` whenever `recommendable` is
`true`.

### `tier` and `scale_note` both answer "a share of WHAT"

`tier` names a subscription level. Some vendors sell none: one fixed price, one
quota, no levels above or below it. `scale_note` is free prose stating how big
the allowance is where the vendor states a size but no level — "roughly 2x to 5x
its API cost", say. Both are operator-stated, published verbatim, and ranked by
nothing. Read either beside `headroom`; a share means little without one.

`null` on both means the operator stated neither, never that no scale exists.

### `draw_note` says how fast ONE model empties the pool

`tier` and `scale_note` size the allowance. `draw_note` states the rate a
single route draws on it — a different question, and the one that decides which
model to send bulk work to.

It matters most where nothing else can tell you. A subscription offering
carries no published rate: `output_usd_per_1m_tokens` reads `null`, and the
feed scores it not at all. Measured 2026-07-30 on one pool of six models: one
billed 10% of the normal rate, and 2% inside a daily window, so it did fifty
times the work per unit drawn. Every rate field on it was `null`, and the two
models beside it were indistinguishable from it.

Prose. Print it, rank nothing by it, parse nothing out of it. Read it before
you pick a model inside one allowance, not just between allowances.

### `not_callable_because` on a row

A row carries `not_callable_because` for the same purpose, with the same value
set. `"headroom"` there means every route in the row is demoted only by a
reading. The routes are still `available` and not `exhausted`, so a caller that
declines to refuse work on a report may keep the whole row. Any other value is
measured. Check it before you drop a row on `callable_now: false` — the row gate
fires on a reading exactly as the route gate does.

Two guarantees, both pinned by tests, because a gate gets built on them.
**`callable_now` is `false` exactly when every route carries a non-null
`not_recommended_because`.** So filtering routes by their own reason is already
correct, and the row field is a summary rather than a fact the routes lack. And
**a row can state a measured reason while holding a route nothing measured** —
one exhausted route beside one merely demoted reads `"exhausted"`, because a
fact outranks a report. Where that distinction matters, gate on the routes.

A reading never clears an exhaustion: a route the maintainer measured as
exhausted stays not recommendable no matter what a later reading says,
even a reading well below 100%. Demotion also never rewrites the Generated
Config — a demoted route stays reachable by exact alias, exactly like an
exhausted one (ADR 0010).

The figure can come from a route's own sub-allowance, never its allowance's
alone. Some routes are capped inside their own Allowance — the operator's Claude
subscription caps `claude-fable-5` inside its weekly quota — and Policy names
the window that measures the cap. That route binds on the WORSE of the Allowance's
own figure and its own window. Measured 2026-07-28, the Allowance read 82% and
fable's own window read 59%, so fable bound at 82% that day; the case this
exists for is the reverse, fable draining while the rest of the Allowance has
room, and a route that read the Allowance alone would then report a healthy
figure for a route about to refuse. This never reaches a sibling route on the
same allowance that names no sub-allowance.

### Read the source of every figure

A row carries `score_source` and each route carries `rate_source`. Three values:
`feed`, `reference`, `operator`.

`reference` means the operator named the same model where the feed DOES publish
it, so the feed's numbers came through a mirror. Rank a `reference` score exactly
as you rank a `feed` score — a quality score describes the model, and the model
is the same. Read a `reference` RATE differently: it is what another vendor
charges, so it ranks the relative burn and never states your bill.

A limit is never borrowed. `context_tokens` always describes the route you will
actually call. A ChatGPT seat states 350000 where the feed states 1050000 for the
same model on the vendor's API, and the seat's figure is the one that refuses.

### An unscored row is not a weak row

Some rows have `score: null`, `score_source: null` and `provider_id: "declared"`.
These are models the operator declared by hand because the upstream feed does not
cover them, with no mirror to borrow a score from — often direct vendor entries,
which are frequently the strongest models on the proxy. They sort last only
because there is no score to sort them by. Their `why` says so. Judge them
yourself; do not read "unscored" as "bad".

### `flat_rate` is not free at the margin

`rate_is_list_price: true` marks a rate as a list price, never an amount billed.
Do not sum it into an invoice. Read it as a burn rate: a subscription pool holds a
finite allowance, and one model can draw on it 50 times faster than another for a
few points of score. Measured 2026-07-28 on one pool, 15.00 per 1M output tokens
against 0.28 — 53.6 times the burn for 34% more score.

Price the output, not the task. Spend the top band on a small, high-leverage
output. Never on bulk code.

## Know what is exhausted

```
litellm-maintainer entitlements --json
```

One entry per ALLOWANCE, with `state` reading `healthy`, `degraded`, `dry` or
`empty`, plus every offering that is admitted but not currently callable, with
its reason and refill time. Each carries the same `allowance_id` a route does,
so the two answers join, and the same `tier` a route does, for the same
reason:

```
<provider-b>-token-plan  dry  shared_pool  flat_rate
  0 of 6 answering
  one shared pool, and every admitted Offering has refused
  earliest refill: 2026-07-29T21:45:00+00:00
```

Use this to answer "why did that model disappear" and "is it worth retrying".
A `refills_at` in the future means waiting works and no call is needed to find
out. A `reason` of `needs_operator` or `authentication_failed` means waiting
does not help and a human has to act — say so rather than retrying.

`entitlement: shared_pool` explains why several models failed together: they
bill from one pool. It is an explanation of what was measured, never a
prediction about a model nobody tested. When one member of a pool reports a
quota exhaustion, the maintainer marks its pool mates for measurement on the
next run — so a stale-looking sibling is usually about to be re-measured, not
forgotten. It never marks them unavailable on the strength of a sibling.

## Your cached model list is probably wrong

This is the part that catches agents out. Your client fetched `/v1/models` once
and cached it; the proxy's config has changed since. Every guidance response
carries a `client_advisory`:

```json
{"client_advisory": {
  "added_last_run": ["claude-zen-laguna-s-2-1-free"],
  "removed_last_run": [{"alias": "claude-<provider>-<model>", "reason": "quota_exhausted",
                        "refills_at": "2026-07-27T00:00:00+00:00"}]}}
```

Both directions matter, and only one is recoverable on your own:

- An alias in the guidance output **is callable by its exact id right now**,
  whether or not your cached list holds it. The proxy resolves by alias, not by
  what your client last fetched. So do not skip a good model because your list
  does not mention it.
- An alias in `removed_last_run` is **no longer served**. Retrying it burns
  turns. Read its `reason` and `refills_at` instead.

## What this will never tell you

There is no remaining balance, credit count or quota percentage anywhere in the
output, and none can be added. The upstream feed publishes no balance, and most
providers expose no endpoint for one. What the tool knows is what was measured:
which routes answered, which refused, why, and when a refusal said it clears.

So do not look for a "how much is left" field, and do not infer one from the
token prices. For a `flat_rate` route, `rate_is_list_price` is true — that price
is the provider's list price, useful for judging how heavily a call drains a
subscription window, never an amount billed.

## Reading the cost basis

| `cost_basis` | Means |
| --- | --- |
| `free` | no cost; usually rate-limited, so pace bulk work |
| `flat_rate` | no marginal cost, but drains a subscription window |
| `metered` | bills per token at the stated rate |
| `passthrough` | billed to the calling client's own credential |
| `unknown` | the feed states no rate; treat as billable |

## When the answer looks wrong

Check the `warnings` array in the JSON before trusting a surprising result.

- **"stale catalogue"** — the feed document is old, so the selection ran on an
  out-of-date model list. `litellm-maintainer fetch` refreshes it.
- **"Health State is empty"** — nothing has been probed, so every model claims
  to work and none has been verified. `litellm-maintainer probe` measures them.
- **Everything is dry, or a command fails** — run `litellm-maintainer doctor`.
  It checks credentials, feed age, whether the proxy answers, and which
  providers no probe has reached. Each failed check names the command that fixes
  it. It exits 0 when everything passes and 1 when anything failed.

## Do not write to the configuration

These three commands only read. The CLI does have verbs that edit the operator's
policy (`litellm-maintainer policy withhold`, `approve-candidate`,
`set-entitlement`), and they are the operator's decisions to make, not yours.

If you conclude that a model should be withheld or a provider marked as one
shared pool, **print the exact command and let the human run it**:

> `claude-foo` has failed authentication on the last three runs. Consider:
> `litellm-maintainer policy withhold openrouter:vendor/foo --reason "auth failing since 2026-07-26"`

## A worked orchestration

You have twelve mechanical refactors to delegate and one hard architectural
question.

```
litellm-maintainer guidance --for coding --prefer free --limit 3 --json   # the twelve
litellm-maintainer guidance --for reasoning --limit 3 --json              # the hard one
```

Take the first `callable_now` row from each. Use the free or flat-rate alias for
the batch so the twelve calls cost nothing marginal, and the highest-scoring
reasoning alias for the hard question, cost being the lesser concern for one
call. Keep each row's remaining routes; if an alias starts refusing mid-batch,
fail over to the next route rather than re-querying.

If a call fails and you want to know whether to wait:

```
litellm-maintainer entitlements --json
```

A future `refills_at` means wait. `needs_operator` means tell the human.

## Deeper reference

`references/output-schema.md` documents every field of both JSON outputs, the
exit codes, and the full flag list. Read it when you need a field this page does
not name, or when you are parsing the output programmatically rather than
reading the first row.
