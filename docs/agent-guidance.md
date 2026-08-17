# Guidance for an orchestrating agent

This page is for an agent that dispatches work through this proxy, and
for the human who wires one up. It documents `guidance` and
`entitlements`: the two commands an agent reads before it calls a model.

## What guidance is for

`guidance` picks a model before you dispatch work to it. It ranks the
Canonical Models this proxy currently offers, by one quality axis, and
lists every Route to each one. It never runs a call and never writes a
file.

```
litellm-maintainer guidance --policy $LITELLM_MAINTAINER_HOME/policy.yaml --for coding
```

## The axes

Pass one axis with `--for`: `coding`, `reasoning`, `agentic`, or `speed`.
These are the Feed's own score names, not scores this project invents.
`coding` is the default.

## A row is a Canonical Model; its Routes are a failover order

One Guidance Row names one Canonical Model. Its `routes` list names
every Alias that reaches that model, ordered cheapest first. Call the
first Route. If it refuses, call the next one in the same row; the order
already accounts for cost, so a failover reads the list in order and
stops at the first Route that answers.

Rows themselves rank by the requested axis, highest first. Routes within
a row never mix into that ranking; the two orderings stay separate.

## Read the source of every figure

A row states `score_source`. Each Route states `rate_source`. Three
values:

- `feed` — an Offering this proxy serves, as the Feed states it.
- `reference` — a **Reference Model**: the same model served elsewhere.
  The operator named the Canonical Model id, and the Feed's numbers for it
  come through. Nothing here was measured against the Route you will call.
- `operator` — the operator wrote it in Policy.

Rank on a `reference` score exactly as you rank a `feed` score: a quality
score describes the model, and the model is the same. Read a `reference`
RATE differently — it is what another vendor charges, so it ranks the
relative burn and never states your bill.

A limit is never borrowed. `context_tokens` on a Route always describes
that Route's own endpoint. A ChatGPT seat states 350000 where the Feed
states 1050000 for the same model on the vendor's API, and the seat's
figure is the one that refuses.

## An unscored row is not a weak row

The Feed does not publish a Declared Offering, so the Feed scores it only
through a Reference Model. Without one, its `score` is `null`, its
`provider_id` is `declared`, its `score_source` is `null`, and its `why`
says the operator declared it.

Such a row ranks last, because the sort has no score to place it by.
Do not read that position as a verdict. A direct vendor entry is often
the strongest model the proxy serves, and it is exactly the kind of
Offering the Feed does not cover. Rank an unscored row with your own
knowledge of the model it names.

Its `cost_basis` is whatever Policy states. Without a statement it is
`passthrough` when the calling client supplies the credential, and
`unknown` when the proxy holds the credential and the Feed states no rate.
Its `entitlement` reads `declared`, which is the third value that field
takes; the other two, `shared_pool` and `per_model`, apply to a provider
the Feed publishes.

## Which allowance pays for this Route

Each Route carries `allowance_id`: what gets billed. Two Routes sharing that
string share one ceiling, so draining one drains the other.

Read it, never `provider_id`, when you want to refuse or cap capacity. Every
Declared Route reports `provider_id: "declared"`, so filtering on that drops
two subscription seats and a private host together. On this proxy, refusing
one allowance costs 4 Routes of 70; refusing the `declared` bucket cost 20.

The forms are `provider:<id>` for a Feed provider, and `pool:<name>`,
`credential:<VARIABLE>` or `alias:<alias>` for a Declared one. Use the whole
string as a key; the prefix is part of it.

Never derive it yourself from an Alias. The Aliases do encode the seat, and
the naming rule is an operator setting, so a guess breaks the day it changes.

`entitlements` reports one entry per allowance under the same id, so pick a
Route here and read its ceiling there.

## How drawn is this Route's allowance right now

Each Route carries `headroom`, either an object or `null`. When present, it
states how much of the Route's Allowance a source has measured:
`used_percent`, `window_minutes`, `resets_at`, and `age_seconds` — how long
ago the source, not this proxy, took the reading.

`null` means one of four things: no source is mapped to this Allowance, no
Reading has been captured yet, every window in the Reading has expired, or
you asked for `guidance` before `headroom refresh` first ran. `null` never
means the allowance is free. Do not print it as `0%`, and do not treat its
absence as good news.

The figure is measured, and it lags: another caller can spend from the same
credential between the Reading and the moment you read it, so never treat
it as a reservation.

On most operators' Policy, `headroom` still never changes `recommendable` or
a Route's order — a drained Route ranks and sorts exactly as it did before
this field existed, and you must read `headroom` yourself and decide whether
to route around it.

An operator MAY turn on `headroom.demote_at_full` in Policy. Where that flag
is on, a Route whose `headroom.used_percent` reads 100 stops being
`recommendable`, and a row whose every Route reads 100 reports
`callable_now: false` — the same demotion ADR 0010 already applies to an
observed quota exhaustion, applied here to a Reading instead. The flag
defaults off, because the Reading travels through a hand-written mapping
that can rot; check `not_recommended_because` on a demoted Route to tell a
measured refusal from a report through that mapping. This never Excludes
the Route: it stays in the Generated Config and you can still reach it, and
a Reading never clears an exhaustion Health State already recorded.

This is the BINDING figure only: the one window, among the source's several,
that is most used. `entitlements` publishes the full window set for the same
Allowance, including the ones that are not binding; `guidance` does not, so
that a Route stays one line to read.

A Route's figure can come from its own Sub-allowance, not from its parent
Allowance. The operator's Claude subscription caps `claude-fable-5` inside its
own weekly pool: Policy names the window that measures the cap, and that
Route then binds on the WORSE of its Allowance's own windows and that
window. Measured 2026-07-28: the parent read 82% while the fable window read
59%, so fable bound at 82% that day — but the case Policy exists for is the
reverse, fable running dry with the rest of the Allowance untouched, and a Route
that ignored its own window would then report the parent's healthy figure
while it was about to refuse. Containment runs one way: the Sub-allowance's
own figure never reaches a sibling Route on the same Allowance that names no
Sub-allowance.

## The Tier states what the Headroom is a share of

Each Route carries `tier`, a string or `null`. It names the subscription
level the Route's Allowance bills under, as the operator states it in
Policy — for example `"claude-max-5x"`.

Read it beside `headroom`. A Headroom states a SHARE, and a share is
scale-free: 89% used means far more work on a Claude Max 5x seat than on
a Max 1x seat, and the percentage alone cannot tell you which one you
are looking at.

`tier` is a label. Nothing here verifies it, ranks by it, or derives
anything from it — print it verbatim beside the Headroom figure and read
it with your own knowledge of what the Tier means. `null` means the
operator named no Tier for this Allowance; it is not a claim that none
exists. `entitlements` publishes the same string for the same Allowance,
so the two answers never disagree.

## A fair-use allowance needs naming, not failover

`fair_use: true` means the allowance is unmetered under a fair-use clause,
with no number stating the line. Its `cost_basis` still reads `flat_rate`,
because that is who bills — the flag is the part that is not safe.

So a Role accepting `flat_rate` by default will fail over into it
unthrottled. Require it be named, the way `metered` and `passthrough` already
must be, and pace bulk work when you do use it.

It changes no ranking. A fair-use Route sorts by its cost basis like any
other, so the flag never quietly hides one behind a worse Route.

## A flat-rate Route is not free at the margin

`rate_is_list_price` is `true` on a `free` or `flat_rate` Route. The rate
beside it is a list price, so never sum it into an invoice. Read it as a
burn rate instead: a subscription pool holds a finite allowance, and one
model can draw on it 50 times faster than another for a few points of
score.

**What the field claims, exactly.** It is derived from `cost_basis` and
reads nothing about the rate. It licenses one thing: rank Routes by this
figure, never add it to an invoice. It does NOT claim the Offering is
subscription-included — a `free` Route reads `true` here too, and its
rate of 0 is its real bill.

`true` beside a `null` rate carries no information, because there is no
figure to rank. Read `cost_basis` for a claim about the Offering itself;
this field only restates it.

**This is the number that ranks Routes inside one Allowance.** Where an
Allowance holds several Routes and they state different rates, that
spread is the draw comparison — `provider:opencode-go` spans 0.28 to 15
across 17 Routes. Where every rate in a pool reads `null`, no such
comparison exists, and `draw_note` is the only thing that states it. See
"How fast one Offering empties its Allowance".

Price the output, not the task. Cost scales with tokens produced, so a
short answer from a top model is cheap and a long one is not.

## Bulk work: --prefer

```
litellm-maintainer guidance --policy $LITELLM_MAINTAINER_HOME/policy.yaml --for coding --prefer free
```

`--prefer free` or `--prefer flat_rate` sorts rows into a cost tier
before the axis score, for a batch of calls where cost matters more than
rank. Omit it to rank by score alone.

Add `--limit <n>` to cap how many rows come back. Add `--json` for
`--format json`, or pass `--format markdown` to produce a page a human
can read.

## The JSON shape

```
litellm-maintainer guidance --policy $LITELLM_MAINTAINER_HOME/policy.yaml --for coding --json
```

The top level carries:

- `schema_version`: parse against this, not against the field list.
- `axis`, `prefer`: the request you made.
- `derived_at`: when this answer was computed.
- `feed_generated_at`: when the Feed Document behind this answer was
  built.
- `warnings`: for example, that `--limit` cut rows.
- `client_advisory`: see below.
- `rows`: the ranked list.

Each row carries `canonical_model_id`, `display_name`, `score` (on the
requested axis, `null` if unscored), `score_source`, `scores` (every axis
this row carries), `capabilities`, `callable_now`, `why` (one line stating
why the row ranks where it does), and `routes`.

Each route carries `alias`, `offering_id`, `provider_id`, `cost_basis`,
`allowance_id`, `fair_use`, `tier`, `headroom`, `entitlement`, `available`,
`rate_is_list_price`, `input_usd_per_1m_tokens`, `output_usd_per_1m_tokens`,
`rate_source`, `context_tokens`, `max_output_tokens`, `wide_alias`,
`exhausted`, `excluded`, `recommendable`, `not_recommended_because`,
`reason`, and `refills_at`. `reason` and `refills_at` are set only when the Route
currently refuses. `headroom` is `null` unless a source measures this
Route's Allowance; see "How drawn is this Route's allowance right now"
above. `tier` is `null` unless the operator named one in Policy's
`allowances` block; see "The Tier states what the Headroom is a share of"
above.

## `available` does not mean the Route works

Warning: the proxy serves Offerings this tool knows to be failing. Read
`recommendable` to choose a Route. `available` answers a different
question and will mislead you.

`available` states that the Alias is in the Generated Config, so the
proxy resolves it. It does not state that a call succeeds.

An Offering the maintainer called and was refused becomes Excluded. It
KEEPS its entry in the config, because writing that file restarts the
proxy and ends every session in flight, and a measurement that reverses
itself must not do that. So the config is a superset of what answers.
See ADR 0014.

Two fields tell them apart:

    available      the Alias is in the config; the proxy resolves it
    recommendable  available AND not Excluded AND not exhausted AND
                   not demoted by a Headroom

A Route reading `available: true` beside `recommendable: false` is
correct and common. Calling it is not an error either — you reach the
provider's own message, such as an authentication error naming the
vendor, rather than "model not found" from the proxy, which names
nothing and invites a retry.

`excluded` states the Excluded fact on its own, beside `exhausted`.

`not_recommended_because` is `null` when `recommendable` is `true`.
Otherwise it names one of three causes: `"health"` — the Offering is
Excluded; `"exhausted"` — a recorded quota exhaustion has not cleared; or
`"headroom"` — `headroom.demote_at_full` demoted it on a 100% Reading. The
first two are facts the maintainer measured by calling the Offering; the
third is a report through a Policy mapping that can rot. Read this field
before assuming a demoted Route was ever actually called and refused.

## The Client Advisory

A calling client commonly caches `/v1/models` once and keeps using that
cached list. The Generated Config changes underneath it: a run adds
Aliases, removes Aliases, and Health State moves independently of
either.

The proxy resolves a call by exact Alias, not by what the client cached.
Two consequences follow, and `client_advisory` states both:

- An Alias not yet in your cached list is callable right now, by its
  exact id. Do not wait for a refresh before you use it.
- An Alias in `removed_last_run` fails if you call it. Read its `reason`
  and `refills_at` instead of retrying it; retrying a removed Alias
  wastes a call on an Offering that will not answer.

`added_last_run` and `removed_last_run` both come from the Previous-run
record, so they name only what changed on the last run, not the full
history of drift.

## Asking for reasoning

Send `reasoning_effort` on the request. Pass one of `low`, `medium`,
`high`, `xhigh` or `max`. The proxy pins no value, so the choice is
yours on every call.

litellm translates the parameter for the provider it dials. For the
Claude 5 family it becomes adaptive thinking plus an effort level.
Anthropic's own default on that family is `high`, so send a value only
when you want a different depth.

Sending it never fails. The proxy sets `litellm_settings.drop_params:
true`, so litellm discards the parameter for a model that does not accept
it rather than refusing the call. You therefore do not need a capability
check before you send it.

Read a row's `capabilities` for `reasoning` when you want to know whether
the depth will change anything. A Discovered row takes that list from the
Feed. A Declared row carries a list only when the operator stated one. An
empty list on a Declared row means "not stated", never "not supported".

Two caveats. A higher effort spends more output tokens, so raise
`max_tokens` with it. And no command here reports how much thinking a
model did before you call it.

## Reading a model's Stated Limit

Each Route carries `context_tokens` and `max_output_tokens`. Both are
Stated Limits: a figure a source stated, never one this project derived
from a model name. The Feed states them for a Discovered Offering. The
operator states them for a Declared Offering.

A `null` means no source stated the figure. Read it as unknown, never as
small. Do not fall back to a remembered default. litellm's own fallback
asserts 200000 tokens for any Claude-shaped model it does not know. That
figure is wrong for every current Claude model except Haiku.

An Alias may exist only so your client budgets the right window. A
`[1m]` suffix names the same model and sends the same request as the
plain Alias. It exists because some clients read their context budget out
of the model name. See CONTEXT.md, "Client-Facing Variant".

Do not assume your client reads this figure. Claude Code does not: it
budgets from the Alias name alone. Claude Code also takes an explicit
number, `CLAUDE_CODE_MAX_CONTEXT_TOKENS`, which applies to a whole
session rather than to one model. Pick the Alias when a session mixes
models, and the variable when a session serves one.

## What can I spend through right now

```
litellm-maintainer entitlements --policy $LITELLM_MAINTAINER_HOME/policy.yaml
```

`entitlements` answers that question, allowance by allowance: the
Entitlement kind, the cost, how many Offerings answer now, and the
earliest refill time for one that does not.

Each entry carries `allowance_id`, the same key `guidance` puts on a Route,
so the two answers join. A Declared allowance gets a full entry — a private
host used to report one aggregate count with no `state` and no refill time,
so the only way to find its ceiling was to hit it.

Each entry also carries `tier`, the same string `guidance` puts on a Route
for the same Allowance — `null` unless the operator named one in Policy's
`allowances` block. Read it beside `headroom`: a Headroom is a share of the
Tier's own ceiling, so the percentage alone does not say how much work that
is.

WARNING: `declared` still reports the same Offerings in aggregate, kept so an
older consumer does not break. Never sum `answering` across `entitlements`
AND `declared`; that counts every Declared Offering twice.

## No command reports a remaining balance

No command here states how much credit or quota is left on any provider.
Nothing this project can read knows that number. The Feed publishes no
balance field that holds one; a local ledger would measure only this
proxy's own consumption, which goes wrong the moment another client
spends from the same pool. See
[ADR 0005](./adr/0005-guidance-reports-what-we-measured.md) for the three
sources considered and rejected.

Do not build an agent that expects a balance from `guidance` or
`entitlements`. Both report what was measured: which Routes answer,
which refuse, why, and when a refusal said it clears. Neither reports
what remains.

## Two usage patterns

**An agent picks a model before dispatch.** Run `guidance --for coding
--json`, parse `rows`, and call the first `recommendable` Route in the
first `callable_now` row. On a failure, read the Route's `reason`, then
move to the next Route in the same row before moving to the next row.

Read `recommendable`, never `available`. See the section below.

**A scheduled project task keeps project docs current.** Run
`guidance --format markdown > docs/models.md` on the same schedule as
your other project automation. The redirect is the caller's; `guidance`
itself writes nothing.
