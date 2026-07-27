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

## An unscored row is not a weak row

The Feed does not publish a Declared Offering, so the Feed states no
score for it. A Declared Offering still gets a row. Its `score` is
`null`, its `provider_id` is `declared`, and its `why` says the operator
declared it.

Such a row ranks last, because the sort has no score to place it by.
Do not read that position as a verdict. A direct vendor entry is often
the strongest model the proxy serves, and it is exactly the kind of
Offering the Feed does not cover. Rank an unscored row with your own
knowledge of the model it names.

Its `cost_basis` is `passthrough` when the calling client supplies the
credential, and `unknown` when the proxy holds the credential and the
Feed states no rate. Its `entitlement` reads `declared`, which is the
third value that field takes; the other two, `shared_pool` and
`per_model`, apply to a provider the Feed publishes.

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
requested axis, `null` if unscored), `scores` (every axis this row
carries), `capabilities`, `callable_now`, `why` (one line stating why the
row ranks where it does), and `routes`.

Each route carries `alias`, `offering_id`, `provider_id`, `cost_basis`,
`entitlement`, `available`, `rate_is_list_price`,
`input_usd_per_1m_tokens`, `output_usd_per_1m_tokens`, `context_tokens`,
`max_output_tokens`, `reason`, and `refills_at`. `reason` and
`refills_at` are set only when the Route currently refuses.

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

`entitlements` answers that question, provider by provider: the
Entitlement kind, the cost, how many Offerings answer now, and the
earliest refill time for one that does not.

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
--json`, parse `rows`, and call the first `available` Route in the
first row. On a failure, read the Route's `reason`, then move to the
next Route in the same row before moving to the next row.

**A scheduled project task keeps project docs current.** Run
`guidance --format markdown > docs/models.md` on the same schedule as
your other project automation. The redirect is the caller's; `guidance`
itself writes nothing.
