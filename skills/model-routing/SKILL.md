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

### An unscored row is not a weak row

Some rows have `score: null` and `provider_id: "declared"`. These are models the
operator declared by hand because the upstream feed does not cover them — often
direct vendor entries, which are frequently the strongest models on the proxy.
They sort last only because there is no score to sort them by. Their `why` says
so. Judge them yourself; do not read "unscored" as "bad".

## Know what is exhausted

```
litellm-maintainer entitlements --json
```

One entry per provider, with `state` reading `healthy`, `degraded`, `dry` or
`empty`, plus every offering that is admitted but not currently callable, with
its reason and refill time:

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
