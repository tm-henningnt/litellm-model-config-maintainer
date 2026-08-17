# Guidance ranks Canonical Models and never claims a balance

**Superseded 2026-07-28 by ADR 0013.** ClinePass and OpenCode Go, named
below as Entitlements that publish nothing, now publish a figure through
`codexbar`. ADR 0013 states what changed and what did not. The reasoning
below is kept as written, for the record of why the answer was once no.

`guidance` answers "what should I use for this kind of work" for a
calling agent. It ranks Canonical Models by one of the Feed's own
quality scores, filtered to what Health State says answers now, and
gives each row its routes in cost order. It reports cost as the Feed's
own token rates plus an Entitlement kind. It never reports how much
credit or quota is left, because nothing we can read knows that.

## Why there is no remaining-balance number

The Feed publishes no balance. Its `pricing.free.quota` field holds a
rate-limit tier string on 37 of 1164 Offerings and null elsewhere;
provider records carry only authentication and signup facts. A number
would have to come from one of three places, and each was rejected.

**A local token ledger.** Account for every call the proxy serves and
report the burn. It measures our consumption, never the provider's
remaining balance, so any spend from another client makes it wrong with
no symptom. It also needs a seventh file and a success-path callback.

**Per-provider balance adapters.** Authoritative where an endpoint
exists, absent for most Entitlements — ClinePass, OpenCode Go and Gemini
publish nothing — and a new per-provider maintenance burden of exactly
the kind `docs/gotchas.md` documents.

**Operator-declared budgets.** Policy states the plan size and we
subtract. It puts provider-limit bookkeeping back into our files, which
is the provider's job, and the figure is a fiction the moment anything
else spends from the same plan.

What we do know is what we measured: which Offerings answer, which
refused, why, and when a refusal said it would reset. Guidance reports
that, and says nothing more.

## Why a row is a Canonical Model

In the audited Feed, 345 of 618 Canonical Models have more than one
route; `deepseek-v4-flash` has eight and `glm-5.2` has seven. A ranked
list of Aliases would name one model seven times before reaching the
second. A row is therefore the model, and the routes hang off it in cost
order, which doubles as a failover order.

## Where one model's Routes disagree

Each Offering carries its own quality record, so two Routes to one
Canonical Model can state different scores. The row takes the highest,
because the score describes the model rather than the Route. In the
audited Feed no Canonical Model's Routes disagree on the coding score, so
the rule changes nothing today. It could mislead later: if a provider
ever serves a degraded or quantised variant under the same Canonical
Model, the row shows the better sibling's score while `--prefer` may
still steer to the cheaper, weaker Route. Accepted, and recorded here so
the next reader knows it was a choice.

`canonical_model.confidence` is ignored for the same reason it is safe
today: every medium-confidence grouping in the audited Feed is a genuine
same-model merge. Read it if a wrong merge ever appears.

## Why the ordering is not one number

Models descend by the requested score. Routes ascend by what they cost.
`--prefer free` or `--prefer flat_rate` re-sorts the model list into cost
tiers for bulk work. A single weighted composite was rejected: the
weights would be arbitrary, the result unexplainable, and every
recalibration would silently reshuffle every caller's choice.

## Consequences

The response carries a `schema_version`, because callers parse it.

The response carries a Client Advisory. A client caches `/v1/models` and
the config changes underneath it, so guidance states the config's
generation time, records that an Alias is callable by exact id whether or
not the client's cached list holds it, and names the Aliases added and
removed on the last run — a removal with its reason and reset time, so a
caller stops retrying a dead Alias. The Previous-run record already holds
those two sets, so this needs no new file. Drift older than the last run
is not reported; a caller that needs more can diff its own cached list
against the current one.
