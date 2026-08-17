# The Allowance is published, never inferred from a name

`guidance` said which model to call. It never said whose allowance paid for
it. Every Declared Route reported `provider_id: "declared"` and nothing
else, so a client could not tell one subscription seat from another, and
could not refuse a fair-use host without refusing every Declared Route with
it.

Measured 2026-07-28 by a downstream orchestrator: excluding the whole
`declared` bucket took an ordinary Role from 58 Routes to 42. On the same
instance after this change, refusing only the fair-use allowance costs 4
Routes of 70 instead of 20, and both subscription seats survive.

A Route now carries `allowance_id`, and an Entitlement entry exists per
Allowance.

## The credential names the Allowance, not `group`

The request named Policy's `group` line as the source, slugified.

`group` is a heading the Generated Config prints. `policy.py` says it "names
nothing the code acts on", and `plan.py` reads it once, to label a section of
a file. Deriving a behavioural key from a display sentence means an operator
rewriting a comment moves a client's cap, with nothing to show it moved.
Slugifying does not fix that; it only makes the failure quieter.

The credential is already the answer. ADR 0009 records the rule — the
credential identifies an Entitlement, because the credential is what gets
billed — and `declared_pool_id` has implemented it since. Against the live
Policy it produces exactly the grouping the request asked for, with no new
operator input:

| Allowance id | Offerings |
| --- | --- |
| `pool:claude-subscription` | 4 Claude direct |
| `credential:EXAMPLE_CHATGPT_SEAT1_WORKER_KEY` | 6 |
| `credential:EXAMPLE_CHATGPT_SEAT2_WORKER_KEY` | 6 |
| `credential:EXAMPLE_PRIVATE_HOST_API_KEY` | 4 |

A Feed provider's Allowance is `provider:<id>`: the Feed states one
`credential_hint` per provider, so a provider IS an allowance there. One
field answers for both kinds.

Three properties the shape defends:

- **`os.environ/` is stripped.** The id carries a variable name, never a
  value. A variable name is not a secret; a value is, and this field must not
  be able to hold one.
- **It is never `null`.** `declared_pool_id` answers `None` for an Offering
  with neither a named pool nor a credential, because it answers "which pool
  propagates a Probe" and such an Offering propagates to nobody. This answers
  "who is billed", and the answer is itself — so it falls back to
  `alias:<alias>`. A `null` would read as one shared allowance for every
  unpooled Offering.
- **The namespace prefix stays.** A provider called `gdm` and a credential
  called `gdm` are not one allowance, and a client uses the whole string as a
  key.

Never derive it from an Alias. The Aliases here do encode the seat —
`claude-chatgpt1-*` against `claude-chatgpt2-*` — and that is exactly the
guess the field exists to prevent: the naming rule is an operator setting, so
a guess breaks the day it changes.

## `fair_use` is not a Cost Basis

The request asked for `fair_use` as a sixth Cost Basis, so that a downstream
Role accepting `free` and `flat_rate` by default would have to name it.

The problem it names is real. A fair-use host is declared `flat_rate`, which
costs nothing at the margin, so it sits in every Role's failover path and a
bulk batch whose free Routes drain walks into it unthrottled.

It is still a separate field. A Cost Basis answers **who bills**, and a
fair-use plan bills flat rate — that is simply true, and overloading the
field would make it answer two questions at once. Load tolerance is the
second question, so it gets the second field. The precedent is `recommendable`
beside `available`: a distinct question got a distinct field, and clients were
told which to filter on.

Two consequences were checked:

- **It changes no ranking.** A `fair_use` Route sorts by its cost basis like
  any other. `_BASIS_ORDER`, `PREFERABLE_BASES`, `_basis_rank` and
  `rate_is_list_price` are untouched, so no consumer's five-value map breaks
  and the cost-basis canary test stays green.
- **`False`, never `None`.** A Policy that says nothing claims the allowance
  takes load normally. That is the safe reading, and it means an older Policy
  does not read as unknown risk.

## A provider may state its own Cost Basis

Added in the same change, for the same reason in a different place: the Feed
cannot see an account's plan.

Measured 2026-07-28 — the Feed marks Groq `paid` or `unknown` on an account
where every call is free, and prices Gemini per token where a Google One AI
Plus subscription covers it. Both therefore read as spend, and a caller
instructed to treat `metered` and `unknown` as money avoided capacity already
paid for. `providers.<id>.cost_basis` states what the provider costs THIS
account.

It changes what `guidance` and `entitlements` report and nothing else. It
never filters an Offering, never reaches the Generated Config, and never
overrides the Feed's token rates: a rate is a number the Feed measured, and
this is a statement about who bills.

## Which schema moves, and which must not

**`guidance.SCHEMA_VERSION` stays `"3"`.** A downstream client pins major 3
and fails loudly on anything else, so a bump it was not told about takes the
whole proxy away from it. `allowance_id` and `fair_use` are additive: a
consumer that ignores them parses exactly what it parsed before.

This corrects a misleading precedent. Both earlier bumps accompanied a new
Route field, so the pattern reads as "a new field bumps". Each was really for
the row folding that shipped alongside — a variant folding onto its sibling,
then a Reference Model folding onto a model's row — because a consumer
counting rows got a different answer afterwards. The fields rode along.

**`entitlements.SCHEMA_VERSION` rises `"1"` → `"2"`.** That one is not
additive. Iterating `entitlements` now yields Offerings it never yielded, and
`declared` still reports the same ones, so a consumer summing both
double-counts. `declared` is kept anyway, because removing it would break a
consumer for no gain.

The same bump carries a correction: `declared` counted each Client-Facing
Variant as an Offering of its own, reporting the operator's 20 Declared
Offerings as 24. A variant shares its sibling's Health Key (ADR 0007), so it
was never an Offering. The two readings now agree.

## Consequences

Declared entries are appended after the sorted Feed providers, so a consumer
indexing `entitlements` reads what it always read.

Nothing here reaches the Generated Config. `allowance_id` and `fair_use` are
guidance and entitlements only — verified by generating and diffing against
the live config, which was byte-identical.

An Allowance is a reporting key, not a limit. Nothing in this project knows
how much of an allowance is left; ADR 0005 still stands.
