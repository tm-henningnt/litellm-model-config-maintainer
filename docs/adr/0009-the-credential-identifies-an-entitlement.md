# The credential identifies an Entitlement, not the provider

An Entitlement grouped Offerings by their Feed provider. It now groups
them by the credential they are billed to. A Discovered Offering's
credential comes from the Feed's per-provider `credential_hint`, so
that case is unchanged; a Declared Offering's comes from its own
`litellm_params.api_key`.

## Why the provider was the wrong key

The operator runs two ChatGPT subscriptions as two worker proxies, six
Aliases each, both reached as `openai/`. They are two separate
allowances: seat 1 running dry says nothing at all about seat 2. Any
provider-level field calls them one pool and would propagate seat 1's
exhaustion onto six Offerings that are fine.

Their credentials already separate them —
`LITELLM_CHATGPT_SEAT1_WORKER_KEY` and `...SEAT2...` — and they do it
with nothing to configure. A third seat appears as its own pool the
moment it is declared, with no Policy edit. That property is the same
one that stopped `chatgpt_role_fix` going stale: a rule that reads the
world beats a list someone has to remember to update.

The rule is also just true. The credential is what the provider bills.
Two Offerings billed to one key share an allowance whatever else
differs about them, and two billed to different keys do not, however
similar they look.

## Considered options

**A pool name on every Declared Offering.** Explicit and obvious in the
file, and a hand-maintained list — the artifact CONTEXT.md says this
project exists to remove. Kept as an override, not as the rule.

**A top-level `entitlements:` block listing pools and members.** The
most expressive, and it duplicates the Alias list, so the two drift.

## Consequences

The rule errs in two known ways. Two keys billed to one account
under-group, so an exhaustion propagates less far than it could. One
key spanning a subscription plus pay-as-you-go over-groups, so it
propagates further than it should. `entitlement_pool` names the pool
explicitly for both.

Neither error is dangerous, and that is the point. An Entitlement
propagates only ATTENTION — a sibling is marked due for a Probe, never
Excluded (ADR 0004). So over-grouping costs a few probe calls and
under-grouping misses an optimisation. Getting this wrong cannot evict
a working Offering.

A Passthrough Auth Offering carries no credential at all: the caller
supplies it. So it joins no pool unless Policy names one. That is
consistent — we cannot know what allowance a credential we never see is
billed to.
