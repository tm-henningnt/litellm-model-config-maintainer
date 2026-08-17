# An Entitlement explains a failure; it never propagates one

Policy declares each provider's Entitlement as `shared_pool` or
`per_model`. A shared pool is tempting to reason forward from: if one
Offering reports `quota_exhausted` and the provider bills from one pool,
every sibling is presumably dry too. We do not draw that conclusion. The
declaration changes how a report reads. It never writes Health State,
and it never removes an Offering from the Generated Config.

## Why forward inference is wrong here

Two recorded cases defeat it.

A provider can refuse one tier and serve another. Gemini refused every
Pro-tier model for quota while every Flash, Gemma and Lite model kept
answering. `quota_exhausted` is therefore not an account-wide fact.

A pool can run dry and keep a free tier alive. ClinePass paid credits
were exhausted while `deepseek-v4-flash`, `laguna-s-2.1` and
`step-3.7-flash` still answered, from the same provider id. A propagated
guess would have removed the three working free routes at the exact
moment they became the only ones left.

The Feed cannot separate those pools for us. All eleven `cline-pass`
Offerings are `subscription_included`; the free ones look identical to
the paid ones. Any pool split would be a hand-maintained list, which is
the artifact this project exists to remove.

## What the declaration buys instead

Reporting the observed split, in the provider's own terms: "3 of 11
answering, 8 quota-exhausted, earliest refill 2026-07-27T00:00Z, still
answering: …". That is more useful to a caller than "exhausted", and it
cannot be wrong, because every part of it was measured. `shared_pool`
tells the reader why the eight failed together and whether to expect the
three to follow.

## Consequences

Learning a sibling's state costs a Probe. We accept that cost as the
price of never evicting a working Offering on a guess.

We now pay that cost promptly rather than whenever freshness happens to
expire. A quota exhaustion marks its pool mates `probe_due`, so the
next sweep measures them (`reduce._pool_siblings_to_mark`,
`entitlements.pool_siblings`). This propagates ATTENTION, not a verdict.
A marked sibling is not Excluded, stays in the Generated Config and
keeps serving; the mark only says "worth measuring". Both cases above
still come out right, because the Flash and free routes get probed,
answer, and stay.

Two things never propagate a mark. A Passthrough Auth Offering's quota
belongs to the calling client, not to our Entitlement, so it says
nothing about the pool; and a Passthrough Auth sibling is never marked,
because the Prober cannot probe one and the mark would never clear.

The resulting burst is bounded by the provider's own `pacing` rule.
Pool mates share one provider by definition, and `probe_offerings`
groups by provider and applies that provider's `concurrency` and
`minimum_interval_seconds` to the group.

The Entitlement view is derived on read, from Feed Document, Policy and
Health State. It is not a file and has no writer, so it cannot go stale
and cannot disagree with `status`.
