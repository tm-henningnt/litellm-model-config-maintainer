# An exhausted Route is demoted, not Excluded

A Route whose recorded quota exhaustion has not cleared stays
`available` and stops being `recommendable`. It remains in the
Generated Config and a caller can still reach it. `guidance` will not
answer with it, and a row whose every Route is Exhausted reports
`callable_now` as false.

This produces a combination that looks wrong at a glance and is not:
`available: true` beside `callable_now: false`.

## The harm it fixes

A Passthrough Auth Offering is never Excluded on a quota exhaustion.
The credential is the caller's, so the exhaustion belongs to that
caller and not to the Offering (CONTEXT.md, "Passthrough Auth"). All
seven of the operator's Claude Aliases are Passthrough Auth.

`best_route` was "the first available Route", and `available` means
only "not Excluded". So when the Claude weekly quota ran out, the hook
recorded `quota_exhausted` with a reset time, the record kept
`excluded=False`, and the row reported `callable_now: true`. The
`model-routing` skill tells an agent, in as many words, to take the
first row whose `callable_now` is true. Every field needed to know
better — `reason`, `refills_at` — was already on the Route, and the one
field agents are told to trust ignored all of it.

## Why not simply Exclude it

Excluding trades a clear failure for a confusing one. The caller gets
"model not found" from the proxy instead of the provider's own "your
quota resets at 09:00", and a caller whose own allowance is intact
loses an Offering that would have answered for them.

Worse, it can wedge. Nothing clears an exclusion on a Passthrough Auth
Offering except the clock: the Prober cannot test one, and the
Observation Journal records only failures, so no success is ever
observed. An exhaustion stating no reset time would Exclude it forever.

Measurement is not available here, so the choice was between reporting
and guessing. Reporting is what ADR 0004 already says an Entitlement is
for.

## Consequences

`available` and `recommendable` now mean different things, and callers
must read the second. `as_dict` emits both, plus `exhausted`, so a
consumer can see why a Route was passed over.

An exhaustion stating no reset time expires after
`schedule.maximum_staleness_hours`. Nothing else could ever clear it,
and hiding a working Offering forever is worse than recommending a
possibly-exhausted one a day later.

Demotion is per Route, so a row with a healthy sibling Route stays
callable and simply fails over — which is what the Route order is for.
