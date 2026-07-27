# One writer at a time, not merely one writer

ADR 0001 gives each state file exactly one writing ROLE, and concludes
that no file needs locking. That conclusion is wrong for Health State,
because three processes now act in the maintainer role at once: the
launchd tick, the Journal watcher, and the operator running `probe` by
hand. Each reads Health State, folds in what it measured, and writes the
result back, so a concurrent update is silently lost. A maintainer
process therefore takes a lock in the instance directory before that
read-modify-write, and releases it after.

## Why this is worth a lock

The write is already atomic, so nothing corrupts. What is lost is an
update, and one particular loss matters.

The watcher records a quota failure with the reset time parsed from the
provider's message. The tick, which read Health State a second earlier,
writes its own copy over the top. The reset time is gone. That value is
what lets an Offering recover on the clock with no further call, which
the spec calls the thing that "makes recovery work for an Offering that
is expensive to call, heavily rate limited, or impossible to call at
all". Losing it converts a free recovery into an expensive one, and
leaves no symptom to notice.

## Considered options

**Accept the race.** Losses are rare and mostly self-correct on the next
Probe. Rejected for the reset-time case above: it does not self-correct,
because the recovery it enables is the one that needs no Probe.

**Stop the watcher running the pipeline.** Have it append a marker and
let the next tick act. This removes the concurrency outright. Rejected:
it gives up "a real quota failure acted on within seconds", which is the
point of watching the Journal at all.

**A lock inside the proxy callback.** Already rejected in ADR 0001, for
the same reasons: a lock in the request path, and a killed proxy leaves
it stale.

## Consequences

The rule to state is "one writer at a time per file", not "one writer".
ADR 0001's role partition still holds and is still the primary
invariant; this ADR narrows only its no-locking conclusion, and only for
Health State.

A lock can go stale if a maintainer process is killed. The lock records
the holding pid so a later process can detect and break a lock whose
owner is gone.

The tick exits in milliseconds when the gate says it is not due, and it
takes the lock only when it proceeds. So contention stays rare, and the
common case pays nothing.
