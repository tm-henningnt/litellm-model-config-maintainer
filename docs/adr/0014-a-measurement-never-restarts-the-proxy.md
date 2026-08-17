# A measurement never restarts the proxy

An Offering the machine observed to be unusable stays in Generated
Config. It becomes Excluded, so `guidance` stops recommending it, and a
caller can still reach it. Only two facts remove an Offering from the
file: Withheld and Gone. That state is called Unlisted.

This generalises ADR 0010 from one cause to all of them. ADR 0010
demoted an Exhausted Route rather than Excluding it. The argument it
made for a quota exhaustion holds for every condition the Prober and the
Observation Journal record.

## The harm it fixes

Every write to Generated Config restarts the proxy, because that file is
the one the litellm `--reload` watcher reads (`cli.py`, above
`rendered_config_is_unchanged`). A restart ends every session in flight.

Health State changes its mind often. Measured 2026-08-01 in
`state/tick.out.log`: 58 writes, with the Alias count oscillating 102,
100, 102, 100, 98. Each step is one restart. On the night of 2026-07-31 a
single Offering, `qwencloud-token-plan:qwen3.8-max-preview`, produced
four writes in twenty minutes as one transient authentication failure
Excluded it and the next Probe restored it.

So the operator's sessions were being killed by a Health State
disagreeing with itself, about one Offering out of a hundred.

## The rule

Generated Config may be driven only by facts that are deliberate or
terminal. It must never be driven by a measurement that oscillates.

| Cause | Oscillates | Cost of ignoring the advice | Result |
| --- | --- | --- | --- |
| Health exclusion | yes | an error | Excluded, still listed |
| Withheld | no | money | Unlisted |
| Gone | no | an error, permanently | Unlisted |

## Why Withheld still Unlists

Advice is enough where ignoring it costs an error. It is never enough
where ignoring it costs money.

A Withheld line such as "subscription ending, renewal unconfirmed"
exists to stop spend. An advisory cannot stop a worker from spending: a
dispatched agent reads `guidance` at selection and then calls the Alias
directly. Removing the Alias is the only control that binds. The same
reasoning admits Gone, for a different reason — it never recovers, so
listing it helps nobody, and it cannot flap.

## What this costs

Generated Config becomes a superset of what answers. A caller that
ignores `guidance` now meets a broken Route more often than before. That
burden lands on three readers: a human calling the proxy by hand, a
client serving a cached `/v1/models` list, and any agent that selects an
Alias without reading `recommendable`.

This is a deliberate transfer. ADR 0010 already accepted it for one
cause; this extends it. The compensation is that the caller reads the
provider's own message — "your quota resets at 09:00", or an
authentication error naming the vendor — instead of the proxy's "model
not found", which names nothing and invites a retry.

`available` and `recommendable` now differ for many more Routes than
before. A caller that reads only `available` was already wrong under ADR
0010. It is now wrong more often.

## What was rejected

**Keep Excluding, and make the proxy reload without a restart.** This is
the real fix and it is not ours to make. It waits on litellm serving a
new config without dropping connections. Recorded here so the decision
can be revisited when that lands.

**Debounce the writes.** Hold an exclusion for N minutes before writing,
so a flap costs one write instead of four. It reduces the restarts
without removing them, it adds a timer to a system whose transforms are
pure, and it still restarts the proxy for a condition that a caller
could simply have been told about.

**Unlist on the operator's command only.** Drop automatic removal
entirely, including Withheld. Rejected because it makes the money case
unenforceable, which is the one case advice cannot cover.

## Consequences

`plan` stops reading `record.excluded` when it builds the offered set.
It keeps reading Withheld and Gone. Two call sites change and the
transform stays pure.

Health State is unchanged. `reduce` still records every exclusion, and
`guidance` still reads it to set `recommendable`. Only the Generator's
use of it changes, which is the seam that keeps this small.

Writes now come from Policy edits, Feed changes and Gone. All three are
operator-paced or slow, so a restart becomes an event with a cause the
operator can name.

Nothing reaps the file. An Offering that breaks permanently without ever
classifying Gone stays listed. `Gone` is reachable but has fired zero
times to date, so a reaper is not built yet. Build one when the first
such Offering appears, not before.
