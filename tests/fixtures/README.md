# Why the Feed fixtures are pinned

`feed-audited.json` and `feed-current.json` are frozen copies, not live
fetches. The live Feed has changed since the audit: offering counts,
capability flags and availability values all moved between the two
snapshots and moved again after.

A test that fetched the Feed live would fail for reasons that have
nothing to do with this code. It would report a drift in the Feed's own
data, not a bug in `plan`, `classify` or `reduce`.

Pinning these two snapshots gives the acceptance test in ticket 10 a
fixed target: the audited snapshot reproduces the operator's config, and
the current snapshot reproduces the known, already-documented
differences. See `.scratch/maintainer-v1/spec.md`, "Seam 3: plan".

Nothing may overwrite either snapshot.
