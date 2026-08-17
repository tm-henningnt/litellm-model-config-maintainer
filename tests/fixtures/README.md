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

## What each fixture is for

| File | May it be regenerated? |
| --- | --- |
| `feed-audited.json`, `feed-current.json` | No. Frozen Feed snapshots. |
| `expected-config.yaml` | **Never.** Read the next section. |
| `policy-pinned.yaml` | Yes, freely. |

## policy-pinned.yaml is the Policy every pinned test reads

It is synthetic. It names the public providers the committed Feed
carries, so the Discovery path is exercised, and it holds no operator
credential, host, subscription or seat. Every credential is an
`EXAMPLE_` name and the private host is `.invalid`.

No test may read the operator's own Policy at
`~/.config/litellm-maintainer/policy.yaml`. That file is private, it is
absent on every other machine, and every edit to it would break the
suite. One test reads the live proxy config, and it skips when that file
is absent.

Re-derive any count a test pins after you change this file. Each such
test states the command in its own docstring.

## expected-config.yaml must never be regenerated

It is the Generated Config the operator built and verified BY HAND on
2026-07-25, copied before the tool first wrote to that path. Its value
is that a human checked it.

A fixture the tool wrote proves only that the tool agrees with itself.

The tests that compared generated output to this file are retired. Their
input no longer exists: the Policy that produced this config was never
committed, no surviving copy reproduces it, and the live Policy matches
none of its 78 Aliases because it now sets `alias_prefix: ""` and
`alias_separator: "--"`. Twenty-two of the frozen Aliases name decisions
the operator took deliberately, such as the six direct ChatGPT entries
retired on 2026-07-26 when the seat workers replaced them.

The file stays here as the record of a migration that finished. Read
the note at the top of `test_acceptance.py` before you act on it.
