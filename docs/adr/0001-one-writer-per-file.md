# Every state file has exactly one writer

The operator's intent, the machine's observations, and the litellm config
all change on different schedules and for different reasons. We split
them into five files and gave each exactly one writer: Policy is written
only by a human, the Observation Journal only by the proxy, Health State
only by the maintainer, Generated Config only by the Generator, and the
previous-run record only by the code that calls
`notify.write_previous_run_state`. No process ever writes a file another
process also writes, so no write can clobber another.

The last clause of that rule needed narrowing. Three processes can act
in the maintainer role at once, so Health State does need a lock. See
ADR 0002. The role partition below still holds.

## Considered options

**One file.** A single `policy.yaml` holding both the operator's choices
and a machine-updated `health:` section. Rejected: the tools would fight
the human. Re-enable a provider by hand and the next quota check
disables it again, or worse, overwrites the edit mid-save.

**Two files, health inside the generated config.** Store health as
`model_info` on each entry and read it back on the next run. Rejected:
an Offering excluded for quota has no entry to carry its own state, so
the config would have to keep advertising models it cannot serve just to
remember why.

**Let the proxy callback write Health State directly, under a lock.**
Rejected: it puts a lock acquisition inside the request path, a killed
proxy can leave a stale lock, and it discards history — you would see
the current state but not that something failed six times today.

## Consequences

The proxy cannot update Health State itself; it appends observation
events and the maintainer reduces them. That indirection is the price of
the invariant, and it buys failure history for free.

The Prober's worklist must come from Policy rather than from Generated
Config. An Excluded Offering is absent from the config, so anything
probing *through* the proxy could never discover that it recovered.
Auto-re-enable depends on this.

State files must not live where the proxy's `--reload` watcher can see
them. It watches `*.py`, `.env`, and the config basename, so keeping
state in `*.json` / `*.jsonl` outside the config directory avoids a
reload loop — the proxy writes the Journal itself, and a watched Journal
would restart the proxy on every recorded failure.
