# Operations

This page lists the jobs you repeat once the tool runs on a schedule.
Each section states what the situation looks like, then the exact
command.

Editing `policy.yaml` by hand stays fully supported. The Operator
Surface (`litellm-maintainer policy <verb>`) exists so a one-line
decision does not need an editor. Every verb takes the lock, refuses a
write when Policy changed on disk since it read the file, and prints
the diff it applied. Your comments and key order survive either way.

Add `--dry-run` to any `policy` verb to see its diff without writing
anything.

## Approve a Candidate

`status` or a run's own report names a Candidate: a Discovered Offering
that clears your structural filters but carries no quality score. It is
never added on its own.

```
litellm-maintainer policy approve-candidate <offering-id> --policy $LITELLM_MAINTAINER_HOME/policy.yaml
```

`<offering-id>` has the form `<provider>:<provider_model_id>`.

## Withhold an Offering

You want to stop using an Offering for a reason the Feed cannot know:
unclear billing, a subscription ending. State the reason; a bare id ages
badly.

```
litellm-maintainer policy withhold <offering-id> --reason "billing terms unclear" --policy $LITELLM_MAINTAINER_HOME/policy.yaml
```

A Withheld Offering is never probed. Only you clear it.

## Unwithhold an Offering

The reason above no longer applies.

```
litellm-maintainer policy unwithhold <offering-id> --policy $LITELLM_MAINTAINER_HOME/policy.yaml
```

## Pin an Alias

An upstream identifier would otherwise derive an ugly Alias, or a client
already depends on a name you must keep.

```
litellm-maintainer policy set-alias <offering-id> <alias> --policy $LITELLM_MAINTAINER_HOME/policy.yaml
```

## Declare an Entitlement as shared_pool

A provider bills every Offering from one shared pool rather than per
model. Declaring this changes only how `entitlements` and `guidance`
report that provider. It never excludes an Offering and never marks a
sibling Offering unusable; see
[ADR 0004](./adr/0004-entitlement-signals-never-propagate.md).

```
litellm-maintainer policy set-entitlement <provider-id> shared_pool --policy $LITELLM_MAINTAINER_HOME/policy.yaml
```

`per_model` is the default. Pass `per_model` here to return a provider to
it.

## Force a Probe when a plan refills early

Health State recorded a reset time, but the provider refilled before
that time passed. Waiting for the clock would leave a working Offering
Excluded for no reason.

```
litellm-maintainer probe --feed $LITELLM_MAINTAINER_HOME/feed.json --policy $LITELLM_MAINTAINER_HOME/policy.yaml --force
```

Add `--provider <provider-id>` to force only one provider.

## Read status

You want to see what the Generated Config currently offers, and why
anything is missing from it.

```
litellm-maintainer status --feed $LITELLM_MAINTAINER_HOME/feed.json --policy $LITELLM_MAINTAINER_HOME/policy.yaml
```

`status` names every Offered, Excluded, Withheld, Sunsetting, and
Candidate Offering, plus any Withheld entry the Feed no longer publishes.

## Read entitlements

You want to know what you can spend through right now, provider by
provider.

```
litellm-maintainer entitlements --feed $LITELLM_MAINTAINER_HOME/feed.json --policy $LITELLM_MAINTAINER_HOME/policy.yaml
```

`entitlements` states each provider's Entitlement kind, its cost, how
many of its Offerings answer right now, and the earliest refill time for
one that does not. It reports no balance and no remaining credit; see
[ADR 0005](./adr/0005-guidance-reports-what-we-measured.md) for why.

## Roll back a bad config

`generate` or a scheduled `run` produced a Generated Config you do not
want live.

```
litellm-maintainer rollback --home $LITELLM_MAINTAINER_HOME
```

`rollback` restores the most recent snapshot kept under
`safety.snapshot_count`. Restart or reload the proxy afterward so it
reads the restored file.

## When doctor fails a check

```
litellm-maintainer doctor --policy $LITELLM_MAINTAINER_HOME/policy.yaml
```

Read each `[FAIL]` line. Every one names the exact command that fixes
it: set a named credential variable, run `fetch` to refresh a stale Feed
Document, run `probe` to populate an empty Health State, or remove a
Withheld entry the Feed no longer publishes. Fix the checks in the order
`doctor` prints them; a later check can depend on an earlier one.

### The local litellm patches

Two checks read the litellm the proxy runs, not this package's own:

```
[OK] litellm_patch.chatgpt_stream: ... carries the patch.
[OK] litellm_patch.usage_only_chunk: ... carries the patch.
```

Both name a defect in litellm's own transform layer. No config setting
or callback reaches either one, so each is a patch to the installed
litellm. `docs/gotchas.md` records the reasoning and the exact edit.

Warning: `uv tool upgrade litellm` replaces that tree and removes both
edits. Nothing else reports the loss. The Generated Config does not
change, the proxy starts, and `/v1/models` still lists every Alias; the
models stop answering. Run `doctor` after every litellm upgrade.

`doctor` finds the tree from the `litellm` executable on `PATH`. Name it
directly when the proxy runs from somewhere else:

```
litellm-maintainer doctor --litellm-path /path/to/site-packages/litellm
```

A check that cannot read its file passes, and says so. Only a file read
without the marker fails.

One false alarm is possible. litellm fixing a defect upstream also
removes the marker, so a `[FAIL]` means "the patched behaviour is not
proven present", not "litellm is broken". Read `docs/gotchas.md`, decide
which case you are in, then either re-apply the patch or remove the
entry from `REQUIRED_PATCHES` in `litellm_maintainer/litellm_patches.py`.
