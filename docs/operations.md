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

## Map an Allowance to a Headroom source

No Headroom appears until you state the mapping. The tool infers none of
it. Do this once per Allowance.

**Verify the source against the provider's own dashboard first.** Nothing
here can catch a source that was wrong from the start.

1.  List what the source measures. Run `codexbar --format json`.
2.  Read `providerID` and `accountEmail` on each entry. The source key is
    `codexbar:<providerID>/<accountEmail>`. Some providers publish no
    `accountEmail`; their key ends at the slash.
3.  Read the same provider's slot labels. Run `codexbar --provider <id>`.
    The JSON never labels the three slots; the text output does.
4.  Decide the shape from those labels. `Session`, `5-hour`, `Weekly` and
    `Monthly` name nested time windows, so a plain string is correct.
    `Pro` and `Flash` name models, so state a `windows` mapping instead.
5.  Find the Allowance id. Run
    `litellm-maintainer entitlements --json | jq -r '.entitlements[].allowance_id'`.
6.  Open the provider's dashboard. Compare its figure against the
    source's. Do not continue while the two disagree, unless you
    establish which is right.
7.  Write the entry under `headroom.sources` in your Policy, keyed on the
    Allowance id. Copy the shapes from `policy.example.yaml`.
8.  State the Tier under `allowances`, on the same key.
9.  Check the mapping. Run
    `litellm-maintainer doctor --policy $LITELLM_MAINTAINER_HOME/policy.yaml --env <env file>`
    and read every `headroom.*` check.
10. Take a first Reading. Run
    `litellm-maintainer headroom refresh --policy $LITELLM_MAINTAINER_HOME/policy.yaml --env <env file>`.
11. Read the result back with `litellm-maintainer entitlements --json`.

A provider that caps one model inside a wider window needs a `members`
entry too. Claude's Fable window is that case. Read the `members` comments
in `policy.example.yaml` before you write one.

## Install the headroom-refresh job

You mapped an Allowance to a codexbar source in `headroom.sources`, and
you want Headroom State to stay current on its own.

```
litellm-maintainer headroom install --policy $LITELLM_MAINTAINER_HOME/policy.yaml --target-dir ~/Library/LaunchAgents
```

This writes a SECOND launchd job, separate from the scheduled tick. It
runs `headroom refresh` on Policy's `headroom.interval_minutes`, 15
minutes by default. It never runs `run`, and it never blocks the tick
or the Observation Journal watcher: `headroom refresh` takes Headroom
State's own lock, never the maintainer lock.

The command prints the file it wrote and a `launchctl` command. Run
that command yourself to register the job:

```
launchctl load -w ~/Library/LaunchAgents/no.tallmaker.litellm-maintainer.headroom-refresh.plist
```

Pass `--env` with the absolute path to your credential file. Without
it, codexbar may resolve no credential, since launchd runs a job from
`/`.

When `headroom.sources` is empty, `install` writes no job. If a job
from an earlier Policy is still on disk, it says so and names the file,
rather than leaving an orphaned job unmentioned.

Verify the job runs by reading Headroom State's own timestamp after a
few intervals:

```
cat $LITELLM_MAINTAINER_HOME/state/headroom.json
```

`read_at` on each record should advance between reads, with no command
run by hand.

To remove the job:

```
litellm-maintainer headroom uninstall --target-dir ~/Library/LaunchAgents
```

This prints the `launchctl unload` command first, then removes the
plist file. Run the printed command yourself; `uninstall` never calls
`launchctl`.

## Repair a scheduled job that stopped

`runs.log` stops gaining lines, or `launchctl list` shows a non-zero exit.

Read the real state first. `launchctl list` reports the last exit code,
not whether the job still runs:

```
launchctl print "gui/$(id -u)/no.tallmaker.litellm-maintainer.tick"
```

`state = spawn scheduled` beside `last exit code = 78: EX_CONFIG` means
launchd parked the job. It stops respawning a job that exits `EX_CONFIG`,
so correcting the fault does not restore the schedule.

Reload it:

```
launchctl unload ~/Library/LaunchAgents/no.tallmaker.litellm-maintainer.tick.plist
launchctl load -w ~/Library/LaunchAgents/no.tallmaker.litellm-maintainer.tick.plist
```

Confirm within one interval:

```
launchctl list | grep tallmaker
tail -2 $LITELLM_MAINTAINER_HOME/state/runs.log
```

A line reading `did not start:` names an import that failed and the
exception behind it. Fix that, then reload again — the line says so,
because fixing alone is not enough.

## Fix a rotted Headroom mapping

`doctor` failed a `headroom.*` check, or `guidance`/`entitlements` printed
a warning that a mapped Allowance's Headroom stopped refreshing.

```
litellm-maintainer doctor --policy $LITELLM_MAINTAINER_HOME/policy.yaml --env .env.local
```

Always pass `--env`. Without it every credential reports missing, and
the real findings hide among the false ones.

Read the failing check's `remedy` line. It names the exact fix:

- `headroom.mapped.<allowance_id>` FAILED, "matches no Reading": codexbar
  renamed the provider or the account logged out. Run
  `codexbar --format json` by hand, read the identity it states now, and
  correct `headroom.sources.<allowance_id>` in Policy.
- `headroom.mapped.<allowance_id>` FAILED, "matches N Readings": a second
  account appeared and the key no longer discriminates it from the
  first. Name the account explicitly:
  `headroom.sources.<allowance_id>: "codexbar:<providerID>/<accountEmail>"`.
- `headroom.binary` FAILED: install the binary named in
  `headroom.command`, or correct that field.
- `headroom.window.<allowance_id>.<slot>` FAILED: codexbar stopped
  publishing a slot named in `headroom.sources.<allowance_id>.windows`.
  Run `codexbar --provider <id> --format json`, confirm the slot still
  exists, and correct the mapping.
- `headroom.member.unclaimed.<allowance_id>` FAILED: an admitted Health
  Key on this Allowance answers, but no `members` entry claims it. Run
  `codexbar --provider <id>`, read the labels, and add the Health Key to
  `headroom.sources.<allowance_id>.members` under the slot it measures.
- `headroom.member.empty.<allowance_id>.<slot>` FAILED: a declared slot
  names no Health Key at all. Add one under
  `headroom.sources.<allowance_id>.members.<slot>`.
- `headroom.member.unknown.<allowance_id>.<slot>.<health_key>` FAILED: the
  named Health Key matches no Offering the Feed publishes and no Declared
  Offering's Alias — a typo, or a model the Feed dropped. Correct or
  remove it.
- `headroom.all_accounts.unreachable.<provider_id>` FAILED:
  `headroom.all_accounts_providers` names a provider no `headroom.sources`
  entry reaches. Map a source to it, or remove it from
  `all_accounts_providers`.
- `headroom.all_accounts.unmarked.<provider_id>` FAILED: two
  `headroom.sources` entries share a `providerID`, and that provider is
  not named in `all_accounts_providers`. Add it — see "Map a provider
  with two accounts" below.
- `headroom.refresh_interval` FAILED: you edited
  `headroom.interval_minutes` and did not re-install the job. Run:

```
litellm-maintainer headroom install --policy $LITELLM_MAINTAINER_HOME/policy.yaml --target-dir ~/Library/LaunchAgents
```

Then reload the job with the `launchctl` command `install` prints.

`doctor` reports the same fault as `headroom.refresh_current`. It renders
`headroom_source_warnings`, the function `guidance` and `entitlements`
already publish, so the two cannot disagree. Measured 2026-07-29: the job
was written to disk and never registered, Headroom State sat 4.9 hours
stale, every figure kept publishing, and `doctor` exited 0. Writing the
plist is not installing it — run the `launchctl load` command the install
prints, then confirm:

```
launchctl list | grep headroom-refresh
```

A `guidance` or `entitlements` warning naming a stale Headroom, with no
matching `doctor` failure, points at the refresh job itself rather than
the mapping: check that `no.tallmaker.litellm-maintainer.headroom-refresh`
is loaded (`launchctl list | grep headroom-refresh`), and read
`state/headroom-refresh.err.log` for what it last reported.

A warning naming an allowance that has "never been refreshed" is
ordinary right after you first declare it: give the job one interval to
run, then check again.

## Map a provider with two accounts

You hold two subscriptions with one codexbar provider — two ChatGPT
seats are the operator's own case. `codexbar --provider codex` alone
returns one Reading, so `headroom.sources` cannot join the second
account to its Allowance.

Find each account's id:

```
codexbar --provider codex --all-accounts --format json
```

Read `identity.accountEmail` in each entry. That is the value that goes
after the slash in `headroom.sources`.

State the provider once, in `all_accounts_providers`, and map each
account under its own Allowance:

```yaml
headroom:
  all_accounts_providers: ["codex"]
  sources:
    "credential:EXAMPLE_CHATGPT_SEAT1_WORKER_KEY": "codexbar:codex/one@example.com"
    "credential:EXAMPLE_CHATGPT_SEAT2_WORKER_KEY": "codexbar:codex/two@example.com"
```

Name the provider id once. Two `sources` entries billed to it would
otherwise carry the same flag twice, with nothing to say which copy is
authoritative.

Never add a provider to `all_accounts_providers` to find out whether it
has two accounts. State it only once you have read two accounts back
from the command above. A plain call for a provider with one account
mapped there costs one extra `--all-accounts` call every refresh for no
reason.

## Find out why a model is missing

You expected an Alias and it is not there, or a caller reports it does
not work.

```
litellm-maintainer explain <offering-id-or-alias>
```

`explain` walks the whole path — Feed, Policy, Health State, Generated
Config, and the running proxy — and names the stage that stopped it. It
takes an Offering id or an Alias.

Read the stop's KIND before you act on it:

- **Decision** — Policy or the Feed stopped it and nothing is broken.
  The line names the construct responsible, such as
  `providers.<id>.pricing`. Change that line, or agree with it.
- **Fault** — something is stale or broken. Repair it.

A stage after the stop reports `unknown`. So does the proxy stage when
the proxy cannot be reached: an absent answer is not a negative answer.

Pass `--no-proxy` to skip the live check. Pass `--json` for a machine
reader.

## Apply a change now

You fixed something and you do not want to wait for the tick, whose
interval is an hour.

Warning: this restarts the proxy and ends every session in flight.
Choose the moment — between waves of work, never inside one.

```
litellm-maintainer deploy --env <env file>
```

`deploy` generates and writes, ignoring the schedule's interval. It runs
the safety rail, and refuses on a rail failure unless you pass `--force`.
It takes the maintainer lock, so it cannot run beside a tick.

It resolves the file the proxy reads from the installed tick job's own
`--out`, so pass no path. Where no job is installed it refuses rather
than guessing: a guess writes the right config where nothing reads it,
which looks exactly like a deploy that did nothing.

An unchanged config is not written, and it says so.

Note that `probe` alone never writes a config. Clearing a Health State
record and deploying it are two steps.

## Read status

You want to see what the Generated Config currently offers, and why
anything is missing from it.

```
litellm-maintainer status --feed $LITELLM_MAINTAINER_HOME/feed.json --policy $LITELLM_MAINTAINER_HOME/policy.yaml
```

`status` names every Offered, Excluded, Unlisted, Withheld, Sunsetting,
and Candidate Offering, plus any Withheld entry the Feed no longer
publishes, and every Alias the proxy last refused as one it does not
serve.

Read Excluded and Unlisted as different states. An **Excluded** Offering
is still in the config and a caller can still reach it; it is not
recommended. An **Unlisted** one is absent from the file. Only Withheld
and Gone Unlist an Offering — see
[ADR 0014](./adr/0014-a-measurement-never-restarts-the-proxy.md).

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

Pin fastapi on every install. litellm imports a name fastapi removed in
0.140.7, and litellm's own version range still admits it:

```
uv tool install "litellm[proxy]==1.97.0" --with "fastapi==0.140.6" --force
```

See `docs/gotchas.md`, "litellm needs a fastapi its own version range
admits but breaks on".

`doctor` finds the tree from the `litellm` executable on `PATH`. Name it
directly when the proxy runs from somewhere else, AND whenever you run
`doctor` from this repository:

```
litellm-maintainer doctor --litellm-path /path/to/site-packages/litellm
```

A check that cannot read its file passes, and says so. Only a file read
without the marker fails.

Two false alarms are possible.

litellm fixing a defect upstream also removes the marker, so a `[FAIL]`
means "the patched behaviour is not proven present", not "litellm is
broken". Read `docs/gotchas.md`, decide which case you are in, then
either re-apply the patch or remove the entry from `REQUIRED_PATCHES` in
`litellm_maintainer/litellm_patches.py`.

A litellm in the repository's own `.venv` shadows the proxy's tree on
`PATH`, so `doctor` reads a stock copy and reports both patches missing.
Read the path in the failure first: a path under the project directory
is this case. See `docs/gotchas.md`, "doctor reads the litellm on PATH,
and a project venv shadows it".
