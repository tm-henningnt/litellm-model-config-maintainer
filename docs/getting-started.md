# Getting started

This page takes you from a clean checkout to a running proxy with a
schedule. Follow the steps in order. Each step depends on the one before
it.

## 1. Install the tool

Clone this repository, then install it in editable mode:

```
pip install -e .
```

This registers the `litellm-maintainer` console script. Every command
below uses that script. If your shell cannot find it, run
`.venv/bin/python -m litellm_maintainer.cli` instead, from the repository
root.

The tool needs `pyyaml` and `httpx`, and nothing else. It reads JSON and
YAML and writes files; it never calls a model. To run the tests as well,
install the dev extra:

```
pip install -e '.[dev]'
```

That adds `pytest` and `litellm`. The tests need litellm to import the
three modules under `providers/`, which run inside the proxy. The tool
never imports them: `deploy` copies them as bytes.

## 2. Install the litellm proxy

The maintainer writes a config; it does not run a proxy. Install litellm
with its proxy extra:

```
pip install 'litellm[proxy]'
```

## 3. Set your instance directory

The maintainer keeps your Policy, your Feed Document, your Health State,
and your Generated Config in one directory. Set it:

```
export LITELLM_MAINTAINER_HOME="$HOME/.config/litellm-maintainer"
```

This is also the default, so the variable is optional. Set it anyway if
you run more than one instance, or if you keep your data outside your
home directory. None of your data lives inside this repository, so a
careless `git add` in the repository cannot publish it.

Create the directory:

```
mkdir -p "$LITELLM_MAINTAINER_HOME"
```

## 4. Obtain a Feed Document

If another process already produces a Feed Document for you, copy it to
`$LITELLM_MAINTAINER_HOME/feed.json` and skip to step 5.

Otherwise, download it yourself with `fetch`. You have no Policy yet, so
name the address on the command line:

```
litellm-maintainer fetch --url https://example.invalid/feed.json
```

If your Feed needs a credential, set an environment variable and name it
with `--credential-env`. The command never takes the token itself:

```
export FEED_TOKEN="your-feed-credential"
litellm-maintainer fetch --url https://example.invalid/feed.json --credential-env FEED_TOKEN
```

After step 5 writes your Policy, move the address into it, so the
scheduled run refreshes the Feed for you:

```yaml
feed:
  url: "https://example.invalid/feed.json"
  credential_env: "FEED_TOKEN"
  maximum_age_hours: 24
```

Later fetches then need no flags:

```
litellm-maintainer fetch --policy $LITELLM_MAINTAINER_HOME/policy.yaml
```

`fetch` writes `$LITELLM_MAINTAINER_HOME/feed.json` only after the
download parses and carries a plausible number of Offerings. A failed
download leaves any earlier Feed Document in place and reports the
failure; it never blanks your Feed.

## 5. Write a starter Policy

Run `init`. It reads the Feed Document and writes a Policy naming every
provider the Feed publishes, with the environment variable each
provider's credential comes from:

```
litellm-maintainer init --feed $LITELLM_MAINTAINER_HOME/feed.json
```

This writes `$LITELLM_MAINTAINER_HOME/policy.yaml`. Open it and read the
comments. `init` never writes a credential into Policy.

## 6. Set your credential variables

Run `doctor`. It names the credential variable each provider in your
Policy needs:

```
litellm-maintainer doctor --policy $LITELLM_MAINTAINER_HOME/policy.yaml
```

Set every variable it names, for example `export OPENROUTER_API_KEY=...`.
Some checks still fail at this point. That is expected: you have not yet
probed anything or generated a config. Later steps clear those checks.

## 7. Validate the Policy

```
litellm-maintainer validate --policy $LITELLM_MAINTAINER_HOME/policy.yaml
```

`validate` reports every Policy section it parsed. Fix any error before
you continue.

## 8. Run a first, cheap Probe

Probe one provider first, to keep the first live sweep cheap:

```
litellm-maintainer probe --feed $LITELLM_MAINTAINER_HOME/feed.json --policy $LITELLM_MAINTAINER_HOME/policy.yaml --provider <one-provider-id>
```

Replace `<one-provider-id>` with a provider id from your Policy. Add
`--dry-run` first if you want to see the worklist before any call runs.

## 9. Generate the config

```
litellm-maintainer generate --feed $LITELLM_MAINTAINER_HOME/feed.json --policy $LITELLM_MAINTAINER_HOME/policy.yaml
```

This writes `$LITELLM_MAINTAINER_HOME/config.yaml`, the Generated
Config. An Offering appears in it only when Policy admits it and Health
State does not exclude it.

## 10. Point the proxy at the Generated Config

Start litellm with that file:

```
litellm --config $LITELLM_MAINTAINER_HOME/config.yaml
```

## 11. Install the failure callback

The maintainer learns from real traffic through the Observation Journal.
`providers/journal_failure_callback.py` writes that Journal.

Warning: an unregistered callback is silent. The proxy serves every
request normally and the Journal stays empty forever. Nothing else can
tell "no failures happened" from "no failures were recorded". Step 14
checks this for you.

Copy the file next to your `config.yaml`. Then register it in **Policy**,
not in the config:

```yaml
proxy_settings:
  litellm_settings:
    callbacks:
      - journal_failure_callback.observation_journal_callback
```

Policy is the right place because the Generator overwrites the Generated
Config on every run. An edit made directly to `config.yaml` is lost.
Run `generate` again after this edit.

Register it on the **main proxy only**. Do not register it on a worker
proxy. A worker knows an Offering under its own name, which carries no
seat identity: `claude-chatgpt1-gpt-5.6-sol` and
`claude-chatgpt2-gpt-5.6-sol` both reach a worker that calls the model
`claude-gpt-5.6-sol`. An entry written there names a key Health State
does not hold, and it could not tell an exhausted seat from a healthy
one. The main proxy is an ordinary client of the worker, so it records
the seat-qualified Alias by itself when the worker fails.

Set `LITELLM_MAINTAINER_HOME` in the proxy's own process environment if
your instance directory is not the default. Read the module's own
docstring for the full detail; this callback never raises into the
proxy and never writes Health State.

## 12. Check the running proxy

```
litellm-maintainer smoke --feed $LITELLM_MAINTAINER_HOME/feed.json --policy $LITELLM_MAINTAINER_HOME/policy.yaml
```

`smoke` checks one Offering per distinct translation rule, through the
proxy you just started. Add `--dry-run` first if you want to see which
rule it would check before any call runs.

## 13. Install the schedule

Nothing in this project runs on its own until you finish this step. A
fully configured instance, a registered callback and a growing Journal
still change nothing without it. Health State only moves when you run a
command by hand.

This step installs exactly one background job: a launchd tick that runs
`litellm-maintainer run` every 60 seconds. That tick is the only thing
this tool leaves running on your machine. It probes nothing on most
ticks: `schedule.interval_minutes` in Policy decides whether a tick does
real work, and the tick exits at once when it does not.

The tick is also what reads the Observation Journal. A failure the proxy
records makes the next tick due, whatever the interval says, and that
run confirms the ambiguous entries and probes nothing else.

Do not run `watch` as a service. It is a foreground debugging command
that does the same work. Two of them contend for the maintainer's lock,
and a daemon that dies stays dead with nothing to show for it.

On macOS, write the launchd job:

```
litellm-maintainer install --policy $LITELLM_MAINTAINER_HOME/policy.yaml --feed $LITELLM_MAINTAINER_HOME/feed.json
```

`install` writes the plist. It never calls `launchctl`. Run the command
it prints yourself, to register the job:

```
launchctl load ~/Library/LaunchAgents/no.tallmaker.litellm-maintainer.tick.plist
```

On any other platform, add a crontab line instead. This example ticks
every five minutes; the tool's own `schedule.interval_minutes` still
decides whether a given tick does anything:

```
*/5 * * * * /path/to/.venv/bin/litellm-maintainer run --policy $LITELLM_MAINTAINER_HOME/policy.yaml --feed $LITELLM_MAINTAINER_HOME/feed.json
```

## 14. Confirm the install

```
litellm-maintainer doctor --policy $LITELLM_MAINTAINER_HOME/policy.yaml
```

Every check should now pass. If one fails, `doctor` names the command
that fixes it.

Two checks cover the ways this system can be fully configured and still
do nothing:

- `journal.callback_registered[...]` — one per served proxy config.
  It fails only for the main proxy's config, which `doctor` recognises
  by the Generator's own header. A hand-written worker config is
  reported and never failed, because a worker records nothing by
  design.
- `schedule.tick_installed` — whether the launchd plist exists.
  Writing the plist and registering it are two steps; this check fails
  until the file is there.

To check the whole install, including what is actually running:

```
litellm-maintainer doctor --policy $LITELLM_MAINTAINER_HOME/policy.yaml
launchctl list | grep tallmaker
```

`doctor` reads `~/.config/litellm` for served configs and
`~/Library/LaunchAgents` for the tick. Pass `--served-config-dir` or
`--target-dir` when yours live elsewhere.

## What now

Read [operations.md](./operations.md) for the jobs you repeat: approving
a Candidate, withholding an Offering, rolling back a bad config. Read
[agent-guidance.md](./agent-guidance.md) if an agent dispatches work
through this proxy. Read [gotchas.md](./gotchas.md) before you add a
provider; it lists the provider and litellm traps this project already
paid for.
