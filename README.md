# litellm-model-config-maintainer

Keeps a litellm proxy's `config.yaml` current. The tool reads a model
discovery feed, applies your policy, checks that the models answer, and
writes the config.

Status: implemented. The design is recorded in [CONTEXT.md](./CONTEXT.md)
and [docs/adr/](./docs/adr/). Start with
[docs/getting-started.md](./docs/getting-started.md).

## The problem

A litellm config can list models from many providers at once: OpenRouter,
Groq, Gemini, OpenCode Go, OpenCode Zen, Cline, ClinePass, a Qwen token
plan, plus direct vendor APIs. Each provider uses its own model
identifiers, base URL, and credential convention.

That config decays. Providers add free models and withdraw them. Vendors
deprecate identifiers. Plans exhaust their quota and refill. Some
endpoints return non-standard response shapes. Nobody wants to re-test
80 models by hand.

## Design

Six files. Each has exactly one writer, so no process can overwrite
another's work. See
[ADR 0001](./docs/adr/0001-one-writer-per-file.md).

| File | Writer | Holds |
| --- | --- | --- |
| `policy.yaml` | the operator, by hand or through the Operator Surface | selection rules, names, withheld models, schedule |
| `feed.json` | Fetch | the Feed's own published Offerings, as of one download |
| `state/observations.jsonl` | the proxy | failures seen in real traffic |
| `state/health.json` | the maintainer | current health per Offering |
| `config.yaml` | the Generator | the output litellm reads |
| Previous-run record | the code that ends a run | what the last run offered and reported, for "added" vs. "new" |

A model reaches the config when your policy admits it and its health
permits it.

Read [CONTEXT.md](./CONTEXT.md) for the vocabulary. The terms are
precise, and the code uses them.

### What it does

- **Selects** models per provider. Take a whole subscription pool. Take
  only free models from a large mirror. Name specific models elsewhere.
- **Translates** feed data into litellm parameters, with a verified
  per-provider rule table.
- **Probes** models directly, with a request rate set per provider, so a
  free tier does not report false failures.
- **Excludes** a model that fails, and restores it when it recovers. A
  quota that states its own reset time recovers on the clock, with no
  further calls.
- **Reports** what changed. It notifies you only when the offered models
  change or a decision needs you.
- **Refuses** to write an implausible result. It keeps snapshots.
- **Answers** what you can spend through right now, and which model to
  use for a task, via `entitlements` and `guidance`.
- **States each model's token limits**, from what the source states, so a
  caller reading `/v1/models` is told the real window rather than a guess.
  See [ADR 0006](./docs/adr/0006-a-stated-limit-comes-from-a-source.md).

### One feature you probably do not need

Some clients derive their context budget from the model **name** rather
than from the token limits the proxy reports. Such a client asks for a 1M
model, budgets 200,000 tokens, and compacts against a window it could have
used.

For those, the Generator can add a **Client-Facing Variant**: a second
Alias for the same model, carrying a suffix the client recognises. It sends
an identical request; only the name differs.

This is opt-in and off by default. Omit `client_facing_variants` from your
Policy and none is generated. **If your harness honours the proxy's
`model_info`, leave it out.** You need nothing here. The feature would only
double some of your Alias names.

Read [ADR 0007](./docs/adr/0007-an-alias-may-exist-for-the-clients-reading.md)
before you enable it. It records what was measured and on which client. It
also records the one real hazard: a wider budget is a claim about the
client, never about the provider.

## Your data stays out of this repository

The repository holds the tool and an example policy. Nothing else.

Your real policy, state, and generated config live in
`$LITELLM_MAINTAINER_HOME`, which defaults to
`~/.config/litellm-maintainer`. No path inside the repository holds your
data, so a careless `git add` cannot publish it.

## Read this first

[docs/gotchas.md](./docs/gotchas.md) lists the provider and litellm traps
we hit. Two of them cost a day each. Read it before you add a provider.

## Set it up with an agent

You can do the setup by hand from
[docs/getting-started.md](./docs/getting-started.md). You can also hand it
to a coding agent. Clone this repository, open an agent in it, and paste
the prompt below.

The prompt makes the agent read the design documents first. It then
checks what this machine already has, so a part-built or fully working
instance is not overwritten. It ends by installing the `model-routing`
skill and writing your routing rules.

An existing installation is the common case. The prompt handles it: the
agent reports what it found and asks where to start, and you can send it
straight to the routing rules.

````
Set up litellm-model-config-maintainer with me. Some of this may already
be done on this machine. Find out before you change anything.

Read these files before you do anything else:

- CONTEXT.md — the vocabulary. Use these terms exactly as defined.
- docs/getting-started.md — the install path. Follow its order.
- docs/gotchas.md — the provider traps. Two of them cost a day each.
- docs/agent-guidance.md — the `guidance` and `entitlements` commands.
- skills/model-routing/SKILL.md — the skill you will install at the end.

Step 0. Find out what already exists. Run these. They only read.

    command -v litellm-maintainer
    echo "${LITELLM_MAINTAINER_HOME:-$HOME/.config/litellm-maintainer}"
    ls "${LITELLM_MAINTAINER_HOME:-$HOME/.config/litellm-maintainer}"
    litellm-maintainer doctor
    ls ~/.claude/skills/model-routing .claude/skills/model-routing

`doctor` gives most of the answer in one command. It checks the Policy,
every credential, the Feed's age, whether the proxy answers, which
providers no Probe has reached, whether the failure callback is
registered, and whether the schedule tick is installed. Each failed
check names the command that fixes it. Read its output rather than
guessing from the file listing.

Then report, as a short table: which of steps 1 to 12 are already done,
which failed a check, and which are missing. Ask me where to start.

Never run a step whose result already exists. `init` and `generate` both
overwrite; running them on a working instance destroys real work. A
failed `doctor` check is a repair, not a reinstall — run the one command
that check names and nothing more.

If I tell you I only want the routing guidance, do step 0, then go
straight to step 13.

Otherwise walk me through the steps below, one per turn. Show me the
exact command, run it after I confirm, then show me its output before you
continue. Stop and ask whenever a step needs a decision only I can make:
my feed URL, which providers to admit, which credentials I hold.

1.  Install the tool and the litellm proxy. Confirm both run.
2.  Set LITELLM_MAINTAINER_HOME. Create the directory.
3.  Fetch a Feed Document. Ask me for the URL.
4.  Write a starter Policy with `init`. Read it back to me. Explain what
    it admits per provider before I keep it.
5.  Ask me which credential variables I hold. Write them to the env file
    the proxy sources. Never write a credential into this repository.
6.  Validate the Policy. Fix what it reports.
7.  Run `probe --dry-run` first and show me the worklist. Then probe one
    provider live with `--provider` so the first sweep stays cheap.
8.  Generate the config. Report what it admitted, what it withheld, and
    what awaits my approval.
9.  Point the proxy at the Generated Config. Start the proxy. Run
    `smoke`. A passing smoke check is the only proof the config works.
10. Install the failure callback, following getting-started step 11.
    Copy the callback file next to the served config. Register it in my
    Policy, never in the Generated Config, then regenerate. Restart the
    proxy and confirm the callback loaded. Register it on the main
    proxy only, never on a worker.
11. Install the schedule. Run `install`, which writes the launchd plist
    and never calls `launchctl`. Show me the `launchctl load` command it
    prints and let me run it. On a platform without launchd, give me the
    crontab line from getting-started step 13 instead.
12. Run `doctor` again. It must exit 0 before you call the setup done.
    Compare it against the step 0 run and tell me what changed.

Then install the skill and write my routing rules. Do this part even when
steps 1 to 12 were already done:

13. Ask me whether the skill belongs to every project
    (`~/.claude/skills/`) or only this one (`.claude/skills/`). Copy
    `skills/model-routing/` there. Say whether it replaced an existing
    copy, and tell me if the two differed.
14. Run `litellm-maintainer guidance --for coding --json` once yourself
    and read its `warnings` array. A "stale catalogue" warning means the
    rankings come from an old Feed. Tell me, and offer `fetch`, before you
    write anything that quotes a score.
15. Ask me where the routing rules belong: my global agent instructions,
    or this project's.
16. Write the rules. They must tell an agent to run
    `litellm-maintainer guidance --for <axis> --json` before it picks a
    model, to add `--prefer free` or `--prefer flat_rate` for bulk work,
    and to read each row's route list as a failover order. State that
    `--prefer` sorts rows and removes nothing: a failover walk can reach
    paid routes, so an agent that must not spend filters on `cost_basis`
    itself. Do not write a table of model names into the rules. A
    hardcoded table is the exact failure the skill exists to prevent.
17. Keep the rules short. Long instructions load into every session and
    cost context. Point at the skill; do not restate it.
18. If I ask for a snapshot to read myself, write one with
    `litellm-maintainer guidance --for coding --format markdown` to its
    own file. Put today's date and the word "snapshot" on its first line,
    so nobody mistakes it for live data.

Rules for you, throughout:

- Never write my policy, my state, or my credentials into this
  repository. They live in $LITELLM_MAINTAINER_HOME.
- Never run the `policy` write verbs yourself. Print the command and let
  me run it. Those are my decisions.
- Report every failure with its output. Do not tell me a step passed
  unless you saw it pass.
````

### The skill

[skills/model-routing/](./skills/model-routing/) holds the skill this
repository distributes. It teaches an agent to answer "which model, and
how do I reach it" from `guidance` and `entitlements` rather than from a
cached model list. Install it by copying the directory:

```
cp -R skills/model-routing ~/.claude/skills/     # every project
cp -R skills/model-routing .claude/skills/       # this project only
```

## Repository layout

```
policy.example.yaml         a worked example to copy
providers/                  custom litellm provider handlers
scripts/                    direct and proxy model checks
skills/model-routing/       the routing skill, for a coding agent
docs/getting-started.md     install and first run, one path
docs/operations.md          the jobs you repeat
docs/agent-guidance.md      guidance and entitlements, for an orchestrating agent
docs/gotchas.md             provider and litellm traps
docs/adr/                   architecture decisions
CONTEXT.md                  the vocabulary
```
