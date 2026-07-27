# litellm Model Config Maintainer

Keeps a litellm proxy's `config.yaml` current: which models the proxy
offers, under which names, sourced from a model discovery feed plus the
operator's own judgment.

## Language

### Sources and artifacts

**Feed**:
The external model discovery service that publishes providers and their
offerings. Authoritative for facts, never for our choices.
_Avoid_: catalog, registry, index

**Offering**:
One provider-specific way to call a model. Every Offering is either
Discovered or Declared.
_Avoid_: model (ambiguous — see Canonical Model)

**Discovered Offering**:
An Offering the Feed publishes, identified as
`<provider>:<provider_model_id>`.

**Declared Offering**:
An Offering the operator writes into Policy, for a provider the Feed
does not cover. Passed through to Generated Config verbatim, so it
cannot be affected by Feed changes.
_Avoid_: manual model, local offering ("local" means locally-hosted in
the Feed's own vocabulary)

**Canonical Model**:
The underlying model an Offering exposes. One Canonical Model can reach
us through several Offerings — `glm-5.2` arrives via OpenCode Go,
ClinePass, and the Qwen Token Plan.

**Entitlement**:
The operator's spending relationship with one allowance. It states
whether that allowance is one shared pool or per-model limits, what a
call costs us, and what our own Health State currently says about the
Offerings billed to it. A provider is who serves the model; an
Entitlement is what using them costs us. Policy declares the kind as
`shared_pool` or `per_model`, and `per_model` is the default, because
it never over-claims. The kind explains an observation. It never
propagates one — see ADR 0004.

**The credential identifies the Entitlement**, not the provider,
because the credential is what gets billed. Two ChatGPT seats behind
one `openai/` prefix are two Entitlements, separated by their two
credentials with nothing to configure. A provider is the usual case
rather than the rule: the Feed states one credential per provider, so a
Discovered Offering's Entitlement and its provider coincide. See ADR
0009. Policy may name a pool explicitly when the credential misleads.

An Entitlement has no state and is never turned off. Only an Offering
is Withheld or Excluded. A `shared_pool` exhaustion makes its pool
mates worth measuring; it does not decide anything about them.
_Avoid_: lane, quota, budget, plan, provider (the provider serves; the
Entitlement bills). Never write "disable an Entitlement" or "an
Entitlement is down": there is no such state. Name what actually
happened to which Offering.

**Passthrough Auth**:
An Offering whose credentials come from the calling client rather than
from the proxy. Two things follow. The Prober cannot test it, because a
probe carries the wrong credentials. And its quota and authentication
failures belong to one caller, not to the Offering, so they are recorded
and reported but never Exclude it. Other failure kinds still do.
_Avoid_: unprobeable (that is a consequence, not the cause)

**Alias**:
The name a client asks the proxy for; litellm's `model_name`. Ours are
prefixed `claude-`. Distinct from the Offering's `provider_model_id`.
_Avoid_: model name, alias name

**Stated Limit**:
The token figures a source states for an Offering: its context window
and its maximum output. Written to Generated Config's `model_info`, and
never derived from a model name. The Feed states one for a Discovered
Offering; the operator states one for a Declared Offering. An Offering
no source describes carries none, and absence reads as unknown rather
than as small — see ADR 0006.
_Avoid_: token limit, context limit, window size. `limits`,
`context_tokens`, `max_input_tokens` and `max_output_tokens` are the
Feed's and litellm's own field names. Write a field name verbatim when
the field is what you mean. Write this term when the fact is.

**Client-Facing Variant**:
An Alias that reaches the same Offering with the same wire request as
another Alias. It exists for a client that derives its context budget
from the Alias name instead of from the Stated Limit the proxy reports.
The provider never sees the difference.

Optional, and off unless Policy asks for it. A client that honours
`model_info` needs none: it reads the Stated Limit and budgets correctly
from the plain Alias.

Two Aliases that differ only this way must state the same Stated Limit,
because litellm holds one entry per model string — see ADR 0007. A
Declared Offering states which Alias it widens; nothing is inferred from
a name. That a client reads a name this way is measured per client, never
assumed.
A Variant and the Alias it widens share ONE Health Key, because they
are one Offering under two names. Health cannot differ where the wire
request does not.
_Avoid_: duplicate, alias variant, 1m alias (write the full term; "the
variant" alone reads as any second copy)

**Sub-allowance**:
An Offering capped inside its own Entitlement's pool. Its exhaustion
says nothing about the pool; the pool's exhaustion still reaches it.
Containment runs one way: out, no; in, yes.

At most half the operator's Claude weekly quota may go to
`claude-fable-5`, so fable can run out with the rest untouched, and the
whole quota can run out with fable's own half unspent. Stated in
Policy, never inferred, and it names no percentage — the provider sets
that and can change it without telling us.
_Avoid_: sub-pool, nested quota, tier

**Exhausted**:
A Route whose recorded quota exhaustion has not yet cleared. It stays
in the Generated Config and a caller can still reach it, so it is not
Excluded. It is simply not recommended: `guidance` will not answer with
it, and a row whose every Route is Exhausted reports `callable_now` as
false.

This is the only state a Passthrough Auth Offering's quota exhaustion
can produce, because that quota belongs to the caller and never
Excludes the Offering — see ADR 0010. An exhaustion stating no reset
time clears after the schedule's maximum staleness, since nothing else
can ever clear it.
_Avoid_: unavailable, down, disabled (it answers for a caller whose own
allowance is intact)

**Health Key**:
The name under which Health State records an Offering. A Discovered
Offering's Health Key is its Offering id; a Declared Offering's is its
Alias, because it has no Feed id. The proxy knows only the Alias, so an
Observation must be translated to the Health Key before it is folded in.
_Avoid_: offering key, model key, id

### The six files

Each has exactly one writer. That rule is the point.

**Policy**:
The operator's declaration of which Offerings may be used, under which
Aliases, and which are deliberately withheld and why. Written only by
the operator: by hand in an editor, or through the Operator Surface. No
part of the run path writes it — see ADR 0003.
_Avoid_: config, settings, rules

**Feed Document**:
The Feed's published document, as it exists on our disk. The Feed is
the authority for what it says; this file is our copy of one moment of
it. Written only by Fetch, and promoted only after it parses and passes
the plausibility check, so a failed download leaves the last good copy
in place.
_Avoid_: feed file, cache, snapshot (a snapshot is a Generated Config
kept for rollback)

**Health State**:
The machine's record of whether each Offering currently answers, and
when to try again if it does not. Written only by the maintainer.
_Avoid_: status, cache, availability (the Feed already uses
`availability` for its own claim)

**Observation Journal**:
An append-only record of failures seen while serving real traffic,
written by the proxy itself. The maintainer reduces it into Health
State, so real usage informs health without a second writer touching
Health State.

Only the main proxy writes it. A worker proxy knows an Offering under
its own name, which carries no seat identity, so an entry it wrote
would name a Health Key that does not exist.

An entry makes the next scheduled run due, whatever the interval says.
A failure a real caller hit is the point of the file; waiting out an
interval to read it would be slower than not having it.
_Avoid_: log, events, error log

**Generated Config**:
The litellm `config.yaml`. Derived output — an Offering appears only if
Policy admits it and Health State does not exclude it. Written only by
the Generator.
_Avoid_: config.yaml (when the distinction from Policy matters)

**Previous-run record**:
The maintainer's own small record of what the last run offered and
reported as Candidates, so a notification can tell "added" or "new"
apart from "unchanged". Written only by the code that calls
`notify.write_previous_run_state`, once, at the end of a run.
_Avoid_: cache

### The tools

**Prober**:
Calls Offerings to find out whether they work. The maintainer folds its
results into Health State. Its worklist comes from Policy, not from
Generated Config, so it can still reach an Offering that is currently
Excluded. It skips Withheld Offerings, since only a human clears those.
_Avoid_: tester, checker

**Probe**:
One attempt to call an Offering. It yields exactly one of five
outcomes: Answered, Self-Healing, Needs Operator, Gone, or Inconclusive.
Each names a consequence, not a cause.
_Avoid_: test, check

**Answered**:
The Offering returned a completion a client can read. It clears an
exclusion.
_Avoid_: success, passed, ok

**Self-Healing**:
The Offering is Excluded and retried without asking, because the
condition passes on its own.
_Avoid_: failed, transient, retryable

**Needs Operator**:
The Offering is Excluded and named in the report, because only a human
decision clears the condition.
_Avoid_: failed, fatal, error

**Gone**:
The identifier no longer answers for this account. The Offering is
Excluded and the report advises removal from Policy. Distinct from the
Feed's `deprecated` and `retired`, which describe its catalogue and not
whether a call succeeds.
_Avoid_: failed, deprecated, retired, removed

**Inconclusive**:
A Probe that measured nothing usable, because the failure is
attributable to our own request rate rather than the Offering. It never
changes Health State. Distinguishing this from a genuine failure is
what stops the Prober from evicting healthy Offerings.
_Avoid_: unknown, error, skipped, failed

**Generator**:
Reads Feed, Policy, and Health State; writes Generated Config.

**Fetch**:
Downloads the Feed and writes the Feed Document. The only writer of
that file. It promotes a download only after the document parses and
holds a plausible number of Offerings; otherwise it keeps the previous
Feed Document and reports the failure.
_Avoid_: sync, update, refresh

**Operator Surface**:
The commands through which the operator inspects the system and records
a decision. The only writer of Policy other than an editor. It holds
the lock, refuses a write when Policy changed on disk since it read the
file, and prints the change it made. It is never part of the run path.
An agent driving these commands acts as the operator's instrument.
_Avoid_: admin, UI, TUI, console

### States an Offering can be in

**Withheld**:
An Offering the operator has chosen not to use, for a reason the Feed
cannot know (billing unclear, subscription ending). Cleared only by a
human.
_Avoid_: disabled, blocked

**Excluded**:
An Offering observed to be unusable, so the Generator leaves it out.
Cleared automatically, by either of two paths: a later Probe succeeds,
or a recorded reset time passes. The second path needs no Probe at all,
which is what lets an Offering recover even when calling it is
expensive, rate-limited, or impossible.
_Avoid_: disabled, broken, failed

**Candidate**:
A Discovered Offering that clears Policy's structural filters but
carries no quality score, so it waits for the operator to admit or
reject it. It is reported, never silently added and never silently
dropped. Declared Offerings are never Candidates — declaring one is
already the decision.
_Avoid_: pending, unapproved, quarantined

**Sunsetting**:
A Discovered Offering the Feed reports as leaving its provider's
catalogue, and which our own Health State records as having Answered.
The Feed's own success record does not count: the Feed observes it with
the Feed owner's credentials, not ours. It stays offered, and the report
names it every run so the operator can migrate before it stops. The Feed
gives the warning; our record decides whether we keep it.
The two disagree legitimately: a withdrawn identifier commonly serves for
a grace period.
_Avoid_: deprecated, retired (those are the Feed's words for its own
catalogue, not statements about whether a call succeeds)

### Guidance

**Guidance Row**:
One model offered now, ranked against the others by one of the Feed's
own quality scores, carrying every Route that reaches it. A row names
the model; its Routes name the Aliases. A Discovered Offering's row is
its Canonical Model, never its Alias, because most Canonical Models
have several Routes and a flat list of Aliases would repeat one model
before naming a second. A Declared Offering has no Canonical Model, so
its row is its Alias and carries no score.
_Avoid_: recommendation, suggestion, pick

**Route**:
One Alias through which a Guidance Row's Canonical Model can be
reached, with its Entitlement and its current health. Routes within a
row are ordered by what they cost, cheapest first, so the order doubles
as a failover order.
_Avoid_: option, path, deployment

**Client Advisory**:
The part of a guidance answer that describes the drift between the
Generated Config and the model list a client cached earlier. The proxy
resolves a call by Alias, not by what the client last fetched, so an
Alias added since then is callable at once. An Alias removed since then
is not, and the Advisory gives the reason and the reset time so a
caller stops retrying a dead Alias.
_Avoid_: warning, note, stale list
