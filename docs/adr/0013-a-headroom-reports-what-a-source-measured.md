# A Headroom reports what a source measured

Supersedes ADR 0005.

ADR 0005 held that `guidance` never reports how much of an allowance is
left, because nothing readable knew a balance. It named ClinePass and
OpenCode Go among the Entitlements that "publish nothing".

Both now publish a figure, through `codexbar`, a menu-bar tool that reads
each provider's own usage endpoint. Measured 2026-07-28: `codexbar` knows
64 providers and nine answered, `clinepass` and `opencodego` among them.
The factual premise of ADR 0005 no longer holds for these two.

## What changed, and what did not

ADR 0005 gave three reasons to reject a balance figure. The first two
concerned a figure we would compute ourselves: a local ledger measures our
own calls, not the provider's balance; an operator-declared budget is
bookkeeping that is the provider's job, not ours. Neither reason applies
to `codexbar`, because `codexbar` reads the provider's own account, not
our request log.

The third reason concerned per-provider balance adapters directly: most
Entitlements had no such endpoint, and building one per provider was "a
new per-provider maintenance burden of exactly the kind `docs/gotchas.md`
documents". `codexbar` is exactly such an adapter. Its burden is real —
four of its providers were failing on 2026-07-28, from expired browser
cookies and stalled OAuth flows — and it belongs to `codexbar`'s own
authors, not to this project. This repository adds one parser for
`codexbar`'s own JSON shape and builds no plugin interface. A generic
seam would invite a second adapter, and the second adapter returns the
burden here.

One objection in ADR 0005 survives whole. A figure `codexbar` reports is
a report, not a reservation: another client can spend from the same
credential between one reading and the next, so the figure lags and can
be wrong the moment it is read. `guidance` never treats it as a claim on
future capacity.

## Decision

`guidance` and `entitlements` publish a **Headroom**: how much of an
Allowance is spent, as `codexbar` measured it, stored on disk between
reads.

Three things hold, together:

1. **The figure is measured, and it lags.** It is a Reading with the
   source's own timestamp, never our own timestamp, and it describes a
   moment already past. It is not a reservation, and it never becomes
   one.
2. **The shared-credential objection survives ADR 0005 intact.** A second
   caller on the same credential can spend between the Reading and the
   dispatch that reads it. A Headroom narrows the blind spot; it does not
   remove it.
3. **Guidance publishes the figure with its source**, exactly as it
   already publishes `score_source`, `rate_source`, `cost_basis_source`
   (ADR 0011, ADR 0012) and `allowance_source`. A reader must be able to
   tell a measured figure from an absent one, and tell which tool
   measured it.

An Allowance with no declared source publishes `headroom: null`. Null
never demotes a Route, and null never reassures a caller that capacity is
free.

## What this does not do

A Headroom never rewrites the Generated Config. `guidance` may demote a
Route on a Headroom, the same way ADR 0010 already demotes a Route on an
observed exhaustion. That demotion stays off until the Readings prove
themselves over real weeks, because the first Route it can demote is the
operator's main agent. Nothing about Policy, the Prober, or Excluded and
Withheld Offerings changes here.

Coverage stays partial. Three of the operator's twelve Allowances carry a
Headroom on 2026-07-29: `pool:claude-subscription`, `provider:cline` and
`provider:opencode-go`. The rest publish `headroom: null`, exactly as
every Allowance did before this ADR.

**A provider's three window slots must be nested time windows.** A
Binding Window is the worst live window, because nested windows constrain
one Allowance at once. Measured 2026-07-29, Gemini fills the same three
slots with one quota per model — Pro, Flash and Flash Lite. Its free plan
includes no Pro, so Pro reads fully drawn while the other two read
untouched, and the worst-of rule then reports a drained Allowance that
answers normally. Gemini is therefore not mapped, as this ADR first
wrote it. Read the labels in `codexbar --provider <id>` before mapping
one. See `docs/gotchas.md`, "codexbar's three window slots do not mean
one thing".

**Update, 2026-07-29 (tickets 09 and 10).** Gemini is now mappable.
Policy names each of the three slots as its own Sub-allowance
(`headroom.sources.<id>.windows`), and names which Health Key draws on
each one (`members`). A Route reads its own slot instead of the parent's
worst-of figure. `members` reaches a Feed provider's own Offerings, not
only a Declared Offering, so `gemini` running `mode: all` gets a figure
per model from the one Reading above — the useful half this ADR left out
when it first shipped.

## Consequences

`CONTEXT.md` gains four terms — Headroom, Reading, Binding Window, Headroom
State — and a seventh file, alongside the six named in ADR 0001.

`guidance.SCHEMA_VERSION` and `entitlements.SCHEMA_VERSION` are unchanged.
A Headroom adds fields; it changes the meaning of none.

ADR 0005 stays on disk, marked superseded, because a reader must still see
why the answer was once no, and what changed in the world to make it
different.
