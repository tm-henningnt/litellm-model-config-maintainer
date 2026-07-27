# An Alias may exist for the client's reading alone

The Policy declares `claude-opus-5[1m]` beside `claude-opus-5`. Both
Aliases reach the same Offering and send the same request. Anthropic
never sees the difference. We call such an Alias a **Client-Facing
Variant**. This record states why one is not a duplicate.

## What the suffix is, and what it is not

It is not an Anthropic model ID. Anthropic's own model list names
`claude-opus-5` and nothing else. Claude Opus 5, Claude Sonnet 5 and
Claude Fable 5 each accept 1M input tokens. That figure is both the
default and the maximum. Anthropic's documentation states plainly that no
smaller context variant exists.

It is not a beta header either. The `context-1m-2025-08-07` beta applies
to an earlier generation. On this family the window needs no header, and
we send none.

It is what Claude Code calls its 1M-budget variant. The operator's
Claude Code settings hold `model: opus[1m]`, which resolves to
`claude-opus-5[1m]`.

Never put the suffix in `litellm_params.model`. litellm strips a bracket
suffix only on the Bedrock path; on the Anthropic path it reaches the
provider verbatim. It also defeats the cost-map lookup: measured,
`claude-opus-5[1m]` resolves to 200000/64000 from the regex rule ADR 0006
describes, while `claude-opus-5` resolves to 1000000/128000.

## Why a separate Alias rather than one

The client decides its own context budget, and it decides from the name
it asked for. Ask the proxy for `claude-opus-5` and Claude Code budgets
200,000 tokens, then compacts almost every turn against a model that
accepts 1,000,000. The operator reported exactly that symptom.

The proxy cannot correct a client's arithmetic. It can only offer a name
the client reads the way the operator intends. So a Client-Facing
Variant records the client's reading. It changes nothing about the call.

## The claim, now measured

This decision rested on one claim: **Claude Code derives its context
budget from the Alias name.** We wrote the Aliases before we verified it.

Measured 2026-07-26, in Claude Code against this proxy. One session, two
Aliases, `/context` read after each:

| Alias asked for | `/context` reports | auto-compact window |
| --- | --- | --- |
| `claude-opus-5` | 6.5k / 200k | 200k |
| `claude-opus-5[1m]` | 21.4k / 1m | none shown |

The claim holds. The record stands.

The measurement says more than the claim did. At the moment of both
readings the proxy reported `max_input_tokens: 1000000` for **both**
Aliases, as `/v1/models` confirmed. Claude Code still budgeted 200k for
the plain Alias. So it does not read the proxy's Stated Limit at all. It
reads the Alias name and nothing else.

### The suffix works on any Alias, whatever model is behind it

Measured 2026-07-27, and it refutes what this record first assumed. We
expected the client to resolve `[1m]` against a base model it recognises,
so that only a first-party Claude name could carry it. It does not.

`claude-chatgpt1-gpt-5.6-luna[1m]` reports `85.4k/1m`. That Alias names no
Claude model. It reaches a local worker on a ChatGPT subscription, through
an `openai/` prefix.

The same reading also proves the Stated Limit is not involved. A sibling
seat was overwriting that Alias's Stated Limit at the time, and `generate`
reported the collision. The budget still read 1M.

So `[1m]` is a pure client-side budget marker. It needs no Stated Limit,
no recognised base model, and nothing on the wire. Any Alias may carry
one.

## The client has an explicit override too

Alibaba Cloud's own Claude Code setup for the Qwen models sets
`CLAUDE_CODE_MAX_CONTEXT_TOKENS=983616` beside `ANTHROPIC_BASE_URL`. So
the budget is not only inferred from a name. The client also takes a
number directly.

That does not replace a Client-Facing Variant, for two reasons.

The variable is one value for the whole session. A caller that switches
model with `/model` keeps the same budget. One session therefore cannot
hold a 200k model and a 1M model at their true sizes. An Alias carries
its own budget, and the operator measured the switch working.

The variable lives in the caller's environment, which the proxy does not
own. Policy reaches every caller; a shell profile reaches one.

Recommend it to a caller that dedicates a session to one model. Note also
that Alibaba sets 983,616 rather than 1,048,576. A client budget is the
caller's to choose. It need not equal the Stated Limit.

## Consequences

The Client-Facing Variant is load-bearing, not a precaution. Without it
this client cannot be told the window per Alias, by any means this proxy
has.

A Stated Limit and a Client-Facing Variant fix different things, and
neither substitutes for the other. ADR 0006 fixes what the proxy reports
to a caller that reads a listing. This record fixes what a caller that
reads only names believes. A future client may do either.

Two Aliases that differ only this way must state the same Stated Limit.
litellm holds one cost-map entry per `litellm_params.model`, so whichever
entry registers last defines both. Measured twice: a sibling that stated
a Stated Limit replaced its partner's correct figures, and a sibling that
stated none inherited its partner's. `generate` warns when two such
Aliases disagree.

The plain Alias therefore states its Stated Limit explicitly, even though
litellm resolves it correctly on its own. Without the declaration the
reported figure would come from whichever sibling registered last, which
is an accident rather than a decision.

A Client-Facing Variant needs no Stated Limit. The two are independent, so
declare one only when a listing-reading caller needs the figure. Declare
none on an Offering whose true window no source states. A variant still
widens the client's budget there, and an unverified figure would add a
claim without adding a capability.

A model whose window a client already budgets correctly gets no
Client-Facing Variant. `claude-haiku-4-5` accepts 200,000 tokens, which is
the default budget anyway.

**A wider budget is a claim about the client, never about the backend.**
The suffix makes a client willing to send 1M tokens. It does not make a
provider accept them. Where the true window is unverified, a variant moves
the failure from an early compaction to a late refusal, deep into a long
session. Prefer a variant on an Offering whose window a source states, and
measure before adding one anywhere else.

The Alias count grows with the client's vocabulary, not with the
provider's. A future client that reads some other suffix needs its own
Client-Facing Variant. This record is the precedent for adding one.

## The pair also shares one health record

The same argument reaches Health State. A Client-Facing Variant is one
Offering under two names, so `DeclaredOffering.health_key` resolves a
variant to the Alias it widens, and the pair holds a single record.

Two records were not a harmless nuance. They cannot legitimately
disagree, and when they drifted the effect was concrete: an exhausted
quota Excluded `claude-opus-5` and left `claude-opus-5[1m]` in the
Generated Config — the same wire request to the same provider, certain
to fail, offered to a client as though it worked.
