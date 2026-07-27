# A Probe failure we cannot read Excludes; a traffic failure does not

`classify` fails closed. A failure it does not recognise returns
`needs_operator`, which Excludes the Offering until a human clears it.
An observation from the Observation Journal takes the `inconclusive`
bucket instead, so it changes no Health State. The Prober keeps the
fail-closed default.

## Why the same failure means two different things

The Prober sends a known-good synthetic request. Nothing about that
request can be wrong, so every failure it sees belongs to the Offering.
Excluding on an unreadable message is right there: something is broken
and only a human can say what.

Real traffic is not like that. The client picks the prompt, the tools,
the sampling parameters and the token count, so the client can cause
the failure. An over-long prompt returns:

```
400 — prompt is too long: 312000 tokens > 200000 maximum
```

No `classify` rule matches that wording. `_OPERATOR_STATUSES` holds
401, 402 and 403 only, so a 400 reaches the fail-closed default. Under
one shared rule, one oversized request would Exclude `claude-sonnet-5`
from the Generated Config until the operator cleared it by hand. Every
client would hold the power to remove any model. The `[1m]`
Client-Facing Variants and the 350k-window seats make an over-long
prompt ordinary, not hypothetical.

## Considered options

**Add rules for 400, 413 and 422.** Precise, and incomplete on the
first provider nobody audited. A keyword list is the artifact this
project exists to remove.

**Classify in the maintainer, not the hook.** The cleanest separation,
but the Journal would then hold raw provider text on every failure, and
`redact.py` exists because that text carries credentials.

**Fail open on the Journal path.** One branch, so it cannot be
incomplete, and it fails in the safe direction. Chosen.

## What the re-bucketing may not change

`journal_outcome` changes the BUCKET and keeps the reason. `bucket`
names the consequence; `reason` names the condition. The condition
really is a failure this project does not recognise. Rewriting the
reason to `unmeasured` would call a real failure a non-event, which is
the conflation `classify.py` forbids beside those two names.

## Consequences

A genuine Offering failure whose wording `classify` does not know goes
unnoticed by the Journal path until a Probe finds it. We accept
slower-and-safe over fast-and-sometimes-catastrophic.

Silence is the risk this creates, so an unclassified observation
carries the provider's message, redacted and truncated, and `run`
prints every one with its Alias and a count. That message names the
rule `classify` is missing. Add the rule because ten of them arrived,
never because one was guessed at.

The message is stored for an `unrecognized_failure` and nothing else,
so the credential exposure is bounded to the cases that teach us
something, and it shrinks as rules are added. The hook builds its
redaction map from `os.environ`, not from a dotenv file alone:
`docs/gotchas.md` records that `load_dotenv()` does not overwrite an
existing variable, so the file can disagree with what the proxy sent.
