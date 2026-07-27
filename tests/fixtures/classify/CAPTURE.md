# classify fixtures: where each payload came from

Nine of the eleven fixtures are real provider responses, captured on
2026-07-25 by one direct HTTPS call each. Two could not be captured. Each
file states its own provenance in a `provenance` key. Trust that key.

Every fixture holds the provider, the HTTP status and the verbatim body.
Every fixture also holds `expected_outcome` and `expected_reset_at`, which
are the values ticket 03 asserts against.

## Warning: do not refresh these files

A captured body is evidence. Re-running a capture changes the reset time, the
request id and the quota state, and it can silently turn a failure into a
success when a plan refills. Nothing may overwrite these files.

## Redaction

Each body passed through the redaction map before it reached a file. One
account identifier survived redaction and was scrubbed by name: the OpenRouter
`user_id`. It reads `<SCRUBBED:openrouter-account-id>`.

The finished directory was checked against every value in `.env.local`. No
credential value appears. No `sk-` token and no bearer value appears.

## Captured live

| file | provider | status | condition |
| --- | --- | --: | --- |
| `gemini-deprecated.json` | gemini | 404 | the vendor removed the identifier |
| `gemini-quota.json` | gemini | 429 | quota refused at `limit: 0` |
| `opencode-go-plan.json` | opencode-go | 400 | the plan does not include the model |
| `opencode-go-gateway.json` | opencode-go | 500 | gateway error |
| `openrouter-gone.json` | openrouter | 404 | the free slug was retired |
| `cline-string-error.json` | cline | 400 | error is a string beside `success: false` |
| `cline-envelope.json` | cline | 200 | success wrapped in `data`, no top-level `choices` |
| `qwen-quota-openai.json` | qwencloud-token-plan | 429 | quota, openai shape |
| `qwen-quota-anthropic.json` | qwencloud-token-plan | 429 | quota, anthropic shape |

The last two are one plan over two protocols. They state the same reset time
in prose, in different JSON. That pair is the reason the spec says classify
must parse the message text.

## Not captured

`opencode-go-rate-limit.json` is reconstructed. Its message is quoted verbatim
from `docs/gotchas.md`, section "Probe concurrency creates false failures".
The envelope follows the provider's own error shape, taken from
`opencode-go-plan.json`. It was not captured on purpose. To produce one we
must trip a provider's limiter deliberately, and the spec tells the Prober to
avoid exactly that.

`transport-timeout.json` is synthesised. A timeout produces no response body,
so there is nothing to capture. It carries `"http_status": null`,
`"body": null` and `"transport": "timeout"`.

## Three findings the capture produced

Read these before you write classify.

**A reset time is not always a timestamp.** The Gemini body states
`Please retry in 32.368329668s.`, which is a relative delay. The Qwen bodies
state `The quota will reset at 07-29 21:45:00 UTC.`, which is an absolute
time. classify must handle both forms, or state plainly which one it ignores.

**A prose reset time can omit the year.** The Qwen message gives month, day
and time only. classify resolves the year from its `now` parameter. That is
one more reason `now` is a parameter and classify is pure.

**A quota error is not always self-healing.** The Gemini body reports
`limit: 0` on every quota metric. A limit of zero means the plan does not
include the model, so no wait clears it. A transient quota states a non-zero
limit. So the fixture expects `needs_operator`, not `self_healing`, and the
zero limit is the signal that separates them.
