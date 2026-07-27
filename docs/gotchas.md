# Provider and litellm gotchas

These traps come from building and testing a large multi-provider litellm
config. Read this file before you add a provider. Each item cost real
debugging time.

## Warning: a wrong provider prefix sends your traffic to the wrong vendor

litellm routes on the prefix of `litellm_params.model`. It accepts only
provider names it knows. Two different failures follow from this.

An unknown prefix fails loudly. `model: poolside/laguna-m.1:free` gives
`LLM Provider NOT provided`, because `poolside` is not a litellm
provider.

A *known* prefix fails silently, which is worse. An OpenRouter model id
such as `cohere/north-mini-code:free` starts with `cohere`, so litellm
sends the request to Cohere's real API with your OpenRouter key. You get
an authentication error from a vendor you did not intend to call.

Always prefix with the routing provider, not the model vendor. Write
`openrouter/cohere/north-mini-code:free`.

## Some providers wrap successful responses

One provider returns successful chat completions inside an envelope:

```json
{ "data": { "choices": [ ... ] }, "success": true }
```

Its error responses are *not* wrapped. They arrive as plain
`{"error": {...}}` at the top level.

This combination is deceptive. A standard OpenAI-compatible client parses
the errors correctly and fails only on success, with a message such as
`provider returned a response with no 'choices'`. The integration looks
half-working when it is actually zero-working.

To fix it, write a `CustomLLM` subclass that unwraps the envelope. The
Generator registers it under `litellm_settings.custom_provider_map`
whenever an Alias routes through it. Never write that map by hand.

Check the provider's *model listing* endpoint for the same pattern. A
provider that nests its model list under `data` often nests completions
too.

### Only the non-streaming body is wrapped

Measured 2026-07-26 against `https://api.cline.bot/api/v1`. With
`stream: true` the same provider answers with ordinary OpenAI chunks and
no envelope. So the handler must implement all four `CustomLLM` methods.
Implement only `completion` and `acompletion`, and every streaming call
dies on the base class's `Not implemented yet!` — after the client has
already received `message_start`, which leaves it waiting.

### The Feed may not declare the wrapper

Routing is data-driven: an Offering declares the wrapper at
`endpoint.protocol_options.response_envelope_key`, and `translate.py`
never reads a provider id. A Feed revision that stops publishing the key
therefore silently drops the handler, and every SUCCESS breaks again.

Declare it in the Policy when the Feed omits it:

```yaml
providers:
  cline:
    mode: all
    response_envelope_key: "data"
```

The Feed's own declaration always wins. Confirm the wrapper before you
write this: call the provider directly and read the raw body. A body with
the payload under one key and no top-level `choices` is wrapped.

Watch for the symptom that hides. `generate` prints a warning naming
every Alias that loses the handler on this run. Read it: the aliases still
resolve, and errors still look normal, so nothing else reports the fault
until a call succeeds.

## litellm guesses a Claude context window, then the proxy nulls it

Warning: a model absent from litellm's cost map does not report an
unknown context window. It reports a wrong one.

litellm's cost map ships a `fallback_generalizations` block. Each rule
pairs a regex with a `model_info`. The rule `claude-family-baseline`
matches `claude-[a-z]+-\d+(?:[-.]\d+)?` and asserts
`max_input_tokens: 200000` and `max_output_tokens: 64000`.

Measured 2026-07-26:

| `litellm_params.model` | exact map | resolves to |
| --- | --- | --- |
| `anthropic/claude-opus-5` | hit | 1000000 / 128000 |
| `openrouter/anthropic/claude-opus-5` | miss | 200000 / 64000 |
| `cline/anthropic/claude-opus-5` | miss | 200000 / 64000 |
| `openai/claude-gpt-5.6-sol` | miss | 200000 / 64000 |

Read the last row twice. Our Alias convention prefixes every name with
`claude-`, so the worker model `claude-gpt-5.6-sol` matches the Claude
rule. A ChatGPT model inherits a figure about a Claude model.

Inside the proxy the symptom changes again. `Router` startup calls
`register_model` for every deployment. That writes an all-null cost-map
entry for a model which is not already an exact key. A rule runs only
after an exact lookup misses, so the null entry shadows the regex:

```
before Router():  "openrouter/anthropic/claude-opus-5" in model_cost -> False
after  Router():  -> True, {max_input_tokens: None, max_output_tokens: None}
```

So the same route reports 200000 outside the proxy and nothing inside it.
Neither figure is the Feed's.

To fix it, state the Stated Limit in `model_info`. The Generator does
this for every Discovered Offering the Feed describes. Declare it by hand
for a Declared Offering. Read ADR 0006 first.

## model_info registers globally, so two Aliases contaminate each other

Warning: an Alias that states no Stated Limit can still report one. It
inherits its sibling's.

litellm holds one cost-map entry per `litellm_params.model`. Two entries
that share that string share one Stated Limit. The entry registered last
defines both.

Measured 2026-07-26 with two deployments on one model string. The first
stated 400000/100000 and the second stated nothing. Both reported
400000/100000. A second test replaced a correct exact-map entry with a
sibling's figures. The plain `anthropic/claude-opus-5` then reported
204800/32768.

State the same Stated Limit on every Alias that shares a model string, or
state none. Two cases exist here already: the two ChatGPT worker seats
per model, and a Client-Facing Variant with its plain sibling.

`generate` prints a warning naming both Aliases, the shared model string,
both figures and which Alias wins. It stays silent when they agree.

## A Stated Limit reports; it does not refuse traffic

The Generator writes no `router_settings` block, and
`enable_pre_call_checks` stays off. This is deliberate.

With that switch on, the router counts the tokens of every request and
raises `ContextWindowExceededError` when the count exceeds
`model_info.max_input_tokens`. A Stated Limit then decides whether a call
runs, not only what a listing says. So a wrong figure from the Feed
becomes an outage instead of a wrong number.

Keep a Stated Limit as metadata. It feeds the model listing,
`/model/info`, `/model_group/info` and budget reservation. It changes no
request.

The switch has a second effect. The router's own invalid-parameter
deployment filter runs only when `drop_params` is false. This proxy sets
`drop_params: true`, so that filter is already inactive.

## The [1m] suffix belongs in an Alias, never in litellm_params.model

Warning: `[1m]` is not an Anthropic model ID. Anthropic's model list
names `claude-opus-5` and no bracketed form.

Claude Opus 5, Claude Sonnet 5 and Claude Fable 5 accept 1M input tokens
as the default and the maximum. There is no smaller variant and no
`context-1m` beta header to send.

`[1m]` is what Claude Code calls its 1M-budget variant. The client reads
its own context budget out of the model name it asked for. Ask the proxy
for `claude-opus-5` and Claude Code budgets 200000 tokens, then compacts
against a model that accepts 1000000.

Put the suffix in the Alias. Route the Alias to the plain model:

```yaml
- alias: "claude-opus-5[1m]"
  litellm_params:
    model: "anthropic/claude-opus-5"
```

Never put it in `litellm_params.model`. litellm strips a bracket suffix
only on the Bedrock path. On the Anthropic path it reaches the provider
verbatim. It also defeats the cost-map lookup: `claude-opus-5[1m]`
resolves to 200000/64000 through the regex rule above, while
`claude-opus-5` resolves to 1000000/128000.

### A Stated Limit does not reach this client

Measured 2026-07-26 in Claude Code, against this proxy. One session, two
Aliases, `/context` read after each:

| Alias asked for | `/context` reports | auto-compact window |
| --- | --- | --- |
| `claude-opus-5` | 6.5k / 200k | 200k |
| `claude-opus-5[1m]` | 21.4k / 1m | none shown |

At both readings the proxy reported `max_input_tokens: 1000000` for
**both** Aliases. So this client ignores the figure. It reads the Alias
name and nothing else.

Do not expect a Stated Limit to fix a client's budget. The two are
independent: a Stated Limit corrects what a listing reports, and a
Client-Facing Variant corrects what a name-reading client believes. Fix
both, and test the client itself — the listing proves nothing about it.

## Claude Code cannot discover models through a Passthrough Auth proxy

Warning: the discovery request is never sent. Nothing reaches the proxy
log, so the fault reads as a proxy fault and is not one.

Claude Code fills its `/model` picker from the proxy's `/v1/models`. Set
`CLAUDE_CODE_ENABLE_GATEWAY_MODEL_DISCOVERY=1` beside
`ANTHROPIC_BASE_URL`. Discovery then runs at startup and caches the
result to `~/.claude/cache/gateway-models.json`.

Discovery sends exactly one credential header: `ANTHROPIC_AUTH_TOKEN` as
a bearer token, or else the resolved API key in `x-api-key`. Passthrough
Auth sets neither variable. The credential travels in
`ANTHROPIC_CUSTOM_HEADERS` as `x-litellm-api-key`, which serves an
inference call and does not satisfy this rule.

Measured 2026-07-27, Claude Code 2.1.220, against this proxy:

| Client credential | `/v1/models` in the log | picker |
| --- | --- | --- |
| `ANTHROPIC_CUSTOM_HEADERS` only | none | built-ins only |
| `ANTHROPIC_AUTH_TOKEN`, dummy value | one call at startup | every Alias |

The proxy answers correctly. A direct `GET /v1/models?limit=1000` returns
200 with 79 Aliases, no redirect, well inside the 3-second timeout. All
79 start with `claude-`, which is the prefix filter discovery applies.

No proxy repairs this, so do not build one. Claude Code puts
`ANTHROPIC_AUTH_TOKEN` in `Authorization` *instead of* the passthrough
credential. A shim between the client and the proxy therefore receives
the dummy token and never sees the credential it replaced. The loss
happens in the client, upstream of everything we run. A worker proxy
solves the opposite case, where a credential is added downstream.

Two remedies exist, and neither restores the full list.

`ANTHROPIC_CUSTOM_MODEL_OPTION` adds one entry to the picker, with
`_NAME`, `_DESCRIPTION` and `_SUPPORTED_CAPABILITIES` beside it. One, not
a list. Set `_SUPPORTED_CAPABILITIES` for a non-Claude model behind a
`claude-` Alias: capability detection matches the model id, so effort and
thinking stay off without it.

Otherwise read the list from `guidance` instead of the picker. It reports
`callable_now`, the cost basis and a failover order, which the picker
cannot.

One idea is untested. Discovery caches to a file, and the picker falls
back to that file when a fresh list does not arrive. A throwaway startup
carrying a dummy token could prime the cache for a later session that
carries none. Whether the no-credential path reads the cache, or skips
the read with the request, is unmeasured. Test it before you rely on it.

## A list-shaped system prompt fails on the ChatGPT backend

Warning: the error names the role, and the role is not the cause. The
shape of the content is.

Measured 2026-07-27 against a live worker. Four requests, differing only
in the system message:

| message | result |
| --- | --- |
| `{"role": "system", "content": "be terse"}` | 200 |
| `{"role": "developer", "content": "be terse"}` | 200 |
| `{"role": "system", "content": [{"type": "text", ...}]}` | 400 |
| `{"role": "developer", "content": [{"type": "text", ...}]}` | 400 |

Both 400s read `{"detail":"System messages are not allowed"}`. Renaming
the role does not help, so a role-only fix looks correct and changes
nothing.

The cause sits in litellm's Responses bridge,
`completion_extras/litellm_responses_transformation/transformation.py`:

```python
if role == "system":
    if isinstance(content, str):
        instructions = content       # hoisted; no system message is sent
    else:
        input_items.append(...)      # stays a message the backend refuses
```

A string becomes the top-level `instructions` field, so no system message
reaches the backend at all. A list stays an input message.

Flatten the instruction to a string. Read the `text` of each block and
join the blocks. Drop a `cache_control` marker, because this backend does
not read one.

Flatten an instruction role only. A user or assistant turn with
list-shaped content works, and flattening one would discard an image.

This hides behind a client. Claude Code sends its system prompt as a list
of content blocks on every request, so every Claude Code call failed while
a `curl` with a string system prompt succeeded. Test with the shape your
client sends, not with the simplest shape that works.

## A hook that discovers its targets from a config file can read the wrong one

Warning: a worker proxy does not serve `config.yaml`. A hook that looks
for that name reads a config the worker does not serve, and it fails
silently.

`chatgpt_role_fix` must know which Aliases target the `chatgpt/` provider,
because litellm calls the hook before it resolves an Alias to a
deployment. It reads that set from a config file.

Measured 2026-07-26. A worker runs with
`--config chatgpt-worker.yaml`. Every path candidate the hook tried —
the working directory, then its own directory — resolved to `config.yaml`
beside that file instead. The hook read the main proxy's config.

The fault hid for one release, because the wrong file held the right
names. The main Policy once declared six direct `chatgpt/` Offerings under
the same Aliases the worker uses. The operator retired those six. The
alias set became empty, the hook rewrote nothing, and every worker request
failed:

```
litellm.BadRequestError: ChatgptException -
{"detail":"System messages are not allowed"}
```

Nothing warned, because reading the file succeeded. It simply held no
`chatgpt/` entry.

Ask litellm for the path it loaded. `litellm.proxy.proxy_server.
user_config_file_path` holds the `--config` value, so it cannot name the
wrong file. Import it inside a function and guard it, because an error at
import time stops the proxy from starting. Fall back to
`LITELLM_CONFIG_PATH` and `CONFIG_FILE_PATH` before any directory guess.

Then warn on an empty target set. A registered hook that matches nothing
is a misconfiguration, never a valid state. That warning is what turns
this class of fault from silent into obvious.

The same shape applies to any hook keyed on a provider prefix. Two proxies
sharing one directory is the trap; the file name is not the identity of
the config.

## The proxy environment can differ from your .env file

`load_dotenv()` does not overwrite a variable that already exists in the
environment. A stale export in your shell profile therefore wins over
the correct value in `.env`.

The symptom is confusing. Every model fails with an authentication error
through the proxy, but the same credentials work when you call the
provider directly.

Do not debug `config.yaml` first. Inspect the proxy process instead. On
macOS, run `ps -E -p <pid>` and compare each variable against `.env`.
Watch for two cases: a placeholder value, and a variable with a similar
but different name.

Beware: that command prints live secrets. Do not save its output.

## Probe concurrency creates false failures

Free tiers limit requests per minute. Concurrent health checks trip those
limits, and the provider returns 429. A naive checker records this as
"the model is broken".

We measured this. Nine verdicts changed between two runs of an identical
config. One model gave FAIL, then PASS, then FAIL across three runs. One
provider reported `Worker local total request limit reached (180/32)` —
our own checker caused that.

Treat a rate-limit failure as a third outcome, not as a failure. Record
"measured nothing" and leave the previous state alone. Set a concurrency
limit and a minimum interval for each provider separately. Subscription
providers tolerate parallel calls. Free tiers need one call at a time.

## A catalogue can report a model as available when it is not

Verify every model with a real call. A discovery feed or catalogue states
what is *worth* testing. It does not prove what works.

We found three kinds of disagreement, all on entries the catalogue marked
`available` with a fresh timestamp:

- A model returned 400. The provider's own plan tier did not include it.
- A model returned 502 on every attempt.
- A model returned 404. The vendor had deprecated it for new accounts.

A benchmark score also proves nothing about reachability. A model can
score well and still be a dead identifier.

## Model identifiers need normalisation per provider

Providers state ids in their own form. Gemini reports
`models/gemini-3.5-flash`. litellm needs `gemini/gemini-3.5-flash`, so
you must remove the `models/` prefix.

A provider can also expose one model over more than one protocol. Do not
assume a documented SDK route is the only route. Test the alternative
before you build a special case for it — we removed one special case this
way.

## litellm cannot price a generic openai/ model

litellm reads its own price map by provider and model name. A native
prefix such as `groq/` or `openrouter/` resolves and cost tracking works.

A generic `openai/<model>` with an explicit `api_base` does not resolve.
`get_model_info()` raises, and litellm reports the spend as zero.

To fix it, supply the rates yourself in `model_info`:

```yaml
model_info:
  input_cost_per_token: 0.0000014
  output_cost_per_token: 0.0000044
```

Be careful with subscription plans. Their published rates are list
prices, not amounts you pay. Mark such entries so a reader cannot mistake
the figures for an invoice.

## Duplicate model_name values do not raise an error

litellm treats two entries with the same `model_name` as one
load-balancing group. It splits traffic between them.

This hides a mistake. Two entries can point at different models, and the
proxy starts normally. Validate that every `model_name` is unique before
you write the file.

## Claude 5 models reject temperature=0

The Claude 5 family accepts `temperature=1` only. A health check that
sends `temperature=0` fails on every Claude 5 model.

Send no `temperature` unless you need one. Alternatively, set
`litellm_settings.drop_params: true`.

## The Responses API needs a list

Models that use the Responses API need `input` as a list of message
objects. A bare string returns `{"detail": "Input must be a list"}`.

```python
litellm.responses(
    input=[{"role": "user", "content": "..."}],
    model="...",
)
```

## Quota errors carry a reset time, in prose

Providers tell you when a quota resets. No provider we tested puts the
value in a machine-readable field. Two routes to the same plan returned
different JSON shapes, and both buried the same timestamp in the message
text.

Parse the message to get the reset time. Store it. Then skip the model
until that time passes, and restore it afterwards without a call. A
recovery on the clock costs nothing, so it also works for a model you
cannot check directly.

## A stream-only provider makes litellm break non-streaming callers

The ChatGPT subscription backend accepts streaming requests only. The
litellm `chatgpt` provider therefore sets `stream = True` in
`llms/chatgpt/responses/transformation.py`, whatever the caller asked for.

`llms/custom_httpx/llm_http_handler.py` then reads that flag back:

```python
stream = bool(stream or data.get("stream"))   # data is the transformed request
...
is_stream_request = bool(stream)
```

The flag now hides the caller's intent. The handler takes the streaming
branch, so a caller that asked for one JSON object receives an SSE body
with `content-type: text/event-stream`. The caller then fails on
`raw_response.json()`.

The symptom points at the wrong layer. A client reports a missing field,
such as `usage.input_tokens`, because it parsed an error body. The model
answered correctly, and the text sits inside the error message.

The provider already holds the repair. Its
`transform_response_api_response` detects SSE and folds it into one
Responses object. That code sits on the non-streaming branch, so the
forced flag makes it unreachable.

Test both directions per Alias. A stream-only provider can pass every
streaming check and still fail every non-streaming one.

Beware: the fix is a patch to installed litellm, so `uv tool upgrade
litellm` removes it. Keep the caller's intent in its own variable, and
gate the branch on that variable:

```python
client_requested_stream = bool(stream)   # before the transform runs
...
is_stream_request = client_requested_stream
```

Patch the sync `response_api_handler` and the async
`async_response_api_handler`. Both carry the same two lines. Restart every
proxy that loads the `chatgpt` provider afterwards; a proxy that only
forwards to such a worker needs no patch.

## An OpenCode Go assistant turn must be a plain string

The OpenCode Go gateway accepts a structured content array on a `user`
turn. It rejects one on an `assistant` turn. Measured 2026-07-26 against
`deepseek-v4-flash`:

| Assistant turn `content` | Result |
| --- | --- |
| `"Hello!"` | 200 |
| `[{"type": "output_text", ...}]` | 400 |
| `[{"type": "input_text", ...}]` | 400 |
| `[{"type": "text", ...}]` | 400 |
| `{"type": "message", "role": "assistant", ...}` | 400 |

The error text is `Error from provider (Console Go): Upstream request
failed`. It names no field.

litellm's Anthropic-to-Responses adapter always writes an array, so the
first turn answers and every later turn fails. Request size is not the
cause: a 300 KB single-turn request answers.

Route `/v1/messages` through chat/completions instead. Set this under
`proxy_settings.litellm_settings` in your Policy:

```yaml
use_chat_completions_url_for_anthropic_messages: true
```

The flag applies to every `openai/` provider, and litellm offers no
per-deployment override. Check each Alias that uses that prefix after you
set it.

## litellm requires created_at, and some providers omit it

The OpenAI Responses specification includes `created_at`. Not every
provider that answers with `"object": "response"` sends the field. OpenCode
Go returns `id`, `object`, `model`, `output`, `stop_reason` and `usage`,
and nothing else.

`llms/openai/responses/transformation.py` reads the field by subscript, so
a missing field raises `KeyError`. A bare `except Exception` then reports
the provider body as an `OpenAIError`. The log shows a correct provider
response next to a stack trace. The model answered.

The flag above avoids this on `/v1/messages`, because chat/completions
never reads `created_at`. A direct call to `/v1/responses` still fails.
Read the field with `.get("created_at")` if you need that endpoint: the
helper already substitutes the current time for `None`.

## A usage-only final chunk can truncate a stream

A provider may end a chat/completions stream with a chunk whose `choices`
list is empty, carrying only `usage`. This is legal. litellm's
Anthropic streaming adapter indexes `chunk.choices[0]` before it checks the
merge path, so the chunk raises `IndexError`.

The generator dies inside the SSE body. Status 200 was already sent, so the
client receives a stream with no `message_delta` and no `message_stop`, and
waits for an end that never arrives.

Two symptoms follow, and the second looks unrelated: the stream never
terminates, and reported usage is `0/0` because the usage chunk crashed
instead of merging.

Fix the order, not the index. The adapter discards the translated chunk on
the merge path, so merge first and translate second. Patch the sync and the
async iterator; both carry the same block.

Do not read a well-formed body as a well-formed stream. Count the events
per Alias: one `message_start`, balanced `content_block_start` and
`content_block_stop`, then `message_delta` and `message_stop`.

## The maintainer can observe its own traffic and re-trigger on it

Warning: any maintainer call that goes through the proxy is recorded by
the proxy's failure callback, exactly like a real client's call. If a
recorded failure also makes the next tick due, the maintainer feeds
itself.

Measured 2026-07-27. The tick ran 7 times in 7 minutes. The chain:

1. The tick finishes a run with the post-write smoke check, which calls
   the proxy once per translation rule.
2. Those calls failed, so `journal_failure_callback` appended 36
   entries.
3. An unprocessed Journal entry elapses `schedule.interval_minutes`
   (`schedule.due`, `journal_pending`).
4. The next tick therefore ran at once, smoke-checked again, and
   appended more entries.

The interval rule that normally prevents a tick storm was bypassed by
the very signal the run had just created.

Two things stop it, and both are needed.

Break the cycle structurally. A journal-triggered run makes no proxy
traffic of its own: no smoke check, no Feed fetch, and no probe beyond
confirming an ambiguous entry. An ordinary tick still smoke-checks, so a
genuine smoke failure still reaches Health State, but the chain ends
after exactly one extra run.

Do not read a credential from `os.environ` alone. Every smoke call in
the loop above failed with `No api key passed in.`, because
`cmd_run` read `LITELLM_MASTER_KEY` from the process environment while
the launchd job exported nothing. Resolve through
`cli._credential_resolver`, which reads the environment and then the
`--env` file. Six translation rules reported FAILED for weeks for this
reason alone; all six pass once the key resolves.

Fail-open is what kept this from being serious. All 36 entries were
`unrecognized_failure`, which `reduce.journal_outcome` re-buckets to
`inconclusive`, so none of them changed Health State (ADR 0008). Under
`classify`'s fail-closed default the first tick would have Excluded 36
Aliases.
