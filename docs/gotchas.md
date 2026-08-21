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

## A forwarded client header overrides an anthropic/ route's own key

Warning: never give a provider an `anthropic/` prefix when the provider
holds its own credential. Use an OpenAI-shaped route instead.

The proxy sets `general_settings.forward_client_headers_to_llm_api:
true`. Passthrough Auth needs it: a `claude-*` direct Alias carries no
`api_key`, and the caller's own credential must reach Anthropic. The
setting forwards that caller's `Authorization` header to EVERY provider,
not only to Anthropic.

An `anthropic/` route authenticates with `x-api-key`. The forwarded
`Authorization` header therefore survives beside it, and the provider
reads the caller's token. The Qwen Token Plan answers HTTP 401
`InvalidApiKey`. The configured key is correct and never reaches the
provider.

An `openai/` route writes `Authorization` itself, so it overwrites the
forwarded header. A `gemini/` route authenticates with `x-goog-api-key`
and never collides. Both are safe.

`extra_headers` does not repair an `anthropic/` route. Forwarding runs
after it and wins. Measured 2026-08-02: a route with the correct key
written literally into `extra_headers.Authorization` still answered 401.

The symptom looks provider-specific, because only anthropic-shaped
routes fail while every other provider answers. Reproduce it with the
caller's own header shape. A call that authenticates with the master key
alone sends no second credential, so it succeeds and hides the fault.

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

### Open: `openai/qwen3.7-max` states two different figures

Four Aliases share `openai/qwen3.7-max`. OpenCode Go and the Qwen Token
Plan both serve this model, and the Feed states a different output limit
for each Offering:

    claude-opencode-go-qwen3.7-max         max_output_tokens 65536
    claude-qwen-token-plan-qwen3.7-max     max_output_tokens 131072

One figure therefore defines all four. Routing is correct, because each
Alias keeps its own `api_base`. Only the reported limit is wrong for one
provider.

To close it, state 65536 on both. The lower figure never over-promises.
The decision belongs to the operator, because the figures come from the
Feed per Offering.

The five other shared Qwen model strings agree on their figures:
`qwen3.7-plus`, `qwen3.6-flash`, `glm-5.2`, `deepseek-v4-pro` and
`qwen3.8-max-preview`. They need no action.

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

We found five kinds of disagreement, all on entries the catalogue marked
`available` with a fresh timestamp:

- A model returned 400. The provider's own plan tier did not include it.
- A model returned 502 on every attempt.
- A model returned 404. The vendor had deprecated it for new accounts.
- A model returned 403. The vendor gates it to its own product surfaces.
  Measured 2026-08-03: `cline-free/glm-5.2` answered "only available via
  Cline product surfaces". A raw API call cannot reach it.
- A gateway relayed a 404 from the catalogue behind it. Measured
  2026-08-03: Cline answered HTTP 500 for `poolside/laguna-m.1:free`,
  and the body carried OpenRouter's own refusal, "No endpoints found".
  The Feed still published the Offering. classify names no rule for the
  message, so the Offering sits Excluded until the Feed drops it.

A benchmark score also proves nothing about reachability. A model can
score well and still be a dead identifier.

The reverse also happens. A provider's own `/v1/models` can omit a model
it serves. Measured 2026-07-28 against `https://example-private-host.invalid/api/v1`: the
list held 8 ids, two of them chat models. Two ids absent from the list,
`minimax-m3` and `qwen35-397b-a17b`, each answered a chat completion with
HTTP 200 and a real reply.

So do not read a provider's model list as its roster. Read it as a hint,
and confirm a model with one call. This is why a Declared Offering is the
operator's statement and never a discovery: nothing here polls that
endpoint to decide what the proxy serves.

## One provider states a failure inside a 2xx stream

Read the frames, not the status. Measured 2026-08-21: Cline answers
HTTP 200, sends one frame carrying
`{"error":{"code":"stream_initialization_failed", ...}}`, then `[DONE]`.
The frame holds no `choices`. A reader that counts chunks sees an empty
stream and reports a body with no completion, so `classify` calls it
`malformed_response` and asks the operator to look at a five-second
rate limit.

`read_stream` now keeps the first error frame in `StreamedRead.error`.
Both live callers build their body with `prober._streamed_body`, which
passes that frame to `classify` when no chunk arrived. The provider
handler raises on the same frame, because the alternative is an empty
answer with no error, and nothing retries an empty answer.

## One provider fails a non-streaming call that streams

Test both ways before you trust a verdict. Cline answers HTTP 500
"empty response content" when a non-streamed completion carries empty
`content`. A reasoning model empties it by spending a small `max_tokens`
budget on reasoning.

Measured 2026-08-21 on `liquid/lfm-2.5-2.6b:free`,
`stepfun/step-3.7-flash`, `cohere/north-mini-code:free` and
`poolside/laguna-xs-2.1:free`. All four failed with `max_tokens=16`.
All four answered with `max_tokens=400`, each after 132 to 617
characters of reasoning. All four stream correctly at `max_tokens=16`.

Every Probe and the smoke check stream, so neither meets this
condition. A client that does not stream, with a small budget, meets it
every call.

## Cline resells OpenRouter, and rewrites the upstream condition

Read a Cline failure as OpenRouter's condition in Cline's words. The
catalogue at `/api/v1/ai/cline/models` holds 419 models in OpenRouter's
own schema, and the errors name the origin: "failed to invoke model
'z-ai/glm-5.2:free' from Openrouter".

Four rewrites, measured 2026-08-21:

- An upstream 429 streaming becomes HTTP 200 plus an error frame.
- An upstream 429 non-streaming becomes HTTP 500, `error` a plain string.
- Empty `content` non-streaming becomes HTTP 500 "empty response content".
- Cline's own `cline-free/*` namespace answers 403 "only available via
  Cline product surfaces", with `error` an OBJECT and no `success` key.

The catalogue also lists ids the inference path rejects.
`stealth/ox-alpha` answered 404 "model not found" through Cline while
OpenRouter served the same id 200.

## One measurement does not withhold a provider's whole free tier

Withhold the Offering you measured, and no other. A 403 on
`cline-free/glm-5.2` was recorded on 2026-08-03, and its one-line reason
was then carried on 13 Cline Offerings. Measured 2026-08-21 by calling
all 13 directly: 9 answered, 2 refused with that 403, and 1 relayed
OpenRouter's own refusal.

The error compounds, because `probe` skips a withheld Offering. The
reason blocks the measurement that would test the reason, so a wrong
line never expires. Nine free models sat unreachable for 18 days.

## A refusal can state a limit the model list omits

A provider that publishes no window often names it in an error. Ask for an
impossible output and read the 400:

```
curl -s -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" \
  -d '{"model":"<id>","messages":[{"role":"user","content":"hi"}],"max_tokens":99999999}' \
  "$BASE_URL/chat/completions"
```

Measured 2026-07-28 against `https://example-private-host.invalid/api/v1`, whose `/v1/models`
carries no limits at all:

```
{"error":"Backend error: max_tokens=99999999 cannot be greater than
 max_model_len=max_total_tokens=1048576. ..."}
```

All four models answered this way, and every figure matched the Feed
mirror's `context_tokens` exactly. One call per model, no bisection.

Two cautions. Read where the message puts the number: `max_total_tokens`
is ONE budget for input plus output, so it is not a `max_output_tokens`,
and stating it as one promises a full-window reply to a caller who also
sent input. And read the error's own shape — this host returns a bare
string under `error`, not the usual `error.message`, so a `jq
'.error.message'` prints `null` and the limit looks absent.

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

## A timeout is a failure the provider never stated

Warning: a timeout shares a bucket with three conditions the provider
DID state. Read the reason, never the bucket, wherever the code asks
whether a failure speaks for itself.

`classify` maps every transport condition to `self_healing` with the
reason `timeout`. No response arrived, so there is no provider message
to read. The same bucket also carries a quota exhaustion, a gateway
error and a rate limit, and each of those quotes the provider's own
wording.

A journal-triggered run probes only what needs confirming. It used to
read the bucket, so a timeout took the self-identifying path and
Excluded the Offering on real traffic alone.

Measured 2026-07-31 on `claude-chatgpt1-gpt-5.6-sol`, the slowest model
on that seat:

1. A `claude -p` run pinned to that Alias sent a long agentic request.
2. Two requests raised `litellm.exceptions.Timeout`.
3. The tick woke early on `journal_pending` and printed `0 confirming
   Probe(s), no sweep`.
4. `reduce` Excluded the Offering. It excludes on the FIRST failing
   event; `failure_count` is recorded and no code reads it as a
   threshold.
5. The Generated Config went from 102 Aliases to 101, under a job that
   was still running against that Alias.

The Offering answered a Probe 23 hours before and answered one again
afterwards. Nothing was wrong with it. The client's own deadline, a slow
model and a busy worker all arrive here as the same condition.

Recovery made it worse instead of self-correcting. A timeout carries no
reset time, so `reduce._apply_reset_expiry` cannot clear the exclusion by
the clock. Only a Probe can. The journal-triggered run also reset
`last_run_at`, which postponed the next sweep from 14:03Z to 14:58Z --
so the run that Excluded the Offering delayed the one Probe that could
restore it by 55 minutes.

The repair is one condition in `cli._needs_confirming`: a timeout gets
exactly one confirming Probe, like an `inconclusive` outcome. The Probe
overrules the observation, because `reduce` applies a Probe outcome last.
An Offering that answers keeps its place; one that times out on an
eight-token known-good request too is Excluded on two independent
measurements.

Two cases the repair does not reach. A Passthrough Auth Offering is
never probed, so a timeout still Excludes it with no confirmation and no
clock recovery. Only the operator clears that, by editing Health State.
And the interval reset stands: a journal-triggered run still postpones
the next sweep.

## The Generated Config holds models that do not answer

Warning: this is deliberate. A model in `config.yaml` is not a claim
that a call to it will succeed, and `status` naming an Offering Excluded
does not mean it left the file.

Every write to the Generated Config restarts the proxy, because that
file is the one the litellm `--reload` watcher reads, and a restart ends
every session in flight. Health State changes its mind often, so writing
on a measurement killed the operator's work for conditions that reversed
themselves minutes later. ADR 0014 states the rule: only a deliberate or
terminal fact may drive a write.

So an Excluded Offering stays in the file and stops being recommended.
Only Withheld and Gone remove one, and that state is Unlisted.

Three things follow, and each has surprised a reader:

**`available: true` beside `recommendable: false` is now common.** It was
rare, and produced only by an exhausted quota (ADR 0010). It is now the
ordinary shape of a failing Offering. A caller reading `available` alone
selects Routes the maintainer called and was refused by.

**Calling one is not an error.** You reach the provider's own message —
"your quota resets at 09:00", or an authentication error naming the
vendor — instead of "model not found" from the proxy, which names
nothing and invites a retry. That is the compensation for the rule.

**Nothing reaps the file.** An Offering that breaks permanently without
ever classifying Gone stays listed. `Gone` has fired zero times to date,
so no reaper is built. Build one when the first such Offering appears.

To find out why any particular Alias behaves as it does, run
`litellm-maintainer explain <offering-or-alias>`.

## A 401 states that one request was refused, not that a key is invalid

Warning: an authentication failure reads as the most self-identifying
failure there is. It names a cause in plain words. It still does not name
whose fault the cause is.

`classify` maps HTTP 401, 402 and 403 to `needs_operator` with the reason
`authentication_failed` (`_OPERATOR_STATUSES`). That bucket Excludes the
Offering, and it carries no reset time, so the clock can never clear it.
Only a Probe can.

A journal-triggered run used to Exclude on that reason with no confirming
Probe, because `cli._needs_confirming` selected on conditions the provider
never stated and a 401 is stated.

Measured 2026-07-31 on `qwencloud-token-plan:qwen3.8-max-preview`:

1. Real traffic returned an invalid-key error, twice over several minutes.
2. The tick woke early on `journal_pending` and printed `0 confirming
   Probe(s), no sweep`.
3. `reduce` Excluded the Offering. `failure_count` reached 7.
4. The Generated Config went from 102 Aliases to 100.

The credential was valid throughout. The same key answered ten direct
requests out of ten, and the five pool siblings that share it read
`recommendable: true` the whole time. A revoked credential cannot refuse
one Offering and serve five.

The Offering flapped rather than parked, which made it harder to read.
`prober._is_fresh` returns False for an Excluded record whose `reset_at`
is null, so the next sweep probed it and it returned -- until the next
journal 401 removed it again. The tick log showed `Offering removed`,
`Offering added`, `Offering removed`.

The repair adds this reason to `cli._needs_confirming`. The argument is
not the timeout's: `reduce._PASSTHROUGH_EXEMPT_REASONS` already holds
`authentication_failed`, because a Passthrough Auth Offering carries the
caller's credential. Where the proxy owns the credential the doubt is
smaller, and it is not zero. So confirm the failure rather than exempt it.
A Probe sends a known-good request under the proxy's own credential, which
is the measurement that separates a refused request from a dead key. A
genuinely revoked credential fails the Probe too and is Excluded on two
measurements.

To repair one by hand, probe the provider and then write the config:

```
litellm-maintainer probe --provider <id> --env .env.local \
  --feed ~/.config/litellm-maintainer/feed.json \
  --policy ~/.config/litellm-maintainer/policy.yaml \
  --home ~/.config/litellm-maintainer
```

`probe` clears Health State and writes no config, so the proxy keeps
refusing until `generate` runs. Pass the tick's own `--out`
(`~/.config/litellm/config.yaml`). The `generate` default writes to the
instance directory, which the proxy does not read.

One divergence this repair leaves. `watcher.py` still selects on
`bucket == INCONCLUSIVE`, so the foreground `watch` command confirms
neither a timeout nor an auth failure. `watch` is a debugging tool and not
the production path, but it now behaves differently from the tick. Move
the predicate to `classify` to share it: `cli` imports `watcher` at module
level, so `watcher` cannot import `cli`.

## A rotted Headroom mapping looks exactly like an unmapped Allowance

Every part of the Headroom capability degrades to one symptom: no
Headroom. `guidance` and `entitlements` publish `headroom: null` for an
Allowance nobody mapped, and for one whose mapping just broke. The two
states read the same, and most Allowances here are the first kind. A
break is invisible unless something names it.

`doctor` names it, in five separate checks:

1. A declared `headroom.sources` entry that matches no Reading codexbar
   publishes right now. codexbar renamed a provider, or the account
   logged out.
2. A declared source that matches SEVERAL Readings. The key stopped
   discriminating one account from another — the exact failure ADR 0009
   and the headroom spec's decision 4 refuse to paper over.
3. Policy's `headroom.command` binary is not on the PATH.
4. A declared slot (`headroom.sources.<id>.windows`) the Reading no
   longer publishes. Measured 2026-07-28: codexbar's own shape moved
   inside one capture session. At 18:48Z the Claude Reading carried an
   extra window, `claude-weekly-scoped-all-model`, at 82%, beside
   `claude-weekly-scoped-fable` at 59%. At 20:52Z it carried only the
   fable window; the all-model figure had moved into `secondary`. A
   declared slot can stop being published with no other symptom at all.
   Three more checks (ticket 10) name a `members` mapping gone stale: an
   admitted Health Key no member claims, a declared slot with no members,
   and a member naming no known Health Key. All three read from Policy
   and the Feed alone, so they fire with no live codexbar Reading at all.
5. An installed refresh job whose `StartInterval` no longer matches
   Policy's `headroom.interval_minutes`. The job bakes its interval into
   the plist at install time, because it has no gate of its own to
   re-read Policy against on every tick (unlike the scheduled tick, whose
   `due` reads Policy fresh every run). An operator who edits the
   interval and does not run `headroom install` again keeps the old
   cadence, silently, forever.

Checks 1, 2 and 4 need a LIVE reading from codexbar, not Headroom State
on disk. `headroom refresh` keeps a stale Reading under its Allowance
FOREVER once one match ever succeeded — a provider that starts erroring,
or that codexbar renamed, never removes its old record, so the file on
disk cannot tell "still matching" from "matched once, months ago". Only
asking codexbar again can say a mapping stopped working right now. This
is the one case `doctor` makes a second network call for, beside the
proxy check it has always made; it makes this one only when Policy
declares a `headroom` source at all.

Two failures must never be confused with each other, and both `doctor`
and the `warnings` array on `guidance`/`entitlements` keep them apart.
A window past its own `resets_at` is VOID: the quota refilled, the
figure describes a period that ended, and this is normal, self-correcting
behaviour that needs no warning at all. A Reading whose OWN copy time
(`read_at`, ours, never codexbar's `updatedAt`) has not moved in several
refresh intervals is STALE: the refresh job stopped, or codexbar has
been erroring for this provider on every run since. Reporting the first
as the second trains an operator to ignore a real warning; reporting the
second as the first hides a dead job behind "the quota must have
refilled".

A machine that declares no `headroom.sources` at all raises none of this:
no finding, no warning, no extra network call. Silence is the correct
state for a capability that is switched off.

## codexbar's three window slots do not mean one thing

`primary`, `secondary` and `tertiary` carry a different KIND per provider.
`codexbar` documents no meaning for them, and the JSON never labels them.
The text output does. Run `codexbar --provider <id>` and read the labels.

Measured 2026-07-29:

```
claude      Session / Weekly              nested time windows
clinepass   5-hour / Weekly / Monthly     nested time windows
opencodego  5-hour / Weekly / Monthly     nested time windows
gemini      Pro / Flash / Flash Lite      one quota per MODEL
```

A Binding Window is the worst live window, because nested time windows all
constrain the same Allowance at once: a spent weekly window binds however
fresh the 5-hour one reads. That rule is correct for the first three.

It is wrong for the fourth. Gemini's slots are three models, and one says
nothing about the next. On the free plan Pro reads `usedPercent: 100`,
because that plan includes no Pro at all, while Flash and Flash Lite read
0. Bind on the worst and the whole Allowance reports fully drawn while two
of its three models are free. With `headroom.demote_at_full` set, every
Gemini Route then stops being recommended.

That is the ClinePass trap inverted. ClinePass reads 0% on two windows and
100% on a third, so a reader that picks one slot reports free capacity that
will refuse. Gemini reads 100% on one slot of three, so a reader that takes
the worst reports a drained Allowance that answers. Both publish a number
the operator cannot act on.

`usedPercent` is not the problem. It counts what was spent, and it counts
it correctly: Pro at "0% left" is `usedPercent: 100`. What differs is what
the slot is a quota OF.

**Read the text labels before mapping a provider.** Two tells mark the
per-model shape: the labels name models rather than periods, and every
slot states the same `windowMinutes` — Gemini stated 1440 three times,
where ClinePass states 300, 10080 and 43200.

A per-model quota is a Sub-allowance in this project's language. Since
2026-07-29, Policy can name one of these three slots directly:

```yaml
"provider:gemini":
  source: "codexbar:gemini/operator@example.com"
  windows:
    primary: "gemini-pro"
    secondary: "gemini-flash"
    tertiary: "gemini-flash-lite"
  members:
    "gemini-flash": ["gemini:gemini-3-flash-preview", "gemini:gemini-3.5-flash"]
```

A slot named this way leaves the worst-of computation above. `members`
says which Health Key draws on it — a Feed Offering's own id, or a
Declared Offering's Alias, matched EXACTLY, never as a glob, a prefix or
a regular expression: `gemini-3*-flash*`, written for Flash, also
matches `gemini-3.1-flash-lite` (measured 2026-07-29). Only a Route whose
Health Key `members` lists binds on the slot; a slot left unnamed in
`windows` stays a parent window and keeps binding every Route, exactly as
before. Where every slot is named, the Allowance publishes no Headroom of
its own — correct, since nothing then caps it as a whole.

Two shapes need a word each, and both come from the same measurement: the
operator's own free Gemini plan, mapped 2026-07-29.

**A slot nobody draws on keeps its entry, with an empty list.** The free
plan includes no Pro, so no Pro model is admitted, and `primary` reads
100% used. An absent slot key and an empty list mean opposite things:
absent says "nobody has assigned this yet" and fails
`headroom.member.empty`; an empty list says "nothing admitted draws on it"
and passes. Do not drop the slot from `windows` to make the check pass. An
undeclared slot rejoins the parent's worst-of computation, so the whole
Allowance would report exhausted while Flash and Flash Lite sit untouched.

**A Health Key that draws on no published window goes in `unmeasured`.**
The same account serves Gemma, and codexbar publishes no window for it.
Leaving the key out fails `headroom.member.unclaimed`, which is right to
fire — an unassigned key is normally a gap. Listing it under a slot states
something false. So state it:

```yaml
  unmeasured:
    - "gemini:gemma-4-26b-a4b-it"
    - "gemini:gemma-4-31b-it"
```

Every Route on such a key reads `headroom: null` either way. The list is
how the operator says the silence is deliberate, and `doctor` reports an
entry naming no known Health Key.

Since 2026-07-29 (ticket 10) `members` reaches a Feed provider's own
Offerings, not only a Declared Offering: `gemini` running `mode: all`
from the Feed can now claim a slot the same way a hand-declared Offering
always could, keyed by its own Offering id. `policy.example.yaml` still
maps Gemini through three Declared Offerings, because a public example
needs no live Feed credential; the mechanism is the same either way.

## Verify a source against the provider's own dashboard before you map it

`doctor` reports a mapping that broke. Nothing reports a source that was
wrong all along. Check each provider by hand, once, before you map it, and
compare a drawn Allowance against an untouched one.

Measured 2026-07-29 against the operator's own dashboards:

```
claude       89% weekly, 19% 5-hour, 61% fable    exact
clinepass    100% monthly, 0% on two others       exact
codex        0% used                              exact
codex        reset time                           SYNTHESISED
opencodego   0% weekly                            WRONG: the dashboard read 43%
gemini       Pro / Flash / Flash Lite             a shape the dashboard does not show
```

**A source reading `(local)` is worth confirming, never assuming wrong.**
`codexbar --provider opencodego` states `(local)` as its source, so it
reports what the CLI on this machine recorded rather than what the server
counts. Its weekly figure read 0% against a dashboard read minutes earlier
at 43%, and the provider was unmapped on that evidence. The operator then
checked, and codexbar was right. It is mapped again.

Two lessons, and the second matters more. A local source is worth
confirming before you trust it. And one disagreement is not proof: read
both again, at the same moment, before you act on the difference.

**Codex states a reset it computes.** `resetsAt` reads exactly
`updatedAt` plus the window, recomputed on every call, so it moves 15
minutes further out on each refresh and never arrives. The dashboard
stated 12:12 where codexbar stated 10:16Z. The USED figure is correct;
the reset is not. Read `resets_at` from a provider that states a real one
— Claude states `Jul 30 18:59Z` and the dashboard agrees to the minute.

**Gemini's slots are not the dashboard's windows.** The dashboard states a
5-hour window and a weekly window, both 0%. codexbar states three
per-model figures instead, and its Pro figure of 100% corresponds to
nothing the dashboard shows: the free plan grants no Pro at all. The Flash
figures do agree with the dashboard's 0%, so a Flash Route reads
correctly. Nothing captures the aggregate windows the dashboard shows,
because codexbar does not publish them for this provider.

**A Headroom states what is spent. It never states how fast.** The
operator's `claude-fable-5` may draw at most half the Claude weekly
allowance, and it draws faster per token than its siblings. Measured
2026-07-29: the weekly window read 89% and fable's own window read 61%, so
every Claude Route bound at 89% and fable looked no worse than the rest.
Reading that as "fable is as safe as the others" is wrong, and no figure
here can say so. Price the output, not the Headroom.

## A stale Tier reads exactly like a current one

`policy.allowances.<id>.tier` states the Tier an Allowance bills under
(CONTEXT.md, "Tier"). It is the operator's own word, typed once, and
nothing in this project reads it back against anything live.

The hazard is the same standing one the `members` mapping carries: a
fact the operator states once and the world can change underneath.
Move a Claude Allowance from Tier `claude-max-5x` to Tier `claude-max-1x`,
forget to edit `allowances`, and every Headroom for that Allowance keeps
reporting a share of the Tier it no longer holds. A Reading of 50% used
against `claude-max-5x` is five times the work of 50% against
`claude-max-1x`, and the figure alone cannot tell you which Tier you are
looking at.

`doctor` reports only what it can see: an `allowances` entry naming an
`allowance_id` no Offering reaches (a typo, or a provider removed from
Policy). It cannot check whether the STRING still names the right Tier,
because nothing this project calls states a Tier a caller could compare
it against — the same reason `codexbar`'s own `loginMethod` field cannot
stand in for one (see the ticket that added this field: `"6"` for
Claude, `"team"` for Codex, `"Free"` for Gemini, `"API key"` for
ClinePass — none of those names a Tier).

Re-check `allowances` by hand after every Tier change on every mapped
credential. Nothing here will ever remind you.

`doctor` reports one thing it CAN see, since 2026-07-29:
`allowances.tier_unstated.<allowance_id>` fires where an Allowance
publishes a Headroom and `allowances` names it nowhere. Reported by an
agent consumer: 24 of 40 Routes carrying a Reading carried no Tier, and 20
of the 24 were one Allowance.

**A present `allowances` entry silences it, with or without `tier`.** That
is the only way to tell two states apart, and neither a Route nor
`entitlements` can express it — both publish `tier: null` for each. An
entry with no `tier` key says the operator looked and found no
subscription level to state. No entry at all says nobody looked. Same rule
as an empty `members` list: silence stated is not silence by default.

The check skips an Allowance no Offering reaches. A Tier there describes
nothing, and the mapping itself is the fault.

## codexbar drops an extra window between calls, and restores it

An `extraRateWindows` entry is not stable across consecutive calls.

Measured 2026-07-29 on the operator's Claude Allowance:
`claude-weekly-scoped-fable` was present, absent, then present again
across three `codexbar --format json` calls about one minute apart. No
Policy changed. No vendor release landed. Repeating the call with
`--provider claude` alone reproduced both answers.

Two things follow, and both are already handled:

- **`headroom refresh` writes what it read.** A Reading taken during a
  drop holds no extra window, so every Offering that draws on it reports
  no Headroom until the next Reading catches the window again. It reports
  no Headroom; it never reports free capacity.
- **`doctor` does not fail a `members` key on the flap.**
  `headroom.member.unreachable.<id>.<key>` passes when the live Reading
  omits a key and Headroom State holds a Reading that publishes it, and
  its detail says so. It still fails when neither publishes the key,
  which is the real rot this check exists to catch.

Do not correct a `members` line because one call disagreed with it. Run
the command again first.

A related earlier measurement, on 2026-07-28: codexbar published
`claude-weekly-scoped-all-model` at 18:48Z and had dropped it by 20:52Z.
That looked like a vendor removing a window. Read against the 2026-07-29
measurement, it is more likely the same flap.

## An Allowance rollup can be pessimistic, not only optimistic

A Sub-allowance usually binds TIGHTER than its parent. `claude-fable-5` can
drain while the Claude weekly quota still reports room, so a reader that
takes the Allowance figure gets a Route that is about to refuse. That
direction is documented above.

The reverse happens too, and it costs capacity rather than a refusal.

Measured 2026-07-29 on a free Gemini plan. `provider:gemini` reports
`primary.used_percent: 100.0`. That window is the Pro quota, the plan
includes no Pro at all, and both Pro Offerings are Withheld:

```
gemini:gemini-2.5-pro: 429 quota exceeded — Pro tier needs Cloud billing
gemini:gemini-3.1-pro-preview: 429 quota exceeded — Pro tier needs Cloud billing
```

Three admitted Flash and Flash Lite Routes on the same Allowance read 0%.
So the Allowance looks exhausted while every Offering it actually serves is
untouched. Excluding it discards working capacity.

`binding: null` already stated that nothing caps the Allowance as a whole.
An agent consumer read a non-null `primary` beside it and missed the point,
which is fair: a 100% figure is louder than an absent one.

So each window now states `admitted_members`, the Health Keys that draw on
it. Read it before you act on any Allowance-level figure:

```
litellm-maintainer entitlements --json | jq -r '.entitlements[]
  | .allowance_id as $a | .headroom // empty
  | (.primary, .secondary, .tertiary, (.extra_windows[]?.window))
  | select(. != null)
  | select(.used_percent > 0 and .admitted_members == [])
  | "\($a)\t\(.sub_allowance_id)\t\(.used_percent)%"'
```

**Walk `extra_windows` too.** A Sub-allowance named directly, rather than
through a slot, lives there and carries both fields. The first published
version of this query read the three slots only, so a pessimistic extra
window was invisible to it (reported 2026-07-29). `claude-weekly-scoped-fable`
is one: remove the Declared Offering that draws on it and leave the
`members` entry, and it reads `admitted_members: []`.

**An empty LIST and `null` are different claims.** An empty list means
Policy declares members for this window and none is admitted. `null` means
Policy declares no membership at all, which is the ordinary case: a parent
window governs every Offering on the Allowance. The first draft of that
query used `length == 0`, which matched both, and it named five healthy
windows as idle capacity.

`admitted` here means what Policy admits, so an Excluded Offering counts.
Reading only what is answering would empty a window's list the moment its
one model failed a Probe.

### The query cannot see an Allowance that admits nothing

It reports a window whose Policy DECLARES membership. Where Policy declares
none, `admitted_members` is `null` and the query is silent — correctly, since
a parent window governs every Offering on the Allowance.

That leaves one shape unreported: an Allowance whose parent window reads full
while Policy admits nothing at all. Measured 2026-07-29:

```
provider:cline-pass   used_percent 100.0   state: empty   in_scope: 0
                      all 11 Offerings Withheld
                      "subscription ending, renewal unconfirmed"
```

Nothing is discarded here, because there is nothing to spend. The hazard is
what a silent detector implies. A reader who runs the query, sees one row,
and then meets 100% on `cline-pass` concludes it is live capacity that is
spent — rather than capacity that no longer exists.

`in_scope` answers it, and no Headroom field does:

```
litellm-maintainer entitlements --json | jq -r '.entitlements[]
  | select(.in_scope == 0)
  | "\(.allowance_id)\tadmits nothing\t\(.withheld) withheld"'
```

**The Reading is not suppressed on such an Allowance, and will not be.**
Suppressing it would publish `headroom: null`, which means UNMEASURED — and
this Allowance is measured. The operator mapped it, the source reads it, and
`doctor` checks that mapping. Hiding a true figure to spare a reader one join
is the same trade this project refused for the Gemini Pro window, and the
answer is the same: state the fact, never hide the measurement.

## Writing the refresh plist is not installing it

`headroom install` writes the launchd plist and prints a `launchctl load`
command. It never calls `launchctl` itself. So the install looks complete,
and nothing runs.

Measured 2026-07-29. The plist sat on disk for five hours. Headroom State's
`read_at` never moved past the last manual refresh. Every Reading kept
publishing, `age_seconds` grew, and no figure was wrong — only old.

Two things hid it:

- **A stale Headroom fails nothing.** `guidance` and `entitlements` publish
  the old figures with a warning in `warnings`, which a caller reading a
  single field never sees.
- **`doctor` exited 0.** Its one related check, `headroom.refresh_interval`,
  reported "not checked: no headroom-refresh job is installed" and passed.
  A check that cannot measure must not fail — correct on its own, and it
  meant an absent job produced no failing check anywhere.

`doctor` now fires `headroom.refresh_current` where any mapped Allowance
has not refreshed in `HEADROOM_STALE_MULTIPLIER` intervals. Confirm an
install with:

```
launchctl list | grep headroom-refresh
```

An empty result means the job is not registered, whatever is on disk.

A downstream consumer reported its Readings as coming "from the installed
`headroom refresh` job". They came from a manual run. The figures were real
and current when read, so the findings held — but nobody had verified the
pipeline ran on its own, and the claim went unchallenged for a day.

## A launchd job inherits almost nothing, and one failure is silent

The refresh job runs a third-party binary. It gets neither your shell's
`PATH` nor your login environment, and each gap fails differently.

Measured 2026-07-30, on one machine.

**No `PATH` holding the binary. Loud.** `headroom.command: "codexbar"`
resolved from a shell and not from the job. The job ran every 15 minutes
for 17 hours, failed every run with
`[Errno 2] No such file or directory: 'codexbar'`, and Headroom State never
moved past the last manual refresh.

**No `USER`. Silent, and worse.** With `USER` unset, `codexbar` returned
`usage: null` for the Claude provider in 10 of 10 runs, and a figure in 10
of 10 runs with it set. Nothing else changed. The refresh then reported:

```
updated 5 of 6 mapped Allowances
```

No error, and no name. A Reading that cannot be read keeps the previous
one, by design — so one Allowance quietly froze while the other five stayed
current, and the count was the only tell.

Two defences, and use both:

- **State `headroom.command` as an absolute path**, never a bare name.
- **`headroom install` writes `USER` and `PATH` into the plist**, from the
  environment you install from. Reinstall after changing either.

Reinstalling is not enough on its own. `launchctl` reads the plist at load
time, so unload and load again:

```
litellm-maintainer headroom install --policy $LITELLM_MAINTAINER_HOME/policy.yaml \
  --target-dir ~/Library/LaunchAgents --env <absolute path>
launchctl unload ~/Library/LaunchAgents/no.tallmaker.litellm-maintainer.headroom-refresh.plist
launchctl load -w ~/Library/LaunchAgents/no.tallmaker.litellm-maintainer.headroom-refresh.plist
```

Then prove it, rather than assuming:

```
launchctl kickstart -k "gui/$(id -u)/no.tallmaker.litellm-maintainer.headroom-refresh"
cat $LITELLM_MAINTAINER_HOME/state/headroom-refresh.out.log
```

It must read `updated N of N`. Any smaller number names no Allowance, so
compare `read_at` per record in `state/headroom.json` to find which froze.

`doctor.headroom.binary` does not catch either fault. It resolves the
command against YOUR `PATH`, not the job's, so it answers "can I run this"
where the question is "can the job run this".

## launchd stops respawning a job that exits EX_CONFIG

Fixing the code does not restart the schedule. You must reload the job.

Measured 2026-07-30. A syntax error in `policy.py` made the CLI fail at
import. The tick job exited, launchd recorded `78: EX_CONFIG`, and it
stopped scheduling the job entirely. The fault was corrected minutes
later. The tick stayed dead for three hours.

Two things hid it, and each is worse than it sounds.

**`runs.log` went silent rather than reporting.** The failure was at
import time, so no project code ran and nothing could append a line. A
reader tailing the log sees entries simply stop, which reads as a
scheduler that is not firing rather than a program that will not start.

**`launchctl list` showed the job present.** It reports the LAST exit
code, so `78` sat there looking like a historical failure. Only
`launchctl print` names the real state:

```
launchctl print "gui/$(id -u)/no.tallmaker.litellm-maintainer.tick"
```

```
state = spawn scheduled
runs = 2581
last exit code = 78: EX_CONFIG
```

`spawn scheduled` with a stale run count and `EX_CONFIG` means parked,
not running. The project never returns 78 itself: that is launchd's own
reading of a process that died before it could do anything.

Two defences are now in place.

`litellm_maintainer.tick_entry` is what both plists invoke, never
`litellm_maintainer.cli`. It imports the standard library and `paths` —
nothing that can carry a project fault — then imports the CLI inside a
`try`. A failure appends one line to `runs.log` naming the exception and
telling the reader to reload the job, then re-raises so the traceback
still reaches `tick.err.log`.

`doctor` fires `headroom.refresh_current` when a mapped Allowance stops
refreshing, which catches the headroom job going the same way.

To repair a parked job:

```
launchctl unload ~/Library/LaunchAgents/no.tallmaker.litellm-maintainer.tick.plist
launchctl load -w ~/Library/LaunchAgents/no.tallmaker.litellm-maintainer.tick.plist
```

Then confirm the exit code is 0, and that `runs.log` gains a line within
one interval.

## litellm needs a fastapi its own version range admits but breaks on

Warning: pin fastapi when you install litellm. An unpinned upgrade
starts a proxy that cannot import.

litellm imports `get_flat_dependant` from `fastapi.dependencies.utils`.
fastapi removed that name in 0.140.7. litellm declares
`fastapi>=0.136.3,<1.0`, so the range admits every version that fails.
`uv tool upgrade litellm` resolves fastapi to the newest release and the
proxy dies at startup.

Read the FIRST traceback. The real fault is:

```
ImportError: cannot import name 'get_flat_dependant' from 'fastapi.dependencies.utils'
```

A second traceback follows it and hides it:

```
ModuleNotFoundError: No module named 'proxy_server'
```

That second message names no dependency, so it sends the reader to the
config or the install layout instead of to fastapi.

Install the tool with the pin. The receipt keeps it, so a later upgrade
holds the pin too:

```
uv tool install "litellm[proxy]==1.97.0" --with "fastapi==0.140.6" --force
```

Confirm the pin with `cat ~/.local/share/uv/tools/litellm/uv-receipt.toml`.

Keep the pin until litellm stops importing the name. Test the installed
tree before you remove it:

```
grep -rn "get_flat_dependant" ~/.local/share/uv/tools/litellm/lib/python*/site-packages/litellm/
```

An empty result means litellm no longer needs the old fastapi.

## doctor reads the litellm on PATH, and a project venv shadows it

Warning: run `doctor` from this repo with `--litellm-path`. Without it,
`doctor` can read the wrong litellm and report both patches missing.

`doctor` finds the tree from the `litellm` executable on `PATH`
(`litellm_patches.litellm_source_root`). The `dev` extra installs
litellm into the repo's `.venv`, and `uv run` puts `.venv/bin` ahead of
`~/.local/bin`. So one `uv run --extra dev python -m pytest` is enough
to make every later `uv run litellm-maintainer doctor` inspect the
repo's test copy.

The test copy is stock litellm and carries no patch. `doctor` then
prints the same two `[FAIL]` lines a real patch loss prints. Nothing in
the wording separates the two cases.

Name the proxy's own tree:

```
litellm-maintainer doctor --litellm-path ~/.local/share/uv/tools/litellm/lib/python3.13/site-packages/litellm
```

Read the path in the failure before you re-apply a patch. A path under
the project directory is this trap, not a patch loss. Confirm it with:

```
uv run python -c "import shutil; print(shutil.which('litellm'))"
```
