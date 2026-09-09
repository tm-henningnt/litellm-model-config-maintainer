# LiteLLM 1.99/1.100 and the OpenCode session header

Research date: 2026-09-08

Scope: comparison of the local patch file, the checked-in proxy header policy,
LiteLLM's official v1.99.0 and v1.100.0 release notes and tagged source, the
actual LiteLLM 1.100.0 Docker image used by this repository, and first-party
OpenCode material. A patched image was built and the proxy stack was restarted
under OrbStack; no real provider request or live upstream packet capture was
performed.

## Findings

### ChatGPT non-streaming request intent: the local patch still appears needed

Both [v1.99.0](https://docs.litellm.ai/release_notes/v1.99.0/v1-99-0) and
[v1.100.0](https://docs.litellm.ai/release_notes/v1.100.0/v1-100-0) retain the
same relevant sequence in the [v1.99.0 HTTP
handler](https://github.com/BerriAI/litellm/blob/v1.99.0/litellm/llms/custom_httpx/llm_http_handler.py#L2595-L2614)
and [v1.100.0 HTTP
handler](https://github.com/BerriAI/litellm/blob/v1.100.0/litellm/llms/custom_httpx/llm_http_handler.py#L2626-L2660): it captures `stream`, transforms the request, folds the transformed
`data["stream"]` back into `stream`, and then uses that value as the handler's
streaming decision. The [v1.99.0 ChatGPT
transform](https://github.com/BerriAI/litellm/blob/v1.99.0/litellm/llms/chatgpt/responses/transformation.py#L61-L84)
and [v1.100.0 ChatGPT
transform](https://github.com/BerriAI/litellm/blob/v1.100.0/litellm/llms/chatgpt/responses/transformation.py#L64-L87)
remain provider-forced streaming.

That means a caller that did not request streaming can still be treated as a
streaming caller after the provider transform. The local
`client_requested_stream` patch in
[`docker/apply_litellm_patches.py`](../../docker/apply_litellm_patches.py)
still addresses the right boundary. Neither release note identifies this
request-intent fix. The conclusion is based on tagged-source inspection, not a
runtime test.

### Anthropic usage-only SSE chunk: source and Docker artifact differ

The original failure was an Anthropic streaming chunk with `choices=[]` (for
example, a final usage-only chunk) reaching code that indexed `chunk.choices[0]`.
LiteLLM's official commit
[`0b809cf7`](https://github.com/BerriAI/litellm/commit/0b809cf7d68e368b7a7ee90d7134c4841c944e3c)
adds choiceless-chunk handling, including merging usage into a held stop chunk,
and guards both sync and async iterators before the normal `choices[0]` path.
That code is present in the official [v1.99.0
tag](https://github.com/BerriAI/litellm/blob/v1.99.0/litellm/llms/anthropic/experimental_pass_through/adapters/streaming_iterator.py#L351-L369)
and [v1.100.0
tag](https://github.com/BerriAI/litellm/blob/v1.100.0/litellm/llms/anthropic/experimental_pass_through/adapters/streaming_iterator.py#L351-L369),
with the early guards in the [sync loop](https://github.com/BerriAI/litellm/blob/v1.100.0/litellm/llms/anthropic/experimental_pass_through/adapters/streaming_iterator.py#L528-L535)
and the corresponding async loop.

The tagged GitHub source contains this fix, but the official
`docker.litellm.ai/berriai/litellm:1.100.0` image inspected for this deployment
still has the old `chunk.choices[0]` sequence and does not contain the early
choiceless-chunk guard. The v1.99 and v1.100 release notes warn that Docker and
PyPI artifacts were built from different commits. Consequently, the local
patch remains required for the Docker artifact currently deployed here. The
v1.100 release note also lists
[PR #34382](https://github.com/BerriAI/litellm/pull/34382), but that is a
separate generic `stream_chunk_builder` fix for chunks without a `choices` key;
it should not be confused with the Anthropic adapter fix above. The local
patch marker/doctor bookkeeping was left in place and the patch still applies
cleanly to the 1.100.0 Docker artifact.

### OpenCode header and LiteLLM proxy operation

The first-party [OpenCode X post](https://x.com/opencode/status/2095410501400289576)
says that some OpenCode Go integrations are missing `x-opencode-session`, which
prevents prompt-cache optimization, and warns that missing-header requests may
start failing on 2026-09-06. OpenCode's first-party source corroborates that
this is an operational request header: its [Zen handler reads the exact
header](https://github.com/anomalyco/opencode/blob/dff8fbc149fb7492e4f07b713ac31ea70d9a541c/packages/console/app/src/routes/zen/util/handler.ts#L124-L135)
and preserves the `x-opencode-*` headers on new-inference forwarding
([header handling](https://github.com/anomalyco/opencode/blob/dff8fbc149fb7492e4f07b713ac31ea70d9a541c/packages/console/app/src/routes/zen/util/handler.ts#L230-L264)).

LiteLLM's [v1.99 proxy source](https://github.com/BerriAI/litellm/blob/v1.99.0/litellm/proxy/litellm_pre_call_utils.py#L1004-L1023)
and [v1.100 proxy source](https://github.com/BerriAI/litellm/blob/v1.100.0/litellm/proxy/litellm_pre_call_utils.py#L1034-L1053)
say that, when
`general_settings.forward_client_headers_to_llm_api` is enabled, it forwards
client headers selected by its `x-*` rule; `x-opencode-session` matches that
rule. See the [v1.99 forwarding switch](https://github.com/BerriAI/litellm/blob/v1.99.0/litellm/proxy/litellm_pre_call_utils.py#L1177-L1180)
and [v1.100 forwarding switch](https://github.com/BerriAI/litellm/blob/v1.100.0/litellm/proxy/litellm_pre_call_utils.py#L1207-L1210).
The checked-in example policy already enables this global switch at
[`policy.example.yaml`](../../policy.example.yaml#L622-L623).

Operationally:

- If the OpenCode request arrives with `x-opencode-session` and this forwarding
  setting remains enabled, LiteLLM should pass it through on the normal proxy
  path. No special static header is needed merely to preserve it.
- LiteLLM does not invent the per-session value. If the client/tool omits the
  header, a proxy setting cannot reconstruct the correct identity; a static
  `extra_headers` value would incorrectly share one session across requests.
- The maintainer's direct Prober and proxy smoke checker are synthetic clients:
  they now generate one fresh session id per OpenCode Go check. The direct
  Prober reuses it across a retry, while the smoke checker makes one request
  per selected rule and therefore needs no retry state.
- The current switch is broad: it forwards matching custom `x-*` headers to
  providers, not only this OpenCode header. If that breadth is unacceptable,
  forwarding should be scoped operationally/model-group-wise rather than
  silently assuming the header is present.

The current OpenCode core also uses other session/cache metadata such as
`x-session-affinity`, `X-Session-Id`, and `promptCacheKey` in one request path
([source](https://github.com/anomalyco/opencode/blob/dff8fbc149fb7492e4f07b713ac31ea70d9a541c/packages/core/src/session/runner/llm.ts#L204-L214)).
That does not prove that every OpenCode client path sends
`x-opencode-session`; the post's requirement should therefore be verified on
the actual client integration.

## Docker/OrbStack note

The [Dockerfile](../../docker/Dockerfile#L17) and
[compose file](../../docker/compose.yaml) now target LiteLLM 1.100.0. The v1.99
and v1.100 release notes publish Docker tags and warn that
their Docker and PyPI artifacts were built from different commits while
expected to be functionally equivalent. The actual image was checked inside
OrbStack: it reports LiteLLM 1.100.0, lacks the tagged-source Anthropic guard,
and accepts both local patches through the patch verifier. The proxy and
ChatGPT worker are running the patched 1.100.0 image; proxy liveness is HTTP
200.

## Research limitations

The research web reader returned `Internal Error ()` for the X URL. A direct
request to the first-party X page did return its HTML metadata, including the
post text, so the post was partially accessible; no authenticated X timeline
or replies were inspected. All LiteLLM/OpenCode corroboration above came from
official release notes or immutable first-party GitHub source/commit links.
No real provider request or upstream packet capture was performed; header
handling was verified against the running configuration and LiteLLM's
installed forwarding helper.
