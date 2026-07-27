# A Stated Limit comes from a source, never from a name

The Generated Config carries each Offering's Stated Limit in
`model_info`. The Feed states one for a Discovered Offering. The operator
states one for a Declared Offering. We write a figure only when a source
states it. We never derive one from a model name, a model family, or a
table in this repository.

This ADR records one asymmetry. `pricing.cost_model_info` suppresses
itself for a native litellm prefix, and `limits.limits_model_info` does
not. That looks like an inconsistency, so the next reader will otherwise
remove it.

## Why cost suppresses itself and a Stated Limit does not

Cost suppression rests on a true claim. litellm prices a native prefix
such as `openrouter/` or `groq/` correctly, from its own map. Our own
rate would add nothing there. It could also disagree with the spend
report.

The same claim about a Stated Limit is false. Measured on 2026-07-26
against the installed litellm:

| `litellm_params.model` | exact cost-map entry | resolves to |
| --- | --- | --- |
| `anthropic/claude-opus-5` | hit | 1000000 / 128000 |
| `openrouter/anthropic/claude-opus-5` | miss | 200000 / 64000 |
| `cline/anthropic/claude-opus-5` | miss | 200000 / 64000 |
| `openai/claude-gpt-5.6-sol` | miss | 200000 / 64000 |

The Feed states 1000000 / 128000 for the OpenRouter route, and
Anthropic's own documentation agrees. So a native prefix does not imply
that litellm knows the window.

## Where the wrong number comes from

litellm's cost map ships a `fallback_generalizations` block. A capability
rule pairs a regex with a `model_info`. The rule `claude-family-baseline`
matches `claude-[a-z]+-\d+(?:[-.]\d+)?` and asserts `max_input_tokens:
200000`, `max_output_tokens: 64000`. Any Claude-shaped model absent from
the exact map inherits it.

Our own Alias convention makes this worse. Every Alias here starts with
`claude-`. So a ChatGPT seat model such as `claude-gpt-5.6-sol` matches
the Claude rule. It inherits a figure about a different vendor.

Inside the proxy the symptom changes. `Router` startup writes an
all-null cost-map entry for every deployment model that is not already an
exact key. A rule runs only after an exact lookup misses, so that null
entry shadows the regex. The same route therefore reports 200000 outside
the proxy and nothing inside it. Neither reading is the Feed's.
`docs/gotchas.md` holds the reproduction.

## What we write, and what we refuse to write

We write `model_info.max_input_tokens` and
`model_info.max_output_tokens`, and only when the Feed states a positive
integer. A missing, `null` or non-positive figure is unstated, and the
key is absent. Absence must read as unknown, never as small.

We never write `model_info.max_tokens`. litellm's `trim_messages` falls
back from the input limit to `max_tokens`, so stating both makes the pair
ambiguous.

We never write a limit into `litellm_params`. A `max_tokens` there is sent
to the provider and caps every caller that omits one. A
`max_input_tokens` there is read by nothing and travels as an unknown
completion keyword.

## Consequences

A Stated Limit is metadata. It feeds the proxy's model listing,
`/model/info`, `/model_group/info` and budget reservation. It never
changes a request body.

We do not emit `router_settings` and we do not enable
`enable_pre_call_checks`. That switch is the only one that makes a Stated
Limit change behaviour instead of reporting. With it on, the router counts the
tokens of every request. It then refuses one that exceeds the figure. A
wrong Feed figure would become an outage rather than a wrong number in a
listing.

An Offering the Feed does not describe carries no Stated Limit. The
operator may state one on a Declared Offering. Nothing derives one.
