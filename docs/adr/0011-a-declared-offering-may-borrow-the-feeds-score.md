# A Declared Offering may borrow the Feed's score, never its limit

A Declared Offering reaches the proxy through a route the Feed does not
publish: a private regional host, a subscription seat, a direct vendor
key. So the Feed states no score for it, no token rate, and no pricing
kind. `guidance` reported all three as absent, and reported the cost basis
as `unknown`.

Both statements were about the FEED's coverage. Neither was about the
model. The effect was concrete, measured 2026-07-28 on the operator's own
instance: 23 Aliases — including every ChatGPT seat and both direct Claude
models, which are the strongest models the proxy serves — sorted below
every scored free model, and read as billable to an agent that follows
the `model-routing` skill's rule to treat `unknown` as spend.

## The decision

A Declared Offering may name a **Reference Model**: the Canonical Model id
the Feed publishes for the SAME model. Its Route then joins that model's
Guidance Row, and the row takes the Feed's score, display name and
capabilities from the Reference Model.

Policy states two further values the Feed cannot know: `cost_basis`, and a
`pricing` block of token rates.

## Why the score transfers and the limit does not

A quality score describes the model. Two providers serving one model
score the same, which is why a Guidance Row is a Canonical Model and not
an Alias. So the Feed's score for `openai/gpt-5.6-sol` is a statement
about the model a ChatGPT seat reaches, whatever route reaches it.

A limit describes the endpoint. Measured 2026-07-27 by bisection: the
ChatGPT subscription backend accepted 369,603 tokens and refused about
398,000, while the Feed states 1,050,000 for the same model on the OpenAI
API. Borrowing that figure would trade an early compaction for a hard
refusal. ADR 0006 already forbids deriving a Stated Limit from anything
but a source, and the source for this endpoint is the operator's own
measurement.

So a Reference Model supplies a score. It never supplies
`max_input_tokens` or `max_output_tokens`.

## Why a rate is a reference, not a bill

A mirror's rate is what ANOTHER vendor charges for the model. It ranks the
relative burn, which is the number that matters on a flat-rate pool: a
subscription holds a finite allowance, and each model draws on it at its
own rate. Measured 2026-07-28 on one pool, `kimi-k3` drew 15.00 per 1M
output tokens against `mimo-v2.5`'s 0.28 — 53.6 times the burn for 34%
more score. Without any rate at all, an agent ranking by score alone
drained that pool in one day.

Two rules keep the figure honest:

- Read only a `paid` mirror. A free mirror states 0.00, which describes
  that mirror's promotion. Reading it would report every model with a free
  tier as costless everywhere, and nothing in the output would show it.
- Read the cheapest paid mirror. Mirrors disagree: `openai/gpt-5.6-terra`
  read 1.25/7.50 through one provider and 2.50/15.00 through another on
  2026-07-28, and the lower pair matched the vendor's own published price.

`rate_is_list_price` already marks a `free` or `flat_rate` rate as a list
price rather than an amount billed, which is the same distinction
`pricing.py` writes into the Generated Config.

## Every number states its source

A row carries `score_source` and each Route carries `rate_source`. The
values are `feed`, `reference` and `operator`.

This is not decoration. A caller weighs the three differently, and the
alternative is a figure that looks measured and is not. `capabilities_are_operator_stated`
already set this precedent for one field; these two extend it to the rest.

`why` says the same thing in words, so an operator reading the text output
sees it without knowing the field names.

## No cost metadata reaches the Generated Config

A Declared Offering's `model_info` reaches the Generated Config verbatim,
and nothing added here changes that. The `pricing` block feeds `guidance`
only.

A flat-rate rate summed into litellm's spend log would report a bill that
nobody was sent. Revisit this for a `metered` Declared Offering, where the
rate IS a bill; the operator has none today.

## Consequences

A Reference Model the Feed drops yields no score, and `guidance` warns.
That warning is easy to miss — the row still appears and still answers —
so `doctor` carries a check per Reference Model, alongside the check for a
stale Withheld line.

The guidance schema version rose to `3`.

Nothing here changes what the proxy serves, what the Prober probes, or
what a call costs. It changes what a caller is told.
