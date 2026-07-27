"""Alias derivation.

See CONTEXT.md, "Alias", and the spec's "Naming" section. `derive_alias`
holds the mechanical rule. `alias_for` applies it to one Discovered
Offering, honouring a Policy override first.

Moved here from `litellm_maintainer.policy`. `policy.py` re-exports both
names, so an existing `from litellm_maintainer.policy import
derive_alias` still works. Import `Policy` only for type checking, to
avoid a circular import: `policy.py` imports this module at run time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm_maintainer.policy import Policy


def derive_alias(provider_label: str, provider_model_id: str, alias_prefix: str) -> str:
    """Derive an Alias from a provider label and a model identifier.

    The rule, in four steps. Warning: apply the steps in this order.

    1. Lowercase the model identifier.
    2. Keep only the part after the last `/`. This drops a vendor path
       segment such as `google/` or `nvidia/`.
    3. Replace each `:` with `-`. A variant marker such as `:free`
       becomes an ordinary token.
    4. Split on `-`. Drop a leading or a trailing token that the
       provider label already contains. Do not drop a token in the
       middle.

    Join the surviving tokens with `-`. The Alias is
    `alias_prefix + provider_label + "-" + the joined tokens`.

    Step 4 stops a repeat. The label `gemini` plus the model
    `gemini-3.5-flash` gives `claude-gemini-3.5-flash`, not
    `claude-gemini-gemini-3.5-flash`. The label `cline-free` plus the
    model `google/gemma-4-31b-it:free` gives
    `claude-cline-free-gemma-4-31b-it`.

    The rule reproduces all 68 of the operator's Discovered Aliases.
    `alias_overrides` is empty. See `tests/test_naming_and_collisions.py`
    and `tests/test_policy.py`.
    """
    tokens = [
        token
        for token in provider_model_id.lower().split("/")[-1].replace(":", "-").split("-")
        if token
    ]
    label_tokens = set(provider_label.lower().split("-"))
    while tokens and tokens[0] in label_tokens:
        tokens.pop(0)
    while tokens and tokens[-1] in label_tokens:
        tokens.pop()
    return f"{alias_prefix}{provider_label}-" + "-".join(tokens)


def alias_for(policy: "Policy", offering_id: str) -> str:
    """Return the Alias for a Discovered Offering.

    Use the `alias_overrides` entry when Policy holds one. Otherwise
    derive the Alias with `derive_alias`. `offering_id` has the form
    `<provider>:<provider_model_id>`.
    """
    override = policy.naming.alias_overrides.get(offering_id)
    if override is not None:
        return override
    provider_id, _, provider_model_id = offering_id.partition(":")
    label = policy.naming.provider_labels.get(provider_id, provider_id)
    return derive_alias(label, provider_model_id, policy.naming.alias_prefix)
