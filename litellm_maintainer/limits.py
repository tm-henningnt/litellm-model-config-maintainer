"""Stated Limits: the token figures a source states for an Offering.

See CONTEXT.md, "Stated Limit", and
`docs/adr/0006-a-stated-limit-comes-from-a-source.md` — the reason this
module does not mirror `pricing.cost_model_info`'s native-prefix
suppression.

Every function here is a pure transform. It takes an Offering, or the
entries `plan.py` has already built, and returns a value. It performs no
network call, no filesystem read and no clock read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from litellm_maintainer.feed import Offering

# The Feed's field names, and the `model_info` key each one becomes.
# litellm reads `max_input_tokens` and `max_output_tokens` from
# `model_info` for its model listing, `/model/info`, `/model_group/info`
# and budget reservation. Neither ever changes a request body.
_FIELD_TO_MODEL_INFO_KEY = {
    "context_tokens": "max_input_tokens",
    "max_output_tokens": "max_output_tokens",
}

# `model_info.max_tokens` is deliberately never written, and this map is
# the only source of keys. litellm's `trim_messages` falls back from
# `max_input_tokens` to `max_tokens`, so stating both makes the pair
# ambiguous for a reader and for litellm.


def _stated(value: Any) -> int | None:
    """`value` as a positive int, or `None` when the source states nothing.

    The Feed publishes `null` for a figure it does not know, and has been
    seen to publish a figure for one field and not the other. A `bool` is
    rejected even though Python counts it as an int: `True` is not a
    token count.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value <= 0:
        return None
    return value


def limits_model_info(offering: Offering) -> dict[str, Any]:
    """The Stated Limits of one Offering, as `model_info` keys.

    Writes each key only when the Feed states a positive int for it, so a
    stated context window is not withheld because the output figure is
    absent, and an absent figure reads as unknown rather than as small.
    Returns `{}` when the Feed states neither.

    Unlike `pricing.cost_model_info` this does not suppress itself for a
    native litellm prefix. Cost is suppressed there because litellm
    prices such a model correctly. Limits are not: litellm resolved
    `openrouter/anthropic/claude-opus-5` — a native prefix — to
    200000/64000 against the Feed's 1000000/128000, from a regex rule
    over model names. A native prefix does not imply litellm knows the
    window. See ADR 0006.
    """
    info: dict[str, Any] = {}
    for field, key in _FIELD_TO_MODEL_INFO_KEY.items():
        stated = _stated(offering.limits.get(field))
        if stated is not None:
            info[key] = stated
    return info


@dataclass(frozen=True)
class LimitCollision:
    """Two or more Aliases that share a model string and disagree on limits.

    litellm holds one cost-map entry per `litellm_params.model`, so the
    last entry written defines every sibling. Measured on 2026-07-26: an
    entry carrying limits replaced its sibling's correct figures, and an
    entry carrying no limits at all inherited its sibling's.

    Reported, never refused. The condition is litellm handling something
    badly rather than an error in Policy, and a legitimate config reaches
    it — two ChatGPT seats point at one model string, and so do a
    Client-Facing Variant and its plain sibling.
    """

    model: str
    # Alias to its stated limits, in the order the entries were built. The
    # last one is the one litellm keeps.
    stated_by_alias: tuple[tuple[str, dict[str, Any]], ...]

    @property
    def winner(self) -> str:
        """The Alias whose figures litellm keeps: the last one registered."""
        return self.stated_by_alias[-1][0]

    @property
    def message(self) -> str:
        stated = "; ".join(
            f"{alias} states {info or 'none'}" for alias, info in self.stated_by_alias
        )
        return (
            f"Stated Limit collision on {self.model!r}: {stated}. litellm holds "
            f"one entry per model string, so {self.winner!r} defines them all. "
            "State the same Stated Limit on every Alias that shares a model "
            "string, or none."
        )


def find_limit_collisions(
    entries: list[dict[str, Any]],
) -> tuple[LimitCollision, ...]:
    """Built entries that share a model string and disagree on a Stated Limit.

    Reads the entries `plan` has already built, Declared and Discovered
    alike, so the check sees the same facts that reach the file rather
    than re-deriving them.

    Silent when the siblings agree, because agreement is the normal case
    and a warning there is noise an operator learns to ignore. An entry
    that states no limits still counts as a disagreement against a
    sibling that states some: the silent one inherits.
    """
    by_model: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for entry in entries:
        model = (entry.get("litellm_params") or {}).get("model")
        if not isinstance(model, str):
            continue
        alias = entry.get("model_name")
        if not isinstance(alias, str):
            # Every built entry carries one. Skip rather than report a
            # collision against an Alias this function had to invent.
            continue
        info = entry.get("model_info") or {}
        stated = {
            key: info[key]
            for key in _FIELD_TO_MODEL_INFO_KEY.values()
            if key in info
        }
        by_model.setdefault(model, []).append((alias, stated))

    collisions = []
    for model, stated_by_alias in sorted(by_model.items()):
        if len(stated_by_alias) < 2:
            continue
        distinct = {tuple(sorted(info.items())) for _, info in stated_by_alias}
        if len(distinct) < 2:
            continue
        collisions.append(
            LimitCollision(model=model, stated_by_alias=tuple(stated_by_alias))
        )
    return tuple(collisions)
