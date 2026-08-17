"""Derives a starter Policy from a Feed snapshot.

Policy is hand-written by the operator (CONTEXT.md, "Policy"; ADR
0003). This module never writes Policy on its own. It only helps a new
operator produce a first draft: a Policy that names every provider the
Feed publishes, with the most conservative selection rule, so the
operator edits a working file instead of starting from a blank page.

`build_starter_policy` is pure. It reads a `feed.Feed` already loaded by
the caller and returns a `StarterPolicy` holding the YAML text. It
performs no file read, no network call and no clock read, so generating
twice from the same Feed gives byte-identical text (every collection is
sorted; nothing here reads a clock or an environment variable).

`write_starter_policy` is the one function in this module that touches
a file. It refuses to overwrite an existing Policy unless the caller
passes `force=True`, and it writes through a temporary file and a
rename, the same atomic-write pattern `fetch.py` and `health.py` use,
so a reader never observes a half-written Policy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import tempfile

from litellm_maintainer.feed import Feed

# The default Alias prefix. Matches `policy.example.yaml`'s own
# `naming.alias_prefix`, so a generated Policy reads like the example a
# new operator already copied from.
DEFAULT_ALIAS_PREFIX = "claude-"


@dataclass(frozen=True)
class StarterPolicy:
    """A generated, not-yet-reviewed Policy.

    `text` is the full YAML document, ready to write to disk. It always
    parses under `policy.parse_policy`. `provider_count` is the number
    of providers named in `text`. `notes` lists what the operator must
    still decide by hand; none of it is a defect in the generated file.
    """

    text: str
    provider_count: int
    notes: tuple[str, ...]


def _label_for(provider_id: str) -> str:
    """Derive a short, readable `naming.provider_labels` value.

    Take the part after the last `/`, so a vendor path segment drops
    the same way `naming.derive_alias` drops one from a model
    identifier. Lowercase it. Replace `_` with `-`, since a Policy
    label reads better as one token style throughout.
    """
    label = provider_id.strip().lower().split("/")[-1]
    return label.replace("_", "-")


def _credential_comment(provider_id: str, credential_hint: str | None) -> str:
    """One comment line naming the provider's credential variable.

    States the variable the Feed names for this provider, read from
    `feed.Provider.credential_hint` (`feed.py`). States plainly when
    the Feed names none, so the operator does not go looking for a
    hint that does not exist.
    """
    if credential_hint:
        return f"    # Credential: the Feed names {credential_hint} for {provider_id}."
    return f"    # Credential: the Feed states no credential_hint for {provider_id}."


def _provider_block(provider_id: str, credential_hint: str | None) -> str:
    lines = [
        f"  {provider_id}:",
        _credential_comment(provider_id, credential_hint),
        "    mode: all",
        "    entitlement: per_model",
    ]
    return "\n".join(lines)


def build_starter_policy(feed: Feed, *, alias_prefix: str = DEFAULT_ALIAS_PREFIX) -> StarterPolicy:
    """Derive a starter Policy from `feed`.

    Names every provider the Feed publishes (`feed.providers`), each
    with `mode: all` and `entitlement: per_model`. `per_model` is the
    Policy default (`policy.PER_MODEL`) and the reading that never
    over-claims (CONTEXT.md, "Entitlement"; ADR 0004), so a starter
    Policy never assumes a shared pool it has not confirmed.

    Never sets a `pricing` filter. `policy.example.yaml` explains why:
    an Offering with no stated price carries the pricing kind
    `unknown`, not `paid`, and a provider can be free to the operator's
    account for a reason the Feed cannot see (a subscription, a
    grandfathered plan). A pricing filter would admit nothing there. A
    comment in the generated file repeats this warning next to
    `providers`.

    Every collection is sorted before it is written, and nothing here
    reads a clock or an environment variable. Two calls on the same
    Feed produce byte-identical `text`.
    """
    provider_ids = sorted(feed.providers)

    provider_blocks = "\n".join(
        _provider_block(provider_id, feed.providers[provider_id].credential_hint)
        for provider_id in provider_ids
    )
    if not provider_blocks:
        provider_blocks = "  {}"

    label_lines = "\n".join(
        f'    {provider_id}: "{_label_for(provider_id)}"' for provider_id in provider_ids
    )
    if not label_lines:
        label_lines = "    {}"

    missing_credential = [
        provider_id
        for provider_id in provider_ids
        if not feed.providers[provider_id].credential_hint
    ]

    text = _RENDER_TEMPLATE.format(
        alias_prefix=alias_prefix,
        provider_blocks=provider_blocks,
        label_lines=label_lines,
        generated_at=feed.generated_at or "unknown",
    )

    notes: list[str] = [
        "Review naming.provider_labels. Each label comes from the provider id;"
        " rename any label you find unclear.",
        "Review approved_candidates. An Offering with no quality score is a"
        " Candidate until you add its id here.",
        "Review withheld. Add an Offering id here, with a reason, for anything"
        " you do not want probed or offered.",
        "Set pacing per provider. The default (concurrency 2, 5 second"
        " interval) fits neither a rate-limited free tier nor a paid"
        " subscription well.",
        "Uncomment the feed block and set url and credential_env before you"
        " run fetch.",
    ]
    for provider_id in missing_credential:
        notes.append(
            f"The Feed states no credential_hint for {provider_id}."
            " Find its credential variable in the provider's own docs."
        )

    return StarterPolicy(
        text=text,
        provider_count=len(provider_ids),
        notes=tuple(notes),
    )


_RENDER_TEMPLATE = """\
# Starter Policy, generated from a Feed snapshot (feed.generated_at:
# {generated_at}).
#
# This file is a draft, not a finished Policy. Review every section
# before you run generate against it. Nothing in this project writes
# to Policy at run time; edit this file by hand from here on.
#
# Validate a Policy with:
#   python -m litellm_maintainer.cli validate --policy policy.yaml

# ─────────────────────────────────────────────────────────────
# providers
#
# One entry per provider the Feed publishes. Every entry starts at
# `mode: all` with `entitlement: per_model`. `mode: all` takes every
# Offering the provider publishes, after the baseline filter (tool use
# required; image, audio, video, embedding and safety Offerings
# excluded). `per_model` never over-claims: it treats each Offering's
# failure as its own, never as a sign its siblings also fail. Change
# a provider to `mode: named` to list only the Offerings you want.
#
# WARNING: do not add a `pricing` filter here. An Offering with no
# stated price carries the pricing kind `unknown`, not `paid`. A
# provider can be free to your account for a reason the Feed cannot
# see: a subscription, a grandfathered plan, a company account. A
# `pricing` filter admits nothing there, because the Feed never marks
# such an Offering `free`. Leave the filter out for a provider whose
# free tier is an account entitlement.
# ─────────────────────────────────────────────────────────────
providers:
{provider_blocks}

# ─────────────────────────────────────────────────────────────
# quality
#
# The quality gate. An Offering scoring at or above
# minimum_coding_score is admitted. An Offering with no score is a
# Candidate: reported, never added, until you approve it below.
# ─────────────────────────────────────────────────────────────
quality:
  minimum_coding_score: 20

# ─────────────────────────────────────────────────────────────
# approved_candidates
#
# Offering ids you admit despite a missing quality score. Empty here:
# review each reported Candidate, then add its id.
# ─────────────────────────────────────────────────────────────
approved_candidates: []

# ─────────────────────────────────────────────────────────────
# naming
#
# provider_labels maps a provider id to the label used in an Alias.
# Each label below is derived from its provider id. Rename any label
# you find unclear; the Alias for every Offering from that provider
# changes with it.
# ─────────────────────────────────────────────────────────────
naming:
  alias_prefix: "{alias_prefix}"
  provider_labels:
{label_lines}
  alias_overrides: {{}}

# ─────────────────────────────────────────────────────────────
# withheld
#
# Offering id to reason. Empty here. Add an entry for an Offering you
# do not want probed or offered; only removing the entry clears it.
# ─────────────────────────────────────────────────────────────
withheld: {{}}

# ─────────────────────────────────────────────────────────────
# declared
#
# An Offering the Feed does not publish. Empty here. See
# policy.example.yaml for the shape of a Declared Offering entry.
# ─────────────────────────────────────────────────────────────
declared: []

# ─────────────────────────────────────────────────────────────
# pacing
#
# Per-provider Probe pacing. `default` applies to a provider with no
# entry of its own. Give a free tier a low concurrency and a long
# interval; give a subscription provider a higher concurrency.
# ─────────────────────────────────────────────────────────────
pacing:
  default:
    concurrency: 2
    minimum_interval_seconds: 5

# ─────────────────────────────────────────────────────────────
# schedule
#
# enabled turns the schedule on or off. interval_minutes sets the run
# interval. require_proxy skips a run while the proxy is down.
# maximum_staleness_hours forces a run despite a down proxy once the
# config has gone this long without one.
# ─────────────────────────────────────────────────────────────
schedule:
  enabled: true
  interval_minutes: 60
  require_proxy: true
  maximum_staleness_hours: 24

# ─────────────────────────────────────────────────────────────
# safety
#
# maximum_removal_share refuses a write that would remove more than
# this share of the current Aliases. snapshot_count sets how many
# prior Generated Config snapshots to keep for rollback.
# ─────────────────────────────────────────────────────────────
safety:
  maximum_removal_share: 0.25
  snapshot_count: 10

# ─────────────────────────────────────────────────────────────
# feed
#
# Commented out. Uncomment and fill in your own url and
# credential_env before you run fetch. credential_env names an
# environment variable; it never holds the token itself.
#
# feed:
#   url: "https://example.invalid/feed.json"
#   credential_env: "MY_FEED_TOKEN"
#   maximum_age_hours: 24
# ─────────────────────────────────────────────────────────────
"""


def write_starter_policy(starter: StarterPolicy, path: Path, *, force: bool = False) -> None:
    """Write `starter.text` to `path`.

    Refuse to overwrite an existing file unless `force` is `True`. On
    refusal, raise `FileExistsError` and leave the existing file
    untouched. Write through a temporary file in `path`'s own
    directory, then rename it into place, so a reader never observes a
    partial Policy and no temporary file survives a successful write.
    """
    if path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite the existing Policy at {path}; pass force=True to replace it"
        )

    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(starter.text)
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise
