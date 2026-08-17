"""The Generator's write adapter.

Writes a config document, produced by `plan`, to the Generated Config
path as YAML, and reads the config already there before a new write
replaces it. This module is an adapter: it performs the one read and
the one write, and holds no decision. The safety checks that decide
whether to write at all — snapshot, removal-share refusal, rollback,
the envelope-downgrade report — live in `litellm_maintainer.safety`.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import yaml

HEADER = (
    "# GENERATED FILE. Do not edit by hand.\n"
    "#\n"
    "# litellm_maintainer writes this file. A hand edit is lost on the\n"
    "# next run. Edit the Policy instead: see CONTEXT.md.\n"
)


_RULE = "  # " + "\u2500" * 60


def _render_entry(entry: dict[str, Any], note: str | None) -> str:
    """One `model_list` entry as an indented YAML list item.

    `yaml.safe_dump` renders the entry on its own, then every line is
    indented by four and the leading `model_name:` takes the `- `. This
    keeps the values YAML writes them, quoting and all, and adds only
    indentation and the note comment.
    """
    dumped = yaml.safe_dump(entry, sort_keys=False, default_flow_style=False).rstrip("\n")
    lines = dumped.split("\n")
    out = [f"  - {lines[0]}"]
    if note:
        out.append(f"    # {note}")
    out.extend(f"    {line}" for line in lines[1:])
    return "\n".join(out)


def render_config(
    config: dict[str, Any],
    annotations: dict[str, Any] | None = None,
) -> str:
    """Render a config document to YAML text, with the generated-by header.

    `annotations` maps an Alias to an object carrying `group` and `note`
    (`plan.AliasAnnotation`). When given, `model_list` is written with a
    heading per group and a note beside each Alias, and a blank line
    between entries, so a human can scroll the file. Comments carry no
    meaning to litellm, and the parsed document is identical either way.

    Without `annotations` this renders the plain `yaml.safe_dump` form.
    Every other top-level key renders that way regardless: only
    `model_list` is grouped.
    """
    model_list = config.get("model_list")
    if not annotations or not isinstance(model_list, list) or not model_list:
        body = yaml.safe_dump(config, sort_keys=False, default_flow_style=False)
        return f"{HEADER}\n{body}"

    chunks: list[str] = ["model_list:"]
    current_group: str | None = None
    for entry in model_list:
        alias = entry.get("model_name")
        annotation = annotations.get(alias)
        group = getattr(annotation, "group", None) if annotation else None
        note = getattr(annotation, "note", None) if annotation else None
        if group and group != current_group:
            chunks.append("")
            chunks.append(_RULE)
            chunks.append(f"  # {group}")
            chunks.append(_RULE)
            current_group = group
        chunks.append("")
        chunks.append(_render_entry(entry, note))
    body = "\n".join(chunks) + "\n"

    rest = {key: value for key, value in config.items() if key != "model_list"}
    if rest:
        body += "\n" + yaml.safe_dump(rest, sort_keys=False, default_flow_style=False)
    return f"{HEADER}\n{body}"


def write_config(
    config: dict[str, Any],
    path: Path,
    annotations: dict[str, Any] | None = None,
) -> None:
    """Write `config` to `path` as YAML, with the generated-by header.

    `annotations` is passed to `render_config`, which uses it for the
    `model_list` headings and notes. Omit it for the plain form.

    Create the parent directory when it does not exist yet.

    Write atomically: a temporary file in the same directory, then a
    rename onto `path`, the same discipline `health.write_health`
    applies. The Generated Config is the one file the proxy's
    `--reload` watcher reads, so a plain in-place write that dies
    half-way would hand the proxy a truncated YAML document — and the
    next run would read that truncated file as "no previous config",
    which silently disarms the removal-share safety rail.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    text = render_config(config, annotations)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        os.unlink(tmp_name)
        raise


def rendered_config_is_unchanged(
    config: dict[str, Any],
    path: Path,
    annotations: dict[str, Any] | None = None,
) -> bool:
    """State whether writing `config` to `path` would change its bytes.

    Warning: the Generated Config is the one file the proxy's
    `--reload` watcher reads. Every write restarts the proxy, dropping
    the requests in flight. A run the Observation Journal triggered can
    fire within a minute of the last one, so an unconditional write
    turns a burst of failures into a burst of restarts -- which drops
    the very calls the run is reacting to.

    `deploy_provider_modules` already applies this discipline, and for
    the same reason: "a copy that writes identical bytes still triggers
    a reload for nothing."

    Compare the rendered TEXT, not the parsed document. The text is
    what the proxy's watcher sees. `HEADER` carries no timestamp, so an
    unchanged plan really does render identical bytes.

    Return `False` when `path` cannot be read at all. An unreadable
    target is a reason to write, never a reason to skip.
    """
    try:
        return path.read_text() == render_config(config, annotations)
    except OSError:
        return False


def read_previous_config(path: Path) -> dict[str, Any] | None:
    """Read the Generated Config already at `path`, before a new write.

    Returns `None` when `path` does not exist yet (the first run ever)
    or does not parse as YAML. Never raises: a previous config this
    reader cannot make sense of gives the safety rail nothing to
    compare against, which is the same as a first run, not a reason to
    stop the new run.

    `litellm_maintainer.safety.detect_envelope_downgrades` and
    `removed_aliases` read this value. Nothing here strips the
    `HEADER` comment first; `yaml.safe_load` ignores YAML comments on
    its own.
    """
    try:
        raw_text = path.read_text()
    except FileNotFoundError:
        return None
    try:
        document = yaml.safe_load(raw_text)
    except yaml.YAMLError:
        return None
    if not isinstance(document, dict):
        return None
    return document
