"""Redaction of credential values from text.

Prior art: `scripts/test_models.py`. That script builds a map from every
non-empty `.env` value to a label, then replaces each value in output
before printing. This module follows the same approach and adds a
regex net for a bare `sk-...` token or a `Bearer ...` header, in case a
credential appears without a matching map entry.
"""

from __future__ import annotations

import re
from pathlib import Path

MIN_VALUE_LENGTH = 8

_SK_TOKEN_RE = re.compile(r"sk-[A-Za-z0-9_\-]{10,}")
_BEARER_RE = re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]{10,}")


def parse_dotenv_file(env_path: Path) -> dict[str, str]:
    """Parse a dotenv-style file into a `NAME -> value` mapping.

    Read `NAME=value` lines. Ignore a blank line and a `#` comment.
    Strip whitespace from both `NAME` and `value`. Shared by
    `build_redaction_map` and `litellm_maintainer.cli._credential_resolver`,
    the two places this project reads `.env.local`.
    """
    values: dict[str, str] = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            values[name.strip()] = value.strip()
    return values


def build_redaction_map(env_path: Path | None = None) -> dict[str, str]:
    """Build a map from credential value to a placeholder.

    Read `env_path` as a dotenv-style file: `NAME=value` lines, blank
    lines and `#` comments ignored. Map every value at least
    `MIN_VALUE_LENGTH` characters long to `<REDACTED:NAME>`. Skip a
    shorter value, because it produces false matches.
    """
    mapping: dict[str, str] = {}
    if env_path is None:
        return mapping
    for name, value in parse_dotenv_file(env_path).items():
        if len(value) < MIN_VALUE_LENGTH:
            continue
        mapping[value] = f"<REDACTED:{name}>"
    return mapping


def redact(text: str, mapping: dict[str, str]) -> str:
    """Replace every mapped credential value in `text`.

    Replace the longest values first, so one value that is a substring
    of another cannot leave a partial value behind. Then apply a regex
    net for a bare `sk-...` token or a `Bearer ...` header, which
    catches a credential with no matching map entry.
    """
    for value in sorted(mapping, key=len, reverse=True):
        if value in text:
            text = text.replace(value, mapping[value])
    text = _SK_TOKEN_RE.sub("<REDACTED:sk-token>", text)
    text = _BEARER_RE.sub(r"\1<REDACTED:bearer-token>", text)
    return text
