"""Shared test fixtures.

`FIXTURES` names the frozen fixture directory. `load_fixture` reads a
named JSON or YAML fixture from it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

FIXTURES = Path(__file__).parent / "fixtures"


def _load(path: Path):
    if path.suffix == ".json":
        with open(path) as f:
            return json.load(f)
    if path.suffix in (".yaml", ".yml"):
        with open(path) as f:
            return yaml.safe_load(f)
    raise ValueError(f"unsupported fixture type: {path.suffix}")


@pytest.fixture
def load_fixture():
    """Load a named fixture from `tests/fixtures/`.

    Pass a path relative to `tests/fixtures/`, for example
    `"classify/gemini-deprecation_notice.json"`. Returns the parsed
    JSON or YAML value.
    """

    def _loader(relative_path: str):
        return _load(FIXTURES / relative_path)

    return _loader
