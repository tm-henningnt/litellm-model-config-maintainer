#!/usr/bin/env python3
"""Stand-in for codexbar, for `real_codexbar_runner` tests only.

Never the real binary. Prints `codexbar-sample.json` verbatim,
whichever arguments it is called with: the tests that need
argument-dependent behaviour use an in-process fake runner instead of
this script.
"""

from __future__ import annotations

import sys
from pathlib import Path

FIXTURE = Path(__file__).parent / "codexbar-sample.json"


def main() -> int:
    sys.stdout.write(FIXTURE.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
