"""Whether the litellm the proxy runs still carries our local patches.

Two defects in litellm break Aliases this instance offers. Each has a
one-place fix, applied to the installed litellm rather than to any file
in this repository, because the fault is in litellm's own transform
layer and no callback or config setting reaches it. `docs/gotchas.md`
records both, with the reasoning and the exact edit.

`uv tool upgrade litellm` replaces the installed tree and removes both
edits. Nothing then reports the loss: the Generated Config does not
change, the proxy starts, and `/v1/models` still lists every Alias.
The models simply stop answering. This module gives `doctor` a way to
say so, with the file to edit named.

A marker is a short string the patch introduces and stock litellm does
not hold. Reading a marker is weaker than measuring behaviour, and it
has one false alarm worth knowing: litellm fixing a defect upstream
also removes the marker. A failed check therefore means "the patched
behaviour is not proven present", not "litellm is broken". Read
`docs/gotchas.md`, confirm which case you are in, and retire the entry
here when litellm carries the fix itself.

The proxy usually runs a different litellm from this package's own. The
maintainer installs litellm as a library; the proxy is commonly a `uv
tool` install with its own tree. So never inspect the imported module:
locate the tree the proxy runs, and read that.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

# One entry per patch. `marker` must appear in `relative_path` when the
# patch is present. Keep each `remedy` short and name the file.
REQUIRED_PATCHES: tuple[tuple[str, str, str, str], ...] = (
    (
        "chatgpt_stream",
        "llms/custom_httpx/llm_http_handler.py",
        "client_requested_stream",
        "re-apply the chatgpt stream patch to "
        "llms/custom_httpx/llm_http_handler.py; see docs/gotchas.md, "
        "'A stream-only provider makes litellm break non-streaming callers'",
    ),
    (
        "usage_only_chunk",
        "llms/anthropic/experimental_pass_through/adapters/streaming_iterator.py",
        "if not chunk.choices",
        "re-apply the usage-only-chunk patch to "
        "llms/anthropic/experimental_pass_through/adapters/streaming_iterator.py; "
        "see docs/gotchas.md, 'A usage-only final chunk can truncate a stream'",
    ),
)


@dataclass(frozen=True)
class PatchStatus:
    """What one patch inspection found.

    `present` is `True` when the marker was found, `False` when the file
    was read and the marker was absent, and `None` when the file could
    not be read at all. `None` is not a failure: an operator who runs
    the proxy elsewhere has nothing for this check to read.
    """

    name: str
    present: bool | None
    detail: str
    remedy: str


def litellm_source_root(executable: str | None = None) -> Path | None:
    """The `litellm` package directory the proxy's own interpreter loads.

    Ask the interpreter that runs the `litellm` executable where it
    imports litellm from. This follows a `uv tool` install, a venv, and
    a system install alike, and it never confuses the proxy's tree with
    this package's own.

    Return `None` when no `litellm` executable is on `PATH`, or when it
    cannot report a path.
    """
    resolved = executable or shutil.which("litellm")
    if resolved is None:
        return None
    interpreter = Path(resolved).resolve().parent / "python"
    if not interpreter.exists():
        interpreter = Path(resolved).resolve().parent / "python3"
    if not interpreter.exists():
        return None
    try:
        completed = subprocess.run(
            [str(interpreter), "-c", "import litellm, pathlib; print(pathlib.Path(litellm.__file__).parent)"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    candidate = Path(completed.stdout.strip())
    return candidate if candidate.is_dir() else None


def inspect_patches(source_root: Path | None) -> tuple[PatchStatus, ...]:
    """Read every required patch's marker under `source_root`.

    `source_root` is the litellm package directory, as
    `litellm_source_root` returns. Pass `None` when it is unknown: every
    result is then `present=None`, which `doctor` reports without
    failing.
    """
    results: list[PatchStatus] = []
    for name, relative_path, marker, remedy in REQUIRED_PATCHES:
        if source_root is None:
            results.append(
                PatchStatus(
                    name=name,
                    present=None,
                    detail="the litellm the proxy runs was not located, so no patch was read",
                    remedy=remedy,
                )
            )
            continue
        target = source_root / relative_path
        try:
            text = target.read_text()
        except OSError as error:
            results.append(
                PatchStatus(
                    name=name,
                    present=None,
                    detail=f"cannot read {relative_path}: {error.strerror or error}",
                    remedy=remedy,
                )
            )
            continue
        if marker in text:
            results.append(
                PatchStatus(
                    name=name,
                    present=True,
                    detail=f"{relative_path} carries the patch.",
                    remedy=remedy,
                )
            )
        else:
            results.append(
                PatchStatus(
                    name=name,
                    present=False,
                    detail=(
                        f"{relative_path} does not carry the patch. An upgrade "
                        "removes it, and the Aliases it repairs stop answering "
                        "with no other symptom."
                    ),
                    remedy=remedy,
                )
            )
    return tuple(results)
