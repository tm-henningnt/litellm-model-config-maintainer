"""Copy provider modules next to the proxy config.

The proxy watches its config directory for Python file changes and
reloads on every write. A copy that writes identical bytes still
triggers a reload for nothing. `deploy_provider_modules` compares
content before it writes, so a run that changes nothing leaves the
proxy alone.
"""

from __future__ import annotations

from pathlib import Path


def deploy_provider_modules(source_dir: Path, target_dir: Path) -> list[Path]:
    """Copy each `*.py` file from `source_dir` to `target_dir`.

    Skip a file when the target already holds identical content. Write
    only a new or changed file. Return the list of files written, each
    path inside `target_dir`.

    Compare file content, not the modification time. Create
    `target_dir` when it does not exist. Leave a file in `target_dir`
    alone when `source_dir` holds no file of that name.
    """
    target_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for source_path in sorted(source_dir.glob("*.py")):
        target_path = target_dir / source_path.name
        content = source_path.read_bytes()

        if target_path.is_file() and target_path.read_bytes() == content:
            continue

        target_path.write_bytes(content)
        written.append(target_path)

    return written
