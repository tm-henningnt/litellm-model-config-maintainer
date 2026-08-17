"""Tests for the provider module deploy step."""

from __future__ import annotations

import time
from pathlib import Path

from litellm_maintainer.deploy import deploy_provider_modules


def test_a_new_module_is_copied_and_returned(tmp_path: Path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    (source_dir / "one_provider.py").write_text("VALUE = 1\n")

    written = deploy_provider_modules(source_dir, target_dir)

    assert written == [target_dir / "one_provider.py"]
    assert (target_dir / "one_provider.py").read_text() == "VALUE = 1\n"


def test_an_identical_module_is_not_copied_and_not_returned(tmp_path: Path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()
    (source_dir / "one_provider.py").write_text("VALUE = 1\n")
    (target_dir / "one_provider.py").write_text("VALUE = 1\n")

    # Force the target's mtime into the past so a spurious rewrite is
    # visible as a change in modification time.
    old_time = time.time() - 1000
    target_path = target_dir / "one_provider.py"
    import os

    os.utime(target_path, (old_time, old_time))
    mtime_before = target_path.stat().st_mtime

    written = deploy_provider_modules(source_dir, target_dir)

    assert written == []
    assert target_path.stat().st_mtime == mtime_before


def test_a_changed_module_is_copied_and_returned(tmp_path: Path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()
    (source_dir / "one_provider.py").write_text("VALUE = 2\n")
    (target_dir / "one_provider.py").write_text("VALUE = 1\n")

    written = deploy_provider_modules(source_dir, target_dir)

    assert written == [target_dir / "one_provider.py"]
    assert (target_dir / "one_provider.py").read_text() == "VALUE = 2\n"


def test_an_unrelated_target_file_survives(tmp_path: Path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    source_dir.mkdir()
    target_dir.mkdir()
    (source_dir / "one_provider.py").write_text("VALUE = 1\n")
    (target_dir / "config.yaml").write_text("model_list: []\n")

    deploy_provider_modules(source_dir, target_dir)

    assert (target_dir / "config.yaml").read_text() == "model_list: []\n"


def test_the_target_dir_is_created_when_missing(tmp_path: Path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "does_not_exist_yet"
    source_dir.mkdir()
    (source_dir / "one_provider.py").write_text("VALUE = 1\n")

    written = deploy_provider_modules(source_dir, target_dir)

    assert target_dir.is_dir()
    assert written == [target_dir / "one_provider.py"]
