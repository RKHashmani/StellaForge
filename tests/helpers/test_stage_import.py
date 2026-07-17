"""Tests for the stage-script loader in ``tests.helpers.stage_import``.

These use throwaway scripts written under ``tmp_path`` so they do not depend on
any real ``stages/`` file or on the order in which other test modules load stage
scripts. Every probe module is registered under a distinctive stem and removed
from ``sys.modules`` on teardown so it cannot leak into the rest of the suite.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from tests.helpers.stage_import import REPO_ROOT, load_stage_module


# `load_stage_module` registers modules in the global `sys.modules`. This fixture snapshots that registry before the
# test and deletes any entries the test added afterward. A test using it can load throwaway modules without leaking them
# into later tests.
@pytest.fixture
def clean_sys_modules() -> None:
    """Remove any ``sys.modules`` entries a test adds, so probes do not leak."""
    before = set(sys.modules)
    try:
        yield
    finally:
        for name in set(sys.modules) - before:
            del sys.modules[name]


def _relpath(script: Path) -> str:
    """Repo-relative path string that ``load_stage_module`` joins onto the repo root."""
    return os.path.relpath(script, REPO_ROOT)


# `load_stage_module` should cache by file, running a script's body only once. This writes a probe script that appends
# an "x" to a counter file every time its body runs, loads it twice, and asserts both loads return the very same module
# object and that the counter contains a single "x" (the body ran once, not once per call).
def test_reloading_same_file_returns_cached_and_executes_once(
    tmp_path: Path, clean_sys_modules: None
) -> None:
    counter = tmp_path / "exec_count.txt"
    script = tmp_path / "_stage_import_memo_probe.py"
    script.write_text(
        f'with open({str(counter)!r}, "a", encoding="utf-8") as _f:\n'
        f'    _f.write("x")\n'
        f"VALUE = 42\n"
    )
    relpath = _relpath(script)

    first = load_stage_module(relpath)
    second = load_stage_module(relpath)

    assert first is second
    assert first.VALUE == 42
    assert counter.read_text() == "x"  # body ran exactly once, not once per call


# The loader keys modules by filename stem, so two different files sharing a stem would collide. This creates two files
# with the same name in different folders, loads the first, and asserts loading the second raises ImportError whose
# message names both paths, so the collision fails loudly instead of one file silently shadowing the other.
def test_stem_collision_with_different_file_raises(tmp_path: Path, clean_sys_modules: None) -> None:
    dir_a = tmp_path / "col_a"
    dir_b = tmp_path / "col_b"
    dir_a.mkdir()
    dir_b.mkdir()
    script_a = dir_a / "_stage_import_collision_probe.py"
    script_b = dir_b / "_stage_import_collision_probe.py"
    script_a.write_text("VALUE = 'a'\n")
    script_b.write_text("VALUE = 'b'\n")

    load_stage_module(_relpath(script_a))
    with pytest.raises(ImportError) as excinfo:
        load_stage_module(_relpath(script_b))

    message = str(excinfo.value)
    assert "col_a" in message  # names the cached file's path
    assert "col_b" in message  # names the requested file's path


# If a script raises while being imported, the loader must not leave a half-initialized module cached (which a later
# load could mistake for success). This writes a script that raises on import, asserts that error propagates, and then
# asserts the module's name is absent from `sys.modules`, confirming the loader cleaned up after the failure.
def test_failed_load_leaves_no_cached_module(tmp_path: Path, clean_sys_modules: None) -> None:
    script = tmp_path / "_stage_import_boom_probe.py"
    script.write_text("raise RuntimeError('boom during import')\n")

    with pytest.raises(RuntimeError, match="boom during import"):
        load_stage_module(_relpath(script))

    assert "_stage_import_boom_probe" not in sys.modules
