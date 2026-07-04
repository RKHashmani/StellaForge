"""Import a stage script that has no package ``__init__``.

Scripts under ``stages/`` run as standalone files and have no ``__init__.py``, so
they cannot be imported as ``stages.stageN.<module>``. ``load_stage_module`` loads
one by file path so its pure helpers (argument parsers, math, post-processing) can
be unit tested in the ``test`` env. Those scripts import their solver packages
lazily inside functions, so loading a module does not require the solver installed.
A script's own directory is added to ``sys.path`` so a module-level import of a
sibling script resolves.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_stage_module(relpath: str, name: str | None = None) -> ModuleType:
    """Load a stage script by repo-relative path and return the executed module.

    Parameters
    ----------
    relpath : str
        Script path relative to the repo root, e.g.
        ``"stages/stage2-boozer/run_boozer.py"``.
    name : str, optional
        Module name to register under; defaults to the file stem.

    Returns
    -------
    ModuleType
        The loaded module, with its top-level names bound.
    """
    path = REPO_ROOT / relpath
    if not path.is_file():
        raise FileNotFoundError(f"stage module not found: {path}")
    mod_name = name or path.stem
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    # Some stage scripts import a sibling script by bare module name at module top,
    # so the script's own directory must be importable before it is executed.
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module
