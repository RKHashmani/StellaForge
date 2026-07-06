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


def load_stage_module(relpath: str) -> ModuleType:
    """Load a stage script by repo-relative path and return the executed module.

    A module is registered in ``sys.modules`` under its file stem. Re-loading the
    same file returns the cached module without re-executing it. Requesting a
    different file whose stem collides with an already-loaded module raises
    ``ImportError`` rather than silently shadowing the cached one.

    Parameters
    ----------
    relpath : str
        Script path relative to the repo root, e.g.
        ``"stages/stage2-boozer/run_boozer.py"``.

    Returns
    -------
    ModuleType
        The loaded module, with its top-level names bound.

    Raises
    ------
    FileNotFoundError
        If ``relpath`` does not resolve to a file under the repo root.
    ImportError
        If the file stem is already registered from a different file, or if an
        import spec cannot be built for the file.
    """
    path = REPO_ROOT / relpath
    if not path.is_file():
        raise FileNotFoundError(f"stage module not found: {path}")
    mod_name = path.stem
    cached = sys.modules.get(mod_name)
    if cached is not None:
        cached_file = getattr(cached, "__file__", None)
        if cached_file is not None and Path(cached_file).resolve() == path.resolve():
            return cached
        raise ImportError(
            f"stage module name {mod_name!r} is already loaded from {cached_file}; "
            f"cannot reload it from a different file {path}"
        )
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    # A stage script may import a sibling script by bare module name at module top,
    # so the script's directory must be importable before the script is executed.
    # These sys.path entries are session-lifetime: one is added per stage directory
    # and never removed, because those sibling imports must keep resolving for every
    # later load in the same pytest process. No cleanup is done here on purpose.
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # A body that raises leaves a half-initialized module in sys.modules; drop
        # it so a failed load cannot be mistaken for a cached success later.
        sys.modules.pop(mod_name, None)
        raise
    return module
