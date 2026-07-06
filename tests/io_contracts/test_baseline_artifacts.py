"""Opportunistic contract checks against the real pipeline baseline outputs.

The other ``tests/io_contracts`` modules build synthetic files and prove the validators in
``src/io_contracts.py`` flag the right defects. This module is the complement: it runs those
same five file validators against the actual, gitignored baseline artifacts produced by a
local pipeline run, so a re-run whose upstream solver drifted (a renamed dataset, a
transposed axis, a dropped variable from a breaking dependency bump) is caught here rather
than silently downstream.

Discovery reads the committed ``inputs/quick_run/config.yaml`` and resolves paths through
``src.utils.resolve_pipeline_paths`` for both output layouts: the plain forward pass
(``outputs/quick_run/stageN_.../``) and every closed-loop iteration
(``outputs/quick_run/loop/iter_*/output/stageN_.../``), mirroring how the loop driver and
``tests/orchestration/test_ouroboros.py`` build iteration paths. Whichever of those exist on
this machine are validated.

Because ``outputs/`` is gitignored, CI and fresh clones have no baseline: each parametrized
case then finds zero files and skips with a clear reason instead of failing. The whole module
is marked ``requires_baseline`` (registered in ``pytest.ini``) so it can also be selected or
excluded explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from src.io_contracts import (
    validate_boozmn,
    validate_neopax_fluxes,
    validate_sfincs_flux,
    validate_transport_solution,
    validate_wout,
)

pytestmark = pytest.mark.requires_baseline

REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = REPO_ROOT / "inputs/quick_run/config.yaml"

# (output-artifact key, its file validator), in forward-pass chain order.
_ARTIFACT_VALIDATORS: tuple[tuple[str, Callable[[Path], None]], ...] = (
    ("s1_output", validate_wout),
    ("s2_output", validate_boozmn),
    ("s3_output", validate_sfincs_flux),
    ("s4_output", validate_neopax_fluxes),
    ("s5_output", validate_transport_solution),
)
_ARTIFACT_KEYS = tuple(key for key, _ in _ARTIFACT_VALIDATORS)


def _discover_baseline_artifacts() -> dict[str, list[Path]]:
    """Collect every existing local baseline file for the five output artifacts.

    Reads the quick_run config once, resolves the plain forward-pass layout plus every
    ``loop/iter_*/output`` closed-loop layout, and keeps only the paths present on disk.

    Returns
    -------
    dict[str, list[Path]]
        Maps each artifact key in ``_ARTIFACT_KEYS`` to its existing files (empty when the
        gitignored baseline is absent, e.g. in CI).
    """
    import yaml

    from src.utils import resolve_pipeline_paths

    config = yaml.safe_load(_CONFIG_PATH.read_text())

    layouts: list[dict[str, str]] = [resolve_pipeline_paths(config)]
    # Closed-loop iteration output roots (each holds a full stageN_<name>/ tree of its
    # own), derived from the config's output_dir like the loop driver derives them.
    loop_root = REPO_ROOT / config["output_dir"] / "loop"
    for output_dir in sorted(loop_root.glob("iter_*/output")):
        # Mirror the iteration call shape used in tests/orchestration/test_ouroboros.py:
        # the sibling input/ folder is the iteration's input_dir.
        input_dir = output_dir.parent / "input"
        layouts.append(
            resolve_pipeline_paths(config, input_dir=str(input_dir), output_dir=str(output_dir))
        )

    found: dict[str, list[Path]] = {key: [] for key in _ARTIFACT_KEYS}
    for layout in layouts:
        for key in _ARTIFACT_KEYS:
            # resolve_pipeline_paths returns repo-relative paths for the plain layout and
            # absolute paths for the loop layouts; REPO_ROOT / <either> normalizes both.
            path = REPO_ROOT / layout[key]
            if path.exists():
                found[key].append(path)
    return found


@pytest.fixture(scope="module")
def baseline_artifacts() -> dict[str, list[Path]]:
    """Discover the local baseline artifacts once for the whole module."""
    return _discover_baseline_artifacts()


@pytest.mark.parametrize(
    ("key", "validator"),
    _ARTIFACT_VALIDATORS,
    ids=[key for key, _ in _ARTIFACT_VALIDATORS],
)
def test_baseline_artifact_satisfies_contract(
    baseline_artifacts: dict[str, list[Path]],
    key: str,
    validator: Callable[[Path], None],
) -> None:
    """Every discovered baseline file for this artifact must still satisfy its contract."""
    files = baseline_artifacts[key]
    if not files:
        pytest.skip(f"no local baseline file found for {key} (gitignored outputs/ absent); nothing to validate")
    # A ContractError from any file is the drift signal we want surfaced.
    for path in files:
        validator(path)
