"""Tests for the Stage 5 post-processing convergence signal.

These reuse the real ``pressure_converged`` and ``build_signal`` from
``stage5_post_processing.py``, loaded by path. That script imports its sibling
``fit_vmec_pressure_from_transport_h5`` at module top, so loading it also exercises
the sibling-import support in ``load_stage_module``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tests.helpers.stage_import import load_stage_module
from tests.helpers.synthetic import write_transport_solution

post = load_stage_module("stages/stage5-post-processing/stage5_post_processing.py")


def _static(value: float, n_species: int = 3, n_rho: int = 5) -> np.ndarray:
    return np.full((n_species, n_rho), value)


def test_pressure_converged_false_for_static_profile(tmp_path: Path) -> None:
    rho = np.linspace(0.0, 1.0, 5)
    f = write_transport_solution(tmp_path / "static.h5", rho=rho, pressure=_static(2.0))
    assert not post.pressure_converged(f, rel_tol=1e-2)


def test_pressure_converged_true_for_small_change(tmp_path: Path) -> None:
    rho = np.linspace(0.0, 1.0, 5)
    slice0 = _static(2.0)
    pressure_3d = np.stack([slice0, slice0 * 1.0001])  # 0.01% change between slices
    f = write_transport_solution(tmp_path / "small.h5", rho=rho, pressure=pressure_3d)
    assert post.pressure_converged(f, rel_tol=1e-2)


def test_pressure_converged_false_for_large_change(tmp_path: Path) -> None:
    rho = np.linspace(0.0, 1.0, 5)
    slice0 = _static(2.0)
    pressure_3d = np.stack([slice0, slice0 * 2.0])  # 100% change between slices
    f = write_transport_solution(tmp_path / "large.h5", rho=rho, pressure=pressure_3d)
    assert not post.pressure_converged(f, rel_tol=1e-2)


def test_build_signal_halts_on_nonpositive_pressure(tmp_path: Path) -> None:
    rho = np.linspace(0.0, 1.0, 5)
    pressure = _static(2.0)
    pressure[:, 2] = -1.0  # summed total pressure is non-positive at one radius
    f = write_transport_solution(tmp_path / "halt.h5", rho=rho, pressure=pressure)
    assert post.build_signal(f, rel_tol=1e-2) == {"converged": False, "halt": True}


def test_build_signal_converged_without_halt(tmp_path: Path) -> None:
    rho = np.linspace(0.0, 1.0, 5)
    slice0 = _static(2.0)
    pressure_3d = np.stack([slice0, slice0 * 1.0001])
    f = write_transport_solution(tmp_path / "ok.h5", rho=rho, pressure=pressure_3d)
    assert post.build_signal(f, rel_tol=1e-2) == {"converged": True, "halt": False}
