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


# build_signal's three tolerance arguments, at the stage script's own argparse defaults.
_TOLERANCES = {"rel_tol": 1.0e-2, "turbulence_rtol": 1.0e-2, "turbulence_atol": 1.0e-8}


# `pressure_converged` decides whether the loop has settled by comparing the last two time slices of the pressure
# profile. A static (single-slice) profile has no change to measure, so convergence cannot be confirmed: this writes
# such a file and asserts the function returns False.
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


# `build_signal` produces the `{converged, halt, rerun_stage4}` dict the loop reads. A non-physical (non-positive)
# total pressure means the run has gone bad and should stop. This forces the total pressure negative at one radius and
# asserts the signal is halt=True (and converged=False), so the loop aborts. The halt return omits `rerun_stage4`
# entirely, and it is reached before any temperature is read, which is why this file needs no temperature.
def test_build_signal_halts_on_nonpositive_pressure(tmp_path: Path) -> None:
    rho = np.linspace(0.0, 1.0, 5)
    pressure = _static(2.0)
    pressure[:, 2] = -1.0  # summed total pressure is non-positive at one radius
    f = write_transport_solution(tmp_path / "halt.h5", rho=rho, pressure=pressure)
    assert post.build_signal(f, **_TOLERANCES) == {"converged": False, "halt": True}


# The non-halt return also reports `rerun_stage4` from ion-temperature drift, so this file carries a temperature with
# two distinct time slices: one slice would take the fewer-than-two-slices warning path and return False without
# measuring anything. The 0.01% drift is well inside turbulence_rtol, so the measured metric stays below 1.
def test_build_signal_converged_without_halt(tmp_path: Path) -> None:
    rho = np.linspace(0.0, 1.0, 5)
    slice0 = _static(2.0)
    temperature0 = _static(1.5)
    f = write_transport_solution(
        tmp_path / "ok.h5",
        rho=rho,
        pressure=np.stack([slice0, slice0 * 1.0001]),
        temperature=np.stack([temperature0, temperature0 * 1.0001]),
    )
    assert post.build_signal(f, **_TOLERANCES) == {"converged": True, "halt": False, "rerun_stage4": False}
