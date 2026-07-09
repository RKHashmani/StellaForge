"""Tests for the Stage 5 pressure-fit helpers.

These reuse the real functions from ``fit_vmec_pressure_from_transport_h5.py``, loaded
by path because ``stages/`` has no package ``__init__``, so the tests pin the actual
fit and file-rewrite behavior rather than a copy. The script imports h5py/numpy only,
so it loads with no solver present.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tests.helpers.stage_import import load_stage_module
from tests.helpers.synthetic import write_transport_solution

fit_mod = load_stage_module("stages/stage5-post-processing/fit_vmec_pressure_from_transport_h5.py")

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _profiles(n_species: int = 3, n_rho: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho = np.linspace(0.0, 1.0, n_rho)
    density = np.linspace(1.0, 2.0, n_species * n_rho).reshape(n_species, n_rho)
    temperature = np.linspace(2.0, 3.0, n_species * n_rho).reshape(n_species, n_rho)
    return rho, density, temperature


# A round-trip test for the polynomial fitter. It builds pressure data from known coefficients [3, -2, 0.5], fits a
# degree-2 power series back to that data, and asserts the recovered coefficients match the originals. Recovering a
# known analytical answer is a strong correctness check.
def test_fit_power_series_recovers_known_polynomial() -> None:
    rho = np.linspace(0.0, 1.0, 6)
    s = rho**2
    coeffs_true = np.array([3.0, -2.0, 0.5])  # low-order first: 3 - 2 s + 0.5 s^2
    p = coeffs_true[0] + coeffs_true[1] * s + coeffs_true[2] * s**2
    coeffs = fit_mod._fit_power_series(s, p, degree=2)
    assert_allclose(coeffs, coeffs_true, atol=1e-9)


# Total pressure can be supplied directly, or reconstructed from temperature x density. This writes one file each way
# (with matching inputs) and asserts `_load_total_pressure` returns the same total for both, that it equals the
# species-summed pressure, and that a static profile reports no time index. Confirms the two supported input forms
# agree.
def test_load_total_pressure_temperature_density_matches_pressure(tmp_path: Path) -> None:
    rho, density, temperature = _profiles()
    pressure = density * temperature
    file_td = write_transport_solution(tmp_path / "td.h5", rho=rho, temperature=temperature, density=density)
    file_p = write_transport_solution(tmp_path / "p.h5", rho=rho, pressure=pressure)

    rho_td, total_td, idx_td = fit_mod._load_total_pressure(file_td, time_index=-1, final_time=False)
    rho_p, total_p, idx_p = fit_mod._load_total_pressure(file_p, time_index=-1, final_time=False)

    assert_allclose(total_td, total_p, rtol=1e-12)
    assert_allclose(total_p, np.sum(pressure, axis=0), rtol=1e-12)
    assert_allclose(rho_td, rho, rtol=1e-12)
    assert idx_td is None and idx_p is None  # static profile has no resolved time index


# For a time-resolved file, `_load_total_pressure` should be able to pick a specific time slice. This writes a
# 2-time-step pressure, then asserts that `final_time=True` selects the last slice (time index 1) and `time_index=0`
# selects the first, with the returned totals matching each slice's sum.
def test_load_total_pressure_final_time_selects_last_slice(tmp_path: Path) -> None:
    rho, density, temperature = _profiles()
    slice0 = density * temperature
    slice1 = 2.0 * slice0
    pressure_3d = np.stack([slice0, slice1])  # (n_time=2, n_species, n_rho)
    f3d = write_transport_solution(tmp_path / "p3d.h5", rho=rho, pressure=pressure_3d)

    _, total_final, idx_final = fit_mod._load_total_pressure(f3d, time_index=-1, final_time=True)
    _, total_initial, idx_initial = fit_mod._load_total_pressure(f3d, time_index=0, final_time=False)

    assert idx_final == 1 and idx_initial == 0
    assert_allclose(total_final, np.sum(slice1, axis=0), rtol=1e-12)
    assert_allclose(total_initial, np.sum(slice0, axis=0), rtol=1e-12)


def test_load_total_pressure_missing_rho_raises(tmp_path: Path) -> None:
    bad = tmp_path / "no_rho.h5"
    with h5py.File(bad, "w") as f:
        f.create_dataset("pressure", data=np.ones((3, 5)))
    with pytest.raises(KeyError):
        fit_mod._load_total_pressure(bad, time_index=-1, final_time=False)


# This is the feedback step that turns the fitted pressure back into a VMEC input file for the next loop iteration.
# Starting from a committed template fixture, it writes a new file with the fitted coefficients and asserts the pressure
# keys are set correctly (`PMASS_TYPE`, `PRES_SCALE`, and the `AM` coefficient line). The expected `AM` line is
# hard-coded (not built by the writer's own formatter) so a bug in that formatter can't hide itself. It also asserts the
# template's `AM` line was replaced in place (not duplicated) and that the committed fixture file itself is left
# unmodified.
def test_write_vmec_input_rewrites_pressure_keys(tmp_path: Path) -> None:
    coeffs = np.array([3.0, -2.0, 0.5])
    out = tmp_path / "vmec_out.txt"
    fixture = DATA_DIR / "vmec_indata_min.txt"

    result = fit_mod._write_vmec_input_with_pressure_fit(fixture, coeffs, output_path=out)

    assert result == out
    text = out.read_text()
    assert "PMASS_TYPE = 'power_series'" in text
    assert "PRES_SCALE = 1.0000000000000000E+00" in text
    # Literal AM line for coeffs [3.0, -2.0, 0.5], independent of the writer's own formatter,
    # so a bug in _format_am_line cannot mask itself.
    expected_am_line = "AM = 3.0000000000000000E+00, -2.0000000000000000E+00, 5.0000000000000000E-01"
    assert expected_am_line in text
    assert "AM = 0.0" not in text  # the template AM line is replaced in place, not left behind or duplicated
    assert "AM = 0.0" in fixture.read_text()  # committed fixture left unmodified
