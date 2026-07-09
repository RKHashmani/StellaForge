"""Tests for the ``transport_solution.h5`` contract in ``src/io_contracts.py``.

The contract is the union of the Stage 3/4/5 transport-snapshot loaders: required
``rho``/``density``/``temperature``/``Er`` (``Er`` with no species axis), optional
``pressure``/``ts``. Malformed cases run on the inner checker; the valid and missing-Er
files exercise the h5 read plus the raise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.io_contracts import ContractError, _check_transport_solution, validate_transport_solution
from tests.helpers.synthetic import write_transport_solution


def _valid_data(n_species: int = 3, n_rho: int = 5) -> dict[str, np.ndarray | None]:
    profile = np.ones((n_species, n_rho))
    return {
        "rho": np.linspace(0.0, 1.0, n_rho),
        "density": profile.copy(),
        "temperature": profile.copy(),
        "pressure": None,
        "Er": np.ones(n_rho),
        "ts": None,
    }


def test_valid_inner_passes() -> None:
    assert _check_transport_solution(_valid_data()) == []


def test_missing_temperature_flagged() -> None:
    data = _valid_data()
    data["temperature"] = None
    assert "missing required field 'temperature'" in _check_transport_solution(data)


def test_missing_er_flagged() -> None:
    data = _valid_data()
    data["Er"] = None
    assert "missing required field 'Er'" in _check_transport_solution(data)


# `Er` is a per-radius field with no species axis, so it must have one fewer dimension than the species-resolved
# `density`. This wrongly gives `Er` a (species, radius) 2D shape and asserts the checker flags that `Er`'s rank should
# be "one less" than density's.
def test_er_with_species_axis_flagged() -> None:
    data = _valid_data(n_species=3, n_rho=5)
    data["Er"] = np.ones((3, 5))  # wrongly given a species axis
    assert any("Er" in problem and "one less" in problem for problem in _check_transport_solution(data))


# Every profile's last axis must span the radial grid `rho`. This gives `density` a trailing axis of 4 while `rho` has
# length 5, and asserts the checker flags the "trailing axis" mismatch.
def test_rho_axis_mismatch_flagged() -> None:
    data = _valid_data(n_rho=5)
    data["density"] = np.ones((3, 4))  # trailing axis 4 != len(rho) 5
    assert any("density" in problem and "trailing axis" in problem for problem in _check_transport_solution(data))


def test_nonfinite_density_flagged() -> None:
    data = _valid_data()
    data["density"][0, 0] = np.nan
    assert "'density' contains non-finite values" in _check_transport_solution(data)


# `rho` is the 1D radial coordinate. This gives it a 2D shape and asserts the checker flags that `rho` "must be 1D".
def test_rho_not_1d_flagged() -> None:
    data = _valid_data()
    data["rho"] = np.ones((2, 3))  # rho must be a 1D radial coordinate
    assert any("rho" in problem and "must be 1D" in problem for problem in _check_transport_solution(data))


# `pressure` is optional, but when present it must share `density`'s rank (static vs time-resolved). This gives a 3D
# (time-resolved) pressure while density is 2D (static) and asserts the checker flags the ndim mismatch between the two.
def test_pressure_rank_mismatch_flagged() -> None:
    data = _valid_data(n_species=3, n_rho=5)
    data["pressure"] = np.ones((4, 3, 5))  # 3D pressure while density is 2D
    assert any(
        "pressure" in problem and "density" in problem and "ndim" in problem
        for problem in _check_transport_solution(data)
    )


def test_er_trailing_axis_mismatch_flagged() -> None:
    data = _valid_data(n_rho=5)
    data["Er"] = np.ones(4)  # trailing axis 4 != len(rho) 5
    assert any("Er" in problem and "trailing axis" in problem for problem in _check_transport_solution(data))


# Sets up a time-resolved solution (density/temperature are 3D and `Er` 2D over 4 time steps) but makes the time axis
# `ts` length 3 instead of 4. Asserts the checker flags that `ts`'s length disagrees with the profiles' time axis, so
# the timestamps and the data stay in sync.
def test_ts_length_mismatch_flagged() -> None:
    n_time, n_species, n_rho = 4, 3, 5
    data = _valid_data(n_species=n_species, n_rho=n_rho)
    data["density"] = np.ones((n_time, n_species, n_rho))
    data["temperature"] = np.ones((n_time, n_species, n_rho))
    data["Er"] = np.ones((n_time, n_rho))
    data["ts"] = np.arange(n_time - 1, dtype=float)  # length 3 != time axis 4
    assert any("ts" in problem and "time axis" in problem for problem in _check_transport_solution(data))


def test_valid_file_passes(tmp_path: Path) -> None:
    rho = np.linspace(0.0, 1.0, 5)
    profile = np.ones((3, 5))
    written = write_transport_solution(tmp_path / "t.h5", rho=rho, density=profile, temperature=profile, er=np.ones(5))
    assert validate_transport_solution(written) is None


# End-to-end failure path: writes a real file omitting the `er` argument (so `Er` is absent) and asserts
# `validate_transport_solution` raises a ContractError mentioning `Er`.
def test_file_missing_er_raises(tmp_path: Path) -> None:
    rho = np.linspace(0.0, 1.0, 5)
    profile = np.ones((3, 5))
    written = write_transport_solution(tmp_path / "t.h5", rho=rho, density=profile, temperature=profile)
    with pytest.raises(ContractError, match="Er"):
        validate_transport_solution(written)
