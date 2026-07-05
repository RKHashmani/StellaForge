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


def test_er_with_species_axis_flagged() -> None:
    data = _valid_data(n_species=3, n_rho=5)
    data["Er"] = np.ones((3, 5))  # wrongly given a species axis
    assert any("Er" in problem and "one less" in problem for problem in _check_transport_solution(data))


def test_rho_axis_mismatch_flagged() -> None:
    data = _valid_data(n_rho=5)
    data["density"] = np.ones((3, 4))  # trailing axis 4 != len(rho) 5
    assert any("density" in problem and "trailing axis" in problem for problem in _check_transport_solution(data))


def test_valid_file_passes(tmp_path: Path) -> None:
    rho = np.linspace(0.0, 1.0, 5)
    profile = np.ones((3, 5))
    written = write_transport_solution(tmp_path / "t.h5", rho=rho, density=profile, temperature=profile, er=np.ones(5))
    assert validate_transport_solution(written) is None


def test_file_missing_er_raises(tmp_path: Path) -> None:
    rho = np.linspace(0.0, 1.0, 5)
    profile = np.ones((3, 5))
    written = write_transport_solution(tmp_path / "t.h5", rho=rho, density=profile, temperature=profile)
    with pytest.raises(ContractError, match="Er"):
        validate_transport_solution(written)
