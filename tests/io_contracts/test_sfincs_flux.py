"""Tests for the ``sfincs_jax_flux_profiles.h5`` contract in ``src/io_contracts.py``.

Built to the Stage 3 writer: top-level ``r``/``rHat``/``rho`` and ``(n_species,
n_radii)`` ``Gamma``/``Q``/``Upar``, bytes ``species_names``, and a root
``axis_zero_padded`` attribute. Malformed cases run on the inner checker; the valid and
wrong-radius files exercise the h5 read plus the raise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.io_contracts import ContractError, _check_sfincs_flux, validate_sfincs_flux
from tests.helpers.synthetic import write_sfincs_flux


def _valid_data(n_species: int = 3, n_radii: int = 5) -> dict[str, np.ndarray | None]:
    rho = np.linspace(0.0, 1.0, n_radii)
    flux = np.ones((n_species, n_radii))
    return {
        "r": rho.copy(),
        "rHat": rho.copy(),
        "rho": rho,
        "Gamma": flux.copy(),
        "Q": flux.copy(),
        "Upar": np.zeros((n_species, n_radii)),
        "species_names": np.array([b"e", b"D", b"T"]),
    }


def test_valid_inner_passes() -> None:
    assert _check_sfincs_flux(_valid_data(), {"axis_zero_padded": True}) == []


def test_missing_gamma_flagged() -> None:
    data = _valid_data()
    data["Gamma"] = None
    assert "missing required field 'Gamma'" in _check_sfincs_flux(data, {"axis_zero_padded": True})


def test_species_axis_mismatch_flagged() -> None:
    data = _valid_data(n_species=3, n_radii=5)
    data["Q"] = np.ones((2, 5))  # 2 species but species_names has 3
    assert any("Q" in problem and "species axis" in problem for problem in _check_sfincs_flux(data, {"axis_zero_padded": True}))


def test_missing_attr_flagged() -> None:
    assert "missing required root attribute 'axis_zero_padded'" in _check_sfincs_flux(_valid_data(), {})


def test_valid_file_passes(tmp_path: Path) -> None:
    rho = np.linspace(0.0, 1.0, 5)
    flux = np.ones((3, 5))
    written = write_sfincs_flux(tmp_path / "s.h5", rho=rho, gamma=flux, q=flux)
    assert validate_sfincs_flux(written) is None


def test_file_wrong_radius_axis_raises(tmp_path: Path) -> None:
    rho = np.linspace(0.0, 1.0, 5)
    flux = np.ones((3, 4))  # radius axis 4 != len(rho) 5
    written = write_sfincs_flux(tmp_path / "s.h5", rho=rho, gamma=flux, q=flux, upar=np.zeros((3, 4)))
    with pytest.raises(ContractError, match="radius axis"):
        validate_sfincs_flux(written)
