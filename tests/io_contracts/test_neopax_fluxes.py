"""Tests for the ``neopax_fluxes.h5`` contract in ``src/io_contracts.py``.

Built to the Stage 4 writer: top-level ``rho``/``r`` and ``(n_species, n_radii)``
``Gamma``/``Q``/``Upar`` (``Upar`` all zeros), plus a ``meta`` group carrying vlen-utf8
``species_names`` and unit attributes. Malformed cases run on the inner checker; the
valid and non-zero-``Upar`` files exercise the h5 read plus the raise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.io_contracts import ContractError, _check_neopax_fluxes, validate_neopax_fluxes
from tests.helpers.synthetic import write_neopax_fluxes


def _valid(n_species: int = 3, n_radii: int = 5) -> tuple[dict[str, np.ndarray | None], np.ndarray, dict[str, object]]:
    rho = np.linspace(0.0, 1.0, n_radii)
    flux = np.ones((n_species, n_radii))
    data = {"rho": rho, "r": rho.copy(), "Gamma": flux.copy(), "Q": flux.copy(), "Upar": np.zeros((n_species, n_radii))}
    species = np.array(["e", "D", "T"], dtype=object)
    attrs = {"particle_flux_units": "m^-2 s^-1", "heat_flux_units": "eV m^-2 s^-1", "minor_radius_m": 1.0}
    return data, species, attrs


def test_valid_inner_passes() -> None:
    data, species, attrs = _valid()
    assert _check_neopax_fluxes(data, species, attrs) == []


def test_nonzero_upar_flagged() -> None:
    data, species, attrs = _valid()
    data["Upar"] = np.ones((3, 5))
    assert "'Upar' must be all zeros for neopax_fluxes" in _check_neopax_fluxes(data, species, attrs)


def test_missing_meta_species_flagged() -> None:
    data, _, attrs = _valid()
    assert "missing required field 'meta/species_names'" in _check_neopax_fluxes(data, None, attrs)


def test_wrong_units_flagged() -> None:
    data, species, attrs = _valid()
    attrs["particle_flux_units"] = "wrong"
    assert any("particle_flux_units" in problem for problem in _check_neopax_fluxes(data, species, attrs))


def test_missing_minor_radius_flagged() -> None:
    data, species, attrs = _valid()
    del attrs["minor_radius_m"]
    assert "missing required meta attribute 'minor_radius_m'" in _check_neopax_fluxes(data, species, attrs)


def test_valid_file_passes(tmp_path: Path) -> None:
    rho = np.linspace(0.0, 1.0, 5)
    flux = np.ones((3, 5))
    written = write_neopax_fluxes(tmp_path / "n.h5", rho=rho, gamma=flux, q=flux)
    assert validate_neopax_fluxes(written) is None


def test_file_nonzero_upar_raises(tmp_path: Path) -> None:
    rho = np.linspace(0.0, 1.0, 5)
    flux = np.ones((3, 5))
    written = write_neopax_fluxes(tmp_path / "n.h5", rho=rho, gamma=flux, q=flux, upar=np.ones((3, 5)))
    with pytest.raises(ContractError, match="Upar"):
        validate_neopax_fluxes(written)
