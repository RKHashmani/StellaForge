"""Tests for the Stage 4 SPECTRAX-GK radial-scan helpers.

These reuse the pure functions from ``spectrax_gk_radial_scan.py``, loaded by path.
``_spectrax_flux_to_neopax_units`` converts a gyro-Bohm flux to NEOPAX physical
units; ``_expand_axis_zero_if_needed`` prepends the magnetic-axis radius (rho=0,
zero flux) when the scan skipped it. SPECTRAX-GK runs via subprocess, so loading
the module and calling these helpers needs no solver.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.constants import elementary_charge, proton_mass

from tests.helpers.stage_import import load_stage_module

scan = load_stage_module("stages/stage4-turbulence/spectrax_gk_radial_scan.py")


# --- _spectrax_flux_to_neopax_units ---

def test_flux_units_q_over_gamma_is_temperature_in_ev() -> None:
    kwargs = dict(density_ref_state=0.5, temperature_ref_keV=2.0, mass_ref_mp=2.0, rho_star_physical=0.01)
    gamma = scan._spectrax_flux_to_neopax_units(3.0, kind="Gamma", **kwargs)
    q = scan._spectrax_flux_to_neopax_units(3.0, kind="Q", **kwargs)
    assert_allclose(q / gamma, 2.0 * 1.0e3, rtol=1e-12)  # Q carries an extra factor of T expressed in eV


def test_flux_units_absolute_scale_matches_reference_convention() -> None:
    # Absolute pin for the reference-species convention: Gamma = gamma_hat * n_ref * vth_ref * rho_star^2
    # with vth_ref = sqrt(2 T e / m). Here gamma_hat=3, n=0.5e20 m^-3, T=2000 eV, mass=2 m_p,
    # rho_star=0.01, so the trailing rho_star^2 = 1e-4 factor is exercised end to end. The ratio-only
    # tests below cannot see a silent drift in this overall unit convention.
    vth_ref = np.sqrt(2.0 * 2000.0 * elementary_charge / (2.0 * proton_mass))
    expected_gamma = 3.0 * 0.5e20 * vth_ref * 1.0e-4
    gamma = scan._spectrax_flux_to_neopax_units(
        3.0, density_ref_state=0.5, temperature_ref_keV=2.0, mass_ref_mp=2.0, rho_star_physical=0.01, kind="Gamma"
    )
    assert_allclose(gamma, expected_gamma, rtol=1e-12)


def test_flux_units_scale_as_rho_star_squared() -> None:
    base = dict(density_ref_state=1.0, temperature_ref_keV=1.0, mass_ref_mp=1.0, kind="Gamma")
    small = scan._spectrax_flux_to_neopax_units(1.0, rho_star_physical=0.01, **base)
    large = scan._spectrax_flux_to_neopax_units(1.0, rho_star_physical=0.02, **base)
    assert_allclose(large / small, 4.0, rtol=1e-12)  # gyro-Bohm scaling goes as rho_star^2


def test_flux_units_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):
        scan._spectrax_flux_to_neopax_units(
            1.0, density_ref_state=1.0, temperature_ref_keV=1.0, mass_ref_mp=1.0, rho_star_physical=0.01, kind="bogus"
        )


# --- _expand_axis_zero_if_needed ---

def _expand_inputs(n_species: int = 3, n: int = 2) -> dict:
    return {
        "rho": np.linspace(0.5, 1.0, n),
        "r_physical": np.linspace(0.5, 1.0, n),
        "rho_index": np.arange(1, n + 1),
        "torflux": np.linspace(0.25, 1.0, n),
        "er": np.linspace(1.0, 2.0, n),
        "heat_flux": np.linspace(1.0, 2.0, n),
        "particle_flux": np.linspace(1.0, 2.0, n),
        "heat_flux_species": np.ones((n, n_species)),
        "particle_flux_species": np.ones((n, n_species)),
        "gamma_neopax": np.arange(1.0, n_species * n + 1.0).reshape(n_species, n),
        "q_neopax": np.arange(1.0, n_species * n + 1.0).reshape(n_species, n) + 10.0,
    }


def test_expand_passthrough_without_source_rho() -> None:
    arrays = _expand_inputs()
    out = scan._expand_axis_zero_if_needed({}, **arrays)
    # no source_rho -> arrays returned unchanged (radius axis stays length 2)
    assert out[0].shape == (2,)
    assert_allclose(out[0], arrays["rho"])
    assert out[9].shape == arrays["gamma_neopax"].shape


def test_expand_prepends_axis_zero() -> None:
    manifest = {"source_rho": [0.0, 0.5, 1.0], "source_er": [0.0, 1.5, 3.0], "geometry": {"a_minor": 2.0}}
    arrays = _expand_inputs()  # rho size 2, rho_index [1, 2]
    (full_rho, full_r, full_rho_index, full_torflux, full_er, _heat, _particle,
     _heat_s, _particle_s, full_gamma, full_q) = scan._expand_axis_zero_if_needed(manifest, **arrays)

    assert_allclose(full_rho, [0.0, 0.5, 1.0], rtol=1e-12)
    assert_allclose(full_r, [0.0, 1.0, 2.0], rtol=1e-12)            # a_minor * full_rho
    assert full_rho_index.tolist() == [0, 1, 2]
    assert_allclose(full_torflux, [0.0, 0.25, 1.0], rtol=1e-12)     # full_rho ** 2
    assert_allclose(full_er, [0.0, 1.5, 3.0], rtol=1e-12)  # Er from manifest source_er, not the per-run er
    assert full_gamma.shape == (3, 3)
    assert_allclose(full_gamma[:, 0], 0.0)              # prepended axis column is zero-filled
    assert_allclose(full_gamma[:, 1:], arrays["gamma_neopax"], rtol=1e-12)
    assert_allclose(full_q[:, 1:], arrays["q_neopax"], rtol=1e-12)


def test_expand_er_zero_filled_on_source_er_size_mismatch() -> None:
    manifest = {"source_rho": [0.0, 0.5, 1.0], "source_er": [0.0, 1.5], "geometry": {"a_minor": 2.0}}
    arrays = _expand_inputs()  # rho size 2, rho_index [1, 2]
    out = scan._expand_axis_zero_if_needed(manifest, **arrays)
    # source_er length (2) does not match the expanded radius axis (3): the axis still expands,
    # but Er drops to zeros rather than misaligning.
    assert_allclose(out[0], [0.0, 0.5, 1.0], rtol=1e-12)
    assert_allclose(out[4], 0.0)


def test_expand_passthrough_on_index_mismatch() -> None:
    manifest = {"source_rho": [0.0, 0.5, 1.0], "geometry": {"a_minor": 2.0}}
    arrays = _expand_inputs()
    arrays["rho_index"] = np.array([0, 1])  # not [1, 2] -> guard trips, no expansion
    out = scan._expand_axis_zero_if_needed(manifest, **arrays)
    assert out[0].shape == (2,)
