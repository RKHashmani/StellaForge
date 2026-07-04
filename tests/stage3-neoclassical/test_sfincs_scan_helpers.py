"""Tests for the Stage 3 sfincs_jax radial-scan helpers.

These reuse the pure functions from ``sfincs_jax_radial_scan.py``, loaded by path.
``_choose_radius_indices`` selects which flux surfaces to solve; the analytical
snapshot builder synthesises species-resolved profiles from config. The solver is
imported lazily inside the worker, so loading the module and calling these helpers
needs no solver.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tests.helpers.stage_import import load_stage_module

scan = load_stage_module("stages/stage3-neoclassical/sfincs_jax_radial_scan.py")


# --- _choose_radius_indices ---

def test_choose_radius_default_skips_axis() -> None:
    rho = np.linspace(0.0, 1.0, 5)  # index 0 is rho = 0
    idxs = scan._choose_radius_indices(rho, explicit=None, rho_min=None, rho_max=None, num_radii=None)
    assert idxs == [1, 2, 3, 4]


def test_choose_radius_explicit_sorted_and_deduped() -> None:
    rho = np.linspace(0.0, 1.0, 5)
    idxs = scan._choose_radius_indices(rho, explicit=[3, 1, 1], rho_min=None, rho_max=None, num_radii=None)
    assert idxs == [1, 3]


def test_choose_radius_explicit_out_of_range_raises() -> None:
    rho = np.linspace(0.0, 1.0, 5)
    with pytest.raises(IndexError):
        scan._choose_radius_indices(rho, explicit=[7], rho_min=None, rho_max=None, num_radii=None)


def test_choose_radius_num_radii_subsamples() -> None:
    rho = np.linspace(0.0, 1.0, 5)  # candidates after axis skip: [1, 2, 3, 4]
    idxs = scan._choose_radius_indices(rho, explicit=None, rho_min=None, rho_max=None, num_radii=2)
    assert idxs == [1, 4]  # endpoints of the candidate span


def test_choose_radius_empty_filter_raises() -> None:
    rho = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError):
        scan._choose_radius_indices(rho, explicit=None, rho_min=2.0, rho_max=None, num_radii=None)


# --- _build_standard_analytical_snapshot ---

def test_snapshot_shapes() -> None:
    snap = scan._build_standard_analytical_snapshot({}, n_species=3, n_radial=5)
    assert snap.rho.shape == (5,)
    assert snap.density.shape == (3, 5)
    assert snap.temperature.shape == (3, 5)
    assert snap.er.shape == (5,)
    assert snap.time_value is None


def test_snapshot_core_and_edge_values() -> None:
    snap = scan._build_standard_analytical_snapshot({}, n_species=3, n_radial=5)
    # defaults: n0 = 4.21, n_edge = 0.6, T0 = 17.8, T_edge = 0.7; electron scale = 1.0
    assert_allclose(snap.density[0, 0], 4.21)
    assert_allclose(snap.density[0, -1], 0.6)
    assert_allclose(snap.temperature[:, 0], 17.8)
    assert_allclose(snap.temperature[:, -1], 0.7)


def test_snapshot_species_density_ratios() -> None:
    cfg = {"profiles": {"deuterium_ratio": 0.6, "tritium_ratio": 0.4}}
    snap = scan._build_standard_analytical_snapshot(cfg, n_species=3, n_radial=5)
    assert_allclose(snap.density[1], 0.6 * snap.density[0])
    assert_allclose(snap.density[2], 0.4 * snap.density[0])
