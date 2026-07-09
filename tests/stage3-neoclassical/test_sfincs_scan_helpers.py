"""Tests for the Stage 3 sfincs_jax radial-scan helpers.

These reuse the pure functions from ``sfincs_jax_radial_scan.py``, loaded by path.
``_choose_radius_indices`` selects which flux surfaces to solve; the analytical
snapshot builder synthesises species-resolved profiles from config. The solver is
imported lazily inside the worker, so loading the module and calling these helpers
needs no solver.
"""

from __future__ import annotations

from types import ModuleType

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tests.helpers.stage_import import load_stage_module

scan = load_stage_module("stages/stage3-neoclassical/sfincs_jax_radial_scan.py")


# --- _choose_radius_indices ---

# `_choose_radius_indices` picks which flux surfaces the scan will solve. With no options given, it should skip the
# magnetic axis (rho = 0, at index 0) because a solve there is rarely useful. Given 5 radii, this asserts it returns
# indices [1, 2, 3, 4], i.e. everything except the axis.
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


# `num_radii` thins the candidate surfaces down to a smaller evenly-spaced sample. After skipping the axis the
# candidates are [1, 2, 3, 4]; asking for 2 should keep the two endpoints, so this asserts the result is [1, 4]. This
# lets a user run a cheaper, coarser scan.
def test_choose_radius_num_radii_subsamples() -> None:
    rho = np.linspace(0.0, 1.0, 5)  # candidates after axis skip: [1, 2, 3, 4]
    idxs = scan._choose_radius_indices(rho, explicit=None, rho_min=None, rho_max=None, num_radii=2)
    assert idxs == [1, 4]  # endpoints of the candidate span


# The rho range is 0 to 1, so a `rho_min` of 2.0 filters out every surface. Rather than return an empty list (which
# would make the scan do nothing), this asserts it raises ValueError so the impossible filter is reported clearly.
def test_choose_radius_empty_filter_raises() -> None:
    rho = np.linspace(0.0, 1.0, 5)
    with pytest.raises(ValueError):
        scan._choose_radius_indices(rho, explicit=None, rho_min=2.0, rho_max=None, num_radii=None)


# --- _build_standard_analytical_snapshot ---

# The Stage 3 and Stage 4 scan scripts each carry their own copy of
# _build_standard_analytical_snapshot. Running the snapshot tests against both copies
# guards against the duplicated helpers silently drifting apart. The Stage 4 module is
# also loaded by the Stage 4 test file; load_stage_module caches by file, so this reuses it.
spectrax_scan = load_stage_module("stages/stage4-turbulence/spectrax_gk_radial_scan.py")

SNAPSHOT_MODULES = [
    pytest.param(scan, id="sfincs_jax"),
    pytest.param(spectrax_scan, id="spectrax_gk"),
]


# `_build_standard_analytical_snapshot` synthesises test profiles (density, temperature, etc.) from config, used when no
# real upstream profiles are available. This builds a 3-species, 5-radius snapshot and asserts every array has the
# expected shape and that a static (non-time-resolved) snapshot has no time value. Parametrized to run against both
# Stage 3's and Stage 4's copy of the function.
@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
def test_snapshot_shapes(module: ModuleType) -> None:
    snap = module._build_standard_analytical_snapshot({}, n_species=3, n_radial=5)
    assert snap.rho.shape == (5,)
    assert snap.density.shape == (3, 5)
    assert snap.temperature.shape == (3, 5)
    assert snap.er.shape == (5,)
    assert snap.time_value is None


@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
def test_snapshot_core_and_edge_values(module: ModuleType) -> None:
    snap = module._build_standard_analytical_snapshot({}, n_species=3, n_radial=5)
    # defaults: n0 = 4.21, n_edge = 0.6, T0 = 17.8, T_edge = 0.7; electron scale = 1.0
    assert_allclose(snap.density[0, 0], 4.21, rtol=1e-12)
    assert_allclose(snap.density[0, -1], 0.6, rtol=1e-12)
    assert_allclose(snap.temperature[:, 0], 17.8, rtol=1e-12)
    assert_allclose(snap.temperature[:, -1], 0.7, rtol=1e-12)


# The ion densities are set as fractions of the electron density (species 0). Configuring deuterium at 0.6 and tritium
# at 0.4, this asserts species 1's density equals 0.6x and species 2's equals 0.4x the electron density at every radius,
# confirming the ratio config is applied correctly.
@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
def test_snapshot_species_density_ratios(module: ModuleType) -> None:
    cfg = {"profiles": {"deuterium_ratio": 0.6, "tritium_ratio": 0.4}}
    snap = module._build_standard_analytical_snapshot(cfg, n_species=3, n_radial=5)
    assert_allclose(snap.density[1], 0.6 * snap.density[0], rtol=1e-12)
    assert_allclose(snap.density[2], 0.4 * snap.density[0], rtol=1e-12)
