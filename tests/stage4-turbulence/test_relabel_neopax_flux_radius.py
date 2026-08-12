"""Tests for the Stage 4 to Stage 5 radial-grid adapter.

The behaviour under test is that the flux file's ``r`` grid ends up exactly on ``a_minor * rho`` for
the minor-radius convention NEOPAX builds its own grid on, so every knot sits at the radius NEOPAX
means by it rather than one about a percent away, and NEOPAX's grid stays inside the knots instead of
running off the end into NaN.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

from common import neopax_geometry as geometry
from tests.helpers.stage_import import load_stage_module

relabel = load_stage_module("stages/stage4-turbulence/relabel_neopax_flux_radius.py")

# Measured from the W7-X equilibrium this adapter was written for. NEOPAX's grid is the wider one
# here, which is the direction that leaves its outermost cell off the end of the flux file.
W7X_VOLUME_P = 26.468223986450294
W7X_R00_BOOZER = 5.395695758496847
W7X_RMAJOR_P = 5.5
W7X_A_B = 0.4985099181504102
W7X_AMINOR_P = 0.49441695148839837

# Measured from the HSX equilibrium `inputs/quick_run` ships. The skew runs the other way here, so
# nothing in this script may assume a fixed direction.
HSX_VOLUME_P = 0.35303080303674406
HSX_R00_BOOZER = 1.2370115143386635
HSX_A_B = 0.1202415479823416
HSX_AMINOR_P = 0.12150780545936438

N_R = 9


def _write_flux_file(path: Path, *, edge_radius: float, n_r: int = N_R) -> np.ndarray:
    """Write a minimal neopax_fluxes.h5 on ``r = edge_radius * rho`` and return that grid."""
    rho = np.linspace(0.0, 1.0, n_r)
    r = edge_radius * rho
    with h5py.File(path, "w") as f:
        f.create_dataset("rho", data=rho)
        f.create_dataset("r", data=r)
        f.create_dataset("Q", data=np.tile(np.linspace(1.0, 0.1, n_r), (2, 1)))
        f.create_dataset("Gamma", data=np.zeros((2, n_r)))
    return r


def _write_wout(path: Path, *, volume_p: float | None, r_major: float, a_minor: float | None) -> Path:
    """Write a VMEC wout holding only the scalars this script reads."""
    from netCDF4 import Dataset

    with Dataset(path, "w") as ds:
        if volume_p is not None:
            ds.createVariable("volume_p", "f8")[...] = volume_p
        ds.createVariable("Rmajor_p", "f8")[...] = r_major
        if a_minor is not None:
            ds.createVariable("Aminor_p", "f8")[...] = a_minor
    return path


def _write_boozmn(path: Path, *, r00: float, with_rmnc: bool = True) -> Path:
    """Write a Boozer file whose boundary surface carries the given R00."""
    from netCDF4 import Dataset

    n_surf, n_mode = 3, 4
    with Dataset(path, "w") as ds:
        ds.createDimension("pack_rad", n_surf)
        ds.createDimension("mn_mode", n_mode)
        if with_rmnc:
            rmnc_b = np.zeros((n_surf, n_mode))
            rmnc_b[:, 0] = np.linspace(0.5 * r00, r00, n_surf)  # R00 rises to its boundary value
            ds.createVariable("rmnc_b", "f8", ("pack_rad", "mn_mode"))[:] = rmnc_b
    return path


# --- minor_radius_from_volume ---

def test_minor_radius_from_volume_reproduces_the_w7x_a_b():
    """The formula must match what NEOPAX derives, or the relabelling targets the wrong grid."""
    assert_allclose(geometry.minor_radius_from_volume(W7X_VOLUME_P, W7X_R00_BOOZER), W7X_A_B, rtol=1e-12)


# The two radii disagree in opposite directions on the two shipped equilibria, a_b/Aminor_p being
# 1.00828 for W7-X and 0.98958 for HSX, so a fixed sign must never be assumed.
def test_the_two_radii_disagree_in_both_directions():
    """W7-X puts NEOPAX's grid outside the flux file's; HSX puts it inside."""
    w7x_ratio = geometry.minor_radius_from_volume(W7X_VOLUME_P, W7X_R00_BOOZER) / W7X_AMINOR_P
    hsx_ratio = geometry.minor_radius_from_volume(HSX_VOLUME_P, HSX_R00_BOOZER) / HSX_AMINOR_P
    assert_allclose(w7x_ratio, 1.00828, rtol=1e-4)
    assert_allclose(hsx_ratio, 0.98958, rtol=1e-4)


@pytest.mark.parametrize(("volume_p", "r_major"), [(0.0, 5.0), (-1.0, 5.0), (26.0, 0.0), (26.0, -5.0), (np.nan, 5.0)])
def test_minor_radius_from_volume_rejects_invalid_geometry(volume_p, r_major):
    with pytest.raises(ValueError):
        geometry.minor_radius_from_volume(volume_p, r_major)


# --- read_neopax_minor_radius ---

def test_read_neopax_minor_radius_uses_the_boundary_r00(tmp_path):
    """The radius takes R00 from the Boozer file's last surface, as NEOPAX does."""
    wout = _write_wout(tmp_path / "wout.nc", volume_p=W7X_VOLUME_P, r_major=W7X_RMAJOR_P, a_minor=W7X_AMINOR_P)
    boozer = _write_boozmn(tmp_path / "boozmn.nc", r00=W7X_R00_BOOZER)

    assert_allclose(geometry.read_neopax_minor_radius(wout, boozer), W7X_A_B, rtol=1e-12)


def test_read_neopax_minor_radius_reports_a_wout_without_volume(tmp_path):
    wout = _write_wout(tmp_path / "wout.nc", volume_p=None, r_major=W7X_RMAJOR_P, a_minor=W7X_AMINOR_P)
    boozer = _write_boozmn(tmp_path / "boozmn.nc", r00=W7X_R00_BOOZER)

    with pytest.raises(KeyError, match="volume_p"):
        geometry.read_neopax_minor_radius(wout, boozer)


def test_read_neopax_minor_radius_reports_a_boozer_file_without_rmnc(tmp_path):
    wout = _write_wout(tmp_path / "wout.nc", volume_p=W7X_VOLUME_P, r_major=W7X_RMAJOR_P, a_minor=W7X_AMINOR_P)
    boozer = _write_boozmn(tmp_path / "boozmn.nc", r00=W7X_R00_BOOZER, with_rmnc=False)

    with pytest.raises(KeyError, match="rmnc_b"):
        geometry.read_neopax_minor_radius(wout, boozer)


# --- neopax_radial_grid ---

def test_neopax_radial_grid_matches_neopax_construction():
    """NEOPAX builds r_grid = linspace(0, 1, n_r) * a; the relabelled grid must equal it exactly."""
    rho = np.linspace(0.0, 1.0, N_R)
    assert_allclose(relabel.neopax_radial_grid(rho, W7X_A_B), np.linspace(0.0, 1.0, N_R) * W7X_A_B, rtol=0, atol=0)


def test_neopax_radial_grid_rejects_grid_that_stops_short_of_edge():
    """A grid not reaching rho_edge would still leave NEOPAX's outermost cell uncovered."""
    with pytest.raises(ValueError, match="must reach rho_edge"):
        relabel.neopax_radial_grid(np.linspace(0.0, 0.9, N_R), W7X_A_B)


# NEOPAX faces its cells at linspace(0, rho_edge, n_radial + 1), so a run with [geometry].rho_edge < 1 -- the
# Trinity3D W7-X benchmark uses 0.7 -- writes Stage 4 fluxes out to rho_edge, not 1. The outer
# check has to follow rho_edge or it would reject exactly the grid it is meant to accept.
def test_neopax_radial_grid_accepts_a_grid_that_reaches_a_non_unit_rho_edge():
    """A rho_edge = 0.7 grid is the correct full grid for that run, not a truncated one."""
    rho = np.linspace(0.0, 0.7, N_R)
    assert_allclose(relabel.neopax_radial_grid(rho, W7X_A_B, rho_edge=0.7), rho * W7X_A_B, rtol=0, atol=0)


# Coverage is one-sided, so a grid running past rho_edge is fine, since the extra knots sit outside
# the transport domain and an interpolation never reads them. Only a shortfall creates a NaN.
def test_neopax_radial_grid_accepts_a_grid_wider_than_rho_edge():
    rho = np.linspace(0.0, 1.0, N_R)
    assert_allclose(relabel.neopax_radial_grid(rho, W7X_A_B, rho_edge=0.7), rho * W7X_A_B, rtol=0, atol=0)


@pytest.mark.parametrize("bad_edge", [0.0, -0.5, float("nan")])
def test_neopax_radial_grid_rejects_a_nonphysical_rho_edge(bad_edge):
    """rho_edge scales the whole grid, so a non-positive or non-finite value is never recoverable."""
    with pytest.raises(ValueError, match="rho_edge must be finite and positive"):
        relabel.neopax_radial_grid(np.linspace(0.0, 0.7, N_R), W7X_A_B, rho_edge=bad_edge)


@pytest.mark.parametrize("rho", [
    np.array([0.0, 0.5, 0.4, 1.0]),          # not increasing
    np.array([-0.1, 0.5, 1.0]),              # starts below zero
    np.array([0.0, np.nan, 1.0]),            # non-finite
])
def test_neopax_radial_grid_rejects_malformed_rho(rho):
    with pytest.raises(ValueError):
        relabel.neopax_radial_grid(rho, W7X_A_B)


# NEOPAX grids from 0, so a file starting further in leaves its inner cells outside the knots just
# as surely as a short outer end does. Stage 4's rho_min / num_radii / rho_indices options reach
# exactly this shape, because `collect` re-inserts the axis only for a scan that covered every
# other surface.
def test_neopax_radial_grid_rejects_a_grid_that_starts_inside_the_axis():
    with pytest.raises(ValueError, match="must start at 0"):
        relabel.neopax_radial_grid(np.linspace(0.1, 1.0, N_R), W7X_A_B)


# Interior spacing is not constrained on purpose, since a sparse grid covering the same interval is
# what the subsampling options exist to produce, and NEOPAX interpolates onto its own grid from it.
def test_neopax_radial_grid_accepts_a_sparse_grid_that_spans_the_interval():
    rho = np.array([0.0, 0.3, 0.95, 1.0])
    assert_allclose(relabel.neopax_radial_grid(rho, W7X_A_B), rho * W7X_A_B, rtol=0, atol=0)


# Coverage admits no tolerance, because interpax has none. A knot placed just inside a target still
# leaves that target outside the knots and NaN at it, so any tolerant comparison would pass grids
# that are still broken. These two are the smallest shortfalls a tolerant check would have missed.
@pytest.mark.parametrize(("rho", "rho_edge", "message"), [
    (np.concatenate([[5.0e-13], np.linspace(0.2, 1.0, N_R - 1)]), 1.0, "must start at 0"),
    (np.linspace(0.0, 1.0 - 5.0e-10, N_R), 1.0, "must reach rho_edge"),
])
def test_neopax_radial_grid_rejects_a_shortfall_no_tolerance_would_catch(rho, rho_edge, message):
    assert np.isclose(rho[0], 0.0, atol=1e-12) or np.isclose(rho[-1], rho_edge, rtol=1e-9), (
        "these cases exist because a tolerant comparison would accept them"
    )
    with pytest.raises(ValueError, match=message):
        relabel.neopax_radial_grid(rho, W7X_A_B, rho_edge=rho_edge)


# --- relabel_flux_radius ---

def test_relabel_moves_grid_onto_neopax_edge(tmp_path):
    """The whole point: after relabelling, the edge sits on a_b, not Aminor_p."""
    path = tmp_path / "neopax_fluxes.h5"
    _write_flux_file(path, edge_radius=W7X_AMINOR_P)

    result = relabel.relabel_flux_radius(path, W7X_A_B)

    assert result["changed"] is True
    with h5py.File(path, "r") as f:
        r = np.asarray(f["r"][...])
        rho = np.asarray(f["rho"][...])
    assert_allclose(r, W7X_A_B * rho, rtol=0, atol=0)
    assert_allclose(r[-1], W7X_A_B, rtol=1e-12)


def test_relabel_leaves_flux_values_untouched(tmp_path):
    """Only the coordinate is rewritten; Stage 4's gyro-Bohm normalization must not shift."""
    path = tmp_path / "neopax_fluxes.h5"
    _write_flux_file(path, edge_radius=W7X_AMINOR_P)
    with h5py.File(path, "r") as f:
        q_before = np.asarray(f["Q"][...])
        gamma_before = np.asarray(f["Gamma"][...])

    relabel.relabel_flux_radius(path, W7X_A_B)

    with h5py.File(path, "r") as f:
        assert_allclose(np.asarray(f["Q"][...]), q_before, rtol=0, atol=0)
        assert_allclose(np.asarray(f["Gamma"][...]), gamma_before, rtol=0, atol=0)


def test_relabel_is_idempotent(tmp_path):
    """Re-running on an already-relabelled file is a no-op, so reruns cannot compound the shift."""
    path = tmp_path / "neopax_fluxes.h5"
    _write_flux_file(path, edge_radius=W7X_AMINOR_P)

    relabel.relabel_flux_radius(path, W7X_A_B)
    with h5py.File(path, "r") as f:
        r_once = np.asarray(f["r"][...])

    second = relabel.relabel_flux_radius(path, W7X_A_B)

    assert second["changed"] is False
    with h5py.File(path, "r") as f:
        assert_allclose(np.asarray(f["r"][...]), r_once, rtol=0, atol=0)


def test_relabel_dry_run_does_not_write(tmp_path):
    path = tmp_path / "neopax_fluxes.h5"
    r_before = _write_flux_file(path, edge_radius=W7X_AMINOR_P)

    result = relabel.relabel_flux_radius(path, W7X_A_B, dry_run=True)

    assert result["changed"] is False
    with h5py.File(path, "r") as f:
        assert_allclose(np.asarray(f["r"][...]), r_before, rtol=0, atol=0)


def test_relabel_records_provenance_attrs(tmp_path):
    """A relabelled file must be self-describing, so a stray one is recognisable later."""
    path = tmp_path / "neopax_fluxes.h5"
    _write_flux_file(path, edge_radius=W7X_AMINOR_P)

    relabel.relabel_flux_radius(path, W7X_A_B)

    with h5py.File(path, "r") as f:
        attrs = f["meta"].attrs
        assert bool(attrs["neopax_radius_relabel_applied"]) is True
        assert attrs["neopax_radius_relabel_convention"] == "boozer_volume"
        assert_allclose(float(attrs["neopax_radius_relabel_a_minor_m"]), W7X_A_B, rtol=1e-12)
        assert_allclose(float(attrs["neopax_radius_relabel_original_r_edge_m"]), W7X_AMINOR_P, rtol=1e-12)


# The recorded original radius is the file's outermost knot, a_minor * rho[-1], which on this grid
# ending at rho_edge = 0.7 is 0.7 * a_minor and not the minor radius itself. The two must not be
# conflated.
def test_relabel_records_the_outermost_knot_not_the_minor_radius(tmp_path):
    rho_edge = 0.7
    path = tmp_path / "neopax_fluxes.h5"
    with h5py.File(path, "w") as f:
        rho = np.linspace(0.0, rho_edge, N_R)
        f.create_dataset("rho", data=rho)
        f.create_dataset("r", data=W7X_AMINOR_P * rho)

    relabel.relabel_flux_radius(path, W7X_A_B, rho_edge=rho_edge)

    with h5py.File(path, "r") as f:
        attrs = f["meta"].attrs
        assert_allclose(float(attrs["neopax_radius_relabel_a_minor_m"]), W7X_A_B, rtol=1e-12)
        assert_allclose(float(attrs["neopax_radius_relabel_original_r_edge_m"]), W7X_AMINOR_P * rho_edge, rtol=1e-12)


# minor_radius_m and conversion record the length the gyro-Bohm fluxes were converted with, so they
# must keep describing that conversion after a relabel. Only the coordinate's own description moves,
# or the file would claim its flux values were built from a radius they never saw.
def test_relabel_leaves_the_flux_normalization_metadata_alone(tmp_path):
    path = tmp_path / "neopax_fluxes.h5"
    _write_flux_file(path, edge_radius=W7X_AMINOR_P)
    with h5py.File(path, "r+") as f:
        meta = f.require_group("meta")
        meta.attrs["minor_radius_m"] = W7X_AMINOR_P
        meta.attrs["conversion"] = "Gamma_r = a * Gamma_rho"
        meta.attrs["radial_flux_coordinate"] = "saved Gamma/Q are converted to r = a*rho"

    relabel.relabel_flux_radius(path, W7X_A_B)

    with h5py.File(path, "r") as f:
        attrs = f["meta"].attrs
        assert_allclose(float(attrs["minor_radius_m"]), W7X_AMINOR_P, rtol=1e-12)
        assert attrs["conversion"] == "Gamma_r = a * Gamma_rho"
        assert "boozer_volume" in attrs["radial_flux_coordinate"]
        assert "minor_radius_m" in attrs["radial_flux_coordinate"]


def test_relabel_requires_rho_dataset(tmp_path):
    path = tmp_path / "neopax_fluxes.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("r", data=np.linspace(0.0, W7X_AMINOR_P, N_R))

    with pytest.raises(KeyError, match="rho"):
        relabel.relabel_flux_radius(path, W7X_A_B)


def test_relabel_rejects_an_unknown_convention(tmp_path):
    path = tmp_path / "neopax_fluxes.h5"
    _write_flux_file(path, edge_radius=W7X_AMINOR_P)

    with pytest.raises(ValueError, match="convention must be one of"):
        relabel.relabel_flux_radius(path, W7X_A_B, convention="boozer_aminor")


# --- the defect this exists to prevent ---

def test_original_w7x_grid_leaves_neopax_outermost_cell_uncovered():
    """Regression guard on the root cause: NEOPAX's last cell centre sits outside the old knots."""
    r_data = W7X_AMINOR_P * np.linspace(0.0, 1.0, N_R)
    r_grid = W7X_A_B * np.linspace(0.0, 1.0, N_R)
    uncovered = r_grid > r_data[-1]
    assert uncovered.sum() == 1, "exactly one cell centre should fall outside, matching the observed 8/10 faces"
    assert uncovered[-1]

    relabelled = relabel.neopax_radial_grid(np.linspace(0.0, 1.0, N_R), W7X_A_B)
    assert not np.any(r_grid > relabelled[-1]), "after relabelling nothing may fall outside the knots"


# The HSX skew runs the other way, so its unrelabelled grid covers NEOPAX's and produces no NaN.
# Relabelling still matters there, because it is what makes the FD lagged response usable at all.
def test_original_hsx_grid_covers_neopax_but_still_does_not_match_it():
    """The adapter is not only about coverage; FD needs the two grids equal to 1e-12."""
    r_data = HSX_AMINOR_P * np.linspace(0.0, 1.0, N_R)
    r_grid = HSX_A_B * np.linspace(0.0, 1.0, N_R)
    assert not np.any(r_grid > r_data[-1])
    assert not np.allclose(r_data, r_grid, rtol=1e-12, atol=0.0)

    relabelled = relabel.neopax_radial_grid(np.linspace(0.0, 1.0, N_R), HSX_A_B)
    assert_allclose(relabelled, r_grid, rtol=0, atol=0)
