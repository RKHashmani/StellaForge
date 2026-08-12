"""Test the NEOPAX profile adapter shared by Stages 3 and 4.

``neopax_profiles.py`` supplies the face state from three profile sources. It can evaluate the
analytical model, read prescribed arrays, or read ``transport_solution.h5``. The radial scans select
one of these sources. These tests cover all three paths. A W7-X golden case compares the analytical
path with a real NEOPAX run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from common.neopax_profiles import (
    SpeciesMeta,
    _analytical_density_temperature,
    _analytical_er,
    build_analytical_face_state,
    build_prescribed_face_state,
    read_transport_face_state,
)


def _species(*entries: tuple[str, float]) -> list[SpeciesMeta]:
    """Build species rows from ``(name, charge)`` pairs and add an unused placeholder mass."""
    return [SpeciesMeta(name=name, charge=charge, mass_mp=1.0) for name, charge in entries]


# Species names select entries in per-species boundary tables. Without a boundary block, only the
# species count affects the profile formula. Charges locate the electron row. The short forms of
# ``c_density`` and ``n_scale`` omit this row.
THREE_SPECIES = _species(("e", -1.0), ("D", 1.0), ("T", 1.0))
TWO_SPECIES = _species(("e", -1.0), ("ion", 1.0))

# Each call must set ``n0``, ``n_edge``, ``T0`` and ``T_edge``. The NEOPAX fallback values
# already use SI units. Its model would incorrectly apply the reference scaling again.
# These values express the four fallbacks in adapter units: 1e20 m^-3 and keV.
CORE_PROFILES = {"n0": 4.21, "n_edge": 0.6, "T0": 17.8, "T_edge": 0.7}


def _charges(species: list[SpeciesMeta]) -> list[float]:
    """Return charges in profile row order."""
    return [entry.charge for entry in species]


# Analytical face state

# The analytical builder creates profiles when upstream profiles are unavailable. This test builds
# three species on five faces. It checks all shapes and the absence of a solution time.
def test_snapshot_shapes() -> None:
    snap = build_analytical_face_state(
        {"profiles": CORE_PROFILES}, species=THREE_SPECIES, n_faces=5, minor_radius=1.0
    )
    assert snap.rho.shape == (5,)
    assert snap.density.shape == (3, 5)
    assert snap.temperature.shape == (3, 5)
    assert snap.er.shape == (5,)
    assert snap.density_grad.shape == (3, 5)
    assert snap.temperature_grad.shape == (3, 5)
    assert snap.time_value is None


# The edge closure uses the two outermost centers. Therefore, the builder requires three faces.
# The prescribed profile source has the same minimum.
@pytest.mark.parametrize("n_faces", [0, 1, 2])
def test_snapshot_requires_at_least_three_faces(n_faces: int) -> None:
    with pytest.raises(ValueError, match="at least 3 faces"):
        build_analytical_face_state(
            {"profiles": CORE_PROFILES}, species=THREE_SPECIES, n_faces=n_faces, minor_radius=1.0
        )


# Every NEOPAX profile uses these four parameters. Their fallback values are not usable.
# The adapter rejects an omitted parameter before it can seed the loop.
@pytest.mark.parametrize("key", ["n0", "n_edge", "T0", "T_edge"])
def test_analytical_model_requires_the_four_core_parameters(key: str) -> None:
    profiles = {name: value for name, value in CORE_PROFILES.items() if name != key}
    with pytest.raises(ValueError, match=rf"\[profiles\]\.{key} is required"):
        _analytical_density_temperature(profiles, np.array([0.0]), rho_edge=1.0, charges=_charges(THREE_SPECIES))


# NEOPAX evaluates this formula on its cell centers. This test isolates the formula from face
# reconstruction. The profile meets its configured core and edge values. Without ``c_density``,
# species 0 has full density and each ion has half density.
def test_analytical_model_meets_the_configured_core_and_edge() -> None:
    density, temperature = _analytical_density_temperature(
        CORE_PROFILES, np.array([0.0, 1.0]), rho_edge=1.0, charges=_charges(THREE_SPECIES)
    )
    assert_allclose(density[0, 0], 4.21, rtol=1e-12)
    assert_allclose(density[0, -1], 0.6, rtol=1e-12)
    assert_allclose(temperature[:, 0], 17.8, rtol=1e-12)
    assert_allclose(temperature[:, -1], 0.7, rtol=1e-12)


# The shape functions use ``rho_edge``, not the last sample. The outermost center stops half a cell
# before the edge. Use of that center as the normalizer would reach the edge value one cell early.
# Shape checks would not detect this error.
def test_analytical_model_normalizes_by_rho_edge_not_the_last_sample() -> None:
    rho = np.array([0.25, 0.5])
    density, _ = _analytical_density_temperature(CORE_PROFILES, rho, rho_edge=1.0, charges=[1.0])
    assert_allclose(density[0], 0.6 + (4.21 - 0.6) * (1.0 - rho**2.0), rtol=1e-12)


# Ion densities are fractions of the electron density in species row 0. This case sets deuterium to
# 0.6 and tritium to 0.4. Linear face reconstruction preserves both ratios exactly.
def test_snapshot_species_density_ratios() -> None:
    cfg = {"profiles": {**CORE_PROFILES, "deuterium_ratio": 0.6, "tritium_ratio": 0.4}}
    snap = build_analytical_face_state(cfg, species=THREE_SPECIES, n_faces=5, minor_radius=1.0)
    assert_allclose(snap.density[1], 0.6 * snap.density[0], rtol=1e-12)
    assert_allclose(snap.density[2], 0.4 * snap.density[0], rtol=1e-12)


def _w7x_like_profiles() -> dict:
    """Return the two-species profile arrays from the W7-X validation input."""
    return {
        "geometry": {"n_radial": 5, "rho_edge": 0.7},
        "profiles": {
            "model": "standard_analytical",
            "n0": [0.35, 0.35],
            "n_edge": [0.29, 0.29],
            "T0": [6.7, 1.0],
            "T_edge": [0.8, 0.8],
            "density_shape_power": [2.0, 2.0],
            "temperature_shape_power": [2.0, 2.0],
            "temperature_shape_alpha": [2.0, 1.0],
            "er0_scale": 0.0,
            "er0_peak_rho": 0.8,
        },
    }


def _bounded_profiles() -> dict:
    """Add the W7-X validation boundary blocks to the profile config.

    Each field has a zero-gradient Neumann axis and per-species Dirichlet edge values.
    The species have different edge values. A collapsed boundary table therefore fails the test.
    """
    cfg = _w7x_like_profiles()
    cfg["boundary"] = {
        "density": {
            "left": {"type": "neumann", "gradient": {"default": 0.0}},
            "right": {"type": "dirichlet", "value": {"e": 0.29, "ion": 0.31}},
        },
        "temperature": {
            "left": {"type": "neumann", "gradient": {"default": 0.0}},
            "right": {"type": "dirichlet", "value": {"e": 0.8, "ion": 0.9}},
        },
    }
    return cfg


# Each scalar profile setting can also be a per-species list. This case gives the species different
# core temperatures and a common edge temperature. The test checks both endpoints.
def test_analytical_model_accepts_per_species_arrays() -> None:
    cfg = _w7x_like_profiles()
    density, temperature = _analytical_density_temperature(
        cfg["profiles"], np.array([0.0, 0.7]), rho_edge=0.7, charges=_charges(TWO_SPECIES)
    )
    assert_allclose(temperature[0, 0], 6.7, rtol=1e-12)  # electron core T0
    assert_allclose(temperature[1, 0], 1.0, rtol=1e-12)  # ion core T0
    assert_allclose(temperature[:, -1], 0.8, rtol=1e-12)  # shared T_edge
    assert_allclose(density[:, 0], 0.35, rtol=1e-12)
    assert_allclose(density[:, -1], 0.29, rtol=1e-12)


# ``temperature_shape_alpha`` is the outer exponent in the profile formula. This case uses equal
# powers and different alpha values. Each species must follow its own curve and normalized shape.
def test_analytical_model_applies_shape_alpha() -> None:
    cfg = _w7x_like_profiles()
    x = np.linspace(0.0, 1.0, 5)
    _, temperature = _analytical_density_temperature(
        cfg["profiles"], 0.7 * x, rho_edge=0.7, charges=_charges(TWO_SPECIES)
    )
    for i, (t0, alpha) in enumerate([(6.7, 2.0), (1.0, 1.0)]):
        expected = 0.8 + (t0 - 0.8) * (1.0 - x**2.0) ** alpha
        assert_allclose(temperature[i], expected, rtol=1e-12)
    # Normalised shapes differ between the two species.
    shape_e = (temperature[0] - 0.8) / (6.7 - 0.8)
    shape_i = (temperature[1] - 0.8) / (1.0 - 0.8)
    assert not np.allclose(shape_e, shape_i)


# A Dirichlet edge puts each configured value exactly on the edge face. This result comes from the
# face operator. Differentiation of interpolated face values cannot enforce it.
def test_snapshot_dirichlet_edge_face_carries_the_configured_value() -> None:
    snap = build_analytical_face_state(_bounded_profiles(), species=TWO_SPECIES, n_faces=6, minor_radius=0.5)
    assert_allclose(snap.density[:, -1], [0.29, 0.31], atol=0.0)
    assert_allclose(snap.temperature[:, -1], [0.8, 0.9], atol=0.0)


# A zero-gradient Neumann axis gives exactly zero for both gradients. A centered stencil across the
# axis gives a small nonzero slope. This difference is a boundary error, not roundoff.
def test_snapshot_neumann_axis_face_gradient_is_exactly_zero() -> None:
    snap = build_analytical_face_state(_bounded_profiles(), species=TWO_SPECIES, n_faces=6, minor_radius=0.5)
    assert_allclose(snap.density_grad[:, 0], 0.0, atol=0.0)
    assert_allclose(snap.temperature_grad[:, 0], 0.0, atol=0.0)


# The face operator uses ``minor_radius * rho`` and converts its result back with the same radius.
# The radius cancels when boundary coefficients contain no length. This applies to Dirichlet,
# zero-gradient Neumann, and absent boundary blocks. A Robin decay length makes the result depend on
# radius. This case detects omission of either the metre grid or the conversion.
def test_snapshot_gradients_are_minor_radius_invariant_without_a_length_scale() -> None:
    cfg = _w7x_like_profiles()
    unit = build_analytical_face_state(cfg, species=TWO_SPECIES, n_faces=6, minor_radius=1.0)
    halved = build_analytical_face_state(cfg, species=TWO_SPECIES, n_faces=6, minor_radius=0.5)
    assert_allclose(halved.density_grad, unit.density_grad, rtol=1e-12)
    assert_allclose(halved.temperature_grad, unit.temperature_grad, rtol=1e-12)

    cfg["boundary"] = {"temperature": {"right": {"type": "robin", "decay_length": 0.1}}}
    unit_robin = build_analytical_face_state(cfg, species=TWO_SPECIES, n_faces=6, minor_radius=1.0)
    halved_robin = build_analytical_face_state(cfg, species=TWO_SPECIES, n_faces=6, minor_radius=0.5)
    assert not np.allclose(halved_robin.temperature_grad, unit_robin.temperature_grad)


# [geometry].rho_edge sets the radial extent. This case requires ``linspace(0, 0.7, n)``.
# The shape functions use ``x = rho / rho_edge``.
def test_snapshot_grid_honours_rho_edge() -> None:
    snap = build_analytical_face_state(_w7x_like_profiles(), species=TWO_SPECIES, n_faces=5, minor_radius=1.0)
    assert_allclose(snap.rho, np.linspace(0.0, 0.7, 5), rtol=1e-12)


def test_snapshot_grid_defaults_rho_edge() -> None:
    cfg = _w7x_like_profiles()
    del cfg["geometry"]["rho_edge"]
    snap = build_analytical_face_state(cfg, species=TWO_SPECIES, n_faces=5, minor_radius=1.0)
    assert_allclose(snap.rho, np.linspace(0.0, 1.0, 5), rtol=1e-12)


def _er_cfg(right_type: str) -> dict:
    """The quick run's Er configuration on a 5-cell unit grid, with the edge type under test."""
    return {
        "geometry": {"rho_edge": 1.0},
        "profiles": {**CORE_PROFILES, "er0_scale": 100.0, "er0_peak_rho": 0.8},
        "boundary": {
            "Er": {"left": {"type": "dirichlet", "value": 0.0}, "right": {"type": right_type}},
        },
    }


# NEOPAX evaluates the ``Er`` parabola on centers and then reconstructs the faces. The centers are
# ``[7, 15, 15, 7, -9]`` kV/m. The Dirichlet axis sets 0.0. Interior faces average adjacent centers.
# The zero-gradient edge gives ``(9 * -9 - 7) / 8 = -11``.
# Direct face evaluation gives -20 at the edge. Stage 3 writes the reconstructed value to sfincs.
@pytest.mark.parametrize("right_type", ["floating_ambipolar_edge", "ambipolar_edge_root"])
def test_snapshot_er_is_reconstructed_from_the_cell_centers(right_type: str) -> None:
    snap = build_analytical_face_state(_er_cfg(right_type), species=THREE_SPECIES, n_faces=6, minor_radius=0.5)
    assert_allclose(snap.er, [0.0, 11.0, 15.0, 11.0, -1.0, -11.0], rtol=1e-12, atol=0.0)
    assert snap.er[-1] < 0.0  # x = 1 > er0_peak_rho = 0.8
    assert not np.allclose(snap.er, 100.0 * snap.rho * (0.8 - snap.rho))


# The ambipolar edge types select a solve, not a boundary closure. NEOPAX converts both to a
# zero-gradient Neumann edge before reconstruction. The adapter must reject any other unsupported
# type instead of applying this conversion.
def test_snapshot_er_rejects_an_unimplemented_edge_type() -> None:
    with pytest.raises(ValueError, match="not implemented here"):
        build_analytical_face_state(_er_cfg("ambipolar_something_else"), species=THREE_SPECIES, n_faces=6,
                                    minor_radius=0.5)


# The T3D reference case has no radial electric field. Therefore, ``er0_scale = 0.0`` must give
# exact zeros at all radii.
def test_snapshot_er_is_zero_when_scale_is_zero() -> None:
    snap = build_analytical_face_state(_w7x_like_profiles(), species=TWO_SPECIES, n_faces=5, minor_radius=1.0)
    assert_allclose(snap.er, np.zeros(5), atol=0.0)


# Electron-omitted short form

# NEOPAX inserts 1.0 at the electron row in short ``c_density`` and ``n_scale`` arrays.
# All other per-species arrays require the full length. The adapter must accept the same forms.

# These profiles are constant in radius and species. Only the two scale factors distinguish rows.
FLAT_PROFILES = {"n0": 2.0, "n_edge": 2.0, "T0": 1.0, "T_edge": 1.0}

# The last row is the electron. Incorrect insertion at row 0 is therefore visible.
ELECTRON_LAST = _species(("D", 1.0), ("T", 1.0), ("e", -1.0))
ALL_POSITIVE = _species(("D", 1.0), ("T", 1.0), ("He", 2.0))


def _density_factors(profiles: dict, species: list[SpeciesMeta]) -> np.ndarray:
    """Return density scale factors on the flat base profile."""
    density, _ = _analytical_density_temperature(
        {**FLAT_PROFILES, **profiles},
        np.array([0.0]),
        rho_edge=1.0,
        charges=_charges(species),
    )
    return density[:, 0] / 2.0


@pytest.mark.parametrize("key", ["c_density", "n_scale"])
@pytest.mark.parametrize(
    ("species", "expected"),
    [(THREE_SPECIES, [1.0, 3.0, 5.0]), (ELECTRON_LAST, [3.0, 5.0, 1.0])],
    ids=["electron_first", "electron_last"],
)
def test_short_form_inserts_one_at_the_electron(
    key: str, species: list[SpeciesMeta], expected: list[float]
) -> None:
    # Keep ``c_density`` flat when another key is under test.
    # This avoids the three-species ion defaults.
    profiles = {"c_density": [1.0, 1.0, 1.0], key: [3.0, 5.0]}
    assert_allclose(_density_factors(profiles, species), expected, rtol=1e-12)


# Shipped configs use full-length arrays. Every row must pass through unchanged.
@pytest.mark.parametrize("key", ["c_density", "n_scale"])
def test_full_length_form_is_unchanged(key: str) -> None:
    profiles = {"c_density": [1.0, 1.0, 1.0], key: [2.0, 3.0, 5.0]}
    assert_allclose(_density_factors(profiles, THREE_SPECIES), [2.0, 3.0, 5.0], rtol=1e-12)


# With two species, the short form has one entry. NEOPAX applies it only to the ion.
# The electron keeps 1.0. The array must not act as a scalar.
@pytest.mark.parametrize("key", ["c_density", "n_scale"])
def test_two_species_short_form_is_not_a_scalar_broadcast(key: str) -> None:
    assert_allclose(_density_factors({key: [4.0]}, TWO_SPECIES), [1.0, 4.0], rtol=1e-12)


# NEOPAX requires full-length arrays for other per-species keys.
# The adapter must reject a short array.
def test_short_form_is_rejected_for_a_key_neopax_requires_full_length() -> None:
    with pytest.raises(ValueError, match="Expected 3 profile factors, got 2"):
        _density_factors({"c_temperature": [3.0, 5.0]}, THREE_SPECIES)


# NEOPAX recognizes an electron only when the lowest charge is negative. An all-ion set has no
# insertion row. The adapter must reject its short form.
@pytest.mark.parametrize("key", ["c_density", "n_scale"])
def test_short_form_is_rejected_without_a_negative_charge(key: str) -> None:
    with pytest.raises(ValueError, match="Expected 3 profile factors, got 2"):
        _density_factors({"c_density": [1.0, 1.0, 1.0], key: [3.0, 5.0]}, ALL_POSITIVE)


# A one-entry sequence is not a scalar. NEOPAX applies only a bare scalar to all species.
# Acceptance of the sequence would defer failure to the transport solver.
@pytest.mark.parametrize("key", ["c_density", "c_temperature", "n_scale", "T_scale", "density_shape_power"])
def test_a_one_entry_sequence_is_not_a_scalar_broadcast(key: str) -> None:
    with pytest.raises(ValueError, match="profile factors, got 1"):
        _density_factors({"c_density": [1.0, 1.0, 1.0], key: [7.0]}, THREE_SPECIES)


# A bare scalar applies to all species for every supported key.
@pytest.mark.parametrize("key", ["c_density", "n_scale"])
def test_a_scalar_broadcasts_across_every_species(key: str) -> None:
    assert_allclose(_density_factors({"c_density": [1.0, 1.0, 1.0], key: 7.0}, THREE_SPECIES), [7.0, 7.0, 7.0],
                    rtol=1e-12)


# ``_build_state`` copies [species].charge_qp into [profiles].charge_qp when necessary.
# This occurs before profile evaluation. Species charges therefore locate the electron by default.
# All repository configs use this path.
@pytest.mark.parametrize("key", ["c_density", "n_scale"])
def test_the_species_charges_locate_the_electron_when_the_block_omits_charge_qp(key: str) -> None:
    cfg = {"profiles": {**FLAT_PROFILES, "c_density": [1.0, 1.0, 1.0], key: [3.0, 5.0]}}
    snap = build_analytical_face_state(cfg, species=ELECTRON_LAST, n_faces=4, minor_radius=1.0)
    # The profile is flat. Its axis value is the scale factor times the base density of 2.0.
    assert_allclose(snap.density[:, 0], np.array([3.0, 5.0, 1.0]) * 2.0, rtol=1e-12)


# [profiles].charge_qp has priority when present. Here it places the electron in a different row.
# The inserted 1.0 must follow that row.
@pytest.mark.parametrize("key", ["c_density", "n_scale"])
def test_the_profiles_charges_override_the_species_charges(key: str) -> None:
    cfg = {"profiles": {**FLAT_PROFILES, "c_density": [1.0, 1.0, 1.0], key: [3.0, 5.0],
                        "charge_qp": _charges(ELECTRON_LAST)}}
    snap = build_analytical_face_state(cfg, species=THREE_SPECIES, n_faces=4, minor_radius=1.0)
    assert_allclose(snap.density[:, 0], np.array([3.0, 5.0, 1.0]) * 2.0, rtol=1e-12)


# This test repeats the pinned rules against a live NEOPAX installation. It uses ``_build_state``,
# which is the real orchestrator entry point. The charge fallback occurs there, not in
# ``build_profiles``. Thus, the test detects an upstream change to the expansion rules.
def test_short_form_matches_live_neopax() -> None:
    pytest.importorskip("NEOPAX")

    from NEOPAX._orchestrator import _build_species, _build_state

    class _Field:
        r_grid = np.array([0.0])
        r_grid_half = np.array([1.0])

    for species in (THREE_SPECIES, ELECTRON_LAST):
        profiles = {"model": "standard_analytical", **FLAT_PROFILES, "c_density": [1.0, 1.0, 1.0],
                    "n_scale": [3.0, 5.0]}
        for charge_source in ("species", "profiles"):
            # Both sources provide the same charges.
            # Their locations must not change expansion.
            config = {
                "species": {"n_species": len(species), "names": [entry.name for entry in species],
                            "mass_mp": [entry.mass_mp for entry in species],
                            "charge_qp": _charges(species) if charge_source == "species" else [1.0] * len(species)},
                "profiles": profiles if charge_source == "species" else {**profiles,
                                                                         "charge_qp": _charges(species)},
            }
            upstream = np.asarray(
                _build_state(config, _Field(), _build_species(config)).density
            )[:, 0]
            assert_allclose(_density_factors(profiles, species) * 2.0, upstream, rtol=1e-12)


# Prescribed face state

# The prescribed block uses m^-3, eV and kV/m. The face state uses 1e20 m^-3, keV and
# kV/m. The block has no radial coordinate. The builder converts units and rebuilds the grid from
# [geometry].
#
# Plain arrays contain ``n_radial`` center values. NEOPAX reads these arrays.
# The ``*_face`` arrays contain ``n_radial + 1`` face values. The scans use these face values.
# Fixtures give the grids different magnitudes to expose use of the wrong grid.
# The ``*_grad_face`` arrays contain NEOPAX gradients in SI units per unit rho.
def _prescribed_cfg(n_species: int = 3, n_radial: int = 5) -> dict:
    density = [[(s + 1) * (r + 1) * 1.0e19 for r in range(n_radial)] for s in range(n_species)]
    temperature = [[(s + 1) * (r + 1) * 1.0e3 for r in range(n_radial)] for s in range(n_species)]
    density_face = [[(s + 1) * (r + 1) * 2.0e19 for r in range(n_radial + 1)] for s in range(n_species)]
    temperature_face = [[(s + 1) * (r + 1) * 2.0e3 for r in range(n_radial + 1)] for s in range(n_species)]
    density_grad_face = [[(s + 1) * (r + 1) * 3.0e19 for r in range(n_radial + 1)] for s in range(n_species)]
    temperature_grad_face = [[(s + 1) * (r + 1) * -3.0e3 for r in range(n_radial + 1)] for s in range(n_species)]
    return {
        "geometry": {"n_radial": n_radial, "rho_edge": 0.7},
        "profiles": {
            "model": "prescribed",
            "density": density,
            "temperature": temperature,
            "Er": [float(r) for r in range(n_radial)],
            "density_face": density_face,
            "temperature_face": temperature_face,
            "Er_face": [10.0 + r for r in range(n_radial + 1)],
            "density_grad_face": density_grad_face,
            "temperature_grad_face": temperature_grad_face,
        },
    }


# The state uses ``n_radial + 1`` faces across ``[0, rho_edge]``. The builder converts the face
# arrays from SI. Centered arrays would put fewer values at incorrect radii.
# Gradients use the same reference constants as their profiles. They remain per unit rho, so this
# path does not use the minor radius.
def test_prescribed_snapshot_units_and_grid() -> None:
    snap = build_prescribed_face_state(_prescribed_cfg(), n_species=3)
    expected = np.array([[(s + 1) * (r + 1) for r in range(6)] for s in range(3)], dtype=float)
    assert_allclose(snap.density, expected * 0.2, rtol=1e-12)  # 2e19 m^-3 becomes 0.2.
    assert_allclose(snap.temperature, expected * 2.0, rtol=1e-12)  # 2e3 eV becomes 2 keV.
    assert_allclose(snap.density_grad, expected * 0.3, rtol=1e-12)  # 3e19 per rho becomes 0.3.
    assert_allclose(snap.temperature_grad, expected * -3.0, rtol=1e-12)  # -3e3 becomes -3 keV.
    assert_allclose(snap.er, 10.0 + np.arange(6.0), rtol=1e-12)  # kV/m stays unchanged.
    assert_allclose(snap.rho, np.linspace(0.0, 0.7, 6), rtol=1e-12)
    assert snap.time_value is None


# Without [geometry].rho_edge, the builder uses the default full-radius face grid.
def test_prescribed_snapshot_defaults_rho_edge() -> None:
    cfg = _prescribed_cfg()
    del cfg["geometry"]["rho_edge"]
    snap = build_prescribed_face_state(cfg, n_species=3)
    assert_allclose(snap.rho, np.linspace(0.0, 1.0, 6), rtol=1e-12)


# An old or hand-written block can contain only centered arrays. The adapter cannot safely create
# face values or gradients. NEOPAX builds both with the run's [boundary] models.
# Local extrapolation could differ at a non-Dirichlet boundary.
# The error identifies the missing key.
@pytest.mark.parametrize(
    "key", ["density_face", "temperature_face", "Er_face", "density_grad_face", "temperature_grad_face"]
)
def test_prescribed_snapshot_requires_the_face_arrays(key: str) -> None:
    cfg = _prescribed_cfg()
    del cfg["profiles"][key]
    with pytest.raises(ValueError, match=rf"\[profiles\]\.{key}"):
        build_prescribed_face_state(cfg, n_species=3)


# Face arrays have one more radius than centered arrays. A short face array omits the outer edge and
# shifts the other samples inward by half a cell.
@pytest.mark.parametrize("key", ["density_face", "density_grad_face", "temperature_grad_face"])
def test_prescribed_snapshot_face_width_validation(key: str) -> None:
    cfg = _prescribed_cfg()
    cfg["profiles"][key] = [row[:-1] for row in cfg["profiles"][key]]
    with pytest.raises(ValueError, match=rf"\[profiles\]\.{key}"):
        build_prescribed_face_state(cfg, n_species=3)
    cfg = _prescribed_cfg()
    cfg["profiles"]["Er_face"] = cfg["profiles"]["Er_face"][:-1]
    with pytest.raises(ValueError, match=r"\[profiles\]\.Er_face"):
        build_prescribed_face_state(cfg, n_species=3)


# Only ``prescribed`` activates this builder. The feedback writer always emits this value.
# Rejection of the ``given`` synonym keeps Stage 3 and Stage 4 consistent with the pipeline output.
@pytest.mark.parametrize("model", ["standard_analytical", "given"])
def test_prescribed_snapshot_rejects_other_models(model: str) -> None:
    cfg = _prescribed_cfg()
    cfg["profiles"]["model"] = model
    with pytest.raises(ValueError, match="requires 'prescribed'"):
        build_prescribed_face_state(cfg, n_species=3)


@pytest.mark.parametrize("key", ["density", "temperature", "Er"])
def test_prescribed_snapshot_requires_each_array(key: str) -> None:
    cfg = _prescribed_cfg()
    del cfg["profiles"][key]
    with pytest.raises(ValueError, match=rf"\[profiles\]\.{key}"):
        build_prescribed_face_state(cfg, n_species=3)


def test_prescribed_snapshot_shape_validation() -> None:
    with pytest.raises(ValueError, match="species rows"):
        build_prescribed_face_state(_prescribed_cfg(), n_species=2)
    cfg = _prescribed_cfg()
    cfg["profiles"]["Er"] = [0.0, 1.0]
    with pytest.raises(ValueError, match=r"\[profiles\]\.Er"):
        build_prescribed_face_state(cfg, n_species=3)
    cfg = _prescribed_cfg()
    cfg["geometry"]["n_radial"] = 7
    with pytest.raises(ValueError, match=r"\[geometry\]\.n_radial"):
        build_prescribed_face_state(cfg, n_species=3)
    with pytest.raises(ValueError, match="at least 3 faces"):
        build_prescribed_face_state(_prescribed_cfg(n_radial=1), n_species=3)


# The minimum applies to faces. NEOPAX uses the two outermost centers for edge extrapolation.
# Therefore, two cells and three faces are sufficient.
def test_prescribed_snapshot_accepts_two_cells() -> None:
    snap = build_prescribed_face_state(_prescribed_cfg(n_radial=2), n_species=3)
    assert_allclose(snap.rho, np.linspace(0.0, 0.7, 3), rtol=1e-12)


# Transport face state

# The transport source reads ``transport_solution.h5`` instead of the feedback block. It uses the
# same face grid as the prescribed source. Centered values exclude both radial endpoints.
# A flux file on those centers would stop short at both ends. NEOPAX or the relabel step rejects it.

# This minor radius relates the two fixture face grids. Its value is not 1.0.
# An omitted, doubled, or inverted gradient conversion therefore gives a different result.
TRANSPORT_MINOR_RADIUS = 0.5


# These values define the fixture clock. The reader requires both scalars in every file.
# The fixtures have no ``ts`` axis, so these tests cover presence and the sign of ``next_dt``.
TRANSPORT_FINAL_TIME = 0.25
TRANSPORT_NEXT_DT = 0.1


def _write_transport_h5(
    path: Path, *, n_radial: int = 5, n_species: int = 3, faces: bool = True, time_resolved: bool = True
) -> Path:
    """Write a transport solution with the pinned NEOPAX layout.

    NEOPAX writes a time series with shape ``(n_time, n_species, n_rho)``. A static slice has shape
    ``(n_species, n_rho)``. The reader must distinguish these ranks. The leading static axis is the
    species axis. Incorrect time indexing would return one species row. ``offset`` distinguishes
    the final time slice.
    """
    from tests.helpers.synthetic import write_transport_solution

    offset = 100.0 if time_resolved else 0.0
    ramp_c = np.arange(n_species * n_radial, dtype=float).reshape(n_species, n_radial)
    ramp_f = np.arange(n_species * (n_radial + 1), dtype=float).reshape(n_species, n_radial + 1)
    face_grid = np.linspace(0.0, 1.0, n_radial + 1)

    def lay(a: np.ndarray) -> np.ndarray:
        """Return ``a`` as one static slice or two slices ending at ``a + offset``."""
        return np.stack([a, a + offset]) if time_resolved else a

    extra = {}
    if faces:
        extra = dict(
            rho_face=face_grid,
            r_grid_half=TRANSPORT_MINOR_RADIUS * face_grid,
            density_face=lay(5.0 + 0.1 * ramp_f),
            temperature_face=lay(7.0 + 0.2 * ramp_f),
            er_face=lay(np.linspace(-2.0, 4.0, n_radial + 1)),
            density_grad_face=lay(0.3 + 0.01 * ramp_f),
            temperature_grad_face=lay(-0.4 - 0.02 * ramp_f),
        )
    return write_transport_solution(
        path,
        rho=0.5 * (face_grid[:-1] + face_grid[1:]),
        density=lay(1.0 + 0.1 * ramp_c),
        temperature=lay(2.0 + 0.2 * ramp_c),
        er=lay(np.linspace(-1.0, 3.0, n_radial)),
        final_time=TRANSPORT_FINAL_TIME,
        next_dt=TRANSPORT_NEXT_DT,
        **extra,
    )


# The reader must return face values on the face grid, as the prescribed source does.
# Face and center fixtures use different magnitudes to expose selection of centered data.
# The two file ranks test the distinction between time and species axes. Incorrect static indexing
# returns one species row and scalar ``Er``. Later shape checks can accept that result.
# The gradient datasets use metre-valued ``r_grid_half``. The reader converts them with the radius
# implied by the two face grids.
@pytest.mark.parametrize("time_resolved", [True, False], ids=["time_resolved", "static"])
def test_transport_snapshot_takes_the_face_grid(time_resolved: bool, tmp_path: Path) -> None:
    path = _write_transport_h5(tmp_path / "transport_solution.h5", time_resolved=time_resolved)
    snap = read_transport_face_state(path, time_index=-1)
    ramp_f = np.arange(3 * 6, dtype=float).reshape(3, 6)
    offset = 100.0 if time_resolved else 0.0  # the last of two slices, or the only one
    assert_allclose(snap.rho, np.linspace(0.0, 1.0, 6), rtol=1e-12)
    assert_allclose(snap.density, offset + 5.0 + 0.1 * ramp_f, rtol=1e-12)
    assert_allclose(snap.temperature, offset + 7.0 + 0.2 * ramp_f, rtol=1e-12)
    assert_allclose(snap.er, offset + np.linspace(-2.0, 4.0, 6), rtol=1e-12)
    assert_allclose(snap.density_grad, TRANSPORT_MINOR_RADIUS * (offset + 0.3 + 0.01 * ramp_f), rtol=1e-12)
    assert_allclose(snap.temperature_grad, TRANSPORT_MINOR_RADIUS * (offset - 0.4 - 0.02 * ramp_f), rtol=1e-12)


# The conversion requires ``r_grid_half = a * rho_face`` for one radius ``a``. A bent metre grid
# causes radius-dependent gradient errors. Shape and finiteness checks do not detect them.
def test_transport_snapshot_rejects_a_nonlinear_metre_grid(tmp_path: Path) -> None:
    from tests.helpers.synthetic import write_transport_solution

    face_grid = np.linspace(0.0, 1.0, 6)
    ramp_f = np.arange(3 * 6, dtype=float).reshape(3, 6)
    path = write_transport_solution(
        tmp_path / "bent.h5",
        rho=0.5 * (face_grid[:-1] + face_grid[1:]),
        density=np.ones((3, 5)), temperature=np.ones((3, 5)), er=np.zeros(5),
        rho_face=face_grid,
        r_grid_half=TRANSPORT_MINOR_RADIUS * face_grid**2,
        density_face=5.0 + 0.1 * ramp_f,
        temperature_face=7.0 + 0.2 * ramp_f,
        er_face=np.linspace(-2.0, 4.0, 6),
        density_grad_face=0.3 + 0.01 * ramp_f,
        temperature_grad_face=-0.4 - 0.02 * ramp_f,
        final_time=TRANSPORT_FINAL_TIME,
        next_dt=TRANSPORT_NEXT_DT,
    )
    with pytest.raises(ValueError, match="not rho_face scaled by one minor radius"):
        read_transport_face_state(path, time_index=-1)


# A solution from an old NEOPAX version can lack face datasets. Centered data covers neither radial
# endpoint. The reader rejects the file and identifies the missing data.
def test_transport_snapshot_requires_the_face_datasets(tmp_path: Path) -> None:
    path = _write_transport_h5(tmp_path / "old_pin.h5", faces=False)
    with pytest.raises(KeyError, match="density_faces"):
        read_transport_face_state(path, time_index=-1)


# A file without the NEOPAX centered state is not a transport solution. The reader reports the
# missing core layout before it reports missing face data.
def test_transport_snapshot_requires_the_cell_centered_datasets(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "not_a_solution.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("r", data=np.linspace(0.0, 1.0, 6))
    with pytest.raises(KeyError, match="missing required NEOPAX transport datasets"):
        read_transport_face_state(path, time_index=-1)


# A static solution contains one slice. Only the default index selects it.
# Silent selection of index 0 would record an index that the reader did not honor.
# The mismatch would appear later when the feedback writer reads the same file.
@pytest.mark.parametrize("bad", [0, 1, -2])
def test_static_transport_snapshot_rejects_a_non_default_time_index(bad: int, tmp_path: Path) -> None:
    path = _write_transport_h5(tmp_path / "static.h5", time_resolved=False)
    with pytest.raises(ValueError, match="single static profile slice"):
        read_transport_face_state(path, time_index=bad)


# Every solution needs the clock. A file without it predates the applicable NEOPAX version.
# Its slices cannot be checked against the run time.
@pytest.mark.parametrize("dropped", ["final_time", "next_dt"])
def test_transport_snapshot_requires_the_clock(dropped: str, tmp_path: Path) -> None:
    import h5py

    path = _write_transport_h5(tmp_path / "clockless.h5")
    with h5py.File(path, "a") as f:
        del f[dropped]
    with pytest.raises(KeyError, match="transport clock datasets"):
        read_transport_face_state(path, time_index=-1)


# NEOPAX allocates all save slots before the run. It fills a slot after the clock passes its time.
# An early stop leaves zero-filled trailing slots, while ``final_time`` records the reached time.
# Any slice can then contain unwritten state. The reader rejects the complete file.
@pytest.mark.parametrize("time_index", [-1, 0])
def test_transport_snapshot_rejects_a_save_grid_stopping_short_of_the_run(time_index: int, tmp_path: Path) -> None:
    import h5py

    path = _write_transport_h5(tmp_path / "stopped_short.h5")
    with h5py.File(path, "a") as f:
        f.create_dataset("ts", data=np.array([0.0, TRANSPORT_FINAL_TIME / 2.0]))
    with pytest.raises(ValueError, match="describe different spans"):
        read_transport_face_state(path, time_index=time_index)


# A file dated through ``final_time`` is valid. This control isolates the early-stop defect above.
def test_transport_snapshot_accepts_a_save_grid_reaching_the_run(tmp_path: Path) -> None:
    import h5py

    path = _write_transport_h5(tmp_path / "reached.h5")
    with h5py.File(path, "a") as f:
        f.create_dataset("ts", data=np.array([0.0, TRANSPORT_FINAL_TIME]))
    assert read_transport_face_state(path, time_index=-1).time_value == TRANSPORT_FINAL_TIME


def _drop_in(path: Path, name: str, value: np.ndarray) -> Path:
    """Set one dataset of a written solution, keeping every other dataset as written."""
    import h5py

    with h5py.File(path, "a") as f:
        if name in f:
            del f[name]
        f.create_dataset(name, data=value)
    return path


# These mutations create cross-dataset defects in a valid static file. Each dataset remains
# plausible by itself. Only layout checks can detect the mismatch.
@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"ts": np.array([TRANSPORT_FINAL_TIME, 999.0])}, "holds 2 timestamps"),
        ({"density_faces": np.full((3, 6), np.nan)}, "non-finite values in the selected slice"),
        ({"density_faces": np.ones((3, 5))}, "does not match"),
        ({"temperature_faces": np.ones((3, 7))}, "does not match"),
        ({"Er_faces": np.zeros((3, 6))}, "inconsistent face dataset ranks"),
        ({"Er_faces": np.zeros(7)}, "expected"),
        (
            {"density": np.ones((2, 3, 5)), "temperature": np.ones((2, 3, 5)), "Er": np.ones((2, 5))},
            "disagree in rank",
        ),
        ({"density": np.ones((2, 5)), "temperature": np.ones((2, 5))}, "disagree in species count"),
        ({"rho_face": np.linspace(0.0, 1.0, 7)}, "one more face than centers"),
        ({"next_dt": np.float64(0.0)}, "would not advance on"),
    ],
    ids=[
        "static_two_timestamps", "non_finite_face", "face_radius_short", "face_temperature_width",
        "er_species_axis", "er_length", "rank_split", "species_split", "face_count", "zero_dt",
    ],
)
def test_transport_snapshot_rejects_cross_dataset_defects(
    overrides: dict[str, np.ndarray], match: str, tmp_path: Path
) -> None:
    path = _write_transport_h5(tmp_path / "defect.h5", time_resolved=False)
    for name, value in overrides.items():
        _drop_in(path, name, value)
    with pytest.raises(ValueError, match=match):
        read_transport_face_state(path, time_index=-1)


# W7-X analytical golden case

# The oracle is the ``ts = 0`` slice from
# ``outputs/_ab_evidence/rerun_8d73deb/transport_solution.h5``. NEOPAX produced it from
# ``outputs/_ab_evidence/ab_common_input.toml`` with the analytical model.
# The slice uses NEOPAX cell centers and boundary models. It therefore tests the complete analytical
# path. The ignored oracle file cannot run in CI, so this test includes its values.
# The gradients below use profile units per unit rho. The file values use per metre and include the
# ``W7X_MINOR_RADIUS`` factor.
W7X_MINOR_RADIUS = 0.4987183806149038

# These are the [geometry], [species], [profiles] and [boundary] blocks
# from the oracle input.
# The builder receives the minor radius directly, so equilibrium file paths are not necessary.
W7X_CONFIG = {
    "geometry": {"n_radial": 8, "rho_edge": 0.7},
    "species": {"names": ["e", "ion"]},
    "profiles": {
        "model": "standard_analytical",
        "n0": [0.35, 0.35],
        "n_edge": [0.29, 0.29],
        "T0": [6.7, 1.0],
        "T_edge": [0.8, 0.8],
        "density_shape_power": [2.0, 2.0],
        "density_shape_alpha": [1.0, 1.0],
        "temperature_shape_power": [2.0, 2.0],
        "temperature_shape_alpha": [2.0, 1.0],
        "er0_scale": 0.0,
        "er0_peak_rho": 0.8,
    },
    "boundary": {
        "density": {
            "left": {"type": "neumann", "gradient": {"default": 0.0}},
            "right": {"type": "dirichlet", "value": {"e": 0.29, "ion": 0.29}},
        },
        "temperature": {
            "left": {"type": "neumann", "gradient": {"default": 0.0}},
            "right": {"type": "dirichlet", "value": {"e": 0.8, "ion": 0.8}},
        },
    },
}

W7X_N_FACES = 9  # [geometry].n_radial + 1

W7X_DENSITY_FACE = [
    [0.35, 0.348828125, 0.346015625, 0.341328125, 0.33476562499999996, 0.32632812499999997,
     0.31601562499999997, 0.30382812499999995, 0.29],
    [0.35, 0.348828125, 0.346015625, 0.341328125, 0.33476562499999996, 0.32632812499999997,
     0.31601562499999997, 0.30382812499999995, 0.29],
]

W7X_TEMPERATURE_FACE = [
    [6.699189758300781, 6.473222351074219, 5.9481857299804695, 5.1307418823242195, 4.107316589355468,
     2.9989059448242177, 1.961076354980469, 1.1839645385742188, 0.8],
    [1.0000000000000002, 0.99609375, 0.9867187499999999, 0.97109375, 0.9492187500000001,
     0.9210937499999999, 0.8867187500000001, 0.84609375, 0.8],
]

W7X_DENSITY_GRAD_FACE = [
    [0.0, -0.02142857142857088, -0.04285714285714315, -0.06428571428571421, -0.0857142857142859,
     -0.10714285714285736, -0.12857142857142825, -0.15, -0.17142857142857162],
    [0.0, -0.02142857142857088, -0.04285714285714315, -0.06428571428571421, -0.0857142857142859,
     -0.10714285714285736, -0.12857142857142825, -0.15, -0.17142857142857162],
]

W7X_TEMPERATURE_GRAD_FACE = [
    [0.0, -4.13197544642856, -7.8688616071428745, -10.81556919642857, -12.577008928571443,
     -12.75809151785714, -10.963727678571422, -6.7988281249999964, -0.37039620535714424],
    [0.0, -0.07142857142857306, -0.14285714285714327, -0.21428571428571397, -0.2857142857142863,
     -0.3571428571428573, -0.42857142857142866, -0.4999999999999989, -0.5714285714285697],
]


# This case compares the analytical path with a real NEOPAX run. It tests the complete path and the
# port. Iteration 1 builds this face state. Iteration 2 reads the state from a solution.
# A mismatch would change the values that GKX and sfincs receive between iterations.
# Maximum relative differences are 0.0 for density and 1.3e-16 for temperature.
# They are 8.6e-15 for density gradient and 4.8e-14 for temperature gradient.
# NumPy and JAX operation order causes these differences. The tolerance is 1e-12.
# This value is about twenty times the largest numerical difference and below any unit error.
def test_w7x_analytical_snapshot_reproduces_neopax() -> None:
    snap = build_analytical_face_state(
        W7X_CONFIG, species=TWO_SPECIES, n_faces=W7X_N_FACES, minor_radius=W7X_MINOR_RADIUS
    )
    assert_allclose(snap.density, W7X_DENSITY_FACE, rtol=1e-12, atol=0.0)
    assert_allclose(snap.temperature, W7X_TEMPERATURE_FACE, rtol=1e-12, atol=0.0)
    assert_allclose(snap.density_grad, W7X_DENSITY_GRAD_FACE, rtol=1e-12, atol=0.0)
    assert_allclose(snap.temperature_grad, W7X_TEMPERATURE_GRAD_FACE, rtol=1e-12, atol=0.0)


# This independent check does not use NEOPAX values. Density and ion temperature are quadratic in
# rho. The three-point stencil is exact for these profiles. The gradients have these closed forms:
#     dn/drho = -2 * (n0 - n_edge) * rho / rho_edge**2
#     dT/drho = -2 * (T0 - T_edge) * rho / rho_edge**2
# Both comparisons agree to approximately 3e-14. This test detects a constant error in per-rho
# normalization. A refreshed golden file could contain the same error on both sides.
def test_w7x_quadratic_channels_match_their_closed_form_gradient() -> None:
    snap = build_analytical_face_state(
        W7X_CONFIG, species=TWO_SPECIES, n_faces=W7X_N_FACES, minor_radius=W7X_MINOR_RADIUS
    )
    rho_edge = float(W7X_CONFIG["geometry"]["rho_edge"])
    density_slope = -2.0 * (0.35 - 0.29) * snap.rho / rho_edge**2
    temperature_slope = -2.0 * (1.0 - 0.8) * snap.rho / rho_edge**2
    assert_allclose(snap.density_grad, [density_slope, density_slope], rtol=1e-12, atol=0.0)
    assert_allclose(snap.temperature_grad[1], temperature_slope, rtol=1e-12, atol=0.0)

    # ``temperature_shape_alpha = 2`` makes electron temperature quartic. A quadratic slope cannot
    # describe it. Such a slope differs by at least 15 percent away from the axis.
    # At the edge, it differs by a factor of 44. This check prevents comparison of one formula with
    # itself. It also detects an ignored ``temperature_shape_alpha`` value.
    quartic_channel = snap.temperature_grad[0, 1:]
    electron_slope = -2.0 * (6.7 - 0.8) * snap.rho[1:] / rho_edge**2
    assert np.min(np.abs(quartic_channel - electron_slope) / np.abs(quartic_channel)) > 0.1


# Three profile sources for one physical state

# Iteration 1 uses the analytical profile source. Later iterations use the prescribed source.
# The transport source reads the same state from the solution. Later code does not reconcile these
# sources. A mismatch would change values passed to GKX and sfincs between iterations.
# These tests build the W7-X golden state through all three paths.


# The real W7-X run had no radial electric field. It cannot distinguish direct face evaluation from
# center-based reconstruction. Therefore, this comparison adds a nonzero field.
W7X_ER_CONFIG = {**W7X_CONFIG, "profiles": {**W7X_CONFIG["profiles"], "er0_scale": 100.0}}


def _unconstrained_face_values(centers: np.ndarray, centers_x: np.ndarray, faces_x: np.ndarray) -> np.ndarray:
    """Calculate NEOPAX face values for a field with no [boundary] block.

    A zero axis gradient puts the first center value on the axis face. Interior faces use linear
    interpolation between adjacent centers. The edge face extrapolates the two outermost centers.
    ``W7X_ER_CONFIG`` has no [boundary.Er] block, so ``Er`` uses this closure.

    Parameters
    ----------
    centers : numpy.ndarray
        Field on the cell centers, shape ``(n_radial,)``.
    centers_x, faces_x : numpy.ndarray
        Cell-center and cell-face coordinates, shapes ``(n_radial,)`` and ``(n_radial + 1,)``.

    Returns
    -------
    numpy.ndarray
        Field on the faces, shape ``(n_radial + 1,)``.
    """
    weight = (faces_x[1:-1] - centers_x[:-1]) / (centers_x[1:] - centers_x[:-1])
    interior = centers[:-1] + weight * (centers[1:] - centers[:-1])
    return np.concatenate([[centers[0]], interior, [1.5 * centers[-1] - 0.5 * centers[-2]]])


def _w7x_cell_centered_state() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the W7-X face grid and analytical state on its cells."""
    rho_edge = float(W7X_ER_CONFIG["geometry"]["rho_edge"])
    faces = np.linspace(0.0, rho_edge, W7X_N_FACES, dtype=np.float64)
    centers = 0.5 * (faces[:-1] + faces[1:])
    density, temperature = _analytical_density_temperature(
        W7X_ER_CONFIG["profiles"], centers, rho_edge=rho_edge, charges=_charges(TWO_SPECIES)
    )
    return faces, density, temperature, _analytical_er(W7X_ER_CONFIG["profiles"], centers, rho_edge=rho_edge)


def _w7x_face_er() -> np.ndarray:
    """Return W7-X ``Er`` on faces after center-based NEOPAX reconstruction."""
    faces, _, _, er_centers = _w7x_cell_centered_state()
    return _unconstrained_face_values(er_centers, 0.5 * (faces[:-1] + faces[1:]), faces)


def _w7x_prescribed_cfg() -> dict:
    """The W7-X golden state as a prescribed [profiles] block.

    Each density array includes the 1e20 reference. Each temperature array includes the 1e3
    reference.
    The block uses SI, while the face state uses 1e20 m^-3 and keV.
    The function validates centered arrays but does not use them in the face state.
    Therefore, they contain the analytical formula on the centers bounded by the golden faces.

    Returns
    -------
    dict
        A config ``build_prescribed_face_state`` accepts.
    """
    rho_edge = float(W7X_ER_CONFIG["geometry"]["rho_edge"])
    _, density_centers, temperature_centers, er_centers = _w7x_cell_centered_state()
    return {
        "geometry": {"n_radial": W7X_N_FACES - 1, "rho_edge": rho_edge},
        "profiles": {
            "model": "prescribed",
            "density": (density_centers * 1.0e20).tolist(),
            "temperature": (temperature_centers * 1.0e3).tolist(),
            "Er": er_centers.tolist(),
            "density_face": (np.asarray(W7X_DENSITY_FACE) * 1.0e20).tolist(),
            "temperature_face": (np.asarray(W7X_TEMPERATURE_FACE) * 1.0e3).tolist(),
            "Er_face": _w7x_face_er().tolist(),
            "density_grad_face": (np.asarray(W7X_DENSITY_GRAD_FACE) * 1.0e20).tolist(),
            "temperature_grad_face": (np.asarray(W7X_TEMPERATURE_GRAD_FACE) * 1.0e3).tolist(),
        },
    }


def _write_w7x_transport_h5(path: Path) -> Path:
    """Write the W7-X golden state as a NEOPAX transport solution.

    A transport solution stores face values in face-state units. They need no conversion.
    Division by the minor radius converts both gradients to per metre on ``r_grid_half``.
    NEOPAX uses this representation.

    Parameters
    ----------
    path : Path
        Destination HDF5 file.

    Returns
    -------
    Path
        The written file path.
    """
    from tests.helpers.synthetic import write_transport_solution

    faces, density_centers, temperature_centers, er_centers = _w7x_cell_centered_state()
    return write_transport_solution(
        path,
        rho=0.5 * (faces[:-1] + faces[1:]),
        density=density_centers,
        temperature=temperature_centers,
        er=er_centers,
        rho_face=faces,
        r_grid_half=W7X_MINOR_RADIUS * faces,
        density_face=np.asarray(W7X_DENSITY_FACE),
        temperature_face=np.asarray(W7X_TEMPERATURE_FACE),
        er_face=_w7x_face_er(),
        density_grad_face=np.asarray(W7X_DENSITY_GRAD_FACE) / W7X_MINOR_RADIUS,
        temperature_grad_face=np.asarray(W7X_TEMPERATURE_GRAD_FACE) / W7X_MINOR_RADIUS,
        final_time=TRANSPORT_FINAL_TIME,
        next_dt=TRANSPORT_NEXT_DT,
    )


# The comparison uses ``er0_scale = 100.0``. The analytical path evaluates the parabola on centers
# and reconstructs its faces. NEOPAX uses this method. Direct face evaluation differs at every face.
# The edge difference is 45 percent.
def test_the_three_profile_sources_agree_on_the_w7x_state(tmp_path: Path) -> None:
    analytical = build_analytical_face_state(
        W7X_ER_CONFIG, species=TWO_SPECIES, n_faces=W7X_N_FACES, minor_radius=W7X_MINOR_RADIUS
    )
    prescribed = build_prescribed_face_state(_w7x_prescribed_cfg(), n_species=2)
    from_solution = read_transport_face_state(
        _write_w7x_transport_h5(tmp_path / "transport_solution.h5"), time_index=-1
    )
    assert np.any(np.abs(analytical.er) > 1.0)  # a zero field would make the Er comparison vacuous
    for field in ("rho", "density", "temperature", "er", "density_grad", "temperature_grad"):
        # These sources read the same values through different conversions. One uses SI reference
        # constants. The other converts from per metre. Both conversions round-trip exactly here.
        # The 1e-15 tolerance gives approximately four units in the last place.
        assert_allclose(getattr(prescribed, field), getattr(from_solution, field), rtol=1e-15, atol=0.0)
        # The analytical source runs the face operator again. NumPy operation order differs from
        # the JAX order used for the golden values. The result agrees within 5e-14.
        assert_allclose(getattr(analytical, field), getattr(from_solution, field), rtol=1e-12, atol=0.0)
