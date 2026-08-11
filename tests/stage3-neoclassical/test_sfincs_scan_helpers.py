"""Tests for the Stage 3 sfincs_jax radial-scan helpers.

These reuse the pure functions from ``sfincs_jax_radial_scan.py``, loaded by path.
``_choose_radius_indices`` selects which flux surfaces to solve; the analytical
snapshot builder synthesises species-resolved profiles from config. The solver is
imported lazily inside the worker, so loading the module and calling these helpers
needs no solver.
"""

from __future__ import annotations

from pathlib import Path
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
gkx_scan = load_stage_module("stages/stage4-turbulence/gkx_radial_scan.py")

SNAPSHOT_MODULES = [
    pytest.param(scan, id="sfincs_jax"),
    pytest.param(gkx_scan, id="gkx"),
]


# `_build_standard_analytical_snapshot` synthesises test profiles (density, temperature, etc.) from config, used when no
# real upstream profiles are available. This builds a 3-species, 5-radius snapshot and asserts every array has the
# expected shape and that a static (non-time-resolved) snapshot has no time value. Parametrized to run against both
# Stage 3's and Stage 4's copy of the function.
@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
def test_snapshot_shapes(module: ModuleType) -> None:
    snap = module._build_standard_analytical_snapshot({}, n_species=3, n_points=5)
    assert snap.rho.shape == (5,)
    assert snap.density.shape == (3, 5)
    assert snap.temperature.shape == (3, 5)
    assert snap.er.shape == (5,)
    assert snap.time_value is None


@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
def test_snapshot_core_and_edge_values(module: ModuleType) -> None:
    snap = module._build_standard_analytical_snapshot({}, n_species=3, n_points=5)
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
    snap = module._build_standard_analytical_snapshot(cfg, n_species=3, n_points=5)
    assert_allclose(snap.density[1], 0.6 * snap.density[0], rtol=1e-12)
    assert_allclose(snap.density[2], 0.4 * snap.density[0], rtol=1e-12)


def _w7x_like_profiles() -> dict:
    """A 2-species [profiles] block in the per-species-array form used by inputs/w7-x_t3d_like."""
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


# Every scalar [profiles] knob is equally accepted as a per-species list. T0 = [6.7, 1.0] with a shared
# T_edge = 0.8 gives each species its own core temperature meeting at a common edge value, so this asserts the
# electron and ion columns start at 6.7 and 1.0 respectively and both end at 0.8.
@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
def test_snapshot_accepts_per_species_arrays(module: ModuleType) -> None:
    snap = module._build_standard_analytical_snapshot(_w7x_like_profiles(), n_species=2, n_points=5)
    assert_allclose(snap.temperature[0, 0], 6.7, rtol=1e-12)  # electron core T0
    assert_allclose(snap.temperature[1, 0], 1.0, rtol=1e-12)  # ion core T0
    assert_allclose(snap.temperature[:, -1], 0.8, rtol=1e-12)  # shared T_edge
    assert_allclose(snap.density[:, 0], 0.35, rtol=1e-12)
    assert_allclose(snap.density[:, -1], 0.29, rtol=1e-12)


# temperature_shape_alpha is the outer exponent in edge + (core - edge) * (1 - x**power)**alpha. With
# alpha = [2.0, 1.0] and equal powers, this asserts each species matches its own analytical curve and that the two
# normalised shapes come out different.
@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
def test_snapshot_applies_shape_alpha(module: ModuleType) -> None:
    snap = module._build_standard_analytical_snapshot(_w7x_like_profiles(), n_species=2, n_points=5)
    x = np.linspace(0.0, 1.0, 5)
    for i, (t0, alpha) in enumerate([(6.7, 2.0), (1.0, 1.0)]):
        expected = 0.8 + (t0 - 0.8) * (1.0 - x**2.0) ** alpha
        assert_allclose(snap.temperature[i], expected, rtol=1e-12)
    # Normalised shapes differ between the two species.
    shape_e = (snap.temperature[0] - 0.8) / (6.7 - 0.8)
    shape_i = (snap.temperature[1] - 0.8) / (1.0 - 0.8)
    assert not np.allclose(shape_e, shape_i)


# [geometry].rho_edge sets the extent of the radial grid. With rho_edge = 0.7 this asserts the snapshot's rho is
# linspace(0, 0.7, n) rather than always reaching 1.0; the shape functions are evaluated on x = rho / rho_edge.
@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
def test_snapshot_grid_honours_rho_edge(module: ModuleType) -> None:
    snap = module._build_standard_analytical_snapshot(_w7x_like_profiles(), n_species=2, n_points=5)
    assert_allclose(snap.rho, np.linspace(0.0, 0.7, 5), rtol=1e-12)


@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
def test_snapshot_grid_defaults_rho_edge(module: ModuleType) -> None:
    cfg = _w7x_like_profiles()
    del cfg["geometry"]["rho_edge"]
    snap = module._build_standard_analytical_snapshot(cfg, n_species=2, n_points=5)
    assert_allclose(snap.rho, np.linspace(0.0, 1.0, 5), rtol=1e-12)


# The radial electric field is the parabola er0_scale * x * (er0_peak_rho - x) in normalised x. It crosses zero at
# x = er0_peak_rho, so this also asserts the outermost point (x = 1 > 0.8) comes out negative. The shared fixture pins
# er0_scale = 0.0, so a non-zero scale is set here.
@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
def test_snapshot_er_is_neopax_parabola(module: ModuleType) -> None:
    cfg = _w7x_like_profiles()
    cfg["profiles"]["er0_scale"] = 100.0
    snap = module._build_standard_analytical_snapshot(cfg, n_species=2, n_points=5)
    x = np.linspace(0.0, 1.0, 5)
    assert_allclose(snap.er, 100.0 * x * (0.8 - x), rtol=1e-12)
    assert snap.er[-1] < 0.0  # x = 1 > er0_peak_rho = 0.8


# The T3D reference case runs with no radial electric field, so the fixture's er0_scale = 0.0 must flatten the parabola
# to exactly zero at every radius. atol = 0.0 pins exact zeros rather than merely small values.
@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
def test_snapshot_er_is_zero_when_scale_is_zero(module: ModuleType) -> None:
    snap = module._build_standard_analytical_snapshot(_w7x_like_profiles(), n_species=2, n_points=5)
    assert_allclose(snap.er, np.zeros(5), atol=0.0)


# --- _build_prescribed_snapshot ---

# Both stage scripts also carry a _build_prescribed_snapshot twin that reads the closed loop's fed-back [profiles]
# arrays, so it gets the same dual-module drift guard. The prescribed block stores SI values (density in m^-3,
# temperature in eV, Er in kV/m) while the snapshot uses 1e20 m^-3 and keV, and the block carries no radial
# coordinate, so the builder must divide the units back and reconstruct the transport grid from [geometry].
#
# The block carries both of NEOPAX's radial grids. The plain arrays hold n_radial cell-center values, which is what
# NEOPAX itself reads back; the `*_face` arrays hold n_radial + 1 values on the faces, which is where these scans
# sample fluxes because that is the grid NEOPAX interpolates a flux file onto. The snapshot is built from the face
# arrays, so the fixtures give the two grids disjoint magnitudes and a reader taking the wrong one cannot pass.
def _prescribed_cfg(n_species: int = 3, n_radial: int = 5) -> dict:
    density = [[(s + 1) * (r + 1) * 1.0e19 for r in range(n_radial)] for s in range(n_species)]
    temperature = [[(s + 1) * (r + 1) * 1.0e3 for r in range(n_radial)] for s in range(n_species)]
    density_face = [[(s + 1) * (r + 1) * 2.0e19 for r in range(n_radial + 1)] for s in range(n_species)]
    temperature_face = [[(s + 1) * (r + 1) * 2.0e3 for r in range(n_radial + 1)] for s in range(n_species)]
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
        },
    }


# The snapshot must come off the face grid: n_radial + 1 points spanning [0, rho_edge], carrying the `*_face` values
# converted out of SI. Taking the centered arrays instead would place n_radial values at the wrong radii, and nothing
# downstream would report it.
@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
def test_prescribed_snapshot_units_and_grid(module: ModuleType) -> None:
    snap = module._build_prescribed_snapshot(_prescribed_cfg(), n_species=3)
    expected = np.array([[(s + 1) * (r + 1) for r in range(6)] for s in range(3)], dtype=float)
    assert_allclose(snap.density, expected * 0.2, rtol=1e-12)  # 2e19 m^-3 becomes 0.2 in 1e20 m^-3 units
    assert_allclose(snap.temperature, expected * 2.0, rtol=1e-12)  # 2e3 eV becomes 2 keV
    assert_allclose(snap.er, 10.0 + np.arange(6.0), rtol=1e-12)  # kV/m passes through unchanged
    assert_allclose(snap.rho, np.linspace(0.0, 0.7, 6), rtol=1e-12)
    assert snap.time_value is None


# A template without [geometry].rho_edge reconstructs the default full-radius face grid.
@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
def test_prescribed_snapshot_defaults_rho_edge(module: ModuleType) -> None:
    cfg = _prescribed_cfg()
    del cfg["geometry"]["rho_edge"]
    snap = module._build_prescribed_snapshot(cfg, n_species=3)
    assert_allclose(snap.rho, np.linspace(0.0, 1.0, 6), rtol=1e-12)


# A hand-written block, or one from a NEOPAX predating the staggered grid, carries only the centered arrays. There is
# no safe way to invent the face values here: NEOPAX builds them under the run's own boundary models, so an
# extrapolation guessed at this end would disagree with the solver wherever the outer boundary is not a plain
# Dirichlet. The builder therefore refuses and names both the missing key and where it comes from.
@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
@pytest.mark.parametrize("key", ["density_face", "temperature_face", "Er_face"])
def test_prescribed_snapshot_requires_the_face_arrays(module: ModuleType, key: str) -> None:
    cfg = _prescribed_cfg()
    del cfg["profiles"][key]
    with pytest.raises(ValueError, match=rf"\[profiles\]\.{key}"):
        module._build_prescribed_snapshot(cfg, n_species=3)


# The face arrays carry exactly one more radius than the centered ones. A face array sized like the centers would
# leave the outermost face unsampled and shift every other value inward by half a cell.
@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
def test_prescribed_snapshot_face_width_validation(module: ModuleType) -> None:
    cfg = _prescribed_cfg()
    cfg["profiles"]["density_face"] = [row[:-1] for row in cfg["profiles"]["density_face"]]
    with pytest.raises(ValueError, match=r"\[profiles\]\.density_face"):
        module._build_prescribed_snapshot(cfg, n_species=3)
    cfg = _prescribed_cfg()
    cfg["profiles"]["Er_face"] = cfg["profiles"]["Er_face"][:-1]
    with pytest.raises(ValueError, match=r"\[profiles\]\.Er_face"):
        module._build_prescribed_snapshot(cfg, n_species=3)


# Only "prescribed" activates the builder. NEOPAX's "given" synonym is deliberately rejected too:
# the loop's feedback writer always emits "prescribed", and accepting a second spelling here would
# let Stage 3/4 configs drift from what the rest of the pipeline writes.
@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
@pytest.mark.parametrize("model", ["standard_analytical", "given"])
def test_prescribed_snapshot_rejects_other_models(module: ModuleType, model: str) -> None:
    cfg = _prescribed_cfg()
    cfg["profiles"]["model"] = model
    with pytest.raises(ValueError, match="requires 'prescribed'"):
        module._build_prescribed_snapshot(cfg, n_species=3)


@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
@pytest.mark.parametrize("key", ["density", "temperature", "Er"])
def test_prescribed_snapshot_requires_each_array(module: ModuleType, key: str) -> None:
    cfg = _prescribed_cfg()
    del cfg["profiles"][key]
    with pytest.raises(ValueError, match=rf"\[profiles\]\.{key}"):
        module._build_prescribed_snapshot(cfg, n_species=3)


@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
def test_prescribed_snapshot_shape_validation(module: ModuleType) -> None:
    with pytest.raises(ValueError, match="species rows"):
        module._build_prescribed_snapshot(_prescribed_cfg(), n_species=2)
    cfg = _prescribed_cfg()
    cfg["profiles"]["Er"] = [0.0, 1.0]
    with pytest.raises(ValueError, match=r"\[profiles\]\.Er"):
        module._build_prescribed_snapshot(cfg, n_species=3)
    cfg = _prescribed_cfg()
    cfg["geometry"]["n_radial"] = 7
    with pytest.raises(ValueError, match=r"\[geometry\]\.n_radial"):
        module._build_prescribed_snapshot(cfg, n_species=3)
    with pytest.raises(ValueError, match="at least 3 faces"):
        module._build_prescribed_snapshot(_prescribed_cfg(n_radial=1), n_species=3)


# The floor is on faces, not cells, because that is where the gradients are taken. Two cells give three faces, which
# is enough for np.gradient's edge_order=2, and also clears NEOPAX's own two-cell minimum for extrapolating its outer
# boundary value. One cell gives two faces and is rejected above.
@pytest.mark.parametrize("module", SNAPSHOT_MODULES)
def test_prescribed_snapshot_accepts_two_cells(module: ModuleType) -> None:
    snap = module._build_prescribed_snapshot(_prescribed_cfg(n_radial=2), n_species=3)
    assert_allclose(snap.rho, np.linspace(0.0, 0.7, 3), rtol=1e-12)


# --- prepare dispatch ---

REPO_ROOT = Path(__file__).resolve().parents[2]

# A complete prescribed common-input template for prepare-level tests. The species block carries the
# charge/mass arrays the species parser requires; the profile arrays are SI on the 5-point grid.
PRESCRIBED_TOML = """\
[species]
names = ["e", "D", "T"]
charge_qp = [-1.0, 1.0, 1.0]
mass_mp = [0.000544617, 2.0, 3.0]

[geometry]
n_radial = 5

[profiles]
model = "prescribed"
density = [[1.0e19, 1.1e19, 1.2e19, 1.3e19, 1.4e19], [5.0e18, 5.5e18, 6.0e18, 6.5e18, 7.0e18], [5.0e18, 5.5e18, 6.0e18, 6.5e18, 7.0e18]]
temperature = [[1000.0, 900.0, 800.0, 700.0, 600.0], [950.0, 850.0, 750.0, 650.0, 550.0], [940.0, 840.0, 740.0, 640.0, 540.0]]
Er = [0.0, 1.0, 2.0, 3.0, 4.0]
density_face = [[0.95e19, 1.05e19, 1.15e19, 1.25e19, 1.35e19, 1.45e19], [4.75e18, 5.25e18, 5.75e18, 6.25e18, 6.75e18, 7.25e18], [4.75e18, 5.25e18, 5.75e18, 6.25e18, 6.75e18, 7.25e18]]
temperature_face = [[1050.0, 950.0, 850.0, 750.0, 650.0, 550.0], [1000.0, 900.0, 800.0, 700.0, 600.0, 500.0], [990.0, 890.0, 790.0, 690.0, 590.0, 490.0]]
Er_face = [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
"""


# The snapshot tests above call _build_prescribed_snapshot directly, so the dispatch branch in _prepare that routes
# --profiles-source=prescribed to it (with no transport solution) would go untested without this. Running the real
# _prepare against a prescribed template and the tracked sfincs template must succeed without any transport file,
# record the source in the manifest, and scan exactly the transport face grid minus the magnetic axis.
# [geometry].n_radial = 5 cells means 6 faces at linspace(0, 1, 6), of which the axis face is dropped, so 5 surfaces
# are scanned.
def test_prepare_dispatches_prescribed_source(tmp_path: Path) -> None:
    config = tmp_path / "common_input.toml"
    config.write_text(PRESCRIBED_TOML)
    args = scan.build_parser().parse_args([
        "prepare",
        "--common-config", str(config),
        "--sfincs-template", str(REPO_ROOT / "inputs/quick_run/sfincs_input.HSX_vacuum_ns201_quickrun"),
        "--output-dir", str(tmp_path / "out"),
        "--profiles-source", "prescribed",
    ])
    manifest, pending = scan._prepare(args)
    assert manifest["profiles_source"] == "prescribed"
    assert manifest["source_transport_solution"] is None
    assert [run["rho"] for run in manifest["runs"]] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1.0])
    assert len(pending) == 5


# --- transport_h5: reading a NEOPAX transport solution directly ---

# The third profile source reads transport_solution.h5 itself rather than the [profiles] block the loop writes. It
# needs the same face grid as the prescribed path and for the same reason: NEOPAX's evolved state sits on the cell
# centers, which span neither rho = 0 nor rho = rho_edge, so a snapshot built from them gives a flux file that stops
# short at both ends and NEOPAX rejects it (or, where a relabel step runs first, that step does).
TRANSPORT_READERS = [
    pytest.param(scan, "_load_transport_snapshot", id="sfincs_jax"),
    pytest.param(gkx_scan, "_infer_transport_snapshot", id="gkx"),
]


def _write_transport_h5(
    path: Path, *, n_radial: int = 5, n_species: int = 3, faces: bool = True, time_resolved: bool = True
) -> Path:
    """Write a transport solution laid out the way the pinned NEOPAX writes one.

    NEOPAX emits either a time series shaped ``(n_time, n_species, n_rho)`` or a single static slice
    shaped ``(n_species, n_rho)``, and both readers have to tell those ranks apart: a static file's
    leading axis is species, so indexing it as time silently returns one species row. ``offset``
    separates the two time slices so the last one is distinguishable.
    """
    from tests.helpers.synthetic import write_transport_solution

    offset = 100.0 if time_resolved else 0.0
    ramp_c = np.arange(n_species * n_radial, dtype=float).reshape(n_species, n_radial)
    ramp_f = np.arange(n_species * (n_radial + 1), dtype=float).reshape(n_species, n_radial + 1)
    face_grid = np.linspace(0.0, 1.0, n_radial + 1)

    def lay(a: np.ndarray) -> np.ndarray:
        """Return ``a`` as a static slice, or as two time slices whose last one is ``a + offset``."""
        return np.stack([a, a + offset]) if time_resolved else a

    extra = {}
    if faces:
        extra = dict(
            rho_face=face_grid,
            density_face=lay(5.0 + 0.1 * ramp_f),
            temperature_face=lay(7.0 + 0.2 * ramp_f),
            er_face=lay(np.linspace(-2.0, 4.0, n_radial + 1)),
        )
    return write_transport_solution(
        path,
        rho=0.5 * (face_grid[:-1] + face_grid[1:]),
        density=lay(1.0 + 0.1 * ramp_c),
        temperature=lay(2.0 + 0.2 * ramp_c),
        er=lay(np.linspace(-1.0, 3.0, n_radial)),
        **extra,
    )


# The snapshot must land on the face grid with the face values, exactly as the prescribed path does. The face fixtures
# use disjoint magnitudes from the centered ones, so a reader still taking f["density"] cannot pass.
# `time_resolved` also puts the rank split on trial. A static file's leading axis is species, not time, so a reader
# indexing it as time returns one species row and a scalar Er while every shape check downstream still looks plausible.
@pytest.mark.parametrize(("module", "reader"), TRANSPORT_READERS)
@pytest.mark.parametrize("time_resolved", [True, False], ids=["time_resolved", "static"])
def test_transport_snapshot_takes_the_face_grid(
    module: ModuleType, reader: str, time_resolved: bool, tmp_path: Path
) -> None:
    path = _write_transport_h5(tmp_path / "transport_solution.h5", time_resolved=time_resolved)
    snap = getattr(module, reader)(path, time_index=-1)
    ramp_f = np.arange(3 * 6, dtype=float).reshape(3, 6)
    offset = 100.0 if time_resolved else 0.0  # the last of two slices, or the only one
    assert_allclose(snap.rho, np.linspace(0.0, 1.0, 6), rtol=1e-12)
    assert_allclose(snap.density, offset + 5.0 + 0.1 * ramp_f, rtol=1e-12)
    assert_allclose(snap.temperature, offset + 7.0 + 0.2 * ramp_f, rtol=1e-12)
    assert_allclose(snap.er, offset + np.linspace(-2.0, 4.0, 6), rtol=1e-12)


# A solution from a NEOPAX predating the staggered grid carries no face datasets. Falling back to the centers would
# hand NEOPAX a grid covering neither end of [0, rho_edge], so the reader refuses and names what is missing.
@pytest.mark.parametrize(("module", "reader"), TRANSPORT_READERS)
def test_transport_snapshot_requires_the_face_datasets(module: ModuleType, reader: str, tmp_path: Path) -> None:
    path = _write_transport_h5(tmp_path / "old_pin.h5", faces=False)
    with pytest.raises(KeyError, match="density_faces"):
        getattr(module, reader)(path, time_index=-1)


# A static solution holds one slice, so any --time-index other than the default -1 names a slice that does not exist.
# Silently reading slice 0 anyway records a time_index in the manifest that the run did not honour, and the mismatch
# only surfaces later when Stage 5's feedback writer rejects the same index.
@pytest.mark.parametrize(("module", "reader"), TRANSPORT_READERS)
@pytest.mark.parametrize("bad", [0, 1, -2])
def test_static_transport_snapshot_rejects_a_non_default_time_index(
    module: ModuleType, reader: str, bad: int, tmp_path: Path
) -> None:
    path = _write_transport_h5(tmp_path / "static.h5", time_resolved=False)
    with pytest.raises(ValueError, match="single static profile slice"):
        getattr(module, reader)(path, time_index=bad)
