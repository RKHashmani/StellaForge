"""Tests for the Stage 3 sfincs_jax radial-scan helpers.

These reuse the pure functions from ``sfincs_jax_radial_scan.py``, loaded by path.
``_choose_radius_indices`` selects which flux surfaces to solve, and
``_prepare_input_text`` writes one surface's face state into a sfincs namelist. The
profile sources those helpers consume are shared with Stage 4 and live in
``common.neopax_profiles``, covered by ``tests/common/test_neopax_profiles.py``. The
solver is imported lazily inside the worker, so loading the module and calling these
helpers needs no solver.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from tests.helpers.stage_import import load_stage_module

scan = load_stage_module("stages/stage3-neoclassical/sfincs_jax_radial_scan.py")


# Radius selection

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


# Prepare dispatch

REPO_ROOT = Path(__file__).resolve().parents[2]

# Complete prescribed input for prepare tests. The [species] block supplies charge and mass. The
# [profiles] arrays use SI values on five cells. Face arrays are linear in rho, with constant
# gradients per unit rho.
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
density_grad_face = [[5.0e18, 5.0e18, 5.0e18, 5.0e18, 5.0e18, 5.0e18], [2.5e18, 2.5e18, 2.5e18, 2.5e18, 2.5e18, 2.5e18], [2.5e18, 2.5e18, 2.5e18, 2.5e18, 2.5e18, 2.5e18]]
temperature_grad_face = [[-500.0, -500.0, -500.0, -500.0, -500.0, -500.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0], [-500.0, -500.0, -500.0, -500.0, -500.0, -500.0]]
"""


# Shared tests call ``build_prescribed_face_state`` directly. This test covers dispatch from the
# prescribed profile-source option. It uses no transport solution. The real prepare step records the
# source and scans the face grid without the magnetic axis. Here, [geometry].n_radial gives five
# cells and six faces. Removing the axis leaves five scan surfaces.
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


# sfincs input text

def _namelist_values(text: str, key: str) -> np.ndarray:
    """Read one namelist entry back out of the emitted input text as a float array."""
    match = re.search(rf"(?m)^[ \t]*{re.escape(key)}[ \t]*=[ \t]*([^!\n\r]+)$", text)
    assert match is not None, f"{key} is absent from the emitted namelist"
    return np.asarray([float(token) for token in match.group(1).split()], dtype=float)


# sfincs reads ``dNHatdrNs`` and ``dTHatdrNs`` against rho. The snapshot stores gradients per unit
# rho. The writer must not scale, negate or recalculate them. The test uses distinct constant
# gradients for both channels. It detects incorrect units, signs and channel selection.
def test_prepare_input_text_writes_the_snapshot_gradients_unchanged() -> None:
    cfg = tomllib.loads(PRESCRIBED_TOML)
    snapshot = scan.build_prescribed_face_state(cfg, n_species=3)
    species = scan._parse_species_from_config(cfg)
    template = (REPO_ROOT / "inputs/quick_run/sfincs_input.HSX_vacuum_ns201_quickrun").read_text()
    radius_index = 3
    text = scan._prepare_input_text(
        template_text=template,
        species=species,
        snapshot=snapshot,
        radius_index=radius_index,
        include_phi1=None,
        resolution_overrides={},
        solver_tolerance=None,
    )
    assert_allclose(_namelist_values(text, "dNHatdrNs"), snapshot.density_grad[:, radius_index], rtol=1e-12)
    assert_allclose(_namelist_values(text, "dTHatdrNs"), snapshot.temperature_grad[:, radius_index], rtol=1e-12)
    assert_allclose(_namelist_values(text, "dNHatdrNs"), [0.05, 0.025, 0.025], rtol=1e-12)
    assert_allclose(_namelist_values(text, "dTHatdrNs"), [-0.5, -0.5, -0.5], rtol=1e-12)
