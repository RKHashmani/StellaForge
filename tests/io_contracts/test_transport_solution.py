"""Tests for the ``transport_solution.h5`` contract in ``src/io_contracts.py``.

The contract covers the fields read by the Stage 3/4/5 transport-snapshot loaders. NEOPAX
solves on a staggered finite-volume grid, so the file carries both halves and both are
required: the cell-centered ``rho``/``density``/``temperature``/``Er`` and the face-grid
``rho_face``/``density_faces``/``temperature_faces``/``Er_faces``, with ``Er`` and
``Er_faces`` carrying no species axis. Optional: ``pressure``/``pressure_faces``/``ts``.
``temperature`` is required and ``pressure`` optional because the Stage 3 loader accepts
either and derives the missing one, while the Stage 4 loader hard-requires ``temperature``
with no pressure fallback; the face datasets follow the same rule.

Beyond presence, the invariants are about axes rather than ranks: the two grids differ only
in their trailing radial axis, ``rho_face`` holds one more point than ``rho`` and increases
strictly outward with ``rho`` at its midpoints, and ``ts`` holds one timestamp per slice.
Malformed cases run on the inner checker; the valid, missing-Er and missing-face files
exercise the h5 read plus the raise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.io_contracts import ContractError, _check_transport_solution, validate_transport_solution
from tests.helpers.synthetic import write_transport_solution


def _staggered_grid(n_rho: int) -> tuple[np.ndarray, np.ndarray]:
    """NEOPAX's staggered grid: ``(centers, faces)`` for n_rho cells, centers midway between faces."""
    rho_face = np.linspace(0.0, 1.0, n_rho + 1)
    return 0.5 * (rho_face[:-1] + rho_face[1:]), rho_face


def _valid_data(n_species: int = 3, n_rho: int = 5) -> dict[str, np.ndarray | None]:
    """A minimal valid solution on both of NEOPAX's grids: n_rho cell centers and n_rho + 1 faces."""
    profile = np.ones((n_species, n_rho))
    face_profile = np.ones((n_species, n_rho + 1))
    rho, rho_face = _staggered_grid(n_rho)
    return {
        "rho": rho,
        "density": profile.copy(),
        "temperature": profile.copy(),
        "pressure": None,
        "Er": np.ones(n_rho),
        "ts": None,
        "rho_face": rho_face,
        "density_faces": face_profile.copy(),
        "temperature_faces": face_profile.copy(),
        "pressure_faces": None,
        "Er_faces": np.ones(n_rho + 1),
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


# `Er` is a per-radius field with no species axis, so it must have one fewer dimension than the species-resolved
# `density`. This wrongly gives `Er` a (species, radius) 2D shape and asserts the checker flags that `Er`'s rank should
# be "one less" than density's.
def test_er_with_species_axis_flagged() -> None:
    data = _valid_data(n_species=3, n_rho=5)
    data["Er"] = np.ones((3, 5))  # wrongly given a species axis
    assert any("Er" in problem and "one less" in problem for problem in _check_transport_solution(data))


# Every profile's last axis must span the radial grid `rho`. This gives `density` a trailing axis of 4 while `rho` has
# length 5, and asserts the checker flags the "trailing axis" mismatch.
def test_rho_axis_mismatch_flagged() -> None:
    data = _valid_data(n_rho=5)
    data["density"] = np.ones((3, 4))  # trailing axis 4 != len(rho) 5
    assert any("density" in problem and "trailing axis" in problem for problem in _check_transport_solution(data))


def test_nonfinite_density_flagged() -> None:
    data = _valid_data()
    data["density"][0, 0] = np.nan
    assert "'density' contains non-finite values" in _check_transport_solution(data)


# `rho` is the 1D radial coordinate. This gives it a 2D shape and asserts the checker flags that `rho` "must be 1D".
def test_rho_not_1d_flagged() -> None:
    data = _valid_data()
    data["rho"] = np.ones((2, 3))  # rho must be a 1D radial coordinate
    assert any("rho" in problem and "must be 1D" in problem for problem in _check_transport_solution(data))


# `pressure` is optional, but when present it must share every one of `density`'s non-radial axes. This gives a 3D
# (time-resolved) pressure while density is 2D (static), the coarsest form of that disagreement.
def test_pressure_rank_mismatch_flagged() -> None:
    data = _valid_data(n_species=3, n_rho=5)
    data["pressure"] = np.ones((4, 3, 5))  # 3D pressure while density is 2D
    assert any(
        "pressure" in problem and "density" in problem and "leading axes" in problem
        for problem in _check_transport_solution(data)
    )


def test_er_trailing_axis_mismatch_flagged() -> None:
    data = _valid_data(n_rho=5)
    data["Er"] = np.ones(4)  # trailing axis 4 != len(rho) 5
    assert any("Er" in problem and "trailing axis" in problem for problem in _check_transport_solution(data))


# Sets up a time-resolved solution (density/temperature are 3D and `Er` 2D over 4 time steps) but makes the time axis
# `ts` length 3 instead of 4. Asserts the checker flags that `ts`'s length disagrees with the profiles' time axis, so
# the timestamps and the data stay in sync.
def test_ts_length_mismatch_flagged() -> None:
    n_time, n_species, n_rho = 4, 3, 5
    data = _valid_data(n_species=n_species, n_rho=n_rho)
    data["density"] = np.ones((n_time, n_species, n_rho))
    data["temperature"] = np.ones((n_time, n_species, n_rho))
    data["Er"] = np.ones((n_time, n_rho))
    data["ts"] = np.arange(n_time - 1, dtype=float)  # length 3 != time axis 4
    assert any("ts" in problem and "time axis" in problem for problem in _check_transport_solution(data))


def test_valid_file_passes(tmp_path: Path) -> None:
    rho, rho_face = _staggered_grid(5)
    profile = np.ones((3, 5))
    face_profile = np.ones((3, 6))
    written = write_transport_solution(
        tmp_path / "t.h5", rho=rho, density=profile, temperature=profile, er=np.ones(5),
        rho_face=rho_face, density_face=face_profile,
        temperature_face=face_profile, er_face=np.ones(6),
    )
    assert validate_transport_solution(written) is None


# End-to-end failure path: writes a real file omitting the `er` argument (so `Er` is absent) and asserts
# `validate_transport_solution` raises a ContractError mentioning `Er`.
def test_file_missing_er_raises(tmp_path: Path) -> None:
    rho, _ = _staggered_grid(5)
    profile = np.ones((3, 5))
    written = write_transport_solution(tmp_path / "t.h5", rho=rho, density=profile, temperature=profile)
    with pytest.raises(ContractError, match="Er"):
        validate_transport_solution(written)


# --- the face grid ---

# NEOPAX solves on a staggered grid, and the consumers of this file split over which half they read. Stages 3 and 4
# sample fluxes on the faces and the loop's feedback writer copies the face state into the prescribed block; the
# pressure fit and convergence signal read only the centers, so a plain forward pass completes without faces. The
# contract requires them regardless, holding the artifact to what a closed-loop iteration needs.
@pytest.mark.parametrize("field", ["rho_face", "density_faces", "temperature_faces", "Er_faces"])
def test_missing_face_field_flagged(field: str) -> None:
    data = _valid_data()
    data[field] = None
    assert f"missing required field '{field}'" in _check_transport_solution(data)


# The faces bound the cells, so there is exactly one more face than center. Any other length means the two grids
# describe different geometry, and a reader pairing them would silently misalign radius with value.
def test_face_grid_must_have_one_more_point_than_the_centers() -> None:
    data = _valid_data(n_rho=5)
    data["rho_face"] = np.linspace(0.0, 1.0, 5)  # as many faces as cells
    problems = _check_transport_solution(data)
    # The elementwise geometry checks would raise on these mismatched shapes, so they stay gated behind this
    # length check.
    assert any("rho_face" in problem and "one more" in problem for problem in problems)


# The centers are not an independent grid but the midpoints of the faces, which is how NEOPAX builds them. A
# file whose `rho` is the face linspace itself, or is written outside-in, keeps the right point count and
# passes every axis check, so counting alone cannot catch it.
@pytest.mark.parametrize(
    "rho",
    [np.linspace(0.0, 1.0, 5), np.array([0.9, 0.7, 0.5, 0.3, 0.1])],
    ids=["node_grid", "descending"],
)
def test_centres_that_are_not_the_face_midpoints_flagged(rho: np.ndarray) -> None:
    data = _valid_data(n_rho=5)
    data["rho"] = rho
    assert any("midpoints" in problem for problem in _check_transport_solution(data))


# The faces march outward from the axis. A shuffled face grid still has the right length, and its midpoints can
# even stay inside [0, 1], so nothing but monotonicity distinguishes it from a grid whose cells have positive width.
def test_non_monotonic_face_grid_flagged() -> None:
    data = _valid_data(n_rho=5)
    data["rho_face"] = np.array([0.0, 0.6, 0.2, 0.8, 0.4, 1.0])
    assert any("rho_face" in problem and "strictly" in problem for problem in _check_transport_solution(data))


# Each face profile spans the face grid, not the cell grid. Sizing one like the centers would leave the outermost
# face unsampled while every other shape check still passed.
@pytest.mark.parametrize("field", ["density_faces", "temperature_faces"])
def test_face_profile_on_the_centre_grid_flagged(field: str) -> None:
    data = _valid_data(n_species=3, n_rho=5)
    data[field] = np.ones((3, 5))  # trailing axis is n_rho, not n_rho + 1
    assert any(field in problem and "trailing axis" in problem for problem in _check_transport_solution(data))


# Er_faces carries no species axis, exactly as Er does on the centers, so its rank must stay one below the face
# profiles'.
def test_er_faces_with_species_axis_flagged() -> None:
    data = _valid_data(n_species=3, n_rho=5)
    data["Er_faces"] = np.ones((3, 6))  # wrongly given a species axis
    assert any("Er_faces" in problem and "one less" in problem for problem in _check_transport_solution(data))


# The centered and face states advance together, so a file mixing a static face state with a time-resolved centered
# one (or the reverse) describes no single instant and must be rejected.
def test_face_and_centre_rank_mismatch_flagged() -> None:
    data = _valid_data(n_species=3, n_rho=5)
    data["density"] = np.ones((2, 3, 5))
    data["temperature"] = np.ones((2, 3, 5))
    data["Er"] = np.ones((2, 5))
    assert any("density_faces" in problem and "leading axes" in problem for problem in _check_transport_solution(data))


# End-to-end: a file written without the face datasets, as a NEOPAX predating the staggered grid would write it, is
# rejected by the file-level validator. Such a file could still drive a forward pass, whose consumers read only the
# centers, but not a loop iteration -- and the contract holds the artifact to the latter.
def test_file_without_the_face_datasets_raises(tmp_path: Path) -> None:
    profile = np.ones((3, 5))
    rho, _ = _staggered_grid(5)
    written = write_transport_solution(
        tmp_path / "old_pin.h5", rho=rho,
        density=profile, temperature=profile, er=np.ones(5),
    )
    with pytest.raises(ContractError, match="density_faces"):
        validate_transport_solution(written)


# --- leading axes, not just rank ---

# Two arrays can share a rank and still describe different numbers of times or species. Only the trailing axis is
# a radial coordinate, so everything before it, time and species, must match exactly between the centered and face
# states, which are the same instant seen on two grids.
def test_face_state_with_a_different_time_axis_flagged() -> None:
    data = _valid_data(n_species=3, n_rho=5)
    data["density"] = np.ones((2, 3, 5))
    data["temperature"] = np.ones((2, 3, 5))
    data["Er"] = np.ones((2, 5))
    data["density_faces"] = np.ones((3, 3, 6))       # 3 times against the centers' 2
    data["temperature_faces"] = np.ones((3, 3, 6))
    data["Er_faces"] = np.ones((3, 6))
    assert any("density_faces" in problem and "leading axes" in problem for problem in _check_transport_solution(data))


def test_face_state_with_a_different_species_count_flagged() -> None:
    data = _valid_data(n_species=3, n_rho=5)
    data["density_faces"] = np.ones((2, 6))          # 2 species against the centers' 3
    data["temperature_faces"] = np.ones((2, 6))
    assert any("density_faces" in problem and "leading axes" in problem for problem in _check_transport_solution(data))


# Er shares density's time axis and drops only the species axis, so a same-rank Er over a different number of times
# is still incoherent.
def test_er_with_a_different_time_axis_flagged() -> None:
    data = _valid_data(n_species=3, n_rho=5)
    data["density"] = np.ones((2, 3, 5))
    data["temperature"] = np.ones((2, 3, 5))
    data["Er"] = np.ones((4, 5))                     # 4 times against density's 2
    data["density_faces"] = np.ones((2, 3, 6))
    data["temperature_faces"] = np.ones((2, 3, 6))
    data["Er_faces"] = np.ones((2, 6))
    assert any("'Er'" in problem and "leading axes" in problem for problem in _check_transport_solution(data))


# The optional pressure arrays are species-resolved like density, so they must carry the same species and time axes.
@pytest.mark.parametrize(("field", "density_field"), [("pressure", "density"), ("pressure_faces", "density_faces")])
def test_pressure_with_a_different_species_count_flagged(field: str, density_field: str) -> None:
    data = _valid_data(n_species=3, n_rho=5)
    n_points = data[density_field].shape[-1]
    data[field] = np.ones((2, n_points))             # 2 species against density's 3
    assert any(field in problem and "leading axes" in problem for problem in _check_transport_solution(data))


# --- ts against a static solution ---

# A static solution is one slice, so `ts` describes it with exactly one timestamp. The Stage 3/4 loaders and the
# feedback writer all date such a file from ts[0], so extra entries name times no slice in the file corresponds to.
def test_static_profiles_with_more_than_one_timestamp_flagged() -> None:
    data = _valid_data(n_species=3, n_rho=5)
    data["ts"] = np.array([1.0, 2.0])  # two timestamps for a single static slice
    assert any("ts" in problem and "single static" in problem for problem in _check_transport_solution(data))


# One slice, one timestamp, which every reader reports identically.
def test_static_profiles_with_one_timestamp_passes() -> None:
    data = _valid_data(n_species=3, n_rho=5)
    data["ts"] = np.array([1.5])
    assert _check_transport_solution(data) == []
