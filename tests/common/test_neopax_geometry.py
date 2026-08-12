"""Test the minor radius from the two transport face grids.

``minor_radius_from_face_grids`` supplies the conversion from per metre to per unit rho.
Every transport solution consumer uses its result. The radius must be consistent with the grids.
It must also be a positive length. These tests cover linear grids that imply invalid radii.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.testing import assert_allclose

from common.neopax_geometry import minor_radius_from_face_grids

FACES = np.linspace(0.0, 1.0, 6)


def test_a_uniformly_scaled_grid_gives_its_scale() -> None:
    assert_allclose(minor_radius_from_face_grids(0.4 * FACES, FACES, source="probe"), 0.4, rtol=1e-15)


# Both ``-0.4 * rho_face`` and ``0.4 * rho_face`` are linear. A linearity check cannot
# distinguish them. A negative radius reverses ``dTHatdrNs`` and ``tprim``.
# A zero radius makes all converted gradients zero. Later checks do not detect these errors.
@pytest.mark.parametrize("scale", [-0.4, 0.0], ids=["negative", "zero"])
def test_a_radius_that_is_not_positive_is_rejected(scale: float) -> None:
    with pytest.raises(ValueError, match="implies a minor radius"):
        minor_radius_from_face_grids(scale * FACES, FACES, source="probe")


# The outermost normalized face is the plasma edge. A zero value causes division by zero.
# The resultant NaN could bypass a linearity comparison.
def test_a_normalized_grid_ending_at_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="rho_face ends at"):
        minor_radius_from_face_grids(np.zeros(6), np.zeros(6), source="probe")


@pytest.mark.parametrize(
    ("r_half", "rho"),
    [
        (np.array([]), np.array([])),
        (np.array([0.4]), np.array([1.0])),
        (np.zeros((2, 3)), np.zeros((2, 3))),
    ],
    ids=["empty", "one_face", "two_dimensional"],
)
def test_a_grid_that_cannot_span_a_cell_is_rejected(r_half: np.ndarray, rho: np.ndarray) -> None:
    with pytest.raises(ValueError, match="must be 1-D with at least 2 faces"):
        minor_radius_from_face_grids(r_half, rho, source="probe")


@pytest.mark.parametrize("bad", [np.nan, np.inf], ids=["nan", "inf"])
def test_a_non_finite_grid_is_rejected(bad: float) -> None:
    r_half = 0.4 * FACES
    r_half[2] = bad
    with pytest.raises(ValueError, match="holds non-finite values"):
        minor_radius_from_face_grids(r_half, FACES, source="probe")


# A bent grid causes a radius-dependent error in each converted quantity.
def test_a_nonlinear_metre_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="not rho_face scaled by one minor radius"):
        minor_radius_from_face_grids(0.4 * FACES**2, FACES, source="probe")


# The message gives the face index and both values. This data identifies a bent grid in the log.
def test_the_nonlinear_message_names_the_worst_face() -> None:
    with pytest.raises(ValueError, match=r"At face 1, r_grid_half = "):
        minor_radius_from_face_grids(np.array([0.0, 9.0, 0.16, 0.24, 0.32, 0.4]), FACES, source="probe")
