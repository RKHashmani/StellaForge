"""Test the NumPy port of the NEOPAX face operator.

``profile_gradients.py`` reproduces the NEOPAX staggered finite-volume reconstruction.
Stages 3 and 4 use this reconstruction to calculate face-grid radial gradients.
The golden values come from NEOPAX ``_face_profile_gradient`` and ``_face_profile`` calls.
They test the port without a NEOPAX installation. A separate test compares seeded cases with
the live solver when NEOPAX is available.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pytest
from numpy.testing import assert_allclose

from common import profile_gradients as pg

# NumPy and JAX operation order causes a difference of a few units in the last place.
# This tolerance includes that difference.
RTOL = 1.0e-12


class GoldenCase(NamedTuple):
    """One pinned comparison against NEOPAX.

    Attributes
    ----------
    id : str
        Name shown by pytest for the parametrized case.
    field : str
        Field name looked up under [boundary].
    species : tuple of str
        Species names in the order the profile rows use.
    config : dict or None
        Parsed config holding the [boundary.<field>] block, or None for a field with no block.
    profile : list
        Cell-centered profile.
    faces : list of float
        Face coordinates.
    gradient : list
        Face derivatives NEOPAX produces for this case.
    values : list
        Face values NEOPAX produces for this case.
    """

    id: str
    field: str
    species: tuple[str, ...]
    config: dict[str, Any] | None
    profile: list[Any]
    faces: list[float]
    gradient: list[Any]
    values: list[Any]


GOLDEN_CASES: list[GoldenCase] = [
    # This nonuniform grid has no density boundary block. The axis derivative is zero.
    # The edge value is ``1.5 * u[-1] - 0.5 * u[-2]``.
    GoldenCase(
        id="no_block_nonuniform_two_species",
        field="density",
        species=("e", "ion"),
        config=None,
        profile=[
            [1.0, 0.95, 0.86, 0.72, 0.55, 0.42, 0.33],
            [2.4, 2.31, 2.05, 1.7, 1.25, 0.9, 0.6],
        ],
        faces=[0.0, 0.08, 0.2, 0.36, 0.6, 0.78, 0.9, 1.0],
        gradient=[
            [
                0.0, -0.48809523809523675, -0.6394957983193266, -0.6893147502903605, -0.814285714285714,
                -0.8610722610722613, -0.8232954545454556, -0.8999999999999999,
            ],
            [
                0.0, -0.8202380952380901, -1.863445378151262, -1.7116724738675952, -2.1587301587301586,
                -2.3787878787878776, -2.7443181818181865, -3.000000000000002,
            ],
        ],
        values=[
            [
                1.0, 0.98, 0.9114285714285714, 0.804, 0.6228571428571429, 0.47200000000000003, 0.3709090909090909,
                0.28500000000000003,
            ],
            [2.4, 2.364, 2.1985714285714284, 1.91, 1.4428571428571428, 1.04, 0.7363636363636363, 0.4499999999999999],
        ],
    ),

    # This case has the W7-X validation shape. A default entry sets a zero axis gradient.
    # A per-species table sets the edge values.
    GoldenCase(
        id="neumann_axis_dirichlet_edge_two_species",
        field="density",
        species=("e", "ion"),
        config={
            "boundary": {
                "density": {
                    "left": {"type": "neumann", "gradient": {"default": 0.0}},
                    "right": {"type": "dirichlet", "value": {"e": 0.29, "ion": 0.31}},
                },
            },
        },
        profile=[
            [1.0, 0.95, 0.86, 0.72, 0.55, 0.42, 0.33],
            [2.4, 2.31, 2.05, 1.7, 1.25, 0.9, 0.6],
        ],
        faces=[0.0, 0.08, 0.2, 0.36, 0.6, 0.78, 0.9, 1.0],
        gradient=[
            [
                0.0, -0.48809523809523675, -0.6394957983193266, -0.6893147502903605, -0.814285714285714,
                -0.8610722610722613, -0.8170454545454556, -0.6969696969696989,
            ],
            [
                0.0, -0.8202380952380901, -1.863445378151262, -1.7116724738675952, -2.1587301587301586,
                -2.3787878787878776, -2.9193181818181864, -6.121212121212126,
            ],
        ],
        values=[
            [
                1.00625, 0.98, 0.9114285714285714, 0.804, 0.6228571428571429, 0.47200000000000003,
                0.3709090909090909, 0.29,
            ],
            [2.41125, 2.364, 2.1985714285714284, 1.91, 1.4428571428571428, 1.04, 0.7363636363636363, 0.31],
        ],
    ),

    # This uniform grid has three species. One species overrides the default axis gradient.
    # The Robin closure calculates its edge value from the decay length. It ignores the value table.
    GoldenCase(
        id="neumann_axis_robin_edge_three_species",
        field="temperature",
        species=("e", "D", "T"),
        config={
            "boundary": {
                "temperature": {
                    "left": {"type": "neumann", "gradient": {"default": 0.0, "D": -0.25}},
                    "right": {"type": "robin", "value": {"default": 1.0}, "decay_length": {"default": 0.05}},
                },
            },
        },
        profile=[
            [3.0, 2.85, 2.6, 2.25, 1.85, 1.4, 1.0, 0.7],
            [1.5, 1.44, 1.33, 1.18, 0.99, 0.78, 0.58, 0.41],
            [0.5, 0.49, 0.46, 0.42, 0.36, 0.29, 0.22, 0.16],
        ],
        faces=[0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0],
        gradient=[
            [
                0.0, -1.1999999999999993, -2.0, -2.8000000000000007, -3.1999999999999993, -3.6000000000000014,
                -3.1999999999999993, -2.4000000000000004, -6.838709677348761,
            ],
            [
                -0.25, -0.4800000000000004, -0.879999999999999, -1.200000000000001, -1.5199999999999996,
                -1.6799999999999997, -1.6000000000000005, -1.3599999999999999, -4.012903225765028,
            ],
            [
                0.0, -0.08000000000000007, -0.23999999999999977, -0.3200000000000003, -0.48, -0.56,
                -0.5599999999999998, -0.48, -1.574193548370847,
            ],
        ],
        values=[
            [3.01875, 2.925, 2.725, 2.425, 2.05, 1.625, 1.2, 0.85, 0.34193548387427675],
            [1.51921875, 1.47, 1.385, 1.255, 1.085, 0.885, 0.6799999999999999, 0.495, 0.20064516129226428],
            [0.50125, 0.495, 0.475, 0.44, 0.39, 0.32499999999999996, 0.255, 0.19, 0.07870967742011654],
        ],
    ),

    # This 1-D profile has scalar boundary values. The result must also be 1-D.
    GoldenCase(
        id="dirichlet_scalar_values_single_profile",
        field="Er",
        species=("e",),
        config={
            "boundary": {
                "Er": {
                    "left": {"type": "dirichlet", "value": 0.0},
                    "right": {"type": "dirichlet", "value": 0.12},
                },
            },
        },
        profile=[0.9, 0.83, 0.71, 0.55, 0.4, 0.28, 0.19],
        faces=[0.0, 0.08, 0.2, 0.36, 0.6, 0.78, 0.9, 1.0],
        gradient=[
            24.233333333333327, -0.6869047619047614, -0.8605042016806715, -0.808362369337979, -0.7214285714285715,
            -0.8020979020979019, -0.8545454545454559, -1.4242424242424259,
        ],
        values=[0.0, 0.872, 0.7785714285714286, 0.646, 0.4642857142857143, 0.328, 0.23090909090909092, 0.12],
    ),

    # This 1-D profile uses a Robin axis and a Neumann edge.
    GoldenCase(
        id="robin_axis_neumann_edge_single_profile",
        field="temperature",
        species=("e",),
        config={
            "boundary": {
                "temperature": {
                    "left": {"type": "robin", "decay_length": 0.3},
                    "right": {"type": "neumann", "gradient": -0.75},
                },
            },
        },
        profile=[0.9, 0.83, 0.71, 0.55, 0.4, 0.28, 0.19],
        faces=[0.0, 0.08, 0.2, 0.36, 0.6, 0.78, 0.9, 1.0],
        gradient=[
            2.692592592584614, -0.6869047619047614, -0.8605042016806715, -0.808362369337979, -0.7214285714285715,
            -0.8020979020979019, -0.8197798295454558, -0.75,
        ],
        values=[
            0.8077777777780768, 0.872, 0.7785714285714286, 0.646, 0.4642857142857143, 0.328, 0.23090909090909092,
            0.14781250000000004,
        ],
    ),

    # Neither Dirichlet boundary gives a value. Each boundary face therefore uses its adjacent cell.
    GoldenCase(
        id="dirichlet_both_sides_without_values",
        field="density",
        species=("e", "ion"),
        config={
            "boundary": {
                "density": {
                    "left": {"type": "dirichlet"},
                    "right": {"type": "dirichlet"},
                },
            },
        },
        profile=[
            [1.0, 0.95, 0.86, 0.72, 0.55, 0.42, 0.33],
            [2.4, 2.31, 2.05, 1.7, 1.25, 0.9, 0.6],
        ],
        faces=[0.0, 0.08, 0.2, 0.36, 0.6, 0.78, 0.9, 1.0],
        gradient=[
            [
                0.1666666666666668, -0.48809523809523675, -0.6394957983193266, -0.6893147502903605,
                -0.814285714285714, -0.8610722610722613, -0.7670454545454556, 0.27272727272727276,
            ],
            [
                0.29999999999999505, -0.8202380952380901, -1.863445378151262, -1.7116724738675952,
                -2.1587301587301586, -2.3787878787878776, -2.556818181818186, 0.9090909090909113,
            ],
        ],
        values=[
            [1.0, 0.98, 0.9114285714285714, 0.804, 0.6228571428571429, 0.47200000000000003, 0.3709090909090909, 0.33],
            [2.4, 2.364, 2.1985714285714284, 1.91, 1.4428571428571428, 1.04, 0.7363636363636363, 0.6],
        ],
    ),

    # Three cells, the smallest profile that still uses the quadratic interior stencil.
    GoldenCase(
        id="three_cells_no_block",
        field="density",
        species=("e", "ion"),
        config=None,
        profile=[
            [1.0, 0.7, 0.4],
            [2.0, 1.3, 0.9],
        ],
        faces=[0.0, 0.25, 0.7, 1.0],
        gradient=[
            [0.0, -0.8650246305418716, -0.828571428571428, -0.9999999999999992],
            [0.0, -2.128735632183907, -1.104761904761904, -1.3333333333333328],
        ],
        values=[
            [1.0, 0.8928571428571428, 0.52, 0.2500000000000001],
            [2.0, 1.75, 1.06, 0.7000000000000001],
        ],
    ),

    # Two cells give one centered interior slope. Fewer than four faces change the boundary spacing.
    # The boundary stencil then uses the boundary cell width.
    GoldenCase(
        id="two_cells_no_block",
        field="density",
        species=("e", "ion"),
        config=None,
        profile=[
            [1.0, 0.6],
            [2.0, 1.1],
        ],
        faces=[0.0, 0.4, 1.0],
        gradient=[
            [0.0, -0.8000000000000002, -0.666666666666667],
            [0.0, -1.8, -1.5],
        ],
        values=[
            [1.0, 0.84, 0.3999999999999999],
            [2.0, 1.64, 0.6500000000000001],
        ],
    ),

    # These constrained two-cell profiles test the narrow-grid stencil spacing.
    GoldenCase(
        id="two_cells_neumann_axis_dirichlet_edge",
        field="density",
        species=("e", "ion"),
        config={
            "boundary": {
                "density": {
                    "left": {"type": "neumann", "gradient": {"default": 0.0}},
                    "right": {"type": "dirichlet", "value": {"default": 0.25}},
                },
            },
        },
        profile=[
            [1.0, 0.6],
            [2.0, 1.1],
        ],
        faces=[0.0, 0.4, 1.0],
        gradient=[
            [0.0, -0.8000000000000002, -1.3333333333333333],
            [0.0, -1.8, -3.277777777777778],
        ],
        values=[
            [1.05, 0.84, 0.25],
            [2.1125, 1.64, 0.25],
        ],
    ),

    # A single cell has two boundary faces and no interior stencil.
    GoldenCase(
        id="one_cell_neumann_axis_dirichlet_edge",
        field="density",
        species=("e", "ion"),
        config={
            "boundary": {
                "density": {
                    "left": {"type": "neumann", "gradient": {"default": 0.0}},
                    "right": {"type": "dirichlet", "value": {"default": 0.25}},
                },
            },
        },
        profile=[
            [0.8],
            [1.6],
        ],
        faces=[0.0, 1.0],
        gradient=[
            [0.0, -1.4666666666666668],
            [0.0, -3.6],
        ],
        values=[
            [0.8, 0.25],
            [1.6, 0.25],
        ],
    ),
]


def _port_faces(case: GoldenCase) -> tuple[np.ndarray, np.ndarray]:
    """Run the port end to end for one golden case.

    Parameters
    ----------
    case : GoldenCase
        The pinned case to evaluate.

    Returns
    -------
    tuple of numpy.ndarray
        Face derivatives and face values from the port.
    """
    model = None if case.config is None else pg.boundary_model_from_config(case.config, case.field, case.species)
    constraints = pg.face_constraints(model, case.profile, case.faces)
    gradient = pg.face_gradient(case.profile, case.faces, **constraints._asdict())
    values = pg.face_value(case.profile, case.faces, **constraints._asdict())
    return gradient, values


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[case.id for case in GOLDEN_CASES])
def test_golden_face_gradient(case: GoldenCase) -> None:
    gradient, _ = _port_faces(case)
    expected = np.asarray(case.gradient, dtype=np.float64)
    assert gradient.shape == expected.shape
    assert_allclose(gradient, expected, rtol=RTOL, atol=0.0)


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=[case.id for case in GOLDEN_CASES])
def test_golden_face_value(case: GoldenCase) -> None:
    _, values = _port_faces(case)
    expected = np.asarray(case.values, dtype=np.float64)
    assert values.shape == expected.shape
    assert_allclose(values, expected, rtol=RTOL, atol=0.0)


def test_single_species_row_matches_flat_profile() -> None:
    """A one-row 2-D profile and the same profile as a flat array must give the same faces."""
    profile = np.array([1.0, 0.9, 0.75, 0.55, 0.3])
    faces = np.array([0.0, 0.15, 0.35, 0.6, 0.85, 1.0])
    flat = pg.face_gradient(profile, faces, right_value=0.2)
    stacked = pg.face_gradient(profile[None, :], faces, right_value=[0.2])
    assert flat.shape == (6,)
    assert stacked.shape == (1, 6)
    assert_allclose(stacked[0], flat, rtol=0.0, atol=0.0)


def test_face_constraints_without_model_extrapolates_the_edge() -> None:
    """Use zero axis gradient and linear edge extrapolation without a [boundary] block."""
    profile = np.array([[1.0, 0.8, 0.5], [2.0, 1.4, 1.0]])
    faces = np.array([0.0, 0.3, 0.7, 1.0])
    constraints = pg.face_constraints(None, profile, faces)
    assert constraints.left_value is None
    assert constraints.right_grad is None
    assert_allclose(constraints.left_grad, np.zeros(2), rtol=0.0, atol=0.0)
    assert_allclose(constraints.right_value, [1.5 * 0.5 - 0.5 * 0.8, 1.5 * 1.0 - 0.5 * 1.4], rtol=RTOL, atol=0.0)


def test_boundary_model_absent_field_returns_none() -> None:
    assert pg.boundary_model_from_config({"boundary": {"density": {}}}, "temperature", ("e",)) is None
    assert pg.boundary_model_from_config({}, "density", ("e",)) is None


def test_boundary_model_orders_species_and_applies_default() -> None:
    config = {
        "boundary": {
            "density": {
                "left": {"type": "neumann", "gradient": {"default": 0.0}},
                "right": {"type": "dirichlet", "value": {"default": 0.2, "e": 0.4}},
            }
        }
    }
    model = pg.boundary_model_from_config(config, "density", ("D", "e", "T"))
    assert model.left_type == "neumann"
    assert model.right_type == "dirichlet"
    assert_allclose(model.right_value, [0.2, 0.4, 0.2], rtol=0.0, atol=0.0)
    assert_allclose(model.left_gradient, [0.0, 0.0, 0.0], rtol=0.0, atol=0.0)
    assert model.right_gradient is None
    assert model.right_decay_length is None


def test_unsupported_boundary_type_raises() -> None:
    config = {"boundary": {"Er": {"left": {"type": "dirichlet"}, "right": {"type": "floating_ambipolar_edge"}}}}
    with pytest.raises(ValueError, match="floating_ambipolar_edge"):
        pg.boundary_model_from_config(config, "Er", ("e",))


def test_missing_species_entry_without_default_raises() -> None:
    config = {"boundary": {"density": {"right": {"type": "dirichlet", "value": {"e": 0.3}}}}}
    with pytest.raises(ValueError, match="no entry for species 'ion'"):
        pg.boundary_model_from_config(config, "density", ("e", "ion"))


def test_face_grid_length_is_checked() -> None:
    with pytest.raises(ValueError, match="n_radial \\+ 1"):
        pg.face_gradient(np.array([1.0, 0.5]), np.array([0.0, 1.0]))


def test_face_grid_must_be_strictly_increasing() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        pg.face_gradient(np.array([1.0, 0.5]), np.array([0.0, 0.6, 0.6]))


def test_constraint_length_is_checked() -> None:
    with pytest.raises(ValueError, match="one per species"):
        pg.face_gradient(np.zeros((2, 3)), np.linspace(0.0, 1.0, 4), right_value=[0.1, 0.2, 0.3])


def test_profile_rank_is_checked() -> None:
    with pytest.raises(ValueError, match="1-D .* or 2-D"):
        pg.face_gradient(np.zeros((2, 2, 3)), np.linspace(0.0, 1.0, 4))


def test_matches_neopax_on_random_inputs() -> None:
    """Compare the port with live NEOPAX on seeded grids, profiles and boundary types."""
    pytest.importorskip("NEOPAX")
    import jax.numpy as jnp
    from NEOPAX._boundary_conditions import build_boundary_condition_model
    from NEOPAX._transport_flux_models import _face_profile, _face_profile_gradient

    rng = np.random.default_rng(20260811)
    for _ in range(40):
        n_cell = int(rng.integers(1, 12))
        n_species = int(rng.integers(1, 4))
        faces = np.concatenate([[0.0], np.cumsum(rng.uniform(0.05, 1.0, size=n_cell))])
        faces = faces / faces[-1]
        profile = rng.uniform(0.1, 3.0, size=(n_species, n_cell))
        names = ("e", "D", "T")[:n_species]
        sides = {}
        for side in ("left", "right"):
            sides[side] = {
                "type": str(rng.choice(pg.SUPPORTED_BOUNDARY_TYPES)),
                "value": {"default": float(rng.uniform(0.1, 1.0))},
                "gradient": {"default": float(rng.uniform(-2.0, 2.0))},
                "decay_length": {"default": float(rng.uniform(0.02, 1.0))},
            }
        config = {"boundary": {"density": sides}}

        for use_config in (None, config):
            if use_config is None:
                bc_model = None
            else:
                bc_model = build_boundary_condition_model(
                    use_config["boundary"]["density"], dr=float(np.diff(faces)[0]), species_names=list(names)
                )
            reference_gradient = np.asarray(
                _face_profile_gradient(jnp.asarray(profile), jnp.asarray(faces), bc_model=bc_model)
            )
            reference_values = np.asarray(_face_profile(jnp.asarray(profile), jnp.asarray(faces), bc_model=bc_model))

            model = None if use_config is None else pg.boundary_model_from_config(use_config, "density", names)
            constraints = pg.face_constraints(model, profile, faces)
            assert_allclose(
                pg.face_gradient(profile, faces, **constraints._asdict()), reference_gradient, rtol=RTOL, atol=1.0e-12
            )
            assert_allclose(
                pg.face_value(profile, faces, **constraints._asdict()), reference_values, rtol=RTOL, atol=1.0e-12
            )
