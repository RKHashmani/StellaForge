"""Reproduce the NEOPAX finite-volume operator on the profile face grid.

NEOPAX stores each profile on ``n_radial`` cell centers. It reconstructs face values with the
configured radial boundary constraints. This module provides a NumPy port of that reconstruction.
Its gradients agree with the gradients that NEOPAX builds internally.
Differentiation of the reconstructed face values does not give the same result.

``face_gradient`` ports ``CellVariable.face_grad``. ``face_value`` ports the linear form of
``CellVariable.face_value``. ``boundary_model_from_config`` and ``face_constraints`` convert the
boundary configuration to per-species face constraints.

All arrays use float64 values. The last axis is radial. A 1-D input gives a 1-D result. Profiles
with multiple species have shape ``(n_species, n_radial)``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# Boundary condition types accepted in [boundary.<field>.<side>].
SUPPORTED_BOUNDARY_TYPES = ("dirichlet", "neumann", "robin")

# This offset prevents zero decay lengths in the Robin denominators. NEOPAX uses the same offset.
_DECAY_EPS = 1.0e-12


def _as_2d(values: npt.ArrayLike, name: str) -> tuple[np.ndarray, bool]:
    """Convert ``values`` to a float64 profile and return a squeeze flag.

    Parameters
    ----------
    values : array_like
        Profile shaped ``(n_radial,)`` or ``(n_species, n_radial)``.
    name : str
        Argument name used in the error message.

    Returns
    -------
    tuple of (numpy.ndarray, bool)
        The 2-D profile and a flag that identifies a 1-D input.

    Raises
    ------
    ValueError
        If ``values`` is not 1-D or 2-D, or holds no radial points.
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim not in (1, 2):
        raise ValueError(f"{name} must be 1-D (n_radial,) or 2-D (n_species, n_radial), got shape {arr.shape}")
    if arr.shape[-1] < 1:
        raise ValueError(f"{name} must hold at least one radial point, got shape {arr.shape}")
    if arr.ndim == 1:
        return arr[None, :], True
    return arr, False


def _as_face_grid(face_grid: npt.ArrayLike, n_cell: int) -> np.ndarray:
    """Validate the face coordinates against a profile with ``n_cell`` cells.

    Parameters
    ----------
    face_grid : array_like
        Face coordinates, expected to hold ``n_cell + 1`` entries.
    n_cell : int
        Number of cell centers in the profile.

    Returns
    -------
    numpy.ndarray
        The face coordinates as float64.

    Raises
    ------
    ValueError
        If the grid is not 1-D, does not hold ``n_cell + 1`` entries, or is not strictly increasing.
    """
    faces = np.asarray(face_grid, dtype=np.float64)
    if faces.ndim != 1:
        raise ValueError(f"face_grid must be 1-D, got shape {faces.shape}")
    if faces.shape[0] != n_cell + 1:
        raise ValueError(
            f"face_grid must hold n_radial + 1 = {n_cell + 1} coordinates for a profile with {n_cell} cells, "
            f"got {faces.shape[0]}"
        )
    if not np.all(np.diff(faces) > 0.0):
        raise ValueError("face_grid must be strictly increasing, since its spacings appear in every denominator")
    return faces


def _as_column(value: npt.ArrayLike | None, n_species: int, name: str) -> np.ndarray | None:
    """Convert a face constraint to a ``(n_species, 1)`` column.

    Parameters
    ----------
    value : array_like or None
        Scalar constraint, or one entry per species.
    n_species : int
        Number of species rows in the profile.
    name : str
        Argument name used in the error message.

    Returns
    -------
    numpy.ndarray or None
        The constraint as a float64 column, or None if ``value`` is None.

    Raises
    ------
    ValueError
        If ``value`` is neither a scalar nor one entry per species.
    """
    if value is None:
        return None
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full((n_species, 1), float(arr))
    if arr.shape == (n_species,):
        return arr[:, None]
    if arr.shape == (n_species, 1):
        return arr
    raise ValueError(f"{name} must be a scalar or hold {n_species} entries, one per species, got shape {arr.shape}")


def _quadratic_face_derivative(
    x_eval: npt.ArrayLike,
    x0: npt.ArrayLike,
    x1: npt.ArrayLike,
    x2: npt.ArrayLike,
    y0: npt.ArrayLike,
    y1: npt.ArrayLike,
    y2: npt.ArrayLike,
) -> np.ndarray:
    """Evaluate the derivative of the quadratic through three points.

    The quadratic passes through ``(x0, y0)``, ``(x1, y1)`` and ``(x2, y2)``.
    The Lagrange basis gives its derivative at ``x_eval``.

    Parameters
    ----------
    x_eval : array_like
        Coordinate at which the derivative is evaluated.
    x0, x1, x2 : array_like
        Distinct interpolation coordinates.
    y0, y1, y2 : array_like
        Values at ``x0``, ``x1`` and ``x2``. These broadcast against the coordinates.

    Returns
    -------
    numpy.ndarray
        The derivative, broadcast over the inputs.
    """
    l0_prime = (2.0 * x_eval - x1 - x2) / ((x0 - x1) * (x0 - x2))
    l1_prime = (2.0 * x_eval - x0 - x2) / ((x1 - x0) * (x1 - x2))
    l2_prime = (2.0 * x_eval - x0 - x1) / ((x2 - x0) * (x2 - x1))
    return y0 * l0_prime + y1 * l1_prime + y2 * l2_prime


def _left_face_value(u: np.ndarray, left_value: np.ndarray | None, left_grad: np.ndarray | None,
                     half_width: float) -> np.ndarray:
    """Axis face value implied by the constraints, preferring a value over a gradient.

    Parameters
    ----------
    u : numpy.ndarray
        Cell-centered profile shaped ``(n_species, n_radial)``.
    left_value : numpy.ndarray or None
        Axis face value column.
    left_grad : numpy.ndarray or None
        Axis face derivative column.
    half_width : float
        Half the width of the innermost cell.

    Returns
    -------
    numpy.ndarray
        Axis face value shaped ``(n_species, 1)``.
    """
    if left_value is not None:
        return np.broadcast_to(left_value, (u.shape[0], 1)).copy()
    if left_grad is not None:
        return u[:, :1] - left_grad * half_width
    return u[:, :1]


def _right_face_value(u: np.ndarray, right_value: np.ndarray | None, right_grad: np.ndarray | None,
                      half_width: float) -> np.ndarray:
    """Edge face value implied by the constraints, preferring a value over a gradient.

    Parameters
    ----------
    u : numpy.ndarray
        Cell-centered profile shaped ``(n_species, n_radial)``.
    right_value : numpy.ndarray or None
        Edge face value column.
    right_grad : numpy.ndarray or None
        Edge face derivative column.
    half_width : float
        Half the width of the outermost cell.

    Returns
    -------
    numpy.ndarray
        Edge face value shaped ``(n_species, 1)``.
    """
    if right_value is not None:
        return np.broadcast_to(right_value, (u.shape[0], 1)).copy()
    if right_grad is not None:
        return u[:, -1:] + right_grad * half_width
    return u[:, -1:]


def face_gradient(
    values_on_centers: npt.ArrayLike,
    face_grid: npt.ArrayLike,
    *,
    left_value: npt.ArrayLike | None = None,
    left_grad: npt.ArrayLike | None = 0.0,
    right_value: npt.ArrayLike | None = None,
    right_grad: npt.ArrayLike | None = None,
) -> np.ndarray:
    """Evaluate a cell-centered profile derivative on the face grid.

    This function reproduces ``CellVariable.face_grad`` from NEOPAX. A gradient constraint directly
    sets a boundary derivative. Otherwise, a one-sided difference sets the boundary derivative.
    This difference uses the face value and the adjacent cell center. A quadratic through the three
    nearest centers sets each interior derivative. The last interior stencil also uses the edge
    face value.
    This stencil stays inside the domain.

    Parameters
    ----------
    values_on_centers : array_like
        Profile on the cell centers. Its shape is ``(n_radial,)`` or
        ``(n_species, n_radial)``.
    face_grid : array_like
        The ``n_radial + 1`` face coordinates, strictly increasing.
    left_value : array_like, optional
        Axis face value, scalar or one entry per species.
    left_grad : array_like, optional
        Axis face derivative, scalar or one entry per species. It sets the axis entry of the result.
        The default is 0.0, which is the NEOPAX default for a profile cell variable.
    right_value : array_like, optional
        Edge face value. It also enters the outermost interior stencil.
    right_grad : array_like, optional
        Edge face derivative, scalar or one entry per species. It sets the edge entry of the result.

    Returns
    -------
    numpy.ndarray
        Derivatives on the ``n_radial + 1`` faces, shaped ``(n_radial + 1,)`` for a 1-D input and
        ``(n_species, n_radial + 1)`` otherwise.

    Raises
    ------
    ValueError
        If the profile is not 1-D or 2-D.
        If ``face_grid`` does not contain ``n_radial + 1`` strictly increasing coordinates.
        If a constraint is not a scalar or one entry per species.
    """
    u, squeeze = _as_2d(values_on_centers, "values_on_centers")
    n_species, n_cell = u.shape
    faces = _as_face_grid(face_grid, n_cell)
    lv = _as_column(left_value, n_species, "left_value")
    lg = _as_column(left_grad, n_species, "left_grad")
    rv = _as_column(right_value, n_species, "right_value")
    rg = _as_column(right_grad, n_species, "right_grad")

    widths = np.diff(faces)
    half_width_left = widths[0] / 2.0
    half_width_right = widths[-1] / 2.0
    right_face = _right_face_value(u, rv, rg, half_width_right)

    if lg is not None:
        left_slope = np.broadcast_to(lg, (n_species, 1)).copy()
    else:
        left_slope = (u[:, :1] - _left_face_value(u, lv, lg, half_width_left)) / half_width_left
    if rg is not None:
        right_slope = np.broadcast_to(rg, (n_species, 1)).copy()
    else:
        right_slope = (right_face - u[:, -1:]) / half_width_right

    if n_cell == 1:
        grads = np.concatenate([left_slope, right_slope], axis=-1)
    elif n_cell == 2:
        x = 0.5 * (faces[1:] + faces[:-1])
        slope = (u[:, 1:2] - u[:, 0:1]) / (x[1] - x[0])
        grads = np.concatenate([left_slope, slope, right_slope], axis=-1)
    else:
        x = 0.5 * (faces[1:] + faces[:-1])
        x_eval = faces[1:-1]
        inner = _quadratic_face_derivative(
            x_eval[:-1], x[:-2], x[1:-1], x[2:], u[:, :-2], u[:, 1:-1], u[:, 2:]
        )
        last_inner = _quadratic_face_derivative(
            faces[-2], x[-2], x[-1], faces[-1], u[:, -2:-1], u[:, -1:], right_face
        )
        grads = np.concatenate([left_slope, inner, last_inner, right_slope], axis=-1)

    return grads[0] if squeeze else grads


def face_value(
    values_on_centers: npt.ArrayLike,
    face_grid: npt.ArrayLike,
    *,
    left_value: npt.ArrayLike | None = None,
    left_grad: npt.ArrayLike | None = 0.0,
    right_value: npt.ArrayLike | None = None,
    right_grad: npt.ArrayLike | None = None,
) -> np.ndarray:
    """Cell-centered profile interpolated onto the face grid.

    This function reproduces the linear form of ``CellVariable.face_value`` from NEOPAX.
    A value constraint sets a boundary face value. Otherwise, the gradient constraint extrapolates
    the adjacent cell center. Linear interpolation between centers sets the interior values.

    Parameters
    ----------
    values_on_centers : array_like
        Profile on the cell centers. Its shape is ``(n_radial,)`` or
        ``(n_species, n_radial)``.
    face_grid : array_like
        The ``n_radial + 1`` face coordinates, strictly increasing.
    left_value : array_like, optional
        Axis face value, scalar or one entry per species. It overrides ``left_grad``.
    left_grad : array_like, optional
        Axis face derivative, scalar or one entry per species. The default is 0.0.
        This value is the NEOPAX default for a profile cell variable.
    right_value : array_like, optional
        Edge face value, scalar or one entry per species. It overrides ``right_grad``.
    right_grad : array_like, optional
        Edge face derivative, scalar or one entry per species.

    Returns
    -------
    numpy.ndarray
        Values on the ``n_radial + 1`` faces, shaped ``(n_radial + 1,)`` for a 1-D input and
        ``(n_species, n_radial + 1)`` otherwise.

    Raises
    ------
    ValueError
        If the profile is not 1-D or 2-D.
        If ``face_grid`` does not contain ``n_radial + 1`` strictly increasing coordinates.
        If a constraint is not a scalar or one entry per species.
    """
    u, squeeze = _as_2d(values_on_centers, "values_on_centers")
    n_species, n_cell = u.shape
    faces = _as_face_grid(face_grid, n_cell)
    lv = _as_column(left_value, n_species, "left_value")
    lg = _as_column(left_grad, n_species, "left_grad")
    rv = _as_column(right_value, n_species, "right_value")
    rg = _as_column(right_grad, n_species, "right_grad")

    widths = np.diff(faces)
    left_face = _left_face_value(u, lv, lg, widths[0] / 2.0)
    right_face = _right_face_value(u, rv, rg, widths[-1] / 2.0)

    if n_cell == 1:
        values = np.concatenate([left_face, right_face], axis=-1)
    else:
        x = 0.5 * (faces[1:] + faces[:-1])
        weight = (faces[1:-1] - x[:-1]) / (x[1:] - x[:-1])
        inner = u[:, :-1] + weight * (u[:, 1:] - u[:, :-1])
        values = np.concatenate([left_face, inner, right_face], axis=-1)

    return values[0] if squeeze else values


@dataclass(frozen=True)
class BoundaryModel:
    """Radial boundary condition types and coefficients for one profile field.

    Attributes
    ----------
    left_type, right_type : str
        One of ``SUPPORTED_BOUNDARY_TYPES``, lower case.
    left_value, right_value : numpy.ndarray or None
        Prescribed boundary face values, read by the dirichlet branch.
    left_gradient, right_gradient : numpy.ndarray or None
        Prescribed boundary face derivatives, read by the neumann branch.
    left_decay_length, right_decay_length : numpy.ndarray or None
        Robin decay lengths. The closure divides the boundary value by this length. The resultant
        derivative points inward.
    """

    left_type: str
    right_type: str
    left_value: np.ndarray | None = None
    right_value: np.ndarray | None = None
    left_gradient: np.ndarray | None = None
    right_gradient: np.ndarray | None = None
    left_decay_length: np.ndarray | None = None
    right_decay_length: np.ndarray | None = None


class FaceConstraints(NamedTuple):
    """Face constraints for one field, with one entry per species in each array.

    Attributes
    ----------
    left_value, right_value : numpy.ndarray or None
        Boundary face values shaped ``(n_species,)``.
    left_grad, right_grad : numpy.ndarray or None
        Boundary face derivatives shaped ``(n_species,)``.
    """

    left_value: np.ndarray | None
    left_grad: np.ndarray | None
    right_value: np.ndarray | None
    right_grad: np.ndarray | None


def _species_entries(entry: Any, species_names: Sequence[str], field: str, side: str, key: str) -> np.ndarray:
    """Order a config entry into one float64 value per species.

    Parameters
    ----------
    entry : Any
        Scalar, sequence, or table keyed by species name with an optional ``default`` entry.
    species_names : sequence of str
        Species names in the order the profile rows use.
    field, side, key : str
        Config location, used in the error message.

    Returns
    -------
    numpy.ndarray
        The entry as float64. A table gives an array with shape ``(n_species,)``.

    Raises
    ------
    ValueError
        If a table names neither a species nor a ``default``.
    """
    if not isinstance(entry, Mapping):
        return np.asarray(entry, dtype=np.float64)
    default = entry.get("default")
    ordered = []
    for name in species_names:
        if name in entry:
            ordered.append(entry[name])
        elif default is not None:
            ordered.append(default)
        else:
            raise ValueError(
                f"[boundary.{field}.{side}.{key}] has no entry for species {name!r} and no 'default' entry"
            )
    return np.asarray(ordered, dtype=np.float64)


def _side_type(side_cfg: Mapping[str, Any], field: str, side: str) -> str:
    """Read and validate the boundary condition type of one side.

    Parameters
    ----------
    side_cfg : mapping
        Contents of [boundary.<field>.<side>].
    field, side : str
        Config location, used in the error message.

    Returns
    -------
    str
        Lower-case type. The default is ``dirichlet``.

    Raises
    ------
    ValueError
        If the type is not one of ``SUPPORTED_BOUNDARY_TYPES``.
    """
    bc_type = str(side_cfg.get("type", "dirichlet")).strip().lower()
    if bc_type not in SUPPORTED_BOUNDARY_TYPES:
        raise ValueError(
            f"[boundary.{field}.{side}] has type {bc_type!r}, which is not implemented here. "
            f"Supported types are {', '.join(SUPPORTED_BOUNDARY_TYPES)}."
        )
    return bc_type


def boundary_model_from_config(config: Mapping[str, Any], field: str,
                               species_names: Sequence[str]) -> BoundaryModel | None:
    """Build the boundary model of one field from a parsed NEOPAX common input.

    Read [boundary.<field>.left] and [boundary.<field>.right]. Each coefficient is a scalar or a
    table keyed by species. An optional ``default`` applies to unnamed species. The Robin closure
    ignores ``value`` and calculates its boundary value from ``decay_length``.

    Parameters
    ----------
    config : mapping
        Parsed ``common_input.toml``.
    field : str
        Field name under [boundary], such as ``density`` or ``temperature``.
    species_names : sequence of str
        Species names in the order the profile rows use.

    Returns
    -------
    BoundaryModel or None
        None if the config has no [boundary.<field>] block. NEOPAX then uses a zero axis
        gradient and an extrapolated edge.

    Raises
    ------
    ValueError
        If a type is unsupported or a table has no applicable value.
    """
    boundary_cfg = config.get("boundary")
    if not isinstance(boundary_cfg, Mapping):
        return None
    field_cfg = boundary_cfg.get(field)
    if not isinstance(field_cfg, Mapping):
        return None

    coefficients: dict[str, np.ndarray | None] = {}
    types: dict[str, str] = {}
    for side in ("left", "right"):
        side_cfg = field_cfg.get(side, {})
        if not isinstance(side_cfg, Mapping):
            raise ValueError(f"[boundary.{field}.{side}] must be a table, got {type(side_cfg).__name__}")
        types[side] = _side_type(side_cfg, field, side)
        for key in ("value", "gradient", "decay_length"):
            entry = side_cfg.get(key)
            coefficients[f"{side}_{key}"] = (
                None if entry is None else _species_entries(entry, species_names, field, side, key)
            )

    logger.debug("boundary model for %r: left type %r, right type %r", field, types["left"], types["right"])
    return BoundaryModel(
        left_type=types["left"],
        right_type=types["right"],
        left_value=coefficients["left_value"],
        right_value=coefficients["right_value"],
        left_gradient=coefficients["left_gradient"],
        right_gradient=coefficients["right_gradient"],
        left_decay_length=coefficients["left_decay_length"],
        right_decay_length=coefficients["right_decay_length"],
    )


def _match_species_length(value: npt.ArrayLike, n_species: int) -> np.ndarray:
    """Resize a per-species coefficient to ``n_species`` entries.

    A scalar applies to all species. The function truncates a long array.
    It extends a short array with its last entry. NEOPAX uses the same rules.

    Parameters
    ----------
    value : array_like
        Scalar or per-species coefficient.
    n_species : int
        Number of species rows in the profile.

    Returns
    -------
    numpy.ndarray
        Coefficient shaped ``(n_species,)``.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.full((n_species,), float(arr))
    arr = arr.reshape(-1)
    if arr.shape[0] == n_species:
        return arr
    if arr.shape[0] > n_species:
        return arr[:n_species]
    return np.pad(arr, (0, n_species - arr.shape[0]), mode="edge")


def _boundary_cell_spacing(faces: np.ndarray, side: str) -> float:
    """Stencil spacing used by the boundary closures.

    Parameters
    ----------
    faces : numpy.ndarray
        Face coordinates.
    side : str
        ``left`` or ``right``.

    Returns
    -------
    float
        Distance between the two centers nearest the boundary. For fewer than four faces, use the
        boundary cell width.
    """
    if faces.shape[0] >= 4:
        centers = 0.5 * (faces[1:] + faces[:-1])
        return float(centers[-1] - centers[-2]) if side == "right" else float(centers[1] - centers[0])
    return float(faces[-1] - faces[-2]) if side == "right" else float(faces[1] - faces[0])


def _left_constraints(model: BoundaryModel, u: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Axis face value and derivative implied by the boundary model.

    Parameters
    ----------
    model : BoundaryModel
        Boundary condition types and coefficients.
    u : numpy.ndarray
        Cell-centered profile shaped ``(n_species, n_radial)``.
    faces : numpy.ndarray
        Face coordinates.

    Returns
    -------
    tuple of numpy.ndarray
        Axis face value and derivative, each shaped ``(n_species,)``.

    Raises
    ------
    ValueError
        If the model declares a left type that is not implemented.
    """
    n_species = u.shape[0]
    dx = _boundary_cell_spacing(faces, "left")
    # The axis face is half a spacing from u_ip1. It is three halves of a spacing from u_ip2.
    # These distances set the stencil weights.
    u_ip1 = u[:, 0]
    u_ip2 = u[:, 1] if u.shape[1] >= 2 else u[:, 0]

    if model.left_type == "dirichlet":
        lv = u[:, 0] if model.left_value is None else _match_species_length(model.left_value, n_species)
        return lv, (-8.0 * lv + 9.0 * u_ip1 - u_ip2) / (3.0 * dx)
    if model.left_type == "neumann":
        if model.left_gradient is None:
            lg = np.zeros(n_species)
        else:
            lg = _match_species_length(model.left_gradient, n_species)
        return (-3.0 * dx * lg + 9.0 * u_ip1 - u_ip2) / 8.0, lg
    if model.left_type == "robin":
        decay = (
            np.ones(n_species)
            if model.left_decay_length is None
            else _match_species_length(model.left_decay_length, n_species)
        )
        lv = (9.0 * u_ip1 - u_ip2) / (8.0 + 3.0 * dx / (decay + _DECAY_EPS))
        return lv, lv / (decay + _DECAY_EPS)
    raise ValueError(f"unsupported left boundary type {model.left_type!r}")


def _right_constraints(model: BoundaryModel, u: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Edge face value and derivative implied by the boundary model.

    Parameters
    ----------
    model : BoundaryModel
        Boundary condition types and coefficients.
    u : numpy.ndarray
        Cell-centered profile shaped ``(n_species, n_radial)``.
    faces : numpy.ndarray
        Face coordinates.

    Returns
    -------
    tuple of numpy.ndarray
        Edge face value and derivative, each shaped ``(n_species,)``.

    Raises
    ------
    ValueError
        If the model declares a right type that is not implemented.
    """
    n_species = u.shape[0]
    dx = _boundary_cell_spacing(faces, "right")
    # The edge face is half a spacing from u_im1. It is three halves of a spacing from u_im2.
    # These distances set the stencil weights.
    u_im1 = u[:, -1]
    u_im2 = u[:, -2] if u.shape[1] >= 2 else u[:, -1]

    if model.right_type == "dirichlet":
        rv = u[:, -1] if model.right_value is None else _match_species_length(model.right_value, n_species)
        return rv, (8.0 * rv - 9.0 * u_im1 + u_im2) / (3.0 * dx)
    if model.right_type == "neumann":
        if model.right_gradient is None:
            rg = np.zeros(n_species)
        else:
            rg = _match_species_length(model.right_gradient, n_species)
        return (3.0 * dx * rg + 9.0 * u_im1 - u_im2) / 8.0, rg
    if model.right_type == "robin":
        decay = (
            np.ones(n_species)
            if model.right_decay_length is None
            else _match_species_length(model.right_decay_length, n_species)
        )
        rv = (9.0 * u_im1 - u_im2) / (8.0 + 3.0 * dx / (decay + _DECAY_EPS))
        return rv, -rv / (decay + _DECAY_EPS)
    raise ValueError(f"unsupported right boundary type {model.right_type!r}")


def face_constraints(model: BoundaryModel | None, values_on_centers: npt.ArrayLike,
                     face_grid: npt.ArrayLike) -> FaceConstraints:
    """Convert a boundary model to constraints for ``face_gradient`` and ``face_value``.

    Without a model, NEOPAX uses a zero axis derivative. It extrapolates the edge value from the two
    outermost centers. With a model, this function returns a value and derivative for each side.
    ``face_value`` uses the value.
    ``face_gradient`` uses the derivative.

    Parameters
    ----------
    model : BoundaryModel or None
        Boundary condition types and coefficients, or None for the unconstrained closure.
    values_on_centers : array_like
        Profile on the cell centers. Its shape is ``(n_radial,)`` or
        ``(n_species, n_radial)``.
    face_grid : array_like
        The ``n_radial + 1`` face coordinates, strictly increasing.

    Returns
    -------
    FaceConstraints
        Constraints with one entry per species. A 1-D profile counts as one species.

    Raises
    ------
    ValueError
        If the profile, grid or boundary type is invalid.
    """
    u, _ = _as_2d(values_on_centers, "values_on_centers")
    faces = _as_face_grid(face_grid, u.shape[-1])

    if model is None:
        edge = 1.5 * u[:, -1] - 0.5 * u[:, -2] if u.shape[1] >= 2 else u[:, -1]
        return FaceConstraints(left_value=None, left_grad=np.zeros(u.shape[0]), right_value=edge, right_grad=None)

    left_value, left_grad = _left_constraints(model, u, faces)
    right_value, right_grad = _right_constraints(model, u, faces)
    return FaceConstraints(left_value=left_value, left_grad=left_grad, right_value=right_value, right_grad=right_grad)
