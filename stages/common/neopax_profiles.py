"""Supply the NEOPAX profile state on the transport face grid.

Stages 3 and 4 scan the ``n_radial + 1`` faces where NEOPAX interpolates flux files.
Three profile sources supply this state. The analytical source reproduces the NEOPAX profile model.
The prescribed source reads the feedback arrays in [profiles]. The transport source reads
``transport_solution.h5``. This module gives both scans the same adapter and ``FaceState`` type.

Density uses 1e20 m^-3. Temperature uses keV. ``Er`` uses kV/m.
Density and temperature gradients use the applicable profile unit per unit rho.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from common.neopax_geometry import minor_radius_from_face_grids
from common.profile_gradients import (
    BoundaryModel,
    boundary_model_from_config,
    face_constraints,
    face_gradient,
    face_value,
)

# These cell-centered datasets identify the NEOPAX ``transport_solution.h5`` layout.
_TRANSPORT_CORE_DATASETS = ("rho", "density", "temperature", "Er")

# A NEOPAX ``transport_solution.h5`` must contain these face-grid datasets.
# ``r_grid_half`` is ``rho_face`` in metres. The gradient datasets use this grid.
_TRANSPORT_FACE_DATASETS = (
    "rho_face",
    "density_faces",
    "temperature_faces",
    "Er_faces",
    "r_grid_half",
    "density_grad_faces",
    "temperature_grad_faces",
)

# These scalars contain the time reached and the next step size.
# The file also contains the aliases ``t_final`` and ``dt_next``.
# The alias ``t_final`` differs from the configured ``[transport_solver].t_final`` value.
_TRANSPORT_CLOCK_DATASETS = ("final_time", "next_dt")

NEOPAX_DENSITY_REFERENCE_M3 = 1.0e20
NEOPAX_TEMPERATURE_REFERENCE_EV = 1.0e3

# NEOPAX changes these [boundary.Er.right] types to a zero-gradient Neumann edge.
# ``_normalized_boundary_cfg_for_transport`` makes this change before face reconstruction.
_ER_AMBIPOLAR_EDGE_TYPES = ("floating_ambipolar_edge", "ambipolar_edge_root")

# ``Er`` has one row and is not species-resolved. Its boundary table does not use a species name.
_ER_ROW_NAMES = ("Er",)

# These four profile parameters have no usable default. Each tuple gives the accepted names in
# NEOPAX lookup order.
_REQUIRED_PROFILE_KEYS = (
    ("n0", "ni0", "ne0"),
    ("n_edge", "nib", "neb"),
    ("T0", "ti0", "te0"),
    ("T_edge", "tib", "teb"),
)


@dataclass(frozen=True)
class SpeciesMeta:
    """One entry of the NEOPAX [species] block.

    Attributes
    ----------
    name : str
        Species name, which the [boundary] blocks key their per-species tables by.
    charge : float
        Charge in proton charges, from ``charge_qp``.
    mass_mp : float
        Mass in proton masses, from ``mass_mp``.
    """

    name: str
    charge: float
    mass_mp: float


@dataclass(frozen=True)
class FaceState:
    """NEOPAX's profile state on the ``n_radial + 1`` transport cell faces.

    Attributes
    ----------
    rho : numpy.ndarray
        Face radii, shape ``(nr,)``, in normalized radius.
    density : numpy.ndarray
        Shape ``(ns, nr)``, in 1e20 m^-3.
    temperature : numpy.ndarray
        Shape ``(ns, nr)``, in keV.
    er : numpy.ndarray
        Shape ``(nr,)``, in kV/m.
    density_grad : numpy.ndarray
        d(density)/d(rho), shape ``(ns, nr)``.
    temperature_grad : numpy.ndarray
        d(temperature)/d(rho), shape ``(ns, nr)``.
    time_value : float or None
        Solution time of the slice, or None for a state built from config.
    """

    rho: np.ndarray
    density: np.ndarray
    temperature: np.ndarray
    er: np.ndarray
    density_grad: np.ndarray
    temperature_grad: np.ndarray
    time_value: float | None


def _electron_index(charges: Sequence[float], n_species: int) -> int | None:
    """Return the index NEOPAX treats as the electron, or None when it recognises no electron.

    NEOPAX selects the first species with the lowest charge. This charge must be negative.
    A short charge list makes the electron unknown.

    Parameters
    ----------
    charges : sequence of float
        Charges in proton charges, in the row order of the profile arrays.
    n_species : int
        Number of species the run expects.

    Returns
    -------
    int or None
        Electron row index, or None when no electron is recognised.
    """
    values = [float(value) for value in charges]
    if len(values) < n_species:
        return None
    index = min(range(n_species), key=lambda i: values[i])
    return index if values[index] < 0.0 else None


def _effective_charges(profile_cfg: dict[str, Any], species: Sequence[SpeciesMeta]) -> list[float]:
    """Return the charges that the analytical model uses to locate the electron.

    ``_build_state`` copies [species].charge_qp into the profile block when necessary.
    This operation occurs before profile evaluation.
    Therefore, [profiles].charge_qp has priority. [species].charge_qp is the fallback.

    Parameters
    ----------
    profile_cfg : dict[str, Any]
        The parsed [profiles] section.
    species : sequence of SpeciesMeta
        Species in the row order of the profile arrays, from [species].

    Returns
    -------
    list of float
        Charges in proton charges, in the row order of the profile arrays.
    """
    override = profile_cfg.get("charge_qp")
    if override is None:
        return [float(entry.charge) for entry in species]
    return [float(value) for value in override]


def _match_species_factors(
    raw: Any, n_species: int, *, default: float, electron_index: int | None = None
) -> np.ndarray:
    """Expand one [profiles] factor with the NEOPAX rules.

    A missing value uses the default. A scalar applies to all species.
    A full array passes through unchanged. ``electron_index`` also permits an array with
    ``n_species - 1`` entries. The function inserts ``1.0`` at the electron position.
    NEOPAX permits this short form only for ``c_density`` and ``n_scale``.

    A sequence with one entry does not act as a scalar. ``_expand_species_values`` rejects it when
    its length is incorrect.

    Parameters
    ----------
    raw : Any
        The config value, a scalar, a sequence, or None.
    n_species : int
        Number of species the run expects.
    default : float
        Value used when ``raw`` is None.
    electron_index : int or None
        Electron row, or None to require the full length.

    Returns
    -------
    numpy.ndarray
        Shape ``(n_species,)``.

    Raises
    ------
    ValueError
        If ``raw`` is a sequence of a length this key does not accept.
    """
    if raw is None:
        return np.full((n_species,), float(default), dtype=np.float64)
    if isinstance(raw, (int, float)):
        return np.full((n_species,), float(raw), dtype=np.float64)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 0:
        return np.full((n_species,), float(arr), dtype=np.float64)
    if arr.size == n_species:
        return arr
    if electron_index is not None and arr.size == n_species - 1:
        expanded = np.ones((n_species,), dtype=np.float64)
        expanded[[i for i in range(n_species) if i != electron_index]] = arr
        return expanded
    accepted = f"{n_species}" if electron_index is None else f"{n_species} or {n_species - 1}"
    raise ValueError(f"Expected {accepted} profile factors, got {arr.size}")


def _required_profile_value(profile_cfg: dict[str, Any], aliases: Sequence[str]) -> Any:
    """Read one of the profile parameters that carries no usable default.

    Parameters
    ----------
    profile_cfg : dict[str, Any]
        The parsed [profiles] section.
    aliases : sequence of str
        Spellings NEOPAX accepts for the parameter, in its own lookup order.

    Returns
    -------
    Any
        The config value under the first alias the block sets.

    Raises
    ------
    ValueError
        If the block sets none of the aliases.
    """
    for key in aliases:
        if key in profile_cfg:
            return profile_cfg[key]
    raise ValueError(
        f"[profiles].{aliases[0]} is required. NEOPAX's fallback for it is already in SI and the model "
        f"then applies the 1e20 m^-3 and keV scaling on top, so an omitted value builds a profile 1e20 "
        f"(density) or 1e3 (temperature) times the one meant. Accepted spellings are {', '.join(aliases)}."
    )


def _analytical_density_temperature(
    profile_cfg: dict[str, Any],
    rho: np.ndarray,
    *,
    rho_edge: float,
    charges: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the analytical density and temperature model at the given radii.

    Parameters
    ----------
    profile_cfg : dict[str, Any]
        The parsed [profiles] section.
    rho : numpy.ndarray
        Radii to evaluate at, in normalized radius.
    rho_edge : float
        Normalizer of the shape functions. ``rho[-1]`` is only the outermost sample.
        It is not the plasma edge when ``rho`` contains cell centers.
    charges : sequence of float
        Effective charges in proton charges, one per species, already resolved by
        :func:`_effective_charges`. Their length is the species count.

    Returns
    -------
    tuple of numpy.ndarray
        ``(density, temperature)``, each shaped ``(n_species, rho.size)``, in 1e20 m^-3 and keV.

    Raises
    ------
    ValueError
        If a profile factor has an invalid number of entries.
        If ``n0``, ``n_edge``, ``T0`` or ``T_edge`` is absent.
    """
    rho = np.asarray(rho, dtype=np.float64)
    x = rho / max(float(rho_edge), 1.0e-30)
    n_species = len(charges)
    electron_index = _electron_index(charges, n_species)

    n0_raw, n_edge_raw, t0_raw, t_edge_raw = (
        _required_profile_value(profile_cfg, aliases) for aliases in _REQUIRED_PROFILE_KEYS
    )
    n0 = _match_species_factors(n0_raw, n_species, default=0.0)
    n_edge = _match_species_factors(n_edge_raw, n_species, default=0.0)
    t0 = _match_species_factors(t0_raw, n_species, default=0.0)
    t_edge = _match_species_factors(t_edge_raw, n_species, default=0.0)
    density_shape_power = _match_species_factors(profile_cfg.get("density_shape_power", 2.0), n_species, default=2.0)
    density_shape_alpha = _match_species_factors(profile_cfg.get("density_shape_alpha", 1.0), n_species, default=1.0)
    temperature_shape_power = _match_species_factors(profile_cfg.get("temperature_shape_power", 2.0), n_species, default=2.0)
    temperature_shape_alpha = _match_species_factors(profile_cfg.get("temperature_shape_alpha", 1.0), n_species, default=1.0)

    c_density = profile_cfg.get("c_density")
    if c_density is None and n_species == 3:
        deuterium_ratio = float(profile_cfg.get("deuterium_ratio", 0.5))
        tritium_ratio = float(profile_cfg.get("tritium_ratio", 0.5))
        c_density = [1.0, deuterium_ratio, tritium_ratio]

    density_species_scale = _match_species_factors(
        c_density, n_species, default=1.0, electron_index=electron_index
    )
    temperature_species_scale = _match_species_factors(profile_cfg.get("c_temperature"), n_species, default=1.0)
    density_global_scale = _match_species_factors(
        profile_cfg.get("n_scale", 1.0), n_species, default=1.0, electron_index=electron_index
    )
    temperature_global_scale = _match_species_factors(profile_cfg.get("T_scale", 1.0), n_species, default=1.0)

    density = np.zeros((n_species, rho.size), dtype=np.float64)
    temperature = np.zeros((n_species, rho.size), dtype=np.float64)
    for i in range(n_species):
        density_shape = (1.0 - x**density_shape_power[i]) ** density_shape_alpha[i]
        temperature_shape = (1.0 - x**temperature_shape_power[i]) ** temperature_shape_alpha[i]
        density_base = n_edge[i] + (n0[i] - n_edge[i]) * density_shape
        temperature_base = t_edge[i] + (t0[i] - t_edge[i]) * temperature_shape
        density[i, :] = density_species_scale[i] * density_global_scale[i] * density_base
        temperature[i, :] = temperature_species_scale[i] * temperature_global_scale[i] * temperature_base

    return density, temperature


def _analytical_er(profile_cfg: dict[str, Any], rho: np.ndarray, *, rho_edge: float) -> np.ndarray:
    """Evaluate the analytical radial electric field at the given radii.

    Parameters
    ----------
    profile_cfg : dict[str, Any]
        The parsed [profiles] section.
    rho : numpy.ndarray
        Radii to evaluate at, in normalized radius.
    rho_edge : float
        Normalizer of the parabola, as in ``_analytical_density_temperature``.

    Returns
    -------
    numpy.ndarray
        ``Er`` shaped ``(rho.size,)``, in kV/m.
    """
    x = np.asarray(rho, dtype=np.float64) / max(float(rho_edge), 1.0e-30)
    er0_scale = float(profile_cfg.get("er0_scale", 100.0))
    er0_peak_rho = float(profile_cfg.get("er0_peak_rho", 0.8))
    return er0_scale * x * (er0_peak_rho - x)


def _er_boundary_model(cfg: dict[str, Any]) -> BoundaryModel | None:
    """Build the ``Er`` boundary model after the NEOPAX ambipolar edge conversion.

    ``floating_ambipolar_edge`` and ``ambipolar_edge_root`` select a solve, not a boundary closure.
    NEOPAX does not send these types to its face operator.
    ``_normalized_boundary_cfg_for_transport`` converts them first. It uses a zero-gradient
    Neumann edge. This function uses the same conversion for the analytical profile source.

    Parameters
    ----------
    cfg : dict[str, Any]
        Parsed common-input TOML, read for [boundary.Er].

    Returns
    -------
    BoundaryModel or None
        None when the config has no [boundary.Er] block.

    Raises
    ------
    ValueError
        If a side declares a type that is neither an ambipolar edge nor one the operator implements.
    """
    boundary_cfg = cfg.get("boundary")
    er_cfg = boundary_cfg.get("Er") if isinstance(boundary_cfg, dict) else None
    if not isinstance(er_cfg, dict):
        return None
    right_cfg = er_cfg.get("right")
    if isinstance(right_cfg, dict) and str(right_cfg.get("type", "")).strip().lower() in _ER_AMBIPOLAR_EDGE_TYPES:
        right_cfg = {**right_cfg, "type": "neumann", "gradient": right_cfg.get("gradient", 0.0)}
        er_cfg = {**er_cfg, "right": right_cfg}
    return boundary_model_from_config({"boundary": {"Er": er_cfg}}, "Er", _ER_ROW_NAMES)


def _neopax_face_state(
    values_on_centers: np.ndarray,
    *,
    boundary: BoundaryModel | None,
    rho_faces: np.ndarray,
    minor_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build NEOPAX face values and per-rho gradients from a centered profile.

    The operator uses the metre grid ``minor_radius * rho_faces``. Boundary coefficients therefore
    keep their configured units. The operator returns a derivative per metre. Multiplication by the
    same radius converts it to a derivative per unit rho.

    Parameters
    ----------
    values_on_centers : numpy.ndarray
        Profile on the ``n_radial`` cell centers, shaped ``(n_species, n_radial)`` for a
        species-resolved field or ``(n_radial,)`` for ``Er``.
    boundary : BoundaryModel or None
        Boundary model of this field, or None for NEOPAX's unconstrained closure.
    rho_faces : numpy.ndarray
        The ``n_radial + 1`` faces, in normalized radius.
    minor_radius : float
        NEOPAX's minor radius, in metres per unit rho.

    Returns
    -------
    tuple of numpy.ndarray
        ``(values_face, grad_face)`` with ``n_radial + 1`` radial entries.
        The second array uses the profile unit per unit rho.
    """
    face_grid_m = float(minor_radius) * np.asarray(rho_faces, dtype=np.float64)
    constraints = face_constraints(boundary, values_on_centers, face_grid_m)
    values_face = face_value(values_on_centers, face_grid_m, **constraints._asdict())
    grad_face = face_gradient(values_on_centers, face_grid_m, **constraints._asdict())
    return values_face, grad_face * float(minor_radius)


def build_analytical_face_state(
    cfg: dict[str, Any],
    *,
    species: Sequence[SpeciesMeta],
    n_faces: int,
    minor_radius: float,
) -> FaceState:
    """Evaluate the analytical model and reconstruct its faces with the NEOPAX method.

    NEOPAX stores its state on ``n_faces - 1`` cell centers. It applies the run's [boundary]
    blocks during face reconstruction. This function uses the same centers and reconstruction.
    Thus, the first iteration supplies the same face state that later solutions contain.
    The ``Er`` reconstruction also uses this path. It first applies the NEOPAX ambipolar edge
    conversion to [boundary.Er].

    Parameters
    ----------
    cfg : dict[str, Any]
        Parsed common-input TOML, read for [profiles], [geometry] and [boundary].
    species : sequence of SpeciesMeta
        Species in profile row order, from [species]. The names select boundary table entries.
        The charges locate the electron if [profiles].charge_qp does not override them.
    n_faces : int
        Number of cell faces to build. A transport grid has ``n_radial + 1`` faces.
    minor_radius : float
        NEOPAX's minor radius, in metres per unit rho.

    Returns
    -------
    FaceState
        Face profiles and per-rho face gradients on ``linspace(0, rho_edge, n_faces)``.

    Raises
    ------
    ValueError
        If ``n_faces`` gives too few cells for the edge closure.
        If ``n0``, ``n_edge``, ``T0`` or ``T_edge`` is absent.
        If a boundary block uses an unsupported type.
    """
    # The edge closure uses the two outermost centers. Therefore, both profile sources require at
    # least two cells.
    if int(n_faces) < 3:
        raise ValueError(
            f"n_faces is {int(n_faces)}; the analytical profiles are sampled on the cells the faces "
            "bound and the outer boundary closure reaches two of them, so at least 3 faces are needed"
        )
    species_names = [entry.name for entry in species]
    profile_cfg = cfg.get("profiles", {})
    rho_edge = float(cfg.get("geometry", {}).get("rho_edge", 1.0))
    rho_faces = np.linspace(0.0, rho_edge, int(n_faces), dtype=np.float64)
    rho_centers = 0.5 * (rho_faces[:-1] + rho_faces[1:])

    density_centers, temperature_centers = _analytical_density_temperature(
        profile_cfg,
        rho_centers,
        rho_edge=rho_edge,
        charges=_effective_charges(profile_cfg, species),
    )
    density_face, density_grad = _neopax_face_state(
        density_centers,
        boundary=boundary_model_from_config(cfg, "density", species_names),
        rho_faces=rho_faces,
        minor_radius=minor_radius,
    )
    temperature_face, temperature_grad = _neopax_face_state(
        temperature_centers,
        boundary=boundary_model_from_config(cfg, "temperature", species_names),
        rho_faces=rho_faces,
        minor_radius=minor_radius,
    )
    er_face, _ = _neopax_face_state(
        _analytical_er(profile_cfg, rho_centers, rho_edge=rho_edge),
        boundary=_er_boundary_model(cfg),
        rho_faces=rho_faces,
        minor_radius=minor_radius,
    )
    return FaceState(
        rho=rho_faces,
        density=density_face,
        temperature=temperature_face,
        er=er_face,
        density_grad=density_grad,
        temperature_grad=temperature_grad,
        time_value=None,
    )


def _require_profile_array(profile_cfg: dict[str, Any], key: str, *, ndim: int) -> np.ndarray:
    """Read a required [profiles] array as float64 and check its dimensionality.

    Parameters
    ----------
    profile_cfg : dict[str, Any]
        The parsed [profiles] section.
    key : str
        Array name to read.
    ndim : int
        Dimensionality the array must have.

    Returns
    -------
    numpy.ndarray
        The array, as float64.

    Raises
    ------
    ValueError
        If the key is absent or the array has a different dimensionality.
    """
    if key not in profile_cfg:
        raise ValueError(f"[profiles].{key} is required when --profiles-source=prescribed")
    array = np.asarray(profile_cfg[key], dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"[profiles].{key} must be {ndim}-D, got shape {array.shape}")
    return array


def _require_face_arrays(
    profile_cfg: dict[str, Any], *, n_species: int, n_radial: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Read and shape-check the [profiles] face arrays against ``n_radial`` cells.

    Parameters
    ----------
    profile_cfg : dict[str, Any]
        The parsed [profiles] section.
    n_species : int
        Number of species the run expects.
    n_radial : int
        Number of transport cells, so the arrays must carry ``n_radial + 1`` radial entries.

    Returns
    -------
    tuple of numpy.ndarray
        ``density_face``, ``temperature_face``, ``er_face``, ``density_grad_face`` and
        ``temperature_grad_face``. Values use SI units. Gradients use units per rho.

    Raises
    ------
    ValueError
        If an array is absent or does not span the ``n_radial + 1`` cell faces.
    """
    n_faces = n_radial + 1
    density_face = _require_profile_array(profile_cfg, "density_face", ndim=2)
    temperature_face = _require_profile_array(profile_cfg, "temperature_face", ndim=2)
    er_face = _require_profile_array(profile_cfg, "Er_face", ndim=1)
    density_grad_face = _require_profile_array(profile_cfg, "density_grad_face", ndim=2)
    temperature_grad_face = _require_profile_array(profile_cfg, "temperature_grad_face", ndim=2)
    for key, array in (
        ("density_face", density_face),
        ("temperature_face", temperature_face),
        ("density_grad_face", density_grad_face),
        ("temperature_grad_face", temperature_grad_face),
    ):
        if array.shape != (n_species, n_faces):
            raise ValueError(
                f"[profiles].{key} has shape {array.shape}, expected {(n_species, n_faces)}: one row per "
                f"species and one column per cell face, which is [geometry].n_radial + 1"
            )
    if er_face.size != n_faces:
        raise ValueError(
            f"[profiles].Er_face holds {er_face.size} points, expected {n_faces} to span the same cell "
            "faces as [profiles].density_face"
        )
    return density_face, temperature_face, er_face, density_grad_face, temperature_grad_face


def build_prescribed_face_state(cfg: dict[str, Any], *, n_species: int) -> FaceState:
    """Build a face state from the prescribed arrays in [profiles].

    The block contains both NEOPAX radial grids. ``density``, ``temperature`` and ``Er`` contain
    one value per cell center. NEOPAX reads these arrays, so this function validates them.
    The scans do not use them. The three ``*_face`` arrays contain one value per cell face.
    The two ``*_grad_face`` arrays also use the cell faces. These five arrays build the face state.
    The scans use this grid because NEOPAX interpolates flux files on it.

    This function does not reconstruct face values or gradients from centered values. NEOPAX builds
    them with the run's boundary models. Missing face arrays therefore cause an error.
    Local extrapolation could disagree with the solver at a non-Dirichlet boundary. The quick run
    has such a boundary at its Robin temperature edge.

    The prescribed density uses m^-3, and temperature uses eV. Their gradients use the same units
    per unit rho. Division by the NEOPAX references gives 1e20 m^-3 and keV.
    ``Er`` uses kV/m in both representations. The arrays contain no radial coordinate.
    Therefore, this function rebuilds the grid from [geometry], as the transport solver does.

    Parameters
    ----------
    cfg : dict[str, Any]
        Parsed common-input TOML.
    n_species : int
        Number of species the run expects, taken from [species].names.

    Returns
    -------
    FaceState
        Face profiles on the reconstructed face grid, in face state units.

    Raises
    ------
    ValueError
        If [profiles].model is not ``prescribed`` or an array is absent.
        If the arrays disagree with each other or with [geometry].n_radial.
    """
    profile_cfg = cfg.get("profiles", {})
    model = str(profile_cfg.get("model", "")).strip().lower()
    if model != "prescribed":
        raise ValueError(f"[profiles].model is {model!r}; --profiles-source=prescribed requires 'prescribed'")
    density = _require_profile_array(profile_cfg, "density", ndim=2)
    temperature = _require_profile_array(profile_cfg, "temperature", ndim=2)
    er = _require_profile_array(profile_cfg, "Er", ndim=1)
    if temperature.shape != density.shape:
        raise ValueError(
            f"[profiles].temperature has shape {temperature.shape}, expected the [profiles].density "
            f"shape {density.shape}"
        )
    if density.shape[0] != n_species:
        raise ValueError(
            f"[profiles].density holds {density.shape[0]} species rows, expected {n_species} to match "
            "the number of entries in [species].names"
        )
    n_radial = int(density.shape[1])
    if er.size != n_radial:
        raise ValueError(f"[profiles].Er holds {er.size} points, expected {n_radial} to match [profiles].density")
    geometry_cfg = cfg.get("geometry", {})
    if "n_radial" not in geometry_cfg:
        raise ValueError("[geometry].n_radial is required when --profiles-source=prescribed")
    if n_radial != int(geometry_cfg["n_radial"]):
        raise ValueError(
            f"[profiles] arrays hold {n_radial} radial points, expected {int(geometry_cfg['n_radial'])} "
            "to match [geometry].n_radial"
        )
    # NEOPAX extrapolates the outer boundary from the two outermost centers.
    # Therefore, the minimum is two cells and three faces.
    if n_radial + 1 < 3:
        raise ValueError(
            f"[profiles] arrays hold {n_radial} cells, giving {n_radial + 1} faces; the outer boundary "
            "closure needs at least 3 faces"
        )

    density_face, temperature_face, er_face, density_grad_face, temperature_grad_face = _require_face_arrays(
        profile_cfg, n_species=n_species, n_radial=n_radial
    )
    rho_edge = float(geometry_cfg.get("rho_edge", 1.0))
    return FaceState(
        rho=np.linspace(0.0, rho_edge, n_radial + 1, dtype=np.float64),
        density=density_face / NEOPAX_DENSITY_REFERENCE_M3,
        temperature=temperature_face / NEOPAX_TEMPERATURE_REFERENCE_EV,
        er=er_face,
        density_grad=density_grad_face / NEOPAX_DENSITY_REFERENCE_M3,
        temperature_grad=temperature_grad_face / NEOPAX_TEMPERATURE_REFERENCE_EV,
        time_value=None,
    )


@dataclass(frozen=True)
class TransportSolution:
    """One NEOPAX solution slice on both radial grids, with the run clock.

    Attributes
    ----------
    rho : numpy.ndarray
        Cell centers, shape ``(n_radial,)``. Spans neither ``rho = 0`` nor ``rho = rho_edge``.
    density, temperature : numpy.ndarray
        Cell-centered profiles, shape ``(ns, n_radial)``, in 1e20 m^-3 and keV.
    er : numpy.ndarray
        Cell-centered radial electric field, shape ``(n_radial,)``, in kV/m.
    rho_face : numpy.ndarray
        Cell faces, shape ``(n_radial + 1,)``, spanning ``[0, rho_edge]`` inclusive.
    density_face, temperature_face, er_face : numpy.ndarray
        The same three fields on the faces, in the same units.
    density_grad_face, temperature_grad_face : numpy.ndarray
        Face gradients with shape ``(ns, n_radial + 1)``. The values use profile units per unit rho.
    minor_radius : float
        Metres per unit rho relating the file's two face grids.
    time_value : float or None
        Solution time of the slice, or None when the file records no ``ts`` axis.
    final_time, next_dt : float
        Run clock in seconds. These values give the time reached and the next step size.
    """

    rho: np.ndarray
    density: np.ndarray
    temperature: np.ndarray
    er: np.ndarray
    rho_face: np.ndarray
    density_face: np.ndarray
    temperature_face: np.ndarray
    er_face: np.ndarray
    density_grad_face: np.ndarray
    temperature_grad_face: np.ndarray
    minor_radius: float
    time_value: float | None
    final_time: float
    next_dt: float


def _check_slice_ranks(named: dict[str, np.ndarray], *, h5_path: Path, grid: str) -> int:
    """Check one grid layout and return the density rank.

    Parameters
    ----------
    named : dict[str, numpy.ndarray]
        The three datasets of one grid, keyed by their names in the file, density first.
    h5_path : Path
        File the datasets came from, quoted in the error message.
    grid : str
        ``center`` or ``face`` for the error message.

    Returns
    -------
    int
        Rank of the density dataset, 2 for a static solution and 3 for a time-resolved one.

    Raises
    ------
    ValueError
        If the density rank is not 2 or 3. Also if temperature or ``Er`` has an incompatible rank.
        ``Er`` has no species axis, so its rank is one less than the density rank.
    """
    (density_name, density), (temperature_name, temperature), (er_name, er) = named.items()
    if density.ndim not in (2, 3):
        raise ValueError(
            f"{h5_path} dataset '{density_name}' must be 2D (n_species, n_rho) or 3D "
            f"(n_time, n_species, n_rho), got shape {density.shape}"
        )
    if temperature.ndim != density.ndim or er.ndim != density.ndim - 1:
        raise ValueError(
            f"{h5_path} has inconsistent {grid} dataset ranks: {density_name} {density.shape}, "
            f"{temperature_name} {temperature.shape}, {er_name} {er.shape}"
        )
    return int(density.ndim)


def _check_slice_shapes(
    centered: dict[str, np.ndarray],
    faced: dict[str, np.ndarray],
    *,
    rho: np.ndarray,
    rho_face: np.ndarray,
    h5_path: Path,
) -> None:
    """Check each profile against its grid and check the staggered layout.

    Rank checks alone accept arrays with different species counts or radial lengths. Downstream
    indexing would then truncate or misalign data silently.

    Parameters
    ----------
    centered : dict[str, numpy.ndarray]
        The three cell-centered datasets, keyed by their names in the file, density first.
    faced : dict[str, numpy.ndarray]
        The three face datasets, keyed likewise.
    rho : numpy.ndarray
        Cell centers.
    rho_face : numpy.ndarray
        Cell faces.
    h5_path : Path
        File the datasets came from, quoted in the error messages.

    Raises
    ------
    ValueError
        If a grid is not 1-D or the number of faces is incorrect.
        If density, temperature or ``Er`` has an incompatible shape.
        If a radial axis does not match its grid, or the species counts differ between grids.
    """
    for grid_name, grid in (("rho", rho), ("rho_face", rho_face)):
        if grid.ndim != 1:
            raise ValueError(f"{h5_path} dataset '{grid_name}' must be 1D, got shape {grid.shape}")
    if rho_face.size != rho.size + 1:
        raise ValueError(
            f"{h5_path} carries {rho_face.size} faces for {rho.size} cell centers; the faces bound "
            "the cells, so there is one more face than centers"
        )
    for named, grid_name, grid in ((centered, "rho", rho), (faced, "rho_face", rho_face)):
        (density_name, density), (temperature_name, temperature), (er_name, er) = named.items()
        if temperature.shape != density.shape:
            raise ValueError(
                f"{h5_path} dataset '{temperature_name}' has shape {temperature.shape}, which does "
                f"not match '{density_name}' {density.shape}"
            )
        if density.shape[-1] != grid.size:
            raise ValueError(
                f"{h5_path} dataset '{density_name}' has trailing axis {density.shape[-1]}, which is "
                f"not the {grid.size}-point '{grid_name}' grid"
            )
        if er.shape != density.shape[:-2] + (grid.size,):
            raise ValueError(
                f"{h5_path} dataset '{er_name}' has shape {er.shape}, expected "
                f"{density.shape[:-2] + (grid.size,)} against '{density_name}' {density.shape}"
            )
    if centered["density"].shape[-2] != faced["density_faces"].shape[-2]:
        raise ValueError(
            f"{h5_path} centered and face states disagree in species count: density "
            f"{centered['density'].shape}, density_faces {faced['density_faces'].shape}"
        )


def _resolve_time_index(n_time: int | None, time_index: int, *, h5_path: Path) -> int:
    """Resolve a possibly negative time index against a solution's time axis.

    Parameters
    ----------
    n_time : int or None
        Length of the time axis, or None for a static solution.
    time_index : int
        Requested slice, counting back from the last when negative.
    h5_path : Path
        File being read, quoted in the error messages.

    Returns
    -------
    int
        Index into the time axis, or 0 for a static solution.

    Raises
    ------
    ValueError
        If the solution is static and ``time_index`` is not ``-1``.
    IndexError
        If ``time_index`` falls outside the time axis.
    """
    if n_time is None:
        if int(time_index) != -1:
            raise ValueError(
                f"{h5_path} stores a single static profile slice with no time axis, so --time-index "
                f"{time_index} cannot be selected"
            )
        return 0
    idx = int(time_index)
    if idx < 0:
        idx += n_time
    if idx < 0 or idx >= n_time:
        raise IndexError(f"time index {time_index} out of range for {h5_path}")
    return idx


def read_transport_solution(h5_path: Path, *, time_index: int = -1) -> TransportSolution:
    """Read and validate one NEOPAX solution slice on both radial grids.

    This function is the only parser for the transport solution layout. Other transport readers
    select data from its result. Therefore, each check here applies to every consumer.

    NEOPAX writes the evolved state on ``n_radial`` cell centers. It reconstructs ``n_radial + 1``
    faces with the run's [boundary] models. The ``*_faces`` datasets contain this state.

    The file also contains face gradients. Differentiation of the face values does not reproduce
    them. The file stores gradients per metre. This function converts them to gradients per rho.

    The file must also contain the run clock. ``next_dt`` must be positive.
    The last saved slice must have the time ``final_time``. An early stop leaves empty trailing
    slots.
    In that case, this function rejects all slices in the file.

    Parameters
    ----------
    h5_path : Path
        Transport solution to read.
    time_index : int
        Time slice to take, counting back from the last when negative.

    Returns
    -------
    TransportSolution
        The selected slice on both grids, with per-rho face gradients.

    Raises
    ------
    KeyError
        If the file does not contain the transport layout.
        If a required face or clock dataset is absent.
    ValueError
        If ranks, shapes or time axes disagree.
        If a static file uses a time index other than ``-1``.
        If one scale factor does not relate the two face grids.
        If the selected slice or clock cannot seed the next iteration.
    IndexError
        If ``time_index`` falls outside the file's time axis.
    """
    with h5py.File(h5_path, "r") as f:
        keys = set(f.keys())
        missing_core = [name for name in _TRANSPORT_CORE_DATASETS if name not in keys]
        if missing_core:
            raise KeyError(f"{h5_path} is missing required NEOPAX transport datasets: {', '.join(missing_core)}.")
        missing = [name for name in _TRANSPORT_FACE_DATASETS if name not in keys]
        if missing:
            raise KeyError(
                f"{h5_path} is missing the transport face datasets: {', '.join(missing)}. They are written "
                "only by NEOPAX revisions that solve on the staggered cell/face grid and export the gradients "
                "they build there, and the cell-centered datasets cannot stand in: they span neither rho = 0 "
                "nor rho = rho_edge, so a flux file built from them leaves both ends of NEOPAX's own grid "
                "outside its knots."
            )
        missing_clock = [name for name in _TRANSPORT_CLOCK_DATASETS if name not in keys]
        if missing_clock:
            raise KeyError(
                f"{h5_path} is missing the transport clock datasets: {', '.join(missing_clock)}. They are "
                "written only by NEOPAX revisions that export the clock the next iteration resumes from, so "
                "an older solution cannot seed that iteration."
            )
        data = {
            name: np.asarray(f[name][()], dtype=np.float64)
            for name in (*_TRANSPORT_CORE_DATASETS, *_TRANSPORT_FACE_DATASETS, *_TRANSPORT_CLOCK_DATASETS)
        }
        ts = np.asarray(f["ts"][()], dtype=np.float64) if "ts" in keys else None

    rho, rho_face, r_grid_half = data["rho"], data["rho_face"], data["r_grid_half"]
    centered = {name: data[name] for name in ("density", "temperature", "Er")}
    faced = {name: data[name] for name in ("density_faces", "temperature_faces", "Er_faces")}
    gradients = {name: data[name] for name in ("density_grad_faces", "temperature_grad_faces")}

    center_rank = _check_slice_ranks(centered, h5_path=h5_path, grid="center")
    face_rank = _check_slice_ranks(faced, h5_path=h5_path, grid="face")
    if center_rank != face_rank:
        raise ValueError(
            f"{h5_path} center and face profiles disagree in rank: density {centered['density'].shape}, "
            f"density_faces {faced['density_faces'].shape}"
        )
    _check_slice_shapes(centered, faced, rho=rho, rho_face=rho_face, h5_path=h5_path)
    # Each face gradient matches its source profile. ``r_grid_half`` matches ``rho_face``.
    # Compare complete shapes to enforce both requirements.
    for name, arr, source_name, source in (
        ("density_grad_faces", gradients["density_grad_faces"], "density_faces", faced["density_faces"]),
        ("temperature_grad_faces", gradients["temperature_grad_faces"], "temperature_faces",
         faced["temperature_faces"]),
        ("r_grid_half", r_grid_half, "rho_face", rho_face),
    ):
        if arr.shape != source.shape:
            raise ValueError(
                f"{h5_path} dataset '{name}' has shape {arr.shape}, which does not match "
                f"'{source_name}' {source.shape}"
            )

    # A static solution has shape ``(n_species, n_faces)``.
    # A time-resolved solution has shape ``(n_time, n_species, n_faces)``.
    # Incorrect time indexing of a static file returns one species row and a scalar ``Er``.
    n_time = None if center_rank == 2 else int(centered["density"].shape[0])
    if n_time is not None and any(arr.shape[0] != n_time for arr in (*centered.values(), *faced.values())):
        raise ValueError(
            f"{h5_path} time axes disagree: density {centered['density'].shape}, "
            f"temperature {centered['temperature'].shape}, Er {centered['Er'].shape}, "
            f"density_faces {faced['density_faces'].shape}"
        )
    idx = _resolve_time_index(n_time, time_index, h5_path=h5_path)
    time_value = _slice_time(ts, idx, n_time=n_time, h5_path=h5_path)

    def take(arr: np.ndarray) -> np.ndarray:
        """Select the chosen slice, which is the whole array on a static solution."""
        return arr if n_time is None else np.asarray(arr[idx], dtype=np.float64)

    minor_radius = minor_radius_from_face_grids(r_grid_half, rho_face, source=str(h5_path))
    solution = TransportSolution(
        rho=rho,
        density=take(centered["density"]),
        temperature=take(centered["temperature"]),
        er=take(centered["Er"]),
        rho_face=rho_face,
        density_face=take(faced["density_faces"]),
        temperature_face=take(faced["temperature_faces"]),
        er_face=take(faced["Er_faces"]),
        density_grad_face=take(gradients["density_grad_faces"]) * minor_radius,
        temperature_grad_face=take(gradients["temperature_grad_faces"]) * minor_radius,
        minor_radius=minor_radius,
        time_value=time_value,
        final_time=float(data["final_time"]),
        next_dt=float(data["next_dt"]),
    )
    _check_finite_slice(solution, h5_path=h5_path)
    _check_clock(solution, last_saved_time=None if ts is None else float(ts[-1]), h5_path=h5_path)
    return solution


def _slice_time(ts: np.ndarray | None, idx: int, *, n_time: int | None, h5_path: Path) -> float | None:
    """Date the selected slice from the file's time axis.

    Parameters
    ----------
    ts : numpy.ndarray or None
        The ``ts`` dataset, or None when the file carries none.
    idx : int
        Resolved index of the selected slice.
    n_time : int or None
        Length of the profile time axis, or None for a static solution.
    h5_path : Path
        File being read, quoted in the error message.

    Returns
    -------
    float or None
        Time of the slice, or None when the file carries no ``ts``.

    Raises
    ------
    ValueError
        If ``ts`` is not 1-D or its length does not match the profiles.
        A static solution must have exactly one timestamp.
    """
    if ts is None:
        return None
    expected = 1 if n_time is None else n_time
    if ts.ndim != 1:
        raise ValueError(f"{h5_path} dataset 'ts' must be 1D (n_time,), got shape {ts.shape}")
    if ts.shape[0] != expected:
        layout = "a single static profile slice" if n_time is None else f"the {expected}-slice time axis"
        raise ValueError(f"{h5_path} dataset 'ts' holds {ts.shape[0]} timestamps, expected {expected} for {layout}")
    return float(ts[idx])


def _check_finite_slice(solution: TransportSolution, *, h5_path: Path) -> None:
    """Reject a selected slice holding non-finite values.

    Parameters
    ----------
    solution : TransportSolution
        The assembled slice.
    h5_path : Path
        File it came from, quoted in the error message.

    Raises
    ------
    ValueError
        If any profile, grid or clock entry of the slice is not finite.
    """
    for name in (
        "rho", "density", "temperature", "er", "rho_face", "density_face", "temperature_face",
        "er_face",
        "density_grad_face", "temperature_grad_face", "final_time", "next_dt",
    ):
        value = getattr(solution, name)
        if value is not None and not np.all(np.isfinite(value)):
            raise ValueError(f"{h5_path} dataset '{name}' holds non-finite values in the selected slice")


def _check_clock(solution: TransportSolution, *, last_saved_time: float | None, h5_path: Path) -> None:
    """Check that a solution's clock describes the saved slices.

    Parameters
    ----------
    solution : TransportSolution
        The assembled slice.
    last_saved_time : float or None
        Final entry of the file's ``ts`` axis, or None when the file carries none.
    h5_path : Path
        File it came from, quoted in the error messages.

    Raises
    ------
    ValueError
        If ``next_dt`` is not positive, or the last saved slice is dated before the time the run
        reached.
    """
    if solution.next_dt <= 0.0:
        raise ValueError(
            f"{h5_path} dataset 'next_dt' is {solution.next_dt!r}; it becomes the next iteration's step size, "
            "which the solver would not advance on"
        )
    # NEOPAX fills a save slot only after the clock passes its time.
    # An early stop leaves empty trailing slots.
    # The last saved time then differs from ``final_time``.
    # No slice in such a file is reliable.
    if last_saved_time is not None and not math.isclose(
        last_saved_time, solution.final_time, rel_tol=1e-9, abs_tol=0.0
    ):
        raise ValueError(
            f"{h5_path} dates its last saved slice at ts = {last_saved_time!r} while 'final_time' is "
            f"{solution.final_time!r}; the run and its save grid describe different spans"
        )


def read_transport_face_state(h5_path: Path, *, time_index: int) -> FaceState:
    """Read one NEOPAX solution slice on its face grid.

    NEOPAX evolves its state on ``n_radial`` cell centers. These centers exclude ``rho = 0`` and
    ``rho = rho_edge``. A flux file on the centers would stop short at both ends.
    NEOPAX interpolation returns NaN outside the flux grid. Therefore, this function reads the
    ``n_radial + 1`` face values. The prescribed feedback block uses the same face state.

    The function also reads face gradients from the file. NEOPAX builds them from centered values
    and the run's boundary models. Face values alone cannot reproduce these gradients.
    The file stores them per metre on ``r_grid_half``. This function converts them to per unit rho.

    Parameters
    ----------
    h5_path : Path
        Transport solution to read.
    time_index : int
        Time slice to take, counting back from the last when negative.

    Returns
    -------
    FaceState
        Face state and per-rho face gradients on the transport face grid.

    Raises
    ------
    KeyError
        If the file does not contain the transport layout.
        If a required face or clock dataset is absent. Centered data cannot safely replace it.
    ValueError
        If a static file uses a time index other than ``-1``.
        If one scale factor does not relate the face grids.
        If an early stop left empty save slots.
    IndexError
        If ``time_index`` falls outside the file's time axis.
    """
    solution = read_transport_solution(h5_path, time_index=time_index)
    return FaceState(
        rho=solution.rho_face,
        density=solution.density_face,
        temperature=solution.temperature_face,
        er=solution.er_face,
        density_grad=solution.density_grad_face,
        temperature_grad=solution.temperature_grad_face,
        time_value=solution.time_value,
    )
