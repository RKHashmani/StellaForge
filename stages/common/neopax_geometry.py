"""Calculate the minor radius that NEOPAX uses for its radial grid.

NEOPAX uses ``a = sqrt(volume_p / (2 pi^2 R00_boozer))``. This radius differs from the VMEC
``Aminor_p`` value.
NEOPAX sets ``r_grid_half = a * rho_grid_half``. A transport solution can therefore supply the
radius from its two face grids.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from netCDF4 import Dataset

# This tolerance compares ``r_grid_half`` with ``a * rho_face``. It is a fraction of the
# outermost face radius.
_FACE_GRID_LINEARITY_RTOL = 1.0e-9


def minor_radius_from_volume(volume_p: float, r_major: float) -> float:
    """Return the minor radius implied by a plasma volume and a major radius.

    Evaluate ``sqrt(volume_p / (2 pi^2 R))``. The Boozer ``R00`` gives the NEOPAX ``a_b`` value.
    The VMEC ``Rmajor_p`` gives the VMEC ``Aminor_p`` value. This difference in major radius
    causes the difference in minor radius.

    Parameters
    ----------
    volume_p : float
        Plasma volume from the VMEC ``wout`` file, in m^3.
    r_major : float
        Major radius, in m.

    Returns
    -------
    float
        The minor radius, in m.

    Raises
    ------
    ValueError
        If either input is non-positive or non-finite.
    """
    if not np.isfinite(volume_p) or volume_p <= 0.0:
        raise ValueError(f"volume_p must be finite and positive, got {volume_p!r}.")
    if not np.isfinite(r_major) or r_major <= 0.0:
        raise ValueError(f"r_major must be finite and positive, got {r_major!r}.")
    return float(np.sqrt(volume_p / (2.0 * np.pi**2 * r_major)))


def minor_radius_from_face_grids(
    r_grid_half: npt.ArrayLike,
    rho_face: npt.ArrayLike,
    *,
    source: str,
) -> float:
    """Return the minor radius relating a transport solution's two face grids.

    Evaluate ``r_grid_half[-1] / rho_face[-1]``. Check this factor at every face. The result
    converts quantities per metre to quantities per rho.

    Parameters
    ----------
    r_grid_half : array_like
        Face radii in metres, shape ``(n_radial + 1,)``.
    rho_face : array_like
        The same faces in normalized radius, shape ``(n_radial + 1,)``.
    source : str
        Identifier quoted in the error message, normally the file the grids came from.

    Returns
    -------
    float
        The minor radius, in metres per unit rho.

    Raises
    ------
    ValueError
        If a grid is not 1-D, contains fewer than two faces, or contains a non-finite value.
        If the grid shapes differ, or the outermost normalized face is not positive.
        If the radius is not finite and positive, or one scale factor does not relate the grids.

    Notes
    -----
    A nonuniform scale factor causes a radius-dependent error in each converted quantity.
    Shape and finiteness checks do not detect this error.

    A linearity check accepts both ``-a * rho_face`` and ``a * rho_face``. A negative radius
    reverses each converted gradient. A zero radius makes each converted gradient zero.
    Therefore, the function checks the sign and magnitude separately.
    """
    r_half = np.asarray(r_grid_half, dtype=np.float64)
    rho = np.asarray(rho_face, dtype=np.float64)
    if r_half.shape != rho.shape:
        raise ValueError(f"{source} r_grid_half has shape {r_half.shape}, which does not match rho_face {rho.shape}.")
    for name, arr in (("r_grid_half", r_half), ("rho_face", rho)):
        if arr.ndim != 1 or arr.shape[0] < 2:
            raise ValueError(f"{source} {name} must be 1-D with at least 2 faces, got shape {arr.shape}.")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{source} {name} holds non-finite values.")
    if rho[-1] <= 0.0:
        raise ValueError(
            f"{source} rho_face ends at {float(rho[-1])!r}; the outermost normalized face is the plasma edge "
            "and must be positive for the two grids to imply a radius."
        )
    a_minor = float(r_half[-1] / rho[-1])
    if not np.isfinite(a_minor) or a_minor <= 0.0:
        raise ValueError(
            f"{source} implies a minor radius of {a_minor!r} from r_grid_half[-1] / rho_face[-1]. A radius "
            "that is not positive reverses or flattens every gradient rescaled by it."
        )
    deviation = np.abs(r_half - a_minor * rho)
    if not np.all(deviation <= _FACE_GRID_LINEARITY_RTOL * abs(float(r_half[-1]))):
        worst = int(np.argmax(deviation))
        raise ValueError(
            f"{source} r_grid_half is not rho_face scaled by one minor radius. At face {worst}, "
            f"r_grid_half = {float(r_half[worst])!r} against a * rho_face = {float(a_minor * rho[worst])!r} "
            f"for a = r_grid_half[-1] / rho_face[-1] = {a_minor!r}."
        )
    return a_minor


def read_neopax_minor_radius(wout_path: Path, boozer_path: Path) -> float:
    """Read the equilibrium files and return the ``boozer_volume`` minor radius.

    Parameters
    ----------
    wout_path : Path
        VMEC ``wout_*.nc`` equilibrium file supplying ``volume_p``.
    boozer_path : Path
        Boozer ``boozmn_*.nc`` file that supplies ``rmnc_b``. Its boundary ``R00`` makes this
        radius differ from VMEC ``Aminor_p``.

    Returns
    -------
    float
        The minor radius to relabel onto, in m.

    Raises
    ------
    KeyError
        If a required variable is absent from either file.
    """
    with Dataset(wout_path, mode="r") as vfile:
        if "volume_p" not in vfile.variables:
            raise KeyError(f"VMEC file {wout_path} has no 'volume_p' variable.")
        volume_p = float(vfile.variables["volume_p"][:])

    with Dataset(boozer_path, mode="r") as bfile:
        if "rmnc_b" not in bfile.variables:
            raise KeyError(f"Boozer file {boozer_path} has no 'rmnc_b' variable.")
        r00_boozer = float(bfile.variables["rmnc_b"][:][-1, 0])
    return minor_radius_from_volume(volume_p, r00_boozer)
