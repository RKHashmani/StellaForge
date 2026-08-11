"""Rewrite the Stage 4 flux file's radial grid onto the minor-radius convention NEOPAX uses.

Stage 4 writes ``neopax_fluxes.h5`` on ``r = Aminor_p * rho`` using VMEC's ``Aminor_p``, while
NEOPAX builds its own grid as ``r_grid = a * rho``. The two minor radii disagree, so this script
is the Stage 4 to Stage 5 grid adapter that puts the flux file on NEOPAX's grid.

Conventions
-----------
``boozer_volume``
    ``a = sqrt(volume_p / (2 pi^2 R00_boozer))`` with ``R00`` taken from the Boozer file
    (``rmnc_b[-1, 0]``). This is what NEOPAX builds today (``_geometry_models.py``), and the only
    convention defined so far. The config names it rather than switching a boolean, so a NEOPAX that
    grids differently becomes a new name here and a config edit rather than a revert.

VMEC satisfies ``volume_p = 2 pi^2 Rmajor_p Aminor_p^2``, so this differs from Stage 4's
``Aminor_p`` by exactly the ``Rmajor_p`` versus ``R00_boozer`` substitution. Which one is larger is
a property of the equilibrium and not a fixed bias, measured at ``a_b/Aminor_p - 1 = +0.83 %`` for
W7-X and ``-1.04 %`` for HSX.

Why the mismatch matters
------------------------
NEOPAX interpolates the flux file onto its own ``r_grid`` with ``interpax.interp1d``, which
defaults to ``extrap=False`` and returns NaN outside the file's knots rather than raising. Where
NEOPAX's grid is the wider one its outermost cell falls off the end, the NaN propagates into the
pressure RHS, and the Radau solver's first Newton residual is NaN at ``t = 0``, so every step is
rejected and the solve advances zero steps. Newer NEOPAX revisions reject such a grid at model
construction instead. ``[turbulence] lagged_response_mode = "fd"`` is stricter again and requires
the two grids to be equal to within ``1e-12``, which only an exact relabelling achieves.

Rewriting ``r`` as ``a * rho`` puts every knot at the radius NEOPAX means by it, which removes the
systematic radial shift. Where the flux file also samples the same ``rho`` points NEOPAX grids, the
knots then coincide with its targets and the interpolation is an exact identity. A scan that
subsampled radii still interpolates between knots, but from correctly placed ones. Only the ``r``
dataset is rewritten, so the gyro-Bohm normalization Stage 4 applied to the flux values is
unaffected.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
from netCDF4 import Dataset

logger = logging.getLogger(__name__)

# Minor-radius conventions this script can relabel onto, matching the stage4.neopax_radius_relabel
# config values.
CONVENTIONS: tuple[str, ...] = ("boozer_volume",)

# Relative tolerance for deciding the flux grid already sits on NEOPAX's grid, in which case this
# script is a no-op. Chosen well below the ~1e-2 mismatch it exists to remove.
_ALREADY_RELABELLED_RTOL = 1.0e-12


def minor_radius_from_volume(volume_p: float, r_major: float) -> float:
    """Return the minor radius implied by a plasma volume and a major radius.

    Evaluates ``sqrt(volume_p / (2 pi^2 R))``. Passing the Boozer ``R00`` gives NEOPAX's ``a_b``,
    and passing VMEC's ``Rmajor_p`` gives VMEC's ``Aminor_p``, which is the whole of the difference
    between the two radii.

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


def read_neopax_minor_radius(wout_path: Path, boozer_path: Path) -> float:
    """Read the equilibrium files and return the ``boozer_volume`` minor radius.

    Parameters
    ----------
    wout_path : Path
        VMEC ``wout_*.nc`` equilibrium file supplying ``volume_p``.
    boozer_path : Path
        Boozer ``boozmn_*.nc`` file supplying ``rmnc_b``, whose boundary ``R00`` is what makes this
        differ from VMEC's own ``Aminor_p``.

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


def neopax_radial_grid(rho: np.ndarray, a_minor: float, *, rho_edge: float = 1.0) -> np.ndarray:
    """Return the flux-file radial grid expressed in NEOPAX's minor-radius convention.

    Parameters
    ----------
    rho : np.ndarray
        Normalized radial label from the flux file, shape ``(n_r,)``.
    a_minor : float
        Minor radius to relabel onto, in m, from :func:`read_neopax_minor_radius`.
    rho_edge : float, optional
        NEOPAX's ``[geometry].rho_edge``, i.e. the outer end of the grid it builds as
        ``linspace(0, rho_edge, n_radial + 1)``. Default ``1.0``, NEOPAX's own default.

    Returns
    -------
    np.ndarray
        ``a_minor * rho``, shape ``(n_r,)``, in m.

    Raises
    ------
    ValueError
        If ``rho`` is negative, or is not a finite, strictly increasing profile covering
        ``[0, rho_edge]``. Either end falling short leaves the matching end of NEOPAX's grid outside
        the relabelled knots, where interpolation returns NaN.

    Notes
    -----
    NEOPAX solves on a staggered finite-volume grid: ``n_radial`` cells whose ``n_radial + 1``
    bounding faces are ``linspace(0, rho_edge, n_radial + 1)``. Fluxes live on the faces, so that is
    the grid a flux file is interpolated onto, and its ends are 0 and ``rho_edge``. Both ends are
    therefore checked. A Stage 4 scan that subsamples radii through ``rho_min``, ``num_radii`` or
    ``rho_indices`` writes a flux file whose grid starts above 0, since the axis re-insertion in
    ``collect`` only fires for a scan covering every face but the axis.

    The two coverage comparisons are exact and one-sided, not tolerant. interpax admits no tolerance
    of its own, so a knot even one ulp inside a target leaves that target outside the grid. Covering
    *more* than ``[0, rho_edge]`` is accepted, since extra knots outside the transport domain are
    harmless to an interpolation.

    Interior spacing is deliberately unconstrained for the same reason: a sparse or oversampled grid
    covering the interval is a legitimate input for NEOPAX to interpolate from, and rejecting it
    would break both the subsampling options above and an explicit ``--analytical-n-radii``.
    ``[turbulence] lagged_response_mode = "fd"`` is stricter and needs the two grids equal point for
    point, which NEOPAX itself enforces to ``1e-12`` at model construction against its own
    ``r_grid_half`` rather than this script guessing at ``n_radial``.
    """
    rho = np.asarray(rho, dtype=np.float64)
    if rho.ndim != 1 or rho.size < 2:
        raise ValueError(f"rho must be a 1-D profile with at least 2 points, got shape {rho.shape}.")
    if not np.all(np.isfinite(rho)):
        raise ValueError("rho contains non-finite entries.")
    if np.any(np.diff(rho) <= 0.0):
        raise ValueError("rho must be strictly increasing.")
    if not np.isfinite(rho_edge) or rho_edge <= 0.0:
        raise ValueError(f"rho_edge must be finite and positive, got {rho_edge!r}.")
    if rho[0] < 0.0:
        raise ValueError(f"rho is a normalized radius and must not be negative, got rho[0] = {rho[0]!r}.")
    # Coverage is compared exactly, not within a tolerance, because interpax has none of its own. A
    # knot one ulp inside NEOPAX's target leaves that target outside the grid, interpolating to NaN.
    if rho[0] > 0.0:
        raise ValueError(
            f"rho must start at 0 for the relabelled grid to cover NEOPAX's innermost cell, got "
            f"rho[0] = {rho[0]!r}. A Stage 4 scan that subsamples radii (rho_min, num_radii, "
            "rho_indices) writes such a grid, and relabelling would not remove the NaN."
        )
    if rho[-1] < rho_edge:
        raise ValueError(
            f"rho must reach rho_edge = {rho_edge!r} for the relabelled grid to cover NEOPAX's "
            f"outermost cell, got rho[-1] = {rho[-1]!r}. Relabelling would not remove the NaN."
        )
    return a_minor * rho


def relabel_flux_radius(
    flux_path: Path,
    a_minor: float,
    *,
    convention: str = "boozer_volume",
    rho_edge: float = 1.0,
    dry_run: bool = False,
) -> dict[str, float | bool | str]:
    """Rewrite the ``r`` dataset of a NEOPAX flux file as ``a_minor * rho``.

    Idempotent, so a file already on ``a_minor * rho`` is left untouched and reruns cannot compound
    the shift.

    Parameters
    ----------
    flux_path : Path
        ``neopax_fluxes.h5`` written by the Stage 4 ``collect`` subcommand.
    a_minor : float
        Minor radius to relabel onto, in m, from :func:`read_neopax_minor_radius`.
    convention : str, optional
        The convention ``a_minor`` came from, recorded in the file's ``meta`` attributes. Default
        ``"boozer_volume"``.
    rho_edge : float, optional
        NEOPAX's ``[geometry].rho_edge``. Default ``1.0``.
    dry_run : bool, optional
        Report what would change without writing. Default ``False``.

    Returns
    -------
    dict
        ``{"r_edge_old": float, "r_edge_new": float, "convention": str, "changed": bool}`` where
        ``r_edge_old`` is the flux file's original edge radius.

    Raises
    ------
    KeyError
        If the file lacks an ``r`` or ``rho`` dataset.
    ValueError
        If ``r`` and ``rho`` disagree in shape, or if ``convention`` is unknown.
    """
    if convention not in CONVENTIONS:
        raise ValueError(f"convention must be one of {', '.join(repr(c) for c in CONVENTIONS)}, got {convention!r}.")

    mode = "r" if dry_run else "r+"
    with h5py.File(flux_path, mode) as f:
        for name in ("r", "rho"):
            if name not in f:
                raise KeyError(f"Flux file {flux_path} has no '{name}' dataset; cannot relabel its radial grid.")
        rho = np.asarray(f["rho"][...], dtype=np.float64)
        r_old = np.asarray(f["r"][...], dtype=np.float64)
        r_new = neopax_radial_grid(rho, a_minor, rho_edge=rho_edge)
        if r_old.shape != r_new.shape:
            raise ValueError(f"Flux file {flux_path} has r.shape={r_old.shape} but rho.shape={rho.shape}.")

        result: dict[str, float | bool | str] = {
            "r_edge_old": float(r_old[-1]),
            "r_edge_new": float(r_new[-1]),
            "convention": convention,
            "changed": False,
        }
        if np.allclose(r_old, r_new, rtol=_ALREADY_RELABELLED_RTOL, atol=0.0):
            logger.info(
                "%s is already on the %s grid (edge r = %.10f m); nothing to do.", flux_path, convention, r_new[-1]
            )
            return result

        logger.info(
            "relabelling %s onto %s: edge radius %.10f m -> %.10f m (%.4f%% change, %d points)",
            flux_path, convention, r_old[-1], r_new[-1], 100.0 * (r_new[-1] - r_old[-1]) / r_old[-1], rho.size,
        )
        if dry_run:
            logger.info("dry run: no write performed.")
            return result

        f["r"][...] = r_new
        meta = f.require_group("meta")
        meta.attrs["neopax_radius_relabel_applied"] = True
        meta.attrs["neopax_radius_relabel_convention"] = convention
        meta.attrs["neopax_radius_relabel_a_minor_m"] = float(a_minor)
        # The file's own outermost knot, a_minor*rho[-1], which equals the minor radius only on a
        # grid running all the way to rho = 1.
        meta.attrs["neopax_radius_relabel_original_r_edge_m"] = float(r_old[-1])
        meta.attrs["neopax_radius_relabel_note"] = (
            "r rewritten as a_minor*rho so every knot sits at the radius NEOPAX means by it. Where "
            "the grids also coincide point for point its interpolation becomes an identity. Flux "
            "values are unchanged."
        )
        # minor_radius_m and conversion record the length the gyro-Bohm fluxes were converted with
        # and must keep describing that conversion, so only the coordinate's own description moves.
        meta.attrs["radial_flux_coordinate"] = (
            f"r relabelled onto the {convention} minor radius a = {a_minor!r} m; saved Gamma/Q were "
            "converted with minor_radius_m and are unchanged"
        )
    result["changed"] = True
    return result


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser for this script."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--flux-file", required=True, type=Path, help="neopax_fluxes.h5 to relabel in place")
    parser.add_argument("--wout", required=True, type=Path, help="VMEC wout_*.nc supplying the plasma volume")
    parser.add_argument("--boozer", required=True, type=Path, help="Boozer boozmn_*.nc supplying rmnc_b")
    parser.add_argument(
        "--convention",
        choices=CONVENTIONS,
        default="boozer_volume",
        help="minor-radius convention NEOPAX builds its radial grid on (default: boozer_volume)",
    )
    parser.add_argument(
        "--rho-edge",
        type=float,
        default=1.0,
        help="NEOPAX's [geometry].rho_edge; the flux file's rho grid must reach it (default: 1.0)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report the change without writing")
    return parser


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO, format="[relabel-neopax-radius] %(message)s")
    args = build_parser().parse_args()
    a_minor = read_neopax_minor_radius(args.wout, args.boozer)
    logger.info("%s minor radius = %.10f m", args.convention, a_minor)
    relabel_flux_radius(
        args.flux_file, a_minor, convention=args.convention, rho_edge=args.rho_edge, dry_run=args.dry_run
    )


if __name__ == "__main__":
    main()
