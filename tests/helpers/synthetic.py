"""Synthetic HDF5 and NetCDF builders for the pipeline's inter-stage file contracts.

These write the minimal fields a stage reader consumes, so post-processing helpers and
the contract validators can be unit tested without running a solver. Fields follow the
reader/writer code, not the spec docs. Each builder produces a valid file in one call:
the caller supplies the core physics arrays and any single field can be overridden (or,
for the NetCDF builders, dropped via ``omit``) to construct a deliberately-broken file.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np


def write_transport_solution(
    path: Path,
    *,
    rho: np.ndarray,
    pressure: np.ndarray | None = None,
    temperature: np.ndarray | None = None,
    density: np.ndarray | None = None,
    er: np.ndarray | None = None,
) -> Path:
    """Write a synthetic ``transport_solution.h5`` from caller-supplied profiles.

    Parameters
    ----------
    path : Path
        Destination HDF5 file.
    rho : np.ndarray
        Radial coordinate, shape ``(n_rho,)``.
    pressure, temperature, density : np.ndarray, optional
        Species-resolved profiles shaped ``(n_species, n_rho)`` for a static profile
        or ``(n_time, n_species, n_rho)`` for a time-resolved one. Provide either
        ``pressure`` or both ``temperature`` and ``density`` so a reader can form a
        total pressure.
    er : np.ndarray, optional
        Radial electric field ``Er``, shape ``(n_rho,)`` static or ``(n_time, n_rho)``
        time-resolved. Carries no species axis. Required by the Stage 3/4 readers.

    Returns
    -------
    Path
        The written file path.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("rho", data=np.asarray(rho, dtype=float))
        for name, arr in (("pressure", pressure), ("temperature", temperature), ("density", density), ("Er", er)):
            if arr is not None:
                f.create_dataset(name, data=np.asarray(arr, dtype=float))
    return path


def write_sfincs_flux(
    path: Path,
    *,
    rho: np.ndarray,
    gamma: np.ndarray,
    q: np.ndarray,
    r: np.ndarray | None = None,
    upar: np.ndarray | None = None,
    species_names: Sequence[str] = ("e", "D", "T"),
    axis_zero_padded: bool = False,
) -> Path:
    """Write a synthetic ``sfincs_jax_flux_profiles.h5`` neoclassical flux file.

    Parameters
    ----------
    path : Path
        Destination HDF5 file.
    rho : np.ndarray
        Radial coordinate, shape ``(n_radii,)``.
    gamma, q : np.ndarray
        Particle and heat flux, shape ``(n_species, n_radii)``.
    r : np.ndarray, optional
        Effective radius written to both ``r`` and ``rHat``; defaults to ``rho``.
    upar : np.ndarray, optional
        Parallel flow, shape ``(n_species, n_radii)``; defaults to zeros.
    species_names : sequence of str
        Species labels, stored as fixed-length bytes.
    axis_zero_padded : bool
        Root attribute recording whether an axis (rho=0) row was prepended.

    Returns
    -------
    Path
        The written file path.
    """
    rho_arr = np.asarray(rho, dtype=float)
    r_arr = rho_arr if r is None else np.asarray(r, dtype=float)
    gamma_arr = np.asarray(gamma, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    upar_arr = np.zeros_like(gamma_arr) if upar is None else np.asarray(upar, dtype=float)
    with h5py.File(path, "w") as f:
        f.create_dataset("r", data=r_arr)
        f.create_dataset("rHat", data=r_arr)
        f.create_dataset("rho", data=rho_arr)
        f.create_dataset("Gamma", data=gamma_arr)
        f.create_dataset("Q", data=q_arr)
        f.create_dataset("Upar", data=upar_arr)
        f.create_dataset("species_names", data=np.asarray([str(s).encode("utf-8") for s in species_names]))
        f.attrs["axis_zero_padded"] = bool(axis_zero_padded)
    return path


def write_neopax_fluxes(
    path: Path,
    *,
    rho: np.ndarray,
    gamma: np.ndarray,
    q: np.ndarray,
    r: np.ndarray | None = None,
    upar: np.ndarray | None = None,
    species_names: Sequence[str] = ("e", "D", "T"),
    minor_radius_m: float = 1.0,
) -> Path:
    """Write a synthetic ``neopax_fluxes.h5`` turbulence flux file.

    Parameters
    ----------
    path : Path
        Destination HDF5 file.
    rho : np.ndarray
        Radial coordinate, shape ``(n_radii,)``.
    gamma, q : np.ndarray
        Particle and heat flux, shape ``(n_species, n_radii)``.
    r : np.ndarray, optional
        Physical radius; defaults to ``rho``.
    upar : np.ndarray, optional
        Parallel flow, shape ``(n_species, n_radii)``; defaults to zeros to match the writer.
    species_names : sequence of str
        Species labels, stored in ``meta`` as variable-length UTF-8 strings.
    minor_radius_m : float
        Minor radius written to ``meta`` attributes.

    Returns
    -------
    Path
        The written file path.
    """
    rho_arr = np.asarray(rho, dtype=float)
    r_arr = rho_arr if r is None else np.asarray(r, dtype=float)
    gamma_arr = np.asarray(gamma, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    upar_arr = np.zeros_like(gamma_arr) if upar is None else np.asarray(upar, dtype=float)
    with h5py.File(path, "w") as f:
        f.create_dataset("rho", data=rho_arr)
        f.create_dataset("r", data=r_arr)
        f.create_dataset("Gamma", data=gamma_arr)
        f.create_dataset("Q", data=q_arr)
        f.create_dataset("Upar", data=upar_arr)
        meta = f.create_group("meta")
        vlen = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset("species_names", data=np.asarray(list(species_names), dtype=object), dtype=vlen)
        meta.attrs["particle_flux_units"] = "m^-2 s^-1"
        meta.attrs["heat_flux_units"] = "eV m^-2 s^-1"
        meta.attrs["minor_radius_m"] = float(minor_radius_m)
    return path


def write_wout(path: Path, *, ns: int = 4, mnmax: int = 3, nfp: int = 1, omit: Sequence[str] = ()) -> Path:
    """Write a synthetic VMEC ``wout`` NetCDF file with the Stage-2-consumed geometry subset.

    Parameters
    ----------
    path : Path
        Destination NetCDF file.
    ns, mnmax : int
        Radial surface count and Fourier mode count (the two dimension sizes).
    nfp : int
        Number of field periods (a scalar variable).
    omit : sequence of str
        Variable names to skip, for building a file missing a required field. NetCDF ties
        every variable to a dimension, so dropping one is the way to make a broken file.

    Returns
    -------
    Path
        The written file path.
    """
    from netCDF4 import Dataset

    with Dataset(path, "w") as ds:
        ds.createDimension("ns", ns)
        ds.createDimension("mnmax", mnmax)
        for name in ("rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc"):
            if name not in omit:
                ds.createVariable(name, "f8", ("ns", "mnmax"))[:] = np.ones((ns, mnmax))
        for name in ("iotas", "phi", "phipf"):
            if name not in omit:
                ds.createVariable(name, "f8", ("ns",))[:] = np.linspace(0.0, 1.0, ns)
        if "xm" not in omit:
            ds.createVariable("xm", "f8", ("mnmax",))[:] = np.arange(mnmax, dtype=float)
        if "xn" not in omit:
            ds.createVariable("xn", "f8", ("mnmax",))[:] = np.zeros(mnmax)
        if "nfp" not in omit:
            ds.createVariable("nfp", "i4")[...] = int(nfp)
    return path


def write_boozmn(
    path: Path, *, n_surf: int = 3, mnboz: int = 4, nfp: int = 1, omit: Sequence[str] = ()
) -> Path:
    """Write a synthetic Boozer ``boozmn`` NetCDF file with the Stage-3-consumed subset.

    Parameters
    ----------
    path : Path
        Destination NetCDF file.
    n_surf, mnboz : int
        Boozer surface count and mode count (the two dimension sizes).
    nfp : int
        Number of field periods (a scalar variable).
    omit : sequence of str
        Variable names to skip, for building a file missing a required field.

    Returns
    -------
    Path
        The written file path.
    """
    from netCDF4 import Dataset

    with Dataset(path, "w") as ds:
        ds.createDimension("radius", n_surf)
        ds.createDimension("mn_mode", mnboz)
        if "bmnc_b" not in omit:
            ds.createVariable("bmnc_b", "f8", ("radius", "mn_mode"))[:] = np.ones((n_surf, mnboz))
        for name in ("iota_b", "buco_b", "bvco_b", "phi_b", "phip_b"):
            if name not in omit:
                ds.createVariable(name, "f8", ("radius",))[:] = np.linspace(0.0, 1.0, n_surf)
        if "ixm_b" not in omit:
            ds.createVariable("ixm_b", "i4", ("mn_mode",))[:] = np.arange(mnboz)
        if "ixn_b" not in omit:
            ds.createVariable("ixn_b", "i4", ("mn_mode",))[:] = np.zeros(mnboz)
        if "jlist" not in omit:
            ds.createVariable("jlist", "i4", ("radius",))[:] = np.arange(2, 2 + n_surf)
        if "nfp_b" not in omit:
            ds.createVariable("nfp_b", "i4")[...] = int(nfp)
    return path
