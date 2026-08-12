"""Synthetic HDF5 and NetCDF builders for the pipeline's inter-stage file contracts.

These write the minimal fields a stage reader consumes, so post-processing helpers and
the contract validators can be unit tested without running a solver. Fields follow the
reader/writer code, not the spec docs. Each builder produces a valid file in one call. The
HDF5 builders take the core physics arrays from the caller; the NetCDF builders use fixed,
self-consistent dimensions mirroring the real solver layout. To construct a
deliberately-broken file, override a field on an HDF5 builder or drop a required variable
from a NetCDF builder via ``omit``.
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
    rho_face: np.ndarray | None = None,
    pressure_face: np.ndarray | None = None,
    temperature_face: np.ndarray | None = None,
    density_face: np.ndarray | None = None,
    er_face: np.ndarray | None = None,
    r_grid_half: np.ndarray | None = None,
    density_grad_face: np.ndarray | None = None,
    temperature_grad_face: np.ndarray | None = None,
    final_time: float | None = None,
    next_dt: float | None = None,
) -> Path:
    """Write a synthetic ``transport_solution.h5`` from caller-supplied profiles.

    NEOPAX stores evolved state on ``n_rho`` cell centers. It stores face quantities on the
    ``n_rho + 1`` bounding faces. Face dataset names are plural, but ``rho_face`` is singular.
    This naming matches ``write_transport_hdf5``.

    Parameters
    ----------
    path : Path
        Destination HDF5 file.
    rho : np.ndarray
        Cell-center radial coordinate, shape ``(n_rho,)``.
    pressure, temperature, density : np.ndarray, optional
        Species profiles. Static profiles have shape ``(n_species, n_rho)``. Time-resolved
        profiles have shape ``(n_time, n_species, n_rho)``. Supply ``pressure`` or both
        ``temperature`` and ``density``.
    er : np.ndarray, optional
        Radial electric field ``Er``, shape ``(n_rho,)`` static or ``(n_time, n_rho)``
        time-resolved. Carries no species axis. Required by the Stage 3/4 readers.
    rho_face : np.ndarray, optional
        Face radial coordinate, shape ``(n_rho + 1,)``.
    pressure_face, temperature_face, density_face, er_face : np.ndarray, optional
        Face-grid counterparts of the profiles above, carrying ``n_rho + 1`` radial
        entries. Written as ``pressure_faces``, ``temperature_faces``, ``density_faces``
        and ``Er_faces``.
    r_grid_half : np.ndarray, optional
        The face grid in metres, shape ``(n_rho + 1,)``. NEOPAX builds it as its minor radius
        times ``rho_face``, which is the scale the face gradients are per.
    density_grad_face, temperature_grad_face : np.ndarray, optional
        Radial gradients of the face profiles against ``r_grid_half``, shaped like
        ``density_face``. Written as ``density_grad_faces`` and ``temperature_grad_faces``.
    final_time, next_dt : float, optional
        Solver clock in seconds. These values contain the reached time and the next step. The
        function writes scalar datasets and omits values set to ``None``.

    Returns
    -------
    Path
        The written file path.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("rho", data=np.asarray(rho, dtype=float))
        if rho_face is not None:
            f.create_dataset("rho_face", data=np.asarray(rho_face, dtype=float))
        if r_grid_half is not None:
            f.create_dataset("r_grid_half", data=np.asarray(r_grid_half, dtype=float))
        for name, scalar in (("final_time", final_time), ("next_dt", next_dt)):
            if scalar is not None:
                f.create_dataset(name, data=np.float64(scalar))
        for name, arr in (
            ("pressure", pressure),
            ("temperature", temperature),
            ("density", density),
            ("Er", er),
            ("pressure_faces", pressure_face),
            ("temperature_faces", temperature_face),
            ("density_faces", density_face),
            ("Er_faces", er_face),
            ("density_grad_faces", density_grad_face),
            ("temperature_grad_faces", temperature_grad_face),
        ):
            if arr is not None:
                f.create_dataset(name, data=np.asarray(arr, dtype=float))
    return path


def write_sfincs_flux(
    path: Path,
    *,
    rho: np.ndarray,
    gamma: np.ndarray,
    q: np.ndarray,
    upar: np.ndarray | None = None,
) -> Path:
    """Write a synthetic ``sfincs_jax_flux_profiles.h5`` neoclassical flux file.

    Parameters
    ----------
    path : Path
        Destination HDF5 file.
    rho : np.ndarray
        Radial coordinate, shape ``(n_radii,)``; also written to ``r`` and ``rHat``.
    gamma, q : np.ndarray
        Particle and heat flux, shape ``(n_species, n_radii)`` with three species.
    upar : np.ndarray, optional
        Parallel flow, shape ``(n_species, n_radii)``; defaults to zeros.

    Returns
    -------
    Path
        The written file path.

    Notes
    -----
    The ``species_names`` dataset (``e``, ``D``, ``T``) and the ``axis_zero_padded`` root
    attribute are written with fixed values because the contract validator consumes them.
    """
    rho_arr = np.asarray(rho, dtype=float)
    gamma_arr = np.asarray(gamma, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    upar_arr = np.zeros_like(gamma_arr) if upar is None else np.asarray(upar, dtype=float)
    with h5py.File(path, "w") as f:
        f.create_dataset("r", data=rho_arr)
        f.create_dataset("rHat", data=rho_arr)
        f.create_dataset("rho", data=rho_arr)
        f.create_dataset("Gamma", data=gamma_arr)
        f.create_dataset("Q", data=q_arr)
        f.create_dataset("Upar", data=upar_arr)
        f.create_dataset("species_names", data=np.asarray([b"e", b"D", b"T"]))
        f.attrs["axis_zero_padded"] = False
    return path


def write_neopax_fluxes(
    path: Path,
    *,
    rho: np.ndarray,
    gamma: np.ndarray,
    q: np.ndarray,
    upar: np.ndarray | None = None,
) -> Path:
    """Write a synthetic ``neopax_fluxes.h5`` turbulence flux file.

    Parameters
    ----------
    path : Path
        Destination HDF5 file.
    rho : np.ndarray
        Radial coordinate, shape ``(n_radii,)``; also written to ``r``.
    gamma, q : np.ndarray
        Particle and heat flux, shape ``(n_species, n_radii)`` with three species.
    upar : np.ndarray, optional
        Parallel flow, shape ``(n_species, n_radii)``; defaults to zeros to match the writer.

    Returns
    -------
    Path
        The written file path.

    Notes
    -----
    The ``meta`` group's ``species_names`` (``e``, ``D``, ``T``), flux-unit attributes, and
    ``minor_radius_m`` are written with fixed values because the contract validator consumes them.
    """
    rho_arr = np.asarray(rho, dtype=float)
    gamma_arr = np.asarray(gamma, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    upar_arr = np.zeros_like(gamma_arr) if upar is None else np.asarray(upar, dtype=float)
    with h5py.File(path, "w") as f:
        f.create_dataset("rho", data=rho_arr)
        f.create_dataset("r", data=rho_arr)
        f.create_dataset("Gamma", data=gamma_arr)
        f.create_dataset("Q", data=q_arr)
        f.create_dataset("Upar", data=upar_arr)
        meta = f.create_group("meta")
        vlen = h5py.string_dtype(encoding="utf-8")
        meta.create_dataset("species_names", data=np.asarray(["e", "D", "T"], dtype=object), dtype=vlen)
        meta.attrs["particle_flux_units"] = "m^-2 s^-1"
        meta.attrs["heat_flux_units"] = "eV m^-2 s^-1"
        meta.attrs["minor_radius_m"] = 1.0
    return path


def write_wout(path: Path, *, omit: Sequence[str] = ()) -> Path:
    """Write a synthetic VMEC ``wout`` NetCDF file with the Stage-2-consumed geometry subset.

    Mirrors the real VMEC layout: the shape coefficients ``rmnc``/``zmns``/``lmns`` sit on
    the ``mn_mode`` grid (indexed by ``xm``/``xn``) and the field-strength coefficients
    ``bmnc``/``bsubumnc``/``bsubvmnc`` on the denser ``mn_mode_nyq`` grid (indexed by
    ``xm_nyq``/``xn_nyq``), which is why the two mode dimensions differ in size. A scalar
    ``Aminor_p`` supplies the minor radius the Stage 4 reader needs.

    Parameters
    ----------
    path : Path
        Destination NetCDF file.
    omit : sequence of str
        Variable names to skip, for building a file missing a required field. NetCDF ties
        every variable to a dimension, so dropping one is the way to make a broken file.

    Returns
    -------
    Path
        The written file path.
    """
    from netCDF4 import Dataset

    ns = 4
    mn_mode = 3
    mn_mode_nyq = 5
    with Dataset(path, "w") as ds:
        ds.createDimension("radius", ns)
        ds.createDimension("mn_mode", mn_mode)
        ds.createDimension("mn_mode_nyq", mn_mode_nyq)
        for name in ("rmnc", "zmns", "lmns"):
            if name not in omit:
                ds.createVariable(name, "f8", ("radius", "mn_mode"))[:] = np.ones((ns, mn_mode))
        for name in ("bmnc", "bsubumnc", "bsubvmnc"):
            if name not in omit:
                ds.createVariable(name, "f8", ("radius", "mn_mode_nyq"))[:] = np.ones((ns, mn_mode_nyq))
        for name in ("iotas", "phi", "phipf"):
            if name not in omit:
                ds.createVariable(name, "f8", ("radius",))[:] = np.linspace(0.0, 1.0, ns)
        if "xm" not in omit:
            ds.createVariable("xm", "f8", ("mn_mode",))[:] = np.arange(mn_mode, dtype=float)
        if "xn" not in omit:
            ds.createVariable("xn", "f8", ("mn_mode",))[:] = np.zeros(mn_mode)
        if "xm_nyq" not in omit:
            ds.createVariable("xm_nyq", "f8", ("mn_mode_nyq",))[:] = np.arange(mn_mode_nyq, dtype=float)
        if "xn_nyq" not in omit:
            ds.createVariable("xn_nyq", "f8", ("mn_mode_nyq",))[:] = np.zeros(mn_mode_nyq)
        if "nfp" not in omit:
            ds.createVariable("nfp", "i4")[...] = 1
        if "Aminor_p" not in omit:
            ds.createVariable("Aminor_p", "f8")[...] = 0.3
    return path


def write_boozmn(path: Path, *, omit: Sequence[str] = ()) -> Path:
    """Write a synthetic Boozer ``boozmn`` NetCDF file with the Stage-3-consumed subset.

    Mirrors the real booz_xform_jax layout: ``bmnc_b`` and ``jlist`` sit on the packed
    ``pack_rad`` surface grid while the profiles ``iota_b``/``buco_b``/``bvco_b`` span the
    full ``radius`` grid, so those dimensions differ in size. The toroidal-flux profiles
    ``phi_b``/``phip_b`` are not written, matching the real file.

    Parameters
    ----------
    path : Path
        Destination NetCDF file.
    omit : sequence of str
        Variable names to skip, for building a file missing a required field.

    Returns
    -------
    Path
        The written file path.
    """
    from netCDF4 import Dataset

    pack_rad = 3
    radius = pack_rad + 1
    mnboz = 4
    with Dataset(path, "w") as ds:
        ds.createDimension("radius", radius)
        ds.createDimension("pack_rad", pack_rad)
        ds.createDimension("mn_mode", mnboz)
        if "bmnc_b" not in omit:
            ds.createVariable("bmnc_b", "f8", ("pack_rad", "mn_mode"))[:] = np.ones((pack_rad, mnboz))
        for name in ("iota_b", "buco_b", "bvco_b"):
            if name not in omit:
                ds.createVariable(name, "f8", ("radius",))[:] = np.linspace(0.0, 1.0, radius)
        if "ixm_b" not in omit:
            ds.createVariable("ixm_b", "i4", ("mn_mode",))[:] = np.arange(mnboz)
        if "ixn_b" not in omit:
            ds.createVariable("ixn_b", "i4", ("mn_mode",))[:] = np.zeros(mnboz)
        if "jlist" not in omit:
            ds.createVariable("jlist", "i4", ("pack_rad",))[:] = np.arange(2, 2 + pack_rad)
        if "nfp_b" not in omit:
            ds.createVariable("nfp_b", "i4")[...] = 1
    return path
