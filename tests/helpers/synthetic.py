"""Synthetic HDF5 builders for the pipeline's inter-stage file contracts.

These write the minimal datasets a stage reader consumes, so post-processing helpers
can be unit tested without running a solver. Datasets follow the reader code: flat,
top-level, lowercase, species-resolved profiles.
"""

from __future__ import annotations

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

    Returns
    -------
    Path
        The written file path.
    """
    with h5py.File(path, "w") as f:
        f.create_dataset("rho", data=np.asarray(rho, dtype=float))
        for name, arr in (("pressure", pressure), ("temperature", temperature), ("density", density)):
            if arr is not None:
                f.create_dataset(name, data=np.asarray(arr, dtype=float))
    return path
