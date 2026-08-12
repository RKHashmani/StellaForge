"""Validate files that pass between pipeline stages.

Each validator checks the fields that the next stage reads. This check detects renamed datasets,
transposed axes and removed variables before a later stage fails.

Each pure ``_check_*`` function checks data in memory and returns all problems. Each outer
``validate_*`` function reads a file and raises :class:`ContractError` for those problems.
The outer functions import I/O libraries only when required.

The contracts follow the reader and writer code when repository code defines the interface. The
Stage 4 scan also reads both NetCDF files. It uses the WOUT radius values and the BOOZMN mode
coefficients. The validators therefore check those fields and their mode-number arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .signal_contract import ContractError, _raise_if


def _require(arr: np.ndarray | None, name: str, problems: list[str]) -> bool:
    """Record a problem and return ``False`` when a required field is absent."""
    if arr is None:
        problems.append(f"missing required field '{name}'")
        return False
    return True


def _finite(arr: np.ndarray | None, name: str, problems: list[str]) -> None:
    """Record a problem when a floating-point array holds non-finite values."""
    if arr is not None and np.issubdtype(arr.dtype, np.floating) and not np.all(np.isfinite(arr)):
        problems.append(f"'{name}' contains non-finite values")


def _check_1d(arr: np.ndarray, name: str, expected_len: int | None, problems: list[str]) -> None:
    """Record a problem when an array is not 1D of ``expected_len`` (skipped if ``None``)."""
    if arr.ndim != 1:
        problems.append(f"'{name}' must be 1D, got shape {arr.shape}")
    elif expected_len is not None and arr.shape[0] != expected_len:
        problems.append(f"'{name}' length {arr.shape[0]} != expected {expected_len}")


def _check_flux_2d(
    arr: np.ndarray, name: str, n_species: int | None, n_radii: int | None, problems: list[str]
) -> None:
    """Record problems for a ``(n_species, n_radii)`` flux array's rank, axes, and finiteness."""
    if arr.ndim != 2:
        problems.append(f"'{name}' must be 2D (n_species, n_radii), got shape {arr.shape}")
    else:
        if n_species is not None and arr.shape[0] != n_species:
            problems.append(f"'{name}' species axis {arr.shape[0]} != n_species {n_species}")
        if n_radii is not None and arr.shape[1] != n_radii:
            problems.append(f"'{name}' radius axis {arr.shape[1]} != len(rho) {n_radii}")
    _finite(arr, name, problems)


def _as_text(value: Any) -> str:
    """Decode an HDF5 attribute to ``str`` whether stored as bytes or text."""
    return value.decode() if isinstance(value, bytes) else str(value)


def _check_rho(rho: np.ndarray | None, problems: list[str]) -> int | None:
    """Check the shared required 1D ``rho`` coordinate, returning its length or ``None``."""
    n_rho: int | None = None
    if _require(rho, "rho", problems):
        if rho.ndim != 1:
            problems.append(f"'rho' must be 1D, got shape {rho.shape}")
        else:
            n_rho = rho.shape[0]
        _finite(rho, "rho", problems)
    return n_rho


def _check_species_names(species: np.ndarray | None, name: str, problems: list[str]) -> int | None:
    """Check a required 1D species-label array (``name`` labels it), returning its count or ``None``."""
    n_species: int | None = None
    if _require(species, name, problems):
        if species.ndim != 1:
            problems.append(f"'{name}' must be 1D, got shape {species.shape}")
        else:
            n_species = species.shape[0]
    return n_species


# transport_solution.h5 ------------------------------------------------------------------

_TRANSPORT_FIELDS = (
    "rho", "density", "temperature", "pressure", "Er", "ts",
    "rho_face", "density_faces", "temperature_faces", "pressure_faces", "Er_faces",
)


def _check_species_profile(
    arr: np.ndarray, name: str, n_rho: int | None, problems: list[str], *, coord: str = "rho"
) -> None:
    """Record problems for a species-resolved profile (static or time-resolved).

    ``coord`` names the radial coordinate the trailing axis must span, so the same check serves the
    cell-centered profiles against ``rho`` and the face profiles against ``rho_face``.
    """
    if arr.ndim not in (2, 3):
        problems.append(
            f"'{name}' must be 2D (n_species, n_rho) or 3D (n_time, n_species, n_rho), got shape {arr.shape}"
        )
    elif n_rho is not None and arr.shape[-1] != n_rho:
        problems.append(f"'{name}' trailing axis {arr.shape[-1]} != len({coord}) {n_rho}")
    _finite(arr, name, problems)


def _check_face_grid(
    rho_face: np.ndarray | None, rho: np.ndarray | None, n_rho: int | None, problems: list[str]
) -> int | None:
    """Check the required 1D ``rho_face`` coordinate against ``rho``, returning its length or ``None``.

    The faces bound the cells, so a solution on ``n_rho`` centers has exactly ``n_rho + 1`` faces, they march
    strictly outward, and each center sits midway between the two faces bounding it. A grid failing any of those
    describes different geometry than ``rho`` does.

    The last two checks compare the grids elementwise, so they run only once the length check has passed;
    ``n_rho`` of ``None`` (``rho`` missing or not 1D) skips all three.
    """
    n_face: int | None = None
    if _require(rho_face, "rho_face", problems):
        if rho_face.ndim != 1:
            problems.append(f"'rho_face' must be 1D, got shape {rho_face.shape}")
        else:
            n_face = rho_face.shape[0]
            if n_rho is not None and rho is not None:
                if n_face != n_rho + 1:
                    problems.append(
                        f"'rho_face' holds {n_face} faces, expected one more than the {n_rho} cell centers in 'rho'"
                    )
                else:
                    _check_face_geometry(rho_face, rho, problems)
        _finite(rho_face, "rho_face", problems)
    return n_face


def _check_face_geometry(rho_face: np.ndarray, rho: np.ndarray, problems: list[str]) -> None:
    """Record problems when the faces do not march outward or do not bracket the centers.

    Called only for grids whose lengths already agree (``n_rho + 1`` faces for ``n_rho`` centers).
    The midpoint rule is NEOPAX's own definition of the grid, ``_geometry_models.py``:
    ``rho_grid = 0.5 * (rho_grid_half[:-1] + rho_grid_half[1:])``. Strictly increasing faces plus
    that rule make the centers strictly increasing too, so the centers need no separate check.
    Default ``numpy.allclose`` tolerances absorb the file's float32 storage.
    """
    increasing = np.diff(rho_face) > 0.0
    if not np.all(increasing):
        first = int(np.argmin(increasing))
        problems.append(
            f"'rho_face' must increase strictly outward, but rho_face[{first}] = {rho_face[first]} is followed "
            f"by rho_face[{first + 1}] = {rho_face[first + 1]}"
        )

    midpoints = 0.5 * (rho_face[:-1] + rho_face[1:])
    if not np.allclose(rho, midpoints):
        worst = int(np.argmax(np.abs(rho - midpoints)))
        problems.append(
            f"'rho' must hold the midpoints of 'rho_face', which is how NEOPAX defines its cell centers "
            f"(rho_grid = 0.5 * (rho_grid_half[:-1] + rho_grid_half[1:])), but rho[{worst}] = {rho[worst]} "
            f"!= {midpoints[worst]}"
        )


def _check_transport_state(
    data: dict[str, np.ndarray | None],
    *,
    names: tuple[str, str, str, str],
    coord: str,
    n_points: int | None,
    problems: list[str],
) -> np.ndarray | None:
    """Check one radial grid's worth of transport state, returning its density array or ``None``.

    Parameters
    ----------
    data : dict
        The datasets read from the file.
    names : tuple of str
        ``(density, temperature, pressure, Er)`` dataset names for this grid, so the same checks
        serve the cell-centered state and its face counterpart.
    coord : str
        Name of the radial coordinate these profiles span, quoted in messages.
    n_points : int or None
        Length of that coordinate, or ``None`` when it could not be determined.
    problems : list of str
        Accumulator appended to in place.

    Returns
    -------
    numpy.ndarray or None
        The density array, which the caller uses to cross-check ranks between the two grids.
    """
    density_name, temperature_name, pressure_name, er_name = names
    density = data.get(density_name)
    temperature = data.get(temperature_name)
    for name, arr in ((density_name, density), (temperature_name, temperature)):
        if _require(arr, name, problems):
            _check_species_profile(arr, name, n_points, problems, coord=coord)
    pressure = data.get(pressure_name)
    if pressure is not None:
        _check_species_profile(pressure, pressure_name, n_points, problems, coord=coord)
        # Only the trailing axis is radial, so everything before it, time and species, must match density's.
        if density is not None and pressure.shape[:-1] != density.shape[:-1]:
            problems.append(
                f"'{pressure_name}' leading axes {pressure.shape[:-1]} != '{density_name}' leading axes "
                f"{density.shape[:-1]} ({pressure_name} must share {density_name}'s time and species axes)"
            )

    er = data.get(er_name)
    if _require(er, er_name, problems):
        if er.ndim not in (1, 2):
            problems.append(f"'{er_name}' must be 1D (n_rho,) or 2D (n_time, n_rho), got shape {er.shape}")
        elif n_points is not None and er.shape[-1] != n_points:
            problems.append(f"'{er_name}' trailing axis {er.shape[-1]} != len({coord}) {n_points}")
        _finite(er, er_name, problems)
        # Er drops the species axis but keeps density's time axis, so its leading axes are density's
        # with the species entry removed.
        if density is not None and density.ndim in (2, 3) and er.ndim == density.ndim - 1:
            if er.shape[:-1] != density.shape[:-2]:
                problems.append(
                    f"'{er_name}' leading axes {er.shape[:-1]} != '{density_name}' time axis "
                    f"{density.shape[:-2]} ({er_name} shares the time axis and drops only the species axis)"
                )

    if density is not None and temperature is not None and density.shape != temperature.shape:
        problems.append(f"'{density_name}' shape {density.shape} != '{temperature_name}' shape {temperature.shape}")
    # Er advances with the same time axis as the profiles but carries no species axis,
    # so it must have exactly one fewer dimension than density.
    if (
        density is not None
        and er is not None
        and density.ndim in (2, 3)
        and er.ndim in (1, 2)
        and er.ndim != density.ndim - 1
    ):
        problems.append(
            f"'{er_name}' ndim {er.ndim} should be one less than '{density_name}' ndim {density.ndim} "
            f"({er_name} carries no species axis)"
        )
    return density


def _check_transport_solution(data: dict[str, np.ndarray | None]) -> list[str]:
    """Check the transport fields that the Stage 3, 4 and 5 loaders read.

    NEOPAX uses a staggered finite-volume grid. The file contains state on ``n_rho`` cell centers
    and on ``n_rho + 1`` faces. This contract requires both states.

    The feedback writer and the Stage 3 and 4 scans read the face state. The pressure fit and the
    convergence signal read the centered state. A forward pass can use only the centered state,
    but a closed-loop iteration also requires the face state.

    Stage 3 can derive ``temperature`` from ``pressure``. Stage 4 requires ``temperature``. The
    contract therefore requires ``temperature`` and keeps ``pressure`` optional on both grids.

    The required centered fields are ``rho``, ``density``, ``temperature`` and ``Er``. The required
    face fields are ``rho_face``, ``density_faces``, ``temperature_faces`` and ``Er_faces``.
    ``pressure``, ``pressure_faces`` and ``ts`` are optional.

    Each profile uses its trailing axis for radius. The other axes must agree across related
    fields. ``Er`` and ``Er_faces`` omit only the species axis.

    ``rho_face`` has one more point than ``rho`` and increases outward. Each center is the midpoint
    of its two faces. A time-resolved ``ts`` has one value per slice. A static ``ts`` has one value.

    This function does not check face gradients, ``r_grid_half``, the minor radius or the clock.
    ``validate_transport_solution`` first calls the shared loader, which checks those values.
    """
    problems: list[str] = []

    rho = data.get("rho")
    n_rho = _check_rho(rho, problems)
    n_face = _check_face_grid(data.get("rho_face"), rho, n_rho, problems)

    density = _check_transport_state(
        data, names=("density", "temperature", "pressure", "Er"), coord="rho", n_points=n_rho, problems=problems
    )
    density_faces = _check_transport_state(
        data,
        names=("density_faces", "temperature_faces", "pressure_faces", "Er_faces"),
        coord="rho_face",
        n_points=n_face,
        problems=problems,
    )
    # Both grids describe the same instant. All axes except the trailing radial axis must agree.
    if density is not None and density_faces is not None and density.shape[:-1] != density_faces.shape[:-1]:
        problems.append(
            f"'density_faces' leading axes {density_faces.shape[:-1]} != 'density' leading axes "
            f"{density.shape[:-1]} (the centered and face states are one instant on two grids, so they "
            "share their time and species axes and differ only in the radial one)"
        )

    ts = data.get("ts")
    if ts is not None:
        if ts.ndim != 1:
            problems.append(f"'ts' must be 1D (n_time,), got shape {ts.shape}")
        elif density is not None and density.ndim == 3 and ts.shape[0] != density.shape[0]:
            problems.append(f"'ts' length {ts.shape[0]} != time axis {density.shape[0]}")
        elif density is not None and density.ndim == 2 and ts.shape[0] != 1:
            # A static solution has one slice and one timestamp.
            # A longer ``ts`` names absent slices.
            problems.append(
                f"'ts' length {ts.shape[0]} != 1 for a single static profile slice"
            )
        _finite(ts, "ts", problems)

    return problems


def validate_transport_solution(path: Path | str) -> None:
    """Validate a NEOPAX ``transport_solution.h5`` against the transport-loader contract.

    The validator first reads the file with ``common.neopax_profiles.read_transport_solution``,
    the loader every face-state consumer runs. That read checks the face state, the face
    gradients, the minor radius and the clock. ``_check_transport_solution`` then checks what the
    loader does not read, the centered state with its optional pressure and the geometry between
    the two radial grids.

    Parameters
    ----------
    path : Path or str
        HDF5 file written by the transport stage.

    Raises
    ------
    ContractError
        If any required dataset is missing or malformed.

    Notes
    -----
    ``common.neopax_profiles`` imports only when ``stages/`` is on the import path.
    ``pytest.ini`` and the Snakefile's container ``PYTHONPATH`` both add it.
    """
    import h5py

    from common.neopax_profiles import read_transport_solution

    try:
        read_transport_solution(Path(path), time_index=-1)
    except (KeyError, ValueError, IndexError) as exc:
        raise ContractError(f"{path} violates the transport_solution.h5 contract: {exc}") from exc
    with h5py.File(path, "r") as f:
        data = {k: (np.asarray(f[k][()]) if k in f else None) for k in _TRANSPORT_FIELDS}
    _raise_if(_check_transport_solution(data), f"{path} violates the transport_solution.h5 contract")


# sfincs_jax_flux_profiles.h5 ------------------------------------------------------------

_SFINCS_FIELDS = ("r", "rHat", "rho", "Gamma", "Q", "Upar", "species_names")


def _check_sfincs_flux(data: dict[str, np.ndarray | None], attrs: dict[str, Any]) -> list[str]:
    """Check the Stage 3 neoclassical flux profiles written by the sfincs radial scan."""
    problems: list[str] = []

    n_radii = _check_rho(data.get("rho"), problems)

    for name in ("r", "rHat"):
        arr = data.get(name)
        if _require(arr, name, problems):
            _check_1d(arr, name, n_radii, problems)
            _finite(arr, name, problems)

    n_species = _check_species_names(data.get("species_names"), "species_names", problems)

    for name in ("Gamma", "Q", "Upar"):
        arr = data.get(name)
        if _require(arr, name, problems):
            _check_flux_2d(arr, name, n_species, n_radii, problems)

    if "axis_zero_padded" not in attrs:
        problems.append("missing required root attribute 'axis_zero_padded'")

    return problems


def validate_sfincs_flux(path: Path | str) -> None:
    """Validate a ``sfincs_jax_flux_profiles.h5`` against the Stage 3 flux contract.

    Raises
    ------
    ContractError
        If any required dataset or the ``axis_zero_padded`` attribute is missing or malformed.
    """
    import h5py

    with h5py.File(path, "r") as f:
        data = {k: (np.asarray(f[k][()]) if k in f else None) for k in _SFINCS_FIELDS}
        attrs = dict(f.attrs)
    _raise_if(_check_sfincs_flux(data, attrs), f"{path} violates the sfincs_jax_flux_profiles.h5 contract")


# neopax_fluxes.h5 -----------------------------------------------------------------------

_NEOPAX_FIELDS = ("rho", "r", "Gamma", "Q", "Upar")
_NEOPAX_UNIT_ATTRS = (("particle_flux_units", "m^-2 s^-1"), ("heat_flux_units", "eV m^-2 s^-1"))


def _check_neopax_fluxes(
    data: dict[str, np.ndarray | None], species_names: np.ndarray | None, meta_attrs: dict[str, Any]
) -> list[str]:
    """Check the Stage 4 turbulence flux profiles written for NEOPAX consumption."""
    problems: list[str] = []

    n_radii = _check_rho(data.get("rho"), problems)

    r = data.get("r")
    if _require(r, "r", problems):
        _check_1d(r, "r", n_radii, problems)
        _finite(r, "r", problems)

    n_species = _check_species_names(species_names, "meta/species_names", problems)

    for name in ("Gamma", "Q", "Upar"):
        arr = data.get(name)
        if _require(arr, name, problems):
            _check_flux_2d(arr, name, n_species, n_radii, problems)

    # The writer always emits Upar as zeros; a non-zero column signals a wrong or
    # mislabeled array rather than a valid parallel-flow field.
    upar = data.get("Upar")
    if upar is not None and np.issubdtype(upar.dtype, np.floating) and upar.size and not np.all(upar == 0.0):
        problems.append("'Upar' must be all zeros for neopax_fluxes")

    for attr, expected in _NEOPAX_UNIT_ATTRS:
        if attr not in meta_attrs:
            problems.append(f"missing required meta attribute '{attr}'")
        elif _as_text(meta_attrs[attr]) != expected:
            problems.append(f"meta attribute '{attr}' = {_as_text(meta_attrs[attr])!r}, expected {expected!r}")
    if "minor_radius_m" not in meta_attrs:
        problems.append("missing required meta attribute 'minor_radius_m'")

    return problems


def validate_neopax_fluxes(path: Path | str) -> None:
    """Validate a ``neopax_fluxes.h5`` against the Stage 4 flux contract.

    Raises
    ------
    ContractError
        If any required dataset, the ``meta/species_names`` dataset, or a required
        ``meta`` attribute is missing or malformed.
    """
    import h5py

    with h5py.File(path, "r") as f:
        data = {k: (np.asarray(f[k][()]) if k in f else None) for k in _NEOPAX_FIELDS}
        species_names: np.ndarray | None = None
        meta_attrs: dict[str, Any] = {}
        if "meta" in f:
            meta = f["meta"]
            if "species_names" in meta:
                species_names = np.asarray(meta["species_names"][()])
            meta_attrs = dict(meta.attrs)
    _raise_if(
        _check_neopax_fluxes(data, species_names, meta_attrs),
        f"{path} violates the neopax_fluxes.h5 contract",
    )


# wout_*.nc ------------------------------------------------------------------------------

# rmnc/zmns/lmns live on the mn_mode grid indexed by xm/xn; bmnc/bsubumnc/bsubvmnc live on
# the denser Nyquist grid indexed by xm_nyq/xn_nyq. Each family is checked against its own
# mode count, so they are tracked separately.
_WOUT_2D = ("rmnc", "zmns", "lmns")
_WOUT_2D_NYQ = ("bmnc", "bsubumnc", "bsubvmnc")
_WOUT_1D = ("iotas", "phi", "phipf")
_WOUT_MODES = ("xm", "xn")
_WOUT_MODES_NYQ = ("xm_nyq", "xn_nyq")
# Minor-radius scalars the Stage 4 reader needs to set the reference length; Aminor_p
# directly, or volume_p with Rmajor_p to derive it.
_WOUT_MINOR_RADIUS = ("Aminor_p", "volume_p", "Rmajor_p")
_WOUT_FIELDS = (
    _WOUT_2D + _WOUT_2D_NYQ + _WOUT_1D + _WOUT_MODES + _WOUT_MODES_NYQ + _WOUT_MINOR_RADIUS + ("nfp",)
)


def _check_wout_2d(arr: np.ndarray, name: str, ns: int | None, mnmax: int | None, problems: list[str]) -> None:
    """Record problems for a wout Fourier-coefficient array shaped ``(ns, mode count)``."""
    if arr.ndim != 2:
        problems.append(f"'{name}' must be 2D (ns, mode count), got shape {arr.shape}")
    else:
        if ns is not None and arr.shape[0] != ns:
            problems.append(f"'{name}' radial axis {arr.shape[0]} != ns {ns}")
        if mnmax is not None and arr.shape[1] != mnmax:
            problems.append(f"'{name}' mode axis {arr.shape[1]} != mode count {mnmax}")
    _finite(arr, name, problems)


def _check_wout(data: dict[str, np.ndarray | None]) -> list[str]:
    """Check the Stage-2-consumed VMEC ``wout`` geometry subset.

    VMEC writes the shape coefficients ``rmnc``/``zmns``/``lmns`` on the ``mn_mode`` grid
    indexed by ``xm``/``xn``, and the field-strength coefficients
    ``bmnc``/``bsubumnc``/``bsubvmnc`` on the denser Nyquist grid indexed by
    ``xm_nyq``/``xn_nyq``. Each coefficient family is checked against the mode count of its
    own grid, and both families share the radial axis ``ns``. The coefficients are
    meaningless without those mode-number arrays and the field-period count ``nfp``, so all
    are required. The minor-radius scalars the Stage 4 reader needs are required too:
    ``Aminor_p``, or ``volume_p`` together with ``Rmajor_p`` to derive it.
    """
    problems: list[str] = []

    xm = data.get("xm")
    rmnc = data.get("rmnc")
    mnmax: int | None = None
    if xm is not None and xm.ndim == 1:
        mnmax = xm.shape[0]
    elif rmnc is not None and rmnc.ndim == 2:
        mnmax = rmnc.shape[1]

    xm_nyq = data.get("xm_nyq")
    bmnc = data.get("bmnc")
    mnmax_nyq: int | None = None
    if xm_nyq is not None and xm_nyq.ndim == 1:
        mnmax_nyq = xm_nyq.shape[0]
    elif bmnc is not None and bmnc.ndim == 2:
        mnmax_nyq = bmnc.shape[1]

    iotas = data.get("iotas")
    ns: int | None = None
    if iotas is not None and iotas.ndim == 1:
        ns = iotas.shape[0]
    elif rmnc is not None and rmnc.ndim == 2:
        ns = rmnc.shape[0]

    for name in _WOUT_2D:
        arr = data.get(name)
        if _require(arr, name, problems):
            _check_wout_2d(arr, name, ns, mnmax, problems)
    for name in _WOUT_2D_NYQ:
        arr = data.get(name)
        if _require(arr, name, problems):
            _check_wout_2d(arr, name, ns, mnmax_nyq, problems)

    for name in _WOUT_1D:
        arr = data.get(name)
        if _require(arr, name, problems):
            _check_1d(arr, name, ns, problems)
            _finite(arr, name, problems)

    for name in _WOUT_MODES:
        arr = data.get(name)
        if _require(arr, name, problems):
            _check_1d(arr, name, mnmax, problems)
            _finite(arr, name, problems)
    for name in _WOUT_MODES_NYQ:
        arr = data.get(name)
        if _require(arr, name, problems):
            _check_1d(arr, name, mnmax_nyq, problems)
            _finite(arr, name, problems)

    if data.get("nfp") is None:
        problems.append("missing required field 'nfp'")

    aminor = data.get("Aminor_p")
    if aminor is None and not (data.get("volume_p") is not None and data.get("Rmajor_p") is not None):
        problems.append("missing required minor-radius scalars: 'Aminor_p' or both 'volume_p' and 'Rmajor_p'")

    return problems


def validate_wout(path: Path | str) -> None:
    """Validate a VMEC ``wout`` NetCDF file against the Stage-2-consumed geometry subset.

    Raises
    ------
    ContractError
        If any consumed variable is missing or its shape is inconsistent.
    """
    from netCDF4 import Dataset

    with Dataset(path, "r") as ds:
        data = {k: (np.asarray(ds.variables[k][...]) if k in ds.variables else None) for k in _WOUT_FIELDS}
    _raise_if(_check_wout(data), f"{path} violates the wout NetCDF contract")


# boozmn_*.nc ----------------------------------------------------------------------------

_BOOZMN_2D = ("bmnc_b",)
_BOOZMN_1D_PROFILES = ("iota_b", "buco_b", "bvco_b")
# Some BOOZ_XFORM writers (including the in-repo booz_xform_jax) omit the toroidal-flux
# profiles phi_b/phip_b; they are optional, but drift-checked when present.
_BOOZMN_1D_OPTIONAL = ("phi_b", "phip_b")
_BOOZMN_MODES = ("ixm_b", "ixn_b")
_BOOZMN_FIELDS = _BOOZMN_2D + _BOOZMN_1D_PROFILES + _BOOZMN_1D_OPTIONAL + _BOOZMN_MODES + ("jlist", "nfp_b")


def _check_boozmn(data: dict[str, np.ndarray | None]) -> list[str]:
    """Check the Stage-3-consumed Boozer ``boozmn`` subset.

    ``bmnc_b`` is indexed ``[surface, mode]``, so its mode axis must agree with the
    ``ixm_b``/``ixn_b`` mode-number arrays. The 1D profiles are only checked for rank
    and finiteness: BOOZ_XFORM variants disagree on whether they span every VMEC surface
    or only the ``jlist`` subset, so their length is not cross-checked against ``bmnc_b``.
    The toroidal-flux profiles ``phi_b``/``phip_b`` are optional because some writers omit
    them; when a file does provide them, they are still checked for rank and finiteness.
    """
    problems: list[str] = []

    bmnc = data.get("bmnc_b")
    ixm = data.get("ixm_b")
    mnboz: int | None = None
    if ixm is not None and ixm.ndim == 1:
        mnboz = ixm.shape[0]
    elif bmnc is not None and bmnc.ndim == 2:
        mnboz = bmnc.shape[1]

    if _require(bmnc, "bmnc_b", problems):
        if bmnc.ndim != 2:
            problems.append(f"'bmnc_b' must be 2D (surfaces, mnboz), got shape {bmnc.shape}")
        elif mnboz is not None and bmnc.shape[1] != mnboz:
            problems.append(f"'bmnc_b' mode axis {bmnc.shape[1]} != mnboz {mnboz}")
        _finite(bmnc, "bmnc_b", problems)

    for name in _BOOZMN_1D_PROFILES:
        arr = data.get(name)
        if _require(arr, name, problems):
            if arr.ndim != 1:
                problems.append(f"'{name}' must be 1D (per-surface profile), got shape {arr.shape}")
            _finite(arr, name, problems)

    for name in _BOOZMN_1D_OPTIONAL:
        arr = data.get(name)
        if arr is not None:
            if arr.ndim != 1:
                problems.append(f"'{name}' must be 1D (per-surface profile), got shape {arr.shape}")
            _finite(arr, name, problems)

    for name in _BOOZMN_MODES:
        arr = data.get(name)
        if _require(arr, name, problems):
            _check_1d(arr, name, mnboz, problems)

    jlist = data.get("jlist")
    if _require(jlist, "jlist", problems) and jlist.ndim != 1:
        problems.append(f"'jlist' must be 1D, got shape {jlist.shape}")

    if data.get("nfp_b") is None:
        problems.append("missing required field 'nfp_b'")

    return problems


def validate_boozmn(path: Path | str) -> None:
    """Validate a Boozer ``boozmn`` NetCDF file against the Stage-3-consumed subset.

    Raises
    ------
    ContractError
        If any consumed variable is missing or its mode axis is inconsistent.
    """
    from netCDF4 import Dataset

    with Dataset(path, "r") as ds:
        data = {k: (np.asarray(ds.variables[k][...]) if k in ds.variables else None) for k in _BOOZMN_FIELDS}
    _raise_if(_check_boozmn(data), f"{path} violates the boozmn NetCDF contract")
