#!/usr/bin/env python3
"""Write a NEOPAX ``common_input.toml`` whose ``[profiles]`` section prescribes a transport solution slice.

It reads one time slice of a NEOPAX ``transport_solution.h5`` and emits a copy of a
``common_input.toml`` template in which the whole ``[profiles]`` section is replaced by a
prescribed-profiles block holding that slice's density, temperature, and radial electric field.
Everything outside ``[profiles]`` is copied through byte-identically, so the emitted file is a
drop-in template for the next loop iteration. The loop driver seeds iteration ``k+1`` from the
transport solution of iteration ``k``, and every stage of that iteration then reads the same
prescribed profiles instead of re-deriving them from analytical parameters.

NEOPAX solves on a staggered finite-volume grid, so the block carries two radial grids and the
readers differ in which one they take:

- ``density`` / ``temperature`` / ``Er`` hold ``[geometry].n_radial`` **cell-center** values. This
  is NEOPAX's own contract for a prescribed block: ``PrescribedProfileModel`` requires exactly one
  value per cell and derives the outer boundary itself. Since the loop feeds this same file back to
  NEOPAX, these arrays cannot grow to the face count.
- ``density_face`` / ``temperature_face`` / ``Er_face`` hold ``n_radial + 1`` **face** values, which
  is where Stages 3 and 4 sample fluxes because that is the grid NEOPAX interpolates a flux file
  onto. They are copied verbatim from the solution's ``*_faces`` datasets rather than reconstructed
  here: NEOPAX builds them under that run's own boundary models, so any extrapolation invented at
  this end would disagree with the solver wherever the outer boundary is not a plain Dirichlet.
  NEOPAX ignores these extra keys when it reads the block back.

The h5 stores density in 1e20 m^-3, temperature in keV, and Er in kV/m. The emitted block is
SI, so density is scaled by 1e20, temperature by 1e3, and Er passes through unchanged in kV/m.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path
import re
from typing import Any

import h5py
import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


NEOPAX_DENSITY_REFERENCE_M3 = 1.0e20
NEOPAX_TEMPERATURE_REFERENCE_EV = 1.0e3

# Cell-center datasets, then their face counterparts, both required of a NEOPAX transport_solution.h5.
_CENTER_DATASETS = ("rho", "density", "temperature", "Er")
_FACE_DATASETS = ("rho_face", "density_faces", "temperature_faces", "Er_faces")


@dataclasses.dataclass(frozen=True)
class TransportSlice:
    """One time slice of a NEOPAX transport solution, on both of its radial grids.

    Attributes
    ----------
    rho : numpy.ndarray
        Cell centers, shape ``(n_radial,)``. Excludes both ``rho = 0`` and ``rho = rho_edge``.
    density, temperature : numpy.ndarray
        Cell-centered profiles, shape ``(n_species, n_radial)``, in 1e20 m^-3 and keV.
    er : numpy.ndarray
        Cell-centered radial electric field, shape ``(n_radial,)``, in kV/m.
    rho_face : numpy.ndarray
        Cell faces, shape ``(n_radial + 1,)``, spanning ``[0, rho_edge]`` inclusive.
    density_face, temperature_face : numpy.ndarray
        Face profiles, shape ``(n_species, n_radial + 1)``, in the same units as the centers.
    er_face : numpy.ndarray
        Face radial electric field, shape ``(n_radial + 1,)``, in kV/m.
    time_value : float or None
        Slice time in seconds, or ``None`` when the file records no ``ts`` axis.
    """

    rho: np.ndarray
    density: np.ndarray
    temperature: np.ndarray
    er: np.ndarray
    rho_face: np.ndarray
    density_face: np.ndarray
    temperature_face: np.ndarray
    er_face: np.ndarray
    time_value: float | None


def _resolve_time_index(n_times: int, *, time_index: int, h5_path: Path) -> int:
    """Normalize a possibly negative time index against the number of saved slices.

    Parameters
    ----------
    n_times : int
        Length of the time axis.
    time_index : int
        Requested index. Negative values count back from the last slice.
    h5_path : Path
        Source file, quoted in the error message.

    Returns
    -------
    int
        The non-negative index into the time axis.

    Raises
    ------
    ValueError
        If the requested index falls outside the time axis.
    """
    idx = int(time_index)
    if idx < 0:
        idx = n_times + idx
    if idx < 0 or idx >= n_times:
        raise ValueError(f"time index {time_index} is out of range for the {n_times} time slices in {h5_path}")
    return idx


def _check_profile_ranks(
    density_all: np.ndarray,
    temperature_all: np.ndarray,
    er_all: np.ndarray,
    *,
    h5_path: Path,
    names: tuple[str, str, str],
) -> None:
    """Check that a density/temperature/Er triple has consistent ranks, raising ``ValueError`` if not."""
    density_name, temperature_name, er_name = names
    if density_all.ndim not in (2, 3):
        raise ValueError(
            f"{h5_path} dataset '{density_name}' must be 2D (n_species, n_rho) or 3D "
            f"(n_time, n_species, n_rho), got shape {density_all.shape}"
        )
    # Er shares density's axes minus the species axis, so its rank is always one lower.
    if temperature_all.ndim != density_all.ndim or er_all.ndim != density_all.ndim - 1:
        raise ValueError(
            f"{h5_path} has inconsistent dataset ranks: {density_name} {density_all.shape}, "
            f"{temperature_name} {temperature_all.shape}, {er_name} {er_all.shape}"
        )


def _load_final_profiles(h5_path: Path, *, time_index: int) -> TransportSlice:
    """Read one time slice of a NEOPAX transport solution, on both of its radial grids.

    Parameters
    ----------
    h5_path : Path
        Path to ``transport_solution.h5``.
    time_index : int
        Slice to read. Negative values count back from the last slice; a file storing a
        single static slice accepts only the default ``-1``.

    Returns
    -------
    TransportSlice
        Cell-centered and face profiles in the file's units (1e20 m^-3, keV, kV/m).

    Raises
    ------
    KeyError
        If a required dataset is absent. The face datasets are required, so a solution from a
        NEOPAX predating the staggered grid is rejected here rather than producing a block the
        next iteration's Stage 3 and Stage 4 would reject.
    ValueError
        If ranks or time axes disagree, the requested slice does not exist, or the slice
        holds non-finite values.
    """
    with h5py.File(h5_path, "r") as f:
        keys = set(f.keys())
        missing = [name for name in (*_CENTER_DATASETS, *_FACE_DATASETS) if name not in keys]
        if missing:
            raise KeyError(
                f"{h5_path} is missing the required transport datasets: {', '.join(missing)}. "
                "The face datasets are written only by NEOPAX revisions that solve on the staggered "
                "cell/face grid, so an older transport solution cannot seed the next loop iteration."
            )
        data = {name: np.asarray(f[name][()], dtype=np.float64) for name in (*_CENTER_DATASETS, *_FACE_DATASETS)}
        ts = np.asarray(f["ts"][()], dtype=np.float64) if "ts" in keys else None

    rho, density_all, temperature_all, er_all = (data[name] for name in _CENTER_DATASETS)
    rho_face, density_face_all, temperature_face_all, er_face_all = (data[name] for name in _FACE_DATASETS)
    _check_profile_ranks(density_all, temperature_all, er_all, h5_path=h5_path, names=_CENTER_DATASETS[1:])
    _check_profile_ranks(
        density_face_all, temperature_face_all, er_face_all, h5_path=h5_path, names=_FACE_DATASETS[1:]
    )
    if density_face_all.ndim != density_all.ndim:
        raise ValueError(
            f"{h5_path} center and face profiles disagree in rank: density {density_all.shape}, "
            f"density_faces {density_face_all.shape}"
        )

    centered = (density_all, temperature_all, er_all)
    faced = (density_face_all, temperature_face_all, er_face_all)
    if density_all.ndim == 2:
        if int(time_index) != -1:
            raise ValueError(
                f"{h5_path} stores a single static profile slice with no time axis, so --time-index "
                f"{time_index} cannot be selected"
            )
        # A static solution's ts holds exactly one timestamp, which dates that slice. The Stage 3 and
        # Stage 4 loaders read it the same way, so the file is dated identically in every stage.
        time_value = None if ts is None else float(ts[0])
    else:
        n_times = density_all.shape[0]
        if any(arr.shape[0] != n_times for arr in (*centered[1:], *faced)):
            raise ValueError(
                f"{h5_path} time axes disagree: density {density_all.shape}, "
                f"temperature {temperature_all.shape}, Er {er_all.shape}, "
                f"density_faces {density_face_all.shape}"
            )
        idx = _resolve_time_index(n_times, time_index=int(time_index), h5_path=h5_path)
        centered = tuple(arr[idx] for arr in centered)
        faced = tuple(arr[idx] for arr in faced)
        time_value = None if ts is None else float(ts[idx])

    for name, arr in zip((*_CENTER_DATASETS, *_FACE_DATASETS), (rho, *centered, rho_face, *faced), strict=True):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{h5_path} dataset '{name}' holds non-finite values in the selected slice")

    return TransportSlice(
        rho=rho, density=centered[0], temperature=centered[1], er=centered[2],
        rho_face=rho_face, density_face=faced[0], temperature_face=faced[1], er_face=faced[2],
        time_value=time_value,
    )


def _validate_against_template(cfg: dict[str, Any], *, slice_: TransportSlice) -> list[str]:
    """Check that a transport slice can be prescribed into a template, returning the species names.

    Parameters
    ----------
    cfg : dict
        Parsed template TOML.
    slice_ : TransportSlice
        The transport slice to prescribe, on both radial grids.

    Returns
    -------
    list of str
        ``[species].names`` from the template, in row order.

    Raises
    ------
    KeyError
        If the template lacks ``[species].names`` or ``[geometry].n_radial``.
    ValueError
        If the species count, shapes, or either radial grid disagree with the template, or the
        grid is too coarse for the readers' radial gradients. A mismatch fails this iteration
        instead of prescribing unusable profiles to the next one.

    Notes
    -----
    The prescribed block stores profile values with no radial coordinate, so every reader rebuilds
    the grid from ``[geometry]`` alone. Both grids are therefore checked against what NEOPAX builds
    from ``n_radial`` and ``rho_edge``: the faces as ``linspace(0, rho_edge, n_radial + 1)`` and the
    centers as their midpoints. A solution gridded any other way would be misplaced in radius by the
    next iteration's readers, with nothing downstream able to detect it.
    """
    species_cfg = cfg.get("species")
    if not isinstance(species_cfg, dict) or "names" not in species_cfg:
        raise KeyError("The template must define [species].names to label the prescribed profile rows")
    names = [str(name) for name in species_cfg["names"]]

    geometry_cfg = cfg.get("geometry")
    if not isinstance(geometry_cfg, dict) or "n_radial" not in geometry_cfg:
        raise KeyError("The template must define [geometry].n_radial to describe the prescribed radial grid")
    n_radial = int(geometry_cfg["n_radial"])
    rho_edge = float(geometry_cfg.get("rho_edge", 1.0))

    rho, density, temperature, er = slice_.rho, slice_.density, slice_.temperature, slice_.er
    if density.ndim != 2:
        raise ValueError(f"Expected a species-resolved density slice of shape (n_species, n_rho), got {density.shape}")
    if density.shape[0] != len(names):
        raise ValueError(
            f"The transport solution has {density.shape[0]} species but the template [species].names lists "
            f"{len(names)}: {names}"
        )
    if temperature.shape != density.shape:
        raise ValueError(f"temperature shape {temperature.shape} does not match density shape {density.shape}")
    if er.shape != rho.shape:
        raise ValueError(f"Er shape {er.shape} does not match rho shape {rho.shape}")
    if density.shape[1] != rho.size:
        raise ValueError(f"density radius axis {density.shape[1]} does not match len(rho) {rho.size}")
    if rho.size != n_radial:
        raise ValueError(
            f"The transport solution has {rho.size} radii but the template [geometry].n_radial is {n_radial}"
        )
    # The readers difference the face arrays, so the floor is a face count: three faces for np.gradient's
    # edge_order=2, which is two cells. Two cells also clears NEOPAX's own minimum for extrapolating its
    # outer boundary value from the cell-centered arrays it reads back out of this block.
    if rho.size + 1 < 3:
        raise ValueError(
            f"The transport solution has {rho.size} cells, giving {rho.size + 1} faces; the "
            "prescribed-profiles readers need at least 3 faces for radial gradients"
        )

    expected_faces = np.linspace(0.0, rho_edge, n_radial + 1)
    expected_centers = 0.5 * (expected_faces[:-1] + expected_faces[1:])
    if not np.allclose(rho, expected_centers):
        raise ValueError(
            f"The transport solution rho {rho.tolist()} does not match the template's cell centers "
            f"midpoints of linspace(0, {rho_edge}, {n_radial + 1}) = {expected_centers.tolist()}"
        )
    if not np.allclose(slice_.rho_face, expected_faces):
        raise ValueError(
            f"The transport solution rho_face {slice_.rho_face.tolist()} does not match the template's "
            f"faces linspace(0, {rho_edge}, {n_radial + 1}) = {expected_faces.tolist()}"
        )

    for label, arr in (("density_faces", slice_.density_face), ("temperature_faces", slice_.temperature_face)):
        if arr.shape != (len(names), n_radial + 1):
            raise ValueError(
                f"The transport solution {label} has shape {arr.shape}, expected "
                f"{(len(names), n_radial + 1)} for {len(names)} species on {n_radial + 1} faces"
            )
    if slice_.er_face.shape != (n_radial + 1,):
        raise ValueError(
            f"The transport solution Er_faces has shape {slice_.er_face.shape}, expected {(n_radial + 1,)}"
        )
    return names


def _format_float_list(values: np.ndarray) -> str:
    """Render a 1D array as a single-line TOML float array using shortest round-trip literals."""
    return "[" + ", ".join(repr(float(v)) for v in np.asarray(values).ravel()) + "]"


def _format_float_table(values: np.ndarray) -> str:
    """Render a 2D array as a single-line TOML array of float arrays, one inner array per row."""
    return "[" + ", ".join(_format_float_list(row) for row in np.asarray(values)) + "]"


def _render_profiles_block(*, species_names: list[str], slice_: TransportSlice) -> str:
    """Render the replacement ``[profiles]`` section text, converting the slice into SI.

    Parameters
    ----------
    species_names : list of str
        Row labels, in the row order of the slice's profile arrays.
    slice_ : TransportSlice
        The transport slice to prescribe, in the file's units (1e20 m^-3, keV, kV/m).

    Returns
    -------
    str
        The section text, starting at the ``[profiles]`` header, LF-terminated, ending with
        one blank line.

    Notes
    -----
    Arrays are emitted on one line each, so no value line can begin with a column-0 ``[`` and
    the section boundary stays unambiguous. None of the emitted keys are rewritten by the
    parse-time path-resolution step.

    The cell-centered arrays come first because they are the ones NEOPAX itself reads back; the
    ``*_face`` arrays that follow are read only by Stages 3 and 4 and ignored by NEOPAX.
    """
    if slice_.time_value is None:
        provenance = [
            "# Prescribed kinetic profiles taken from the previous closed-loop iteration's transport solution."
        ]
    else:
        provenance = [
            f"# Prescribed kinetic profiles taken from the transport time slice t = {float(slice_.time_value)!r} s"
            " of the",
            "# previous closed-loop iteration's transport solution.",
        ]
    lines = [
        "[profiles]",
        *provenance,
        'model = "prescribed"',
        f"# Per-species number density in m^-3; rows follow [species].names ({', '.join(species_names)}),",
        "# columns follow the [geometry] cell centers, the midpoints of linspace(0, rho_edge, n_radial + 1).",
        "# NEOPAX reads these; its prescribed-profile model takes one value per cell and no boundary face.",
        f"density = {_format_float_table(slice_.density * NEOPAX_DENSITY_REFERENCE_M3)}",
        "# Per-species temperature in eV; same row and column ordering as density.",
        f"temperature = {_format_float_table(slice_.temperature * NEOPAX_TEMPERATURE_REFERENCE_EV)}",
        "# Radial electric field in kV/m on the same cell-center grid.",
        f"Er = {_format_float_list(slice_.er)}",
        "# The same three profiles on the n_radial + 1 cell faces, linspace(0, rho_edge, n_radial + 1).",
        "# Stages 3 and 4 sample fluxes here, because that is the grid NEOPAX interpolates a flux file onto.",
        "# Copied from the transport solution's own face state, so the boundary faces carry the values the",
        "# solver used under this run's boundary conditions rather than an extrapolation invented downstream.",
        f"density_face = {_format_float_table(slice_.density_face * NEOPAX_DENSITY_REFERENCE_M3)}",
        f"temperature_face = {_format_float_table(slice_.temperature_face * NEOPAX_TEMPERATURE_REFERENCE_EV)}",
        f"Er_face = {_format_float_list(slice_.er_face)}",
        "",
        "",
    ]
    return "\n".join(lines)


def _replace_profiles_section(text: str, block: str) -> str:
    """Swap the ``[profiles]`` section of a TOML document for ``block``, leaving the rest untouched.

    Parameters
    ----------
    text : str
        The full template text.
    block : str
        Replacement section text, LF-terminated. It is reterminated to the line endings
        ``text`` already uses (the committed templates are CRLF).

    Returns
    -------
    str
        The template with the ``[profiles]`` section replaced.

    Raises
    ------
    ValueError
        If the text has no column-0 ``[profiles]`` header.

    Notes
    -----
    The replaced span runs from the header line to the next column-0 ``[`` or to end of file,
    keeping same-named keys in other sections intact.
    """
    header = re.search(r"^\[profiles\][ \t]*(?:#[^\n]*)?\r?$", text, flags=re.MULTILINE)
    if header is None:
        raise ValueError("The template has no [profiles] section to replace")
    tail = text[header.end() :]
    next_section = re.search(r"^\[", tail, flags=re.MULTILINE)
    end = len(text) if next_section is None else header.end() + next_section.start()
    newline = "\r\n" if "\r\n" in text else "\n"
    return text[: header.start()] + block.replace("\n", newline) + text[end:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a common_input.toml whose [profiles] section prescribes a transport solution slice.",
    )
    parser.add_argument("transport_h5", type=Path, help="Path to the NEOPAX transport_solution.h5 to feed back")
    parser.add_argument("template_toml", type=Path, help="Path to the common_input.toml to copy")
    parser.add_argument(
        "--output-toml",
        type=Path,
        required=True,
        help="Path to write the prescribed-profiles copy to. The template is never modified in place.",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        default=-1,
        help="Transport time slice to prescribe. Negative values count back from the last slice.",
    )
    args = parser.parse_args()

    template_text = args.template_toml.read_bytes().decode("utf-8")
    cfg = tomllib.loads(template_text)

    slice_ = _load_final_profiles(args.transport_h5, time_index=int(args.time_index))
    species_names = _validate_against_template(cfg, slice_=slice_)

    block = _render_profiles_block(species_names=species_names, slice_=slice_)
    output_text = _replace_profiles_section(template_text, block)
    args.output_toml.parent.mkdir(parents=True, exist_ok=True)
    args.output_toml.write_bytes(output_text.encode("utf-8"))

    slice_label = "static slice" if slice_.time_value is None else f"slice t = {slice_.time_value} s"
    print(
        f"wrote {args.output_toml}: prescribed [profiles] for {len(species_names)} species "
        f"({', '.join(species_names)}) on {slice_.rho.size} cell centers and "
        f"{slice_.rho_face.size} faces from {slice_label}"
    )


if __name__ == "__main__":
    main()
