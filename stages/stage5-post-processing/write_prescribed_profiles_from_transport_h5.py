#!/usr/bin/env python3
"""Write a NEOPAX ``common_input.toml`` whose ``[profiles]`` section prescribes a transport solution slice.

It reads one time slice of a NEOPAX ``transport_solution.h5`` and emits a copy of a
``common_input.toml`` template in which the whole ``[profiles]`` section is replaced by a
prescribed-profiles block holding that slice's density, temperature, and radial electric field.
Everything outside ``[profiles]`` is copied through byte-identically, so the emitted file is a
drop-in template for the next loop iteration. The loop driver seeds iteration ``k+1`` from the
transport solution of iteration ``k``, and every stage of that iteration then reads the same
prescribed profiles instead of re-deriving them from analytical parameters.

The h5 stores density in 1e20 m^-3, temperature in keV, and Er in kV/m. The emitted block is
SI, so density is scaled by 1e20, temperature by 1e3, and Er passes through unchanged in kV/m.
"""

from __future__ import annotations

import argparse
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


def _load_final_profiles(
    h5_path: Path,
    *,
    time_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float | None]:
    """Read one time slice of a NEOPAX transport solution.

    Parameters
    ----------
    h5_path : Path
        Path to ``transport_solution.h5``.
    time_index : int
        Slice to read. Negative values count back from the last slice; a file storing a
        single static slice accepts only the default ``-1``.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray, float or None)
        ``(rho, density, temperature, er, time_value)`` in the file's units (1e20 m^-3, keV,
        kV/m). ``density`` and ``temperature`` have shape ``(n_species, n_rho)``, ``rho`` and
        ``er`` have shape ``(n_rho,)``, and ``time_value`` is the slice time in seconds when
        the file records a ``ts`` axis.

    Raises
    ------
    KeyError
        If a required dataset is absent.
    ValueError
        If ranks or time axes disagree, the requested slice does not exist, or the slice
        holds non-finite values.
    """
    with h5py.File(h5_path, "r") as f:
        keys = set(f.keys())
        for name in ("rho", "density", "temperature", "Er"):
            if name not in keys:
                raise KeyError(f"{h5_path} is missing the required transport dataset '{name}'")
        rho = np.asarray(f["rho"][()], dtype=np.float64)
        density_all = np.asarray(f["density"][()], dtype=np.float64)
        temperature_all = np.asarray(f["temperature"][()], dtype=np.float64)
        er_all = np.asarray(f["Er"][()], dtype=np.float64)
        ts = np.asarray(f["ts"][()], dtype=np.float64) if "ts" in keys else None

    if density_all.ndim not in (2, 3):
        raise ValueError(
            f"{h5_path} dataset 'density' must be 2D (n_species, n_rho) or 3D (n_time, n_species, n_rho), "
            f"got shape {density_all.shape}"
        )
    # Er shares density's axes minus the species axis, so its rank is always one lower.
    if temperature_all.ndim != density_all.ndim or er_all.ndim != density_all.ndim - 1:
        raise ValueError(
            f"{h5_path} has inconsistent dataset ranks: density {density_all.shape}, "
            f"temperature {temperature_all.shape}, Er {er_all.shape}"
        )

    if density_all.ndim == 2:
        if int(time_index) != -1:
            raise ValueError(
                f"{h5_path} stores a single static profile slice with no time axis, so --time-index "
                f"{time_index} cannot be selected"
            )
        density, temperature, er, time_value = density_all, temperature_all, er_all, None
    else:
        n_times = density_all.shape[0]
        if temperature_all.shape[0] != n_times or er_all.shape[0] != n_times:
            raise ValueError(
                f"{h5_path} time axes disagree: density {density_all.shape}, "
                f"temperature {temperature_all.shape}, Er {er_all.shape}"
            )
        idx = _resolve_time_index(n_times, time_index=int(time_index), h5_path=h5_path)
        density, temperature, er = density_all[idx], temperature_all[idx], er_all[idx]
        time_value = None if ts is None else float(ts[idx])

    for name, arr in (("rho", rho), ("density", density), ("temperature", temperature), ("Er", er)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{h5_path} dataset '{name}' holds non-finite values in the selected slice")
    return rho, density, temperature, er, time_value


def _validate_against_template(
    cfg: dict[str, Any],
    *,
    rho: np.ndarray,
    density: np.ndarray,
    temperature: np.ndarray,
    er: np.ndarray,
) -> list[str]:
    """Check that a transport slice can be prescribed into a template, returning the species names.

    Parameters
    ----------
    cfg : dict
        Parsed template TOML.
    rho : numpy.ndarray
        Radial grid of the transport solution, shape ``(n_rho,)``.
    density, temperature : numpy.ndarray
        Species-resolved profiles, shape ``(n_species, n_rho)``.
    er : numpy.ndarray
        Radial electric field, shape ``(n_rho,)``.

    Returns
    -------
    list of str
        ``[species].names`` from the template, in row order.

    Raises
    ------
    KeyError
        If the template lacks ``[species].names`` or ``[geometry].n_radial``.
    ValueError
        If the species count, shapes, or radial grid disagree with the template, or the grid
        is too coarse for the readers' radial gradients. A mismatch fails this iteration
        instead of prescribing unusable profiles to the next one.
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
    if rho.size < 3:
        raise ValueError(
            f"The transport solution has {rho.size} radii; the prescribed-profiles readers need at least 3 "
            "for radial gradients"
        )

    # The prescribed block stores profile values only, so the readers rebuild the radial grid
    # from [geometry] alone. Refuse to emit a block whose grid they would rebuild differently.
    expected_rho = np.linspace(0.0, rho_edge, rho.size)
    if not np.allclose(rho, expected_rho):
        raise ValueError(
            f"The transport solution rho {rho.tolist()} does not match the template grid "
            f"linspace(0, {rho_edge}, {n_radial}) = {expected_rho.tolist()}"
        )
    return names


def _format_float_list(values: np.ndarray) -> str:
    """Render a 1D array as a single-line TOML float array using shortest round-trip literals."""
    return "[" + ", ".join(repr(float(v)) for v in np.asarray(values).ravel()) + "]"


def _format_float_table(values: np.ndarray) -> str:
    """Render a 2D array as a single-line TOML array of float arrays, one inner array per row."""
    return "[" + ", ".join(_format_float_list(row) for row in np.asarray(values)) + "]"


def _render_profiles_block(
    *,
    species_names: list[str],
    density_si: np.ndarray,
    temperature_ev: np.ndarray,
    er_kv_m: np.ndarray,
    time_value: float | None,
) -> str:
    """Render the replacement ``[profiles]`` section text.

    Parameters
    ----------
    species_names : list of str
        Row labels, in the row order of ``density_si`` and ``temperature_ev``.
    density_si : numpy.ndarray
        Number density in m^-3, shape ``(n_species, n_rho)``.
    temperature_ev : numpy.ndarray
        Temperature in eV, shape ``(n_species, n_rho)``.
    er_kv_m : numpy.ndarray
        Radial electric field in kV/m, shape ``(n_rho,)``.
    time_value : float or None
        Slice time in seconds, quoted in a provenance comment when known.

    Returns
    -------
    str
        The section text, starting at the ``[profiles]`` header, LF-terminated, ending with
        one blank line.

    Notes
    -----
    Arrays are emitted on one line each, so no value line can begin with a column-0 ``[`` and
    the section boundary stays unambiguous. The block holds only ``model``, ``density``,
    ``temperature``, and ``Er``, none of which the parse-time path-resolution step rewrites.
    """
    if time_value is None:
        provenance = [
            "# Prescribed kinetic profiles taken from the previous closed-loop iteration's transport solution."
        ]
    else:
        provenance = [
            f"# Prescribed kinetic profiles taken from the transport time slice t = {float(time_value)!r} s of the",
            "# previous closed-loop iteration's transport solution.",
        ]
    lines = [
        "[profiles]",
        *provenance,
        'model = "prescribed"',
        f"# Per-species number density in m^-3; rows follow [species].names ({', '.join(species_names)}),",
        "# columns follow the [geometry] radial grid linspace(0, rho_edge, n_radial).",
        f"density = {_format_float_table(density_si)}",
        "# Per-species temperature in eV; same row and column ordering as density.",
        f"temperature = {_format_float_table(temperature_ev)}",
        "# Radial electric field in kV/m on the same radial grid.",
        f"Er = {_format_float_list(er_kv_m)}",
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

    rho, density, temperature, er, time_value = _load_final_profiles(
        args.transport_h5,
        time_index=int(args.time_index),
    )
    species_names = _validate_against_template(cfg, rho=rho, density=density, temperature=temperature, er=er)

    block = _render_profiles_block(
        species_names=species_names,
        density_si=density * NEOPAX_DENSITY_REFERENCE_M3,
        temperature_ev=temperature * NEOPAX_TEMPERATURE_REFERENCE_EV,
        er_kv_m=er,
        time_value=time_value,
    )
    output_text = _replace_profiles_section(template_text, block)
    args.output_toml.parent.mkdir(parents=True, exist_ok=True)
    args.output_toml.write_bytes(output_text.encode("utf-8"))

    slice_label = "static slice" if time_value is None else f"slice t = {time_value} s"
    print(
        f"wrote {args.output_toml}: prescribed [profiles] for {len(species_names)} species "
        f"({', '.join(species_names)}) on {rho.size} radii from {slice_label}"
    )


if __name__ == "__main__":
    main()
