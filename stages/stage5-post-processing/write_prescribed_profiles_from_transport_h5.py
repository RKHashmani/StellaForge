#!/usr/bin/env python3
"""Write transport profiles to the [profiles] block of ``common_input.toml``.

The writer reads the final slice of ``transport_solution.h5``. It replaces [profiles] and advances
``t0`` and ``dt`` in [transport_solver]. It keeps all other template text unchanged.

The block contains state on both parts of the staggered grid:

* ``density``, ``temperature`` and ``Er`` contain [geometry].n_radial cell-center values. NEOPAX
  requires one value per cell and calculates the outer boundary.
* ``density_face``, ``temperature_face`` and ``Er_face`` contain ``n_radial + 1`` face values.
  Stages 3 and 4 sample fluxes on these faces. NEOPAX ignores these extra keys.
* ``density_grad_face`` and ``temperature_grad_face`` contain gradients on the same faces. NEOPAX
  calculates them from the centered state and the run's [boundary] settings.

The file stores density in 1e20 m^-3, temperature in keV and ``Er`` in kV/m. The writer converts
density and temperature to SI values. It keeps ``Er`` in kV/m.

The file stores face gradients per metre. The writer converts them to gradients per unit rho. It
uses the minor radius ``r_grid_half[-1] / rho_face[-1]`` for this conversion.

The writer sets ``t0`` to ``final_time`` and ``dt`` to ``next_dt``. It does not change ``t_final``.
When ``final_time`` reaches ``t_final``, it leaves both keys unchanged. Stage 5 then reports
``horizon``, and the loop stops before NEOPAX can produce an empty solution.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
from typing import Any

import numpy as np

# Stage 3, Stage 4 and this writer use the same loader and unit constants.
from common.neopax_profiles import (
    NEOPAX_DENSITY_REFERENCE_M3,
    NEOPAX_TEMPERATURE_REFERENCE_EV,
    TransportSolution,
    read_transport_solution,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


logger = logging.getLogger(__name__)


def _load_final_profiles(h5_path: Path) -> TransportSolution:
    """Read the final NEOPAX solution slice on both radial grids.

    The shared loader validates the file for every consumer. This writer always uses the last saved
    slice because ``final_time`` and ``next_dt`` describe the end of the run.

    Parameters
    ----------
    h5_path : Path
        Path to ``transport_solution.h5``.

    Returns
    -------
    TransportSolution
        Centered and face profiles in file units. The face gradients use unit rho, and the object
        includes the run clock.

    Raises
    ------
    KeyError
        If a required dataset is absent.
    ValueError
        If the layout, values, grids or clock are invalid.
    """
    return read_transport_solution(h5_path, time_index=-1)


def _validate_against_template(cfg: dict[str, Any], *, slice_: TransportSolution) -> list[str]:
    """Check a transport slice against a template and return the species names.

    Parameters
    ----------
    cfg : dict
        Parsed template TOML.
    slice_ : TransportSolution
        The transport slice to prescribe, on both radial grids.

    Returns
    -------
    list of str
        [species].names in row order.

    Raises
    ------
    KeyError
        If the template lacks [species].names or [geometry].n_radial.
    ValueError
        If the species, shapes or radial grids disagree, or the grid is too coarse.

    Notes
    -----
    [profiles] does not store radial coordinates. Each reader rebuilds both grids from [geometry].
    This function checks the solution against those grids.
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
    # NEOPAX calculates the edge value from the two outermost centers. This requires three faces.
    if rho.size + 1 < 3:
        raise ValueError(
            f"The transport solution has {rho.size} cells, giving {rho.size + 1} faces; the outer "
            "boundary closure needs at least 3 faces"
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

    for label, arr in (
        ("density_faces", slice_.density_face),
        ("temperature_faces", slice_.temperature_face),
        ("density_grad_faces", slice_.density_grad_face),
        ("temperature_grad_faces", slice_.temperature_grad_face),
    ):
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


def _render_profiles_block(*, species_names: list[str], slice_: TransportSolution) -> str:
    """Render the replacement [profiles] block and convert the slice to SI values.

    Parameters
    ----------
    species_names : list of str
        Row labels, in the row order of the slice's profile arrays.
    slice_ : TransportSolution
        Slice in file units. Its face gradients are already per unit rho.

    Returns
    -------
    str
        LF text that starts with [profiles] and ends with one blank line.

    Notes
    -----
    Each array stays on one line. A value line cannot look like a section header. NEOPAX reads the
    centered arrays and ignores the face arrays.
    """
    if slice_.time_value is None:
        provenance = [
            "# Prescribed profiles from the previous closed-loop transport solution."
        ]
    else:
        provenance = [
            f"# Prescribed profiles from transport time slice t = {float(slice_.time_value)!r} s.",
            "# The previous closed-loop transport solution supplied this slice.",
        ]
    lines = [
        "[profiles]",
        *provenance,
        'model = "prescribed"',
        "# Number density by species in m^-3.",
        f"# Rows follow [species].names: {', '.join(species_names)}.",
        "# Columns follow the [geometry] cell centers.",
        "# The centers are midpoints of linspace(0, rho_edge, n_radial + 1).",
        "# NEOPAX reads one value per cell and calculates the boundary face.",
        f"density = {_format_float_table(slice_.density * NEOPAX_DENSITY_REFERENCE_M3)}",
        "# Temperature by species in eV. Rows and columns follow density.",
        f"temperature = {_format_float_table(slice_.temperature * NEOPAX_TEMPERATURE_REFERENCE_EV)}",
        "# Radial electric field in kV/m on the same cell-center grid.",
        f"Er = {_format_float_list(slice_.er)}",
        "# The same profiles on n_radial + 1 cell faces.",
        "# The faces are linspace(0, rho_edge, n_radial + 1).",
        "# Stages 3 and 4 sample fluxes on this grid.",
        "# The transport solution supplies the face state.",
        "# These values include the run's [boundary] conditions.",
        f"density_face = {_format_float_table(slice_.density_face * NEOPAX_DENSITY_REFERENCE_M3)}",
        f"temperature_face = {_format_float_table(slice_.temperature_face * NEOPAX_TEMPERATURE_REFERENCE_EV)}",
        f"Er_face = {_format_float_list(slice_.er_face)}",
        "# Face gradients in m^-3 and eV per unit rho.",
        "# NEOPAX calculates them from the centered state and [boundary] conditions.",
        "# The transport solution supplies these gradients on the rho grid.",
        f"density_grad_face = {_format_float_table(slice_.density_grad_face * NEOPAX_DENSITY_REFERENCE_M3)}",
        "temperature_grad_face = "
        + _format_float_table(slice_.temperature_grad_face * NEOPAX_TEMPERATURE_REFERENCE_EV),
        "",
        "",
    ]
    return "\n".join(lines)


def _section_span(text: str, section: str) -> tuple[int, int] | None:
    """Locate one TOML section and return its character span.

    Parameters
    ----------
    text : str
        The full TOML document.
    section : str
        Bare section name, without its brackets.

    Returns
    -------
    tuple of int, or None
        Half-open span for the header and body. Return ``None`` when no column-zero header exists.
    """
    header = re.search(rf"^\[{re.escape(section)}\][ \t]*(?:#[^\n]*)?\r?$", text, flags=re.MULTILINE)
    if header is None:
        return None
    tail = text[header.end() :]
    next_section = re.search(r"^\[", tail, flags=re.MULTILINE)
    end = len(text) if next_section is None else header.end() + next_section.start()
    return header.start(), end


def _replace_profiles_section(text: str, block: str) -> str:
    """Replace [profiles] with ``block`` and keep all other text unchanged.

    Parameters
    ----------
    text : str
        The full template text.
    block : str
        LF replacement text. The function converts it to the template's line endings.

    Returns
    -------
    str
        Template with [profiles] replaced.

    Raises
    ------
    ValueError
        If the text has no column-zero [profiles] header.

    Notes
    -----
    The span ends at the next column-zero section header or at the end of the file.
    """
    span = _section_span(text, "profiles")
    if span is None:
        raise ValueError("The template has no [profiles] section to replace")
    start, end = span
    newline = "\r\n" if "\r\n" in text else "\n"
    return text[:start] + block.replace("\n", newline) + text[end:]


def _replace_section_key(section_text: str, *, section: str, key: str, value: str) -> str:
    """Replace one column-zero key value in one section.

    Parameters
    ----------
    section_text : str
        Text of a single section, as delimited by :func:`_section_span`.
    section : str
        Bare section name for error messages.
    key : str
        Key to rewrite, matched only at the start of a line.
    value : str
        Text placed after ``=`` and before any trailing comment. Spacing, comments and line endings
        remain unchanged.

    Returns
    -------
    str
        The section text with that one value rewritten.

    Raises
    ------
    KeyError
        If the key does not appear exactly once at column zero.
    """
    pattern = re.compile(rf"^({re.escape(key)}[ \t]*=[ \t]*)[^\r\n#]*?(?=[ \t]*(?:#|\r?$))", flags=re.MULTILINE)
    rewritten, count = pattern.subn(lambda match: match.group(1) + value, section_text)
    if count != 1:
        raise KeyError(f"The template's [{section}] section assigns '{key}' {count} times, expected exactly once")
    return rewritten


def _advance_transport_clock(text: str, cfg: dict[str, Any], *, final_time: float, next_dt: float) -> str:
    """Advance ``t0`` and ``dt`` in [transport_solver] to the solution clock.

    Parameters
    ----------
    text : str
        The full TOML document to rewrite.
    cfg : dict
        Parsed template TOML with [transport_solver].t_final.
    final_time : float
        Time the transport run reached, written to ``t0``.
    next_dt : float
        Step the transport run was about to take, written to ``dt``.

    Returns
    -------
    str
        Updated document. Return the original document when the run reaches ``t_final``.

    Raises
    ------
    KeyError
        If [transport_solver] is incomplete or assigns ``t0`` or ``dt`` more than once.
    ValueError
        If ``final_time`` did not advance beyond the starting ``t0``.

    Notes
    -----
    ``t_final`` is the absolute end time and does not move. NEOPAX accepts ``t0 >= t_final`` and can
    return an empty solution. This function prevents another iteration after the horizon.
    """
    solver_cfg = cfg.get("transport_solver")
    if not isinstance(solver_cfg, dict) or "t_final" not in solver_cfg:
        raise KeyError(
            "The template must define [transport_solver].t_final, which bounds how far the transport "
            "clock may be advanced"
        )
    t_final = float(solver_cfg["t_final"])
    t0 = float(solver_cfg.get("t0", 0.0))
    if final_time <= t0:
        raise ValueError(
            f"the transport run reached final_time {final_time!r} from [transport_solver].t0 {t0!r}, so it "
            "integrated nothing. Writing that back as the next iteration's t0 would stall the loop at this "
            "instant while the profiles kept changing under it."
        )
    if final_time >= t_final:
        logger.info(
            "transport horizon reached, final_time %r against t_final %r, so [transport_solver].t0 and dt "
            "were not advanced. The emitted file seeds no further iteration, because the loop stops here.",
            final_time, t_final,
        )
        return text

    span = _section_span(text, "transport_solver")
    if span is None:
        raise KeyError("The template has no [transport_solver] section header to advance the transport clock in")
    start, end = span
    section_text = text[start:end]
    for key, value in (("t0", final_time), ("dt", next_dt)):
        section_text = _replace_section_key(section_text, section="transport_solver", key=key, value=repr(value))
    logger.info("advanced [transport_solver] to t0 = %r, dt = %r, against t_final = %r", final_time, next_dt, t_final)
    return text[:start] + section_text + text[end:]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
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
    args = parser.parse_args()

    template_text = args.template_toml.read_bytes().decode("utf-8")
    cfg = tomllib.loads(template_text)

    slice_ = _load_final_profiles(args.transport_h5)
    species_names = _validate_against_template(cfg, slice_=slice_)

    block = _render_profiles_block(species_names=species_names, slice_=slice_)
    output_text = _replace_profiles_section(template_text, block)
    output_text = _advance_transport_clock(output_text, cfg, final_time=slice_.final_time, next_dt=slice_.next_dt)
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
