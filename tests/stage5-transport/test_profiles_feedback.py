"""Tests for the Stage 5 prescribed-profiles feedback writer.

These exercise the pure helpers and the CLI of ``write_prescribed_profiles_from_transport_h5.py``,
loaded by path. The script closes the design loop: it reads one time slice of a NEOPAX
``transport_solution.h5`` and emits a copy of a ``common_input.toml`` template whose whole
``[profiles]`` section is replaced by a prescribed block holding that slice. Two properties carry
the contract: the unit conversion out of the file's units (1e20 m^-3 -> m^-3, keV -> eV, kV/m
unchanged) and the byte-identity of everything outside ``[profiles]``, which is what makes the
emitted file a drop-in template for the next iteration. NEOPAX is never imported, so no solver is
needed.
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pytest
from numpy.testing import assert_array_equal

from tests.helpers.stage_import import load_stage_module
from tests.helpers.synthetic import write_transport_solution

writer = load_stage_module("stages/stage5-post-processing/write_prescribed_profiles_from_transport_h5.py")

REPO_ROOT = Path(__file__).resolve().parents[2]
# The committed quick_run template: 3 species (e, D, T), [geometry].n_radial = 5 and no rho_edge, so
# a slice feeds back only on the grid linspace(0, 1, 5). It is CRLF-terminated and its [profiles]
# section is followed by further sections, which is the layout every tracked-template test relies on.
TRACKED_TEMPLATE = REPO_ROOT / "inputs/quick_run/common_input.toml"


def _face_grid(n_radial: int = 5, rho_edge: float = 1.0) -> np.ndarray:
    """NEOPAX's face grid: the ``n_radial + 1`` cell boundaries spanning ``[0, rho_edge]``."""
    return np.linspace(0.0, rho_edge, n_radial + 1)


def _center_grid(n_radial: int = 5, rho_edge: float = 1.0) -> np.ndarray:
    """NEOPAX's cell centers: face-grid midpoints, so neither 0 nor ``rho_edge`` is a member."""
    faces = _face_grid(n_radial, rho_edge)
    return 0.5 * (faces[:-1] + faces[1:])


def _make_slice(offset: float = 0.0, n_species: int = 3, n_rho: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one transport slice in file units: density in 1e20 m^-3, temperature in keV, Er in kV/m.

    Every entry differs from every other so a transposed, reordered, or wrongly sliced array cannot
    pass an equality assertion by coincidence. ``offset`` separates consecutive time slices.
    """
    ramp = np.arange(n_species * n_rho, dtype=float).reshape(n_species, n_rho)
    return 1.0 + 0.1 * ramp + offset, 2.0 + 0.2 * ramp + offset, np.linspace(-1.0, 3.0, n_rho) + offset


def _make_face_slice(
    offset: float = 0.0, n_species: int = 3, n_rho: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Face-grid counterpart of :func:`_make_slice`, on ``n_rho + 1`` radii.

    The base values are deliberately disjoint from ``_make_slice``'s, so a reader that takes the
    cell-centered arrays where it should take the face ones cannot pass by coincidence.
    """
    ramp = np.arange(n_species * (n_rho + 1), dtype=float).reshape(n_species, n_rho + 1)
    return 5.0 + 0.1 * ramp + offset, 7.0 + 0.2 * ramp + offset, np.linspace(-2.0, 4.0, n_rho + 1) + offset


def _auto_faces(density: np.ndarray, n_rho: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a face triple whose rank matches ``density``, so only the centered arrays are on trial.

    The rank-consistency tests deliberately break one centered array; they should not also have to
    restate a matching set of face arrays just to reach the check they are about.
    """
    arr = np.asarray(density)
    if arr.ndim != 3:
        return _make_face_slice(n_rho=n_rho)
    slices = [_make_face_slice(offset=float(i), n_rho=n_rho) for i in range(arr.shape[0])]
    return tuple(np.stack([s[k] for s in slices]) for k in range(3))  # type: ignore[return-value]


def _write_h5(
    path: Path,
    *,
    rho: np.ndarray,
    density: np.ndarray,
    temperature: np.ndarray,
    er: np.ndarray,
    ts: Sequence[float] | None = None,
    faces: tuple[np.ndarray, np.ndarray, np.ndarray] | None | str = "auto",
    rho_face: np.ndarray | None = None,
) -> Path:
    """Write a transport solution, appending the ``ts`` time axis the shared builder cannot write.

    ``faces`` is a ``(density, temperature, er)`` triple on the face grid. The default ``"auto"``
    derives a rank-matching triple from ``density``; passing ``None`` omits the face datasets
    entirely, which is how a solution from a NEOPAX predating the staggered grid is simulated.
    """
    if faces == "auto":
        faces = _auto_faces(density, n_rho=int(np.asarray(rho).size))
    face_density, face_temperature, face_er = (None, None, None) if faces is None else faces
    write_transport_solution(
        path,
        rho=rho,
        density=density,
        temperature=temperature,
        er=er,
        rho_face=_face_grid(int(np.asarray(rho).size)) if rho_face is None and faces is not None else rho_face,
        density_face=face_density,
        temperature_face=face_temperature,
        er_face=face_er,
    )
    if ts is not None:
        with h5py.File(path, "a") as f:
            f.create_dataset("ts", data=np.asarray(ts, dtype=float))
    return path


def _make_transport_slice(
    n_radial: int = 5, rho_edge: float = 1.0, n_species: int = 3, **overrides: Any
) -> Any:
    """Build a ``TransportSlice`` consistent with a template of ``n_radial`` cells.

    Any field can be overridden to put exactly one of ``_validate_against_template``'s checks on
    trial while every other check still passes.
    """
    density, temperature, er = _make_slice(n_species=n_species, n_rho=n_radial)
    density_face, temperature_face, er_face = _make_face_slice(n_species=n_species, n_rho=n_radial)
    fields: dict[str, Any] = {
        "rho": _center_grid(n_radial, rho_edge),
        "density": density,
        "temperature": temperature,
        "er": er,
        "rho_face": _face_grid(n_radial, rho_edge),
        "density_face": density_face,
        "temperature_face": temperature_face,
        "er_face": er_face,
        "time_value": None,
    }
    return writer.TransportSlice(**{**fields, **overrides})


def _write_time_resolved(path: Path, *, n_times: int = 3, ts: Sequence[float] | None = None) -> Path:
    """Write a 3-D (n_time, n_species, n_rho) solution whose slice ``i`` is ``_make_slice(offset=i)``."""
    slices = [_make_slice(offset=float(i)) for i in range(n_times)]
    face_slices = [_make_face_slice(offset=float(i)) for i in range(n_times)]
    return _write_h5(
        path,
        rho=_center_grid(),
        density=np.stack([s[0] for s in slices]),
        temperature=np.stack([s[1] for s in slices]),
        er=np.stack([s[2] for s in slices]),
        ts=ts,
        rho_face=_face_grid(),
        faces=(
            np.stack([s[0] for s in face_slices]),
            np.stack([s[1] for s in face_slices]),
            np.stack([s[2] for s in face_slices]),
        ),
    )


def _write_static(path: Path) -> Path:
    """Write a 2-D (n_species, n_rho) solution holding a single static slice with no time axis."""
    density, temperature, er = _make_slice()
    return _write_h5(
        path,
        rho=_center_grid(),
        density=density,
        temperature=temperature,
        er=er,
        rho_face=_face_grid(),
        faces=_make_face_slice(),
    )


def _run_writer(
    monkeypatch: pytest.MonkeyPatch,
    h5_path: Path,
    template: Path,
    output: Path,
    *extra_args: str,
) -> str:
    """Run the CLI end to end through ``main()`` with a patched argv and return the emitted text."""
    monkeypatch.setattr(
        "sys.argv",
        ["write_prescribed_profiles_from_transport_h5", str(h5_path), str(template),
         "--output-toml", str(output), *extra_args],
    )
    writer.main()
    return output.read_bytes().decode("utf-8")


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split a TOML document at column-0 section headers into ``(header line, section text)`` pairs.

    The first pair holds whatever precedes the first header, under an empty header name. Section
    text keeps its own terminators so two documents compare byte for byte.
    """
    starts = [m.start() for m in re.finditer(r"^\[", text, flags=re.MULTILINE)]
    sections: list[tuple[str, str]] = []
    for start, end in zip([0, *starts], [*starts, len(text)], strict=True):
        chunk = text[start:end]
        sections.append((chunk.splitlines()[0] if chunk.startswith("[") else "", chunk))
    return sections


def _template_cfg(n_radial: int = 5, names: Sequence[str] = ("e", "D", "T"), **geometry: float) -> dict:
    """The minimal parsed-template shape ``_validate_against_template`` consumes."""
    return {"species": {"names": list(names)}, "geometry": {"n_radial": n_radial, **geometry}}


# --- main(): happy path against the tracked template ---

# The end-to-end contract in one test. Running the CLI against the committed quick_run template and a
# three-slice transport solution must produce a parseable TOML whose [profiles] section holds exactly
# model/density/temperature/Er, with the last slice converted out of the file's units: density scaled
# by 1e20 into m^-3, temperature by 1e3 into eV, Er passed through in kV/m. Equality is exact rather
# than approximate because the writer emits shortest round-trip float literals, so any drift in the
# formatting would show up here. The final assertion pins the row labelling: rows stay in the
# transport solution's own order and are documented against [species].names in template order.
def test_main_prescribes_the_final_slice_of_the_tracked_template(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h5_path = _write_time_resolved(tmp_path / "transport_solution.h5", n_times=3, ts=[0.0, 1.0e-6, 2.5e-6])
    text = _run_writer(monkeypatch, h5_path, TRACKED_TEMPLATE, tmp_path / "out.toml")

    cfg = tomllib.loads(text)
    assert cfg["profiles"]["model"] == "prescribed"

    density, temperature, er = _make_slice(offset=2.0)  # slice 2 of 3 is the default --time-index -1
    assert_array_equal(np.asarray(cfg["profiles"]["density"]), density * 1.0e20)
    assert_array_equal(np.asarray(cfg["profiles"]["temperature"]), temperature * 1.0e3)
    assert_array_equal(np.asarray(cfg["profiles"]["Er"]), er)
    assert f"rows follow [species].names ({', '.join(cfg['species']['names'])})" in text


# --- main(): the face arrays Stages 3 and 4 read ---

# NEOPAX evolves its state on cell centers, but Stages 3 and 4 sample fluxes on the faces, so the
# emitted block has to carry both grids. The face values are copied from the solution's own
# `*_faces` datasets rather than re-derived here: NEOPAX builds them under that run's boundary
# models, so an extrapolation invented at this end would disagree with the solver wherever the outer
# boundary is anything but a plain Dirichlet. This asserts the face keys are emitted with n_radial+1
# columns, converted into the same SI units as the centered arrays. The face fixtures use disjoint
# base values from the centered ones, so copying the wrong grid across cannot pass.
def test_emitted_block_carries_the_face_arrays_from_the_solution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h5_path = _write_time_resolved(tmp_path / "transport_solution.h5", n_times=3, ts=[0.0, 1.0e-6, 2.5e-6])
    text = _run_writer(monkeypatch, h5_path, TRACKED_TEMPLATE, tmp_path / "out.toml")

    cfg = tomllib.loads(text)
    assert set(cfg["profiles"]) == {
        "model", "density", "temperature", "Er", "density_face", "temperature_face", "Er_face",
    }

    density_face, temperature_face, er_face = _make_face_slice(offset=2.0)
    assert_array_equal(np.asarray(cfg["profiles"]["density_face"]), density_face * 1.0e20)
    assert_array_equal(np.asarray(cfg["profiles"]["temperature_face"]), temperature_face * 1.0e3)
    assert_array_equal(np.asarray(cfg["profiles"]["Er_face"]), er_face)
    # [geometry].n_radial = 5 cells, so 6 faces against 5 centers.
    assert np.asarray(cfg["profiles"]["density_face"]).shape[1] == len(cfg["profiles"]["density"][0]) + 1


# A transport solution written by a NEOPAX predating the staggered grid carries no `*_faces`
# datasets. Emitting a block without the face arrays would defer the failure to Stage 3 or Stage 4 of
# the next iteration, far from its cause, so the writer refuses here and names the missing datasets.
def test_solution_without_face_datasets_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    density, temperature, er = _make_slice()
    h5_path = _write_h5(
        tmp_path / "transport_solution.h5",
        rho=_center_grid(), density=density, temperature=temperature, er=er,
        faces=None,  # a solution predating the staggered grid
    )
    monkeypatch.setattr(
        "sys.argv",
        ["write_prescribed_profiles_from_transport_h5", str(h5_path), str(TRACKED_TEMPLATE),
         "--output-toml", str(tmp_path / "out.toml")],
    )
    with pytest.raises(KeyError, match=r"density_faces"):
        writer.main()


# --- main(): everything outside [profiles] survives ---

# The replacement is scoped to the [profiles] section, and this is the assertion that proves it. Both
# documents are split at column-0 headers and every section other than [profiles] must come back byte
# for byte, in the same order. The [sources] section is called out explicitly because it carries its
# own `temperature = [...]` key: a key-name-based rewrite instead of a section-scoped one would
# clobber it, and nothing else in the template would notice. The line-ending assertion pins that the
# emitted block is reterminated to the template's own endings rather than leaving the file mixed.
def test_sections_outside_profiles_are_byte_identical(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    template_bytes = TRACKED_TEMPLATE.read_bytes()
    template_text = template_bytes.decode("utf-8")
    out_path = tmp_path / "out.toml"
    text = _run_writer(monkeypatch, _write_static(tmp_path / "transport_solution.h5"), TRACKED_TEMPLATE, out_path)

    template_sections = _split_sections(template_text)
    output_sections = _split_sections(text)
    assert [name for name, _ in output_sections] == [name for name, _ in template_sections]
    for (name, expected), (_, actual) in zip(template_sections, output_sections, strict=True):
        if name != "[profiles]":
            assert actual == expected, f"section {name!r} was rewritten"

    sources_lines = [line for line in template_text.splitlines() if line.startswith("temperature = [")]
    assert len(sources_lines) == 1  # the [sources] source-term list; the template's [profiles] has no such key
    assert sources_lines[0] in text

    newline = b"\r\n" if b"\r\n" in template_bytes else b"\n"
    out_bytes = out_path.read_bytes()
    assert out_bytes.count(b"\n") == out_bytes.count(newline)  # every terminator matches the template's


# --- main(): idempotency ---

# The emitted file is itself a valid template for the next iteration, so feeding it back with the same
# transport solution must reproduce it exactly. This catches a block that fails to round-trip through
# its own section-boundary detection, for instance an array literal that wraps onto a line starting
# with "[" or a provenance comment that accumulates on every pass.
def test_rewriting_its_own_output_is_byte_identical(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    h5_path = _write_time_resolved(tmp_path / "transport_solution.h5", n_times=3, ts=[0.0, 1.0e-6, 2.5e-6])
    first = tmp_path / "first.toml"
    second = tmp_path / "second.toml"
    _run_writer(monkeypatch, h5_path, TRACKED_TEMPLATE, first)
    _run_writer(monkeypatch, h5_path, first, second)
    assert second.read_bytes() == first.read_bytes()


# --- main(): [profiles] last, LF-terminated template ---

def _write_lf_template(path: Path) -> Path:
    """Write a minimal LF-terminated template whose final section is ``[profiles]``."""
    lines = [
        "# scratch template", "", "[geometry]", "n_radial = 5", "",
        "[species]", 'names = ["e", "D", "T"]', "",
        "[sources]", 'temperature = ["dt_reaction"]', "",
        "[profiles]", 'model = "standard_analytical"', "n0 = 4.21", "",
    ]
    path.write_bytes("\n".join(lines).encode("utf-8"))
    return path


# When [profiles] is the final section there is no following column-0 header to bound the replaced
# span, so the writer must run it to end of file. This also covers the LF branch of the line-ending
# logic that the CRLF tracked template cannot reach, so the output must stay free of carriage
# returns, and the absence of "n0" proves the old analytical keys were removed rather than appended to.
def test_profiles_as_last_section_of_an_lf_template(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    template = _write_lf_template(tmp_path / "template.toml")
    out_path = tmp_path / "out.toml"
    text = _run_writer(monkeypatch, _write_static(tmp_path / "transport_solution.h5"), template, out_path)

    cfg = tomllib.loads(text)
    assert cfg["profiles"]["model"] == "prescribed"
    assert cfg["sources"]["temperature"] == ["dt_reaction"]  # the preceding section is untouched
    assert b"\r" not in out_path.read_bytes()
    assert "n0 = 4.21" not in text


# --- --time-index ---

# A positive index addresses the time axis directly, so slice 1 of three must be prescribed rather
# than the default last slice. The provenance comment must quote that slice's own ts entry.
def test_time_index_selects_the_requested_slice(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    h5_path = _write_time_resolved(tmp_path / "transport_solution.h5", n_times=3, ts=[0.0, 1.0e-6, 2.5e-6])
    text = _run_writer(monkeypatch, h5_path, TRACKED_TEMPLATE, tmp_path / "out.toml", "--time-index", "1")

    profiles = tomllib.loads(text)["profiles"]
    density, temperature, er = _make_slice(offset=1.0)
    assert_array_equal(np.asarray(profiles["density"]), density * 1.0e20)
    assert_array_equal(np.asarray(profiles["temperature"]), temperature * 1.0e3)
    assert_array_equal(np.asarray(profiles["Er"]), er)
    assert "transport time slice t = 1e-06 s" in text


# `_resolve_time_index` is the only place negative indices are normalized, so it is pinned directly:
# -1 is the last slice, -n the first, and a non-negative index passes through unchanged.
def test_resolve_time_index_counts_back_from_the_last_slice() -> None:
    h5_path = Path("transport_solution.h5")
    assert writer._resolve_time_index(4, time_index=-1, h5_path=h5_path) == 3
    assert writer._resolve_time_index(4, time_index=-4, h5_path=h5_path) == 0
    assert writer._resolve_time_index(4, time_index=2, h5_path=h5_path) == 2


# Out-of-range indices must fail loudly instead of silently clamping to an endpoint, in both
# directions: 3 is one past the last slice of a three-slice file and -4 one before the first.
@pytest.mark.parametrize("time_index", [3, -4])
def test_time_index_out_of_range_raises(tmp_path: Path, time_index: int) -> None:
    h5_path = _write_time_resolved(tmp_path / "transport_solution.h5", n_times=3)
    with pytest.raises(ValueError, match="out of range for the 3 time slices"):
        writer._load_final_profiles(h5_path, time_index=time_index)


# A 2-D file holds one static slice and no time axis, so the default -1 reads it while any other
# index, positive or negative, is meaningless and must be rejected rather than quietly ignored.
def test_static_file_accepts_only_the_default_time_index(tmp_path: Path) -> None:
    h5_path = _write_static(tmp_path / "static.h5")
    density, temperature, er = _make_slice()
    density_face, temperature_face, er_face = _make_face_slice()

    loaded = writer._load_final_profiles(h5_path, time_index=-1)
    assert_array_equal(loaded.rho, _center_grid())
    assert_array_equal(loaded.density, density)
    assert_array_equal(loaded.temperature, temperature)
    assert_array_equal(loaded.er, er)
    assert_array_equal(loaded.rho_face, _face_grid())
    assert_array_equal(loaded.density_face, density_face)
    assert_array_equal(loaded.temperature_face, temperature_face)
    assert_array_equal(loaded.er_face, er_face)
    assert loaded.time_value is None

    for bad in (0, 1, -2):
        with pytest.raises(ValueError, match="stores a single static profile slice"):
            writer._load_final_profiles(h5_path, time_index=bad)


# The provenance comment names the slice time only when the file records a ts axis. A time-resolved file without ts
# falls back to the bare provenance line, which pins that ts presence drives the comment rather than array rank. The
# positive with-ts case is pinned by test_time_index_selects_the_requested_slice.
def test_time_resolved_file_without_ts_omits_slice_time(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    no_ts = _write_time_resolved(tmp_path / "no_ts.h5", n_times=2)
    text = _run_writer(monkeypatch, no_ts, TRACKED_TEMPLATE, tmp_path / "no_ts.toml")
    assert "time slice t =" not in text
    assert "# Prescribed kinetic profiles taken from the previous closed-loop iteration" in text


# --- _load_final_profiles: file-level error contract ---

# A transport solution written by an older or different solver may simply lack a dataset. That is a
# missing-key condition, so it must surface as KeyError naming the dataset, not as an index error
# deeper in the slicing.
def test_missing_dataset_raises_key_error(tmp_path: Path) -> None:
    _, temperature, er = _make_slice()
    density_face, temperature_face, er_face = _make_face_slice()
    path = write_transport_solution(tmp_path / "no_density.h5", rho=_center_grid(),
                                    temperature=temperature, er=er, rho_face=_face_grid(),
                                    density_face=density_face, temperature_face=temperature_face,
                                    er_face=er_face)
    with pytest.raises(KeyError, match="missing the required transport datasets: density"):
        writer._load_final_profiles(path, time_index=-1)


# A NaN or infinity anywhere in the selected slice would be written straight into the next
# iteration's template and only fail much later inside a solver, so it is rejected at read time with
# the offending dataset named.
def test_non_finite_values_in_the_slice_raise(tmp_path: Path) -> None:
    density, temperature, er = _make_slice()
    density[1, 2] = np.nan
    path = _write_h5(tmp_path / "nan.h5", rho=_center_grid(), density=density,
                     temperature=temperature, er=er)
    with pytest.raises(ValueError, match="dataset 'density' holds non-finite values"):
        writer._load_final_profiles(path, time_index=-1)


# Density must be 2-D or 3-D; anything else leaves the slicing undefined and must be reported against
# the density dataset rather than producing a mis-shaped block.
def test_density_of_unsupported_rank_raises(tmp_path: Path) -> None:
    rho = _center_grid()
    path = _write_h5(tmp_path / "flat.h5", rho=rho, density=rho, temperature=rho, er=rho)
    with pytest.raises(ValueError, match="dataset 'density' must be 2D"):
        writer._load_final_profiles(path, time_index=-1)


# Er shares density's time and radius axes but carries no species axis, so its rank must always be
# exactly one lower. Both a temperature that disagrees with density and an Er that carries a spurious
# species axis are rank inconsistencies and must be caught before any slicing happens.
@pytest.mark.parametrize("broken", ["temperature", "Er"])
def test_inconsistent_dataset_ranks_raise(tmp_path: Path, broken: str) -> None:
    rho = _center_grid()
    density, temperature, er = _make_slice()
    if broken == "temperature":
        # 3-D density against a 2-D temperature.
        arrays = dict(density=np.stack([density, density]), temperature=temperature, er=np.stack([er, er]))
    else:
        # 2-D density against an Er that wrongly carries the species axis.
        arrays = dict(density=density, temperature=temperature, er=np.stack([er, er, er]))
    path = _write_h5(tmp_path / f"{broken}.h5", rho=rho, **arrays)
    with pytest.raises(ValueError, match="inconsistent dataset ranks"):
        writer._load_final_profiles(path, time_index=-1)


# All three datasets are 3-D here but disagree on the length of the time axis, so a single index
# cannot address the same instant in each. The mismatch is reported instead of indexing whichever
# array happens to be long enough.
def test_disagreeing_time_axes_raise(tmp_path: Path) -> None:
    rho = _center_grid()
    density, temperature, er = _make_slice()
    path = _write_h5(tmp_path / "axes.h5", rho=rho, density=np.stack([density, density]),
                     temperature=np.stack([temperature] * 3), er=np.stack([er, er]))
    with pytest.raises(ValueError, match="time axes disagree"):
        writer._load_final_profiles(path, time_index=-1)


# --- _validate_against_template ---

# The prescribed block stores profile values only, so the readers rebuild both radial grids from
# [geometry] alone. A slice gridded for rho_edge = 0.7 is therefore valid against a template
# declaring that and invalid against one defaulting to 1.0, even though both declare n_radial = 5.
def test_rho_grid_must_match_the_template_geometry() -> None:
    assert writer._validate_against_template(
        _template_cfg(rho_edge=0.7), slice_=_make_transport_slice(rho_edge=0.7)
    ) == ["e", "D", "T"]
    with pytest.raises(ValueError, match="does not match the template's cell centers"):
        writer._validate_against_template(_template_cfg(), slice_=_make_transport_slice(rho_edge=0.7))


# The centers and the faces are checked separately, so a solution whose centers happen to land right
# but whose faces do not is still rejected. Without this the flux grid Stages 3 and 4 sample on could
# drift away from the one NEOPAX interpolates onto with nothing reporting it.
def test_face_grid_must_match_the_template_geometry() -> None:
    with pytest.raises(ValueError, match="rho_face .* does not match the template's faces"):
        writer._validate_against_template(
            _template_cfg(), slice_=_make_transport_slice(rho_face=np.linspace(0.0, 0.9, 6))
        )


# The face arrays carry one more radial entry than the centered ones. A face array sized like the
# centers would leave the outermost face unsampled, so the widths are pinned explicitly.
def test_face_arrays_must_carry_one_more_radius_than_the_centers() -> None:
    slice_ = _make_transport_slice()
    with pytest.raises(ValueError, match=r"density_faces has shape .* expected \(3, 6\)"):
        writer._validate_against_template(
            _template_cfg(), slice_=_make_transport_slice(density_face=slice_.density_face[:, :-1])
        )
    with pytest.raises(ValueError, match=r"Er_faces has shape .* expected \(6,\)"):
        writer._validate_against_template(
            _template_cfg(), slice_=_make_transport_slice(er_face=slice_.er_face[:-1])
        )


# The block has no way to name its own rows or its own grid, so a template missing either piece of
# metadata cannot be filled in. Both are missing-key conditions and must raise KeyError naming the
# absent table key rather than defaulting to something plausible.
@pytest.mark.parametrize(
    ("missing", "message"),
    [("species", r"\[species\]\.names"), ("geometry", r"\[geometry\]\.n_radial")],
)
def test_template_missing_required_tables_raises_key_error(missing: str, message: str) -> None:
    cfg = _template_cfg()
    del cfg[missing]
    with pytest.raises(KeyError, match=message):
        writer._validate_against_template(cfg, slice_=_make_transport_slice())


# A slice with more species rows than the template names would emit rows nobody can identify, so the
# counts must agree exactly and the error must quote both sides.
def test_species_count_mismatch_raises() -> None:
    with pytest.raises(ValueError, match=r"3 species but the template \[species\]\.names lists 2"):
        writer._validate_against_template(_template_cfg(names=("e", "D")), slice_=_make_transport_slice())


# The readers size the prescribed arrays from [geometry].n_radial, so a slice on a different number of
# radii must fail here rather than being silently truncated or broadcast downstream.
def test_n_radial_mismatch_raises() -> None:
    with pytest.raises(ValueError, match=r"5 radii but the template \[geometry\]\.n_radial is 7"):
        writer._validate_against_template(_template_cfg(n_radial=7), slice_=_make_transport_slice())


# The three arrays must agree with each other before they are written as one block: density must be
# species-resolved, temperature must match density exactly, and Er must sit on the same radial grid
# as rho. Each mismatch names the array it is about.
def test_slice_shape_mismatches_raise() -> None:
    slice_ = _make_transport_slice()
    with pytest.raises(ValueError, match="species-resolved density slice"):
        writer._validate_against_template(_template_cfg(), slice_=_make_transport_slice(density=slice_.density[0]))
    with pytest.raises(ValueError, match="temperature shape"):
        writer._validate_against_template(
            _template_cfg(), slice_=_make_transport_slice(temperature=slice_.temperature[:, :4])
        )
    with pytest.raises(ValueError, match="Er shape"):
        writer._validate_against_template(_template_cfg(), slice_=_make_transport_slice(er=slice_.er[:4]))


# The prescribed-profiles readers difference the arrays radially, which needs at least three points.
# A two-radius template and a matching two-radius slice are self-consistent and pass every other
# check, so this guard is the only thing standing between them and an unusable feedback file.
def test_fewer_than_three_radii_raises() -> None:
    with pytest.raises(ValueError, match="at least 3 faces"):
        writer._validate_against_template(_template_cfg(n_radial=1), slice_=_make_transport_slice(n_radial=1))


# Two cells give three faces, which is enough to difference.
def test_two_cells_pass() -> None:
    assert writer._validate_against_template(
        _template_cfg(n_radial=2), slice_=_make_transport_slice(n_radial=2)
    ) == ["e", "D", "T"]


# --- _replace_profiles_section ---

# A template with nothing to replace is a configuration error, not something to append a section to,
# because the emitted file would then carry whatever profile model the reader defaults to.
def test_template_without_a_profiles_section_raises() -> None:
    with pytest.raises(ValueError, match=r"no \[profiles\] section"):
        writer._replace_profiles_section('[general]\r\nmode = "transport"\r\n', "[profiles]\n")


# The header line may carry a trailing comment, which must not stop it being recognised; the comment
# goes away with the rest of the replaced section and the following section is left where it was.
def test_profiles_header_with_a_trailing_comment_is_replaced() -> None:
    text = '[profiles] # analytical\nmodel = "standard_analytical"\n\n[sources]\nx = 1\n'
    out = writer._replace_profiles_section(text, '[profiles]\nmodel = "prescribed"\n\n')
    assert out == '[profiles]\nmodel = "prescribed"\n\n[sources]\nx = 1\n'


# A static solution carrying one timestamp is dated by it, exactly as the Stage 3 and Stage 4 loaders date theirs from
# ts[0]. The contract guarantees a static solution's ts holds exactly one entry, so every stage reads the same time.
def test_static_solution_with_a_timestamp_records_its_slice_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    h5_path = _write_static(tmp_path / "static.h5")
    with h5py.File(h5_path, "a") as f:
        f.create_dataset("ts", data=np.array([2.5e-6]))
    text = _run_writer(monkeypatch, h5_path, TRACKED_TEMPLATE, tmp_path / "out.toml")
    assert "transport time slice t = 2.5e-06 s" in text
