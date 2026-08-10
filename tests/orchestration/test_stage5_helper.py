"""Tests for the ``src.stage5_helper`` readers of the NEOPAX template.

NEOPAX is configured by a TOML file, not CLI flags. ``prepare_neopax_config``
writes a path-resolved copy of the shared template under the run's Stage 5 output
dir, rewriting its five path fields *relative to that copy's own directory* (NEOPAX
runs there) and never touching the committed template. These tests pin the rewrite
targets, the trailing slash on ``transport_output_dir``, and template immutability.
``read_rho_edge`` reads the radial grid's outer edge back out of the same template,
which is what keeps the Stage 4 relabelling step on the grid NEOPAX will build.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.stage5_helper import prepare_neopax_config, read_rho_edge

_TEMPLATE = (
    'vmec_file = "PLACEHOLDER"\n'
    'boozer_file = "PLACEHOLDER"\n'
    'neoclassical_file = "PLACEHOLDER"\n'
    'turbulence_file = "PLACEHOLDER"\n'
    'transport_output_dir = "PLACEHOLDER"\n'
)


def _prepare(tmp_path: Path) -> tuple[Path, Path]:
    """Lay out a tmp run and resolve the NEOPAX config; return (template, resolved)."""
    template = tmp_path / "inputs" / "common_input.toml"
    template.parent.mkdir(parents=True)
    template.write_text(_TEMPLATE)

    out = tmp_path / "out"
    resolved = out / "stage5_transport" / "common_input_updated.toml"  # parent not pre-created
    prepare_neopax_config(
        s5_config_template=str(template),
        s5_resolved_config=str(resolved),
        s1_output=str(out / "stage1_equilibrium" / "wout.nc"),
        s2_output=str(out / "stage2_boozer" / "boozmn.nc"),
        s3_output=str(out / "stage3_neoclassical" / "sfincs_flux.h5"),
        s4_output=str(out / "stage4_turbulence" / "neopax_fluxes.h5"),
        s5_output_dir=str(out / "stage5_transport"),
    )
    return template, resolved


# NEOPAX (Stage 5) is configured by a TOML file, not CLI flags. `prepare_neopax_config` writes a copy of the template
# into the Stage 5 output dir and rewrites its five input-path fields so each points at the right upstream artifact,
# relative to that copy's own location. This reads the copy and asserts all five fields now hold the expected
# `../stageN/...` relative path, and that the output dir field resolves to `./` (the copy's own directory).
def test_rewrites_five_paths_relative_and_quoted(tmp_path: Path) -> None:
    _, resolved = _prepare(tmp_path)
    text = resolved.read_text()  # also asserts the parent dir was created
    assert 'vmec_file = "../stage1_equilibrium/wout.nc"' in text
    assert 'boozer_file = "../stage2_boozer/boozmn.nc"' in text
    assert 'neoclassical_file = "../stage3_neoclassical/sfincs_flux.h5"' in text
    assert 'turbulence_file = "../stage4_turbulence/neopax_fluxes.h5"' in text
    # The output dir is the copy's own dir, so it resolves to "./" with a trailing slash.
    assert 'transport_output_dir = "./"' in text


# The rewrite must happen on the copy, never the shared template. After running the same preparation, this reads the
# original template file back and asserts its contents are byte-for-byte unchanged, proving the committed template is
# never mutated by a run.
def test_template_left_unmodified(tmp_path: Path) -> None:
    template, _ = _prepare(tmp_path)
    assert template.read_text() == _TEMPLATE


# --- read_rho_edge ---

def _write_template(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "common_input.toml"
    path.write_text(body)
    return path


def test_read_rho_edge_returns_the_configured_value(tmp_path: Path) -> None:
    assert read_rho_edge(str(_write_template(tmp_path, "[geometry]\nrho_edge = 0.7\n"))) == 0.7


# NEOPAX's own default is 1.0, so a template that never mentions rho_edge grids out to the boundary. Both an absent key
# and an absent [geometry] table have to resolve to it, or the relabelling step would reject a perfectly good grid.
@pytest.mark.parametrize("body", ["[geometry]\nn_radial = 5\n", "[species]\nn_species = 1\n"])
def test_read_rho_edge_defaults_to_one(tmp_path: Path, body: str) -> None:
    assert read_rho_edge(str(_write_template(tmp_path, body))) == 1.0


# rho_edge scales every stage's radial grid, so a value outside (0, 1] is never recoverable downstream. It is caught at
# Snakefile parse time, before any stage runs. A bool passes isinstance(x, int) in Python, hence its own case here.
@pytest.mark.parametrize("value", ['"0.7"', "true", "1.5", "0.0", "-0.5", "nan"])
def test_read_rho_edge_rejects_a_value_outside_the_unit_interval(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError, match=r"\[geometry\].rho_edge"):
        read_rho_edge(str(_write_template(tmp_path, f"[geometry]\nrho_edge = {value}\n")))


# --- [profiles] validation ---

# Stage 5 is the last rule in the DAG, so a prescribed [profiles] block NEOPAX cannot load used to
# surface only after Stages 1-4 had run. The closed-loop driver writes such a block into the
# template from iteration 2 onward. These pin that a malformed one is rejected at parse time.

_SPECIES_AND_GEOMETRY = '[species]\nnames = ["e", "ion"]\n\n[geometry]\nn_radial = 3\n\n'


def _prepare_with_profiles(tmp_path: Path, profiles: str) -> None:
    """Resolve a config whose ``[profiles]`` section is ``profiles``, raising on invalid ones."""
    template = tmp_path / "inputs" / "common_input.toml"
    template.parent.mkdir(parents=True, exist_ok=True)
    template.write_text(_TEMPLATE + "\n" + _SPECIES_AND_GEOMETRY + profiles)
    out = tmp_path / "out"
    prepare_neopax_config(
        s5_config_template=str(template),
        s5_resolved_config=str(out / "stage5_transport" / "common_input_updated.toml"),
        s1_output=str(out / "stage1_equilibrium" / "wout.nc"),
        s2_output=str(out / "stage2_boozer" / "boozmn.nc"),
        s3_output=str(out / "stage3_neoclassical" / "sfincs_flux.h5"),
        s4_output=str(out / "stage4_turbulence" / "neopax_fluxes.h5"),
        s5_output_dir=str(out / "stage5_transport"),
    )


def test_scalar_analytical_parameters_are_accepted(tmp_path: Path) -> None:
    _prepare_with_profiles(tmp_path, '[profiles]\nmodel = "standard_analytical"\nn0 = 4.21\nT0 = 17.8\n')


# NEOPAX coerces per-species lists for every analytical parameter, shape exponents included, so
# parse-time validation must let them through.
def test_per_species_lists_under_the_analytical_model_are_accepted(tmp_path: Path) -> None:
    _prepare_with_profiles(
        tmp_path,
        '[profiles]\nmodel = "standard_analytical"\n'
        "n0 = [4.21, 4.21]\nn_edge = [0.4, 0.4]\n"
        "T0 = [6.7, 1.0]\nT_edge = [0.2, 0.2]\n"
        "density_shape_power = [1.0, 1.0]\ntemperature_shape_power = [2.0, 2.0]\n"
        "density_shape_alpha = [1.5, 1.5]\ntemperature_shape_alpha = [2.0, 1.0]\n",
    )


_VALID_PRESCRIBED = (
    '[profiles]\nmodel = "prescribed"\n'
    "density = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]\n"
    "temperature = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]\n"
    "Er = [0.0, 0.0, 0.0]\n"
)


def test_valid_prescribed_block_is_accepted(tmp_path: Path) -> None:
    _prepare_with_profiles(tmp_path, _VALID_PRESCRIBED)


@pytest.mark.parametrize(
    "profiles, message",
    [
        pytest.param(
            '[profiles]\nmodel = "prescribed"\ndensity = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]\n',
            r"\[profiles\].temperature is missing",
            id="missing-array",
        ),
        pytest.param(
            _VALID_PRESCRIBED.replace("density = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]", "density = [[1.0, 2.0, 3.0]]"),
            r"holds 1 species rows, expected 2",
            id="species-row-count",
        ),
        pytest.param(
            _VALID_PRESCRIBED.replace("Er = [0.0, 0.0, 0.0]", "Er = [0.0, 0.0]"),
            r"rows must hold 3 points",
            id="radial-point-count",
        ),
        pytest.param(
            _VALID_PRESCRIBED.replace("density = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]", "density = [1.0, 2.0, 3.0]"),
            r"must be a 2-D array",
            id="wrong-rank",
        ),
    ],
)
def test_malformed_prescribed_blocks_are_rejected(tmp_path: Path, profiles: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _prepare_with_profiles(tmp_path, profiles)


# Every tracked run config must pass the parse-time validation.
@pytest.mark.parametrize("run", ["w7-x_quick_run", "w7-x_t3d_validation", "quick_run"])
def test_tracked_configs_pass_validation(tmp_path: Path, run: str) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    template = repo_root / "inputs" / run / "common_input.toml"
    out = tmp_path / "out"
    prepare_neopax_config(
        s5_config_template=str(template),
        s5_resolved_config=str(out / "stage5_transport" / "common_input_updated.toml"),
        s1_output=str(out / "stage1_equilibrium" / "wout.nc"),
        s2_output=str(out / "stage2_boozer" / "boozmn.nc"),
        s3_output=str(out / "stage3_neoclassical" / "sfincs_flux.h5"),
        s4_output=str(out / "stage4_turbulence" / "neopax_fluxes.h5"),
        s5_output_dir=str(out / "stage5_transport"),
    )
