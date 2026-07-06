"""Tests for ``src.stage3_helper.radial_scan_cmd``.

``radial_scan_cmd`` turns the ``stage3.sfincs_jax`` config block into the single
shell command that the Snakefile's ``rule stage3_sfincs`` runs. The exact string
it builds is the contract with that rule, so this pins it: the static base flags
(including the literal ``{input.*}`` placeholders Snakemake substitutes at run
time), the not-None-only optional flags, the tri-state booleans, and the
gpu-only ``--gpu-ids`` flag.
"""

from __future__ import annotations

import re

import pytest

from src.stage3_helper import radial_scan_cmd
from tests.helpers.stage_import import load_stage_module

_STAGE3_SCRIPT = "stages/stage3-neoclassical/sfincs_jax_radial_scan.py"
# Snakemake substitutes {input.*}/{output.*} at run time; swap them for literal path
# tokens so the emitted command parses as a plain argument vector here.
_PLACEHOLDER = re.compile(r"\{(?:input|output)\.[A-Za-z0-9_]+\}")
_scan = load_stage_module(_STAGE3_SCRIPT)


def cmd(**overrides) -> str:
    """Build a command with quick-run-like defaults, overriding only what a test varies."""
    base = dict(
        docker_prefix="docker run --rm",
        image="ghcr.io/driftless-star/driftless-star:stage-3-sfincs-cpu",
        stage_cfg={},
        output_dir="outputs/quick_run/stage3_neoclassical",
        device="cpu",
    )
    base.update(overrides)
    return radial_scan_cmd(**base)


def test_base_command_with_empty_config() -> None:
    # Empty config + cpu emits exactly the static base parts; this equality also
    # pins the literal {input.*} placeholders, which must survive verbatim so
    # Snakemake can substitute them at rule-execution time.
    assert cmd() == (
        "docker run --rm ghcr.io/driftless-star/driftless-star:stage-3-sfincs-cpu "
        "python stages/stage3-neoclassical/sfincs_jax_radial_scan.py "
        "--common-config {input.common_config} "
        "--sfincs-template {input.config_file} "
        "--wout-path {input.wout} "
        "--output-dir outputs/quick_run/stage3_neoclassical "
        "--backend cpu"
    )


def test_only_non_none_optionals_appear() -> None:
    out = cmd(stage_cfg={"ntheta": 31, "nx": 64, "nzeta": None})
    assert "--ntheta 31" in out
    assert "--nx 64" in out
    assert "--nzeta" not in out             # explicit None -> omitted
    assert "--profiles-source" not in out   # absent key -> omitted


@pytest.mark.parametrize(
    "key, on, off",
    [
        ("plot", "--plot", "--no-plot"),
        ("verbose_workers", "--verbose-workers", "--no-verbose-workers"),
    ],
)
def test_tristate_bools(key: str, on: str, off: str) -> None:
    # Token membership (not substring) so --plot does not match inside --no-plot.
    assert on in cmd(stage_cfg={key: True}).split()
    assert off in cmd(stage_cfg={key: False}).split()
    absent = cmd(stage_cfg={}).split()
    assert on not in absent and off not in absent


def test_gpu_ids_only_on_gpu() -> None:
    cfg = {"gpu_ids": "0,1"}
    on_gpu = cmd(stage_cfg=cfg, device="gpu")
    assert "--gpu-ids 0,1" in on_gpu
    assert "--gpu-ids" not in cmd(stage_cfg=cfg, device="cpu")  # cpu suppresses it
    assert "--gpu-ids" not in cmd(stage_cfg={}, device="gpu")   # gpu but no ids


def test_emitted_flags_parse_with_stage_script() -> None:
    # Cross-check: every flag radial_scan_cmd can emit must be one the stage script's own
    # parser accepts. Populate all optionals/toggles, then feed the argument vector to the
    # script's build_parser. A helper flag the script renamed or dropped makes argparse
    # sys.exit(2); this catches helper-vs-script drift the exact-string tests cannot.
    stage_cfg = {
        "profiles_source": "analytical",
        "neopax_result": "outputs/quick_run/stage5_transport/transport_solution.h5",
        "ntheta": 31,
        "nzeta": 15,
        "nxi": 12,
        "nx": 64,
        "solver_tolerance": 1.0e-6,
        "max_parallel": 8,
        "gpu_ids": "0,1",
        "plot": False,            # emits --no-plot
        "verbose_workers": True,  # emits --verbose-workers
    }
    concrete = _PLACEHOLDER.sub("placeholder_path", cmd(stage_cfg=stage_cfg, device="gpu"))
    tokens = concrete.split()
    argv = tokens[tokens.index(_STAGE3_SCRIPT) + 1:]
    try:
        _scan.build_parser().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"stage script parser rejected a helper-emitted flag: argv={argv} (exit {exc.code})")
