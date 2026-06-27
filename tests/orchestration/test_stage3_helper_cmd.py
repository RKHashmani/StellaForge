"""Tests for ``src.stage3_helper.radial_scan_cmd``.

``radial_scan_cmd`` turns the ``stage3.sfincs_jax`` config block into the single
shell command that the Snakefile's ``rule stage3_sfincs`` runs. The exact string
it builds is the contract with that rule, so this pins it: the static base flags
(including the literal ``{input.*}`` placeholders Snakemake substitutes at run
time), the not-None-only optional flags, the tri-state booleans, and the
gpu-only ``--gpu-ids`` flag.
"""

from __future__ import annotations

import pytest

from src.stage3_helper import radial_scan_cmd


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
