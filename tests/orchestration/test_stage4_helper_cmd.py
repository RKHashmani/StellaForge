"""Tests for ``src.stage4_helper.radial_scan_cmd``.

``radial_scan_cmd`` turns the ``stage4.spectrax_gk`` config block into the single
shell command that the Snakefile's ``rule stage4_spectrax`` runs. The exact string
it builds is the contract with that rule, so this pins it: the static base flags
(the ``--vmec-file-override`` and ``--boozer-file-override`` geometry inputs plus
the literal ``{input.*}`` placeholders Snakemake substitutes at run time), the
not-None-only optional flags (where ``t_max`` is emitted as ``--t-final``), the
four tri-state bool toggles, and the gpu-only ``--gpu-ids`` flag.
"""

from __future__ import annotations

import re

import pytest

from src.stage4_helper import radial_scan_cmd
from tests.helpers.stage_import import load_stage_module

_STAGE4_SCRIPT = "stages/stage4-turbulence/spectrax_gk_radial_scan.py"
# Snakemake substitutes {input.*}/{output.*} at run time; swap them for literal path
# tokens so the emitted command parses as a plain argument vector here.
_PLACEHOLDER = re.compile(r"\{(?:input|output)\.[A-Za-z0-9_]+\}")
_scan = load_stage_module(_STAGE4_SCRIPT)


def cmd(**overrides) -> str:
    """Build a command with quick-run-like defaults, overriding only what a test varies."""
    base = dict(
        docker_prefix="docker run --rm",
        image="ghcr.io/driftless-star/driftless-star:stage-4-spectrax-cpu",
        stage_cfg={},
        output_dir="outputs/quick_run/stage4_turbulence",
        device="cpu",
    )
    base.update(overrides)
    return radial_scan_cmd(**base)


# `radial_scan_cmd` builds the shell command the Snakefile runs for Stage 4. With an empty config and cpu device, this
# pins the exact base command string, including the two geometry-input overrides and the literal `{input.*}`
# placeholders Snakemake fills in with real paths at run time.
def test_base_command_with_empty_config() -> None:
    # Empty config + cpu emits exactly the static base parts; this equality also
    # pins the literal {input.*} placeholders (including the two geometry
    # overrides) that Snakemake substitutes at rule-execution time.
    assert cmd() == (
        "docker run --rm ghcr.io/driftless-star/driftless-star:stage-4-spectrax-cpu "
        "python stages/stage4-turbulence/spectrax_gk_radial_scan.py "
        "--common-config {input.common_config} "
        "--spectrax-template {input.config_file} "
        "--vmec-file-override {input.wout} "
        "--boozer-file-override {input.boozer} "
        "--output-dir outputs/quick_run/stage4_turbulence "
        "--backend cpu"
    )


# Optional flags appear only when set. This also checks a rename: the config key `t_max` is emitted as the `--t-final`
# flag. It asserts the two set values appear (with the rename applied), the None-valued key is omitted, and a key absent
# from the config is omitted.
def test_only_non_none_optionals_appear() -> None:
    out = cmd(stage_cfg={"t_max": 250.0, "ntheta": 16, "ny": None})
    assert "--t-final 250.0" in out          # t_max is renamed to --t-final
    assert "--ntheta 16" in out
    assert "--ny" not in out                 # explicit None -> omitted
    assert "--profiles-source" not in out    # absent key -> omitted


# Boolean toggles are tri-state: True emits `--flag`, False emits `--no-flag`, absent emits neither. Parametrized over
# Stage 4's four toggles, this verifies all three cases. It compares whole tokens so a short flag like `--plot` isn't
# falsely matched inside `--no-plot` or `--plot-run-heat-traces`.
@pytest.mark.parametrize(
    "key, on, off",
    [
        ("plot", "--plot", "--no-plot"),
        ("plot_run_heat_traces", "--plot-run-heat-traces", "--no-plot-run-heat-traces"),
        ("verbose_workers", "--verbose-workers", "--no-verbose-workers"),
        ("collect_even_if_failures", "--collect-even-if-failures", "--no-collect-even-if-failures"),
    ],
)
def test_tristate_bools(key: str, on: str, off: str) -> None:
    # Token membership (not substring) so --plot does not match inside --no-plot
    # or --plot-run-heat-traces.
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


# A drift guard. The exact-string tests only check the helper against itself; this checks it against the real stage
# script. It configures every optional and toggle (including the `t_max` -> `--t-final` rename), strips out the
# Snakemake placeholders, and feeds the resulting flags into the stage script's own argparse parser (`build_parser`). If
# the script ever renames or drops a flag, argparse exits (`sys.exit(2)`) and this test fails, catching the mismatch.
def test_emitted_flags_parse_with_stage_script() -> None:
    stage_cfg = {
        "profiles_source": "analytical",
        "neopax_result": "outputs/quick_run/stage5_transport/transport_solution.h5",
        "nx": 12,
        "ny": 12,
        "ntheta": 16,
        "t_max": 250.0,
        "average_window": 1.0,
        "sample_stride": 50,
        "diagnostics_stride": 1,
        "max_parallel": 1,
        "gpu_ids": "0,1",
        "plot": False,                     # emits --no-plot
        "plot_run_heat_traces": True,      # emits --plot-run-heat-traces
        "verbose_workers": False,          # emits --no-verbose-workers
        "collect_even_if_failures": True,  # emits --collect-even-if-failures
    }
    concrete = _PLACEHOLDER.sub("placeholder_path", cmd(stage_cfg=stage_cfg, device="gpu"))
    tokens = concrete.split()
    argv = tokens[tokens.index(_STAGE4_SCRIPT) + 1:]
    try:
        _scan.build_parser().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"stage script parser rejected a helper-emitted flag: argv={argv} (exit {exc.code})")
