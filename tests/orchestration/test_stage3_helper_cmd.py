"""Tests for the ``src.stage3_helper`` per-phase command composers.

``prepare_cmd``, ``run_one_cmd``, and ``collect_cmd`` turn the ``stage3.sfincs_jax``
config block into the three shell commands that the Snakefile's per-phase Stage 3
rules run. The exact strings are the contract with those rules, so this pins them:
the static base flags (including the literal ``{input.*}`` and ``{wildcards.surf}``
placeholders Snakemake substitutes at run time), the not-None-only optional flags
on ``prepare``, the tri-state booleans, and the absence of scan-level parallelism
and GPU flags now that surface concurrency belongs to ``snakemake --cores`` and
device assignment to the docker prefix.
"""

from __future__ import annotations

import argparse
import inspect
import re
from collections.abc import Callable

import pytest

from src.stage3_helper import collect_cmd, prepare_cmd, run_one_cmd
from tests.helpers.stage_import import load_stage_module

_STAGE3_SCRIPT = "stages/stage3-neoclassical/sfincs_jax_radial_scan.py"
# Snakemake substitutes {input.*}/{output.*} at run time; swap them for literal path
# tokens so the emitted command parses as a plain argument vector here.
_PLACEHOLDER = re.compile(r"\{(?:input|output)\.[A-Za-z0-9_]+\}")
_scan = load_stage_module(_STAGE3_SCRIPT)

# Every prepare-phase optional key with a quick-run-like value, used both to exercise
# each flag individually and to prove that a fully-populated config never leaks
# scan-level parallelism flags. max_parallel and gpu_ids are retired keys that older
# configs may still carry; every composer must ignore them.
_PREPARE_OPTIONALS: list[tuple[str, str, object]] = [
    ("profiles_source",    "--profiles-source",    "prescribed"),  # the loop's iteration-2+ value, not the default
    ("neopax_result",      "--neopax-result",      "outputs/quick_run/stage5_transport/transport_solution.h5"),
    ("ntheta",             "--ntheta",             5),
    ("nzeta",              "--nzeta",              11),
    ("nxi",                "--nxi",                12),
    ("nx",                 "--nx",                 4),
    ("solver_tolerance",   "--solver-tolerance",   1e-06),
    ("analytical_n_radii", "--analytical-n-radii", 51),
    ("rho_indices",        "--rho-indices",        "1,5,10"),
    ("rho_min",            "--rho-min",            0.1),
    ("rho_max",            "--rho-max",            0.9),
    ("num_radii",          "--num-radii",          4),
]

_FULL_CFG: dict = {key: value for key, _, value in _PREPARE_OPTIONALS} | {
    "gpu_ids": "0,1",
    "plot": False,
    "verbose_workers": True,
    "max_parallel": 8,
}


def compose(composer: Callable[..., str], **overrides) -> str:
    """Build a command with quick-run-like defaults, overriding only what a test varies.

    Each composer takes exactly the arguments its phase uses, so the shared defaults are narrowed to the ones the
    called composer accepts.
    """
    base = dict(
        docker_prefix="docker run --rm",
        image="ghcr.io/driftless-star/driftless-star:stage-3-sfincs-cpu",
        stage_cfg={},
        output_dir="outputs/quick_run/stage3_neoclassical",
        device="cpu",
    )
    base.update(overrides)
    accepted = inspect.signature(composer).parameters
    return composer(**{name: value for name, value in base.items() if name in accepted})


def parse_with_stage_script(command: str) -> argparse.Namespace:
    """Feed a composed command's argv into the real stage script parser and return the namespace.

    Fails the test if argparse rejects any flag (argparse exits via ``sys.exit(2)``).
    """
    concrete = _PLACEHOLDER.sub("placeholder_path", command).replace("{wildcards.surf}", "rho_001_r0p1000")
    tokens = concrete.split()
    argv = tokens[tokens.index(_STAGE3_SCRIPT) + 1:]
    try:
        return _scan.build_parser().parse_args(argv)
    except SystemExit as exc:
        pytest.fail(f"stage script parser rejected a composer-emitted flag: argv={argv} (exit {exc.code})")


# `prepare_cmd` builds the shell command that writes per-surface payloads and the scan manifest. With an empty config
# and cpu device it should emit just the fixed base command. This pins that exact string, including the `prepare`
# subcommand token and the literal `{input.*}` placeholders that Snakemake fills in with real file paths at run time.
def test_prepare_base_command_with_empty_config() -> None:
    assert compose(prepare_cmd) == (
        "docker run --rm ghcr.io/driftless-star/driftless-star:stage-3-sfincs-cpu "
        "python stages/stage3-neoclassical/sfincs_jax_radial_scan.py prepare "
        "--common-config {input.common_config} "
        "--sfincs-template {input.config_file} "
        "--wout-path {input.wout} "
        "--boozer-path {input.boozer} "
        "--output-dir outputs/quick_run/stage3_neoclassical "
        "--backend cpu"
    )


# `run_one_cmd` builds the per-surface worker command. This pins the exact string: the payload path is derived from the
# output dir plus the literal `{wildcards.surf}` placeholder (Snakemake substitutes the surface's run-directory name at
# rule-execution time), and nothing beyond the payload and backend is emitted with an empty config.
def test_run_one_base_command_with_empty_config() -> None:
    assert compose(run_one_cmd) == (
        "docker run --rm ghcr.io/driftless-star/driftless-star:stage-3-sfincs-cpu "
        "python stages/stage3-neoclassical/sfincs_jax_radial_scan.py run-one "
        "--payload outputs/quick_run/stage3_neoclassical/runs/{wildcards.surf}/payload.json "
        "--backend cpu"
    )


# `collect_cmd` builds the reduction command that folds per-surface results into the flux-profile HDF5. With an empty
# config it should emit only the `collect` subcommand and the output dir; no backend flag exists on the reduction step.
def test_collect_base_command_with_empty_config() -> None:
    assert compose(collect_cmd) == (
        "docker run --rm ghcr.io/driftless-star/driftless-star:stage-3-sfincs-cpu "
        "python stages/stage3-neoclassical/sfincs_jax_radial_scan.py collect "
        "--output-dir outputs/quick_run/stage3_neoclassical"
    )


# Each prepare-phase optional key set on its own must emit its flag-value pair exactly once; an absent key must emit
# nothing. Parametrizing over the full optional table also locks the config-key to CLI-flag spelling (for example
# analytical_n_radii becomes --analytical-n-radii).
@pytest.mark.parametrize("key, flag, value", _PREPARE_OPTIONALS)
def test_prepare_optional_flag_emitted_exactly_once(key: str, flag: str, value: object) -> None:
    out = compose(prepare_cmd, stage_cfg={key: value})
    assert f"{flag} {value}" in out
    assert out.split().count(flag) == 1
    assert flag not in compose(prepare_cmd, stage_cfg={})


# verbose_workers is consumed at prepare time (baked into each payload.json), so its toggle lives on `prepare` and is
# tri-state: True emits the on-flag, False the off-flag, and an absent key emits neither so the script default applies.
# Token membership is used so --verbose-workers is not falsely found inside --no-verbose-workers.
def test_prepare_verbose_workers_tristate() -> None:
    assert "--verbose-workers" in compose(prepare_cmd, stage_cfg={"verbose_workers": True}).split()
    assert "--no-verbose-workers" in compose(prepare_cmd, stage_cfg={"verbose_workers": False}).split()
    absent = compose(prepare_cmd, stage_cfg={}).split()
    assert "--verbose-workers" not in absent and "--no-verbose-workers" not in absent


# Plotting happens in the reduction step, so the plot toggle lives on `collect` and is tri-state: True emits --plot,
# False emits --no-plot, absent emits neither. Token membership so --plot is not falsely found inside --no-plot.
def test_collect_plot_tristate() -> None:
    assert "--plot" in compose(collect_cmd, stage_cfg={"plot": True}).split()
    assert "--no-plot" in compose(collect_cmd, stage_cfg={"plot": False}).split()
    absent = compose(collect_cmd, stage_cfg={}).split()
    assert "--plot" not in absent and "--no-plot" not in absent


# Device assignment lives in the docker run prefix, so the worker command itself carries no GPU flag in either mode
# and the worker pins the one device its container exposes. The worker phase reads no stage config at all, so a
# retired gpu_ids key left in a config file has no path to this command.
def test_run_one_never_emits_gpu_ids() -> None:
    assert "--gpu-ids" not in compose(run_one_cmd, device="gpu")
    assert "--gpu-ids" not in compose(run_one_cmd, device="cpu")
    assert "stage_cfg" not in inspect.signature(run_one_cmd).parameters


# A drift guard. The exact-string tests only check the composer against itself; this checks it against the real stage
# script. It configures every optional and toggle, substitutes the Snakemake placeholders, and feeds the argv into the
# stage script's own `build_parser`. Parsing must succeed and dispatch to cmd_prepare, so a renamed or dropped flag on
# the prepare subparser fails here.
def test_prepare_flags_parse_and_dispatch_to_cmd_prepare() -> None:
    args = parse_with_stage_script(compose(prepare_cmd, stage_cfg=_FULL_CFG))
    assert args.func.__name__ == "cmd_prepare"


# A drift guard for the worker phase: the composed run-one command (gpu variant) must parse with the real stage script
# and dispatch to cmd_run_one, and the payload path must land on the run-one --payload argument rather than on any
# flat-parser attribute. gpu_ids staying at the parser's own "0" default proves the composer sent no id list, so the
# worker pins device ordinal 0, the one device a pinned container exposes.
def test_run_one_flags_parse_and_dispatch_to_cmd_run_one() -> None:
    args = parse_with_stage_script(compose(run_one_cmd, device="gpu"))
    assert args.func.__name__ == "cmd_run_one"
    assert args.payload == "outputs/quick_run/stage3_neoclassical/runs/rho_001_r0p1000/payload.json"
    assert args.gpu_ids == "0"


# A drift guard for the reduction phase. The flat (no-subcommand) parser defines its own --output-dir and --plot
# defaults; this asserts the collect subparser's parsed values win, so the composed --output-dir and --no-plot reach
# cmd_collect instead of being shadowed by flat-parser defaults.
def test_collect_flags_parse_and_dispatch_to_cmd_collect() -> None:
    args = parse_with_stage_script(compose(collect_cmd, stage_cfg=_FULL_CFG))
    assert args.func.__name__ == "cmd_collect"
    assert args.output_dir == "outputs/quick_run/stage3_neoclassical"
    assert args.plot is False
    assert args.command == "collect"


# Surface-level concurrency now belongs to `snakemake --cores`, so even a fully-populated config (including the retired
# max_parallel key) must never make any composer emit a scan-level parallelism flag.
def test_no_composer_emits_scan_level_parallelism_flags() -> None:
    for composer in (prepare_cmd, run_one_cmd, collect_cmd):
        for device in ("cpu", "gpu"):
            out = compose(composer, stage_cfg=_FULL_CFG, device=device)
            assert "--max-parallel" not in out
            assert "--collect-even-if-failures" not in out
