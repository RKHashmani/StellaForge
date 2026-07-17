"""Dry-run tests of the Snakemake forward-pass DAG.

``snakemake -n`` parses the Snakefile and plans the job graph without running any
container, so it catches wiring and parse-time errors with no Docker.

Overriding ``output_dir`` to a tmp dir keeps the Snakefile's parse-time
``prepare_neopax_config`` write, and every planned artifact path, out of the repo,
and ``--runtime-source-cache-path`` keeps Snakemake's own runtime source cache under
tmp. This is not full filesystem hermeticity: Snakemake still writes ``.snakemake/``
metadata into the repo root during a dry run, but that directory is gitignored.
``--workflow-profile none`` stops a user-level profile from injecting flags, so the
planned DAG is the same everywhere.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from src.utils import resolve_pipeline_paths

REPO_ROOT = Path(__file__).resolve().parents[2]
FORWARD_RULES = ("stage1_vmec", "stage2_boozer", "stage3_sfincs", "stage4_spectrax", "stage5_neopax")


def _dry_run(tmp_path: Path, targets: list[str], config_overrides: list[str]) -> subprocess.CompletedProcess:
    """Plan the quick_run DAG with ``snakemake -n``, redirecting every write under ``tmp_path``."""
    return subprocess.run(
        ["snakemake", "-n", *targets,
         "--configfile", "inputs/quick_run/config.yaml",
         "--workflow-profile", "none",
         "--runtime-source-cache-path", f"{tmp_path}/srccache",
         "--config", f"output_dir={tmp_path}/out", *config_overrides],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


# This runs the default forward pass and asserts it plans successfully (exit code 0), that all five stage rules are
# scheduled, and that the post-processing rule is NOT scheduled, because the default target is a pure forward pass with
# no loop-closing step. This catches Snakefile wiring/parse errors without Docker.
def test_forward_pass_dag_dry_run(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=[])
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    for rule in FORWARD_RULES:
        assert rule in output, f"rule {rule} not scheduled:\n{output}"
    assert "stage5_post_processing" not in output  # rule all is a pure forward pass


# When you explicitly ask Snakemake to build the convergence-signal file (the loop-closing target), the post-processing
# rule SHOULD be pulled in. This resolves that target path, dry-runs with it as the goal, and asserts
# `stage5_post_processing` now appears in the plan, confirming the loop-closing rule wires up on demand.
def test_s5_signal_target_schedules_post_processing(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "inputs/quick_run/config.yaml").read_text())
    s5_signal = resolve_pipeline_paths(config, output_dir=f"{tmp_path}/out")["s5_signal"]
    result = _dry_run(tmp_path, targets=[s5_signal], config_overrides=[])
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    # Requesting the convergence signal pulls in the loop-closing rule and formats its
    # shell string, which rule all (a pure forward pass) never reaches.
    assert "stage5_post_processing" in output, output


# The Snakefile validates the `device` config at parse time, before any job runs. This dry-runs with `device=bogus` and
# asserts the run fails (non-zero exit) with a message saying device must be 'cpu' or 'gpu', confirming a bad value is
# caught early rather than deep into an execution.
def test_invalid_device_fails_at_parse(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=["device=bogus"])
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    # The Snakefile's parse-time device guard rejects anything but cpu/gpu.
    assert "must be 'cpu' or 'gpu'" in output, output
