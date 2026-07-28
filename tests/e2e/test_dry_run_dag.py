"""Dry-run tests of the Snakemake forward-pass DAG.

``snakemake -n`` parses the Snakefile and plans the job graph without running any
container, so it catches wiring and parse-time errors with no Docker.

Stages 3 and 4 fan out one job per flux surface behind ``prepare`` checkpoints, so
a dry-run plans only up to each checkpoint plus the deferred ``collect`` gathers;
the per-surface ``run_one`` layer enters the DAG only after a real ``prepare``
writes its manifest.

Overriding ``output_dir`` to a tmp dir keeps the Snakefile's parse-time
``prepare_neopax_config`` write, and every planned artifact path, out of the repo,
and ``--runtime-source-cache-path`` keeps Snakemake's own runtime source cache under
tmp. This is not full filesystem hermeticity: Snakemake still writes ``.snakemake/``
metadata into the repo root during a dry run, but that directory is gitignored.
``--workflow-profile none`` stops a user-level profile from injecting flags, so the
planned DAG is the same everywhere.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import yaml

from src.ouroboros import _write_loop_overrides
from src.utils import resolve_pipeline_paths

REPO_ROOT = Path(__file__).resolve().parents[2]
FORWARD_RULES = (
    "stage1_vmec",
    "stage2_boozer",
    "stage3_prepare",
    "stage3_collect",
    "stage4_prepare",
    "stage4_collect",
    "stage5_neopax",
)
# The per-surface run_one rules are gated behind the prepare checkpoints, so a plan built
# before any manifest exists must not schedule them.
DEFERRED_RULES = ("stage3_run_one", "stage4_run_one")


def _dry_run(
    tmp_path: Path,
    targets: list[str],
    config_overrides: list[str],
    extra_configfiles: list[str] | None = None,
    printshellcmds: bool = False,
) -> subprocess.CompletedProcess:
    """Plan the quick_run DAG with ``snakemake -n``, redirecting every write under ``tmp_path``.

    ``extra_configfiles`` are appended after the base config file under the same single
    ``--configfile`` flag, exactly as the loop driver passes its overrides file, and
    ``printshellcmds`` adds ``-p`` so planned shell commands can be asserted on.
    """
    return subprocess.run(
        ["snakemake", "-n", *(["-p"] if printshellcmds else []), *targets,
         "--configfile", "inputs/quick_run/config.yaml", *(extra_configfiles or []),
         "--workflow-profile", "none",
         "--runtime-source-cache-path", f"{tmp_path}/srccache",
         "--config", f"output_dir={tmp_path}/out", *config_overrides],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


# This runs the default forward pass and asserts it plans successfully (exit code 0), that every stage rule visible
# before the checkpoints run is scheduled (including both deferred collect gathers), and that the post-processing rule
# is NOT scheduled, because the default target is a pure forward pass with no loop-closing step. This catches Snakefile
# wiring/parse errors without Docker.
def test_forward_pass_dag_dry_run(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=[])
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    for rule in FORWARD_RULES:
        assert rule in output, f"rule {rule} not scheduled:\n{output}"
    assert "stage5_post_processing" not in output  # rule all is a pure forward pass
    # Dry-run visibility ends at the unexecuted prepare checkpoints: the per-surface jobs
    # must be absent and Snakemake must announce that the DAG grows after the checkpoints.
    for rule in DEFERRED_RULES:
        assert rule not in output, f"per-surface rule {rule} planned before its checkpoint ran:\n{output}"
    assert "checkpoint jobs" in output, output


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


# From iteration 2 the loop driver appends its loop_overrides.yaml after the base config file, switching both stage
# prepares to the prescribed profiles carried by the seeded common_input.toml. This plans the loop-closing target with
# and without that overrides file (written by the real driver helper) and asserts the printed shell commands flip from
# analytical to prescribed, and that the post-processing rule always invokes the prescribed-profiles writer.
def test_loop_overrides_switch_stage_prepares_to_prescribed(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "inputs/quick_run/config.yaml").read_text())
    s5_signal = resolve_pipeline_paths(config, output_dir=f"{tmp_path}/out")["s5_signal"]

    baseline = _dry_run(tmp_path, targets=[s5_signal], config_overrides=[], printshellcmds=True)
    output = baseline.stdout + baseline.stderr
    assert baseline.returncode == 0, output
    assert output.count("--profiles-source analytical") == 2, output
    assert "write_prescribed_profiles_from_transport_h5.py" in output, output
    assert "--output-toml" in output, output

    overrides = _write_loop_overrides(tmp_path)
    result = _dry_run(tmp_path, targets=[s5_signal], config_overrides=[],
                      extra_configfiles=[str(overrides)], printshellcmds=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count("--profiles-source prescribed") == 2, output
    assert "--profiles-source analytical" not in output, output


# The Snakefile validates the `device` config at parse time, before any job runs. This dry-runs with `device=bogus` and
# asserts the run fails (non-zero exit) with a message saying device must be 'cpu' or 'gpu', confirming a bad value is
# caught early rather than deep into an execution.
def test_invalid_device_fails_at_parse(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=["device=bogus"])
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    # The Snakefile's parse-time device guard rejects anything but cpu/gpu.
    assert "must be 'cpu' or 'gpu'" in output, output


# A perturbed fd_gradients sibling run-directory must be schedulable by stage4_run_one exactly like a baseline
# surface: its basename satisfies the shared SURF_PATTERN wildcard constraint, so asking Snakemake to build that
# run.diagnostics.csv plans the per-surface rule. A sibling whose channel letter falls outside n/t violates the
# constraint, so no rule can produce it and the DAG build aborts at parse time.
def test_perturbed_surface_target_matches_run_one(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "inputs/quick_run/config.yaml").read_text())
    stage4_dir = resolve_pipeline_paths(config, output_dir=f"{tmp_path}/out")["stage4_dir"]
    good = f"{stage4_dir}/runs/rho_003_r0p2500_fd_n_D/run.diagnostics.csv"
    result = _dry_run(tmp_path, targets=[good], config_overrides=[])
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "stage4_run_one" in output, output
    # A malformed sibling (fd_x is not a valid density/temperature channel) matches no rule.
    bad = f"{stage4_dir}/runs/rho_003_r0p2500_fd_x_D/run.diagnostics.csv"
    result = _dry_run(tmp_path, targets=[bad], config_overrides=[])
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "No rule to produce" in output, output


# With a real manifest on disk and its upstream equilibrium and Boozer outputs already present, the stage4_prepare
# checkpoint is up to date, so a dry run expands its gather and schedules one stage4_run_one per manifest entry,
# including the perturbed sibling. The dummy upstream outputs are written before the manifest and backdated so the
# checkpoint output stays the newest of its inputs and never re-runs; the checkpoint's remaining config inputs
# already live under inputs/quick_run/. Only the manifest basenames matter, so container-absolute run_dir paths work.
def test_perturbed_manifest_expands_fan_out(tmp_path: Path) -> None:
    config = yaml.safe_load((REPO_ROOT / "inputs/quick_run/config.yaml").read_text())
    paths = resolve_pipeline_paths(config, output_dir=f"{tmp_path}/out")
    for key in ("s1_output", "s2_output"):
        artifact = Path(paths[key])
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("")
        backdated = time.time() - 100
        os.utime(artifact, (backdated, backdated))
    manifest = Path(paths["stage4_manifest"])
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(
        json.dumps(
            {
                "runs": [
                    {"run_dir": "/container/abs/runs/rho_001_r0p2500"},
                    {"run_dir": "/container/abs/runs/rho_001_r0p2500_fd_n_D"},
                ]
            }
        )
    )
    result = _dry_run(tmp_path, targets=[paths["s4_output"]], config_overrides=[])
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "stage4_run_one" in output, output
    assert "rho_001_r0p2500_fd_n_D" in output, output
