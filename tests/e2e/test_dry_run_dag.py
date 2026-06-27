"""Dry-run test of the Snakemake forward-pass DAG.

``snakemake -n`` parses the Snakefile and plans the job graph without running any
container, so it catches wiring and parse-time errors with no Docker. This test
asserts the dry run exits 0 from the committed quick_run config and that all five
forward-pass rules are scheduled, while ``stage5_post_processing`` is not (the
default ``rule all`` target is a pure forward pass).

Overriding ``output_dir`` to a tmp dir keeps the Snakefile's parse-time
``prepare_neopax_config`` write, and every planned artifact path, out of the repo.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FORWARD_RULES = ("stage1_vmec", "stage2_boozer", "stage3_sfincs", "stage4_spectrax", "stage5_neopax")


def test_forward_pass_dag_dry_run(tmp_path: Path) -> None:
    result = subprocess.run(
        ["snakemake", "-n",
         "--configfile", "inputs/quick_run/config.yaml",
         "--config", f"output_dir={tmp_path}/out"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    for rule in FORWARD_RULES:
        assert rule in output, f"rule {rule} not scheduled:\n{output}"
    assert "stage5_post_processing" not in output  # rule all is a pure forward pass
