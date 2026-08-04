"""Tests for ``executors/htcondor/job_wrapper.sh``, the script that starts Snakemake on a cluster node.

Which Snakemake the wrapper launches matters. The submit host writes each job's command line using its
own Snakemake's flags, so the execute node needs a matching version to understand them, and a mismatch
surfaces as an unhelpful "unrecognized arguments" error. The matching one is baked into the execute
node's container image at a fixed path, and ``tests/orchestration/test_pixi_envs.py`` is what keeps the
two versions equal.

The behaviour tested here is what the wrapper does when that fixed path is missing. It must name the
path it looked for and exit non-zero, rather than fall back to any ``snakemake`` it happens to find on
the node's PATH -- an unknown version would either fail confusingly later or produce a run that only
looks successful.

Only that failure case is tested. Succeeding requires the container image's Snakemake, which cannot
exist on a developer machine, so real cluster jobs cover the success path instead.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WRAPPER = REPO_ROOT / "executors/htcondor/job_wrapper.sh"
RUNTIME_SNAKEMAKE = "/app/.pixi/envs/htcondor-runtime/bin/snakemake"

# Skip inside the image itself, where the path does resolve and the wrapper runs it. The assertions below
# would then fail even though the wrapper is behaving correctly.
inside_image = pytest.mark.skipif(
    Path(RUNTIME_SNAKEMAKE).exists(),
    reason=f"{RUNTIME_SNAKEMAKE} exists, so the wrapper resolves it and the failure path cannot be reached",
)


# A decoy `snakemake` sits first on PATH. A wrapper that fell back to PATH would find it and exit 0 printing
# the sentinel; one that resolves a single fixed path cannot see it. Running the wrapper directly, rather than
# through a shell, exercises its executable bit too.
@inside_image
def test_missing_runtime_snakemake_fails_instead_of_falling_back_to_path(tmp_path: Path) -> None:
    sentinel = "DECOY-SNAKEMAKE-WAS-EXECUTED"
    decoy_bin = tmp_path / "bin"
    decoy_bin.mkdir()
    decoy = decoy_bin / "snakemake"
    decoy.write_text(f"#!/bin/sh\necho {sentinel}\nexit 0\n")
    decoy.chmod(0o755)

    env = {**os.environ, "PATH": f"{decoy_bin}:{os.environ['PATH']}"}
    env.pop("_CONDOR_JOB_IWD", None)  # let repo_root fall back to cwd, i.e. tmp_path

    result = subprocess.run([str(WRAPPER), "--version"], cwd=tmp_path, env=env, capture_output=True, text=True)

    assert result.returncode != 0, f"wrapper exited 0 with stdout={result.stdout!r}"
    assert sentinel not in result.stdout + result.stderr, "the wrapper fell back to a PATH snakemake"
    assert RUNTIME_SNAKEMAKE in result.stderr, f"the missing path is not named in stderr: {result.stderr!r}"
    assert "[job_wrapper] execute-node" not in result.stderr, "reported resolving a Snakemake it did not run"
