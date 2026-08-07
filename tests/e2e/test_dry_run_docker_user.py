"""Dry-run tests of the container user the Snakefile plans into every launch.

``snakemake -n -p`` formats every stage's shell command without starting a container, so
the ``docker_user`` modes can be told apart from the planned text alone on a machine with
no Docker.

What is pinned here is that the mode reaches every launch, not just the first. The flag
decides whether a container can read the bind-mounted repo at all on a rootless daemon,
so a single launch built from a stale hardcoded prefix would fail the run while the plan
looked correct. Only the explicit modes are asserted on, since the default ``"auto"``
resolves against the daemon of whichever machine runs the suite.

The ``_dry_run`` helper is shared with the DAG tests, which documents the isolation it
applies to keep every write under tmp.
"""

from __future__ import annotations

from pathlib import Path

from tests.e2e.test_dry_run_dag import _dry_run

# Snakefile spellings for the two container users. The host substitution is planned literally and evaluated when the
# job runs, so it appears verbatim in the plan.
HOST_USER_FLAG = '--user "$(id -u):$(id -g)"'
ROOT_USER_FLAG = "--user 0:0"


# A rootful daemon needs every launch to drop to the invoking user, or the run leaves root-owned files scattered
# through the user's working tree. Counting against planned launches catches a rule that was given a prefix built
# before the mode was resolved.
def test_host_mode_plans_the_user_flag_on_every_launch(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=["docker_user=host"], printshellcmds=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count("docker run") > 0, output
    assert output.count(HOST_USER_FLAG) == output.count("docker run"), output


# A rootless runtime already maps the invoking user onto container uid 0, and passing the host uid anyway leaves the
# container as an unprivileged subuid that cannot read /work. Naming uid 0 explicitly rather than omitting the flag
# keeps the resolved user the same if a stage image ever declares its own USER, so every launch must carry it and none
# may carry the host spelling.
def test_root_mode_plans_the_root_flag_on_every_launch(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=["docker_user=root"], printshellcmds=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert output.count("docker run") > 0, output
    assert output.count(ROOT_USER_FLAG) == output.count("docker run"), output
    assert HOST_USER_FLAG not in output, output


# Generated ConStellaration runs default to /staging, outside the repository bind at /work. Every container must
# therefore receive the absolute output root as a same-path bind mount. The shared dry-run helper already overrides
# output_dir to an absolute tmp path, so this pins the external-run behavior without touching /staging.
def test_absolute_run_root_is_bound_on_every_launch(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=["docker_user=root"], printshellcmds=True)
    output = result.stdout + result.stderr
    bind_flag = f"-v {tmp_path}/out:{tmp_path}/out"
    assert result.returncode == 0, output
    assert output.count("docker run") > 0, output
    assert output.count(bind_flag) == output.count("docker run"), output


# A mistyped mode must be caught while the DAG is being built, before any job runs, and the message must name the
# value, so the failure is actionable rather than surfacing hours later as an unreadable bind mount.
def test_malformed_mode_fails_at_parse_naming_the_value(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=["docker_user=rootless"])
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "docker_user" in output, output
    assert "rootless" in output, output
