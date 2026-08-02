"""Dry-run tests of the Stage 4 radial-grid relabelling step the Snakefile plans into ``stage4_collect``.

``snakemake -n -p`` formats the rule's shell command without starting a container, so the whole
composed line can be asserted on. What matters here is the composition rather than the flags, which
``tests/orchestration/test_stage4_helper_cmd.py`` already pins. What has to hold is that the step is
absent by default, and that when present it is chained and grouped so that a collect failure still
fails the rule and both commands' output still reaches the log.

The ``_dry_run`` helper is shared with the DAG tests, which documents the isolation it applies to
keep every write under tmp.
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.e2e.test_dry_run_dag import _dry_run

_RELABEL_SCRIPT = "stages/stage4-turbulence/relabel_neopax_flux_radius.py"


def _disabled_configfile(tmp_path: Path) -> str:
    """Write an overrides config that turns the step off, as the loop driver layers its own.

    ``--config`` cannot express this, because snakemake stringifies every scalar inside a nested
    dict override, so a nested ``false`` arrives as ``"false"`` and is rejected. A layered config file
    keeps YAML's own types, which is also how the loop driver passes its overrides.
    """
    path = tmp_path / "relabel_off.yaml"
    path.write_text("stage4:\n  neopax_radius_relabel: false\n")
    return str(path)


def _collect_shell_line(output: str) -> str:
    """Return the planned shell command for stage4_collect."""
    for line in output.splitlines():
        if line.startswith("Shell command:") and "spectrax_gk_radial_scan.py collect" in line:
            return line
    raise AssertionError(f"no stage4_collect shell command planned:\n{output}")


# Setting the key to false removes the step entirely rather than running a container that decides to do nothing, so a
# run that opts out plans exactly the pipeline that existed before the step.
def test_relabel_absent_when_disabled(tmp_path: Path) -> None:
    result = _dry_run(
        tmp_path,
        targets=[],
        config_overrides=[],
        extra_configfiles=[_disabled_configfile(tmp_path)],
        printshellcmds=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert _RELABEL_SCRIPT not in output, output


# The shipped quick_run config names a convention, so the step joins stage4_collect as a second container launch. `&&`
# rather than `;` so a failed collect never relabels a stale or missing flux file.
def test_relabel_is_chained_onto_collect_by_default(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=[], printshellcmds=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    line = _collect_shell_line(output)
    assert "&& docker run" in line, line
    assert f"python {_RELABEL_SCRIPT}" in line, line
    assert "--convention boozer_volume" in line, line


# The regression this guards: `a && b 2>&1 | tee log` binds the pipe to `b` alone, so the collect output silently stops
# reaching the rule log the moment the relabel is switched on. Both commands must sit inside the group the pipe reads.
def test_both_commands_are_grouped_before_the_log_pipe(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=[], printshellcmds=True)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    line = _collect_shell_line(output)
    body = re.sub(r"^Shell command:\s*", "", line)
    assert body.startswith("( "), body
    grouped, _, piped = body.partition(" ) 2>&1 | tee ")
    assert piped, f"the grouped commands must be piped to the log as a unit: {body}"
    assert "spectrax_gk_radial_scan.py collect" in grouped, grouped
    assert _RELABEL_SCRIPT in grouped, grouped


# Rewriting one HDF5 dataset needs no GPU. On a gpu run the collect launch takes a slot and a device, and the relabel
# launch that follows it must take neither, or it would queue for a slot it has no use for.
def test_relabel_launch_takes_no_gpu_or_slot(tmp_path: Path) -> None:
    result = _dry_run(
        tmp_path,
        targets=[],
        config_overrides=["gpu_ids=all"],
        printshellcmds=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    line = _collect_shell_line(output)
    relabel_launch = line[line.index("&& docker run"):]
    assert "--gpus" not in relabel_launch, relabel_launch
    assert "gpu_slots" not in relabel_launch, relabel_launch
    assert "--gpus" in line[: line.index("&& docker run")], line  # the collect launch still gets one


# A mistyped convention must be caught while the DAG is being built, before any stage runs, and the message must name
# the value. The check sits outside the Stage 4 rule block on purpose, so a run that freezes Stage 4 still catches it.
def test_malformed_convention_fails_at_parse_naming_the_value(tmp_path: Path) -> None:
    result = _dry_run(tmp_path, targets=[], config_overrides=["stage4={'neopax_radius_relabel': 'boozer'}"])
    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "neopax_radius_relabel" in output, output
    assert "boozer" in output, output
