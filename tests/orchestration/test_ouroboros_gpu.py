"""Tests for the GPU pool flags of ``src.ouroboros`` (the closed-loop driver).

``--gpu-ids`` and ``--jobs-per-gpu`` let one invocation place its containers without
editing the committed run config, so the flags have to reach every iteration and have
to be validated before the loop starts building iteration trees. These tests pin both
halves of that, plus the requirement that an invocation which passes neither flag
produces the command line it always did.

The isolation tricks are the ones documented in ``tests/orchestration/test_ouroboros.py``,
whose ``_write_config`` helper is reused here. Paths stay under tmp and the two
file-touching collaborators are faked, so the control flow runs with no solver and no
Snakemake.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
import yaml

import src.ouroboros as ouroboros
from tests.orchestration.test_ouroboros import _write_config

# The argv a forward pass builds before any GPU flag is added. Overrides passed as --config entries have to come last,
# after the dir overrides, because snakemake's --config takes every following token until the next flag.
BASE_ARGV = [
    "snakemake", "out/converge_status.json", "--cores", "4",
    "--configfile", "cfg.yaml",
    "--config", "input_dir=in", "output_dir=out",
]


def _captured_argv(monkeypatch: pytest.MonkeyPatch, **kwargs) -> list[str]:
    """Run one forward pass against a fake ``subprocess.run`` and return the argv it was handed.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace ``subprocess.run`` so no Snakemake starts.
    **kwargs
        Extra keyword arguments for ``run_forward_pass``, on top of the fixed target, dirs, cores, config and root.

    Returns
    -------
    list[str]
        The command the driver would have run.
    """
    captured: dict = {}

    def fake_subprocess_run(cmd, **kw):
        captured["cmd"] = cmd

    monkeypatch.setattr(ouroboros.subprocess, "run", fake_subprocess_run)
    ouroboros.run_forward_pass(
        target="out/converge_status.json",
        input_dir="in",
        output_dir="out",
        cores=4,
        config_path=Path("cfg.yaml"),
        repo_root=Path("/repo"),
        **kwargs,
    )
    return captured["cmd"]


def _write_config_with_device(tmp_path: Path) -> Path:
    """Write a run config that still carries the retired top-level ``device`` key.

    Parameters
    ----------
    tmp_path : Path
        Directory the config and its input/output dirs live under.

    Returns
    -------
    Path
        The written config file.
    """
    config_path = _write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["device"] = "gpu"
    config_path.write_text(yaml.safe_dump(config))
    return config_path


# The GPU settings must reach snakemake as --config entries so they beat the run config file and every overrides file
# the loop layers on top of it. This pins them landing at the very end of the --config group, as raw key=value text that
# snakemake re-parses exactly as it would the same values written in a config file.
def test_extra_config_lands_at_the_end_of_the_config_group(monkeypatch: pytest.MonkeyPatch) -> None:
    cmd = _captured_argv(monkeypatch, extra_config=["gpu_ids=4,5", "jobs_per_gpu=2"])

    assert cmd == [*BASE_ARGV, "gpu_ids=4,5", "jobs_per_gpu=2"]


# An invocation that passes neither flag must run the command line it ran before the flags existed, so adding them
# cannot have changed any established run. Passing None explicitly and leaving the argument out must both do that.
def test_no_extra_config_leaves_the_command_line_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _captured_argv(monkeypatch, extra_config=None) == BASE_ARGV
    assert _captured_argv(monkeypatch) == BASE_ARGV


# The flags describe the machine the whole invocation runs on, not one pass of it, so every iteration must be given the
# same placement. This records what each forward pass received across two iterations and requires both to carry the
# flags verbatim, since an iteration that dropped them would fall back to the config file's cpu setting mid-run.
def test_flags_reach_every_iteration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(ouroboros, "_seed_iteration_inputs", lambda **kw: None)

    forwarded: list = []

    def fake_run(*, target, extra_config=None, **kw):
        forwarded.append(extra_config)
        signal = Path(target)
        signal.parent.mkdir(parents=True, exist_ok=True)
        signal.write_text(json.dumps({"status": "continue"}))  # neither halt nor converged -> keep looping

    monkeypatch.setattr(ouroboros, "run_forward_pass", fake_run)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path), "--max-iters", "2",
                                     "--gpu-ids", "4,5", "--jobs-per-gpu", "2"])
    ouroboros.main()

    assert forwarded == [["gpu_ids=4,5", "jobs_per_gpu=2"], ["gpu_ids=4,5", "jobs_per_gpu=2"]]


# Raising jobs_per_gpu for a cpu run oversubscribes nothing, because a cpu run has no device to share out. The driver
# resolves the settings from an overlaid copy of the config right after reading it, so the combination is rejected
# before the first iteration tree is created and the user loses no work to a flag typo. The flags are checked as the
# config values they become, so "null" has to reach the same guard the config file's own null would.
def test_cpu_run_with_a_raised_job_count_is_rejected_before_any_iteration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(ouroboros, "_seed_iteration_inputs", lambda **kw: None)
    monkeypatch.setattr(ouroboros, "run_forward_pass", lambda **kw: None)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path),
                                     "--gpu-ids", "null", "--jobs-per-gpu", "2"])

    with pytest.raises(ValueError, match=re.escape("config['jobs_per_gpu'] is 2 but config['gpu_ids'] asks")):
        ouroboros.main()

    assert not (tmp_path / "out" / "loop").exists()


# A run config written before the key was replaced would otherwise run the whole loop on cpu images while the user
# believes they asked for GPUs. The driver applies the same guard the Snakefile does, at startup, and its message names
# the replacement key so the config can be migrated.
def test_retired_device_key_is_rejected_at_startup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = _write_config_with_device(tmp_path)
    monkeypatch.setattr(ouroboros, "_seed_iteration_inputs", lambda **kw: None)
    monkeypatch.setattr(ouroboros, "run_forward_pass", lambda **kw: None)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path)])

    with pytest.raises(ValueError, match="gpu_ids"):
        ouroboros.main()

    assert not (tmp_path / "out" / "loop").exists()
