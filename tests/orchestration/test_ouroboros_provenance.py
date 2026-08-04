"""Tests for the per-iteration configuration record written by ``src.ouroboros``, the closed-loop driver.

The verbatim config copy in each iteration's ``input/`` directory describes the run as it was written,
not as it was launched. The driver's placement flags (``--gpu-ids``, ``--jobs-per-gpu``,
``--container-runtime``) are handed to Snakemake as ``--config`` entries, which override the config
file's values without editing the file itself. ``--gpu-ids`` is the dangerous half of that gap, because
it decides whether every stage runs in a cpu or a gpu container image.

``effective_config.yaml`` closes the gap by recording the config with this invocation's overrides
applied. These tests check that it captures those overrides, that one is written for every iteration
using that iteration's own directories, and that it sits beside the verbatim copy rather than replacing
it -- the pair records both what was supplied and what actually ran.

The tests exercise the driver's real control flow with its two file-touching collaborators replaced by
stand-ins, so no solver and no Snakemake ever start and every path stays under pytest's ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import src.ouroboros as ouroboros
from src.utils import resolve_pipeline_paths
from tests.orchestration.test_ouroboros import _write_config


def _fake_forward_pass(*, target: str, **kwargs) -> None:
    """Stand in for ``run_forward_pass``, writing only the Stage 5 signal the loop reads back.

    The signal is neither a halt nor a convergence, so the loop runs every iteration it was asked for.
    """
    signal = Path(target)
    signal.parent.mkdir(parents=True, exist_ok=True)
    signal.write_text(json.dumps({}))


def _write_config_with_gpu_defaults(tmp_path: Path) -> Path:
    """Write a run config with the committed GPU defaults (``gpu_ids`` null, ``jobs_per_gpu`` 1, as in
    ``inputs/quick_run/config.yaml``), so an override has something to beat."""
    config_path = _write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    config["gpu_ids"] = None
    config["jobs_per_gpu"] = 1
    config_path.write_text(yaml.safe_dump(config))
    return config_path


def _run_loop(monkeypatch: pytest.MonkeyPatch, config_path: Path, *flags: str, max_iters: int = 1) -> None:
    """Run the driver with both file-touching collaborators faked, which leaves the effective config
    written exactly as a real run writes it while no input is copied and no Snakemake starts."""
    monkeypatch.setattr(ouroboros, "_seed_iteration_inputs", lambda **kw: None)
    monkeypatch.setattr(ouroboros, "run_forward_pass", _fake_forward_pass)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path),
                                     "--max-iters", str(max_iters), *flags])
    ouroboros.main()


def _effective(tmp_path: Path, n: int) -> dict:
    """Return the parsed ``<output_dir>/loop/iter_n/input/effective_config.yaml``."""
    return yaml.safe_load((tmp_path / "out" / "loop" / f"iter_{n}" / "input" / "effective_config.yaml").read_text())


# The flags travel as --config entries and write to no file, so without this record a loop launched with
# --gpu-ids all leaves gpu_ids: null on disk -- describing cpu images for work that ran on gpu ones. Each
# recorded value is checked against what the config file supplied, so a passing assertion cannot just be
# the default agreeing with the flag.
def test_effective_config_records_the_command_line_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config_with_gpu_defaults(tmp_path)
    base = yaml.safe_load(config_path.read_text())

    _run_loop(monkeypatch, config_path, "--gpu-ids", "all", "--jobs-per-gpu", "2",
              "--container-runtime", "apptainer")

    recorded = _effective(tmp_path, 1)
    assert recorded["gpu_ids"] == "all"
    assert recorded["jobs_per_gpu"] == 2
    assert recorded["container_runtime"] == "apptainer"
    assert (base["gpu_ids"], base["jobs_per_gpu"]) == (None, 1)
    # The Snakefile defaults an absent container_runtime to docker, so the flag really did change it.
    assert "container_runtime" not in base


# With no flag passed, the record must match the run config apart from the iteration's two directories.
# Comparing the whole mapping means an extra key -- a container_runtime default the driver does not own,
# say -- fails as loudly as a wrong value.
def test_effective_config_without_flags_matches_the_run_config(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config_with_gpu_defaults(tmp_path)
    base = yaml.safe_load(config_path.read_text())

    _run_loop(monkeypatch, config_path)

    assert _effective(tmp_path, 1) == {
        **base,
        "input_dir": f"{tmp_path}/out/loop/iter_1/input",
        "output_dir": f"{tmp_path}/out/loop/iter_1/output",
    }


# Each pass runs with --config input_dir/output_dir pointing at its own iter_N tree, so a record holding the
# base config's directories would name files that pass never touched. The final assertion holds the base
# directories at {tmp}/in and {tmp}/out, so the per-iteration ones cannot pass merely because the fixture
# already produced iteration-shaped paths.
def test_effective_config_is_written_for_every_iteration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path)
    base = yaml.safe_load(config_path.read_text())

    _run_loop(monkeypatch, config_path, "--gpu-ids", "4,5", max_iters=3)

    for n in (1, 2, 3):
        recorded = _effective(tmp_path, n)
        assert recorded["gpu_ids"] == "4,5"
        assert recorded["input_dir"] == f"{tmp_path}/out/loop/iter_{n}/input"
        assert recorded["output_dir"] == f"{tmp_path}/out/loop/iter_{n}/output"
    assert (base["input_dir"], base["output_dir"]) == (f"{tmp_path}/in", f"{tmp_path}/out")


# The two records answer different questions -- what the user supplied, and what the pass ran with -- so the
# new file has to sit beside the verbatim copy rather than replace it. The real _seed_iteration_inputs runs
# here, over base inputs created first, since that is the collaborator that writes the verbatim copy.
def test_effective_config_sits_alongside_the_verbatim_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config_with_gpu_defaults(tmp_path)
    base_p = resolve_pipeline_paths(yaml.safe_load(config_path.read_text()))
    for key in ("s1_input", "s3_config", "s4_config", "s5_config"):
        base_file = Path(base_p[key])
        base_file.parent.mkdir(parents=True, exist_ok=True)
        base_file.write_text(f"base {key}")

    monkeypatch.setattr(ouroboros, "run_forward_pass", _fake_forward_pass)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path), "--max-iters", "1",
                                     "--gpu-ids", "all"])
    ouroboros.main()

    iter1_input = tmp_path / "out" / "loop" / "iter_1" / "input"
    assert (iter1_input / config_path.name).read_text() == config_path.read_text()
    assert yaml.safe_load((iter1_input / config_path.name).read_text())["gpu_ids"] is None
    assert _effective(tmp_path, 1)["gpu_ids"] == "all"
