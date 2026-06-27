"""Tests for ``src.ouroboros`` (the closed-loop driver).

The driver runs repeated Snakemake forward passes, feeding each iteration's
evolved Stage 1 boundary into the next. These tests pin the orchestration logic
without running Snakemake or copying real files, using two isolation tricks:

- Paths stay in tmp. ``resolve_pipeline_paths`` builds every path as
  ``f"{output_dir}/..."`` and ``ouroboros._abs`` passes absolute paths through
  unchanged, so giving the config an absolute ``tmp_path`` ``output_dir`` makes
  every signal file the loop reads and writes land under tmp, never the repo.
- I/O stays mocked. Monkeypatching the two file-touching collaborators
  (``_seed_iteration_inputs``, ``run_forward_pass``) isolates the pure
  control-flow logic, while the fake ``run_forward_pass`` still writes the signal
  JSON the loop reads back, so the halt/converged branch is genuinely exercised.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

import src.ouroboros as ouroboros
from src.utils import resolve_pipeline_paths


def _write_config(tmp_path: Path) -> Path:
    """Write a run config whose input/output dirs are absolute under tmp_path."""
    config = {
        "run_name": "testrun",
        "input_dir": str(tmp_path / "in"),
        "output_dir": str(tmp_path / "out"),
        "filenames": {
            "s1_input": "vmec_input.{run_name}",
            "s1_output": "wout_{run_name}.nc",
            "s2_output": "boozmn_{run_name}.nc",
            "s3_config": "sfincs_input.{run_name}",
            "s3_output": "sfincs_flux.h5",
            "s4_config": "{run_name}.toml",
            "s4_output": "neopax_fluxes.h5",
            "s5_config": "common_input.toml",
            "s5_output": "transport_solution.h5",
            "s5_signal": "converge_status.json",
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_run_forward_pass_builds_snakemake_argv(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_subprocess_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs

    monkeypatch.setattr(ouroboros.subprocess, "run", fake_subprocess_run)
    ouroboros.run_forward_pass(
        target="out/converge_status.json",
        input_dir="in",
        output_dir="out",
        cores=4,
        config_path=Path("cfg.yaml"),
        repo_root=Path("/repo"),
    )
    assert captured["cmd"] == [
        "snakemake", "out/converge_status.json", "--cores", "4",
        "--configfile", "cfg.yaml",
        "--config", "input_dir=in", "output_dir=out",
    ]
    assert captured["kwargs"]["cwd"] == Path("/repo")
    assert captured["kwargs"]["check"] is True


def test_main_rejects_nonpositive_max_iters(monkeypatch: pytest.MonkeyPatch) -> None:
    # The guard fires before any config read or forward pass.
    monkeypatch.setattr("sys.argv", ["ouroboros", "--max-iters", "0"])
    with pytest.raises(ValueError, match="max-iters"):
        ouroboros.main()


def test_loop_seeds_from_previous_feedback(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    base_p = resolve_pipeline_paths(config)
    base_out = config["output_dir"]
    iter1_p = resolve_pipeline_paths(
        config,
        input_dir=f"{base_out}/loop/iter_1/input",
        output_dir=f"{base_out}/loop/iter_1/output",
    )

    seeded: list[str] = []
    monkeypatch.setattr(ouroboros, "_seed_iteration_inputs",
                        lambda **kw: seeded.append(kw["s1_source"]))

    def fake_run(*, target, **kw):
        signal = Path(target)
        signal.parent.mkdir(parents=True, exist_ok=True)
        signal.write_text(json.dumps({}))  # neither halt nor converged -> keep looping

    monkeypatch.setattr(ouroboros, "run_forward_pass", fake_run)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path), "--max-iters", "2"])
    ouroboros.main()

    # Iteration 1 seeds the base boundary; iteration 2 seeds the prior iteration's feedback.
    assert seeded == [base_p["s1_input"], iter1_p["s1_feedback"]]


@pytest.mark.parametrize("signal", [{"halt": True}, {"converged": True}])
def test_loop_short_circuits_on_terminal_signal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, signal: dict
) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(ouroboros, "_seed_iteration_inputs", lambda **kw: None)

    calls: list[str] = []

    def fake_run(*, target, **kw):
        calls.append(target)
        p = Path(target)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(signal))

    monkeypatch.setattr(ouroboros, "run_forward_pass", fake_run)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path), "--max-iters", "3"])
    ouroboros.main()

    # A halt or converged signal after iteration 1 stops the loop despite max-iters=3.
    assert len(calls) == 1
