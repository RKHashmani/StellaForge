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


# `run_forward_pass` should shell out to Snakemake with a specific command. Rather than actually run it, this uses
# monkeypatch to replace `subprocess.run` with a fake that records the command it was handed, then asserts the exact
# argument list (target, cores, config file, dir overrides) and that it runs in the repo root with `check=True` (so a
# failing Snakemake call raises).
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


# Snakemake's --configfile is a nargs="+" argument, so extra config files must be appended after the base config file
# under the same single flag (a second --configfile occurrence would replace the first). This pins that argv shape:
# extras go between the base config file and the --config overrides.
def test_run_forward_pass_appends_extra_configfiles(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    def fake_subprocess_run(cmd, **kwargs):
        captured["cmd"] = cmd

    monkeypatch.setattr(ouroboros.subprocess, "run", fake_subprocess_run)
    ouroboros.run_forward_pass(
        target="out/converge_status.json",
        input_dir="in",
        output_dir="out",
        cores=4,
        config_path=Path("cfg.yaml"),
        repo_root=Path("/repo"),
        extra_configfiles=[Path("in/loop_overrides.yaml")],
    )
    assert captured["cmd"] == [
        "snakemake", "out/converge_status.json", "--cores", "4",
        "--configfile", "cfg.yaml", "in/loop_overrides.yaml",
        "--config", "input_dir=in", "output_dir=out",
    ]


# Checks input validation and its ordering. It runs `main()` with `--max-iters 0` and a config path that doesn't exist.
# The test asserts it raises ValueError about max-iters (not a file-not-found error), proving the loop-count guard runs
# before the config is ever read.
def test_main_rejects_nonpositive_max_iters(monkeypatch: pytest.MonkeyPatch) -> None:
    # A nonexistent --config pins the ordering: the max-iters guard must fire before any
    # config read, so main() raises ValueError here rather than FileNotFoundError.
    monkeypatch.setattr("sys.argv", ["ouroboros", "--max-iters", "0", "--config", "/no/such/config.yaml"])
    with pytest.raises(ValueError, match="max-iters"):
        ouroboros.main()


# Verifies the "ouroboros" feedback for both evolving inputs. Each iteration should seed the Stage 1 boundary and the
# shared common_input template from the previous iteration's feedback artifacts. It monkeypatches the two file-touching
# helpers (so no solver runs, but a signal file is still written), records the (s1_source, s5_source) pair each
# iteration is handed, and asserts iteration 1 seeds from the base inputs while iteration 2 seeds from iteration 1's
# feedback outputs.
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

    seeded: list[tuple[str, str]] = []
    monkeypatch.setattr(ouroboros, "_seed_iteration_inputs",
                        lambda **kw: seeded.append((kw["s1_source"], kw["s5_source"])))

    def fake_run(*, target, **kw):
        signal = Path(target)
        signal.parent.mkdir(parents=True, exist_ok=True)
        signal.write_text(json.dumps({}))  # neither halt nor converged -> keep looping

    monkeypatch.setattr(ouroboros, "run_forward_pass", fake_run)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path), "--max-iters", "2"])
    ouroboros.main()

    assert seeded == [
        (base_p["s1_input"], base_p["s5_config"]),
        (iter1_p["s1_feedback"], iter1_p["s5_config_feedback"]),
    ]


# The loop tests above mock _seed_iteration_inputs, so this exercises the real copies once. Stage 3/4 configs and the
# run config must come from the base inputs while s1_input and s5_config come from the s1_source/s5_source arguments.
# Distinct file contents prove each destination received its intended source rather than a sibling's.
def test_seed_iteration_inputs_copies_each_source(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)
    config = yaml.safe_load(config_path.read_text())
    base_p = resolve_pipeline_paths(config)
    iter_p = resolve_pipeline_paths(
        config,
        input_dir=f"{config['output_dir']}/loop/iter_2/input",
        output_dir=f"{config['output_dir']}/loop/iter_2/output",
    )
    for key in ("s3_config", "s4_config"):
        base = Path(base_p[key])
        base.parent.mkdir(parents=True, exist_ok=True)
        base.write_text(f"base {key}")
    s1_source = tmp_path / "evolved_boundary"
    s1_source.write_text("evolved boundary")
    s5_source = tmp_path / "prescribed_common_input.toml"
    s5_source.write_text("prescribed profiles")

    ouroboros._seed_iteration_inputs(
        repo_root=tmp_path, base_p=base_p, iter_p=iter_p,
        s1_source=str(s1_source), s5_source=str(s5_source), config_path=config_path,
    )

    assert Path(iter_p["s3_config"]).read_text() == "base s3_config"
    assert Path(iter_p["s4_config"]).read_text() == "base s4_config"
    assert Path(iter_p["s1_input"]).read_text() == "evolved boundary"
    assert Path(iter_p["s5_config"]).read_text() == "prescribed profiles"
    assert (Path(iter_p["input_dir"]) / config_path.name).read_text() == config_path.read_text()


# From iteration 2 the loop must pass an overrides file that flips Stages 3/4 to the prescribed profiles carried by the
# seeded common_input.toml. This records the extra_configfiles each forward pass receives and checks the file's content:
# none on iteration 1, and on iteration 2 a real YAML file setting profiles_source to prescribed for both stages.
def test_loop_passes_prescribed_overrides_from_second_iteration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path)
    monkeypatch.setattr(ouroboros, "_seed_iteration_inputs", lambda **kw: None)

    extras: list = []

    def fake_run(*, target, extra_configfiles=None, **kw):
        extras.append(extra_configfiles)
        signal = Path(target)
        signal.parent.mkdir(parents=True, exist_ok=True)
        signal.write_text(json.dumps({}))

    monkeypatch.setattr(ouroboros, "run_forward_pass", fake_run)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path), "--max-iters", "2"])
    ouroboros.main()

    assert extras[0] is None
    assert len(extras[1]) == 1
    overrides = yaml.safe_load(extras[1][0].read_text())
    assert overrides["stage3"]["sfincs_jax"]["profiles_source"] == "prescribed"
    assert overrides["stage4"]["spectrax_gk"]["profiles_source"] == "prescribed"
    # The overrides file lives inside the iteration's input dir, alongside the seeded inputs.
    assert str(extras[1][0]).startswith(f"{tmp_path}/out/loop/iter_2/input")


# The loop should stop early when a run reports it has converged or been told to halt. `@pytest.mark.parametrize` runs
# this twice, once with a `halt` signal and once with a `converged` signal. Each fake run writes that signal, and the
# test asserts only one iteration happened even though max-iters was 3, proving the terminal signal short-circuits the
# loop.
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
