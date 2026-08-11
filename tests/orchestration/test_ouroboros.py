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


def _write_config(tmp_path: Path, loop: dict | None = None) -> Path:
    """Write a run config whose input/output dirs are absolute under tmp_path.

    Passing ``loop`` adds that block to the config, which is how the per-stage rerun flags reach the driver.
    """
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
    if loop is not None:
        config["loop"] = loop
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


def test_run_forward_pass_can_disable_shared_workdir_lock(monkeypatch: pytest.MonkeyPatch) -> None:
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
        nolock=True,
    )

    assert "--nolock" in captured["cmd"]


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
# none on iteration 1, and on iteration 2 a real YAML file setting profiles_source to prescribed for both stages. A run
# that freezes nothing must not emit a loop block either, since that block is what makes the Snakefile drop stage rules.
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
    assert overrides["stage4"]["gkx"]["profiles_source"] == "prescribed"
    assert "loop" not in overrides
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


# A stage can only be frozen when every stage feeding it is frozen too, so freezing Stage 3 while Stage 1 still reruns
# is rejected. The driver resolves the flags right after reading the config and before it creates anything, so a config
# that can never run costs no files. This asserts main() raises and that no iteration tree was left behind.
def test_main_rejects_frozen_stage_fed_by_a_rerunning_stage(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, loop={"rerun": {"stage3": False}})
    monkeypatch.setattr(ouroboros, "_seed_iteration_inputs", lambda **kw: None)
    monkeypatch.setattr(ouroboros, "run_forward_pass", lambda **kw: None)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path)])

    with pytest.raises(ValueError, match="stage3"):
        ouroboros.main()

    assert not (tmp_path / "out" / "loop").exists()


# reuse_output_dir is the actuation signal the driver writes into loop_overrides.yaml for iterations 2 and later, so a
# run config carrying it would freeze stages already in iteration 1 and in plain forward passes. The driver must reject
# it at startup, before any iteration tree is created.
def test_main_rejects_reuse_output_dir_in_run_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, loop={"reuse_output_dir": f"{tmp_path}/reuse"})
    monkeypatch.setattr(ouroboros, "_seed_iteration_inputs", lambda **kw: None)
    monkeypatch.setattr(ouroboros, "run_forward_pass", lambda **kw: None)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path)])

    with pytest.raises(ValueError, match="reuse_output_dir"):
        ouroboros.main()

    assert not (tmp_path / "out" / "loop").exists()


# With every stage frozen no artifact can change between iterations, so the loop has nothing left to feed forward and
# must stop after one forward pass however many iterations were asked for. The fake run writes a signal that is neither
# a halt nor a convergence, so a second pass would follow if the all-frozen case did not cap the iteration count.
def test_loop_runs_a_single_pass_when_every_stage_is_frozen(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    frozen = {"stage1": False, "stage2": False, "stage3": False, "stage4": False, "stage5": False}
    config_path = _write_config(tmp_path, loop={"rerun": frozen})
    monkeypatch.setattr(ouroboros, "_seed_iteration_inputs", lambda **kw: None)

    calls: list[str] = []

    def fake_run(*, target, **kw):
        calls.append(target)
        signal = Path(target)
        signal.parent.mkdir(parents=True, exist_ok=True)
        signal.write_text(json.dumps({}))

    monkeypatch.setattr(ouroboros, "run_forward_pass", fake_run)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path), "--max-iters", "3"])
    ouroboros.main()

    assert len(calls) == 1


# Freezing Stage 1 pins the boundary, so every iteration must be seeded from the base boundary and the evolved boundary
# that post-processing still writes must be passed over. The profile feedback is independent of that choice, so
# iteration 2 must still take its common_input from iteration 1's prescribed-profiles output.
def test_frozen_stage1_keeps_seeding_the_base_boundary(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, loop={"rerun": {"stage1": False}})
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
        signal.write_text(json.dumps({}))

    monkeypatch.setattr(ouroboros, "run_forward_pass", fake_run)
    monkeypatch.setattr("sys.argv", ["ouroboros", "--config", str(config_path), "--max-iters", "2"])
    ouroboros.main()

    assert seeded == [
        (base_p["s1_input"], base_p["s5_config"]),
        (base_p["s1_input"], iter1_p["s5_config_feedback"]),
    ]


def _overrides_of_second_iteration(monkeypatch: pytest.MonkeyPatch, config_path: Path) -> dict:
    """Run two loop iterations and return iteration 2's parsed overrides file.

    Only the two file-touching collaborators are faked, so the overrides file is written exactly as a real run would
    write it while no solver runs and no input is copied.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to fake the collaborators and the command line.
    config_path : Path
        Run config handed to the driver.

    Returns
    -------
    dict
        Parsed contents of the overrides file iteration 2 was run with.
    """
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
    return yaml.safe_load(extras[1][0].read_text())


# The overrides file is the only channel telling the Snakefile which stages to leave out of an iteration, so with Stages
# 1 and 2 frozen it must carry the complete five-stage flag map, not just the frozen ones, along with the iteration 1
# output tree their artifacts stay in. The prescribed switch for the two stages that still rerun must survive alongside.
def test_overrides_carry_rerun_flags_and_the_reuse_tree(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config_path = _write_config(tmp_path, loop={"rerun": {"stage1": False, "stage2": False}})

    overrides = _overrides_of_second_iteration(monkeypatch, config_path)

    assert overrides["stage3"]["sfincs_jax"]["profiles_source"] == "prescribed"
    assert overrides["stage4"]["gkx"]["profiles_source"] == "prescribed"
    assert overrides["loop"]["rerun"] == {
        "stage1": False, "stage2": False, "stage3": True, "stage4": True, "stage5": True,
    }
    assert overrides["loop"]["reuse_output_dir"] == f"{tmp_path}/out/loop/iter_1/output"


# A frozen stage never runs, so switching it to prescribed profiles would describe a run that does not happen. With
# Stages 1 and 3 frozen the overrides file must drop the stage3 block entirely while still switching the Stage 4 run and
# still carrying the loop block, since Stage 4 and Stage 5 do rerun and read the prescribed profiles.
def test_overrides_omit_the_prescribed_switch_for_a_frozen_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = _write_config(tmp_path, loop={"rerun": {"stage1": False, "stage3": False}})

    overrides = _overrides_of_second_iteration(monkeypatch, config_path)

    assert "stage3" not in overrides
    assert overrides["stage4"]["gkx"]["profiles_source"] == "prescribed"
    assert overrides["loop"]["rerun"]["stage3"] is False
    assert overrides["loop"]["reuse_output_dir"] == f"{tmp_path}/out/loop/iter_1/output"


# The Snakefile only switches into reuse mode when it sees loop.reuse_output_dir, so a call that passes no rerun flags
# must keep producing the bare prescribed switch for Stages 3 and 4 with no loop block attached, leaving every run that
# does not freeze anything exactly as it was.
def test_write_loop_overrides_defaults_to_the_prescribed_switch_only(tmp_path: Path) -> None:
    path = ouroboros._write_loop_overrides(tmp_path)

    assert yaml.safe_load(path.read_text()) == {
        "stage3": {"sfincs_jax": {"profiles_source": "prescribed"}},
        "stage4": {"gkx": {"profiles_source": "prescribed"}},
    }


# Naming a frozen stage without naming the tree its artifacts live in would leave the Snakefile unable to find them, so
# that combination is a driver bug rather than a bad config and must fail at the write instead of producing a file the
# Snakefile would reject later.
def test_write_loop_overrides_requires_a_reuse_tree_when_a_stage_is_frozen(tmp_path: Path) -> None:
    flags = {"stage1": False, "stage2": True, "stage3": True, "stage4": True, "stage5": True}

    with pytest.raises(ValueError, match="reuse_output_dir"):
        ouroboros._write_loop_overrides(tmp_path, rerun=flags)
