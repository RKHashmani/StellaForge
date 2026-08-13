"""Ouroboros: closed-loop driver (Stage 5 -> Stage 1 pressure feedback, Stage 5 -> Stages 3/4/5 profile feedback).

Runs the forward pass repeatedly, feeding each iteration's evolved Stage 1 input and
common input into the next. After the first iteration, the driver also switches Stages 3 and 4
to ``profiles_source: prescribed`` via a config overrides file, so every profile consumer
reads the transport-evolved profiles. Every iteration runs under its own
``<output_dir>/loop/iter_N/``, where ``input/`` holds that pass's seeded inputs, a verbatim copy of the
run config, and the ``effective_config.yaml`` recording that config with this invocation's command-line
overrides applied, while ``output/`` holds the stage outputs that pass computed (feedback artifacts and
convergence signal). The committed inputs under ``input_dir`` are only ever read, never modified.

An optional ``loop.rerun`` block in the run config flags stages false to freeze them. When a stage is
frozen the overrides file also carries the flag map and the iteration 1 tree, so the Snakefile reads that
stage's artifacts from there. With every stage frozen the driver runs a single forward pass and stops.

The ``--gpu-ids``, ``--jobs-per-gpu`` and ``--container-runtime`` flags override the run config's
matching top-level keys for every iteration of one invocation, leaving the config file itself
untouched. They reach Snakemake as ``--config`` entries, so each iteration's ``effective_config.yaml``
is what records the values that pass ran with.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from .signal_contract import validate_signal
from .utils import LOOP_STAGES, resolve_gpu_settings, resolve_pipeline_paths, resolve_rerun_flags

logger = logging.getLogger(__name__)


def _snakemake_command(profile: str | None) -> list[str]:
    """Return the normal CLI or the submit-side HTCondor compatibility CLI."""
    if profile and "htcondor" in profile.lower():
        return [sys.executable, "-m", "src.snakemake_htcondor_compat"]
    return ["snakemake"]


def _abs(repo_root: Path, path: str) -> Path:
    """Resolve a pipeline path against ``repo_root`` (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else repo_root / p


def _seed(src: Path, dst: Path) -> None:
    """Copy ``src`` to ``dst``, creating ``dst``'s parent, to materialize an iteration input."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _seed_iteration_inputs(
    *,
    repo_root: Path,
    base_p: dict[str, str],
    iter_p: dict[str, str],
    s1_source: str,
    s5_source: str,
    config_path: Path,
) -> None:
    """Populate one iteration's ``input/`` directory.

    The Stage 3/4 configs and the run config are copied from the base inputs every
    iteration. The Stage 1 boundary comes from ``s1_source`` and the ``common_input``
    template from ``s5_source``, which point at the base files on iteration 1 and at
    the previous iteration's evolved boundary and prescribed profiles afterward.

    Parameters
    ----------
    repo_root : Path
        Repository root; pipeline paths are resolved against it.
    base_p, iter_p : dict[str, str]
        ``resolve_pipeline_paths`` output for the base config and for this iteration.
    s1_source : str
        Path to the Stage 1 boundary to seed as this iteration's ``s1_input``.
    s5_source : str
        Path to the ``common_input`` template to seed as this iteration's ``s5_config``.
    config_path : Path
        Run config file, copied into the iteration input dir as a record.
    """
    for key in ("s3_config", "s4_config"):
        _seed(_abs(repo_root, base_p[key]), _abs(repo_root, iter_p[key]))
    _seed(config_path, _abs(repo_root, iter_p["input_dir"]) / config_path.name)
    _seed(_abs(repo_root, s1_source), _abs(repo_root, iter_p["s1_input"]))
    _seed(_abs(repo_root, s5_source), _abs(repo_root, iter_p["s5_config"]))


def _write_loop_overrides(
    iter_input_dir: Path,
    *,
    rerun: dict[str, bool] | None = None,
    reuse_output_dir: str | None = None,
) -> Path:
    """Write the config overrides that iterations 2 and later are run with.

    Stages 3 and 4 are switched to ``profiles_source: prescribed`` when they rerun,
    matching the profiles the seeded ``common_input.toml`` carries. When any stage is
    frozen the file also adds the rerun flags and the iteration 1 tree.

    Parameters
    ----------
    iter_input_dir : Path
        The iteration's ``input/`` directory; the overrides file is written inside it
        so every input that shaped the pass is recorded there.
    rerun : dict[str, bool], optional
        Rerun flag for every stage, as returned by ``resolve_rerun_flags``. ``None``
        means every stage reruns.
    reuse_output_dir : str, optional
        Output directory of iteration 1, required whenever a stage is frozen and unused
        otherwise.

    Returns
    -------
    Path
        Path of the written overrides file.

    Raises
    ------
    ValueError
        If a stage is frozen while ``reuse_output_dir`` is ``None``.
    """
    flags = {stage: True for stage in LOOP_STAGES} if rerun is None else rerun
    frozen = [stage for stage in LOOP_STAGES if not flags[stage]]
    if frozen and reuse_output_dir is None:
        raise ValueError(
            f"reuse_output_dir is required to write overrides that freeze {frozen}, because the Snakefile can only "
            "reuse a frozen stage's artifacts once it is told which output tree holds them."
        )

    switched = [stage for stage in ("stage3", "stage4") if flags[stage]]
    if not frozen:
        text = (
            "# Written by the loop driver for iterations 2 and later: the seeded\n"
            "# common_input.toml carries prescribed profiles from the previous iteration's\n"
            "# transport solution, and Stages 3 and 4 must read those instead of the\n"
            "# analytical profile parameters.\n"
        )
    elif switched:
        names = " and ".join(f"Stage {stage.removeprefix('stage')}" for stage in switched)
        text = (
            "# Written by the loop driver for iterations 2 and later. The seeded common_input.toml\n"
            "# carries prescribed profiles from the previous iteration's transport solution, and\n"
            f"# {names} must read those instead of the analytical profile parameters.\n"
        )
    else:
        text = ""
    for stage, section in (("stage3", "sfincs_jax"), ("stage4", "gkx")):
        if flags[stage]:
            text += f"{stage}:\n  {section}:\n    profiles_source: prescribed\n"
    if frozen:
        text += (
            "# The loop block tells the Snakefile which stages are frozen and where iteration 1 wrote\n"
            "# its outputs, so this pass reads their artifacts from that tree.\n"
            "loop:\n"
            "  rerun:\n"
        )
        text += "".join(f"    {stage}: {str(flags[stage]).lower()}\n" for stage in LOOP_STAGES)
        text += f"  reuse_output_dir: {json.dumps(reuse_output_dir)}\n"

    overrides = iter_input_dir / "loop_overrides.yaml"
    overrides.parent.mkdir(parents=True, exist_ok=True)
    overrides.write_text(text)
    return overrides


def _write_effective_config(iter_input_dir: Path, config: dict, *, input_dir: str, output_dir: str) -> Path:
    """Record the configuration one iteration is run with.

    The verbatim copy of the run config beside this file records what the user supplied, but this
    invocation's command-line overrides reach Snakemake through ``--config`` and would otherwise leave
    no trace, so a loop launched with ``--gpu-ids all`` would be recorded as the cpu run its config file
    still describes. It sits beside that copy rather than replacing it, so the pair records both.
    ``_write_loop_overrides`` records the per-iteration deltas in the same directory.

    Parameters
    ----------
    iter_input_dir : Path
        The iteration's ``input/`` directory, where the seeded inputs and the loop overrides live too.
    config : dict
        Run config with this invocation's command-line overrides already applied.
    input_dir, output_dir : str
        This iteration's directories, replacing the base config's because they are the ones it used.

    Returns
    -------
    Path
        Path of the written config file.
    """
    recorded = {**config, "input_dir": input_dir, "output_dir": output_dir}
    iter_input_dir.mkdir(parents=True, exist_ok=True)
    effective = iter_input_dir / "effective_config.yaml"
    effective.write_text(
        "# Written by the loop driver. The run config with this invocation's command-line overrides\n"
        "# applied and under this iteration's own directories. From iteration 2 the loop_overrides.yaml\n"
        "# beside it layers the prescribed-profiles switch on top of this, and the verbatim copy of the\n"
        "# run config alongside both records what the user supplied.\n"
        + yaml.safe_dump(recorded, sort_keys=True)
    )
    return effective


def run_forward_pass(
    *,
    target: str,
    input_dir: str,
    output_dir: str,
    cores: int,
    config_path: Path,
    repo_root: Path,
    extra_configfiles: list[Path] | None = None,
    extra_config: list[str] | None = None,
    profile: str | None = None,
    nolock: bool = False,
    htcondor_jobdir: str | None = None,
) -> None:
    """
    Run one Snakemake forward pass for an iteration.

    Parameters
    ----------
    extra_configfiles : list[Path], optional
        Config files merged on top of ``config_path`` (e.g. the loop overrides that
        switch Stages 3/4 to prescribed profiles from iteration 2 on).
    extra_config : list[str], optional
        Further ``key=value`` entries for the ``--config`` group (e.g. the GPU pool
        settings this invocation was given), taking precedence over every config file.
    profile : str, optional
        Snakemake profile directory (e.g. ``executors/htcondor/profiles/htcondor-gpu``). Sends the
        iteration's jobs to a cluster executor instead of running them locally.
    nolock : bool, optional
        Disable Snakemake's repository-wide working-directory lock. This is safe only when concurrent
        controllers have disjoint input/output trees, as the batch runner guarantees.
    htcondor_jobdir : str, optional
        Per-run directory for HTCondor submit, error, output, and unified event-log files. Parallel
        controllers must not share this directory.
    """
    logger.info("Forward pass [%s]: snakemake %s --cores %d", output_dir, target, cores)
    # Extra config files are appended under the single --configfile flag. A second flag
    # occurrence would replace the first and silently drop the base config. Later files
    # take precedence in the deep merge, and --config key=value overrides apply last.
    cmd = [*_snakemake_command(profile), target, "--cores", str(cores)]
    if nolock:
        cmd.append("--nolock")
    if profile:
        cmd += ["--profile", profile]
    if htcondor_jobdir:
        cmd += ["--htcondor-jobdir", htcondor_jobdir]
    cmd += ["--configfile", str(config_path)]
    cmd += [str(p) for p in (extra_configfiles or [])]
    cmd += ["--config", f"input_dir={input_dir}", f"output_dir={output_dir}", *(extra_config or [])]
    subprocess.run(cmd, cwd=repo_root, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Closed-loop driver (Stage 5 -> Stage 1 pressure feedback)."
    )
    parser.add_argument("--config", type=Path, default=Path("inputs/quick_run/config.yaml"),
                        help="Pipeline run config (default: inputs/quick_run/config.yaml).")
    parser.add_argument("--max-iters", type=int, default=3,
                        help="Cap on loop iterations; the convergence signal can stop the loop earlier (default: 3).")
    parser.add_argument("--cores", type=int, default=4,
                        help="Cores passed to 'snakemake --cores' (default: 4).")
    parser.add_argument("--gpu-ids", type=str, default=None,
                        help="Override the config's gpu_ids for every iteration (null, all, or a comma-separated "
                             "id list).")
    parser.add_argument("--jobs-per-gpu", type=int, default=None,
                        help="Override the config's jobs_per_gpu for every iteration.")
    parser.add_argument("--profile", type=str, default=None,
                        help="Snakemake profile directory passed to every iteration (e.g. "
                             "executors/htcondor/profiles/htcondor-gpu to run on HTCondor).")
    parser.add_argument("--container-runtime", type=str, default=None, choices=["docker", "apptainer"],
                        help="Override the config's container_runtime for every iteration. A cluster "
                             "profile normally needs 'apptainer', since the execute nodes have no "
                             "Docker daemon.")
    parser.add_argument("--nolock", action="store_true",
                        help="Disable Snakemake's working-directory lock. Intended for parallel controllers "
                             "whose run input/output trees do not overlap.")
    parser.add_argument("--htcondor-jobdir", type=str, default=None,
                        help="Per-run HTCondor log directory. Parallel controllers must use distinct values.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if args.max_iters < 1:
        raise ValueError(f"--max-iters must be >= 1, got {args.max_iters}")

    repo_root = Path(__file__).resolve().parent.parent
    config_path = args.config if args.config.is_absolute() else repo_root / args.config
    config = yaml.safe_load(config_path.read_text())
    rerun = resolve_rerun_flags(config)
    if (config.get("loop") or {}).get("reuse_output_dir") is not None:
        raise ValueError(
            "config['loop']['reuse_output_dir'] is written by the driver into each iteration's loop_overrides.yaml "
            "and must not appear in the run config, where it would freeze stages already in iteration 1."
        )

    # The run config with this invocation's command-line overrides applied. It both validates the GPU settings before
    # any iteration tree is created and is recorded per iteration as what that pass ran with. yaml.safe_load turns
    # "null" into None and "4" into an integer, while "all" and "4,5" stay strings.
    effective = dict(config)
    if args.gpu_ids is not None:
        effective["gpu_ids"] = yaml.safe_load(args.gpu_ids)
    if args.jobs_per_gpu is not None:
        effective["jobs_per_gpu"] = args.jobs_per_gpu
    if args.container_runtime is not None:
        effective["container_runtime"] = args.container_runtime
    resolve_gpu_settings(effective)

    # The flags are forwarded as raw text so snakemake re-parses each value exactly as a config file would.
    config_overrides: list[str] = []
    if args.gpu_ids is not None:
        config_overrides.append(f"gpu_ids={args.gpu_ids}")
    if args.jobs_per_gpu is not None:
        config_overrides.append(f"jobs_per_gpu={args.jobs_per_gpu}")
    if args.container_runtime is not None:
        config_overrides.append(f"container_runtime={args.container_runtime}")

    base_out = config["output_dir"]
    base_p = resolve_pipeline_paths(config)

    max_iters = args.max_iters
    if not any(rerun.values()):
        logger.info("Every stage is frozen by loop.rerun, so no artifact can change between iterations; the driver "
                    "runs a single forward pass and stops, ignoring --max-iters %d.", max_iters)
        max_iters = 1
    # Only a frozen stage needs the iteration 1 tree.
    reuse_output_dir = None if all(rerun.values()) else f"{base_out}/loop/iter_1/output"

    # Each iteration has a full forward pass in its own '<output_dir>/loop/iter_N/' tree.
    # The prior iteration supplies the Stage 1 boundary and the ``common_input`` template.
    # This template contains prescribed profiles and the advanced transport clock.
    # Base inputs supply all other files. From iteration 2, an override makes Stages 3 and 4 read
    # prescribed profiles from ``common_input``. Separate trees force each enabled stage to run
    # again. A frozen stage reads its artifacts from the iteration 1 tree. A frozen Stage 1 uses the
    # base boundary, so no later iteration reads the evolved boundary.
    prev_p: dict[str, str] | None = None
    n = 0
    for n in range(1, max_iters + 1):
        iter_in = f"{base_out}/loop/iter_{n}/input"
        iter_out = f"{base_out}/loop/iter_{n}/output"
        logger.info("=== Iteration %d of %d (output=%s) ===", n, max_iters, iter_out)
        iter_p = resolve_pipeline_paths(config, input_dir=iter_in, output_dir=iter_out)

        s1_source = base_p["s1_input"] if n == 1 or not rerun["stage1"] else prev_p["s1_feedback"]
        s5_source = base_p["s5_config"] if n == 1 else prev_p["s5_config_feedback"]
        _seed_iteration_inputs(repo_root=repo_root, base_p=base_p, iter_p=iter_p,
                               s1_source=s1_source, s5_source=s5_source, config_path=config_path)

        iter_input_dir = _abs(repo_root, iter_p["input_dir"])
        _write_effective_config(iter_input_dir, effective, input_dir=iter_in, output_dir=iter_out)
        extra_configfiles = None if n == 1 else [
            _write_loop_overrides(iter_input_dir, rerun=rerun, reuse_output_dir=reuse_output_dir)
        ]
        run_forward_pass(target=iter_p["s5_signal"], input_dir=iter_in, output_dir=iter_out,
                         cores=args.cores, config_path=config_path, repo_root=repo_root,
                         extra_configfiles=extra_configfiles, extra_config=config_overrides or None,
                         profile=args.profile, nolock=args.nolock,
                         htcondor_jobdir=args.htcondor_jobdir)

        prev_p = iter_p  # post-processing wrote both feedback artifacts; they seed iteration n+1
        signal = json.loads(_abs(repo_root, iter_p["s5_signal"]).read_text())
        validate_signal(signal)
        status = signal["status"]
        if status == "halted":
            logger.warning("Iteration %d: Stage 5 signalled a halt (pressure not sustained); "
                           "stopping. Restart from different initial conditions.", n)
            break
        if status == "converged":
            logger.info("Converged at iteration %d; stopping.", n)
            break
        if status == "horizon":
            logger.info("Iteration %d reached [transport_solver].t_final, so no transport time is left to "
                        "advance into; stopping. Raise t_final to keep evolving.", n)
            break
    logger.info("Loop finished after %d iteration(s); records under %s/loop/", n, base_out)


if __name__ == "__main__":
    main()
