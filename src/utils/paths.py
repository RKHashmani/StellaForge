"""Pipeline path derivation

``resolve_pipeline_paths`` is the single source of truth mapping a run's config
to every file and directory the pipeline uses. The Snakefile and the closed-loop
driver (``src/ouroboros.py``) both import it so they never derive paths differently.
"""

from __future__ import annotations

# Output-file key -> its per-stage output subdirectory name.
_STAGE_SUBDIRS: dict[str, str] = {
    "s1_output": "stage1_equilibrium",
    "s2_output": "stage2_boozer",
    "s3_output": "stage3_neoclassical",
    "s4_output": "stage4_turbulence",
    "s5_output": "stage5_transport",
    "s5_signal": "stage5_post_processing",
}

# Basename of the resolved NEOPAX config written under the Stage 5 output dir.
RESOLVED_COMMON_CONFIG = "common_input_updated.toml"


def resolve_pipeline_paths(
    config: dict,
    input_dir: str | None = None,
    output_dir: str | None = None,
) -> dict[str, str]:
    """Turn one run's config into every concrete path the pipeline reads or writes.

    Parameters
    ----------
    config : dict
        Parsed run config; must contain ``run_name``, ``input_dir``,
        ``output_dir``, and ``filenames``.
    input_dir, output_dir : str, optional
        Override the config's directories (``None`` keeps the config value). The
        loop driver passes these to run an iteration inside its own
        ``outputs/<run>/loop/iter_N/{input,output}`` sandbox.

    Returns
    -------
    dict[str, str]
        Repo-relative paths, in five groups:

        - the resolved ``input_dir`` and ``output_dir``;
        - input files ``s1_input``, ``s3_config``, ``s4_config``, ``s5_config``
          (``s5_config`` is the shared ``common_input`` template);
        - output artifacts ``s1_output``..``s5_output``, ``s5_signal``;
        - per-stage output dirs ``stage1_dir``..``stage5_post_dir``;
        - paths generated under ``outputs/``: ``s5_resolved_config`` (the
          path-resolved NEOPAX copy NEOPAX actually runs) and ``s1_feedback``
          (the evolved Stage 1 boundary the loop feeds to the next iteration).

    Examples
    --------
    >>> config = {
    ...     "run_name": "demo",
    ...     "input_dir": "inputs/demo",
    ...     "output_dir": "outputs/demo",
    ...     "filenames": {
    ...         "s1_input": "vmec_input.{run_name}",
    ...         "s3_config": "sfincs_input.{run_name}",
    ...         "s4_config": "{run_name}.toml",
    ...         "s5_config": "common_input.toml",
    ...         "s1_output": "wout_{run_name}.nc",
    ...         "s2_output": "boozmn_{run_name}.nc",
    ...         "s3_output": "sfincs_jax_flux_profiles.h5",
    ...         "s4_output": "neopax_fluxes.h5",
    ...         "s5_output": "transport_solution.h5",
    ...         "s5_signal": "converge_status.json",
    ...     },
    ... }
    >>> resolve_pipeline_paths(config)["s1_output"]
    'outputs/demo/stage1_equilibrium/wout_demo.nc'
    >>> resolve_pipeline_paths(config, output_dir="outputs/demo/loop/iter_1/output")["s5_signal"]
    'outputs/demo/loop/iter_1/output/stage5_post_processing/converge_status.json'
    """
    run_name = config["run_name"]
    input_dir = input_dir if input_dir is not None else config["input_dir"]
    output_dir = output_dir if output_dir is not None else config["output_dir"]
    files = config["filenames"]

    def fn(key: str) -> str:
        return files[key].format(run_name=run_name)

    def stage_dir(key: str) -> str:
        return f"{output_dir}/{_STAGE_SUBDIRS[key]}"

    def out(key: str) -> str:
        return f"{stage_dir(key)}/{fn(key)}"

    return {
        "input_dir": input_dir,
        "output_dir": output_dir,
        # Input files.
        "s1_input": f"{input_dir}/{fn('s1_input')}",
        "s3_config": f"{input_dir}/{fn('s3_config')}",
        "s4_config": f"{input_dir}/{fn('s4_config')}",
        "s5_config": f"{input_dir}/{fn('s5_config')}",
        # Output artifacts.
        "s1_output": out("s1_output"),
        "s2_output": out("s2_output"),
        "s3_output": out("s3_output"),
        "s4_output": out("s4_output"),
        "s5_output": out("s5_output"),
        "s5_signal": out("s5_signal"),
        # Per-stage output dirs.
        "stage1_dir": stage_dir("s1_output"),
        "stage2_dir": stage_dir("s2_output"),
        "stage3_dir": stage_dir("s3_output"),
        "stage4_dir": stage_dir("s4_output"),
        "stage5_dir": stage_dir("s5_output"),
        "stage5_post_dir": stage_dir("s5_signal"),
        # Generated under outputs/.
        "s5_resolved_config": f"{stage_dir('s5_output')}/{RESOLVED_COMMON_CONFIG}",
        "s1_feedback": f"{stage_dir('s5_signal')}/{fn('s1_input')}",
    }
