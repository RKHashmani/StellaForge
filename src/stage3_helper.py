"""Stage 3 (neoclassical) shell-command composition for the Snakemake workflow.

The Stage 3 ``sfincs_jax`` radial-scan script accepts many optional flags; this
module turns the user-facing ``config.yaml`` ``stage3.sfincs_jax`` block into the
per-phase shell commands run by the Snakefile's ``stage3_prepare`` checkpoint and
its ``stage3_run_one``/``stage3_collect`` rules.
"""

from __future__ import annotations

_SCRIPT = "stages/stage3-neoclassical/sfincs_jax_radial_scan.py"

# (config_key, cli_flag) accepted by the `prepare` subcommand; emitted as `<flag> <value>` when set.
_PREPARE_OPTIONAL_FLAGS: list[tuple[str, str]] = [
    ("profiles_source",    "--profiles-source"),
    ("neopax_result",      "--neopax-result"),
    ("ntheta",             "--ntheta"),
    ("nzeta",              "--nzeta"),
    ("nxi",                "--nxi"),
    ("nx",                 "--nx"),
    ("solver_tolerance",   "--solver-tolerance"),
    ("analytical_n_radii", "--analytical-n-radii"),
    ("rho_indices",        "--rho-indices"),
    ("rho_min",            "--rho-min"),
    ("rho_max",            "--rho-max"),
    ("num_radii",          "--num-radii"),
]

# verbose_workers is baked into each per-surface payload at prepare time.
_PREPARE_BOOL_FLAGS: list[tuple[str, str, str]] = [
    ("verbose_workers", "--verbose-workers", "--no-verbose-workers"),
]

# Plotting happens in the collect step.
_COLLECT_BOOL_FLAGS: list[tuple[str, str, str]] = [
    ("plot", "--plot", "--no-plot"),
]


def _append_optional_flags(parts: list[str], stage_cfg: dict, table: list[tuple[str, str]]) -> None:
    """Append ``<flag> <value>`` to ``parts`` for each table entry whose config value is not None."""
    for key, flag in table:
        value = stage_cfg.get(key)
        if value is not None:
            parts.append(f"{flag} {value}")


def _append_bool_flags(parts: list[str], stage_cfg: dict, table: list[tuple[str, str, str]]) -> None:
    """Append the on-flag for True and the off-flag for False; a missing/None key appends nothing."""
    for key, on, off in table:
        value = stage_cfg.get(key)
        if value is True:
            parts.append(on)
        elif value is False:
            parts.append(off)


def prepare_cmd(
    *,
    docker_prefix: str,
    image: str,
    stage_cfg: dict,
    output_dir: str,
    device: str,
) -> str:
    """Compose the Stage 3 ``sfincs_jax`` ``prepare`` shell command.

    Concretely: build the CLI arguments for the scan script's ``prepare``
    subcommand from ``stage_cfg`` and wrap them in a ``docker run`` invocation
    of the stage image. Nothing executes here; the returned string is the
    command Snakemake runs when the checkpoint's job fires.

    Parameters
    ----------
    docker_prefix : str
        ``docker run ...`` prefix prepared by the Snakefile.
    image : str
        Container image for Stage 3 (e.g. ``ghcr.io/.../stage-3-sfincs-cpu``).
    stage_cfg : dict
        The ``config.yaml`` ``stage3.sfincs_jax`` block.
    output_dir : str
        Stage 3 output directory (already ``{run_name}``-substituted).
    device : str
        ``"cpu"`` or ``"gpu"``; recorded into the manifest as the run backend.

    Returns
    -------
    str
        A single-line ``prepare`` subcommand that writes the per-surface
        payloads and scan manifest. ``{input.*}`` placeholders remain literal
        so Snakemake substitutes them at rule-execution time.
    """
    parts = [
        f"{docker_prefix} {image}",
        f"python {_SCRIPT} prepare",
        "--common-config {input.common_config}",
        "--sfincs-template {input.config_file}",
        "--wout-path {input.wout}",
        f"--output-dir {output_dir}",
        f"--backend {device}",
    ]
    _append_optional_flags(parts, stage_cfg, _PREPARE_OPTIONAL_FLAGS)
    _append_bool_flags(parts, stage_cfg, _PREPARE_BOOL_FLAGS)
    return " ".join(parts)


def run_one_cmd(
    *,
    docker_prefix: str,
    image: str,
    output_dir: str,
    device: str,
) -> str:
    """Compose the Stage 3 ``sfincs_jax`` ``run-one`` shell command.

    Parameters
    ----------
    docker_prefix : str
        ``docker run ...`` prefix prepared by the Snakefile.
    image : str
        Container image for Stage 3 (e.g. ``ghcr.io/.../stage-3-sfincs-cpu``).
    output_dir : str
        Stage 3 output directory (already ``{run_name}``-substituted).
    device : str
        ``"cpu"`` or ``"gpu"``; selects the worker's JAX backend. Device
        assignment lives in ``docker_prefix``.

    Returns
    -------
    str
        A single-line ``run-one`` subcommand that solves one surface. The
        ``{wildcards.surf}`` placeholder names the per-surface run directory and
        stays literal so Snakemake substitutes it at rule-execution time.
    """
    parts = [
        f"{docker_prefix} {image}",
        f"python {_SCRIPT} run-one",
        f"--payload {output_dir}/runs/{{wildcards.surf}}/payload.json",
        f"--backend {device}",
    ]
    return " ".join(parts)


def collect_cmd(
    *,
    docker_prefix: str,
    image: str,
    stage_cfg: dict,
    output_dir: str,
) -> str:
    """Compose the Stage 3 ``sfincs_jax`` ``collect`` shell command.

    Parameters
    ----------
    docker_prefix : str
        ``docker run ...`` prefix prepared by the Snakefile.
    image : str
        Container image for Stage 3 (e.g. ``ghcr.io/.../stage-3-sfincs-cpu``).
    stage_cfg : dict
        The ``config.yaml`` ``stage3.sfincs_jax`` block.
    output_dir : str
        Stage 3 output directory (already ``{run_name}``-substituted).

    Returns
    -------
    str
        A single-line ``collect`` subcommand that reduces the per-surface
        results into the flux-profile HDF5.
    """
    parts = [
        f"{docker_prefix} {image}",
        f"python {_SCRIPT} collect",
        f"--output-dir {output_dir}",
    ]
    _append_bool_flags(parts, stage_cfg, _COLLECT_BOOL_FLAGS)
    return " ".join(parts)
