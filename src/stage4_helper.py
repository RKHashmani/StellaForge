"""Stage 4 (turbulence) shell-command composition for the Snakemake workflow.

The Stage 4 SPECTRAX-GK radial-scan script accepts many optional flags; this
module turns the user-facing ``config.yaml`` ``stage4.spectrax_gk`` block into
the per-phase shell commands run by the Snakefile's ``stage4_prepare``
checkpoint and its ``stage4_run_one``/``stage4_collect`` rules.
"""

from __future__ import annotations

_SCRIPT = "stages/stage4-turbulence/spectrax_gk_radial_scan.py"

# (config_key, cli_flag) accepted by the `prepare` subcommand; emitted as `<flag> <value>` when set.
# The config key t_max is spelled --t-final, an accepted alias whose argparse dest is t_max.
_PREPARE_OPTIONAL_FLAGS: list[tuple[str, str]] = [
    ("profiles_source",    "--profiles-source"),
    ("neopax_result",      "--neopax-result"),
    ("nx",                 "--nx"),
    ("ny",                 "--ny"),
    ("ntheta",             "--ntheta"),
    ("t_max",              "--t-final"),
    ("sample_stride",      "--sample-stride"),
    ("diagnostics_stride", "--diagnostics-stride"),
    ("analytical_n_radii", "--analytical-n-radii"),
    ("rho_indices",        "--rho-indices"),
    ("rho_min",            "--rho-min"),
    ("rho_max",            "--rho-max"),
    ("num_radii",          "--num-radii"),
]

# Flux averaging and plotting happen in the collect step. Config t_max is deliberately not re-emitted
# at collect: it reaches the manifest via prepare, and collect's --t-final falls back to the manifest
# value, keeping a single source of truth for the time window.
_COLLECT_OPTIONAL_FLAGS: list[tuple[str, str]] = [
    ("average_window", "--average-window"),
]

_COLLECT_BOOL_FLAGS: list[tuple[str, str, str]] = [
    ("plot",                 "--plot",                 "--no-plot"),
    ("plot_run_heat_traces", "--plot-run-heat-traces", "--no-plot-run-heat-traces"),
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
    """Compose the Stage 4 SPECTRAX-GK ``prepare`` shell command.

    Concretely: build the CLI arguments for the scan script's ``prepare``
    subcommand from ``stage_cfg`` and wrap them in a ``docker run`` invocation
    of the stage image. Nothing executes here; the returned string is the
    command Snakemake runs when the checkpoint's job fires.

    Parameters
    ----------
    docker_prefix : str
        ``docker run ...`` prefix prepared by the Snakefile.
    image : str
        Container image for Stage 4 (e.g. ``ghcr.io/.../stage-4-spectrax-cpu``).
    stage_cfg : dict
        The ``config.yaml`` ``stage4.spectrax_gk`` block.
    output_dir : str
        Stage 4 output directory (already ``{run_name}``-substituted).
    device : str
        ``"cpu"`` or ``"gpu"``; accepted for a uniform composer signature. The
        prepare step only writes the manifest and runtime TOMLs, and its parser
        takes no backend flag, so no device flag is emitted.

    Returns
    -------
    str
        A single-line ``prepare`` subcommand that writes the per-radius manifest
        and SPECTRAX runtime TOMLs. ``{input.*}`` placeholders remain literal so
        Snakemake substitutes them at rule-execution time.
    """
    parts = [
        f"{docker_prefix} {image}",
        f"python {_SCRIPT} prepare",
        "--common-config {input.common_config}",
        "--spectrax-template {input.config_file}",
        "--vmec-file-override {input.wout}",
        "--boozer-file-override {input.boozer}",
        f"--output-dir {output_dir}",
    ]
    _append_optional_flags(parts, stage_cfg, _PREPARE_OPTIONAL_FLAGS)
    return " ".join(parts)


def run_one_cmd(
    *,
    docker_prefix: str,
    image: str,
    stage_cfg: dict,
    output_dir: str,
    device: str,
) -> str:
    """Compose the Stage 4 SPECTRAX-GK ``run-one`` shell command.

    Parameters
    ----------
    docker_prefix : str
        ``docker run ...`` prefix prepared by the Snakefile.
    image : str
        Container image for Stage 4 (e.g. ``ghcr.io/.../stage-4-spectrax-cpu``).
    stage_cfg : dict
        The ``config.yaml`` ``stage4.spectrax_gk`` block.
    output_dir : str
        Stage 4 output directory (already ``{run_name}``-substituted).
    device : str
        ``"cpu"`` or ``"gpu"``; controls the JAX backend and GPU pinning.

    Returns
    -------
    str
        A single-line ``run-one`` subcommand that evolves one surface. The
        ``{wildcards.surf}`` placeholder names the manifest run to execute and
        stays literal so Snakemake substitutes it at rule-execution time.
    """
    parts = [
        f"{docker_prefix} {image}",
        f"python {_SCRIPT} run-one",
        f"--manifest {output_dir}/manifest.json",
        "--run-name {wildcards.surf}",
        f"--backend {device}",
    ]
    # The run-one worker pins a single device via --gpu-id. The raw config string is passed through
    # unchanged; splitting a multi-GPU list round-robin across surfaces is a deferred follow-up.
    if device == "gpu" and stage_cfg.get("gpu_ids") is not None:
        parts.append(f"--gpu-id {stage_cfg['gpu_ids']}")
    # --verbose-worker is a plain store_true with no negative form, so only an explicit config True
    # emits it; False and a missing key both leave the worker at its quiet default.
    if stage_cfg.get("verbose_workers") is True:
        parts.append("--verbose-worker")
    return " ".join(parts)


def collect_cmd(
    *,
    docker_prefix: str,
    image: str,
    stage_cfg: dict,
    output_dir: str,
    device: str,
) -> str:
    """Compose the Stage 4 SPECTRAX-GK ``collect`` shell command.

    Parameters
    ----------
    docker_prefix : str
        ``docker run ...`` prefix prepared by the Snakefile.
    image : str
        Container image for Stage 4 (e.g. ``ghcr.io/.../stage-4-spectrax-cpu``).
    stage_cfg : dict
        The ``config.yaml`` ``stage4.spectrax_gk`` block.
    output_dir : str
        Stage 4 output directory (already ``{run_name}``-substituted).
    device : str
        ``"cpu"`` or ``"gpu"``; accepted for a uniform composer signature and
        not used by the reduction step.

    Returns
    -------
    str
        A single-line ``collect`` subcommand that reduces per-radius
        diagnostics into ``flux_summary.h5`` and ``neopax_fluxes.h5``.
    """
    parts = [
        f"{docker_prefix} {image}",
        f"python {_SCRIPT} collect",
        f"--manifest {output_dir}/manifest.json",
        f"--out {output_dir}/flux_summary.h5",
        f"--neopax-flux-out {output_dir}/neopax_fluxes.h5",
    ]
    _append_optional_flags(parts, stage_cfg, _COLLECT_OPTIONAL_FLAGS)
    _append_bool_flags(parts, stage_cfg, _COLLECT_BOOL_FLAGS)
    return " ".join(parts)
