# driftless-star MVP Snakemake workflow

import json
import posixpath

from src import stage3_helper, stage4_helper, stage5_helper
from src.utils import resolve_pipeline_paths, RESOLVED_COMMON_CONFIG

# Require an explicit run config
if not config:
    raise ValueError(
        "No config loaded. Pass --configfile inputs/<run>/config.yaml "
        "(e.g. snakemake --configfile inputs/quick_run/config.yaml --cores 4)."
    )
_missing = [k for k in ("run_name", "input_dir", "output_dir", "filenames") if k not in config]
if _missing:
    raise ValueError(f"config is missing required key(s): {_missing}.")

RUN_NAME = config["run_name"]

DEVICE = config.get("device", "cpu")
if DEVICE not in ("cpu", "gpu"):
    raise ValueError(
        f"config['device'] must be 'cpu' or 'gpu', got {DEVICE!r}."
    )

GPU_FLAG       = "--gpus all " if DEVICE == "gpu" else ""
STAGE1_IMG     = f"ghcr.io/driftless-star/driftless-star:stage-1-vmec-{DEVICE}"
STAGE2_IMG     = f"ghcr.io/driftless-star/driftless-star:stage-2-booz-jax-{DEVICE}"
STAGE3_JAX_IMG = f"ghcr.io/driftless-star/driftless-star:stage-3-sfincs-{DEVICE}"
STAGE4_IMG     = f"ghcr.io/driftless-star/driftless-star:stage-4-spectrax-{DEVICE}"
STAGE5_IMG     = f"ghcr.io/driftless-star/driftless-star:stage-5-neopax-{DEVICE}"

# --user: make bind-mounted writes host-owned (Linux docker otherwise writes as root).
# -e HOME=/tmp: pixi activation needs a writable HOME after dropping root.
DOCKER_PREFIX = (
    f'docker run --rm --pull=missing {GPU_FLAG}'
    '--user "$(id -u):$(id -g)" '
    '-e HOME=/tmp '
    '-v "$PWD:/work" -w /work'
)

shell.executable("bash")
# Propagate failures through `cmd | tee {log}` pipelines so a crashed stage
# does not look successful just because tee exited 0.
shell.prefix("set -o pipefail; ")

P = resolve_pipeline_paths(config)
S1_INPUT  = P["s1_input"]
S1_OUTPUT = P["s1_output"]
S2_OUTPUT = P["s2_output"]
S3_CONFIG = P["s3_config"]
S3_MANIFEST = P["stage3_manifest"]
S3_OUTPUT = P["s3_output"]
S4_CONFIG = P["s4_config"]
S4_MANIFEST = P["stage4_manifest"]
S4_OUTPUT = P["s4_output"]
S5_CONFIG = P["s5_config"]
S5_OUTPUT = P["s5_output"]
S5_SIGNAL = P["s5_signal"]
S1_FEEDBACK = P["s1_feedback"]

STAGE3_CFG = config["stage3"]["sfincs_jax"]
STAGE4_CFG = config["stage4"]["spectrax_gk"]

# Stage 5 post-processing convergence threshold (see the `convergence` block in inputs/<run>/config.yaml).
PRESSURE_REL_TOL = config.get("convergence", {}).get("pressure_rel_tol", 1.0e-2)

# Write a path-resolved copy of the NEOPAX template under outputs/ and run that (template untouched).
stage5_helper.prepare_neopax_config(
    s5_config_template=S5_CONFIG,
    s5_resolved_config=P["s5_resolved_config"],
    s1_output=S1_OUTPUT,
    s2_output=S2_OUTPUT,
    s3_output=S3_OUTPUT,
    s4_output=S4_OUTPUT,
    s5_output_dir=P["stage5_dir"],
)


# Terminal artifact of the MVP forward pass.
rule all:
    input:
        S5_OUTPUT,

rule stage1_vmec:
    input:  S1_INPUT
    output: S1_OUTPUT
    log:    f"{P['stage1_dir']}/{RUN_NAME}.log"
    shell:
        f"{DOCKER_PREFIX} {STAGE1_IMG} "
        f"vmec_jax {{input}} --output {{output}}"
        " 2>&1 | tee {log}"

rule stage2_boozer:
    input:  S1_OUTPUT
    output: S2_OUTPUT
    log:    f"{P['stage2_dir']}/{RUN_NAME}.log"
    shell:
        f"{DOCKER_PREFIX} {STAGE2_IMG} "
        "python stages/stage2-boozer/run_boozer.py --wout {input} --output {output}"
        " 2>&1 | tee {log}"


# Per-surface run-directory basenames e.g. rho_012_r0p4898 follow the pattern:
# zero-padded radial-grid index then the normalized flux-surface radius (e.g. rho=0.4898).
# Stage 4 fd_gradients mode adds perturbed siblings, e.g. rho_012_r0p4898_fd_n_D.
SURF_PATTERN = r"rho_\d+_r[0-9p]+(?:_fd_[nt]_\w+)?"

checkpoint stage3_prepare:
    input:
        config_file = S3_CONFIG,
        wout        = S1_OUTPUT,
        common_config = S5_CONFIG,
    output:
        S3_MANIFEST,
    log:
        f"{P['stage3_dir']}/{RUN_NAME}.prepare.log"
    shell:
        stage3_helper.prepare_cmd(
            docker_prefix=DOCKER_PREFIX,
            image=STAGE3_JAX_IMG,
            stage_cfg=STAGE3_CFG,
            output_dir=P["stage3_dir"],
            device=DEVICE,
        ) + " 2>&1 | tee {log}"

rule stage3_run_one:
    input:
        manifest = S3_MANIFEST,
    output:
        f"{P['stage3_dir']}/runs/{{surf}}/result.json",
    wildcard_constraints:
        surf = SURF_PATTERN,
    log:
        f"{P['stage3_dir']}/runs/{{surf}}/run.log"
    shell:
        stage3_helper.run_one_cmd(
            docker_prefix=DOCKER_PREFIX,
            image=STAGE3_JAX_IMG,
            stage_cfg=STAGE3_CFG,
            output_dir=P["stage3_dir"],
            device=DEVICE,
        ) + " 2>&1 | tee {log}"

def stage3_surface_results(wildcards):
    """List every per-surface result.json named by the Stage 3 manifest."""
    manifest_path = checkpoints.stage3_prepare.get().output[0]
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    return [
        f"{P['stage3_dir']}/runs/{run['run_subdir']}/result.json"
        for run in manifest["runs"]
    ]

rule stage3_collect:
    input:
        manifest = S3_MANIFEST,
        results  = stage3_surface_results,
    output:
        S3_OUTPUT,
    log:
        f"{P['stage3_dir']}/{RUN_NAME}.collect.log"
    shell:
        stage3_helper.collect_cmd(
            docker_prefix=DOCKER_PREFIX,
            image=STAGE3_JAX_IMG,
            stage_cfg=STAGE3_CFG,
            output_dir=P["stage3_dir"],
            device=DEVICE,
        ) + " 2>&1 | tee {log}"

checkpoint stage4_prepare:
    input:
        config_file = S4_CONFIG,
        wout        = S1_OUTPUT,
        boozer      = S2_OUTPUT,
        common_config = S5_CONFIG,
    output:
        S4_MANIFEST,
    log:
        f"{P['stage4_dir']}/{RUN_NAME}.prepare.log"
    shell:
        stage4_helper.prepare_cmd(
            docker_prefix=DOCKER_PREFIX,
            image=STAGE4_IMG,
            stage_cfg=STAGE4_CFG,
            output_dir=P["stage4_dir"],
            device=DEVICE,
        ) + " 2>&1 | tee {log}"

rule stage4_run_one:
    input:
        manifest = S4_MANIFEST,
    output:
        f"{P['stage4_dir']}/runs/{{surf}}/run.diagnostics.csv",
    wildcard_constraints:
        surf = SURF_PATTERN,
    log:
        f"{P['stage4_dir']}/runs/{{surf}}/run.log"
    shell:
        stage4_helper.run_one_cmd(
            docker_prefix=DOCKER_PREFIX,
            image=STAGE4_IMG,
            stage_cfg=STAGE4_CFG,
            output_dir=P["stage4_dir"],
            device=DEVICE,
        ) + " 2>&1 | tee {log}"

def stage4_surface_diagnostics(wildcards):
    """List every per-surface diagnostics CSV named by the Stage 4 manifest.

    Stage 4 manifest entries carry no run_subdir key, only the container-absolute
    run_dir, so the host-side path is rebuilt from its POSIX basename.
    """
    manifest_path = checkpoints.stage4_prepare.get().output[0]
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    return [
        f"{P['stage4_dir']}/runs/{posixpath.basename(run['run_dir'])}/run.diagnostics.csv"
        for run in manifest["runs"]
    ]

rule stage4_collect:
    input:
        manifest = S4_MANIFEST,
        diagnostics = stage4_surface_diagnostics,
    output:
        S4_OUTPUT,
    log:
        f"{P['stage4_dir']}/{RUN_NAME}.collect.log"
    shell:
        stage4_helper.collect_cmd(
            docker_prefix=DOCKER_PREFIX,
            image=STAGE4_IMG,
            stage_cfg=STAGE4_CFG,
            output_dir=P["stage4_dir"],
            device=DEVICE,
        ) + " 2>&1 | tee {log}"

rule stage5_neopax:
    input:
        config_file = S5_CONFIG,
        wout    = S1_OUTPUT,
        boozer  = S2_OUTPUT,
        neo_h5  = S3_OUTPUT,
        turb_h5 = S4_OUTPUT,
    output:
        S5_OUTPUT,
    log:
        f"{P['stage5_dir']}/{RUN_NAME}.log"
    shell:
        f"{DOCKER_PREFIX} {STAGE5_IMG} "
        f"sh -c \"cd {P['stage5_dir']} && neopax {RESOLVED_COMMON_CONFIG}\""
        " 2>&1 | tee {log}"

# Stage 5 post-processing closes the optimization loop and writes a convergence signal.
# The evolved Stage 1 boundary is written under outputs/ (S1_FEEDBACK), never onto the
# committed input. `rule all` stays S5_OUTPUT, so a plain `snakemake` is a pure forward pass.
rule stage5_post_processing:
    input:  S5_OUTPUT
    output:
        signal   = S5_SIGNAL,
        feedback = S1_FEEDBACK,
    log:    f"{P['stage5_post_dir']}/{RUN_NAME}.log"
    shell:
        f'{DOCKER_PREFIX} {STAGE5_IMG} sh -c "'
        'python stages/stage5-post-processing/fit_vmec_pressure_from_transport_h5.py '
        f'write-input {{input}} {S1_INPUT} --output-input {{output.feedback}} && '
        'python stages/stage5-post-processing/stage5_post_processing.py '
        f'--transport {{input}} --signal {{output.signal}} --pressure-rel-tol {PRESSURE_REL_TOL}"'
        " 2>&1 | tee {log}"
