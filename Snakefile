# driftless-star MVP Snakemake workflow

import json
import posixpath
import shlex
from pathlib import Path

from src import stage3_helper, stage4_helper, stage5_helper
from src.utils import (
    resolve_docker_user,
    resolve_gpu_settings,
    resolve_pipeline_paths,
    resolve_rerun_flags,
    RESOLVED_COMMON_CONFIG,
)

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

GPU = resolve_gpu_settings(config)
DEVICE = GPU.device

# Per-stage rerun flags for the closed loop, validated at parse time so a bad combination fails before any job runs.
# Freezing a stage means reading its artifacts from an earlier pass, which takes an address from loop.reuse_output_dir.
# A plain forward pass and the loop's first iteration therefore always run every stage.
RERUN = resolve_rerun_flags(config)
REUSE_OUTPUT_DIR = (config.get("loop") or {}).get("reuse_output_dir")
if REUSE_OUTPUT_DIR is None:
    RERUN = dict.fromkeys(RERUN, True)
elif not isinstance(REUSE_OUTPUT_DIR, str):
    raise ValueError(f"config['loop']['reuse_output_dir'] must be a path string, got {REUSE_OUTPUT_DIR!r}.")
elif not RERUN["stage5"]:
    raise ValueError(
        "config['loop'] freezes every stage while naming a reuse tree, leaving no stage to produce the requested "
        "target. The loop driver runs a single iteration for an all-frozen config instead of naming a reuse tree."
    )

CONTAINER_RUNTIME = config.get("container_runtime", "docker")
if CONTAINER_RUNTIME not in ("docker", "apptainer"):
    raise ValueError(
        "config['container_runtime'] must be 'docker' or 'apptainer', "
        f"got {CONTAINER_RUNTIME!r}."
    )

STAGE1_IMG     = f"ghcr.io/driftless-star/driftless-star:stage-1-vmec-{DEVICE}"
STAGE2_IMG     = f"ghcr.io/driftless-star/driftless-star:stage-2-booz-jax-{DEVICE}"
STAGE3_JAX_IMG = f"ghcr.io/driftless-star/driftless-star:stage-3-sfincs-{DEVICE}"
STAGE4_IMG     = f"ghcr.io/driftless-star/driftless-star:stage-4-spectrax-{DEVICE}"
STAGE5_IMG     = f"ghcr.io/driftless-star/driftless-star:stage-5-neopax-{DEVICE}"


def _apptainer_bind_flags(cfg: dict) -> str:
    """Bind the workdir and absolute roots needed by nested Apptainer jobs."""
    bind_specs = ['"$PWD:/work"']
    paths = [cfg.get("input_dir"), cfg.get("output_dir")]
    paths.append((cfg.get("loop") or {}).get("reuse_output_dir"))
    for path in paths:
        if not path:
            continue
        p = Path(path)
        if p.is_absolute():
            bind_specs.append(f'"{p}:{p}"')
    return " ".join(f"--bind {spec}" for spec in dict.fromkeys(bind_specs))


def _docker_bind_flags(cfg: dict) -> str:
    """Bind the workdir and absolute run roots needed by Docker jobs."""
    bind_flags = ['-v "$PWD:/work"']
    paths = [cfg.get("input_dir"), cfg.get("output_dir")]
    paths.append((cfg.get("loop") or {}).get("reuse_output_dir"))
    for path in paths:
        if not path:
            continue
        p = Path(path)
        if p.is_absolute():
            bind_flags.append(f"-v {shlex.quote(f'{p}:{p}')}")
    return " ".join(dict.fromkeys(bind_flags))


def _apptainer_image_ref(image: str) -> str:
    """Map OCI-style workflow names to the native SIF images published to GHCR."""
    if "://" in image or image.endswith(".sif"):
        return image
    repo, tag = image.rsplit(":", 1)
    return f"oras://{repo}:apptainer-{tag}"


# Docker keeps the newer per-device slot allocator. On HTCondor, each Apptainer
# job already receives one isolated GPU, which `--nv` exposes to the nested SIF.
if CONTAINER_RUNTIME == "docker":
    gpu_flag = ""
    slot_prefix = ""
    if DEVICE == "gpu":
        gpu_flag = "--gpus device=@GPU_ID@ "
        slot_prefix = (
            f"python -m src.gpu_slots --gpu-ids {','.join(GPU.pool) if GPU.pool else 'all'} "
            f"--jobs-per-gpu {GPU.jobs_per_gpu} --lock-dir .snakemake/gpu_slots -- "
        )
    # --user makes bind-mounted writes host-owned; HOME must be writable for pixi activation.
    docker_tail = (
        f'{resolve_docker_user(config)}'
        '-e HOME=/tmp '
        f'{_docker_bind_flags(config)} -w /work '
    )
    CONTAINER_PREFIX = f"{slot_prefix}docker run --rm --pull=missing {gpu_flag}{docker_tail}"
    # File-rewrite helpers need neither a GPU flag nor a scheduling slot.
    CONTAINER_PREFIX_CPU = f"docker run --rm --pull=missing {docker_tail}"

    def container_image_location(image: str) -> str:
        return image
else:
    gpu_flag = "--nv " if DEVICE == "gpu" else ""
    apptainer_tail = f'{_apptainer_bind_flags(config)} --pwd /work '
    # --unsquash avoids FUSE-based SIF mounts on execute nodes whose parent
    # HTCondor container does not expose /dev/fuse.
    CONTAINER_PREFIX = f"apptainer run --unsquash {gpu_flag}{apptainer_tail}"
    CONTAINER_PREFIX_CPU = f"apptainer run --unsquash {apptainer_tail}"

    def container_image_location(image: str) -> str:
        return _apptainer_image_ref(image)


def container_image_ref(image: str) -> str:
    return f"{CONTAINER_PREFIX}{container_image_location(image)}"

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
S5_CONFIG_FEEDBACK = P["s5_config_feedback"]

# Only cross-stage artifacts redirect. Manifests, run dirs, and logs belong to rules defined only when a stage reruns.
if REUSE_OUTPUT_DIR is not None:
    P_REUSE = resolve_pipeline_paths(config, output_dir=REUSE_OUTPUT_DIR)
    if not RERUN["stage1"]:
        S1_OUTPUT = P_REUSE["s1_output"]
    if not RERUN["stage2"]:
        S2_OUTPUT = P_REUSE["s2_output"]
    if not RERUN["stage3"]:
        S3_OUTPUT = P_REUSE["s3_output"]
    if not RERUN["stage4"]:
        S4_OUTPUT = P_REUSE["s4_output"]

STAGE3_CFG = config["stage3"]["sfincs_jax"]
STAGE4_CFG = config["stage4"]["spectrax_gk"]
# Resolved here rather than beside its rule so a misspelled convention is caught even when the run freezes Stage 4.
RADIUS_RELABEL_CONVENTION = stage4_helper.resolve_radius_relabel(config)

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

# A frozen stage's rules are omitted from the workflow, so its reuse-tree artifacts cannot be rebuilt or overwritten.
if RERUN["stage1"]:
    rule stage1_vmec:
        input:  S1_INPUT
        output: S1_OUTPUT
        log:    f"{P['stage1_dir']}/{RUN_NAME}.log"
        shell:
            f"{container_image_ref(STAGE1_IMG)} "
            f"vmec_jax {{input}} --output {{output}}"
            " 2>&1 | tee {log}"

if RERUN["stage2"]:
    rule stage2_boozer:
        input:  S1_OUTPUT
        output: S2_OUTPUT
        log:    f"{P['stage2_dir']}/{RUN_NAME}.log"
        shell:
            f"{container_image_ref(STAGE2_IMG)} "
            "python stages/stage2-boozer/run_boozer.py --wout {input} --output {output}"
            " 2>&1 | tee {log}"


# Per-surface run-directory basenames e.g. rho_012_r0p4898 follow the pattern:
# zero-padded radial-grid index then the normalized flux-surface radius (e.g. rho=0.4898).
# Stage 4 fd_gradients mode adds perturbed siblings, e.g. rho_012_r0p4898_fd_n_D.
SURF_PATTERN = r"rho_\d+_r[0-9p]+(?:_fd_[nt]_\w+)?"

if RERUN["stage3"]:
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
                docker_prefix=CONTAINER_PREFIX,
                image=container_image_location(STAGE3_JAX_IMG),
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
                docker_prefix=CONTAINER_PREFIX,
                image=container_image_location(STAGE3_JAX_IMG),
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
                docker_prefix=CONTAINER_PREFIX,
                image=container_image_location(STAGE3_JAX_IMG),
                stage_cfg=STAGE3_CFG,
                output_dir=P["stage3_dir"],
            ) + " 2>&1 | tee {log}"

if RERUN["stage4"]:
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
                docker_prefix=CONTAINER_PREFIX,
                image=container_image_location(STAGE4_IMG),
                stage_cfg=STAGE4_CFG,
                output_dir=P["stage4_dir"],
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
                docker_prefix=CONTAINER_PREFIX,
                image=container_image_location(STAGE4_IMG),
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

    # Stage 4 writes the flux file on VMEC's Aminor_p while NEOPAX interpolates it onto a grid built
    # from its own minor radius, so the collected grid is rewritten onto NEOPAX's convention here.
    NEOPAX_RADIUS_RELABEL_CMD = " && " + stage4_helper.relabel_cmd(
        docker_prefix=CONTAINER_PREFIX_CPU,
        image=container_image_location(STAGE4_IMG),
        flux_file=S4_OUTPUT,
        wout=S1_OUTPUT,
        boozer=S2_OUTPUT,
        convention=RADIUS_RELABEL_CONVENTION,
        rho_edge=stage5_helper.read_rho_edge(S5_CONFIG),
    ) if RADIUS_RELABEL_CONVENTION else ""

    rule stage4_collect:
        input:
            manifest = S4_MANIFEST,
            diagnostics = stage4_surface_diagnostics,
        output:
            S4_OUTPUT,
        log:
            f"{P['stage4_dir']}/{RUN_NAME}.collect.log"
        shell:
            # Grouped so the pipe captures both commands, since `a && b | tee` would bind the pipe
            # to b alone and drop the collect output from the log.
            "( " + stage4_helper.collect_cmd(
                docker_prefix=CONTAINER_PREFIX,
                image=container_image_location(STAGE4_IMG),
                stage_cfg=STAGE4_CFG,
                output_dir=P["stage4_dir"],
            ) + NEOPAX_RADIUS_RELABEL_CMD + " ) 2>&1 | tee {log}"

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
        f"{container_image_ref(STAGE5_IMG)} "
        f"sh -c \"cd {P['stage5_dir']} && neopax {RESOLVED_COMMON_CONFIG}\""
        " 2>&1 | tee {log}"

# Stage 5 post-processing closes the optimization loop and writes a convergence signal.
# Both the evolved Stage 1 input (S1_FEEDBACK) and the common input (S5_CONFIG_FEEDBACK)
# land under outputs/. `rule all` stays S5_OUTPUT, so a plain `snakemake` is a pure forward pass.
rule stage5_post_processing:
    input:
        transport     = S5_OUTPUT,
        s1_input      = S1_INPUT,
        common_config = S5_CONFIG,
    output:
        signal            = S5_SIGNAL,
        feedback          = S1_FEEDBACK,
        profiles_feedback = S5_CONFIG_FEEDBACK,
    log:    f"{P['stage5_post_dir']}/{RUN_NAME}.log"
    shell:
        f'{container_image_ref(STAGE5_IMG)} sh -c "'
        'python stages/stage5-post-processing/fit_vmec_pressure_from_transport_h5.py '
        'write-input {input.transport} {input.s1_input} --output-input {output.feedback} && '
        'python stages/stage5-post-processing/write_prescribed_profiles_from_transport_h5.py '
        '{input.transport} {input.common_config} --output-toml {output.profiles_feedback} && '
        'python stages/stage5-post-processing/stage5_post_processing.py '
        f'--transport {{input.transport}} --signal {{output.signal}} --pressure-rel-tol {PRESSURE_REL_TOL}"'
        " 2>&1 | tee {log}"
