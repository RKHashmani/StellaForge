# driftless-star MVP Snakemake workflow

from pathlib import Path

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
    """Bind repo-relative workdir plus any absolute input/output roots."""
    bind_specs = ['"$PWD:/work"']
    for key in ("input_dir", "output_dir"):
        path = cfg.get(key)
        if not path:
            continue
        p = Path(path)
        if p.is_absolute():
            bind_specs.append(f'"{p}:{p}"')
    return " ".join(f"--bind {spec}" for spec in dict.fromkeys(bind_specs))


def _apptainer_image_ref(image: str) -> str:
    """Map workflow image refs to native Apptainer image locations."""
    if "://" in image or image.endswith(".sif"):
        return image

    repo, tag = image.rsplit(":", 1)
    return f"oras://{repo}:apptainer-{tag}"

# --user: make bind-mounted writes host-owned (Linux docker otherwise writes as root).
# -e HOME=/tmp: pixi activation needs a writable HOME after dropping root.
if CONTAINER_RUNTIME == "docker":
    GPU_FLAG = "--gpus all " if DEVICE == "gpu" else ""
    CONTAINER_PREFIX = (
        f'docker run --rm --pull=missing {GPU_FLAG}'
        '--user "$(id -u):$(id -g)" '
        '-e HOME=/tmp '
        '-v "$PWD:/work" -w /work '
    )
    def container_image_ref(image: str) -> str:
        return f"{CONTAINER_PREFIX}{image}"
else:
    GPU_FLAG = "--nv " if DEVICE == "gpu" else ""
    CONTAINER_PREFIX = (
        # `--unsquash` avoids FUSE-based SIF mounts on execute nodes that do
        # not expose /dev/fuse inside the parent HTCondor container.
        f'apptainer run --unsquash {GPU_FLAG}'
        f'{_apptainer_bind_flags(config)} '
        '--pwd /work '
    )
    def container_image_ref(image: str) -> str:
        return f"{CONTAINER_PREFIX}{_apptainer_image_ref(image)}"

shell.executable("bash")
# Propagate failures through `cmd | tee {log}` pipelines so a crashed stage
# does not look successful just because tee exited 0.
shell.prefix("set -o pipefail; ")

P = resolve_pipeline_paths(config)
S1_INPUT  = P["s1_input"]
S1_OUTPUT = P["s1_output"]
S2_OUTPUT = P["s2_output"]
S3_CONFIG = P["s3_config"]
S3_OUTPUT = P["s3_output"]
S4_CONFIG = P["s4_config"]
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
        f"{container_image_ref(STAGE1_IMG)} "
        f"vmec_jax {{input}} --output {{output}}"
        " 2>&1 | tee {log}"

rule stage2_boozer:
    input:  S1_OUTPUT
    output: S2_OUTPUT
    log:    f"{P['stage2_dir']}/{RUN_NAME}.log"
    shell:
        f"{container_image_ref(STAGE2_IMG)} "
        "python stages/stage2-boozer/run_boozer.py --wout {input} --output {output}"
        " 2>&1 | tee {log}"

rule stage3_sfincs:
    input:
        config_file = S3_CONFIG,
        wout        = S1_OUTPUT,
        common_config = S5_CONFIG,
    output:
        S3_OUTPUT,
    log:
        f"{P['stage3_dir']}/{RUN_NAME}.log"
    shell:
        stage3_helper.radial_scan_cmd(
            docker_prefix=CONTAINER_PREFIX,
            image=_apptainer_image_ref(STAGE3_JAX_IMG) if CONTAINER_RUNTIME == "apptainer" else STAGE3_JAX_IMG,
            stage_cfg=STAGE3_CFG,
            output_dir=P["stage3_dir"],
            device=DEVICE,
        ) + " 2>&1 | tee {log}"

rule stage4_spectrax:
    input:
        config_file = S4_CONFIG,
        wout        = S1_OUTPUT,
        boozer      = S2_OUTPUT,
        common_config = S5_CONFIG,
    output:
        S4_OUTPUT,
    log:
        f"{P['stage4_dir']}/{RUN_NAME}.log"
    shell:
        stage4_helper.radial_scan_cmd(
            docker_prefix=CONTAINER_PREFIX,
            image=_apptainer_image_ref(STAGE4_IMG) if CONTAINER_RUNTIME == "apptainer" else STAGE4_IMG,
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
        f"{container_image_ref(STAGE5_IMG)} "
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
        f'{container_image_ref(STAGE5_IMG)} sh -c "'
        'python stages/stage5-post-processing/fit_vmec_pressure_from_transport_h5.py '
        f'write-input {{input}} {S1_INPUT} --output-input {{output.feedback}} && '
        'python stages/stage5-post-processing/stage5_post_processing.py '
        f'--transport {{input}} --signal {{output.signal}} --pressure-rel-tol {PRESSURE_REL_TOL}"'
        " 2>&1 | tee {log}"
