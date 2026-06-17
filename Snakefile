# driftless-star MVP Snakemake workflow

from src import stage3, stage4, stage5

configfile: "config.yaml"

RUN_NAME = config["run_name"]
# Substitute {run_name} into each directory value; literal paths pass through unchanged.
DIRS = {k: v.format(run_name=RUN_NAME) for k, v in config["directories"].items()}
FILES = config["filenames"]


# Substitute {run_name} into the filename for `key`; literal names pass through unchanged.
def filename(key):
    return FILES[key].format(run_name=RUN_NAME)

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

S1_INPUT  = f"{DIRS['stage1_input']}/{filename('s1_input')}"
S1_OUTPUT = f"{DIRS['stage1_output']}/{filename('s1_output')}"
S2_OUTPUT = f"{DIRS['stage2_output']}/{filename('s2_output')}"

STAGE3_CFG = config["stage3"]["sfincs_jax"]
S3_CONFIG  = f"{DIRS['stage3_input']}/{filename('s3_config')}"
S3_OUTPUT  = f"{DIRS['stage3_output']}/{filename('s3_output')}"

STAGE4_CFG = config["stage4"]["spectrax_gk"]
S4_CONFIG  = f"{DIRS['stage4_input']}/{filename('s4_config')}"
S4_OUTPUT  = f"{DIRS['stage4_output']}/{filename('s4_output')}"
S4_WANDB_POLL_INTERVAL_MINUTES = STAGE4_CFG.get("wandb_poll_interval_minutes", 10)
S4_WANDB_STOP_FILE = f"{DIRS['stage4_output']}/.wandb-monitor.done"
S4_WANDB_MONITOR_LOG = f"{DIRS['stage4_output']}/wandb_monitor.log"

S5_CONFIG  = f"{DIRS['stage5_input']}/{filename('s5_config')}"
S5_OUTPUT  = f"{DIRS['stage5_output']}/{filename('s5_output')}"
S5_SIGNAL  = f"{DIRS['stage5_post_processing_output']}/{filename('s5_signal')}"

stage5.prepare_neopax_config(
    s5_config=S5_CONFIG,
    s5_output_dir=DIRS["stage5_output"],
    s1_output=S1_OUTPUT,
    s2_output=S2_OUTPUT,
    s3_output=S3_OUTPUT,
    s4_output=S4_OUTPUT,
)


# Terminal artifact of the MVP forward pass.
rule all:
    input:
        S5_OUTPUT,

rule stage1_vmec:
    input:  S1_INPUT
    output: S1_OUTPUT
    log:    f"{DIRS['stage1_output']}/{RUN_NAME}.log"
    shell:
        f"{DOCKER_PREFIX} {STAGE1_IMG} "
        f"vmec_jax {{input}} --outdir {DIRS['stage1_output']}"
        " 2>&1 | tee {log}"

rule stage2_boozer:
    input:  S1_OUTPUT
    output: S2_OUTPUT
    log:    f"{DIRS['stage2_output']}/{RUN_NAME}.log"
    shell:
        f"{DOCKER_PREFIX} {STAGE2_IMG} "
        "python stages/stage2-boozer/run_boozer.py --wout {input} --output {output}"
        " 2>&1 | tee {log}"

rule stage3_sfincs:
    input:
        config_file = S3_CONFIG,
        wout        = S1_OUTPUT,
        neopax_config = S5_CONFIG,
    output:
        S3_OUTPUT,
    log:
        f"{DIRS['stage3_output']}/{RUN_NAME}.log"
    shell:
        stage3.radial_scan_cmd(
            docker_prefix=DOCKER_PREFIX,
            image=STAGE3_JAX_IMG,
            stage_cfg=STAGE3_CFG,
            output_dir=DIRS["stage3_output"],
            device=DEVICE,
        ) + " 2>&1 | tee {log}"

rule stage4_spectrax:
    input:
        config_file = S4_CONFIG,
        wout        = S1_OUTPUT,
        boozer      = S2_OUTPUT,
        neopax_config = S5_CONFIG,
    output:
        S4_OUTPUT,
    log:
        f"{DIRS['stage4_output']}/{RUN_NAME}.log"
    shell:
        f"""
        mkdir -p {DIRS['stage4_output']}
        rm -f {S4_WANDB_STOP_FILE}
        python tests/stage4-turbulence/SPECTRAX-GK/log_wandb_stage4_concurrent.py \
            --output-dir {DIRS['stage4_output']} \
            --progress-log {{log}} \
            --poll-interval-minutes {S4_WANDB_POLL_INTERVAL_MINUTES} \
            --stop-file {S4_WANDB_STOP_FILE} \
            > {S4_WANDB_MONITOR_LOG} 2>&1 &
        monitor_pid=$!
        set +e
        {stage4.radial_scan_cmd(
            docker_prefix=DOCKER_PREFIX,
            image=STAGE4_IMG,
            stage_cfg=STAGE4_CFG,
            output_dir=DIRS["stage4_output"],
            device=DEVICE,
        )} 2>&1 | tee {{log}}
        stage_status=${{{{PIPESTATUS[0]}}}}
        touch {S4_WANDB_STOP_FILE}
        wait "$monitor_pid"
        monitor_status=$?
        if (( stage_status != 0 )); then
            exit "$stage_status"
        fi
        if (( monitor_status != 0 )); then
            echo "warning: Stage 4 WandB monitor failed; see {S4_WANDB_MONITOR_LOG}" >&2
        fi
        exit 0
        """

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
        f"{DIRS['stage5_output']}/{RUN_NAME}.log"
    shell:
        f"{DOCKER_PREFIX} {STAGE5_IMG} "
        f"sh -c \"cd {DIRS['stage5_input']} && neopax {filename('s5_config')}\""
        " 2>&1 | tee {log}"

# Stage 5 Post-Processing closes the optimization loop and writes a convergence signal.
# The new Stage 1 input (S1_INPUT) is UNDECLARED to prevent making the DAG cyclic.
# `rule all` stays S5_OUTPUT, so a plain `snakemake` remains a pure, non-mutating forward pass.
rule stage5_post_processing:
    input:  S5_OUTPUT
    output: S5_SIGNAL
    log:    f"{DIRS['stage5_post_processing_output']}/{RUN_NAME}.log"
    shell:
        f'{DOCKER_PREFIX} {STAGE5_IMG} sh -c "'
        'python stages/stage5-post-processing/fit_vmec_pressure_from_transport_h5.py '
        f'write-input {{input}} {S1_INPUT} && '
        'python stages/stage5-post-processing/stage5_post_processing.py '
        '--transport {input} --signal {output}"'
        " 2>&1 | tee {log}"

rule clean:
    shell:
        f"""
        rm -rf {DIRS['stage1_output']} {DIRS['stage2_output']} \
               {DIRS['stage3_output']} {DIRS['stage4_output']} \
               {DIRS['stage5_output']} {DIRS['stage5_post_processing_output']}
        """
