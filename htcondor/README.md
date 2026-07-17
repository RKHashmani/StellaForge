# Running on HTCondor cluster Guide

This directory contains the files used to run the pipeline on CHTC with HTCondor. Having a staging repo for input and output files is required for nodes input and output to compute.

A brief demo for running the quick_run exmaple is as follows. In actual runs, replace with actual config and inputs.

In root dir (i.e. /driftless-star), run with:
```
pixi run -e pipeline snakemake \
    --profile profiles/htcondor-gpu \
    --configfile /staging/YOUR_STAGING_DIR/driftless-star/inputs/quick_run/config.yaml \
    --config \
        device=gpu container_runtime=apptainer \
        input_dir=/staging/YOUR_STAGING_DIR/driftless-star/inputs/quick_run \
        output_dir=/staging/YOUR_STAGING_DIR/driftless-star/outputs/quick_run
```

If `LockException` is recevied, it is usually because there is one previously unfinished run. To unlock, run:
```
pixi run -e pipeline snakemake --unlock \
    --profile htcondor/profiles/htcondor-gpu \
    --configfile /staging/YOUR_STAGING_DIR/driftless-star/inputs/quick_run/config.yaml \
    --config \
        device=gpu container_runtime=apptainer \
        input_dir=/staging/YOUR_STAGING_DIR/driftless-star/inputs/quick_run \
        output_dir=/staging/YOUR_STAGING_DIR/driftless-star/outputs/quick_run \
```
Then rerun with the `--rerun-incomplete` flag set to complete the previous run.

## Main Idea

The parent image from `apptainer.def` is used only to provide a stable runtime
for remote HTCondor jobs. Inside that parent image, the workflow may launch the
stage-specific images for VMEC, BOOZ_XFORM, SFINCS, SPECTRAX-GK, and NEOPAX.

So the layering is:

1. HTCondor launches the parent image `chtc-runtime.sif`.
2. Inside that parent image, Snakemake executes one rule.
3. That rule launches the stage container command.

## What Each File Does

- `../chtc-runtime.sif`
  The built parent runtime image consumed by HTCondor `universe=container` jobs. Needs to be in the root.

- `profiles/htcondor/job_wrapper.sh`
  Job wrapper for snakemake HTCondor plugin.

- `profiles/htcondor-gpu/config.yaml`
  Congif for GPU-based runs on HTCondor cluster.

- `container/apptainer.def`
  Builds the parent runtime image `chtc-runtime.sif`. It can be used to created the sif container from scratch(via `apptainer build chtc-runtime.sif apptainer.def`). It provides:
  - the `chtc-submit` Pixi environment,
  - `snakemake`,
  - `apptainer`,
  - the HTCondor Snakemake executor plugin.

- `container/apptainer_submit.sub`
  A helper submit file for image-building experiments.
