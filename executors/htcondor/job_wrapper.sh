#!/usr/bin/env bash

set -euo pipefail

repo_root="${_CONDOR_JOB_IWD:-}"
if [[ -z "${repo_root}" ]]; then
  repo_root="$(pwd -P)"
fi
runtime_snakemake="/app/.pixi/envs/htcondor-runtime/bin/snakemake"

export HOME="${HOME:-$PWD}"
if [[ ! -w "${HOME}" ]]; then
  export HOME="$PWD"
fi

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${repo_root}/.snakemake/apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${APPTAINER_CACHEDIR}}"
mkdir -p "${APPTAINER_CACHEDIR}"

if [[ ! -x "${runtime_snakemake}" ]]; then
  echo "[job_wrapper] FATAL: no executable Snakemake at ${runtime_snakemake}" >&2
  echo "[job_wrapper] The parent Apptainer image is missing, or was built from an environment other than" >&2
  echo "[job_wrapper] htcondor-runtime. Rebuild htcondor-runtime.sif from executors/htcondor/apptainer.def" >&2
  echo "[job_wrapper] and check the profile's universe/container_image resources." >&2
  echo "[job_wrapper] pwd=$(pwd -P)" >&2
  echo "[job_wrapper] repo_root=${repo_root}" >&2
  echo "[job_wrapper] PATH=${PATH}" >&2
  command -v snakemake >&2 || true
  command -v apptainer >&2 || true
  ls -ld /app /app/.pixi /app/.pixi/envs \
         /app/.pixi/envs/htcondor-runtime /app/.pixi/envs/htcondor-runtime/bin >&2 || true
  ls -l "${runtime_snakemake}" >&2 || true
  exit 1
fi

echo "[job_wrapper] execute-node $("${runtime_snakemake}" --version 2>&1 | head -1) from ${runtime_snakemake}" >&2

exec "${runtime_snakemake}" "$@"
