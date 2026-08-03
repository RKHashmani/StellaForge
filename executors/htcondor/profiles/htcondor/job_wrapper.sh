#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${_CONDOR_JOB_IWD:-}"
if [[ -z "${repo_root}" ]]; then
  repo_root="$(pwd -P)"
fi
pipeline_snakemake="${repo_root}/.pixi/envs/pipeline/bin/snakemake"
runtime_snakemake="/app/.pixi/envs/chtc_submit/bin/snakemake"

export HOME="${HOME:-$PWD}"
if [[ ! -w "${HOME}" ]]; then
  export HOME="$PWD"
fi

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${repo_root}/.snakemake/apptainer-cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${TMPDIR:-/tmp}}"
export SINGULARITY_CACHEDIR="${SINGULARITY_CACHEDIR:-${APPTAINER_CACHEDIR}}"
mkdir -p "${APPTAINER_CACHEDIR}"

echo "[job_wrapper] pwd=$(pwd -P)" >&2
echo "[job_wrapper] repo_root=${repo_root}" >&2
echo "[job_wrapper] PATH=${PATH}" >&2
echo "[job_wrapper] pipeline_snakemake=${pipeline_snakemake}" >&2
echo "[job_wrapper] runtime_snakemake=${runtime_snakemake}" >&2
command -v snakemake >&2 || true
command -v apptainer >&2 || true

ls -ld /app /app/.pixi /app/.pixi/envs /app/.pixi/envs/chtc_submit /app/.pixi/envs/chtc_submit/bin >&2 || true
ls -l "${pipeline_snakemake}" "${runtime_snakemake}" >&2 || true

if [[ -x "${pipeline_snakemake}" ]]; then
  exec "${pipeline_snakemake}" "$@"
fi

if [[ -x "${runtime_snakemake}" ]]; then
  exec "${runtime_snakemake}" "$@"
fi

exec snakemake "$@"
