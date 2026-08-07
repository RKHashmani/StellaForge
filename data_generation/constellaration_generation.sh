#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# A killed Snakemake controller can leave a stale repository lock. Never
# remove it while another controller is genuinely active in this repository.
ACTIVE_CONTROLLERS=()
while read -r process_id; do
  [[ -n "${process_id}" ]] || continue
  process_cwd="$(readlink -e "/proc/${process_id}/cwd" 2>/dev/null || true)"
  if [[ "${process_cwd}" == "${REPO_ROOT}" ]]; then
    ACTIVE_CONTROLLERS+=("${process_id}")
  fi
done < <(pgrep -u "$(id -u)" -f '(^|/)(snakemake)( |$)|python([0-9.]*)? -m src\.(ouroboros|snakemake_htcondor)' || true)

if (( ${#ACTIVE_CONTROLLERS[@]} > 0 )); then
  echo "Refusing to unlock: an active Snakemake/Ouroboros process is using this repository." >&2
  ps -o pid,etime,cmd -p "$(IFS=,; echo "${ACTIVE_CONTROLLERS[*]}")" >&2 || true
  echo "Stop that process, or wait for it to finish, before restarting this batch." >&2
  exit 1
fi

echo "Checking for a stale Snakemake repository lock..."
pixi run -e pipeline snakemake \
  --unlock \
  --cores 1 \
  --snakefile data_generation/unlock.smk

exec pixi run constellaration-batch \
  --output-root /staging/groups/driftless_star/constellaration_runs \
  --profile executors/htcondor/profiles/htcondor-gpu \
  --container-runtime apptainer \
  --gpu-ids all \
  --loop-iters 10 \
  --max-parallel 10 \
  --keep-going \
  --cores 8 \
  "$@"
