#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

if (( $# == 0 )); then
  echo "usage: pipeline.sh {generate|launch|batch|test} [arguments...]" >&2
  exit 2
fi

COMMAND="$1"
shift
cd "${REPO_ROOT}"

case "${COMMAND}" in
  generate|launch|batch)
    exec pixi run --manifest-path "${REPO_ROOT}/pixi.toml" -e pipeline \
      python -m experiments.constellaration.runs "${COMMAND}" "$@"
    ;;
  test)
    exec pixi run --manifest-path "${REPO_ROOT}/pixi.toml" -e pipeline \
      pytest experiments/constellaration/tests "$@"
    ;;
  *)
    echo "unknown experiment command: ${COMMAND}" >&2
    exit 2
    ;;
esac
