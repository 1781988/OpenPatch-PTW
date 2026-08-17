#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-core}"
RUN_NAME="${2:-icassp_main}"
CONFIG="${CONFIG:-configs/openpatch_ptw.yaml}"

python run_all_experiments.py \
  --config "${CONFIG}" \
  --run-name "${RUN_NAME}" \
  --mode "${MODE}"
