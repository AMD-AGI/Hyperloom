#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################
#
# mlperf_agentic_client.sh -- AgentX client that drives MLPerf ``utility/run_agentic.sh``
# against localhost:$PORT.
#
# Hyperloom settles PORT, the served model, the flow, the trajectory count and
# the concurrency; preflight checks the harness, dataset and tokenizer. The
# server is inference_suite's: this boots it from the config
# MLPERF_SERVER_CONFIG names and reaps it through MLPERF_SERVER_PID_DIR on exit.
# Without MLPERF_SERVER_CONFIG the server is off this node and already up.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
log() { echo "[mlperf_agentic_client] $*"; }

: "${CONC:?CONC required (the AgentX switch projects it from the benchmark config)}"
: "${PORT:?PORT required}"
: "${MLPERF_AGENTIC_MODEL:?MLPERF_AGENTIC_MODEL required}"
: "${MLPERF_AGENTIC_FLOW:?MLPERF_AGENTIC_FLOW required}"
: "${AGENTIC_NUM_TRAJECTORIES:?AGENTIC_NUM_TRAJECTORIES required}"
: "${AGENTIC_CONCURRENCY:?AGENTIC_CONCURRENCY required}"
: "${AGENTIC_DATASET_PATH:?AGENTIC_DATASET_PATH required}"
: "${MLPERF_TOKENIZER_DIR:?MLPERF_TOKENIZER_DIR required}"
: "${MLPERF_ENDPOINTS_DIR:?MLPERF_ENDPOINTS_DIR required}"
MODEL_KEY="$MLPERF_AGENTIC_MODEL"

RESULT_DIR="${RESULT_DIR:-$(pwd)}"
RESULT_FILENAME="${RESULT_FILENAME:-inferencex_result}"
ART="${RESULT_DIR}/mlperf_artifacts"
rm -rf "$ART"
mkdir -p "$RESULT_DIR" "$ART"

if [ -n "${MLPERF_SERVER_CONFIG:-}" ]; then
  : "${MLPERF_SERVER_PID_DIR:?MLPERF_SERVER_PID_DIR required with MLPERF_SERVER_CONFIG}"
  : "${MLPERF_SUITE_PYTHON:?MLPERF_SUITE_PYTHON required with MLPERF_SERVER_CONFIG}"
  SERVER_RUN_DIR="${RESULT_DIR}/mlperf_server_round"
  stop_server() {
    if ! "$MLPERF_SUITE_PYTHON" -m inference_suite stop --pid-dir "$MLPERF_SERVER_PID_DIR"; then
      log "WARN: a server process outlived SIGKILL; ${MLPERF_SERVER_PID_DIR}/server.pid stays for the next reap"
    fi
  }
  trap stop_server EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  # A run_dir is written once; an earlier round's server under this RESULT_DIR was reaped on its exit.
  rm -rf "$SERVER_RUN_DIR"
  log "booting the server through inference_suite (run_dir ${SERVER_RUN_DIR})"
  if ! "$MLPERF_SUITE_PYTHON" -m inference_suite run --config "$MLPERF_SERVER_CONFIG" --run-dir "$SERVER_RUN_DIR"; then
    log "ERROR: the server did not boot; see ${SERVER_RUN_DIR}/report.json"
    exit 3
  fi
else
  log "no server config: driving the server already up on port ${PORT}"
fi

HARDWARE="${MLPERF_AGENTIC_HARDWARE:-mi355x}"
export RESULTS_DIR="$ART"
export AGENTIC_DATASET_PATH MLPERF_TOKENIZER_DIR AGENTIC_NUM_TRAJECTORIES AGENTIC_CONCURRENCY

log "mlperf flow=${MLPERF_AGENTIC_FLOW} trajectories=${AGENTIC_NUM_TRAJECTORIES} model=${MODEL_KEY} hw=${HARDWARE} conc=${AGENTIC_CONCURRENCY} dataset=${AGENTIC_DATASET_PATH}"
set +e
(
  cd "$MLPERF_ENDPOINTS_DIR"
  bash utility/run_agentic.sh "$MLPERF_AGENTIC_FLOW" "$MODEL_KEY" "$HARDWARE" perf
)
HARNESS_RC=$?
set -e
if [ "$HARNESS_RC" -ne 0 ]; then
  log "ERROR: run_agentic.sh failed (rc=${HARNESS_RC}); not mapping a result"
  exit "$HARNESS_RC"
fi

SUMMARY="$(find "$ART" -name 'result_summary.json' -print -quit 2>/dev/null || true)"
if [ -z "$SUMMARY" ]; then
  log "ERROR: no result_summary.json produced under ${ART}"
  exit 1
fi
# Inline accuracy lands in scores.json beside the summary. Without it the
# mapped result carries no accuracy and the accuracy gate refuses the KEEP.
SCORES="$(find "$ART" -name 'scores.json' -print -quit 2>/dev/null || true)"
MAP_ARGS=("$SUMMARY" "${RESULT_DIR}/${RESULT_FILENAME}.json")
if [ -n "$SCORES" ]; then
  MAP_ARGS+=("$SCORES")
fi
python3 "${BENCH_DIR}/map_mlperf.py" "${MAP_ARGS[@]}"
log "mapped ${SUMMARY} -> ${RESULT_DIR}/${RESULT_FILENAME}.json"
