#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################
#
# mlperf_agentic_client.sh — AgentX client that keeps Magpie server lifecycle
# and drives MLPerf ``utility/run_agentic.sh`` against localhost:30000.
#
# MAGPIE_RUN_PHASE=server / client matches aiperf_client.sh. Search uses
# smoke.yaml (150 trajectories); KEEP validation sets MLPERF_AGENTIC_FLOW=full.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
log() { echo "[mlperf_agentic_client] $*"; }

: "${MODEL:?MODEL required}"
: "${CONC:?CONC required (the AgentX switch projects it from the benchmark config)}"
PORT="${PORT:-30000}"

RESULT_DIR="${RESULT_DIR:-$(pwd)}"
RESULT_FILENAME="${RESULT_FILENAME:-inferencex_result}"
ART="${RESULT_DIR}/mlperf_artifacts"
rm -rf "$ART"
mkdir -p "$RESULT_DIR" "$ART"

if [ "${MAGPIE_RUN_PHASE:-full}" = "client" ]; then
  log "client-only mode: reusing caller-managed server on port ${PORT}"
else
  FRAMEWORK="${FRAMEWORK:-}"
  GPU="$(printf '%s' "${GPU_TYPE:-${RUNNER_TYPE:-mi355x}}" | tr '[:upper:]' '[:lower:]')"
  BUILTIN="${AGENTX_SERVER_SCRIPT:-${FRAMEWORK}_${GPU}.sh}"
  if [ -z "${AGENTX_SERVER_SCRIPT:-}" ] && [ -z "$FRAMEWORK" ]; then
    log "ERROR: FRAMEWORK unset and AGENTX_SERVER_SCRIPT not provided; cannot resolve the builtin server script"
    exit 2
  fi
  if [ ! -f "${BENCH_DIR}/${BUILTIN}" ]; then
    log "ERROR: builtin server script not found: ${BENCH_DIR}/${BUILTIN}"
    exit 2
  fi

  PIDFILE="${RESULT_DIR}/agentx_server.pid"
  rm -f "$PIDFILE"
  SERVER_PID=""

  cleanup() {
    [ "${AGENTX_KEEP_SERVER:-0}" = "1" ] && return 0
    if [ -n "${SERVER_PID:-}" ]; then
      log "tearing down server pid=${SERVER_PID}"
      kill -TERM "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
      _i=0
      while [ "$_i" -lt 10 ]; do
        kill -0 "${SERVER_PID}" 2>/dev/null || break
        sleep 2
        _i=$((_i + 1))
      done
      if kill -0 "${SERVER_PID}" 2>/dev/null; then
        log "server survived SIGTERM after grace period; sending SIGKILL"
        kill -KILL "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null || true
      fi
    fi
    command -v fuser >/dev/null 2>&1 && fuser -k "${PORT}/tcp" 2>/dev/null || true
  }
  trap cleanup EXIT INT TERM

  _KEEPALIVE_S="${AGENTX_HTTP_KEEP_ALIVE_S:-900}"
  _ka_target=""
  case "$BUILTIN" in
    *vllm*) _ka_target=vllm ;;
    *sglang*) _ka_target=sglang ;;
    *)
      case "${FRAMEWORK:-}" in
        *vllm*) _ka_target=vllm ;;
        *sglang*) _ka_target=sglang ;;
      esac
      ;;
  esac
  case "$_ka_target" in
    vllm) export VLLM_HTTP_TIMEOUT_KEEP_ALIVE="${VLLM_HTTP_TIMEOUT_KEEP_ALIVE:-$_KEEPALIVE_S}" ;;
    sglang) export SGLANG_TIMEOUT_KEEP_ALIVE="${SGLANG_TIMEOUT_KEEP_ALIVE:-$_KEEPALIVE_S}" ;;
  esac

  log "delegating server boot -> ${BUILTIN} (PROFILE=${PROFILE:-0}) PORT=${PORT}"
  MAGPIE_RUN_PHASE=server MAGPIE_SERVER_PID_FILE="$PIDFILE" \
    PORT="$PORT" RESULT_DIR="$RESULT_DIR" \
    bash "${BENCH_DIR}/${BUILTIN}"
  SERVER_PID="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -z "${SERVER_PID:-}" ]; then
    log "ERROR: builtin server phase wrote no pid to ${PIDFILE}; refusing to run (would risk a GPU leak)"
    exit 3
  fi
  log "server up (pid=${SERVER_PID}) on port ${PORT}"
fi

MLPERF_ROOT="${MLPERF_ENDPOINTS_DIR:-/opt/mlperf-endpoints}"
if [ ! -f "${MLPERF_ROOT}/utility/run_agentic.sh" ]; then
  log "ERROR: MLPerf harness missing at ${MLPERF_ROOT}/utility/run_agentic.sh"
  exit 2
fi
if [ -z "${AGENTIC_DATASET_PATH:-}" ] || [ ! -e "${AGENTIC_DATASET_PATH}" ]; then
  log "ERROR: AGENTIC_DATASET_PATH is missing or unreadable: ${AGENTIC_DATASET_PATH:-unset}"
  exit 2
fi
if [ -z "${MLPERF_TOKENIZER_DIR:-}" ] || [ ! -d "${MLPERF_TOKENIZER_DIR}" ]; then
  log "ERROR: MLPERF_TOKENIZER_DIR is missing: ${MLPERF_TOKENIZER_DIR:-unset}"
  exit 2
fi

FLOW="${MLPERF_AGENTIC_FLOW:-smoke_test}"
MODEL_KEY="${MLPERF_AGENTIC_MODEL:-kimi-k3}"
HARDWARE="${MLPERF_AGENTIC_HARDWARE:-mi355x}"
export AGENTIC_CONCURRENCY="${AGENTIC_CONCURRENCY:-$CONC}"
export RESULTS_DIR="$ART"
export AGENTIC_DATASET_PATH
export MLPERF_TOKENIZER_DIR
if [ "$FLOW" = "smoke_test" ]; then
  export AGENTIC_NUM_TRAJECTORIES="${AGENTIC_NUM_TRAJECTORIES:-150}"
else
  export AGENTIC_NUM_TRAJECTORIES="${AGENTIC_NUM_TRAJECTORIES:-613}"
fi

NONCANON=()
[ "$FLOW" = "smoke_test" ] && NONCANON+=("flow=smoke_test(canonical full/613)")
[ "${AGENTIC_NUM_TRAJECTORIES}" != "613" ] && NONCANON+=("entries=${AGENTIC_NUM_TRAJECTORIES}(canonical 613)")
export AGENTX_NONCANONICAL_REASONS=""
if [ ${#NONCANON[@]} -gt 0 ]; then
  _reasons="$(IFS=,; echo "${NONCANON[*]}")"
  export AGENTX_NONCANONICAL_REASONS="$_reasons"
  log "SMOKE: non-canonical MLPerf workload [${_reasons}] -- KEEP requires a full 613 validation"
fi

log "mlperf flow=${FLOW} model=${MODEL_KEY} hw=${HARDWARE} conc=${AGENTIC_CONCURRENCY} dataset=${AGENTIC_DATASET_PATH}"
set +e
(
  cd "$MLPERF_ROOT"
  bash utility/run_agentic.sh "$FLOW" "$MODEL_KEY" "$HARDWARE" perf
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
# Inline accuracy lands in scores.json beside the summary; the accuracy/ path is
# what a separate acc-only pass would write, and K3 has no such dataset. Both are
# checked so neither layout silently maps to "no accuracy" and blocks every KEEP.
ACCURACY="$(find "$ART" -name 'scores.json' -print -quit 2>/dev/null || true)"
if [ -z "$ACCURACY" ]; then
  ACCURACY="$(find "$ART" -path '*/accuracy/accuracy_results.json' -print -quit 2>/dev/null || true)"
fi
MAP_ARGS=("$SUMMARY" "${RESULT_DIR}/${RESULT_FILENAME}.json")
if [ -n "$ACCURACY" ]; then
  MAP_ARGS+=("$ACCURACY")
fi
python3 "${BENCH_DIR}/map_mlperf.py" "${MAP_ARGS[@]}"
log "mapped ${SUMMARY} -> ${RESULT_DIR}/${RESULT_FILENAME}.json"
