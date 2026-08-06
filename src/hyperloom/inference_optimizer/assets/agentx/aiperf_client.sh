#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################
#
# aiperf_client.sh — AgentX benchmark client for Magpie's benchmark path.
#
# Design: DELEGATE the server phase to the maintained per-framework builtin
# (vllm_mi300x.sh / sglang_mi300x.sh) via MAGPIE_RUN_PHASE=server, so server
# boot + torch-profiler enabling stay correct across frameworks and versions
# (no profiler flags reimplemented here). Then run `aiperf profile` (AgentX
# weka-trace scenario) as the client, and map its export into the InferenceX
# result schema. On single node the client owns profiling: InferenceX's
# benchmark_serving.py self-triggers /start_profile, and aiperf does not, so
# when PROFILE=1 this script self-brackets a /start_profile..stop_profile window
# around aiperf's steady state (see the PROFILE branch below).
#
# Inputs (from Magpie env): MODEL, TP, PORT, MAX_MODEL_LEN, CONC, RESULT_DIR,
#   RESULT_FILENAME, PROFILE, EXTRA_VLLM_ARGS, FRAMEWORK, GPU_TYPE/RUNNER_TYPE.
# AgentX knobs (AGENTX_ prefix; NOT AIPERF_, which aiperf's own settings read):
#   AGENTX_DATASET, AGENTX_MAX_CTX, AGENTX_NUM_ENTRIES,
#   AGENTX_WARMUP_DURATION, AGENTX_NUM_WARMUP_SESSIONS, AGENTX_KEEP_SERVER,
#   AGENTX_PROFILE_WARMUP_S, AGENTX_PROFILE_WINDOW_S,
#   AGENTX_SERVER_SCRIPT (override builtin name), AIPERF_BIN.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
log() { echo "[aiperf_client] $*"; }

: "${MODEL:?MODEL required}"
PORT="${PORT:-8000}"
CONC="${CONC:-16}"
RESULT_DIR="${RESULT_DIR:-$(pwd)}"
RESULT_FILENAME="${RESULT_FILENAME:-inferencex_result}"
ART="${RESULT_DIR}/aiperf_artifacts"
# Start from a clean artifact dir so a prior round's export can never be
# mis-read as this round's result (and so `find` matches exactly one file).
rm -rf "$ART"
mkdir -p "$RESULT_DIR" "$ART"

# ── Resolve the per-framework builtin server script ──────────────────────────
FRAMEWORK="${FRAMEWORK:-}"
GPU="$(printf '%s' "${GPU_TYPE:-${RUNNER_TYPE:-mi300x}}" | tr '[:upper:]' '[:lower:]')"
BUILTIN="${AGENTX_SERVER_SCRIPT:-${FRAMEWORK}_${GPU}.sh}"
# The AgentX switch injects FRAMEWORK from benchmark.framework; a missing value
# (and no explicit AGENTX_SERVER_SCRIPT) is misconfiguration -- fail loud rather
# than silently defaulting to a framework and booting the wrong server.
if [ -z "${AGENTX_SERVER_SCRIPT:-}" ] && [ -z "$FRAMEWORK" ]; then
  log "ERROR: FRAMEWORK unset and AGENTX_SERVER_SCRIPT not provided; cannot resolve the builtin server script"
  exit 2
fi
if [ ! -f "${BENCH_DIR}/${BUILTIN}" ]; then
  log "ERROR: builtin server script not found: ${BENCH_DIR}/${BUILTIN}"
  exit 2
fi

# ── Server phase: delegate to builtin (correct boot + profiler per framework) ─
PIDFILE="${RESULT_DIR}/agentx_server.pid"
rm -f "$PIDFILE"
SERVER_PID=""  # set after boot; cleanup guards ${SERVER_PID:-} + a port fallback

cleanup() {
  [ "${AGENTX_KEEP_SERVER:-0}" = "1" ] && return 0
  if [ -n "${SERVER_PID:-}" ]; then
    log "tearing down server pid=${SERVER_PID}"
    kill -TERM "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    # vLLM can ignore/stall on SIGTERM (graceful shutdown hangs after a large
    # profiler-trace flush), leaking the GPUs. Escalate to SIGKILL if the
    # process group is still alive after a grace period.
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
  # Belt-and-suspenders: free the port even if the pid was unknown/stale, so a
  # server that booted without a recorded pid can never leak the GPUs.
  command -v fuser >/dev/null 2>&1 && fuser -k "${PORT}/tcp" 2>/dev/null || true
}
# Install the trap BEFORE booting the server: if the builtin starts the server
# then returns nonzero, set -e aborts here and the EXIT trap still fires (the
# port fallback reaps a server booted without a recorded pid) — no leak window.
trap cleanup EXIT INT TERM

log "delegating server boot -> ${BUILTIN} (PROFILE=${PROFILE:-0})"
MAGPIE_RUN_PHASE=server MAGPIE_SERVER_PID_FILE="$PIDFILE" \
  PORT="$PORT" RESULT_DIR="$RESULT_DIR" \
  bash "${BENCH_DIR}/${BUILTIN}"
SERVER_PID="$(cat "$PIDFILE" 2>/dev/null || true)"

# Fail loud if the builtin server phase did not record a pid: proceeding would
# run a benchmark against a server we cannot reliably tear down.
if [ -z "${SERVER_PID:-}" ]; then
  log "ERROR: builtin server phase wrote no pid to ${PIDFILE}; refusing to run (would risk a GPU leak)"
  exit 3
fi
log "server up (pid=${SERVER_PID}) on port ${PORT}"

# ── Resolve served model name (a reused server may expose a different id) ─────
SERVE_MODEL="$MODEL"
_served="$(curl -sf "http://localhost:${PORT}/v1/models" 2>/dev/null \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["data"][0]["id"])' 2>/dev/null || true)"
[ -n "$_served" ] && SERVE_MODEL="$_served"

# ── Clamp requested context to what the model can actually serve ──────────────
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAXCTX="${AGENTX_MAX_CTX:-$MAX_MODEL_LEN}"
if [ "$MAXCTX" -gt "$MAX_MODEL_LEN" ] 2>/dev/null; then MAXCTX="$MAX_MODEL_LEN"; fi

DS="${AGENTX_DATASET:-semianalysis-cc-traces-weka-with-subagents}"
NENT="${AGENTX_NUM_ENTRIES:-16}"
WARMDUR="${AGENTX_WARMUP_DURATION:-45}"
WARMSESS="${AGENTX_NUM_WARMUP_SESSIONS:-$CONC}"

# aiperf has no "0 warmup" value: it treats UNSET flags as "no warmup phase" and
# rejects an explicit 0 (loadgen.warmup_duration/num_sessions error). So build
# the warmup flags conditionally and OMIT them to disable warmup.
WARMUP_ARGS=()
if [ -n "${WARMDUR}" ] && [ "${WARMDUR}" != "0" ]; then
  WARMUP_ARGS+=(--warmup-duration "$WARMDUR")
fi
if [ -n "${WARMSESS}" ] && [ "${WARMSESS}" != "0" ]; then
  WARMUP_ARGS+=(--num-warmup-sessions "$WARMSESS")
fi

# aiperf reads AIPERF_-prefixed env into its own pydantic settings; scrub any
# stray exported ones (keep AIPERF_BIN, which is ours) so they can't corrupt it.
while IFS='=' read -r _k _; do
  case "$_k" in
    AIPERF_BIN) : ;;
    AIPERF_*) unset "$_k" 2>/dev/null || true ;;
  esac
done < <(env)

AIPERF="${AIPERF_BIN:-aiperf}"
log "aiperf model=${SERVE_MODEL} dataset=${DS} conc=${CONC} maxctx=${MAXCTX} warmup=${WARMDUR}s/${WARMSESS}sess"

run_aiperf() {
  "$AIPERF" profile \
    --url "http://localhost:${PORT}" \
    --model "$SERVE_MODEL" \
    --endpoint-type chat --streaming --use-server-token-count \
    --public-dataset "$DS" \
    --num-dataset-entries "$NENT" \
    --max-context-length "$MAXCTX" \
    --concurrency "$CONC" \
    ${WARMUP_ARGS[@]+"${WARMUP_ARGS[@]}"} \
    --artifact-dir "$ART" --ui simple
}

AIPERF_RC=0
if [ "${PROFILE:-0}" = "1" ]; then
  # Single-node trace capture: aiperf (a generic client) never triggers vLLM's
  # /start_profile the way InferenceX's benchmark_serving.py does, so this script
  # self-brackets a wall-clock profiling window around aiperf's steady state.
  # The profiler is already ENABLED on the server (builtin server phase adds the
  # framework's --profiler-config/env when PROFILE=1); /start_profile begins
  # recording and /stop_profile flushes the trace to torch_profiler_dir for
  # TraceLens. Only fires under PROFILE=1, so measurement rounds pay no cost.
  PWARM="${AGENTX_PROFILE_WARMUP_S:-60}"
  PWIN="${AGENTX_PROFILE_WINDOW_S:-20}"
  log "PROFILE=1: self-bracketing profile window (delay=${PWARM}s window=${PWIN}s)"
  run_aiperf & APID=$!
  sleep "$PWARM"
  if kill -0 "$APID" 2>/dev/null; then
    if curl -sf -X POST "http://localhost:${PORT}/start_profile" >/dev/null 2>&1; then
      log "start_profile OK"
    else
      log "WARN start_profile failed (trace may be empty)"
    fi
    sleep "$PWIN"
    curl -sf -X POST "http://localhost:${PORT}/stop_profile" >/dev/null 2>&1 \
      && log "stop_profile OK" || log "WARN stop_profile failed"
  else
    log "WARN aiperf finished before profile window opened; raise AGENTX_NUM_ENTRIES or lower AGENTX_PROFILE_WARMUP_S"
  fi
  wait "$APID" || AIPERF_RC=$?
else
  run_aiperf || AIPERF_RC=$?
fi
log "aiperf exit=${AIPERF_RC}"

# Do not map a failed run: a nonzero aiperf must fail the benchmark, not emit a
# (possibly partial) result that Magpie would record as success.
if [ "$AIPERF_RC" -ne 0 ]; then
  log "ERROR: aiperf failed (rc=${AIPERF_RC}); not mapping a result"
  exit "$AIPERF_RC"
fi

# ── Map aiperf export -> InferenceX result schema ────────────────────────────
# ``-print -quit`` (no pipe) avoids a find|head SIGPIPE that would abort under
# ``set -o pipefail`` before the emptiness guard below.
PJ="$(find "$ART" -name 'profile_export_aiperf.json' -print -quit)"
if [ -z "$PJ" ]; then
  log "ERROR: no profile_export_aiperf.json produced"
  exit 1
fi
python3 "${BENCH_DIR}/map_aiperf.py" "$PJ" "${RESULT_DIR}/${RESULT_FILENAME}.json"
log "mapped -> ${RESULT_DIR}/${RESULT_FILENAME}.json"
