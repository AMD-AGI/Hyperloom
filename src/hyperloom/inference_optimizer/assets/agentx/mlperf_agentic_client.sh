#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################
#
# mlperf_agentic_client.sh — AgentX client that keeps Magpie server lifecycle
# and drives MLPerf ``utility/run_agentic.sh`` against localhost:30000.
#
# MAGPIE_RUN_PHASE=server / client matches aiperf_client.sh. Search measures
# smoke.yaml (150 trajectories) against a smoke baseline. The canonical 613
# confirmation sets MLPERF_AGENTIC_FLOW=full once, on the final stack.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
log() { echo "[mlperf_agentic_client] $*"; }

: "${MODEL:?MODEL required}"
: "${CONC:?CONC required (the AgentX switch projects it from the benchmark config)}"
# The harness always dials localhost:30000. A different PORT would benchmark
# whatever already answers there, then have the exit trap kill it.
if [ -n "${PORT:-}" ] && [ "$PORT" != "30000" ]; then
  log "ERROR: MLPerf harness targets localhost:30000; refusing PORT=${PORT}"
  exit 2
fi
PORT=30000
export PORT

MODEL_KEY="${MLPERF_AGENTIC_MODEL:-kimi-k3}"
case "$MODEL_KEY" in
  kimi-k3|kimi_k3) ;;
  *)
    log "ERROR: MLPerf agentic client only measures kimi-k3 (got ${MODEL_KEY})"
    exit 2
    ;;
esac

_port_open() {
  python3 - "$1" <<'PY'
import socket
import sys

sock = socket.socket()
sock.settimeout(0.5)
try:
    sys.exit(0 if sock.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
finally:
    sock.close()
PY
}

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

  if _port_open "$PORT"; then
    log "ERROR: localhost:${PORT} is already accepting connections; refusing to benchmark a server this round did not start"
    exit 2
  fi

  PIDFILE="${RESULT_DIR}/agentx_server.pid"
  rm -f "$PIDFILE"
  SERVER_PID=""
  MLPERF_OWN_PORT=0

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
    if [ "${MLPERF_OWN_PORT:-0}" = "1" ]; then
      command -v fuser >/dev/null 2>&1 && fuser -k "${PORT}/tcp" 2>/dev/null || true
    fi
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
  python3 - "$PORT" "$SERVER_PID" <<'PY'
import os
import sys

port = int(sys.argv[1])
server = int(sys.argv[2])
hexport = f"{port:04X}"
inodes = set()
for name in ("/proc/net/tcp", "/proc/net/tcp6"):
    try:
        lines = open(name, encoding="utf-8").read().splitlines()[1:]
    except OSError:
        continue
    for line in lines:
        parts = line.split()
        if len(parts) < 10 or parts[3] != "0A":
            continue
        if parts[1].rsplit(":", 1)[-1].upper() != hexport:
            continue
        inodes.add(parts[9])
if not inodes:
    sys.stderr.write(f"ERROR: nothing is listening on port {port} after server start\n")
    sys.exit(1)

def ancestors(pid: int) -> set[int]:
    seen: set[int] = set()
    while pid and pid not in seen:
        seen.add(pid)
        try:
            stat = open(f"/proc/{pid}/stat", encoding="utf-8").read()
        except OSError:
            break
        pid = int(stat.rsplit(")", 1)[1].split()[1])
    return seen

owned = False
for pid_name in os.listdir("/proc"):
    if not pid_name.isdigit():
        continue
    fd_dir = f"/proc/{pid_name}/fd"
    try:
        fds = os.listdir(fd_dir)
    except OSError:
        continue
    for fd in fds:
        try:
            target = os.readlink(f"{fd_dir}/{fd}")
        except OSError:
            continue
        if target.startswith("socket:[") and target[8:-1] in inodes:
            if server in ancestors(int(pid_name)):
                owned = True
                break
    if owned:
        break
if not owned:
    sys.stderr.write(
        f"ERROR: listener on port {port} is not the server this round started (pid {server})\n"
    )
    sys.exit(1)
PY
  MLPERF_OWN_PORT=1
fi

python3 - "$PORT" "$MODEL_KEY" <<'PY'
import json
import sys
import urllib.request

port, want = sys.argv[1], sys.argv[2].lower()
url = f"http://127.0.0.1:{port}/v1/models"
try:
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.load(resp)
except Exception as exc:
    sys.stderr.write(f"ERROR: {url} did not answer after boot: {exc}\n")
    sys.exit(1)
ids = [str(item.get("id") or "") for item in (payload.get("data") or []) if isinstance(item, dict)]
if want not in {item.lower() for item in ids}:
    sys.stderr.write(f"ERROR: served models {ids} do not include {want}\n")
    sys.exit(1)
print(f"[mlperf_agentic_client] served model ok: {ids}")
PY

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
HARDWARE="${MLPERF_AGENTIC_HARDWARE:-mi355x}"
# CONC is the concurrency this round is recorded under. A stale
# AGENTIC_CONCURRENCY from the baseline YAML must not outrank it.
if [ -n "${CONC:-}" ]; then
  export AGENTIC_CONCURRENCY="$CONC"
else
  export AGENTIC_CONCURRENCY="${AGENTIC_CONCURRENCY:-16}"
fi
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
  log "SMOKE: non-canonical MLPerf workload [${_reasons}] -- measurable for search; canonical submission is the 613 confirmation"
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
