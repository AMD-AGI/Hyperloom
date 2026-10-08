#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################
#
# mlperf_agentic_client.sh — AgentX client that keeps Magpie server lifecycle
# and drives MLPerf ``utility/run_agentic.sh`` against localhost:30000.
#
# MAGPIE_RUN_PHASE=server / client matches aiperf_client.sh. The AgentX switch
# settles PORT, the served model, the flow, the trajectory count and the
# concurrency; preflight checks the harness, dataset and tokenizer. This script
# checks only what it alone can see: who owns the port, and what is served.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
log() { echo "[mlperf_agentic_client] $*"; }

: "${MODEL:?MODEL required}"
: "${CONC:?CONC required (the AgentX switch projects it from the benchmark config)}"
: "${PORT:?PORT required}"
: "${MLPERF_AGENTIC_MODEL:?MLPERF_AGENTIC_MODEL required}"
: "${MLPERF_AGENTIC_FLOW:?MLPERF_AGENTIC_FLOW required}"
: "${AGENTIC_NUM_TRAJECTORIES:?AGENTIC_NUM_TRAJECTORIES required}"
: "${AGENTIC_CONCURRENCY:?AGENTIC_CONCURRENCY required}"
: "${AGENTIC_DATASET_PATH:?AGENTIC_DATASET_PATH required}"
: "${MLPERF_TOKENIZER_DIR:?MLPERF_TOKENIZER_DIR required}"
: "${MLPERF_ENDPOINTS_DIR:?MLPERF_ENDPOINTS_DIR required}"
export PORT
MODEL_KEY="$MLPERF_AGENTIC_MODEL"

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
  # A framework whose keep-alive variable this script does not know -- atom's is not published -- takes it by name
  # here rather than by a guess hard-coded above. An agentic turn can idle for minutes between tool results, so a
  # server defaulting to a short keep-alive will drop the connection mid-trajectory.
  #
  # The name is untrusted: it arrives through ``benchmark.envs`` like any variant env. It is read by indirect
  # expansion rather than eval, and refused unless it is a shell identifier -- eval on a crafted name would run
  # commands in this shell. ``env_safety`` also blocks it from variant and external sources; this is the second
  # layer, since a legitimate operator export reaches here without passing through that filter.
  if [ -n "${AGENTX_KEEP_ALIVE_ENV:-}" ]; then
    case "$AGENTX_KEEP_ALIVE_ENV" in
      [!A-Za-z_]* | *[!A-Za-z0-9_]*)
        log "ERROR: AGENTX_KEEP_ALIVE_ENV=${AGENTX_KEEP_ALIVE_ENV} is not a variable name"
        exit 2
        ;;
    esac
    export "${AGENTX_KEEP_ALIVE_ENV}=${!AGENTX_KEEP_ALIVE_ENV:-$_KEEPALIVE_S}"
    log "keep-alive: ${AGENTX_KEEP_ALIVE_ENV}=${!AGENTX_KEEP_ALIVE_ENV}"
  fi

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
