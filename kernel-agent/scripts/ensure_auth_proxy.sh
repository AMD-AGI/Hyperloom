#!/usr/bin/env bash
# kernel-agent OOB auth-proxy supervisor.
#
# The auth-proxy (auth_proxy.py from OOB) listens on 127.0.0.1:4002 and
# rewrites the agent CLIs' x-api-key header into Authorization: Bearer for
# the AMD LLM gateway. Without it, claude/codex CLI requests get HTTP 401
# "token not present" because the gateway's auth is non-standard.
#
# This script is idempotent and meant to be sourced or invoked before any
# OOB-dependent kernel-agent tool runs:
#
#   bash $WORKSPACE_PATH/kernel-agent/scripts/ensure_auth_proxy.sh
#
# Behaviour:
#   * If port :4002 is open AND a benign HTTP probe gets ANY HTTP status
#     back, treat the proxy as healthy and noop.
#   * If port is open but the probe times out (proxy stuck), kill any
#     auth_proxy.py PIDs and relaunch.
#   * If port is closed, launch the proxy.
#   * On launch, wait up to 10s for the port to bind. Print the resolved
#     PROXY_ANTHROPIC_BASE_URL / PROXY_OPENAI_BASE_URL so callers can eval
#     the output if they want to import the values.
#
# The script reads OOB_BASE_URL_VAL / OOB_API_KEY_VAL via the same env-var
# fallback chain as install.sh. If OOB_BASE_URL is unset, the script logs
# a warning and exits 0 (the proxy is genuinely unnecessary in that mode).

set -euo pipefail

HYPERLOOM_ROOT="${HYPERLOOM_ROOT:-/opt/hyperloom}"
PROXY_PY="${PROXY_PY:-${HYPERLOOM_ROOT}/OOB/oob_cli/auth_proxy.py}"
PROXY_PORT="${AUTH_PROXY_PORT:-4002}"
LOG_DIR="${HYPERLOOM_ROOT}/logs"
PROBE_TIMEOUT="${AUTH_PROXY_PROBE_TIMEOUT:-2}"
START_WAIT_SEC="${AUTH_PROXY_START_WAIT:-10}"

OOB_BASE_URL_VAL="${OOB_BASE_URL:-${OPENAI_BASE_URL:-${ANTHROPIC_BASE_URL:-${LLM_API_BASE:-}}}}"
OOB_API_KEY_VAL="${OOB_API_KEY:-${SAFE_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-${ANTHROPIC_API_KEY:-${OPENAI_API_KEY:-}}}}}"

log()  { echo "[ensure-auth-proxy] $*"; }
warn() { echo "[ensure-auth-proxy WARN] $*" >&2; }

# Returns 0 (open) / 1 (closed). Uses bash's /dev/tcp; cheap and avoids
# calling external curl/nc when we just need a TCP-level liveness check.
port_open() {
  (echo > "/dev/tcp/127.0.0.1/${PROXY_PORT}") >/dev/null 2>&1
}

# Returns 0 if proxy responds with any HTTP status (proof of life), 1 if
# the connection times out / hangs / is refused. We don't care WHICH
# status — even a 4xx from upstream means the proxy is forwarding.
proxy_responds() {
  if ! command -v curl >/dev/null 2>&1; then
    # No curl → fall back to TCP-level liveness only.
    port_open
    return $?
  fi
  local http_code
  http_code=$(curl -s -o /dev/null -w '%{http_code}' \
                   --max-time "$PROBE_TIMEOUT" \
                   "http://127.0.0.1:${PROXY_PORT}/" 2>/dev/null || echo 000)
  case "$http_code" in
    "" | 000) return 1 ;;  # connect/timeout failure
    *)        return 0 ;;  # any HTTP status counts as "alive"
  esac
}

# Best-effort kill of any python auth_proxy.py owning :4002 so the relaunch
# does not race with a stuck process. We use pkill on the script path; if
# pkill is unavailable, fall back to fuser.
kill_stuck_proxy() {
  if command -v pkill >/dev/null 2>&1; then
    pkill -f "auth_proxy.py" >/dev/null 2>&1 || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser -k "${PROXY_PORT}/tcp" >/dev/null 2>&1 || true
  fi
  # Give the OS a moment to release the port.
  sleep 1
}

start_proxy() {
  if [ ! -f "$PROXY_PY" ]; then
    warn "auth_proxy.py not found at ${PROXY_PY}; OOB Bearer rewrite skipped"
    return 1
  fi
  if [ -z "$OOB_BASE_URL_VAL" ]; then
    warn "OOB_BASE_URL/OPENAI_BASE_URL not set; cannot start auth-proxy"
    return 1
  fi
  local scheme host parsed_port path port_target
  scheme=$(echo "$OOB_BASE_URL_VAL" | grep -oP '^https?' || true)
  host=$(echo "$OOB_BASE_URL_VAL" | grep -oP '(?<=://)[^:/]+' || true)
  parsed_port=$(echo "$OOB_BASE_URL_VAL" | grep -oP '(?<=:)\d+(?=/)' || true)
  if [ -n "$parsed_port" ]; then
    port_target="$parsed_port"
  elif [ "$scheme" = "https" ]; then
    port_target=443
  else
    port_target=80
  fi
  path=$(echo "$OOB_BASE_URL_VAL" | grep -oP '(?<=://)[^/]+(/.+)' | grep -oP '/.*' || true)

  log "starting auth-proxy on :${PROXY_PORT} -> ${scheme}://${host}:${port_target}${path}"
  mkdir -p "$LOG_DIR"
  LLM_PROXY_SCHEME="$scheme" \
  LLM_PROXY_HOST="$host" \
  LLM_PROXY_PORT="$port_target" \
  AUTH_PROXY_PORT="$PROXY_PORT" \
  PROXY_AUTH_TOKEN="$OOB_API_KEY_VAL" \
    nohup python3 "$PROXY_PY" >"${LOG_DIR}/oob-auth-proxy.log" 2>&1 &

  local elapsed=0
  while [ "$elapsed" -lt "$START_WAIT_SEC" ]; do
    if port_open; then
      log "auth-proxy bound :${PROXY_PORT} after ${elapsed}s"
      # Stash for the caller (and for SKILL.md instructions).
      PROXY_ANTHROPIC_BASE_URL="http://127.0.0.1:${PROXY_PORT}$(echo "$path" | sed 's|/v1$||')"
      PROXY_OPENAI_BASE_URL="http://127.0.0.1:${PROXY_PORT}${path}"
      echo "PROXY_ANTHROPIC_BASE_URL=${PROXY_ANTHROPIC_BASE_URL}"
      echo "PROXY_OPENAI_BASE_URL=${PROXY_OPENAI_BASE_URL}"
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  warn "auth-proxy did not bind :${PROXY_PORT} within ${START_WAIT_SEC}s; see ${LOG_DIR}/oob-auth-proxy.log"
  return 1
}

main() {
  if proxy_responds; then
    log "auth-proxy already healthy on :${PROXY_PORT}"
    return 0
  fi

  if port_open; then
    warn "port :${PROXY_PORT} open but probe timed out — proxy is stuck; killing and relaunching"
    kill_stuck_proxy
  fi

  start_proxy
}

main "$@"
