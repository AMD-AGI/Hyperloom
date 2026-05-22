#!/usr/bin/env bash
# kernel-agent OOB auth-proxy supervisor.
#
# The auth-proxy (auth_proxy.py from OOB) listens on 127.0.0.1:4010 by
# default (:4002 is often occupied by dfdaemon on shared dev hosts).
# rewrites the agent CLIs' x-api-key header into Authorization: Bearer for
# the AMD LLM gateway. Without it, claude/codex CLI requests get HTTP 401
# "token not present" because the gateway's auth is non-standard.
#
# This script is idempotent and meant to be sourced or invoked before any
# OOB-dependent kernel-agent tool runs:
#
#   bash $REPO_ROOT/kernel-agent/scripts/ensure_auth_proxy.sh
#
# Behaviour:
#   * If the proxy forwards GET <upstream>/models with Bearer auth (HTTP 200),
#     treat it as healthy and noop.
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

# HYPERLOOM_ROOT defaults to the writable source-mirrors location under
# $USER_DATA_PATH/runtime/source-mirrors (set by install.sh). The auth-proxy
# log lands next to it for unified monitoring. Operators may still pin
# HYPERLOOM_ROOT manually if they want the proxy script + log to live
# elsewhere (e.g. legacy /opt/hyperloom deployments).
USER_DATA_PATH="${USER_DATA_PATH:-/workspace/hyperloom}"
HYPERLOOM_RUNTIME_DIR="${HYPERLOOM_RUNTIME_DIR:-${USER_DATA_PATH}/runtime}"
HYPERLOOM_ROOT="${HYPERLOOM_ROOT:-${HYPERLOOM_RUNTIME_DIR}/source-mirrors}"
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_HYPERLOOM_REPO="$(cd "${_SCRIPT_DIR}/../.." && pwd)"
PROXY_PY="${PROXY_PY:-${HYPERLOOM_ROOT}/OOB/oob_cli/auth_proxy.py}"
if [ ! -f "$PROXY_PY" ]; then
  PROXY_PY="${HYPERLOOM_ROOT}/OOB/auth_proxy.py"
fi
if [ ! -f "$PROXY_PY" ]; then
  PROXY_PY="${_HYPERLOOM_REPO}/OOB/auth_proxy.py"
fi
PROXY_PORT="${AUTH_PROXY_PORT:-4010}"
# Auth-proxy stdout/stderr lands under $USER_DATA_PATH/logs/auth-proxy/
# by default so a single $USER_DATA_PATH tail covers it. Override
# AUTH_PROXY_LOG_DIR if you want the legacy ${HYPERLOOM_ROOT}/logs location.
LOG_DIR="${AUTH_PROXY_LOG_DIR:-${USER_DATA_PATH}/logs/auth-proxy}"
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

# Returns 0 when the listener forwards LLM catalog traffic (HTTP 200 on
# GET .../models with Bearer). Empty 404 on :4002 usually means dfdaemon,
# not auth_proxy — do not treat that as healthy.
proxy_forwards_llm() {
  if ! command -v curl >/dev/null 2>&1; then
    port_open
    return $?
  fi
  if [ -z "$OOB_BASE_URL_VAL" ] || [ -z "$OOB_API_KEY_VAL" ]; then
    return 1
  fi
  derive_proxy_urls
  local models_url="${PROXY_OPENAI_BASE_URL%/}/models"
  local http_code=""
  http_code=$(curl -sk -o /dev/null -w '%{http_code}' \
    --max-time "$PROBE_TIMEOUT" \
    -H "Authorization: Bearer ${OOB_API_KEY_VAL}" \
    "$models_url" 2>/dev/null) || true
  [ "$http_code" = "200" ]
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

# Derive the proxy-side URLs from $OOB_BASE_URL_VAL. Sets the global vars
# PROXY_ANTHROPIC_BASE_URL / PROXY_OPENAI_BASE_URL. Idempotent; safe to call
# from both the healthy-noop and the just-started branches of main(). The
# Anthropic SDK appends "/v1" itself, so the Anthropic URL strips a trailing
# "/v1" while the OpenAI URL keeps the path verbatim.
derive_proxy_urls() {
  local path
  path=$(echo "$OOB_BASE_URL_VAL" | grep -oP '(?<=://)[^/]+(/.+)' | grep -oP '/.*' || true)
  PROXY_ANTHROPIC_BASE_URL="http://127.0.0.1:${PROXY_PORT}$(echo "$path" | sed 's|/v1$||')"
  PROXY_OPENAI_BASE_URL="http://127.0.0.1:${PROXY_PORT}${path}"
}

# Print the PROXY_*_BASE_URL key=value lines for callers that source/parse
# this script's stdout (install.sh's write_env_file does this). Returns
# non-zero if OOB_BASE_URL_VAL is empty so install.sh can warn loudly
# instead of silently emitting an env.sh missing the proxy URLs.
emit_proxy_urls() {
  if [ -z "$OOB_BASE_URL_VAL" ]; then
    return 1
  fi
  derive_proxy_urls
  echo "PROXY_ANTHROPIC_BASE_URL=${PROXY_ANTHROPIC_BASE_URL}"
  echo "PROXY_OPENAI_BASE_URL=${PROXY_OPENAI_BASE_URL}"
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
  local scheme host parsed_port port_target
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

  derive_proxy_urls

  log "starting auth-proxy on :${PROXY_PORT} -> ${scheme}://${host}:${port_target}"
  mkdir -p "$LOG_DIR"
  LLM_PROXY_SCHEME="$scheme" \
  LLM_PROXY_HOST="$host" \
  LLM_PROXY_PORT="$port_target" \
  AUTH_PROXY_PORT="$PROXY_PORT" \
  PROXY_AUTH_TOKEN="$OOB_API_KEY_VAL" \
    nohup python3 "$PROXY_PY" >"${LOG_DIR}/oob-auth-proxy.log" 2>&1 &

  local elapsed=0
  while [ "$elapsed" -lt "$START_WAIT_SEC" ]; do
    if proxy_forwards_llm; then
      log "auth-proxy forwarding on :${PROXY_PORT} after ${elapsed}s"
      emit_proxy_urls
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  warn "auth-proxy did not bind :${PROXY_PORT} within ${START_WAIT_SEC}s; see ${LOG_DIR}/oob-auth-proxy.log"
  return 1
}

main() {
  if proxy_forwards_llm; then
    log "auth-proxy already healthy on :${PROXY_PORT} (GET /models -> 200)"
    # Always emit PROXY_*_BASE_URL on the healthy-noop path too — install.sh
    # parses our stdout to populate the pod-local kernel-agent env. Without this, that env
    # silently lacks ANTHROPIC_BASE_URL/OPENAI_BASE_URL whenever the proxy
    # was already healthy at install time, and externally-preset upstream
    # URLs leak into Claude/Codex CLIs.
    emit_proxy_urls || warn "OOB_BASE_URL not set; PROXY_*_BASE_URL not emitted"
    return 0
  fi

  if port_open; then
    if [ "$PROXY_PORT" = "4002" ]; then
      warn "port :4002 is open but does not forward /models (often dfdaemon, not auth_proxy); use AUTH_PROXY_PORT=4010"
    else
      warn "port :${PROXY_PORT} open but GET /models != 200 — killing auth_proxy.py and relaunching"
    fi
    kill_stuck_proxy
  fi

  start_proxy
}

main "$@"
