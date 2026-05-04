#!/usr/bin/env bash
# Kernel Agent installer.
#
# Base install is intentionally small and deterministic:
#   - ray[default]==2.44.1 + click<8.3.0
#   - TraceLens editable install + CLI verification
#
# Backends are lazy: install only what a request needs, or use --all-backends.

set -euo pipefail

WORKSPACE_PATH="${WORKSPACE_PATH:-/workspace}"
KERNEL_AGENT_ROOT="${KERNEL_AGENT_ROOT:-${WORKSPACE_PATH}/kernel-agent}"
HYPERLOOM_ROOT="${HYPERLOOM_ROOT:-/opt/hyperloom}"
HYPERLOOM_BUNDLE="${HYPERLOOM_BUNDLE:-/wekafs/fully-local}"
TRACELENS_ROOT="${TRACELENS_ROOT:-/wekafs/hyperloom/TraceLens-internal}"
GEAK_REPO="${GEAK_REPO:-https://github.com/AMD-AGI/GEAK.git}"
# Default to the LitellmModel-fixed branch. Upstream main works ONLY when the
# model is reached via amd_llm + AMD LLM Gateway (anthropic SDK direct). On
# the AMD primus-safe OpenAI-compat proxy that we have here, main's litellm
# path does not normalize tool-call args and returns Python repr (single
# quotes) for `submit({summary: [...]}`), making _parse_llm_response crash
# with "Expecting property name in double quotes char 2". The PR branch
# `feature/xiaofei/claw` adds use_amd_openai_compatible_litellm_route /
# litellm_model_name_for_completion / format_messages_openai_chat which
# normalize the OpenAI-compat path. Verified end-to-end on r36.
GEAK_BRANCH="${GEAK_BRANCH:-feature/xiaofei/claw}"
OOB_SRC="${OOB_SRC:-${HYPERLOOM_BUNDLE}/OOB}"
GEAK_CONFIG="${GEAK_CONFIG:-${HYPERLOOM_ROOT}/geak-config/local.yaml}"
# litellm picks the protocol from the model_name prefix:
#  - bare 'claude-...' -> Anthropic /v1/messages route (gateway rejects with 401)
#  - 'openai/...'      -> OpenAI /v1/chat/completions route (gateway accepts)
# The AMD gateway only speaks OpenAI chat-completions, so we force the prefix.
_GEAK_RAW_MODEL="${GEAK_MODEL_NAME:-claude-opus-4-7}"
case "${_GEAK_RAW_MODEL}" in
  openai/*|anthropic/*|bedrock/*|azure/*|vertex_ai/*) GEAK_MODEL_NAME_VAL="${_GEAK_RAW_MODEL}" ;;
  *) GEAK_MODEL_NAME_VAL="openai/${_GEAK_RAW_MODEL}" ;;
esac
# GEAK/OOB use the user's LiteLLM-compatible endpoint. The canonical env is
# OPENAI_BASE_URL + SAFE_API_KEY; keep fallbacks for older launchers.
GEAK_API_KEY_VAL="${GEAK_API_KEY:-${SAFE_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-${AMD_API_KEY:-${AMD_LLM_API_KEY:-${LLM_API_KEY:-${OPENAI_API_KEY:-}}}}}}}"
GEAK_BASE_URL_VAL="${GEAK_BASE_URL:-${OPENAI_BASE_URL:-${ANTHROPIC_BASE_URL:-${LLM_API_BASE:-}}}}"
OOB_API_KEY_VAL="${OOB_API_KEY:-${SAFE_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-${ANTHROPIC_API_KEY:-${OPENAI_API_KEY:-}}}}}"
OOB_BASE_URL_VAL="${OOB_BASE_URL:-${OPENAI_BASE_URL:-${ANTHROPIC_BASE_URL:-}}}"

WITH_GEAK=0
WITH_OOB=0
WITH_LLM=0
CHECK_ONLY=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: install.sh [options]

Base install always ensures:
  ray[default]==2.44.1, click<8.3.0, TraceLens CLI

Options:
  --with-geak        Install GEAK CLI/config only
  --with-oob         Install OOB + claude/codex CLI auth only
  --with-llm         Write LLM proxy env/auth only
  --all-backends     Install GEAK + OOB + LLM pieces
  --backend NAME     Install one backend: geak | oob | llm
  --check-only       Verify current environment, do not install
  --dry-run          Print actions without running installs
  -h, --help         Show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --with-geak) WITH_GEAK=1 ;;
    --with-oob) WITH_OOB=1 ;;
    --with-llm) WITH_LLM=1 ;;
    --all-backends) WITH_GEAK=1; WITH_OOB=1; WITH_LLM=1 ;;
    --backend)
      shift
      case "${1:-}" in
        geak) WITH_GEAK=1 ;;
        oob) WITH_OOB=1 ;;
        llm) WITH_LLM=1 ;;
        *) echo "[kernel-agent] ERROR: unknown backend '${1:-}'" >&2; exit 2 ;;
      esac
      ;;
    --check-only) CHECK_ONLY=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[kernel-agent] ERROR: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

log() { echo "[kernel-agent] $*"; }
warn() { echo "[kernel-agent WARN] $*" >&2; }
die() { echo "[kernel-agent ERROR] $*" >&2; exit 1; }

run() {
  log "$*"
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    "$@"
  fi
}

ensure_python() {
  python3 --version >/dev/null || die "python3 is required"
  python3 -m pip --version >/dev/null || die "pip is required"
}

ensure_ray() {
  log "ensuring ray[default]==2.44.1 and click<8.3.0"
  if [ "$CHECK_ONLY" -eq 0 ]; then
    run python3 -m pip install --quiet --no-cache-dir "click<8.3.0" "ray[default]==2.44.1"
  fi
  if [ "$DRY_RUN" -eq 0 ]; then
    python3 - <<'PY'
import ray, sys
if ray.__version__ != "2.44.1":
    raise SystemExit(f"ray version mismatch: {ray.__version__} != 2.44.1")
print(f"[kernel-agent] ray version: {ray.__version__}")
PY
  fi
}

ensure_tracelens() {
  if [ ! -d "$TRACELENS_ROOT" ] && [ -d "${HYPERLOOM_BUNDLE}/TraceLens-internal" ]; then
    TRACELENS_ROOT="${HYPERLOOM_BUNDLE}/TraceLens-internal"
  fi
  if [ ! -d "$TRACELENS_ROOT" ]; then
    if [ "$DRY_RUN" -eq 1 ]; then
      warn "TraceLens root not found: $TRACELENS_ROOT"
      return
    fi
    die "TraceLens root not found: $TRACELENS_ROOT"
  fi
  log "ensuring TraceLens CLI from $TRACELENS_ROOT"
  if [ "$CHECK_ONLY" -eq 0 ]; then
    run bash -lc "cd '$TRACELENS_ROOT' && python3 -m pip install -q --no-cache-dir -e ."
  fi
  if [ "$DRY_RUN" -eq 0 ]; then
    command -v TraceLens_generate_perf_report_pytorch >/dev/null 2>&1 \
      || die "TraceLens_generate_perf_report_pytorch not found after install"
    TraceLens_generate_perf_report_pytorch --help >/dev/null
  fi
}

ensure_geak() {
  log "ensuring GEAK backend"
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    mkdir -p "${HYPERLOOM_ROOT}" "$(dirname "$GEAK_CONFIG")"
  fi
  if ! command -v geak >/dev/null 2>&1; then
    if [ ! -d "${HYPERLOOM_ROOT}/geak/.git" ]; then
      run git clone --depth 1 -b "$GEAK_BRANCH" "$GEAK_REPO" "${HYPERLOOM_ROOT}/geak"
    fi
    run python3 -m pip install -q --no-cache-dir -e "${HYPERLOOM_ROOT}/geak"
  else
    log "geak already installed: $(command -v geak)"
  fi
  if [ "$CHECK_ONLY" -eq 0 ]; then
    if [ -z "$GEAK_API_KEY_VAL" ] || [ -z "$GEAK_BASE_URL_VAL" ]; then
      warn "GEAK_API_KEY/GEAK_BASE_URL not fully set; writing config with current values"
    fi
    if [ "$DRY_RUN" -eq 0 ]; then
      cat > "$GEAK_CONFIG" <<EOF
model:
  model_class: litellm
  model_name: ${GEAK_MODEL_NAME_VAL}
  api_key: ${GEAK_API_KEY_VAL}
  base_url: ${GEAK_BASE_URL_VAL}
  model_kwargs:
    max_tokens: 16384
EOF
      chmod 600 "$GEAK_CONFIG"
    else
      log "would write GEAK config: $GEAK_CONFIG"
    fi
  fi
  if [ "$DRY_RUN" -eq 0 ]; then
    command -v geak >/dev/null 2>&1 || die "geak CLI not found"
  fi
}

ensure_oob() {
  log "ensuring OOB backend"
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    mkdir -p "${HYPERLOOM_ROOT}/OOB"
  fi
  if ! command -v oob >/dev/null 2>&1; then
    if [ -d "$OOB_SRC" ]; then
      if [ ! -d "${HYPERLOOM_ROOT}/OOB/oob_cli" ]; then
        run cp -r "$OOB_SRC" "${HYPERLOOM_ROOT}/OOB/oob_cli"
      fi
      if [ -f "${HYPERLOOM_ROOT}/OOB/oob_cli/requirements.txt" ]; then
        run python3 -m pip install -q --no-cache-dir -r "${HYPERLOOM_ROOT}/OOB/oob_cli/requirements.txt"
      fi
      run python3 -m pip install -q --no-cache-dir -e "${HYPERLOOM_ROOT}/OOB/oob_cli"
    else
      warn "OOB source not found: $OOB_SRC"
    fi
  else
    log "oob already installed: $(command -v oob)"
  fi

  if command -v node >/dev/null 2>&1; then
    if ! npm --version >/dev/null 2>&1; then
      log "system npm broken; reinstalling nodejs 20 from nodesource"
      if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
        apt-get -y purge libnode-dev libnode72 nodejs nodejs-doc npm >/dev/null 2>&1 || true
        curl -fsSL https://deb.nodesource.com/setup_20.x 2>/dev/null | bash - >/dev/null 2>&1 \
          || warn "nodesource setup failed"
        apt-get -y install nodejs >/dev/null 2>&1 || warn "nodejs install failed"
      fi
    fi
    if command -v npm >/dev/null 2>&1 && ! command -v claude >/dev/null 2>&1; then
      run npm config set prefix /usr/local
      run npm install -g @anthropic-ai/claude-code
    fi
    if command -v npm >/dev/null 2>&1 && ! command -v codex >/dev/null 2>&1; then
      run npm config set prefix /usr/local
      run npm install -g @openai/codex@0.100.0
    fi
  else
    warn "node not found; skipping claude/codex npm CLI install"
  fi

  ensure_llm_auth_files
}

ensure_llm_auth_files() {
  log "ensuring LLM/OOB auth files"
  if [ "$CHECK_ONLY" -eq 1 ]; then
    return
  fi
  if [ -z "$OOB_API_KEY_VAL" ]; then
    warn "OOB/Anthropic API key not set; auth files not written"
    return
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would write /root/.claude/config.json and /root/.codex/auth.json"
    return
  fi
  mkdir -p /root/.claude /root/.codex
  cat > /root/.claude/config.json <<EOF
{
  "theme": "dark",
  "hasCompletedOnboarding": true,
  "primaryApiKey": "${OOB_API_KEY_VAL}",
  "customApiUrl": "${OOB_BASE_URL_VAL}"
}
EOF
  chmod 600 /root/.claude/config.json
  cat > /root/.codex/auth.json <<EOF
{
  "auth_mode": "apikey",
  "OPENAI_API_KEY": "${OOB_API_KEY_VAL}"
}
EOF
  chmod 600 /root/.codex/auth.json
}

# Start the OOB auth-proxy on :4002 that rewrites Bearer headers for the AMD
# LLM gateway. Without this proxy, claude/codex CLI requests get 401
# "token not present" because the gateway's auth is non-standard.
ensure_auth_proxy() {
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  if [ -z "$OOB_BASE_URL_VAL" ]; then
    warn "OOB_BASE_URL not set; cannot start auth-proxy"
    return 0
  fi
  local proxy_py="${HYPERLOOM_ROOT}/OOB/oob_cli/auth_proxy.py"
  if [ ! -f "$proxy_py" ]; then
    warn "auth_proxy.py not found at $proxy_py; OOB Bearer rewrite skipped"
    return 0
  fi
  local port=4002
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

  if ss -tlnp 2>/dev/null | grep -q ":${port} "; then
    log "OOB auth-proxy already listening on :${port}"
  else
    log "starting OOB auth-proxy on :${port} -> ${scheme}://${host}:${port_target}${path}"
    mkdir -p "${HYPERLOOM_ROOT}/logs"
    LLM_PROXY_SCHEME="$scheme" LLM_PROXY_HOST="$host" LLM_PROXY_PORT="$port_target" \
      AUTH_PROXY_PORT="$port" PROXY_AUTH_TOKEN="$OOB_API_KEY_VAL" \
      nohup python3 "$proxy_py" >"${HYPERLOOM_ROOT}/logs/oob-auth-proxy.log" 2>&1 &
    sleep 2
    if ! ss -tlnp 2>/dev/null | grep -q ":${port} "; then
      warn "auth-proxy did not bind :${port}; see ${HYPERLOOM_ROOT}/logs/oob-auth-proxy.log"
      return 0
    fi
  fi
  PROXY_ANTHROPIC_BASE_URL="http://127.0.0.1:${port}$(echo "$path" | sed 's|/v1$||')"
  PROXY_OPENAI_BASE_URL="http://127.0.0.1:${port}${path}"
  log "auth-proxy ready: ANTHROPIC_BASE_URL=${PROXY_ANTHROPIC_BASE_URL}"
  log "auth-proxy ready: OPENAI_BASE_URL=${PROXY_OPENAI_BASE_URL}"
}

# Write a kernel-agent env file users should source so subsequent CLI calls
# (and Ray workers via runtime_env) pick up the proxy-rewritten URLs.
write_env_file() {
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  local env_file="${KERNEL_AGENT_ROOT}/env.sh"
  mkdir -p "${KERNEL_AGENT_ROOT}"
  {
    echo '#!/bin/sh'
    echo "# kernel-agent runtime env (regenerated by install.sh)"
    [ -n "${PROXY_ANTHROPIC_BASE_URL:-}" ] && echo "export ANTHROPIC_BASE_URL='${PROXY_ANTHROPIC_BASE_URL}'"
    [ -n "${PROXY_OPENAI_BASE_URL:-}" ] && echo "export OPENAI_BASE_URL='${PROXY_OPENAI_BASE_URL}'"
    [ -n "${OOB_API_KEY_VAL}" ] && {
      echo "export SAFE_API_KEY='${OOB_API_KEY_VAL}'"
      echo "export ANTHROPIC_API_KEY='${OOB_API_KEY_VAL}'"
      echo "export ANTHROPIC_AUTH_TOKEN='${OOB_API_KEY_VAL}'"
      echo "export OPENAI_API_KEY='${OOB_API_KEY_VAL}'"
      echo "export OOB_API_KEY='${OOB_API_KEY_VAL}'"
    }
    [ -n "${OOB_BASE_URL_VAL}" ] && echo "export OOB_BASE_URL='${OOB_BASE_URL_VAL}'"
    [ -n "${GEAK_CONFIG}" ] && echo "export GEAK_CONFIG='${GEAK_CONFIG}'"
    [ -n "${GEAK_MODEL_NAME_VAL}" ] && echo "export GEAK_MODEL_NAME='${GEAK_MODEL_NAME_VAL}'"
    [ -n "${GEAK_API_KEY_VAL}" ] && echo "export GEAK_API_KEY='${GEAK_API_KEY_VAL}'"
    [ -n "${GEAK_BASE_URL_VAL}" ] && echo "export GEAK_BASE_URL='${GEAK_BASE_URL_VAL}'"
  } > "$env_file"
  chmod 600 "$env_file"
  log "wrote ${env_file} (source it before running kernel-agent tools)"
}

report_status() {
  log "root: ${KERNEL_AGENT_ROOT}"
  log "ray: $(python3 - <<'PY' 2>/dev/null || echo missing
try:
    import ray
    print(ray.__version__)
except Exception:
    raise SystemExit(1)
PY
)"
  for tool in TraceLens_generate_perf_report_pytorch geak oob claude codex; do
    if command -v "$tool" >/dev/null 2>&1; then
      log "found ${tool}: $(command -v "$tool")"
    else
      warn "${tool} not found"
    fi
  done
}

main() {
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    mkdir -p "${KERNEL_AGENT_ROOT}/runs"
  fi
  ensure_python
  ensure_ray
  ensure_tracelens

  [ "$WITH_GEAK" -eq 1 ] && ensure_geak
  [ "$WITH_OOB" -eq 1 ] && ensure_oob
  [ "$WITH_LLM" -eq 1 ] && ensure_llm_auth_files
  if [ "$WITH_OOB" -eq 1 ] || [ "$WITH_LLM" -eq 1 ]; then
    ensure_auth_proxy
  fi
  write_env_file

  report_status
  log "install complete"
}

main "$@"
