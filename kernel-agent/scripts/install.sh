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

# Credentials fallback: env always wins. If SAFE_API_KEY or OPENAI_BASE_URL
# is missing from env, source $REPO_ROOT/.env (defaults to $(pwd)) but
# protect any keys already set in env from being overwritten by .env.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
if [ -z "${SAFE_API_KEY:-}" ] || [ -z "${OPENAI_BASE_URL:-}" ]; then
  if [ -f "$REPO_ROOT/.env" ]; then
    _snap_safe="${SAFE_API_KEY-}"
    _snap_url="${OPENAI_BASE_URL-}"
    set -a
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.env"
    set +a
    [ -n "$_snap_safe" ] && export SAFE_API_KEY="$_snap_safe"
    [ -n "$_snap_url" ]  && export OPENAI_BASE_URL="$_snap_url"
    unset _snap_safe _snap_url
    echo "[kernel-agent] loaded credentials fallback from $REPO_ROOT/.env (env wins)"
  fi
fi
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
# GEAK's LitellmModel (feature/xiaofei/claw) auto-routes bare claude-* model
# names to OpenAI ChatCompletion when api_base contains llm-proxy/openai. So we
# pass GEAK_MODEL_NAME through unchanged; do NOT prepend openai/ here.
GEAK_MODEL_NAME_VAL="${GEAK_MODEL_NAME:-claude-opus-4-7}"
# GEAK/OOB use the user's LiteLLM-compatible endpoint. The canonical env is
# OPENAI_BASE_URL + SAFE_API_KEY; keep fallbacks for older launchers.
GEAK_API_KEY_VAL="${GEAK_API_KEY:-${SAFE_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-${AMD_API_KEY:-${AMD_LLM_API_KEY:-${LLM_API_KEY:-${OPENAI_API_KEY:-}}}}}}}"
GEAK_BASE_URL_VAL="${GEAK_BASE_URL:-${OPENAI_BASE_URL:-${ANTHROPIC_BASE_URL:-${LLM_API_BASE:-}}}}"
OOB_API_KEY_VAL="${OOB_API_KEY:-${SAFE_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-${ANTHROPIC_API_KEY:-${OPENAI_API_KEY:-}}}}}"
OOB_BASE_URL_VAL="${OOB_BASE_URL:-${OPENAI_BASE_URL:-${ANTHROPIC_BASE_URL:-}}}"

# Install everything by default. The previous lazy `--with-geak / --with-oob`
# scheme caused recurring "OOB proxy not running, request errored, found
# the missing service after the fact" issues — when the resident skill
# triggered a kernel-opt that needed claude/codex but install.sh had only
# brought up GEAK, the auth-proxy was missing and every CLI request 401'd.
# Per user direction: "kernel-agent skills 不区别别的, 直接全部安装". The
# old --with-* / --all-backends / --backend flags are accepted but no-op
# for backwards compatibility with existing call sites.
WITH_GEAK=1
WITH_OOB=1
WITH_LLM=1
CHECK_ONLY=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: install.sh [options]

Always installs (no --with-* selectivity any more):
  ray[default]==2.44.1, click<8.3.0, TraceLens CLI,
  GEAK CLI/config, OOB + claude/codex CLI auth, LLM proxy env/auth,
  and the OOB auth-proxy on :4002 (via ensure_auth_proxy.sh).

Options:
  --check-only       Verify current environment, do not install
  --dry-run          Print actions without running installs
  -h, --help         Show this help

Legacy options (accepted but no-op, kept for backwards compat):
  --with-geak / --with-oob / --with-llm / --all-backends / --backend NAME
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --with-geak|--with-oob|--with-llm|--all-backends)
      # No-op: install.sh always installs everything now. Accepted for
      # backwards compat with older call sites / docs.
      ;;
    --backend)
      shift
      case "${1:-}" in
        geak|oob|llm) ;;  # no-op, see above
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
# In --check-only mode, downgrade post-install verification failures to a
# warning so report_status can still enumerate what's missing. The caller
# explicitly asked us NOT to install; failing on the first missing piece
# defeats the point of check-only.
verify_die() {
  if [ "$CHECK_ONLY" -eq 1 ]; then warn "$1"; else die "$1"; fi
}

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
    if [ "$DRY_RUN" -eq 1 ] || [ "$CHECK_ONLY" -eq 1 ]; then
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
    if command -v TraceLens_generate_perf_report_pytorch >/dev/null 2>&1; then
      TraceLens_generate_perf_report_pytorch --help >/dev/null
    else
      verify_die "TraceLens_generate_perf_report_pytorch not found after install"
    fi
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
    command -v geak >/dev/null 2>&1 || verify_die "geak CLI not found"
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

# Delegate to the standalone idempotent supervisor script. It handles the
# port probe + curl probe + stuck-proxy restart cases. Sourcing it lets us
# pick up its PROXY_*_BASE_URL exports for write_env_file.
ensure_auth_proxy() {
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  local script
  script="$(dirname "$0")/ensure_auth_proxy.sh"
  if [ ! -f "$script" ]; then
    warn "ensure_auth_proxy.sh not found at $script"
    return 0
  fi
  # Re-export the env vars the helper expects, then capture its KEY=VALUE
  # output and source it back so PROXY_*_BASE_URL land in this shell.
  local out
  if ! out=$(
    HYPERLOOM_ROOT="$HYPERLOOM_ROOT" \
    OOB_BASE_URL="$OOB_BASE_URL_VAL" \
    OOB_API_KEY="$OOB_API_KEY_VAL" \
    bash "$script" 2>&1
  ); then
    warn "ensure_auth_proxy.sh failed; OOB requests may 401"
    echo "$out" | sed 's/^/  /' >&2
    return 0
  fi
  echo "$out" | sed 's/^/[ensure-auth-proxy] /'
  while IFS='=' read -r key value; do
    case "$key" in
      PROXY_ANTHROPIC_BASE_URL) PROXY_ANTHROPIC_BASE_URL="$value" ;;
      PROXY_OPENAI_BASE_URL)    PROXY_OPENAI_BASE_URL="$value" ;;
    esac
  done <<<"$out"
}

# Write a kernel-agent env file users should source so subsequent CLI calls
# (and Ray workers via runtime_env) pick up the proxy-rewritten URLs.
write_env_file() {
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  # ensure_auth_proxy.sh now always emits PROXY_*_BASE_URL on success
  # (both the just-started and the healthy-noop branches). If we still don't
  # have them, the supervisor either failed or OOB_BASE_URL was empty —
  # either way env.sh would silently lack ANTHROPIC_BASE_URL/OPENAI_BASE_URL,
  # which is the exact failure mode that lets externally-preset upstream
  # URLs leak into Claude/Codex CLIs and 401-hang the SDK. Warn loudly so
  # the install operator notices instead of debugging at runtime.
  if [ -z "${PROXY_ANTHROPIC_BASE_URL:-}" ] || [ -z "${PROXY_OPENAI_BASE_URL:-}" ]; then
    warn "PROXY_*_BASE_URL not captured from ensure_auth_proxy.sh; env.sh will lack ANTHROPIC_BASE_URL/OPENAI_BASE_URL"
    warn "This means an externally-preset ANTHROPIC_BASE_URL will reach Claude CLI directly and hang on gateway 401"
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

  # Always install everything; ensure_oob also calls ensure_llm_auth_files.
  ensure_geak
  ensure_oob
  ensure_auth_proxy
  write_env_file

  report_status
  log "install complete"
}

main "$@"
