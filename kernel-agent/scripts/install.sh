#!/usr/bin/env bash
# Kernel Agent installer.
#
# Base install is intentionally small and deterministic:
#   - ray[default]==2.44.1 + click<8.3.0
#   - Node.js/npm for claude/codex CLIs
#   - TraceLens editable install + CLI verification
#
# The installer prepares all kernel-agent backends in one pass.

set -euo pipefail

WORKSPACE_PATH="${WORKSPACE_PATH:-/workspace}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-${WORKSPACE_PATH}}"
KERNEL_AGENT_ROOT="${KERNEL_AGENT_ROOT:-${WORKSPACE_PATH}/kernel-agent}"
HYPERLOOM_KERNEL_AGENT_ROOT="${HYPERLOOM_KERNEL_AGENT_ROOT:-${KERNEL_AGENT_ROOT}}"
INFERENCE_OPTIMIZER_SESSION_DIR="${INFERENCE_OPTIMIZER_SESSION_DIR:-/workspace/hyperloom}"
HYPERLOOM_RUNTIME_DIR="${HYPERLOOM_RUNTIME_DIR:-${INFERENCE_OPTIMIZER_SESSION_DIR}/runtime}"
KERNEL_AGENT_ENV="${KERNEL_AGENT_ENV:-${HYPERLOOM_RUNTIME_DIR}/kernel-agent.env.sh}"
HYPERLOOM_ROOT="${HYPERLOOM_ROOT:-/opt/hyperloom}"
HYPERLOOM_BUNDLE="${HYPERLOOM_BUNDLE:-/wekafs/fully-local}"
MAGPIE_DIR="${MAGPIE_DIR:-${WORKSPACE_ROOT}/Magpie}"
# Resolve MAGPIE_PYTHON dynamically. The previous default
# ${MAGPIE_DIR}/venv/bin/python assumed a Magpie-private venv, but
# inference_optimizer/scripts/install.sh's ensure_magpie() does
# `pip install -e $MAGPIE_DIR` into the driver Python's site-packages
# (or the container image pre-installs it that way) — no venv is ever
# created at $MAGPIE_DIR/venv. Mirrors _resolve_magpie_python() in
# inference_optimizer/orchestrator/action_executors/_grid_runner.py:
#   $MAGPIE_PYTHON env > python3 on PATH that can `import Magpie`
#     > /opt/venv/bin/python (if it exists) > python3 on PATH.
_resolve_magpie_python() {
  if [ -n "${MAGPIE_PYTHON:-}" ]; then
    printf '%s' "$MAGPIE_PYTHON"
    return 0
  fi
  local candidate
  candidate="$(command -v python3 2>/dev/null || true)"
  if [ -n "$candidate" ] && "$candidate" -c "import Magpie" >/dev/null 2>&1; then
    printf '%s' "$candidate"
    return 0
  fi
  if [ -x /opt/venv/bin/python ]; then
    printf '%s' /opt/venv/bin/python
    return 0
  fi
  printf '%s' "${candidate:-/opt/venv/bin/python}"
}
MAGPIE_PYTHON="$(_resolve_magpie_python)"
PYTHONPATH="${MAGPIE_DIR}:${PYTHONPATH:-}"
INFERENCEX_PATH="${INFERENCEX_PATH:-}"
TRACELENS_ROOT="${TRACELENS_ROOT:-/wekafs/hyperloom/TraceLens-internal}"
# Writable mirror for TraceLens when $TRACELENS_ROOT is on a read-only mount
# (e.g. /wekafs/...). Mirrors the OOB pattern: cp -r the read-only source into
# ${HYPERLOOM_ROOT}/TraceLens-internal once, then pip install -e the mirror so
# editable build artifacts land on a writable filesystem. See ensure_tracelens.
TRACELENS_MIRROR_DIR="${TRACELENS_MIRROR_DIR:-${HYPERLOOM_ROOT}/TraceLens-internal}"

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
# Pin GEAK to the first release that ships RAG MCP retrieval and cross-session
# memory together. Keep this overridable so future GEAK fixes can move Hyperloom
# forward without reworking the installer contract.
GEAK_REF="${GEAK_REF:-v3.1.0}"
OOB_SRC="${OOB_SRC:-${HYPERLOOM_BUNDLE}/OOB}"
GEAK_CONFIG="${GEAK_CONFIG:-${HYPERLOOM_RUNTIME_DIR}/geak-config/local.yaml}"
# Pass GEAK_MODEL_NAME through unchanged; GEAK owns provider-specific routing.
GEAK_MODEL_NAME_VAL="${GEAK_MODEL_NAME:-claude-opus-4-7}"
RAG_INDEX_DIR="${HOME}/.cache/amd-ai-devtool/semantic-index"
GEAK_RAG_INDEX_DEVICE_VAL="${GEAK_RAG_INDEX_DEVICE:-cuda}"
GEAK_MEMORY_STORE_PATH_VAL="${GEAK_MEMORY_STORE_PATH:-/wekafs/hyperloom/geak-memory/memory.db}"
GEAK_SAVE_TO_KNOWLEDGE_BASE_VAL="${GEAK_SAVE_TO_KNOWLEDGE_BASE:-1}"
GEAK_MEMORY_MIN_SPEEDUP_VAL="${GEAK_MEMORY_MIN_SPEEDUP:-1.20}"
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
# Per user direction: "kernel-agent skills do not differentiate, just install everything". The
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
  Node.js/npm, GEAK CLI/config, OOB + claude/codex CLI auth,
  LLM proxy env/auth, and the OOB auth-proxy on :4002
  (via ensure_auth_proxy.sh).

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

ensure_node() {
  log "ensuring Node.js/npm for claude/codex CLIs"
  if command -v node >/dev/null 2>&1 && npm --version >/dev/null 2>&1; then
    log "node: $(command -v node) ($(node --version 2>/dev/null || echo unknown))"
    log "npm: $(command -v npm) ($(npm --version 2>/dev/null || echo unknown))"
    return 0
  fi

  if [ "$CHECK_ONLY" -eq 1 ]; then
    warn "node/npm missing; claude/codex CLI install would be skipped"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would install Node.js 20 from NodeSource because node/npm is missing"
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    verify_die "node/npm missing and apt-get is unavailable; install Node.js 20 manually"
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    log "curl missing; installing curl/ca-certificates before NodeSource setup"
    apt-get update >/dev/null
    apt-get -y install ca-certificates curl gnupg >/dev/null
  fi

  log "installing Node.js 20 from NodeSource"
  apt-get -y purge libnode-dev libnode72 nodejs nodejs-doc npm >/dev/null 2>&1 || true
  if ! curl -fsSL https://deb.nodesource.com/setup_20.x 2>/dev/null | bash - >/dev/null 2>&1; then
    verify_die "NodeSource setup failed; cannot install Node.js/npm for claude/codex CLIs"
    return 0
  fi
  if ! apt-get -y install nodejs >/dev/null; then
    verify_die "nodejs install failed; claude/codex CLI install cannot proceed"
    return 0
  fi
  command -v node >/dev/null 2>&1 || verify_die "node CLI not found after nodejs install"
  npm --version >/dev/null 2>&1 || verify_die "npm not usable after nodejs install"
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

# Idempotently bring up a Ray head node. Kernel backends submit Ray tasks with
# `num_gpus>=1`; if no head is running (or one is running with --num-gpus=0)
# kernel optimization will hang forever even when GPUs are idle. We:
#   1. detect a live Ray head via `ray status` (any successful return = live)
#   2. if absent, force-stop any half-started Ray and start a fresh head with
#      all visible GPUs advertised
#   3. tolerate the no-GPU case (CPU-only dev box) so `--check-only` stays
#      non-fatal in environments without ROCm
ensure_ray_started() {
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  if ! command -v ray >/dev/null 2>&1; then
    warn "ray CLI missing; cannot start ray head"
    return 0
  fi
  if ray status >/dev/null 2>&1; then
    log "ray head already running"
    return 0
  fi
  log "no live ray head detected; starting one"
  ray stop --force >/dev/null 2>&1 || true
  local num_gpus
  num_gpus="$(python3 - <<'PY' 2>/dev/null || echo 0
try:
    import torch
    print(torch.cuda.device_count() or 0)
except Exception:
    print(0)
PY
)"
  if [ "${RAY_NUM_GPUS:-}" != "" ]; then
    num_gpus="$RAY_NUM_GPUS"
  fi
  log "starting ray head with --num-gpus=${num_gpus}"
  if ! ray start --head --disable-usage-stats \
       --num-gpus="$num_gpus" --include-dashboard=false >/dev/null; then
    warn "ray start failed; kernel optimization will hang. Check ROCm visibility."
    return 0
  fi
  ray status >/dev/null 2>&1 || warn "ray status reports no live head after start"
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
  # Read-only source guard (mirrors the OOB cp -r pattern). When
  # $TRACELENS_ROOT is on a read-only mount (the WekaFS default), pip
  # install -e fails because it must write *.egg-info into the source
  # tree, and at runtime tools/tracelens_analysis.py re-runs the same
  # editable install in a subprocess on every select_kernels request,
  # producing a tight failure loop. Detecting unwritable source up front
  # and mirroring to ${HYPERLOOM_ROOT}/TraceLens-internal (parallel to
  # ${HYPERLOOM_ROOT}/geak / ${HYPERLOOM_ROOT}/OOB/oob_cli) lets both
  # the install-time and the runtime pip install land on a writable
  # filesystem. write_env_file() emits the resulting TRACELENS_ROOT into
  # the pod-local kernel-agent env so subsequent CLI subprocesses (setsid nohup inference_optimizer
  # optimize → kernel-agent → tracelens_analysis.py) inherit the mirror.
  if [ "$CHECK_ONLY" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    if ! ( : > "$TRACELENS_ROOT/.hl_write_test" ) 2>/dev/null; then
      log "TraceLens root not writable ($TRACELENS_ROOT); mirroring to $TRACELENS_MIRROR_DIR"
      mkdir -p "$(dirname "$TRACELENS_MIRROR_DIR")"
      if [ ! -d "$TRACELENS_MIRROR_DIR" ]; then
        run cp -r "$TRACELENS_ROOT" "$TRACELENS_MIRROR_DIR"
      else
        log "TraceLens mirror already present: $TRACELENS_MIRROR_DIR"
      fi
      TRACELENS_ROOT="$TRACELENS_MIRROR_DIR"
      export TRACELENS_ROOT
    else
      rm -f "$TRACELENS_ROOT/.hl_write_test"
    fi
  fi
  log "ensuring TraceLens CLI from $TRACELENS_ROOT"
  if [ "$CHECK_ONLY" -eq 0 ]; then
    run bash -lc "cd '$TRACELENS_ROOT' && python3 -m pip install -q --no-cache-dir -e ."
  fi
  if [ "$DRY_RUN" -eq 0 ]; then
    # TraceLens #124: prefer the inference variant (correct entry for
    # vLLM/SGLang traces). Fall back to the legacy CLI for older builds.
    if command -v TraceLens_generate_perf_report_pytorch_inference >/dev/null 2>&1; then
      TraceLens_generate_perf_report_pytorch_inference --help >/dev/null
      log "TraceLens perf CLI verified: TraceLens_generate_perf_report_pytorch_inference (#124)"
    elif command -v TraceLens_generate_perf_report_pytorch >/dev/null 2>&1; then
      TraceLens_generate_perf_report_pytorch --help >/dev/null
      warn "TraceLens_generate_perf_report_pytorch_inference not found; using legacy TraceLens_generate_perf_report_pytorch"
    else
      verify_die "Neither TraceLens_generate_perf_report_pytorch_inference nor TraceLens_generate_perf_report_pytorch found after install"
    fi
  fi
}

ensure_geak() {
  log "ensuring GEAK backend"
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    mkdir -p "${HYPERLOOM_ROOT}" "$(dirname "$GEAK_CONFIG")" "$(dirname "$GEAK_MEMORY_STORE_PATH_VAL")"
  fi
  if [ ! -d "${HYPERLOOM_ROOT}/geak/.git" ]; then
    run git clone --depth 1 --branch "$GEAK_REF" "$GEAK_REPO" "${HYPERLOOM_ROOT}/geak"
  else
    log "GEAK checkout already present: ${HYPERLOOM_ROOT}/geak"
  fi
  if [ "$CHECK_ONLY" -eq 0 ]; then
    run python3 -m pip install -q --no-cache-dir "${HYPERLOOM_ROOT}/geak"
    run python3 -m pip install -q --no-cache-dir "${HYPERLOOM_ROOT}/geak/mcp_tools/rag-mcp"
  else
    log "check-only: skipping GEAK and rag-mcp installation"
  fi
  if [ "$CHECK_ONLY" -eq 0 ]; then
    if [ "$DRY_RUN" -eq 0 ]; then
      if [ -z "$GEAK_API_KEY_VAL" ] || [ -z "$GEAK_BASE_URL_VAL" ]; then
        die "Cannot generate GEAK litellm config: SAFE_API_KEY and OPENAI_BASE_URL are required"
      fi
      cat > "$GEAK_CONFIG" <<EOF
model:
  model_class: litellm
  model_name: ${GEAK_MODEL_NAME_VAL}
  api_key: ${GEAK_API_KEY_VAL}
  base_url: ${GEAK_BASE_URL_VAL}
  model_kwargs:
    max_tokens: 16384
tools:
  rag: true
EOF
      chmod 600 "$GEAK_CONFIG"
      grep -Eq '^[[:space:]]*model_class:[[:space:]]*litellm[[:space:]]*$' "$GEAK_CONFIG" \
        || die "GEAK config must force model_class: litellm: $GEAK_CONFIG"
    else
      if [ -z "$GEAK_API_KEY_VAL" ] || [ -z "$GEAK_BASE_URL_VAL" ]; then
        warn "GEAK_API_KEY/GEAK_BASE_URL not fully set"
      fi
      log "would write GEAK config: $GEAK_CONFIG"
    fi
  else
    if [ "$DRY_RUN" -eq 0 ]; then
      if [ -z "$GEAK_API_KEY_VAL" ] || [ -z "$GEAK_BASE_URL_VAL" ]; then
        warn "GEAK_API_KEY/GEAK_BASE_URL not fully set"
      fi
      if [ -f "$GEAK_CONFIG" ]; then
        grep -Eq '^[[:space:]]*model_class:[[:space:]]*litellm[[:space:]]*$' "$GEAK_CONFIG" \
          || warn "GEAK config does not force model_class: litellm: $GEAK_CONFIG"
      else
        warn "GEAK config missing: $GEAK_CONFIG"
      fi
    fi
  fi
  if [ "$DRY_RUN" -eq 0 ]; then
    command -v geak >/dev/null 2>&1 || verify_die "geak CLI not found"
  fi
}

ensure_rag_index() {
  if [ -d "$RAG_INDEX_DIR" ] && [ -n "$(ls -A "$RAG_INDEX_DIR" 2>/dev/null)" ]; then
    log "RAG index already present at $RAG_INDEX_DIR"
    return
  fi
  if [ "$CHECK_ONLY" -eq 1 ]; then
    warn "RAG index missing at $RAG_INDEX_DIR"
    return
  fi
  log "building RAG index at $RAG_INDEX_DIR on device=${GEAK_RAG_INDEX_DEVICE_VAL} (first run downloads ~1.3 GB embedding model)"
  run bash -lc "cd '${HYPERLOOM_ROOT}/geak' && python3 scripts/build_index.py --force --device '${GEAK_RAG_INDEX_DEVICE_VAL}'"
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
      run python3 -m pip install -q --no-cache-dir "${HYPERLOOM_ROOT}/OOB/oob_cli"
    else
      warn "OOB source not found: $OOB_SRC"
    fi
  else
    log "oob already installed: $(command -v oob)"
  fi

  if ! command -v npm >/dev/null 2>&1; then
    if [ "$DRY_RUN" -eq 1 ]; then
      log "would install claude/codex npm CLIs after Node.js/npm is installed"
      ensure_llm_auth_files
      return 0
    fi
    verify_die "npm not found; ensure_node must run before ensure_oob"
    return 0
  fi
  if ! command -v claude >/dev/null 2>&1; then
    run npm config set prefix /usr/local
    run npm install -g @anthropic-ai/claude-code
  fi
  if ! command -v codex >/dev/null 2>&1; then
    run npm config set prefix /usr/local
    run npm install -g @openai/codex@0.100.0
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

# Write a pod-local kernel-agent env file users should source so subsequent CLI calls
# (and Ray workers via runtime_env) pick up the proxy-rewritten URLs.
write_env_file() {
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  # ensure_auth_proxy.sh now always emits PROXY_*_BASE_URL on success
  # (both the just-started and the healthy-noop branches). If we still don't
  # have them, the supervisor either failed or OOB_BASE_URL was empty —
  # either way the kernel-agent env would silently lack ANTHROPIC_BASE_URL/OPENAI_BASE_URL,
  # which is the exact failure mode that lets externally-preset upstream
  # URLs leak into Claude/Codex CLIs and 401-hang the SDK. Warn loudly so
  # the install operator notices instead of debugging at runtime.
  if [ -z "${PROXY_ANTHROPIC_BASE_URL:-}" ] || [ -z "${PROXY_OPENAI_BASE_URL:-}" ]; then
    warn "PROXY_*_BASE_URL not captured from ensure_auth_proxy.sh; kernel-agent env will lack ANTHROPIC_BASE_URL/OPENAI_BASE_URL"
    warn "This means an externally-preset ANTHROPIC_BASE_URL will reach Claude CLI directly and hang on gateway 401"
  fi
  local env_file="${KERNEL_AGENT_ENV}"
  mkdir -p "$(dirname "$env_file")"
  {
    echo '#!/bin/sh'
    echo "# kernel-agent runtime env (regenerated by install.sh)"
    [ -n "${INFERENCE_OPTIMIZER_SESSION_DIR:-}" ] && echo "export INFERENCE_OPTIMIZER_SESSION_DIR='${INFERENCE_OPTIMIZER_SESSION_DIR}'"
    [ -n "${HYPERLOOM_RUNTIME_DIR:-}" ] && echo "export HYPERLOOM_RUNTIME_DIR='${HYPERLOOM_RUNTIME_DIR}'"
    [ -n "${KERNEL_AGENT_ENV:-}" ] && echo "export KERNEL_AGENT_ENV='${KERNEL_AGENT_ENV}'"
    [ -n "${HYPERLOOM_KERNEL_AGENT_ROOT:-}" ] && echo "export HYPERLOOM_KERNEL_AGENT_ROOT='${HYPERLOOM_KERNEL_AGENT_ROOT}'"
    [ -n "${KERNEL_AGENT_ROOT:-}" ] && echo "export KERNEL_AGENT_ROOT='${KERNEL_AGENT_ROOT}'"
    [ -n "${WORKSPACE_ROOT:-}" ] && echo "export WORKSPACE_ROOT='${WORKSPACE_ROOT}'"
    [ -n "${WORKSPACE_PATH:-}" ] && echo "export WORKSPACE_PATH='${WORKSPACE_PATH}'"
    [ -n "${MAGPIE_DIR:-}" ] && echo "export MAGPIE_DIR='${MAGPIE_DIR}'"
    [ -n "${MAGPIE_PYTHON:-}" ] && echo "export MAGPIE_PYTHON='${MAGPIE_PYTHON}'"
    [ -n "${PYTHONPATH:-}" ] && echo "export PYTHONPATH='${PYTHONPATH}'"
    [ -n "${INFERENCEX_PATH:-}" ] && echo "export INFERENCEX_PATH='${INFERENCEX_PATH}'"
    [ -n "${PROXY_ANTHROPIC_BASE_URL:-}" ] && echo "export ANTHROPIC_BASE_URL='${PROXY_ANTHROPIC_BASE_URL}'"
    [ -n "${PROXY_OPENAI_BASE_URL:-}" ] && echo "export OPENAI_BASE_URL='${PROXY_OPENAI_BASE_URL}'"
    [ -n "${OOB_API_KEY_VAL}" ] && {
      echo "export SAFE_API_KEY='${OOB_API_KEY_VAL}'"
      echo "export ANTHROPIC_API_KEY='${OOB_API_KEY_VAL}'"
      echo "export ANTHROPIC_AUTH_TOKEN='${OOB_API_KEY_VAL}'"
      echo "export OPENAI_API_KEY='${OOB_API_KEY_VAL}'"
      echo "export OOB_API_KEY='${OOB_API_KEY_VAL}'"
      echo "export AMD_LLM_API_KEY='${OOB_API_KEY_VAL}'"
      echo "export LLM_GATEWAY_KEY='${OOB_API_KEY_VAL}'"
    }
    [ -n "${OOB_BASE_URL_VAL}" ] && echo "export OOB_BASE_URL='${OOB_BASE_URL_VAL}'"
    # Pin TRACELENS_ROOT to the (possibly mirrored) value resolved by
    # ensure_tracelens(). This is what lets setsid nohup inference_optimizer
    # optimize → kernel-agent/tools/tracelens_analysis.py inherit the writable
    # mirror instead of falling back to the read-only /wekafs default.
    [ -n "${TRACELENS_ROOT:-}" ] && echo "export TRACELENS_ROOT='${TRACELENS_ROOT}'"
    [ -n "${GEAK_CONFIG}" ] && echo "export GEAK_CONFIG='${GEAK_CONFIG}'"
    [ -n "${GEAK_MODEL_NAME_VAL}" ] && echo "export GEAK_MODEL_NAME='${GEAK_MODEL_NAME_VAL}'"
    [ -n "${GEAK_API_KEY_VAL}" ] && echo "export GEAK_API_KEY='${GEAK_API_KEY_VAL}'"
    [ -n "${GEAK_BASE_URL_VAL}" ] && echo "export GEAK_BASE_URL='${GEAK_BASE_URL_VAL}'"
    [ -n "${GEAK_MEMORY_STORE_PATH_VAL}" ] && echo "export GEAK_MEMORY_STORE_PATH='${GEAK_MEMORY_STORE_PATH_VAL}'"
    [ -n "${GEAK_SAVE_TO_KNOWLEDGE_BASE_VAL}" ] && echo "export GEAK_SAVE_TO_KNOWLEDGE_BASE='${GEAK_SAVE_TO_KNOWLEDGE_BASE_VAL}'"
    [ -n "${GEAK_MEMORY_MIN_SPEEDUP_VAL}" ] && echo "export GEAK_MEMORY_MIN_SPEEDUP='${GEAK_MEMORY_MIN_SPEEDUP_VAL}'"
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
  # TraceLens perf-report CLI: report whichever variant is available
  # (#124 prefers the _inference suffix; the legacy CLI is acceptable as
  # a fallback, the dispatcher in tools/tracelens_analysis.py picks at runtime).
  if command -v TraceLens_generate_perf_report_pytorch_inference >/dev/null 2>&1; then
    log "found TraceLens_generate_perf_report_pytorch_inference: $(command -v TraceLens_generate_perf_report_pytorch_inference)"
  elif command -v TraceLens_generate_perf_report_pytorch >/dev/null 2>&1; then
    warn "TraceLens_generate_perf_report_pytorch_inference not found; using legacy TraceLens_generate_perf_report_pytorch: $(command -v TraceLens_generate_perf_report_pytorch)"
  else
    warn "TraceLens perf-report CLI not found (looked for both _inference and legacy)"
  fi
  for tool in geak oob claude codex; do
    if command -v "$tool" >/dev/null 2>&1; then
      log "found ${tool}: $(command -v "$tool")"
    else
      warn "${tool} not found"
    fi
  done
  if [ -d "${HYPERLOOM_ROOT}/geak/.git" ]; then
    log "GEAK ref: $(git -C "${HYPERLOOM_ROOT}/geak" describe --tags --always 2>/dev/null || echo unknown)"
  else
    warn "GEAK checkout missing at ${HYPERLOOM_ROOT}/geak"
  fi
  if python3 -c "import rag_mcp" >/dev/null 2>&1; then
    log "rag-mcp installed: yes"
  else
    warn "rag-mcp not installed"
  fi
  if [ -d "$RAG_INDEX_DIR" ] && [ -n "$(ls -A "$RAG_INDEX_DIR" 2>/dev/null)" ]; then
    log "RAG index: present at $RAG_INDEX_DIR"
  else
    warn "RAG index missing at $RAG_INDEX_DIR"
  fi
  log "RAG index build device: ${GEAK_RAG_INDEX_DEVICE_VAL}"
  if grep -q "rag: true" "$GEAK_CONFIG" 2>/dev/null; then
    log "tools.rag enabled in $GEAK_CONFIG"
  else
    warn "tools.rag not enabled in $GEAK_CONFIG"
  fi
  log "GEAK memory store: ${GEAK_MEMORY_STORE_PATH_VAL}"
}

main() {
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    mkdir -p "${KERNEL_AGENT_ROOT}/runs"
  fi
  ensure_python
  ensure_node
  ensure_ray
  ensure_ray_started
  ensure_tracelens

  # Always install everything; ensure_oob also calls ensure_llm_auth_files.
  ensure_geak
  ensure_rag_index
  ensure_oob
  ensure_auth_proxy
  write_env_file

  report_status
  log "install complete"
}

main "$@"
