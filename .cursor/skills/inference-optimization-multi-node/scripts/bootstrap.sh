#!/bin/bash
# Hyperloom BYOI Bootstrap (Method B)
# Installs Hyperloom remote RayJob dependencies into a user-provided image.
# Design: deploy/local/DESIGN.md section 7
# Idempotent: safe to re-run; per-step skip checks short-circuit completed work.

set -uo pipefail

# ---------- logging ----------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${BLUE}[bootstrap]${NC} $*"; }
ok()   { echo -e "${GREEN}[bootstrap]${NC} $*"; }
warn() { echo -e "${YELLOW}[bootstrap WARN]${NC} $*"; }
err()  { echo -e "${RED}[bootstrap ERROR]${NC} $*" >&2; }

# ---------- config (env-overridable, see deploy/local/DESIGN.md) ----------
HYPERLOOM_BUNDLE="${HYPERLOOM_BUNDLE:-/wekafs/fully-local}"
HYPERLOOM_ROOT="${HYPERLOOM_ROOT:-/opt/hyperloom}"
INTELLIKIT_REPO="${INTELLIKIT_REPO:-https://github.com/AMDResearch/intellikit.git}"
INTELLIKIT_SHA="${INTELLIKIT_SHA:-bcbfa0252df9d55f3aab68c95dd3ce45ccbe5b46}"
LOG_DIR="${LOG_DIR:-/var/log/hyperloom}"
AGENT_WORKSPACE_ROOT="${AGENT_WORKSPACE_ROOT:-/tmp/agent-workspaces}"
SKILL_ROOT_DEFAULT="$HYPERLOOM_ROOT/.cursor/skills/inference-optimization"
DONE_MARKER="$HYPERLOOM_ROOT/.bootstrap_done"
ENV_FILE="/etc/profile.d/hyperloom-env.sh"

FRAMEWORK="${FRAMEWORK:-sglang}"

# ---------- short-circuits ----------
if [ "${SKIP_BOOTSTRAP:-0}" = "1" ]; then
    warn "SKIP_BOOTSTRAP=1 set, exiting without changes"
    exit 0
fi

FORCE=0
if [ "${1:-}" = "--force" ]; then
    FORCE=1
fi

if [ -f "$DONE_MARKER" ] && [ "$FORCE" -eq 0 ]; then
    ok "bootstrap already completed (marker: $DONE_MARKER)"
    ok "use '$0 --force' to re-run"
    exit 0
fi

mkdir -p "$LOG_DIR" "$AGENT_WORKSPACE_ROOT" "$HYPERLOOM_ROOT/.cursor/skills"

# =====================================================================
# Step 1 — hard requirements (no auto-install; fail fast on missing)
# =====================================================================
log "Step 1/6: probing hard requirements"
HARD_FAIL=0

if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" 2>/dev/null; then
    err "Python >= 3.10 required (found: $(python3 --version 2>&1 || echo 'none'))"
    HARD_FAIL=1
fi

if ! command -v rocm-smi >/dev/null 2>&1 && ! command -v amd-smi >/dev/null 2>&1; then
    err "ROCm GPU tools not found (need rocm-smi or amd-smi)"
    HARD_FAIL=1
fi

if ! python3 -c "import ${FRAMEWORK}" 2>/dev/null; then
    err "Framework '${FRAMEWORK}' not importable; install sglang or vllm in user image first"
    HARD_FAIL=1
fi

# WekaFS bundle layout (validated against actual /wekafs/fully-local layout)
for sub in OOB TraceLens-internal inference_optimization/InferenceX; do
    if [ ! -d "$HYPERLOOM_BUNDLE/$sub" ]; then
        err "WekaFS bundle missing: $HYPERLOOM_BUNDLE/$sub"
        HARD_FAIL=1
    fi
done

if [ "$HARD_FAIL" -eq 1 ]; then
    err "hard requirement check failed; cannot continue"
    exit 1
fi
ok "Step 1 done — Python / GPU / ${FRAMEWORK} / WekaFS bundle present"

# =====================================================================
# Step 2 — soft system deps (install what's missing)
# =====================================================================
log "Step 2/6: ensuring system packages"

NEEDS_APT=()
for tool in pip git curl; do
    command -v "$tool" >/dev/null 2>&1 || NEEDS_APT+=("$tool")
done
# gnupg package provides 'gpg' command (not 'gnupg') on Debian/Ubuntu
command -v gpg >/dev/null 2>&1 || NEEDS_APT+=("gnupg")
if [ ! -f /etc/ssl/certs/ca-certificates.crt ]; then
    NEEDS_APT+=("ca-certificates")
fi

if [ "${#NEEDS_APT[@]}" -gt 0 ]; then
    log "  apt installing: ${NEEDS_APT[*]}"
    apt-get update -qq >/dev/null 2>&1 || warn "apt-get update failed"
    apt-get install -y --no-install-recommends "${NEEDS_APT[@]}" >/dev/null 2>&1 \
        || warn "apt install partially failed"
fi

NODE_OK=0
if command -v node >/dev/null 2>&1; then
    NODE_VER=$(node --version 2>/dev/null | grep -oP '\d+' | head -1 || echo 0)
    [ "${NODE_VER:-0}" -ge 20 ] && NODE_OK=1
fi
if [ "$NODE_OK" -eq 0 ]; then
    log "  installing Node.js 20 (nodesource)"
    curl -fsSL https://deb.nodesource.com/setup_20.x 2>/dev/null | bash - >/dev/null 2>&1 \
        && apt-get install -y nodejs >/dev/null 2>&1 \
        || warn "Node.js 20 install failed (npm CLIs in Step 4 will skip)"
fi

if [ ! -f /usr/local/share/ca-certificates/amd-root-ca.crt ]; then
    log "  installing AMD CA certificates"
    curl -fsSL https://raw.githubusercontent.com/AMD-AGI/Primus-SaFE/main/Scripts/setup-certs/amd-root-ca.crt \
        -o /usr/local/share/ca-certificates/amd-root-ca.crt 2>/dev/null || warn "amd-root-ca download failed"
    curl -fsSL https://raw.githubusercontent.com/AMD-AGI/Primus-SaFE/main/Scripts/setup-certs/amd-issuing-ca.crt \
        -o /usr/local/share/ca-certificates/amd-issuing-ca.crt 2>/dev/null || warn "amd-issuing-ca download failed"
    update-ca-certificates >/dev/null 2>&1 || warn "update-ca-certificates failed"
fi

ok "Step 2 done"

# =====================================================================
# Step 3 — external deps (GitHub + PyPI)
# =====================================================================
log "Step 3/6: installing external deps (intellikit + Ray)"

if ! python3 -c "import metrix" 2>/dev/null; then
    log "  cloning intellikit @ ${INTELLIKIT_SHA}"
    git clone "$INTELLIKIT_REPO" "$HYPERLOOM_ROOT/intellikit" 2>/dev/null \
        || warn "intellikit clone failed (may already exist)"
    git -C "$HYPERLOOM_ROOT/intellikit" checkout "$INTELLIKIT_SHA" 2>/dev/null \
        || warn "intellikit checkout failed"
    pip install -q --no-cache-dir -e "$HYPERLOOM_ROOT/intellikit/metrix/" 2>/dev/null \
        || warn "intellikit metrix install failed"
else
    log "  intellikit already installed, skipping"
fi

if ! command -v ray >/dev/null 2>&1; then
    log "  installing ray + click<8.3"
    pip install -q --no-cache-dir ray "click<8.3" 2>/dev/null || warn "ray/click install failed"
else
    log "  ray already installed, skipping"
fi

# Ray <= 2.44 deepcopy() in scripts.py breaks under click >= 8.3 (Sentinel change).
# Always pin click<8.3 even if ray was preinstalled, since other deps may have
# upgraded click past the safe boundary.
CLICK_VER=$(python3 -c "import click; print(click.__version__)" 2>/dev/null || echo "0")
CLICK_MAJOR=$(echo "$CLICK_VER" | cut -d. -f1)
CLICK_MINOR=$(echo "$CLICK_VER" | cut -d. -f2)
if [ "${CLICK_MAJOR:-0}" -gt 8 ] || { [ "${CLICK_MAJOR:-0}" -eq 8 ] && [ "${CLICK_MINOR:-0}" -ge 3 ]; }; then
    log "  click ${CLICK_VER} is too new for ray; downgrading to click<8.3"
    pip install -q --no-cache-dir "click<8.3" 2>/dev/null || warn "click downgrade failed"
fi

ok "Step 3 done"

# =====================================================================
# Step 4 — copy Hyperloom components from WekaFS + install
# =====================================================================
log "Step 4/6: copying Hyperloom components from WekaFS"

if ! python3 -c "import TraceLens" 2>/dev/null; then
    log "  cp TraceLens-internal -> $HYPERLOOM_ROOT/TraceLens"
    cp -r "$HYPERLOOM_BUNDLE/TraceLens-internal" "$HYPERLOOM_ROOT/TraceLens"
    pip install -q --no-cache-dir -e "$HYPERLOOM_ROOT/TraceLens" 2>/dev/null \
        || warn "TraceLens pip install failed"
else
    log "  TraceLens already installed, skipping"
fi

# OOB on WekaFS is flat (cli.py / auth_proxy.py at the top); we wrap it in
# /opt/hyperloom/OOB/oob_cli/ so paths match Method A exactly.
if ! command -v oob >/dev/null 2>&1; then
    log "  cp OOB -> $HYPERLOOM_ROOT/OOB/oob_cli"
    mkdir -p "$HYPERLOOM_ROOT/OOB"
    cp -r "$HYPERLOOM_BUNDLE/OOB" "$HYPERLOOM_ROOT/OOB/oob_cli"
    if [ -f "$HYPERLOOM_ROOT/OOB/oob_cli/requirements.txt" ]; then
        pip install -q --no-cache-dir -r "$HYPERLOOM_ROOT/OOB/oob_cli/requirements.txt" 2>/dev/null \
            || warn "OOB requirements install failed"
    fi
    pip install -q --no-cache-dir -e "$HYPERLOOM_ROOT/OOB/oob_cli" 2>/dev/null \
        || warn "OOB editable install failed"
else
    log "  oob CLI already installed, skipping"
fi

# Inject AMD CA into certifi bundle (Python httpx/requests use certifi, not system CA)
if [ -f /usr/local/share/ca-certificates/amd-root-ca.crt ]; then
    CERTIFI_BUNDLE=$(python3 -c "import certifi; print(certifi.where())" 2>/dev/null || true)
    if [ -n "$CERTIFI_BUNDLE" ] && ! grep -q "AMD" "$CERTIFI_BUNDLE" 2>/dev/null; then
        log "  injecting AMD CA into certifi bundle ($CERTIFI_BUNDLE)"
        cat /usr/local/share/ca-certificates/amd-root-ca.crt \
            /usr/local/share/ca-certificates/amd-issuing-ca.crt >> "$CERTIFI_BUNDLE" 2>/dev/null \
            || warn "certifi CA injection failed"
    fi
fi

if command -v node >/dev/null 2>&1; then
    if ! command -v claude >/dev/null 2>&1 || ! command -v codex >/dev/null 2>&1; then
        log "  installing npm CLIs (claude-code + codex@0.100.0)"
        npm install -g @anthropic-ai/claude-code @openai/codex@0.100.0 >/dev/null 2>&1 \
            || warn "npm CLI install failed"
    else
        log "  claude + codex CLIs already installed, skipping"
    fi
else
    warn "node not available; skipping npm CLI install (claude/codex backends unavailable)"
fi

ok "Step 4 done"

# =====================================================================
# Step 5 — write OOB / agent CLI auth files
# =====================================================================
log "Step 5/6: writing OOB auth files"

OOB_API_KEY_VAL="${OOB_API_KEY:-${SAFE_API_KEY:-${AMD_LLM_API_KEY:-${LLM_API_KEY:-}}}}"

if [ -n "$OOB_API_KEY_VAL" ] && [ ! -f /root/.claude/config.json ]; then
    mkdir -p /root/.claude
    cat > /root/.claude/config.json <<CLAUDE_EOF
{
  "theme": "dark",
  "hasCompletedOnboarding": true,
  "primaryApiKey": "${OOB_API_KEY_VAL}"
}
CLAUDE_EOF
    chmod 600 /root/.claude/config.json
    log "  wrote /root/.claude/config.json"
fi

if [ -n "$OOB_API_KEY_VAL" ] && [ ! -f /root/.codex/auth.json ]; then
    mkdir -p /root/.codex
    cat > /root/.codex/auth.json <<CODEX_EOF
{
  "auth_mode": "apikey",
  "OPENAI_API_KEY": "${OOB_API_KEY_VAL}"
}
CODEX_EOF
    chmod 600 /root/.codex/auth.json
    log "  wrote /root/.codex/auth.json"
fi

ok "Step 5 done"

# =====================================================================
# Step 6 — start services + write env file + done marker
# =====================================================================
log "Step 6/6: starting services + writing env"

# GPU count detection (env > visible-devices > smi)
detect_gpu_count() {
    if [ -n "${GPUS_PER_NODE:-}" ]; then
        echo "$GPUS_PER_NODE"
    elif [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
        echo "$HIP_VISIBLE_DEVICES" | tr ',' '\n' | wc -l
    elif [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
        echo "$ROCR_VISIBLE_DEVICES" | tr ',' '\n' | wc -l
    elif command -v amd-smi >/dev/null 2>&1; then
        amd-smi list 2>/dev/null | grep -c "^GPU:" || echo 0
    elif command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --showid 2>/dev/null | grep -c "^[0-9]" || echo 0
    else
        echo 0
    fi
}
GPU_COUNT=$(detect_gpu_count)

# Ray head
if ! ray status >/dev/null 2>&1; then
    log "  starting Ray head (--num-gpus=${GPU_COUNT})"
    RAY_GPU_OPT=""
    [ "${GPU_COUNT:-0}" -gt 0 ] 2>/dev/null && RAY_GPU_OPT="--num-gpus=${GPU_COUNT}"
    ray start --head --port=6379 \
              --dashboard-host=0.0.0.0 --dashboard-port=8265 \
              ${RAY_GPU_OPT} >"$LOG_DIR/ray-head.log" 2>&1 \
        || warn "Ray start failed (see $LOG_DIR/ray-head.log)"
else
    log "  Ray already running, skipping"
fi

# Core42 direct gateway settings.
#
# Claude Code fails through the local OOB auth-proxy on Core42 because the
# proxy path/header adaptation can return 404 for valid Claude models. Keep the
# proxy available for debugging, but never write it into ANTHROPIC_BASE_URL.
CORE42_LLM_PROXY_V1="${OOB_BASE_URL:-https://core42.example-internal-host.invalid/api/v1/llm-proxy/v1}"
CORE42_LLM_PROXY_BASE="${CORE42_LLM_PROXY_V1%/v1}"
if [ "$CORE42_LLM_PROXY_BASE" = "$CORE42_LLM_PROXY_V1" ]; then
    CORE42_LLM_PROXY_BASE="${CORE42_LLM_PROXY_BASE%/}"
fi

# Optional OOB auth-proxy (Bearer-rewriting reverse proxy on :4002 for AMD LLM gateway).
# Do not export it as ANTHROPIC_BASE_URL; Claude/OOB must use the direct gateway.
PROXY_ANTHROPIC_URL=""
PROXY_OPENAI_URL=""
if [ -n "${OOB_BASE_URL:-}" ]; then
    AUTH_PROXY_PORT=4002
    PROXY_PY="$HYPERLOOM_ROOT/OOB/oob_cli/auth_proxy.py"

    LLM_PROXY_SCHEME=$(echo "$OOB_BASE_URL" | grep -oP '^https?' || true)
    LLM_PROXY_HOST=$(echo "$OOB_BASE_URL" | grep -oP '(?<=://)[^:/]+' || true)
    PARSED_PORT=$(echo "$OOB_BASE_URL" | grep -oP '(?<=:)\d+(?=/)' || true)
    if [ -n "$PARSED_PORT" ]; then
        LLM_PROXY_PORT="$PARSED_PORT"
    elif [ "$LLM_PROXY_SCHEME" = "https" ]; then
        LLM_PROXY_PORT=443
    else
        LLM_PROXY_PORT=80
    fi
    LLM_PATH=$(echo "$OOB_BASE_URL" | grep -oP '(?<=://)[^/]+(/.+)' | grep -oP '/.*' || true)
    ANTHROPIC_PATH=$(echo "$LLM_PATH" | sed 's|/v1$||')
    PROXY_ANTHROPIC_URL="http://127.0.0.1:${AUTH_PROXY_PORT}${ANTHROPIC_PATH}"
    PROXY_OPENAI_URL="http://127.0.0.1:${AUTH_PROXY_PORT}${LLM_PATH}"

    if ! ss -tlnp 2>/dev/null | grep -q ":${AUTH_PROXY_PORT} "; then
        if [ -f "$PROXY_PY" ]; then
            log "  starting OOB auth-proxy on :${AUTH_PROXY_PORT}"
            export LLM_PROXY_SCHEME LLM_PROXY_HOST LLM_PROXY_PORT AUTH_PROXY_PORT
            export PROXY_AUTH_TOKEN="${OOB_API_KEY_VAL}"
            nohup python3 "$PROXY_PY" >"$LOG_DIR/oob-auth-proxy.log" 2>&1 &
            sleep 1
        else
            warn "auth_proxy.py not found at $PROXY_PY; OOB Bearer rewrite skipped"
        fi
    else
        log "  OOB auth-proxy already running on :${AUTH_PROXY_PORT}"
    fi
fi

# Write /etc/profile.d/hyperloom-env.sh — sourced on every SSH login
log "  writing $ENV_FILE"
{
    echo '#!/bin/sh'
    echo "# Hyperloom BYOI env (auto-generated by bootstrap.sh on $(date -u +%FT%TZ))"
    echo 'export PATH="/opt/venv/bin:$PATH"'
    echo 'export MODE="${MODE:-remote}"'
    echo "export FRAMEWORK='${FRAMEWORK}'"
    echo 'export OOB_CLI="${OOB_CLI:-oob}"'
    echo "export INFERENCEX_PATH='${INFERENCEX_PATH:-$HYPERLOOM_BUNDLE/inference_optimization/InferenceX}'"
    echo "export SKILL_ROOT='${SKILL_ROOT:-$SKILL_ROOT_DEFAULT}'"
    echo 'export SCRIPTS_DIR="${SKILL_ROOT}/scripts"'
    echo 'export OOB_RAY_CLI="python3 ${SCRIPTS_DIR}/oob_ray_submit.py"'
    echo "export TRACELENS_ROOT='${HYPERLOOM_ROOT}/TraceLens'"
    echo "export AGENT_WORKSPACE_ROOT='${AGENT_WORKSPACE_ROOT}'"
    echo "export KERNEL_OPT_BACKENDS='${KERNEL_OPT_BACKENDS:-codex,claude}'"
    echo "export PYTHONPATH=\"\${PYTHONPATH:+\$PYTHONPATH:}$HYPERLOOM_ROOT/OOB\""
    echo "export AMD_LLM_API_KEY='${AMD_LLM_API_KEY:-${LLM_API_KEY:-${LLM_GATEWAY_KEY:-}}}'"
    echo "export ANTHROPIC_API_KEY='${OOB_API_KEY_VAL}'"
    echo "export OPENAI_API_KEY='${OOB_API_KEY_VAL}'"
    echo "export OOB_BASE_URL='${CORE42_LLM_PROXY_V1}'"
    echo "export ANTHROPIC_BASE_URL='${CORE42_LLM_PROXY_BASE}'"
    echo 'export ANTHROPIC_CUSTOM_HEADERS="Authorization: Bearer ${ANTHROPIC_API_KEY}"'
    echo "export OPENAI_BASE_URL='${CORE42_LLM_PROXY_V1}'"
    [ -n "${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}" ] && \
        echo "export SSL_CERT_FILE='${SSL_CERT_FILE:-/etc/ssl/certs/ca-certificates.crt}'"
    echo "export NODE_EXTRA_CA_CERTS='${NODE_EXTRA_CA_CERTS:-/etc/ssl/certs/ca-certificates.crt}'"
} > "$ENV_FILE"
chmod 644 "$ENV_FILE"

touch "$DONE_MARKER"

# shellcheck disable=SC1090
. "$ENV_FILE"

ok "Step 6 done"

echo ""
ok "Bootstrap complete"
echo "  HYPERLOOM_BUNDLE: $HYPERLOOM_BUNDLE"
echo "  HYPERLOOM_ROOT:   $HYPERLOOM_ROOT"
echo "  OOB_CLI:          $(command -v oob 2>/dev/null || echo missing)"
echo "  INFERENCEX_PATH:  ${INFERENCEX_PATH:-$HYPERLOOM_BUNDLE/inference_optimization/InferenceX}"
echo "  SKILL_ROOT:       ${SKILL_ROOT:-$SKILL_ROOT_DEFAULT}"
echo "  GPU count:        ${GPU_COUNT}"
echo "  Ray:              $(ray status 2>/dev/null | head -1 || echo 'not running')"
echo "  Marker:           $DONE_MARKER"
echo "  Env file:         $ENV_FILE (sourced on SSH login)"
echo ""
