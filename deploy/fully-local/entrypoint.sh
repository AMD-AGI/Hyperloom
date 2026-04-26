#!/bin/bash
set -e

# Ensure `python` is available even if only `python3` exists
if ! command -v python &>/dev/null && command -v python3 &>/dev/null; then
    ln -s "$(command -v python3)" /usr/local/bin/python
fi

LOG_DIR=/var/log/hyperloom
mkdir -p "$LOG_DIR" "${NFS_BASE_PATH:-/tmp/geak-data}"

# --- Graceful shutdown ---
cleanup() {
    echo "[entrypoint] Shutting down..."
    if [ -n "${AUTH_PROXY_PID:-}" ] && [ "$AUTH_PROXY_PID" -gt 0 ] 2>/dev/null; then
        kill "$AUTH_PROXY_PID" 2>/dev/null || true
    fi
    ray stop --force 2>/dev/null || true
    wait
    exit 0
}
trap cleanup SIGTERM SIGINT

# --- GPU inventory: env vars > hardware scan ---
if [ -n "${GPUS_PER_NODE:-}" ]; then
    GPU_COUNT="$GPUS_PER_NODE"
elif [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
    GPU_COUNT=$(echo "$HIP_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
elif [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
    GPU_COUNT=$(echo "$ROCR_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
elif command -v amd-smi &>/dev/null; then
    GPU_COUNT=$(amd-smi list 2>/dev/null | grep -c "^GPU:" || echo 0)
elif command -v rocm-smi &>/dev/null; then
    GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -c "^[0-9]" || echo 0)
else
    GPU_COUNT=0
fi

AUTH_PROXY_PID=0

export MODE="${MODE:-fully-local}"

# OOB runs as a per-task CLI in fully-local mode (`oob run ...`); no persistent service.
export OOB_CLI="${OOB_CLI:-oob}"

echo "============================================"
echo "  Hyperloom — Local Mode (containerized)"
echo "============================================"
echo "  GPU:          ${GPU_COUNT} detected"
echo "  Framework:    ${FRAMEWORK:-sglang}"
echo "  GEAK:         CLI (geak) via Ray scheduler"
echo "  GEAK model:   ${GEAK_MODEL_NAME:-claude-opus-4-7}"
echo "  GEAK config:  ${GEAK_CONFIG:-/opt/hyperloom/geak-config/local.yaml}"
echo "  GEAK data:    ${NFS_BASE_PATH:-/tmp/geak-data}"
echo "  Ray head:     :6379 (dashboard :8265)"
echo "  OOB:          CLI (${OOB_CLI})"
echo "  TraceLens:    CLI (pip-installed)"
echo "  SSH:          :22"
echo "  Logs:         ${LOG_DIR}/"
echo "============================================"

# --- Map OOB vars to provider-specific runtime vars ---
export ANTHROPIC_API_KEY="${OOB_API_KEY:-}"
export OPENAI_API_KEY="${OOB_API_KEY:-}"
export ANTHROPIC_BASE_URL="${OOB_BASE_URL:-}"
export OPENAI_BASE_URL="${OOB_BASE_URL:-}"

# Alias LLM_API_KEY -> AMD_LLM_API_KEY for GEAK CLI (geak reads AMD_LLM_API_KEY)
export AMD_LLM_API_KEY="${AMD_LLM_API_KEY:-${LLM_API_KEY:-${LLM_GATEWAY_KEY:-}}}"

# Render GEAK LiteLLM config from template (model/key/base resolved from env)
GEAK_TEMPLATE=/opt/hyperloom/geak-config/template.yaml
GEAK_CONFIG="${GEAK_CONFIG:-/opt/hyperloom/geak-config/local.yaml}"

# Fallback for dev mode where /opt/hyperloom is bind-mounted from host repo
if [ ! -f "$GEAK_TEMPLATE" ] && [ -f "/opt/hyperloom/deploy/fully-local/geak-litellm.yaml" ]; then
    GEAK_TEMPLATE="/opt/hyperloom/deploy/fully-local/geak-litellm.yaml"
fi

if [ -f "$GEAK_TEMPLATE" ]; then
    mkdir -p "$(dirname "$GEAK_CONFIG")"
    _model="${GEAK_MODEL_NAME:-claude-opus-4-7}"
    _key="${GEAK_API_KEY:-${LLM_API_KEY:-${AMD_LLM_API_KEY:-}}}"
    _url="${GEAK_BASE_URL:-${LLM_API_BASE:-}}"
    sed -e "s|__GEAK_MODEL_NAME__|${_model}|g" \
        -e "s|__GEAK_API_KEY__|${_key}|g" \
        -e "s|__GEAK_BASE_URL__|${_url}|g" \
        "$GEAK_TEMPLATE" > "$GEAK_CONFIG"
    chmod 600 "$GEAK_CONFIG"
else
    echo "[WARN] GEAK template not found at $GEAK_TEMPLATE"
fi
export GEAK_CONFIG

# --- Export env vars for SSH sessions (docker run -e vars are invisible to sshd) ---
{
    echo 'export PATH="/opt/venv/bin:$PATH"'
    for var in MODE FRAMEWORK GEAK_LOCAL KERNEL_OPT_BACKENDS NFS_BASE_PATH \
               INFERENCEX_PATH LLM_API_KEY LLM_API_BASE AMD_LLM_API_KEY LLM_GATEWAY_KEY \
               GEAK_CONFIG GEAK_MODEL_NAME GEAK_API_KEY GEAK_BASE_URL \
               AGENT_WORKSPACE_ROOT OOB_CLI OOB_HOME \
               OOB_API_KEY OOB_BASE_URL OOB_LOCAL \
               ANTHROPIC_API_KEY ANTHROPIC_BASE_URL OPENAI_API_KEY OPENAI_BASE_URL \
               HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPUS_PER_NODE; do
        [ -n "${!var:-}" ] && echo "export ${var}='${!var}'"
    done
} > /etc/profile.d/hyperloom-env.sh

# --- SSH server ---
/usr/sbin/sshd

# --- Local Ray head (GPU task scheduler for GEAK) ---
RAY_GPU_OPT=""
if [ "${GPU_COUNT:-0}" -gt 0 ] 2>/dev/null; then
    RAY_GPU_OPT="--num-gpus=${GPU_COUNT}"
fi
ray start --head \
    --port=6379 \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    ${RAY_GPU_OPT} \
    > "$LOG_DIR/ray-head.log" 2>&1 || true

# --- OOB CLI prep (Claude Code + Codex backends; invoked per-task by `oob run`) ---
# Provision agent CLI auth files and an optional auth proxy. The MCP/REST service
# is no longer started — the skill calls `oob run ...` directly per task.
if [ -d /opt/OOB/oob_cli ]; then
    mkdir -p "${AGENT_WORKSPACE_ROOT:-/tmp/agent-workspaces}"

    if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
        mkdir -p /root/.claude
        cat > /root/.claude/config.json <<CLAUDE_EOF
{
  "theme": "dark",
  "hasCompletedOnboarding": true,
  "primaryApiKey": "${ANTHROPIC_API_KEY}"
}
CLAUDE_EOF
        chmod 600 /root/.claude/config.json
    fi

    if [ -n "${OPENAI_API_KEY:-}" ]; then
        mkdir -p /root/.codex
        cat > /root/.codex/auth.json <<CODEX_EOF
{
  "auth_mode": "apikey",
  "OPENAI_API_KEY": "${OPENAI_API_KEY}"
}
CODEX_EOF
        chmod 600 /root/.codex/auth.json
    fi

    # Bearer-auth rewrite for AMD LLM gateway: claude/codex CLIs only know
    # x-api-key, so route them through a localhost proxy that injects Bearer.
    if [ -n "${ANTHROPIC_BASE_URL:-}" ]; then
        export LLM_PROXY_SCHEME=$(echo "$ANTHROPIC_BASE_URL" | grep -oP '^https?' || true)
        export LLM_PROXY_HOST=$(echo "$ANTHROPIC_BASE_URL" | grep -oP '(?<=://)[^:/]+' || true)
        PARSED_PORT=$(echo "$ANTHROPIC_BASE_URL" | grep -oP '(?<=:)\d+(?=/)' || true)
        if [ -n "$PARSED_PORT" ]; then
            export LLM_PROXY_PORT="$PARSED_PORT"
        elif [ "$LLM_PROXY_SCHEME" = "https" ]; then
            export LLM_PROXY_PORT=443
        else
            export LLM_PROXY_PORT=80
        fi
        LLM_PATH=$(echo "$ANTHROPIC_BASE_URL" | grep -oP '(?<=://)[^/]+(/.+)' | grep -oP '/.*' || true)
        export AUTH_PROXY_PORT=4002
        export PROXY_AUTH_TOKEN="${ANTHROPIC_API_KEY:-${OPENAI_API_KEY:-}}"
        python /opt/OOB/oob_cli/auth_proxy.py \
          > "$LOG_DIR/oob-auth-proxy.log" 2>&1 &
        AUTH_PROXY_PID=$!
        sleep 1
        ANTHROPIC_PATH=$(echo "$LLM_PATH" | sed 's|/v1$||')
        export ANTHROPIC_BASE_URL="http://127.0.0.1:${AUTH_PROXY_PORT}${ANTHROPIC_PATH}"
        # Codex musl-rustls may not honor corporate CAs;
        # route through the same proxy (HTTP -> HTTPS upstream).
        export OPENAI_BASE_URL="http://127.0.0.1:${AUTH_PROXY_PORT}${LLM_PATH}"
    fi
fi

# --- Health check: wait for ports (timeout 30s) ---
wait_for_port() {
    local port=$1 name=$2 timeout=30 elapsed=0
    while ! ss -tlnp 2>/dev/null | grep -q ":${port} " && [ $elapsed -lt $timeout ]; do
        sleep 1
        elapsed=$((elapsed + 1))
    done
    if [ $elapsed -ge $timeout ]; then
        echo "  [WARN] ${name} on :${port} not ready after ${timeout}s — check ${LOG_DIR}/"
    else
        echo "  [OK]   ${name} on :${port} (${elapsed}s)"
    fi
}

echo ""
echo "Waiting for services..."
wait_for_port 6379 "Ray head"
wait_for_port 8265 "Ray dashboard"
[ "$AUTH_PROXY_PID" -gt 0 ] && wait_for_port "${AUTH_PROXY_PORT:-4002}" "OOB auth-proxy"

echo ""
echo "${GPU_COUNT} GPU(s) available. Framework '${FRAMEWORK:-sglang}' ready."
echo "GEAK tasks are scheduled via Ray (dashboard: http://localhost:8265)"
echo "Connect: Cursor Remote SSH → localhost:<mapped-ssh-port> → open /opt/hyperloom"
echo ""

# --- Keep alive: restart crashed services ---
while true; do
    if ! ray status > /dev/null 2>&1; then
        echo "[$(date)] Ray head down, restarting..."
        ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 \
            ${RAY_GPU_OPT} \
            >> "$LOG_DIR/ray-head.log" 2>&1 || true
    fi
    if [ "$AUTH_PROXY_PID" -gt 0 ] && ! kill -0 "$AUTH_PROXY_PID" 2>/dev/null; then
        echo "[$(date)] OOB auth-proxy crashed, restarting..."
        python /opt/OOB/oob_cli/auth_proxy.py \
          >> "$LOG_DIR/oob-auth-proxy.log" 2>&1 &
        AUTH_PROXY_PID=$!
    fi
    sleep 5
done
