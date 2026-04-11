#!/bin/bash
set -e

LOG_DIR=/var/log/hyperloom
mkdir -p "$LOG_DIR" "${NFS_BASE_PATH:-/tmp/geak-data}"

# --- Graceful shutdown ---
cleanup() {
    echo "[entrypoint] Shutting down..."
    kill "$TRACELENS_PID" "$GEAK_API_PID" "$GEAK_MCP_PID" \
         "$OOB_MCP_PID" "$AUTH_PROXY_PID" 2>/dev/null || true
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

TRACELENS_PORT=${TRACELENS_PORT:-8001}
GEAK_MCP_PORT=${GEAK_MCP_PORT:-8002}
OOB_MCP_PORT=${OOB_MCP_PORT:-8003}
OOB_MCP_PID=0
AUTH_PROXY_PID=0

echo "============================================"
echo "  Hyperloom — Local Mode (containerized)"
echo "============================================"
echo "  GPU:          ${GPU_COUNT} detected"
echo "  Framework:    ${FRAMEWORK:-sglang}"
echo "  GEAK data:    ${NFS_BASE_PATH:-/tmp/geak-data}"
echo "  TraceLens:    :${TRACELENS_PORT}"
echo "  GEAK MCP:     :${GEAK_MCP_PORT}"
echo "  OOB Agent:    :${OOB_MCP_PORT}"
echo "  SSH:          :22"
echo "  Logs:         ${LOG_DIR}/"
echo "============================================"

# --- Map LLM_API_KEY to all aliases GEAK might look for ---
if [ -n "${LLM_API_KEY:-}" ]; then
    export AMD_LLM_API_KEY="${AMD_LLM_API_KEY:-$LLM_API_KEY}"
    export LLM_GATEWAY_KEY="${LLM_GATEWAY_KEY:-$LLM_API_KEY}"
fi

# --- Map OOB vars to provider-specific runtime vars ---
export ANTHROPIC_API_KEY="${OOB_API_KEY:-}"
export OPENAI_API_KEY="${OOB_API_KEY:-}"
export ANTHROPIC_BASE_URL="${OOB_BASE_URL:-}"
export OPENAI_BASE_URL="${OOB_BASE_URL:-}"

# --- Export env vars for SSH sessions (docker run -e vars are invisible to sshd) ---
{
    for var in MODE FRAMEWORK GEAK_LOCAL KERNEL_OPT_BACKENDS NFS_BASE_PATH DATABASE_PATH \
               INFERENCEX_PATH LLM_API_KEY LLM_API_BASE AMD_LLM_API_KEY LLM_GATEWAY_KEY \
               TRACELENS_PORT GEAK_MCP_PORT OOB_MCP_PORT AGENT_WORKSPACE_ROOT \
               OOB_API_KEY OOB_BASE_URL \
               HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPUS_PER_NODE; do
        [ -n "${!var:-}" ] && echo "export ${var}='${!var}'"
    done
} > /etc/profile.d/hyperloom-env.sh

# --- SSH server ---
/usr/sbin/sshd

# --- TraceLens MCP ---
TRACELENS_PORT=$TRACELENS_PORT \
TRACELENS_HOST=0.0.0.0 \
  python -m TraceLens.AgenticMode.MCPServer \
  > "$LOG_DIR/tracelens.log" 2>&1 &
TRACELENS_PID=$!

# --- GEAK REST API (backend for MCP tools) ---
cd /opt/geak
GEAK_LOCAL=true \
HOST=0.0.0.0 \
NFS_BASE_PATH="${NFS_BASE_PATH:-/tmp/geak-data}" \
DATABASE_PATH="${DATABASE_PATH:-/tmp/geak-data/geak.db}" \
  python -m server.main \
  > "$LOG_DIR/geak-api.log" 2>&1 &
GEAK_API_PID=$!

# --- GEAK MCP ---
GEAK_LOCAL=true \
HOST=0.0.0.0 \
MCP_PORT=$GEAK_MCP_PORT \
NFS_BASE_PATH="${NFS_BASE_PATH:-/tmp/geak-data}" \
DATABASE_PATH="${DATABASE_PATH:-/tmp/geak-data/geak.db}" \
  python -m server.mcp.http_server \
  > "$LOG_DIR/geak-mcp.log" 2>&1 &
GEAK_MCP_PID=$!

# --- OOB Agent MCP (Claude Code + Codex backends) ---
if [ -d /opt/oob-mcp/agent_mcp_server ]; then
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
        python3 /opt/oob-mcp/agent_mcp_server/auth_proxy.py \
          > "$LOG_DIR/oob-auth-proxy.log" 2>&1 &
        AUTH_PROXY_PID=$!
        sleep 1
        ANTHROPIC_PATH=$(echo "$LLM_PATH" | sed 's|/v1$||')
        export ANTHROPIC_BASE_URL="http://127.0.0.1:${AUTH_PROXY_PORT}${ANTHROPIC_PATH}"
    fi

    PYTHONPATH=/opt/oob-mcp \
    MCP_PORT=$OOB_MCP_PORT \
    AGENT_WORKSPACE_ROOT="${AGENT_WORKSPACE_ROOT:-/tmp/agent-workspaces}" \
    ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
    ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-}" \
    OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
    OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
      python -m agent_mcp_server.server \
      > "$LOG_DIR/oob-mcp.log" 2>&1 &
    OOB_MCP_PID=$!
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
wait_for_port "$TRACELENS_PORT" "TraceLens"
wait_for_port 8000 "GEAK API"
wait_for_port "$GEAK_MCP_PORT" "GEAK MCP"
[ "$OOB_MCP_PID" -gt 0 ] && wait_for_port "$OOB_MCP_PORT" "OOB Agent"

# --- Auto-configure GEAK LLM from env vars ---
if [ -n "${LLM_API_KEY:-}" ] && [ -n "${LLM_API_BASE:-}" ]; then
    echo "Configuring GEAK LLM backend..."
    curl -s -X POST http://localhost:8000/api/v1/config/model \
      -H "Content-Type: application/json" \
      -H "X-API-Key: local-mcp" \
      -d "{\"model_class\":\"litellm\",\"model_name\":\"openai/gpt-4\",\"model_kwargs\":{\"api_base\":\"${LLM_API_BASE}\",\"api_key\":\"${LLM_API_KEY}\"}}" \
      > /dev/null 2>&1 && echo "  [OK]   GEAK LLM configured" || echo "  [WARN] GEAK LLM config failed"
fi

echo ""
echo "${GPU_COUNT} GPU(s) available. Framework '${FRAMEWORK:-sglang}' ready."
echo "Connect: Cursor Remote SSH → localhost:<mapped-ssh-port> → open /opt/hyperloom"
echo ""

# --- Keep alive: restart crashed services ---
while true; do
    if ! kill -0 "$TRACELENS_PID" 2>/dev/null; then
        echo "[$(date)] TraceLens crashed, restarting..."
        cd /opt/TraceLens
        TRACELENS_PORT=$TRACELENS_PORT TRACELENS_HOST=0.0.0.0 \
          python -m TraceLens.AgenticMode.MCPServer \
          >> "$LOG_DIR/tracelens.log" 2>&1 &
        TRACELENS_PID=$!
    fi
    if ! kill -0 "$GEAK_API_PID" 2>/dev/null; then
        echo "[$(date)] GEAK API crashed, restarting..."
        cd /opt/geak
        GEAK_LOCAL=true HOST=0.0.0.0 \
        NFS_BASE_PATH="${NFS_BASE_PATH:-/tmp/geak-data}" \
        DATABASE_PATH="${DATABASE_PATH:-/tmp/geak-data/geak.db}" \
          python -m server.main \
          >> "$LOG_DIR/geak-api.log" 2>&1 &
        GEAK_API_PID=$!
    fi
    if ! kill -0 "$GEAK_MCP_PID" 2>/dev/null; then
        echo "[$(date)] GEAK MCP crashed, restarting..."
        cd /opt/geak
        GEAK_LOCAL=true HOST=0.0.0.0 MCP_PORT=$GEAK_MCP_PORT \
        NFS_BASE_PATH="${NFS_BASE_PATH:-/tmp/geak-data}" \
        DATABASE_PATH="${DATABASE_PATH:-/tmp/geak-data/geak.db}" \
          python -m server.mcp.http_server \
          >> "$LOG_DIR/geak-mcp.log" 2>&1 &
        GEAK_MCP_PID=$!
    fi
    if [ "$OOB_MCP_PID" -gt 0 ] && ! kill -0 "$OOB_MCP_PID" 2>/dev/null; then
        echo "[$(date)] OOB Agent MCP crashed, restarting..."
        PYTHONPATH=/opt/oob-mcp \
        MCP_PORT=$OOB_MCP_PORT \
        AGENT_WORKSPACE_ROOT="${AGENT_WORKSPACE_ROOT:-/tmp/agent-workspaces}" \
        ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
        ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-}" \
        OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
        OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
          python -m agent_mcp_server.server \
          >> "$LOG_DIR/oob-mcp.log" 2>&1 &
        OOB_MCP_PID=$!
    fi
    sleep 5
done
