#!/bin/bash
# Auto-start Hyperloom MCP services on login (idempotent)
# Uses port checks instead of pgrep (pgrep is unreliable with hostPID: true)
if [ -z "$HYPERLOOM_STARTED" ] && [ -f /opt/entrypoint.sh ]; then
    mkdir -p /var/log/hyperloom "${NFS_BASE_PATH:-/tmp/geak-data}"

    # Map LLM_API_KEY to all aliases GEAK might look for
    if [ -n "${LLM_API_KEY:-}" ]; then
        export AMD_LLM_API_KEY="${AMD_LLM_API_KEY:-$LLM_API_KEY}"
        export LLM_GATEWAY_KEY="${LLM_GATEWAY_KEY:-$LLM_API_KEY}"
    fi

    if ! curl -s --max-time 1 http://localhost:${TRACELENS_PORT:-8001}/mcp > /dev/null 2>&1; then
        echo "[Hyperloom] Starting TraceLens MCP..."
        TRACELENS_PORT=${TRACELENS_PORT:-8001} TRACELENS_HOST=0.0.0.0 \
          python -m TraceLens.AgenticMode.MCPServer \
          > /var/log/hyperloom/tracelens.log 2>&1 &
    fi
    if ! curl -s --max-time 1 http://localhost:8000/health > /dev/null 2>&1; then
        echo "[Hyperloom] Starting GEAK REST API..."
        cd /opt/geak && \
        GEAK_LOCAL=true HOST=0.0.0.0 \
        NFS_BASE_PATH="${NFS_BASE_PATH:-/tmp/geak-data}" \
        DATABASE_PATH="${DATABASE_PATH:-/tmp/geak-data/geak.db}" \
          python -m server.main \
          > /var/log/hyperloom/geak-api.log 2>&1 &
        cd - >/dev/null
    fi
    if ! curl -s --max-time 1 http://localhost:${GEAK_MCP_PORT:-8002}/ > /dev/null 2>&1; then
        echo "[Hyperloom] Starting GEAK MCP..."
        cd /opt/geak && \
        GEAK_LOCAL=true HOST=0.0.0.0 MCP_PORT=${GEAK_MCP_PORT:-8002} \
        NFS_BASE_PATH="${NFS_BASE_PATH:-/tmp/geak-data}" \
        DATABASE_PATH="${DATABASE_PATH:-/tmp/geak-data/geak.db}" \
          python -m server.mcp.http_server \
          > /var/log/hyperloom/geak-mcp.log 2>&1 &
        cd - >/dev/null
    fi
    if [ -d /opt/oob-mcp/agent_mcp_server ] && \
       ! curl -s --max-time 1 http://localhost:${OOB_MCP_PORT:-8003}/ > /dev/null 2>&1; then
        echo "[Hyperloom] Starting OOB Agent MCP..."
        if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
            mkdir -p /root/.claude
            printf '{"theme":"dark","hasCompletedOnboarding":true,"primaryApiKey":"%s"}' \
                "${ANTHROPIC_API_KEY}" > /root/.claude/config.json
            chmod 600 /root/.claude/config.json
        fi
        if [ -n "${OPENAI_API_KEY:-}" ]; then
            mkdir -p /root/.codex
            printf '{"auth_mode":"apikey","OPENAI_API_KEY":"%s"}' \
                "${OPENAI_API_KEY}" > /root/.codex/auth.json
            chmod 600 /root/.codex/auth.json
        fi
        PYTHONPATH=/opt/oob-mcp \
        MCP_PORT=${OOB_MCP_PORT:-8003} \
        AGENT_WORKSPACE_ROOT="${AGENT_WORKSPACE_ROOT:-/tmp/agent-workspaces}" \
        ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-}" \
        ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-}" \
        OPENAI_API_KEY="${OPENAI_API_KEY:-}" \
        OPENAI_BASE_URL="${OPENAI_BASE_URL:-}" \
          python -m agent_mcp_server.server \
          > /var/log/hyperloom/oob-mcp.log 2>&1 &
    fi

    # Auto-configure GEAK LLM if env vars are set
    if [ -n "${LLM_API_KEY:-}" ] && [ -n "${LLM_API_BASE:-}" ]; then
        (
            sleep 5
            curl -s -X POST http://localhost:8000/api/v1/config/model \
              -H "Content-Type: application/json" \
              -H "X-API-Key: local-mcp" \
              -d "{\"model_class\":\"litellm\",\"model_name\":\"openai/gpt-4\",\"model_kwargs\":{\"api_base\":\"${LLM_API_BASE}\",\"api_key\":\"${LLM_API_KEY}\"}}" \
              > /dev/null 2>&1 && echo "[Hyperloom] GEAK LLM configured" || echo "[Hyperloom] GEAK LLM config failed"
        ) &
    fi

    export HYPERLOOM_STARTED=1
fi
