# Hyperloom CLI auto-start script
# Sourced by /etc/profile.d/hyperloom.sh on first login

export MODE="${MODE:-fully-local}"

if [ -z "$HYPERLOOM_STARTED" ] && [ "${MODE:-}" = "fully-local" ]; then
    export HYPERLOOM_STARTED=1
    mkdir -p /var/log/hyperloom "${NFS_BASE_PATH:-/tmp/geak-data}"
    
    # Map OOB vars to provider-specific runtime vars
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

    # OOB runs as a per-task CLI (`oob run ...`); no persistent service to start.
    export OOB_CLI="${OOB_CLI:-oob}"

    # Detect GPU count when entrypoint.sh did not run first.
    if [ -z "${GPU_COUNT:-}" ]; then
        if [ -n "${GPUS_PER_NODE:-}" ]; then
            GPU_COUNT="$GPUS_PER_NODE"
        elif [ -n "${HIP_VISIBLE_DEVICES:-}" ]; then
            GPU_COUNT=$(echo "$HIP_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
        elif [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
            GPU_COUNT=$(echo "$ROCR_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
        elif command -v amd-smi >/dev/null 2>&1; then
            GPU_COUNT=$(amd-smi list 2>/dev/null | grep -c "^GPU:" || echo 0)
        elif command -v rocm-smi >/dev/null 2>&1; then
            GPU_COUNT=$(rocm-smi --showid 2>/dev/null | grep -c "^[0-9]" || echo 0)
        else
            GPU_COUNT=0
        fi
    fi

    # Start Ray if not running
    if ! ray status > /dev/null 2>&1; then
        echo "Starting Ray head..."
        RAY_GPU_OPT=""
        if [ "${GPU_COUNT:-0}" -gt 0 ] 2>/dev/null; then
            RAY_GPU_OPT="--num-gpus=${GPU_COUNT}"
        fi
        ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 ${RAY_GPU_OPT} > /dev/null 2>&1 || true
    fi

    # Rewrite OOB traffic through the local auth proxy.
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
        ANTHROPIC_PATH=$(echo "$LLM_PATH" | sed 's|/v1$||')
        export AUTH_PROXY_PORT=4002
        export PROXY_AUTH_TOKEN="${ANTHROPIC_API_KEY:-${OPENAI_API_KEY:-}}"

        if ! ss -tlnp 2>/dev/null | grep -q ":4002 "; then
            echo "Starting OOB auth proxy..."
            python3 /opt/OOB/oob_cli/auth_proxy.py > /var/log/hyperloom/oob-auth-proxy.log 2>&1 &
        fi

        export ANTHROPIC_BASE_URL="http://127.0.0.1:${AUTH_PROXY_PORT}${ANTHROPIC_PATH}"
        export OPENAI_BASE_URL="http://127.0.0.1:${AUTH_PROXY_PORT}${LLM_PATH}"
    fi
fi
