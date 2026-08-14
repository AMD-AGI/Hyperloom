#!/bin/bash

ENV_FILE="/opt/Hyperloom/.env"

set_env_var() {
    local key="$1"
    local value="$2"
    local escaped_value

    escaped_value=$(printf '%s' "$value" | sed 's/[\\&|]/\\&/g')

    if grep -qE "^[#[:space:]]*${key}=" "$ENV_FILE"; then
        sed -i "s|^[#[:space:]]*${key}=.*|${key}=${escaped_value}|g" "$ENV_FILE"
    else
        printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
    fi
}

if [ -n "${OPENAI_API_KEY}" ]; then
    set_env_var "OPENAI_API_KEY" "${OPENAI_API_KEY}"
fi

if [ -n "${OPENAI_BASE_URL}" ]; then
    set_env_var "OPENAI_BASE_URL" "${OPENAI_BASE_URL}"
fi

TRACELENS_ROOT=${TRACELENS_ROOT:-"/opt/TraceLens"}
set_env_var "TRACELENS_ROOT" "${TRACELENS_ROOT}"

if [ -n "${TRACELENS_INTERNAL_ROOT}" ]; then
    set_env_var "TRACELENS_INTERNAL_ROOT" "${TRACELENS_INTERNAL_ROOT}"
fi

if [ -n "${INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS}" ]; then
    set_env_var "INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS" "${INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS}"
fi

# Container images ship a writable /workspace; a bare-metal host off root has
# neither it nor permission to create it, so the mkdir below would abort.
_default_workspace_root() {
  if [ -w /workspace ]; then printf '%s' /workspace/hyperloom; else printf '%s' "$(pwd -P)/session"; fi
}
USER_DATA_PATH=${USER_DATA_PATH:-"$(_default_workspace_root)"}
set_env_var "USER_DATA_PATH" "${USER_DATA_PATH}"


tail -f /dev/null