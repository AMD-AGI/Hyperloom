#!/usr/bin/env bash
# =============================================================================
# apply_quiet_config.sh
#
# Functions to temporarily quieten the training YAML config for short trials.
# Reduces Megatron stderr noise at the source while preserving :::MLLOG output.
#
# Usage:
#   source apply_quiet_config.sh
#   quiet_yaml "$EXP"          # backup + apply quiet settings
#   ... run trial ...
#   restore_yaml "$EXP"        # restore original
# =============================================================================

quiet_yaml() {
    local yaml="${1:?YAML path required}"

    if [ ! -f "$yaml" ]; then
        echo "ERROR: quiet_yaml: $yaml not found" >&2
        return 1
    fi

    if [ -f "${yaml}.bak" ]; then
        echo "WARNING: quiet_yaml: backup already exists, restoring first" >&2
        restore_yaml "$yaml"
    fi

    cp "$yaml" "${yaml}.bak"
    sed -i 's/stderr_sink_level: DEBUG/stderr_sink_level: WARNING/' "$yaml"
    sed -i 's/log_interval: 32/log_interval: 999999/' "$yaml"
}

restore_yaml() {
    local yaml="${1:?YAML path required}"

    if [ -f "${yaml}.bak" ]; then
        mv "${yaml}.bak" "$yaml"
    fi
}

is_yaml_quiet() {
    local yaml="${1:?YAML path required}"
    [ -f "${yaml}.bak" ]
}
