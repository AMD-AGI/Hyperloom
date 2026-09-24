#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Source this asset, then load missing exports from $REPO_ROOT/.env.
setup_dotenv_is_authoritative() {
  [ -f "$REPO_ROOT/.env" ] || return 1
  [ "${HYPERLOOM_SETUP_ENV_AUTHORITATIVE:-0}" = 1 ] ||
    grep -q '^HYPERLOOM_RUN_MODE=' "$REPO_ROOT/.env" 2>/dev/null
}

runtime_env_var_is_readonly() {
  local declaration
  declaration="$(declare -p "$1" 2>/dev/null)" || return 1
  [[ "$declaration" =~ ^declare\ -[^[:space:]]*r[^[:space:]]*\  ]]
}

# Decode only quote escapes, consuming continuation lines from the same input.
# REPLY preserves trailing newlines; no parameter, command, arithmetic, tilde or glob expansion.
_runtime_env_read_quoted_value() {
  local quote="${1:0:1}" pending="${1:1}" decoded="" char continuation=0
  while :; do
    if [ -z "$pending" ]; then
      if ! IFS= read -r pending && [ -z "$pending" ]; then
        printf '%s\n' '[runtime-env ERROR] unterminated quoted value in .env' >&2
        return 1
      fi
      pending="${pending%$'\r'}"
      [ "$continuation" -eq 1 ] || decoded+=$'\n'
      continuation=0
      continue
    fi
    char="${pending:0:1}"
    pending="${pending:1}"
    if [ "$char" = "$quote" ] && [[ "$pending" != *[![:space:]]* ]]; then
      REPLY="$decoded"
      return 0
    fi
    if [ "$quote" = '"' ] && [ "$char" = '\' ]; then
      case "${pending:0:1}" in
        '\'|'"'|'$'|'`')
          decoded+="${pending:0:1}"
          pending="${pending:1}"
          continue ;;
        '') continuation=1; continue ;;
      esac
    fi
    decoded+="$char"
  done
}

load_dotenv_no_clobber() {
  DOTENV_LOADED_COUNT=0
  [ -f "$REPO_ROOT/.env" ] || return 0
  local loaded=0 authoritative=0 readonly_root=0
  setup_dotenv_is_authoritative && authoritative=1
  runtime_env_var_is_readonly USER_DATA_PATH && readonly_root=1
  local pass raw key value REPLY
  # Resolve the caller/file mode before any Python key, regardless of line order.
  for pass in mode values; do
    while IFS= read -r raw || [ -n "$raw" ]; do
      raw="${raw#"${raw%%[![:space:]]*}"}"
      raw="${raw%$'\r'}"
      [ -z "$raw" ] && continue
      case "$raw" in \#*) continue ;; esac
      case "$raw" in export\ *) raw="${raw#export }" ;; esac
      case "$raw" in *=*) ;; *) continue ;; esac
      key="${raw%%=*}"
      value="${raw#*=}"
      key="${key%"${key##*[![:space:]]}"}"
      value="${value#"${value%%[![:space:]]*}"}"
      case "$value" in
        \"*|\'*)
          _runtime_env_read_quoted_value "$value" || return 1
          value="$REPLY" ;;
        *) value="${value%"${value##*[![:space:]]}"}" ;;
      esac
      if [[ ! "$key" =~ ^[a-zA-Z_][a-zA-Z0-9_]*$ ]]; then
        printf '%s\n' '[runtime-env ERROR] invalid variable name in .env' >&2
        return 1
      fi
      if [ "$pass" = mode ]; then
        [ "$key" = HYPERLOOM_RUN_MODE ] || continue
      else
        [ "$key" = HYPERLOOM_RUN_MODE ] && continue
        if [ "${HYPERLOOM_RUN_MODE:-}" = docker ]; then
          case "$key" in PYTHON|VIRTUAL_ENV|INFERENCE_OPTIMIZER_FORCE_PYTHON) continue ;; esac
        fi
      fi
      if [ "$key" = USER_DATA_PATH ] && [ "$readonly_root" -eq 1 ]; then
        if [ "${USER_DATA_PATH:-}" != "$value" ] &&
           { [ "$authoritative" -eq 1 ] || [ -z "${USER_DATA_PATH:-}" ]; }; then
          printf '%s\n' '[runtime-env ERROR] readonly USER_DATA_PATH conflicts with .env workspace root' >&2
          return 1
        fi
        continue
      fi
      if [ -z "${!key:-}" ]; then
        export "$key=$value" || return 1
        loaded=$((loaded + 1))
      fi
    done < "$REPO_ROOT/.env"
  done
  DOTENV_LOADED_COUNT="$loaded"
  return 0
}
