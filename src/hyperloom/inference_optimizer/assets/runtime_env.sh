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

# Which workspace a run uses, and so where its kernel-agent.env.sh lands, is decided here and nowhere else:
# install.sh sets its paths through runtime_env_forget_stale_workspace and runtime_env_derive_runtime_paths, and
# preflight (cli/preflight_env.py) asks runtime_env_kernel_agent_env_path for the file install.sh wrote.
#
#   - A .env that `hyperloom setup` wrote (setup_dotenv_is_authoritative) names the workspace: the caller's
#     USER_DATA_PATH and the paths derived from it are what an earlier workspace left in the shell, so they are
#     dropped and .env supplies them.
#   - Any other .env only fills what the caller left unset. It is shared by every session that runs from the
#     checkout, so a copy of a derived path there belongs to whichever workspace wrote it last: when the caller
#     names its own USER_DATA_PATH these come from that root, never from .env. This is how several sessions share
#     one checkout, each exporting only its own USER_DATA_PATH.
RUNTIME_ENV_WORKSPACE_DERIVED_KEYS="HYPERLOOM_RUNTIME_DIR KERNEL_AGENT_ENV HYPERLOOM_ROOT"

# Drop the caller's workspace selection when .env is setup-written; a readonly USER_DATA_PATH cannot be dropped.
runtime_env_forget_stale_workspace() {
  setup_dotenv_is_authoritative || return 0
  runtime_env_var_is_readonly USER_DATA_PATH || unset USER_DATA_PATH
  unset HYPERLOOM_RUNTIME_DIR KERNEL_AGENT_ENV HYPERLOOM_ROOT
}

# Fill the runtime paths from USER_DATA_PATH; call once the workspace is selected.
runtime_env_derive_runtime_paths() {
  HYPERLOOM_RUNTIME_DIR="${HYPERLOOM_RUNTIME_DIR:-${USER_DATA_PATH}/runtime}"
  KERNEL_AGENT_ENV="${KERNEL_AGENT_ENV:-${HYPERLOOM_RUNTIME_DIR}/kernel-agent.env.sh}"
  # Legacy variable kept for compatibility; open-source checkouts use install.sh's _open_source_root.
  HYPERLOOM_ROOT="${HYPERLOOM_ROOT:-${HYPERLOOM_RUNTIME_DIR}/source-mirrors}"
}

load_dotenv_no_clobber() {
  DOTENV_LOADED_COUNT=0
  [ -f "$REPO_ROOT/.env" ] || return 0
  local loaded=0 authoritative=0 readonly_root=0 caller_root=0
  setup_dotenv_is_authoritative && authoritative=1
  runtime_env_var_is_readonly USER_DATA_PATH && readonly_root=1
  [ -n "${USER_DATA_PATH:-}" ] && caller_root=1
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
      if [ "$caller_root" -eq 1 ] && [[ " $RUNTIME_ENV_WORKSPACE_DERIVED_KEYS " == *" $key "* ]]; then
        continue
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

# Print the kernel-agent.env.sh install.sh writes for the checkout $1 (default $REPO_ROOT) under the caller's
# environment, or nothing when no workspace is named. Run it in a command substitution: it loads .env into the shell
# that runs it.
runtime_env_kernel_agent_env_path() {
  local REPO_ROOT="${1:-${REPO_ROOT:-}}"
  if [ -n "$REPO_ROOT" ]; then
    runtime_env_forget_stale_workspace
    load_dotenv_no_clobber || return 1
  fi
  if [ -z "${KERNEL_AGENT_ENV:-}" ]; then
    [ -n "${HYPERLOOM_RUNTIME_DIR:-}${USER_DATA_PATH:-}" ] || return 0
    runtime_env_derive_runtime_paths
  fi
  printf '%s\n' "$KERNEL_AGENT_ENV"
}
