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

# Which workspace a run uses, and so which kernel-agent.env.sh it gets, is decided here and nowhere else:
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
#
# kernel-agent.env.sh carries credentials and is sourced, so it never lives under USER_DATA_PATH, which every pod of
# the user shares and can write: by default it is in a directory of this account's own on the container's disk,
# one per runtime dir so the workspaces of one container keep apart. The temporary directory is the one place every
# container has that is its own. Its name is predictable, so every reader and writer holds it to
# runtime_env_private_kernel_agent_env. The runtime dir is keyed absolute (runtime_env_absolute_path), as preflight
# keys it after `hyperloom` has made USER_DATA_PATH absolute.
runtime_env_derive_runtime_paths() {
  HYPERLOOM_RUNTIME_DIR="${HYPERLOOM_RUNTIME_DIR:-${USER_DATA_PATH}/runtime}"
  runtime_env_forget_legacy_kernel_agent_env
  if [ -z "${KERNEL_AGENT_ENV:-}" ]; then
    local key
    key="$(runtime_env_absolute_path "$HYPERLOOM_RUNTIME_DIR" | sha256sum)"
    KERNEL_AGENT_ENV="${TMPDIR:-/tmp}/hyperloom-$(id -u)-${key:0:16}/kernel-agent.env.sh"
  fi
  # Legacy variable kept for compatibility; open-source checkouts use install.sh's _open_source_root.
  HYPERLOOM_ROOT="${HYPERLOOM_ROOT:-${HYPERLOOM_RUNTIME_DIR}/source-mirrors}"
}

# Print $1 as an absolute path, without a newline: a leading ~ or ~name expanded as Python's os.path.expanduser
# expands it, a relative path put under the physical working directory (os.getcwd(), which is what `hyperloom` makes
# USER_DATA_PATH absolute against), and empty and . components dropped. Nothing else is resolved, neither .. nor a
# link. Preflight's copy is _absolute_path in cli/preflight_env.py; test_install_session_paths.py holds the two equal.
runtime_env_absolute_path() {
  local path="$1" name home="" found=0 rest out="" part
  if [ "${path:0:1}" = '~' ]; then
    name="${path%%/*}"
    name="${name:1}"
    if [ -z "$name" ] && [ "${HOME+set}" = set ]; then
      home="$HOME" found=1
    elif [[ -z "$name" || "$name" == *[!0-9]* ]] && home="$(_runtime_env_passwd_home "${name:-$(id -u)}")"; then
      # getent reads an all-digit key as a uid, where Python looks up a user of that name.
      found=1
    fi
    if [ "$found" -eq 1 ]; then
      home="${home%"${home##*[!/]}"}"
      path="${home}${path:$((${#name} + 1))}"
      path="${path:-/}"
    fi
  fi
  case "$path" in /*) ;; *) path="$(pwd -P)/$path" ;; esac
  rest="$path/"
  while [ -n "$rest" ]; do
    part="${rest%%/*}"
    rest="${rest#*/}"
    case "$part" in '' | .) ;; *) out+="/$part" ;; esac
  done
  printf '%s' "${out:-/}"
}

# The home directory of the account named, or numbered, $1 in the passwd database; status 1 when there is none.
_runtime_env_passwd_home() {
  local entry
  entry="$(getent passwd -- "$1" 2>/dev/null)" || return 1
  IFS=: read -r _ _ _ _ _ entry _ <<< "$entry"
  printf '%s' "$entry"
}

# Where kernel-agent.env.sh was written before it left the user's directory: the runtime dir, and the default one.
runtime_env_legacy_kernel_agent_env_paths() {
  [ -z "${HYPERLOOM_RUNTIME_DIR:-}" ] || printf '%s\n' "${HYPERLOOM_RUNTIME_DIR}/kernel-agent.env.sh"
  [ -z "${USER_DATA_PATH:-}" ] || [ "${HYPERLOOM_RUNTIME_DIR:-}" = "${USER_DATA_PATH}/runtime" ] ||
    printf '%s\n' "${USER_DATA_PATH}/runtime/kernel-agent.env.sh"
}

# A KERNEL_AGENT_ENV that names one of those places is what an older install left in .env or in a shell that sourced
# the file; the path is derived again rather than written to or read from there.
runtime_env_forget_legacy_kernel_agent_env() {
  local legacy
  [ -n "${KERNEL_AGENT_ENV:-}" ] || return 0
  while IFS= read -r legacy; do
    [ "${KERNEL_AGENT_ENV:-}" != "$legacy" ] || unset KERNEL_AGENT_ENV
  done < <(runtime_env_legacy_kernel_agent_env_paths)
}

# Remove the copy an older install left in those places, which other accounts' pods can rewrite; one that cannot be
# removed is named, never read.
runtime_env_remove_legacy_kernel_agent_env() {
  local legacy
  while IFS= read -r legacy; do
    [ -e "$legacy" ] || [ -L "$legacy" ] || continue
    rm -f -- "$legacy" 2>/dev/null ||
      printf '[runtime-env WARN] cannot remove the old shared copy of the kernel-agent env file %s; it is not read\n' \
        "$legacy" >&2
  done < <(runtime_env_legacy_kernel_agent_env_paths)
}

# kernel-agent.env.sh carries credentials and is sourced, and its default place is a predictable name in a
# temporary directory other accounts can write. So it is written and read only in a directory of this account's, not
# a link, that its group and others cannot write, and read only as a regular file of this account's, not a link, that
# its group and others cannot write. This is that rule's one owner: install.sh's writers call it with `write` (which
# makes the directory 0700 when missing; the file itself is renamed into place), and runtime_env_kernel_agent_env_path,
# behind the documented source line and the slurm job, calls it to read. Preflight reads the file in Python
# (_kernel_agent_env_problem in cli/preflight_env.py, with the same words); test_install_session_paths.py holds every
# reader to the same refusals. Status 1 and a refusal on stderr naming the entry and why, never what the file holds.
runtime_env_private_kernel_agent_env() {
  local file="$1" mode="${2:-read}" dir entry="" problem=""
  dir="$(dirname -- "$file")"
  if [ "$mode" = write ] && [ ! -e "$dir" ] && [ ! -L "$dir" ]; then
    # shellcheck disable=SC2174  # 0700 is for the file's own directory; the parents it may make are only ancestors
    mkdir -p -m 700 -- "$dir" || return 1
  fi
  if [ -e "$dir" ] || [ -L "$dir" ]; then
    entry="$dir" problem="$(_runtime_env_not_private "$dir" directory)"
  fi
  if [ -z "$problem" ] && [ "$mode" = read ] && { [ -e "$file" ] || [ -L "$file" ]; }; then
    entry="$file" problem="$(_runtime_env_not_private "$file" 'regular file')"
  fi
  [ -n "$problem" ] || return 0
  printf '[runtime-env ERROR] refusing the kernel-agent env file %s: %s %s; %s\n' "$file" "$entry" "$problem" \
    "it carries credentials and is sourced, so it is only written and read in a directory of this account's that no other account can write. Set KERNEL_AGENT_ENV to a file in such a directory on this container's own disk, or TMPDIR to such a directory, and re-run install.sh." >&2
  return 1
}

# Why the entry $1, which should be a $2 (`directory` or `regular file`), fails that rule; nothing when it passes.
_runtime_env_not_private() {
  local owner_mode uid
  uid="$(id -u)"
  if [ -L "$1" ]; then
    printf 'is a symbolic link'
  elif { [ "$2" = directory ] && [ ! -d "$1" ]; } || { [ "$2" != directory ] && [ ! -f "$1" ]; }; then
    printf 'is not a %s' "$2"
  elif ! owner_mode="$(stat -c '%u %a' -- "$1")"; then
    printf 'cannot be inspected'
  elif [ "${owner_mode% *}" != "$uid" ]; then
    printf 'belongs to uid %s, not to this account (uid %s)' "${owner_mode% *}" "$uid"
  elif (( 8#${owner_mode#* } & 8#022 )); then
    printf 'can be written by its group or others (mode %s)' "${owner_mode#* }"
  fi
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
# environment, or nothing when no workspace is named; refuse it (status 1, nothing printed) when another account
# could have written it (runtime_env_private_kernel_agent_env). Run it in a command substitution: it loads .env into
# the shell that runs it.
runtime_env_kernel_agent_env_path() {
  local REPO_ROOT="${1:-${REPO_ROOT:-}}"
  if [ -n "$REPO_ROOT" ]; then
    runtime_env_forget_stale_workspace
    load_dotenv_no_clobber || return 1
  fi
  runtime_env_forget_legacy_kernel_agent_env
  if [ -z "${KERNEL_AGENT_ENV:-}" ]; then
    [ -n "${HYPERLOOM_RUNTIME_DIR:-}${USER_DATA_PATH:-}" ] || return 0
    runtime_env_derive_runtime_paths
  fi
  runtime_env_private_kernel_agent_env "$KERNEL_AGENT_ENV" || return 1
  printf '%s\n' "$KERNEL_AGENT_ENV"
}
