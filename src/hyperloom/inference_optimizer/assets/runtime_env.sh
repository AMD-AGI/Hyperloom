#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Source this asset, then load missing exports from $REPO_ROOT/.env.
# A .env another account can write (runtime_env_dotenv_not_private) never names the workspace: that account would drop
# the caller's USER_DATA_PATH, PYTHONPATH and runtime paths by writing one line, and the .env lends none of them back.
setup_dotenv_is_authoritative() {
  [ -f "$REPO_ROOT/.env" ] || return 1
  [ -z "$(runtime_env_dotenv_not_private "$REPO_ROOT/.env")" ] || return 1
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

# Which workspace a run uses, and so which kernel-agent.env.sh it gets, follows one rule, kept here: install.sh sets
# its paths through runtime_env_forget_stale_workspace and runtime_env_derive_runtime_paths, and the documented source
# line and the slurm job ask runtime_env_kernel_agent_env_path for the file install.sh wrote. Preflight does not call
# this file: cli/preflight_env.py mirrors the rule in Python (_workspace_selection, _kernel_agent_env_candidate), and
# test_install_session_paths.py holds both to the same answer.
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
# behind the documented source line and the slurm job, calls it to read. A reader also refuses a directory or file that
# does not exist: a path printed for it could be created by another account before the caller sources it. Preflight
# reads the file in Python (_kernel_agent_env_problem in cli/preflight_env.py, with the same words);
# test_install_session_paths.py holds every reader to the same refusals. Status 1 and why on stderr, naming the entry,
# never what the file holds.
runtime_env_private_kernel_agent_env() {
  local file="$1" mode="${2:-read}" dir
  dir="$(dirname -- "$file")"
  if [ "$mode" = write ] && [ ! -e "$dir" ] && [ ! -L "$dir" ]; then
    # shellcheck disable=SC2174  # 0700 is for the file's own directory; the parents it may make are only ancestors
    mkdir -p -m 700 -- "$dir" || return 1
  fi
  _runtime_env_private_entry "$file" "$dir" directory || return 1
  [ "$mode" = read ] || return 0
  _runtime_env_private_entry "$file" "$file" 'regular file'
}

# Status 0 when $2, the kernel-agent env file $1 or its directory, exists and is a $3 that passes the rule; else
# status 1 and why on stderr.
_runtime_env_private_entry() {
  local problem
  if [ ! -e "$2" ] && [ ! -L "$2" ]; then
    printf '[runtime-env ERROR] no kernel-agent env file at %s: %s does not exist; %s\n' "$1" "$2" \
      "install.sh writes it on this container's own disk, so run install.sh in this container first, with the same workspace." >&2
    return 1
  fi
  problem="$(_runtime_env_not_private "$2" "$3")"
  [ -n "$problem" ] || return 0
  printf '[runtime-env ERROR] refusing the kernel-agent env file %s: %s %s; %s\n' "$1" "$2" "$problem" \
    "it carries credentials and is sourced, so it is only written and read in a directory of this account's that no other account can write. Set KERNEL_AGENT_ENV to a file in such a directory on this container's own disk, or TMPDIR to such a directory, and re-run install.sh." >&2
  return 1
}

# Why the entry $1, which should be a $2 (`directory` or `regular file`), fails that rule; nothing when it passes.
# $3, when given, is one more uid the entry may belong to (runtime_env_dotenv_not_private: root on a user's checkout).
_runtime_env_not_private() {
  local owner_mode uid
  uid="$(id -u)"
  if [ -L "$1" ]; then
    printf 'is a symbolic link'
  elif { [ "$2" = directory ] && [ ! -d "$1" ]; } || { [ "$2" != directory ] && [ ! -f "$1" ]; }; then
    printf 'is not a %s' "$2"
  elif ! owner_mode="$(stat -c '%u %a' -- "$1")"; then
    printf 'cannot be inspected'
  elif [ "${owner_mode% *}" != "$uid" ] && [ "${owner_mode% *}" != "${3:-$uid}" ]; then
    printf 'belongs to uid %s, not to this account (uid %s)' "${owner_mode% *}" "$uid"
  elif (( 8#${owner_mode#* } & 8#022 )); then
    printf 'can be written by its group or others (mode %s)' "${owner_mode#* }"
  fi
}

# The credentials install_kernel_tools.sh write_env_file keeps in a private checkout's .env. A checkout another account
# can write (runtime_env_dotenv_not_private), such as one bind-mounted from shared storage, keeps none there: they go to
# this container's kernel-agent env file alone, write_env_file removes any left in .env
# (runtime_env_remove_dotenv_credentials), and a reader that took them from .env takes them from that file
# (runtime_env_dotenv_credentials_elsewhere, runtime_env_load_kernel_agent_credentials, behind install.sh's
# preflight_load_dotenv). Preflight's copy of this list is _DOTENV_CREDENTIAL_KEYS in cli/preflight_env.py;
# test_dotenv_credentials_private.py holds the two equal.
RUNTIME_ENV_DOTENV_CREDENTIAL_KEYS="ANTHROPIC_BASE_URL ANTHROPIC_API_KEY ANTHROPIC_CUSTOM_HEADERS CLAUDE_CODE_OAUTH_TOKEN"
# The retired spellings of those credentials (with the DeepSeek configuration that normalizes into them). A .env that
# keeps no credentials (runtime_env_dotenv_credentials_elsewhere) lends none of these either: load_dotenv_no_clobber
# skips them there, so a gateway URL, headers or key another account writes into a shared .env never pairs with this
# account's key, and write_env_file removes them with the keys above. Preflight's copy is
# _DOTENV_RETIRED_CREDENTIAL_KEYS in cli/preflight_env.py.
RUNTIME_ENV_DOTENV_RETIRED_CREDENTIAL_KEYS="ANTHROPIC_AUTH_TOKEN DEEPSEEK_API_KEY DEEPSEEK_BASE_URL DEEPSEEK_MODEL SAFE_API_KEY AMD_LLM_API_KEY LLM_GATEWAY_KEY GEAK_API_KEY"

# Why another account can write the checkout's .env $1, or nothing when this account alone can: the checkout directory
# is held to runtime_env_private_kernel_agent_env's rule for a directory and the file, when there is one, to its rule
# for a regular file (_runtime_env_not_private: a link, another account's, or writable by its group or others).
# The checkout is judged where it is (cd -P), so one reached through a link -- install.sh keeps the logical path,
# preflight resolves its own -- gets the same answer from both; .env itself is never followed. Root acting on a
# checkout one other uid owns (the documented Docker flow: a container's root on the user's bind-mounted checkout)
# accepts that uid as its own: root can already write it, and that uid already writes the code root runs from it.
# Preflight's copy is _dotenv_not_private in cli/preflight_env.py, with the same words.
runtime_env_dotenv_not_private() {
  local dotenv="$1" dir problem resolved owner=""
  dir="$(dirname -- "$dotenv")"
  if [ -d "$dir" ] && resolved="$(cd -P -- "$dir" 2>/dev/null && pwd -P)"; then
    dir="$resolved"
    dotenv="$dir/$(basename -- "$dotenv")"
    if [ "$(id -u)" = 0 ]; then
      owner="$(stat -c '%u' -- "$dir" 2>/dev/null)" || owner=""
    fi
  fi
  problem="$(_runtime_env_not_private "$dir" directory "$owner")"
  if [ -n "$problem" ]; then
    printf '%s %s' "$dir" "$problem"
  elif [ -e "$dotenv" ] || [ -L "$dotenv" ]; then
    problem="$(_runtime_env_not_private "$dotenv" 'regular file' "$owner")"
    [ -z "$problem" ] || printf '%s %s' "$dotenv" "$problem"
  fi
}

# Why this checkout's credentials are in the kernel-agent env file rather than in its .env $1, or nothing when .env
# keeps them: another account can write .env (runtime_env_dotenv_not_private), or it carries
# HYPERLOOM_DOTENV_CREDENTIALS=kernel-agent-env. write_env_file leaves that line in a .env it kept them out of, because
# its own writes turn a .env another account could write in a private directory into one of this account's. Ask once
# load_dotenv_no_clobber has loaded .env. Preflight's copy is _dotenv_credentials_elsewhere in cli/preflight_env.py.
runtime_env_dotenv_credentials_elsewhere() {
  local not_private
  not_private="$(runtime_env_dotenv_not_private "$1")"
  if [ -n "$not_private" ]; then
    printf 'another account can write it (%s)' "$not_private"
  elif [ "${HYPERLOOM_DOTENV_CREDENTIALS:-}" = kernel-agent-env ]; then
    printf 'it carries HYPERLOOM_DOTENV_CREDENTIALS=kernel-agent-env'
  fi
}

# Remove from the checkout's .env $1 every assignment of RUNTIME_ENV_DOTENV_CREDENTIAL_KEYS,
# RUNTIME_ENV_DOTENV_RETIRED_CREDENTIAL_KEYS and the keys $2..., with the continuation lines of a quoted value, and
# keep every other byte. The lines are told apart as load_dotenv_no_clobber reads them. What is kept is written beside
# the file (mktemp in its directory), given its mode and renamed over it, so a reader sees the old file or the new one,
# never part of one. A link is always replaced, never written through: by a regular file of what it kept, or, for a
# link to anything but a regular file, removed, so no later write of .env lands in what it points to. Nothing is
# written when there is nothing to remove from a regular file (runtime_env_keep_lent_dotenv_lines always writes).
# Status 1 and why on stderr when it cannot be done, a .env this account cannot read or one that is not a regular file
# included.
runtime_env_remove_dotenv_credentials() {
  _runtime_env_rewrite_dotenv 0 "$@"
}

# runtime_env_remove_dotenv_credentials for a .env another account can write (runtime_env_dotenv_not_private), which
# also removes every assignment load_dotenv_no_clobber would not lend from it (runtime_env_dotenv_lent), with the
# continuation lines of a quoted value, and every other line but a blank one or a comment. The install's own writes
# replace such a .env with a file of this account's, which a later reader may find private: what another account
# planted there (NODE_OPTIONS, OPENAI_BASE_URL, LD_PRELOAD) must be gone before, or the rewrite lends it from then on.
# The file is always replaced, even when nothing is removed, by one of mode 0600 that no other account can append to
# before the install's next write copies it; as root, it keeps the owner of a .env that the checkout's owner owns, so
# the user's own .env stays the user's. Call it before the first write of such a .env.
runtime_env_keep_lent_dotenv_lines() {
  _runtime_env_rewrite_dotenv 1 "$@"
}

# The two above: $1 is 1 to keep only what a .env another account can write lends, 0 to remove the credentials alone.
_runtime_env_rewrite_dotenv() {
  local lent_only="$1" dotenv="$2" keys mode=600 tmp line text key value nl quote="" drop=0 removed=0 failed=0 REPLY
  local owner dir_owner
  shift 2
  keys=" $RUNTIME_ENV_DOTENV_CREDENTIAL_KEYS $RUNTIME_ENV_DOTENV_RETIRED_CREDENTIAL_KEYS $* "
  if [ -L "$dotenv" ] && [ ! -f "$dotenv" ]; then
    rm -f -- "$dotenv" && return 0
    printf '[runtime-env ERROR] cannot remove the link %s\n' "$dotenv" >&2
    return 1
  fi
  if [ ! -f "$dotenv" ]; then
    [ -e "$dotenv" ] || return 0
    printf '[runtime-env ERROR] cannot remove the credentials from %s: it is not a regular file\n' "$dotenv" >&2
    return 1
  fi
  [ -L "$dotenv" ] || mode="$(stat -c '%a' -- "$dotenv")" || return 1
  [ "$lent_only" -eq 0 ] || mode=600
  tmp="$(mktemp "$(dirname -- "$dotenv")/.env.XXXXXX")" || return 1
  if [ "$lent_only" -eq 1 ] && [ "$(id -u)" = 0 ] && [ ! -L "$dotenv" ]; then
    owner="$(stat -c '%u:%g' -- "$dotenv")" && dir_owner="$(stat -c '%u' -- "$(dirname -- "$dotenv")")" || failed=1
    if [ "$failed" -eq 0 ] && [ "${owner%%:*}" = "$dir_owner" ]; then chown -- "$owner" "$tmp" || failed=1; fi
  fi
  while :; do
    nl=$'\n'
    if ! IFS= read -r line; then
      [ -n "$line" ] || break
      nl=''
    fi
    if [ -n "$quote" ]; then
      ! _runtime_env_quote_ends "$quote" "$line" || quote=""
    else
      drop=0
      text="${line#"${line%%[![:space:]]*}"}"
      text="${text%$'\r'}"
      text="${text#export }"
      if [[ "$text" == *=* && "$text" != \#* ]]; then
        key="${text%%=*}"
        key="${key%"${key##*[![:space:]]}"}"
        [[ "$keys" != *" $key "* ]] || drop=1
        text="${text#*=}"
        text="${text#"${text%%[![:space:]]*}"}"
        value="${text%"${text##*[![:space:]]}"}"
        case "$text" in
          \"* | \'*)
            if _runtime_env_quote_ends "${text:0:1}" "${text:1}"; then
              # The value as load_dotenv_no_clobber reads it; one it cannot read on this line alone is not lent.
              value=$'\n'
              if _runtime_env_read_quoted_value "$text" < /dev/null 2>/dev/null; then value="$REPLY"; fi
            else
              quote="${text:0:1}"
            fi ;;
        esac
        if [ "$lent_only" -eq 1 ] && { [ -n "$quote" ] || ! runtime_env_dotenv_lent "$key" "$value"; }; then
          drop=1
        fi
      elif [ "$lent_only" -eq 1 ] && [ -n "$text" ] && [[ "$text" != \#* ]]; then
        drop=1
      fi
    fi
    if [ "$drop" -eq 1 ]; then
      removed=$((removed + 1))
    else
      printf '%s%s' "$line" "$nl" || failed=1
    fi
  done < "$dotenv" > "$tmp" || failed=1
  if [ "$failed" -eq 1 ]; then
    printf '[runtime-env ERROR] cannot remove the credentials from %s: it could not be read, or what it keeps written\n' \
      "$dotenv" >&2
  elif [ -n "$quote" ]; then
    printf '[runtime-env ERROR] cannot remove the credentials from %s: a quoted value in it never ends\n' "$dotenv" >&2
  elif { [ "$removed" -eq 0 ] && [ ! -L "$dotenv" ] && [ "$lent_only" -eq 0 ]; } || { chmod -- "$mode" "$tmp" && mv -fT -- "$tmp" "$dotenv"; }; then
    rm -f -- "$tmp"
    return 0
  fi
  rm -f -- "$tmp"
  return 1
}

# Status 0 when $2, the rest of a line, ends a value quoted with $1 as _runtime_env_read_quoted_value ends one: its last
# character other than a blank is that quote, not escaped (for a double quote) by an odd run of backslashes before it.
# Status 1 while the value runs on to the next line.
_runtime_env_quote_ends() {
  local text="${2%"${2##*[![:space:]]}"}" run
  [ -n "$text" ] && [ "${text: -1}" = "$1" ] || return 1
  [ "$1" = '"' ] || return 0
  text="${text%?}"
  run="${text##*[!\\]}"
  [ $((${#run} % 2)) -eq 0 ]
}

# Export, as data and only where the variable is unset or empty, the RUNTIME_ENV_DOTENV_CREDENTIAL_KEYS lines of the
# kernel-agent env file $1 that install.sh wrote in this container: what a checkout's .env no longer carries once
# another account can write it. Nothing when there is no such file; status 1, with runtime_env_private_kernel_agent_env's
# refusal on stderr, for one another account could have written, which is not read. RUNTIME_ENV_CREDENTIALS_LOADED
# counts what was exported. Preflight's copy is _kernel_agent_env_credentials in cli/preflight_env.py.
runtime_env_load_kernel_agent_credentials() {
  local file="$1" line key value REPLY
  RUNTIME_ENV_CREDENTIALS_LOADED=0
  [ -e "$file" ] || [ -L "$file" ] || return 0
  runtime_env_private_kernel_agent_env "$file" || return 1
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#"${line%%[![:space:]]*}"}"
    line="${line#export }"
    key="${line%%=*}"
    key="${key%"${key##*[![:space:]]}"}"
    [[ "$line" == *=* && " $RUNTIME_ENV_DOTENV_CREDENTIAL_KEYS " == *" $key "* ]] || continue
    value="${line#*=}"
    value="${value#"${value%%[![:space:]]*}"}"
    case "$value" in
      \"* | \'*)
        _runtime_env_read_quoted_value "$value" || return 1
        value="$REPLY" ;;
      *) value="${value%"${value##*[![:space:]]}"}" ;;
    esac
    [ -n "$value" ] && [ -z "${!key:-}" ] || continue
    export "$key=$value"
    RUNTIME_ENV_CREDENTIALS_LOADED=$((RUNTIME_ENV_CREDENTIALS_LOADED + 1))
  done < "$file"
}

# Print the lines that export the credential $1 with the value $2 from a sourced kernel-agent env file as a fallback,
# not an override: a value already in the environment is kept, and a different one is announced, never printed. The
# same lines as install_kernel_tools.sh write_env_file's _emit_credential_fallback; test_dotenv_allow_list.py holds
# the two equal.
runtime_env_emit_credential_fallback() {
  echo "if [ -n \"\${${1}:-}\" ]; then"
  echo "  [ \"\${${1}}\" = '${2}' ] || \\"
  echo "    echo '[kernel-agent] ${1} differs from the install-time snapshot in this file; keeping the value already in the environment (re-run install.sh to refresh it)' >&2"
  echo "else"
  echo "  export ${1}='${2}'"
  echo "fi"
}

# Put the credentials KEY=VALUE... into this container's kernel-agent env file $1, for a checkout whose .env keeps
# none (install_baremetal.sh, which runs before install.sh writes the file whole). Each KEY given loses what the file
# held for it, and is written (runtime_env_emit_credential_fallback) when its VALUE is not empty; every other line of a
# file that passes runtime_env_private_kernel_agent_env stays, and one that does not is not read. The file is written
# beside itself and renamed into place, as write_env_file writes it. Status 1 and why on stderr when it cannot be.
runtime_env_write_kernel_agent_credentials() {
  local file="$1" dir written line pair keys="" skip="" key
  shift
  for pair in "$@"; do keys+=" ${pair%%=*}"; done
  keys+=" "
  runtime_env_private_kernel_agent_env "$file" write || return 1
  dir="$(dirname -- "$file")"
  written="$(mktemp "${dir}/.kernel-agent.env.XXXXXX")" || return 1
  {
    if { [ -e "$file" ] || [ -L "$file" ]; } && runtime_env_private_kernel_agent_env "$file" 2>/dev/null; then
      while IFS= read -r line || [ -n "$line" ]; do
        if [ -n "$skip" ]; then
          [ "$line" != 'fi' ] || skip=""
          continue
        fi
        key="${line#"${line%%[![:space:]]*}"}"
        # shellcheck disable=SC2016 # the literal first line runtime_env_emit_credential_fallback writes
        if [[ "$key" == 'if [ -n "${'*':-}" ]; then' ]]; then
          # shellcheck disable=SC2016 # the same literal
          key="${key#'if [ -n "${'}"
          key="${key%%:-*}"
          [[ "$keys" != *" $key "* ]] || { skip=1; continue; }
        elif [[ "$key" == export\ *=* ]]; then
          key="${key#export }"
          key="${key%%=*}"
          [[ "$keys" != *" $key "* ]] || continue
        fi
        printf '%s\n' "$line"
      done < "$file"
    else
      echo '#!/bin/sh'
      echo '# kernel-agent runtime env (credentials from install_baremetal.sh; install.sh regenerates the rest)'
    fi
    for pair in "$@"; do
      [ -z "${pair#*=}" ] || runtime_env_emit_credential_fallback "${pair%%=*}" "${pair#*=}"
    done
  } > "$written" || { rm -f -- "$written"; return 1; }
  mv -f -- "$written" "$file" || { rm -f -- "$written"; return 1; }
}

# What a .env another account can write (runtime_env_dotenv_not_private) lends: each NAME=VALUE,VALUE... is a name and
# the values it may carry there, DOTENV_NOT_PRIVATE_LENT_VALUES in hyperloom/common/env_safety.py, which says why so
# few. test_dotenv_allow_list.py holds the two equal and runtime_env_dotenv_lent to
# is_lent_not_private_dotenv_assignment.
RUNTIME_ENV_DOTENV_NOT_PRIVATE_LENT="FRAMEWORK=atom,custom,sglang,vllm,xdit HF_DATASETS_OFFLINE=0,1 HF_HUB_OFFLINE=0,1 \
HYPERLOOM_DOTENV_CREDENTIALS=kernel-agent-env \
HYPERLOOM_FRAMEWORK_ENV=isolated,shared HYPERLOOM_LLM_MODE=anthropic,deepseek,openai HYPERLOOM_RUN_MODE=baremetal,docker"

# Status 0 when a .env another account can write lends the variable named $1 (any case, as preflight compares it) with
# the value $2, compared whole. The name is held to ASCII before it is folded: under a UTF-8 locale ${1^^} turns a long
# s into S, and the name, written back as given, would then be one load_dotenv_no_clobber refuses whole. The letters are
# spelled out because a range or class can take in other letters in some locales.
RUNTIME_ENV_ASCII_NAME_RE='^[ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_][ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_]*$'
runtime_env_dotenv_lent() {
  local upper entry
  [[ "$1" =~ $RUNTIME_ENV_ASCII_NAME_RE ]] || return 1
  upper="${1^^}"
  for entry in $RUNTIME_ENV_DOTENV_NOT_PRIVATE_LENT; do
    [ "${entry%%=*}" = "$upper" ] || continue
    [[ ",${entry#*=}," == *",$2,"* && "$2" != *,* ]]
    return
  done
  return 1
}

# Export, where unset or empty, the assignments of $REPO_ROOT/.env. A .env that keeps no credentials
# (runtime_env_dotenv_credentials_elsewhere: another account can write it, or it carries
# HYPERLOOM_DOTENV_CREDENTIALS=kernel-agent-env) lends none: its RUNTIME_ENV_DOTENV_CREDENTIAL_KEYS and
# RUNTIME_ENV_DOTENV_RETIRED_CREDENTIAL_KEYS lines are skipped and counted in DOTENV_SKIPPED_CREDENTIAL_COUNT, and
# the credentials come from the shell or this container's kernel-agent env file alone. A .env another account can write
# lends, besides, only the names and values runtime_env_dotenv_lent allows: every other line is skipped, its name (never
# its value) in DOTENV_REFUSED_KEYS and in one warning on stderr, so another account's LD_PRELOAD, NODE_OPTIONS,
# PYTHONPATH, HTTPS_PROXY, GEAK_REPO or OPENAI_BASE_URL never reaches this account's shell. A private .env is read as before: install_kernel_tools.sh and
# install_baremetal.sh themselves keep names there (PYTHONPATH, MAGPIE_PYTHON, TRACELENS_INTERNAL_ROOT, TL_EXTENSION)
# that preflight's list does not carry and the documented shell flows read back. Preflight's copy of both skips is in
# _load_dotenv_fallback in cli/preflight_env.py.
load_dotenv_no_clobber() {
  DOTENV_LOADED_COUNT=0
  # shellcheck disable=SC2034 # read by install.sh preflight_load_dotenv
  DOTENV_SKIPPED_CREDENTIAL_COUNT=0
  DOTENV_REFUSED_KEYS=""
  [ -f "$REPO_ROOT/.env" ] || return 0
  local loaded=0 skipped=0 authoritative=0 readonly_root=0 caller_root=0 skip_credentials=0 not_private refused=""
  not_private="$(runtime_env_dotenv_not_private "$REPO_ROOT/.env")"
  [ -z "$not_private" ] || skip_credentials=1
  [ "${HYPERLOOM_DOTENV_CREDENTIALS:-}" != kernel-agent-env ] || skip_credentials=1
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
        if [ "$key" = HYPERLOOM_DOTENV_CREDENTIALS ]; then
          [ "$value" != kernel-agent-env ] || skip_credentials=1
          continue
        fi
        [ "$key" = HYPERLOOM_RUN_MODE ] || continue
        if [ -n "$not_private" ] && ! runtime_env_dotenv_lent "$key" "$value"; then
          [[ " $refused " == *" $key "* ]] || refused+="${refused:+ }$key"
          continue
        fi
      else
        [ "$key" = HYPERLOOM_RUN_MODE ] && continue
        if [ "$skip_credentials" -eq 1 ] &&
           [[ " $RUNTIME_ENV_DOTENV_CREDENTIAL_KEYS $RUNTIME_ENV_DOTENV_RETIRED_CREDENTIAL_KEYS " == *" $key "* ]]; then
          skipped=$((skipped + 1))
          continue
        fi
        if [ -n "$not_private" ] && ! runtime_env_dotenv_lent "$key" "$value"; then
          [[ " $refused " == *" $key "* ]] || refused+="${refused:+ }$key"
          continue
        fi
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
  # shellcheck disable=SC2034 # read by install.sh preflight_load_dotenv
  DOTENV_SKIPPED_CREDENTIAL_COUNT="$skipped"
  # shellcheck disable=SC2034 # for the caller; the warning below names them too
  DOTENV_REFUSED_KEYS="$refused"
  # Said once per .env and set of names, though install.sh loads .env twice.
  if [ -n "$refused" ] && [ "${_RUNTIME_ENV_DOTENV_REFUSED_SAID:-}" != "$REPO_ROOT/.env:$refused" ]; then
    _RUNTIME_ENV_DOTENV_REFUSED_SAID="$REPO_ROOT/.env:$refused"
    printf '[runtime-env WARN] not loaded from %s, which another account can write (%s): %s; export any of them you need in this shell\n' \
      "$REPO_ROOT/.env" "$not_private" "$refused" >&2
  fi
  return 0
}

# Print the kernel-agent.env.sh install.sh wrote for the checkout $1 (default $REPO_ROOT) under the caller's
# environment. Status 1 and nothing printed, with why on stderr, when no workspace is named, when install.sh has not
# written the file in this container, or when another account could have written it
# (runtime_env_private_kernel_agent_env); a caller sources the path only on status 0:
#   kernel_agent_env_file="$(. .../runtime_env.sh; runtime_env_kernel_agent_env_path)" && . "$kernel_agent_env_file"
# Run it in a command substitution: it loads .env into the shell that runs it.
runtime_env_kernel_agent_env_path() {
  local REPO_ROOT="${1:-${REPO_ROOT:-}}"
  if [ -n "$REPO_ROOT" ]; then
    runtime_env_forget_stale_workspace
    load_dotenv_no_clobber || return 1
  fi
  runtime_env_forget_legacy_kernel_agent_env
  if [ -z "${KERNEL_AGENT_ENV:-}" ]; then
    if [ -z "${HYPERLOOM_RUNTIME_DIR:-}${USER_DATA_PATH:-}" ]; then
      printf '%s\n' "[runtime-env ERROR] no kernel-agent env file to load: none of KERNEL_AGENT_ENV, HYPERLOOM_RUNTIME_DIR, USER_DATA_PATH is set; export USER_DATA_PATH and run install.sh in this container first." >&2
      return 1
    fi
    runtime_env_derive_runtime_paths
  fi
  runtime_env_private_kernel_agent_env "$KERNEL_AGENT_ENV" || return 1
  printf '%s\n' "$KERNEL_AGENT_ENV"
}
