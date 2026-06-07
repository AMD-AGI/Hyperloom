#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc. All rights reserved.

# Local Mode bootstrap for a fresh Hyperloom checkout.

set -euo pipefail

DRY_RUN=0
CHECK_ONLY=0

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_script_dir}/../.." && pwd)}"
# Capture whether USER_DATA_PATH was provided BEFORE applying the default so we
# can warn loudly on the silent fallback. ${VAR:+1} is empty when VAR is unset
# or empty, which is exactly the case the :- default below would absorb.
_user_data_was_set="${USER_DATA_PATH:+1}"
USER_DATA_PATH="${USER_DATA_PATH:-/workspace/hyperloom}"
if [ -z "${_user_data_was_set}" ]; then
  echo "[install WARN] USER_DATA_PATH not set; defaulting to /workspace/hyperloom. Set USER_DATA_PATH to persist artifacts under your data root." >&2
fi
HYPERLOOM_RUNTIME_DIR="${HYPERLOOM_RUNTIME_DIR:-${USER_DATA_PATH}/runtime}"
DEPS_ROOT_EXPLICIT=0
if [ -n "${HYPERLOOM_DEPS_ROOT:-}" ]; then
  DEPS_ROOT_EXPLICIT=1
else
  HYPERLOOM_DEPS_ROOT="${HYPERLOOM_RUNTIME_DIR}/source-mirrors"
fi
LOCAL_SETUP_ENV="${LOCAL_SETUP_ENV:-${HYPERLOOM_RUNTIME_DIR}/local-setup.env.sh}"

PRIMUS_CLAW_REPO="${PRIMUS_CLAW_REPO:-https://github.com/AMD-AGI/Primus-Claw.git}"
INFERENCEX_REPO="${INFERENCEX_REPO:-https://github.com/SemiAnalysisAI/InferenceX.git}"
TRACELENS_REPO="${TRACELENS_REPO:-https://github.com/AMD-AGI/TraceLens.git}"
TRACELENS_REF="${TRACELENS_REF:-c35c787ef31f0425fa0028a605ffc8c60a737c2c}"
# Preferred container-local checkout for the public repo when operators install
# TraceLens manually. The internal extension is private: Hyperloom keeps NO
# repo URL, ref, or default path for it. It is used only when an operator
# explicitly sets TRACELENS_INTERNAL_ROOT to an existing checkout
# (open-source-only otherwise). There is no separate on/off toggle.
TRACELENS_DEFAULT_ROOT="${TRACELENS_DEFAULT_ROOT:-/workspace/TraceLens}"

usage() {
  cat <<'EOF'
Usage: inference_optimizer/scripts/local_setup.sh [options]

Bootstraps Local Mode from only a Hyperloom checkout plus credentials.
It clones missing dependency repos and writes a local env file. Runtime
dependency checks and installation remain owned by the Cursor agent via
inference_optimizer/SKILL.md when optimization starts.

Options:
  --dry-run             Print planned actions without cloning or writing
  --check-only          Verify existing dependency checkouts, do not write env
  --deps-root PATH      Directory for dependency checkouts
  --session-dir PATH    Session directory; defaults to $USER_DATA_PATH or /workspace/hyperloom
  -h, --help            Show this help

Advanced env overrides:
  REPO_ROOT, USER_DATA_PATH, HYPERLOOM_DEPS_ROOT, LOCAL_SETUP_ENV,
  OOB_SRC, INFERENCEX_PATH, TRACELENS_ROOT, TRACELENS_INTERNAL_ROOT,
  PRIMUS_CLAW_REPO, INFERENCEX_REPO,
  TRACELENS_REPO, TRACELENS_REF,
  TRACELENS_INTERNAL_ROOT (path to an existing internal extension checkout;
    set to enable it, otherwise open-source-only)
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --check-only) CHECK_ONLY=1 ;;
    --deps-root)
      [ "$#" -ge 2 ] || { echo "[local-setup] ERROR: --deps-root requires PATH" >&2; exit 2; }
      shift
      HYPERLOOM_DEPS_ROOT="${1:-}"
      DEPS_ROOT_EXPLICIT=1
      ;;
    --session-dir)
      [ "$#" -ge 2 ] || { echo "[local-setup] ERROR: --session-dir requires PATH" >&2; exit 2; }
      shift
      USER_DATA_PATH="${1:-}"
      HYPERLOOM_RUNTIME_DIR="${USER_DATA_PATH}/runtime"
      if [ "$DEPS_ROOT_EXPLICIT" -eq 0 ]; then
        HYPERLOOM_DEPS_ROOT="${HYPERLOOM_RUNTIME_DIR}/source-mirrors"
      fi
      LOCAL_SETUP_ENV="${HYPERLOOM_RUNTIME_DIR}/local-setup.env.sh"
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[local-setup] ERROR: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

log() { echo "[local-setup] $*"; }
warn() { echo "[local-setup WARN] $*" >&2; }
die() { echo "[local-setup ERROR] $*" >&2; exit 1; }

run() {
  log "$*"
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    "$@"
  fi
}

shell_quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

write_export() {
  printf 'export %s=%s\n' "$1" "$(shell_quote "$2")"
}

credential_status() {
  local name="$1"
  local value="${!name:-}"
  if [ -n "$value" ]; then
    log "${name}: set"
    return 0
  fi
  if [ -f "${REPO_ROOT}/.env" ] && grep -Eq "^[[:space:]]*${name}=" "${REPO_ROOT}/.env"; then
    log "${name}: set in .env"
    return 0
  fi
  warn "${name}: not set"
}

ensure_git_available() {
  if ! command -v git >/dev/null 2>&1; then
    die "git is required to clone Hyperloom dependency repositories"
  fi
}

clone_or_update() {
  local name="$1"
  local repo="$2"
  local dest="$3"
  local ref="${4:-}"

  if [ -d "${dest}/.git" ]; then
    if [ "$CHECK_ONLY" -eq 1 ]; then
      log "${name}: existing checkout ${dest}"
      return 0
    fi
    run git -C "$dest" fetch --all --tags --prune
    if [ -n "$ref" ]; then
      run git -C "$dest" checkout "$ref"
      run git -C "$dest" pull --ff-only || true
    fi
    return 0
  fi

  if [ -e "$dest" ]; then
    die "${name} destination exists but is not a git checkout: ${dest}"
  fi
  if [ "$CHECK_ONLY" -eq 1 ]; then
    die "${name} checkout missing: ${dest}"
  fi

  log "${name}: clone ${repo} -> ${dest}"
  if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$(dirname "$dest")"
  fi
  run git clone "$repo" "$dest"
  if [ -n "$ref" ]; then
    run git -C "$dest" checkout "$ref"
  fi
}

_read_dotenv_var() {
  local name="$1"
  if [ -f "${REPO_ROOT}/.env" ]; then
    grep -E "^[[:space:]]*${name}=" "${REPO_ROOT}/.env" \
      | tail -n 1 \
      | sed -E "s/^[[:space:]]*${name}=//; s/^[\"' ]//; s/[\"' ]$//"
  fi
}

# .env.template historically used TRACELENS_*=\ as a visual "empty" hint;
# treat that (and whitespace-only values) as unset so local_setup clones
# under $HYPERLOOM_DEPS_ROOT instead of dying on a non-existent "\" path.
_is_placeholder_path_value() {
  local value="${1:-}"
  case "$value" in
    ""|'\'|'\\') return 0 ;;
  esac
  local trimmed="${value//[[:space:]]/}"
  [ -z "$trimmed" ] && return 0
  [ "$trimmed" = '\' ] && return 0
  return 1
}

_normalize_trace_env_roots() {
  local name value
  for name in TRACELENS_ROOT TRACELENS_INTERNAL_ROOT; do
    value="${!name:-}"
    if [ -n "$value" ] && _is_placeholder_path_value "$value"; then
      unset "$name"
    fi
  done
}

_resolve_existing_checkout() {
  local var_name="$1"
  local default_root="$2"
  local value=""

  _normalize_trace_env_roots
  value="${!var_name:-}"
  if [ -z "$value" ]; then
    value="$(_read_dotenv_var "$var_name" || true)"
  fi
  if _is_placeholder_path_value "$value"; then
    value=""
  fi
  if [ -z "$value" ] && [ -d "${default_root}/.git" ]; then
    value="${default_root}"
  fi
  if [ -n "$value" ]; then
    [ -d "$value" ] || die "${var_name} is set but does not exist: ${value}"
    log "${var_name}: using existing ${value}"
    printf -v "$var_name" '%s' "$value"
    export "$var_name"
    return 0
  fi
  return 1
}

resolve_tracelens() {
  if _resolve_existing_checkout TRACELENS_ROOT "$TRACELENS_DEFAULT_ROOT"; then
    :
  else
    TRACELENS_ROOT="${HYPERLOOM_DEPS_ROOT}/TraceLens"
    clone_or_update "TraceLens" "$TRACELENS_REPO" "$TRACELENS_ROOT" "$TRACELENS_REF"
    export TRACELENS_ROOT
    log "TRACELENS_ROOT: ${TRACELENS_ROOT}"
  fi

  # TraceLens-internal is opt-in: resolved only when TRACELENS_INTERNAL_ROOT is
  # explicitly provided (env or .env). With no value Hyperloom stays on the
  # open-source-only setup (no roofline gap / MI355+ MAF). No separate toggle.
  _normalize_trace_env_roots
  local internal_root="${TRACELENS_INTERNAL_ROOT:-}"
  if [ -z "$internal_root" ]; then
    internal_root="$(_read_dotenv_var TRACELENS_INTERNAL_ROOT || true)"
  fi
  if _is_placeholder_path_value "$internal_root"; then
    internal_root=""
  fi
  if [ -z "$internal_root" ]; then
    log "TraceLens-internal: not requested (open-source-only; set TRACELENS_INTERNAL_ROOT to enable)"
    return 0
  fi

  # Internal is never cloned by Hyperloom (no URL is kept). The operator must
  # provide an existing checkout; a missing path falls back to open-source-only.
  TRACELENS_INTERNAL_ROOT="$internal_root"
  if [ ! -d "$TRACELENS_INTERNAL_ROOT" ]; then
    warn "TRACELENS_INTERNAL_ROOT set but not found: ${TRACELENS_INTERNAL_ROOT}; falling back to open-source-only (provide an existing internal checkout to enable)"
    unset TRACELENS_INTERNAL_ROOT
    return 0
  fi
  export TRACELENS_INTERNAL_ROOT
  log "TRACELENS_INTERNAL_ROOT: using existing ${TRACELENS_INTERNAL_ROOT}"
}

resolve_oob_src() {
  if [ -n "${OOB_SRC:-}" ]; then
    [ -d "$OOB_SRC" ] || die "OOB_SRC is set but does not exist: ${OOB_SRC}"
    log "OOB_SRC: using existing ${OOB_SRC}"
    return 0
  fi

  local root="${HYPERLOOM_DEPS_ROOT}/Primus-Claw"
  clone_or_update "Primus-Claw" "$PRIMUS_CLAW_REPO" "$root" ""
  OOB_SRC="${root}/OOB"
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    [ -d "$OOB_SRC" ] || die "Primus-Claw checkout does not contain OOB/: ${OOB_SRC}"
  fi
  export OOB_SRC
  log "OOB_SRC: ${OOB_SRC}"
}

resolve_inferencex() {
  if [ -n "${INFERENCEX_PATH:-}" ]; then
    [ -d "$INFERENCEX_PATH" ] || die "INFERENCEX_PATH is set but does not exist: ${INFERENCEX_PATH}"
    log "INFERENCEX_PATH: using existing ${INFERENCEX_PATH}"
    return 0
  fi

  INFERENCEX_PATH="${HYPERLOOM_DEPS_ROOT}/InferenceX"
  clone_or_update "InferenceX" "$INFERENCEX_REPO" "$INFERENCEX_PATH" ""
  export INFERENCEX_PATH
  log "INFERENCEX_PATH: ${INFERENCEX_PATH}"
}

write_local_env() {
  if [ "$DRY_RUN" -eq 1 ] || [ "$CHECK_ONLY" -eq 1 ]; then
    log "would write local env: ${LOCAL_SETUP_ENV}"
    return 0
  fi

  mkdir -p "$(dirname "$LOCAL_SETUP_ENV")"
  {
    echo '#!/bin/sh'
    echo '# Generated by inference_optimizer/scripts/local_setup.sh'
    write_export REPO_ROOT "$REPO_ROOT"
    write_export USER_DATA_PATH "$USER_DATA_PATH"
    write_export HYPERLOOM_RUNTIME_DIR "$HYPERLOOM_RUNTIME_DIR"
    write_export HYPERLOOM_DEPS_ROOT "$HYPERLOOM_DEPS_ROOT"
    write_export OOB_SRC "$OOB_SRC"
    write_export INFERENCEX_PATH "$INFERENCEX_PATH"
    write_export TRACELENS_ROOT "$TRACELENS_ROOT"
    if [ -n "${TRACELENS_INTERNAL_ROOT:-}" ]; then
      write_export TRACELENS_INTERNAL_ROOT "$TRACELENS_INTERNAL_ROOT"
      write_export TL_EXTENSION "TraceLens_internal"
    fi
  } > "$LOCAL_SETUP_ENV"
  chmod 600 "$LOCAL_SETUP_ENV"
  log "wrote ${LOCAL_SETUP_ENV}"
}

print_next_steps() {
  local quoted_env quoted_user_data
  quoted_env="$(shell_quote "$LOCAL_SETUP_ENV")"
  quoted_user_data="$(shell_quote "$USER_DATA_PATH")"
  cat <<EOF
[local-setup] local setup complete

Open this folder in Cursor as the workspace:
  ${REPO_ROOT}

Paste this into Cursor Chat and fill in your workload:

@inference_optimizer/SKILL.md

Optimize inference for this workload:
- Model: /path/to/your/model
- Framework: sglang
- GPU: MI300X
- TP: 8
- CONC: 64
- ISL: 1024
- OSL: 1024
- Precision: bf16
- Goal: improve throughput by at least 10%
- Budget: 24 hours

Before launch, run exactly:
\`\`\`bash
source ${quoted_env}
export USER_DATA_PATH=${quoted_user_data}
\`\`\`

Requirements:
1. Report the session ID, log path, PID, and initial health check result.
2. Monitor the process every 300s until the optimization is complete or failed.
EOF
}

main() {
  log "REPO_ROOT=${REPO_ROOT}"
  log "USER_DATA_PATH=${USER_DATA_PATH}"
  log "HYPERLOOM_DEPS_ROOT=${HYPERLOOM_DEPS_ROOT}"
  credential_status SAFE_API_KEY
  credential_status OPENAI_BASE_URL
  credential_status CURSOR_API_KEY || true

  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    mkdir -p "$HYPERLOOM_DEPS_ROOT" "$HYPERLOOM_RUNTIME_DIR"
  fi

  ensure_git_available
  resolve_oob_src
  resolve_inferencex
  resolve_tracelens
  write_local_env

  print_next_steps
}

main "$@"
