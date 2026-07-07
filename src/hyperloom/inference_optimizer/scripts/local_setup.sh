#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc. All rights reserved.

# Local Mode bootstrap for a fresh Hyperloom checkout.

set -euo pipefail

DRY_RUN=0
CHECK_ONLY=0

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_script_dir}/../../../.." && pwd)}"
# Capture whether USER_DATA_PATH was provided BEFORE applying the default so we
# can warn loudly on the silent fallback. ${VAR:+1} is empty when VAR is unset
# or empty, which is exactly the case the :- default below would absorb.
_user_data_was_set="${USER_DATA_PATH:+1}"
USER_DATA_PATH="${USER_DATA_PATH:-/workspace/hyperloom}"
if [ -z "${_user_data_was_set}" ]; then
  echo "[install WARN] USER_DATA_PATH not set; defaulting to /workspace/hyperloom. Set USER_DATA_PATH to persist artifacts under your data root." >&2
fi
HYPERLOOM_RUNTIME_DIR="${HYPERLOOM_RUNTIME_DIR:-${USER_DATA_PATH}/runtime}"
# Pod-local base for auto-cloned open-source deps, decoupled from USER_DATA_PATH
# so a shared (WekaFS) workspace root never collocates concurrent pods' checkouts.
# Default is a pod-internal, non-ephemeral dir (NOT /tmp): a tmp-reaper wiping
# /tmp mid-run left TRACELENS_ROOT dangling and broke trace_analyze (#722).
HYPERLOOM_DEPS_ROOT="${HYPERLOOM_DEPS_ROOT:-${HYPERLOOM_OPEN_SOURCE_ROOT:-/opt/hyperloom/open-source-repos}}"
_open_source_root="${HYPERLOOM_DEPS_ROOT}"
LOCAL_SETUP_ENV="${LOCAL_SETUP_ENV:-${HYPERLOOM_RUNTIME_DIR}/local-setup.env.sh}"

KERNEL_FORGE_REPO="${KERNEL_FORGE_REPO:-https://github.com/AMD-AGI/KernelForge.git}"
INFERENCEX_REPO="${INFERENCEX_REPO:-https://github.com/SemiAnalysisAI/InferenceX.git}"
INFERENCEX_REF="${INFERENCEX_REF:-2035a2117ad22403376359be0064dfa2c078c59b}"
TRACELENS_REPO="${TRACELENS_REPO:-https://github.com/AMD-AGI/TraceLens.git}"
# TraceLens v0.7.0 integration (#474): head of release/hyperloom_integration_v0.7.0.
TRACELENS_REF="${TRACELENS_REF:-35bbb6380cf69a2655ee28260b02b5f2dc481744}"
# Optional operator hint for a pre-existing manual checkout. Left EMPTY by
# default: the pod-local ${_open_source_root}/TraceLens is the sole implicit
# default (resolved in resolve_tracelens), so a stale /workspace/TraceLens is
# never silently adopted, which would bypass the /opt clone+pin path (#722).
# The internal extension is private: Hyperloom keeps NO repo URL, ref, or
# default path for it; used only when TRACELENS_INTERNAL_ROOT is set.
TRACELENS_DEFAULT_ROOT="${TRACELENS_DEFAULT_ROOT:-}"

usage() {
  cat <<'EOF'
Usage: src/hyperloom/inference_optimizer/scripts/local_setup.sh [options]

Bootstraps Local Mode from only a Hyperloom checkout plus credentials.
It clones missing dependency repos and writes a local env file. Runtime
dependency checks and installation remain owned by the Cursor agent via
src/hyperloom/inference_optimizer/SKILL.md when optimization starts.

Options:
  --dry-run             Print planned actions without cloning or writing
  --check-only          Verify existing dependency checkouts, do not write env
  --deps-root PATH      Directory for dependency checkouts
  --session-dir PATH    Session directory; defaults to $USER_DATA_PATH or /workspace/hyperloom
  -h, --help            Show this help

Advanced env overrides:
  REPO_ROOT, USER_DATA_PATH, HYPERLOOM_DEPS_ROOT, LOCAL_SETUP_ENV,
  OOB_SRC, INFERENCEX_PATH, TRACELENS_ROOT, TRACELENS_INTERNAL_ROOT,
  KERNEL_FORGE_REPO, INFERENCEX_REPO, INFERENCEX_REF,
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
      ;;
    --session-dir)
      [ "$#" -ge 2 ] || { echo "[local-setup] ERROR: --session-dir requires PATH" >&2; exit 2; }
      shift
      USER_DATA_PATH="${1:-}"
      HYPERLOOM_RUNTIME_DIR="${USER_DATA_PATH}/runtime"
      # Deps root stays pod-local and does NOT follow --session-dir.
      LOCAL_SETUP_ENV="${HYPERLOOM_RUNTIME_DIR}/local-setup.env.sh"
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[local-setup] ERROR: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# Re-sync after arg parsing: --deps-root may have changed HYPERLOOM_DEPS_ROOT.
_open_source_root="${HYPERLOOM_DEPS_ROOT}"
# Export the canonical open-source-root key so a same-shell `install.sh` /
# optimize invocation that did NOT source local-setup.env.sh still resolves the
# SAME default TraceLens path (install.sh / paths / handler / tool read only
# HYPERLOOM_OPEN_SOURCE_ROOT). Without this, a --deps-root / HYPERLOOM_DEPS_ROOT
# override would leave those consumers on /opt and mis-classify managed vs
# override (#722).
export HYPERLOOM_OPEN_SOURCE_ROOT="${_open_source_root}"

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
  # When "atomic", a fresh clone + ref checkout happens in a temp sibling and is
  # renamed into place only after both succeed, so a concurrent reader never
  # sees a half-cloned/unpinned $dest (#722). Used for TraceLens, whose path is
  # read live by trace_analyze. Keep in lockstep with the twin implementations:
  # kernel-agent/scripts/install.sh (ensure_tracelens) and
  # kernel-agent/tools/tracelens_analysis.py (_ensure_tracelens_checkout).
  local mode="${5:-}"

  if [ -d "${dest}/.git" ]; then
    if [ "$CHECK_ONLY" -eq 1 ]; then
      log "${name}: existing checkout ${dest}"
      return 0
    fi
    if [ -n "$ref" ]; then
      # Realign to $ref via the shallow SHA-aware fetch used by
      # kernel-agent/scripts/install.sh (ensure_tracelens): `fetch origin <ref>`
      # + detached FETCH_HEAD checkout works for both a branch name and a raw
      # commit SHA on a real (shallow) GitHub remote, unlike `checkout <sha>`
      # which needs the object already present locally (#722 / PR#789).
      run git -C "$dest" fetch --depth 1 origin "$ref"
      run git -C "$dest" checkout -q FETCH_HEAD
    else
      run git -C "$dest" fetch --all --tags --prune
    fi
    return 0
  fi

  if [ -e "$dest" ]; then
    # atomic mode targets the installer-MANAGED default checkout (TraceLens,
    # read live by trace_analyze): a dir without .git is a half-done/crashed
    # clone, so drop it and rebuild, matching kernel-agent/scripts/install.sh
    # (ensure_tracelens) and tracelens_analysis.py (_ensure_tracelens_checkout).
    # Non-atomic deps keep the fail-fast guard so an operator path is never
    # silently wiped (#722 / PR#789).
    if [ "$mode" = "atomic" ]; then
      if [ "$DRY_RUN" -eq 1 ]; then
        log "would: rm -rf ${dest} (incomplete, not a git repo) then clone"
      elif [ "$CHECK_ONLY" -eq 1 ]; then
        warn "${name}: checkout at ${dest} is not a git repo (check-only, skipping rebuild)"
        return 0
      else
        warn "${name}: checkout at ${dest} is not a git repo; rebuilding"
        rm -rf "$dest"
      fi
    else
      die "${name} destination exists but is not a git checkout: ${dest}"
    fi
  fi
  if [ "$CHECK_ONLY" -eq 1 ]; then
    die "${name} checkout missing: ${dest}"
  fi

  log "${name}: clone ${repo} -> ${dest}"
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would: git clone ${repo} ${dest}${ref:+ (checkout ${ref})}"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  if [ "$mode" = "atomic" ]; then
    local tmp ok=1
    tmp="$(dirname "$dest")/.$(basename "$dest").clone.$$"
    rm -rf "$tmp"
    git clone "$repo" "$tmp" || ok=0
    if [ "$ok" -eq 1 ] && [ -n "$ref" ]; then
      # Pin via shallow SHA-aware fetch + detached FETCH_HEAD, matching the twin
      # implementations (install.sh ensure_tracelens, tracelens_analysis.py):
      # `checkout <sha>` needs the object present locally, which a shallow clone
      # may lack; `fetch origin <ref>` works for a branch name or raw SHA (#722).
      git -C "$tmp" fetch --depth 1 origin "$ref" || ok=0
      [ "$ok" -eq 1 ] && { git -C "$tmp" checkout -q FETCH_HEAD || ok=0; }
    fi
    if [ "$ok" -eq 0 ]; then
      rm -rf "$tmp"
      die "${name}: clone/checkout ${ref:-default} failed; refusing to publish a partial checkout at ${dest}"
    fi
    mv "$tmp" "$dest"
    return 0
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

# Normalized (placeholder-stripped) TRACELENS_ROOT from env or .env; empty when
# unset or a historical "\" / whitespace placeholder. Mirrors the cleaning in
# _resolve_existing_checkout so the "explicit override?" decision cannot be
# fooled by a placeholder that later gets discarded (#722 / PR#789).
_normalized_tracelens_root_value() {
  local value="${TRACELENS_ROOT:-}"
  if [ -z "$value" ]; then
    value="$(_read_dotenv_var TRACELENS_ROOT || true)"
  fi
  _is_placeholder_path_value "$value" && value=""
  printf '%s' "$value"
}

# Canonicalize a path (resolve symlinks/.. , strip trailing slash) so the
# default-vs-override comparison matches the Python side's Path.resolve(); a
# trailing-slash / symlinked spelling of the default must not read as override.
# Empty input yields empty output; unresolvable paths fall back to the trimmed
# literal so a not-yet-cloned default still compares correctly (#722 / PR#789).
# Keep in lockstep with the twin helper in kernel-agent/scripts/install.sh.
_canonicalize_path() {
  local p="${1:-}"
  [ -z "$p" ] && return 0
  readlink -f -- "$p" 2>/dev/null || printf '%s' "${p%/}"
}

resolve_tracelens() {
  # An EXPLICIT override (TRACELENS_ROOT via env/.env, or an explicitly-set
  # TRACELENS_DEFAULT_ROOT) is operator-maintained: adopt it as-is, never
  # re-pin. The IMPLICIT pod-local default (${_open_source_root}/TraceLens) is
  # installer-managed: always run clone_or_update so an existing checkout is
  # fetched/checked out to TRACELENS_REF (not left on a stale SHA) and a missing
  # one is atomically cloned+pinned (#722 / PR#789).
  _normalize_trace_env_roots
  local _explicit="" _default_root _norm_root _norm_default_root
  # Explicit override ONLY when a placeholder-stripped, canonicalized path
  # (TRACELENS_ROOT or TRACELENS_DEFAULT_ROOT) points OUTSIDE the pod-local
  # default. The default path — even when re-exported into env/.env or spelled
  # out via TRACELENS_DEFAULT_ROOT — stays installer-managed so a stale checkout
  # is still realigned to TRACELENS_REF, not silently adopted (#722 / PR#789).
  # Matches the path-based override test in kernel-agent/scripts/install.sh and
  # the handler/tool.
  _default_root="$(_canonicalize_path "${_open_source_root}/TraceLens")"
  _norm_root="$(_canonicalize_path "$(_normalized_tracelens_root_value)")"
  _norm_default_root=""
  if ! _is_placeholder_path_value "${TRACELENS_DEFAULT_ROOT:-}"; then
    _norm_default_root="$(_canonicalize_path "${TRACELENS_DEFAULT_ROOT:-}")"
  fi
  if { [ -n "$_norm_root" ] && [ "$_norm_root" != "$_default_root" ]; } \
     || { [ -n "$_norm_default_root" ] && [ "$_norm_default_root" != "$_default_root" ]; }; then
    _explicit=1
  fi
  if [ -n "$_explicit" ] && _resolve_existing_checkout TRACELENS_ROOT \
       "${TRACELENS_DEFAULT_ROOT:-${_open_source_root}/TraceLens}"; then
    :
  else
    TRACELENS_ROOT="${_open_source_root}/TraceLens"
    # Atomic: TraceLens is read live by trace_analyze; never publish a
    # half-cloned/unpinned tree at $TRACELENS_ROOT (#722). clone_or_update
    # realigns an existing checkout to $TRACELENS_REF.
    clone_or_update "TraceLens" "$TRACELENS_REPO" "$TRACELENS_ROOT" "$TRACELENS_REF" atomic
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

  local root="${_open_source_root}/KernelForge"
  clone_or_update "KernelForge" "$KERNEL_FORGE_REPO" "$root" ""
  OOB_SRC="${root}/OOB"
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    [ -d "$OOB_SRC" ] || die "KernelForge checkout does not contain OOB/: ${OOB_SRC}"
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

  INFERENCEX_PATH="${_open_source_root}/InferenceX"
  clone_or_update "InferenceX" "$INFERENCEX_REPO" "$INFERENCEX_PATH" "$INFERENCEX_REF"
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
    echo '# Generated by src/hyperloom/inference_optimizer/scripts/local_setup.sh'
    write_export REPO_ROOT "$REPO_ROOT"
    write_export USER_DATA_PATH "$USER_DATA_PATH"
    write_export HYPERLOOM_RUNTIME_DIR "$HYPERLOOM_RUNTIME_DIR"
    write_export HYPERLOOM_DEPS_ROOT "$HYPERLOOM_DEPS_ROOT"
    # Also export the canonical open-source-root key so install.sh /
    # hyperloom.inference_optimizer.paths / the handler / the tool resolve the SAME
    # default TraceLens path when this env file is sourced. Without it, a
    # --deps-root / HYPERLOOM_DEPS_ROOT override would leave those consumers on
    # /opt/hyperloom/open-source-repos and mis-classify managed vs override (#722).
    write_export HYPERLOOM_OPEN_SOURCE_ROOT "$_open_source_root"
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

@src/hyperloom/inference_optimizer/SKILL.md

Optimize inference for this workload:
- Model: /path/to/your/model
- Framework: sglang
- GPU: MI300X
- TP: 8
- CONC: 64
- ISL: 1024
- OSL: 1024
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
