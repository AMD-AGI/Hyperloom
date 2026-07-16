#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Local Mode bootstrap for a fresh Hyperloom checkout.
#
# Scope: clone the PRIVATE KernelForge repo and write local-setup.env.sh
# (exports FORGE_PATH). Open-source deps (Magpie / InferenceX / TraceLens) are
# owned by install.sh; bare-metal invokes this only when the kernel backend
# order includes forge.

set -euo pipefail

DRY_RUN=0
CHECK_ONLY=0
PRINT_NEXT_STEPS=1

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
# Pod-local base for the private KernelForge checkout. Keep it decoupled from
# USER_DATA_PATH so shared (WekaFS) workspaces never collocate pod checkouts.
HYPERLOOM_DEPS_ROOT="${HYPERLOOM_DEPS_ROOT:-${HYPERLOOM_OPEN_SOURCE_ROOT:-/opt/hyperloom/open-source-repos}}"
_open_source_root="${HYPERLOOM_DEPS_ROOT}"
LOCAL_SETUP_ENV="${LOCAL_SETUP_ENV:-${HYPERLOOM_RUNTIME_DIR}/local-setup.env.sh}"

# Only the private KernelForge repo is cloned here; open-source deps
# (Magpie / InferenceX / TraceLens) are owned by install.sh.
KERNEL_FORGE_REPO="${KERNEL_FORGE_REPO:-https://github.com/AMD-AGI/KernelForge.git}"

usage() {
  cat <<'EOF'
Usage: src/hyperloom/inference_optimizer/assets/local_setup.sh [options]

Clones the private KernelForge checkout and writes a local env file exporting
FORGE_PATH. Open-source deps (Magpie / InferenceX / TraceLens) are installed by
install.sh, not here. Bare-metal installs call this only when the kernel backend
order includes forge.

Options:
  --dry-run             Print planned actions without cloning or writing
  --check-only          Verify existing dependency checkouts, do not write env
  --deps-root PATH      Directory for dependency checkouts
  --user-data-path PATH Writable artifact root; defaults to $USER_DATA_PATH or /workspace/hyperloom
  --session-dir PATH    Alias for --user-data-path (backward compatible)
  --no-next-steps       Do not print the standalone launch prompt
  -h, --help            Show this help

Advanced env overrides:
  REPO_ROOT, USER_DATA_PATH, HYPERLOOM_DEPS_ROOT, LOCAL_SETUP_ENV,
  FORGE_PATH, KERNEL_FORGE_ROOT, KERNEL_FORGE_PATH, KERNEL_FORGE_REPO
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
    --user-data-path|--session-dir)
      [ "$#" -ge 2 ] || { echo "[local-setup] ERROR: $1 requires PATH" >&2; exit 2; }
      shift
      USER_DATA_PATH="${1:-}"
      HYPERLOOM_RUNTIME_DIR="${USER_DATA_PATH}/runtime"
      # Deps root stays pod-local and does NOT follow --session-dir.
      LOCAL_SETUP_ENV="${HYPERLOOM_RUNTIME_DIR}/local-setup.env.sh"
      ;;
    --no-next-steps) PRINT_NEXT_STEPS=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[local-setup] ERROR: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# Re-sync after arg parsing: --deps-root may have changed HYPERLOOM_DEPS_ROOT.
_open_source_root="${HYPERLOOM_DEPS_ROOT}"
# Export the canonical open-source-root key so same-shell install.sh / optimize
# invocations resolve the same default open-source dependency paths.
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
    if [ -n "$ref" ]; then
      # Realign to $ref via the shallow SHA-aware fetch used by
      # src/hyperloom/agents/kernel/scripts/install.sh (ensure_tracelens): `fetch origin <ref>`
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
    die "${name} destination exists but is not a git checkout: ${dest}"
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
  if ! run git clone "$repo" "$dest"; then
    return 1
  fi
  if [ -n "$ref" ]; then
    run git -C "$dest" checkout "$ref"
  fi
  return 0
}

resolve_forge() {
  if [ -n "${FORGE_PATH:-}" ]; then
    [ -d "$FORGE_PATH" ] || die "FORGE_PATH is set but does not exist: ${FORGE_PATH}"
    KERNEL_FORGE_ROOT="${KERNEL_FORGE_ROOT:-$FORGE_PATH}"
    export FORGE_PATH KERNEL_FORGE_ROOT
    log "FORGE_PATH: using existing ${FORGE_PATH}"
    return 0
  fi

  # KernelForge is a separate repo cloned only for the opt-in forge kernel
  # backend. Treat the clone as best-effort: when it is unavailable (no access
  # or not yet public), warn and continue so the rest of local setup still
  # succeeds. The default backend order is geak, which does not require
  # KernelForge; forge is opt-in via KERNEL_OPT_BACKEND_ORDER=forge.
  local root="${_open_source_root}/KernelForge"
  if clone_or_update "KernelForge" "$KERNEL_FORGE_REPO" "$root" ""; then
    FORGE_PATH="${FORGE_PATH:-$root}"
    KERNEL_FORGE_ROOT="${KERNEL_FORGE_ROOT:-$FORGE_PATH}"
    export FORGE_PATH KERNEL_FORGE_ROOT
    log "FORGE_PATH: ${FORGE_PATH}"
  else
    warn "KernelForge checkout unavailable (${KERNEL_FORGE_REPO}); skipping forge backend setup."
    warn "The forge kernel backend (KERNEL_OPT_BACKEND_ORDER=forge) will be unavailable; the default 'geak' backend does not require KernelForge."
  fi
}

write_local_env() {
  if [ "$DRY_RUN" -eq 1 ] || [ "$CHECK_ONLY" -eq 1 ]; then
    log "would write local env: ${LOCAL_SETUP_ENV}"
    return 0
  fi

  mkdir -p "$(dirname "$LOCAL_SETUP_ENV")"
  {
    echo '#!/bin/sh'
    echo '# Generated by src/hyperloom/inference_optimizer/assets/local_setup.sh'
    write_export REPO_ROOT "$REPO_ROOT"
    write_export USER_DATA_PATH "$USER_DATA_PATH"
    write_export HYPERLOOM_RUNTIME_DIR "$HYPERLOOM_RUNTIME_DIR"
    write_export HYPERLOOM_DEPS_ROOT "$HYPERLOOM_DEPS_ROOT"
    # Also export the canonical open-source-root key so install.sh /
    # hyperloom.inference_optimizer.session.paths / the handler / the tool resolve the SAME
    # default open-source dep paths when this env file is sourced. Without it, a
    # --deps-root / HYPERLOOM_DEPS_ROOT override would leave those consumers on
    # /opt/hyperloom/open-source-repos and mis-classify managed vs override (#722).
    write_export HYPERLOOM_OPEN_SOURCE_ROOT "$_open_source_root"
    if [ -n "${FORGE_PATH:-}" ]; then
      write_export FORGE_PATH "$FORGE_PATH"
      write_export KERNEL_FORGE_ROOT "$KERNEL_FORGE_ROOT"
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
- TP: 1
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

  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    mkdir -p "$HYPERLOOM_DEPS_ROOT" "$HYPERLOOM_RUNTIME_DIR"
  fi

  ensure_git_available
  resolve_forge
  write_local_env

  if [ "$PRINT_NEXT_STEPS" -eq 1 ]; then
    print_next_steps
  else
    log "local setup complete"
  fi
}

main "$@"
