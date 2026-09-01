#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Local Mode bootstrap for a fresh Hyperloom checkout.
#
# Scope: write local-setup.env.sh, the env file every later step sources
# (REPO_ROOT / USER_DATA_PATH / HYPERLOOM_RUNTIME_DIR / HYPERLOOM_DEPS_ROOT /
# HYPERLOOM_CACHE_DIR). Open-source deps (Magpie / InferenceX / TraceLens) are
# owned by install.sh.
#
# It used to also clone the private KernelForge repo and export $FORGE_PATH,
# because the forge kernel backend lived in that separate checkout. forge now
# ships inside this distribution as the `kernelforge` package, so there is
# nothing to clone and nothing left to point at: no code reads $FORGE_PATH any
# more, and it is not on env_safety's forwarding allowlist either. The dev
# override that replaced it is $KERNELFORGE_PROJECT_ROOT.

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
# Container images ship a writable /workspace; a bare-metal host off root has
# neither it nor permission to create it, so the mkdir below would abort.
_default_workspace_root() {
  # The nearest existing ancestor decides: -w is false for a path that does not
  # exist yet, which would divert root off a /workspace it can still create.
  _ws_probe=/workspace
  while [ ! -e "$_ws_probe" ] && [ "$_ws_probe" != / ]; do _ws_probe=$(dirname "$_ws_probe"); done
  if [ -w "$_ws_probe" ]; then printf '%s' /workspace/hyperloom; else printf '%s' "$(pwd -P)/session"; fi
}
USER_DATA_PATH="${USER_DATA_PATH:-$(_default_workspace_root)}"
if [ -z "${_user_data_was_set}" ]; then
  echo "[install WARN] USER_DATA_PATH not set; defaulting to ${USER_DATA_PATH}. Set USER_DATA_PATH to persist artifacts under your data root." >&2
fi
HYPERLOOM_RUNTIME_DIR="${HYPERLOOM_RUNTIME_DIR:-${USER_DATA_PATH}/runtime}"
# Pod-local base for dependency checkouts. Keep it decoupled from
# USER_DATA_PATH so shared (WekaFS) workspaces never collocate pod checkouts.
HYPERLOOM_DEPS_ROOT="${HYPERLOOM_DEPS_ROOT:-${HYPERLOOM_CACHE_DIR:-${REPO_ROOT}/.cache}}"
_open_source_root="${HYPERLOOM_DEPS_ROOT}"
LOCAL_SETUP_ENV="${LOCAL_SETUP_ENV:-${HYPERLOOM_RUNTIME_DIR}/local-setup.env.sh}"

usage() {
  cat <<'EOF'
Usage: src/hyperloom/inference_optimizer/assets/local_setup.sh [options]

Writes the local env file every later step sources. Open-source deps
(Magpie / InferenceX / TraceLens) are installed by install.sh, not here; the
forge kernel backend ships in this distribution and needs no checkout.

Options:
  --dry-run             Print planned actions without writing
  --check-only          Do not write the env file
  --deps-root PATH      Directory for dependency checkouts
  --user-data-path PATH Writable artifact root; defaults to $USER_DATA_PATH or /workspace/hyperloom
  --session-dir PATH    Alias for --user-data-path (backward compatible)
  --no-next-steps       Do not print the standalone launch prompt
  -h, --help            Show this help

Advanced env overrides:
  REPO_ROOT, USER_DATA_PATH, HYPERLOOM_DEPS_ROOT, LOCAL_SETUP_ENV
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
export HYPERLOOM_CACHE_DIR="${_open_source_root}"

log() { echo "[local-setup] $*"; }
warn() { echo "[local-setup WARN] $*" >&2; }
die() { echo "[local-setup ERROR] $*" >&2; exit 1; }

shell_quote() {
  printf "'%s'" "$(printf '%s' "$1" | sed "s/'/'\\\\''/g")"
}

write_export() {
  printf 'export %s=%s\n' "$1" "$(shell_quote "$2")"
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
    # the $REPO_ROOT/.cache default and mis-classify managed vs override (#722).
    write_export HYPERLOOM_CACHE_DIR "$_open_source_root"
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

  write_local_env

  if [ "$PRINT_NEXT_STEPS" -eq 1 ]; then
    print_next_steps
  else
    log "local setup complete"
  fi
}

main "$@"
