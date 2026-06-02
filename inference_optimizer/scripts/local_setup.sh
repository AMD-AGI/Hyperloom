#!/usr/bin/env bash
# Local Mode bootstrap for a fresh Hyperloom checkout.

set -euo pipefail

DRY_RUN=0
CHECK_ONLY=0

_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${_script_dir}/../.." && pwd)}"
USER_DATA_PATH="${USER_DATA_PATH:-/workspace/hyperloom}"
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
# TraceLens: open-source AMD-AGI/TraceLens. Carries the full TraceLens.*
# Python package, the TraceLens_generate_perf_report_* CLIs, the agent
# skill bundle, and the SGLang/vLLM patch tree (examples/custom_workflows/
# inference_analysis/) that _server_patcher reads at runtime. After the
# 2026-05-18 OSS migration this is the only repo Hyperloom needs for the
# baseline feature set.
TRACELENS_REPO="${TRACELENS_REPO:-https://github.com/AMD-AGI/TraceLens.git}"
# Default to ``main`` until a TraceLens release that includes PR #668
# (FlyDSL MoE perfmodel) is tagged. Pin to the resulting tag (e.g.
# ``v0.5.0``) for reproducible installs once it lands.
TRACELENS_REF="${TRACELENS_REF:-main}"
# Optional TraceLens-internal extension (private). Adds roofline numbers,
# gains estimates, and MI355/MI455 MAF data on top of the open-source
# impact-score-only report. OFF by default — opt in by exporting
# TRACELENS_INSTALL_INTERNAL=1 (clones into ${HYPERLOOM_DEPS_ROOT}/TraceLens-internal)
# or by setting TRACELENS_INTERNAL_ROOT to point at an existing checkout.
TRACELENS_INTERNAL_REPO="${TRACELENS_INTERNAL_REPO:-https://github.com/AMD-AGI/TraceLens-internal.git}"
TRACELENS_INTERNAL_REF="${TRACELENS_INTERNAL_REF:-main}"
TRACELENS_INSTALL_INTERNAL="${TRACELENS_INSTALL_INTERNAL:-0}"

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
  TRACELENS_INSTALL_INTERNAL, TRACELENS_INTERNAL_REPO, TRACELENS_INTERNAL_REF
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

resolve_tracelens() {
  if [ -n "${TRACELENS_ROOT:-}" ]; then
    [ -d "$TRACELENS_ROOT" ] || die "TRACELENS_ROOT is set but does not exist: ${TRACELENS_ROOT}"
    log "TRACELENS_ROOT: using existing ${TRACELENS_ROOT}"
    return 0
  fi

  TRACELENS_ROOT="${HYPERLOOM_DEPS_ROOT}/TraceLens"
  clone_or_update "TraceLens" "$TRACELENS_REPO" "$TRACELENS_ROOT" "$TRACELENS_REF"
  export TRACELENS_ROOT
  log "TRACELENS_ROOT: ${TRACELENS_ROOT}"
}

resolve_tracelens_internal() {
  # Opt-in: ON when TRACELENS_INTERNAL_ROOT is already exported (operator
  # provided an existing checkout) or TRACELENS_INSTALL_INTERNAL is truthy.
  # OFF by default keeps Local Mode on the open-source-only path (impact
  # score, no roofline, MI300/MI325 MAF only).
  local opt_in=0
  if [ -n "${TRACELENS_INTERNAL_ROOT:-}" ]; then opt_in=1; fi
  case "${TRACELENS_INSTALL_INTERNAL:-0}" in
    1|true|TRUE|yes|YES) opt_in=1 ;;
  esac
  if [ "$opt_in" -eq 0 ]; then
    log "TraceLens-internal: skipped (set TRACELENS_INSTALL_INTERNAL=1 to opt in)"
    return 0
  fi

  if [ -n "${TRACELENS_INTERNAL_ROOT:-}" ]; then
    [ -d "$TRACELENS_INTERNAL_ROOT" ] || die "TRACELENS_INTERNAL_ROOT is set but does not exist: ${TRACELENS_INTERNAL_ROOT}"
    log "TRACELENS_INTERNAL_ROOT: using existing ${TRACELENS_INTERNAL_ROOT}"
    export TRACELENS_INTERNAL_ROOT
    return 0
  fi

  TRACELENS_INTERNAL_ROOT="${HYPERLOOM_DEPS_ROOT}/TraceLens-internal"
  clone_or_update "TraceLens-internal" "$TRACELENS_INTERNAL_REPO" \
    "$TRACELENS_INTERNAL_ROOT" "$TRACELENS_INTERNAL_REF"
  export TRACELENS_INTERNAL_ROOT
  log "TRACELENS_INTERNAL_ROOT: ${TRACELENS_INTERNAL_ROOT}"
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
  resolve_tracelens_internal
  write_local_env

  print_next_steps
}

main "$@"
