#!/usr/bin/env bash
# Inference Optimizer installer.
#
# Owns the inference_optimizer-side bare-image setup so SKILL.md does
# not have to hand-roll it. Idempotent — every step skips if the
# artifact is already present.
#
# Stack (in order):
#   1. inference_optimizer + extras (pulls in claude_agent_sdk via
#      pyproject `[test]` extra)
#   2. Magpie (benchmark engine) into $WORKSPACE_ROOT/Magpie
#   3. InferenceX checkout detection (sets INFERENCEX_PATH for runtime)
#   4. Delegates to kernel-agent/scripts/install.sh for ray, ray-head
#      bring-up, Node/npm, TraceLens, GEAK, OOB and the auth-proxy. kernel-agent
#      itself is the canonical owner of those — we just chain to it
#      so users have a single entry point.
#
# kernel-agent's install.sh owns Ray + ray start, TraceLens, GEAK, OOB
# auth-proxy. inference_optimizer's install.sh owns Magpie / InferenceX
# / the inference_optimizer Python package itself. The two are
# composable: kernel-agent works standalone; inference_optimizer drags
# kernel-agent in via this script.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
INFERENCE_OPTIMIZER_SESSION_DIR="${INFERENCE_OPTIMIZER_SESSION_DIR:-/workspace/hyperloom}"
HYPERLOOM_RUNTIME_DIR="${HYPERLOOM_RUNTIME_DIR:-${INFERENCE_OPTIMIZER_SESSION_DIR}/runtime}"
KERNEL_AGENT_ENV="${KERNEL_AGENT_ENV:-${HYPERLOOM_RUNTIME_DIR}/kernel-agent.env.sh}"
HYPERLOOM_ROOT="${HYPERLOOM_ROOT:-/opt/hyperloom}"
KERNEL_AGENT_ROOT="${KERNEL_AGENT_ROOT:-${REPO_ROOT}/kernel-agent}"
MAGPIE_REPO="${MAGPIE_REPO:-https://github.com/AMD-AGI/Magpie.git}"
MAGPIE_DIR="${MAGPIE_DIR:-${WORKSPACE_ROOT}/Magpie}"

DRY_RUN=0
CHECK_ONLY=0
SKIP_KERNEL_AGENT=0

usage() {
  cat <<'EOF'
Usage: inference_optimizer/scripts/install.sh [options]

Installs:
  - inference_optimizer Python package (with claude_agent_sdk via [test])
  - Magpie (cloned to $WORKSPACE_ROOT/Magpie)
  - Detects/exports INFERENCEX_PATH
  - Chains to kernel-agent/scripts/install.sh for Ray + ray-head start,
    Node/npm, TraceLens, GEAK, OOB CLI, and the OOB auth-proxy.

Options:
  --check-only          Verify only, do not install
  --dry-run             Print actions without running them
  --skip-kernel-agent   Skip the chained kernel-agent installer
  -h, --help            Show this help

Env overrides:
  REPO_ROOT, WORKSPACE_ROOT, KERNEL_AGENT_ROOT, MAGPIE_REPO, MAGPIE_DIR,
  INFERENCEX_PATH, PYTHON, TRACELENS_ROOT, INFERENCE_OPTIMIZER_SESSION_DIR,
  HYPERLOOM_RUNTIME_DIR, KERNEL_AGENT_ENV, HYPERLOOM_ROOT
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --check-only) CHECK_ONLY=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --skip-kernel-agent) SKIP_KERNEL_AGENT=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[inference-optimizer] ERROR: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

log() { echo "[inference-optimizer] $*"; }
warn() { echo "[inference-optimizer WARN] $*" >&2; }
die() { echo "[inference-optimizer ERROR] $*" >&2; exit 1; }

run() {
  log "$*"
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    "$@"
  fi
}

# --- 0. Resolve PYTHON ---
# Prefer the existing PYTHON env (callers may have already pinned it),
# then /opt/venv/bin/python (default for hyperloom containers), then
# whatever python3 is on PATH. We do NOT hardcode /opt/venv/bin into
# subsequent PATH; we only use this binary to drive `pip install`.
resolve_python() {
  if [ -n "${PYTHON:-}" ] && [ -x "$PYTHON" ]; then
    return 0
  fi
  if [ -x "/opt/venv/bin/python" ]; then
    PYTHON="/opt/venv/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    PYTHON="$(command -v python3)"
    return 0
  fi
  die "no usable python found (set PYTHON, install python3, or mount /opt/venv)"
}

resolve_python
log "PYTHON=${PYTHON}"
log "REPO_ROOT=${REPO_ROOT}"
log "WORKSPACE_ROOT=${WORKSPACE_ROOT}"
log "HYPERLOOM_ROOT=${HYPERLOOM_ROOT}"
log "KERNEL_AGENT_ROOT=${KERNEL_AGENT_ROOT}"
log "KERNEL_AGENT_ENV=${KERNEL_AGENT_ENV}"
export INFERENCE_OPTIMIZER_SESSION_DIR HYPERLOOM_RUNTIME_DIR KERNEL_AGENT_ENV
export HYPERLOOM_KERNEL_AGENT_ROOT="${HYPERLOOM_KERNEL_AGENT_ROOT:-${KERNEL_AGENT_ROOT}}"

# pip --break-system-packages when PYTHON is the system interpreter
# (e.g. bare ubuntu/debian image without a venv). Detect by comparing
# sys.prefix vs sys.base_prefix; equal == not in venv.
PIP_EXTRA=()
if "$PYTHON" - <<'PY' 2>/dev/null
import sys
raise SystemExit(0 if sys.prefix == sys.base_prefix else 1)
PY
then
  PIP_EXTRA=(--break-system-packages)
  log "non-venv PYTHON; pip will use --break-system-packages"
fi

# --- 1. inference_optimizer + claude_agent_sdk via [test] ---
ensure_inference_optimizer() {
  log "ensuring inference_optimizer package + claude_agent_sdk extras"
  if [ "$CHECK_ONLY" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    "$PYTHON" -m pip install --quiet "${PIP_EXTRA[@]}" -e "${REPO_ROOT}[test]"
  fi
  "$PYTHON" - <<'PY' || die "inference_optimizer not importable after install"
import inference_optimizer  # noqa: F401
PY
  if "$PYTHON" -c "import claude_agent_sdk" >/dev/null 2>&1; then
    log "claude_agent_sdk OK"
  else
    warn "claude_agent_sdk not importable after install (Coordinator will fail)"
    [ "$CHECK_ONLY" -eq 1 ] || die "claude_agent_sdk missing"
  fi
}

# --- 2. Magpie ---
ensure_magpie() {
  log "ensuring Magpie at ${MAGPIE_DIR}"
  if "$PYTHON" -c "import Magpie" >/dev/null 2>&1; then
    log "Magpie already importable"
    return 0
  fi
  if [ "$CHECK_ONLY" -eq 1 ]; then
    warn "Magpie not importable (check-only mode, skipping clone/install)"
    return 0
  fi
  if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$WORKSPACE_ROOT"
  fi
  if [ ! -f "$MAGPIE_DIR/setup.py" ] && [ ! -f "$MAGPIE_DIR/pyproject.toml" ]; then
    log "cloning Magpie from $MAGPIE_REPO"
    run git clone --depth 1 "$MAGPIE_REPO" "$MAGPIE_DIR"
  fi
  if [ "$DRY_RUN" -eq 0 ]; then
    "$PYTHON" -m pip install --quiet "${PIP_EXTRA[@]}" -e "$MAGPIE_DIR"
    "$PYTHON" -c "import Magpie" >/dev/null
    log "Magpie installed OK"
  fi
}

# --- 3. InferenceX detection ---
ensure_inferencex() {
  if [ -n "${INFERENCEX_PATH:-}" ] && [ -d "$INFERENCEX_PATH" ]; then
    log "INFERENCEX_PATH = $INFERENCEX_PATH (preserved from env)"
    return 0
  fi
  for candidate in \
      "$MAGPIE_DIR/InferenceX" \
      "/wekafs/hyperloom/InferenceX" \
      "/opt/hyperloom/InferenceX" \
      "/wekafs/fully-local/inference_optimization/InferenceX"
  do
    if [ -d "$candidate" ]; then
      INFERENCEX_PATH="$candidate"
      export INFERENCEX_PATH
      log "INFERENCEX_PATH = $INFERENCEX_PATH"
      return 0
    fi
  done
  warn "InferenceX not found. GSM8K eval will fail without it. Set"
  warn "INFERENCEX_PATH or clone https://github.com/SemiAnalysisAI/InferenceX"
  warn "into \$MAGPIE_DIR/InferenceX."
}

# --- 4. Chain to kernel-agent ---
chain_kernel_agent() {
  if [ "$SKIP_KERNEL_AGENT" -eq 1 ]; then
    log "skipping kernel-agent installer (--skip-kernel-agent)"
    return 0
  fi
  local script="${KERNEL_AGENT_ROOT}/scripts/install.sh"
  if [ ! -f "$script" ]; then
    warn "kernel-agent installer not found at $script"
    return 0
  fi
  log "delegating ray + TraceLens + GEAK + OOB + auth-proxy to ${script}"
  export REPO_ROOT WORKSPACE_ROOT KERNEL_AGENT_ROOT MAGPIE_DIR HYPERLOOM_ROOT
  export INFERENCE_OPTIMIZER_SESSION_DIR HYPERLOOM_RUNTIME_DIR KERNEL_AGENT_ENV
  export HYPERLOOM_KERNEL_AGENT_ROOT="${HYPERLOOM_KERNEL_AGENT_ROOT:-${KERNEL_AGENT_ROOT}}"
  [ -n "${INFERENCEX_PATH:-}" ] && export INFERENCEX_PATH
  local args=()
  [ "$CHECK_ONLY" -eq 1 ] && args+=(--check-only)
  [ "$DRY_RUN" -eq 1 ] && args+=(--dry-run)
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would run: bash '$script' ${args[*]}"
    return 0
  fi
  bash "$script" "${args[@]}"
}

ensure_inference_optimizer
ensure_magpie
ensure_inferencex
chain_kernel_agent

log "install complete"
log "next: source ${KERNEL_AGENT_ENV}, then run inference_optimizer.cli"
