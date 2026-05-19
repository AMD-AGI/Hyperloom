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
#   2. Magpie (benchmark engine) into $HYPERLOOM_RUNTIME_DIR/Magpie
#      (= $USER_DATA_PATH/runtime/Magpie by default)
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

# Single artefact root: everything writable defaults to $USER_DATA_PATH so
# operators can monitor a run end-to-end by tailing one directory. Magpie
# clone, source mirrors, generated env / GEAK config, and the pod-local
# auth-proxy state all derive from $HYPERLOOM_RUNTIME_DIR.
# Removed envs: WORKSPACE_ROOT / WORKSPACE_PATH (collapsed into USER_DATA_PATH).
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
USER_DATA_PATH="${USER_DATA_PATH:-/workspace/hyperloom}"
HYPERLOOM_RUNTIME_DIR="${HYPERLOOM_RUNTIME_DIR:-${USER_DATA_PATH}/runtime}"
KERNEL_AGENT_ENV="${KERNEL_AGENT_ENV:-${HYPERLOOM_RUNTIME_DIR}/kernel-agent.env.sh}"
HYPERLOOM_ROOT="${HYPERLOOM_ROOT:-${HYPERLOOM_RUNTIME_DIR}/source-mirrors}"
KERNEL_AGENT_ROOT="${KERNEL_AGENT_ROOT:-${REPO_ROOT}/kernel-agent}"
FRAMEWORK_AGENT_ROOT="${FRAMEWORK_AGENT_ROOT:-${REPO_ROOT}/framework-agent}"
MAGPIE_REPO="${MAGPIE_REPO:-https://github.com/AMD-AGI/Magpie.git}"
MAGPIE_DIR="${MAGPIE_DIR:-${HYPERLOOM_RUNTIME_DIR}/Magpie}"

DRY_RUN=0
CHECK_ONLY=0
SKIP_KERNEL_AGENT=0
SKIP_FRAMEWORK_AGENT=0

usage() {
  cat <<'EOF'
Usage: inference_optimizer/scripts/install.sh [options]

Installs:
  - inference_optimizer Python package (with claude_agent_sdk via [test])
  - Magpie (cloned to $HYPERLOOM_RUNTIME_DIR/Magpie by default)
  - Detects/exports INFERENCEX_PATH
  - Chains to kernel-agent/scripts/install.sh for Ray + ray-head start,
    Node/npm, TraceLens, GEAK, OOB CLI, and the OOB auth-proxy.
  - Chains to framework-agent/scripts/install.sh for the `fa` CLI
    consumed by `--framework-pr-discover` / `--framework-pr` at
    optimize-time. framework-agent is fully standalone; the chain
    just makes the `fa` binary available on PATH inside the same
    sandbox without operators having to run a second installer.

Options:
  --check-only           Verify only, do not install
  --dry-run              Print actions without running them
  --skip-kernel-agent    Skip the chained kernel-agent installer
  --skip-framework-agent Skip the chained framework-agent installer
  -h, --help             Show this help

Env overrides:
  REPO_ROOT, KERNEL_AGENT_ROOT, FRAMEWORK_AGENT_ROOT, MAGPIE_REPO,
  MAGPIE_DIR, INFERENCEX_PATH, PYTHON, TRACELENS_ROOT, USER_DATA_PATH,
  HYPERLOOM_RUNTIME_DIR, KERNEL_AGENT_ENV, HYPERLOOM_ROOT
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --check-only) CHECK_ONLY=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --skip-kernel-agent) SKIP_KERNEL_AGENT=1 ;;
    --skip-framework-agent) SKIP_FRAMEWORK_AGENT=1 ;;
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
log "USER_DATA_PATH=${USER_DATA_PATH}"
log "HYPERLOOM_RUNTIME_DIR=${HYPERLOOM_RUNTIME_DIR}"
log "HYPERLOOM_ROOT=${HYPERLOOM_ROOT}"
log "KERNEL_AGENT_ROOT=${KERNEL_AGENT_ROOT}"
log "KERNEL_AGENT_ENV=${KERNEL_AGENT_ENV}"
log "MAGPIE_DIR=${MAGPIE_DIR}"
export USER_DATA_PATH HYPERLOOM_RUNTIME_DIR KERNEL_AGENT_ENV
export HYPERLOOM_KERNEL_AGENT_ROOT="${HYPERLOOM_KERNEL_AGENT_ROOT:-${KERNEL_AGENT_ROOT}}"
# Pre-create the writable runtime root so ensure_magpie / chain_kernel_agent
# never race on missing parents (Magpie's pip install -e writes egg-info
# under MAGPIE_DIR; install.sh of kernel-agent writes geak-config /
# kernel-agent.env.sh into HYPERLOOM_RUNTIME_DIR).
if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
  mkdir -p "${HYPERLOOM_RUNTIME_DIR}"
fi

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
    mkdir -p "$(dirname "$MAGPIE_DIR")"
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
      "${HYPERLOOM_RUNTIME_DIR}/InferenceX" \
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
  export REPO_ROOT KERNEL_AGENT_ROOT MAGPIE_DIR HYPERLOOM_ROOT
  export USER_DATA_PATH HYPERLOOM_RUNTIME_DIR KERNEL_AGENT_ENV
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

# --- 5. Chain to framework-agent ---
# Mirrors chain_kernel_agent but for the `fa` CLI used by
# --framework-pr-discover. framework-agent's installer is fully
# self-contained (zero shared state with kernel-agent), so we just
# delegate. Failures here are non-fatal: the IO main path still
# works without fa; only --framework-pr-discover requires it.
chain_framework_agent() {
  if [ "$SKIP_FRAMEWORK_AGENT" -eq 1 ]; then
    log "skipping framework-agent installer (--skip-framework-agent)"
    return 0
  fi
  local script="${FRAMEWORK_AGENT_ROOT}/scripts/install.sh"
  if [ ! -f "$script" ]; then
    warn "framework-agent installer not found at $script; --framework-pr-discover will be unavailable"
    return 0
  fi
  log "delegating fa CLI install to ${script}"
  export REPO_ROOT FRAMEWORK_AGENT_ROOT
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would run: bash '$script'"
    return 0
  fi
  bash "$script" || warn "framework-agent install returned non-zero; --framework-pr-discover will fail at runtime"
}

ensure_inference_optimizer
ensure_magpie
ensure_inferencex
chain_kernel_agent
chain_framework_agent

_probe_framework_source_roots() {
  log "probing framework source roots for INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS"
  local roots
  roots="$("$PYTHON" - <<'PY'
from inference_optimizer.orchestrator.framework_paths import probe_framework_source_roots_for_env
print(probe_framework_source_roots_for_env())
PY
)"
  if [ -z "$roots" ]; then
    warn "no framework source roots discovered"
    return 0
  fi
  log "discovered framework roots: $roots"
  if [ "$DRY_RUN" -eq 1 ] || [ "$CHECK_ONLY" -eq 1 ]; then
    log "would append INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS=$roots to ${KERNEL_AGENT_ENV}"
    return 0
  fi
  mkdir -p "$(dirname "$KERNEL_AGENT_ENV")"
  if [ -f "$KERNEL_AGENT_ENV" ] && grep -q '^export INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS=' "$KERNEL_AGENT_ENV" 2>/dev/null; then
    sed -i "s|^export INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS=.*|export INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS=${roots}|" "$KERNEL_AGENT_ENV"
  else
    {
      echo ""
      echo "# Framework source roots for PolicyGate + flag discovery (auto-probed)"
      echo "export INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS=${roots}"
    } >> "$KERNEL_AGENT_ENV"
  fi
}

_probe_framework_source_roots

log "install complete"
log "next: source ${KERNEL_AGENT_ENV}, then run inference_optimizer.cli"
