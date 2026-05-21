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
#   2b. Atomic-write patch for Magpie._prepare_benchmark_scripts
#       (bugs.md §C #1 root-cause fix; fail-loud)
#   3. InferenceX checkout: clone latest from upstream (no SHA pin yet),
#      sets INFERENCEX_PATH for runtime
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
MAGPIE_REPO="${MAGPIE_REPO:-https://github.com/AMD-AGI/Magpie.git}"
MAGPIE_DIR="${MAGPIE_DIR:-${HYPERLOOM_RUNTIME_DIR}/Magpie}"
INFERENCEX_REPO="${INFERENCEX_REPO:-https://github.com/SemiAnalysisAI/InferenceX.git}"
INFERENCEX_DEFAULT_DIR="${INFERENCEX_DEFAULT_DIR:-${HYPERLOOM_RUNTIME_DIR}/InferenceX}"

DRY_RUN=0
CHECK_ONLY=0
SKIP_KERNEL_AGENT=0

usage() {
  cat <<'EOF'
Usage: inference_optimizer/scripts/install.sh [options]

Installs:
  - inference_optimizer Python package (with claude_agent_sdk via [test])
  - Magpie (cloned to $HYPERLOOM_RUNTIME_DIR/Magpie by default)
  - Detects/exports INFERENCEX_PATH
  - Chains to kernel-agent/scripts/install.sh for Ray + ray-head start,
    Node/npm, TraceLens, GEAK, OOB CLI, and the OOB auth-proxy.

Options:
  --check-only          Verify only, do not install
  --dry-run             Print actions without running them
  --skip-kernel-agent   Skip the chained kernel-agent installer
  -h, --help            Show this help

Env overrides:
  REPO_ROOT, KERNEL_AGENT_ROOT, MAGPIE_REPO, MAGPIE_DIR,
  INFERENCEX_REPO, INFERENCEX_DEFAULT_DIR, INFERENCEX_PATH,
  PYTHON, TRACELENS_ROOT, USER_DATA_PATH,
  HYPERLOOM_RUNTIME_DIR, KERNEL_AGENT_ENV, HYPERLOOM_ROOT,
  PATCH_MAGPIE (=1; set 0 only if upstream Magpie atomic-write
  PR is already merged into your clone)
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
log "USER_DATA_PATH=${USER_DATA_PATH}"
log "HYPERLOOM_RUNTIME_DIR=${HYPERLOOM_RUNTIME_DIR}"
log "HYPERLOOM_ROOT=${HYPERLOOM_ROOT}"
log "KERNEL_AGENT_ROOT=${KERNEL_AGENT_ROOT}"
log "KERNEL_AGENT_ENV=${KERNEL_AGENT_ENV}"
log "MAGPIE_DIR=${MAGPIE_DIR}"
log "INFERENCEX_REPO=${INFERENCEX_REPO}"
log "INFERENCEX_DEFAULT_DIR=${INFERENCEX_DEFAULT_DIR}"
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

# --- 2b. Atomic-write patch for Magpie._prepare_benchmark_scripts ---
# Hyperloom bugs.md §C #1 (vllm_mi300x.sh / sglang_mi300x.sh sourced by a
# leaked bash while a new Magpie subprocess is mid-`shutil.copy2` →
# `syntax error near unexpected token 'fi'`). Magpie is invoked as a
# subprocess, so monkey-patching from the Coordinator process does not
# reach it; we patch the cloned source in place at install time. The
# patcher itself is idempotent + flock-serialised + atomic-rename
# (see `_magpie_patcher.py`), so re-runs are O(1) no-ops.
#
# Escalation: this is a known root-cause fix. A `False` return means
# Magpie was refactored upstream and our pattern no longer matches; we
# `die` so the operator notices instead of silently shipping a broken
# install. Override the gate via PATCH_MAGPIE=0 for the (rare) case
# where you've already landed an upstream PR locally.
ensure_magpie_atomic_scripts_patch() {
  if [ "${PATCH_MAGPIE:-1}" -eq 0 ]; then
    log "PATCH_MAGPIE=0 — skipping Magpie atomic-write patch (caller asserts upstream already fixed)"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would apply Hyperloom #C1 atomic-write patch to ${MAGPIE_DIR}/Magpie/modes/benchmark/benchmarker.py"
    return 0
  fi
  log "applying Hyperloom #C1 atomic-write patch to Magpie._prepare_benchmark_scripts"
  MAGPIE_DIR="$MAGPIE_DIR" "$PYTHON" - <<'PY' || die "Magpie atomic-write patch failed; see warnings above. bugs.md §C #1 is NOT mitigated. Set PATCH_MAGPIE=0 to skip if you know what you are doing."
import os, sys
from inference_optimizer.orchestrator.action_executors._magpie_patcher import (
    ensure_magpie_atomic_scripts_patch,
)
ok = ensure_magpie_atomic_scripts_patch(os.environ["MAGPIE_DIR"])
sys.exit(0 if ok else 1)
PY
  log "Magpie #C1 patch OK"
}

# --- 3. InferenceX checkout: fresh clone from upstream ---
#
# Previously this function scanned a list of shared-filesystem candidates
# (`/wekafs/hyperloom/InferenceX`, `/wekafs/fully-local/.../InferenceX`,
# etc.) and pointed every install at whichever it found first. That
# multi-install / shared-checkout layout is the upstream source of the
# concurrent-write races in bugs.md §C #1 — every fresh Magpie
# subprocess `shutil.copy2`'d its scripts on top of the same shared
# files, while bash interpreters from neighbouring installs were
# `source`-ing them. Cloning a per-install copy here eliminates the
# cross-install fan-in (Magpie's in-place atomic-write patch then
# closes the intra-install race window — both fixes are needed; this
# one alone is not sufficient).
#
# Policy:
#   * INFERENCEX_PATH set and exists -> preserve verbatim. This is the
#     dev / CI override (caller is explicitly opting out of fresh
#     clones, e.g. iterating on a local edit).
#   * Otherwise -> always `git clone --depth 1` from INFERENCEX_REPO
#     into INFERENCEX_DEFAULT_DIR. If a clone already exists there from
#     a previous install we leave it as-is (idempotent re-runs) — the
#     per-install isolation guarantee is already met, and re-cloning
#     would just churn benchmark scripts that the Magpie patch already
#     keeps consistent on disk.
#   * No SHA pin yet (deferred). We record whatever commit `git clone`
#     resolves into the session manifest (see manifest.py) so failed
#     runs can be reproduced.
ensure_inferencex() {
  if [ -n "${INFERENCEX_PATH:-}" ] && [ -d "$INFERENCEX_PATH" ]; then
    log "INFERENCEX_PATH = $INFERENCEX_PATH (preserved from env; skipping fresh clone)"
    export INFERENCEX_PATH
    return 0
  fi
  INFERENCEX_PATH="$INFERENCEX_DEFAULT_DIR"
  if [ -d "$INFERENCEX_PATH/.git" ] || [ -d "$INFERENCEX_PATH/benchmarks" ]; then
    log "InferenceX already cloned at ${INFERENCEX_PATH}; preserving existing checkout"
    export INFERENCEX_PATH
    return 0
  fi
  if [ "$CHECK_ONLY" -eq 1 ]; then
    warn "InferenceX not present at ${INFERENCEX_PATH} (check-only mode, skipping clone)"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would: git clone --depth 1 ${INFERENCEX_REPO} ${INFERENCEX_PATH}"
    export INFERENCEX_PATH
    return 0
  fi
  log "cloning fresh InferenceX from ${INFERENCEX_REPO} -> ${INFERENCEX_PATH}"
  mkdir -p "$(dirname "$INFERENCEX_PATH")"
  if ! git clone --depth 1 "$INFERENCEX_REPO" "$INFERENCEX_PATH"; then
    warn "InferenceX clone failed. GSM8K eval will fail without it. Set"
    warn "INFERENCEX_PATH to a pre-cloned tree to skip this step."
    return 0
  fi
  export INFERENCEX_PATH
  log "InferenceX cloned at ${INFERENCEX_PATH}"
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

ensure_inference_optimizer
ensure_magpie
ensure_magpie_atomic_scripts_patch
ensure_inferencex
chain_kernel_agent

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
