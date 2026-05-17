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

# Ray/K8s subprocesses may inherit a minimal PATH; git/apt live under /usr/bin.
# Prepend the standard system bins so multi-node RayJob subprocesses (and any
# K8s-spawned child shell) still resolve git/apt/python3 when callers only
# prepend /opt/venv/bin.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"

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
  INFERENCEX_PATH, PYTHON, TRACELENS_ROOT, USER_DATA_PATH,
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
#
# bare-image bootstrap fallback: when nothing in the search order
# exists AND apt-get is available (Debian/Ubuntu sandbox), we try a
# best-effort `apt-get install -y python3 python3-venv python3-pip`
# before giving up. This makes install.sh truly "bare-image" capable
# (the file's own doc comment claims it) instead of punting back to
# the caller. The apt-get call is gated by:
#   * apt-get binary present
#   * not --check-only / --dry-run (those modes are non-mutating)
#   * INFERENCE_OPTIMIZER_SKIP_APT_BOOTSTRAP env unset (escape hatch)
# Any failure here still hits die() with the original error message,
# so callers on non-apt images keep the same exit semantics.
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

  # Bare-image bootstrap (Debian/Ubuntu only). Skipped silently when
  # apt-get is missing (RHEL/Alpine/etc.) or the operator opted out.
  if command -v apt-get >/dev/null 2>&1 \
      && [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ] \
      && [ -z "${INFERENCE_OPTIMIZER_SKIP_APT_BOOTSTRAP:-}" ]; then
    log "no python3 found; attempting bare-image apt bootstrap " \
        "(set INFERENCE_OPTIMIZER_SKIP_APT_BOOTSTRAP=1 to disable)"
    export DEBIAN_FRONTEND=noninteractive
    if apt-get update -qq >/dev/null 2>&1 \
        && apt-get install -y --no-install-recommends \
              python3 python3-venv python3-pip >/dev/null 2>&1; then
      if command -v python3 >/dev/null 2>&1; then
        PYTHON="$(command -v python3)"
        log "apt bootstrap succeeded: PYTHON=$PYTHON"
        return 0
      fi
    fi
    warn "apt bootstrap failed; falling through to die()"
  fi

  die "no usable python found (set PYTHON, install python3, mount /opt/venv, " \
      "or run on an apt-based image so install.sh can bootstrap python3 itself)"
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

# --- 4. InferenceX bench_serving runtime deps ---
#
# `benchmark_serving.py` lives under InferenceX (not under Magpie's
# pyproject.toml), so `pip install -e Magpie` does NOT pull its client-side
# dependencies. Without these, every Magpie variant launch dies with
# `ModuleNotFoundError: No module named 'aiohttp'` (or transformers,
# huggingface_hub, datasets, ...) BEFORE the sglang server is even hit.
#
# We install into the same $PYTHON that Magpie uses (resolved to
# `/opt/venv/bin/python3` on Claw sandboxes via the active PATH at run
# time). The version pins are intentionally loose: these are stable
# client-only packages and we want to inherit whatever the container's
# base image already has rather than forcing churn.
_BENCH_SERVING_DEPS=(
  aiohttp
  tqdm
  numpy
  requests
  transformers
  huggingface_hub
  datasets
  pandas
)

ensure_bench_serving_deps() {
  log "ensuring InferenceX benchmark_serving client deps in $PYTHON"
  local missing=()
  for m in "${_BENCH_SERVING_DEPS[@]}"; do
    # Map pip name -> import name (only aiohttp/etc. happen to match).
    local import_name="$m"
    case "$m" in
      huggingface_hub) import_name="huggingface_hub" ;;
    esac
    if ! "$PYTHON" -c "import ${import_name}" >/dev/null 2>&1; then
      missing+=("$m")
    fi
  done
  if [ ${#missing[@]} -eq 0 ]; then
    log "bench_serving deps already satisfied"
    return 0
  fi
  log "installing missing bench_serving deps: ${missing[*]}"
  if [ "$CHECK_ONLY" -eq 1 ]; then
    warn "check-only mode; would install: ${missing[*]}"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "dry-run; skipping pip install"
    return 0
  fi
  "$PYTHON" -m pip install --quiet --no-cache-dir \
    "${PIP_EXTRA[@]}" "${missing[@]}" \
    || die "failed to install bench_serving deps: ${missing[*]}"
  for m in "${missing[@]}"; do
    "$PYTHON" -c "import ${m}" >/dev/null 2>&1 \
      || die "bench_serving dep ${m} still not importable after install"
  done
  log "bench_serving deps installed OK"
}

# --- 5. Chain to kernel-agent ---
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
ensure_inferencex
ensure_bench_serving_deps
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
