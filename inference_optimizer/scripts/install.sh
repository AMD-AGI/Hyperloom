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
# On hyperloom / sgl-workspace containers the canonical ROCm stack lives in
# /opt/venv (preinstalled torch+rocm, sglang, vllm, aiter, sgl_kernel,
# triton, Magpie, inference_optimizer, claude_agent_sdk, ray). Always
# prefer that interpreter — bare-image PYTHONs (e.g. /usr/bin/python3) on a
# ROCm pod silently pull plain `torch` from PyPI on `pip install -e .[test]`,
# which is the NVIDIA CUDA wheel and crashes downstream RAG / baseline
# steps with "Found no NVIDIA driver". Operators who really need a custom
# interpreter can opt out with INFERENCE_OPTIMIZER_FORCE_PYTHON=1.
resolve_python() {
  if [ -x "/opt/venv/bin/python" ] && [ "${INFERENCE_OPTIMIZER_FORCE_PYTHON:-0}" != "1" ]; then
    if [ -n "${PYTHON:-}" ] && [ "${PYTHON}" != "/opt/venv/bin/python" ]; then
      log "preferring /opt/venv/bin/python over PYTHON=${PYTHON} (canonical ROCm stack)"
      log "  set INFERENCE_OPTIMIZER_FORCE_PYTHON=1 to honor PYTHON verbatim"
    fi
    PYTHON="/opt/venv/bin/python"
    return 0
  fi
  if [ -n "${PYTHON:-}" ] && [ -x "$PYTHON" ]; then
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
# Export PYTHON + prepend its bin dir so the chained kernel-agent installer's
# bare `python3 -m pip ...` calls (kernel-agent/scripts/install.sh) land in
# the same interpreter. Otherwise PATH-only resolution can split the
# installation across two different pythons.
export PYTHON
PATH="$(dirname "$PYTHON"):${PATH:-/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin}"
export PATH

# --- 0a. Torch compatibility gate (ROCm-aware) ---
# If rocm-smi reports devices, the resolved PYTHON must already have a
# ROCm-built torch importable. Two failure modes we explicitly catch:
#   1. torch missing entirely on a ROCm pod -- letting pip install proceed
#      will pull the NVIDIA CUDA wheel from PyPI (default `torch`).
#   2. torch present but built against CUDA (torch.version.hip is None)
#      -- the chained RAG-index step auto-detects device=cuda and crashes
#      at torch._C._cuda_init() with "Found no NVIDIA driver".
ensure_torch_compatible_with_gpu() {
  if ! command -v rocm-smi >/dev/null 2>&1; then
    return 0
  fi
  if ! rocm-smi --showid >/dev/null 2>&1; then
    return 0
  fi
  local probe
  probe="$("$PYTHON" - <<'PY' 2>/dev/null || true
import json, sys
out = {"rc": 0}
try:
    import torch
    out["torch_version"] = torch.__version__
    out["hip"] = getattr(torch.version, "hip", None)
    out["cuda_str"] = getattr(torch.version, "cuda", None)
except Exception as exc:
    out["rc"] = 2
    out["error"] = type(exc).__name__ + ": " + str(exc)[:200]
print(json.dumps(out))
PY
)"
  if [ -z "$probe" ]; then
    warn "torch probe produced no output (PYTHON=${PYTHON})"
    return 0
  fi
  local rc; rc="$("$PYTHON" -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('rc',0))" "$probe" 2>/dev/null || echo 0)"
  if [ "$rc" = "2" ]; then
    warn "torch is NOT importable from PYTHON=${PYTHON}"
    warn "this pod has ROCm GPUs (rocm-smi works) -- letting pip install proceed"
    warn "would pull plain 'torch' from PyPI (= NVIDIA CUDA wheel) and break"
    warn "downstream RAG / baseline / kernel steps with 'Found no NVIDIA driver'."
    warn "Fixes (pick one):"
    warn "  * use the canonical ROCm stack:   unset PYTHON; install.sh will pick /opt/venv"
    warn "  * install the ROCm torch wheel:    \"\$PYTHON\" -m pip install --pre torch --index-url https://download.pytorch.org/whl/rocm6.x"
    warn "  * opt out of this gate:            INFERENCE_OPTIMIZER_FORCE_PYTHON=1 INFERENCE_OPTIMIZER_SKIP_TORCH_GATE=1 install.sh"
    if [ "${INFERENCE_OPTIMIZER_SKIP_TORCH_GATE:-0}" != "1" ]; then
      die "refusing to install on ROCm pod with no torch in PYTHON=${PYTHON}"
    fi
    warn "INFERENCE_OPTIMIZER_SKIP_TORCH_GATE=1 set; continuing despite missing torch"
    return 0
  fi
  local hip; hip="$("$PYTHON" -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('hip') or '')" "$probe" 2>/dev/null || echo "")"
  local tv;  tv="$("$PYTHON"  -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('torch_version') or '')" "$probe" 2>/dev/null || echo "")"
  if [ -z "$hip" ]; then
    warn "torch=${tv} in PYTHON=${PYTHON} is NOT a ROCm build (torch.version.hip is None)"
    warn "but this pod reports ROCm GPUs via rocm-smi. RAG-index / baseline / kernel"
    warn "steps will crash at torch._C._cuda_init() with 'Found no NVIDIA driver'."
    warn "Fixes (pick one):"
    warn "  * use the canonical ROCm stack:   unset PYTHON; install.sh will pick /opt/venv"
    warn "  * install the ROCm torch wheel:    \"\$PYTHON\" -m pip install --force-reinstall --pre torch --index-url https://download.pytorch.org/whl/rocm6.x"
    warn "  * opt out of this gate:            INFERENCE_OPTIMIZER_SKIP_TORCH_GATE=1 install.sh"
    if [ "${INFERENCE_OPTIMIZER_SKIP_TORCH_GATE:-0}" != "1" ]; then
      die "refusing to install: torch=${tv} is CUDA-built on a ROCm pod"
    fi
    warn "INFERENCE_OPTIMIZER_SKIP_TORCH_GATE=1 set; continuing despite torch/GPU mismatch"
    return 0
  fi
  log "torch=${tv} (hip=${hip}) -- ROCm-compatible OK"
}

ensure_torch_compatible_with_gpu
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
# sys.prefix vs sys.base_prefix; equal == not in venv. The flag was added
# in pip 23.0.1; older pips reject it as an unknown option, so we probe
# `pip install --break-system-packages --help` before adopting it.
PIP_EXTRA=()
if "$PYTHON" - <<'PY' 2>/dev/null
import sys
raise SystemExit(0 if sys.prefix == sys.base_prefix else 1)
PY
then
  if "$PYTHON" -m pip install --break-system-packages --help >/dev/null 2>&1; then
    PIP_EXTRA=(--break-system-packages)
    log "non-venv PYTHON; pip will use --break-system-packages"
  else
    pip_ver="$("$PYTHON" -m pip --version 2>&1 | awk '{print $2}')"
    warn "non-venv PYTHON detected (PYTHON=${PYTHON}) but pip ${pip_ver}"
    warn "is too old for --break-system-packages (requires >= 23.0.1)."
    warn "Fixes (pick one):"
    warn "  * use the canonical ROCm stack: unset PYTHON; install.sh will pick /opt/venv"
    warn "  * create a venv:                python3 -m venv \"\$USER_DATA_PATH/venv\" \\"
    warn "                                  && \"\$USER_DATA_PATH/venv/bin/python\" -m pip install -U pip wheel \\"
    warn "                                  && export PYTHON=\"\$USER_DATA_PATH/venv/bin/python\""
    warn "  * upgrade system pip:           \"\$PYTHON\" -m pip install --user -U 'pip>=23.0.1'"
    die "refusing to run pip without a working --break-system-packages on a non-venv interpreter"
  fi
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

ensure_inference_optimizer
ensure_magpie
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
