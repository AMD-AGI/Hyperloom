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
#      bring-up, Node/npm, TraceLens, GEAK, OOB and CLI auth-file setup.
#      kernel-agent itself is the canonical owner of those — we just
#      chain to it so users have a single entry point.
#
# kernel-agent's install.sh owns Ray + ray start, TraceLens, GEAK, OOB
# CLI auth files. inference_optimizer's install.sh owns Magpie /
# InferenceX / the inference_optimizer Python package itself. The two
# are composable: kernel-agent works standalone; inference_optimizer
# drags kernel-agent in via this script.

set -euo pipefail

# Ray/K8s subprocesses may inherit a minimal PATH; git/apt live under /usr/bin.
# Prepend the standard system bins so multi-node RayJob subprocesses (and any
# K8s-spawned child shell) still resolve git/apt/python3 when callers only
# prepend /opt/venv/bin.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"

# Single artefact root: everything writable defaults to $USER_DATA_PATH so
# operators can monitor a run end-to-end by tailing one directory. Magpie
# clone, source mirrors, and generated env / GEAK config all derive from
# $HYPERLOOM_RUNTIME_DIR.
# Removed envs: WORKSPACE_ROOT / WORKSPACE_PATH (collapsed into USER_DATA_PATH).
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
DOTENV_LOADED_COUNT=0

load_dotenv_no_clobber() {
  DOTENV_LOADED_COUNT=0
  [ -f "$REPO_ROOT/.env" ] || return 0
  local loaded=0
  local raw key value
  while IFS= read -r raw || [ -n "$raw" ]; do
    raw="${raw#"${raw%%[![:space:]]*}"}"
    raw="${raw%"${raw##*[![:space:]]}"}"
    [ -z "$raw" ] && continue
    case "$raw" in \#*) continue ;; esac
    case "$raw" in export\ *) raw="${raw#export }" ;; esac
    case "$raw" in *=*) ;; *) continue ;; esac
    key="${raw%%=*}"
    value="${raw#*=}"
    key="${key%"${key##*[![:space:]]}"}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    case "$value" in
      \"*\") value="${value#\"}"; value="${value%\"}" ;;
      \'*\') value="${value#\'}"; value="${value%\'}" ;;
    esac
    [ -z "$key" ] && continue
    if [ -z "${!key:-}" ]; then
      export "$key=$value"
      loaded=$((loaded + 1))
    fi
  done < "$REPO_ROOT/.env"
  DOTENV_LOADED_COUNT="$loaded"
  return 0
}

# Load .env before deriving USER_DATA_PATH / HYPERLOOM_RUNTIME_DIR so a
# freshly-copied .env.template can be the single configuration entrypoint.
# The loader is no-clobber: explicit shell exports always win.
load_dotenv_no_clobber
USER_DATA_PATH="${USER_DATA_PATH:-/workspace/hyperloom}"
HYPERLOOM_RUNTIME_DIR="${HYPERLOOM_RUNTIME_DIR:-${USER_DATA_PATH}/runtime}"
KERNEL_AGENT_ENV="${KERNEL_AGENT_ENV:-${HYPERLOOM_RUNTIME_DIR}/kernel-agent.env.sh}"
HYPERLOOM_ROOT="${HYPERLOOM_ROOT:-${HYPERLOOM_RUNTIME_DIR}/source-mirrors}"
KERNEL_AGENT_ROOT="${KERNEL_AGENT_ROOT:-${REPO_ROOT}/kernel-agent}"
FRAMEWORK_AGENT_ROOT="${FRAMEWORK_AGENT_ROOT:-${REPO_ROOT}/framework-agent}"
MAGPIE_REPO="${MAGPIE_REPO:-https://github.com/AMD-AGI/Magpie.git}"
MAGPIE_DIR="${MAGPIE_DIR:-${HYPERLOOM_RUNTIME_DIR}/Magpie}"
INFERENCEX_REPO="${INFERENCEX_REPO:-https://github.com/SemiAnalysisAI/InferenceX.git}"
INFERENCEX_DEFAULT_DIR="${INFERENCEX_DEFAULT_DIR:-${HYPERLOOM_RUNTIME_DIR}/InferenceX}"

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
    Node/npm, TraceLens, GEAK, and OOB CLI auth.
  - Chains to framework-agent/scripts/install.sh for the `fa` CLI
    used by the `framework_pr` bandit arm at optimize-time.
    framework-agent is fully standalone; the chain just makes the
    `fa` binary available on PATH inside the same sandbox without
    operators having to run a second installer.

Options:
  --check-only           Verify only, do not install
  --dry-run              Print actions without running them
  --skip-kernel-agent    Skip the chained kernel-agent installer
  --skip-framework-agent Skip the chained framework-agent installer
  -h, --help             Show this help

Env overrides:
  REPO_ROOT, KERNEL_AGENT_ROOT, FRAMEWORK_AGENT_ROOT, MAGPIE_REPO,
  MAGPIE_DIR, INFERENCEX_REPO, INFERENCEX_DEFAULT_DIR, INFERENCEX_PATH,
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

# Preflight credential validation. Mirrors the gate in
# kernel-agent/scripts/install.sh so users invoking the inference-optimizer
# installer directly (the canonical entrypoint) get the same fail-fast
# behaviour as users running kernel-agent on its own. Without this, a
# missing SAFE_API_KEY / OPENAI_BASE_URL slips past pip install, Magpie
# clone, InferenceX clone (~10+ minutes of work) and only surfaces when the
# chained kernel-agent installer reaches GEAK config generation.
#
# Loader (env wins; never overwrites a key that is already set):
#   env > $REPO_ROOT/.env
#
# Strict mode by design: --check-only / --dry-run is the only path that
# downgrades the die to a warn (introspection mode, no install runs).
preflight_load_dotenv() {
  load_dotenv_no_clobber
  if [ "${DOTENV_LOADED_COUNT:-0}" -gt 0 ]; then
    log "loaded ${DOTENV_LOADED_COUNT} missing var(s) from $REPO_ROOT/.env (env wins)"
  fi
}

preflight_validate_credentials() {
  preflight_load_dotenv
  local missing=()
  [ -z "${SAFE_API_KEY:-}" ]    && missing+=("SAFE_API_KEY")
  [ -z "${OPENAI_BASE_URL:-}" ] && missing+=("OPENAI_BASE_URL")
  if [ "${#missing[@]}" -eq 0 ]; then
    log "credentials preflight: SAFE_API_KEY + OPENAI_BASE_URL present"
    return 0
  fi
  local env_file_status
  if [ -f "$REPO_ROOT/.env" ]; then
    env_file_status="present"
  else
    env_file_status="not found"
  fi
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    warn "missing credential(s): ${missing[*]} (.env=${env_file_status}); " \
         "continuing because --check-only / --dry-run is active. The " \
         "chained kernel-agent installer will still fail later unless " \
         "these are set before a real install."
    return 0
  fi
  cat >&2 <<EOF
[inference-optimizer ERROR] Missing required credential(s): ${missing[*]}

Tried loading from:
  - shell environment
  - \$REPO_ROOT/.env  (${env_file_status}: ${REPO_ROOT}/.env)

Fix one of:
  1. Copy .env from a working worktree into this one:
       cp /path/to/main-worktree/.env "${REPO_ROOT}/.env"
  2. Export directly into the shell before re-running:
       export SAFE_API_KEY=sk-xxxxx
       export OPENAI_BASE_URL=https://gateway.example.com/v1
EOF
  exit 2
}
preflight_validate_credentials

# --- 0. Resolve PYTHON ---
# On hyperloom / sgl-workspace containers the canonical ROCm stack lives in
# /opt/venv (preinstalled torch+rocm, sglang, vllm, aiter, sgl_kernel,
# triton, Magpie, inference_optimizer, claude_agent_sdk, ray). Always
# prefer that interpreter — bare-image PYTHONs (e.g. /usr/bin/python3) on a
# ROCm pod silently pull plain `torch` from PyPI on `pip install -e .[test]`,
# which is the NVIDIA CUDA wheel and crashes downstream RAG / baseline
# steps with "Found no NVIDIA driver". Operators who really need a custom
# interpreter can opt out with INFERENCE_OPTIMIZER_FORCE_PYTHON=1.
#
# bare-image bootstrap fallback: when nothing in the search order exists AND
# apt-get is available (Debian/Ubuntu sandbox), try a best-effort
# `apt-get install -y python3 python3-venv python3-pip` before giving up.
# Gated by apt-get present, not --check-only / --dry-run, and
# INFERENCE_OPTIMIZER_SKIP_APT_BOOTSTRAP unset.
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
  log "delegating ray + TraceLens + GEAK + OOB CLI auth to ${script}"
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
# Mirrors chain_kernel_agent but for the `fa` CLI used by the
# `framework_pr` bandit arm. framework-agent's installer is fully
# self-contained (zero shared state with kernel-agent), so we just
# delegate. Failures here are non-fatal: the IO main path still
# works without fa; only `framework_pr` arm ticks require it.
chain_framework_agent() {
  if [ "$SKIP_FRAMEWORK_AGENT" -eq 1 ]; then
    log "skipping framework-agent installer (--skip-framework-agent)"
    return 0
  fi
  local script="${FRAMEWORK_AGENT_ROOT}/scripts/install.sh"
  if [ ! -f "$script" ]; then
    warn "framework-agent installer not found at $script; framework_pr arm will be unavailable"
    return 0
  fi
  log "delegating fa CLI install to ${script}"
  # Pass through the resolved PYTHON so framework-agent's installer picks
  # the same /opt/venv interpreter (avoids /usr/bin/pip 22.0.2 which fails
  # the PEP 660 build_editable hook on our pyproject).
  export REPO_ROOT FRAMEWORK_AGENT_ROOT
  export VENV_PYTHON="${PYTHON}"
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would run: bash '$script' (VENV_PYTHON=${VENV_PYTHON})"
    return 0
  fi
  bash "$script" || warn "framework-agent install returned non-zero; framework_pr arm will fail at runtime"
}

ensure_inference_optimizer
ensure_magpie
ensure_magpie_atomic_scripts_patch
ensure_inferencex
ensure_bench_serving_deps
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

# ---------------------------------------------------------------------------
# framework-agent (sibling skill — drives the standalone FRAMEWORK_PR
# phase via ``fa phase-discover`` for batch enumeration. The
# Coordinator's executor handles the apply/bench loop directly, so
# ``phase-fetch`` / ``phase-emit-proposal`` ship for ad-hoc use but
# are not on the inference_optimizer hot path). Owns its own python
# deps and venv layout; we only need to invoke its installer.
#
# Install is ON by default to match the runtime default
# (``SharedState.framework_phase_enabled = True``). Opt out by
# exporting ``INFERENCE_OPTIMIZER_NO_FRAMEWORK=1`` before install
# (mirrors the runtime ``--no-framework`` CLI flag).
#
# Back-compat: the legacy ``INFERENCE_OPTIMIZER_FRAMEWORK_AGENT_ENABLED=0``
# knob is still honoured for one release with a deprecation warning so
# operator scripts don't break. Remove on the next cleanup pass.
# ---------------------------------------------------------------------------
ensure_framework_agent() {
  if [ -n "${INFERENCE_OPTIMIZER_FRAMEWORK_AGENT_ENABLED:-}" ]; then
    warn "INFERENCE_OPTIMIZER_FRAMEWORK_AGENT_ENABLED is deprecated; use INFERENCE_OPTIMIZER_NO_FRAMEWORK=1 to opt out"
    if [ "${INFERENCE_OPTIMIZER_FRAMEWORK_AGENT_ENABLED}" = "0" ]; then
      log "framework-agent: skipped (legacy INFERENCE_OPTIMIZER_FRAMEWORK_AGENT_ENABLED=0)"
      return 0
    fi
  fi
  if [ "${INFERENCE_OPTIMIZER_NO_FRAMEWORK:-0}" = "1" ]; then
    log "framework-agent: skipped (INFERENCE_OPTIMIZER_NO_FRAMEWORK=1)"
    return 0
  fi
  local fa_dir="${INFERENCE_OPTIMIZER_REPO:-$(pwd)}/framework-agent"
  if [ ! -d "$fa_dir" ]; then
    warn "framework-agent: directory missing at $fa_dir — skipping"
    return 0
  fi
  if [ ! -f "$fa_dir/scripts/install.sh" ]; then
    warn "framework-agent: $fa_dir/scripts/install.sh missing — skipping"
    return 0
  fi
  log "framework-agent: installing from $fa_dir"
  if [ "$DRY_RUN" -eq 1 ] || [ "$CHECK_ONLY" -eq 1 ]; then
    log "would run: bash '$fa_dir/scripts/install.sh'"
    return 0
  fi
  bash "$fa_dir/scripts/install.sh"
  log "framework-agent: install complete"
}

ensure_framework_agent

log "install complete"
log "kernel-agent env file written: ${KERNEL_AGENT_ENV}"
log "  HYPERLOOM_KERNEL_AGENT_ROOT=${HYPERLOOM_KERNEL_AGENT_ROOT}"
log ""
log "next steps — pick ONE:"
log "  (a) source ${KERNEL_AGENT_ENV}, then run inference_optimizer.cli"
log "  (b) just launch inference_optimizer.cli — preflight will auto-source"
log "      \$KERNEL_AGENT_ENV (or \$USER_DATA_PATH/runtime/kernel-agent.env.sh)"
log "      via _load_kernel_agent_env_fallback() if HYPERLOOM_KERNEL_AGENT_ROOT"
log "      is unset (added May 2026 after the R1 N14 stall — see"
log "      design/roofline-v2.md §6.6 if it exists)."
log ""
log "If you skip BOTH and HYPERLOOM_KERNEL_AGENT_ROOT stays unset, the"
log "roofline composite action's trace_analyze sub-step will fail with"
log "  'HYPERLOOM_KERNEL_AGENT_ROOT is not set'"
log "and the whole optimisation loop stalls (PolicyGate blocks every"
log "downstream action on a missing TraceLens snapshot)."
