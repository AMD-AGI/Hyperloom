#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc. All rights reserved.

# Inference Optimizer installer.
#
# Owns the inference_optimizer-side bare-image setup so SKILL.md does
# not have to hand-roll it. Idempotent — every step skips if the
# artifact is already present.
#
# Stack (in order):
#   1. inference_optimizer + extras (pulls in claude_agent_sdk via
#      pyproject `[test]` extra)
#   2. Magpie (benchmark engine) into the pod-local open-source repo tree,
#      pinned to MAGPIE_REF (a commit SHA or tag)
#   2b. Atomic-write patch for Magpie._prepare_benchmark_scripts
#       (bugs.md §C #1 root-cause fix; fail-soft — a no-op when the
#       MAGPIE_REF target already has upstream atomic copying)
#   3. InferenceX checkout: clone from upstream pinned to INFERENCEX_REF
#      (a commit SHA), sets INFERENCEX_PATH for runtime
#   4. Delegates to src/hyperloom/agents/kernel/scripts/install.sh for ray, ray-head
#      bring-up, TraceLens, GEAK and LLM gateway env setup.
#      kernel-agent itself is the canonical owner of those — we just
#      chain to it so users have a single entry point.
#
# kernel-agent's install.sh owns Ray + ray start, TraceLens, GEAK and
# LLM gateway env. inference_optimizer's install.sh owns Magpie /
# InferenceX / the inference_optimizer Python package itself. The two
# are composable: kernel-agent works standalone; inference_optimizer
# drags kernel-agent in via this script.
#
# Open-source deps (InferenceX / TraceLens) are cloned here or by the
# chained kernel-agent installer.

set -euo pipefail

# Ray/K8s subprocesses may inherit a minimal PATH; git/apt live under /usr/bin.
# Prepend the standard system bins so multi-node RayJob subprocesses (and any
# K8s-spawned child shell) still resolve git/apt/python3 when callers only
# prepend /opt/venv/bin.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
# Re-assert the active virtualenv ahead of the system bins prepended above.
# Callers may only put the venv on PATH (e.g. /venv/bin) or activate it via
# $VIRTUAL_ENV; otherwise the system-bins prepend shadows the venv python3
# with /usr/bin/python3, whose apt-managed packages (e.g. packaging) have no
# RECORD file and break `pip install`/uninstall. Probe the activated venv
# first, then the common ROCm image locations (/opt/venv, /venv).
for _venv_bin in "${VIRTUAL_ENV:+${VIRTUAL_ENV}/bin}" /opt/venv/bin /venv/bin; do
  if [ -n "${_venv_bin}" ] && [ -x "${_venv_bin}/python" ]; then
    export PATH="${_venv_bin}:$PATH"
    break
  fi
done

# Single artefact root: everything writable defaults to $USER_DATA_PATH so
# operators can monitor a run end-to-end by tailing one directory. Magpie
# clone, source mirrors, and generated env / GEAK config all derive from
# $HYPERLOOM_RUNTIME_DIR.
# Removed envs: WORKSPACE_ROOT / WORKSPACE_PATH (collapsed into USER_DATA_PATH).
_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_repo_root() {
  if [ -n "${REPO_ROOT:-}" ]; then
    printf '%s\n' "$REPO_ROOT"
    return 0
  fi
  local source_root packaged_root
  source_root="$(cd "${_script_dir}/../../../.." && pwd)"
  packaged_root="$(cd "${_script_dir}/../../.." && pwd)"
  if [ -f "${source_root}/pyproject.toml" ]; then
    printf '%s\n' "$source_root"
  else
    printf '%s\n' "$packaged_root"
  fi
}

REPO_ROOT="$(resolve_repo_root)"
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
# Capture whether USER_DATA_PATH was provided BEFORE applying the default so we
# can warn loudly on the silent fallback. ${VAR:+1} is empty when VAR is unset
# or empty, which is exactly the case the :- default below would absorb.
_user_data_was_set="${USER_DATA_PATH:+1}"
USER_DATA_PATH="${USER_DATA_PATH:-/workspace/hyperloom}"
if [ -z "${_user_data_was_set}" ]; then
  echo "[install WARN] USER_DATA_PATH not set; defaulting to /workspace/hyperloom. Set USER_DATA_PATH to persist artifacts under your data root." >&2
fi
HYPERLOOM_RUNTIME_DIR="${HYPERLOOM_RUNTIME_DIR:-${USER_DATA_PATH}/runtime}"
KERNEL_AGENT_ENV="${KERNEL_AGENT_ENV:-${HYPERLOOM_RUNTIME_DIR}/kernel-agent.env.sh}"
# Legacy variable kept for compatibility; open-source checkouts use _open_source_root.
HYPERLOOM_ROOT="${HYPERLOOM_ROOT:-${HYPERLOOM_RUNTIME_DIR}/source-mirrors}"
# Pod-local base for auto-cloned open-source deps, decoupled from USER_DATA_PATH
# so a shared (WekaFS) workspace root never collocates concurrent pods' checkouts.
# Default is a pod-internal, non-ephemeral dir (NOT /tmp): a tmp-reaper wiping
# /tmp mid-run left TRACELENS_ROOT dangling and broke trace_analyze (#722).
_open_source_root="${HYPERLOOM_OPEN_SOURCE_ROOT:-/opt/hyperloom/open-source-repos}"
# tree-reform.MD P2.5: kernel-agent/framework-agent live under the hyperloom
# package tree in both source and pip-installed layouts. A missing pyproject at
# REPO_ROOT means setup is running from a pip --target workspace rather than a
# source checkout, so the editable self-install step below is skipped.
_hyperloom_pkg_root="$(cd "${_script_dir}/../.." && pwd)"
HYPERLOOM_PACKAGED_INSTALL=0
if [ ! -f "${REPO_ROOT}/pyproject.toml" ] && [ -d "${_hyperloom_pkg_root}/agents/kernel" ]; then
  HYPERLOOM_PACKAGED_INSTALL=1
fi
KERNEL_AGENT_ROOT="${KERNEL_AGENT_ROOT:-${_hyperloom_pkg_root}/agents/kernel}"
FRAMEWORK_AGENT_ROOT="${FRAMEWORK_AGENT_ROOT:-${_hyperloom_pkg_root}/agents/framework}"
# tree-reform.MD P2.5: framework-agent was promoted from a sibling
# ``framework-agent/`` checkout into the in-tree ``hyperloom`` src-layout
# namespace (``src/hyperloom/agents/framework``); it no longer has its own
# installer/venv, so FRAMEWORK_AGENT_ROOT now just points at that in-tree
# package (still overridable) and the old chain_framework_agent() delegation
# below is a no-op.
MAGPIE_REPO="${MAGPIE_REPO:-https://github.com/AMD-AGI/Magpie.git}"
# Pin Magpie to a release commit/tag instead of the default branch. Operators can
# re-pin with MAGPIE_REF=<tag|sha>.
MAGPIE_REF="${MAGPIE_REF:-0171222c532db6fc5cb174667db66e34f1d9dd98}"
MAGPIE_PACKAGE_SPEC="${MAGPIE_PACKAGE_SPEC:-magpie-eval @ git+${MAGPIE_REPO}@${MAGPIE_REF}}"
# MAGPIE_PATH points install.sh AND the Python optimizer (cli.py /
# _grid_runner.py / manifest.py) at Magpie's import root. When unset by the
# operator, ensure_magpie resolves it from the pip-installed package; explicit
# overrides remain supported for local source checkouts / debugging.
MAGPIE_PATH_EXPLICIT=0
if [ -n "${MAGPIE_PATH:-}" ]; then
  MAGPIE_PATH_EXPLICIT=1
fi
MAGPIE_PATH="${MAGPIE_PATH:-${_open_source_root}/Magpie}"
INFERENCEX_REPO="${INFERENCEX_REPO:-https://github.com/SemiAnalysisAI/InferenceX.git}"
# Pin InferenceX to a current default-branch HEAD *commit SHA* so the
# per-install clone is reproducible (same rationale as MAGPIE_REF). Operators
# can re-pin with INFERENCEX_REF=<tag|branch|sha>.
INFERENCEX_REF="${INFERENCEX_REF:-2035a2117ad22403376359be0064dfa2c078c59b}"
INFERENCEX_DEFAULT_DIR="${INFERENCEX_DEFAULT_DIR:-${_open_source_root}/InferenceX}"

DRY_RUN=0
CHECK_ONLY=0
SKIP_KERNEL_AGENT=0

usage() {
  cat <<'EOF'
Usage: src/hyperloom/inference_optimizer/assets/install.sh [options]

Installs:
  - inference_optimizer Python package (with claude_agent_sdk via [test])
  - langfuse SDK, but ONLY when HYPERLOOM_LANGFUSE_ENABLE is on in the
    environment / .env (opt-in live trace push; skipped otherwise)
  - Magpie (pip-installed from MAGPIE_PACKAGE_SPEC)
  - Detects/exports INFERENCEX_PATH
  - Chains to src/hyperloom/agents/kernel/scripts/install.sh for Ray + ray-head start,
    TraceLens, GEAK, and LLM gateway env.
  - The `fa` CLI (used by the Coordinator-owned FRAMEWORK_AGENT phase at
    optimize-time, candidate discovery via `fa phase-discover`) is provided
    by this same editable install (tree-reform.MD P2.5 promoted
    framework-agent into src/hyperloom/agents/framework/, so it no longer
    has its own separate installer/venv to chain to).

Options:
  --check-only           Verify only, do not install
  --dry-run              Print actions without running them
  --skip-kernel-agent    Skip the chained kernel-agent installer
  -h, --help             Show this help

Env overrides:
  REPO_ROOT, KERNEL_AGENT_ROOT, FRAMEWORK_AGENT_ROOT, MAGPIE_REPO,
  MAGPIE_REF (commit SHA / tag / branch the Magpie package is pinned to;
    default is a commit that already copies benchmark scripts atomically),
  MAGPIE_PACKAGE_SPEC, MAGPIE_PATH, INFERENCEX_REPO,
  INFERENCEX_REF (commit SHA / tag / branch the InferenceX clone is pinned
    to; default is a current upstream HEAD SHA),
  INFERENCEX_DEFAULT_DIR, INFERENCEX_PATH,
  PYTHON, TRACELENS_ROOT,
  TRACELENS_INTERNAL_ROOT (set to enable the optional internal extension;
    unset => open-source-only),
  USER_DATA_PATH,
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

# Truthy/falsy test for boolean-ish env vars. Numeric `-eq` comparisons choke on
# string values (`[ false -eq 0 ]` errors and reads as true under set -e), so a
# user writing MAGPIE_PATCH_STRICT=false would get the OPPOSITE of intent. Accept
# the common spellings case-insensitively; returns success (0) when falsy.
is_falsy() {
  case "$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')" in
    0|false|no|off|"") return 0 ;;
    *) return 1 ;;
  esac
}

run() {
  log "$*"
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    "$@"
  fi
}

# Clone a dependency pinned to $ref into $dir, mirroring the GEAK pin in
# src/hyperloom/agents/kernel/scripts/install.sh. `git clone --branch` only accepts
# tags/branches, not raw SHAs, so a 7-40 hex char ref triggers a shallow
# fetch-checkout dance instead (GitHub serves shallow SHA fetches via
# uploadpack.allowReachableSHA1InWant=true). DRY_RUN / CHECK_ONLY are honoured
# through the shared `run` helper. On success the checkout has a valid HEAD at
# $ref, so manifest.py's _git_revision_at() still resolves the pinned commit.
# Returns non-zero (stopping at the first failed step) so callers can choose
# fail-loud (Magpie) or fail-soft (InferenceX).
git_fetch_pinned() {
  local repo="$1" dir="$2" ref="$3" label="$4"
  if [[ "$ref" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
    log "fetching ${label} pinned to commit ${ref} (shallow fetch-checkout)"
    run git init -q "$dir" || return 1
    run git -C "$dir" remote add origin "$repo" || return 1
    run git -C "$dir" fetch --depth 1 origin "$ref" || return 1
    run git -C "$dir" checkout -q FETCH_HEAD || return 1
  else
    log "cloning ${label} pinned to ref ${ref} (--branch)"
    run git clone --depth 1 --branch "$ref" "$repo" "$dir" || return 1
  fi
  return 0
}

# Serialize concurrent installs that share one open-source checkout root
# (Magpie / InferenceX, plus GEAK / TraceLens via the chained
# kernel-agent installer). With no lock, two installs race and corrupt each
# other's half-cloned checkouts (observed: GEAK src/minisweagent/... missing,
# repeated install failures). The lock lives in $_open_source_root (pod-local)
# so it tracks exactly what it guards; the chained kernel-agent installer uses
# the same $_open_source_root default, keeping parent/child on one lock path.
# We hold an flock on $_open_source_root/.install.lock via fd 9 from the first
# mirror-mutating step until this process exits (fd closes on exit), so it
# guards every clone/build below and releases automatically at the end.
# Skipped under --check-only / --dry-run (introspection only, no mutation).
# When we chain to kernel-agent's installer we export
# HYPERLOOM_INSTALL_LOCK_HELD=1 so that child does not deadlock re-acquiring
# the same lock on a second open file description.
acquire_install_lock() {
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  if [ "${HYPERLOOM_INSTALL_LOCK_HELD:-0}" = "1" ]; then
    log "install lock already held by parent installer; not re-locking"
    return 0
  fi
  mkdir -p "${_open_source_root}"
  exec 9>"${_open_source_root}/.install.lock"
  if command -v flock >/dev/null 2>&1; then
    log "waiting for install lock: ${_open_source_root}/.install.lock"
    flock 9
    log "acquired install lock"
    export HYPERLOOM_INSTALL_LOCK_HELD=1
  else
    warn "flock not available; concurrent installs may race on dependency checkouts"
  fi
}

# Preflight credential validation. Mirrors src/hyperloom/agents/kernel/scripts/install.sh:
# a usable setup needs Anthropic or DeepSeek credentials. The installer gate
# intentionally does not default or require OpenAI.
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
  local has_url=0 has_key=0
  { [ -n "${ANTHROPIC_BASE_URL:-}" ] || [ -n "${DEEPSEEK_BASE_URL:-}" ] || [ -n "${DEEPSEEK_API_KEY:-}" ]; } && has_url=1
  { [ -n "${ANTHROPIC_API_KEY:-}" ] || [ -n "${ANTHROPIC_AUTH_TOKEN:-}" ] || [ -n "${DEEPSEEK_API_KEY:-}" ]; } && has_key=1
  [ "$has_url" -eq 0 ] && missing+=("ANTHROPIC_BASE_URL or DEEPSEEK_BASE_URL (DeepSeek may omit the URL)")
  [ "$has_key" -eq 0 ] && missing+=("ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or DEEPSEEK_API_KEY")
  if [ "$has_url" -eq 1 ] && [ "$has_key" -eq 1 ]; then
    log "credentials preflight: usable LLM base URL + key present"
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
[inference-optimizer ERROR] Missing required credential group(s): ${missing[*]}

Tried loading from:
  - shell environment
  - \$REPO_ROOT/.env  (${env_file_status}: ${REPO_ROOT}/.env)

Fix one of:
  1. Anthropic:
       export ANTHROPIC_BASE_URL=https://api.anthropic.com
       export ANTHROPIC_API_KEY=sk-ant-...
  2. DeepSeek:
       export DEEPSEEK_API_KEY=sk-...
       # optional: export DEEPSEEK_BASE_URL=https://api.deepseek.com/anthropic
  3. Copy .env from a working worktree into this one:
       cp /path/to/main-worktree/.env "${REPO_ROOT}/.env"
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
# bare `python3 -m pip ...` calls (src/hyperloom/agents/kernel/scripts/install.sh) land in
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
log "open_source_root=${_open_source_root}"
log "KERNEL_AGENT_ROOT=${KERNEL_AGENT_ROOT}"
log "KERNEL_AGENT_ENV=${KERNEL_AGENT_ENV}"
log "MAGPIE_PATH=${MAGPIE_PATH}"
log "INFERENCEX_REPO=${INFERENCEX_REPO}"
log "INFERENCEX_DEFAULT_DIR=${INFERENCEX_DEFAULT_DIR}"
export USER_DATA_PATH HYPERLOOM_RUNTIME_DIR KERNEL_AGENT_ENV
export HYPERLOOM_KERNEL_AGENT_ROOT="${HYPERLOOM_KERNEL_AGENT_ROOT:-${KERNEL_AGENT_ROOT}}"
# Pre-create the writable runtime root so ensure_magpie / chain_kernel_agent
# never race on missing parents (Magpie's pip install -e writes egg-info
# under MAGPIE_PATH; kernel-agent install.sh writes kernel-agent.env.sh into
# HYPERLOOM_RUNTIME_DIR).
if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
  mkdir -p "${HYPERLOOM_RUNTIME_DIR}" "${_open_source_root}"
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
  if [ "$HYPERLOOM_PACKAGED_INSTALL" -eq 1 ]; then
    log "ensuring inference_optimizer runtime deps (packaged install)"
    # The bare wheel ships with empty base deps (pip --target stays clean), so
    # the runtime deps are installed here. The parent package imports with no
    # third-party dep, so this import check runs before the pip install below.
    "$PYTHON" - <<'PY' || die "hyperloom.inference_optimizer not importable from installed wheel"
import hyperloom.inference_optimizer  # noqa: F401
PY
    if [ "$CHECK_ONLY" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
      # PyYAML: hard import-time dep of the CLI startup path. llm extra
      # (claude-agent-sdk/openai/httpx): Coordinator backends.
      "$PYTHON" -m pip install --quiet "${PIP_EXTRA[@]}" \
        "PyYAML>=6.0" "claude-agent-sdk>=0.2.110" "openai>=1.50" "httpx>=0.27"
      # web extra only when critic web tools are enabled (off by default).
      if [ "${CRITIC_WEB_TOOLS_ENABLED:-}" = "true" ] || [ "${CRITIC_WEB_TOOLS_ENABLED:-}" = "1" ]; then
        "$PYTHON" -m pip install --quiet "${PIP_EXTRA[@]}" "markdownify>=0.11" "cachetools>=5.3"
      fi
    fi
    if "$PYTHON" -c "import claude_agent_sdk" >/dev/null 2>&1; then
      log "claude_agent_sdk OK"
    else
      warn "claude_agent_sdk not importable after runtime dep install (Coordinator will fail)"
      [ "$CHECK_ONLY" -eq 1 ] || die "claude_agent_sdk missing"
    fi
    return 0
  fi
  log "ensuring inference_optimizer package + claude_agent_sdk extras"
  if [ "$CHECK_ONLY" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    "$PYTHON" -m pip install --quiet "${PIP_EXTRA[@]}" -e "${REPO_ROOT}[test]"
  fi
  "$PYTHON" - <<'PY' || die "hyperloom.inference_optimizer not importable after install"
import hyperloom.inference_optimizer  # noqa: F401
PY
  if "$PYTHON" -c "import claude_agent_sdk" >/dev/null 2>&1; then
    log "claude_agent_sdk OK"
  else
    warn "claude_agent_sdk not importable after install (Coordinator will fail)"
    [ "$CHECK_ONLY" -eq 1 ] || die "claude_agent_sdk missing"
  fi
}

# --- 1b. forge-gemm-tune (KernelForge deterministic GEMM tuning CLI) ---
_forge_gemm_tune_candidates() {
  # Explicit override first.
  [ -n "${FORGE_GEMM_TUNE_ROOT:-}" ] && printf '%s\n' "$FORGE_GEMM_TUNE_ROOT"
  # KernelForge root aliases used by Hyperloom's forge backend.
  [ -n "${FORGE_PATH:-}" ] && printf '%s\n' "${FORGE_PATH%/}/src/forge_gemm_tune" "${FORGE_PATH%/}/forge_gemm_tune"
  [ -n "${KERNEL_FORGE_ROOT:-}" ] && printf '%s\n' "${KERNEL_FORGE_ROOT%/}/src/forge_gemm_tune" "${KERNEL_FORGE_ROOT%/}/forge_gemm_tune"
  [ -n "${KERNEL_FORGE_PATH:-}" ] && printf '%s\n' "${KERNEL_FORGE_PATH%/}/src/forge_gemm_tune" "${KERNEL_FORGE_PATH%/}/forge_gemm_tune"
}

_resolve_forge_gemm_tune_root() {
  local cand
  while IFS= read -r cand; do
    [ -n "$cand" ] || continue
    if [ -f "${cand%/}/pyproject.toml" ] && { [ -f "${cand%/}/forge_gemm_tune/cli.py" ] || [ -f "${cand%/}/cli.py" ]; }; then
      realpath "$cand" 2>/dev/null || printf '%s\n' "$cand"
      return 0
    fi
  done < <(_forge_gemm_tune_candidates)
  return 1
}

ensure_forge_gemm_tune() {
  local root resolved
  if root="$(_resolve_forge_gemm_tune_root)"; then
    log "ensuring forge-gemm-tune from ${root}"
    if [ "$CHECK_ONLY" -eq 1 ]; then
      "$PYTHON" -c "import forge_gemm_tune" >/dev/null 2>&1 \
        && log "forge-gemm-tune import OK" \
        || warn "forge-gemm-tune not importable (check-only; would install from ${root})"
      return 0
    fi
    if [ "$DRY_RUN" -eq 1 ]; then
      log "would run: ${PYTHON} -m pip install -e ${root}"
      return 0
    fi
    resolved="$("$PYTHON" -c 'import forge_gemm_tune, os; print(os.path.realpath(os.path.dirname(forge_gemm_tune.__file__)))' 2>/dev/null || true)"
    case "$resolved" in
      "$root" | "$root"/*)
        log "forge-gemm-tune already installed from ${root}; skipping editable reinstall"
        ;;
      *)
        "$PYTHON" -m pip install --quiet "${PIP_EXTRA[@]}" -e "$root"
        "$PYTHON" -c "import forge_gemm_tune; import forge_gemm_tune.cli" >/dev/null
        log "forge-gemm-tune installed OK from ${root}"
        ;;
    esac
  else
    if "$PYTHON" -c "import forge_gemm_tune" >/dev/null 2>&1; then
      log "forge-gemm-tune import OK"
    else
      log "forge-gemm-tune source not configured; skipping optional forge GEMM tuning install"
    fi
  fi
}

# --- 2. Magpie ---
# The install state is the pip-installed package specified by
# $MAGPIE_PACKAGE_SPEC. $MAGPIE_PATH remains exported for runtime code that
# needs to inspect Magpie's package files (patcher / manifest / InferenceX
# discovery); when not explicitly set, it resolves to the installed package
# root after import.
ensure_magpie() {
  log "ensuring Magpie package ${MAGPIE_PACKAGE_SPEC}"
  if [ "$CHECK_ONLY" -eq 1 ]; then
    if "$PYTHON" -c "import Magpie" >/dev/null 2>&1; then
      log "Magpie importable"
    else
      warn "Magpie not importable (check-only mode, skipping pip install)"
    fi
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would pip install Magpie package: ${MAGPIE_PACKAGE_SPEC}"
    return 0
  fi
  if [ "$DRY_RUN" -eq 0 ]; then
    if "$PYTHON" -c "import Magpie" >/dev/null 2>&1; then
      log "Magpie already importable; skipping pip install"
    else
      "$PYTHON" -m pip install --quiet "${PIP_EXTRA[@]}" "$MAGPIE_PACKAGE_SPEC"
      "$PYTHON" -c "import Magpie" >/dev/null
      log "Magpie installed OK from ${MAGPIE_PACKAGE_SPEC}"
    fi
    local installed_root
    installed_root="$("$PYTHON" - <<'PY'
from pathlib import Path
import Magpie
print(Path(Magpie.__file__).resolve().parent.parent)
PY
)"
    if [ "$MAGPIE_PATH_EXPLICIT" -eq 0 ]; then
      MAGPIE_PATH="$installed_root"
      export MAGPIE_PATH
      log "MAGPIE_PATH resolved from installed package: ${MAGPIE_PATH}"
    else
      log "MAGPIE_PATH override preserved: ${MAGPIE_PATH}"
    fi
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
# Fail-soft (was fail-loud): a `False` return means the legacy
# `shutil.copy2` block was not found. With MAGPIE_REF now pinned to an
# upstream commit that already copies scripts atomically
# (`_copy_benchmark_script_atomic`), that is the EXPECTED no-op state —
# bugs.md §C #1 is already mitigated upstream, so we `warn` and continue
# instead of aborting every install. (A sibling branch makes the patcher
# itself upstream-aware; this warn is the defense-in-depth complement.) If
# you re-pin MAGPIE_REF to a pre-refactor commit and the patch still cannot
# apply, the script-tearing race is genuinely unpatched — review the
# warning. Override the gate via PATCH_MAGPIE=0 to skip the step entirely.
ensure_magpie_atomic_scripts_patch() {
  if is_falsy "${PATCH_MAGPIE:-1}"; then
    log "PATCH_MAGPIE is falsy — skipping Magpie atomic-write patch (caller asserts upstream already fixed)"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would apply Hyperloom #C1 atomic-write patch to ${MAGPIE_PATH}/Magpie/modes/benchmark/benchmarker.py"
    return 0
  fi
  log "applying Hyperloom #C1 atomic-write patch to Magpie._prepare_benchmark_scripts"
  # Exit-code contract (read below): 0 ok · 2 remote-trust drift only ·
  # 4 GENUINE atomic failure (race unmitigated) · 1 benign atomic no-op.
  if MAGPIE_PATH="$MAGPIE_PATH" "$PYTHON" - <<'PY'
import os, sys
from hyperloom.orchestrator.actions.executors._magpie_patcher import (
    magpie_scripts_patch_status,
)
status = magpie_scripts_patch_status(os.environ["MAGPIE_PATH"])
print(f"_magpie_patcher: atomic_reason={status.atomic_reason} "
      f"atomic_ok={status.atomic_ok} remote_trust_ok={status.remote_trust_ok} "
      f"eval_flag_ok={status.eval_flag_ok}",
      file=sys.stderr)
if status.ok:
    sys.exit(0)
# A GENUINE atomic failure (unrecognized shape / I/O error) means the
# script-tearing race is actually unmitigated — distinct exit so a strict
# install can fail-loud instead of swallowing it as an expected no-op.
if status.atomic_genuine_failure:
    sys.exit(4)
if not status.atomic_ok:
    sys.exit(1)
if not status.remote_trust_ok:
    sys.exit(2)
# Redundant --concurrent-requests eval flag could not be stripped from a
# generic benchmark script (unrecognised shape). Non-fatal: RUN_EVAL=true
# baselines may abort on InferenceX's 'Unknown parameter', but the baseline
# executor's eval-failure fallback re-runs with RUN_EVAL=false, so the run
# still proceeds (without an accuracy gate). Distinct exit so install warns.
if not status.eval_flag_ok:
    sys.exit(5)
# Defensive catch-all: a not-ok status with none of the bits above set should
# never happen, but exit non-zero so we never fall through to exit 0.
sys.exit(3)
PY
  then
    log "Magpie #C1 patch OK"
  else
    rc=$?
    if [ "$rc" -eq 4 ]; then
      # GENUINE failure: the legacy block is gone AND upstream is not atomic
      # (or a read/write error). bugs.md §C #1 (script-tearing race) is NOT
      # mitigated — `profile`/`baseline` can hit `syntax error near unexpected
      # token 'fi'`. Strict mode (default) aborts; a falsy MAGPIE_PATCH_STRICT
      # (0/false/no/off) keeps the legacy fail-soft behaviour and only warns.
      if is_falsy "${MAGPIE_PATCH_STRICT:-1}"; then
        warn "Magpie atomic-write patch GENUINELY failed (race unmitigated); MAGPIE_PATCH_STRICT=${MAGPIE_PATCH_STRICT:-} (falsy), continuing anyway — review _magpie_patcher.py."
      else
        die "Magpie atomic-write patch GENUINELY failed: neither the legacy shutil.copy2 block nor an upstream atomic copy was found in benchmarker.py. bugs.md §C #1 (script-tearing race) is unmitigated. Re-pin MAGPIE_REF to a supported commit, review _magpie_patcher.py, or set MAGPIE_PATCH_STRICT=0 to downgrade to a warning (or PATCH_MAGPIE=0 to skip entirely)."
      fi
    elif [ "$rc" -eq 2 ]; then
      warn "Magpie SGLang remote trust patch did not apply. If MAGPIE_TRUST_REMOTE_CODE=1 is required for custom-code models (for example Kimi/Qwen tokenizer paths), remote benchmark clients may still fail to pass trust; review _magpie_patcher.py or set PATCH_MAGPIE=0 only if this is intentional."
    elif [ "$rc" -eq 5 ]; then
      warn "Magpie redundant --concurrent-requests eval flag could not be stripped from a generic benchmark script (unrecognised run_eval line). RUN_EVAL=true baselines may abort on InferenceX's 'Unknown parameter: --concurrent-requests'; the baseline executor falls back to RUN_EVAL=false (no accuracy gate) so the run still proceeds. Review _magpie_patcher.py if accuracy eval is required."
    else
      # Benign no-op (rc=1): MAGPIE_PATH unset / benchmarker.py missing. With
      # MAGPIE_REF pinned to an upstream-atomic commit the patcher reports
      # ``upstream_atomic`` (exit 0) instead, so this branch is just the
      # missing-tree case — warn and continue. PATCH_MAGPIE=0 skips the step.
      warn "Magpie atomic-write patch skipped (no benchmarker.py under MAGPIE_PATH). Fine for tests/dry-runs; otherwise check MAGPIE_PATH or set PATCH_MAGPIE=0."
    fi
  fi
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
#   * Otherwise -> fetch INFERENCEX_REF from INFERENCEX_REPO into
#     INFERENCEX_DEFAULT_DIR via the shared git_fetch_pinned() dance
#     (SHA-aware shallow fetch-checkout, mirrors the GEAK pin). If a clone
#     already exists there from a previous install we leave it as-is
#     (idempotent re-runs) — the per-install isolation guarantee is already
#     met, and re-cloning would just churn benchmark scripts that the Magpie
#     patch already keeps consistent on disk.
#   * Pinned to INFERENCEX_REF (a commit SHA by default) so a fresh install
#     is reproducible. We still record the resolved commit into the session
#     manifest (see manifest.py / _describe_dep) so runs stay traceable even
#     when an operator overrides INFERENCEX_REF.
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
    log "would fetch InferenceX pinned to INFERENCEX_REF=${INFERENCEX_REF} from ${INFERENCEX_REPO} -> ${INFERENCEX_PATH}"
    export INFERENCEX_PATH
    return 0
  fi
  log "cloning fresh InferenceX pinned to ${INFERENCEX_REF} from ${INFERENCEX_REPO} -> ${INFERENCEX_PATH}"
  mkdir -p "$(dirname "$INFERENCEX_PATH")"
  if ! git_fetch_pinned "$INFERENCEX_REPO" "$INFERENCEX_PATH" "$INFERENCEX_REF" "InferenceX"; then
    warn "InferenceX clone failed. GSM8K eval will fail without it. Set"
    warn "INFERENCEX_PATH to a pre-cloned tree to skip this step."
    return 0
  fi
  export INFERENCEX_PATH
  log "InferenceX cloned at ${INFERENCEX_PATH} (pinned ${INFERENCEX_REF})"
}

# --- 4. InferenceX bench_serving runtime deps ---
#
# `benchmark_serving.py` lives under InferenceX (not under Magpie's
# pyproject.toml), so installing Magpie does NOT pull its client-side
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

# --- 4b. xDiT image-quality gate deps (SSIM + LPIPS) ---
#
# The scriptable xDiT bench wrapper computes an image-quality gate
# (LPIPS / SSIM / MSE vs a BF16 reference). torch/torchvision/numpy ship
# with the pytorch-xdit image, but scikit-image (SSIM) and lpips (LPIPS)
# do NOT — so without this step the gate silently degrades to MSE-only
# (the wrapper now reports ssim_available/lpips_available=false). These
# are pip-name != import-name, so we map them explicitly.
#
# Fail-soft: lpips also pulls AlexNet weights on first use (network), so a
# failed install must NOT abort the whole install — the wrapper degrades
# gracefully (honest *_available=false) rather than crashing the run.
_XDIT_QUALITY_DEPS=(
  "scikit-image:skimage"
  "lpips:lpips"
)

ensure_xdit_quality_deps() {
  log "ensuring xDiT image-quality gate deps (SSIM/LPIPS) in $PYTHON"
  local missing=()
  local pair pip_name import_name
  for pair in "${_XDIT_QUALITY_DEPS[@]}"; do
    pip_name="${pair%%:*}"
    import_name="${pair##*:}"
    if ! "$PYTHON" -c "import ${import_name}" >/dev/null 2>&1; then
      missing+=("$pip_name")
    fi
  done
  if [ ${#missing[@]} -eq 0 ]; then
    log "xDiT quality deps already satisfied"
    return 0
  fi
  if [ "$CHECK_ONLY" -eq 1 ]; then
    warn "check-only mode; would install xDiT quality deps: ${missing[*]}"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "dry-run; skipping xDiT quality dep install"
    return 0
  fi
  log "installing missing xDiT quality deps: ${missing[*]}"
  "$PYTHON" -m pip install --quiet --no-cache-dir \
    "${PIP_EXTRA[@]}" "${missing[@]}" \
    || warn "failed to install xDiT quality deps: ${missing[*]} (gate degrades to MSE-only)"
  for pair in "${_XDIT_QUALITY_DEPS[@]}"; do
    import_name="${pair##*:}"
    "$PYTHON" -c "import ${import_name}" >/dev/null 2>&1 \
      || warn "xDiT quality dep '${import_name}' not importable after install (gate excludes it)"
  done
}

# --- 4c. Langfuse SDK (opt-in live trace push) ---
# The local reports/trace/*.jsonl ledger never needs this. Only the opt-in
# live-Langfuse sink (HYPERLOOM_LANGFUSE_ENABLE=1) imports the SDK, and when
# absent the emitter degrades to a silent no-op — so a run can look "fine"
# while pushing nothing. Operators kept hitting that gap: they flipped the
# flag + set the keys but forgot the separate `pip install '...[trace]'`,
# and only noticed when session_breakdown showed sdk_available=false.
#
# Fix: when (and ONLY when) the Langfuse master switch is on in the loaded
# environment (.env is sourced above by load_dotenv_no_clobber, so the flag
# is visible here), guarantee the SDK is importable — install it on demand,
# mirroring ensure_bench_serving_deps (import-probe first, pip only on miss).
# Switch off => skipped entirely, so environments that don't use Langfuse
# stay lean. Fail-soft: a failed install warns (the emitter's own no-op
# fallback still protects the run) rather than aborting the whole install.
_langfuse_enabled() {
  case "$(printf '%s' "${HYPERLOOM_LANGFUSE_ENABLE:-}" | tr '[:upper:]' '[:lower:]')" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

ensure_langfuse_when_enabled() {
  if ! _langfuse_enabled; then
    log "langfuse: HYPERLOOM_LANGFUSE_ENABLE not set; skipping SDK install (local jsonl ledger is unaffected)"
    return 0
  fi
  if "$PYTHON" -c "import langfuse" >/dev/null 2>&1; then
    log "langfuse: SDK already importable"
    return 0
  fi
  log "langfuse: HYPERLOOM_LANGFUSE_ENABLE is on but SDK missing — installing langfuse"
  if [ "$CHECK_ONLY" -eq 1 ]; then
    warn "langfuse: SDK missing (check-only; would install 'langfuse>=2.0')"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would install 'langfuse>=2.0'"
    return 0
  fi
  if "$PYTHON" -m pip install --quiet --no-cache-dir "${PIP_EXTRA[@]}" "langfuse>=2.0" \
      && "$PYTHON" -c "import langfuse" >/dev/null 2>&1; then
    log "langfuse: SDK installed OK"
  else
    warn "langfuse: SDK install failed; live push will degrade to a no-op (local jsonl ledger still written). Preinstall 'langfuse' in the image or run: \"\$PYTHON\" -m pip install 'langfuse>=2.0'"
  fi
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
  log "delegating ray + TraceLens + GEAK + LLM gateway env to ${script}"
  export REPO_ROOT KERNEL_AGENT_ROOT MAGPIE_PATH HYPERLOOM_ROOT
  export USER_DATA_PATH HYPERLOOM_RUNTIME_DIR KERNEL_AGENT_ENV
  export HYPERLOOM_KERNEL_AGENT_ROOT="${HYPERLOOM_KERNEL_AGENT_ROOT:-${KERNEL_AGENT_ROOT}}"
  [ -n "${INFERENCEX_PATH:-}" ] && export INFERENCEX_PATH
  # Forward the optional internal extension path when provided; unset =>
  # kernel-agent installer stays open-source-only (no separate toggle).
  [ -n "${TRACELENS_INTERNAL_ROOT:-}" ] && export TRACELENS_INTERNAL_ROOT
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
ensure_forge_gemm_tune
ensure_langfuse_when_enabled
# Hold the install lock for the whole mirror-mutating region (Magpie /
# InferenceX clones + the chained kernel-agent GEAK/TraceLens clones).
acquire_install_lock
# Magpie is only needed when the Magpie benchmark backend is active. The
# bypass backend drives InferenceX directly (see benchmark_backend.py), so
# skip the Magpie install/import and its script-patch when bypass is selected.
# Default (unset/blank) stays magpie, preserving existing behavior.
# Mirror Python's resolve_backend_name() normalization (strip THEN lower):
# sed trims ONLY leading/trailing whitespace (like str.strip()), so " bypass" /
# "bypass " skip Magpie here to match runtime, while an internal-space value
# such as "by pass" stays != "bypass" and correctly falls through to Magpie
# (runtime resolves such unknown values back to magpie). A blanket delete of
# ALL whitespace would wrongly collapse "by pass" -> "bypass" and diverge.
HYPERLOOM_BENCHMARK_BACKEND_LC="$(printf '%s' "${HYPERLOOM_BENCHMARK_BACKEND:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | tr '[:upper:]' '[:lower:]')"
if [ "$HYPERLOOM_BENCHMARK_BACKEND_LC" = "bypass" ]; then
  log "benchmark backend is bypass; skipping ensure_magpie + ensure_magpie_atomic_scripts_patch"
else
  ensure_magpie
  ensure_magpie_atomic_scripts_patch
fi
ensure_inferencex
ensure_bench_serving_deps
ensure_xdit_quality_deps
chain_kernel_agent
# tree-reform.MD P2.5: framework-agent was promoted into
# src/hyperloom/agents/framework/ (single hyperloom distribution), so the
# `fa` CLI is already installed by ensure_inference_optimizer() above; no
# more separate chain_framework_agent() delegation to a standalone installer.

_probe_framework_source_roots() {
  log "probing framework source roots for INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS"
  local roots
  roots="$("$PYTHON" - <<'PY'
from hyperloom.orchestrator.framework.paths import probe_framework_source_roots_for_env
print(probe_framework_source_roots_for_env())
PY
)"
  if [ -z "$roots" ]; then
    warn "no framework source roots discovered"
    return 0
  fi
  log "discovered framework roots: $roots"
  # Emit a framework-bucketed one-liner so operators (and the
  # preflight grep) can tell at a glance whether atom was picked up.
  local roots_summary
  roots_summary="$(ROOTS_INPUT="$roots" "$PYTHON" - <<'PY'
import os
from hyperloom.orchestrator.framework.paths import summarise_framework_root_discovery
print(summarise_framework_root_discovery(os.environ.get("ROOTS_INPUT", "")))
PY
)"
  if [ -n "$roots_summary" ]; then
    log "discovered framework roots: $roots_summary"
  fi
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
log "kernel-agent env file written: ${KERNEL_AGENT_ENV}"
log "  HYPERLOOM_KERNEL_AGENT_ROOT=${HYPERLOOM_KERNEL_AGENT_ROOT}"
log ""
log "next steps — pick ONE:"
log "  (a) source ${KERNEL_AGENT_ENV}, then run hyperloom.inference_optimizer.cli"
log "  (b) just launch hyperloom.inference_optimizer.cli — preflight will auto-source"
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
