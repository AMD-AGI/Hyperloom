#!/usr/bin/env bash
# Kernel Agent installer.
#
# Base install is intentionally small and deterministic:
#   - ray[default]==2.44.1 + click<8.3.0
#   - Node.js/npm for claude/codex CLIs and @cursor/sdk
#   - TraceLens editable install + CLI verification
#
# The installer prepares all kernel-agent backends in one pass.

set -euo pipefail

# Ray/K8s subprocesses may inherit a minimal PATH; git/apt/node live under
# /usr/bin even when callers only prepend /opt/venv/bin. Prepend the
# standard system bins so multi-node RayJob children resolve them.
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"
export PATH="/opt/venv/bin:$PATH"

# Default every writable artefact location under $USER_DATA_PATH so a single
# session-dir move relocates Magpie / source mirrors / GEAK config / the
# kernel-agent env file. Operators can still pin individual paths via env
# overrides (HYPERLOOM_ROOT, MAGPIE_DIR, etc.) — the defaults below take
# effect only when the corresponding env var is unset.
#
# REPO_ROOT / KERNEL_AGENT_ROOT default to the on-disk source location
# (this script lives at kernel-agent/scripts/install.sh, so its parent's
# parent is the repo root). Read-only inputs (TRACELENS_ROOT, OOB_SRC,
# HYPERLOOM_BUNDLE, GEAK_MEMORY_STORE_PATH, RAG_INDEX_DIR) stay outside
# USER_DATA_PATH for warm-start latency reasons (decision: keep GEAK
# cross-session memory + RAG embedding cache shared across sessions).
#
# Removed envs: WORKSPACE_PATH / WORKSPACE_ROOT (collapsed into the
# USER_DATA_PATH-rooted defaults). If your launcher exported these,
# either rename to USER_DATA_PATH or simply drop them.
KERNEL_AGENT_ROOT="${KERNEL_AGENT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
HYPERLOOM_KERNEL_AGENT_ROOT="${HYPERLOOM_KERNEL_AGENT_ROOT:-${KERNEL_AGENT_ROOT}}"
USER_DATA_PATH="${USER_DATA_PATH:-/workspace/hyperloom}"
HYPERLOOM_RUNTIME_DIR="${HYPERLOOM_RUNTIME_DIR:-${USER_DATA_PATH}/runtime}"
KERNEL_AGENT_ENV="${KERNEL_AGENT_ENV:-${HYPERLOOM_RUNTIME_DIR}/kernel-agent.env.sh}"
HYPERLOOM_ROOT="${HYPERLOOM_ROOT:-${HYPERLOOM_RUNTIME_DIR}/source-mirrors}"
HYPERLOOM_BUNDLE="${HYPERLOOM_BUNDLE:-/wekafs/fully-local}"
MAGPIE_DIR="${MAGPIE_DIR:-${HYPERLOOM_RUNTIME_DIR}/Magpie}"
# Resolve MAGPIE_PYTHON dynamically. The previous default
# ${MAGPIE_DIR}/venv/bin/python assumed a Magpie-private venv, but
# inference_optimizer/scripts/install.sh's ensure_magpie() does
# `pip install -e $MAGPIE_DIR` into the driver Python's site-packages
# (or the container image pre-installs it that way) — no venv is ever
# created at $MAGPIE_DIR/venv. Mirrors _resolve_magpie_python() in
# inference_optimizer/orchestrator/action_executors/_grid_runner.py:
#   $MAGPIE_PYTHON env > python3 on PATH that can `import Magpie`
#     > /opt/venv/bin/python (if it exists) > python3 on PATH.
_resolve_magpie_python() {
  if [ -n "${MAGPIE_PYTHON:-}" ]; then
    printf '%s' "$MAGPIE_PYTHON"
    return 0
  fi
  local candidate
  candidate="$(command -v python3 2>/dev/null || true)"
  if [ -n "$candidate" ] && "$candidate" -c "import Magpie" >/dev/null 2>&1; then
    printf '%s' "$candidate"
    return 0
  fi
  if [ -x /opt/venv/bin/python ]; then
    printf '%s' /opt/venv/bin/python
    return 0
  fi
  printf '%s' "${candidate:-/opt/venv/bin/python}"
}
MAGPIE_PYTHON="$(_resolve_magpie_python)"
PYTHONPATH="${MAGPIE_DIR}:${PYTHONPATH:-}"
INFERENCEX_PATH="${INFERENCEX_PATH:-}"
# TraceLens checkout root. Points at a checkout of the open-source
# AMD-AGI/TraceLens repo (carries the TraceLens.* Python package, the
# TraceLens_generate_perf_report_* CLIs, the agent skill bundle, and the
# per-version sglang_roofline_patches/sglang_<minor>_<patch>/ tree under
# examples/custom_workflows/inference_analysis/ that _server_patcher
# reads at runtime). ``inference_optimizer/scripts/local_setup.sh``
# clones this for end users in Local Mode; CI / bundled deployments
# preset the path via env. Legacy pre-2026-05-18 snapshots of the
# private TraceLens-internal repo at /wekafs/hyperloom/TraceLens-internal
# still satisfy the layout, so the default falls back to that location
# for shops that have not migrated yet.
TRACELENS_ROOT="${TRACELENS_ROOT:-/wekafs/hyperloom/TraceLens-internal}"
# Writable mirror for TraceLens when $TRACELENS_ROOT is on a read-only mount
# (e.g. /wekafs/...). Mirrors the OOB pattern: cp -r the read-only source into
# ${HYPERLOOM_ROOT}/TraceLens once, then pip install -e the mirror so
# editable build artifacts land on a writable filesystem. See ensure_tracelens.
TRACELENS_MIRROR_DIR="${TRACELENS_MIRROR_DIR:-${HYPERLOOM_ROOT}/TraceLens}"
# Optional TraceLens-internal extension. When set, ensure_tracelens()
# pip installs the private AMD-AGI/TraceLens-internal package on top of
# the open-source TraceLens. The internal extension post-processes the
# generated agent report to add roofline numbers, gains estimates, and
# MI355/MI455 MAF data. Unset (default) keeps Hyperloom on the
# open-source-only path (impact score, no roofline, MI300/MI325 MAF only).
# Provisioned by ``inference_optimizer/scripts/local_setup.sh`` when the
# operator opts in via TRACELENS_INSTALL_INTERNAL=1.
TRACELENS_INTERNAL_ROOT="${TRACELENS_INTERNAL_ROOT:-}"

# Credentials fallback: env always wins. If SAFE_API_KEY or OPENAI_BASE_URL
# is missing from env, source $REPO_ROOT/.env (resolved above from this
# script's parent dir) but protect any keys already set in env from being
# overwritten by .env.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
if [ -z "${SAFE_API_KEY:-}" ] || [ -z "${OPENAI_BASE_URL:-}" ] || [ -z "${CURSOR_API_KEY:-}" ]; then
  if [ -f "$REPO_ROOT/.env" ]; then
    _snap_safe="${SAFE_API_KEY-}"
    _snap_url="${OPENAI_BASE_URL-}"
    _snap_cursor="${CURSOR_API_KEY-}"
    set -a
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.env"
    set +a
    [ -n "$_snap_safe" ] && export SAFE_API_KEY="$_snap_safe"
    [ -n "$_snap_url" ]  && export OPENAI_BASE_URL="$_snap_url"
    [ -n "$_snap_cursor" ] && export CURSOR_API_KEY="$_snap_cursor"
    unset _snap_safe _snap_url _snap_cursor
    echo "[kernel-agent] loaded credentials fallback from $REPO_ROOT/.env (env wins)"
  fi
fi
GEAK_REPO="${GEAK_REPO:-https://github.com/AMD-AGI/GEAK.git}"
# Pin GEAK to the save-and-test-diff-fallthrough fix tip
# (https://github.com/AMD-AGI/GEAK/pull/244, not yet released as a tag).
# We pin to the *commit SHA* of the branch tip, NOT the branch name, so a
# future force-push / rebase upstream cannot silently change what every
# fresh install gets.
# TODO(post-GEAK-PR-244): once PR #244 lands and ships in a new GEAK tag,
# revert this pin to the tag (e.g. v3.2.1) for stronger discoverability.
# Operators can override with GEAK_REF=<tag|branch|sha>.
GEAK_REF="${GEAK_REF:-ec61bdbdb151904ec187a8d89518afb969c53737}"
OOB_SRC="${OOB_SRC:-${HYPERLOOM_BUNDLE}/OOB}"
GEAK_CONFIG="${GEAK_CONFIG:-${HYPERLOOM_RUNTIME_DIR}/geak-config/local.yaml}"
# GEAK talks to the AMD Primus-Safe LiteLLM-compatible /chat/completions
# endpoint.  Force the LiteLLM provider prefix to `openai/` for bare Claude
# model names so LiteLLM uses the OpenAI-compatible transformer instead of the
# Anthropic /v1/messages transformer.  Without this, GEAK gets
# Primus.00009 / NotFound on the same key+URL that works through
# /chat/completions.
GEAK_MODEL_NAME_RAW="${GEAK_MODEL_NAME:-claude-opus-4-7}"
case "${GEAK_MODEL_NAME_RAW}" in
  openai/*|anthropic/*|gpt-*|o1-*|o3-*|o4-*)
    GEAK_MODEL_NAME_VAL="${GEAK_MODEL_NAME_RAW}"
    ;;
  claude-*)
    GEAK_MODEL_NAME_VAL="openai/${GEAK_MODEL_NAME_RAW}"
    ;;
  *)
    GEAK_MODEL_NAME_VAL="${GEAK_MODEL_NAME_RAW}"
    ;;
esac
# Run mode for the GEAK CLI. Drives ``run.mode`` in the generated
# ``$GEAK_CONFIG`` yaml: ``full`` (default) selects the 2 h / 5-round preset
# at ``run.budgets.full`` and ``run.presets.full``; ``quick`` selects the
# 1 h / 2-round preset for smoke tests. GEAK's ``mini.py:435`` mode
# precedence still honours later overrides (CLI ``--mode`` or
# LLM-parsed task hints), but this is the yaml-default operators can set
# at install time without hand-editing $GEAK_CONFIG.
GEAK_RUN_MODE_VAL="${GEAK_RUN_MODE:-full}"
# Validate inline (the ``die`` helper is defined further down; calling it
# from this top-level scope would error with "die: command not found").
case "$GEAK_RUN_MODE_VAL" in
  quick|full) ;;
  *)
    echo "[kernel-agent ERROR] GEAK_RUN_MODE must be 'quick' or 'full'; got '$GEAK_RUN_MODE_VAL'" >&2
    exit 1
    ;;
esac
RAG_INDEX_DIR="${HOME}/.cache/amd-ai-devtool/semantic-index"
# RAG index build device. Resolution:
#   1. If $GEAK_RAG_INDEX_DEVICE is set explicitly, honor it verbatim
#      (operator override; lets CPU-only environments opt out).
#   2. Otherwise auto-detect: prefer GPU when either rocm-smi reports
#      a device or torch.cuda.is_available() returns True; fall back
#      to cpu only when no accelerator is visible.
# Rationale: CPU embedding can take 1.5h+ on the BGE-large model and
# repeatedly triggered zombie installers when the launcher timed out
# mid-build (observed: 58min CPU run vs ~1min cuda run). The kernel-agent
# runtime is always installed on GPU pods (IR-1 in the inference_optimizer
# SKILL gates this), so cuda is the right default.
if [ -z "${GEAK_RAG_INDEX_DEVICE:-}" ]; then
  if command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showid >/dev/null 2>&1; then
    GEAK_RAG_INDEX_DEVICE_VAL="cuda"
  elif python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    GEAK_RAG_INDEX_DEVICE_VAL="cuda"
  else
    GEAK_RAG_INDEX_DEVICE_VAL="cpu"
  fi
else
  GEAK_RAG_INDEX_DEVICE_VAL="${GEAK_RAG_INDEX_DEVICE}"
fi
# When 1, ensure_rag_index runs GEAK scripts/build_index.py after GEAK
# install. Defaults to 1 so GEAK kernel-opt gets RAG-augmented retrieval
# out of the box on the canonical GPU-pod path (cuda auto-detect, BGE-
# large embedding ~1min). Callers that don't want this — e.g. claude-only
# kernel-opt, CPU-only sandbox where BGE-large takes ~1.5h, or any path
# where install.sh latency matters more than RAG quality — should set
# `KERNEL_AGENT_BUILD_GEAK_RAG_INDEX=0` in the launching env (Brain
# propagates Environment block vars from the prompt into sandbox env, so
# operators can flip this per-task without editing this script).
KERNEL_AGENT_BUILD_GEAK_RAG_INDEX_VAL="${KERNEL_AGENT_BUILD_GEAK_RAG_INDEX:-1}"
GEAK_MEMORY_STORE_PATH_VAL="${GEAK_MEMORY_STORE_PATH:-/wekafs/hyperloom/geak-memory/memory.db}"
GEAK_SAVE_TO_KNOWLEDGE_BASE_VAL="${GEAK_SAVE_TO_KNOWLEDGE_BASE:-1}"
GEAK_MEMORY_MIN_SPEEDUP_VAL="${GEAK_MEMORY_MIN_SPEEDUP:-1.20}"
CODEX_MODEL_VAL="${CODEX_MODEL:-gpt-5.4}"
# GEAK/OOB use the user's LiteLLM-compatible endpoint. The canonical env is
# OPENAI_BASE_URL + SAFE_API_KEY; keep fallbacks for older launchers.
GEAK_API_KEY_VAL="${GEAK_API_KEY:-${SAFE_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-${AMD_API_KEY:-${AMD_LLM_API_KEY:-${LLM_API_KEY:-${OPENAI_API_KEY:-}}}}}}}"
GEAK_BASE_URL_VAL="${GEAK_BASE_URL:-${OPENAI_BASE_URL:-${ANTHROPIC_BASE_URL:-${LLM_API_BASE:-}}}}"
# LiteLLM provider-specific base_url normalisation:
#   * openai/* models require the OpenAI-compatible base URL, which in our
#     gateway includes the trailing /v1.  Preserve it.
#   * anthropic/* models use /v1/messages internally; strip a trailing /v1
#     only when the operator explicitly selects an anthropic provider.
case "${GEAK_MODEL_NAME_VAL}" in
  anthropic/claude-*)
    GEAK_BASE_URL_VAL="${GEAK_BASE_URL_VAL%/v1}"
    GEAK_BASE_URL_VAL="${GEAK_BASE_URL_VAL%/}"
    ;;
esac
OOB_API_KEY_VAL="${OOB_API_KEY:-${SAFE_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-${ANTHROPIC_API_KEY:-${OPENAI_API_KEY:-}}}}}"
OOB_BASE_URL_VAL="${OOB_BASE_URL:-${OPENAI_BASE_URL:-${ANTHROPIC_BASE_URL:-}}}"
# Cursor SDK key. Independent issuer (Cursor account, prefix `crsr_...`); never
# inherit from SAFE_API_KEY / OOB_API_KEY because those address the AMD gateway.
# Leave empty if the operator has not provisioned a Cursor key — the cursor
# backend will surface the missing key clearly at run time.
CURSOR_API_KEY_VAL="${CURSOR_API_KEY:-}"
CURSOR_DEFAULT_MODEL_VAL="${CURSOR_DEFAULT_MODEL:-claude-opus-4-7-thinking-xhigh}"

# Install everything by default. The previous lazy `--with-geak / --with-oob`
# scheme caused recurring "missing dependency discovered at request time"
# issues — when the resident skill triggered a kernel-opt that needed
# claude/codex but install.sh had only brought up GEAK, the CLI auth files
# were missing and every request 401'd.
# Per user direction: "kernel-agent skills do not differentiate, just install everything". The
# old --with-* / --all-backends / --backend flags are accepted but no-op
# for backwards compatibility with existing call sites.
WITH_GEAK=1
WITH_OOB=1
WITH_LLM=1
CHECK_ONLY=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: install.sh [options]

Always installs (no --with-* selectivity any more):
  ray[default]==2.44.1, click<8.3.0, TraceLens CLI,
  Node.js/npm, GEAK CLI/config, OOB + claude/codex CLI auth,
  and LLM gateway env/auth (claude/codex CLIs talk to the gateway
  directly; the legacy auth-proxy on :4002 has been retired).

Options:
  --check-only       Verify current environment, do not install
  --dry-run          Print actions without running installs
  -h, --help         Show this help

Environment (optional):
  KERNEL_AGENT_BUILD_GEAK_RAG_INDEX=1   Build the GEAK semantic RAG index in ensure_rag_index (default).
                                        Set 0 to skip — useful for claude-only kernel-opt or CPU-only
                                        sandboxes where BGE-large embedding takes ~1.5h.

Legacy options (accepted but no-op, kept for backwards compat):
  --with-geak / --with-oob / --with-llm / --all-backends / --backend NAME
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --with-geak|--with-oob|--with-llm|--all-backends)
      # No-op: install.sh always installs everything now. Accepted for
      # backwards compat with older call sites / docs.
      ;;
    --backend)
      shift
      case "${1:-}" in
        geak|oob|llm) ;;  # no-op, see above
        *) echo "[kernel-agent] ERROR: unknown backend '${1:-}'" >&2; exit 2 ;;
      esac
      ;;
    --check-only) CHECK_ONLY=1 ;;
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[kernel-agent] ERROR: unknown option '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

log() { echo "[kernel-agent] $*"; }
warn() { echo "[kernel-agent WARN] $*" >&2; }
die() { echo "[kernel-agent ERROR] $*" >&2; exit 1; }
# In --check-only mode, downgrade post-install verification failures to a
# warning so report_status can still enumerate what's missing. The caller
# explicitly asked us NOT to install; failing on the first missing piece
# defeats the point of check-only.
verify_die() {
  if [ "$CHECK_ONLY" -eq 1 ]; then warn "$1"; else die "$1"; fi
}

# Preflight credential validation. The env+.env fallback loader above
# (lines 89-105) only LOADS missing keys; it does not VALIDATE that they
# were actually provided. Without this gate, a missing SAFE_API_KEY or
# OPENAI_BASE_URL would slip past pip install / GEAK clone / aiter JIT
# (~10-20 minutes of work) and only blow up at the final
# generate_geak_litellm_config step (line ~670). Fail fast here so the
# operator can fix .env / export before any expensive work happens.
#
# Strict mode by design: no bypass env var. The chained installer
# steps (GEAK config, OOB CLI auth) all need real credentials, so an
# install without them cannot finish anyway. The only downgrade path
# is --check-only / --dry-run, which is for introspection only and
# does not actually install.
preflight_validate_credentials() {
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
         "continuing because --check-only / --dry-run is active. GEAK " \
         "config generation will still fail later unless these are set " \
         "before a real install."
    return 0
  fi
  cat >&2 <<EOF
[kernel-agent ERROR] Missing required credential(s): ${missing[*]}

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

run() {
  log "$*"
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    "$@"
  fi
}

ensure_python() {
  python3 --version >/dev/null || die "python3 is required"
  python3 -m pip --version >/dev/null || die "pip is required"
}

# Pip constraint file emitted by ensure_rocm_torch_for_geak() that pins the
# on-disk torch so GEAK pip installs cannot swap ROCm torch for PyPI CUDA torch.
GEAK_PIP_CONSTRAINT_FILE=""

ensure_rocm_torch_for_geak() {
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  if [ "${KERNEL_AGENT_SKIP_TORCH_GATE:-0}" = "1" ]; then
    warn "KERNEL_AGENT_SKIP_TORCH_GATE=1 set; not pinning torch for GEAK install"
    return 0
  fi
  # Gate on rocm-smi: only enforce torch hygiene on actual ROCm pods.
  # Non-ROCm hosts (CI/dev) get no interference, no pin.
  if ! command -v rocm-smi >/dev/null 2>&1; then
    return 0
  fi
  if ! rocm-smi --showid >/dev/null 2>&1; then
    return 0
  fi

  # Probe torch via importlib.metadata so the pinned version matches the
  # dist-info string pip uses (torch.__version__ may drop local segments).
  local probe status torch_version hip cuda_str
  probe="$(python3 - <<'PY' 2>/dev/null || true
import importlib.metadata as _m
try:
    import torch
except Exception as exc:
    print("import_error|||" + type(exc).__name__ + ": " + str(exc)[:160])
else:
    try:
        ver = _m.version("torch")
    except Exception:
        ver = getattr(torch, "__version__", "")
    print("|".join([
        "ok",
        ver,
        getattr(torch.version, "hip", None) or "",
        getattr(torch.version, "cuda", None) or "",
    ]))
PY
)"
  IFS='|' read -r status torch_version hip cuda_str <<< "$probe"
  if [ "$status" != "ok" ]; then
    warn "torch not importable from python3 on ROCm pod"
    warn "GEAK rag-mcp would pull torch from PyPI (= NVIDIA CUDA wheel) and corrupt the ROCm stack"
    warn "Fix: use the canonical ROCm Python (usually /opt/venv/bin/python3) or install ROCm torch first"
    warn "Override (NOT recommended): KERNEL_AGENT_SKIP_TORCH_GATE=1"
    die "refusing GEAK install on ROCm pod without an importable torch"
  fi

  # Pin the exact dist-info version (incl. +rocm... local segment) so GEAK
  # transitive deps cannot silently swap the on-disk torch for PyPI CUDA torch.
  GEAK_PIP_CONSTRAINT_FILE="${HYPERLOOM_RUNTIME_DIR}/geak_pip_constraints.txt"
  mkdir -p "$(dirname "$GEAK_PIP_CONSTRAINT_FILE")"
  printf 'torch==%s\n' "${torch_version}" > "$GEAK_PIP_CONSTRAINT_FILE"
  if [ -z "$hip" ]; then
    warn "torch=${torch_version} on ROCm pod is not a ROCm build (hip=none, cuda=${cuda_str:-none}); pinning to block replacement but ROCm stack may be broken"
  else
    log "pinned torch==${torch_version} (hip=${hip}) via ${GEAK_PIP_CONSTRAINT_FILE}"
  fi
}

# PR-D §3: pin `git` and `patch` so the TraceLens server patcher has the
# binaries it expects on every deployment.
#
# Background: `inference_optimizer/orchestrator/action_executors/_server_patcher.py`
# uses two binaries to apply TraceLens patches to vLLM/SGLang installs:
#   * `git apply` — strict path, default; bails immediately on context drift.
#   * `patch -p<N> --fuzz=2` — PR-C fuzzy fallback (tightened from
#     PR-C's original `--fuzz=10` to GNU patch's default `--fuzz=2` in
#     PR-D §6 to reject multi-line context drift that could mis-apply
#     patch CHANGE lines to wrong-but-similar-looking call sites).
#     Still tolerates whitespace and single-line drift, the common
#     point-release case the fuzzy fallback was designed for.
#
# Stripped runtime images (`lmsysorg/sglang:v0.5.9-rocm700-mi30x` and the
# minimal vLLM serving images) sometimes ship without one or both binaries.
# `_server_patcher` fail-softs in that case → `--enable-shape-discovery-
# for-cuda-graph-profile` is silently never injected → graph-replayed
# kernels stay opaque, exactly what #194 §5 was trying to fix.
#
# Apt-installing here is the cheap, framework-agnostic safety net: it's
# the same install path the existing `ensure_node` helper takes for
# Node.js, so it carries no new failure modes.
ensure_patch_tools() {
  log "ensuring git + patch (required by inference_optimizer/_server_patcher fuzzy-fallback path)"
  local need_git=0 need_patch=0
  command -v git >/dev/null 2>&1   || need_git=1
  command -v patch >/dev/null 2>&1 || need_patch=1
  if [ "$need_git" -eq 0 ] && [ "$need_patch" -eq 0 ]; then
    log "git: $(command -v git) ($(git --version 2>/dev/null | head -1))"
    log "patch: $(command -v patch) ($(patch --version 2>/dev/null | head -1))"
    return 0
  fi
  if [ "$CHECK_ONLY" -eq 1 ]; then
    [ "$need_git" -eq 1 ]   && warn "git missing; TraceLens server-patch strict path (\`git apply\`) will fail-soft"
    [ "$need_patch" -eq 1 ] && warn "patch missing; TraceLens server-patch fuzzy fallback (\`patch --fuzz=2\`) will fail-soft"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would apt-get install git/patch because: git=$([ $need_git -eq 1 ] && echo missing || echo present), patch=$([ $need_patch -eq 1 ] && echo missing || echo present)"
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    [ "$need_git" -eq 1 ]   && warn "git missing and apt-get unavailable; install \`git\` manually for TraceLens server patching"
    [ "$need_patch" -eq 1 ] && warn "patch missing and apt-get unavailable; install \`patch\` manually for TraceLens server patching fuzzy fallback"
    return 0
  fi
  local pkgs=()
  [ "$need_git" -eq 1 ]   && pkgs+=("git")
  [ "$need_patch" -eq 1 ] && pkgs+=("patch")
  log "apt-get installing: ${pkgs[*]}"
  apt-get update >/dev/null 2>&1 || warn "apt-get update failed; install may pull stale package indices"
  if ! apt-get -y install "${pkgs[@]}" >/dev/null; then
    warn "apt-get install of ${pkgs[*]} failed; TraceLens server patching may fail-soft on this host"
    return 0
  fi
  command -v git >/dev/null 2>&1   || warn "git still missing after apt-get install"
  command -v patch >/dev/null 2>&1 || warn "patch still missing after apt-get install"
}

# Pin `ts` (from the `moreutils` Debian/Ubuntu package) so timestamp-prefixed
# logging in downstream benchmark wrappers (Magpie's `*_mi*.sh` and any
# `cmd 2>&1 | ts '[%H:%M:%S]'` shim the optimizer fork-execs) doesn't blow
# up with `ts: command not found`.
#
# Background: stripped runtime images (e.g. `lmsysorg/sglang:v0.5.9-rocm700-mi30x`
# and the minimal vLLM serving images) ship without moreutils. When a wrapper
# pipes its stdout/stderr through `ts` for per-line timestamps and `ts` is
# missing, bash propagates exit code 127 up through the pipeline. The driving
# inference_optimizer validate_stack executor sees `subprocess_nonzero`,
# classifies the run as a baseline failure, and loops — burning minutes per
# iteration on a one-line apt fix. moreutils itself is a tiny perl-only
# package (<1 MB with deps), so this is a strict win over the retry cost.
#
# Same shape as ensure_patch_tools(): cheap apt-install with dry-run /
# check-only / no-apt-get fail-soft semantics. fail-soft on install error
# rather than die so that operators on truly air-gapped hosts can still get
# the rest of the toolchain up (the wrapper's `| ts` is a logging nicety,
# not a correctness requirement; the run itself can still produce results).
ensure_moreutils() {
  log "ensuring moreutils (provides \`ts\`; required by benchmark wrappers' timestamped logging shims)"
  if command -v ts >/dev/null 2>&1; then
    log "ts: $(command -v ts)"
    return 0
  fi
  if [ "$CHECK_ONLY" -eq 1 ]; then
    warn "ts missing; benchmark wrappers that pipe through \`| ts\` will fail with exit 127 (\`ts: command not found\`)"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would apt-get install moreutils because ts is missing"
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    warn "ts missing and apt-get unavailable; install \`moreutils\` manually (apt-get install moreutils, or distro equivalent)"
    return 0
  fi
  log "apt-get installing: moreutils"
  apt-get update >/dev/null 2>&1 || warn "apt-get update failed; install may pull stale package indices"
  if ! apt-get -y install moreutils >/dev/null; then
    warn "apt-get install of moreutils failed; benchmark wrappers' \`| ts\` timestamping will fail-soft on this host"
    return 0
  fi
  command -v ts >/dev/null 2>&1 || warn "ts still missing after apt-get install moreutils"
}

ensure_node() {
  log "ensuring Node.js/npm for claude/codex CLIs and @cursor/sdk"
  if command -v node >/dev/null 2>&1 && npm --version >/dev/null 2>&1; then
    log "node: $(command -v node) ($(node --version 2>/dev/null || echo unknown))"
    log "npm: $(command -v npm) ($(npm --version 2>/dev/null || echo unknown))"
    return 0
  fi

  if [ "$CHECK_ONLY" -eq 1 ]; then
    warn "node/npm missing; claude/codex CLI install would be skipped"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would install Node.js 20 from NodeSource because node/npm is missing"
    return 0
  fi
  if ! command -v apt-get >/dev/null 2>&1; then
    verify_die "node/npm missing and apt-get is unavailable; install Node.js 20 manually"
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    log "curl missing; installing curl/ca-certificates before NodeSource setup"
    apt-get update >/dev/null
    apt-get -y install ca-certificates curl gnupg >/dev/null
  fi

  log "installing Node.js 20 from NodeSource"
  apt-get -y purge libnode-dev libnode72 nodejs nodejs-doc npm >/dev/null 2>&1 || true
  local ns_url="https://deb.nodesource.com/setup_20.x"
  local ns_sha="2c4c6683a17b6f4128898a7b521e3c8bb725a99ffaf1b5e32ac97c6fa7d381be"
  local ns_script="/tmp/nodesource_setup_20.x"
  if ! curl -fsSL "$ns_url" -o "$ns_script"; then
    verify_die "NodeSource setup download failed; cannot install Node.js/npm for claude/codex CLIs"
    return 0
  fi
  if ! echo "${ns_sha}  ${ns_script}" | sha256sum -c - >/dev/null 2>&1; then
    verify_die "NodeSource setup SHA256 mismatch; aborting Node.js/npm install"
    return 0
  fi
  if ! bash "$ns_script" >/dev/null 2>&1; then
    verify_die "NodeSource setup failed; cannot install Node.js/npm for claude/codex CLIs"
    return 0
  fi
  if ! apt-get -y install nodejs >/dev/null; then
    verify_die "nodejs install failed; claude/codex CLI install cannot proceed"
    return 0
  fi
  command -v node >/dev/null 2>&1 || verify_die "node CLI not found after nodejs install"
  npm --version >/dev/null 2>&1 || verify_die "npm not usable after nodejs install"
}

ensure_ray() {
  log "ensuring ray[default]==2.44.1 and click<8.3.0"
  if [ "$CHECK_ONLY" -eq 0 ]; then
    run python3 -m pip install --quiet --no-cache-dir --break-system-packages "click<8.3.0" "ray[default]==2.44.1"
  fi
  if [ "$DRY_RUN" -eq 0 ]; then
    python3 - <<'PY'
import ray, sys
if ray.__version__ != "2.44.1":
    raise SystemExit(f"ray version mismatch: {ray.__version__} != 2.44.1")
print(f"[kernel-agent] ray version: {ray.__version__}")
PY
  fi
}

# Idempotently bring up a Ray head node. Kernel backends submit Ray tasks with
# `num_gpus>=1`; if no head is running (or one is running with --num-gpus=0)
# kernel optimization will hang forever even when GPUs are idle. We:
#   1. detect a live Ray head via `ray status` (any successful return = live)
#   2. if absent, force-stop any half-started Ray and start a fresh head with
#      all visible GPUs advertised
#   3. tolerate the no-GPU case (CPU-only dev box) so `--check-only` stays
#      non-fatal in environments without ROCm
ensure_ray_started() {
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  if ! command -v ray >/dev/null 2>&1; then
    warn "ray CLI missing; cannot start ray head"
    return 0
  fi
  if ray status >/dev/null 2>&1; then
    log "ray head already running"
    return 0
  fi
  log "no live ray head detected; starting one"
  ray stop --force >/dev/null 2>&1 || true
  local num_gpus
  num_gpus="$(python3 - <<'PY' 2>/dev/null || echo 0
try:
    import torch
    print(torch.cuda.device_count() or 0)
except Exception:
    print(0)
PY
)"
  if [ "${RAY_NUM_GPUS:-}" != "" ]; then
    num_gpus="$RAY_NUM_GPUS"
  fi
  log "starting ray head with --num-gpus=${num_gpus}"
  if ! ray start --head --disable-usage-stats \
       --num-gpus="$num_gpus" --include-dashboard=false >/dev/null; then
    warn "ray start failed; kernel optimization will hang. Check ROCm visibility."
    return 0
  fi
  ray status >/dev/null 2>&1 || warn "ray status reports no live head after start"
}

ensure_tracelens() {
  # Legacy bundle fallback: the fully-local image ships either a fresh
  # ${HYPERLOOM_BUNDLE}/TraceLens checkout (post-2026-05-18 OSS layout)
  # or, for older bundles, the pre-migration TraceLens-internal mirror.
  if [ ! -d "$TRACELENS_ROOT" ] && [ -d "${HYPERLOOM_BUNDLE}/TraceLens" ]; then
    TRACELENS_ROOT="${HYPERLOOM_BUNDLE}/TraceLens"
  fi
  if [ ! -d "$TRACELENS_ROOT" ] && [ -d "${HYPERLOOM_BUNDLE}/TraceLens-internal" ]; then
    TRACELENS_ROOT="${HYPERLOOM_BUNDLE}/TraceLens-internal"
  fi
  if [ ! -d "$TRACELENS_ROOT" ]; then
    if [ "$DRY_RUN" -eq 1 ] || [ "$CHECK_ONLY" -eq 1 ]; then
      warn "TraceLens root not found: $TRACELENS_ROOT"
      return
    fi
    die "TraceLens root not found: $TRACELENS_ROOT"
  fi
  # Read-only source guard (mirrors the OOB cp -r pattern). When
  # $TRACELENS_ROOT is on a read-only mount (the WekaFS default), pip
  # install -e fails because it must write *.egg-info into the source
  # tree, and at runtime tools/tracelens_analysis.py re-runs the same
  # editable install in a subprocess on every trace_analyze request,
  # producing a tight failure loop. Detecting unwritable source up front
  # and mirroring to ${HYPERLOOM_ROOT}/TraceLens-internal (parallel to
  # ${HYPERLOOM_ROOT}/geak / ${HYPERLOOM_ROOT}/OOB/oob_cli) lets both
  # the install-time and the runtime pip install land on a writable
  # filesystem. write_env_file() emits the resulting TRACELENS_ROOT into
  # the pod-local kernel-agent env so subsequent CLI subprocesses (setsid nohup inference_optimizer
  # optimize → kernel-agent → tracelens_analysis.py) inherit the mirror.
  if [ "$CHECK_ONLY" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    if ! ( : > "$TRACELENS_ROOT/.hl_write_test" ) 2>/dev/null; then
      log "TraceLens root not writable ($TRACELENS_ROOT); mirroring to $TRACELENS_MIRROR_DIR"
      mkdir -p "$(dirname "$TRACELENS_MIRROR_DIR")"
      if [ ! -d "$TRACELENS_MIRROR_DIR" ]; then
        log "mirroring TraceLens to writable dir (large tree; may take minutes): $TRACELENS_ROOT -> $TRACELENS_MIRROR_DIR"
        run cp -r "$TRACELENS_ROOT" "$TRACELENS_MIRROR_DIR"
      else
        log "TraceLens mirror already present: $TRACELENS_MIRROR_DIR"
      fi
      TRACELENS_ROOT="$TRACELENS_MIRROR_DIR"
      export TRACELENS_ROOT
    else
      rm -f "$TRACELENS_ROOT/.hl_write_test"
    fi
  fi
  log "ensuring TraceLens CLI from $TRACELENS_ROOT"
  if [ "$CHECK_ONLY" -eq 0 ]; then
    # Do not use bash -lc: login profiles reset PATH (drops venv) and break pip.
    run sh -c "cd '$TRACELENS_ROOT' && python3 -m pip install -q --no-cache-dir --break-system-packages -e ."
  fi
  # Optional TraceLens-internal extension. Opt-in path: operator (or
  # local_setup.sh) exports TRACELENS_INTERNAL_ROOT pointing at a
  # checkout of AMD-AGI/TraceLens-internal. When unset, Hyperloom stays
  # on the open-source-only report (impact score, no roofline).
  if [ -n "${TRACELENS_INTERNAL_ROOT:-}" ]; then
    if [ ! -d "$TRACELENS_INTERNAL_ROOT" ]; then
      if [ "$DRY_RUN" -eq 1 ] || [ "$CHECK_ONLY" -eq 1 ]; then
        warn "TRACELENS_INTERNAL_ROOT set but not on disk: $TRACELENS_INTERNAL_ROOT"
      else
        die "TRACELENS_INTERNAL_ROOT set but not on disk: $TRACELENS_INTERNAL_ROOT"
      fi
    elif [ "$CHECK_ONLY" -eq 0 ]; then
      log "ensuring TraceLens-internal extension from $TRACELENS_INTERNAL_ROOT"
      run sh -c "cd '$TRACELENS_INTERNAL_ROOT' && python3 -m pip install -q --no-cache-dir --break-system-packages -e ."
    fi
  fi
  if [ "$DRY_RUN" -eq 0 ]; then
    # TraceLens #124: only the inference variant is accepted (the correct
    # entry for vLLM/SGLang traces). Hyperloom is inference-only since
    # v0.4; the legacy training-mode CLI was removed to keep install /
    # runtime in lockstep.
    if command -v TraceLens_generate_perf_report_pytorch_inference >/dev/null 2>&1; then
      TraceLens_generate_perf_report_pytorch_inference --help >/dev/null
      log "TraceLens perf CLI verified: TraceLens_generate_perf_report_pytorch_inference (#124)"
    else
      verify_die "TraceLens_generate_perf_report_pytorch_inference not found after install (Hyperloom is inference-only since v0.4; bump TraceLens-internal)"
    fi
  fi
}

ensure_geak() {
  log "ensuring GEAK backend"
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    mkdir -p "${HYPERLOOM_ROOT}" "$(dirname "$GEAK_CONFIG")" "$(dirname "$GEAK_MEMORY_STORE_PATH_VAL")"
  fi
  if [ ! -d "${HYPERLOOM_ROOT}/geak/.git" ]; then
    # ``git clone --branch`` only accepts tags / branches, not SHAs. Detect
    # a 7-40 hex char SHA and use a fetch-checkout dance instead so the
    # SHA pin above stays shallow. GitHub serves shallow SHA fetches
    # (uploadpack.allowReachableSHA1InWant=true).
    if [[ "$GEAK_REF" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
      run git init -q "${HYPERLOOM_ROOT}/geak"
      run git -C "${HYPERLOOM_ROOT}/geak" remote add origin "$GEAK_REPO"
      run git -C "${HYPERLOOM_ROOT}/geak" fetch --depth 1 origin "$GEAK_REF"
      run git -C "${HYPERLOOM_ROOT}/geak" checkout -q FETCH_HEAD
    else
      run git clone --depth 1 --branch "$GEAK_REF" "$GEAK_REPO" "${HYPERLOOM_ROOT}/geak"
    fi
  else
    log "GEAK checkout already present: ${HYPERLOOM_ROOT}/geak"
  fi
  if [ "$CHECK_ONLY" -eq 0 ]; then
    # Pin the pip flag set so we work in both venv installs (main upstream
    # assumption) and sandbox / system-python installs (multi-node feature
    # branch needs --break-system-packages; pip in a venv treats it as a
    # no-op so the flag is safe to keep unconditionally).
    _PIP_FLAGS="-q --no-cache-dir --break-system-packages"
    ensure_rocm_torch_for_geak
    _PIP_CONSTRAINT_ARGS=""
    if [ -n "${GEAK_PIP_CONSTRAINT_FILE:-}" ] && [ -f "${GEAK_PIP_CONSTRAINT_FILE}" ]; then
      _PIP_CONSTRAINT_ARGS="--constraint ${GEAK_PIP_CONSTRAINT_FILE}"
    fi
    run python3 -m pip install ${_PIP_FLAGS} ${_PIP_CONSTRAINT_ARGS} "${HYPERLOOM_ROOT}/geak"
    # GEAK v3.2.0 ships 4 MCP tools under mcp_tools/; all are imported
    # by the bundled ``minisweagent`` at preprocess time:
    #   * rag-mcp                    — knowledge-base retrieval (tools.rag)
    #   * profiler-mcp               — Metrix-backed instrumented profiling
    #                                  (preprocessor.py import); Metrix is now
    #                                  a PyPI dep declared in its pyproject,
    #                                  not a separate mcp_tools/ folder.
    #   * cross-session-memory-mcp   — GEAK_MEMORY_STORE_PATH retriever
    #   * automated-test-discovery   — pre-fills eval_command harness
    # v3.1.0 -> v3.2.0 change: ``metrix-mcp`` folder was removed and the
    # metrix runtime is now consumed transitively via profiler-mcp's
    # ``dependencies = ["metrix>=0.1.0"]``. Listing it here again would
    # break install with "File ... does not exist".
    for _geak_mcp in rag-mcp profiler-mcp \
                    cross-session-memory-mcp automated-test-discovery; do
      run python3 -m pip install ${_PIP_FLAGS} ${_PIP_CONSTRAINT_ARGS} \
        "${HYPERLOOM_ROOT}/geak/mcp_tools/${_geak_mcp}"
    done
    # Patch GEAK's bundled prompt YAML to remove the misleading
    # ``task_runner.py performance`` example that causes sub-agent
    # LLMs to burn budget on ``find /`` for a non-existent script.
    # Idempotent and fail-soft — see kernel-agent/tools/geak_prompt_patcher.py
    # for the full rationale. Always best-effort; only blocking when
    # the operator explicitly opts in via HYPERLOOM_GEAK_PROMPT_PATCH_REQUIRED=1.
    _geak_patcher="${KERNEL_AGENT_ROOT}/tools/geak_prompt_patcher.py"
    if [ -f "$_geak_patcher" ]; then
      run python3 "$_geak_patcher"
    else
      warn "geak prompt patcher missing at $_geak_patcher; skip"
    fi
  else
    log "check-only: skipping GEAK and mcp_tools installation"
  fi
  if [ "$CHECK_ONLY" -eq 0 ]; then
    if [ "$DRY_RUN" -eq 0 ]; then
      if [ -z "$GEAK_API_KEY_VAL" ] || [ -z "$GEAK_BASE_URL_VAL" ]; then
        die "Cannot generate GEAK litellm config: SAFE_API_KEY and OPENAI_BASE_URL are required"
      fi
      cat > "$GEAK_CONFIG" <<EOF
model:
  model_class: litellm
  model_name: ${GEAK_MODEL_NAME_VAL}
  api_key: ${GEAK_API_KEY_VAL}
  base_url: ${GEAK_BASE_URL_VAL}
  model_kwargs:
    max_tokens: 16384
tools:
  rag: true
run:
  mode: ${GEAK_RUN_MODE_VAL}
  budgets:
    quick:
      total_s: 3600
      preprocess_soft_cap_s: 900
      preprocess_hard_cap_fraction: 0.5
      finalize_grace_s: 300
      kill_buffer_s: 60
    full:
      total_s: 7200
      preprocess_soft_cap_s: 900
      preprocess_hard_cap_fraction: 0.5
      finalize_grace_s: 300
      kill_buffer_s: 60
  presets:
    quick:
      orchestrator:
        max_rounds: 2
    full:
      orchestrator:
        max_rounds: 5
env:
  env:
    PAGER: cat
    MANPAGER: cat
    LESS: -R
    PIP_PROGRESS_BAR: 'off'
    TQDM_DISABLE: '1'
  timeout: 3600
EOF
      chmod 600 "$GEAK_CONFIG"
      grep -Eq '^[[:space:]]*model_class:[[:space:]]*litellm[[:space:]]*$' "$GEAK_CONFIG" \
        || die "GEAK config must force model_class: litellm: $GEAK_CONFIG"
    else
      if [ -z "$GEAK_API_KEY_VAL" ] || [ -z "$GEAK_BASE_URL_VAL" ]; then
        warn "GEAK_API_KEY/GEAK_BASE_URL not fully set"
      fi
      log "would write GEAK config: $GEAK_CONFIG"
    fi
  else
    if [ "$DRY_RUN" -eq 0 ]; then
      if [ -z "$GEAK_API_KEY_VAL" ] || [ -z "$GEAK_BASE_URL_VAL" ]; then
        warn "GEAK_API_KEY/GEAK_BASE_URL not fully set"
      fi
      if [ -f "$GEAK_CONFIG" ]; then
        grep -Eq '^[[:space:]]*model_class:[[:space:]]*litellm[[:space:]]*$' "$GEAK_CONFIG" \
          || warn "GEAK config does not force model_class: litellm: $GEAK_CONFIG"
      else
        warn "GEAK config missing: $GEAK_CONFIG"
      fi
    fi
  fi
  if [ "$DRY_RUN" -eq 0 ]; then
    command -v geak >/dev/null 2>&1 || verify_die "geak CLI not found"
  fi
  patch_geak_minisweagent_runtime
}

patch_geak_minisweagent_runtime() {
  # Hyperloom runtime patch for GEAK-v3 / mini-swe-agent 1.14.4.
  #
  # Two issues were observed during Qwen3-8B GEAK runs:
  #   1. minisweagent.models.litellm_model converted tool schemas to
  #      function.input_schema. LiteLLM's OpenAI-shape Anthropic mapper reads
  #      function.parameters, so every tool reached Claude with an empty schema
  #      and Claude repeatedly emitted bash{}.
  #   2. Bare claude-* model names made LiteLLM choose Anthropic /v1/messages,
  #      while our gateway is OpenAI-compatible /chat/completions. The config
  #      generation above now rewrites bare claude-* to openai/claude-* and
  #      preserves the /v1 base_url.
  #
  # This patch is intentionally idempotent and applies both to the writable GEAK
  # source mirror and to the installed site-packages copy used by the current
  # pod. Keeping it here makes Hyperloom's source tree the durable owner of the
  # workaround; direct runtime edits are only a validation artefact.
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  python3 - <<'PY'
import json
import os
from pathlib import Path

paths: list[Path] = []

mirror = Path(os.environ.get("HYPERLOOM_ROOT", "")) / "geak" / "src" / "minisweagent"
if mirror.exists():
    paths.append(mirror)

try:
    import minisweagent  # type: ignore
    paths.append(Path(minisweagent.__file__).resolve().parent)
except Exception:
    pass

seen: set[Path] = set()
for root in paths:
    root = root.resolve()
    if root in seen:
        continue
    seen.add(root)

    litellm_model = root / "models" / "litellm_model.py"
    if litellm_model.exists():
        text = litellm_model.read_text()
        old = '''        function: dict[str, Any] = {
            "name": name,
            "description": func.get("description", ""),
            "input_schema": func.get(
                "parameters",
                {
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
        }'''
        new = '''        schema = func.get("parameters") or func.get(
            "input_schema",
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
        )
        function: dict[str, Any] = {
            "name": name,
            "description": func.get("description", ""),
            "parameters": schema,
        }'''
        if old in text:
            litellm_model.write_text(text.replace(old, new, 1))
            print(f"[kernel-agent] patched LiteLLM tool schema: {litellm_model}")
        elif '"parameters": schema' in text:
            print(f"[kernel-agent] LiteLLM tool schema already patched: {litellm_model}")
        else:
            print(f"[kernel-agent WARN] LiteLLM schema patch pattern not found: {litellm_model}")

    cfg = root / "config" / "mini_kernel_strategy_list.yaml"
    if cfg.exists():
        text = cfg.read_text()
        marker = "Never call `bash` with `{}` or an empty command."
        if marker not in text:
            needle = (
                "    Your response must contain exactly ONE tool call.\n"
                "    Include a THOUGHT section before your tool call where you explain your reasoning process.\n"
            )
            repl = needle + (
                "    If you choose the `bash` tool, its arguments MUST include a non-empty\n"
                "    `command` string, for example:\n"
                "    {\"command\": \"python3 scripts/task_runner.py correctness\"}.\n"
                "    Never call `bash` with `{}` or an empty command.\n"
            )
            if needle in text:
                cfg.write_text(text.replace(needle, repl, 1))
                print(f"[kernel-agent] patched bash prompt guidance: {cfg}")
            else:
                print(f"[kernel-agent WARN] bash prompt guidance pattern not found: {cfg}")
        else:
            print(f"[kernel-agent] bash prompt guidance already patched: {cfg}")

    tools_json = root / "tools" / "tools.json"
    if tools_json.exists():
        data = json.loads(tools_json.read_text())
        changed = False
        for tool in data:
            if tool.get("name") == "bash":
                desc = (
                    "Execute shell commands directly in bash. REQUIRED: pass a "
                    "non-empty `command` string in the arguments object. Never "
                    "call bash with `{}`."
                )
                cmd_desc = "The non-empty bash command to execute, e.g. `ls -la /tmp`."
                if tool.get("description") != desc:
                    tool["description"] = desc
                    changed = True
                params = tool.setdefault("parameters", {})
                params.setdefault("type", "object")
                params.setdefault("properties", {}).setdefault("command", {})["type"] = "string"
                if params["properties"]["command"].get("description") != cmd_desc:
                    params["properties"]["command"]["description"] = cmd_desc
                    changed = True
                if params.get("required") != ["command"]:
                    params["required"] = ["command"]
                    changed = True
        if changed:
            tools_json.write_text(json.dumps(data, indent=2) + "\n")
            print(f"[kernel-agent] patched bash tool schema docs: {tools_json}")
        else:
            print(f"[kernel-agent] bash tool schema docs already patched: {tools_json}")
PY
}

ensure_rag_index() {
  case "$KERNEL_AGENT_BUILD_GEAK_RAG_INDEX_VAL" in
    0|false|FALSE|no|NO|off|OFF)
      log "skipping GEAK RAG index build (KERNEL_AGENT_BUILD_GEAK_RAG_INDEX=$KERNEL_AGENT_BUILD_GEAK_RAG_INDEX_VAL)"
      return 0
      ;;
  esac
  if [ -d "$RAG_INDEX_DIR" ] && [ -n "$(ls -A "$RAG_INDEX_DIR" 2>/dev/null)" ]; then
    log "RAG index already present at $RAG_INDEX_DIR"
    return
  fi
  if [ "$CHECK_ONLY" -eq 1 ]; then
    warn "RAG index missing at $RAG_INDEX_DIR"
    return
  fi
  log "building RAG index at $RAG_INDEX_DIR on device=${GEAK_RAG_INDEX_DEVICE_VAL} (first run downloads ~1.3 GB embedding model)"
  run sh -c "cd '${HYPERLOOM_ROOT}/geak' && python3 scripts/build_index.py --force --device '${GEAK_RAG_INDEX_DEVICE_VAL}'"
}

ensure_oob() {
  log "ensuring OOB backend"
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    mkdir -p "${HYPERLOOM_ROOT}/OOB"
  fi
  if ! command -v oob >/dev/null 2>&1; then
    if [ -d "$OOB_SRC" ]; then
      if [ ! -d "${HYPERLOOM_ROOT}/OOB/oob_cli" ]; then
        run cp -r "$OOB_SRC" "${HYPERLOOM_ROOT}/OOB/oob_cli"
      fi
      if [ -f "${HYPERLOOM_ROOT}/OOB/oob_cli/requirements.txt" ]; then
        run python3 -m pip install -q --no-cache-dir --break-system-packages -r "${HYPERLOOM_ROOT}/OOB/oob_cli/requirements.txt"
      fi
      run python3 -m pip install -q --no-cache-dir --break-system-packages "${HYPERLOOM_ROOT}/OOB/oob_cli"
    else
      warn "OOB source not found: $OOB_SRC"
    fi
  else
    log "oob already installed: $(command -v oob)"
  fi

  if ! command -v npm >/dev/null 2>&1; then
    if [ "$DRY_RUN" -eq 1 ]; then
      log "would install claude/codex npm CLIs after Node.js/npm is installed"
      ensure_llm_auth_files
      return 0
    fi
    verify_die "npm not found; ensure_node must run before ensure_oob"
    return 0
  fi
  if ! command -v claude >/dev/null 2>&1; then
    run npm config set prefix /usr/local
    run npm install -g @anthropic-ai/claude-code
  fi
  if ! command -v codex >/dev/null 2>&1; then
    run npm config set prefix /usr/local
    run npm install -g @openai/codex@0.100.0
  fi
  # @cursor/sdk is a Node library (not a CLI), used by OOB's cursor backend
  # via `node -e "import '@cursor/sdk'"`. We always install it globally so
  # the OOB cursor path works without per-pod npm setup. Use `require.resolve`
  # against the global root to avoid a redundant install when present.
  if ! NODE_PATH="$(npm root -g 2>/dev/null || true)" \
       node -e "require.resolve('@cursor/sdk')" >/dev/null 2>&1; then
    run npm config set prefix /usr/local
    run npm install -g @cursor/sdk
  fi

  ensure_llm_auth_files
}

ensure_llm_auth_files() {
  log "ensuring LLM/OOB auth files"
  if [ "$CHECK_ONLY" -eq 1 ]; then
    return
  fi
  if [ -z "$OOB_API_KEY_VAL" ]; then
    warn "OOB/Anthropic API key not set; auth files not written"
    return
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would write /root/.claude/config.json and /root/.codex/auth.json"
    return
  fi
  mkdir -p /root/.claude /root/.codex
  # The Anthropic SDK appends /v1 itself, so strip a trailing /v1 from
  # the OpenAI-style upstream URL when writing customApiUrl.
  local _anthropic_url="${OOB_BASE_URL_VAL%/}"
  _anthropic_url="${_anthropic_url%/v1}"
  cat > /root/.claude/config.json <<EOF
{
  "theme": "dark",
  "hasCompletedOnboarding": true,
  "primaryApiKey": "${OOB_API_KEY_VAL}",
  "customApiUrl": "${_anthropic_url}"
}
EOF
  chmod 600 /root/.claude/config.json
  # Codex 0.100.0 reads ~/.codex/auth.json before env and does not fall
  # back when OPENAI_API_KEY is empty; write the real key for direct auth.
  cat > /root/.codex/auth.json <<EOF
{
  "auth_mode": "apikey",
  "OPENAI_API_KEY": "${OOB_API_KEY_VAL}"
}
EOF
  chmod 600 /root/.codex/auth.json
}

# Write a pod-local kernel-agent env file users should source so subsequent CLI calls
# (and Ray workers via runtime_env) pick up the upstream gateway URLs.
write_env_file() {
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  # Warn loudly if OOB_BASE_URL is empty — kernel-agent env would silently
  # lack ANTHROPIC_BASE_URL/OPENAI_BASE_URL and CLIs would resort to whatever
  # was in the operator's shell rc, defeating the point of this file.
  if [ -z "${OOB_BASE_URL_VAL:-}" ]; then
    warn "OOB_BASE_URL empty; kernel-agent env will lack ANTHROPIC_BASE_URL/OPENAI_BASE_URL"
  fi
  # Anthropic SDK appends /v1 itself, so strip a trailing /v1 from the
  # OpenAI-style upstream URL when exporting ANTHROPIC_BASE_URL.
  local _anthropic_url=""
  if [ -n "${OOB_BASE_URL_VAL:-}" ]; then
    _anthropic_url="${OOB_BASE_URL_VAL%/}"
    _anthropic_url="${_anthropic_url%/v1}"
  fi
  local env_file="${KERNEL_AGENT_ENV}"
  mkdir -p "$(dirname "$env_file")"
  {
    echo '#!/bin/sh'
    echo "# kernel-agent runtime env (regenerated by install.sh)"
    [ -n "${USER_DATA_PATH:-}" ] && echo "export USER_DATA_PATH='${USER_DATA_PATH}'"
    [ -n "${HYPERLOOM_RUNTIME_DIR:-}" ] && echo "export HYPERLOOM_RUNTIME_DIR='${HYPERLOOM_RUNTIME_DIR}'"
    [ -n "${KERNEL_AGENT_ENV:-}" ] && echo "export KERNEL_AGENT_ENV='${KERNEL_AGENT_ENV}'"
    [ -n "${HYPERLOOM_KERNEL_AGENT_ROOT:-}" ] && echo "export HYPERLOOM_KERNEL_AGENT_ROOT='${HYPERLOOM_KERNEL_AGENT_ROOT}'"
    [ -n "${KERNEL_AGENT_ROOT:-}" ] && echo "export KERNEL_AGENT_ROOT='${KERNEL_AGENT_ROOT}'"
    [ -n "${MAGPIE_DIR:-}" ] && echo "export MAGPIE_DIR='${MAGPIE_DIR}'"
    [ -n "${MAGPIE_PYTHON:-}" ] && echo "export MAGPIE_PYTHON='${MAGPIE_PYTHON}'"
    [ -n "${PYTHONPATH:-}" ] && echo "export PYTHONPATH='${PYTHONPATH}'"
    [ -n "${INFERENCEX_PATH:-}" ] && echo "export INFERENCEX_PATH='${INFERENCEX_PATH}'"
    [ -n "${_anthropic_url}" ] && echo "export ANTHROPIC_BASE_URL='${_anthropic_url}'"
    [ -n "${OOB_BASE_URL_VAL:-}" ] && echo "export OPENAI_BASE_URL='${OOB_BASE_URL_VAL}'"
    [ -n "${OOB_API_KEY_VAL}" ] && {
      echo "export SAFE_API_KEY='${OOB_API_KEY_VAL}'"
      echo "export ANTHROPIC_API_KEY='${OOB_API_KEY_VAL}'"
      echo "export ANTHROPIC_AUTH_TOKEN='${OOB_API_KEY_VAL}'"
      echo "export OPENAI_API_KEY='${OOB_API_KEY_VAL}'"
      echo "export OOB_API_KEY='${OOB_API_KEY_VAL}'"
      echo "export AMD_LLM_API_KEY='${OOB_API_KEY_VAL}'"
      echo "export LLM_GATEWAY_KEY='${OOB_API_KEY_VAL}'"
    }
    [ -n "${OOB_BASE_URL_VAL}" ] && echo "export OOB_BASE_URL='${OOB_BASE_URL_VAL}'"
    # Cursor backend env. Only export CURSOR_API_KEY if the operator already
    # provided one via env or .env; do not synthesise from SAFE_API_KEY (the
    # Cursor account is a separate issuer).
    [ -n "${CURSOR_API_KEY_VAL}" ] && echo "export CURSOR_API_KEY='${CURSOR_API_KEY_VAL}'"
    [ -n "${CURSOR_DEFAULT_MODEL_VAL}" ] && echo "export CURSOR_DEFAULT_MODEL='${CURSOR_DEFAULT_MODEL_VAL}'"
    # Pin TRACELENS_ROOT to the (possibly mirrored) value resolved by
    # ensure_tracelens(). This is what lets setsid nohup inference_optimizer
    # optimize → kernel-agent/tools/tracelens_analysis.py inherit the writable
    # mirror instead of falling back to the read-only /wekafs default.
    [ -n "${TRACELENS_ROOT:-}" ] && echo "export TRACELENS_ROOT='${TRACELENS_ROOT}'"
    [ -n "${TRACELENS_INTERNAL_ROOT:-}" ] && echo "export TRACELENS_INTERNAL_ROOT='${TRACELENS_INTERNAL_ROOT}'"
    [ -n "${GEAK_CONFIG}" ] && echo "export GEAK_CONFIG='${GEAK_CONFIG}'"
    [ -n "${GEAK_RUN_MODE_VAL}" ] && echo "export GEAK_RUN_MODE='${GEAK_RUN_MODE_VAL}'"
    [ -n "${GEAK_MODEL_NAME_VAL}" ] && echo "export GEAK_MODEL_NAME='${GEAK_MODEL_NAME_VAL}'"
    [ -n "${GEAK_API_KEY_VAL}" ] && echo "export GEAK_API_KEY='${GEAK_API_KEY_VAL}'"
    [ -n "${GEAK_BASE_URL_VAL}" ] && echo "export GEAK_BASE_URL='${GEAK_BASE_URL_VAL}'"
    [ -n "${GEAK_MEMORY_STORE_PATH_VAL}" ] && echo "export GEAK_MEMORY_STORE_PATH='${GEAK_MEMORY_STORE_PATH_VAL}'"
    [ -n "${GEAK_SAVE_TO_KNOWLEDGE_BASE_VAL}" ] && echo "export GEAK_SAVE_TO_KNOWLEDGE_BASE='${GEAK_SAVE_TO_KNOWLEDGE_BASE_VAL}'"
    [ -n "${GEAK_MEMORY_MIN_SPEEDUP_VAL}" ] && echo "export GEAK_MEMORY_MIN_SPEEDUP='${GEAK_MEMORY_MIN_SPEEDUP_VAL}'"
    [ -n "${CODEX_MODEL_VAL}" ] && echo "export CODEX_MODEL='${CODEX_MODEL_VAL}'"
  } > "$env_file"
  chmod 600 "$env_file"
  log "wrote ${env_file} (source it before running kernel-agent tools)"
}

report_status() {
  log "root: ${KERNEL_AGENT_ROOT}"
  log "ray: $(python3 - <<'PY' 2>/dev/null || echo missing
try:
    import ray
    print(ray.__version__)
except Exception:
    raise SystemExit(1)
PY
)"
  # TraceLens perf-report CLI: only the inference variant is accepted
  # (#124). Hyperloom is inference-only since v0.4; the legacy
  # training-mode CLI was removed because its output shape silently
  # breaks downstream fusion / roofline analysis.
  if command -v TraceLens_generate_perf_report_pytorch_inference >/dev/null 2>&1; then
    log "found TraceLens_generate_perf_report_pytorch_inference: $(command -v TraceLens_generate_perf_report_pytorch_inference)"
  else
    warn "TraceLens_generate_perf_report_pytorch_inference not found (Hyperloom is inference-only since v0.4)"
  fi
  for tool in geak oob claude codex; do
    if command -v "$tool" >/dev/null 2>&1; then
      log "found ${tool}: $(command -v "$tool")"
    else
      warn "${tool} not found"
    fi
  done
  for tool in git patch; do
    if command -v "$tool" >/dev/null 2>&1; then
      log "found ${tool}: $(command -v "$tool")"
    else
      warn "${tool} not found (TraceLens server patcher will fail-soft without it)"
    fi
  done
  # @cursor/sdk is a library, not a CLI; verify via require.resolve.
  if NODE_PATH="$(npm root -g 2>/dev/null || true)" \
     node -e "require.resolve('@cursor/sdk')" >/dev/null 2>&1; then
    log "found @cursor/sdk in $(npm root -g 2>/dev/null || echo '?')"
  else
    warn "@cursor/sdk not installed (cursor backend will fail to start)"
  fi
  if [ -n "$CURSOR_API_KEY_VAL" ]; then
    log "CURSOR_API_KEY: set"
  else
    warn "CURSOR_API_KEY not set; cursor backend will 401 if invoked"
  fi
  if [ -d "${HYPERLOOM_ROOT}/geak/.git" ]; then
    log "GEAK ref: $(git -C "${HYPERLOOM_ROOT}/geak" describe --tags --always 2>/dev/null || echo unknown)"
  else
    warn "GEAK checkout missing at ${HYPERLOOM_ROOT}/geak"
  fi
  if python3 -c "import rag_mcp" >/dev/null 2>&1; then
    log "rag-mcp installed: yes"
  else
    warn "rag-mcp not installed"
  fi
  if [ -d "$RAG_INDEX_DIR" ] && [ -n "$(ls -A "$RAG_INDEX_DIR" 2>/dev/null)" ]; then
    log "RAG index: present at $RAG_INDEX_DIR"
  else
    warn "RAG index missing at $RAG_INDEX_DIR"
  fi
  log "RAG index build device: ${GEAK_RAG_INDEX_DEVICE_VAL}"
  if grep -q "rag: true" "$GEAK_CONFIG" 2>/dev/null; then
    log "tools.rag enabled in $GEAK_CONFIG"
  else
    warn "tools.rag not enabled in $GEAK_CONFIG"
  fi
  log "GEAK memory store: ${GEAK_MEMORY_STORE_PATH_VAL}"
}

main() {
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    # KERNEL_AGENT_ROOT is now the source root (read-only checkout); tool
    # outputs land under $USER_DATA_PATH/kernel-agent/runs/<session_id>/
    # (created lazily by the tools themselves). All we need here is the
    # writable runtime tree on $USER_DATA_PATH for the env file + GEAK
    # config + source mirrors.
    mkdir -p "${HYPERLOOM_RUNTIME_DIR}" "${HYPERLOOM_ROOT}"
  fi
  ensure_python
  ensure_node
  ensure_patch_tools
  ensure_moreutils
  ensure_ray
  ensure_ray_started
  ensure_tracelens

  # Always install everything; ensure_oob also calls ensure_llm_auth_files.
  ensure_geak
  ensure_rag_index
  ensure_oob
  write_env_file

  report_status
  log "install complete"
}

main "$@"
