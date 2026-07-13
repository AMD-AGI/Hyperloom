#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc. All rights reserved.

# Kernel-agent installer.
#
# Base install is intentionally small and deterministic:
#   - ray[default]==2.44.1 + click<8.3.0
#   - TraceLens editable install + CLI verification
#   - GEAK per-kernel + e2e optimizer backends
#
# The installer prepares all kernel-agent backends in one pass.

set -euo pipefail

# Ray/K8s subprocesses may inherit a minimal PATH; git/apt/node live under
# /usr/bin even when callers only prepend /opt/venv/bin. Prepend the
# standard system bins so multi-node RayJob children resolve them.
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

# Default every writable artefact location under $USER_DATA_PATH so a single
# session-dir move relocates Magpie / source mirrors / the
# kernel-agent env file. Operators can still pin individual paths via env
# overrides (HYPERLOOM_ROOT, MAGPIE_PATH, etc.) — the defaults below take
# effect only when the corresponding env var is unset.
#
# REPO_ROOT / KERNEL_AGENT_ROOT default to the on-disk source location
# (this script lives at src/hyperloom/agents/kernel/scripts/install.sh, so
# KERNEL_AGENT_ROOT is one level up and REPO_ROOT is five levels up).
# Operator-provided read-only inputs
# (TRACELENS_ROOT, TRACELENS_INTERNAL_ROOT, HYPERLOOM_BUNDLE)
# may stay outside USER_DATA_PATH.
# The default public TraceLens checkout is cloned under USER_DATA_PATH/runtime
# like Magpie/InferenceX so its env path is safe across pods.
#
# Removed envs: WORKSPACE_PATH / WORKSPACE_ROOT (collapsed into the
# USER_DATA_PATH-rooted defaults). If your launcher exported these,
# either rename to USER_DATA_PATH or simply drop them.
KERNEL_AGENT_ROOT="${KERNEL_AGENT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../../.." && pwd)}"
HYPERLOOM_KERNEL_AGENT_ROOT="${HYPERLOOM_KERNEL_AGENT_ROOT:-${KERNEL_AGENT_ROOT}}"
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
HYPERLOOM_BUNDLE="${HYPERLOOM_BUNDLE:-/wekafs/hyperloom}"
# MAGPIE_PATH is the single override shared with the Python runtime; a standalone
# run never clones upstream Magpie when MAGPIE_PATH is set.
MAGPIE_PATH="${MAGPIE_PATH:-${_open_source_root}/Magpie}"
# Resolve MAGPIE_PYTHON dynamically. The previous default
# ${MAGPIE_PATH}/venv/bin/python assumed a Magpie-private venv, but
# src/hyperloom/inference_optimizer/assets/install.sh's ensure_magpie() does
# `pip install -e $MAGPIE_PATH` into the driver Python's site-packages
# (or the container image pre-installs it that way) — no venv is ever
# created at $MAGPIE_PATH/venv. Mirrors _resolve_magpie_python() in
# src/hyperloom/orchestrator/actions/executors/_grid_runner.py:
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
PYTHONPATH="${MAGPIE_PATH}:${PYTHONPATH:-}"
INFERENCEX_PATH="${INFERENCEX_PATH:-}"
# TraceLens base repo is required; the internal extension is OPTIONAL.
#   1. AMD-AGI/TraceLens          -> $TRACELENS_ROOT  (base: skills, patches, CLI, analysis orchestrator)
#   2. AMD-AGI/TraceLens-internal -> $TRACELENS_INTERNAL_ROOT (internal: rehydration module)
# Default base clones the public repo into the workspace runtime tree,
# matching Magpie / InferenceX rather than persisting pod-local mirrors.
# The internal extension is used ONLY when $TRACELENS_INTERNAL_ROOT is set
# (env / .env); leave it unset for the base-only report. No separate toggle.
TRACELENS_REPO="https://github.com/AMD-AGI/TraceLens.git"
# TraceLens v0.8.0 integration (#474): head of
# release/hyperloom_integration_v0.8.0. The optional internal extension tracks
# the matching release/hyperloom_integration_v0.8.0 branch of
# AMD-AGI/TraceLens-internal, but Hyperloom keeps no pin/URL for it — the
# operator supplies it via TRACELENS_INTERNAL_ROOT.
TRACELENS_REF="48f7cf6d1cc7c6d3e0aaee06c9689639021d11e3"
# Operator override iff TRACELENS_ROOT points OUTSIDE the pod-local default.
# The persistent kernel-agent env re-exports the resolved default path, so a
# presence-only check (${VAR:+1}) would misclassify it as an override and skip
# the managed clone/realign, leaving a missing/stale default un-self-healed
# across reruns (#722). Compare against the default path, not env presence.
# Canonicalize a path (resolve symlinks/.. , strip trailing slash) so the
# default-vs-override comparison matches the Python side's Path.resolve();
# unresolvable paths fall back to the trimmed literal so a not-yet-cloned
# default still compares correctly (#722 / PR#789).
# Keep in lockstep with the twin helper in
# src/hyperloom/inference_optimizer/assets/local_setup.sh.
_canonicalize_path() {
  local p="${1:-}"
  [ -z "$p" ] && return 0
  readlink -f -- "$p" 2>/dev/null || printf '%s' "${p%/}"
}
_tracelens_default_root="$(_canonicalize_path "${_open_source_root}/TraceLens")"
_tracelens_root_was_set=""
if [ -n "${TRACELENS_ROOT:-}" ] \
   && [ "$(_canonicalize_path "${TRACELENS_ROOT}")" != "${_tracelens_default_root}" ]; then
  _tracelens_root_was_set=1
fi
TRACELENS_ROOT="${TRACELENS_ROOT:-${_open_source_root}/TraceLens}"
TRACELENS_INTERNAL_ROOT="${TRACELENS_INTERNAL_ROOT:-}"
# Writable mirror when the optional internal extension is on a read-only mount.
TRACELENS_MIRROR_DIR="${TRACELENS_MIRROR_DIR:-${_open_source_root}/TraceLens-internal}"

# Credentials fallback: env always wins. If any supported LLM credential is
# missing from env, source $REPO_ROOT/.env but protect already-set values.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
if [ -z "${SAFE_API_KEY:-}" ] || [ -z "${OPENAI_BASE_URL:-}" ] \
   || [ -z "${OPENAI_API_KEY:-}" ] || [ -z "${ANTHROPIC_BASE_URL:-}" ] \
   || [ -z "${ANTHROPIC_API_KEY:-}" ] || [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ]; then
  if [ -f "$REPO_ROOT/.env" ]; then
    _snap_safe="${SAFE_API_KEY-}"
    _snap_url="${OPENAI_BASE_URL-}"
    _snap_openai_key="${OPENAI_API_KEY-}"
    _snap_anthropic_url="${ANTHROPIC_BASE_URL-}"
    _snap_anthropic_key="${ANTHROPIC_API_KEY-}"
    _snap_anthropic_token="${ANTHROPIC_AUTH_TOKEN-}"
    set -a
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.env"
    set +a
    [ -n "$_snap_safe" ] && export SAFE_API_KEY="$_snap_safe"
    [ -n "$_snap_url" ]  && export OPENAI_BASE_URL="$_snap_url"
    [ -n "$_snap_openai_key" ] && export OPENAI_API_KEY="$_snap_openai_key"
    [ -n "$_snap_anthropic_url" ] && export ANTHROPIC_BASE_URL="$_snap_anthropic_url"
    [ -n "$_snap_anthropic_key" ] && export ANTHROPIC_API_KEY="$_snap_anthropic_key"
    [ -n "$_snap_anthropic_token" ] && export ANTHROPIC_AUTH_TOKEN="$_snap_anthropic_token"
    unset _snap_safe _snap_url _snap_openai_key _snap_anthropic_url
    unset _snap_anthropic_key _snap_anthropic_token
    echo "[kernel-agent] loaded credentials fallback from $REPO_ROOT/.env (env wins)"
  fi
fi
# e2e whole-pipeline optimizer — Hyperloom calls it simply "geak" (formerly the
# standalone PerfSkills repo / GEAK_v4). Its code lives IN GEAK (interface/run_e2e.py
# + e2e_workflow/), tracked on the ``main`` branch. Hyperloom calls
# interface/run_e2e.py at the KERNEL_AGENT phase when
# KERNEL_OPT_BACKEND_ORDER=geak. It owns the GEAK_* handle; operators override
# repo/ref/root with GEAK_REPO / GEAK_REF / GEAK_ROOT. NOTE: only Hyperloom's
# internal naming changed — no upstream GEAK branch was renamed.
GEAK_REPO="${GEAK_REPO:-https://github.com/AMD-AGI/GEAK.git}"
GEAK_ROOT="${GEAK_ROOT:-${_open_source_root}/GEAK}"
GEAK_REF="${GEAK_REF:-main}"
GEAK_E2E_RUNNER="${GEAK_E2E_RUNNER:-${GEAK_ROOT}/interface/run_e2e.py}"
# Run mode for the e2e optimizer budget. ``full`` (default) selects the 3 h
# preset; ``quick`` selects the 70 min smoke preset. The inference_optimizer
# kernel request handler reads $GEAK_RUN_MODE to pick the backend budget
# default, so it is forwarded through the Ray env allowlist.
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
# Split-provider aware per-side credentials. Anthropic side keeps its own
# base URL/key; OpenAI side keeps its own. Single-gateway deploys still fall
# back to SAFE_API_KEY + OPENAI_BASE_URL so behavior is unchanged.
_ANTHROPIC_BASE_URL_VAL="${ANTHROPIC_BASE_URL:-}"
_ANTHROPIC_KEY_VAL="${ANTHROPIC_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-${SAFE_API_KEY:-}}}"
_OPENAI_BASE_URL_VAL="${OPENAI_BASE_URL:-}"
_OPENAI_KEY_VAL="${OPENAI_API_KEY:-${SAFE_API_KEY:-}}"
# Pick a key that matches a given endpoint: OpenAI URL -> OpenAI key,
# Anthropic URL -> Anthropic key, otherwise the single-gateway SAFE fallback.
_key_for_endpoint() {
  local url="$1"
  if [ -n "$_OPENAI_BASE_URL_VAL" ] && [ "$url" = "$_OPENAI_BASE_URL_VAL" ]; then
    printf '%s' "$_OPENAI_KEY_VAL"; return 0
  fi
  if [ -n "$_ANTHROPIC_BASE_URL_VAL" ] && [ "$url" = "$_ANTHROPIC_BASE_URL_VAL" ]; then
    printf '%s' "$_ANTHROPIC_KEY_VAL"; return 0
  fi
  printf '%s' "${SAFE_API_KEY:-${OPENAI_API_KEY:-${ANTHROPIC_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-}}}}"
}
# GEAK uses the user's LiteLLM-compatible endpoint. The canonical env is
# OPENAI_BASE_URL + SAFE_API_KEY; keep fallbacks for older launchers.
GEAK_BASE_URL_VAL="${GEAK_BASE_URL:-${OPENAI_BASE_URL:-${ANTHROPIC_BASE_URL:-${LLM_API_BASE:-}}}}"
# Pair the GEAK key to its endpoint so a split deploy never sends the wrong
# provider's key. Explicit GEAK_API_KEY still wins.
GEAK_API_KEY_VAL="${GEAK_API_KEY:-$(_key_for_endpoint "$GEAK_BASE_URL_VAL")}"
[ -n "$GEAK_API_KEY_VAL" ] || GEAK_API_KEY_VAL="${SAFE_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-${ANTHROPIC_API_KEY:-${AMD_API_KEY:-${AMD_LLM_API_KEY:-${LLM_API_KEY:-${OPENAI_API_KEY:-}}}}}}}"
# install.sh always installs everything. A previous lazy
# "install only the requested backend" scheme caused recurring
# "missing dependency discovered at request time" issues, so the
# installer brings up every kernel-agent backend in one pass.
CHECK_ONLY=0
DRY_RUN=0
# Optional build-time escape hatch: skip Ray daemon startup while still
# installing Ray itself and all downstream dependencies (TraceLens/GEAK).
# Useful in Docker image builds where launching background daemons is fragile.
case "${SKIP_RAY_START:-0}" in
  1|true|TRUE|yes|YES|on|ON) SKIP_RAY_START=1 ;;
  *) SKIP_RAY_START=0 ;;
esac
usage() {
  cat <<'EOF'
Usage: install.sh [options]

Always installs:
  ray[default]==2.44.1, click<8.3.0, TraceLens CLI,
  the GEAK e2e whole-pipeline optimizer, and LLM gateway env/auth.

Options:
  --check-only       Verify current environment, do not install
  --dry-run          Print actions without running installs
  -h, --help         Show this help

Environment (optional):
  SKIP_RAY_START=1                      Skip `ray start --head` during install (default 0).
                                        Installs ray/click but defers daemon startup to runtime.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
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

# Preflight credential validation. A usable setup needs at least one LLM base
# URL and at least one key; single-gateway and split OpenAI/Anthropic are valid.
#
# Strict mode by design: no bypass env var. The chained installer
# steps (the GEAK e2e optimizer) all need real credentials, so an
# install without them cannot finish anyway. The only downgrade path
# is --check-only / --dry-run, which is for introspection only and
# does not actually install.
preflight_validate_credentials() {
  local missing=()
  local has_url=0 has_key=0
  { [ -n "${OPENAI_BASE_URL:-}" ] || [ -n "${ANTHROPIC_BASE_URL:-}" ]; } && has_url=1
  { [ -n "${SAFE_API_KEY:-}" ] || [ -n "${OPENAI_API_KEY:-}" ] \
    || [ -n "${ANTHROPIC_API_KEY:-}" ] || [ -n "${ANTHROPIC_AUTH_TOKEN:-}" ]; } && has_key=1
  [ "$has_url" -eq 0 ] && missing+=("OPENAI_BASE_URL or ANTHROPIC_BASE_URL")
  [ "$has_key" -eq 0 ] && missing+=("SAFE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, or ANTHROPIC_AUTH_TOKEN")
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
         "continuing because --check-only / --dry-run is active. The GEAK " \
         "e2e optimizer will still fail later unless these are set " \
         "before a real install."
    return 0
  fi
  cat >&2 <<EOF
[kernel-agent ERROR] Missing required credential group(s): ${missing[*]}

Tried loading from:
  - shell environment
  - \$REPO_ROOT/.env  (${env_file_status}: ${REPO_ROOT}/.env)

Fix one of:
  1. Single gateway:
       export SAFE_API_KEY=ak-your-safe-apikey
       export OPENAI_BASE_URL=https://gateway.example.com/v1
  2. Split OpenAI or Anthropic:
       export OPENAI_BASE_URL=https://api.openai.com/v1
       export OPENAI_API_KEY=sk-...
       # or
       export ANTHROPIC_BASE_URL=https://api.anthropic.com
       export ANTHROPIC_API_KEY=sk-ant-...
  3. Copy .env from a working worktree into this one:
       cp /path/to/main-worktree/.env "${REPO_ROOT}/.env"
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

# Serialize concurrent installs that share one $USER_DATA_PATH. Installs
# pointed at the same data root share the auto-cloned dependency checkouts —
# GEAK / TraceLens trees below. With no lock, two installs race and
# corrupt each other's half-cloned trees. The lock lives in $_open_source_root
# (pod-local) so it tracks exactly what it guards: same-root installs serialize,
# but separate pod-local roots never block each other. We hold an flock on
# $_open_source_root/.install.lock via fd 9 from the first mirror-mutating step
# until this process exits (fd closes on exit), so it guards every clone/mirror
# below and releases automatically at the end.
# Skipped under --check-only / --dry-run (introspection only, no mutation).
# When inference_optimizer's installer already holds the lock it exports
# HYPERLOOM_INSTALL_LOCK_HELD=1; we honour that and do not re-acquire, which
# would deadlock on a second open file description for the same path.
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

ensure_python() {
  python3 --version >/dev/null || die "python3 is required"
  python3 -m pip --version >/dev/null || die "pip is required"
}

# PR-D §3: pin `git` and `patch` so the TraceLens server patcher has the
# binaries it expects on every deployment.
#
# Background: `src/hyperloom/orchestrator/actions/executors/_server_patcher.py`
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
# Apt-installing here is the cheap, framework-agnostic safety net for the
# TraceLens server patcher, so it carries no new failure modes.
ensure_patch_tools() {
  log "ensuring git + patch (required by src/hyperloom/inference_optimizer/_server_patcher fuzzy-fallback path)"
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

# Minimum soft `ulimit -n` the Ray raylet needs to stay up (issue #433).
# The raylet opens a large number of fds (sockets, plasma store, per-worker
# pipes); at the container default soft limit (1024) it aborts on startup /
# lingers as a zombie that only `ray stop --force` clears. Operators can
# override via RAY_MIN_NOFILE.
RAY_MIN_NOFILE="${RAY_MIN_NOFILE:-65536}"

# Raise this shell's soft open-files limit before `ray start` so the raylet
# child inherits a high enough ceiling (issue #433). Raising the soft limit
# up to the hard cap needs no privileges; only `docker run --ulimit
# nofile=...` at container launch can lift the hard cap, so when the hard cap
# is itself below the target we raise soft as high as allowed and warn.
ensure_fd_limit_for_ray() {
  local soft hard target
  soft="$(ulimit -Sn 2>/dev/null || echo unknown)"
  hard="$(ulimit -Hn 2>/dev/null || echo unknown)"
  case "$soft" in
    ''|*[!0-9]*)
      log "fd-limit: soft nofile unknown ('$soft'); skipping raylet fd preflight (issue #433)"
      return 0
      ;;
  esac
  if [ "$soft" -ge "$RAY_MIN_NOFILE" ]; then
    log "fd-limit: soft nofile=$soft already >= $RAY_MIN_NOFILE"
    return 0
  fi
  target="$RAY_MIN_NOFILE"
  case "$hard" in
    unlimited|''|*[!0-9]*) : ;;
    *) [ "$hard" -lt "$RAY_MIN_NOFILE" ] && target="$hard" ;;
  esac
  if ulimit -Sn "$target" 2>/dev/null; then
    log "fd-limit: raised soft nofile $soft -> $(ulimit -Sn) before 'ray start' (issue #433)"
  else
    warn "fd-limit: could not raise soft nofile to $target (soft=$soft hard=$hard); Ray raylet may be unstable. Launch container with --ulimit nofile=1048576 (issue #433)."
  fi
  case "$hard" in
    unlimited|''|*[!0-9]*) : ;;
    *)
      [ "$hard" -lt "$RAY_MIN_NOFILE" ] && \
        warn "fd-limit: hard nofile cap=$hard < $RAY_MIN_NOFILE; only 'docker run --ulimit nofile=1048576' lifts the hard cap (issue #433)."
      ;;
  esac
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
  if [ "$SKIP_RAY_START" -eq 1 ]; then
    log "skipping ray head startup (SKIP_RAY_START=1)"
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
  # issue #433: raise fd limit BEFORE starting the head so the raylet inherits
  # a ceiling high enough to stay up (container default 1024 makes it abort).
  ensure_fd_limit_for_ray
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

_pip_install_editable() {
  local root="$1"
  local label="$2"
  if [ ! -d "$root" ]; then
    if [ "$DRY_RUN" -eq 1 ] || [ "$CHECK_ONLY" -eq 1 ]; then
      warn "${label} checkout not found: ${root}"
      return 1
    fi
    die "${label} checkout not found: ${root}"
  fi
  if [ "$CHECK_ONLY" -eq 0 ]; then
    local project_name
    project_name="$(_project_name_from_pyproject "$root")"
    if [ -n "$project_name" ] && _local_install_matches_root "$project_name" "$root"; then
      log "${label} already installed from ${root}; skipping reinstall"
      return 0
    fi
  fi
  log "ensuring ${label} editable install from ${root}"
  if [ "$CHECK_ONLY" -eq 0 ]; then
    # Do not use bash -lc: login profiles reset PATH (drops venv) and break pip.
    run sh -c "cd '$root' && python3 -m pip install -q --no-cache-dir --break-system-packages -e ."
  fi
  return 0
}

_project_name_from_pyproject() {
  local root="$1"
  local pyproject="${root%/}/pyproject.toml"
  [ -f "$pyproject" ] || return 0
  python3 - "$pyproject" <<'PY' 2>/dev/null || true
import sys
path = sys.argv[1]
try:
    import tomllib
except Exception:
    import tomli as tomllib  # py<3.11 fallback
with open(path, "rb") as f:
    data = tomllib.load(f)
name = (((data.get("project") or {}).get("name")) or "").strip()
print(name)
PY
}

_local_install_matches_root() {
  local project_name="$1"
  local root="$2"
  [ -n "$project_name" ] || return 1
  python3 - "$project_name" "$root" <<'PY' >/dev/null 2>&1
import importlib.metadata
import json
import os
import sys
from urllib.parse import urlparse, unquote

project = sys.argv[1]
root = os.path.realpath(sys.argv[2])
try:
    dist = importlib.metadata.distribution(project)
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(1)
direct = dist.read_text("direct_url.json")
if not direct:
    raise SystemExit(1)
try:
    payload = json.loads(direct)
except Exception:
    raise SystemExit(1)
url = payload.get("url") or ""
if not url.startswith("file:"):
    raise SystemExit(1)
parsed = urlparse(url)
installed_root = os.path.realpath(unquote(parsed.path))
if os.path.normcase(installed_root) != os.path.normcase(root):
    raise SystemExit(1)
raise SystemExit(0)
PY
}

# Internal extension is opt-in: enabled iff $TRACELENS_INTERNAL_ROOT is set.
_tracelens_internal_enabled() {
  [ -n "${TRACELENS_INTERNAL_ROOT:-}" ]
}

ensure_tracelens() {
  if [ -n "${_tracelens_root_was_set:-}" ]; then
    # Operator override: never auto-clone, but still fail fast when the checkout
    # is missing OR an incomplete non-git tree (a half-done clone lacking .git),
    # mirroring the handler/tool completeness check (#722 / PR#789).
    if [ ! -d "$TRACELENS_ROOT/.git" ]; then
      if [ "$DRY_RUN" -eq 1 ] || [ "$CHECK_ONLY" -eq 1 ]; then
        warn "TraceLens root not found or not a git checkout: $TRACELENS_ROOT"
      else
        die "TraceLens root not found or not a git checkout: $TRACELENS_ROOT"
      fi
    fi
  elif [ ! -d "$TRACELENS_ROOT/.git" ]; then
    if [ "$CHECK_ONLY" -eq 1 ]; then
      warn "TraceLens checkout missing/incomplete at ${TRACELENS_ROOT} (check-only mode, skipping clone)"
    elif [ "$DRY_RUN" -eq 1 ]; then
      [ -e "$TRACELENS_ROOT" ] && log "would: rm -rf ${TRACELENS_ROOT} (incomplete, not a git repo)"
      log "would: git clone --depth 1 ${TRACELENS_REPO} ${TRACELENS_ROOT}"
    else
      # An existing dir without .git is a half-done/incomplete clone (e.g. a
      # crashed installer). On this installer-managed default path, drop it and
      # rebuild so it never lingers as an unusable tree that trace_analyze's
      # self-heal would treat as complete (#722/PR#789) — matches
      # _ensure_tracelens_checkout, which moves aside and re-clones.
      if [ -e "$TRACELENS_ROOT" ]; then
        warn "TraceLens checkout at ${TRACELENS_ROOT} is not a git repo; rebuilding"
        rm -rf "$TRACELENS_ROOT"
      fi
      # Clone AND pin the ref inside a temp sibling, then atomically rename into
      # place only after everything succeeds. Publishing before the ref pin (or
      # on a mid-clone crash) would leave an unpinned/half-cloned $TRACELENS_ROOT
      # that a concurrent reader (trace_analyze self-heal) treats as complete (#722).
      # Keep this temp-clone+pin+atomic-rename in lockstep with the twin
      # implementations: src/hyperloom/inference_optimizer/assets/local_setup.sh
      # (clone_or_update "atomic") and src/hyperloom/agents/kernel/tools/tracelens_analysis.py
      # (_ensure_tracelens_checkout).
      mkdir -p "$(dirname "$TRACELENS_ROOT")"
      _tl_tmp="$(dirname "$TRACELENS_ROOT")/.$(basename "$TRACELENS_ROOT").clone.$$"
      rm -rf "$_tl_tmp"
      if ! git clone --depth 1 "$TRACELENS_REPO" "$_tl_tmp" \
        || ! git -C "$_tl_tmp" fetch --depth 1 origin "$TRACELENS_REF" \
        || ! git -C "$_tl_tmp" checkout -q FETCH_HEAD; then
        rm -rf "$_tl_tmp"
        die "TraceLens clone/pin to ${TRACELENS_REF} failed; refusing to publish an unpinned checkout at ${TRACELENS_ROOT}"
      fi
      mv "$_tl_tmp" "$TRACELENS_ROOT"
    fi
  elif [ -z "${_tracelens_root_was_set:-}" ] && [ -d "$TRACELENS_ROOT/.git" ]; then
    # Existing default checkout: realign to the pinned ref in place.
    if [ "$CHECK_ONLY" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
      run git -C "$TRACELENS_ROOT" fetch --depth 1 origin "$TRACELENS_REF"
      run git -C "$TRACELENS_ROOT" checkout -q FETCH_HEAD
    elif [ "$DRY_RUN" -eq 1 ]; then
      log "would: git -C ${TRACELENS_ROOT} fetch --depth 1 origin ${TRACELENS_REF}"
      log "would: git -C ${TRACELENS_ROOT} checkout -q FETCH_HEAD"
    fi
  fi
  # Internal extension is opt-in via TRACELENS_INTERNAL_ROOT only; no implicit
  # bundle/default path is probed (keeps internal location out of this repo).

  if [ ! -d "$TRACELENS_ROOT" ]; then
    if [ "$DRY_RUN" -eq 1 ] || [ "$CHECK_ONLY" -eq 1 ]; then
      warn "TraceLens root not found: $TRACELENS_ROOT"
    else
      die "TraceLens root not found: $TRACELENS_ROOT"
    fi
  fi
  _pip_install_editable "$TRACELENS_ROOT" "TraceLens (public)" || {
    [ "$DRY_RUN" -eq 1 ] || [ "$CHECK_ONLY" -eq 1 ] || die "install AMD-AGI/TraceLens at TRACELENS_ROOT=${TRACELENS_ROOT}"
  }

  if ! _tracelens_internal_enabled; then
    log "TraceLens-internal: not provided (open-source-only; set TRACELENS_INTERNAL_ROOT to enable)"
    TRACELENS_INTERNAL_ROOT=""
    export TRACELENS_ROOT
    return 0
  fi

  if [ ! -d "$TRACELENS_INTERNAL_ROOT" ]; then
    warn "TRACELENS_INTERNAL_ROOT set but not found: $TRACELENS_INTERNAL_ROOT; falling back to open-source-only (provide an existing internal checkout to enable)"
    TRACELENS_INTERNAL_ROOT=""
    export TRACELENS_ROOT
    return 0
  fi
  # Read-only source guard. When
  # $TRACELENS_INTERNAL_ROOT is on a read-only mount (the WekaFS default), pip
  # install -e fails because it must write *.egg-info into the source
  # tree, and at runtime tools/tracelens_analysis.py re-runs the same
  # editable install in a subprocess on every trace_analyze request,
  # producing a tight failure loop. Detecting unwritable source up front
  # and mirroring to ${HYPERLOOM_ROOT}/TraceLens-internal lets both
  # the install-time and the runtime pip install land on a writable
  # filesystem. write_env_file() emits the resulting TRACELENS_INTERNAL_ROOT into
  # the pod-local kernel-agent env so subsequent CLI subprocesses inherit
  # the mirror.
  if [ "$CHECK_ONLY" -eq 0 ] && [ "$DRY_RUN" -eq 0 ]; then
    if ! ( : > "$TRACELENS_INTERNAL_ROOT/.hl_write_test" ) 2>/dev/null; then
      log "TraceLens-internal root not writable ($TRACELENS_INTERNAL_ROOT); mirroring to $TRACELENS_MIRROR_DIR"
      mkdir -p "$(dirname "$TRACELENS_MIRROR_DIR")"
      if [ ! -d "$TRACELENS_MIRROR_DIR" ]; then
        log "mirroring TraceLens-internal to writable dir (large tree; may take minutes): $TRACELENS_INTERNAL_ROOT -> $TRACELENS_MIRROR_DIR"
        run cp -r "$TRACELENS_INTERNAL_ROOT" "$TRACELENS_MIRROR_DIR"
      else
        log "TraceLens-internal mirror already present: $TRACELENS_MIRROR_DIR"
      fi
      TRACELENS_INTERNAL_ROOT="$TRACELENS_MIRROR_DIR"
      export TRACELENS_INTERNAL_ROOT
    else
      rm -f "$TRACELENS_INTERNAL_ROOT/.hl_write_test"
    fi
  fi
  _pip_install_editable "$TRACELENS_INTERNAL_ROOT" "TraceLens-internal"
  export TRACELENS_ROOT TRACELENS_INTERNAL_ROOT
  if [ "$DRY_RUN" -eq 0 ]; then
    # TraceLens #124: only the inference variant is accepted (the correct
    # entry for vLLM/SGLang traces). Hyperloom is inference-only since
    # v0.4; the legacy training-mode CLI was removed to keep install /
    # runtime in lockstep.
    if command -v TraceLens_generate_perf_report_pytorch_inference >/dev/null 2>&1; then
      TraceLens_generate_perf_report_pytorch_inference --help >/dev/null
      log "TraceLens perf CLI verified: TraceLens_generate_perf_report_pytorch_inference (#124)"
    else
      verify_die "TraceLens_generate_perf_report_pytorch_inference not found after install (Hyperloom is inference-only since v0.4; reinstall TraceLens, plus TraceLens-internal if TRACELENS_INTERNAL_ROOT is set)"
    fi
  fi
}

# Write a pod-local kernel-agent env file users should source so subsequent CLI calls
# (and Ray workers via runtime_env) pick up the upstream gateway URLs.
write_env_file() {
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  local _openai_url="${_OPENAI_BASE_URL_VAL:-${_ANTHROPIC_BASE_URL_VAL:-${LLM_API_BASE:-${GEAK_BASE_URL_VAL:-}}}}"
  local _gateway_key="${_OPENAI_KEY_VAL:-${_ANTHROPIC_KEY_VAL:-${GEAK_API_KEY_VAL:-${LLM_GATEWAY_KEY:-${LLM_API_KEY:-}}}}}"
  # Warn loudly if no gateway URL is resolved — kernel-agent env would silently
  # lack ANTHROPIC_BASE_URL/OPENAI_BASE_URL and CLIs would resort to whatever
  # was in the operator's shell rc, defeating the point of this file.
  if [ -z "${_openai_url:-}" ]; then
    warn "LLM gateway URL empty; kernel-agent env will lack ANTHROPIC_BASE_URL/OPENAI_BASE_URL"
  fi
  # Anthropic base URL: keep an explicit split-provider value as-is; only
  # derive from the OpenAI-style upstream (strip trailing /v1, the Anthropic
  # SDK re-appends it) when no Anthropic endpoint was configured.
  local _anthropic_url=""
  if [ -n "${_ANTHROPIC_BASE_URL_VAL:-}" ]; then
    _anthropic_url="${_ANTHROPIC_BASE_URL_VAL}"
  elif [ -n "${_openai_url:-}" ]; then
    _anthropic_url="${_openai_url%/}"
    _anthropic_url="${_anthropic_url%/v1}"
  fi
  # Per-side keys: a split deploy writes each provider its own key; a single
  # gateway resolves both to the same key.
  local _anthropic_key="${_ANTHROPIC_KEY_VAL:-${_gateway_key}}"
  local _openai_key="${_OPENAI_KEY_VAL:-${_gateway_key}}"
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
    [ -n "${MAGPIE_PATH:-}" ] && echo "export MAGPIE_PATH='${MAGPIE_PATH}'"
    [ -n "${MAGPIE_PYTHON:-}" ] && echo "export MAGPIE_PYTHON='${MAGPIE_PYTHON}'"
    [ -n "${PYTHONPATH:-}" ] && echo "export PYTHONPATH='${PYTHONPATH}'"
    [ -n "${INFERENCEX_PATH:-}" ] && echo "export INFERENCEX_PATH='${INFERENCEX_PATH}'"
    [ -n "${_anthropic_url}" ] && echo "export ANTHROPIC_BASE_URL='${_anthropic_url}'"
    [ -n "${_openai_url:-}" ] && echo "export OPENAI_BASE_URL='${_openai_url}'"
    # Anthropic-side and OpenAI-side keys are written independently so a split
    # deploy (e.g. DeepSeek Anthropic API + native OpenAI) does not cross-send
    # the wrong provider's key. Gateway aliases keep the single-gateway key.
    [ -n "${_anthropic_key}" ] && {
      echo "export ANTHROPIC_API_KEY='${_anthropic_key}'"
      echo "export ANTHROPIC_AUTH_TOKEN='${_anthropic_key}'"
    }
    [ -n "${_openai_key}" ] && echo "export OPENAI_API_KEY='${_openai_key}'"
    [ -n "${_gateway_key}" ] && {
      echo "export SAFE_API_KEY='${_gateway_key}'"
      echo "export AMD_LLM_API_KEY='${_gateway_key}'"
      echo "export LLM_GATEWAY_KEY='${_gateway_key}'"
    }
    # Pin TRACELENS_ROOT and TRACELENS_INTERNAL_ROOT to the (possibly
    # mirrored) values resolved by ensure_tracelens(). This is what lets
    # setsid nohup inference_optimizer optimize →
    # src/hyperloom/agents/kernel/tools/tracelens_analysis.py inherit the writable
    # mirrors instead of falling back to the read-only /wekafs defaults.
    [ -n "${TRACELENS_ROOT:-}" ] && echo "export TRACELENS_ROOT='${TRACELENS_ROOT}'"
    if [ -n "${TRACELENS_INTERNAL_ROOT:-}" ]; then
      echo "export TRACELENS_INTERNAL_ROOT='${TRACELENS_INTERNAL_ROOT}'"
      echo "export TL_EXTENSION='TraceLens_internal'"
    fi
    [ -n "${HYPERLOOM_ROOT:-}" ] && echo "export HYPERLOOM_ROOT='${HYPERLOOM_ROOT}'"
    # e2e optimizer ("geak") checkout + runner (GEAK_ROOT / GEAK_E2E_RUNNER),
    # consumed by src/hyperloom/agents/kernel/tools/backends/geak_runner.py.
    [ -n "${GEAK_E2E_RUNNER}" ] && echo "export GEAK_E2E_RUNNER='${GEAK_E2E_RUNNER}'"
    [ -n "${GEAK_ROOT}" ] && echo "export GEAK_ROOT='${GEAK_ROOT}'"
    # Pin the claude binary the GEAK SDK path uses (else claude_agent_sdk may
    # fall back to its older bundled CLI). run_e2e.py maps this to cli_path.
    _geak_claude_bin=""
    for _c in "${HOME}/.local/bin/claude" "/usr/local/bin/claude" "$(command -v claude 2>/dev/null || true)"; do
      if [ -n "${_c}" ] && [ -x "${_c}" ]; then _geak_claude_bin="${_c}"; break; fi
    done
    [ -n "${_geak_claude_bin}" ] && echo "export GEAK_CLAUDE_BIN='${_geak_claude_bin}'"
    # e2e optimizer budget mode (read by the inference_optimizer kernel request
    # handler to pick the backend budget default) + LLM connection for the runner.
    [ -n "${GEAK_RUN_MODE_VAL}" ] && echo "export GEAK_RUN_MODE='${GEAK_RUN_MODE_VAL}'"
    [ -n "${GEAK_API_KEY_VAL}" ] && echo "export GEAK_API_KEY='${GEAK_API_KEY_VAL}'"
    [ -n "${GEAK_BASE_URL_VAL}" ] && echo "export GEAK_BASE_URL='${GEAK_BASE_URL_VAL}'"
    # GEAK scoring / profiler / shape knobs. These are read by GEAK itself (the
    # Ray actor), but the optimize CLI sources THIS file and its env replaces the
    # launcher's exports -- so any knob not persisted here is silently dropped
    # before reaching the actor (e.g. GEAK_SCORE_TARGET=kernel fell back to wall).
    # Passthrough-if-set (no hardcoded defaults):
    #   GEAK_SCORE_TARGET         -> score best-patch on kernel_ms vs wall (E2E-transferable)
    #   GEAK_SKIP_PROFILE         -> skip the advisory profiler-mcp roofline pass
    #   GEAK_MAX_BENCHMARK_SHAPES -> harness benchmark-shape cap
    [ -n "${GEAK_SCORE_TARGET:-}" ] && echo "export GEAK_SCORE_TARGET='${GEAK_SCORE_TARGET}'"
    [ -n "${GEAK_SKIP_PROFILE:-}" ] && echo "export GEAK_SKIP_PROFILE='${GEAK_SKIP_PROFILE}'"
    [ -n "${GEAK_MAX_BENCHMARK_SHAPES:-}" ] && echo "export GEAK_MAX_BENCHMARK_SHAPES='${GEAK_MAX_BENCHMARK_SHAPES}'"
  } > "$env_file"
  chmod 600 "$env_file"
  log "wrote ${env_file} (source it before running kernel-agent tools)"
}

# Clone the e2e optimizer ("geak"; GEAK@GEAK_v4, formerly PerfSkills): SHA pins
# use a shallow fetch-checkout; tags/branches use git clone --branch. Not
# pip-installed (it's a JS workflow dir); after the checkout we run GEAK's
# setup.sh (Claude Code CLI + py deps) plus claude_agent_sdk.
ensure_geak() {
  log "ensuring e2e optimizer geak (GEAK@${GEAK_REF}, formerly PerfSkills)"
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    mkdir -p "${GEAK_ROOT}"
  fi
  if [ ! -d "${GEAK_ROOT}/.git" ]; then
    if [[ "$GEAK_REF" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
      run git init -q "${GEAK_ROOT}"
      run git -C "${GEAK_ROOT}" remote add origin "$GEAK_REPO"
      run git -C "${GEAK_ROOT}" fetch --depth 1 origin "$GEAK_REF"
      run git -C "${GEAK_ROOT}" checkout -q FETCH_HEAD
    else
      run git clone --depth 1 --branch "$GEAK_REF" "$GEAK_REPO" "${GEAK_ROOT}"
    fi
  else
    log "e2e optimizer checkout already present: ${GEAK_ROOT}"
    if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
      # Keep an existing checkout aligned with the requested GEAK_REF:
      # without this a ref bump (branch/tag/SHA) leaves the runtime pinned
      # to the stale e2e code it first cloned.
      run git -C "${GEAK_ROOT}" fetch --depth 1 origin "$GEAK_REF"
      run git -C "${GEAK_ROOT}" checkout -q --force FETCH_HEAD
    fi
  fi
  if [ "$CHECK_ONLY" -eq 0 ]; then
    # GEAK's own installer: upgrades the Claude Code CLI to the dynamic Workflow
    # floor (>= 2.1.177) and installs its py deps. Idempotent.
    # `env -u REPO_ROOT`: we export REPO_ROOT (Hyperloom's checkout) for our own
    # steps, but GEAK's setup.sh keys its requirements.txt path off REPO_ROOT too
    # (${REPO_ROOT:-...}), so leaking ours makes it look for requirements.txt in
    # the Hyperloom tree and die. Strip it so setup.sh derives its own repo root.
    if [ -x "${GEAK_ROOT}/setup.sh" ]; then
      run env -u REPO_ROOT bash "${GEAK_ROOT}/setup.sh" || warn "GEAK setup.sh failed; Claude Code may be < 2.1.177"
    else
      warn "GEAK setup.sh missing at ${GEAK_ROOT}/setup.sh; using the npm claude from ensure_forge_claude_cli"
    fi
    # setup.sh installs the CLI, not the SDK; run_e2e.py prefers the SDK.
    _PIP_FLAGS="-q --no-cache-dir --break-system-packages"
    run python3 -m pip install ${_PIP_FLAGS} claude-agent-sdk anyio || \
      warn "claude-agent-sdk install failed; run_e2e.py will fall back to the claude CLI"
    if [ ! -f "${GEAK_E2E_RUNNER}" ]; then
      warn "e2e runner not found at ${GEAK_E2E_RUNNER} (interface/ missing — is the checkout on the ${GEAK_REF} branch with the e2e code?)"
    fi
  else
    log "check-only: skipping e2e optimizer sdk installation"
  fi
}

# The forge backend drives the `claude` CLI inside its autonomous loop
# (see forge_submit._apply_fellow_env), so it needs Node/npm, the claude npm
# CLI, and ~/.claude auth.
ensure_forge_claude_cli() {
  log "ensuring claude CLI for the forge backend"
  if [ "$CHECK_ONLY" -eq 1 ]; then
    command -v claude >/dev/null 2>&1 || warn "claude CLI missing; forge backend will fail to drive its fellow"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would install Node.js/npm + @anthropic-ai/claude-code and write ~/.claude/config.json"
    return 0
  fi
  # Node.js 20 from NodeSource when npm is absent (claude CLI is an npm package).
  if ! command -v npm >/dev/null 2>&1; then
    if ! command -v apt-get >/dev/null 2>&1; then
      warn "npm missing and apt-get unavailable; install Node.js 20 manually for the forge claude CLI"
      return 0
    fi
    command -v curl >/dev/null 2>&1 || { apt-get update >/dev/null; apt-get -y install ca-certificates curl gnupg >/dev/null; }
    log "installing Node.js 20 from NodeSource"
    local ns_script="/tmp/nodesource_setup_20.x"
    if curl -fsSL "https://deb.nodesource.com/setup_20.x" -o "$ns_script" \
       && echo "2c4c6683a17b6f4128898a7b521e3c8bb725a99ffaf1b5e32ac97c6fa7d381be  ${ns_script}" | sha256sum -c - >/dev/null 2>&1 \
       && bash "$ns_script" >/dev/null 2>&1; then
      apt-get -y install nodejs >/dev/null || warn "nodejs install failed; forge claude CLI unavailable"
    else
      warn "NodeSource setup failed; forge claude CLI unavailable"
      return 0
    fi
  fi
  if command -v npm >/dev/null 2>&1 && ! command -v claude >/dev/null 2>&1; then
    run npm config set prefix /usr/local
    run npm install -g @anthropic-ai/claude-code
  fi
  # ~/.claude auth from existing provider / GEAK gateway env so the fellow authenticates.
  local _claude_key="${_ANTHROPIC_KEY_VAL:-${SAFE_API_KEY:-${GEAK_API_KEY_VAL:-${_OPENAI_KEY_VAL:-}}}}"
  if [ -n "$_claude_key" ]; then
    mkdir -p /root/.claude
    local _anthropic_url="${_ANTHROPIC_BASE_URL_VAL:-${GEAK_BASE_URL_VAL:-${_OPENAI_BASE_URL_VAL:-}}}"
    _anthropic_url="${_anthropic_url%/}"
    _anthropic_url="${_anthropic_url%/v1}"
    cat > /root/.claude/config.json <<EOF
{
  "theme": "dark",
  "hasCompletedOnboarding": true,
  "primaryApiKey": "${_claude_key}",
  "customApiUrl": "${_anthropic_url}"
}
EOF
    chmod 600 /root/.claude/config.json
  else
    warn "gateway API key not set; ~/.claude/config.json not written (forge fellow auth may fail)"
  fi
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
  for tool in git patch; do
    if command -v "$tool" >/dev/null 2>&1; then
      log "found ${tool}: $(command -v "$tool")"
    else
      warn "${tool} not found (TraceLens server patcher will fail-soft without it)"
    fi
  done
  if [ -d "${GEAK_ROOT}/.git" ]; then
    log "e2e optimizer geak ref: $(git -C "${GEAK_ROOT}" describe --tags --always 2>/dev/null || echo unknown)"
  else
    warn "e2e optimizer geak checkout missing at ${GEAK_ROOT}"
  fi
  if [ -f "${GEAK_E2E_RUNNER}" ]; then
    log "e2e runner present: ${GEAK_E2E_RUNNER}"
  else
    warn "e2e runner missing at ${GEAK_E2E_RUNNER}"
  fi
}

main() {
  if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
    # KERNEL_AGENT_ROOT is now the source root (read-only checkout); tool
    # outputs land under $USER_DATA_PATH/kernel-agent/runs/<session_id>/
    # (created lazily by the tools themselves). All we need here is the
    # writable runtime tree on $USER_DATA_PATH for the env file + source mirrors.
    mkdir -p "${HYPERLOOM_RUNTIME_DIR}" "${_open_source_root}"
  fi
  ensure_python
  ensure_patch_tools
  ensure_moreutils
  ensure_ray
  ensure_ray_started
  # Hold the install lock for the whole source-mutating region (TraceLens /
  # GEAK clones + mirrors). System-package steps above (apt/pip)
  # do not touch source-mirrors, so they stay outside the lock.
  acquire_install_lock
  ensure_tracelens

  # The GEAK e2e whole-pipeline optimizer is always installed; whether it is
  # used at runtime is decided per-session via KERNEL_OPT_BACKEND_ORDER.
  ensure_geak
  ensure_forge_claude_cli
  write_env_file

  report_status
  log "install complete"
}

main "$@"
