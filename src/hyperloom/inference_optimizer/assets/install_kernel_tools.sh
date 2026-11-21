# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Kernel-tool steps of install.sh: TraceLens, GEAK, the Claude CLI and the
# kernel-agent env file. install.sh sources this file in a
# subshell after its own setup (paths, .env, credentials, install lock) and
# calls install_kernel_tools; the file has no entry point of its own.

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  echo "install_kernel_tools.sh is sourced by install.sh; run src/hyperloom/inference_optimizer/assets/install.sh" >&2
  exit 2
fi

# K8s subprocesses may inherit a minimal PATH; prepend the standard system
# bins, then the interpreter install.sh selected, so the bare python3 calls
# below install into its environment.
export PATH="$(dirname -- "$PYTHON"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin${PATH:+:$PATH}"

# Join PYTHONPATH-style entries in order, dropping empties and duplicates so
# repeated composition stays idempotent. Earlier arguments win their position.
_compose_pythonpath() {
  local out="" entry part
  for entry in "$@"; do
    [ -n "$entry" ] || continue
    # Split each argument on ':' so a passed-in PYTHONPATH is de-duplicated too.
    local _ifs="$IFS"
    IFS=':'
    for part in $entry; do
      [ -n "$part" ] || continue
      case ":${out}:" in
        *":${part}:"*) ;;
        *) out="${out:+${out}:}${part}" ;;
      esac
    done
    IFS="$_ifs"
  done
  printf '%s' "$out"
}
# Keep REPO_ROOT on PYTHONPATH so subprocesses can ``import hyperloom`` under a
# ``pip install --target $REPO_ROOT`` layout (the target dir is not on the
# default sys.path). Put REPO_ROOT first, then any pre-existing PYTHONPATH,
# including one install.sh loaded from .env.
PYTHONPATH="$(_compose_pythonpath "${REPO_ROOT:-}" "${PYTHONPATH:-}")"
INFERENCEX_PATH="${INFERENCEX_PATH:-}"
# TraceLens base repo is required; the internal extension is OPTIONAL.
#   1. AMD-AGI/TraceLens          -> $TRACELENS_ROOT  (base: skills, patches, CLI, analysis orchestrator)
#   2. TraceLens-internal -> $TRACELENS_INTERNAL_ROOT (optional rehydration module)
# Default base clones the public repo into the workspace runtime tree
# rather than persisting pod-local mirrors.
# The internal extension is used ONLY when $TRACELENS_INTERNAL_ROOT is set
# (env / .env); leave it unset for the base-only report. No separate toggle.
TRACELENS_REPO="https://github.com/AMD-AGI/TraceLens.git"
# TraceLens SHA.
TRACELENS_REF="1f6d6183852fa9321f7ab10a8587c78ca04e618b"
# Operator override iff TRACELENS_ROOT is not a TraceLens@<sha> checkout in the
# cache root. Installs re-export that path (kernel-agent env, .env), so a
# presence-only check would freeze a rerun on the old checkout instead of
# cloning/realigning to the current TRACELENS_REF (#722).
# Canonicalize a path (resolve symlinks/.. , strip trailing slash) so the
# override comparison matches the Python side's Path.resolve();
# unresolvable paths fall back to the trimmed literal so a not-yet-cloned
# default still compares correctly (#722 / PR#789).
# Keep in lockstep with Python-side path canonicalization helpers.
_canonicalize_path() {
  local p="${1:-}"
  [ -z "$p" ] && return 0
  readlink -f -- "$p" 2>/dev/null || printf '%s' "${p%/}"
}
_is_git_checkout_root() {
  local root="$1" top
  top="$(git -C "$root" rev-parse --show-toplevel 2>/dev/null || true)"
  [ -n "$top" ] && [ "$(_canonicalize_path "$top")" = "$(_canonicalize_path "$root")" ]
}
# Mirror Path.home() (posixpath.expanduser) so paths written here land where the
# Python readers look: a *present* HOME wins even when empty, else the uid's passwd entry.
_home_dir() {
  local h
  if [ -n "${HOME+x}" ]; then
    h="${HOME}"
  else
    h="$(getent passwd "$(id -u)" 2>/dev/null | cut -d: -f6 || true)"
    # Nothing left to resolve; Path.home() raises on this same input.
    [ -n "$h" ] || return 1
  fi
  while [ -n "$h" ] && [ "${h%/}" != "$h" ]; do h="${h%/}"; done
  printf '%s' "${h:-/}"
}
_TRACELENS_SHA="$(_resolve_ref_sha "$TRACELENS_REPO" "$TRACELENS_REF")"
_tracelens_root_was_set=""
if [ -n "${TRACELENS_ROOT:-}" ]; then
  _tracelens_root_canonical="$(_canonicalize_path "${TRACELENS_ROOT}")"
  if [ "$(dirname "${_tracelens_root_canonical}")" = "$(_canonicalize_path "${_open_source_root}")" ] \
     && [[ "$(basename "${_tracelens_root_canonical}")" == TraceLens@* ]]; then
    TRACELENS_ROOT=""
  elif [ ! -e "${_tracelens_root_canonical}" ]; then
    warn "TRACELENS_ROOT=${TRACELENS_ROOT} does not exist; using the pinned TraceLens checkout instead"
    TRACELENS_ROOT=""
  else
    _tracelens_root_was_set=1
  fi
fi
TRACELENS_ROOT="${TRACELENS_ROOT:-${_open_source_root}/TraceLens@${_TRACELENS_SHA}}"
TRACELENS_INTERNAL_ROOT="${TRACELENS_INTERNAL_ROOT:-}"
# Writable mirror when the optional internal extension is on a read-only mount.
TRACELENS_MIRROR_DIR="${TRACELENS_MIRROR_DIR:-${_open_source_root}/TraceLens-internal}"

DOTENV="${REPO_ROOT}/.env"

# kernel-agent consumes ANTHROPIC_* and nothing else. Translating a retired
# DEEPSEEK_* config into the standard variables belongs to whoever owns the
# entrypoint -- assets/install.sh for a chained install, llm_config for the
# Python side -- and both do it before this script is reached. Repeating it
# here bought nothing: the chained caller unsets the retired keys and exports
# the Anthropic side first, so a second translation could never fire. It only
# added a copy of the endpoint derivation that no test covered and that would
# drift out of step with deepseek_compat_env. A retired variable reaching this
# far is a stale leftover; the .env writer below purges it rather than acting
# on it.
# e2e whole-pipeline optimizer — Hyperloom calls it simply "geak" (formerly the
# standalone PerfSkills repo / GEAK_v4). Its code lives IN GEAK (interface/run_e2e.py
# + e2e_workflow/), tracked on the ``main`` branch. Hyperloom calls
# interface/run_e2e.py at the KERNEL_AGENT phase when
# KERNEL_OPT_BACKEND_ORDER=geak. It owns the GEAK_* handle; operators override
# repo/ref/root with GEAK_REPO / GEAK_REF / GEAK_ROOT. NOTE: only Hyperloom's
# internal naming changed — no upstream GEAK branch was renamed.
GEAK_REPO="${GEAK_REPO:-https://github.com/AMD-AGI/GEAK.git}"
GEAK_REF="${GEAK_REF:-main}"
_geak_root_is_operator_override=""
if [ -n "${GEAK_ROOT:-}" ]; then
  # Re-exported GEAK@* cache paths remain installer-managed across reruns.
  _geak_root_canonical="$(_canonicalize_path "${GEAK_ROOT}")"
  _geak_cache_root="$(_canonicalize_path "${_open_source_root}")"
  if [ "$(dirname "${_geak_root_canonical}")" = "${_geak_cache_root}" ] \
     && [[ "$(basename "${_geak_root_canonical}")" == GEAK@* ]]; then
    # As for TraceLens: a cache path re-resolves to the current GEAK_REF, and the runner inside it follows.
    GEAK_ROOT=""
    GEAK_E2E_RUNNER=""
  elif [ ! -e "${_geak_root_canonical}" ]; then
    warn "GEAK_ROOT=${GEAK_ROOT} does not exist; using the GEAK_REF checkout instead"
    GEAK_ROOT=""
    GEAK_E2E_RUNNER=""
  else
    _geak_root_is_operator_override=1
  fi
fi
# GEAK_REF defaults to a branch (`main`), so resolving it to a SHA hits the
# network (git ls-remote). Only do that when GEAK_ROOT was not overridden -- an
# operator-pinned root must not pay for (or fail on) a network round-trip.
if [ -z "${GEAK_ROOT:-}" ]; then
  _GEAK_SHA="$(_resolve_ref_sha "$GEAK_REPO" "$GEAK_REF")"
  GEAK_ROOT="${_open_source_root}/GEAK@${_GEAK_SHA}"
fi
GEAK_E2E_RUNNER="${GEAK_E2E_RUNNER:-${GEAK_ROOT}/interface/run_e2e.py}"
GEAK_CLAUDE_MODEL_VAL="${GEAK_CLAUDE_MODEL:-${CLAUDE_MODEL:-claude-opus-5-5}}"
# Run mode for the GEAKv4 Claude Code workflow. ``full`` (default) selects the
# 2 h / 5-round preset; ``quick`` selects the 1 h / 2-round smoke-test preset.
# GEAK still honours later CLI ``--mode`` or LLM-parsed task-hint overrides.
GEAK_RUN_MODE_VAL="${GEAK_RUN_MODE:-full}"
case "$GEAK_RUN_MODE_VAL" in
  quick|full) ;;
  *) die "GEAK_RUN_MODE must be 'quick' or 'full'; got '$GEAK_RUN_MODE_VAL'" ;;
esac
# GEAK's workflow runtime (interface/runtime/engine) needs Node 20 or newer.
_GEAK_NODE_MIN_MAJOR=20
HYPERLOOM_GEAK_NODE_VERSION="${HYPERLOOM_GEAK_NODE_VERSION:-22.23.3}"
# Split-provider aware per-side credentials: each protocol side keeps its own
# base URL/key. A dual-protocol gateway simply has both pointed at one host.
_ANTHROPIC_BASE_URL_VAL="${ANTHROPIC_BASE_URL:-}"
# API-credits credentials only. CLAUDE_CODE_OAUTH_TOKEN must never reach this
# variable: it feeds ~/.claude/config.json primaryApiKey, which would move a
# subscription run onto API billing.
_ANTHROPIC_KEY_VAL="${ANTHROPIC_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-}}"
# Part of the Anthropic-side credential for a header-authenticated gateway, and
# not derivable from the key, so it is persisted alongside the URL and key.
_ANTHROPIC_CUSTOM_HEADERS_VAL="${ANTHROPIC_CUSTOM_HEADERS:-}"
# GEAK_BASE_URL / GEAK_API_KEY are neither derived nor written here: GEAKv4 runs
# on the Anthropic side via GEAK_CLAUDE_MODEL + Claude Code auth, and an operator
# value reaches it from the environment. write_env_file removes both from the
# shared .env so a stale entry cannot outlive this install.
upsert_dotenv_var() {
  local key="$1" value="$2" tmp found=0 line stripped
  [ -n "$key" ] || return 0
  # write_env_file's dotenv_not_private: a .env another account can write lends only what runtime_env_dotenv_lent
  # allows, so nothing else is written there (the kernel-agent env file carries it).
  [ -z "${dotenv_not_private:-}" ] || runtime_env_dotenv_lent "$key" "$value" || return 0
  tmp="$(mktemp)"
  if [ -f "$DOTENV" ]; then
    while IFS= read -r line || [ -n "$line" ]; do
      stripped="${line#"${line%%[![:space:]]*}"}"
      stripped="${stripped#export }"
      case "$stripped" in
        "${key}="*) printf '%s=%s\n' "$key" "$value" >> "$tmp"; found=1 ;;
        *) printf '%s\n' "$line" >> "$tmp" ;;
      esac
    done < "$DOTENV"
  fi
  [ "$found" -eq 0 ] && printf '%s=%s\n' "$key" "$value" >> "$tmp"
  mkdir -p "$(dirname "$DOTENV")"
  mv "$tmp" "$DOTENV"
  chmod 600 "$DOTENV" 2>/dev/null || true
}

remove_dotenv_var() {
  local key="$1" tmp line stripped
  [ -n "$key" ] || return 0
  [ -f "$DOTENV" ] || return 0
  tmp="$(mktemp)"
  while IFS= read -r line || [ -n "$line" ]; do
    stripped="${line#"${line%%[![:space:]]*}"}"
    stripped="${stripped#export }"
    case "$stripped" in
      "${key}="*) ;;
      *) printf '%s\n' "$line" >> "$tmp" ;;
    esac
  done < "$DOTENV"
  mv "$tmp" "$DOTENV"
  chmod 600 "$DOTENV" 2>/dev/null || true
}

# Remove $1 from .env when its value ends in $2, the shape install.sh derives
# from a USER_DATA_PATH; an operator's value of any other shape is kept.
remove_derived_dotenv_path() {
  local key="$1" suffix="$2" line stripped value
  [ -f "$DOTENV" ] || return 0
  while IFS= read -r line || [ -n "$line" ]; do
    stripped="${line#"${line%%[![:space:]]*}"}"
    stripped="${stripped#export }"
    case "$stripped" in
      "${key}="*)
        value="${stripped#*=}"
        value="${value%$'\r'}"
        value="${value%\"}"; value="${value#\"}"
        value="${value%\'}"; value="${value#\'}"
        value="${value%/}"
        case "$value" in
          *"$suffix") remove_dotenv_var "$key"; return 0 ;;
        esac
        ;;
    esac
  done < "$DOTENV"
}

# In --check-only mode, downgrade post-install verification failures to a
# warning so report_status can still enumerate what's missing. The caller
# explicitly asked us NOT to install; failing on the first missing piece
# defeats the point of check-only.
verify_die() {
  if [ "$CHECK_ONLY" -eq 1 ]; then warn "$1"; else die "$1"; fi
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
# Stripped upstream runtime images (minimal SGLang/vLLM serving images, as
# opposed to the `rocm/hyperloom` ones) sometimes ship without one or both
# binaries.
# `_server_patcher` fail-softs in that case → `--enable-shape-discovery-
# for-cuda-graph-profile` is silently never injected → graph-replayed
# kernels stay opaque, exactly what #194 §5 was trying to fix.
#
# Apt-installing here is the cheap, framework-agnostic safety net for the
# TraceLens server patcher, so it carries no new failure modes.
ensure_patch_tools() {
  log "ensuring git + patch (required by src/hyperloom/orchestrator/actions/executors/_server_patcher.py fuzzy-fallback path)"
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
# logging in downstream benchmark wrappers (any
# `cmd 2>&1 | ts '[%H:%M:%S]'` shim the optimizer fork-execs) doesn't blow
# up with `ts: command not found`.
#
# Background: stripped upstream runtime images (minimal SGLang/vLLM serving
# images, as opposed to the `rocm/hyperloom` ones) ship without moreutils.
# When a wrapper pipes its stdout/stderr through `ts` for per-line timestamps
# and `ts` is missing, bash propagates exit code 127 up through the pipeline.
# The driving inference_optimizer validate_stack executor sees
# `subprocess_nonzero`, classifies the run as a baseline failure, and loops —
# burning minutes per iteration on a one-line apt fix. moreutils itself is a
# tiny perl-only package (<1 MB with deps), so this is a strict win over the
# retry cost.
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

_pip_install_editable() {
  local root="$1"
  local label="$2"
  local extras="${3:-}"
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
    local spec="."
    [ -n "$extras" ] && spec=".[${extras}]"
    # Do not use bash -lc: login profiles reset PATH (drops venv) and break pip.
    run sh -c "cd '$root' && python3 -m pip install -q --no-cache-dir --break-system-packages -e '$spec'"
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
      _STALE_KEYS+=("TRACELENS_ROOT")
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
      # Keep this temp-clone+pin+atomic-rename in lockstep with
      # src/hyperloom/orchestrator/trace_analysis/tracelens_analysis.py
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
  _pip_install_editable "$TRACELENS_ROOT" "TraceLens (public)" "kernel_source" || {
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
  # Editable installation needs writable source for package metadata. Mirror
  # read-only checkouts before the install-time pip call; runtime analysis only
  # checks dependencies. write_env_file() preserves the mirror path for later
  # CLI subprocesses.
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
    # The analysis agent's report step runs through this config-driven CLI.
    if command -v TraceLens_generate_perf_report_from_config >/dev/null 2>&1; then
      TraceLens_generate_perf_report_from_config --help >/dev/null
      log "TraceLens perf CLI verified: TraceLens_generate_perf_report_from_config"
    else
      verify_die "TraceLens_generate_perf_report_from_config not found after install (reinstall TraceLens, plus the optional extension if TRACELENS_INTERNAL_ROOT is set)"
    fi
  fi
}

# Write the container-local kernel-agent env file users should source so subsequent CLI calls
# pick up the upstream gateway URLs.
write_env_file() {
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    return 0
  fi
  # Emit one credential into the env file as a fallback, not an override.
  #
  # Credentials are runtime input owned by .env / the caller; the paths beside
  # them in that file are install-time results owned by this script. Every
  # documented launch sources .env first and the env file second, and only the
  # first install runs install.sh -- so a credential re-exported unconditionally
  # outranks a key the operator just rotated, on every later launch, with nothing
  # to show for it (#1169). Assigning only when unset keeps the snapshot useful
  # for the slurm path that sources the env file alone, and announces a
  # mismatch (never a value) so a rotation that has not reached this file stays
  # visible at launch. Defined here so the emitted block travels with the
  # function body.
  _emit_credential_fallback() {
    local _name="$1"
    local _value="$2"
    echo "if [ -n \"\${${_name}:-}\" ]; then"
    echo "  [ \"\${${_name}}\" = '${_value}' ] || \\"
    echo "    echo '[kernel-agent] ${_name} differs from the install-time snapshot in this file; keeping the value already in the environment (re-run install.sh to refresh it)' >&2"
    echo "else"
    echo "  export ${_name}='${_value}'"
    echo "fi"
  }
  # Strict protocol-side separation: each side keeps its own canonical values;
  # GEAK aliases are never written back to provider slots.
  local _anthropic_url="${_ANTHROPIC_BASE_URL_VAL:-}"
  local _anthropic_key="${_ANTHROPIC_KEY_VAL:-}"
  local _anthropic_headers="${_ANTHROPIC_CUSTOM_HEADERS_VAL:-}"
  # Warn loudly if the Anthropic endpoint is unresolved — kernel-agent env would
  # silently lack a base URL and CLIs would resort to whatever was in the
  # operator's shell rc, defeating the point of this file.
  if [ -z "${_anthropic_url:-}" ]; then
    warn "LLM gateway URL empty; kernel-agent env will lack ANTHROPIC_BASE_URL"
  fi
  runtime_env_remove_legacy_kernel_agent_env
  local env_file="${KERNEL_AGENT_ENV}" env_dir written
  env_dir="$(dirname "$env_file")"
  runtime_env_private_kernel_agent_env "$env_file" write || die "kernel-agent env file not written"
  # Written beside the file (mktemp makes it 0600) and renamed over it, so an earlier copy is replaced whoever owns
  # it, never rewritten in place or chmod-ed.
  written="$(mktemp "${env_dir}/.kernel-agent.env.XXXXXX")"
  {
    echo '#!/bin/sh'
    echo "# kernel-agent runtime env (regenerated by install.sh)"
    [ -n "${USER_DATA_PATH:-}" ] && echo "export USER_DATA_PATH='${USER_DATA_PATH}'"
    [ -n "${HYPERLOOM_RUNTIME_DIR:-}" ] && echo "export HYPERLOOM_RUNTIME_DIR='${HYPERLOOM_RUNTIME_DIR}'"
    [ -n "${KERNEL_AGENT_ENV:-}" ] && echo "export KERNEL_AGENT_ENV='${KERNEL_AGENT_ENV}'"
    [ -n "${PYTHONPATH:-}" ] && echo "export PYTHONPATH='${PYTHONPATH}'"
    [ -n "${INFERENCEX_PATH:-}" ] && echo "export INFERENCEX_PATH='${INFERENCEX_PATH}'"
    # The kernel-agent drives Claude Code, so only the Anthropic side is
    # exported here; gateway/OpenAI credentials are never persisted.
    [ -n "${_anthropic_url}" ] && _emit_credential_fallback ANTHROPIC_BASE_URL "${_anthropic_url}"
    [ -n "${_anthropic_key}" ] && _emit_credential_fallback ANTHROPIC_API_KEY "${_anthropic_key}"
    # A header-authenticated gateway rejects the CLI without this, and it cannot
    # be re-derived from the key, so it is a credential in its own right. The
    # single quotes _emit_credential_fallback writes keep any ${VAR} reference
    # intact for parse_custom_headers to expand.
    [ -n "${_anthropic_headers}" ] && _emit_credential_fallback ANTHROPIC_CUSTOM_HEADERS "${_anthropic_headers}"
    # A subscription token is the Anthropic side on its own: an oauth-only host
    # resolves neither URL nor key, so without this line sourcing the file
    # leaves the kernel-agent with no Anthropic credential at all.
    [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && _emit_credential_fallback CLAUDE_CODE_OAUTH_TOKEN "${CLAUDE_CODE_OAUTH_TOKEN}"
    # Pin TRACELENS_ROOT and TRACELENS_INTERNAL_ROOT to the (possibly
    # mirrored) values resolved by ensure_tracelens(). This is what lets
    # setsid nohup python -m hyperloom optimize →
    # python -m hyperloom.orchestrator.trace_analysis.tracelens_analysis inherit the writable
    # mirrors instead of falling back to the read-only /path defaults.
    [ -n "${TRACELENS_ROOT:-}" ] && echo "export TRACELENS_ROOT='${TRACELENS_ROOT}'"
    if [ -n "${TRACELENS_INTERNAL_ROOT:-}" ]; then
      echo "export TRACELENS_INTERNAL_ROOT='${TRACELENS_INTERNAL_ROOT}'"
      echo "export TL_EXTENSION='TraceLens_internal'"
    fi
    [ -n "${HYPERLOOM_ROOT:-}" ] && echo "export HYPERLOOM_ROOT='${HYPERLOOM_ROOT}'"
    # e2e optimizer ("geak") checkout + runner (GEAK_ROOT / GEAK_E2E_RUNNER),
    # consumed by src/hyperloom/orchestrator/kernel/geak_runner.py.
    [ -n "${GEAK_E2E_RUNNER}" ] && echo "export GEAK_E2E_RUNNER='${GEAK_E2E_RUNNER}'"
    [ -n "${GEAK_ROOT}" ] && echo "export GEAK_ROOT='${GEAK_ROOT}'"
    [ -n "${GEAK_CLAUDE_MODEL_VAL}" ] && echo "export GEAK_CLAUDE_MODEL='${GEAK_CLAUDE_MODEL_VAL}'"
    [ -n "${_claude_path_dir:-}" ] && echo "export PATH='${_claude_path_dir}':\"\$PATH\""
    # Pin the claude binary the GEAK SDK path uses (else claude_agent_sdk may
    # fall back to its older bundled CLI). run_e2e.py maps this to cli_path.
    _geak_claude_bin="$(command -v claude 2>/dev/null || true)"
    [ -n "${_geak_claude_bin}" ] && echo "export GEAK_CLAUDE_BIN='${_geak_claude_bin}'"
    # GEAK runs bare `node` / `codex` when these are unset, and neither is on a stock image's PATH.
    [ -n "${_geak_node_bin:-}" ] && echo "export GEAK_NODE_BIN='${_geak_node_bin}'"
    [ -n "${_geak_codex_bin:-}" ] && echo "export GEAK_CODEX_BIN='${_geak_codex_bin}'"
    # e2e optimizer budget mode (read by the inference_optimizer kernel request
    # handler to pick the backend budget default) + LLM connection for the runner.
    [ -n "${GEAK_RUN_MODE_VAL}" ] && echo "export GEAK_RUN_MODE='${GEAK_RUN_MODE_VAL}'"
    # GEAK_API_KEY / GEAK_BASE_URL are intentionally NOT emitted: they derive
    # from Anthropic/DeepSeek values and would reintroduce
    # cross-provider leakage into the sourced runtime env.
    # GEAK scoring / profiler / shape knobs. These are read by GEAK itself, but
    # the optimize CLI sources THIS file and its env replaces the launcher's
    # exports -- so any knob not persisted here is silently dropped before
    # reaching GEAK (e.g. GEAK_SCORE_TARGET=kernel fell back to wall).
    # Passthrough-if-set (no hardcoded defaults):
    #   GEAK_SCORE_TARGET         -> score best-patch on kernel_ms vs wall (E2E-transferable)
    #   GEAK_SKIP_PROFILE         -> skip the advisory profiler-mcp roofline pass
    #   GEAK_MAX_BENCHMARK_SHAPES -> harness benchmark-shape cap
    [ -n "${GEAK_SCORE_TARGET:-}" ] && echo "export GEAK_SCORE_TARGET='${GEAK_SCORE_TARGET}'"
    [ -n "${GEAK_SKIP_PROFILE:-}" ] && echo "export GEAK_SKIP_PROFILE='${GEAK_SKIP_PROFILE}'"
    [ -n "${GEAK_MAX_BENCHMARK_SHAPES:-}" ] && echo "export GEAK_MAX_BENCHMARK_SHAPES='${GEAK_MAX_BENCHMARK_SHAPES}'"
  } > "$written"
  mv -f -- "$written" "$env_file"
  log "wrote ${env_file} (source it before running kernel-agent tools)"

  # A checkout another account can write -- a group- or other-writable checkout or .env, one of another account's, or a
  # link, as a checkout bind-mounted from shared storage is -- keeps no credentials in .env, where that account could
  # read or rewrite them: they are in the file above alone, and the ones an earlier install or the operator left in .env
  # (with their retired spellings) are removed first, before any write below copies .env; a link there is replaced, or
  # removed when it points at anything but a regular file. They were never loaded (load_dotenv_no_clobber skips them
  # in such a .env), so none of them, another account's included, reached the file above. Readers that took them from
  # .env take them from that file instead (install.sh preflight_load_dotenv, preflight's _load_dotenv_fallback).
  # Asked before the first write below, which replaces .env with a file of this account's.
  # Every other line such a .env would not lend goes with them (runtime_env_keep_lent_dotenv_lines): the writes below
  # leave a file of this account's, which the next reader may find private, so a line another account planted
  # (NODE_OPTIONS, OPENAI_BASE_URL) would be lent from then on.
  local dotenv_not_private
  dotenv_not_private="$(runtime_env_dotenv_not_private "$DOTENV")"
  if [ -n "$dotenv_not_private" ]; then
    runtime_env_keep_lent_dotenv_lines "$DOTENV" ||
      die "the credentials in ${DOTENV} could not be removed, and another account can write it (${dotenv_not_private}); remove them by hand"
    log "credentials kept out of ${DOTENV}, which another account can write (${dotenv_not_private}): they are in ${env_file} alone"
  fi

  # .env is shared by every session run from this checkout, so it keeps only the
  # workspace root: a later run without USER_DATA_PATH in its shell still finds
  # this workspace, and the paths below derive from that root again. They are not
  # persisted (the env file above carries them to this workspace's runtime), and
  # a copy an earlier install wrote is purged, or it would pair another
  # workspace's runtime dir with this root. A value of another shape is an
  # operator's own and stays.
  [ -n "${USER_DATA_PATH:-}" ] && upsert_dotenv_var USER_DATA_PATH "$USER_DATA_PATH"
  remove_derived_dotenv_path HYPERLOOM_RUNTIME_DIR /runtime
  remove_derived_dotenv_path KERNEL_AGENT_ENV /runtime/kernel-agent.env.sh
  remove_derived_dotenv_path HYPERLOOM_ROOT /runtime/source-mirrors
  [ -n "${PYTHONPATH:-}" ] && upsert_dotenv_var PYTHONPATH "$PYTHONPATH"
  remove_dotenv_var INFERENCEX_PATH
  # Persist each provider's own canonical vars (write when present, clear when
  # absent) so an operator's .env creds survive a re-install. Never cross-write
  # and never persist gateway aliases, so the two providers stay separated.
  # Only in a private checkout (see dotenv_not_private above).
  if [ -z "$dotenv_not_private" ]; then
    if [ -n "${_anthropic_url}" ]; then
      upsert_dotenv_var ANTHROPIC_BASE_URL "$_anthropic_url"
    else
      remove_dotenv_var ANTHROPIC_BASE_URL
    fi
    if [ -n "${_anthropic_key}" ]; then
      upsert_dotenv_var ANTHROPIC_API_KEY "$_anthropic_key"
    else
      remove_dotenv_var ANTHROPIC_API_KEY
    fi
    # The subscription token is a credential form of its own, not another
    # spelling of the API key, so it is persisted beside the slot above rather
    # than folded into it. Write-only: an unset token means the operator
    # authenticates some other way this run, not that a stored one should go.
    if [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
      upsert_dotenv_var CLAUDE_CODE_OAUTH_TOKEN "$CLAUDE_CODE_OAUTH_TOKEN"
    fi
    [ -z "${HYPERLOOM_DOTENV_CREDENTIALS:-}" ] || remove_dotenv_var HYPERLOOM_DOTENV_CREDENTIALS
  else
    # The writes above leave a .env of this account's even where another account could write the one they replaced,
    # and a later reader would take it for a private .env that holds the credentials. This line, no secret, says they
    # are in the kernel-agent env file (runtime_env_dotenv_credentials_elsewhere).
    upsert_dotenv_var HYPERLOOM_DOTENV_CREDENTIALS kernel-agent-env
  fi
  # Only the Anthropic side is persisted here. The OpenAI-side entries in the
  # shared .env are owned elsewhere and left untouched -- which is also what
  # keeps the OpenAI half of a dual-protocol gateway alive across a re-install.
  # Retired provider vars never survive one.
  remove_dotenv_var DEEPSEEK_BASE_URL
  remove_dotenv_var DEEPSEEK_API_KEY
  remove_dotenv_var DEEPSEEK_MODEL
  remove_dotenv_var ANTHROPIC_AUTH_TOKEN
  # Legacy gateway key: not read, purged if present.
  remove_dotenv_var SAFE_API_KEY
  remove_dotenv_var AMD_LLM_API_KEY
  remove_dotenv_var LLM_GATEWAY_KEY
  # A pinned cache path written here would read back as an override after the pin moves. The
  # kernel-agent env file carries it to the runtime; an operator's own path is left as written.
  [ -z "${_tracelens_root_was_set:-}" ] && remove_dotenv_var TRACELENS_ROOT
  if [ -n "${TRACELENS_INTERNAL_ROOT:-}" ]; then
    upsert_dotenv_var TRACELENS_INTERNAL_ROOT "$TRACELENS_INTERNAL_ROOT"
    upsert_dotenv_var TL_EXTENSION "TraceLens_internal"
  fi
  # The runner lives in the GEAK checkout, so it follows GEAK_ROOT out of .env.
  if [ -z "${_geak_root_is_operator_override:-}" ]; then
    remove_dotenv_var GEAK_ROOT
    remove_dotenv_var GEAK_E2E_RUNNER
  fi
  [ -n "${GEAK_CLAUDE_MODEL_VAL}" ] && upsert_dotenv_var GEAK_CLAUDE_MODEL "$GEAK_CLAUDE_MODEL_VAL"
  [ -n "${_geak_claude_bin}" ] && upsert_dotenv_var GEAK_CLAUDE_BIN "$_geak_claude_bin"
  [ -n "${_geak_node_bin:-}" ] && upsert_dotenv_var GEAK_NODE_BIN "$_geak_node_bin"
  [ -n "${_geak_codex_bin:-}" ] && upsert_dotenv_var GEAK_CODEX_BIN "$_geak_codex_bin"
  [ -n "${GEAK_RUN_MODE_VAL}" ] && upsert_dotenv_var GEAK_RUN_MODE "$GEAK_RUN_MODE_VAL"
  remove_dotenv_var GEAK_API_KEY
  remove_dotenv_var GEAK_BASE_URL
  [ -n "${GEAK_SCORE_TARGET:-}" ] && upsert_dotenv_var GEAK_SCORE_TARGET "$GEAK_SCORE_TARGET"
  [ -n "${GEAK_SKIP_PROFILE:-}" ] && upsert_dotenv_var GEAK_SKIP_PROFILE "$GEAK_SKIP_PROFILE"
  [ -n "${GEAK_MAX_BENCHMARK_SHAPES:-}" ] && upsert_dotenv_var GEAK_MAX_BENCHMARK_SHAPES "$GEAK_MAX_BENCHMARK_SHAPES"
  log "updated ${DOTENV} with kernel-agent runtime env"
}

# Clone the e2e optimizer ("geak", formerly PerfSkills) for its
# interface/run_e2e.py runner, then pip-install the GEAK package + claude_agent_sdk.
ensure_geak() {
  log "ensuring e2e optimizer geak (GEAK@${GEAK_REF}, formerly PerfSkills)"
  if [ -n "${_geak_root_is_operator_override:-}" ]; then
    if ! _is_git_checkout_root "${GEAK_ROOT}"; then
      if [ "$DRY_RUN" -eq 1 ] || [ "$CHECK_ONLY" -eq 1 ]; then
        warn "operator-supplied GEAK_ROOT is not a git checkout: ${GEAK_ROOT}"
      else
        die "operator-supplied GEAK_ROOT is not a git checkout: ${GEAK_ROOT}"
      fi
    else
      log "using operator-supplied GEAK checkout unchanged: ${GEAK_ROOT}"
    fi
  elif [ ! -d "${GEAK_ROOT}/.git" ]; then
    if [ "$DRY_RUN" -eq 0 ] && [ "$CHECK_ONLY" -eq 0 ]; then
      mkdir -p "${GEAK_ROOT}"
    fi
    if [ "$CHECK_ONLY" -eq 1 ]; then
      _STALE_KEYS+=("GEAK_ROOT")
    fi
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
    _PIP_FLAGS="-q --no-cache-dir --break-system-packages"
    # GEAK is a pip package now: install from the checkout above so the package
    # matches the interface/run_e2e.py we run and honours any GEAK_REPO/GEAK_REF
    # override (local mirror, fork, SSH URL). Its bootstrap installs deps + the
    # Claude Code CLI (>= 2.1.177); GEAK_HOME reuses our checkout so bootstrap
    # skips a second clone.
    if [ -f "${GEAK_ROOT}/pyproject.toml" ] || [ -f "${GEAK_ROOT}/setup.py" ]; then
      run env GEAK_HOME="${GEAK_ROOT}" python3 -m pip install ${_PIP_FLAGS} "${GEAK_ROOT}" || \
        warn "GEAK pip install failed"
    else
      warn "GEAK package metadata missing at ${GEAK_ROOT}; skipping pip install"
    fi
    run python3 -m pip install ${_PIP_FLAGS} claude-agent-sdk anyio || \
      warn "claude-agent-sdk install failed; run_e2e.py will fall back to the claude CLI"
    if [ ! -f "${GEAK_E2E_RUNNER}" ]; then
      warn "e2e runner not found at ${GEAK_E2E_RUNNER} (interface/ missing — is the checkout on the ${GEAK_REF} branch with the e2e code?)"
    fi
  else
    log "check-only: skipping e2e optimizer sdk installation"
  fi
}

# The claude CLI on PATH is the single copy every consumer runs: specialists
# resolve it through GEAK_CLAUDE_BIN or PATH, GEAK is pointed at it through
# GEAK_CLAUDE_BIN, and Forge resolves PATH before its fallback locations. It is
# ensured before ensure_geak, so GEAK's bootstrap finds it and only steps in
# when it is older than GEAK's minimum, installing into the same ~/.local/bin.
# Claude Code's native installer places the CLI in ~/.local/bin, so that
# directory goes first on PATH: a pinned install wins over an image's claude,
# and an existing copy there is reused rather than downloaded again.
ensure_claude_cli() {
  local _claude_home _claude_bin_dir
  _claude_home="$(_home_dir || true)"
  if [ -n "$_claude_home" ]; then
    _claude_bin_dir="${_claude_home}/.local/bin"
    export PATH="${_claude_bin_dir}:${PATH}"
    # write_env_file carries it to every shell that sources the runtime env.
    _claude_path_dir="${_claude_bin_dir}"
  fi
  if [ "$CHECK_ONLY" -eq 1 ]; then
    command -v claude >/dev/null 2>&1 || warn "claude CLI not on PATH; specialists and the kernel backends cannot run it"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would ensure the claude CLI on PATH and write ~/.claude/config.json"
    return 0
  fi
  if [ -n "${HYPERLOOM_CLAUDE_CODE_VERSION:-}" ] || ! command -v claude >/dev/null 2>&1; then
    [ -n "$_claude_home" ] || die "no home directory for uid $(id -u); the claude CLI installs under ~/.local/bin"
    command -v curl >/dev/null 2>&1 || die "curl missing; the claude CLI cannot be installed"
    local _claude_target="${HYPERLOOM_CLAUDE_CODE_VERSION:-latest}"
    log "installing the claude CLI (${_claude_target}) into ${_claude_bin_dir}"
    curl -fsSL --connect-timeout 20 https://claude.ai/install.sh | HOME="$_claude_home" bash -s -- "$_claude_target" \
      || die "the claude CLI installer failed"
  fi
  command -v claude >/dev/null 2>&1 || die "claude CLI is not on PATH after the install"
  # ~/.claude authenticates the Claude Code CLI for Anthropic-compatible flows.
  # Its readers resolve Path.home(), so ~ must come from _home_dir here too.
  local _claude_key="${_ANTHROPIC_KEY_VAL:-}"
  if [ -n "$_claude_key" ]; then
    if [ -z "$_claude_home" ]; then
      warn "no home directory for uid $(id -u) (HOME unset, no passwd entry); ~/.claude/config.json not written"
      return 0
    fi
    mkdir -p "${_claude_home}/.claude"
    local _anthropic_url="${_ANTHROPIC_BASE_URL_VAL:-}"
    _anthropic_url="${_anthropic_url%/}"
    _anthropic_url="${_anthropic_url%/v1}"
    cat > "${_claude_home}/.claude/config.json" <<EOF
{
  "theme": "dark",
  "hasCompletedOnboarding": true,
  "primaryApiKey": "${_claude_key}",
  "customApiUrl": "${_anthropic_url}"
}
EOF
    chmod 600 "${_claude_home}/.claude/config.json"
    log "wrote ${_claude_home}/.claude/config.json"
  elif [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    log "subscription token in use; ~/.claude/config.json left alone (primaryApiKey would override it)"
  else
    warn "Anthropic/DeepSeek key not set; ~/.claude/config.json not written"
  fi
}

# The Codex CLI Hyperloom's own specialists resolve: an operator's GEAK_CODEX_BIN, then PATH, then the
# runtime bundled with the openai-codex SDK that install.sh installs with Hyperloom.
_resolve_geak_codex_bin() {
  "${PYTHON:-python3}" -c 'import os; from hyperloom.common.agent_cli import resolve_codex_executable; print(resolve_codex_executable(os.environ.get("GEAK_CODEX_BIN", "")))' 2>/dev/null || true
}

# Whether GEAK will take its Node runtime: a backend or profile is named, or the keys are OpenAI-only (as in
# cli/geak_binaries.py). With none, GEAK keeps its native path, which runs no Node.
_geak_agent_selected() {
  [ -n "${GEAK_AGENT_BACKEND:-}${GEAK_AGENT_PROFILE:-}" ] && return 0
  case "$(printf '%s' "${GEAK_AGENT_AUTO:-1}" | tr '[:upper:]' '[:lower:]')" in 0|false|no) return 1 ;; esac
  "${PYTHON:-python3}" -c 'import sys; from hyperloom_common.llm.llm_config import is_openai_only; sys.exit(not is_openai_only())' 2>/dev/null
}

_geak_node_unavailable() {
  if _geak_agent_selected; then die "$1"; fi
  warn "$1; GEAK's workflow runtime cannot start until Node ${_GEAK_NODE_MIN_MAJOR}+ is on PATH or GEAK_NODE_BIN is set"
}

# Pin the Node binary GEAK's workflow runtime runs (GEAK_NODE_BIN). An operator's
# value is kept, a Node on PATH that is new enough is reused, and otherwise the
# official build is fetched once into the dependency cache, checked against the
# release's SHASUMS256.txt.
ensure_geak_node() {
  _geak_node_bin=""
  if [ -n "${GEAK_NODE_BIN:-}" ]; then
    _geak_node_bin="${GEAK_NODE_BIN}"
    return 0
  fi
  local _found _major
  _found="$(command -v node 2>/dev/null || true)"
  if [ -n "$_found" ]; then
    _major="$("$_found" --version 2>/dev/null || true)"
    _major="${_major#v}"
    _major="${_major%%.*}"
    case "$_major" in
      ''|*[!0-9]*) ;;
      *)
        if [ "$_major" -ge "$_GEAK_NODE_MIN_MAJOR" ]; then
          _geak_node_bin="$_found"
          return 0
        fi
        ;;
    esac
  fi
  local _arch
  case "$(uname -m)" in
    x86_64) _arch=x64 ;;
    aarch64|arm64) _arch=arm64 ;;
    *) die "no Node build for $(uname -m); set GEAK_NODE_BIN to a Node ${_GEAK_NODE_MIN_MAJOR}+ binary" ;;
  esac
  local _name="node-v${HYPERLOOM_GEAK_NODE_VERSION}-linux-${_arch}"
  local _dir="${_open_source_root}/${_name}"
  if [ -x "${_dir}/bin/node" ]; then
    _geak_node_bin="${_dir}/bin/node"
    return 0
  fi
  if [ "$CHECK_ONLY" -eq 1 ]; then
    warn "no Node ${_GEAK_NODE_MIN_MAJOR}+ on PATH or in ${_dir}; GEAK's workflow runtime cannot start"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would fetch ${_name} into ${_open_source_root} for GEAK"
    return 0
  fi
  command -v curl >/dev/null 2>&1 || { _geak_node_unavailable "curl missing; Node for GEAK cannot be fetched"; return 0; }
  local _base="https://nodejs.org/dist/v${HYPERLOOM_GEAK_NODE_VERSION}" _tmp
  mkdir -p "${_open_source_root}"
  _tmp="$(mktemp -d "${_open_source_root}/.${_name}.XXXXXX")"
  log "fetching ${_name} for GEAK into ${_dir}"
  if ! curl -fsSL --connect-timeout 20 -o "${_tmp}/${_name}.tar.xz" "${_base}/${_name}.tar.xz" \
     || ! curl -fsSL --connect-timeout 20 -o "${_tmp}/SHASUMS256.txt" "${_base}/SHASUMS256.txt"; then
    rm -rf -- "$_tmp"
    _geak_node_unavailable "downloading ${_name} from ${_base} failed"
    return 0
  fi
  if ! (cd "$_tmp" && grep " ${_name}.tar.xz\$" SHASUMS256.txt | sha256sum -c - >/dev/null); then
    rm -rf -- "$_tmp"
    die "${_name}.tar.xz does not match its SHASUMS256.txt entry"
  fi
  # The archive records its builder's uid/gid; root would chown to them, which a one-id user namespace refuses.
  if ! tar --no-same-owner -xJf "${_tmp}/${_name}.tar.xz" -C "$_tmp"; then
    rm -rf -- "$_tmp"
    die "unpacking ${_name}.tar.xz failed (is xz installed?)"
  fi
  mv "${_tmp}/${_name}" "$_dir"
  rm -rf -- "$_tmp"
  _geak_node_bin="${_dir}/bin/node"
}

report_status() {
  # The analysis agent's report step runs through this config-driven CLI.
  if command -v TraceLens_generate_perf_report_from_config >/dev/null 2>&1; then
    log "found TraceLens_generate_perf_report_from_config: $(command -v TraceLens_generate_perf_report_from_config)"
  else
    warn "TraceLens_generate_perf_report_from_config not found"
  fi
  for tool in git patch; do
    if command -v "$tool" >/dev/null 2>&1; then
      log "found ${tool}: $(command -v "$tool")"
    else
      warn "${tool} not found (TraceLens server patcher will fail-soft without it)"
    fi
  done
  if _is_git_checkout_root "${GEAK_ROOT}"; then
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

install_kernel_tools() {
  ensure_python
  ensure_patch_tools
  ensure_moreutils
  # TraceLens / GEAK clones run under the install lock install.sh holds.
  ensure_tracelens
  ensure_claude_cli
  ensure_geak_node
  _geak_codex_bin="$(_resolve_geak_codex_bin)"

  # The GEAK e2e whole-pipeline optimizer is always installed; whether it is
  # used at runtime is decided per-session via KERNEL_OPT_BACKEND_ORDER.
  ensure_geak
  _prune_dep_cache "TraceLens" "GEAK"
  write_env_file

  report_status
  log "kernel-agent tools installed"
}
