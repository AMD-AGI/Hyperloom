#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

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
# (TRACELENS_ROOT, TRACELENS_INTERNAL_ROOT)
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
# Container images ship a writable /workspace; a bare-metal host off root has
# neither it nor permission to create it, so the mkdir below would abort.
_default_workspace_root() {
  # The nearest existing ancestor decides: -w is false for a path that does not
  # exist yet, which would divert root off a /workspace it can still create.
  _ws_probe=/workspace
  while [ ! -e "$_ws_probe" ] && [ "$_ws_probe" != / ]; do _ws_probe=$(dirname "$_ws_probe"); done
  if [ -w "$_ws_probe" ]; then printf '%s' /workspace/hyperloom; else printf '%s' "$(pwd -P)/session"; fi
}
USER_DATA_PATH="${USER_DATA_PATH:-$(_default_workspace_root)}"
if [ -z "${_user_data_was_set}" ]; then
  echo "[install WARN] USER_DATA_PATH not set; defaulting to ${USER_DATA_PATH}. Set USER_DATA_PATH to persist artifacts under your data root." >&2
fi
HYPERLOOM_RUNTIME_DIR="${HYPERLOOM_RUNTIME_DIR:-${USER_DATA_PATH}/runtime}"
KERNEL_AGENT_ENV="${KERNEL_AGENT_ENV:-${HYPERLOOM_RUNTIME_DIR}/kernel-agent.env.sh}"
# Legacy variable kept for compatibility; open-source checkouts use _open_source_root.
HYPERLOOM_ROOT="${HYPERLOOM_ROOT:-${HYPERLOOM_RUNTIME_DIR}/source-mirrors}"
# Writable, repo-local base for auto-cloned deps: $HYPERLOOM_CACHE_DIR else
# $REPO_ROOT/.cache, cloned per revision (<name>@<sha>). Not /tmp (a reaper can
# wipe it mid-run, leaving TRACELENS_ROOT dangling — #722).
_open_source_root="${HYPERLOOM_CACHE_DIR:-${REPO_ROOT}/.cache}"
# MAGPIE_PATH is the single override shared with the Python runtime; this
# installer never fetches Magpie itself, it only records the path.
MAGPIE_PATH="${MAGPIE_PATH:-${_open_source_root}/Magpie}"
# Resolve MAGPIE_PYTHON dynamically. The previous default
# ${MAGPIE_PATH}/venv/bin/python assumed a Magpie-private venv, but
# src/hyperloom/inference_optimizer/assets/install.sh's ensure_magpie() pip
# installs Magpie into the driver Python's site-packages
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
# site-packages/dist-packages is already on the import path; keeping it off
# PYTHONPATH avoids shadowing an isolated vLLM venv's torch (undefined symbol).
_is_python_package_root() {
  local p="${1%/}"
  case "${p##*/}" in
    site-packages | dist-packages) return 0 ;;
    *) return 1 ;;
  esac
}
# MAGPIE_PATH belongs on PYTHONPATH only for a source checkout, not a pip
# install (site-packages, already importable).
_magpie_pythonpath_arg() {
  local p="${MAGPIE_PATH:-}"
  { [ -n "$p" ] && ! _is_python_package_root "$p"; } && printf '%s' "$p"
}
# Keep REPO_ROOT on PYTHONPATH so subprocesses can ``import hyperloom`` under a
# ``pip install --target $REPO_ROOT`` layout (the target dir is not on the
# default sys.path). Put REPO_ROOT first, then MAGPIE_PATH, then any pre-existing
# PYTHONPATH; write_env_file recomposes this the same way just before persisting
# it, so a stale .env sourced later cannot drop REPO_ROOT.
PYTHONPATH="$(_compose_pythonpath "${REPO_ROOT:-}" "$(_magpie_pythonpath_arg)" "${PYTHONPATH:-}")"
INFERENCEX_PATH="${INFERENCEX_PATH:-}"
# TraceLens base repo is required; the internal extension is OPTIONAL.
#   1. AMD-AGI/TraceLens          -> $TRACELENS_ROOT  (base: skills, patches, CLI, analysis orchestrator)
#   2. TraceLens-internal -> $TRACELENS_INTERNAL_ROOT (optional rehydration module)
# Default base clones the public repo into the workspace runtime tree,
# matching Magpie / InferenceX rather than persisting pod-local mirrors.
# The internal extension is used ONLY when $TRACELENS_INTERNAL_ROOT is set
# (env / .env); leave it unset for the base-only report. No separate toggle.
TRACELENS_REPO="https://github.com/AMD-AGI/TraceLens.git"
# TraceLens v1.0 integration: head of
# release/hyperloom_integration_v1.0. The optional internal extension tracks
# the matching release/hyperloom_integration_v1.0 branch of
# AMD-AGI/TraceLens-internal, but Hyperloom keeps no pin/URL for it — the
# operator supplies it via TRACELENS_INTERNAL_ROOT.
TRACELENS_REF="a59a9c165bb64c7c416fd7cf79149803d552e43c"
# Operator override iff TRACELENS_ROOT points OUTSIDE the pod-local default.
# The persistent kernel-agent env re-exports the resolved default path, so a
# presence-only check (${VAR:+1}) would misclassify it as an override and skip
# the managed clone/realign, leaving a missing/stale default un-self-healed
# across reruns (#722). Compare against the default path, not env presence.
# Canonicalize a path (resolve symlinks/.. , strip trailing slash) so the
# default-vs-override comparison matches the Python side's Path.resolve();
# unresolvable paths fall back to the trimmed literal so a not-yet-cloned
# default still compares correctly (#722 / PR#789).
# Keep in lockstep with Python-side path canonicalization helpers.
_canonicalize_path() {
  local p="${1:-}"
  [ -z "$p" ] && return 0
  readlink -f -- "$p" 2>/dev/null || printf '%s' "${p%/}"
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
# Resolve a git ref to a commit SHA (7-40 hex passes through; branch/tag via
# ls-remote, falling back to the raw ref). The SHA keys the per-revision cache.
_resolve_ref_sha() {
  local repo="$1" ref="$2" sha=""
  if [[ "$ref" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
    printf '%s' "$ref"
    return 0
  fi
  sha="$(git ls-remote "$repo" "$ref" 2>/dev/null | awk 'NR==1{print $1}')"
  if [ -z "$sha" ]; then
    # Loud, not silent: a raw-ref cache key drops the per-revision guarantee.
    echo "[kernel-agent WARN] could not resolve '$ref' at $repo to a commit SHA (network or bad ref); using '$ref' as the per-revision cache key -- stale-checkout guard weakened. Pin *_REF to a 40-hex SHA or restore network access." >&2
    sha="$ref"
  fi
  printf '%s' "$sha"
}

# Bound cache growth: keep the newest $HYPERLOOM_CACHE_KEEP (default 3, 0 disables)
# <name>@<sha> checkouts per dep, prune older ones. A moving branch ref (GEAK
# `main`) resolves to a new SHA each HEAD bump, so the cache would grow unbounded.
# Lock-held; the just-installed revision is newest, so always retained.
_prune_dep_cache() {
  local keep="${HYPERLOOM_CACHE_KEEP:-3}"
  case "$keep" in ''|*[!0-9]*) keep=3 ;; esac
  [ "$keep" -eq 0 ] && return 0
  local name stale listing
  for name in "$@"; do
    # `|| true`: no-match glob fails `ls` under `set -euo pipefail`. Collect, then act.
    listing="$(ls -dt "${_open_source_root}/${name}@"* 2>/dev/null | tail -n +"$((keep + 1))" || true)"
    [ -n "$listing" ] || continue
    while IFS= read -r stale; do
      [ -n "$stale" ] && [ -d "$stale" ] || continue
      log "pruning stale dep cache (keeping newest ${keep} ${name}@*): ${stale}"
      rm -rf -- "$stale" 2>/dev/null || true
    done <<EOF
$listing
EOF
  done
}
_TRACELENS_SHA="$(_resolve_ref_sha "$TRACELENS_REPO" "$TRACELENS_REF")"
_tracelens_default_root="$(_canonicalize_path "${_open_source_root}/TraceLens@${_TRACELENS_SHA}")"
_tracelens_root_was_set=""
if [ -n "${TRACELENS_ROOT:-}" ] \
   && [ "$(_canonicalize_path "${TRACELENS_ROOT}")" != "${_tracelens_default_root}" ]; then
  _tracelens_root_was_set=1
fi
TRACELENS_ROOT="${TRACELENS_ROOT:-${_open_source_root}/TraceLens@${_TRACELENS_SHA}}"
TRACELENS_INTERNAL_ROOT="${TRACELENS_INTERNAL_ROOT:-}"
# Writable mirror when the optional internal extension is on a read-only mount.
TRACELENS_MIRROR_DIR="${TRACELENS_MIRROR_DIR:-${_open_source_root}/TraceLens-internal}"

# Credentials fallback: env always wins. If any supported LLM credential is
# missing from env, source $REPO_ROOT/.env but protect already-set values.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
DOTENV="${REPO_ROOT}/.env"

# Vars whose already-resolved value must survive the `. $REPO_ROOT/.env` below.
# The dotenv is a *credentials* fallback, but it is a full env file: sourcing it
# under `set -a` re-imports every path var it happens to carry. A stale entry —
# or one written by a DIFFERENT checkout / workspace sharing this .env — then
# silently overwrites what the caller resolved, and the run installs against one
# workspace while writing its env file into another (#Hyperloom-2 / atom mixup).
# Only non-empty pre-source values are restored, so the dotenv still fills gaps.
_DOTENV_PROTECTED_VARS='REPO_ROOT KERNEL_AGENT_ROOT HYPERLOOM_KERNEL_AGENT_ROOT
USER_DATA_PATH HYPERLOOM_RUNTIME_DIR KERNEL_AGENT_ENV HYPERLOOM_ROOT
MAGPIE_PATH INFERENCEX_PATH TRACELENS_ROOT TRACELENS_INTERNAL_ROOT
GEAK_ROOT GEAK_E2E_RUNNER PYTHONPATH'

if [ -z "${ANTHROPIC_BASE_URL:-}" ] || [ -z "${ANTHROPIC_API_KEY:-}" ] \
   || [ -z "${ANTHROPIC_AUTH_TOKEN:-}" ] || [ -z "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
  if [ -f "$REPO_ROOT/.env" ]; then
    _snap_anthropic_url="${ANTHROPIC_BASE_URL-}"
    _snap_anthropic_key="${ANTHROPIC_API_KEY-}"
    _snap_anthropic_token="${ANTHROPIC_AUTH_TOKEN-}"
    # The subscription token is snapshotted like the other three: without it a
    # stale token in .env silently replaces the one the caller exported.
    _snap_claude_oauth="${CLAUDE_CODE_OAUTH_TOKEN-}"
    # Snapshotted like the credentials above rather than via
    # _DOTENV_PROTECTED_VARS, whose mismatch warning prints the value -- this one
    # usually carries the gateway secret.
    _snap_anthropic_headers="${ANTHROPIC_CUSTOM_HEADERS-}"
    for _v in $_DOTENV_PROTECTED_VARS; do
      eval "_snap_prot_${_v}=\"\${${_v}-}\""
    done
    set -a
    # shellcheck disable=SC1091
    . "$REPO_ROOT/.env"
    set +a
    [ -n "$_snap_anthropic_url" ] && export ANTHROPIC_BASE_URL="$_snap_anthropic_url"
    [ -n "$_snap_anthropic_key" ] && export ANTHROPIC_API_KEY="$_snap_anthropic_key"
    [ -n "$_snap_anthropic_token" ] && export ANTHROPIC_AUTH_TOKEN="$_snap_anthropic_token"
    [ -n "$_snap_claude_oauth" ] && export CLAUDE_CODE_OAUTH_TOKEN="$_snap_claude_oauth"
    [ -n "$_snap_anthropic_headers" ] && export ANTHROPIC_CUSTOM_HEADERS="$_snap_anthropic_headers"
    for _v in $_DOTENV_PROTECTED_VARS; do
      eval "_snap_val=\"\${_snap_prot_${_v}-}\""
      if [ -n "${_snap_val}" ]; then
        eval "_cur_val=\"\${${_v}-}\""
        if [ "${_cur_val}" != "${_snap_val}" ]; then
          echo "[kernel-agent] .env would have overridden ${_v}=${_snap_val} with ${_cur_val}; keeping the caller's value" >&2
        fi
        eval "export ${_v}=\"\${_snap_prot_${_v}}\""
      fi
      unset "_snap_prot_${_v}"
    done
    unset _v _snap_val _cur_val
    unset _snap_anthropic_url _snap_anthropic_key _snap_anthropic_token _snap_claude_oauth _snap_anthropic_headers
    echo "[kernel-agent] loaded credentials fallback from $REPO_ROOT/.env (env wins)"
  fi
fi
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
# GEAK_REF defaults to a branch (`main`), so resolving it to a SHA hits the
# network (git ls-remote). Only do that when GEAK_ROOT was not overridden -- an
# operator-pinned root must not pay for (or fail on) a network round-trip.
if [ -z "${GEAK_ROOT:-}" ]; then
  _GEAK_SHA="$(_resolve_ref_sha "$GEAK_REPO" "$GEAK_REF")"
  GEAK_ROOT="${_open_source_root}/GEAK@${_GEAK_SHA}"
fi
GEAK_E2E_RUNNER="${GEAK_E2E_RUNNER:-${GEAK_ROOT}/interface/run_e2e.py}"
GEAK_CLAUDE_MODEL_VAL="${GEAK_CLAUDE_MODEL:-${CLAUDE_MODEL:-claude-opus-5}}"
# Run mode for the GEAKv4 Claude Code workflow. ``full`` (default) selects the
# 2 h / 5-round preset; ``quick`` selects the 1 h / 2-round smoke-test preset.
# GEAK still honours later CLI ``--mode`` or LLM-parsed task-hint overrides.
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

upsert_dotenv_var() {
  local key="$1" value="$2" tmp found=0 line stripped
  [ -n "$key" ] || return 0
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
# In --check-only mode, downgrade post-install verification failures to a
# warning so report_status can still enumerate what's missing. The caller
# explicitly asked us NOT to install; failing on the first missing piece
# defeats the point of check-only.
verify_die() {
  if [ "$CHECK_ONLY" -eq 1 ]; then warn "$1"; else die "$1"; fi
}

# Preflight credential validation. The installer gate accepts Anthropic or
# DeepSeek and intentionally does not default or require OpenAI.
#
# Strict mode by design: no bypass env var. The chained installer
# steps (the GEAK e2e optimizer) all need real credentials, so an
# install without them cannot finish anyway. The only downgrade path
# is --check-only / --dry-run, which is for introspection only and
# does not actually install.
# Reject one provider's base URL paired with only the other provider's key.
# DeepSeek serves its own Anthropic-compatible endpoint, so its key counts as one.
preflight_reject_cross_provider() {
  local a_key a_endpoint conflict=""
  a_key="${ANTHROPIC_API_KEY:-${ANTHROPIC_AUTH_TOKEN:-${CLAUDE_CODE_OAUTH_TOKEN:-}}}"
  a_endpoint="${ANTHROPIC_BASE_URL:-}"
  # A subscription token only validates against Anthropic itself, so it implies
  # the official endpoint and needs no ANTHROPIC_BASE_URL.
  if [ -z "$a_endpoint" ] && [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    a_endpoint="https://api.anthropic.com"
  fi
  if [ -n "${OPENAI_BASE_URL:-}" ] && [ -z "${OPENAI_API_KEY:-}" ] && [ -n "$a_key" ]; then
    conflict="OPENAI_BASE_URL is set without an OPENAI_API_KEY, while an Anthropic-side key is configured"
  elif [ -n "${ANTHROPIC_BASE_URL:-}" ] && [ -z "$a_key" ] && [ -n "${OPENAI_API_KEY:-}" ]; then
    conflict="ANTHROPIC_BASE_URL is set without an Anthropic-side key, while an OPENAI_API_KEY is configured"
  elif [ -n "${OPENAI_BASE_URL:-}" ] && [ -n "$a_key" ] && [ -z "$a_endpoint" ]; then
    conflict="an Anthropic-side key is configured without ANTHROPIC_BASE_URL, while the OpenAI side points at OPENAI_BASE_URL"
  # Only an explicit ANTHROPIC_BASE_URL signals a gateway-shaped deploy whose
  # OPENAI_API_KEY is likely a gateway key missing its own URL.
  elif [ -n "${ANTHROPIC_BASE_URL:-}" ] && [ -n "${OPENAI_API_KEY:-}" ] && [ -z "${OPENAI_BASE_URL:-}" ]; then
    conflict="OPENAI_API_KEY is configured without OPENAI_BASE_URL, while the Anthropic side points at ANTHROPIC_BASE_URL"
  fi
  [ -z "$conflict" ] && return 0
  if [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    warn "conflicting LLM credentials: ${conflict}; continuing because --check-only / --dry-run is active."
    return 0
  fi
  cat >&2 <<EOF
[kernel-agent ERROR] Conflicting LLM credentials: ${conflict}.

Hyperloom never borrows one provider's key or endpoint for the other. Give each
side its own base URL and key, or drop the other provider's key.
EOF
  return 1
}

preflight_validate_credentials() {
  local missing=()
  local has_anthropic=0 has_openai=0
  # Each side needs its own base URL and key; an unset side is fine.
  { [ -n "${ANTHROPIC_BASE_URL:-}" ] &&
    { [ -n "${ANTHROPIC_API_KEY:-}" ] || [ -n "${ANTHROPIC_AUTH_TOKEN:-}" ]; }; } &&
    has_anthropic=1
  # A subscription token carries its own endpoint, so it is self-consistent alone.
  [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ] && has_anthropic=1
  { [ -n "${OPENAI_BASE_URL:-}" ] && [ -n "${OPENAI_API_KEY:-}" ]; } && has_openai=1
  preflight_reject_cross_provider || return 1
  if [ "$has_anthropic" -eq 1 ] || [ "$has_openai" -eq 1 ]; then
    log "credentials preflight: at least one self-consistent provider side present"
    return 0
  fi
  missing+=("a self-consistent provider side: ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY (DeepSeek may omit the URL), or OPENAI_BASE_URL + OPENAI_API_KEY")
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
  1. Anthropic:
       export ANTHROPIC_BASE_URL=https://api.anthropic.com
       export ANTHROPIC_API_KEY=sk-ant-...
  2. DeepSeek:
       export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic
       export OPENAI_BASE_URL=https://api.deepseek.com/v1
       export ANTHROPIC_API_KEY=sk-...  OPENAI_API_KEY=sk-...
     A gateway serving only its own models also needs the model ids. Known
     hosts default themselves; otherwise set CLAUDE_MODEL (Anthropic side)
     and CODEX_MODEL (OpenAI side).
  3. Claude Max/Pro subscription (run \`claude setup-token\`):
       export CLAUDE_CODE_OAUTH_TOKEN=sk-ant-oat01-...
       # leave ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN unset
  4. Copy .env from a working worktree into this one:
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
# logging in downstream benchmark wrappers (Magpie's `*_mi*.sh` and any
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

RAY_VERSION="${RAY_VERSION:-2.44.1}"
# Ray 2.44.1's CLI currently fails during import with click >= 8.3.0.
RAY_CLI_CLICK_MAX_VERSION="${RAY_CLI_CLICK_MAX_VERSION:-8.3.0}"
RAY_INSTALL_SPEC="ray[default]==${RAY_VERSION}"
CLICK_INSTALL_SPEC="click<${RAY_CLI_CLICK_MAX_VERSION}"

ensure_ray() {
  log "ensuring ${RAY_INSTALL_SPEC} and ${CLICK_INSTALL_SPEC}"
  if [ "$CHECK_ONLY" -eq 0 ]; then
    run python3 -m pip install --quiet --no-cache-dir --break-system-packages "$CLICK_INSTALL_SPEC" "$RAY_INSTALL_SPEC"
  fi
  if [ "$DRY_RUN" -eq 0 ]; then
    RAY_VERSION="$RAY_VERSION" RAY_CLI_CLICK_MAX_VERSION="$RAY_CLI_CLICK_MAX_VERSION" python3 - <<'PY'
import importlib.metadata as md
import os
import re
import sys

import ray

RAY_VERSION = os.environ["RAY_VERSION"]
RAY_CLI_CLICK_MAX_VERSION = os.environ["RAY_CLI_CLICK_MAX_VERSION"]

def _version_tuple(version: str) -> tuple[int, int, int]:
    parts = [int(p) for p in re.findall(r"\d+", version)[:3]]
    parts.extend([0] * (3 - len(parts)))
    return tuple(parts[:3])

if ray.__version__ != RAY_VERSION:
    raise SystemExit(f"ray version mismatch: {ray.__version__} != {RAY_VERSION}")
click_version = md.version("click")
if _version_tuple(click_version) >= _version_tuple(RAY_CLI_CLICK_MAX_VERSION):
    raise SystemExit(f"click version incompatible with Ray CLI: {click_version} >= {RAY_CLI_CLICK_MAX_VERSION}")
try:
    from ray.scripts.scripts import main as _ray_cli_main  # noqa: F401
except Exception as exc:
    raise SystemExit(f"ray CLI import failed: {type(exc).__name__}: {exc}") from exc
print(f"[kernel-agent] ray version: {ray.__version__}, click version: {click_version}")
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
      if [ "$hard" -lt "$RAY_MIN_NOFILE" ]; then
        warn "fd-limit: hard nofile cap=$hard < $RAY_MIN_NOFILE; only 'docker run --ulimit nofile=1048576' lifts the hard cap (issue #433)."
      fi
      ;;
  esac
  # Never leak a non-zero status to the caller: under `set -e` the healthy path
  # (hard cap >= target) left the trailing test as the function's exit status,
  # which aborted the whole installer (issue #433).
  return 0
}

ray_head_has_serving_slot() {
  python3 - <<'PY' >/dev/null 2>&1
import ray

ray.init(address="auto", ignore_reinit_error=True, log_to_driver=False, logging_level="error")
try:
    resources = ray.cluster_resources()
finally:
    ray.shutdown()
raise SystemExit(0 if "serving_slot" in resources else 1)
PY
}

# Idempotently bring up a Ray head node. Kernel backends submit Ray tasks with
# `num_gpus>=1`; if no head is running (or one is running with --num-gpus=0)
# kernel optimization will hang forever even when GPUs are idle. We:
#   1. detect a live Ray head via `ray status`
#   2. reuse it only when it declares the Hyperloom `serving_slot` resource
#   3. otherwise force-stop stale/incompatible local Ray and start a fresh head
#      with all visible GPUs advertised
#   4. tolerate the no-GPU case (CPU-only dev box) so `--check-only` stays
#      non-fatal in environments without ROCm
_free_tcp_port() {
  # Print a currently-free loopback TCP port (probe via an ephemeral bind).
  # Echoes 0 on failure so callers fall back to Ray's default port.
  python3 - <<'PY' 2>/dev/null || echo 0
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    s.bind(("127.0.0.1", 0))
    print(s.getsockname()[1])
finally:
    s.close()
PY
}

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
    if ray_head_has_serving_slot; then
      log "ray head already running with serving_slot"
      return 0
    fi
    warn "ray head already running without serving_slot; restarting local Ray head"
  else
    log "no live ray head detected; starting one"
  fi
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
  # Bind the head to FREE, probed ports instead of Ray's fixed defaults (GCS
  # 6379, client 10001). On spur many sessions share a node's host network, so
  # the fixed ports collide: a later head attaches to an earlier/leftover head's
  # GCS and aborts with a session-name mismatch (ray-session-isolation). No
  # rendezvous is needed -- Ray records the chosen address in the container-
  # private /tmp/ray/ray_current_cluster, which ``ray status`` /
  # ``ray.init(address="auto")`` read. HL_RAY_HEAD_PORT pins the GCS port.
  local ray_port_args=()
  local ray_gcs_port ray_client_port
  ray_gcs_port="${HL_RAY_HEAD_PORT:-$(_free_tcp_port)}"
  ray_client_port="$(_free_tcp_port)"
  [ -n "$ray_gcs_port" ] && [ "$ray_gcs_port" != "0" ] && ray_port_args+=(--port="$ray_gcs_port")
  [ -n "$ray_client_port" ] && [ "$ray_client_port" != "0" ] && ray_port_args+=(--ray-client-server-port="$ray_client_port")
  log "starting ray head with --num-gpus=${num_gpus} port=${ray_gcs_port}"
  # Declare the ``serving_slot`` custom resource so serving-family GPU work
  # (baseline / profile / explore / sweep / gpu_research) routed through the
  # Ray execution backend can hold the whole-machine mutex serialising
  # GPU work. Without it those tasks request an undeclared resource and deadlock
  # PENDING forever, since ensure_ray_cluster connects to this existing head
  # instead of starting its own with the resource.
  if ! ray start --head --disable-usage-stats \
       "${ray_port_args[@]}" \
       --num-gpus="$num_gpus" --include-dashboard=false \
       --resources='{"serving_slot": 1}' >/dev/null; then
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
      # Keep this temp-clone+pin+atomic-rename in lockstep with
      # src/hyperloom/agents/kernel/tools/tracelens_analysis.py
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
  # and mirroring to $TRACELENS_MIRROR_DIR lets both
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
      verify_die "TraceLens_generate_perf_report_pytorch_inference not found after install (Hyperloom is inference-only since v0.4; reinstall TraceLens, plus the optional extension if TRACELENS_INTERNAL_ROOT is set)"
    fi
  fi
}

# Write a pod-local kernel-agent env file users should source so subsequent CLI calls
# (and Ray workers via runtime_env) pick up the upstream gateway URLs.
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
  # for the slurm/Ray paths that source the env file alone, and announces a
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
  # Recompute PYTHONPATH just before persisting it. Sourcing an existing
  # $REPO_ROOT/.env (credentials fallback above) can re-import a stale
  # PYTHONPATH that lacks REPO_ROOT, silently overwriting the value composed at
  # the top of this script. Under a ``pip install --target $REPO_ROOT`` layout
  # that dir holds the hyperloom package and is not on the default sys.path, so
  # subprocesses would fail to ``import hyperloom`` on re-install. Rebuild here
  # (REPO_ROOT first, then MAGPIE_PATH, then any remaining entries) and drop
  # duplicates so repeated installs stay idempotent.
  PYTHONPATH="$(_compose_pythonpath "${REPO_ROOT:-}" "$(_magpie_pythonpath_arg)" "${PYTHONPATH:-}")"
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
    # setsid nohup python -m hyperloom.inference_optimizer.cli optimize →
    # src/hyperloom/agents/kernel/tools/tracelens_analysis.py inherit the writable
    # mirrors instead of falling back to the read-only /path defaults.
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
    [ -n "${GEAK_CLAUDE_MODEL_VAL}" ] && echo "export GEAK_CLAUDE_MODEL='${GEAK_CLAUDE_MODEL_VAL}'"
    # Pin the claude binary the GEAK SDK path uses (else claude_agent_sdk may
    # fall back to its older bundled CLI). run_e2e.py maps this to cli_path.
    _geak_claude_bin=""
    local _probe_home
    _probe_home="$(_home_dir || true)"
    for _c in "${_probe_home:+${_probe_home}/.local/bin/claude}" "/usr/local/bin/claude" "$(command -v claude 2>/dev/null || true)"; do
      if [ -n "${_c}" ] && [ -x "${_c}" ]; then _geak_claude_bin="${_c}"; break; fi
    done
    [ -n "${_geak_claude_bin}" ] && echo "export GEAK_CLAUDE_BIN='${_geak_claude_bin}'"
    # e2e optimizer budget mode (read by the inference_optimizer kernel request
    # handler to pick the backend budget default) + LLM connection for the runner.
    [ -n "${GEAK_RUN_MODE_VAL}" ] && echo "export GEAK_RUN_MODE='${GEAK_RUN_MODE_VAL}'"
    # GEAK_API_KEY / GEAK_BASE_URL are intentionally NOT emitted: they derive
    # from Anthropic/DeepSeek values and would reintroduce
    # cross-provider leakage into the sourced runtime env.
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

  [ -n "${USER_DATA_PATH:-}" ] && upsert_dotenv_var USER_DATA_PATH "$USER_DATA_PATH"
  [ -n "${HYPERLOOM_RUNTIME_DIR:-}" ] && upsert_dotenv_var HYPERLOOM_RUNTIME_DIR "$HYPERLOOM_RUNTIME_DIR"
  [ -n "${KERNEL_AGENT_ENV:-}" ] && upsert_dotenv_var KERNEL_AGENT_ENV "$KERNEL_AGENT_ENV"
  [ -n "${HYPERLOOM_KERNEL_AGENT_ROOT:-}" ] && upsert_dotenv_var HYPERLOOM_KERNEL_AGENT_ROOT "$HYPERLOOM_KERNEL_AGENT_ROOT"
  [ -n "${KERNEL_AGENT_ROOT:-}" ] && upsert_dotenv_var KERNEL_AGENT_ROOT "$KERNEL_AGENT_ROOT"
  [ -n "${MAGPIE_PATH:-}" ] && upsert_dotenv_var MAGPIE_PATH "$MAGPIE_PATH"
  [ -n "${MAGPIE_PYTHON:-}" ] && upsert_dotenv_var MAGPIE_PYTHON "$MAGPIE_PYTHON"
  [ -n "${PYTHONPATH:-}" ] && upsert_dotenv_var PYTHONPATH "$PYTHONPATH"
  [ -n "${INFERENCEX_PATH:-}" ] && upsert_dotenv_var INFERENCEX_PATH "$INFERENCEX_PATH"
  # Persist each provider's own canonical vars (write when present, clear when
  # absent) so an operator's .env creds survive a re-install. Never cross-write
  # and never persist gateway aliases, so the two providers stay separated.
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
  [ -n "${TRACELENS_ROOT:-}" ] && upsert_dotenv_var TRACELENS_ROOT "$TRACELENS_ROOT"
  if [ -n "${TRACELENS_INTERNAL_ROOT:-}" ]; then
    upsert_dotenv_var TRACELENS_INTERNAL_ROOT "$TRACELENS_INTERNAL_ROOT"
    upsert_dotenv_var TL_EXTENSION "TraceLens_internal"
  fi
  [ -n "${HYPERLOOM_ROOT:-}" ] && upsert_dotenv_var HYPERLOOM_ROOT "$HYPERLOOM_ROOT"
  [ -n "${GEAK_E2E_RUNNER}" ] && upsert_dotenv_var GEAK_E2E_RUNNER "$GEAK_E2E_RUNNER"
  [ -n "${GEAK_ROOT}" ] && upsert_dotenv_var GEAK_ROOT "$GEAK_ROOT"
  [ -n "${GEAK_CLAUDE_MODEL_VAL}" ] && upsert_dotenv_var GEAK_CLAUDE_MODEL "$GEAK_CLAUDE_MODEL_VAL"
  [ -n "${_geak_claude_bin}" ] && upsert_dotenv_var GEAK_CLAUDE_BIN "$_geak_claude_bin"
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
    _PIP_FLAGS="-q --no-cache-dir --break-system-packages"
    # GEAK is a pip package now: install from the checkout above so the package
    # matches the interface/run_e2e.py we run and honours any GEAK_REPO/GEAK_REF
    # override (local mirror, fork, SSH URL). Its bootstrap installs deps + the
    # Claude Code CLI (>= 2.1.177); GEAK_HOME reuses our checkout so bootstrap
    # skips a second clone.
    if [ -f "${GEAK_ROOT}/pyproject.toml" ] || [ -f "${GEAK_ROOT}/setup.py" ]; then
      run env GEAK_HOME="${GEAK_ROOT}" python3 -m pip install ${_PIP_FLAGS} "${GEAK_ROOT}" || \
        warn "GEAK pip install failed; Claude Code may be < 2.1.177"
    else
      warn "GEAK package metadata missing at ${GEAK_ROOT}; skipping pip install (Claude Code may be < 2.1.177)"
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

# The forge backend drives the `claude` CLI inside its autonomous loop
# (see forge_submit._apply_kernel_backend_env), so it needs Node/npm, the claude npm
# CLI, and ~/.claude auth.
ensure_forge_claude_cli() {
  log "ensuring claude CLI for the forge backend"
  if [ "$CHECK_ONLY" -eq 1 ]; then
    command -v claude >/dev/null 2>&1 || warn "claude CLI missing; forge backend will fail to drive its kernel_backend"
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
  # Claude Code CLI install. $HYPERLOOM_CLAUDE_CODE_VERSION pins a specific npm
  # version and FORCE-reinstalls it (overriding one already baked into the base
  # image) — needed because newer claude-code releases reject models the gateway
  # still serves (e.g. retired Opus 4). When unset, keep the legacy behaviour:
  # install the latest only when the CLI is absent.
  if command -v npm >/dev/null 2>&1; then
    # A global install needs a writable prefix. Off root /usr/local is not one,
    # and `run` has no failure path, so this would abort the whole installer.
    local _npm_prefix="/usr/local" _npm_home
    if [ ! -w /usr/local/lib ]; then
      _npm_home="$(_home_dir || true)"
      if [ -z "$_npm_home" ]; then
        warn "no writable npm prefix (/usr/local and HOME both unavailable); forge claude CLI unavailable"
        return 0
      fi
      # ~/.local/bin is where the CLI probe in write_env_file already looks.
      _npm_prefix="${_npm_home}/.local"
      mkdir -p "${_npm_prefix}/lib" "${_npm_prefix}/bin"
    fi
    if [ -n "${HYPERLOOM_CLAUDE_CODE_VERSION:-}" ]; then
      run npm config set prefix "${_npm_prefix}"
      run npm install -g "@anthropic-ai/claude-code@${HYPERLOOM_CLAUDE_CODE_VERSION}"
    elif ! command -v claude >/dev/null 2>&1; then
      run npm config set prefix "${_npm_prefix}"
      run npm install -g @anthropic-ai/claude-code
    fi
  fi
  # ~/.claude authenticates the Claude Code CLI for Anthropic-compatible flows.
  # Its readers resolve Path.home(), so ~ must come from _home_dir here too.
  local _claude_key="${_ANTHROPIC_KEY_VAL:-}"
  if [ -n "$_claude_key" ]; then
    local _claude_home
    _claude_home="$(_home_dir || true)"
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
  _prune_dep_cache "TraceLens" "GEAK"
  ensure_forge_claude_cli
  write_env_file

  report_status
  log "install complete"
}

main "$@"
