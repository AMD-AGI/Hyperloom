#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Inference Optimizer installer.
#
# Owns the inference_optimizer-side bare-image setup so SKILL.md does
# not have to hand-roll it. Idempotent — every step skips if the
# artifact is already present.
#
# Stack (in order):
#   1. inference_optimizer + extras (pulls in claude_agent_sdk via
#      pyproject `[test]` extra)
#   2. Magpie (benchmark engine) pip-installed from MAGPIE_PACKAGE_SPEC,
#      pinned to MAGPIE_REF (a commit SHA or tag)
#   2b. Atomic-write patch for Magpie._prepare_benchmark_scripts
#       (root-cause fix for the Hyperloom #C1 script-tearing race;
#       fail-soft — a no-op when the MAGPIE_REF target already has
#       upstream atomic copying)
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

setup_dotenv_is_authoritative() {
  [ -f "$REPO_ROOT/.env" ] || return 1
  grep -q '^HYPERLOOM_RUN_MODE=' "$REPO_ROOT/.env" 2>/dev/null
}

scrub_stale_workspace_env_for_setup_dotenv() {
  setup_dotenv_is_authoritative || return 0
  unset USER_DATA_PATH
  unset HYPERLOOM_RUNTIME_DIR
  unset KERNEL_AGENT_ENV
  unset HYPERLOOM_ROOT
  unset HYPERLOOM_KERNEL_AGENT_ROOT
  unset KERNEL_AGENT_ROOT
  unset FRAMEWORK_AGENT_ROOT
  unset HYPERLOOM_SKILL_PATH
  unset PYTHONPATH
}

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
scrub_stale_workspace_env_for_setup_dotenv
load_dotenv_no_clobber
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
# Resolve a git ref to a commit SHA: 7-40 hex passes through; branch/tag via
# ls-remote (falls back to the raw ref). The SHA keys the per-revision cache.
_resolve_ref_sha() {
  local repo="$1" ref="$2" sha=""
  if [[ "$ref" =~ ^[0-9a-fA-F]{7,40}$ ]]; then
    printf '%s' "$ref"
    return 0
  fi
  sha="$(git ls-remote "$repo" "$ref" 2>/dev/null | awk 'NR==1{print $1}')"
  if [ -z "$sha" ]; then
    # Loud, not silent: a raw-ref cache key drops the per-revision guarantee.
    echo "[inference-optimizer WARN] could not resolve '$ref' at $repo to a commit SHA (network or bad ref); using '$ref' as the per-revision cache key -- stale-checkout guard weakened. Pin *_REF to a 40-hex SHA or restore network access." >&2
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

MAGPIE_REPO="${MAGPIE_REPO:-https://github.com/AMD-AGI/Magpie.git}"
# Pin Magpie to a release commit/tag instead of the default branch. Operators can
# re-pin with MAGPIE_REF=<tag|sha>.
MAGPIE_REF="${MAGPIE_REF:-e6833b8183c6c41adf6038252337550876ca0433}"
MAGPIE_PACKAGE_SPEC="${MAGPIE_PACKAGE_SPEC:-magpie-eval @ git+${MAGPIE_REPO}@${MAGPIE_REF}}"

# aiperf (SemiAnalysis AgentX benchmark client) — pinned to an immutable commit
# on the SemiAnalysisAI org repo. Only needed for AgentX mode (HYPERLOOM_AGENTX).
# Installed fail-soft (see ensure_aiperf): a failure must never block the default
# synthetic path. Operators can override AIPERF_BIN to skip this install.
# Kept in lockstep with INFERENCEX_REF below: this is exactly the aiperf commit
# that InferenceX@${INFERENCEX_REF} carries as its utils/aiperf submodule. The
# leaderboard's scenario invariants live in aiperf, so a mismatched pair silently
# benchmarks against different rules. The previous pin (dc975aaa) predates the
# 062126 corpus and its scenario allowlist rejects it outright.
AIPERF_REPO="${AIPERF_REPO:-https://github.com/SemiAnalysisAI/aiperf.git}"
AIPERF_REF="${AIPERF_REF:-754356e9a39acc6cc6afb242d123bb57c3fb6f75}"
AIPERF_PACKAGE_SPEC="${AIPERF_PACKAGE_SPEC:-aiperf @ git+${AIPERF_REPO}@${AIPERF_REF}}"
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
# Re-pinned to the leaderboard's current head so AgentX replays the same
# scenario, corpus generation and warmup contract the published rows were
# produced with. Keep AIPERF_REF above in lockstep (it is this commit's
# utils/aiperf submodule); re-sync when the corpus generation changes.
INFERENCEX_REF="${INFERENCEX_REF:-3d5581562f643f9bdeb8410cd924e2c70906c966}"
_INFERENCEX_SHA="$(_resolve_ref_sha "$INFERENCEX_REPO" "$INFERENCEX_REF")"
INFERENCEX_DEFAULT_DIR="${INFERENCEX_DEFAULT_DIR:-${_open_source_root}/InferenceX@${_INFERENCEX_SHA}}"

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
  - Clones InferenceX pinned to INFERENCEX_REF and exports INFERENCEX_PATH
  - Chains to src/hyperloom/agents/kernel/scripts/install.sh for Ray + ray-head start,
    TraceLens, GEAK, and LLM gateway env.
  - The `fa` CLI is provided by this same editable install; framework-agent
    lives in src/hyperloom/agents/framework/ and has no separate
    installer/venv to chain to.

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
  PR is already merged into your clone),
  MAGPIE_EVAL_FLAG_STRICT (=1; abort when the redundant
    --concurrent-requests eval flag cannot be removed from a Magpie
    benchmark script. Set 0 only when GSM8K accuracy eval is not
    required — with the flag live, every RUN_EVAL=true baseline aborts)
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
# a usable setup needs at least one self-consistent provider side. A
# dual-protocol gateway such as DeepSeek configures both sides on one host.
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

# Translate a retired DEEPSEEK_* configuration into the standard variables.
# DeepSeek is a dual-protocol gateway, not a third provider: /anthropic speaks
# the Anthropic API and /v1 speaks OpenAI chat-completions, both with the same
# key. Endpoint and model derivation matches
# hyperloom.common.llm_config.deepseek_compat_env.
normalize_legacy_deepseek_env() {
  [ -n "${DEEPSEEK_API_KEY:-}" ] || [ -n "${DEEPSEEK_BASE_URL:-}" ] || return 0
  # Adopt the gateway whole or not at all. Anything already on the Anthropic
  # side means the retired variables are stale leftovers: half-adopting them
  # would send an explicit Anthropic credential to DeepSeek's host.
  if [ -n "${ANTHROPIC_BASE_URL:-}" ] || [ -n "${ANTHROPIC_API_KEY:-}" ] \
     || [ -n "${ANTHROPIC_AUTH_TOKEN:-}" ] || [ -n "${CLAUDE_CODE_OAUTH_TOKEN:-}" ]; then
    warn "DEEPSEEK_* is retired and ignored here: the Anthropic side is already configured"
    return 0
  fi
  local base lowered anthropic_url openai_url model
  base="${DEEPSEEK_BASE_URL:-}"
  base="${base%/}"
  # Case-insensitive match (AMD spells it /Anthropic); ${base%/*} drops the
  # final segment whatever its casing.
  lowered="$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]')"
  case "$lowered" in
    "")           anthropic_url="https://api.deepseek.com/anthropic"; openai_url="https://api.deepseek.com/v1" ;;
    */anthropic)  anthropic_url="$base"; openai_url="${base%/*}/v1" ;;
    */v1)         anthropic_url="${base%/*}/anthropic"; openai_url="$base" ;;
    https://api.deepseek.com|http://api.deepseek.com)
                  anthropic_url="${base}/anthropic"; openai_url="${base}/v1" ;;
    *)            anthropic_url="$base"; openai_url="${base}/v1" ;;
  esac
  model="${DEEPSEEK_MODEL:-deepseek-v4-pro}"
  export ANTHROPIC_BASE_URL="$anthropic_url"
  [ -n "${DEEPSEEK_API_KEY:-}" ] && export ANTHROPIC_API_KEY="$DEEPSEEK_API_KEY"
  [ -n "${CLAUDE_MODEL:-}" ] || export CLAUDE_MODEL="$model"
  # GEAKv4 follows whichever Claude model is actually in effect.
  [ -n "${GEAK_CLAUDE_MODEL:-}" ] || export GEAK_CLAUDE_MODEL="${CLAUDE_MODEL:-$model}"
  # The OpenAI side is adopted only when it is entirely free; otherwise some
  # other gateway already runs there and keeps its own key and model.
  if [ -z "${OPENAI_BASE_URL:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    export OPENAI_BASE_URL="$openai_url"
    [ -n "${DEEPSEEK_API_KEY:-}" ] && export OPENAI_API_KEY="$DEEPSEEK_API_KEY"
    [ -n "${CODEX_MODEL:-}" ] || export CODEX_MODEL="$model"
  fi
  unset DEEPSEEK_API_KEY DEEPSEEK_BASE_URL DEEPSEEK_MODEL
  warn "DEEPSEEK_* is retired; normalized to ANTHROPIC_*/OPENAI_*"
}

# Reject one provider's base URL paired with only the other provider's key.
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
[inference-optimizer ERROR] Conflicting LLM credentials: ${conflict}.

Hyperloom never borrows one provider's key or endpoint for the other. Give each
side its own base URL and key, or drop the other provider's key.
EOF
  return 1
}

preflight_validate_credentials() {
  preflight_load_dotenv
  normalize_legacy_deepseek_env
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
  missing+=("a self-consistent provider side: ANTHROPIC_BASE_URL + ANTHROPIC_API_KEY, or OPENAI_BASE_URL + OPENAI_API_KEY")
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
  2. A dual-protocol gateway such as DeepSeek (same key, both sides):
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
      # The bare wheel ships with empty base deps, so the runtime set is
      # installed here: `llm` (Coordinator backends; openai-codex is the agent
      # runtime an OpenAI-only deployment needs to run at all) and `forge` (the
      # built-in kernel-opt agent, which ships in this wheel and used to be
      # installed separately from a KernelForge checkout, and which pulls
      # `llm` itself). PyYAML rides in on `forge`.
      #
      # Named by EXTRA, not by a hand-copied pin list. The list this replaces
      # restated seven specifiers from pyproject.toml with no sync mechanism:
      # raising a lower bound there left the packaged install path silently
      # holding the old one. pip resolves `[llm,forge]` against the already
      # installed distribution -- verified to need no index for the top-level
      # package -- so this is a metadata read, not a reinstall.
      "$PYTHON" -m pip install --quiet "${PIP_EXTRA[@]}" \
        "hyperloom-inference_optimizer[llm,forge]"
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
    _check_kernelforge_ready
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
  _check_kernelforge_ready
}

# Readiness probe for the built-in kernel-opt agent. It replaces the step that
# pip-installed forge as a separate distribution from a KernelForge checkout
# found via an env pointer: there is no checkout and no separate distribution
# any more, so there is nothing to install -- only something to verify. The
# three imports are the ones whose absence used to be found late:
#   kernelforge.cli      forge-loop's entry point (the 2026-07-28 ModuleNotFoundError)
#   kernelforge.fusion   forge-fuse, which imports fine only if the tree is complete
#   openai_codex         the agent runtime for an OpenAI-only deployment; without
#                        it the provider fallback silently becomes a claude run
#                        that dies at its first turn on "Not logged in"
_check_kernelforge_ready() {
  if "$PYTHON" -c "import kernelforge.cli, kernelforge.fusion" >/dev/null 2>&1; then
    log "kernelforge (built-in kernel-opt agent) OK"
  else
    warn "kernelforge not importable after install; forge kernel attempts will fail"
    [ "$CHECK_ONLY" -eq 1 ] || die "kernelforge missing"
  fi
  if "$PYTHON" -c "import openai_codex" >/dev/null 2>&1; then
    log "openai_codex OK"
  else
    warn "openai_codex not importable; an OpenAI-only deployment cannot construct the forge codex provider"
  fi
}

# --- 1b. forge GEMM tuning (`kernelforge gemm-tune`) ---
# Nothing to install: the tuner is a subpackage of the kernelforge that ships in
# this distribution, so `pip install -e "${REPO_ROOT}[test]"` above already put
# it in place. It used to be its own wheel, resolved from a checkout via
# FORGE_GEMM_TUNE_ROOT / FORGE_PATH and pip-installed editable on the side; that
# whole resolver is gone with the separate distribution. What remains is worth
# keeping as a probe, because a partial install shows up here rather than in the
# middle of a tuning run.
ensure_forge_gemm_tune() {
  # Ask whether the subcommand is REGISTERED, not merely whether the module
  # imports. Tuning runs as `python -m kernelforge.cli gemm-tune run`, and a
  # tree whose module imports fine while the command never registered passes an
  # import probe and then dies mid-run on "No such command 'gemm-tune'".
  # Registration happens at import, so main.commands is populated by then.
  # (Absorbed from the FORGE_PATH-era probe this replaced, which learned the
  # same lesson against a checkout instead of against a packaged subpackage.)
  local probe='
import sys
import kernelforge.gemm_tune.cli  # noqa: F401
from kernelforge.cli import main
sys.exit(0 if "gemm-tune" in getattr(main, "commands", {}) else 1)
'
  if "$PYTHON" -c "$probe" >/dev/null 2>&1; then
    log "kernelforge gemm-tune OK (subcommand registered)"
  else
    # die, not warn. When the tuner was a separate distribution resolved from a
    # checkout, a miss meant "that optional side-install did not happen" and
    # degrading was right. It now ships in the same wheel as everything else
    # this script just verified, so a miss means that wheel is incomplete --
    # the one condition an install script exists to refuse.
    die "kernelforge gemm-tune is not runnable, but it ships in this wheel; the install is incomplete"
  fi
}

# --- 1c. rocprof-compute (rocprofiler-compute) for the forge profiling stage ---
# forge-loop's profiling stage prefers rocprof-compute (roofline / speed-of-light)
# and only falls back to the thin rocprofv3 "PMC" path when the tool is absent.
# KernelForge's resolve_rocpc() looks for the tool at
# `<ROCM_PATH>/libexec/rocprofiler-compute/rocprof_compute_base.py`; the stock
# vllm/sglang ROCm serving images ship rocprofv3 but NOT rocprofiler-compute, so
# every forge run silently degrades to PMC (no roofline -> optimization-potential
# is always estimable=NO). Two pieces are needed: the profiler's Python deps
# (the `forge-profiling` extra, Step 0) and the system tool itself, which pip
# cannot provide — it comes from the ROCm apt package (Step 1).
#
# This step is FAIL-SOFT by design: forge still works on the PMC path, so a
# missing/failed rocprof-compute must NOT abort the install. Every branch logs
# (with the concrete "profiling will degrade to PMC" consequence) so the
# post-mortem observability question — "why did this run profile on PMC?" — is
# answerable from the install log alone.
# rocprof-compute's CSV converter (utils/utils.py, v3->v2) assumes pandas'
# legacy 'object' string dtype. pandas>=3.0 defaults future.infer_string=True,
# so the rocprofv3 counter CSV's Agent_Id ("Agent 9") is read as the new
# StringDtype; the converter's `dtype == "object"` guard then skips its int
# coercion and the subsequent Agent_Id<->Node_Id merge dies ("merge on str and
# int64"). Every counter file is dropped -> rocprof-compute reports "No
# profiling data found" -> forge silently degrades to the PMC path (no roofline;
# optimization-potential estimable=NO). Nothing else in the stack needs
# pandas>=3 (verified: no installed dist requires it), so <3 is conflict-free.
#
# rocprof-compute runs under the interpreter KernelForge's resolve_rocpc() picks:
# it probes sys.executable, then /usr/bin/python3, then `python3` on PATH, and
# uses the FIRST that can run `rocprof-compute --help`. We mirror that probe and
# pin pandas in exactly THAT interpreter (not blindly $PYTHON), so the pin cannot
# be a silent no-op and the log names the env that will actually run the tool.
#
# Two known, accepted deltas vs. resolve_rocpc() (neither is a correctness bug —
# both fall back safely and only affect which interpreter gets pinned):
#   * First candidate: we probe $PYTHON where resolve_rocpc probes the RUNTIME
#     sys.executable. $PYTHON is install-time sys.executable and, in the shared
#     -venv carrier flow, is the SAME interpreter forge runs under, so they agree.
#     If a deployment splits install-time and runtime Python, the pin may land on
#     a non-preferred interpreter — the fallback + warn below make that visible.
#   * `--help` passing proves rocprof-compute's deps import, NOT that its CSV
#     conversion works on this pandas; the pandas<3 pin (below) is what closes
#     that gap. A deeper check (pandas major inside the probe) would belong in
#     KernelForge's resolve_rocpc(), not here.
#
# Fail-soft: a pin failure must NOT abort the install — forge still runs on PMC.

# Echo the interpreter resolve_rocpc() will run rocprof-compute under: the first
# of $PYTHON (install-time sys.executable), /usr/bin/python3, PATH python3 that
# can run `<libexec>/rocprof-compute --help`. Non-zero + no output if none do.
_rocpc_effective_python() {
  local libexec="$1" py seen=" "
  for py in "$PYTHON" /usr/bin/python3 "$(command -v python3 2>/dev/null || true)"; do
    [ -n "$py" ] || continue
    case "$seen" in *" $py "*) continue ;; esac
    seen="${seen}${py} "
    if "$py" "${libexec}/rocprof-compute" --help >/dev/null 2>&1; then
      printf '%s\n' "$py"
      return 0
    fi
  done
  return 1
}

# Print pandas version under $1 and exit: 0 => <3 (ok); 1 => >=3; 3 => absent.
# Decided in Python (robust vs. bash numeric parsing under set -euo pipefail).
_pandas_major_ge3() {
  "$1" - <<'PY'
import sys
try:
    import pandas
except Exception:
    sys.exit(3)
print(pandas.__version__)
sys.exit(1 if int(pandas.__version__.split(".")[0]) >= 3 else 0)
PY
}

_ensure_pandas_lt3_for_rocpc() {
  local py="${1:-$PYTHON}" ver rc why
  if ver="$(_pandas_major_ge3 "$py")"; then rc=0; else rc=$?; fi

  if [ "$rc" -eq 0 ]; then
    log "rocprof-compute: pandas ${ver} is <3 under ${py} (compatible with rocprof-compute's CSV converter); no pin needed"
    return 0
  fi

  # rc==1 (pandas>=3) OR rc==3 (pandas absent): pin pandas<3 in the interpreter
  # that runs rocprof-compute. This runs LAST in install.sh (after
  # chain_kernel_agent — the final pip-installing step), so no later step can
  # re-pull pandas>=3, and the re-check below is the final, truthful state.
  if [ "$rc" -eq 3 ]; then
    why="pandas not yet installed"
  else
    why="pandas ${ver} (>=3) breaks rocprof-compute's CSV converter (Agent_Id StringDtype)"
  fi

  if [ "$CHECK_ONLY" -eq 1 ]; then
    warn "rocprof-compute: ${why} (interpreter ${py}); check-only — would pin 'pandas>=2.2.3,<3'. Until fixed, forge profiling degrades to the PMC path."
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would run: ${py} -m pip install 'pandas>=2.2.3,<3'  (${why})"
    return 0
  fi

  log "rocprof-compute: ${why}; installing 'pandas>=2.2.3,<3' into ${py}"
  "$py" -m pip install --quiet "${PIP_EXTRA[@]}" 'pandas>=2.2.3,<3' \
    || warn "rocprof-compute: 'pip install pandas>=2.2.3,<3' failed under ${py}; forge profiling will stay on the PMC path. Check pip/network."

  # Re-check the SAME interpreter, at the END of install.sh, so the logged
  # outcome reflects what forge will actually import at runtime.
  if ver="$(_pandas_major_ge3 "$py")"; then rc=0; else rc=$?; fi
  if [ "$rc" -eq 0 ]; then
    log "rocprof-compute: pandas ${ver} in ${py}; forge profiling can use rocprof-compute (roofline)"
  else
    warn "rocprof-compute: pandas still incompatible in ${py} (version='${ver}', rc=${rc}); forge profiling will degrade to the PMC path."
  fi
  return 0
}

ensure_rocprof_compute() {
  # UNCONDITIONAL, not gated on KERNEL_OPT_BACKEND_ORDER: install.sh runs at
  # setup time under the default geak backend — the carrier sets
  # KERNEL_OPT_BACKEND_ORDER=forge only later on the optimize command, AFTER
  # install.sh has finished (_incontainer.sh) — so a backend gate here would skip
  # the install and a later forge session would still profile on the PMC path.
  # rocprof-compute (~11 MB) + pandas<3 are only useful for forge but harmless
  # otherwise (pandas<3 is conflict-free), so running always is the safe,
  # ordering-independent choice. The backend value is logged for context only.
  #
  # This used to be gated on the presence of a KernelForge checkout. Vendoring
  # forge in removed the checkout, which would have turned the gate into a
  # permanent skip: roofline profiling silently uninstalled on every pod.
  log "rocprof-compute: ensuring roofline profiling deps (KERNEL_OPT_BACKEND_ORDER='${KERNEL_OPT_BACKEND_ORDER:-}')"

  local rocm_root base
  rocm_root="${ROCM_PATH:-/opt/rocm}"
  base="${rocm_root%/}/libexec/rocprofiler-compute/rocprof_compute_base.py"

  # --- Step 0: the profiler's Python dependencies ---
  # The tool is a Python program: without dash/kaleido/matplotlib/plotille/tqdm
  # and friends it does not run at all. These live in the `forge-profiling`
  # extra. The comment above used to claim the KernelForge root install pulled
  # them in as base deps; it did not — they were in KernelForge's own
  # `profiling` extra, which that install never requested, so this has been
  # missing on every pod. Fail-soft like the rest of this function.
  # `-e`, matching the `pip install -e "${REPO_ROOT}[test]"` further up. pip
  # compares the editable marker in direct_url.json, so a non-editable install
  # of the same local path is a *mismatch* and pip reinstalls: the editable
  # install is replaced by a copy, source edits stop taking effect, and every
  # setup pays a full wheel build of a tree that vendoring doubled in size.
  # Asking for the same shape leaves the install in place and resolves only
  # the extra.
  #
  # Installed by default, with an opt-out rather than an opt-in: an opt-in
  # reproduces the bug this block exists to fix -- profiling silently degrading
  # to the PMC path on every pod that did not know to ask. `SKIP_FORGE_PROFILING=1`
  # is for environments that cannot afford ~20 extra wheels (kaleido and
  # astunparse are exact pins carried over from rocprofiler-compute's own
  # requirements.txt), or that already have them.
  if [ "${SKIP_FORGE_PROFILING:-0}" = "1" ]; then
    log "rocprof-compute: SKIP_FORGE_PROFILING=1 — skipping the forge-profiling extra; profiling degrades to the PMC path"
  elif [ "$CHECK_ONLY" -eq 1 ]; then
    warn "rocprof-compute: check-only — would install -e '${REPO_ROOT}[forge-profiling]'"
  elif [ "$DRY_RUN" -eq 1 ]; then
    log "would run: ${PYTHON} -m pip install -e '${REPO_ROOT}[forge-profiling]'"
  elif [ -n "${REPO_ROOT:-}" ] && [ -f "${REPO_ROOT%/}/pyproject.toml" ]; then
    "$PYTHON" -m pip install --quiet "${PIP_EXTRA[@]}" -e "${REPO_ROOT}[forge-profiling]" \
      || warn "rocprof-compute: installing the forge-profiling extra failed; profiling will degrade to the PMC path. Check pip/network."
  else
    # No source tree to extend, so name the extra's requirements rather than
    # the distribution: `pip install hyperloom-inference_optimizer[...]` would
    # resolve the *distribution* against an index and could overwrite the
    # installation now running with a published build of a different version.
    # Reading Requires-Dist off the installed metadata asks for exactly the
    # profiling dependencies and can touch nothing else.
    local reqs
    reqs="$("$PYTHON" -c '
from importlib.metadata import PackageNotFoundError, requires
from packaging.requirements import Requirement
try:
    specs = requires("hyperloom-inference_optimizer") or []
except PackageNotFoundError:
    raise SystemExit(1)
for spec in specs:
    req = Requirement(spec)
    if req.marker and req.marker.evaluate({"extra": "forge-profiling"}):
        req.marker = None
        print(str(req))
' 2>/dev/null)" || reqs=""
    if [ -n "$reqs" ]; then
      # shellcheck disable=SC2086 -- one requirement per line, split intended
      "$PYTHON" -m pip install --quiet "${PIP_EXTRA[@]}" $reqs \
        || warn "rocprof-compute: installing the forge-profiling extra failed; profiling will degrade to the PMC path. Check pip/network."
    else
      warn "rocprof-compute: could not read the forge-profiling requirements from installed metadata; profiling will degrade to the PMC path"
    fi
  fi

  # --- Step 1: ensure the rocprof-compute tool exists ---
  # It is a ROCm system package (pip cannot provide it). Idempotent: skip the apt
  # install when the file KernelForge's resolve_rocpc() checks is already present.
  if [ -f "$base" ]; then
    log "rocprof-compute already present at ${base}"
  elif [ "$CHECK_ONLY" -eq 1 ]; then
    warn "rocprof-compute not found at ${base} (check-only; would apt-get install rocprofiler-compute). Forge profiling would degrade to the PMC path."
  elif [ "$DRY_RUN" -eq 1 ]; then
    log "would run: apt-get install -y --no-install-recommends rocprofiler-compute"
  elif ! command -v apt-get >/dev/null 2>&1; then
    # No apt (RHEL/Alpine/etc.): cannot install the system package here.
    warn "rocprof-compute: apt-get unavailable; cannot install rocprofiler-compute. Forge profiling will degrade to the PMC path (no roofline; optimization-potential estimable=NO). Bake rocprofiler-compute into the image to enable roofline profiling."
  else
    log "installing rocprofiler-compute (forge profiling backend) via apt into ${rocm_root}"
    # Fail-soft throughout: never let apt failures (locked dpkg, offline mirror,
    # missing package) abort the install. Capture output to a log so a failure is
    # diagnosable (do not swallow apt errors). Try once, refresh index, retry.
    local apt_log="${TMPDIR:-/tmp}/rocpc_apt_$$.log"
    export DEBIAN_FRONTEND=noninteractive
    if ! apt-get install -y --no-install-recommends rocprofiler-compute >"$apt_log" 2>&1; then
      apt-get update -qq >>"$apt_log" 2>&1 || true
      apt-get install -y --no-install-recommends rocprofiler-compute >>"$apt_log" 2>&1 || true
    fi
    # Verify against the SAME path KernelForge's resolve_rocpc() checks.
    if [ -f "$base" ]; then
      log "rocprof-compute installed OK: ${base} present"
    else
      warn "rocprof-compute install did not produce ${base}; forge profiling will degrade to the PMC path (no roofline; optimization-potential estimable=NO). apt output tail (check ROCm repo access / package name for this ROCm version):"
      # Guard BOTH the missing-file case and pipefail: if the redirect above never
      # created $apt_log (e.g. an unwritable TMPDIR), a bare `tail | while` exits
      # non-zero and set -euo pipefail would abort install.sh — the very
      # fail-soft invariant this diagnostic exists to serve. Only tail when the
      # file exists, and swallow any residual pipe failure.
      if [ -f "$apt_log" ]; then
        tail -n 6 "$apt_log" 2>/dev/null | while IFS= read -r _ln; do warn "  apt| ${_ln}"; done || true
      fi
    fi
    rm -f "$apt_log" 2>/dev/null || true
  fi

  # --- Step 2: pin pandas<3 for rocprof-compute's CSV converter ---
  # Pin in the interpreter resolve_rocpc() will actually run the tool under (probe
  # mirrors KernelForge). Runs when the tool is present; in check/dry-run we
  # surface the plan against $PYTHON even before the tool exists.
  if [ -f "$base" ]; then
    local rocpc_py
    if rocpc_py="$(_rocpc_effective_python "$(dirname "$base")")"; then
      [ "$rocpc_py" = "$PYTHON" ] \
        || log "rocprof-compute: resolve_rocpc will run under ${rocpc_py} (not \$PYTHON=${PYTHON}); pinning pandas there"
    else
      rocpc_py="$PYTHON"
      warn "rocprof-compute: could not confirm which interpreter runs 'rocprof-compute --help'; pinning pandas in \$PYTHON=${PYTHON} (best effort — verify forge's runtime interpreter has pandas<3)"
    fi
    _ensure_pandas_lt3_for_rocpc "$rocpc_py"
  elif [ "$CHECK_ONLY" -eq 1 ] || [ "$DRY_RUN" -eq 1 ]; then
    _ensure_pandas_lt3_for_rocpc "$PYTHON"
  fi
  return 0
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

# --- 2a. aiperf (AgentX benchmark client) — fail-soft, AgentX-only ---
# Installs the pinned aiperf for HYPERLOOM_AGENTX; only reached when the caller
# opts in (see the INSTALL_AIPERF / HYPERLOOM_AGENTX gate at the call site). A
# failure here is NON-fatal: it only warns and leaves aiperf absent, so the
# default synthetic path is never blocked by an AgentX-only dependency. Skipped
# when the operator points AIPERF_BIN at their own build.
ensure_aiperf() {
  if [ -n "${AIPERF_BIN:-}" ]; then
    log "AIPERF_BIN set (${AIPERF_BIN}); skipping aiperf install"
    return 0
  fi
  if [ "$CHECK_ONLY" -eq 1 ]; then
    if command -v aiperf >/dev/null 2>&1; then log "aiperf on PATH"; else warn "aiperf not found (check-only)"; fi
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "would pip install aiperf (AgentX): ${AIPERF_PACKAGE_SPEC}"
    return 0
  fi
  # Presence is not enough. Measured: the previous pin (aiperf 0.8.0) carries
  # weka-trace, --scenario and --benchmark-duration, and defines a scenario by
  # the same name -- but with different invariants and an allowlist that predates
  # the current corpus. A presence-only skip therefore left every already
  # provisioned box on the old build after a pin bump, silently, while the
  # preflight's own advice ("install via install.sh") pointed back at this no-op.
  # Record what we installed and reinstall when it no longer matches.
  local stamp="${HYPERLOOM_STATE_DIR:-${HOME}/.hyperloom}/aiperf_installed_ref"
  local -a pip_args=("${PIP_EXTRA[@]}")
  if command -v aiperf >/dev/null 2>&1; then
    if [ "$(cat "$stamp" 2>/dev/null)" = "$AIPERF_REF" ]; then
      log "aiperf on PATH is the pinned ref ${AIPERF_REF:0:8}; skipping install"
      return 0
    fi
    log "aiperf on PATH is not the pinned ref ${AIPERF_REF:0:8} (recorded: $(cat "$stamp" 2>/dev/null || echo none)); reinstalling"
    # Deliberately NOT --no-deps: a newer aiperf may need dependencies the old
    # one did not, and installing the package without them is a worse failure
    # than the stale build we are replacing.
    pip_args+=(--force-reinstall)
  fi
  log "installing aiperf (AgentX): ${AIPERF_PACKAGE_SPEC}"
  if "$PYTHON" -m pip install --quiet "${pip_args[@]}" "$AIPERF_PACKAGE_SPEC"; then
    log "aiperf installed OK"
    # Stamp only after a success, so a failed upgrade retries next run instead
    # of recording a ref that is not what is on disk. Best-effort: an unwritable
    # state dir costs a redundant reinstall, never a wrong skip.
    mkdir -p "$(dirname "$stamp")" 2>/dev/null && printf '%s\n' "$AIPERF_REF" > "$stamp" 2>/dev/null || \
      warn "could not record the installed aiperf ref at ${stamp}; the next run will reinstall"
  else
    warn "aiperf install failed (${AIPERF_PACKAGE_SPEC}); AgentX mode (HYPERLOOM_AGENTX) stays unavailable until aiperf is installed or AIPERF_BIN is set. Default synthetic path is unaffected."
  fi
}

# --- 2b. Atomic-write patch for Magpie._prepare_benchmark_scripts ---
# The Hyperloom #C1 script-tearing race (vllm_mi300x.sh / sglang_mi300x.sh
# sourced by a leaked bash while a new Magpie subprocess is mid-`shutil.copy2` →
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
# the #C1 race is already mitigated upstream, so we `warn` and continue
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
  # INFERENCEX_PATH is passed explicitly: the patcher also has to scrub the
  # InferenceX ``benchmarks/`` copies Magpie executes and teach
  # ``benchmark_lib.sh::run_lm_eval`` to tolerate the flag. This step therefore
  # MUST run after ensure_inferencex has exported INFERENCEX_PATH — see the
  # call ordering at the bottom of this script.
  if MAGPIE_PATH="$MAGPIE_PATH" INFERENCEX_PATH="${INFERENCEX_PATH:-}" "$PYTHON" - <<'PY'
import os, sys
from hyperloom.orchestrator.actions.executors._magpie_patcher import (
    magpie_scripts_patch_status,
)
status = magpie_scripts_patch_status(
    os.environ["MAGPIE_PATH"],
    os.environ.get("INFERENCEX_PATH") or None,
)
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
# eval_flag_ok is False ONLY when a live `run_eval --concurrent-requests`
# survives in a caller script AND InferenceX's run_lm_eval would reject it
# (a defence-in-depth patch that merely could not be applied, with no live
# flag, is NOT counted as a failure -- install-time now matches the run-time
# ensure_eval_concurrency_compat judgement). This is the genuinely fatal case:
# every RUN_EVAL=true baseline aborts on 'Unknown parameter'. Distinct exit so
# install can name the failure mode.
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
      # (or a read/write error). The Hyperloom #C1 script-tearing race is NOT
      # mitigated — `profile`/`baseline` can hit `syntax error near unexpected
      # token 'fi'`. Strict mode (default) aborts; a falsy MAGPIE_PATCH_STRICT
      # (0/false/no/off) keeps the legacy fail-soft behaviour and only warns.
      if is_falsy "${MAGPIE_PATCH_STRICT:-1}"; then
        warn "Magpie atomic-write patch GENUINELY failed (race unmitigated); MAGPIE_PATCH_STRICT=${MAGPIE_PATCH_STRICT:-} (falsy), continuing anyway — review _magpie_patcher.py."
      else
        die "Magpie atomic-write patch GENUINELY failed: neither the legacy shutil.copy2 block nor an upstream atomic copy was found in benchmarker.py. The Hyperloom #C1 script-tearing race is unmitigated. Re-pin MAGPIE_REF to a supported commit, review _magpie_patcher.py, or set MAGPIE_PATCH_STRICT=0 to downgrade to a warning (or PATCH_MAGPIE=0 to skip entirely)."
      fi
    elif [ "$rc" -eq 2 ]; then
      warn "Magpie SGLang remote trust patch did not apply. If MAGPIE_TRUST_REMOTE_CODE=1 is required for custom-code models (for example Kimi/Qwen tokenizer paths), remote benchmark clients may still fail to pass trust; review _magpie_patcher.py or set PATCH_MAGPIE=0 only if this is intentional."
    elif [ "$rc" -eq 5 ]; then
      # Fail-loud by default: a surviving --concurrent-requests aborts EVERY
      # RUN_EVAL=true baseline in InferenceX's run_lm_eval arg parser, no
      # results*.json is written, and the baseline accuracy gate then stops the
      # whole run with `baseline_accuracy_failed`. There is no salvage: the
      # executor deliberately does NOT fall back to RUN_EVAL=false for a genuine
      # baseline (a throughput-only baseline cannot satisfy the accuracy gate).
      # Set MAGPIE_EVAL_FLAG_STRICT=0 only when accuracy eval is genuinely
      # not required for this deployment.
      if is_falsy "${MAGPIE_EVAL_FLAG_STRICT:-1}"; then
        warn "Magpie redundant --concurrent-requests eval flag could not be stripped from a generic benchmark script (unrecognised run_eval line); MAGPIE_EVAL_FLAG_STRICT=${MAGPIE_EVAL_FLAG_STRICT:-} (falsy), continuing anyway — RUN_EVAL=true baselines will abort on InferenceX's 'Unknown parameter: --concurrent-requests'."
      else
        die "Magpie redundant --concurrent-requests eval flag could not be stripped from a generic benchmark script (unrecognised run_eval line), and InferenceX's run_lm_eval could not be taught to tolerate it. Every RUN_EVAL=true baseline will abort with 'Unknown parameter: --concurrent-requests' and the run will stop with baseline_accuracy_failed. Concurrency must flow via EVAL_CONCURRENT_REQUESTS (fallback CONC), not the flag — fix the script's run_eval line or review _magpie_patcher.py. Set MAGPIE_EVAL_FLAG_STRICT=0 to downgrade to a warning if accuracy eval is not required."
      fi
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
# (`/shared/hyperloom/InferenceX`, `/shared/fully-local/.../InferenceX`,
# etc.) and pointed every install at whichever it found first. That
# multi-install / shared-checkout layout is the upstream source of the
# concurrent-write races behind the Hyperloom #C1 script-tearing race —
# every fresh Magpie subprocess `shutil.copy2`'d its scripts on top of
# the same shared files, while bash interpreters from neighbouring
# installs were `source`-ing them. Cloning a per-install copy here
# eliminates the cross-install fan-in (Magpie's in-place atomic-write patch then
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

# --- 4b. Scriptable quality-gate deps (SSIM + LPIPS) ---
#
# Every scriptable workload with a visual output computes its gate the same way
# (LPIPS / SSIM / MSE vs a reference): xDiT does, and so does an
# operator-supplied `--framework custom` bench script. torch/torchvision/numpy
# ship with the ROCm images, but scikit-image (SSIM) and lpips (LPIPS) do NOT —
# so without this step the gate silently degrades to MSE-only (the wrapper
# reports ssim_available/lpips_available=false). These are pip-name !=
# import-name, so we map them explicitly.
#
# Installed unconditionally rather than per framework: the framework is not
# known at install time, and an operator's own script is free to import either
# one. Gating these on `xdit` would disarm the gate for every other scriptable
# workload, and a workload whose gate cannot run scores every candidate zero.
#
# Fail-soft: lpips also pulls AlexNet weights on first use (network), so a
# failed install must NOT abort the whole install — the wrapper degrades
# gracefully (honest *_available=false) rather than crashing the run.
_SCRIPTABLE_QUALITY_DEPS=(
  "scikit-image:skimage"
  "lpips:lpips"
)

# Load-bearing packages an OPTIONAL dep install must never move. On a ROCm pod
# these are vendor ROCm builds from a private index; letting pip's resolver pull
# a PyPI (CUDA) torch to satisfy e.g. lpips' `torch>=0.4.0` silently bricks GPU
# access for EVERY framework sharing this venv (that is exactly how this brick
# shipped: a CUDA torch replaced the ROCm one, exit 0, no visible error).
_SHARED_VENV_CORE_PINS=(torch torchvision torchaudio triton)

# Write a pip constraints file pinning each installed core package to its exact
# current version. `pip install -c <file>` then forbids the resolver from moving
# them, while still letting the requested deps pull their OTHER (safe) deps.
_write_core_constraints() {
  local dest="$1" pkg ver
  : > "$dest"
  for pkg in "${_SHARED_VENV_CORE_PINS[@]}"; do
    ver="$("$PYTHON" -c "import importlib.metadata as m; print(m.version('$pkg'))" 2>/dev/null || true)"
    [ -n "$ver" ] && printf '%s==%s\n' "$pkg" "$ver" >> "$dest"
  done
  return 0
}

# torch's ROCm/HIP version string (empty if torch is absent OR is a non-ROCm
# build). The tripwire below reads it before and after the optional install.
_torch_hip_version() {
  "$PYTHON" -c "import torch; print(torch.version.hip or '')" 2>/dev/null || true
}

# Tripwire: if torch was a ROCm build before the optional install but is now a
# non-ROCm (CUDA-only) build, the resolver swapped the load-bearing wheel. Try a
# best-effort rollback to the pinned versions, then abort HARD — a silent warn
# here is exactly how this poison reached every co-tenant framework before.
_guard_torch_not_clobbered() {
  local constraints="$1" hip_before="$2" hip_after
  [ -n "$hip_before" ] || return 0
  hip_after="$(_torch_hip_version)"
  [ -n "$hip_after" ] && return 0
  warn "scriptable quality deps swapped the ROCm torch for a non-ROCm build; attempting rollback to: $(tr '\n' ' ' < "$constraints")"
  "$PYTHON" -m pip install --quiet --no-cache-dir --force-reinstall --no-deps \
    "${PIP_EXTRA[@]}" -r "$constraints" \
    || warn "rollback reinstall failed (pinned ROCm wheels may need the vendor index); restore torch manually"
  die "optional scriptable quality deps clobbered the load-bearing ROCm torch (was hip=${hip_before}, now a non-ROCm build). Aborting instead of poisoning every framework in this shared venv. Preinstall scikit-image/lpips in the image, or extend _SHARED_VENV_CORE_PINS."
}

ensure_scriptable_quality_deps() {
  log "ensuring scriptable quality-gate deps (SSIM/LPIPS) in $PYTHON"
  local missing=()
  local pair pip_name import_name
  for pair in "${_SCRIPTABLE_QUALITY_DEPS[@]}"; do
    pip_name="${pair%%:*}"
    import_name="${pair##*:}"
    if ! "$PYTHON" -c "import ${import_name}" >/dev/null 2>&1; then
      missing+=("$pip_name")
    fi
  done
  if [ ${#missing[@]} -eq 0 ]; then
    log "scriptable quality deps already satisfied"
    return 0
  fi
  if [ "$CHECK_ONLY" -eq 1 ]; then
    warn "check-only mode; would install scriptable quality deps: ${missing[*]}"
    return 0
  fi
  if [ "$DRY_RUN" -eq 1 ]; then
    log "dry-run; skipping scriptable quality dep install"
    return 0
  fi
  log "installing missing scriptable quality deps: ${missing[*]}"
  # Pin the load-bearing core so this optional install can never move
  # torch/torchvision/triton, and snapshot torch's ROCm build so the tripwire
  # can abort loudly if it got swapped anyway.
  local constraints hip_before
  constraints="$(mktemp)"
  _write_core_constraints "$constraints"
  hip_before="$(_torch_hip_version)"
  "$PYTHON" -m pip install --quiet --no-cache-dir -c "$constraints" \
    "${PIP_EXTRA[@]}" "${missing[@]}" \
    || warn "failed to install scriptable quality deps: ${missing[*]} (gate degrades to MSE-only)"
  _guard_torch_not_clobbered "$constraints" "$hip_before"
  rm -f "$constraints"
  for pair in "${_SCRIPTABLE_QUALITY_DEPS[@]}"; do
    import_name="${pair##*:}"
    "$PYTHON" -c "import ${import_name}" >/dev/null 2>&1 \
      || warn "scriptable quality dep '${import_name}' not importable after install (gate excludes it)"
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

# --- 4d. Per-framework runtime deps (manifest-driven) ---
#
# Scriptable frameworks execute the model author's own code (HY-World-2.0 for
# an operator's own), which imports packages no ROCm serving image ships. A framework
# declares what it needs in assets/framework_deps/<framework>.txt, so onboarding
# the next one is a data file rather than new code, and a framework with no
# manifest (sglang / vllm / atom) is a no-op.
#
# Manifest parsing, the load-bearing refusal list and the torch-clobber
# tripwire all live in hyperloom.inference_optimizer.framework_deps, which the
# CLI preflight also calls, so the two passes cannot drift.
#
# $FRAMEWORK is normally only known at launch (--framework), not at install
# time, which is why preflight owns the pass that covers the documented flow.
# This one is the fast path for operators who export it before installing.
ensure_framework_deps() {
  if [ -z "${FRAMEWORK:-}" ]; then
    log "framework deps: \$FRAMEWORK unset at install time; CLI preflight will handle it at launch"
    return 0
  fi
  local args=(--framework "$FRAMEWORK" --python "$PYTHON" --prefix "[inference-optimizer] framework deps")
  [ "$CHECK_ONLY" -eq 1 ] && args+=(--check-only)
  [ "$DRY_RUN" -eq 1 ] && args+=(--dry-run)
  # Each flag attaches with =: a bare --pip-extra would leave a dashed value
  # unconsumed, and argparse then rejects it as an unrecognized argument.
  if [ ${#PIP_EXTRA[@]} -gt 0 ]; then
    local pip_flag
    for pip_flag in "${PIP_EXTRA[@]}"; do
      args+=("--pip-extra=${pip_flag}")
    done
  fi
  "$PYTHON" -m hyperloom.inference_optimizer.framework_deps "${args[@]}" \
    || die "framework deps for '${FRAMEWORK}' failed fatally; see the error above"
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
fi
ensure_inferencex
# Ordering matters: the Magpie script patch also scrubs the redundant
# `--concurrent-requests` eval flag from the InferenceX `benchmarks/` copies
# Magpie actually executes, and teaches `benchmark_lib.sh::run_lm_eval` to
# tolerate it. Both need $INFERENCEX_PATH, which only ensure_inferencex exports
# — running the patch before it silently skipped those targets and left
# RUN_EVAL=true baselines aborting on 'Unknown parameter'.
if [ "$HYPERLOOM_BENCHMARK_BACKEND_LC" != "bypass" ]; then
  ensure_magpie_atomic_scripts_patch
fi

# aiperf (AgentX client) is an OPT-IN Magpie-path add-on: installed only when the
# operator explicitly asks (INSTALL_AIPERF or HYPERLOOM_AGENTX truthy), so a
# default install grows no extra network/build dependency. If AgentX is turned on
# at runtime without aiperf present, the runtime preflight fails loud with
# guidance (install it, or point AIPERF_BIN at an existing build). Fail-soft
# inside ensure_aiperf. Never for the bypass backend.
if [ "$HYPERLOOM_BENCHMARK_BACKEND_LC" != "bypass" ]; then
  # Strip surrounding whitespace then lowercase, so the installer
  # parses these flags identically to the Python runtime's agentx_enabled()
  # (which .strip()s) — e.g. HYPERLOOM_AGENTX=" on " must be ON in both.
  _agx_want="$(printf '%s' "${INSTALL_AIPERF:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | tr '[:upper:]' '[:lower:]')"
  _agx_sw="$(printf '%s' "${HYPERLOOM_AGENTX:-}" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' | tr '[:upper:]' '[:lower:]')"
  case "${_agx_want}:${_agx_sw}" in
    1:*|true:*|yes:*|on:*|*:1|*:true|*:yes|*:on)
      ensure_aiperf ;;
    *)
      log "aiperf (AgentX) skipped: set INSTALL_AIPERF=1 or HYPERLOOM_AGENTX=1 to install it (or point AIPERF_BIN at an existing build)." ;;
  esac
fi
ensure_bench_serving_deps
ensure_scriptable_quality_deps
ensure_framework_deps
chain_kernel_agent
# rocprof-compute + pandas<3 pin runs LAST — strictly AFTER every pip-installing
# step (chain_kernel_agent included; nothing below installs packages). This makes
# the pandas<3 pin the final word (no later `pip install` can re-pull pandas>=3)
# and its own re-check the truthful end state, not a premature false-positive.
# Unconditional (not gated on the backend): the default-geak install a later
# forge session inherits still gets rocprof-compute + pandas<3.
ensure_rocprof_compute
# tree-reform.MD P2.5: framework-agent was promoted into
# src/hyperloom/agents/framework/ (single hyperloom distribution), so the
# `fa` CLI is already installed by ensure_inference_optimizer() above; no
# more separate chain_framework_agent() delegation to a standalone installer.

_write_specialist_secret_env_opt_in() {
  if [ "$DRY_RUN" -eq 1 ] || [ "$CHECK_ONLY" -eq 1 ]; then
    log "would append HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV=1 to ${KERNEL_AGENT_ENV}"
    return 0
  fi
  mkdir -p "$(dirname "$KERNEL_AGENT_ENV")"
  if [ -f "$KERNEL_AGENT_ENV" ] && grep -q '^export HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV=' "$KERNEL_AGENT_ENV" 2>/dev/null; then
    sed -i 's|^export HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV=.*|export HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV=1|' "$KERNEL_AGENT_ENV"
  else
    {
      echo ""
      echo "# Production bootstrap: specialist subprocesses need env credentials unless claude CLI auth is preconfigured"
      echo "export HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV=1"
    } >> "$KERNEL_AGENT_ENV"
  fi
}

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

_write_specialist_secret_env_opt_in
_probe_framework_source_roots

_prune_dep_cache "InferenceX" "Magpie"
log "install complete"
log "kernel-agent env file written: ${KERNEL_AGENT_ENV}"
log "  HYPERLOOM_KERNEL_AGENT_ROOT=${HYPERLOOM_KERNEL_AGENT_ROOT}"
log ""
log "next steps — pick ONE:"
log "  (a) source ${KERNEL_AGENT_ENV}, then run hyperloom.inference_optimizer.cli"
log "  (b) just launch hyperloom.inference_optimizer.cli — preflight will auto-source"
log "      \$KERNEL_AGENT_ENV (or \$USER_DATA_PATH/runtime/kernel-agent.env.sh)"
log "      via _load_kernel_agent_env_fallback() if HYPERLOOM_KERNEL_AGENT_ROOT"
log "      is unset."
log ""
log "If you skip BOTH and HYPERLOOM_KERNEL_AGENT_ROOT stays unset, the"
log "roofline composite action's trace_analyze sub-step will fail with"
log "  'HYPERLOOM_KERNEL_AGENT_ROOT is not set'"
log "and the whole optimisation loop stalls (PolicyGate blocks every"
log "downstream action on a missing TraceLens snapshot)."
