#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################
#
# aiperf_client.sh — AgentX benchmark client for Magpie's benchmark path.
#
# Design: DELEGATE the server phase to the maintained per-framework builtin
# (vllm_mi300x.sh / sglang_mi300x.sh) via MAGPIE_RUN_PHASE=server, so server
# boot + torch-profiler enabling stay correct across frameworks and versions
# (no profiler flags reimplemented here). Then run `aiperf profile` (AgentX
# weka-trace scenario) as the client, and map its export into the InferenceX
# result schema. On single node the client owns profiling: InferenceX's
# benchmark_serving.py self-triggers /start_profile, and aiperf does not, so
# when PROFILE=1 this script self-brackets a /start_profile..stop_profile window
# after AIPerf reports the measured phase through its progress API.
#
# Inputs (from Magpie env): MODEL, TP, PORT, MAX_MODEL_LEN, CONC, RESULT_DIR,
#   RESULT_FILENAME, PROFILE, EXTRA_VLLM_ARGS, FRAMEWORK, GPU_TYPE/RUNNER_TYPE,
#   AGENTX_CAPTURE_ID, AGENTX_CAPTURE_STATUS_PATH.
# AgentX knobs (AGENTX_ prefix; NOT AIPERF_, which aiperf's own settings read):
#   AGENTX_DATASET / WEKA_LOADER_OVERRIDE (pin the corpus loader),
#   AGENTX_NUM_ENTRIES (corpus cap; default 393 = all),
#   AGENTX_DURATION (measurement window; default 3600),
#   AGENTX_WARMUP_REQUESTS_PER_LANE (default 10),
#   AGENTX_WARMUP_GRACE_PERIOD (max drain wait; default 1800),
#   AGENTX_FAILED_REQUEST_THRESHOLD (error-rate abort ratio; default 0.10),
#   AGENTX_UNSAFE_OVERRIDE (opt into a sub-900s smoke; forces the run
#     non-submittable -- see the smoke note below),
#   AGENTX_REALTIME_METRICS (rolling stats block; default true),
#   AGENTX_DATASET_CONFIG_TIMEOUT (default 1800), AGENTX_LIVE_ASSISTANT,
#   AGENTX_HTTP_TCP_USER_TIMEOUT (no-TCP-progress bound in ms; default 900000,
#     matching upstream's long-context recipes -- aiperf's stock 30s aborts
#     live connections while the server is prefill-bound),
#   AGENTX_HTTP_KEEP_ALIVE_S (server-side idle timeout in s; default 900 -- the
#     other half of AGENTX_HTTP_TCP_USER_TIMEOUT. Exported as the framework's
#     own knob before the server boots; see the keep-alive note below),
#   AGENTX_MMAP_CACHE_DIR (dataset mmap cache; defaults under $HF_HUB_CACHE),
#   AGENTX_MAX_CTX (explicit opt-in client-side context cap; NEVER inferred
#     from $MAX_MODEL_LEN -- see the replay-context note below),
#   AGENTX_TRACE_FLUSH_TIMEOUT_S (how long to wait after /stop_profile for the
#     per-rank trace files to finish writing; default 1800. A 200 from
#     /stop_profile only means the tracer was told to stop -- measured on
#     GLM-5.3 TP=8, the 5.1 GB set was still being written 546s later),
#   AGENTX_TRACE_FIRST_FILE_TIMEOUT_S (separate, shorter bound for the case
#     where NO trace file appears at all -- a failed capture, not a slow one;
#     default 900, clamped to AGENTX_TRACE_FLUSH_TIMEOUT_S. The first rank file
#     landed at t+350s on that same GLM-5.3 capture),
#   AGENTX_KEEP_SERVER, AGENTX_PROFILE_WINDOW_S,
#   AGENTX_PROFILE_WARMUP_S (deprecated and ignored; phase-gated profiling
#     replaced the fixed warmup delay),
#   AGENTX_SERVER_SCRIPT (override builtin name), AIPERF_BIN.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
log() { echo "[aiperf_client] $*"; }

: "${MODEL:?MODEL required}"
PORT="${PORT:-8000}"
# Concurrency is measurement-defining and upstream makes it a hard requirement
# (benchmark_lib.sh check_env_vars exits 1 on an empty CONC). A default here
# would produce a full 3600s scenario-locked run at a concurrency nobody chose,
# and the mapped result records no concurrency at all, so the mismatch would be
# invisible afterwards. The switch always projects CONC; if it ever stops, say so.
: "${CONC:?CONC required (the AgentX switch projects it from the benchmark config)}"
RESULT_DIR="${RESULT_DIR:-$(pwd)}"
RESULT_FILENAME="${RESULT_FILENAME:-inferencex_result}"
ART="${RESULT_DIR}/aiperf_artifacts"
# Start from a clean artifact dir so a prior round's export can never be
# mis-read as this round's result (and so `find` matches exactly one file).
rm -rf "$ART"
mkdir -p "$RESULT_DIR" "$ART"

# ── Resolve the per-framework builtin server script ──────────────────────────
FRAMEWORK="${FRAMEWORK:-}"
GPU="$(printf '%s' "${GPU_TYPE:-${RUNNER_TYPE:-mi300x}}" | tr '[:upper:]' '[:lower:]')"
BUILTIN="${AGENTX_SERVER_SCRIPT:-${FRAMEWORK}_${GPU}.sh}"
# The AgentX switch injects FRAMEWORK from benchmark.framework; a missing value
# (and no explicit AGENTX_SERVER_SCRIPT) is misconfiguration -- fail loud rather
# than silently defaulting to a framework and booting the wrong server.
if [ -z "${AGENTX_SERVER_SCRIPT:-}" ] && [ -z "$FRAMEWORK" ]; then
  log "ERROR: FRAMEWORK unset and AGENTX_SERVER_SCRIPT not provided; cannot resolve the builtin server script"
  exit 2
fi
if [ ! -f "${BENCH_DIR}/${BUILTIN}" ]; then
  log "ERROR: builtin server script not found: ${BENCH_DIR}/${BUILTIN}"
  exit 2
fi

# ── Server phase: delegate to builtin (correct boot + profiler per framework) ─
PIDFILE="${RESULT_DIR}/agentx_server.pid"
rm -f "$PIDFILE"
SERVER_PID=""  # set after boot; cleanup guards ${SERVER_PID:-} + a port fallback

cleanup() {
  [ "${AGENTX_KEEP_SERVER:-0}" = "1" ] && return 0
  if [ -n "${SERVER_PID:-}" ]; then
    log "tearing down server pid=${SERVER_PID}"
    kill -TERM "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null || true
    # vLLM can ignore/stall on SIGTERM (graceful shutdown hangs after a large
    # profiler-trace flush), leaking the GPUs. Escalate to SIGKILL if the
    # process group is still alive after a grace period.
    _i=0
    while [ "$_i" -lt 10 ]; do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 2
      _i=$((_i + 1))
    done
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      log "server survived SIGTERM after grace period; sending SIGKILL"
      kill -KILL "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null || true
    fi
  fi
  # Belt-and-suspenders: free the port even if the pid was unknown/stale, so a
  # server that booted without a recorded pid can never leak the GPUs.
  command -v fuser >/dev/null 2>&1 && fuser -k "${PORT}/tcp" 2>/dev/null || true
}
# Install the trap BEFORE booting the server: if the builtin starts the server
# then returns nonzero, set -e aborts here and the EXIT trap still fires (the
# port fallback reaps a server booted without a recorded pid) — no leak window.
trap cleanup EXIT INT TERM

# ── Server-side keep-alive: the other half of AGENTX_HTTP_TCP_USER_TIMEOUT ────
# AIPerf pins ONE pooled keep-alive connection per agentic session and reuses it
# across that session's turns. The inter-turn gap in an agentic replay is a
# model think-time, not a client delay, and routinely exceeds a serving
# framework's default idle timeout -- vLLM's is 5s (envs.py:
# ``VLLM_HTTP_TIMEOUT_KEEP_ALIVE: int = 5``). When the gap crosses it the server
# closes the socket exactly as the client reuses it, and aiohttp surfaces
# ServerDisconnectedError. AIPerf escalates that to a TERMINAL warmup failure:
# "A root AgentX warmup request failed, so profiling was not started" -- against
# a completely healthy server, with no error anywhere in the server log.
#
# Measured here on a conc=16 K3 round: the server logged an orderly
# "Application shutdown complete" while warmup sat at 64/177, and the only
# symptom was a burst of ServerDisconnectedError on the client. Upstream hit the
# same failure (InferenceX #2371 aborted a c4 arm ~15 min in) and fixes it by
# raising the SERVER idle timeout to match the client's tolerance.
#
# The client half already ships above as AIPERF_HTTP_TCP_USER_TIMEOUT (900s);
# without this the two disagree by 180x. Exported per framework because the knob
# name is framework-specific, and only when the operator has not pinned one.
_KEEPALIVE_S="${AGENTX_HTTP_KEEP_ALIVE_S:-900}"
# Decide from BUILTIN, not from a concatenation. Matching against
# "${FRAMEWORK}${BUILTIN}" glued the two together, so FRAMEWORK=vllm with
# BUILTIN=sglang_mi300x.sh formed "vllmsglang_mi300x.sh", hit the *vllm* arm
# first, and left SGLang on its 5s default -- while this very line went on to
# report 900s. The server then closed the connection mid-warmup and the round
# died as "root AgentX warmup request failed", with the log actively denying the
# cause. BUILTIN is the script that actually boots, so it is the authority;
# FRAMEWORK is only a fallback for a script name that carries no framework, and
# a disagreement between them is worth saying out loud rather than resolving
# silently in either direction.
_ka_target=""
case "$BUILTIN" in
  *vllm*) _ka_target=vllm ;;
  *sglang*) _ka_target=sglang ;;
  *)
    case "${FRAMEWORK:-}" in
      *vllm*) _ka_target=vllm ;;
      *sglang*) _ka_target=sglang ;;
    esac
    ;;
esac
case "${FRAMEWORK:-}" in
  "") ;;
  *"$_ka_target"*) ;;
  *)
    [ -n "$_ka_target" ] && log "WARN FRAMEWORK=${FRAMEWORK} disagrees with the server script ${BUILTIN}; keep-alive follows the script"
    ;;
esac
case "$_ka_target" in
  vllm) export VLLM_HTTP_TIMEOUT_KEEP_ALIVE="${VLLM_HTTP_TIMEOUT_KEEP_ALIVE:-$_KEEPALIVE_S}" ;;
  sglang) export SGLANG_TIMEOUT_KEEP_ALIVE="${SGLANG_TIMEOUT_KEEP_ALIVE:-$_KEEPALIVE_S}" ;;
esac
if [ -n "$_ka_target" ]; then
  log "server keep-alive: ${_ka_target} ${_KEEPALIVE_S}s (client tcp-user-timeout ${AGENTX_HTTP_TCP_USER_TIMEOUT:-900000}ms)"
else
  log "WARN no keep-alive knob for server script ${BUILTIN}; server idle timeout left at its default while the client tolerates ${AGENTX_HTTP_TCP_USER_TIMEOUT:-900000}ms"
fi

log "delegating server boot -> ${BUILTIN} (PROFILE=${PROFILE:-0})"
MAGPIE_RUN_PHASE=server MAGPIE_SERVER_PID_FILE="$PIDFILE" \
  PORT="$PORT" RESULT_DIR="$RESULT_DIR" \
  bash "${BENCH_DIR}/${BUILTIN}"
SERVER_PID="$(cat "$PIDFILE" 2>/dev/null || true)"

# Fail loud if the builtin server phase did not record a pid: proceeding would
# run a benchmark against a server we cannot reliably tear down.
if [ -z "${SERVER_PID:-}" ]; then
  log "ERROR: builtin server phase wrote no pid to ${PIDFILE}; refusing to run (would risk a GPU leak)"
  exit 3
fi
log "server up (pid=${SERVER_PID}) on port ${PORT}"

# ── Resolve served model name (a reused server may expose a different id) ─────
SERVE_MODEL="$MODEL"
_served="$(curl -sf "http://localhost:${PORT}/v1/models" 2>/dev/null \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["data"][0]["id"])' 2>/dev/null || true)"
[ -n "$_served" ] && SERVE_MODEL="$_served"

# ── Replay context: never capped from $MAX_MODEL_LEN ─────────────────────────
# ``--max-context-length`` makes aiperf DROP (not truncate) every trace whose
# peak exceeds it, and $MAX_MODEL_LEN is derived from the synthetic ISL+OSL
# shape the agentic corpus never uses -- so deriving the cap from it silently
# shrinks the corpus to its short-trace tail while every status marker still
# reports a clean run. Upstream's agentic path unsets MAX_MODEL_LEN and never
# emits the flag; the server's own context window is the only limit that
# applies, and a trace that does not fit surfaces honestly as request errors
# (see --failed-request-threshold below) instead of vanishing.
#
# AGENTX_MAX_CTX stays as an explicit operator escape hatch: set it to opt IN
# to a client-side cap. It is never inferred.
CTX_ARGS=()
if [ -n "${AGENTX_MAX_CTX:-}" ]; then
  CTX_ARGS+=(--max-context-length "$AGENTX_MAX_CTX")
fi

# ── Corpus variant: mirror upstream's model-family whitelist ─────────────────
# Upstream picks the trace corpus from a curated MODEL_PREFIX label
# (benchmark_lib.sh resolve_trace_source): the 1M-context families replay the
# unfiltered 062126 corpus, everything else the 256k-capped variant. Hyperloom
# has no such label, so derive the family from the model identity. An unmatched
# model falls back to the 256k variant -- the SAME fallback upstream uses -- and
# says so, rather than failing: being conservative here costs a shorter corpus,
# guessing "full" costs a boot failure or a 4xx storm.
_model_family() {
  printf '%s' "${1##*/}" | tr '[:upper:]' '[:lower:]' | tr -d '._-'
}
_default_loader() {
  case "$(_model_family "$1")" in
    dsv4*|deepseekv4*|glm52*|minimaxm3*|kimik3*)
      printf 'semianalysis_cc_traces_weka_062126' ;;
    *)
      printf 'semianalysis_cc_traces_weka_062126_256k' ;;
  esac
}
# WEKA_LOADER_OVERRIDE is upstream's own per-recipe override; AGENTX_DATASET is
# the Hyperloom-side name kept for compatibility. Either pins the loader.
# Derived once and reused by the deviation check below: two call sites 90 lines
# apart would silently disagree the moment the derivation grows a second input.
# AGENTX_CANONICAL_DATASET lets an operator declare the canonical corpus for a
# model the family whitelist does not cover -- see the deviation check.
CANON_DS="${AGENTX_CANONICAL_DATASET:-$(_default_loader "$MODEL")}"
DS="${AGENTX_DATASET:-${WEKA_LOADER_OVERRIDE:-$CANON_DS}}"
if [ -z "${AGENTX_DATASET:-}${WEKA_LOADER_OVERRIDE:-}" ]; then
  case "$DS" in
    *_256k) log "corpus: ${DS} (model family not in the 1M-context whitelist; set WEKA_LOADER_OVERRIDE to pin another)" ;;
    *)      log "corpus: ${DS}" ;;
  esac
fi

# The with-subagents corpus holds 393 traces; the loader treats this as a
# min(cap, available) ceiling, so 393 means "all of them".
NENT="${AGENTX_NUM_ENTRIES:-393}"
DURATION="${AGENTX_DURATION:-3600}"

# Deterministic agentic cache-pressure warmup: N extra requests per concurrency
# lane on top of the mandatory snapshot primers, then wait (at most) the grace
# period for them to drain before profiling starts. This replaces the old
# --warmup-duration / --num-warmup-sessions pair, which the scenario does not
# use and which measured a different thing entirely.
CANON_WARMUP_PER_LANE=10
CANON_WARMUP_GRACE=1800
WARMLANE="${AGENTX_WARMUP_REQUESTS_PER_LANE:-$CANON_WARMUP_PER_LANE}"
WARMGRACE="${AGENTX_WARMUP_GRACE_PERIOD:-$CANON_WARMUP_GRACE}"

# Validate every measurement-defining knob BEFORE it reaches a comparison, an
# arithmetic expansion, or an awk program. Three concrete failures this closes,
# all of them in knobs the orchestrator forwards verbatim from its own env:
#
#   * ``[ "$WARMLANE" -lt N ]`` with a non-integer ("1.5", "0x2") makes ``[``
#     exit 2. On the left of ``&&`` that status is exempt from ``set -e``, so the
#     non-canonical guard silently does not fire and an illegal configuration is
#     stamped submission_valid=true -- exactly the hole the guard exists to plug.
#   * a non-integer WARMGRACE reaches the ``$(( ))`` in the PROFILE branch and,
#     under ``set -euo pipefail``, aborts a round that was otherwise complete.
#   * FRT is interpolated into an awk PROGRAM BODY below. Unvalidated, a value
#     like ``system("...")`` is executed by awk. Fixed both ways: passed as an
#     awk -v variable (never as program text) AND rejected here.
#
# Fail loud rather than coerce: these values define what was measured, and this
# file's whole contract is that a deviation can never be mistaken for a
# leaderboard run. A typo must stop the round, not silently become canonical.
_require_uint() {  # _require_uint NAME VALUE
  case "$2" in
    "" | *[!0-9]*)
      log "ERROR: $1 must be a non-negative integer, got '$2'"
      exit 2
      ;;
  esac
}
_require_decimal() {  # _require_decimal NAME VALUE -- digits with one optional dot
  case "$2" in
    "" | *[!0-9.]* | *.*.*)
      log "ERROR: $1 must be a decimal number, got '$2'"
      exit 2
      ;;
  esac
}
_require_uint AGENTX_WARMUP_REQUESTS_PER_LANE "$WARMLANE"
_require_uint AGENTX_WARMUP_GRACE_PERIOD "$WARMGRACE"
# DURATION is read above (before these helpers exist) but validated here, since
# it reaches the same `$(( ))` in the profile-delay clamp and the same numeric
# `[ ]` comparison in the canonical check below.
_require_uint AGENTX_DURATION "$DURATION"

# Per-trajectory-tree idle cap. NOT the same thing as the scenario's 10s
# whole-system cap, and NOT scenario-locked -- upstream passes it explicitly
# alongside the scenario. Without it a trace carrying a 20-minute recorded idle
# gap replays that gap in full, and because --benchmark-duration is a fixed
# window the lost time comes straight out of measured requests: throughput lands
# systematically below the published row for the same server config.
IDLEGAP="${AGENTX_TRACE_IDLE_GAP_CAP_SECONDS:-300}"

# aiperf reads AIPERF_-prefixed env into its own pydantic settings; scrub any
# stray exported ones (keep AIPERF_BIN, which is ours) so they can't corrupt it.
# Ours are exported AFTER the scrub so they are authoritative; operators tune
# them through the AGENTX_ names instead.
while IFS='=' read -r _k _; do
  case "$_k" in
    AIPERF_BIN) : ;;
    AIPERF_*) unset "$_k" 2>/dev/null || true ;;
  esac
done < <(env)

# Dataset load + reconstruct + mmap runs 4-14 min on the Weka corpus; aiperf's
# stock 900s Configure-Profiling timeout trips under parallel /tmp contention.
# aiperf validates SERVICE_PROFILE_CONFIGURE_TIMEOUT >= DATASET_CONFIGURATION_TIMEOUT.
DATASET_CONFIG_TIMEOUT="${AGENTX_DATASET_CONFIG_TIMEOUT:-1800}"
_require_uint AGENTX_DATASET_CONFIG_TIMEOUT "$DATASET_CONFIG_TIMEOUT"
export AIPERF_DATASET_CONFIGURATION_TIMEOUT="$DATASET_CONFIG_TIMEOUT"
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="$DATASET_CONFIG_TIMEOUT"
# TCP_USER_TIMEOUT bounds how long Linux tolerates an established connection
# making no progress -- and an agentic turn against a long-context model makes
# no TCP progress for as long as the server is prefill-bound. aiperf's stock
# 30s therefore aborts otherwise-live connections mid-prefill, which surfaces
# as a warmup failure with no server-side error to match it. Upstream's
# Kimi-K3 and DSv4 recipes all export 900000 (15 min) for exactly this, and the
# scrub above would drop an inherited copy, so it has to be re-stated here or
# the request timeout is left to a bound two orders of magnitude too small.
export AIPERF_HTTP_TCP_USER_TIMEOUT="${AGENTX_HTTP_TCP_USER_TIMEOUT:-900000}"
# Pre-canned assistant replay (recorded responses drive later turns).
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="${AGENTX_LIVE_ASSISTANT:-0}"
# Headless realtime metrics are opt-in on current aiperf, and the scrub above
# would drop an inherited copy anyway. Without it the rolling TTFT/ITL/throughput
# block is skipped entirely, which makes --stats-interval inert: a 60-minute
# measurement window then emits nothing until it ends, so a run that is merely
# slow is indistinguishable from one that has wedged. Upstream exports it too.
export AIPERF_UI_REALTIME_METRICS_ENABLED="${AGENTX_REALTIME_METRICS:-true}"
# Content-addressed mmap cache: on a hit this skips loader + tokenizer +
# composer entirely, turning that 4-14 min into ~0 for every run after the
# first. Soft default -- never required, so a bare environment still works.
_mmap_default="${HF_HUB_CACHE:-${HOME:-/tmp}/.cache/huggingface/hub}/aiperf_dataset_mmap"
export AIPERF_DATASET_MMAP_CACHE_DIR="${AGENTX_MMAP_CACHE_DIR:-$_mmap_default}"

AIPERF="${AIPERF_BIN:-aiperf}"

# Abort the run once the error rate exceeds this ratio. aiperf's own default is
# None, i.e. the check is DISABLED -- without the flag a run whose requests
# mostly 4xx still exits 0 and is mapped as a normal measurement, because
# map_aiperf.py carries no error counters. This is the safety net that turns a
# server/client context mismatch into an honest failure instead of a fabricated
# win on the surviving short sessions. Matches upstream's 0.10.
# Declared as one value, like CANON_WARMUP_*/WARMLANE below, so the canonical
# ratio and the default cannot drift apart.
CANON_FRT=0.10
FRT="${AGENTX_FAILED_REQUEST_THRESHOLD:-$CANON_FRT}"
_require_decimal AGENTX_FAILED_REQUEST_THRESHOLD "$FRT"

# ── Non-canonical workloads may run, but may never be submittable ────────────
# The scenario enforces a 900s duration floor, so a shortened AGENTX_DURATION is
# a hard startup abort rather than a quick run -- there would be no way to smoke
# test this path at all. Upstream opts into --unsafe-override below the floor.
#
# --unsafe-override alone is NOT sufficient to make a run non-submittable: aiperf
# stamps submission_valid=false only when the override actually suppressed a
# violation, so forcing the flag at the canonical 3600s (where there is nothing
# to suppress) leaves a fully KEEP-able result. And the scenario has no concept
# of corpus size at all, so shrinking AGENTX_NUM_ENTRIES to 50 traces produces
# submission_valid=true on a workload nothing on the leaderboard ran.
#
# So Hyperloom stamps the verdict itself. Every deviation from the canonical
# workload is collected here and handed to map_aiperf.py, which forces
# submission_valid=false with these reasons attached; benchmark_result.py then
# refuses the measurement. A smoke can be run, and can never be mistaken for a
# leaderboard measurement -- by construction rather than by promise.
CANON_ENTRIES=393
CANON_DURATION=3600
# Warmup is measurement-defining and was missing from this list until a measured
# run exposed the gap: the agentic warmup is what puts the KV/radix cache under
# realistic pressure before the window opens, so replaying at 1 request/lane
# instead of 10 measures a materially emptier cache. It carries no scenario
# marker either -- aiperf has no concept of "how much warmup is enough" -- so a
# reduced-warmup round came back submission_valid=true and looked publishable.
# On a 743B model the canonical 10/lane is a ~2h warmup, which is exactly when
# an operator reaches for this knob, so the hole was reachable in practice.
# (CANON_WARMUP_PER_LANE/CANON_WARMUP_GRACE are declared above, alongside
# WARMLANE/WARMGRACE, so the canonical value and the default can't drift apart.)
# The corpus this model family canonically replays, before any operator pin.
# CANON_DS is resolved with the corpus above. The family whitelist behind it is
# a derivation, not a registry -- a model upstream runs on the full corpus but
# whose slug does not match falls back to the 256k set, and the corpus log line
# tells the operator to pin the right one. Flagging that pin as a deviation
# would make the correct run permanently non-submittable, which is why
# AGENTX_CANONICAL_DATASET exists as a second, explicit knob: pinning alone
# still counts as a deviation.
NONCANON=()
# A pinned corpus is a different workload, and the scenario cannot object: its
# allowlist admits every dated weka variant, so replaying the 061526 set (which
# upstream's own H100/H200 recipes pin) comes back submission_valid=true against
# a leaderboard row measured on 062126. Only the client knows which one this
# model family was supposed to replay, so only the client can say it deviated.
[ "$DS" != "$CANON_DS" ] && NONCANON+=("corpus=${DS}(canonical ${CANON_DS})")
[ "$NENT" != "$CANON_ENTRIES" ] && NONCANON+=("entries=${NENT}(canonical ${CANON_ENTRIES})")
[ "$DURATION" != "$CANON_DURATION" ] && NONCANON+=("duration=${DURATION}s(canonical ${CANON_DURATION}s)")
[ -n "${AGENTX_MAX_CTX:-}" ] && NONCANON+=("client_context_cap=${AGENTX_MAX_CTX}")
[ "${AGENTX_UNSAFE_OVERRIDE:-false}" = "true" ] && NONCANON+=("unsafe_override_forced")
# `-lt`, not `!=`: only a *smaller* value under-pressures the cache or risks
# truncating the drain before it finishes. A larger value is strictly more
# warmup than canonical -- e.g. an operator raising the grace period so a
# large model's warmup has room to drain -- and does not change what gets
# replayed, so it must not be flagged as a deviation.
[ "$WARMLANE" -lt "$CANON_WARMUP_PER_LANE" ] && \
  NONCANON+=("warmup_per_lane=${WARMLANE}(canonical ${CANON_WARMUP_PER_LANE})")
[ "$WARMGRACE" -lt "$CANON_WARMUP_GRACE" ] && \
  NONCANON+=("warmup_grace=${WARMGRACE}s(canonical ${CANON_WARMUP_GRACE}s)")
# The abort threshold is measurement-defining for the same reason warmup is:
# raising it keeps a run alive that upstream's 0.10 would have aborted, and the
# surviving requests are then mapped as a normal measurement. aiperf stamps no
# scenario marker for it -- the threshold is the client's own safety net, not
# something the scenario knows about -- so a loosened round comes back
# submission_valid=true. Only a *larger* ratio deviates; tightening it below
# canonical measures a strictly cleaner run. Compared with awk because the
# ratio is a decimal, which `-lt` cannot handle.
# -v, never string interpolation: the value is DATA to awk, so it can never be
# read as program text. Interpolating it made any caller that can set an
# AGENTX_* env var (the switch forwards them verbatim) able to run arbitrary
# commands inside the container via ``system("...")`` -- and, because the
# injected program returned a status of its own, the non-canonical guard below
# also silently failed to fire.
if awk -v f="$FRT" -v c="$CANON_FRT" 'BEGIN{exit !(f > c)}' 2>/dev/null; then
  NONCANON+=("failed_request_threshold=${FRT}(canonical ${CANON_FRT})")
fi

SMOKE_ARGS=()
if [ "$DURATION" -lt "$CANON_DURATION" ] || [ "${AGENTX_UNSAFE_OVERRIDE:-false}" = "true" ]; then
  # Below the floor the scenario would abort outright; at or above it the flag
  # is harmless (nothing to suppress) and keeps the two paths uniform.
  SMOKE_ARGS+=(--unsafe-override)
fi
# Always exported, empty included: the switch forwards every AGENTX_* key from
# the orchestrator's environment, so an inherited value from a previous run or a
# wrapper would otherwise survive into a canonical run and invalidate it.
export AGENTX_NONCANONICAL_REASONS=""
if [ ${#NONCANON[@]} -gt 0 ]; then
  _reasons="$(IFS=,; echo "${NONCANON[*]}")"
  export AGENTX_NONCANONICAL_REASONS="$_reasons"
  log "SMOKE: non-canonical workload [${_reasons}] -- this result will be stamped submission_valid=false and cannot KEEP"
fi

log "aiperf model=${SERVE_MODEL} corpus=${DS} entries=${NENT} conc=${CONC} duration=${DURATION}s warmup=${WARMLANE}/lane grace=${WARMGRACE}s fail-thresh=${FRT}${AGENTX_MAX_CTX:+ maxctx=${AGENTX_MAX_CTX}}"

# Mirrors upstream benchmark_lib.sh build_replay_cmd(). --scenario locks the
# leaderboard invariants (ignore_eos, streaming, no input truncation, corpus
# allowlist, 900s duration floor, idle-gap cap, cache-bust) and stamps
# metadata.submission_valid; anything conflicting aborts at startup.
#
# Deliberate deviations from upstream, all measurement-neutral:
#   --artifact-dir       upstream says --output-artifact-dir; aiperf accepts
#                        both (GenAI-Perf alias) and this name is what the
#                        test harness parses.
#   --model              upstream uses ${SERVED_MODEL_NAME:-$MODEL}; the probed
#                        /v1/models id is more robust when a server is reused.
#   --max-context-length omitted (see the replay-context note above).
AIPERF_PROGRESS_ARGS=()
run_aiperf() {
  "$AIPERF" profile \
    --scenario inferencex-agentx-mvp \
    --url "http://localhost:${PORT}" \
    --endpoint /v1/chat/completions \
    --endpoint-type chat --streaming --use-server-token-count \
    --model "$SERVE_MODEL" \
    --tokenizer "$MODEL" --tokenizer-trust-remote-code \
    --public-dataset "$DS" \
    --num-dataset-entries "$NENT" \
    --concurrency "$CONC" \
    --benchmark-duration "$DURATION" \
    --random-seed 42 \
    --trajectory-start-min-ratio 0.25 \
    --trajectory-start-max-ratio 0.75 \
    --warmup-requests-per-lane "$WARMLANE" \
    --warmup-grace-period "$WARMGRACE" \
    --trace-idle-gap-cap-seconds "$IDLEGAP" \
    --failed-request-threshold "$FRT" \
    --stats-interval 30 \
    --slice-duration 1.0 \
    --no-gpu-telemetry \
    ${CTX_ARGS[@]+"${CTX_ARGS[@]}"} \
    ${SMOKE_ARGS[@]+"${SMOKE_ARGS[@]}"} \
    ${AIPERF_PROGRESS_ARGS[@]+"${AIPERF_PROGRESS_ARGS[@]}"} \
    --artifact-dir "$ART" --ui simple
}

AIPERF_RC=0
if [ "${PROFILE:-0}" = "1" ]; then
  # Single-node trace capture: aiperf (a generic client) never triggers vLLM's
  # /start_profile the way InferenceX's benchmark_serving.py does, so this script
  # self-brackets a bounded profiling window inside aiperf's measured phase.
  # The profiler is already ENABLED on the server (builtin server phase adds the
  # framework's --profiler-config/env when PROFILE=1); /start_profile begins
  # recording and /stop_profile flushes the trace to torch_profiler_dir for
  # TraceLens. Only fires under PROFILE=1, so measurement rounds pay no cost.
  PWIN="${AGENTX_PROFILE_WINDOW_S:-20}"
  _require_uint AGENTX_PROFILE_WINDOW_S "$PWIN"
  : "${AGENTX_CAPTURE_ID:?AGENTX_CAPTURE_ID required for AgentX profiling}"
  : "${AGENTX_CAPTURE_STATUS_PATH:?AGENTX_CAPTURE_STATUS_PATH required for AgentX profiling}"
  PHASE_GATE="${BENCH_DIR}/aiperf_phase_gate.py"
  # AIPerf owns the AgentX warmup lifecycle, whose duration is not bounded by
  # the flat drain grace on very large models. Process liveness and Magpie's
  # outer profile deadline provide the hard bound.
  PHASE_WAIT_TIMEOUT=0
  CAPTURE_STATUS_FILE="$AGENTX_CAPTURE_STATUS_PATH"
  rm -f "$CAPTURE_STATUS_FILE"
  AIPERF_PROGRESS_PORT=""
  PHASE_GATE_FAILURE_REASON="profiling_phase_unavailable"
  _write_profile_capture_status() {
    _status="$1"
    _reason="$2"
    _phase_start_ns="${3:-0}"
    _decision="${4:-}"
    if [ ! -f "$PHASE_GATE" ]; then
      log "ERROR missing phase gate ${PHASE_GATE}; cannot write trace-capture status"
      return 0
    fi
    if ! python3 "$PHASE_GATE" write-capture-status \
      --output "$CAPTURE_STATUS_FILE" \
      --capture-id "$AGENTX_CAPTURE_ID" \
      --status "$_status" \
      --reason "$_reason" \
      --phase-start-ns "$_phase_start_ns" \
      --requested-window-seconds "$PWIN" \
      --decision-json "$_decision"; then
      log "ERROR failed to write trace-capture status via ${PHASE_GATE}"
    fi
  }
  if [ -n "${AGENTX_PROFILE_WARMUP_S:-}" ]; then
    log "WARN AGENTX_PROFILE_WARMUP_S is ignored: profiling now starts from AIPerf's measured-phase signal"
  fi
  if [ -f "$PHASE_GATE" ]; then
    if AIPERF_PROGRESS_PORT="$(python3 "$PHASE_GATE" pick-port)"; then
      AIPERF_PROGRESS_ARGS=(--api-host 127.0.0.1 --api-port "$AIPERF_PROGRESS_PORT")
    else
      PHASE_GATE_FAILURE_REASON="api_port_allocation_failed"
      log "WARN failed to allocate an AIPerf progress API port; the measurement will run without trace capture"
    fi
  else
    PHASE_GATE_FAILURE_REASON="phase_gate_missing"
    log "WARN missing AIPerf phase gate ${PHASE_GATE}; the measurement will run without trace capture"
  fi
  # A 200 OK from /stop_profile means the tracer was TOLD to stop, not that the
  # trace is on disk. MEASURED on GLM-5.3 (sglang, TP=8, one 20s window): the
  # first file appeared 350s after the call returned, all 8 ranks were present
  # at 391s, and the set was still growing at 546s on its way to 5.1 GB -- the
  # ranks serialise one after another. ``cleanup`` allows 20s before SIGKILL, so
  # every previous capture was killed mid-write: 8 files of plausible size that
  # all fail ``gzip -t``. That is the same corruption seen on a Kimi-K3 capture
  # and blamed at the time on copying the files too early; it was this.
  #
  # So wait for the set to be COMPLETE (one file per rank) and STABLE (total
  # size unchanged across consecutive samples) before returning and letting the
  # teardown run. Bounded, and loud on timeout -- a truncated trace that is
  # reported as a trace is worse than no trace, because TraceLens will read it.
  _trace_dirs() {
    _seen="|"
    # Configured profiler paths are valid before the server creates them; the
    # RESULT_DIR fallback is only evidence when that directory already exists.
    for d in "${SGLANG_TORCH_PROFILER_DIR:-}" "${VLLM_TORCH_PROFILER_DIR:-}"; do
      [ -n "$d" ] || continue
      case "$_seen" in *"|${d}|"*) continue ;; esac
      printf '%s\n' "$d"
      _seen="${_seen}${d}|"
    done
    d="${RESULT_DIR}/torch_trace"
    if [ -d "$d" ]; then
      case "$_seen" in *"|${d}|"*) ;; *) printf '%s\n' "$d" ;; esac
    fi
  }
  _trace_stat() {  # -> "<count> <total bytes>"
    _tc=0; _tb=0
    for _d in $(_trace_dirs); do
      for _f in "$_d"/*trace*; do
        [ -f "$_f" ] || continue
        _tc=$((_tc + 1))
        _sz=$(wc -c < "$_f" 2>/dev/null || echo 0)
        _tb=$((_tb + _sz))
      done
    done
    printf '%s %s' "$_tc" "$_tb"
  }
  _wait_for_trace_flush() {
    # Nothing was ever pointed at a directory, so there is nothing to flush.
    # Without this the loop below can never reach its stable-sample condition
    # (the count stays 0 forever) and burns the whole budget waiting for files
    # that no profiler was configured to write.
    if [ -z "$(_trace_dirs)" ]; then
      log "no profiler output directory is configured; nothing to wait for"
      return 2
    fi
    _want="${TP:-0}"
    case "$_want" in "" | *[!0-9]*) _want=0 ;; esac
    _budget="${AGENTX_TRACE_FLUSH_TIMEOUT_S:-1800}"
    case "$_budget" in "" | *[!0-9]*) _budget=1800 ;; esac
    # A separate, much shorter bound for "no file has appeared at all". A
    # capture that produced zero files is a failed capture (a rejected
    # /start_profile, a profiler that never armed) -- waiting out the full
    # flush budget for it buys nothing. The first rank file landed at t+350s on
    # the GLM-5.3 8-rank measurement, so the default leaves real margin.
    _first="${AGENTX_TRACE_FIRST_FILE_TIMEOUT_S:-900}"
    case "$_first" in "" | *[!0-9]*) _first=900 ;; esac
    [ "$_first" -gt "$_budget" ] && _first="$_budget"
    log "waiting for the profiler trace to finish writing (expect ${_want:-?} rank files, bound ${_budget}s, first-file bound ${_first}s)"
    _t0=$(date +%s); _prev=""; _stable=0
    while :; do
      sleep 10
      _now=$(_trace_stat); _cnt="${_now%% *}"; _el=$(( $(date +%s) - _t0 ))
      if [ "$_cnt" -gt 0 ] && [ "$_now" = "$_prev" ]; then
        _stable=$((_stable + 1))
      else
        _stable=0
      fi
      if [ "$_cnt" -eq 0 ] && [ "$_el" -ge "$_first" ]; then
        log "WARN no trace file appeared within ${_first}s of stop_profile; treating the capture as empty. Check that /start_profile was accepted and that the profiler output dir is writable. The benchmark measurement remains available, but trace capture will be marked failed."
        return 3
      fi
      # Three identical samples AND, when TP is known, one file per rank. The
      # count check matters: ranks appear one at a time, so a set that is merely
      # "not growing right now" can still be missing half its ranks.
      if [ "$_stable" -ge 3 ] && { [ "$_want" -eq 0 ] || [ "$_cnt" -ge "$_want" ]; }; then
        log "trace flush complete after ${_el}s: ${_cnt} file(s), $(( ${_now##* } / 1048576 )) MiB"
        return 0
      fi
      if [ "$_el" -ge "$_budget" ]; then
        log "WARN trace flush did not settle within ${_budget}s (${_cnt} file(s), $(( ${_now##* } / 1048576 )) MiB, expected ${_want} ranks). The files are very likely TRUNCATED and will fail gzip -t; raise AGENTX_TRACE_FLUSH_TIMEOUT_S. The benchmark measurement remains available, but trace capture will be marked failed."
        return 4
      fi
      _prev="$_now"
    done
  }
  log "PROFILE=1: waiting for AIPerf's measured phase before opening a ${PWIN}s profile window"
  run_aiperf & APID=$!
  PHASE_START_NS=""
  if [ -n "$AIPERF_PROGRESS_PORT" ] && PHASE_START_NS="$(
    python3 "$PHASE_GATE" wait-phase \
      --api-url "http://127.0.0.1:${AIPERF_PROGRESS_PORT}" \
      --phase profiling \
      --pid "$APID" \
      --timeout-seconds "$PHASE_WAIT_TIMEOUT"
  )"; then
    log "AIPerf measured phase started (start_ns=${PHASE_START_NS})"
    # SGLang takes its capture bounds in the /start_profile BODY, not on the
    # serve line. Hyperloom computes them into $PROFILE_EXTRA_BODY, but only
    # InferenceX's own client ever posted it -- a bare POST leaves the capture
    # unbounded, and the worker then accumulates profiler events in host RAM
    # until the cgroup OOM-killer takes it out mid-run. Forward the body when
    # there is one; vLLM ignores it (its bounds ride on --profiler-config.*).
    _pbody="${PROFILE_EXTRA_BODY:-}"
    if [ -n "$_pbody" ] && [ "$_pbody" != "{}" ]; then
      _pstart=(-H "Content-Type: application/json" -d "$_pbody")
      log "start_profile: forwarding capture bounds ${_pbody}"
    else
      _pstart=()
    fi
    if curl -sf -X POST "${_pstart[@]+"${_pstart[@]}"}" \
         "http://localhost:${PORT}/start_profile" >/dev/null 2>&1; then
      log "start_profile OK"
      if CAPTURE_RESULT="$(
        python3 "$PHASE_GATE" wait-capture-stop \
          --api-url "http://127.0.0.1:${AIPERF_PROGRESS_PORT}" \
          --phase profiling \
          --pid "$APID" \
          --max-window-seconds "$PWIN"
      )"; then
        log "profile capture stop decision: ${CAPTURE_RESULT}"
      else
        CAPTURE_RESULT='{"stop_reason":"capture_gate_failed"}'
        log "WARN capture stop gate failed; stopping the profiler immediately"
      fi
      STOP_OK=0
      if curl -sf -X POST "http://localhost:${PORT}/stop_profile" >/dev/null 2>&1; then
        STOP_OK=1
        log "stop_profile OK"
      else
        log "WARN stop_profile failed"
      fi
      FLUSH_RC=0
      _wait_for_trace_flush || FLUSH_RC=$?
      if [ "$STOP_OK" -eq 1 ] && [ "$FLUSH_RC" -eq 0 ]; then
        _write_profile_capture_status "succeeded" "capture_complete" "$PHASE_START_NS" "$CAPTURE_RESULT"
      elif [ "$STOP_OK" -ne 1 ]; then
        _write_profile_capture_status "failed" "stop_profile_failed" "$PHASE_START_NS" "$CAPTURE_RESULT"
      else
        case "$FLUSH_RC" in
          2) FLUSH_REASON="profiler_output_unconfigured" ;;
          3) FLUSH_REASON="trace_files_missing" ;;
          4) FLUSH_REASON="trace_flush_timeout" ;;
          *) FLUSH_REASON="trace_flush_failed" ;;
        esac
        _write_profile_capture_status "failed" "$FLUSH_REASON" "$PHASE_START_NS" "$CAPTURE_RESULT"
      fi
    else
      log "WARN start_profile failed (trace may be empty)"
      _write_profile_capture_status "failed" "start_profile_failed" "$PHASE_START_NS"
    fi
  else
    log "WARN AIPerf did not expose a measured phase; the measurement will finish without trace capture"
    _write_profile_capture_status "failed" "$PHASE_GATE_FAILURE_REASON"
  fi
  wait "$APID" || AIPERF_RC=$?
else
  run_aiperf || AIPERF_RC=$?
fi
log "aiperf exit=${AIPERF_RC}"

# Do not map a failed run: a nonzero aiperf must fail the benchmark, not emit a
# (possibly partial) result that Magpie would record as success.
if [ "$AIPERF_RC" -ne 0 ]; then
  log "ERROR: aiperf failed (rc=${AIPERF_RC}); not mapping a result"
  exit "$AIPERF_RC"
fi

# ── Map aiperf export -> InferenceX result schema ────────────────────────────
# ``-print -quit`` (no pipe) avoids a find|head SIGPIPE that would abort under
# ``set -o pipefail`` before the emptiness guard below.
PJ="$(find "$ART" -name 'profile_export_aiperf.json' -print -quit)"
if [ -z "$PJ" ]; then
  log "ERROR: no profile_export_aiperf.json produced"
  exit 1
fi
python3 "${BENCH_DIR}/map_aiperf.py" "$PJ" "${RESULT_DIR}/${RESULT_FILENAME}.json"
log "mapped -> ${RESULT_DIR}/${RESULT_FILENAME}.json"
