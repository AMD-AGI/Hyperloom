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
# around aiperf's steady state (see the PROFILE branch below).
#
# Inputs (from Magpie env): MODEL, TP, PORT, MAX_MODEL_LEN, CONC, RESULT_DIR,
#   RESULT_FILENAME, PROFILE, EXTRA_VLLM_ARGS, FRAMEWORK, GPU_TYPE/RUNNER_TYPE.
# AgentX knobs (AGENTX_ prefix; NOT AIPERF_, which aiperf's own settings read):
#   AGENTX_DATASET / WEKA_LOADER_OVERRIDE (pin the corpus loader),
#   AGENTX_NUM_ENTRIES (corpus cap; default 393 = all),
#   AGENTX_DURATION (measurement window; default 3600),
#   AGENTX_WARMUP_REQUESTS_PER_LANE (default 10),
#   AGENTX_WARMUP_GRACE_PERIOD (max drain wait; default 1800),
#   AGENTX_FAILED_REQUEST_THRESHOLD (error-rate abort ratio; default 0.10),
#   AGENTX_DATASET_CONFIG_TIMEOUT (default 1800), AGENTX_LIVE_ASSISTANT,
#   AGENTX_MMAP_CACHE_DIR (dataset mmap cache; defaults under $HF_HUB_CACHE),
#   AGENTX_MAX_CTX (explicit opt-in client-side context cap; NEVER inferred
#     from $MAX_MODEL_LEN -- see the replay-context note below),
#   AGENTX_KEEP_SERVER, AGENTX_PROFILE_WARMUP_S, AGENTX_PROFILE_WINDOW_S,
#   AGENTX_SERVER_SCRIPT (override builtin name), AIPERF_BIN.
set -euo pipefail

BENCH_DIR="$(cd "$(dirname "$0")" && pwd)"
log() { echo "[aiperf_client] $*"; }

: "${MODEL:?MODEL required}"
PORT="${PORT:-8000}"
CONC="${CONC:-16}"
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
DS="${AGENTX_DATASET:-${WEKA_LOADER_OVERRIDE:-$(_default_loader "$MODEL")}}"
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
WARMLANE="${AGENTX_WARMUP_REQUESTS_PER_LANE:-10}"
WARMGRACE="${AGENTX_WARMUP_GRACE_PERIOD:-1800}"

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
export AIPERF_DATASET_CONFIGURATION_TIMEOUT="${AGENTX_DATASET_CONFIG_TIMEOUT:-1800}"
export AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT="${AGENTX_DATASET_CONFIG_TIMEOUT:-1800}"
# Pre-canned assistant replay (recorded responses drive later turns).
export AIPERF_DATASET_WEKA_LIVE_ASSISTANT_RESPONSES="${AGENTX_LIVE_ASSISTANT:-0}"
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
FRT="${AGENTX_FAILED_REQUEST_THRESHOLD:-0.10}"

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
    --failed-request-threshold "$FRT" \
    --stats-interval 30 \
    --slice-duration 1.0 \
    --no-gpu-telemetry \
    ${CTX_ARGS[@]+"${CTX_ARGS[@]}"} \
    --artifact-dir "$ART" --ui simple
}

AIPERF_RC=0
if [ "${PROFILE:-0}" = "1" ]; then
  # Single-node trace capture: aiperf (a generic client) never triggers vLLM's
  # /start_profile the way InferenceX's benchmark_serving.py does, so this script
  # self-brackets a wall-clock profiling window around aiperf's steady state.
  # The profiler is already ENABLED on the server (builtin server phase adds the
  # framework's --profiler-config/env when PROFILE=1); /start_profile begins
  # recording and /stop_profile flushes the trace to torch_profiler_dir for
  # TraceLens. Only fires under PROFILE=1, so measurement rounds pay no cost.
  # Wall-clock delay measured from aiperf launch, so it has to clear everything
  # that happens before steady state: corpus load/reconstruct (0 on an mmap cache
  # hit, minutes on a miss), the per-lane cache warmup, and the warmup drain.
  # 60s used to be enough when the client replayed a handful of short traces; at
  # the upstream profile it lands squarely inside setup and captures nothing.
  PWARM="${AGENTX_PROFILE_WARMUP_S:-2700}"
  PWIN="${AGENTX_PROFILE_WINDOW_S:-20}"
  log "PROFILE=1: self-bracketing profile window (delay=${PWARM}s window=${PWIN}s)"
  run_aiperf & APID=$!
  sleep "$PWARM"
  if kill -0 "$APID" 2>/dev/null; then
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
    else
      log "WARN start_profile failed (trace may be empty)"
    fi
    sleep "$PWIN"
    curl -sf -X POST "http://localhost:${PORT}/stop_profile" >/dev/null 2>&1 \
      && log "stop_profile OK" || log "WARN stop_profile failed"
  else
    log "WARN aiperf finished before profile window opened; raise AGENTX_NUM_ENTRIES or lower AGENTX_PROFILE_WARMUP_S"
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

# ── Additionally emit the leaderboard-shaped aggregate ───────────────────────
# map_aiperf.py produces Magpie's schema (flat, ms, p99-bearing) -- that is what
# the search loop scores. The leaderboard's own numbers come from a different
# aggregator with a disjoint schema (nested, seconds, and no p99 at all), so one
# mapper cannot serve both. Run upstream's aggregator alongside, purely for
# manual comparison against published rows.
#
# Best-effort by design, with one rule: never take the run down. It needs the
# InferenceX checkout (module-relative imports, cwd at the repo root), which is
# absent in unit-test sandboxes -- so a missing script is a silent skip, and a
# failing one is a warning. What is NOT silent is the outcome: the marker below
# records either the artifact path or the reason it is missing, so "the
# leaderboard numbers quietly stopped being produced" cannot go unnoticed.
_agg_note="${RESULT_DIR}/${RESULT_FILENAME}.leaderboard.txt"
_ix_root="${INFERENCEX_PATH:-$(cd "${BENCH_DIR}/.." 2>/dev/null && pwd)}"
_agg_mod="${_ix_root}/utils/agentic/aggregation/process_agentic_result.py"
if [ ! -f "$_agg_mod" ]; then
  log "leaderboard aggregate: SKIPPED (no ${_agg_mod})"
  printf 'skipped: upstream aggregator not found at %s\n' "$_agg_mod" > "$_agg_note"
else
  # KV_OFFLOADING is required_env for the aggregator; AgentX does not configure
  # KV offloading, so declare that explicitly rather than let it exit.
  if ( cd "$_ix_root" \
       && RESULT_DIR="$RESULT_DIR" \
          AGENTIC_OUTPUT_DIR="$RESULT_DIR" \
          RESULT_FILENAME="${RESULT_FILENAME}.leaderboard" \
          KV_OFFLOADING="${KV_OFFLOADING:-none}" \
          python3 -m utils.agentic.aggregation.process_agentic_result ) >"${_agg_note}.log" 2>&1; then
    log "leaderboard aggregate -> ${RESULT_DIR}/${RESULT_FILENAME}.leaderboard.json"
    printf 'ok: %s\n' "${RESULT_DIR}/${RESULT_FILENAME}.leaderboard.json" > "$_agg_note"
  else
    log "WARN leaderboard aggregate failed; see ${_agg_note}.log (measurement itself is unaffected)"
    printf 'failed: see %s\n' "${_agg_note}.log" > "$_agg_note"
  fi
fi
