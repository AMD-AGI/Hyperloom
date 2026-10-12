#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Runs ONE pre-release validation leg inside an environment that already IS the
# leg's isolated, single-GPU container (whoever scheduled it chose the image and
# the GPU). From the leg id alone it derives the scenario, backend, duration and
# kernel backend, installs this checkout's wheel into a per-leg workspace, writes
# that workspace's .env, then drives the setup prompt and the demo prompt through
# `claude --print` (resuming one session, nudging a turn that ended early) and
# waits for the optimizer's state.json to reach a terminal stop_reason.
#
# HYPERLOOM_RUN_MODE is always baremetal here: a `docker-*` leg means "the
# framework image is the container", not docker-in-docker. The DinD prompts
# (setup-docker-*.md) belong to bootstrap-pre-release.sh, not to this runner.
#
# Usage:
#   run-leg.sh <leg-id>              run the leg; exit 0 = pass, 1 = fail, 2 = bad input
#   run-leg.sh --resolve <leg-id>    print what the leg resolves to and exit; needs no env
#
# Leg id: (baremetal|docker)-(sglang|vllm|atom)[-forge]-(3h|12h). The 3h legs run
# Qwen3-8B (bf16, --no-kernel), the 12h legs Qwen3-14B-FP8; -forge selects the
# KernelForge kernel backend. Combinations without a prompt (e.g. a -forge 3h leg)
# are rejected by --resolve.
#
# Required env (no defaults: every location, model, endpoint and credential comes
# from the caller):
#   CI_VERSION          unique id of this attempt; keys the workspace and the claude
#                       session, so a new attempt never resumes an old conversation
#   LEG_WORK_DIR        writable root; the leg workspace is LEG_WORK_DIR/CI_VERSION/<leg>
#   USER_DATA_PATH      Hyperloom session root (written to .env unchanged)
#   LEG_MODEL_PATH      model directory for this leg, or
#   LEG_MODELS_DIR      a directory holding Qwen3-8B/ and Qwen3-14B-FP8/
#   ANTHROPIC_API_KEY  ANTHROPIC_BASE_URL  CLAUDE_MODEL
#   CLAUDE_CLI_VERSION  only when no `claude` is on PATH
# Optional env: LEG_WHEEL (prebuilt wheel instead of building this checkout),
#   HF_TOKEN, ANTHROPIC_CUSTOM_HEADERS, INFERENCE_OPTIMIZER_SESSION_LAYOUT,
#   GEAK_CLAUDE_MODEL, FORGE_CLAUDE_MODEL, OPENAI_BASE_URL, OPENAI_API_KEY,
#   CODEX_MODEL; timing knobs LEG_SETUP_DEADLINE_S, LEG_SETUP_STALL_S,
#   LEG_SETUP_MAX_TURNS, LEG_TURN_GAP_S, LEG_WAIT_INTERVAL_S, LEG_STALL_GRACE_S,
#   LEG_DEMO_REDRIVES, LEG_DEADLINE_S.
#
# Secrets: the LLM key is written only into the leg's 0600 .env (Hyperloom's setup
# treats that file as authoritative and drops ambient ANTHROPIC_* values), and the
# EXIT trap scrubs it. Nothing here prints a secret value.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
PROMPTS_DIR="$SCRIPT_DIR/prompts/pre-release"

LEG_ID_RE='^(baremetal|docker)-(sglang|vllm|atom)(-forge)?-(3h|12h)$'
# Mirrors SUCCESS_STOP_REASONS in hyperloom/inference_optimizer/breakdown/stop_reasons.py.
CLEAN_STOP_REASONS="target_reached global_converged time_exhausted max_ticks sweep_done robustness_escalated"
# Keys copied from the caller's env into .env when set; secret ones are scrubbed on exit.
PASSTHROUGH_KEYS="ANTHROPIC_CUSTOM_HEADERS GEAK_CLAUDE_MODEL FORGE_CLAUDE_MODEL OPENAI_BASE_URL OPENAI_API_KEY CODEX_MODEL INFERENCE_OPTIMIZER_SESSION_LAYOUT"
SECRET_KEYS="ANTHROPIC_API_KEY ANTHROPIC_CUSTOM_HEADERS OPENAI_API_KEY"

log() { echo "[run-leg $(date -u +%H:%M:%S)] $*"; }
die_usage() { echo "run-leg: $*" >&2; exit 2; }

# Sets LEG SCENARIO BACKEND FORGE HOURS MODEL_NAME SETUP_PROMPT DEMO_PROMPT, or fails.
resolve_leg() {
  LEG="$1"
  [[ "$LEG" =~ $LEG_ID_RE ]] || die_usage "invalid leg id '$LEG' (expected (baremetal|docker)-(sglang|vllm|atom)[-forge]-(3h|12h))"
  SCENARIO="${BASH_REMATCH[1]}"
  BACKEND="${BASH_REMATCH[2]}"
  FORGE=0; [ -n "${BASH_REMATCH[3]}" ] && FORGE=1
  HOURS="${BASH_REMATCH[4]%h}"
  case "$HOURS" in
    3) MODEL_NAME=Qwen3-8B ;;
    12) MODEL_NAME=Qwen3-14B-FP8 ;;
  esac
  if [ "$SCENARIO" = baremetal ]; then
    SETUP_PROMPT="$PROMPTS_DIR/setup-baremetal-${BACKEND}.md"
  else
    SETUP_PROMPT="$PROMPTS_DIR/setup-image-${BACKEND}.md"
  fi
  local suffix=""
  if [ "$BACKEND" = atom ] && [ "$HOURS" != 3 ]; then
    suffix="-atom"
  fi
  [ "$FORGE" = 1 ] && suffix="${suffix}-forge"
  DEMO_PROMPT="$PROMPTS_DIR/demo-${HOURS}h${suffix}.md"
  [ -f "$SETUP_PROMPT" ] || die_usage "leg '$LEG' has no setup prompt ($SETUP_PROMPT)"
  [ -f "$DEMO_PROMPT" ] || die_usage "leg '$LEG' has no demo prompt ($DEMO_PROMPT)"
}

print_resolution() {
  printf 'leg=%s\nscenario=%s\nbackend=%s\nforge=%s\nhours=%s\nmodel=%s\nsetup_prompt=%s\ndemo_prompt=%s\n' \
    "$LEG" "$SCENARIO" "$BACKEND" "$FORGE" "$HOURS" "$MODEL_NAME" \
    "${SETUP_PROMPT#"$SRC_DIR"/}" "${DEMO_PROMPT#"$SRC_DIR"/}"
}

require_env() {
  local missing=() k
  for k in CI_VERSION LEG_WORK_DIR USER_DATA_PATH ANTHROPIC_API_KEY ANTHROPIC_BASE_URL CLAUDE_MODEL; do
    [ -n "${!k:-}" ] || missing+=("$k")
  done
  [ -n "${LEG_MODEL_PATH:-}${LEG_MODELS_DIR:-}" ] || missing+=("LEG_MODEL_PATH or LEG_MODELS_DIR")
  [ "${#missing[@]}" -eq 0 ] || die_usage "missing required env: ${missing[*]}"
  case "$CI_VERSION" in */*|.|..) die_usage "CI_VERSION must be a single path segment" ;; esac
  MODEL_PATH="${LEG_MODEL_PATH:-${LEG_MODELS_DIR%/}/$MODEL_NAME}"
  [ -f "$MODEL_PATH/config.json" ] || die_usage "no config.json under the model path $MODEL_PATH"
}

# Prints the key's value, or the default when it is missing, null, false or the file
# is unreadable. python3, not jq: some framework images cannot install jq.
state_field() {
  python3 - "$1" "$2" "$3" 2>/dev/null <<'PY'
import json, sys

path, key, default = sys.argv[1:4]
try:
    with open(path) as fh:
        value = json.load(fh).get(key)
except Exception:
    value = None
if value is None or value is False or value == "":
    print(default)
elif value is True:
    print("true")
else:
    print(value)
PY
}

is_clean_stop_reason() {
  case " $CLEAN_STOP_REASONS " in *" $1 "*) return 0 ;; *) return 1 ;; esac
}

# Deterministic UUID from (leg, attempt): version nibble 4, variant nibble 8.
leg_session_uuid() {
  local h; h="$(printf '%s|%s' "$1" "$2" | sha1sum | cut -c1-32)"
  printf '%s-%s-4%s-8%s-%s' "${h:0:8}" "${h:8:4}" "${h:12:3}" "${h:15:3}" "${h:18:12}"
}

# Seconds since the newest file under the workspace or the session root was
# written, floored at $2 so a previous attempt's files never count as progress.
leg_idle_s() {
  local since="$1" now="$2" last
  # awk reads all input: `sort | head` would SIGPIPE on a large tree and, under
  # pipefail, end the runner.
  last="$(find "$ROOT" "$USER_DATA_PATH" -type f -printf '%T@\n' 2>/dev/null \
          | awk '$1 > m { m = $1 } END { if (m) printf "%d\n", m }' || true)"
  if [ -n "$last" ] && [ "$last" -ge "$since" ] 2>/dev/null; then since="$last"; fi
  echo $(( now - since ))
}

ensure_claude_cli() {
  if command -v claude >/dev/null 2>&1; then
    log "claude CLI present: $(claude --version 2>/dev/null || echo unknown)"; return 0
  fi
  : "${CLAUDE_CLI_VERSION:?no claude on PATH and CLAUDE_CLI_VERSION is unset}"
  if ! command -v npm >/dev/null 2>&1; then
    local node_version="${NODE_VERSION:-v20.18.0}" node_prefix="$ROOT/.node"
    log "installing Node ${node_version} under $node_prefix"
    mkdir -p "$node_prefix"
    curl -fsSL "https://nodejs.org/dist/${node_version}/node-${node_version}-linux-x64.tar.xz" \
      | tar -xJ -C "$node_prefix" --strip-components=1
    export PATH="$node_prefix/bin:$PATH"
  fi
  log "installing Claude CLI @ ${CLAUDE_CLI_VERSION}"
  npm install -g "@anthropic-ai/claude-code@${CLAUDE_CLI_VERSION}" >"$ROOT/npm-claude.log" 2>&1 \
    || { log "ERROR: claude CLI install failed (see $ROOT/npm-claude.log)"; return 1; }
  command -v claude >/dev/null 2>&1 || { log "ERROR: claude not on PATH after install"; return 1; }
}

install_hyperloom() {
  local wheel="${LEG_WHEEL:-}"
  if [ -z "$wheel" ]; then
    local wheel_dir="$ROOT/wheel"
    rm -rf "$wheel_dir"; mkdir -p "$wheel_dir"
    log "building the wheel from $SRC_DIR"
    pip wheel --no-input --no-deps --wheel-dir "$wheel_dir" "$SRC_DIR" >"$ROOT/pip-wheel.log" 2>&1 \
      || { log "ERROR: wheel build failed (see $ROOT/pip-wheel.log)"; return 1; }
    local built=("$wheel_dir"/hyperloom_inference_optimizer-*.whl)
    wheel="${built[0]}"
  fi
  [ -f "$wheel" ] || { log "ERROR: no wheel at $wheel"; return 1; }
  log "pip install $(basename "$wheel") --target $ROOT"
  pip install --no-input --target "$ROOT" "$wheel" >"$ROOT/pip-install.log" 2>&1 \
    || { log "ERROR: wheel install failed (see $ROOT/pip-install.log)"; return 1; }
}

write_dotenv() {
  local k
  ( umask 077
    {
      echo "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}"
      echo "ANTHROPIC_BASE_URL=${ANTHROPIC_BASE_URL}"
      echo "CLAUDE_MODEL=${CLAUDE_MODEL}"
      for k in $PASSTHROUGH_KEYS; do
        # Double-quoted: a header value carries a space and a colon.
        [ -n "${!k:-}" ] && printf '%s="%s"\n' "$k" "${!k}"
      done
      echo "USER_DATA_PATH=${USER_DATA_PATH}"
      echo "HYPERLOOM_RUN_MODE=baremetal"
      echo "FRAMEWORK=${BACKEND}"
      echo "MODEL_PATH=${MODEL_PATH}"
      [ "$FORGE" = 1 ] && echo "KERNEL_OPT_BACKEND_ORDER=forge"
      true
    } > "$ENV_FILE"
  )
}

scrub_dotenv() {
  local k
  [ -f "$ENV_FILE" ] || return 0
  for k in $SECRET_KEYS; do sed -i "/^${k}=/d" "$ENV_FILE" 2>/dev/null || true; done
}

agent_turn() {
  claude --print --dangerously-skip-permissions "$@" >>"$AGENT_LOG" 2>&1
}

SETUP_RESUME_NUDGE='Your previous turn ended before setup finished. Do NOT restart anything that is already running. Check on the install you launched: if it is still in progress, keep monitoring it and only answer once it has finished. Once it has finished successfully, complete any remaining setup steps and then reply with exactly the completion line the setup instructions asked for. If it has failed, report the failure.'
DEMO_RESUME_NUDGE='Your previous turn ended without leaving a running optimize behind, and nothing has been written under the workspace since, so the work is not progressing. Do NOT fabricate a result. Finish the launch in THIS turn: complete the install if it is still needed, start optimize detached with setsid nohup so it survives the end of this turn, then confirm the nested session run dir and its state.json exist and report their paths.'

# Setup: resume one session until it prints the completion line, bounded by an
# overall deadline, a no-progress window and a turn count.
run_setup() {
  local deadline="${LEG_SETUP_DEADLINE_S:-2700}" stall="${LEG_SETUP_STALL_S:-600}"
  local max_turns="${LEG_SETUP_MAX_TURNS:-30}" turn=0 now idle
  while :; do
    turn=$(( turn + 1 ))
    if [ "$turn" = 1 ]; then
      log "setup turn 1 (session $SESSION_UUID)"
      agent_turn --session-id "$SESSION_UUID" < "$SETUP_PROMPT" \
        || log "WARN: setup turn 1 exited non-zero; anything it detached keeps running"
    else
      log "setup turn $turn (resuming $SESSION_UUID)"
      if ! printf '%s\n' "$SETUP_RESUME_NUDGE" | agent_turn --resume "$SESSION_UUID"; then
        log "WARN: could not resume $SESSION_UUID; re-feeding the setup prompt"
        agent_turn < "$SETUP_PROMPT" || true
      fi
    fi
    if grep -qiE "setup complete: (baremetal|docker)/${BACKEND}" "$AGENT_LOG"; then
      log "setup reported complete on turn $turn"; return 0
    fi
    now="$(date +%s)"; idle="$(leg_idle_s "$LEG_T0" "$now")"
    if [ "$idle" -ge "$stall" ]; then log "ERROR: setup stalled, nothing written for ${idle}s"; return 1; fi
    if [ $(( now - LEG_T0 )) -ge "$deadline" ]; then log "ERROR: setup incomplete after ${deadline}s ($turn turns)"; return 1; fi
    if [ "$turn" -ge "$max_turns" ]; then log "ERROR: setup incomplete after $max_turns turns"; return 1; fi
    log "WARN: setup turn $turn ended early (idle ${idle}s); resuming"
    sleep "${LEG_TURN_GAP_S:-30}"
  done
}

# Newest state.json under the session root written by this attempt, as its dir.
find_session_dir() {
  find "$USER_DATA_PATH" -mindepth 2 -type f -name state.json -newermt "@$LEG_T0" -printf '%T@ %h\n' 2>/dev/null \
    | sort -rn | awk 'NR == 1 { sub(/^[^ ]+ /, ""); print }' || true
}

# Demo: one resumed turn launches optimize detached; then poll state.json until a
# terminal stop_reason with the close sequence done. A launch that left nothing
# running is re-driven a bounded number of times.
run_demo() {
  local interval="${LEG_WAIT_INTERVAL_S:-45}" grace="${LEG_STALL_GRACE_S:-600}"
  local deadline="${LEG_DEADLINE_S:-$(( HOURS * 3600 + 3600 ))}" max_redrives="${LEG_DEMO_REDRIVES:-5}"
  local start now elapsed grace_ts sdir="" state stop close redrives=0
  log "demo ${HOURS}h turn (resuming $SESSION_UUID)"
  if ! agent_turn --resume "$SESSION_UUID" < "$DEMO_PROMPT"; then
    log "WARN: could not resume $SESSION_UUID for the demo; running it standalone"
    agent_turn < "$DEMO_PROMPT" || true
  fi
  start="$(date +%s)"; grace_ts="$start"
  while :; do
    now="$(date +%s)"; elapsed=$(( now - start ))
    if [ -z "$sdir" ]; then
      sdir="$(find_session_dir)"
      if [ -n "$sdir" ]; then
        log "session dir: $sdir"
      elif [ "$(leg_idle_s "$grace_ts" "$now")" -ge "$grace" ]; then
        if [ "$redrives" -ge "$max_redrives" ]; then
          log "ERROR: no state.json after $redrives demo re-drive(s)"; return 1
        fi
        redrives=$(( redrives + 1 ))
        log "WARN: no state.json and no progress; re-driving the launch ($redrives/$max_redrives)"
        if ! printf '%s\n' "$DEMO_RESUME_NUDGE" | agent_turn --resume "$SESSION_UUID"; then
          agent_turn < "$DEMO_PROMPT" || true
        fi
        grace_ts="$(date +%s)"
        continue
      fi
    fi
    if [ -n "$sdir" ]; then
      state="$sdir/state.json"
      stop="$(state_field "$state" stop_reason "")"
      close="$(state_field "$state" close_sequence_done false)"
      # stop_reason first: a baseline_failed run also writes final.json.
      if [ -n "$stop" ] && [ "$close" = true ]; then
        if is_clean_stop_reason "$stop"; then log "PASS: stop_reason=$stop after ${elapsed}s"; return 0; fi
        log "FAIL: stop_reason=$stop is not a clean terminal exit"; return 1
      fi
      if [ -z "$stop" ] && [ "$close" = true ] && [ -f "$sdir/reports/final.json" ]; then
        log "PASS: final.json present and close sequence done after ${elapsed}s"; return 0
      fi
    fi
    if [ "$elapsed" -ge "$deadline" ]; then
      log "ERROR: no terminal stop_reason within ${deadline}s (session dir: ${sdir:-none})"; return 1
    fi
    sleep "$interval"
  done
}

main() {
  if [ "${1:-}" = --resolve ]; then
    [ "$#" -eq 2 ] || die_usage "usage: run-leg.sh --resolve <leg-id>"
    resolve_leg "$2"; print_resolution; return 0
  fi
  [ "$#" -eq 1 ] || die_usage "usage: run-leg.sh <leg-id> | --resolve <leg-id>"
  resolve_leg "$1"
  require_env

  ROOT="${LEG_WORK_DIR%/}/${CI_VERSION}/${LEG}"
  mkdir -p "$ROOT" "$USER_DATA_PATH"
  # One runner per workspace: a second start would drive the same session twice.
  if command -v flock >/dev/null 2>&1; then
    exec 9>"$ROOT/.run-leg.lock"
    flock -n 9 || { log "ERROR: another run-leg.sh holds $ROOT"; exit 2; }
  fi
  ENV_FILE="$ROOT/.env"
  AGENT_LOG="$ROOT/agent.log"
  LEG_T0="$(date +%s)"
  SESSION_UUID="$(leg_session_uuid "$LEG" "$CI_VERSION")"
  trap scrub_dotenv EXIT

  log "leg=$LEG scenario=$SCENARIO backend=$BACKEND forge=$FORGE hours=$HOURS model=$MODEL_PATH"
  log "workspace=$ROOT user_data_path=$USER_DATA_PATH hf_token=$([ -n "${HF_TOKEN:-}" ] && echo set || echo unset)"
  log "setup_prompt=${SETUP_PROMPT#"$SRC_DIR"/} demo_prompt=${DEMO_PROMPT#"$SRC_DIR"/}"

  ensure_claude_cli
  install_hyperloom
  write_dotenv

  # The claude CLI reads ANTHROPIC_MODEL, not Hyperloom's CLAUDE_MODEL.
  export ANTHROPIC_MODEL="$CLAUDE_MODEL" ANTHROPIC_SMALL_FAST_MODEL="$CLAUDE_MODEL"
  export IS_SANDBOX=1 REPO_ROOT="$ROOT" PYTHONPATH="$ROOT:${PYTHONPATH:-}"
  cd "$ROOT"

  run_setup || exit 1
  run_demo || exit 1
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
