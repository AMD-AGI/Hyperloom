#!/usr/bin/env bash
# Run the *Python* Marathon harness (orchestrator + kernel_manager + watchdog + dashboard)
# for up to 24h on NFS so the job survives laptop disconnect. Uses skills under
#   inference_optimization/marathon_harness/skills/
# as the behavioral spec; the implementation is marathon_harness/*.py.
#
# Prerequisites on the GPU pod: Python venv, claude_code_sdk + Node + `claude` CLI,
# ANTHROPIC_API_KEY (or credits) in TBO/.env, inference server reachable when benchmarks run.
#
# Usage:
#   bash run_marathon_python_24h.sh start
#   bash run_marathon_python_24h.sh resume /path/to/DeepSeek-R1-0528-optimized/sessions/YYYYMMDD-HHMMSS
#
# Logs: $BASE_DIR/sessions/marathon_<timestamp>.log and marathon.pid next to the log.
#
set -euo pipefail

ACTION="${1:-start}"
SESSION_ARG="${2:-}"

MARATHON_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TBO_ROOT="$(cd "$MARATHON_ROOT/../.." && pwd)"
BASE_DIR="${BASE_DIR:-/shared_nfs/nehaprakriya/Agentic-InferenceX/DeepSeek-R1-0528-optimized}"
MODEL_NAME="${MODEL_NAME:-DeepSeek-R1-0528}"
MODEL_CLASS="${MODEL_CLASS:-moe_mla}"
MAX_HOURS="${MAX_HOURS:-24}"
FRAMEWORK="${FRAMEWORK:-sglang}"
TP="${TP:-8}"
GPU_TYPE="${GPU_TYPE:-MI355X}"
GPU_COUNT="${GPU_COUNT:-8}"
ENV_FILE="${ENV_FILE:-$TBO_ROOT/.env}"

cd "$MARATHON_ROOT"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "ERROR: env file not found: $ENV_FILE"
  exit 1
fi
if [[ ! -d "$BASE_DIR" ]]; then
  echo "ERROR: base_dir not found: $BASE_DIR"
  exit 1
fi
if [[ ! -f "$BASE_DIR/scripts/launch_server.sh" ]] && [[ ! -d "$BASE_DIR/handoff" ]]; then
  echo "WARNING: expected Agentic-InferenceX layout (scripts/launch_server.sh or handoff/). Continuing."
fi

TS="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$BASE_DIR/sessions/marathon_${TS}.log"
PID_FILE="$BASE_DIR/sessions/marathon_${TS}.pid"
mkdir -p "$BASE_DIR/sessions"

EXTRA=(--model-class "$MODEL_CLASS" --framework "$FRAMEWORK" --tp "$TP" --gpu-type "$GPU_TYPE" --gpu-count "$GPU_COUNT" --max-hours "$MAX_HOURS" --env-file "$ENV_FILE")

if [[ "$ACTION" == "resume" ]]; then
  if [[ -z "$SESSION_ARG" ]]; then
    echo "Usage: $0 resume /path/to/.../sessions/<id>"
    exit 1
  fi
  if [[ ! -d "$SESSION_ARG" ]]; then
    echo "ERROR: session dir not found: $SESSION_ARG"
    exit 1
  fi
  EXTRA+=(--session-dir "$SESSION_ARG" --resume)
  LOG_FILE="$SESSION_ARG/marathon_resume_${TS}.log"
  PID_FILE="$SESSION_ARG/marathon_resume_${TS}.pid"
elif [[ "$ACTION" != "start" ]]; then
  echo "Usage: $0 start | resume <session_dir>"
  exit 1
fi

echo "Marathon (Python) — model=$MODEL_NAME base=$BASE_DIR"
echo "Log: $LOG_FILE"
echo "Skill spec: $MARATHON_ROOT/marathon_harness/skills/SKILL.md"

CMD=(python -m marathon_harness.marathon "$MODEL_NAME" "$BASE_DIR" "${EXTRA[@]}")

nohup "${CMD[@]}" >>"$LOG_FILE" 2>&1 &
echo $! >"$PID_FILE"
disown || true

echo "Started PID $(cat "$PID_FILE") (detached)."
echo "Tail:   tail -f $LOG_FILE"
echo "Stop:   kill \$(cat $PID_FILE)"
echo "State:  ls ${SESSION_ARG:-$BASE_DIR/sessions/*/}/state.json 2>/dev/null | tail -1"
