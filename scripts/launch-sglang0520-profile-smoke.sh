#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
cd "$REPO_ROOT"

_dotenv_prev="$(export -p | grep -v -e '=""$' -e "=''\$")"
set -a; . "$REPO_ROOT/.env"; set +a
eval "$_dotenv_prev"; unset _dotenv_prev

export USER_DATA_PATH="${USER_DATA_PATH:?USER_DATA_PATH missing}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HYPERLOOM_KERNEL_AGENT_ROOT="$REPO_ROOT/src/hyperloom/agents/kernel"
export KERNEL_AGENT_ROOT="$HYPERLOOM_KERNEL_AGENT_ROOT"
ulimit -Sn 65536 || true

. "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${MAGPIE_PATH:-/shared_nfs/yunkai/Hyperloom/.cache/Magpie}:${PYTHONPATH:-}"
export PYTHONPATH="$(printf '%s' "$PYTHONPATH" | tr ':' '\n' | sed "s|/shared_nfs/yunkai/Hyperloom|${REPO_ROOT}|g" | awk '!seen[$0]++' | paste -sd: -)"

RUN_DIR="$USER_DATA_PATH/optimizer_runs"
mkdir -p "$RUN_DIR"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$RUN_DIR/sglang0520_qwen06b_profile_${TS}.log"
LAUNCH_INFO="$RUN_DIR/launch_sglang0520_qwen06b_${TS}.json"
PIDFILE="$RUN_DIR/run_sglang0520_qwen06b_${TS}.pid"
META="$RUN_DIR/sglang0520_qwen06b_${TS}.meta"

MODEL="${MODEL_PATH:-/shared_nfs/models/Qwen3-0.6B}"

{
  echo "REPO_ROOT=$REPO_ROOT"
  echo "MODEL=$MODEL"
  echo "LOG=$LOG"
  echo "LAUNCH_INFO=$LAUNCH_INFO"
  echo "PIDFILE=$PIDFILE"
  echo "started_utc=$TS"
} >"$META"

setsid nohup python3 -m hyperloom.inference_optimizer.cli optimize \
  --model "$MODEL" \
  --framework sglang \
  --model-class dense \
  --tp 1 \
  --conc 64 \
  --isl 1024 \
  --osl 1024 \
  --precision bf16 \
  --max-hours 3 \
  --max-minutes-sweep-pct 0.01 \
  --no-framework-agent \
  --no-kernel \
  --no-enable-roofline \
  --no-enable-conc-sweep \
  --no-research-scout \
  --extra-env "SGLANG_USE_AITER=${SGLANG_USE_AITER:-1}" \
  --launch-info-file "$LAUNCH_INFO" \
  >>"$LOG" 2>&1 &

OPT_PID=$!
echo "$OPT_PID" >"$PIDFILE"
echo "optimizer_pid=$OPT_PID"

# Wait for session_dir in launch info (up to 10 min).
SESSION_DIR=""
for _ in $(seq 1 120); do
  if [[ -f "$LAUNCH_INFO" ]]; then
    SESSION_DIR="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("session_dir") or "")' "$LAUNCH_INFO" 2>/dev/null || true)"
    if [[ -n "$SESSION_DIR" && -d "$SESSION_DIR" ]]; then
      echo "session_dir=$SESSION_DIR"
      echo "session_dir=$SESSION_DIR" >>"$META"
      break
    fi
  fi
  sleep 5
done

if [[ -z "$SESSION_DIR" ]]; then
  echo "WARN: session_dir not resolved yet; monitor will retry from launch info" | tee -a "$LOG"
fi

MONITOR_LOG="$RUN_DIR/stop_after_profile_${TS}.log"
nohup bash "$REPO_ROOT/scripts/stop-after-profile.sh" "$OPT_PID" "$LAUNCH_INFO" "$MONITOR_LOG" >>"$MONITOR_LOG" 2>&1 &
echo "monitor_log=$MONITOR_LOG"
