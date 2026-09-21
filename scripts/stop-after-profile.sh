#!/usr/bin/env bash
# Stop an inference_optimizer run once PRELUDE profile succeeds.
set -euo pipefail

OPT_PID="${1:?optimizer pid}"
SESSION_OR_LAUNCH="${2:?session dir or launch-info json}"
LOG="${3:-/dev/null}"

resolve_session_dir() {
  local arg="$1"
  if [[ -f "$arg" && "$arg" == *.json ]]; then
    python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("session_dir") or "")' "$arg" 2>/dev/null || true
  else
    echo "$arg"
  fi
}

SESSION_DIR="$(resolve_session_dir "$SESSION_OR_LAUNCH")"

deadline=$((SECONDS + 10800))  # 3h safety cap aligned with --max-hours 3

while kill -0 "$OPT_PID" 2>/dev/null; do
  if (( SECONDS > deadline )); then
    echo "[stop-after-profile] 3h cap reached; stopping pid $OPT_PID" | tee -a "$LOG"
    kill -TERM "$OPT_PID" 2>/dev/null || true
    break
  fi

  if [[ -z "$SESSION_DIR" || ! -d "$SESSION_DIR" ]]; then
    SESSION_DIR="$(resolve_session_dir "$SESSION_OR_LAUNCH")"
  fi

  if [[ -n "$SESSION_DIR" && -f "$SESSION_DIR/state.json" ]]; then
    status="$(python3 - <<'PY' "$SESSION_DIR/state.json"
import json, sys
st = json.load(open(sys.argv[1]))
print(st.get("last_profile_status") or "")
PY
)"
    if [[ "$status" == "succeeded" ]]; then
      echo "[stop-after-profile] profile succeeded; stopping pid $OPT_PID" | tee -a "$LOG"
      kill -TERM "$OPT_PID" 2>/dev/null || true
      break
    fi
  fi
  sleep 30
done

wait "$OPT_PID" 2>/dev/null || true
echo "[stop-after-profile] optimizer exited" | tee -a "$LOG"
