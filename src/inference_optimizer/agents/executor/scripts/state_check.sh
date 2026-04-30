#!/usr/bin/env bash
# state_check.sh — extract key fields from $SESSION_DIR/state.json so the
# executor can quickly reason about progress without reading the whole file.
#
# Usage:   bash $AGENT_PKG_DIR/scripts/state_check.sh
# Env:     SESSION_DIR (required)
#
# Output (one field per line):
#   mode=guided_kernel_opt
#   model=DeepSeek-R1-0528 (model_class=moe_mla)
#   time=elapsed=12.3m left=347.7m max=360.0m
#   tput=baseline=420.0 current=482.3 cumulative_gain=14.83%
#   crashes=0
#   current_action=kernel_opt
#   stop_reason=(running)
#   last_decisions:
#     - 2026-04-29T07:01:23 from=conductor changes={current_tput: 482.3, ...}
#     - ...
#
# Read-only. Never modifies state.json.

set -euo pipefail

if [ -z "${SESSION_DIR:-}" ]; then
    echo "ERROR: SESSION_DIR must be set" >&2
    exit 2
fi

SNAP="$SESSION_DIR/state.json"
if [ ! -f "$SNAP" ]; then
    echo "(state.json not found at $SNAP — clock has not flushed yet?)"
    exit 0
fi

python3 - "$SNAP" <<'PY'
import json, sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as fh:
    s = json.load(fh)

mode = s.get("execution_mode", "?")
model_name = s.get("model_name") or s.get("model_path") or "?"
model_class = s.get("model_class", "unknown")
elapsed = s.get("elapsed_minutes", 0.0)
max_m = s.get("max_minutes", 0.0)
left = max(0.0, max_m - elapsed)
base = s.get("baseline_tput", 0.0)
cur = s.get("current_tput", 0.0)
gain = s.get("cumulative_gain", 0.0)
crashes = s.get("crash_count", 0)
current_action = s.get("current_action") or "(idle)"
stop_reason = s.get("stop_reason") or "(running)"

print(f"mode={mode}")
print(f"model={model_name} (model_class={model_class})")
print(f"time=elapsed={elapsed:.2f}m left={left:.2f}m max={max_m:.2f}m")
print(f"tput=baseline={base:.2f} current={cur:.2f} cumulative_gain={gain:.2f}%")
print(f"crashes={crashes}")
print(f"current_action={current_action}")
print(f"stop_reason={stop_reason}")

decisions = s.get("decisions_tail") or []
if decisions:
    print("last_decisions:")
    for d in decisions[-5:]:
        ts = d.get("ts", "?")
        fr = d.get("from", "?")
        ch = d.get("changes", {})
        ch_str = ", ".join(f"{k}={v}" for k, v in ch.items())
        print(f"  - ts={ts} from={fr} changes={{{ch_str}}}")
else:
    print("last_decisions: (none)")
PY
