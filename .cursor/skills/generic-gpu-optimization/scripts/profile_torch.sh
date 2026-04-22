#!/usr/bin/env bash
# =============================================================================
# profile_torch.sh — wrap the BENCH_COMMAND with torch.profiler via PYTHONSTARTUP
# Outputs: $RESULT_DIR/profiles/torch_trace.json
# =============================================================================
set -euo pipefail
[ -n "${RESULT_DIR:-}" ] || { echo "RESULT_DIR not set"; exit 1; }
source "$RESULT_DIR/state.env"
source "$RESULT_DIR/detected.env"
[ -f "$RESULT_DIR/kept_env.sh" ] && source "$RESULT_DIR/kept_env.sh"

mkdir -p "$RESULT_DIR/profiles"
PROF_OUT="$RESULT_DIR/profiles/torch_trace.json"

# Inject a torch.profiler context via a wrapper script.
WRAPPER="$RESULT_DIR/profiles/torch_profile_wrapper.py"
cat > "$WRAPPER" <<PY
import os, runpy, sys
import torch
from torch.profiler import profile, schedule, ProfilerActivity, tensorboard_trace_handler

argv = sys.argv[1:]
if not argv:
    print("usage: torch_profile_wrapper.py <script.py> [args...]"); sys.exit(2)
script, *args = argv
sys.argv = [script] + args

acts = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
sched = schedule(wait=2, warmup=2, active=3, repeat=1)

with profile(activities=acts, schedule=sched,
             on_trace_ready=lambda p: p.export_chrome_trace("$PROF_OUT"),
             record_shapes=False) as p:
    runpy.run_path(script, run_name="__main__")
    p.step()
PY

cd "$REPO_ROOT"
# BENCH_COMMAND is "python <script> [args]". Replace "python" with our wrapper invocation.
WRAPPED=$(echo "$BENCH_COMMAND" | sed -E "s|^python(3?) |python\1 $WRAPPER |")
echo "[profile-torch] $WRAPPED"
bash -c "$WRAPPED"

[ -f "$PROF_OUT" ] || { echo "ERROR: torch.profiler did not produce $PROF_OUT"; exit 1; }
echo "[profile-torch] wrote $PROF_OUT"
