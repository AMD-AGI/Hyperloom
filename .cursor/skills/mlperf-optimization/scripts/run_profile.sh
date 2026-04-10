#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# =============================================================================
# MLPerf Optimization — Profile Run
#
# Runs training with PyTorch profiler enabled to collect kernel traces.
# NOTE: Requires temporarily modifying the YAML config to enable profiling.
#
# Required env vars: MLPERF_DIR, CONFIG_SH, RESULT_DIR
# =============================================================================

: "${MLPERF_DIR:?MLPERF_DIR env var required}"
: "${CONFIG_SH:?CONFIG_SH env var required}"
: "${RESULT_DIR:?RESULT_DIR env var required}"

echo "============================================================"
echo "MLPerf Optimization — Profile"
echo "Results: $RESULT_DIR"
echo "============================================================"

# --- Enable profiling in YAML ---
echo "[1/3] Enabling profiling in YAML..."

cd "$MLPERF_DIR"
source "$CONFIG_SH"
EXP_FILE="$EXP"

python3 -c "
import yaml
with open('$EXP_FILE') as f:
    config = yaml.safe_load(f)
ov = config['modules']['pre_trainer']['overrides']
ov['profile'] = True
ov['use_pytorch_profiler'] = True
ov['profile_step_start'] = 6
ov['profile_step_end'] = 7
with open('$EXP_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print('Profiling enabled')
"

# --- Run profiling (Tier 1 trial) ---
echo "[2/3] Running profiling pass (10 iters, Tier 1)..."

run_mlperf_trial "profile" 1 10

# --- Restore YAML ---
echo "[3/3] Restoring YAML (disabling profiling)..."

python3 -c "
import yaml
with open('$EXP_FILE') as f:
    config = yaml.safe_load(f)
ov = config['modules']['pre_trainer']['overrides']
ov['profile'] = False
ov['profile_step_start'] = 60
ov['profile_step_end'] = 61
with open('$EXP_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print('Profiling disabled')
"

# Find and copy trace
TRACE_FILE=$(find /workspace /tmp /root -name "*.pt.trace.json" -newer "$RESULT_DIR/attempt_profile_raw.log" 2>/dev/null | head -1)
if [ -n "$TRACE_FILE" ]; then
    cp "$TRACE_FILE" "$RESULT_DIR/baseline_trace.json"
    echo "Trace saved to $RESULT_DIR/baseline_trace.json"
    filter_trace "$RESULT_DIR/baseline_trace.json" "$RESULT_DIR/filtered_trace.json"
else
    echo "WARNING: No trace file found"
fi

echo ""
echo "============================================================"
echo "Profile complete. Results: $RESULT_DIR"
echo "============================================================"
