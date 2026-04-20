#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# =============================================================================
# MLPerf Optimization — Profile Run
#
# Runs training with PyTorch profiler enabled to collect kernel traces.
# Produces: baseline_trace.json, filtered_trace.json(.gz), and parsed
# analysis files (profiler_summary.json, categories.json, etc.).
#
# Required env vars: MLPERF_DIR, CONFIG_SH, RESULT_DIR
# =============================================================================

: "${MLPERF_DIR:?MLPERF_DIR env var required}"
: "${CONFIG_SH:?CONFIG_SH env var required}"
: "${RESULT_DIR:?RESULT_DIR env var required}"

TB_TRACE_DIR="$RESULT_DIR/tb_traces"

echo "============================================================"
echo "MLPerf Optimization — Profile"
echo "Results: $RESULT_DIR"
echo "Trace output: $TB_TRACE_DIR"
echo "============================================================"

# --- Enable profiling in YAML ---
echo "[1/7] Enabling profiling in YAML..."

cd "$MLPERF_DIR"
source "$CONFIG_SH"
EXP_FILE="$EXP"

python3 -c "
import yaml, os

TB_TRACE_DIR = '$TB_TRACE_DIR'
os.makedirs(TB_TRACE_DIR, exist_ok=True)

with open('$EXP_FILE') as f:
    config = yaml.safe_load(f)
ov = config['modules']['pre_trainer']['overrides']
ov['profile'] = True
ov['use_pytorch_profiler'] = True
ov['profile_step_start'] = 5
ov['profile_step_end'] = 8
ov['tensorboard_dir'] = TB_TRACE_DIR
ov['torch_profiler_use_gzip'] = True
with open('$EXP_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print('Profiling enabled (steps 5-8, tensorboard_dir=' + TB_TRACE_DIR + ')')
"

# --- Run profiling (Tier 1 trial) ---
echo "[2/7] Running profiling pass (10 iters, Tier 1)..."

run_mlperf_trial "profile" 1 10

# --- Restore YAML ---
echo "[3/7] Restoring YAML (disabling profiling)..."

python3 -c "
import yaml
with open('$EXP_FILE') as f:
    config = yaml.safe_load(f)
ov = config['modules']['pre_trainer']['overrides']
ov['profile'] = False
ov['use_pytorch_profiler'] = True
ov['profile_step_start'] = 60
ov['profile_step_end'] = 61
ov['tensorboard_dir'] = '/workspace/code/tensorboard'
ov['torch_profiler_use_gzip'] = False
with open('$EXP_FILE', 'w') as f:
    yaml.dump(config, f, default_flow_style=False)
print('Profiling disabled, YAML restored')
"

# --- Locate trace: known dir first, then discover_trace() fallback ---
echo "[4/7] Locating and filtering trace..."

TRACE_FOUND=""

# Detect actual tensorboard path from training log (framework ignores tensorboard_dir YAML
# and prints the real path: "args.tensorboard_dir is deprecated, the tensorboard path is: <path>")
ACTUAL_TB_DIR=""
RAW_LOG="$RESULT_DIR/attempt_profile_raw.log"
if [ -f "$RAW_LOG" ]; then
    # Strip ANSI escape codes before extracting path
    ACTUAL_TB_DIR=$(sed 's/\x1b\[[0-9;]*m//g' "$RAW_LOG" \
        | grep -oP 'the tensorboard path is: \K\S+' | tail -1)
fi
if [ -n "$ACTUAL_TB_DIR" ]; then
    echo "Detected actual tensorboard dir from log: $ACTUAL_TB_DIR"
fi

# Search order: actual framework dir → our configured TB_TRACE_DIR → discover_trace()
for SEARCH_DIR in "$ACTUAL_TB_DIR" "$TB_TRACE_DIR"; do
    [ -z "$SEARCH_DIR" ] && continue
    [ -d "$SEARCH_DIR" ] || continue

    # Use find -printf for sorting (handles filenames with [] safely)
    FOUND_GZ=$(find "$SEARCH_DIR" -name "*.pt.trace.json.gz" -type f \
        -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    FOUND_JSON=$(find "$SEARCH_DIR" -name "*.pt.trace.json" -type f \
        -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

    if [ -n "$FOUND_GZ" ]; then
        echo "Torch profiler trace (gzip): $FOUND_GZ"
        python3 << PYEOF
import gzip, json
with gzip.open('$FOUND_GZ', 'rt') as f:
    trace = json.load(f)
with open('$RESULT_DIR/baseline_trace.json', 'w') as f:
    json.dump(trace, f)
PYEOF
        TRACE_FOUND="pytorch_json_gz"
        break
    elif [ -n "$FOUND_JSON" ]; then
        echo "Torch profiler trace (json): $FOUND_JSON"
        cp "$FOUND_JSON" "$RESULT_DIR/baseline_trace.json"
        TRACE_FOUND="pytorch_json"
        break
    fi
done

# Fallback: discover_trace() with marker file and all known dirs
if [ -z "$TRACE_FOUND" ]; then
    echo "No trace in known dirs, falling back to discover_trace()..."
    MARKER="${TRIAL_START_MARKER:-$RAW_LOG}"
    HINT_DIR="${ACTUAL_TB_DIR:-$TB_TRACE_DIR}"
    TRACE_INFO=$(discover_trace "$MARKER" "$RESULT_DIR/baseline_trace.json" "$HINT_DIR" 2>&1)
    TRACE_FOUND=$(echo "$TRACE_INFO" | grep "^TRACE_FORMAT=" | cut -d= -f2)
fi

if [ -n "$TRACE_FOUND" ]; then
    echo "Trace found (format: $TRACE_FOUND)"
    echo "Trace saved to $RESULT_DIR/baseline_trace.json"
    filter_trace "$RESULT_DIR/baseline_trace.json" "$RESULT_DIR/filtered_trace.json"
else
    echo "WARNING: No trace file found in any format (.pt.trace.json, .rpd, .json)"
    echo "Check: profile=true set in YAML? rpdTracerControl loaded in logs?"
fi

# --- Size check: re-filter if too large for TraceLens ---
echo "[5/7] Validating trace for TraceLens..."

MAX_TRACE_MB=50
TRACELENS_TRACE_GZ=""
if [ -f "$RESULT_DIR/filtered_trace.json" ]; then
    TRACE_SIZE_MB=$(python3 -c "import os; print(f'{os.path.getsize(\"$RESULT_DIR/filtered_trace.json\") / 1024 / 1024:.1f}')")
    echo "Filtered trace size: ${TRACE_SIZE_MB}MB (limit: ${MAX_TRACE_MB}MB)"

    if python3 -c "exit(0 if float('$TRACE_SIZE_MB') > $MAX_TRACE_MB else 1)"; then
        echo "Trace exceeds ${MAX_TRACE_MB}MB — re-filtering with compact category set..."
        python3 -c "
import json, os

src = '$RESULT_DIR/filtered_trace.json'
dst = '$RESULT_DIR/filtered_trace_compact.json'
compact_cats = {'kernel', 'gpu_memcpy', 'user_annotation'}

with open(src) as f:
    trace = json.load(f)
orig = len(trace['traceEvents'])
trace['traceEvents'] = [e for e in trace['traceEvents'] if e.get('cat', '') in compact_cats]
filt = len(trace['traceEvents'])
with open(dst, 'w') as f:
    json.dump(trace, f)
size_mb = os.path.getsize(dst) / 1024 / 1024
print(f'Compact filter: {orig} -> {filt} events ({size_mb:.1f}MB)')
"
        TRACELENS_TRACE="$RESULT_DIR/filtered_trace_compact.json"
    else
        TRACELENS_TRACE="$RESULT_DIR/filtered_trace.json"
    fi

    gzip -kf "$TRACELENS_TRACE"
    TRACELENS_TRACE_GZ="${TRACELENS_TRACE}.gz"
    echo "Gzipped trace: $TRACELENS_TRACE_GZ ($(du -h "$TRACELENS_TRACE_GZ" | cut -f1))"
else
    echo "WARNING: No filtered trace to validate"
fi

# --- Validate trace contents ---
echo "[6/7] Verifying trace contents..."

if [ -n "$TRACELENS_TRACE_GZ" ]; then
    python3 -c "
import json, gzip

gz_path = '$TRACELENS_TRACE_GZ'
with gzip.open(gz_path, 'rt') as f:
    trace = json.load(f)

kernel_count = sum(1 for e in trace.get('traceEvents', []) if e.get('cat') == 'kernel')
if kernel_count == 0:
    print('WARNING: Zero kernel events — profiling may have missed GPU activity')
else:
    print(f'Trace validation PASSED — {kernel_count} kernel events')
"
fi

# --- Parse trace: operator summary, categories, GEAK candidates ---
echo "[7/7] Parsing trace (operator summary, categories, GEAK candidates)..."

if [ -f "$RESULT_DIR/baseline_trace.json" ]; then
    python3 "$SKILL_ROOT/scripts/parse_trace.py" \
        --trace-path "$RESULT_DIR/baseline_trace.json" \
        --result-dir "$RESULT_DIR"
else
    echo "WARNING: No baseline_trace.json — skipping parse_trace.py"
fi

echo ""
echo "============================================================"
echo "Profile complete. Results: $RESULT_DIR"
if [ -n "${TRACELENS_TRACE_GZ:-}" ]; then
    echo "TraceLens trace: $TRACELENS_TRACE_GZ"
fi
if [ -f "$RESULT_DIR/categories.json" ]; then
    echo "Categories: $(cat "$RESULT_DIR/categories.json")"
fi
echo "============================================================"
