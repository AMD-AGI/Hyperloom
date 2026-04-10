#!/usr/bin/env bash
# =============================================================================
# mlperf-optimization/scripts/common.sh
#
# Shared helpers for MLPerf training optimization:
#   - Trial tier management (quick / convergence / full)
#   - Log filtering via trial_monitor.py
#   - YAML quiet config via apply_quiet_config.sh
#   - Metric extraction from MLLOG
#   - Process management
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

: "${MLPERF_DIR:=/root/Hyperloom-plus-mlperf/training_optimization/mlperf}"
: "${CONFIG_SH:=$MLPERF_DIR/config_MI355X_1x8x1_fp8.sh}"

source "$SCRIPT_DIR/apply_quiet_config.sh"

_CURRENT_PORT="${MASTER_PORT:-29501}"

# ---------------------------------------------------------------------------
# Port management
# ---------------------------------------------------------------------------
next_port() {
    _CURRENT_PORT=$((_CURRENT_PORT + 1))
    echo "$_CURRENT_PORT"
}

# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------
kill_training() {
    pkill -9 -f "train.py" 2>/dev/null || true
    pkill -9 -f "torchrun" 2>/dev/null || true
    pkill -9 -f "torch.distributed" 2>/dev/null || true
    sleep "${KILL_WAIT_S:-5}"
}

# ---------------------------------------------------------------------------
# Trial tier defaults
# ---------------------------------------------------------------------------
trial_tier_defaults() {
    local tier="${1:?tier required (1, 2, or 3)}"
    case "$tier" in
        1)
            echo "train_iters=100 eval_interval=10000 log_freq=1 timeout_s=3600 quiet=1"
            ;;
        2)
            echo "train_iters=500 eval_interval=50 log_freq=1 timeout_s=7200 quiet=1"
            ;;
        3)
            echo "train_iters= eval_interval= log_freq=32 timeout_s=0 quiet=0"
            ;;
        *)
            echo "ERROR: unknown tier $tier" >&2
            return 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# run_mlperf_trial  — Central trial runner with tier support
#
# Usage:
#   run_mlperf_trial <label> [tier] [train_iters] [extra_env]
#
# Tier 1 (quick):       100 iters, no eval, MLLOG_TRAIN_LOSS_LOG_FREQ=1, 60min timeout
# Tier 2 (convergence): 500 iters, eval enabled, MLLOG_TRAIN_LOSS_LOG_FREQ=1, 120min timeout
# Tier 3 (full):        original config, MLLOG_TRAIN_LOSS_LOG_FREQ=32, no timeout
#
# Prints a TRIAL_RESULT line on stdout for structured parsing.
# Raw log is always preserved at $RESULT_DIR/attempt_<label>_raw.log
# Filtered log at $RESULT_DIR/attempt_<label>.log
# ---------------------------------------------------------------------------
run_mlperf_trial() {
    local label="${1:?label required}"
    local tier="${2:-1}"
    local train_iters_override="${3:-}"
    local extra_env="${4:-}"

    # Resolve tier defaults
    local defaults
    defaults=$(trial_tier_defaults "$tier") || return 1
    local train_iters eval_interval log_freq timeout_s quiet
    eval "$defaults"

    # Override train_iters if caller specified
    if [ -n "$train_iters_override" ]; then
        train_iters="$train_iters_override"
    fi

    local raw_log="${RESULT_DIR:-.}/attempt_${label}_raw.log"
    local filtered_log="${RESULT_DIR:-.}/attempt_${label}.log"

    echo ""
    echo "--- Trial [$label] tier=$tier iters=$train_iters eval_interval=$eval_interval log_freq=$log_freq timeout=${timeout_s}s ---"

    kill_training

    # Apply quiet YAML for tier 1 & 2
    if [ "$quiet" = "1" ] && [ -n "$EXP" ] && [ -f "$EXP" ]; then
        quiet_yaml "$EXP"
    fi

    cd "$MLPERF_DIR"
    source "$CONFIG_SH"

    # Tier-specific overrides
    if [ -n "$train_iters" ]; then
        export PRIMUS_TRAIN_ITERS="$train_iters"
    fi
    if [ -n "$eval_interval" ]; then
        export PRIMUS_EVAL_INTERVAL="$eval_interval"
    fi

    # CRITICAL: override MLLOG_TRAIN_LOSS_LOG_FREQ for short trials
    export MLLOG_TRAIN_LOSS_LOG_FREQ="$log_freq"
    export PYTHONWARNINGS=ignore
    export MASTER_PORT=$(next_port)

    # Apply caller's extra env vars
    if [ -n "$extra_env" ]; then
        for kv in $extra_env; do
            export "$kv"
        done
    fi

    # Recompute derived values after overrides
    if [ -n "$PRIMUS_GLOBAL_BATCH_SIZE" ] && [ -n "$EVAL_SAMPLES_INTERVAL" ]; then
        export PRIMUS_EVAL_INTERVAL=$((EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE))
        if [ -n "$eval_interval" ] && [ "$eval_interval" != "10000" ]; then
            export PRIMUS_EVAL_INTERVAL="$eval_interval"
        fi
    fi

    bash setup_container_symlinks.sh 2>/dev/null

    # Re-source config to pick up derived values, then re-apply our overrides
    source "$CONFIG_SH"
    if [ -n "$train_iters" ]; then
        export PRIMUS_TRAIN_ITERS="$train_iters"
    fi
    if [ -n "$eval_interval" ]; then
        export PRIMUS_EVAL_INTERVAL="$eval_interval"
    fi
    export MLLOG_TRAIN_LOSS_LOG_FREQ="$log_freq"
    export PYTHONWARNINGS=ignore
    if [ -n "$extra_env" ]; then
        for kv in $extra_env; do
            export "$kv"
        done
    fi

    local exit_code=0
    local max_iters_flag=""
    if [ -n "$train_iters" ]; then
        max_iters_flag="--max-iters $train_iters"
    fi

    if [ "$timeout_s" -gt 0 ] 2>/dev/null; then
        timeout "$timeout_s" bash -c 'bash run_and_time.sh' 2>&1 \
            | python3 "$SKILL_ROOT/scripts/trial_monitor.py" \
                --raw-log "$raw_log" $max_iters_flag --label "$label" \
            | tee "$filtered_log"
        exit_code=${PIPESTATUS[0]}
    else
        bash run_and_time.sh 2>&1 \
            | python3 "$SKILL_ROOT/scripts/trial_monitor.py" \
                --raw-log "$raw_log" $max_iters_flag --label "$label" \
            | tee "$filtered_log"
        exit_code=${PIPESTATUS[0]}
    fi

    # Restore YAML
    if [ "$quiet" = "1" ] && [ -n "$EXP" ]; then
        restore_yaml "$EXP"
    fi

    # Handle timeout exit code (124)
    if [ "$exit_code" -eq 124 ]; then
        echo "WARNING: Trial [$label] timed out after ${timeout_s}s" >&2
    fi

    return $exit_code
}

# ---------------------------------------------------------------------------
# Metric extraction — operate on raw log files
# ---------------------------------------------------------------------------

extract_ms_per_iter() {
    local log_file="${1:?log file required}"
    local warmup="${2:-5}"
    local measure="${3:-5}"

    python3 -c "
import json, sys

with open('$log_file') as f:
    lines = f.readlines()

mllog_lines = [l for l in lines if l.startswith(':::MLLOG')]
train_loss_events = []
for line in mllog_lines:
    data = json.loads(line.replace(':::MLLOG ', ''))
    if data['key'] == 'train_loss':
        train_loss_events.append(data)

iter_times_ms = []
for i in range(1, len(train_loss_events)):
    delta = train_loss_events[i]['time_ms'] - train_loss_events[i-1]['time_ms']
    iter_times_ms.append(delta)

if not iter_times_ms:
    print('ERROR: no iteration times found', file=sys.stderr)
    sys.exit(1)

warmup = $warmup
measure = $measure
if len(iter_times_ms) >= warmup + measure:
    measured = iter_times_ms[warmup:warmup + measure]
elif len(iter_times_ms) > warmup:
    measured = iter_times_ms[warmup:]
else:
    measured = iter_times_ms

avg = sum(measured) / len(measured)
print(f'{avg:.1f}')
"
}

extract_mllog_field() {
    local log_file="${1:?log file required}"
    local field="${2:?field name required}"

    python3 -c "
import json
with open('$log_file') as f:
    for line in f:
        if not line.startswith(':::MLLOG'):
            continue
        data = json.loads(line.replace(':::MLLOG ', ''))
        if data['key'] == '$field':
            print(data['value'])
            break
"
}

verify_gbs() {
    local log_file="${1:?log file required}"
    local expected="${2:?expected GBS required}"

    python3 -c "
import json, sys

gbs = None
with open('$log_file') as f:
    for line in f:
        if not line.startswith(':::MLLOG'):
            continue
        data = json.loads(line.replace(':::MLLOG ', ''))
        if data['key'] == 'global_batch_size':
            gbs = data['value']
            break

if gbs is None:
    print('WARNING: could not find GBS in MLLOG', file=sys.stderr)
    sys.exit(0)

if gbs != $expected:
    print(f'ERROR: GBS mismatch: found {gbs}, expected $expected', file=sys.stderr)
    sys.exit(1)

print(f'GBS verified: {gbs}')
"
}

extract_losses() {
    local log_file="${1:?log file required}"

    python3 -c "
import json

with open('$log_file') as f:
    for line in f:
        if not line.startswith(':::MLLOG'):
            continue
        data = json.loads(line.replace(':::MLLOG ', ''))
        if data['key'] == 'train_loss':
            sc = data['metadata'].get('samples_count', '?')
            lr = data['metadata'].get('lr', '?')
            print(f'samples={sc}\tloss={data[\"value\"]:.6f}\tlr={lr}')
"
}

extract_time_to_train() {
    local log_file="${1:?log file required}"

    python3 -c "
import json

events = []
with open('$log_file') as f:
    for line in f:
        if not line.startswith(':::MLLOG'):
            continue
        events.append(json.loads(line.replace(':::MLLOG ', '')))

run_start = next((e['time_ms'] for e in events if e['key'] == 'run_start'), None)
run_stop = next((e for e in events if e['key'] == 'run_stop'), None)

if run_start and run_stop:
    seconds = (run_stop['time_ms'] - run_start) / 1000
    status = run_stop['metadata'].get('status', 'unknown')
    print(f'{seconds:.1f}\t{status}')
else:
    print('ERROR: run_start/run_stop not found')
"
}

# ---------------------------------------------------------------------------
# Helper: parse a TRIAL_RESULT line into shell variables
# Usage: eval "$(parse_trial_result "$line")"
# ---------------------------------------------------------------------------
parse_trial_result() {
    local line="${1:?TRIAL_RESULT line required}"
    python3 -c "
line = '''$line'''
if not line.startswith('TRIAL_RESULT'):
    print('# not a TRIAL_RESULT line')
    exit(0)
parts = line.split()
for part in parts[1:]:
    key, _, val = part.partition('=')
    print(f'TRIAL_{key.upper()}=\"{val}\"')
"
}

# ---------------------------------------------------------------------------
# Helper: compute gain percentage
# ---------------------------------------------------------------------------
compute_gain_pct() {
    local baseline="${1:?baseline value required}"
    local current="${2:?current value required}"
    python3 -c "
b = float('$baseline')
c = float('$current')
if b > 0:
    print(f'{(b - c) / b * 100:.2f}')
else:
    print('0.00')
"
}

# ---------------------------------------------------------------------------
# Helper: detect NaN/Inf in a log file
# ---------------------------------------------------------------------------
detect_nan_in_log() {
    local log_file="${1:?log file required}"
    python3 -c "
import json, math
with open('$log_file') as f:
    for line in f:
        if not line.startswith(':::MLLOG'):
            continue
        data = json.loads(line.replace(':::MLLOG ', ''))
        if data['key'] == 'train_loss':
            v = data.get('value')
            if v is not None and (math.isnan(float(v)) or math.isinf(float(v))):
                print('NaN_DETECTED')
                exit(0)
print('OK')
"
}

# ---------------------------------------------------------------------------
# Chrome trace filtering (for profiling action)
# ---------------------------------------------------------------------------
filter_trace() {
    local src="${1:?source trace path required}"
    local dst="${2:?destination trace path required}"
    python3 -c "
import json, os, gzip

src = '$src'
dst = '$dst'

opener = gzip.open if src.endswith('.gz') else open
with opener(src, 'rt') as f:
    trace = json.load(f)

keep = {'kernel', 'gpu_memcpy', 'gpu_memset', 'cpu_op', 'cuda_runtime',
        'ac2g', 'user_annotation', 'gpu_user_annotation'}
orig = len(trace['traceEvents'])
trace['traceEvents'] = [e for e in trace['traceEvents'] if e.get('cat', '') in keep]
filt = len(trace['traceEvents'])

writer = gzip.open if dst.endswith('.gz') else open
with writer(dst, 'wt') as f:
    json.dump(trace, f)

size_mb = os.path.getsize(dst) / 1024 / 1024
print(f'Filtered: {orig} -> {filt} events ({size_mb:.1f}MB)')
" 2>&1 || return 1
}
