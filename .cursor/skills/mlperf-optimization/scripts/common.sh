#!/usr/bin/env bash
# =============================================================================
# mlperf-optimization/scripts/common.sh
#
# Shared helpers for MLPerf training optimization:
#   - Trial tier management (quick / convergence / long convergence / full)
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
    local tier="${1:?tier required (1, 2, 2L, or 3)}"
    case "$tier" in
        1)
            echo "train_iters=100 eval_interval=10000 log_freq=1 timeout_s=3600 quiet=1"
            ;;
        2)
            echo "train_iters=500 eval_interval=50 log_freq=1 timeout_s=7200 quiet=1"
            ;;
        2L)
            echo "train_iters=2500 eval_interval=50 log_freq=1 timeout_s=14400 quiet=1"
            ;;
        2.5)
            echo "train_iters=1500 eval_interval=50 log_freq=1 timeout_s=10800 quiet=1"
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
# Tier 1   (quick):                100 iters, no eval, MLLOG_TRAIN_LOSS_LOG_FREQ=1, 60min timeout
# Tier 2   (convergence):         500 iters, eval enabled, MLLOG_TRAIN_LOSS_LOG_FREQ=1, 120min timeout
# Tier 2.5 (convergence project): 1500 iters, eval enabled, MLLOG_TRAIN_LOSS_LOG_FREQ=1, 3hr timeout
# Tier 2L  (long convergence):    2500 iters, eval enabled, MLLOG_TRAIN_LOSS_LOG_FREQ=1, 4hr timeout
# Tier 3   (full):                original config, MLLOG_TRAIN_LOSS_LOG_FREQ=32, no timeout
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

    # Timestamp marker for discover_trace(): created BEFORE training starts so
    # any trace file written during the run will have mtime > marker mtime.
    # This avoids the -newer race condition where the raw log finishes writing
    # after the trace file, causing find -newer to skip the trace.
    export TRIAL_START_MARKER="${RESULT_DIR:-.}/_trial_start_${label}"
    touch "$TRIAL_START_MARKER"

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
# project_ttt — Extrapolate TTT from a Tier 2L raw log
#
# Fits log(eval_loss - target) vs log(samples) to a linear model (power-law
# decay), then projects how many samples are needed to reach target. Combined
# with samples/sec (derived from ms/iter and GBS), estimates wall-clock TTT.
#
# Usage:
#   project_ttt <raw_log> <gbs>
#
# Output: "<projected_ttt_seconds>\t<projected_samples>\t<samples_per_sec>\t<r_squared>"
# ---------------------------------------------------------------------------
project_ttt() {
    local log_file="${1:?raw log file required}"
    local gbs="${2:?GBS required}"

    python3 -c "
import json, sys, math

TARGET = 3.34
log_file = '$log_file'
gbs = int('$gbs')

train_loss_events = []
eval_events = []

with open(log_file) as f:
    for line in f:
        if not line.startswith(':::MLLOG'):
            continue
        data = json.loads(line.replace(':::MLLOG ', ''))
        if data['key'] == 'train_loss':
            train_loss_events.append(data)
        elif data['key'] == 'eval_accuracy':
            sc = data.get('metadata', {}).get('samples_count')
            val = data.get('value')
            if sc is not None and val is not None:
                eval_events.append((int(sc), float(val)))

# Compute ms/iter from train_loss timestamps (skip first 5 as warmup)
iter_deltas = []
for i in range(1, len(train_loss_events)):
    delta = train_loss_events[i]['time_ms'] - train_loss_events[i-1]['time_ms']
    iter_deltas.append(delta)

warmup = min(5, len(iter_deltas) // 2)
if len(iter_deltas) > warmup:
    measured = iter_deltas[warmup:]
else:
    measured = iter_deltas

if not measured:
    print('ERROR: no iteration timing data', file=sys.stderr)
    sys.exit(1)

ms_per_iter = sum(measured) / len(measured)
samples_per_sec = gbs / (ms_per_iter / 1000.0)

# Filter eval events: only keep those above target (we can't log negative values)
valid_evals = [(s, v) for s, v in eval_events if v > TARGET]

if len(valid_evals) < 2:
    # Not enough eval points to fit — use last eval_loss to make a rough estimate
    if eval_events:
        last_s, last_v = eval_events[-1]
        if last_v <= TARGET:
            # Already converged within the trial
            print(f'{last_s / samples_per_sec:.1f}\t{last_s}\t{samples_per_sec:.1f}\t1.00')
        else:
            # Crude linear extrapolation from two most recent points
            if len(eval_events) >= 2:
                s1, v1 = eval_events[-2]
                s2, v2 = eval_events[-1]
                if v1 != v2:
                    slope = (v2 - v1) / (s2 - s1)
                    if slope < 0:
                        projected_samples = s2 + (TARGET - v2) / slope
                        projected_ttt = projected_samples / samples_per_sec
                        print(f'{projected_ttt:.1f}\t{projected_samples:.0f}\t{samples_per_sec:.1f}\t0.50')
                        sys.exit(0)
            print(f'-1\t-1\t{samples_per_sec:.1f}\t0.00')
    else:
        print(f'-1\t-1\t{samples_per_sec:.1f}\t0.00')
    sys.exit(0)

# Power-law regression: log(eval_loss - target) vs log(samples)
xs = [math.log(s) for s, v in valid_evals]
ys = [math.log(v - TARGET) for s, v in valid_evals]

n = len(xs)
sum_x = sum(xs)
sum_y = sum(ys)
sum_xy = sum(x * y for x, y in zip(xs, ys))
sum_x2 = sum(x * x for x in xs)

denom = n * sum_x2 - sum_x * sum_x
if abs(denom) < 1e-12:
    print(f'-1\t-1\t{samples_per_sec:.1f}\t0.00')
    sys.exit(0)

slope = (n * sum_xy - sum_x * sum_y) / denom
intercept = (sum_y - slope * sum_x) / n

# slope should be negative (loss decreasing). If positive, extrapolation is invalid.
if slope >= 0:
    print(f'-1\t-1\t{samples_per_sec:.1f}\t0.00', file=sys.stdout)
    sys.exit(0)

# Project to eval_loss = TARGET + epsilon via power law:
#   log(epsilon) = slope * log(s) + intercept  →  s = exp((log(epsilon) - intercept) / slope)
epsilon = 0.01
target_log = math.log(epsilon)
projected_samples = math.exp((target_log - intercept) / slope)

if projected_samples < 0:
    print(f'-1\t-1\t{samples_per_sec:.1f}\t0.00')
    sys.exit(0)

projected_ttt = projected_samples / samples_per_sec

# R-squared
y_mean = sum_y / n
ss_tot = sum((y - y_mean) ** 2 for y in ys)
ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

print(f'{projected_ttt:.1f}\t{projected_samples:.0f}\t{samples_per_sec:.1f}\t{r_squared:.2f}')
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
# Loss efficiency: (first_eval_loss - last_eval_loss) / wall_time_seconds
# Higher = faster convergence per unit time. Used as Layer 2 quick filter
# for convergence-affecting actions.
# ---------------------------------------------------------------------------
compute_loss_efficiency() {
    local log_file="${1:?log file required}"
    python3 -c "
import json, sys

log_file = '$log_file'
eval_events = []
first_ts = None

with open(log_file) as f:
    for line in f:
        if not line.startswith(':::MLLOG'):
            continue
        data = json.loads(line.replace(':::MLLOG ', ''))
        if first_ts is None:
            first_ts = data.get('time_ms', 0)
        if data['key'] == 'eval_accuracy':
            sc = data.get('metadata', {}).get('samples_count')
            val = data.get('value')
            ts = data.get('time_ms', 0)
            if sc is not None and val is not None:
                eval_events.append((int(sc), float(val), float(ts)))

if len(eval_events) < 2 or first_ts is None:
    print('0.0')
    sys.exit(0)

first_eval = eval_events[0]
last_eval = eval_events[-1]
delta_loss = first_eval[1] - last_eval[1]
delta_time_s = (last_eval[2] - first_ts) / 1000.0

if delta_time_s <= 0:
    print('0.0')
    sys.exit(0)

efficiency = delta_loss / delta_time_s
print(f'{efficiency:.6f}')
"
}

# ---------------------------------------------------------------------------
# TTT gain: (baseline_ttt - candidate_ttt) / baseline_ttt * 100
# Positive = candidate is faster (better). Used as Layer 3 final decision
# for convergence-affecting actions.
# ---------------------------------------------------------------------------
compute_ttt_gain_pct() {
    local baseline_ttt="${1:?baseline projected TTT required}"
    local candidate_ttt="${2:?candidate projected TTT required}"
    python3 -c "
b = float('$baseline_ttt')
c = float('$candidate_ttt')
if b > 0 and c > 0:
    print(f'{(b - c) / b * 100:.2f}')
else:
    print('0.00')
"
}

# ---------------------------------------------------------------------------
# Action classification: throughput-only vs convergence-affecting
# Determines which evaluation layers apply in the DFS loop.
# ---------------------------------------------------------------------------
classify_action() {
    local action="${1:?action name required}"
    case "$action" in
        config-selection|convergence-speed|fp8-recipe-tuning|hyperparams)
            echo "convergence-affecting"
            ;;
        *)
            echo "throughput-only"
            ;;
    esac
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
# Eval trajectory extraction — for convergence-speed action
# ---------------------------------------------------------------------------
extract_eval_trajectory() {
    local log_file="${1:?log file required}"

    python3 -c "
import json

with open('$log_file') as f:
    for line in f:
        if not line.startswith(':::MLLOG'):
            continue
        data = json.loads(line.replace(':::MLLOG ', ''))
        if data['key'] == 'eval_accuracy':
            sc = data.get('metadata', {}).get('samples_count', '?')
            val = data.get('value')
            time_ms = data.get('time_ms', 0)
            if val is not None:
                print(f'{sc}\t{val:.6f}\t{time_ms}')
"
}

# ---------------------------------------------------------------------------
# Eval overhead computation — wall time spent in eval phases
# ---------------------------------------------------------------------------
compute_eval_overhead() {
    local log_file="${1:?log file required}"

    python3 -c "
import json

events = []
with open('$log_file') as f:
    for line in f:
        if not line.startswith(':::MLLOG'):
            continue
        events.append(json.loads(line.replace(':::MLLOG ', '')))

eval_starts = [e['time_ms'] for e in events if e['key'] == 'eval_start']
eval_stops  = [e['time_ms'] for e in events if e['key'] == 'eval_stop']

total_eval_ms = 0
for start, stop in zip(eval_starts, eval_stops):
    total_eval_ms += stop - start

num_evals = len(eval_starts)
avg_eval_ms = total_eval_ms / num_evals if num_evals > 0 else 0

print(f'{total_eval_ms / 1000:.1f}\t{num_evals}\t{avg_eval_ms / 1000:.1f}')
"
}

# ---------------------------------------------------------------------------
# Trace discovery (ROCm RPD + standard PyTorch JSON)
#
# On ROCm, PyTorch profiler uses the RPD backend (rpdTracerControl) which
# writes .rpd SQLite databases — NOT .pt.trace.json files. This function
# searches for both formats and converts RPD→JSON when needed.
#
# Search priority:
#   0) Known tensorboard_dir (tb_trace_dir arg) — no -newer, most reliable
#   1) -newer search for *.pt.trace.json
#   2) -newer search for *trace*.json / *chrome*.json / *profiler*.json
#   3) -newer search for *.rpd (+ convert)
#   4) -newer search for *_results.json (rocprofv3)
#   5) Relaxed fallback: most recent *.pt.trace.json within last 30 min
# ---------------------------------------------------------------------------
discover_trace() {
    local after_file="${1:?reference file for -newer required}"
    local output_json="${2:?output JSON path required}"
    local tb_trace_dir="${3:-}"

    local found

    # Helper: decompress .gz trace to output_json (handles filenames with [] safely via heredoc)
    _decompress_gz_trace() {
        local gz_path="$1"
        local out_path="$2"
        python3 << PYEOF
import gzip, json, sys
with gzip.open(sys.argv[1], 'rt') as f:
    trace = json.load(f)
with open(sys.argv[2], 'w') as f:
    json.dump(trace, f)
PYEOF
    }

    # Helper: find most recent file by mtime (handles filenames with [] safely)
    _find_newest() {
        find "$@" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-
    }

    # 0) Priority: search known tensorboard_dir without -newer
    if [ -n "$tb_trace_dir" ] && [ -d "$tb_trace_dir" ]; then
        found=$(_find_newest "$tb_trace_dir" -name "*.pt.trace.json.gz" -type f)
        if [ -n "$found" ]; then
            python3 -c "
import gzip, json, sys
with gzip.open(sys.argv[1], 'rt') as f:
    trace = json.load(f)
with open(sys.argv[2], 'w') as f:
    json.dump(trace, f)
" "$found" "$output_json"
            echo "TRACE_FORMAT=pytorch_json_gz"
            echo "TRACE_SOURCE=$found"
            return 0
        fi

        found=$(_find_newest "$tb_trace_dir" -name "*.pt.trace.json" -type f)
        if [ -n "$found" ]; then
            cp "$found" "$output_json"
            echo "TRACE_FORMAT=pytorch_json"
            echo "TRACE_SOURCE=$found"
            return 0
        fi
    fi

    # 1a) -newer search for gzipped PyTorch Chrome trace
    found=$(_find_newest /workspace /tmp /root -name "*.pt.trace.json.gz" \
        -newer "$after_file" -type f)
    if [ -n "$found" ]; then
        echo "Found gzipped trace (newer): $found" >&2
        python3 -c "
import gzip, json, sys
with gzip.open(sys.argv[1], 'rt') as f:
    trace = json.load(f)
with open(sys.argv[2], 'w') as f:
    json.dump(trace, f)
" "$found" "$output_json"
        echo "TRACE_FORMAT=pytorch_json_gz"
        echo "TRACE_SOURCE=$found"
        return 0
    fi

    # 1b) -newer search for standard PyTorch Chrome trace JSON
    found=$(_find_newest /workspace /tmp /root -name "*.pt.trace.json" \
        -newer "$after_file" -type f)
    if [ -n "$found" ]; then
        cp "$found" "$output_json"
        echo "TRACE_FORMAT=pytorch_json"
        echo "TRACE_SOURCE=$found"
        return 0
    fi

    # 2) -newer search for any Chrome-format JSON
    found=$(_find_newest /workspace /tmp /root \
        \( -name "*trace*.json" -o -name "*chrome*.json" -o -name "*profiler*.json" \) \
        -newer "$after_file" -size +10k -type f)
    if [ -n "$found" ]; then
        cp "$found" "$output_json"
        echo "TRACE_FORMAT=chrome_json"
        echo "TRACE_SOURCE=$found"
        return 0
    fi

    # 3) -newer search for RPD files (.rpd = SQLite3 from ROCm RPD backend)
    found=$(_find_newest /workspace /tmp /root -name "*.rpd" \
        -newer "$after_file" -type f)
    if [ -n "$found" ]; then
        echo "Found RPD trace: $found — converting to Chrome JSON..." >&2
        convert_rpd_to_json "$found" "$output_json"
        if [ $? -eq 0 ] && [ -s "$output_json" ]; then
            echo "TRACE_FORMAT=rpd_converted"
            echo "TRACE_SOURCE=$found"
            return 0
        fi
        echo "WARNING: RPD conversion failed" >&2
    fi

    # 4) -newer search for rocprofv3 results JSON
    found=$(_find_newest /workspace /tmp /root -name "*_results.json" \
        -newer "$after_file" -type f)
    if [ -n "$found" ]; then
        cp "$found" "$output_json"
        echo "TRACE_FORMAT=rocprofv3"
        echo "TRACE_SOURCE=$found"
        return 0
    fi

    # 5a) Relaxed fallback: most recent *.pt.trace.json.gz from the last 30 min
    found=$(_find_newest /workspace /tmp /root -name "*.pt.trace.json.gz" \
        -mmin -30 -type f)
    if [ -n "$found" ]; then
        echo "Relaxed fallback: found recent gzipped trace $found" >&2
        python3 -c "
import gzip, json, sys
with gzip.open(sys.argv[1], 'rt') as f:
    trace = json.load(f)
with open(sys.argv[2], 'w') as f:
    json.dump(trace, f)
" "$found" "$output_json"
        echo "TRACE_FORMAT=pytorch_json_gz_relaxed"
        echo "TRACE_SOURCE=$found"
        return 0
    fi

    # 5b) Relaxed fallback: most recent *.pt.trace.json from the last 30 min
    found=$(_find_newest /workspace /tmp /root -name "*.pt.trace.json" \
        -mmin -30 -type f)
    if [ -n "$found" ]; then
        echo "Relaxed fallback: found recent trace $found" >&2
        cp "$found" "$output_json"
        echo "TRACE_FORMAT=pytorch_json_relaxed"
        echo "TRACE_SOURCE=$found"
        return 0
    fi

    echo "WARNING: No trace file found in any format" >&2
    return 1
}

# ---------------------------------------------------------------------------
# RPD → Chrome JSON conversion
#
# RPD files are SQLite3 databases written by ROCm's rpdTracerControl.
# Convert to Chrome trace JSON using rpd2tracing.py (if available) or
# a fallback that queries the SQLite schema directly.
# ---------------------------------------------------------------------------
convert_rpd_to_json() {
    local rpd_path="${1:?rpd file path required}"
    local json_path="${2:?output json path required}"

    python3 -c "
import sys, json, os

rpd_path = '$rpd_path'
json_path = '$json_path'

# Method 1: Try rpd2tracing.py from rocmProfileData package
try:
    from rpd2tracing import rpd2tracing
    rpd2tracing(rpd_path, json_path)
    size_mb = os.path.getsize(json_path) / 1024 / 1024
    print(f'RPD→JSON via rpd2tracing: {size_mb:.1f}MB', file=sys.stderr)
    sys.exit(0)
except ImportError:
    pass

# Method 2: Direct SQLite query (fallback)
import sqlite3

conn = sqlite3.connect(rpd_path)
cursor = conn.cursor()

# Check which tables exist
tables = {row[0] for row in cursor.execute(
    \"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()}

trace_events = []

if 'rocpd_api' in tables and 'rocpd_op' in tables:
    # Standard RPD schema: rocpd_api (CPU) + rocpd_op (GPU)
    for row in cursor.execute('''
        SELECT api.Name, op.KernelName, op.BeginNs, op.EndNs, op.gpuId,
               op.queueId, op.pid, op.tid
        FROM rocpd_op op
        LEFT JOIN rocpd_api api ON op.api_id = api.id
        ORDER BY op.BeginNs
        LIMIT 200000
    '''):
        api_name, kernel_name, begin_ns, end_ns, gpu_id, queue_id, pid, tid = row
        name = kernel_name or api_name or 'unknown'
        dur_us = (end_ns - begin_ns) / 1000 if begin_ns and end_ns else 0
        ts_us = begin_ns / 1000 if begin_ns else 0
        trace_events.append({
            'name': name, 'cat': 'kernel', 'ph': 'X',
            'ts': ts_us, 'dur': dur_us,
            'pid': pid or gpu_id or 0, 'tid': tid or queue_id or 0,
        })

elif 'api' in tables and 'op' in tables:
    # Alternate schema
    for row in cursor.execute('''
        SELECT Name, KernelName, BeginNs, EndNs, gpuId, pid, tid
        FROM op LEFT JOIN api ON op.api_id = api.id
        ORDER BY BeginNs LIMIT 200000
    '''):
        api_name, kernel_name, begin_ns, end_ns, gpu_id, pid, tid = row
        name = kernel_name or api_name or 'unknown'
        dur_us = (end_ns - begin_ns) / 1000 if begin_ns and end_ns else 0
        ts_us = begin_ns / 1000 if begin_ns else 0
        trace_events.append({
            'name': name, 'cat': 'kernel', 'ph': 'X',
            'ts': ts_us, 'dur': dur_us,
            'pid': pid or gpu_id or 0, 'tid': tid or 0,
        })

conn.close()

if not trace_events:
    print(f'WARNING: RPD file has no ops (tables: {tables})', file=sys.stderr)
    sys.exit(1)

with open(json_path, 'w') as f:
    json.dump({'traceEvents': trace_events}, f)

size_mb = os.path.getsize(json_path) / 1024 / 1024
print(f'RPD→JSON via SQLite fallback: {len(trace_events)} events, {size_mb:.1f}MB',
      file=sys.stderr)
" 2>&1
    return ${PIPESTATUS[0]}
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
