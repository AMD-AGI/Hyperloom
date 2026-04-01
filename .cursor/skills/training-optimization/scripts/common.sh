#!/usr/bin/env bash
# =============================================================================
# training-optimization/scripts/common.sh
#
# Shared helpers for training optimization scripts:
#   - kill_training       - tear down torchrun processes
#   - extract_ms_per_iter - parse ms/iter from training log
#   - verify_gbs          - verify global batch size matches baseline
#   - filter_trace        - shrink Chrome trace JSON for TraceLens
#   - next_port           - get next available master port
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Global port counter (starts from MASTER_PORT or 29500)
_CURRENT_PORT="${MASTER_PORT:-29500}"

next_port() {
    _CURRENT_PORT=$((_CURRENT_PORT + 1))
    echo "$_CURRENT_PORT"
}

kill_training() {
    pkill -9 -f "primus/cli/main.py" 2>/dev/null || true
    pkill -9 -f "megatron" 2>/dev/null || true
    # Kill any orphaned torchrun workers
    pkill -9 -f "torch.distributed" 2>/dev/null || true
    sleep "${KILL_WAIT_S:-5}"
}

# Extract ms/iter from training log (iterations 6–10 average).
# Args: log_file [warmup_iters] [measure_iters]
extract_ms_per_iter() {
    local log_file="${1:?log file required}"
    local warmup="${2:-5}"
    local measure="${3:-5}"

    python3 -c "
import re, sys

with open('$log_file') as f:
    lines = f.readlines()

iter_times = []
for line in lines:
    m = re.search(r'elapsed time per iteration \(ms\):\s*([\d.]+)', line)
    if not m:
        m = re.search(r'time \(ms\):\s*([\d.]+)', line)
    if not m:
        m = re.search(r'iter_time.*?([\d]+\.?\d*)\s*ms', line, re.IGNORECASE)
    if m:
        iter_times.append(float(m.group(1)))

if not iter_times:
    print('ERROR: no iteration times found', file=sys.stderr)
    sys.exit(1)

warmup = $warmup
measure = $measure
if len(iter_times) >= warmup + measure:
    measured = iter_times[warmup:warmup + measure]
elif len(iter_times) > warmup:
    measured = iter_times[warmup:]
else:
    measured = iter_times

avg = sum(measured) / len(measured)
print(f'{avg:.1f}')
"
}

# Verify GBS matches expected value.
# Args: log_file expected_gbs
verify_gbs() {
    local log_file="${1:?log file required}"
    local expected="${2:?expected GBS required}"

    python3 -c "
import re, sys

with open('$log_file') as f:
    content = f.read()

# Try multiple patterns for GBS
patterns = [
    r'global.batch.size[:\s=]+(\d+)',
    r'GBS[:\s=]+(\d+)',
    r'global_batch_size[:\s=]+(\d+)',
]

gbs = None
for pat in patterns:
    m = re.search(pat, content, re.IGNORECASE)
    if m:
        gbs = int(m.group(1))
        break

if gbs is None:
    print('WARNING: could not find GBS in log', file=sys.stderr)
    sys.exit(0)

if gbs != $expected:
    print(f'ERROR: GBS mismatch: found {gbs}, expected $expected', file=sys.stderr)
    sys.exit(1)

print(f'GBS verified: {gbs}')
"
}

# Filter a Chrome trace JSON for TraceLens (drop noisy python_function events).
# Args: source_trace_path destination_trace_path
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

# Run a training attempt and extract ms/iter.
# Args: label overrides...
# Env: CONFIG_YAML, NUM_GPUS, PRIMUS_ROOT, RESULT_DIR, KEPT_OVERRIDES, BASELINE_GBS
run_attempt() {
    local label="${1:?label required}"
    shift
    local extra_overrides="$*"
    local log_file="$RESULT_DIR/attempt_${label}.log"
    local port
    port=$(next_port)

    kill_training

    cd "$PRIMUS_ROOT"
    torchrun --nproc_per_node="$NUM_GPUS" --master_port="$port" \
        -m primus.cli.main train pretrain \
        --config "$CONFIG_YAML" \
        $KEPT_OVERRIDES \
        $extra_overrides \
        profile=false use_pytorch_profiler=false \
        2>&1 | tee "$log_file"

    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo "CRASH"
        return 1
    fi

    # Verify GBS
    if [ -n "${BASELINE_GBS:-}" ]; then
        verify_gbs "$log_file" "$BASELINE_GBS" || { echo "INVALID_GBS"; return 2; }
    fi

    # Extract ms/iter
    extract_ms_per_iter "$log_file"
}
