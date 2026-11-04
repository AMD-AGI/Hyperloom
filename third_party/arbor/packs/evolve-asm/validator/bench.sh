#!/usr/bin/env bash
# Bench validator for evolve-asm pack.
# Runs the kernel bench driver inside the container, collects N timing samples
# via HIP events (device time, not wall time — KernelForge protocol), gates on
# the WEAKEST run, and writes bench_output.json.
#
# Required env vars (injected by arbor harness):
#   BENCH_OUTPUT      host path to write bench_output.json
#   CONTAINER         docker container name
#   KERNEL_BENCH_CMD  full bench command to run inside container
#   N_BENCH           number of timing iterations (default 500)
# Optional:
#   REBUILD_GUARD_CO  path to .co inside container
#   REBUILD_GUARD_S   path to .s inside container (if set, guard is enforced)

set -euo pipefail

: "${BENCH_OUTPUT:?Arbor must inject BENCH_OUTPUT}"
: "${CONTAINER:?Arbor must inject CONTAINER}"
: "${KERNEL_BENCH_CMD:?Arbor must inject KERNEL_BENCH_CMD}"

N_BENCH="${N_BENCH:-500}"

mkdir -p "$(dirname "$BENCH_OUTPUT")"

# ── Rebuild guard ─────────────────────────────────────────────────────────────
if [ -n "${REBUILD_GUARD_CO:-}" ] && [ -n "${REBUILD_GUARD_S:-}" ]; then
    STALE=$(docker exec "$CONTAINER" python3 -c "
import os
co = os.path.getmtime('$REBUILD_GUARD_CO') if os.path.exists('$REBUILD_GUARD_CO') else 0
s  = os.path.getmtime('$REBUILD_GUARD_S')  if os.path.exists('$REBUILD_GUARD_S')  else 0
print('stale' if s > 0 and co < s else 'ok')
" 2>/dev/null || echo "unknown")
    if [ "$STALE" = "stale" ]; then
        python3 -c "
import json
json.dump({'status': 'failed', 'error': 'REBUILD GUARD: .co is older than .s — run patch_co.py first'},
          open('$BENCH_OUTPUT', 'w'), indent=2)
"
        echo "[bench] REBUILD GUARD FAILED: .co older than .s" >&2
        exit 0
    fi
fi

# ── Run bench inside container ────────────────────────────────────────────────
RAW=$(docker exec "$CONTAINER" bash -c "$KERNEL_BENCH_CMD" 2>&1) || {
    python3 -c "
import json
json.dump({'status': 'failed', 'error': 'bench command failed: ${RAW:-no output}'},
          open('$BENCH_OUTPUT', 'w'), indent=2)
"
    exit 0
}

# ── Parse and write bench_output.json ────────────────────────────────────────
echo "$RAW" | python3 - "$BENCH_OUTPUT" << 'PYEOF'
import sys, json, re, math

output_path = sys.argv[1]
raw = sys.stdin.read()

# Parse all gpu_us: lines — one per run
samples = [float(m) for m in re.findall(r'gpu_us[=:\s]+([\d.]+)', raw)]

# Fall back to wall_ms / median_ms if gpu_us not present
if not samples:
    ms_matches = re.findall(r'(?:wall_ms|median_ms)[=:\s]+([\d.]+)', raw)
    samples = [float(m) * 1000.0 for m in ms_matches]

if not samples:
    json.dump({'status': 'failed', 'error': f'no timing data in bench output: {raw[-400:]}'
               }, open(output_path, 'w'), indent=2)
    sys.exit(0)

samples_sorted = sorted(samples)
n = len(samples_sorted)
weakest_us = samples_sorted[-1]   # max = worst = gate pessimistically
median_us  = samples_sorted[n // 2]
min_us     = samples_sorted[0]

# Parse roofline fields (optional — emitted by bench_dsv41flash_moe.py)
flops_m = re.search(r'flops[=:\s]+([\d.eE+]+)', raw)
hbm_m   = re.search(r'hbm_bytes[=:\s]+([\d.eE+]+)', raw)
dtype_m  = re.search(r'\bdtype[=:\s]+([a-zA-Z0-9]+)', raw)
roofline_frac = None
if flops_m and hbm_m:
    try:
        flops     = float(flops_m.group(1))
        hbm_bytes = float(hbm_m.group(1))
        dtype     = (dtype_m.group(1).lower() if dtype_m else 'bf16')
        # gfx950 MAF: FP4=5 PFLOPS, FP8=2.5 PFLOPS, BF16=1.25 PFLOPS; HBM=6.4 TB/s
        maf_f = {'fp4':5e15,'mxfp4':5e15,'fp8':2.5e15,'mxfp8':2.5e15}.get(dtype, 1.25e15)
        maf_b = 6.4e12
        ai    = flops / hbm_bytes
        ridge = maf_f / maf_b
        t     = weakest_us * 1e-6
        ach_f = flops / t
        ach_b = hbm_bytes / t
        roofline_frac = round((ach_f / maf_f) if ai > ridge else (ach_b / maf_b), 4)
    except Exception:
        pass

# Structural signature (for island_db MAP-Elites grid)
sig = {}
for key in ('vgpr', 'barriers', 'spills', 'mfma'):
    m = re.search(rf'sig_{key}[=:\s]+(\d+)', raw, re.IGNORECASE)
    if m:
        sig[key] = int(m.group(1))

result = {
    'status':        'ok',
    'weakest_us':    round(weakest_us, 3),
    'median_us':     round(median_us, 3),
    'min_us':        round(min_us, 3),
    'n_samples':     n,
    'all_samples_us': [round(s, 3) for s in samples],
    'roofline_frac': roofline_frac,
    'sig':           sig,
}
json.dump(result, open(output_path, 'w'), indent=2)
print(f'[bench] weakest={weakest_us:.3f}µs median={median_us:.3f}µs n={n}'
      + (f' roofline={roofline_frac*100:.1f}%' if roofline_frac else ''))
PYEOF
