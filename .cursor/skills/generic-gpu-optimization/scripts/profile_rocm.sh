#!/usr/bin/env bash
# =============================================================================
# profile_rocm.sh — wrap the BENCH_COMMAND with rocprofv3 (or rocprof v2 fallback)
# Outputs: $RESULT_DIR/profiles/rocprof.json
# =============================================================================
set -euo pipefail
[ -n "${RESULT_DIR:-}" ] || { echo "RESULT_DIR not set"; exit 1; }
source "$RESULT_DIR/state.env"
source "$RESULT_DIR/detected.env"
[ -f "$RESULT_DIR/kept_env.sh" ] && source "$RESULT_DIR/kept_env.sh"

mkdir -p "$RESULT_DIR/profiles"
cd "$REPO_ROOT"

if command -v rocprofv3 >/dev/null 2>&1; then
    echo "[profile] using rocprofv3"
    rocprofv3 \
        --kernel-trace --hip-trace \
        --output-format json \
        -d "$RESULT_DIR/profiles/rocprof_run" \
        -- bash -c "$BENCH_COMMAND"
    # Coalesce to a single json
    find "$RESULT_DIR/profiles/rocprof_run" -name '*.json' -exec cat {} \; > \
        "$RESULT_DIR/profiles/rocprof.json"

elif command -v rocprof >/dev/null 2>&1; then
    echo "[profile] using rocprof v2 fallback"
    rocprof --hip-trace --hsa-trace \
        -o "$RESULT_DIR/profiles/rocprof.csv" \
        bash -c "$BENCH_COMMAND"
    # Convert CSV to a minimal json the analyzer understands
    python3 - <<PY > "$RESULT_DIR/profiles/rocprof.json"
import csv, json, sys
rows = list(csv.DictReader(open("$RESULT_DIR/profiles/rocprof.csv")))
events = [{"KernelName": r.get("KernelName") or r.get("Name"),
           "DurationNs": float(r.get("DurationNs") or r.get("Duration") or 0)} for r in rows]
print(json.dumps({"kernels": events}))
PY

else
    echo "ERROR: neither rocprofv3 nor rocprof installed"
    exit 1
fi

echo "[profile] wrote $RESULT_DIR/profiles/rocprof.json"
