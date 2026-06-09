#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc. All rights reserved.

# Placeholder bench body shared by every entry in BENCH_REGISTRY. Writes a
# single marker JSON into $SPECIALIST_BENCH_OUTPUT_DIR so the runner has
# something to journal. Real probe bodies replace this per bench_id; they must
# stay worktree-scoped and must never start a serving process.

set -eu

: "${SPECIALIST_BENCH_OUTPUT_DIR:?SPECIALIST_BENCH_OUTPUT_DIR is required}"
: "${SPECIALIST_BENCH_WORKTREE:?SPECIALIST_BENCH_WORKTREE is required}"

bench_id="${BENCH_ID:-unknown}"
params_json="${SPECIALIST_BENCH_PARAMS_JSON:-}"
if [ -z "${params_json}" ]; then
  params_json="{}"
fi
marker="${SPECIALIST_BENCH_OUTPUT_DIR}/result.json"
mkdir -p "${SPECIALIST_BENCH_OUTPUT_DIR}"
cat > "${marker}" <<JSON
{
  "bench_id": "${bench_id}",
  "status": "placeholder",
  "worktree": "${SPECIALIST_BENCH_WORKTREE}",
  "params": ${params_json},
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

echo "placeholder bench ${bench_id} ok"
