#!/usr/bin/env bash
# Copyright Advanced Micro Devices, Inc. All rights reserved.

# Placeholder bench body shared by every entry in BENCH_REGISTRY until
# real probe implementations land. Writes a single marker JSON into
# $DYNAMIC_BENCH_OUTPUT_DIR so the runner has something to journal.

set -eu

: "${DYNAMIC_BENCH_OUTPUT_DIR:?DYNAMIC_BENCH_OUTPUT_DIR is required}"
: "${DYNAMIC_BENCH_WORKTREE:?DYNAMIC_BENCH_WORKTREE is required}"

bench_id="${BENCH_ID:-unknown}"
marker="${DYNAMIC_BENCH_OUTPUT_DIR}/result.json"
mkdir -p "${DYNAMIC_BENCH_OUTPUT_DIR}"
cat > "${marker}" <<JSON
{
  "bench_id": "${bench_id}",
  "status": "placeholder",
  "worktree": "${DYNAMIC_BENCH_WORKTREE}",
  "ts": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON

echo "placeholder bench ${bench_id} ok"
