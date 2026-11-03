#!/usr/bin/env bash
# The parallel-dossier pack's bench: score_dossier.py's score of the committed drafts.
# Run by the harness on the host (a `document` workload, "docker_container": "@host"),
# with WORKSPACE_ROOT and BENCH_OUTPUT set. Writes {"dossier_score", "parts", ...}.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
python3 "$here/score_dossier.py" "${WORKSPACE_ROOT:?WORKSPACE_ROOT is not set}" \
    --json "${BENCH_OUTPUT:?BENCH_OUTPUT is not set}"
