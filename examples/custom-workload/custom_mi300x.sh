#!/usr/bin/env bash
# Hyperloom custom-workload entrypoint.
#
# Hyperloom looks for `custom_<runner-type>.sh` inside $HYPERLOOM_BYPASS_SCRIPTS_DIR,
# falling back to a lone *.sh in that directory. This example keeps one script and
# symlinks the other runner names onto it, because the workload here does not vary
# by GPU.
#
# Hyperloom injects:
#   RESULT_DIR       where the result JSON must be written
#   RESULT_FILENAME  its basename, without the .json extension ("inferencex_result")
#   RUNNER_TYPE      the resolved runner, e.g. mi300x
#   MODEL            --model, verbatim
#
# Anything else the benchmark reads -- EXAMPLE_WORKERS and EXAMPLE_CHUNK here --
# arrives as ordinary environment, either from --extra-env or from a variant the
# optimizer is trying. That is the whole mechanism: the loop restarts this script
# with different environment and compares what comes back.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "[example] runner=${RUNNER_TYPE:-<unset>} workers=${EXAMPLE_WORKERS:-1} chunk=${EXAMPLE_CHUNK:-100000}"
exec python3 benchmark.py
