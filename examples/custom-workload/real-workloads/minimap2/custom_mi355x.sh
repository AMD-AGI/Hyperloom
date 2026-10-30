#!/usr/bin/env bash
# Hyperloom entrypoint -> dispatches to the workload adapter.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ./_minimap2_bench.sh
