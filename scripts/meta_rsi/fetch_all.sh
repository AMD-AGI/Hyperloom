#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Fetch the round's plans in order (TIERS, default tier1 tier2 tier3) into PULSE_BUNDLES.
# The Pulse key is read from PULSE_API_KEY or a root-only key file, never from the command line.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROUND=${PULSE_ROUND_DIR:?set PULSE_ROUND_DIR to the round directory}
BUNDLES=${PULSE_BUNDLES:?set PULSE_BUNDLES to the archive download directory}
if [ -z "${PULSE_API_KEY:-}" ]; then
  PULSE_API_KEY=$(cat "${PULSE_KEY_FILE:-$HOME/.config/hyperloom-pulse/key}")
fi
export PULSE_API_KEY
mkdir -p "$BUNDLES"
for t in ${TIERS:-tier1 tier2 tier3}; do
  echo "$(date -u +%FT%TZ) start $t"
  python3 "$HERE/pulse.py" fetch --plan "$ROUND/plan_$t.jsonl" --out "$BUNDLES" \
    --log "$ROUND/02_fetch_$t.rows.jsonl" --jobs "${JOBS:-16}" --min-free-gb "${MIN_FREE_GB:-200}"
  echo "$(date -u +%FT%TZ) end $t rc=$?"
done
