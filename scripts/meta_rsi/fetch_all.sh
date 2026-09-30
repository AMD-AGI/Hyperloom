#!/usr/bin/env bash
# Fetch the round's plans in order (TIERS, default tier1 tier2 tier3) into PULSE_BUNDLES.
# The Pulse key is read from PULSE_API_KEY or a root-only key file, never from the command line.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROUND=${PULSE_ROUND_DIR:-/wekafs/csl/Hyperloom-Sessions/meta_rsi/pulse15d}
BUNDLES=${PULSE_BUNDLES:-/root/pulse15d/02_bundles}
if [ -z "${PULSE_API_KEY:-}" ]; then
  PULSE_API_KEY=$(cat "${PULSE_KEY_FILE:-$HOME/.config/hyperloom-pulse/key}")
fi
export PULSE_API_KEY
mkdir -p "$BUNDLES"
for t in ${TIERS:-tier1 tier2 tier3}; do
  echo "$(date -u +%FT%TZ) start $t"
  python3 "$HERE/pulse.py" fetch --plan "$ROUND/plan_$t.jsonl" --out "$BUNDLES" \
    --log "$ROUND/02_fetch_$t.rows.jsonl" --jobs "${JOBS:-16}" --min-free-gb 200
  echo "$(date -u +%FT%TZ) end $t rc=$?"
done
