#!/usr/bin/env bash
# One Meta RSI analysis round, end to end: enumerate and fetch Pulse sessions, account the tokens,
# replay the levers, and pick the A/B scenario. The A/B itself and compare_ab.py run afterwards.
#   PULSE_ROUND_DIR=<dir> PULSE_BUNDLES=<dir> PULSE_RECENT_SINCE=YYYY-MM-DD run_round.sh [LAST_DAYS]
# Needs a SOCKS5 tunnel to the Pulse host on PULSE_SOCKS (default 127.0.0.1:1080) and PYTHONPATH
# pointing at the Hyperloom src the levers are replayed from (defaults to this checkout).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DAYS=${1:-15}
export PULSE_ROUND_DIR=${PULSE_ROUND_DIR:?set PULSE_ROUND_DIR}
export PULSE_BUNDLES=${PULSE_BUNDLES:?set PULSE_BUNDLES}
if [ -z "${PULSE_API_KEY:-}" ]; then
  PULSE_API_KEY=$(cat "${PULSE_KEY_FILE:-$HOME/.config/hyperloom-pulse/key}")
fi
export PULSE_API_KEY
export PYTHONPATH=${PYTHONPATH:-$(cd "$HERE/../.." && pwd)/src}
mkdir -p "$PULSE_ROUND_DIR/01_census" "$PULSE_BUNDLES"
cd "$HERE"

python3 pulse.py enum --last-days "$DAYS" --layers global --out "$PULSE_ROUND_DIR/00_enum_global"
python3 targets.py
python3 pulse.py census --targets "$PULSE_ROUND_DIR/targets_all.txt" --out "$PULSE_ROUND_DIR/01_census/ls.jsonl.gz" || true
python3 pulse.py census-retry --census "$PULSE_ROUND_DIR/01_census/ls.jsonl.gz" \
  --index "$PULSE_ROUND_DIR/00_enum_global/index_rows.jsonl" --out "$PULSE_ROUND_DIR/01_census/retry.jsonl.gz" || true
python3 build_plan.py
./fetch_all.sh

python3 an_ledger.py
python3 an_global.py > /dev/null
python3 an_era.py
python3 an_local.py
python3 an_idle.py > /dev/null
python3 an_specialist.py > /dev/null
python3 an_replay_levers.py
python3 select_scenario.py
echo "analysis in $PULSE_ROUND_DIR/analysis; run the A/B on the top scenario, then compare_ab.py"
