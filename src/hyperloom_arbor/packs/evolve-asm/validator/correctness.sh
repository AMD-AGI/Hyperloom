#!/usr/bin/env bash
# Correctness (SNR) validator for evolve-asm pack.
# Runs the SNR check command inside the container and writes correctness_output.json.
# SNR >= threshold_db is required before timing is accepted.
#
# Required env vars:
#   CORRECTNESS_OUTPUT  host path to write correctness_output.json
#   CONTAINER           docker container name
#   KERNEL_SNR_CMD      full SNR command to run inside container
#   SNR_THRESHOLD_DB    minimum SNR in dB (default 30.0)

set -euo pipefail

: "${CORRECTNESS_OUTPUT:?Arbor must inject CORRECTNESS_OUTPUT}"
: "${CONTAINER:?Arbor must inject CONTAINER}"
: "${KERNEL_SNR_CMD:?Arbor must inject KERNEL_SNR_CMD}"

SNR_THRESHOLD_DB="${SNR_THRESHOLD_DB:-30.0}"

mkdir -p "$(dirname "$CORRECTNESS_OUTPUT")"

RAW=$(docker exec "$CONTAINER" bash -c "$KERNEL_SNR_CMD" 2>&1) || {
    python3 -c "
import json
json.dump({'passed': False, 'score': 0.0, 'error': 'SNR command failed to run'},
          open('$CORRECTNESS_OUTPUT', 'w'), indent=2)
"
    exit 0
}

echo "$RAW" | python3 - "$CORRECTNESS_OUTPUT" "$SNR_THRESHOLD_DB" << 'PYEOF'
import sys, json, re, math

output_path = sys.argv[1]
threshold   = float(sys.argv[2])
raw         = sys.stdin.read()

# Parse snr_db: XX.X
m = re.search(r'snr_db[=:\s]+([\d.]+)', raw)
if m:
    snr = float(m.group(1))
    passed = snr >= threshold
    json.dump({'passed': passed, 'score': float(passed),
               'snr_db': snr, 'threshold_db': threshold,
               'response': f'SNR {snr:.1f} dB ({"PASS" if passed else "FAIL"}, threshold {threshold} dB)'},
              open(output_path, 'w'), indent=2)
    print(f'[correctness] SNR {snr:.1f} dB {"PASS" if passed else "FAIL"}')
    sys.exit(0)

# Fall back to cos_sim: 0.XXXXXX
m = re.search(r'cos[_\-]sim[=:\s]+([\d.]+)', raw, re.IGNORECASE)
if m:
    cos = float(m.group(1))
    passed = cos >= 0.9999
    json.dump({'passed': passed, 'score': float(passed),
               'cos_sim': cos,
               'response': f'cos_sim={cos:.6f} ({"PASS" if passed else "FAIL"})'},
              open(output_path, 'w'), indent=2)
    print(f'[correctness] cos_sim={cos:.6f} {"PASS" if passed else "FAIL"}')
    sys.exit(0)

json.dump({'passed': False, 'score': 0.0,
           'error': f'no snr_db or cos_sim in output: {raw[-300:]}'},
          open(output_path, 'w'), indent=2)
print(f'[correctness] FAIL: no SNR data found')
PYEOF
