#!/usr/bin/env bash
# =============================================================================
# generic-gpu-optimization/scripts/common.sh
# Shared helpers. Source me, or invoke as: ./common.sh <function> <args>
# =============================================================================

set -euo pipefail

# --- kill_workload --------------------------------------------------------
# Kill any lingering benchmark / training processes from the target repo.
kill_workload() {
    local patterns="${1:-bench|benchmark|torchrun}"
    pkill -9 -f "$patterns" 2>/dev/null || true
    sleep 2
}

# --- extract_metric -------------------------------------------------------
# Args: <log_file> <regex_with_one_capture_group>
# Prints the matched value, or empty.
extract_metric() {
    local log="$1"
    local regex="$2"
    [ -f "$log" ] || { echo ""; return 1; }

    python3 - "$log" "$regex" <<'PY'
import re, sys
log_path, regex = sys.argv[1], sys.argv[2]
matches = []
for line in open(log_path, errors="ignore"):
    m = re.search(regex, line)
    if m:
        try:
            matches.append(float(m.group(1)))
        except (ValueError, IndexError):
            pass
if not matches:
    sys.exit(0)
# Use last match (final summary), or median if many
if len(matches) > 5:
    matches.sort()
    print(f"{matches[len(matches)//2]:.4f}")
else:
    print(f"{matches[-1]:.4f}")
PY
}

# --- copy_or_warn ---------------------------------------------------------
copy_or_warn() {
    local src="$1" dst="$2"
    if [ -f "$src" ]; then cp "$src" "$dst"; else echo "WARN: missing $src"; fi
}

# --- next_attempt_id ------------------------------------------------------
next_attempt_id() {
    local results="${1:-$RESULT_DIR/results.tsv}"
    if [ -f "$results" ]; then
        awk -F'\t' 'NR>1 {print $1}' "$results" | sort -n | tail -1 | awk '{print $1+1}'
    else
        echo 0
    fi
}

# --- gpu_idle_check -------------------------------------------------------
gpu_idle_check() {
    local pct
    pct=$(rocm-smi --showuse 2>/dev/null | grep -oE '[0-9]+%' | head -1 | tr -d '%')
    if [ -n "$pct" ] && [ "$pct" -gt 5 ]; then
        echo "WARN: GPU $pct% busy — measurement noise will be high"
    fi
}

# Allow invoking as a CLI
if [ "${BASH_SOURCE[0]}" = "$0" ] && [ $# -gt 0 ]; then
    "$@"
fi
