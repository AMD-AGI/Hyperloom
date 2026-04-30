#!/usr/bin/env bash
# apply_patch.sh — apply one patch + restart server + re-baseline.
#
# Wraps patch_inductor.py (IR-6 soft) + safe server lifecycle (IR-4
# kill_server, IR-5 no pkill -f sglang) + run_baseline.sh re-bench.
# This is the integrate phase (IR-3 mandatory).
#
# Usage:
#   bash apply_patch.sh \
#       --target-file <path>      # source kernel to patch
#       --patch-file <path>       # GEAK/OOB output to overlay
#       [--best-config <path>]    # required when tuning block_size/num_warps
#       [--tuning-keys "block_size,num_warps"]
#       [--skip-rebaseline]       # for dry runs
#
# Env (consumed by run_baseline.sh on re-bench):
#   MODEL, TP, CONC, ISL, OSL, INFERENCEX_PATH, FRAMEWORK
#   PORT (default 8888)
#   SERVER_KILL_WAIT_S (default 10)
#   EVAL_TASK (set to "gsm8k" to also run accuracy gate)
#
# Output: writes log under $RESULT_DIR/apply_patch.log if set.

set -euo pipefail

TARGET_FILE=""
PATCH_FILE=""
BEST_CONFIG=""
TUNING_KEYS=""
SKIP_REBASELINE="0"

while [ $# -gt 0 ]; do
    case "$1" in
        --target-file)    TARGET_FILE="$2"; shift 2 ;;
        --patch-file)     PATCH_FILE="$2";  shift 2 ;;
        --best-config)    BEST_CONFIG="$2"; shift 2 ;;
        --tuning-keys)    TUNING_KEYS="$2"; shift 2 ;;
        --skip-rebaseline) SKIP_REBASELINE="1"; shift ;;
        *)
            echo "ERROR: unknown arg $1" >&2
            exit 2
            ;;
    esac
done

if [ -z "$TARGET_FILE" ] || [ -z "$PATCH_FILE" ]; then
    echo "usage: apply_patch.sh --target-file X --patch-file Y [--best-config Z] [--tuning-keys K] [--skip-rebaseline]" >&2
    exit 2
fi

AGENT_PKG_DIR="${AGENT_PKG_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/..}"
PKG_ROOT="$(cd "$AGENT_PKG_DIR/../.." && pwd)"
PATCH_INDUCTOR="$PKG_ROOT/scripts/patch_inductor.py"
RUN_BASELINE="$PKG_ROOT/scripts/run_baseline.sh"
LOG_PATH="${RESULT_DIR:-/tmp}/apply_patch.log"
mkdir -p "$(dirname "$LOG_PATH")"

log() {
    echo "[$(date -Iseconds)] apply_patch.sh: $*" | tee -a "$LOG_PATH" >&2
}

log "starting: target=$TARGET_FILE patch=$PATCH_FILE best_config=$BEST_CONFIG tuning_keys=$TUNING_KEYS"

# 1. Pre-flight: server kill + GPU memory check (IR-4 + IR-5).
SERVER_KILL_WAIT_S="${SERVER_KILL_WAIT_S:-10}"
PIDS=$(pgrep -f 'python.*-m sglang.launch_server' 2>/dev/null || true)
if [ -n "$PIDS" ]; then
    log "killing existing sglang.launch_server pids: $PIDS"
    # IR-5: targeted kill, NOT pkill -f sglang.
    for pid in $PIDS; do kill "$pid" 2>/dev/null || true; done
fi
PIDS_VLLM=$(pgrep -f 'python.*-m vllm.entrypoints' 2>/dev/null || true)
if [ -n "$PIDS_VLLM" ]; then
    log "killing existing vllm pids: $PIDS_VLLM"
    for pid in $PIDS_VLLM; do kill "$pid" 2>/dev/null || true; done
fi
sleep "$SERVER_KILL_WAIT_S"
log "GPU memory after kill (rocm-smi or nvidia-smi):"
( rocm-smi --showmeminfo vram 2>/dev/null || nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null || true ) | tee -a "$LOG_PATH" >&2

# 2. Apply the patch via patch_inductor.py (IR-6 soft warns on its own).
PATCH_ARGS=(--target-file "$TARGET_FILE")
if [ -n "$BEST_CONFIG" ]; then
    PATCH_ARGS+=(--best-config "$BEST_CONFIG")
fi
if [ -n "$TUNING_KEYS" ]; then
    PATCH_ARGS+=(--tuning-keys "$TUNING_KEYS")
fi
log "patch_inductor.py argv: ${PATCH_ARGS[*]}"
if ! python3 "$PATCH_INDUCTOR" "${PATCH_ARGS[@]}" >> "$LOG_PATH" 2>&1; then
    log "patch_inductor.py failed (rc=$?)"
    exit 1
fi

# 3. Overlay the patch file on the target. Atomic-ish: write to .new,
#    fsync (best-effort), then rename. If patch == target (GEAK rewrote
#    in place), this is a no-op.
if [ "$PATCH_FILE" != "$TARGET_FILE" ]; then
    cp "$PATCH_FILE" "$TARGET_FILE.new"
    sync || true
    mv "$TARGET_FILE.new" "$TARGET_FILE"
    log "overlay done: $PATCH_FILE -> $TARGET_FILE"
fi

# 4. Compute fingerprint so kernel agent can include it in RESPONSE.
FINGERPRINT=$(sha256sum "$TARGET_FILE" | cut -d' ' -f1 | head -c 16)
log "patch_fingerprint=$FINGERPRINT"

# 5. Re-baseline (unless skipped).
if [ "$SKIP_REBASELINE" = "1" ]; then
    log "--skip-rebaseline set; skipping run_baseline.sh"
    exit 0
fi
if [ ! -f "$RUN_BASELINE" ]; then
    log "WARN: run_baseline.sh not found at $RUN_BASELINE; skipping re-bench"
    exit 0
fi
log "running re-baseline: $RUN_BASELINE"
if ! bash "$RUN_BASELINE" >> "$LOG_PATH" 2>&1; then
    RC=$?
    log "run_baseline.sh failed (rc=$RC)"
    exit $RC
fi
log "apply_patch.sh complete; fingerprint=$FINGERPRINT"
exit 0
