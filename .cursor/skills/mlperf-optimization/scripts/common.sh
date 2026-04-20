#!/usr/bin/env bash
# =============================================================================
# mlperf-optimization/scripts/common.sh
#
# Shared helpers for MLPerf training optimization:
#   - Trial tier management (quick / convergence / long convergence / full)
#   - Log filtering via trial_monitor.py
#   - YAML quiet config via apply_quiet_config.sh
#   - Metric extraction from MLLOG
#   - Process management
#
# Metric/trace helpers are thin wrappers over `mlperf_utils.py <subcommand>`.
# Keep bash function names and stdout contracts unchanged — callers parse with
# cut/grep/eval, so any drift here will silently break trial parsing.
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MLPERF_UTILS="$SCRIPT_DIR/mlperf_utils.py"

: "${MLPERF_DIR:=/root/Hyperloom-plus-mlperf/training_optimization/mlperf}"
: "${CONFIG_SH:=$MLPERF_DIR/config_MI355X_1x8x1_fp8.sh}"

source "$SCRIPT_DIR/apply_quiet_config.sh"

_CURRENT_PORT="${MASTER_PORT:-29501}"

# ---------------------------------------------------------------------------
# Port management
# ---------------------------------------------------------------------------
next_port() {
    _CURRENT_PORT=$((_CURRENT_PORT + 1))
    echo "$_CURRENT_PORT"
}

# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------
kill_training() {
    pkill -9 -f "train.py" 2>/dev/null || true
    pkill -9 -f "torchrun" 2>/dev/null || true
    pkill -9 -f "torch.distributed" 2>/dev/null || true
    sleep "${KILL_WAIT_S:-5}"
}

# ---------------------------------------------------------------------------
# Trial tier defaults
# ---------------------------------------------------------------------------
trial_tier_defaults() {
    local tier="${1:?tier required (1, 2, 3, or 4)}"
    case "$tier" in
        1)
            echo "train_iters=100 eval_interval=10000 log_freq=1 timeout_s=3600 quiet=1"
            ;;
        2)
            echo "train_iters=500 eval_interval=50 log_freq=1 timeout_s=7200 quiet=1"
            ;;
        3)
            echo "train_iters=2500 eval_interval=50 log_freq=1 timeout_s=14400 quiet=1"
            ;;
        4)
            echo "train_iters= eval_interval= log_freq=32 timeout_s=0 quiet=0"
            ;;
        *)
            echo "ERROR: unknown tier $tier" >&2
            return 1
            ;;
    esac
}

# ---------------------------------------------------------------------------
# run_mlperf_trial  — Central trial runner with tier support
#
# Usage:
#   run_mlperf_trial <label> [tier] [train_iters] [extra_env]
#
# Tier 1 (quick):            100 iters, no eval, MLLOG_TRAIN_LOSS_LOG_FREQ=1, 60min timeout
# Tier 2 (convergence):      500 iters, eval enabled, MLLOG_TRAIN_LOSS_LOG_FREQ=1, 120min timeout
# Tier 3 (long convergence): 2500 iters, eval enabled, MLLOG_TRAIN_LOSS_LOG_FREQ=1, 4hr timeout
# Tier 4 (full):             original config, MLLOG_TRAIN_LOSS_LOG_FREQ=32, no timeout
#
# Prints a TRIAL_RESULT line on stdout for structured parsing.
# Raw log is always preserved at $RESULT_DIR/attempt_<label>_raw.log
# Filtered log at $RESULT_DIR/attempt_<label>.log
# ---------------------------------------------------------------------------
run_mlperf_trial() {
    local label="${1:?label required}"
    local tier="${2:-1}"
    local train_iters_override="${3:-}"
    local extra_env="${4:-}"

    local defaults
    defaults=$(trial_tier_defaults "$tier") || return 1
    local train_iters eval_interval log_freq timeout_s quiet
    eval "$defaults"

    if [ -n "$train_iters_override" ]; then
        train_iters="$train_iters_override"
    fi

    local raw_log="${RESULT_DIR:-.}/attempt_${label}_raw.log"
    local filtered_log="${RESULT_DIR:-.}/attempt_${label}.log"

    echo ""
    echo "--- Trial [$label] tier=$tier iters=$train_iters eval_interval=$eval_interval log_freq=$log_freq timeout=${timeout_s}s ---"

    kill_training

    # Timestamp marker for discover_trace(): created BEFORE training starts so
    # any trace file written during the run will have mtime > marker mtime.
    export TRIAL_START_MARKER="${RESULT_DIR:-.}/_trial_start_${label}"
    touch "$TRIAL_START_MARKER"

    if [ "$quiet" = "1" ] && [ -n "$EXP" ] && [ -f "$EXP" ]; then
        quiet_yaml "$EXP"
    fi

    cd "$MLPERF_DIR"
    source "$CONFIG_SH"

    if [ -n "$train_iters" ]; then
        export PRIMUS_TRAIN_ITERS="$train_iters"
    fi
    if [ -n "$eval_interval" ]; then
        export PRIMUS_EVAL_INTERVAL="$eval_interval"
    fi

    # CRITICAL: override MLLOG_TRAIN_LOSS_LOG_FREQ for short trials
    export MLLOG_TRAIN_LOSS_LOG_FREQ="$log_freq"
    export PYTHONWARNINGS=ignore
    export MASTER_PORT=$(next_port)

    if [ -n "$extra_env" ]; then
        for kv in $extra_env; do
            export "$kv"
        done
    fi

    if [ -n "$PRIMUS_GLOBAL_BATCH_SIZE" ] && [ -n "$EVAL_SAMPLES_INTERVAL" ]; then
        export PRIMUS_EVAL_INTERVAL=$((EVAL_SAMPLES_INTERVAL / PRIMUS_GLOBAL_BATCH_SIZE))
        if [ -n "$eval_interval" ] && [ "$eval_interval" != "10000" ]; then
            export PRIMUS_EVAL_INTERVAL="$eval_interval"
        fi
    fi

    bash setup_container_symlinks.sh 2>/dev/null

    source "$CONFIG_SH"
    if [ -n "$train_iters" ]; then
        export PRIMUS_TRAIN_ITERS="$train_iters"
    fi
    if [ -n "$eval_interval" ]; then
        export PRIMUS_EVAL_INTERVAL="$eval_interval"
    fi
    export MLLOG_TRAIN_LOSS_LOG_FREQ="$log_freq"
    export PYTHONWARNINGS=ignore
    if [ -n "$extra_env" ]; then
        for kv in $extra_env; do
            export "$kv"
        done
    fi

    local exit_code=0
    local max_iters_flag=""
    if [ -n "$train_iters" ]; then
        max_iters_flag="--max-iters $train_iters"
    fi

    if [ "$timeout_s" -gt 0 ] 2>/dev/null; then
        timeout "$timeout_s" bash -c 'bash run_and_time.sh' 2>&1 \
            | python3 "$SKILL_ROOT/scripts/trial_monitor.py" \
                --raw-log "$raw_log" $max_iters_flag --label "$label" \
            | tee "$filtered_log"
        exit_code=${PIPESTATUS[0]}
    else
        bash run_and_time.sh 2>&1 \
            | python3 "$SKILL_ROOT/scripts/trial_monitor.py" \
                --raw-log "$raw_log" $max_iters_flag --label "$label" \
            | tee "$filtered_log"
        exit_code=${PIPESTATUS[0]}
    fi

    if [ "$quiet" = "1" ] && [ -n "$EXP" ]; then
        restore_yaml "$EXP"
    fi

    if [ "$exit_code" -eq 124 ]; then
        echo "WARNING: Trial [$label] timed out after ${timeout_s}s" >&2
    fi

    return $exit_code
}

# ---------------------------------------------------------------------------
# Metric extraction wrappers — delegate to mlperf_utils.py subcommands.
# Stdout format of each wrapper matches the pre-refactor heredoc exactly.
# ---------------------------------------------------------------------------

extract_ms_per_iter() {
    python3 "$MLPERF_UTILS" extract-ms-per-iter \
        "${1:?log file required}" --warmup "${2:-5}" --measure "${3:-5}"
}

extract_mllog_field() {
    python3 "$MLPERF_UTILS" extract-mllog-field \
        "${1:?log file required}" "${2:?field name required}"
}

verify_gbs() {
    python3 "$MLPERF_UTILS" verify-gbs \
        "${1:?log file required}" "${2:?expected GBS required}"
}

extract_losses() {
    python3 "$MLPERF_UTILS" extract-losses "${1:?log file required}"
}

extract_time_to_train() {
    python3 "$MLPERF_UTILS" extract-ttt "${1:?log file required}"
}

# project_ttt: power-law extrapolation of TTT.
# Output: "<projected_ttt_seconds>\t<projected_samples>\t<samples_per_sec>\t<r_squared>"
project_ttt() {
    python3 "$MLPERF_UTILS" project-ttt \
        "${1:?raw log file required}" "${2:?GBS required}"
}

# parse_trial_result: emits shell assignments for eval "$(parse_trial_result "$line")".
parse_trial_result() {
    python3 "$MLPERF_UTILS" parse-trial-result "${1:?TRIAL_RESULT line required}"
}

compute_gain_pct() {
    python3 "$MLPERF_UTILS" compute-gain-pct \
        "${1:?baseline value required}" "${2:?current value required}"
}

# Loss efficiency — Layer 2 quick filter.
compute_loss_efficiency() {
    python3 "$MLPERF_UTILS" compute-loss-efficiency "${1:?log file required}"
}

# TTT gain — Layer 3 final decision.
compute_ttt_gain_pct() {
    python3 "$MLPERF_UTILS" compute-ttt-gain-pct \
        "${1:?baseline projected TTT required}" "${2:?candidate projected TTT required}"
}

# ---------------------------------------------------------------------------
# Action classification: throughput-only vs convergence-affecting
# Determines which evaluation layers apply in the DFS loop.
# ---------------------------------------------------------------------------
classify_action() {
    local action="${1:?action name required}"
    case "$action" in
        config-selection|convergence-speed|fp8-recipe-tuning|hyperparams)
            echo "convergence-affecting"
            ;;
        *)
            echo "throughput-only"
            ;;
    esac
}

detect_nan_in_log() {
    python3 "$MLPERF_UTILS" detect-nan "${1:?log file required}"
}

extract_eval_trajectory() {
    python3 "$MLPERF_UTILS" extract-eval-trajectory "${1:?log file required}"
}

compute_eval_overhead() {
    python3 "$MLPERF_UTILS" compute-eval-overhead "${1:?log file required}"
}

# ---------------------------------------------------------------------------
# Trace discovery (ROCm RPD + standard PyTorch JSON)
#
# Search priority:
#   0) Known tensorboard_dir (tb_trace_dir arg) — no -newer, most reliable
#   1) -newer search for *.pt.trace.json(.gz)
#   2) -newer search for *trace*.json / *chrome*.json / *profiler*.json
#   3) -newer search for *.rpd (+ convert)
#   4) -newer search for *_results.json (rocprofv3)
#   5) Relaxed fallback: most recent *.pt.trace.json(.gz) within last 30 min
# ---------------------------------------------------------------------------
discover_trace() {
    local after_file="${1:?reference file for -newer required}"
    local output_json="${2:?output JSON path required}"
    local tb_trace_dir="${3:-}"

    local found

    _find_newest() {
        find "$@" -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-
    }

    # 0) Priority: search known tensorboard_dir without -newer
    if [ -n "$tb_trace_dir" ] && [ -d "$tb_trace_dir" ]; then
        found=$(_find_newest "$tb_trace_dir" -name "*.pt.trace.json.gz" -type f)
        if [ -n "$found" ]; then
            python3 "$MLPERF_UTILS" decompress-gz-trace "$found" "$output_json"
            echo "TRACE_FORMAT=pytorch_json_gz"
            echo "TRACE_SOURCE=$found"
            return 0
        fi

        found=$(_find_newest "$tb_trace_dir" -name "*.pt.trace.json" -type f)
        if [ -n "$found" ]; then
            cp "$found" "$output_json"
            echo "TRACE_FORMAT=pytorch_json"
            echo "TRACE_SOURCE=$found"
            return 0
        fi
    fi

    # 1a) -newer search for gzipped PyTorch Chrome trace
    found=$(_find_newest /workspace /tmp /root -name "*.pt.trace.json.gz" \
        -newer "$after_file" -type f)
    if [ -n "$found" ]; then
        echo "Found gzipped trace (newer): $found" >&2
        python3 "$MLPERF_UTILS" decompress-gz-trace "$found" "$output_json"
        echo "TRACE_FORMAT=pytorch_json_gz"
        echo "TRACE_SOURCE=$found"
        return 0
    fi

    # 1b) -newer search for standard PyTorch Chrome trace JSON
    found=$(_find_newest /workspace /tmp /root -name "*.pt.trace.json" \
        -newer "$after_file" -type f)
    if [ -n "$found" ]; then
        cp "$found" "$output_json"
        echo "TRACE_FORMAT=pytorch_json"
        echo "TRACE_SOURCE=$found"
        return 0
    fi

    # 2) -newer search for any Chrome-format JSON
    found=$(_find_newest /workspace /tmp /root \
        \( -name "*trace*.json" -o -name "*chrome*.json" -o -name "*profiler*.json" \) \
        -newer "$after_file" -size +10k -type f)
    if [ -n "$found" ]; then
        cp "$found" "$output_json"
        echo "TRACE_FORMAT=chrome_json"
        echo "TRACE_SOURCE=$found"
        return 0
    fi

    # 3) -newer search for RPD files (.rpd = SQLite3 from ROCm RPD backend)
    found=$(_find_newest /workspace /tmp /root -name "*.rpd" \
        -newer "$after_file" -type f)
    if [ -n "$found" ]; then
        echo "Found RPD trace: $found — converting to Chrome JSON..." >&2
        convert_rpd_to_json "$found" "$output_json"
        if [ $? -eq 0 ] && [ -s "$output_json" ]; then
            echo "TRACE_FORMAT=rpd_converted"
            echo "TRACE_SOURCE=$found"
            return 0
        fi
        echo "WARNING: RPD conversion failed" >&2
    fi

    # 4) -newer search for rocprofv3 results JSON
    found=$(_find_newest /workspace /tmp /root -name "*_results.json" \
        -newer "$after_file" -type f)
    if [ -n "$found" ]; then
        cp "$found" "$output_json"
        echo "TRACE_FORMAT=rocprofv3"
        echo "TRACE_SOURCE=$found"
        return 0
    fi

    # 5a) Relaxed fallback: most recent *.pt.trace.json.gz from the last 30 min
    found=$(_find_newest /workspace /tmp /root -name "*.pt.trace.json.gz" \
        -mmin -30 -type f)
    if [ -n "$found" ]; then
        echo "Relaxed fallback: found recent gzipped trace $found" >&2
        python3 "$MLPERF_UTILS" decompress-gz-trace "$found" "$output_json"
        echo "TRACE_FORMAT=pytorch_json_gz_relaxed"
        echo "TRACE_SOURCE=$found"
        return 0
    fi

    # 5b) Relaxed fallback: most recent *.pt.trace.json from the last 30 min
    found=$(_find_newest /workspace /tmp /root -name "*.pt.trace.json" \
        -mmin -30 -type f)
    if [ -n "$found" ]; then
        echo "Relaxed fallback: found recent trace $found" >&2
        cp "$found" "$output_json"
        echo "TRACE_FORMAT=pytorch_json_relaxed"
        echo "TRACE_SOURCE=$found"
        return 0
    fi

    echo "WARNING: No trace file found in any format" >&2
    return 1
}

# ---------------------------------------------------------------------------
# RPD → Chrome JSON conversion (thin wrapper).
# ---------------------------------------------------------------------------
convert_rpd_to_json() {
    python3 "$MLPERF_UTILS" rpd-to-chrome \
        "${1:?rpd file path required}" "${2:?output json path required}"
}

# ---------------------------------------------------------------------------
# Chrome trace filtering (for profiling action).
# ---------------------------------------------------------------------------
filter_trace() {
    python3 "$MLPERF_UTILS" filter-trace \
        "${1:?source trace path required}" "${2:?destination trace path required}" || return 1
}
