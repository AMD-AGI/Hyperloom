#!/usr/bin/env bash
# Launch a Hyperloom `optimize` session for MiniMax-M3 (vLLM, MI355X) to test
# whether Hyperloom can DISCOVER the shared-expert fusion (PR #46545) on its own,
# starting from STOCK vLLM (no fusion). The discovery path we wired:
#   model_profile detects n_shared_experts -> build_orchestrator_prompt injects an
#   "MoE + shared expert -> consider shared-expert fusion (see KB)" lead ->
#   orchestrator dispatches a specialist with "shared expert fusion" in the task ->
#   select_kb injects kb/fusion/empirical_kb.md (mechanism + source-file map +
#   validation recipe; NO ready-made patch — the agent must write the code).
#
# PREREQUISITE (currently the blocker): the local LLM proxy must be UP, because the
# container can't resolve core42 directly. Bring up the same proxy used for the
# DeepSeek run so that `curl http://localhost:8123/api/v1/llm-proxy/v1/models`
# returns 200, THEN run this script.
#
#   bash scripts/run_minimaxm3_optimize.sh fg     # foreground (watch the 1st run)
#   bash scripts/run_minimaxm3_optimize.sh        # background, survives disconnect
set -euo pipefail

# ── LLM proxy / auth (same bridge as run_dsv4_optimize.sh) ──────────────────────
export SAFE_API_KEY="${SAFE_API_KEY:-ak-n6jl5-EN_63Amus5gSq8vX3ZDz0j41yaqO3Akw-nWc0}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://localhost:8123/api/v1/llm-proxy/v1}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-http://localhost:8123/api/v1/llm-proxy}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-$SAFE_API_KEY}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$SAFE_API_KEY}"

MODE="background"
if [ "${1:-}" = "fg" ] || [ "${1:-}" = "foreground" ] || [ "${FOREGROUND:-0}" = "1" ]; then
    MODE="foreground"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ── Workload (matches the validated 1k/1k MiniMax-M3 point) ──────────────────────
export ISL="${ISL:-1024}"
export OSL="${OSL:-1024}"
export CONCURRENCY="${CONCURRENCY:-64}"
export RANDOM_RANGE_RATIO="${RANDOM_RANGE_RATIO:-1.0}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((ISL + OSL + 256))}"
# InferenceX bench fallback needs this (sources benchmark_lib.sh).
export INFERENCEX_PATH="${INFERENCEX_PATH:-/home/xingran.fan@amd.com/InferenceX}"
export PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx950}"

# Deliberately DO NOT set HYPERLOOM_OPT_HINTS for shared-expert fusion — the point
# is to verify the *auto-detected* lead (from model_profile) surfaces it unaided.

# ── Tunables ─────────────────────────────────────────────────────────────────────
MODEL="${MODEL:-/it-share-4/MiniMax-M3-FP8}"
LAUNCH_SCRIPT="${LAUNCH_SCRIPT:-scripts/launch_minimaxm3_mi355x_serve.sh}"
FRAMEWORK="${FRAMEWORK:-vllm}"
TP="${TP:-4}"
PORT="${PORT:-8951}"
GPUS="${GPUS:-0,1,2,3}"
MAX_HOURS="${MAX_HOURS:-3}"
TARGET_GAIN="${TARGET_GAIN:-10}"
# Gateway exposes claude-opus-4-7 / 4-6.
AGENT_MODEL="${AGENT_MODEL:-claude-opus-4-7}"
# Specialist sub-agents (dispatch_agents) inherit this; the proxy only serves
# claude-opus-4-7/4-6, so make sure the default is one of those (not sonnet).
export AGENT_MODEL
SESSION_DIR="${SESSION_DIR:-}"
if [ -z "$SESSION_DIR" ]; then
    SESSION_DIR='/home/xingran.fan@amd.com/Arbor-sessions/{model_name}-{timestamp}'
fi

RUN_TS="$(date +%Y%m%d-%H%M%S)"
RUN_LOG_DIR="${RUN_LOG_DIR:-$REPO_ROOT/run-logs}"
mkdir -p "$RUN_LOG_DIR"
LOG="$RUN_LOG_DIR/mm3-optimize-$RUN_TS.log"
PIDFILE="$RUN_LOG_DIR/mm3-optimize-$RUN_TS.pid"
ln -sfn "$LOG" "$RUN_LOG_DIR/mm3-optimize-latest.log"
ln -sfn "$PIDFILE" "$RUN_LOG_DIR/mm3-optimize-latest.pid"

# ── Preflight: refuse to start if the LLM proxy is down (saves a confusing crash) ─
if ! curl -sf -m 6 -o /dev/null "${OPENAI_BASE_URL}/models" 2>/dev/null; then
    echo "ERROR: LLM proxy not reachable at ${OPENAI_BASE_URL}/models" >&2
    echo "       Bring up the local llm-proxy (the same one used for DeepSeek), then retry." >&2
    exit 2
fi

echo "Mode:        $MODE"
echo "Model:       $MODEL   ($FRAMEWORK, TP=$TP, PORT=$PORT, GPUS=$GPUS)"
echo "Launch:      $LAUNCH_SCRIPT"
echo "Workload:    ISL=$ISL OSL=$OSL CONC=$CONCURRENCY MAX_MODEL_LEN=$MAX_MODEL_LEN"
echo "Budget:      ${MAX_HOURS}h  target +${TARGET_GAIN}%   agent=$AGENT_MODEL"
echo "Session:     $SESSION_DIR"
echo "Log:         $LOG"
echo

OPTIMIZE_ARGS=(
    --model "$MODEL"
    --launch-script "$LAUNCH_SCRIPT"
    --framework "$FRAMEWORK"
    --tp "$TP"
    --port "$PORT"
    --gpus "$GPUS"
    --max-hours "$MAX_HOURS"
    --target-gain "$TARGET_GAIN"
    --session-dir "$SESSION_DIR"
    --agent-model "$AGENT_MODEL"
)

if [ "$MODE" = "foreground" ]; then
    echo "$$" > "$PIDFILE"
    exec python3 -u -m hyperloom.cli optimize "${OPTIMIZE_ARGS[@]}" 2>&1 | tee "$LOG"
fi

setsid nohup python3 -u -m hyperloom.cli optimize "${OPTIMIZE_ARGS[@]}" \
    < /dev/null > "$LOG" 2>&1 &
CHILD_PID=$!
echo "$CHILD_PID" > "$PIDFILE"
disown "$CHILD_PID" 2>/dev/null || true
echo "Started MiniMax-M3 optimize session in background (PID $CHILD_PID)."
echo "Follow:  tail -f run-logs/mm3-optimize-latest.log"
echo "Stop:    kill -TERM \$(cat run-logs/mm3-optimize-latest.pid)"
