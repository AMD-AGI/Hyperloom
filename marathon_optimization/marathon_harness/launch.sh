#!/usr/bin/env bash
set -euo pipefail

# Marathon Harness — tmux launcher
#
# Usage:
#   ./launch.sh MODEL_NAME BASE_DIR [--max-hours H] [--max-cost-usd $] [extra args...]
#   ./launch.sh --attach          # reattach to running session
#   ./launch.sh --kill            # kill running session
#   ./launch.sh --logs            # tail the log file
#   ./launch.sh --status          # show session dir contents & latest state
#
# Examples:
#   ./launch.sh DeepSeek-R1-0528 /shared_nfs/nehaprakriya/Agentic-InferenceX/DeepSeek-R1-0528-optimized
#   ./launch.sh DeepSeek-R1-0528 /shared_nfs/.../DeepSeek-R1-0528-optimized --max-hours 1 --max-cost-usd 50
#   ./launch.sh DeepSeek-R1-0528 /shared_nfs/.../DeepSeek-R1-0528-optimized --resume
#   ./launch.sh --attach

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="${ENV_FILE:-/shared_nfs/nehaprakriya/TBO/.env}"
TMUX_SESSION="marathon"
LOG_FILE="/tmp/marathon_run.log"

# ── Helper commands ──────────────────────────────────────────────────────────

case "${1:-}" in
    --attach)
        exec tmux attach -t "$TMUX_SESSION"
        ;;
    --kill)
        tmux kill-session -t "$TMUX_SESSION" 2>/dev/null && echo "Session killed." || echo "No session running."
        exit 0
        ;;
    --logs)
        exec tail -f "$LOG_FILE"
        ;;
    --status)
        echo "=== tmux session ==="
        tmux has-session -t "$TMUX_SESSION" 2>/dev/null && echo "  Running" || echo "  Not running"
        echo ""
        echo "=== Log tail ==="
        tail -20 "$LOG_FILE" 2>/dev/null || echo "  No log file"
        echo ""
        echo "=== Latest session dir ==="
        # Find most recent session
        for base in /shared_nfs/nehaprakriya/Agentic-InferenceX/*/sessions; do
            if [ -d "$base" ]; then
                latest=$(ls -td "$base"/*/ 2>/dev/null | head -1)
                if [ -n "$latest" ]; then
                    echo "  $latest"
                    [ -f "$latest/state.json" ] && python3 -c "
import json, sys
s = json.load(open('$latest/state.json'))
print(f\"  Phase:      {s.get('phase','?')}\")
print(f\"  Iteration:  {s.get('iteration_count',0)}\")
print(f\"  Best tput:  {s.get('best_tput_per_gpu',0):.1f} tok/s/GPU\")
print(f\"  Gain:       {s.get('cumulative_gain_pct',0):.1f}%\")
print(f\"  LLM cost:   \${s.get('total_llm_cost_usd',0):.2f}\")
print(f\"  Stack size: {len(s.get('action_stack',[]))}\")
print(f\"  Completed:  {len(s.get('completed_actions',[]))}\")
" 2>/dev/null
                fi
            fi
        done
        exit 0
        ;;
esac

# ── Validate args ────────────────────────────────────────────────────────────

if [[ $# -lt 2 ]]; then
    cat <<'USAGE'
Marathon Harness — tmux launcher

Usage:
  ./launch.sh MODEL_NAME BASE_DIR [options...]

Options (passed to marathon_harness):
  --max-hours H        Wall-clock limit (default: 24)
  --max-cost-usd $     LLM cost limit (default: unlimited)
  --resume             Resume from latest checkpoint
  --dry-run            Print config and exit
  --gpu-type TYPE      GPU type (default: MI355X)
  --gpu-count N        GPU count (default: 8)
  --tp N               Tensor parallelism (default: 8)
  --framework F        sglang or vllm (default: sglang)

Management:
  ./launch.sh --attach   Reattach to running session
  ./launch.sh --kill     Kill running session
  ./launch.sh --logs     Tail the log file
  ./launch.sh --status   Show current state

Examples:
  ./launch.sh DeepSeek-R1-0528 /shared_nfs/.../DeepSeek-R1-0528-optimized --max-hours 4
  ./launch.sh DeepSeek-R1-0528 /shared_nfs/.../DeepSeek-R1-0528-optimized --resume
USAGE
    exit 1
fi

MODEL_NAME="$1"
BASE_DIR="$2"
shift 2
EXTRA_ARGS=("$@")

# ── Preflight checks ────────────────────────────────────────────────────────

if ! command -v tmux &>/dev/null; then
    echo "ERROR: tmux not installed. Run: apt-get install -y tmux"
    exit 1
fi

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Session '$TMUX_SESSION' is already running."
    echo "  Attach:  ./launch.sh --attach"
    echo "  Kill:    ./launch.sh --kill"
    exit 1
fi

if [ ! -d "$BASE_DIR" ]; then
    echo "ERROR: Base dir does not exist: $BASE_DIR"
    exit 1
fi

# ── Source env ───────────────────────────────────────────────────────────────

ENV_SOURCE=""
if [ -f "$ENV_FILE" ]; then
    ENV_SOURCE="set -a; source $ENV_FILE; set +a; "
    echo "Env file: $ENV_FILE"
else
    echo "Warning: No .env file at $ENV_FILE (OOB backends may not work)"
fi

# ── Build command ────────────────────────────────────────────────────────────

CMD="${ENV_SOURCE}cd $PKG_DIR && python3 -m marathon_harness $MODEL_NAME $BASE_DIR ${EXTRA_ARGS[*]:-} 2>&1 | tee $LOG_FILE; echo '--- MARATHON EXITED (code: '\$?') ---'; exec bash"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  Marathon Harness                                           ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "  Model:    $MODEL_NAME"
echo "  Base dir: $BASE_DIR"
echo "  Args:     ${EXTRA_ARGS[*]:-<none>}"
echo "  Log:      $LOG_FILE"
echo "  Session:  $TMUX_SESSION"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# ── Launch tmux ──────────────────────────────────────────────────────────────

tmux new-session -d -s "$TMUX_SESSION" -c "$PKG_DIR" "$CMD"

echo "Marathon is running in tmux."
echo ""
echo "  Attach:   ./launch.sh --attach    (or: tmux attach -t $TMUX_SESSION)"
echo "  Logs:     ./launch.sh --logs      (or: tail -f $LOG_FILE)"
echo "  Status:   ./launch.sh --status"
echo "  Kill:     ./launch.sh --kill      (or: tmux kill-session -t $TMUX_SESSION)"
echo ""
echo "You can safely close your laptop now."
