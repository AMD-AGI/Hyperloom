#!/usr/bin/env bash
#
# Launch the Marathon three-process architecture:
#   Pane 0: Watchdog Supervisor (marathon-inference-optimization/watchdog/SKILL.md)
#   Pane 1: Orchestrator (marathon-inference-optimization/SKILL.md)
#   Pane 2: Kernel Manager (marathon-inference-optimization/kernel-manager/SKILL.md)
#
# Usage:
#   ./launch_marathon.sh <model_name> <result_dir> [skill_root]
#
# Example:
#   ./launch_marathon.sh qwen35-397b /shared_nfs/results/qwen35-397b-mi355x
#   ./launch_marathon.sh deepseek-r1 /shared_nfs/results/deepseek-r1-mi355x /path/to/skills

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_NAME="${1:?Usage: $0 <model_name> <result_dir> [skill_root]}"
RESULT_DIR="${2:?Usage: $0 <model_name> <result_dir> [skill_root]}"
SKILL_ROOT="${3:-/shared_nfs/nehaprakriya/TBO/.cursor/skills}"

MARATHON_SKILL="$SKILL_ROOT/marathon-inference-optimization/SKILL.md"
MANAGER_SKILL="$SKILL_ROOT/marathon-inference-optimization/kernel-manager/SKILL.md"
WATCHDOG_SKILL="$SKILL_ROOT/marathon-inference-optimization/watchdog/SKILL.md"
RCA_SKILL="/shared_nfs/nehaprakriya/agentic-rc/.cursor/skills/training-workload-rca/SKILL.md"
BASE_DIR="${MARATHON_BASE_DIR:-$RESULT_DIR}"
SESSION_NAME="marathon-${MODEL_NAME}"

for skill_file in "$MARATHON_SKILL" "$MANAGER_SKILL" "$WATCHDOG_SKILL"; do
    if [ ! -f "$skill_file" ]; then
        echo "ERROR: Skill not found at $skill_file"
        exit 1
    fi
done

if [ ! -f "$RCA_SKILL" ]; then
    echo "WARNING: RCA skill not found at $RCA_SKILL — Watchdog RCA dispatch will be limited"
fi

mkdir -p "$RESULT_DIR/kernel_manager/rca_reports"
mkdir -p "$RESULT_DIR/kernel_manager/merge_ready"

touch "$RESULT_DIR/kernel_manager/event_log.jsonl"
touch "$RESULT_DIR/kernel_manager/findings.jsonl"
touch "$RESULT_DIR/kernel_manager/work_queue.jsonl"
touch "$RESULT_DIR/kernel_manager/results.jsonl"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session '$SESSION_NAME' already exists. Attach with: tmux attach -t $SESSION_NAME"
    exit 1
fi

WATCHDOG_PROMPT="Read ${WATCHDOG_SKILL}. Also read ${RCA_SKILL} for the RCA methodology. \
You are the Watchdog Supervisor for ${MODEL_NAME} optimization on MI355X. \
RESULT_DIR=${RESULT_DIR}. \
Monitor ${RESULT_DIR}/kernel_manager/event_log.jsonl for new events. \
Write findings to ${RESULT_DIR}/kernel_manager/findings.jsonl. \
Write detailed RCA reports to ${RESULT_DIR}/kernel_manager/rca_reports/. \
Begin monitoring now."

ORCHESTRATOR_PROMPT="Read ${MARATHON_SKILL}. \
You are optimizing ${MODEL_NAME} on MI355X. \
RESULT_DIR=${RESULT_DIR}. \
BASE_DIR=${BASE_DIR} (read-only — contains exploration_tree, results, optimizations from prior runs). \
The Kernel Manager is running in tmux pane 2 and will process work_queue.jsonl entries. \
The Watchdog Supervisor is running in tmux pane 0 and monitors event_log.jsonl. \
Write ALL new data to RESULT_DIR. Read prior run data from BASE_DIR. Never modify files in BASE_DIR. \
Write events to ${RESULT_DIR}/kernel_manager/event_log.jsonl on merge failures and crashes. \
Check ${RESULT_DIR}/kernel_manager/findings.jsonl for Watchdog guidance between DFS actions. \
Begin the Marathon protocol from Step 0 (Warm-Start)."

MANAGER_PROMPT="Read ${MANAGER_SKILL}. \
You are the Kernel Manager for ${MODEL_NAME} optimization on MI355X. \
RESULT_DIR=${RESULT_DIR}. \
Monitor the work queue at ${RESULT_DIR}/kernel_manager/work_queue.jsonl for new kernel targets. \
Write results to ${RESULT_DIR}/kernel_manager/results.jsonl. \
Write merge-ready patches to ${RESULT_DIR}/kernel_manager/merge_ready/. \
Write events to ${RESULT_DIR}/kernel_manager/event_log.jsonl on crashes and failures. \
Check ${RESULT_DIR}/kernel_manager/findings.jsonl for Watchdog RCA guidance between OOB rounds. \
Begin monitoring now."

LOG_DIR="$RESULT_DIR/logs"
mkdir -p "$LOG_DIR"
FILTER="$SCRIPT_DIR/log_filter.py"

tmux new-session -d -s "$SESSION_NAME" -n watchdog -x 220 -y 50

CLAUDE_FLAGS="--permission-mode acceptEdits --add-dir /shared_nfs/nehaprakriya --allowed-tools \"Bash(*),Read,Write,Edit,Glob,Grep\" --output-format stream-json --verbose"

tmux send-keys -t "${SESSION_NAME}:watchdog" \
    "export RESULT_DIR='${RESULT_DIR}' && export BASE_DIR='${BASE_DIR}' && export MODEL_NAME='${MODEL_NAME}' && claude ${CLAUDE_FLAGS} -p '${WATCHDOG_PROMPT}' 2>&1 | python3 ${FILTER} | tee ${LOG_DIR}/watchdog.log" Enter

tmux new-window -t "$SESSION_NAME" -n orchestrator

tmux send-keys -t "${SESSION_NAME}:orchestrator" \
    "export RESULT_DIR='${RESULT_DIR}' && export BASE_DIR='${BASE_DIR}' && export MODEL_NAME='${MODEL_NAME}' && claude ${CLAUDE_FLAGS} -p '${ORCHESTRATOR_PROMPT}' 2>&1 | python3 ${FILTER} | tee ${LOG_DIR}/orchestrator.log" Enter

tmux new-window -t "$SESSION_NAME" -n kernel-mgr

tmux send-keys -t "${SESSION_NAME}:kernel-mgr" \
    "export RESULT_DIR='${RESULT_DIR}' && export BASE_DIR='${BASE_DIR}' && export MODEL_NAME='${MODEL_NAME}' && claude ${CLAUDE_FLAGS} -p '${MANAGER_PROMPT}' 2>&1 | python3 ${FILTER} | tee ${LOG_DIR}/kernel-mgr.log" Enter

echo "================================================"
echo "  Marathon session: $SESSION_NAME"
echo "  Window 0 (watchdog):     ${SESSION_NAME}:watchdog"
echo "  Window 1 (orchestrator): ${SESSION_NAME}:orchestrator"
echo "  Window 2 (kernel-mgr):   ${SESSION_NAME}:kernel-mgr"
echo ""
echo "  Base dir (read):   $BASE_DIR"
echo "  Session dir (write): $RESULT_DIR"
echo ""
echo "  Logs:"
echo "    tail -f $LOG_DIR/orchestrator.log"
echo "    tail -f $LOG_DIR/watchdog.log"
echo "    tail -f $LOG_DIR/kernel-mgr.log"
echo ""
echo "  Work queue:   $RESULT_DIR/kernel_manager/work_queue.jsonl"
echo "  Results:      $RESULT_DIR/kernel_manager/results.jsonl"
echo "  Patches:      $RESULT_DIR/kernel_manager/merge_ready/"
echo "  Event log:    $RESULT_DIR/kernel_manager/event_log.jsonl"
echo "  Findings:     $RESULT_DIR/kernel_manager/findings.jsonl"
echo "  RCA reports:  $RESULT_DIR/kernel_manager/rca_reports/"
echo ""
echo "  Attach:  tmux attach -t $SESSION_NAME"
echo "  Switch:  Ctrl-b n  (next window)"
echo "  Detach:  Ctrl-b d"
echo "================================================"
