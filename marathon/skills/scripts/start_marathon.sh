#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# Pre-launch setup + Marathon launch
#
# Usage:  bash start_marathon.sh <model_name> <result_dir> [skill_root]
#
# Example:
#   bash start_marathon.sh deepseek-r1 /shared_nfs/nehaprakriya/Agentic-InferenceX/DeepSeek-R1-0528-marathon
###############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILL_ROOT="${3:-$(dirname "$(dirname "$SCRIPT_DIR")")}"
LAUNCH_SCRIPT="$SCRIPT_DIR/launch_marathon.sh"

MODEL_NAME="${1:?Usage: $0 <model_name> <base_dir> [skill_root]}"
BASE_DIR="${2:?Usage: $0 <model_name> <base_dir> [skill_root]}"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SESSION_DIR="$BASE_DIR/sessions/$TIMESTAMP"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Marathon Pre-Launch: ${MODEL_NAME} on MI355X${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════${NC}"
echo ""
echo -e "  Base dir (read-only):  $BASE_DIR"
echo -e "  Session dir (writes):  $SESSION_DIR"
echo ""

# ── 1. Source API keys ──────────────────────────────────────────────────────
echo -e "${YELLOW}[1/7]${NC} Sourcing API keys from TBO/.env ..."
TBO_ENV="$(cd "$SKILL_ROOT/../.." && pwd)/.env"
if [ -f "$TBO_ENV" ]; then
    set -a
    source "$TBO_ENV"
    set +a
    echo "  GEAK_AUTH_KEY=${GEAK_AUTH_KEY:0:10}...  ✓"
    echo "  LITELLM_API_KEY=${LITELLM_API_KEY:0:10}...  ✓"
else
    echo -e "${RED}  ERROR: $TBO_ENV not found${NC}"
    exit 1
fi

# ── 2. Verify claude CLI ───────────────────────────────────────────────────
echo -e "${YELLOW}[2/7]${NC} Checking claude CLI ..."
if command -v claude &>/dev/null; then
    echo "  $(claude --version 2>&1)"
else
    echo -e "${RED}  ERROR: claude not found in PATH${NC}"
    exit 1
fi

# ── 3. Verify base directory exists (previous run data) ────────────────────
echo -e "${YELLOW}[3/7]${NC} Checking base directory ..."
if [ -d "$BASE_DIR" ]; then
    echo -e "  ${GREEN}✓${NC} $BASE_DIR"
    if [ -x "$BASE_DIR/scripts/manage.sh" ]; then
        echo ""
        "$BASE_DIR/scripts/manage.sh" status || true
        echo ""
    fi
else
    echo -e "${RED}  ERROR: Base directory not found: $BASE_DIR${NC}"
    echo "  The base directory must contain the previous run (exploration tree, results, etc.)"
    exit 1
fi

# ── 4. Create session directory ────────────────────────────────────────────
echo -e "${YELLOW}[4/7]${NC} Creating session directory ..."
mkdir -p "$SESSION_DIR"
ln -sfn "$BASE_DIR" "$SESSION_DIR/base"
echo -e "  ${GREEN}✓${NC} $SESSION_DIR"
echo -e "  ${GREEN}✓${NC} $SESSION_DIR/base → $BASE_DIR"

# ── 5. Verify server is running ────────────────────────────────────────────
echo -e "${YELLOW}[5/7]${NC} Checking inference server ..."
if curl -sf http://0.0.0.0:8888/health >/dev/null 2>&1; then
    echo -e "  ${GREEN}Server is UP at http://0.0.0.0:8888${NC}"
else
    echo -e "  ${YELLOW}WARNING: Server not responding at http://0.0.0.0:8888${NC}"
    echo "  The orchestrator will need to start it during warm-up."
fi

# ── 6. Create IPC directories and files (in session dir) ───────────────────
echo -e "${YELLOW}[6/7]${NC} Creating IPC files for Watchdog + Kernel Manager ..."
mkdir -p "$SESSION_DIR/kernel_manager/rca_reports"
mkdir -p "$SESSION_DIR/kernel_manager/merge_ready"
touch "$SESSION_DIR/kernel_manager/work_queue.jsonl"
touch "$SESSION_DIR/kernel_manager/results.jsonl"
touch "$SESSION_DIR/kernel_manager/event_log.jsonl"
touch "$SESSION_DIR/kernel_manager/findings.jsonl"
mkdir -p "$SESSION_DIR/results"
echo "  $SESSION_DIR/kernel_manager/"
ls -1 "$SESSION_DIR/kernel_manager/"

# ── 7. Verify skill files exist ───────────────────────────────────────────
echo -e "${YELLOW}[7/7]${NC} Verifying skill files ..."
FAIL=0
for f in \
    "$SKILL_ROOT/marathon-inference-optimization/SKILL.md" \
    "$SKILL_ROOT/marathon-inference-optimization/kernel-manager/SKILL.md" \
    "$SKILL_ROOT/marathon-inference-optimization/watchdog/SKILL.md" \
    "$LAUNCH_SCRIPT"; do
    if [ -f "$f" ]; then
        echo -e "  ${GREEN}✓${NC} $(basename "$(dirname "$f")")/$(basename "$f")"
    else
        echo -e "  ${RED}✗${NC} $f"
        FAIL=1
    fi
done
RCA_SKILL="/shared_nfs/nehaprakriya/agentic-rc/.cursor/skills/training-workload-rca/SKILL.md"
if [ -f "$RCA_SKILL" ]; then
    echo -e "  ${GREEN}✓${NC} training-workload-rca/SKILL.md"
else
    echo -e "  ${YELLOW}~${NC} training-workload-rca/SKILL.md (optional, Watchdog RCA limited)"
fi
if [ "$FAIL" -eq 1 ]; then
    echo -e "${RED}  Missing required skill files. Aborting.${NC}"
    exit 1
fi

# ── All checks passed — launch ─────────────────────────────────────────────
echo ""
echo -e "${GREEN}All pre-checks passed. Launching Marathon ...${NC}"
echo -e "  Base (read):    $BASE_DIR"
echo -e "  Session (write): $SESSION_DIR"
echo ""

export MARATHON_BASE_DIR="$BASE_DIR"
exec bash "$LAUNCH_SCRIPT" "$MODEL_NAME" "$SESSION_DIR" "$SKILL_ROOT"
