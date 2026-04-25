#!/usr/bin/env bash
# Marathon launch wrapper — edit the vars below, then: bash launch.sh
set -euo pipefail

# ── Required: change these per run ──────────────────────────────────────────
export MODEL_NAME="gpt-oss-120b"
export BASE_DIR="/shared_nfs/nehaprakriya/Agentic-InferenceX/gpt-oss-120b-marathon-optimized"
export MODEL_PATH="/shared_nfs/models/gpt-oss-120b"
export MAX_HOURS=24

# ── Auth (from Primus-Claw .env) ────────────────────────────────────────────
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN}"
export ANTHROPIC_BASE_URL="http://127.0.0.1:4002/api/v1/llm-proxy"
export SAFE_API_KEY="${SAFE_API_KEY}"

# ── Hardware / workload ─────────────────────────────────────────────────────
export FRAMEWORK=vllm
export MODEL_CLASS=moe_swa
export GPU_COUNT=8
export GPU_TYPE=MI300X
export TP=1
export EP=1
export PRECISION=fp8
export CONC=64
export ISL=1024
export OSL=1024
export INFERENCEX_PATH=/hyperloom/InferenceX

# ── Kernel optimization backends ────────────────────────────────────────────
export IMAGE=""
export KERNEL_OPT_BACKENDS=claude,codex

# ── Preflight checks ───────────────────────────────────────────────────────
echo "Checking Primus-Claw..."
curl -sf http://localhost:8000/health > /dev/null 2>&1 || { echo "ERROR: Claw backend (:8000) is down"; exit 1; }
curl -sf http://localhost:8100/health > /dev/null 2>&1 || { echo "ERROR: Claw executor (:8100) is down"; exit 1; }
curl -s -H "Authorization: Bearer $SAFE_API_KEY" \
  http://127.0.0.1:4002/api/v1/llm-proxy/v1/models 2>/dev/null \
  | python3 -c "import json,sys; json.load(sys.stdin)['data']" > /dev/null 2>&1 \
  || { echo "ERROR: Auth proxy (:4002) LLM path not working"; exit 1; }
echo "  All healthy."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="/tmp/marathon.log"

echo ""
echo "  Model:     $MODEL_NAME"
echo "  Base dir:  $BASE_DIR"
echo "  Budget:    ${MAX_HOURS}h"
echo "  Log:       $LOG"
echo ""

bash "$SCRIPT_DIR/launcher/run.sh" > "$LOG" 2>&1 &
PID=$!
echo "Marathon launched (PID=$PID)"
echo ""
echo "  Monitor:   tail -f $LOG"
echo "  Attach:    tmux attach -t marathon"
echo "  Stop:      kill $PID"
echo ""
