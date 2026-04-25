# Marathon Launcher (3-Pane tmux)

Runs the marathon inference optimization as 3 independent `claude` CLI agents
in tmux panes (orchestrator, kernel-manager, watchdog), coordinating via JSONL
files on shared NFS. Based on Hyperloom PR #83.

The Python marathon harness (`marathon_harness/`) is **not used** — the `claude`
agents read the protocol specs from `SPEC_ROOT` directly.

## Prerequisites

- `tmux`, `jq` — `apt-get install -y tmux jq`
- Node.js >= 18 — `curl -fsSL https://deb.nodesource.com/setup_22.x | bash - && apt-get install -y nodejs`
- Claude CLI — `npm install -g @anthropic-ai/claude-code`
- Primus-Claw auth proxy running locally (routes LLM calls to SAFE LLM proxy)

`run.sh` auto-installs missing deps in local mode, but it's faster to have them
pre-installed.

## Step 1: Verify Primus-Claw is healthy

```bash
# Backend + executor (expect JSON with "status":"ok")
curl -s http://localhost:8000/health | python3 -c "import json,sys; print(json.load(sys.stdin))" 2>/dev/null && echo "Backend: OK" || echo "Backend: DOWN"
curl -s http://localhost:8100/health | python3 -c "import json,sys; print(json.load(sys.stdin))" 2>/dev/null && echo "Executor: OK" || echo "Executor: DOWN"

# Auth proxy (transparent proxy — no /health endpoint; test the LLM path instead)
curl -s -H "Authorization: Bearer ${ANTHROPIC_AUTH_TOKEN}" \
  http://127.0.0.1:4002/api/v1/llm-proxy/v1/models \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'Auth proxy: OK ({len(d[\"data\"])} models)')" \
  2>/dev/null || echo "Auth proxy: DOWN"
```

If any are down, restart:

```bash
# Restart all of Primus-Claw (backend + executor + auth proxy)
nohup bash /shared_nfs/nehaprakriya/TBO/inference_optimization/marathon_harness/claw_watchdog.sh \
  > /tmp/claw_watchdog.log 2>&1 &

# Or restart just the auth proxy
cd /shared_nfs/nehaprakriya/Primus-Claw/OOB
nohup python3 auth_proxy.py &> /tmp/auth_proxy.log &
```

## Step 2: Set environment variables

```bash
# --- Required: change these per run ---
export MODEL_NAME="DeepSeek-R1-0528"
export BASE_DIR="/shared_nfs/nehaprakriya/Agentic-InferenceX/DeepSeek-R1-0528-optimized"
export MODEL_PATH="/hyperloom/models/DeepSeek-R1-0528"
export MAX_HOURS=4                  # wall-clock budget in hours

# --- Auth (from Primus-Claw, should not need changing) ---
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN}"
export ANTHROPIC_BASE_URL="http://127.0.0.1:4002/api/v1/llm-proxy"
export SAFE_API_KEY="${SAFE_API_KEY}"

# --- Hardware / workload (defaults shown, override as needed) ---
export FRAMEWORK=sglang             # sglang | vllm
export MODEL_CLASS=moe_mla          # dense | moe_mla | moe_swa | moe_mla_nsa
export GPU_COUNT=8
export GPU_TYPE=MI355X
export TP=8
export EP=1
export PRECISION=fp8
export CONC=64
export ISL=1024
export OSL=1024
export INFERENCEX_PATH=/hyperloom/InferenceX

# --- Kernel optimization backends ---
export IMAGE="harbor.oci-slc.example-internal-host.invalid/custom/lmsysorg/sglang:202603270958"
export KERNEL_OPT_BACKENDS=geak,claude,codex
# If IMAGE is empty, GEAK backend is skipped (claude/codex still work)
```

## Step 3: Launch

```bash
bash /shared_nfs/nehaprakriya/TBO/inference_optimization/marathon_launcher/scripts/launcher/run.sh \
  > /tmp/marathon.log 2>&1 &
echo "PID=$!"
```

## Step 4: Monitor

```bash
# Live progress (printed every 60s by the built-in monitor)
tail -f /tmp/marathon.log

# Attach to tmux to watch the panes directly
tmux attach -t marathon
# Ctrl-B N = next pane, Ctrl-B D = detach without stopping

# Quick status check
SESSION_DIR=$(grep "SESSION_DIR=" /tmp/marathon.log | head -1 | cut -d= -f2)
jq '{phase, tput: .current_tput_per_gpu, gain: .cumulative_gain_pct,
     completed: (.completed_actions | length), stack: (.action_stack | length)}' \
  "$SESSION_DIR/state.json"
```

## Step 5: Early stop (if needed)

```bash
SESSION_DIR=$(grep "SESSION_DIR=" /tmp/marathon.log | head -1 | cut -d= -f2)

# Graceful: panes write SESSION_REPORT.md then exit
touch "$SESSION_DIR/STOP_PANE_watchdog" \
      "$SESSION_DIR/STOP_PANE_orchestrator" \
      "$SESSION_DIR/STOP_PANE_kernel-mgr"

# Or kill the background process (cleanup trap handles the rest)
kill %1
```

## Step 6: Read the final report

```bash
SESSION_DIR=$(grep "SESSION_DIR=" /tmp/marathon.log | head -1 | cut -d= -f2)
cat "$SESSION_DIR/SESSION_REPORT.md"
```

## Where things live

```
marathon_launcher/              ← this directory (thin launcher)
├── README.md
├── SKILL.md                    ← skill spec for the calling agent
└── scripts/launcher/
    ├── run.sh                  ← main entrypoint
    ├── pane_orchestrator.md    ← system prompt for orchestrator claude CLI
    ├── pane_kernel_mgr.md      ← system prompt for kernel-mgr claude CLI
    └── pane_watchdog.md        ← system prompt for watchdog claude CLI

SPEC_ROOT (referenced, not copied):
  /hyperloom/Hyperloom/marathon_optimization/marathon_harness/skills/
  ├── SKILL.md                  ← the full DFS protocol (Neha's source of truth)
  ├── KNOWLEDGE-BASE.md
  ├── actions/                  ← per-action execution specs
  ├── kernel-manager/SKILL.md
  ├── watchdog/SKILL.md
  └── ...

Session output:
  /shared_nfs/nehaprakriya/marathon-sessions/<timestamp>/
  ├── state.json
  ├── SESSION_REPORT.md
  ├── kernel_manager/
  │   ├── work_queue.jsonl      (orchestrator → kernel-mgr)
  │   ├── results.jsonl         (kernel-mgr → orchestrator)
  │   ├── event_log.jsonl       (all → watchdog)
  │   ├── findings.jsonl        (watchdog → orchestrator)
  │   └── merge_ready/<id>/
  └── logs/{orchestrator,kernel-mgr,watchdog}.log
```

## What changed from the Python harness

The Python files (`orchestrator.py`, `kernel_manager.py`, `watchdog.py`,
`state.py`, `llm.py`, `gpu_lock.py`, etc.) are replaced by 3 `claude` CLI
agents that read the SKILL.md protocol specs directly. Key differences:

- **3 separate context windows** instead of 1 shared process (survives 24h runs)
- **Auto-restart** via `--continue` loop (up to 50 restarts per pane)
- **No Python scoring** — agents follow the scoring formula from SKILL.md heuristically
- **No asyncio GPU lock** — enforced by prompt constraints (orchestrator owns server lifecycle)
- **LLM calls** route through Primus-Claw auth proxy → SAFE LLM proxy (not claude_code_sdk)
- **MCP tools** (GEAK, OOB, TraceLens) connect to remote OCI endpoints via `--mcp-config`
