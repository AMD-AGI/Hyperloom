# Fully-Local Mode — Complete Execution Reference

This document contains ALL fully-local-mode-specific instructions. Read this **before starting**
when running on a single machine with GPU access inside the Hyperloom Docker container.

**Agent:** Read `SKILL.md` for the orchestrator loop and shared Iron Rules (IR-1 through
IR-7). This file defines fully-local-specific Iron Rules (IR-12 through IR-14), constants,
architecture, and per-action execution overrides.

## Environment

- **Client**: Cursor IDE (remote SSH into container)
- **Runtime**: Local GPU machine (single node, Docker container)
- **GEAK**: CLI (`geak` command) scheduled via Ray (`geak_ray_submit.py`)
- **OOB**: CLI (`oob_ray_submit.py run`) — installed by `pip install -e /opt/oob-mcp/agent_mcp_server`
- **TraceLens**: Local CLI (`pip install -e /opt/TraceLens`)
- **Storage**: Local disk within container (`/tmp/geak-data`, `/opt/hyperloom`, `${OOB_HOME:-~/.oob}`)
- **Ray**: Local head node for GPU task scheduling (`:6379`, dashboard `:8265`)
- **MCP**: NONE — bundled `mcp.json` is intentionally empty; all tooling is invoked as in-container CLIs

## Mode Detection

Selected explicitly by `MODE=fully-local`:

```bash
if [ "${MODE:-}" = "fully-local" ]; then
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-/opt/hyperloom}"
    GEAK_CLI="python3 $SKILL_ROOT/scripts/geak_ray_submit.py"
    OOB_CLI="${OOB_CLI:-oob}"
fi
```

---

## Fully-Local Iron Rules

### IR-12: SaFE MCP is FORBIDDEN

**Do NOT call any SaFE MCP tool.** This includes `workload_create`, `workload_get`,
`workload_stop`, and any other SaFE MCP operation. No RayJobs, no PyTorchJobs.

Violation = immediate run invalidation.

### IR-13: GEAK uses Ray submit, not MCP or REST API

**Do NOT call any `geak_*` MCP tools** (`geak_create_task`, `geak_submit_task`, etc.).
**Do NOT use `geak_client.py`** (REST API client).

All GEAK tasks go through `geak_ray_submit.py` which schedules the `geak` CLI via the local
Ray cluster. Ray handles GPU allocation automatically.

### IR-14: No persistent GEAK service

The GEAK REST API server (`python -m server.main`) does NOT run in fully-local mode.
GEAK is invoked as a CLI tool (`geak`) per-task. Do NOT attempt to start,
configure, or communicate with a GEAK REST API server.

### IR-15: OOB is CLI-only — no MCP, no REST

The OOB MCP server and `oob_client.py` REST flow are GONE in fully-local mode.

**Do NOT call any `agent_*` MCP tools** (`agent_create_task`, `agent_submit_task`,
`agent_get_task`, `agent_get_outputs`, `agent_download_file`, `agent_cancel_task`).
**Do NOT use `oob_client.py`** or `curl http://localhost:8003/...`.

All Codex / Claude kernel-optimization tasks go through a single blocking
`oob_ray_submit.py run -a <agent> ...` invocation (see "OOB in Fully-Local Mode" below).
The 5-step REST flow (create → submit → poll → list outputs → download)
collapses into one CLI call that writes results to the local task workspace.

---

## Architecture

```
Cursor IDE (SSH) --> Skill (SKILL.md)
                      |-> geak_ray_submit.py batch -t kernel_a.md -t kernel_b.md ... --yolo
                      |     |-> Ray head (localhost:6379) schedules tasks
                      |     |-> Each task: geak -t <task>.md --gpu-ids <ray-assigned> --yolo
                      |     \-> Results returned when all tasks complete
                      |
                      |-> oob_ray_submit.py run -a {claude,codex} -p "..." -f kernel.py -o <work_dir>
                      |     |-> Spawns the claude/codex CLI as a subprocess
                      |     |-> Blocks until the agent task finishes (single call)
                      |     \-> Optimized files land in <work_dir>/<task_id>/output/
                      |
                      |-> TraceLens CLI (local, pip install -e)
                      |     \-> Offline trace analysis, no persistent service
                      |
                      |-> Benchmark scripts (direct shell, no exec_on_gpu)
                      |     \-> run_baseline.sh, run_profile.sh, etc.
                      |
                      \-> Results on local disk
```

---

## Key Differences from Other Modes

| Aspect | Fully-Local | Local (LOCAL.md) | Claw (CLAW.md) |
|--------|-------------|------------------|----------------|
| GEAK invocation | `geak_ray_submit.py` (Ray + geak CLI) | GEAK MCP tools | `geak_client.py` (REST API) |
| GEAK service | None (CLI per-task) | MCP server (persistent) | Remote REST API |
| GPU scheduling | Ray local cluster | GEAK MCP handles it | SaFE cluster scheduler |
| OOB | `oob_ray_submit.py run` CLI (per-task subprocess) | OOB MCP tools | `oob_client.py` (REST) |
| OOB service | None (CLI per-task) | MCP server (persistent) | Remote REST API |
| TraceLens | CLI (offline) | CLI + MCP server | CLI (offline) |
| Shell commands | Direct | Direct | `exec_on_gpu` wrapper |
| RayJob | None | None | SaFE-managed |

---

## GEAK in Fully-Local Mode

### Configuration (auto-rendered at container start)

`entrypoint.sh` renders `/opt/hyperloom/geak-config/local.yaml` from `template.yaml`
using these env vars (set via `docker run -e ...`):

| Env var | Default | Notes |
|---|---|---|
| `GEAK_MODEL_NAME` | `claude-opus-4-7` | LiteLLM-format model name |
| `GEAK_API_KEY` | falls back to `LLM_API_KEY` | LLM gateway key |
| `GEAK_BASE_URL` | falls back to `LLM_API_BASE` | OpenAI-compatible endpoint |
| `GEAK_CONFIG` | `/opt/hyperloom/geak-config/local.yaml` | Rendered config path |

`geak_ray_submit.py` auto-injects `--config $GEAK_CONFIG` into every `geak` invocation,
so users do not need to manage GEAK config manually.

### Task file preparation

Before submitting to GEAK, create a task markdown file per kernel candidate:

```bash
TASK_DIR="$RESULT_DIR/geak_tasks"
mkdir -p "$TASK_DIR"

cat > "$TASK_DIR/kernel_${KERNEL_NAME}.md" <<'EOF'
# Task: Optimize ${KERNEL_NAME}

## Kernel Source
<paste full kernel source here>

## Optimization Instructions
Optimize this Triton kernel for AMD MI355X (gfx950, CDNA4).
Hardware: 304 CUs, 256 VGPR/CU, HBM3e ~8 TB/s, MFMA instructions.
Use homogeneous mode. Set max_rounds to 1.
The kernel MUST be optimized to at least 1.5x speedup.
Do NOT search the filesystem with find / or grep -r /.

## Config
kernel_url: /path/to/kernel.py
repo: /path/to/repo
config: default
EOF
```

### Single kernel submission

```bash
$GEAK_CLI run -t "$TASK_DIR/kernel_${KERNEL_NAME}.md" --yolo
```

### Batch submission (IR-1: all candidates in parallel)

```bash
$GEAK_CLI batch \
  -t "$TASK_DIR/kernel_a.md" \
  -t "$TASK_DIR/kernel_b.md" \
  -t "$TASK_DIR/kernel_c.md" \
  -t "$TASK_DIR/kernel_d.md" \
  -t "$TASK_DIR/kernel_e.md" \
  --yolo
```

Ray automatically assigns 1 GPU per task. If there are more tasks than GPUs, excess
tasks queue and start as GPUs become free.

### Checking cluster resources

```bash
$GEAK_CLI status
# Shows: GPU usage (used/total), CPU, memory
```

### GEAK output location

`geak` writes results to its `--output-dir` (defaults to `optimization_logs/` in cwd).
Parse the stdout for the output directory path.

---

## OOB in Fully-Local Mode

OOB has **no persistent service**. The skill calls the `oob` CLI directly per task —
one blocking command replaces the previous create→submit→poll→list→download flow.

### Single-shot invocation

```bash
OOB_CLI="${OOB_CLI:-oob}"  # entrypoint exports this; default is `oob`

# Codex (one round)
$OOB_CLI run \
  -a codex \
  -p "Optimize this Triton kernel ... (see prompt template in kernel-opt/codex.md)" \
  -f $WORK_DIR/kernel.py \
  -o $WORK_DIR/oob_codex_${KERNEL_NAME} \
  --max-turns 20 \
  --timeout 1800 \
  --no-live --json

# Claude (one round)
$OOB_CLI run \
  -a claude \
  -p "Optimize this Triton kernel ... (see prompt template in kernel-opt/claude.md)" \
  -f $WORK_DIR/kernel.py \
  -o $WORK_DIR/oob_claude_${KERNEL_NAME} \
  --max-turns 30 \
  --timeout 1800 \
  --no-live --json
```

`oob_ray_submit.py run` blocks until the agent task reaches a terminal status, prints the final
result JSON to stdout (with `--json`), and exits 0 on `completed`, 1 on `failed`,
130 on Ctrl-C. Output files land at:

```
<output-dir>/tasks/<user>/<task_id>/workspace/optimized_kernel.py    # main result
<output-dir>/tasks/<user>/<task_id>/workspace/execution.log         # subprocess stdout/stderr
<output-dir>/tasks/<user>/<task_id>/workspace/trajectory_round_*.jsonl  # tool-call trajectory

# Use the JSON `.workspace` field to get the workspace dir directly — there is
# no separate `output/` subdir; the agent writes straight into `workspace/`.
```

Parse the `--json` payload to get `task_id`, `status`, `workspace`, and `error_message`:

```bash
RESULT_JSON=$($OOB_CLI run -a codex -p "$PROMPT" -f kernel.py -o $WORK_DIR/codex --json --no-live)
TASK_ID=$(echo "$RESULT_JSON" | jq -r .task_id)
STATUS=$(echo "$RESULT_JSON"  | jq -r .status)
WORKSPACE=$(echo "$RESULT_JSON" | jq -r .workspace)
[ "$STATUS" = "completed" ] && cp "$WORKSPACE/optimized_kernel.py" "$RESULT_DIR/"
```

### Iterative refinement loop (codex / claude only)

`OOB_ROUND_ITERATIONS` (default 3) iterations per round, each one is a separate
`oob_ray_submit.py run` invocation. The skill is responsible for stitching them: append the
previous iteration's verified speedup / compile error into the next prompt.
The original kernel source is always passed via `-f kernel.py`; **never** pass
a previous iteration's output as the input.

See [`../kernel-opt/codex.md`](../kernel-opt/codex.md) and
[`../kernel-opt/claude.md`](../kernel-opt/claude.md) "Fully-Local Execution"
sections for the full pseudocode.

### Cancellation

Send `SIGINT` / `SIGTERM` to the `oob` process — it will request graceful
cancellation of the underlying agent task and return rc=130.

---

## TraceLens in Fully-Local Mode

TraceLens is a CLI tool with no persistent service. Use directly:

```bash
# Generate performance report
TraceLens_generate_perf_report_pytorch_inference \
  --trace-file "$TRACE_DIR/$FILTERED_TRACE_NAME" \
  --output-dir "$RESULT_DIR/tracelens"

# Other TraceLens CLI commands available via pip install
```

---

## Per-Action Fully-Local Overrides

### Setup (`actions/setup.md`)

No RayJob creation. Environment is pre-configured by the container:

```bash
export MODE=fully-local
export FRAMEWORK=${FRAMEWORK:-sglang}
export INFERENCEX_PATH=${INFERENCEX_PATH:-/opt/hyperloom/InferenceX}
export SKILL_ROOT=/opt/hyperloom/.cursor/skills/inference-optimization
export SCRIPTS_DIR="$SKILL_ROOT/scripts"
export GEAK_CLI="python3 $SCRIPTS_DIR/geak_ray_submit.py"
export OOB_CLI="${OOB_CLI:-oob}"

TIMESTAMP=$(date +%Y-%m-%d-%H-%M)
export RESULT_DIR="/tmp/inference-optimization/results/${TIMESTAMP}"
export TRACE_DIR="/tmp/inference-optimization/traces/${TIMESTAMP}"
mkdir -p "$RESULT_DIR" "$TRACE_DIR"

source "$SCRIPTS_DIR/common.sh"
```

### Baseline (`actions/baseline.md`)

All commands run directly in shell (no `exec_on_gpu`):

```bash
bash "$SCRIPTS_DIR/run_baseline.sh"
```

### Profile (`actions/profile.md`)

```bash
bash "$SCRIPTS_DIR/run_profile.sh"
```

Trace files written to `$TRACE_DIR`. TraceLens analyzes locally:

```bash
TraceLens_generate_perf_report_pytorch_inference \
  --trace-file "$TRACE_DIR/$FILTERED_TRACE_NAME" \
  --output-dir "$RESULT_DIR/tracelens"
```

### Kernel Optimization (`actions/kernel-opt.md`)

**GEAK backend:** Use `geak_ray_submit.py` (see "GEAK in Fully-Local Mode" above).

**Codex / Claude backend:** Use `oob_ray_submit.py run` CLI (see "OOB in Fully-Local Mode" above).

**LLM backend:** Direct OpenAI API call (same as other modes).

All backends launch **concurrently** per IR-1.

### Integrate (`actions/integrate.md`)

`patch_inductor.py` operates on local Inductor cache:

```bash
python3 $SCRIPTS_DIR/patch_inductor.py patch \
    --kernel-name $KERNEL_NAME \
    --geak-file $GEAK_OUTPUT_PATH \
    --target-file $STANDALONE_FILE_PATH \
    --best-config '{"XBLOCK": 4, "R0_BLOCK": 2048, "num_warps": 4}'
```

### Sweep (`actions/sweep.md`)

Serial sweep via `run_sweep.sh` (no SaFE parallel option):

```bash
bash "$SCRIPTS_DIR/run_sweep.sh"
```

### Report (`actions/report.md`)

No RayJob cleanup. Generate report and ingest to KB.

---

## Health Timeout

After patching with torch.compile, set `HEALTH_TIMEOUT=1800` (30 min) to allow full
recompilation before health check times out.

## Safe Process Management

Same as `SKILL.md` IR-5. **NEVER use `pkill -f sglang`**. Use targeted kill:

```bash
kill $(pgrep -f 'python.*-m sglang.launch_server') 2>/dev/null
kill $(pgrep -f 'python.*-m vllm.entrypoints') 2>/dev/null
```
