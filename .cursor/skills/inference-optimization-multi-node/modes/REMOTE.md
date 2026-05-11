# Remote Mode — Complete Execution Reference

This document contains ALL remote-mode-specific instructions. Read this **before starting**
when using remote client with SaFE cluster.

**Agent:** Read `SKILL.md` for the orchestrator loop and shared Iron Rules (IR-1 through
IR-7b). This file defines remote-specific Iron Rules (IR-8 through IR-11), constants,
architecture, and per-action execution overrides.

## Environment

- **Client**: Remote client (external orchestration environment)
- **Runtime**: SaFE cluster with multi-node GPU
- **MCP Servers**: SaFE MCP only (RayJob lifecycle)
- **OOB**: CLI (`oob_ray_submit.py run`) for Codex/Claude optimization tasks
- **TraceLens**: Local CLI (`pip install -e /hyperloom/TraceLens-internal`)
- **Storage**: Shared NFS (`/wekafs/` inside Pod, maps to NFS root)

## Mode Detection

Auto-detected when remote client context is present, or user specifies `Mode: remote`.

```bash
MODE="remote"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/wekafs/inference-optimization}"
```

---

## Remote-Mode Iron Rules

### IR-8: Use Ray Dashboard REST for ALL GPU-side commands

ALL commands that run on the Ray cluster MUST be submitted through Ray Dashboard
REST (`POST /api/jobs/`). **NEVER** drive the cluster from the sandbox with Ray Client,
`ray://<head>:10001`, or any custom Ray client submit wrapper.

```bash
# CORRECT
curl -sS -X POST "$RAY_DASHBOARD_URL/api/jobs/" \
  -H "Content-Type: application/json" \
  -d '{"entrypoint":"bash -lc '\''source /etc/profile.d/hyperloom-env.sh && bash $SCRIPTS_DIR/run_baseline.sh'\''"}'

# WRONG — sandbox-side Ray client or custom Ray submit wrapper
```

### IR-9: Main inference workload MUST use `kind: "RayJob"`

The persistent inference cluster **MUST** be `kind: "RayJob"`. Kernel optimization
backends run through the OOB CLI wrapper (`oob_ray_submit.py`) on the RayJob.
The skill itself MUST NOT create PyTorchJob workloads.

### IR-10: SaFE MCP — ONLY `workload_create(kind="RayJob")` and `workload_stop`

In Remote mode, the skill may use SaFE MCP **only** for:

- **`workload_create`** with `kind: "RayJob"` — to create the inference cluster
- **`workload_get`** / **`workload_list`** — to check workload status
- **`workload_stop`** — to stop the RayJob after optimization is complete

**FORBIDDEN SaFE MCP operations:**

- **`workload_delete`** — NEVER delete workloads; use `workload_stop` instead
- **`workload_create` with any kind other than `"RayJob"`** — no PyTorchJob, no other
  types. Kernel optimization is handled by the OOB CLI wrapper on Ray; the skill
  MUST NOT create other workload types directly.

Violation = immediate run invalidation.

### IR-10a: Create a new RayJob; do not use an existing RayJob

At the start of each skill execution, create a fresh RayJob with `workload_create`.
Do NOT attach to, resume, or reuse an existing RayJob from a previous run, even if it
appears healthy; stale state invalidates optimization results.

### IR-11: OOB is CLI-only — no service API

Same as IR-7 in `SKILL.md` — NEVER modify OOB configuration, auth files, test data,
or settings. Use only the approved CLI tools:

- OOB: `oob_ray_submit.py run`
- TraceLens: `TraceLens_generate_perf_report_pytorch_inference` and
  `orchestrator_prepare.py`

---

## Remote-Mode Constants

These supplement the shared constants in `SKILL.md`.

| Constant | Value | Description |
|----------|-------|-------------|
| `RAY_DASHBOARD_PORT` | 8265 | Ray Dashboard REST port on RayJob head node |

### Image Selection (Remote Mode)

Use `KERNEL_OPT_IMAGE` (provided by CI or user). In remote mode, CI should supply the
Ray-patched image for SGLang (e.g., `harbor.../custom/lmsysorg/sglang:202603270958` with
Ray 2.44.1 fix). The same image is used for both the RayJob and kernel-opt backends.

### Image Build (Dockerfile)

The custom SGLang image is based on upstream SGLang with Ray compatibility fixes:

```dockerfile
FROM harbor.oci-slc.primus-safe.amd.com/proxy/lmsysorg/sglang:v0.5.9-rocm700-mi35x
RUN python -m pip install ray[default]==2.44.1 click==8.1.7
```

**Why:** Upstream SGLang image ships a Ray version with a `copy.deepcopy` / `Sentinel` enum
crash bug that breaks RayJob's `ray start` and `ray-job-submitter`. `click==8.1.7` pins
click to avoid API incompatibilities with Ray CLI.

---


## RayJob Architecture

```
Remote Client --> Skill (SKILL.md)
                 |-> SaFE MCP (workload_create kind="RayJob")
                 |     |-> Creates Ray Cluster: head (1 pod) + workers (N pods)
                 |     |-> Exposes Ray Dashboard REST port (8265)
                 |     \-> env.RAY_JOB_ENTRYPOINT = "tail -f /dev/null" (keeps cluster alive)
                 |
                 |-> Ray Dashboard REST
                 |     \-> POST http://<head>:8265/api/jobs/ -> run inside RayJob image
                 |
                 |-> oob_ray_submit.py run -a {claude,codex} -p "..." -f kernel.py -o <work_dir>
                 |     |-> Ray head schedules task with GPU isolation
                 |     |-> Worker spawns claude/codex CLI as subprocess, blocks until done
                 |     \-> Optimized files land in <work_dir>/<task_id>/workspace/
                 |
                 |-> TraceLens CLI
                 |     \-> Offline trace analysis from shared NFS
                 |
                 \-> Benchmark results on shared NFS
```

---

## RayJob Lifecycle

### Create a NEW RayJob

**Exactly ONE RayJob per skill execution.** Create a new RayJob at the start and use it
for the entire run. Do NOT create a second one mid-execution. Do NOT reuse RayJobs from
previous skill executions (they may have stale state). After optimization is complete,
`workload_stop` the RayJob.

> **CRITICAL: `RAY_JOB_ENTRYPOINT` must be base64-encoded.** Plain text causes
> `base64: invalid input` → exit code 127.
> Base64 encoding: `echo -n "tail -f /dev/null" | base64` → `dGFpbCAtZiAvZGV2L251bGw=`

**Single node (NUM_NODES=1):**

```
Tool: workload_create
Args: {
    "display_name": "inference-opt-<model_short_name>",
    "workspace_id": "<user_workspace_id>",
    "kind": "RayJob",
    "images": ["KERNEL_OPT_IMAGE"],
    "resources": [
        {"replica": 1, "cpu": "96", "gpu": "8", "memory": "1024Gi", "sharedMemory": "500Gi", "ephemeralStorage": "500Gi"}
    ],
    "env": { "RAY_JOB_ENTRYPOINT": "dGFpbCAtZiAvZGV2L251bGw=" },
    "is_tolerate_all": true,
    "ttl_seconds_after_finished": 600
}
```

**Multi-node (NUM_NODES>1):**

```
Tool: workload_create
Args: {
    "display_name": "inference-opt-<model_short_name>",
    "workspace_id": "<user_workspace_id>",
    "kind": "RayJob",
    "images": ["KERNEL_OPT_IMAGE", "KERNEL_OPT_IMAGE"],
    "resources": [
        {"replica": 1, "cpu": "96", "gpu": "8", "memory": "1024Gi", "sharedMemory": "256Gi", "ephemeralStorage": "500Gi"},
        {"replica": 1, "cpu": "96", "gpu": "8", "memory": "1024Gi", "sharedMemory": "256Gi", "ephemeralStorage": "500Gi"}
    ],
    "env": { "RAY_JOB_ENTRYPOINT": "dGFpbCAtZiAvZGV2L251bGw=" },
    "is_tolerate_all": true,
    "ttl_seconds_after_finished": 7200
}
```

For N>2 nodes, set `resources[1].replica` to `N-1`. TP = GPU_PER_NODE × NUM_NODES.

### Wait for RayJob Ready + get Ray Dashboard address

```python
import time

RAYJOB_ID = "<workload_id from create response>"

for attempt in range(60):
    result = workload_get(workload_id=RAYJOB_ID)
    phase = result.get("status", {}).get("phase", "")
    if phase == "Running":
        break
    elif phase in ("Failed", "Stopped"):
        raise RuntimeError(f"RayJob failed: {phase}")
    time.sleep(30)

pods = result.get("status", {}).get("pods", [])
head_pod = next((p for p in pods if "head" in p.get("name", "")), pods[0] if pods else None)
HEAD_IP = head_pod.get("ip", head_pod.get("podIP", ""))
RAY_DASHBOARD_URL = f"http://{HEAD_IP}:8265"
```

### Set up remote execution environment

```bash
export MODE=remote
export HEAD_IP="<HEAD_IP>"
export RAY_DASHBOARD_URL="http://<HEAD_IP>:8265"
export MODEL="<model_path_on_shared_nfs>"
export TP=<total_gpu_count>
export CONC=<concurrency>
export ISL=1024
export OSL=256
export FRAMEWORK=<sglang|vllm>
export INFERENCEX_PATH="<path_on_shared_nfs>"
export SKILL_ROOT="${SKILL_ROOT:-/wekafs/yunkai/Hyperloom/.cursor/skills/inference-optimization}"
export SCRIPTS_DIR="$SKILL_ROOT/scripts"
export OOB_RAY_CLI="python3 $SCRIPTS_DIR/oob_ray_submit.py"
export OOB_CLI="${OOB_CLI:-/opt/venv/bin/oob}"
export TRACELENS_ROOT="${TRACELENS_ROOT:-/opt/hyperloom/TraceLens}"

TIMESTAMP=$(date +%Y-%m-%d-%H-%M)
export RESULT_DIR="/wekafs/inference-optimization/results/${TIMESTAMP}"
export TRACE_DIR="/wekafs/inference-optimization/traces/${TIMESTAMP}"

# Verify Ray cluster by submitting a short Ray Dashboard REST job.
curl -sS -X POST "$RAY_DASHBOARD_URL/api/jobs/" \
  -H "Content-Type: application/json" \
  -d '{"entrypoint":"bash -lc '\''source /etc/profile.d/hyperloom-env.sh && python3 - <<\"PY\"\nimport ray\nray.init()\nprint(f\"Nodes: {len(ray.nodes())}\")\nprint(ray.cluster_resources())\nray.shutdown()\nPY'\''"}'
```

### Remote Path Contract

Remote mode uses two distinct filesystems with different ownership:

- **RayJob-side runtime outputs MUST use `/wekafs/...` paths.** This includes
  `$RESULT_DIR`, `$TRACE_DIR`, TraceLens output generated inside the RayJob,
  OOB input/output workspaces, kernel candidate files, re-baseline artifacts,
  and sweep outputs. These paths must be pod-visible and shared across the
  RayJob head/worker pods.
- **Sandbox-side generated files MUST use `/workspace/hyperloom/...` paths.**
  The Claw sandbox may read `/wekafs` as input, but it must not write there.
  Any sandbox-created summaries, manifests, temporary prompts, copied metadata,
  or final user-facing artifacts must be written under `/workspace/hyperloom/`.
- **Final deliverables MUST be present under `/workspace/hyperloom/`.** After
  RayJob work finishes, copy or summarize the relevant `/wekafs` RayJob
  artifacts into `/workspace/hyperloom/optimization_report.md`,
  `/workspace/hyperloom/ci_metrics.json` if used, and any supporting result
  summaries/manifests. Claw uploads `/workspace/hyperloom/` to S3 when the
  session ends.

### Bootstrap / CLI preflight inside RayJob

After the RayJob is running, reuse the BYOI bootstrap logic from
`scripts/bootstrap.sh` to ensure CLI dependencies exist inside the RayJob image.
This installs or verifies OOB, TraceLens, Ray/click compatibility, and the
`codex` / `claude` CLIs. The generated environment file is reusable, but
force `MODE=remote` again after sourcing it.

```bash
# Ray Dashboard REST job entrypoint:
export PATH='/opt/venv/bin:'\"\$PATH\"
export MODE=remote
export SKILL_ROOT='$SKILL_ROOT'
export HYPERLOOM_BUNDLE='\${HYPERLOOM_BUNDLE:-/wekafs/fully-local}'

if [ ! -f /opt/hyperloom/.bootstrap_done ] && [ -d \"\$HYPERLOOM_BUNDLE\" ]; then
  bash '$SCRIPTS_DIR/bootstrap.sh'
fi

[ -f /etc/profile.d/hyperloom-env.sh ] && . /etc/profile.d/hyperloom-env.sh
export MODE=remote
export OOB_CLI='\${OOB_CLI:-/opt/venv/bin/oob}'

command -v oob || echo 'WARN: oob CLI missing'
command -v codex || command -v claude || echo 'WARN: codex/claude CLI missing'
command -v TraceLens_generate_perf_report_pytorch_inference || echo 'WARN: TraceLens CLI missing'
"
```

### Cleanup — Stop RayJob

After the optimization is complete and the report is generated:

```
Tool: workload_stop
Args: { "workload_id": "<RAYJOB_ID>" }
```

Also stop any parallel sweep workloads (if SaFE Option B was used):

```python
sweep_workloads = workload_list(workspace_id=WORKSPACE_ID, kind="RayJob")
for wl in sweep_workloads:
    if wl["displayName"].startswith("sweep-"):
        workload_stop(wl["workloadId"])
```

---

## Per-Action Remote Overrides

The action modules in `actions/*.md` contain shared action logic. Below are the
remote execution details for each action.

### Setup (`actions/setup.md`)

In remote mode, after environment detection, create the RayJob (see "RayJob Lifecycle" above)
before proceeding to classify/baseline.

### OOB in Remote Mode

OOB has no persistent service; each Codex / Claude task is one blocking
`oob_ray_submit.py run` invocation. Ray assigns GPU isolation via
`HIP_VISIBLE_DEVICES`.

```bash
# $OOB_RAY_CLI = "python3 $SCRIPTS_DIR/oob_ray_submit.py"
$OOB_RAY_CLI status

$OOB_RAY_CLI run \
  -a codex \
  -p "Optimize this Triton kernel ... (see prompt template in kernel-opt/codex.md)" \
  -f $WORK_DIR/kernel.py \
  -o $WORK_DIR/oob_codex_${KERNEL_NAME} \
  --max-turns 20 \
  --timeout 1800 \
  --no-live --json
```

Output files land at:

```
<output-dir>/tasks/<user>/<task_id>/workspace/optimized_kernel.py
<output-dir>/tasks/<user>/<task_id>/workspace/execution.log
<output-dir>/tasks/<user>/<task_id>/workspace/trajectory_round_*.jsonl
```

Use the JSON `.workspace` field to get the workspace dir directly. There is no
separate `output/` subdir.

The RayJob image must contain both `/opt/venv/bin/oob` and the selected backend
CLI (`codex` or `claude`) on `PATH`. If `oob_ray_submit.py` forwards a client
`PATH` that hides `/opt/venv/bin`, set `OOB_CLI=/opt/venv/bin/oob`.

Additional DSR1 RayJob pitfalls:

- Always pass pod-visible `/wekafs/...` paths to `-f`, `-o`, and runtime file
  references. Client paths such as `/mnt/weka/...` are not valid inside workers.
- Use `OOB_CLI=/opt/venv/bin/oob` when launching from an external Cursor; the
  forwarded client `PATH` can otherwise hide the pod venv.
- The backend CLI must be installed and authenticated. Codex may need
  `/root/.codex/auth.json` or an equivalent `--api-key` path; API key env alone
  may not be enough.
- `OPENAI_BASE_URL` should point at the Core42 proxy:
  `https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1`.
- `ANTHROPIC_BASE_URL` must point at the Core42 Anthropic-compatible direct
  gateway, not the local OOB auth proxy:
  `https://core42.primus-safe.amd.com/api/v1/llm-proxy`.
- `ANTHROPIC_CUSTOM_HEADERS` must include the bearer header:
  `Authorization: Bearer ${ANTHROPIC_API_KEY}`.
- Do not use `ANTHROPIC_BASE_URL=http://127.0.0.1:4002/...` for Claude/OOB on
  Core42. That auth-proxy path is known to return `404` for valid Claude models.
- **Core42 Claude/OOB auth smoke test:** If `oob run -a claude` or `claude --print`
  fails with a model-not-found/access message for a known-valid model such as
  `claude-opus-4-6` or `claude-opus-4-7`, use this one-shot request only to verify
  that the Core42 Anthropic-compatible endpoint, key, headers, and model name are valid:
  ```bash
  curl -sS "${ANTHROPIC_BASE_URL}/v1/messages" \
    -H "x-api-key: $ANTHROPIC_API_KEY" \
    -H "anthropic-version: 2023-06-01" \
    -H "content-type: application/json" \
    -d '{"model":"claude-opus-4-6","max_tokens":8,"messages":[{"role":"user","content":"Reply OK only."}]}'
  ```
  This request is **not** a kernel optimization path and MUST NOT replace OOB.
  If it succeeds, repair the OOB/Claude environment and rerun `oob_ray_submit.py run`.
  On Core42, Claude Code may fail through the bootstrap auth proxy because the proxy
  path/header adaptation can return `404` even when the upstream model is valid.
  Repair the RayJob environment by using the Core42 Anthropic-style base path
  without `/v1` and injecting the Bearer header:
  ```bash
  export ANTHROPIC_BASE_URL="https://core42.primus-safe.amd.com/api/v1/llm-proxy"
  export ANTHROPIC_CUSTOM_HEADERS="Authorization: Bearer ${ANTHROPIC_API_KEY}"

  claude --bare --print --model claude-opus-4-6 "Reply with OK only."
  ```
  When invoking Claude through `oob_ray_submit.py`, ensure
  `ANTHROPIC_BASE_URL` and `ANTHROPIC_CUSTOM_HEADERS` are forwarded to Ray workers.
  `oob_ray_submit.py` already forwards these variables; if a smoke test still shows
  `127.0.0.1:4002`, rerun `bootstrap.sh --force` with the Core42 direct gateway env
  before submitting OOB work.
- A successful OOB smoke should return `status: completed` and produce a
  workspace under `<output-dir>/tasks/cli/<task_id>/workspace/`.

### TraceLens in Remote Mode

TraceLens is a CLI tool with no persistent service. Use it directly on the
remote client side against trace files on shared NFS:

```bash
mkdir -p "$RESULT_DIR/tracelens/perf_report_csvs"
TraceLens_generate_perf_report_pytorch_inference \
  --profile_json_path "$TRACE_PATH" \
  --output_xlsx_path "$RESULT_DIR/tracelens/perf_report.xlsx" \
  --output_csvs_dir "$RESULT_DIR/tracelens/perf_report_csvs" \
  --gpu_arch_json_path "$TRACELENS_ROOT/TraceLens/AgenticMode/Standalone/utils/arch/$GPU_TYPE.json" \
  --enable_pseudo_ops \
  --group_by_num_kernels \
  --enable_kernel_summary
if [ -f "$RESULT_DIR/tracelens/perf_report_csvs/ops_summary.csv" ]; then
  python3 "$TRACELENS_ROOT/TraceLens/AgenticMode/Standalone/orchestrator_prepare.py" \
    --trace-path "$TRACE_PATH" \
    --platform "$GPU_TYPE" \
    --output-dir "$RESULT_DIR/tracelens"
else
  echo "TraceLens produced GPU-only output; use kernel_summary.csv as fallback"
fi
```

vLLM worker traces can be GPU-only. In that case the inference CLI may generate
only `gpu_timeline.csv` and `kernel_summary.csv`; use `kernel_summary.csv` as the
fallback candidate source.

TraceLens pitfalls seen in RayJob validation:

- Use the CLI, not the TraceLens MCP, for Hyperloom CLI validation.
- If `TraceLens_generate_perf_report_pytorch_inference` is missing on the
  client side, install TraceLens with
  `pip install -e /mnt/weka/yunkai/TraceLens-internal`, or rely on the
  RayJob bootstrap-installed CLI.
- `pandas 3.x` can trigger dtype assignment failures; pin `pandas<3` if the
  TraceLens CLI fails during report generation.

### Baseline (`actions/baseline.md`)

All commands must be submitted as Ray Dashboard REST jobs:

```bash
POST /api/jobs entrypoint: "export MODEL='$MODEL' TP=$TP CONC=$CONC FRAMEWORK=sglang \
  SGLANG_EXTRA_ARGS='--enable-torch-compile --mem-fraction-static 0.6' \
  RESULT_DIR='$RESULT_DIR' TRACE_DIR='$TRACE_DIR' INFERENCEX_PATH='$INFERENCEX_PATH' && \
  bash $SCRIPTS_DIR/run_baseline.sh"
```

Trace files and results are written to shared NFS — accessible from both remote client and
Ray cluster.

### Profile (`actions/profile.md`)

Profiling commands run as Ray Dashboard REST jobs:

Profile control must follow `actions/profile.md`: if `/start_profile` or
`/stop_profile` is used, it must be the built-in endpoint of the profiled
SGLang backend process (usually prefill for PD/MoRI). Do not implement or
launch any custom endpoint/service named `start_profile` or `stop_profile`.
When `run_profile.sh` cannot express a PD/MoRI prefill/decode/router topology,
submit the documented Ray Dashboard REST Python driver described in
`actions/profile.md` instead of forcing the single-server script path.

For PD/MoRI, the canonical remote profile flow is:

1. Launch the profiled backend with `SGLANG_TORCH_PROFILER_DIR="$TRACE_DIR"`
   already present in its environment.
2. Send benchmark traffic to the router/public endpoint.
3. Send `/start_profile` and `/stop_profile` only to the profiled backend
   (usually prefill), not to the router.
4. Use an immediate `/start_profile` payload by default:
   ```json
   {"output_dir":"$TRACE_DIR","activities":["CPU","GPU"],"with_stack":true,"record_shapes":true,"profile_prefix":"prefill"}
   ```
   Do not pass `start_step` / `num_steps` by default; use a bounded step window
   only after an immediate profile works and trace size needs reduction.
5. After `/stop_profile`, poll `$TRACE_DIR` until trace count and total size are
   stable. If the directory is empty, inspect the backend server log for
   `Traces are saved to:` before declaring failure.

```bash
POST /api/jobs entrypoint: "export RUN_CONTEXT_FILE='$RESULT_DIR/run_context.env' && \
  bash $SCRIPTS_DIR/run_profile.sh"
```

After profiling, unset profiler env vars inside the Ray cluster:

```bash
POST /api/jobs entrypoint: "unset PROFILE SGLANG_TORCH_PROFILER_DIR VLLM_TORCH_PROFILER_DIR"
```

Trace files on shared NFS — accessible from both remote client and TraceLens CLI.

Filesystem searches for kernel source must also run as Ray Dashboard REST jobs:

```bash
POST /api/jobs entrypoint: "find /sgl-workspace /opt/venv /tmp/torchinductor_root -maxdepth 4 -name '*.py' \
  -exec grep -l 'KERNEL_NAME' {} \\; 2>/dev/null"
POST /api/jobs entrypoint: "python3 - <<'PY'\nfrom pathlib import Path\nprint(Path('/path/to/standalone_kernel.py').read_text())\nPY"
```

Trace parsing can be done on the remote client side (traces are on shared NFS).

### Backends (`actions/backends.md`)

`ServerArgs` inspection and all backend test commands run as Ray Dashboard REST jobs:

```bash
POST /api/jobs entrypoint: "python3 -c \"
from sglang.srt.server_args import ServerArgs
import inspect
src = inspect.getsource(ServerArgs.__init__)
for line in src.split('\\\\n'):
    if '--' in line and ('backend' in line.lower() or 'enable' in line.lower()):
        print(line.strip())
\""
```

Full backend test loop per switch:

```bash
POST /api/jobs entrypoint: "
# Kill server (safe pattern)
ps aux | grep 'python3 -m sglang' | grep -v grep | grep -v bash | awk '{print \$2}' | xargs -r kill -9 2>/dev/null
sleep $SERVER_KILL_WAIT_S

# Restart with new backend
export MODEL='$MODEL' TP=$TP CONC=$CONC FRAMEWORK=$FRAMEWORK
export SGLANG_EXTRA_ARGS='$BASELINE_ARGS $NEW_BACKEND_SWITCH'
export RESULT_DIR='$RESULT_DIR/backend_test_$SWITCH_NAME'
bash $SCRIPTS_DIR/run_baseline.sh
"
```

After killing, verify Ray cluster is alive:
submit `curl -s http://localhost:8265/api/cluster_status` as a Ray Dashboard REST job.

### Server Params (`actions/params.md`)

All server kill/restart + benchmark commands run as Ray Dashboard REST jobs:

```bash
POST /api/jobs entrypoint: "
# Kill server (safe pattern — do NOT pkill -f sglang)
ps aux | grep 'python3 -m sglang' | grep -v grep | grep -v bash | awk '{print \$2}' | xargs -r kill -9 2>/dev/null
sleep $SERVER_KILL_WAIT_S

# Restart with new param
export MODEL='$MODEL' TP=$TP CONC=$CONC FRAMEWORK=$FRAMEWORK
export SGLANG_EXTRA_ARGS='$BASELINE_ARGS $NEW_PARAM'
export RESULT_DIR='$RESULT_DIR/param_test_$PARAM_NAME'
export INFERENCEX_PATH='$INFERENCEX_PATH'
bash $SCRIPTS_DIR/run_baseline.sh
"
```

After killing, verify Ray cluster by submitting `curl -s http://localhost:8265/api/cluster_status` as a Ray Dashboard REST job.

### Kernel Optimization (`actions/kernel-opt.md`)

Kernel source lives on the RayJob. Use Ray Dashboard REST jobs for all find/cat commands:

```bash
POST /api/jobs entrypoint: "find /tmp/torchinductor_root -maxdepth 4 -name '*.py'"
POST /api/jobs entrypoint: "python3 - <<'PY'\nfrom pathlib import Path\nprint(Path('/path/to/standalone_kernel.py').read_text())\nPY"
```

Image selection: use `KERNEL_OPT_IMAGE` (provided by CI or user).

Submit kernel candidates through OOB only:

```bash
# OOB Codex / Claude
$OOB_RAY_CLI run -a codex -p "$PROMPT" -f "$WORK_DIR/kernel.py" \
  -o "$WORK_DIR/oob_codex_${KERNEL_NAME}" --max-turns 20 --timeout 1800 \
  --no-live --json
```

### Integrate (`actions/integrate.md`)

All patch commands run as Ray Dashboard REST jobs:

```bash
POST /api/jobs entrypoint: "python3 $SCRIPTS_DIR/patch_inductor.py patch \
    --kernel-name $KERNEL_NAME \
    --optimized-file $OOB_OUTPUT_PATH \
    --target-file $STANDALONE_FILE_PATH \
    --best-config '{\"XBLOCK\": 4, \"R0_BLOCK\": 2048, \"num_warps\": 4}'"
```

OOB outputs are on shared NFS or the CLI-reported workspace path. Use the
JSON `.workspace` field from `oob_ray_submit.py run --json`.

**Re-Baseline:**

```bash
POST /api/jobs entrypoint: "
export HEALTH_TIMEOUT=1800
export MODEL='$MODEL' TP=$TP CONC=$CONC FRAMEWORK=$FRAMEWORK
export SGLANG_EXTRA_ARGS='$TUNED_SERVER_ARGS'
export RESULT_DIR='$RESULT_DIR/optimized_$KERNEL_NAME'
bash $SCRIPTS_DIR/run_baseline.sh
"
```

**Revert on Ray cluster:**

```bash
POST /api/jobs entrypoint: "
# Strategy A (Inductor cache): restore .bak files
find /tmp/torchinductor_root -name '*.bak' -exec sh -c 'cp \"\$1\" \"\${1%.bak}\"' _ {} \\;

# Strategy B (framework source): restore backup
cp /path/to/kernel.py.bak /path/to/kernel.py
find /sgl-workspace/aiter -name '__pycache__' -exec rm -rf {} + 2>/dev/null
"
```

#### Multi-Node Kernel Patching

For multi-node RayJob, kernel patching requires patching on ALL nodes:
- **Inductor cache** (Strategy A): Cache is node-local. Must patch on ALL nodes,
  OR use a shared Inductor cache directory on NFS (`TORCHINDUCTOR_CACHE_DIR=/wekafs/inductor_cache`).
- **Framework source** (Strategy B): If framework is installed on shared NFS, patching on
  head is sufficient.

For multi-node patching, submit patch command to each node via Ray:

```python
import ray

@ray.remote
def patch_on_node(patch_script):
    import subprocess
    return subprocess.run(patch_script, shell=True, capture_output=True, text=True)

nodes = ray.nodes()
futures = [patch_on_node.options(
    resources={f"node:{node['NodeManagerAddress'].split(':')[0]}": 0.001}
).remote(patch_script) for node in nodes if node['Alive']]
results = ray.get(futures)
```

For multi-node RayJob, revert on ALL nodes using the same `patch_on_node` pattern.

### Sweep (`actions/sweep.md`)

**Option A: Serial sweep on RayJob via Ray Dashboard REST:**

```bash
POST /api/jobs entrypoint: "export MODEL='$MODEL' TP=$TP INFERENCEX_PATH='$INFERENCEX_PATH' \
  CONC_VALUES='4 16 64' \
  ISL_OSL_CONFIGS='1024:1024 8192:1024 1024:8192' \
  RESULT_DIR='$RESULT_DIR/sweep' \
  SGLANG_EXTRA_ARGS='$TUNED_SERVER_ARGS' && \
  bash $SCRIPTS_DIR/run_sweep.sh"
```

**~~Option B: SaFE MCP parallel sweep~~ — DEPRECATED (violates IR-10)**

> Per IR-10, the skill MUST NOT create SaFE workloads other than the main RayJob.
> Use Option A (serial) or Option B (Ray tasks inside the existing RayJob) instead.

**Option B: Parallel sweep via Ray tasks inside the RayJob:**

Submit a Python driver through Ray Dashboard REST. Inside the RayJob
image, the driver can use in-cluster `ray.init()` (no `address=`) to schedule tasks on the
existing cluster:

```python
import ray

ray.init()

@ray.remote(num_gpus=8)
def run_sweep_config(model, tp, conc, isl, osl, result_dir, extra_args):
    import subprocess, os
    env = {
        "MODEL": model, "TP": str(tp), "CONC": str(conc),
        "ISL": str(isl), "OSL": str(osl),
        "RESULT_DIR": result_dir,
        "SGLANG_EXTRA_ARGS": extra_args,
    }
    cmd = f"bash {SCRIPTS_DIR}/run_baseline.sh"
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, env={**os.environ, **env})

configs = [(64, 1024, 1024), (16, 1024, 1024), (4, 1024, 1024),
           (64, 8192, 1024), (16, 8192, 1024)]
futures = [run_sweep_config.remote(MODEL, TP, c, i, o,
    f"/wekafs/inference-optimization/results/sweep_{TIMESTAMP}/c{c}_i{i}_o{o}",
    TUNED_SERVER_ARGS) for c, i, o in configs]
results = ray.get(futures)
```

**Note:** Requires enough GPUs in the Ray cluster to run configs in parallel. With a single
RayJob (8 GPU), configs run sequentially.

### Report (`actions/report.md`)

Generate or copy the final report and result summaries into
`/workspace/hyperloom/` before stopping the RayJob. The RayJob may keep full raw
artifacts under `/wekafs`, but the user-facing final bundle must be in
`/workspace/hyperloom/` so Claw persists it to S3. After this is complete, stop
the RayJob (see "Cleanup" above).

---

## Multi-Node TP

SGLang and vLLM both support multi-node tensor parallelism:

```bash
# SGLang multi-node (Ray-based)
python3 -m sglang.launch_server --model $MODEL --tp $TP --host 0.0.0.0 --port 8888

# vLLM multi-node
vllm serve $MODEL --tensor-parallel-size $TP --distributed-executor-backend ray --host 0.0.0.0 --port 8000
```

TP value should match total GPU count across all nodes (e.g., 2 nodes × 8 GPU = TP=16).

## NFS Sharing Constraint

OOB CLI output and RayJob patching must share the same NFS:
- OOB CLI writes optimized kernels under `<output-dir>/tasks/<user>/<task_id>/workspace/`
- RayJob's patch step reads from the same NFS path
- Ensure RayJob's volume mount includes the OOB output path

## Safe Process Management

**NEVER use `pkill -f sglang`** — it kills Ray workers. Only use:

```bash
# SGLang:
kill $(pgrep -f 'python.*-m sglang.launch_server') 2>/dev/null
ps aux | grep 'sglang.launch_server' | grep -v grep | grep -v 'ray::' | awk '{print $2}' | xargs -r kill -9 2>/dev/null

# vLLM:
kill $(pgrep -f 'python.*-m vllm.entrypoints') 2>/dev/null
```

After killing, ALWAYS verify Ray cluster is alive:
`curl -s http://<HEAD_IP>:8265/api/cluster_status`
