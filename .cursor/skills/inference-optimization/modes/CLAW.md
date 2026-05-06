# Claw Mode — Complete Execution Reference

This document contains ALL claw-mode-specific instructions. Read this **before starting**
when using Claw client with SaFE cluster.

**Agent:** Read `SKILL.md` for the orchestrator loop and shared Iron Rules (IR-1 through
IR-8). This file defines claw-specific Iron Rules (IR-9 through IR-12), constants,
architecture, and per-action execution overrides.

## Environment

- **Client**: Claw (internal platform, Claude Code-like)
- **Runtime**: SaFE cluster with multi-node GPU
- **MCP Servers**: SaFE MCP + GEAK MCP + OOB GPU Optimizer MCP
- **TraceLens**: Local CLI (`pip install -e /hyperloom/TraceLens-internal`)
- **Storage**: Shared NFS (`/shared_nfs/` inside Pod, maps to NFS root)

## Mode Detection

Auto-detected when Claw client context is present, or user specifies `Mode: claw`.

```bash
if [ "${GEAK_LOCAL:-false}" != "true" ]; then
    MODE="claw"
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared_nfs/inference-optimization}"
fi
```

---

## Claw-Mode Iron Rules

### IR-9: Use `exec_on_gpu` for ALL GPU-side commands

After `source scripts/executor.sh`, ALL commands that run on the Ray cluster MUST go
through `exec_on_gpu()` or `exec_on_gpu_bg()`. **NEVER** manually call `ray_submit.py`
directly.

```bash
# CORRECT — generate YAML config, then exec_on_gpu
exec_on_gpu "magpie benchmark --benchmark-config $RESULT_DIR/config.yaml -o $RESULT_DIR/baseline"

# WRONG — manual ray_submit.py
python3 scripts/ray_submit.py --ray-address ... --command "..."
```

### IR-10: Main inference workload MUST use `kind: "RayJob"`

The persistent inference cluster **MUST** be `kind: "RayJob"`. PyTorchJob is ONLY
created internally by GEAK MCP for kernel optimization — the skill itself MUST NOT
create PyTorchJob workloads.

### IR-11: SaFE MCP — ONLY `workload_create(kind="RayJob")` and `workload_stop`

In Claw mode, the skill may use SaFE MCP **only** for:

- **`workload_create`** with `kind: "RayJob"` — to create the inference cluster
- **`workload_get`** / **`workload_list`** — to check workload status
- **`workload_stop`** — to stop the RayJob after optimization is complete

**FORBIDDEN SaFE MCP operations:**

- **`workload_delete`** — NEVER delete workloads; use `workload_stop` instead
- **`workload_create` with any kind other than `"RayJob"`** — no PyTorchJob, no other
  types. GEAK creates its own PyTorchJobs internally via GEAK MCP; the skill MUST NOT
  create them directly.

Violation = immediate run invalidation.

### IR-12: GEAK configuration is read-only

Same as IR-7 in `SKILL.md` — NEVER modify GEAK configuration, test data, or settings.
Interact with GEAK exclusively through GEAK MCP tool calls.

---

## Claw-Mode Constants

These supplement the shared constants in `SKILL.md`.

| Constant | Value | Description |
|----------|-------|-------------|
| `RAY_CLIENT_PORT` | 10001 | Ray Client port on RayJob head node |

### Image Selection (Claw Mode)

Use `KERNEL_OPT_IMAGE` (provided by CI or user). In claw mode, CI should supply the
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
Claw Client --> Skill (SKILL.md)
                 |-> SaFE MCP (workload_create kind="RayJob")
                 |     |-> Creates Ray Cluster: head (1 pod) + workers (N pods)
                 |     |-> Exposes Ray Client port (10001)
                 |     \-> env.RAY_JOB_ENTRYPOINT = "tail -f /dev/null" (keeps cluster alive)
                 |
                 |-> exec_on_gpu (scripts/executor.sh)
                 |     \-> ray_submit.py -> ray.init("ray://<head>:10001") -> run on cluster
                 |
                 |-> GEAK MCP (remote, unchanged)
                 |     \-> SaFE API -> PyTorchJob -> kernel optimization
                 |
                 |-> TraceLens CLI (local, pip install -e)
                 |     \-> Reads traces from shared NFS
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

### Wait for RayJob Ready + get Ray head address

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
RAY_HEAD_ADDRESS = f"ray://{HEAD_IP}:{RAY_CLIENT_PORT}"
```

### Set up claw execution environment

```bash
export MODE=claw
export RAY_HEAD_ADDRESS="ray://<HEAD_IP>:10001"
export MODEL="<model_path_on_shared_nfs>"
export TP=<total_gpu_count>
export CONC=<concurrency>
export ISL=1024
export OSL=256
export FRAMEWORK=<sglang|vllm>
export INFERENCEX_PATH="<path_on_shared_nfs>"

TIMESTAMP=$(date +%Y-%m-%d-%H-%M)
export RESULT_DIR="/shared_nfs/inference-optimization/results/${TIMESTAMP}"
export TRACE_DIR="/shared_nfs/inference-optimization/traces/${TIMESTAMP}"

source scripts/executor.sh

# Verify Ray cluster
exec_on_gpu "python3 -c \"import ray; ray.init(); print(f'Nodes: {len(ray.nodes())}'); print(ray.cluster_resources()); ray.shutdown()\""
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

## Per-Action Claw Overrides

The action modules in `actions/*.md` contain shared logic for both modes. Below are the
claw-specific execution details for each action.

### Setup (`actions/setup.md`)

In claw mode, after environment detection, create the RayJob (see "RayJob Lifecycle" above)
before proceeding to classify/baseline.

#### Magpie installation on the RayJob head pod

After `workload_get` reports `Running`, install Magpie inside the head pod via
`exec_on_gpu` so subsequent `magpie benchmark` invocations resolve. The path
priority mirrors `actions/setup.md` `_resolve_magpie_path`:

1. `$MAGPIE_PATH` if explicitly exported (must be reachable from the head pod)
2. `/hyperloom/users/8cf535bc3ad11fa15e48157cf3b3f726/Magpie` (current workspace checkout)
3. First match of `/shared_nfs/*/Magpie`
4. Legacy default `/shared_nfs/Magpie`

```bash
# Resolve and install Magpie inside the Ray head pod (idempotent).
exec_on_gpu '
set -e
candidates=(
    "${MAGPIE_PATH:-}"
    "/hyperloom/users/8cf535bc3ad11fa15e48157cf3b3f726/Magpie"
)
for c in "${candidates[@]}"; do
    if [ -n "$c" ] && [ -d "$c" ]; then MAGPIE_PATH="$c"; break; fi
done
if [ -z "${MAGPIE_PATH:-}" ]; then
    MAGPIE_PATH=$(ls -d /shared_nfs/*/Magpie 2>/dev/null | head -1)
fi
MAGPIE_PATH="${MAGPIE_PATH:-/shared_nfs/Magpie}"
export MAGPIE_PATH
echo "Resolved MAGPIE_PATH=$MAGPIE_PATH"
command -v magpie >/dev/null || pip install -e "$MAGPIE_PATH" 2>&1 | tail -5
python3 -c "from Magpie.modes.benchmark.config import SweepMatrix" \
    || { echo "ERROR: Magpie at $MAGPIE_PATH lacks sweep_matrix support" >&2; exit 1; }
'
```

Run this once per RayJob (immediately after `workload_get` shows `Running`).
`actions/sweep.md` and any DFS action that uses `sweep_matrix` will fail without
this step.

### Baseline (`actions/baseline.md`)

All `magpie benchmark` commands must be wrapped with `exec_on_gpu`.

`EXTRA_ARGS_KEY` is `EXTRA_SGLANG_ARGS` for SGLang, `EXTRA_VLLM_ARGS` for vLLM. Compute it
once: `EXTRA_ARGS_KEY="EXTRA_$(echo $FRAMEWORK | tr '[:lower:]' '[:upper:]')_ARGS"`

```bash
# Generate YAML config on shared NFS, then exec_on_gpu
exec_on_gpu "magpie benchmark --benchmark-config $RESULT_DIR/baseline_compile_config.yaml \
  -o $RESULT_DIR/baseline_compile"
```
(The YAML config file is generated by the local action script — see `actions/baseline.md`.)

Results are written to shared NFS — accessible from both Claw client and Ray cluster.

### Profile (`actions/profile.md`)

Profiling via `exec_on_gpu` with Magpie's built-in profiler support:

```bash
exec_on_gpu "magpie benchmark --benchmark-config $RESULT_DIR/profile_config.yaml \
  -o $RESULT_DIR/profile"
```
(The YAML config with `torch_profiler`, `tracelens`, `gap_analysis` enabled is generated by `actions/profile.md`.)

Magpie handles profiler env setup and cleanup internally — no manual `unset PROFILE` needed.

Trace files on shared NFS — accessible from both Claw client and TraceLens CLI.

Filesystem searches for kernel source must also go through `exec_on_gpu`:

```bash
exec_on_gpu "find /sgl-workspace /opt/venv /tmp/torchinductor_root -name '*.py' \
  -exec grep -l 'KERNEL_NAME' {} \\; 2>/dev/null"
exec_on_gpu "cat /path/to/standalone_kernel.py"
```

Trace parsing can be done on the Claw side (traces are on shared NFS).

### Backends (`actions/backends.md`)

`ServerArgs` inspection and all backend test commands run via `exec_on_gpu`:

```bash
exec_on_gpu "python3 -c \"
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
exec_on_gpu "magpie benchmark --benchmark-config $RESULT_DIR/backend_test_${SWITCH_NAME}_config.yaml \
  -o $RESULT_DIR/backend_test_${SWITCH_NAME}"
```
(YAML config generated by `actions/backends.md`.)

After killing, verify Ray cluster is alive:
`exec_on_gpu "curl -s http://localhost:8265/api/cluster_status"`

### Server Params (`actions/params.md`)

All benchmark commands use `exec_on_gpu` with Magpie (server lifecycle managed by Magpie):

```bash
exec_on_gpu "magpie benchmark --benchmark-config $RESULT_DIR/param_test_${PARAM_NAME}_config.yaml \
  -o $RESULT_DIR/param_test_${PARAM_NAME}"
```
(YAML config generated by `actions/params.md`.)

After killing, verify Ray cluster: `exec_on_gpu "curl -s http://localhost:8265/api/cluster_status"`

### Kernel Optimization (`actions/kernel-opt.md`)

Kernel source lives on the RayJob. Use `exec_on_gpu` for all find/cat commands:

```bash
exec_on_gpu "find /tmp/torchinductor_root -name '*.py' | while read f; do ..."
exec_on_gpu "cat /path/to/standalone_kernel.py"
```

Image selection: use `KERNEL_OPT_IMAGE` (provided by CI or user).

### Integrate (`actions/integrate.md`)

All patch commands go through `exec_on_gpu`:

```bash
exec_on_gpu "python3 $SCRIPTS_DIR/patch_inductor.py patch \
    --kernel-name $KERNEL_NAME \
    --geak-file $GEAK_OUTPUT_PATH \
    --target-file $STANDALONE_FILE_PATH \
    --best-config '{\"XBLOCK\": 4, \"R0_BLOCK\": 2048, \"num_warps\": 4}'"
```

GEAK output is on shared NFS: `/shared_nfs/geak/tasks/<user_hash>/<task_id>/output/`.

**Re-Baseline:**

```bash
exec_on_gpu "magpie benchmark --benchmark-config $RESULT_DIR/optimized_${KERNEL_NAME}_config.yaml \
  -o $RESULT_DIR/optimized_${KERNEL_NAME}"
```
(YAML config generated by `actions/integrate.md`.)

**Revert on Ray cluster:**

```bash
exec_on_gpu "
# Strategy A (Inductor cache): restore .bak files
find /tmp/torchinductor_root -name '*.bak' -exec sh -c 'cp \"\$1\" \"\${1%.bak}\"' _ {} \\;

# Strategy B (framework source): restore backup
cp /path/to/kernel.py.bak /path/to/kernel.py
find /sgl-workspace/aiter -name '__pycache__' -exec rm -rf {} + 2>/dev/null
"
```

#### Multi-Node Kernel Patching

For multi-node RayJob, kernel patching requires patching on ALL nodes:
- **Inductor cache** (Strategy A): Cache is local to each node. Must patch on ALL nodes,
  OR use a shared Inductor cache directory on NFS (`TORCHINDUCTOR_CACHE_DIR=/shared_nfs/inductor_cache`).
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

**Option A: Serial sweep on RayJob via `exec_on_gpu`:**

```bash
for CONC_VAL in 4 16 64; do
  for ISL_OSL in "1024:1024" "8192:1024" "1024:8192"; do
    ISL_VAL=${ISL_OSL%%:*}; OSL_VAL=${ISL_OSL##*:}
    exec_on_gpu "magpie benchmark --benchmark-config \
      $RESULT_DIR/sweep/conc${CONC_VAL}_isl${ISL_VAL}_osl${OSL_VAL}_config.yaml \
      -o $RESULT_DIR/sweep/conc${CONC_VAL}_isl${ISL_VAL}_osl${OSL_VAL}"
  done
done
```

**~~Option B: SaFE MCP parallel sweep~~ — DEPRECATED (violates IR-11)**

> Per IR-11, the skill MUST NOT create SaFE workloads other than the main RayJob.
> Use Option A (serial) or Option C (Ray submit) instead.

**Option B: Parallel sweep via Ray submit:**

Submit each config as a separate Ray task (uses the existing RayJob cluster):

```python
import ray

ray.init(address=RAY_HEAD_ADDRESS)

EXTRA_ARGS_KEY = f"EXTRA_{FRAMEWORK.upper()}_ARGS"

@ray.remote(num_gpus=8)
def run_sweep_config(model, tp, conc, isl, osl, result_dir,
                     extra_args_key, extra_args, framework, runner_type, inferencex_path):
    import subprocess, os, yaml
    config = {"benchmark": {
        "framework": framework, "model": model, "precision": "fp8",
        "run_mode": "local", "runner_type": runner_type,
        "inferencex_path": inferencex_path,
        "benchmark_script": f"{framework}_{runner_type}.sh",
        "envs": {"TP": tp, "CONC": conc, "ISL": isl, "OSL": osl,
                 "RANDOM_RANGE_RATIO": 0.5, extra_args_key: extra_args},
        "profiler": {"torch_profiler": {"enabled": False}},
    }}
    config_path = f"{result_dir}/config.yaml"
    os.makedirs(result_dir, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    cmd = f"magpie benchmark --benchmark-config '{config_path}' -o '{result_dir}'"
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, env=os.environ.copy())

configs = [(64, 1024, 1024), (16, 1024, 1024), (4, 1024, 1024),
           (64, 8192, 1024), (16, 8192, 1024)]
futures = [run_sweep_config.remote(
    MODEL, TP, c, i, o,
    f"/shared_nfs/inference-optimization/results/sweep_{TIMESTAMP}/c{c}_i{i}_o{o}",
    EXTRA_ARGS_KEY, TUNED_SERVER_ARGS, FRAMEWORK, RUNNER_TYPE, INFERENCEX_PATH
) for c, i, o in configs]
results = ray.get(futures)
```

**Note:** Requires enough GPUs in the Ray cluster to run configs in parallel. With a single
RayJob (8 GPU), configs run sequentially.

### Report (`actions/report.md`)

After the report is generated, stop the RayJob (see "Cleanup" above).

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

GEAK optimization output and RayJob must share the same NFS:
- GEAK server writes optimized kernels to `NFS_BASE_PATH/tasks/<user>/<task_id>/output/`
- RayJob's patch step reads from the same NFS path
- Ensure RayJob's volume mount includes GEAK's storage path

If GEAK uses separate storage, download kernel via `geak_download_file` before patching.

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
