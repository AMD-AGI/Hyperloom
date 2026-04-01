# Action: Environment Setup

## Inputs
- User-specified MODEL, TP, CONC, ISL, OSL, FRAMEWORK (optional — auto-detect if not provided)

## Procedure

### Step 1: Auto-detect environment

```bash
MODEL=$(ls -d /shared_nfs/*/models/*/ 2>/dev/null | head -1)
GPU_COUNT=$(amd-smi list 2>/dev/null | grep "^GPU:" | wc -l)
GPU_TYPE=$(rocm-smi --showproductname 2>/dev/null | grep "GFX Version" | head -1 | grep -o "gfx[0-9]*")
INFERENCEX_PATH=$(ls -d /shared_nfs/*/InferenceX 2>/dev/null | head -1)

FRAMEWORK="${FRAMEWORK:-sglang}"
if [ "$FRAMEWORK" = "vllm" ]; then
    FRAMEWORK_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null)
else
    FRAMEWORK_VERSION=$(python3 -c "import sglang; print(sglang.__version__)" 2>/dev/null)
fi

TP=$GPU_COUNT
if [ "$TP" -le 1 ]; then CONC=4; elif [ "$TP" -le 4 ]; then CONC=32; else CONC=64; fi
```

User-specified values override auto-detected ones.

### Step 2: Set paths and env vars

```bash
SKILL_ROOT="${SKILL_ROOT:-.cursor/skills/inference-optimization}"
SCRIPTS_DIR="$SKILL_ROOT/scripts"

# Mode detection
if [ "${GEAK_LOCAL:-true}" = "true" ]; then
    MODE="local"
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace/inference-optimization}"
else
    MODE="claw"
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared_nfs/inference-optimization}"
fi

# Source executor backend (enables exec_on_gpu for local/claw dispatch)
source "$SCRIPTS_DIR/executor.sh"

export MODEL="$MODEL"
export TP="$TP"
export CONC="$CONC"
export ISL="${ISL:-1024}"
export OSL="${OSL:-256}"
export FRAMEWORK="${FRAMEWORK:-sglang}"
export INFERENCEX_PATH="$INFERENCEX_PATH"
```

`run_baseline.sh` writes `run_context.env` into `$RESULT_DIR`. Reuse it for subsequent steps.

## Outputs
- All environment variables set
- `$SKILL_ROOT`, `$SCRIPTS_DIR` paths validated
- `$RESULT_DIR` created

### [CLAW] Phase 1: Create RayJob on SaFE

**Skip this section in local mode.** In claw mode, create a persistent Ray cluster before any GPU operations.

> **CRITICAL: `RAY_JOB_ENTRYPOINT` must be base64-encoded.** Plain text causes `base64: invalid input` → exit code 127.
>
> Base64 encoding: `echo -n "tail -f /dev/null" | base64` → `dGFpbCAtZiAvZGV2L251bGw=`

> **Exactly ONE RayJob per skill execution.** At the start, create a new RayJob and use it for the entire run. Do NOT create a second one — if you already created one in this execution, reuse it. Do NOT reuse RayJobs from previous skill executions. After optimization is complete, `workload_stop` the RayJob.

**Single node (NUM_NODES=1):**

```
Tool: workload_create
Args: {
    "display_name": "inference-opt-<model_short_name>",
    "workspace_id": "<user_workspace_id>",
    "kind": "RayJob",
    "images": ["GEAK_IMAGE_SGLANG_RAY"],
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
    "images": ["GEAK_IMAGE_SGLANG_RAY", "GEAK_IMAGE_SGLANG_RAY"],
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

**Wait for RayJob Ready + get Ray head address:**

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

**If head IP is not directly accessible**, check if SaFE exposes a NodePort or Service endpoint for the Ray Client port (10001).

**Set up claw execution environment:**

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

## Failure Handling
- If no model found: ask user for MODEL path
- If no GPUs detected: check `amd-smi` / `rocm-smi` installation
- If InferenceX not found: check `/shared_nfs/*/InferenceX/`
