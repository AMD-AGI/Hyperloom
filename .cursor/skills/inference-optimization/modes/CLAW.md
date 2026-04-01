# Claw Mode — Execution Details

This document supplements `SKILL.md` with claw-mode-specific instructions.
Applies when using Claw client with SaFE cluster (multi-node support).

## Environment

- **Client**: Claw (internal platform, Claude Code-like)
- **Runtime**: SaFE cluster with multi-node GPU
- **MCP Servers**: SaFE MCP + GEAK MCP + TraceLens MCP (all remote HTTP)
- **Storage**: Shared NFS (`/shared_nfs/` inside Pod, maps to NFS root)

## Mode Detection

Auto-detected when Claw client context is present, or user specifies `Mode: claw`.

```bash
if [ "${GEAK_LOCAL:-false}" != "true" ]; then
    MODE="claw"
    WORKSPACE_ROOT="${WORKSPACE_ROOT:-/shared_nfs/inference-optimization}"
fi
```

## RayJob Lifecycle

### Step 1: Create a NEW RayJob

See `SKILL.md` Constants for the image to use. See `actions/setup.md` [CLAW] section for the full `workload_create` payloads (single-node and multi-node).

**Exactly ONE RayJob per skill execution.** At the start, create a new RayJob and use it throughout the entire run. Do NOT create a second one mid-execution. Do NOT reuse RayJobs left over from previous skill executions (they may have stale state). After the run completes, `workload_stop` the RayJob.

Key points:
- `kind: "RayJob"` (IR-7: NEVER use PyTorchJob for the main inference workload)
- `RAY_JOB_ENTRYPOINT` must be base64-encoded (`dGFpbCAtZiAvZGV2L251bGw=`)
- Use `GEAK_IMAGE_SGLANG_RAY` (upstream SGLang + Ray 2.44.1 fix)
- `is_tolerate_all: true` to prevent scheduling failures

### Step 2: Wait for Ready and get Ray address

Poll `workload_get` until `phase == "Running"`, then extract head pod IP.
Ray Client address: `ray://<head_ip>:10001`

### Step 3: Submit tasks via executor

All GPU operations use `exec_on_gpu` from `executor.sh`:

```bash
export MODE=claw
export RAY_HEAD_ADDRESS="ray://<head_ip>:10001"
source scripts/executor.sh

exec_on_gpu "bash scripts/run_baseline.sh"
```

### Step 4: Cleanup

```
Tool: workload_delete
Args: { "workload_id": "<id>" }
```

## Phase-by-Phase Execution

| Phase | Execution |
|-------|-----------|
| Setup | `workload_create(kind="RayJob")` + wait for Ready |
| Baseline | `exec_on_gpu`: launch server + `run_baseline.sh` |
| Profile | `exec_on_gpu`: profiling via `/start_profile` + benchmark |
| TraceLens | TraceLens MCP (traces on shared NFS) |
| Identify Candidates | `exec_on_gpu` for source search; trace parsing on Claw side |
| Backends | `exec_on_gpu`: kill + restart + benchmark with backend switches |
| Server Params | `exec_on_gpu`: kill + restart + benchmark per param |
| GEAK | GEAK MCP (creates SaFE PyTorchJob for kernel optimization) |
| Integrate | `exec_on_gpu`: patch kernel (from shared NFS) + restart + benchmark |
| Sweep | `exec_on_gpu` (serial) or SaFE parallel or Ray submit |
| Report + Cleanup | Claw-side report + `workload_delete` |

## Multi-Node TP

SGLang and vLLM both support multi-node tensor parallelism:

```bash
# SGLang multi-node (Ray-based)
python3 -m sglang.launch_server --model $MODEL --tp $TP --host 0.0.0.0 --port 8888

# vLLM multi-node
vllm serve $MODEL --tensor-parallel-size $TP --distributed-executor-backend ray --host 0.0.0.0 --port 8000
```

TP value should match total GPU count across all nodes (e.g., 2 nodes x 8 GPU = TP=16).

## NFS Sharing Constraint

GEAK optimization output and RayJob must share the same NFS:
- GEAK server writes optimized kernels to `NFS_BASE_PATH/tasks/<user>/<task_id>/output/`
- RayJob's patch step reads from the same NFS path
- Ensure RayJob's volume mount includes GEAK's storage path

If GEAK uses separate storage, download kernel via `geak_download_file` before patching.

## Multi-Node Kernel Patching

After GEAK optimization, patching kernels on multi-node requires:
1. Patch on head node (kernel files are in shared NFS or Inductor cache)
2. Kill inference server (all workers stop)
3. Restart server (Ray automatically re-distributes workers)

For Inductor cache patching (torch.compile mode), the cache is local to each node.
Must patch on ALL nodes or use a shared Inductor cache directory on NFS.

See `actions/integrate.md` [CLAW] 8a-extra for the multi-node patching code.

## Safe Process Management (IR-5)

**NEVER use `pkill -f sglang`** — it kills Ray workers. Only use:

```bash
# SGLang:
kill $(pgrep -f 'python.*-m sglang.launch_server') 2>/dev/null
ps aux | grep 'sglang.launch_server' | grep -v grep | grep -v 'ray::' | awk '{print $2}' | xargs -r kill -9 2>/dev/null

# vLLM:
kill $(pgrep -f 'python.*-m vllm.entrypoints') 2>/dev/null
```

After killing, ALWAYS verify Ray cluster is alive: `curl -s http://<HEAD_IP>:8265/api/cluster_status`
