# Action: Recover — Crash Recovery

**DFS role:** Triggered when the server crashes, a kernel patch causes failure, or the
environment becomes unstable. Replaces the old "2+ crashes = emergency stop" with
intelligent recovery.

## Crash Classification

| Crash type | Signature | Recovery |
|-----------|-----------|----------|
| **Server OOM** | `RuntimeError: out of memory`, `hipErrorOutOfMemory` | Reduce `--mem-fraction-static` by 0.05, reduce `--cuda-graph-max-bs` |
| **CUDA graph capture** | `RuntimeError: wrong! device_gemm`, `CUDA graph capture failure` | Reduce `--cuda-graph-max-bs`, try `--disable-cuda-graph` |
| **Kernel patch crash** | Server dies after kernel patch, `AttributeError`, `NameError` | Revert patch, clear caches, restart |
| **Framework patch crash** | Server dies after Strategy D/E edit, import error | Git checkout, pip reinstall, restart |
| **Model loading OOM** | Dies during shard loading, resource tracker warnings | Reduce TP or `--mem-fraction-static`, check for leaked processes |
| **NCCL/RCCL timeout** | `NCCL timeout`, `watchdog timeout` | Check GPU health, restart with RCCL debug, try `NCCL_TIMEOUT=1800` |
| **Unknown** | No matching signature | Save crash log, restore checkpoint, retry with conservative config |

## Procedure

### Step 1: Classify the crash

```bash
# Extract crash signature from server log
CRASH_LOG=$(tail -100 "$RESULT_DIR/server.log" 2>/dev/null)
CRASH_TYPE="unknown"

if echo "$CRASH_LOG" | grep -q "out of memory\|hipErrorOutOfMemory"; then
    CRASH_TYPE="oom"
elif echo "$CRASH_LOG" | grep -q "CUDA graph\|device_gemm\|wrong!"; then
    CRASH_TYPE="cuda_graph"
elif echo "$CRASH_LOG" | grep -q "AttributeError\|NameError\|ImportError"; then
    CRASH_TYPE="patch_crash"
elif echo "$CRASH_LOG" | grep -q "NCCL\|watchdog\|timeout"; then
    CRASH_TYPE="nccl_timeout"
fi
```

### Step 2: Apply recovery strategy

Each crash type has a recovery chain with exponential backoff.

```python
RECOVERY_CHAINS = {
    "oom": [
        {"action": "reduce_mem_fraction", "delta": -0.05, "wait_s": 10},
        {"action": "reduce_cuda_graph_bs", "divisor": 2, "wait_s": 10},
        {"action": "disable_cuda_graph", "wait_s": 10},
        {"action": "restore_checkpoint", "wait_s": 30},
    ],
    "cuda_graph": [
        {"action": "reduce_cuda_graph_bs", "divisor": 2, "wait_s": 10},
        {"action": "disable_cuda_graph", "wait_s": 10},
        {"action": "restore_checkpoint", "wait_s": 30},
    ],
    "patch_crash": [
        {"action": "revert_last_patch", "wait_s": 10},
        {"action": "clear_all_caches", "wait_s": 10},
        {"action": "restore_checkpoint", "wait_s": 30},
    ],
    "nccl_timeout": [
        {"action": "increase_nccl_timeout", "value": 1800, "wait_s": 30},
        {"action": "check_gpu_health", "wait_s": 60},
        {"action": "restart_with_debug", "wait_s": 30},
    ],
    "unknown": [
        {"action": "restore_checkpoint", "wait_s": 30},
        {"action": "conservative_restart", "wait_s": 60},
    ],
}
```

### Step 3: Attempt recovery with backoff

```python
MAX_RECOVERY_ATTEMPTS = 3
BACKOFF_MULTIPLIER = 2

for attempt in range(MAX_RECOVERY_ATTEMPTS):
    chain = RECOVERY_CHAINS.get(crash_type, RECOVERY_CHAINS["unknown"])
    step = chain[min(attempt, len(chain) - 1)]

    print(f"Recovery attempt {attempt + 1}/{MAX_RECOVERY_ATTEMPTS}: {step['action']}")
    wait_time = step["wait_s"] * (BACKOFF_MULTIPLIER ** attempt)

    apply_recovery_action(step)
    time.sleep(wait_time)

    if try_restart_server():
        print(f"Recovery successful after {attempt + 1} attempts")
        state["crash_log"].append({
            "timestamp": time.time(),
            "type": crash_type,
            "recovery": step["action"],
            "attempts": attempt + 1,
        })
        break
else:
    print("Recovery failed after all attempts — escalating")
    state["crash_count"] += 1
    if state["crash_count"] >= 3:
        print("3+ crashes in session — emergency stop, report partial results")
        # Proceed to report.md with partial results
```

### Step 4: Post-recovery

After successful recovery:
1. Save checkpoint immediately
2. Log crash to KB: `python3 $SKILL_ROOT/kb/kb_ingest.py --category crash_recovery ...`
3. Reduce scores for the action that caused the crash (0.3x multiplier)
4. Resume DFS loop from the next action on the stack

## Outputs
- Updated `state.crash_log` with crash details and recovery action
- Updated `state.crash_count`
- KB entry documenting the crash and fix
- Restored server (if recovery succeeded)

## Accuracy Validation
N/A — recovery restores to a known-good state.

## Failure Handling
- All recovery attempts fail: emergency stop → `actions/report.md` with partial results
- Checkpoint restore fails: full restart from Step 1 with conservative config
