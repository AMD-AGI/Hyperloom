---
name: marathon-inference-optimization
description: |
  Marathon-phase inference optimization (3h-24h+). Operates on the deeper layers of the
  compute stack: Compiler, Kernel/Operator Libraries, Communication, and GPU Runtime.
  Picks up from a Sprint-optimized baseline (or pre-optimized directory) and performs
  deep analysis: kernel dispatch tracing, variant discovery, framework rebuilds, operator
  autotuning, communication topology optimization, and register-constrained kernel rewrites.
  Dreams every 3-4h with KB contribution.
globs:
  - "**/inference*optim*"
  - "**/benchmark*"
  - "**/sglang*"
  - "**/vllm*"
  - "**/kernel*"
  - "**/aiter*"
---

# Marathon Inference Optimization — Deep Stack Orchestrator

> **MANDATORY:** You MUST follow The Protocol below step-by-step. Do NOT freelance.
> Marathon starts from an already-optimized baseline. Do NOT redo Sprint-level work
> (backend switches, basic server params, basic kernel-opt). Go deeper.
> Every optimization attempt MUST go through the DFS loop with scored actions.

**Primary objective:** maximize `tok/s/GPU` by optimizing the deeper layers of the compute stack.
**Hard constraint:** accuracy must not degrade (eval gate).
**Method:** depth-first search over code-level optimization actions, scored by a heuristic.
**Time budget:** 24+ hours. This is the Marathon phase — deep analysis, framework rebuilds, kernel rewrites.
**Prerequisites:** Sprint output (handoff directory) OR a pre-optimized baseline directory.

## Stack Scope

This skill optimizes the **deeper layers** of the compute stack, below where Sprint operates:

```
┌─────────────────────────────────┐
│  Application                    │
│  Serving / Orchestration        │  ← Sprint already optimized these
│  Model                          │
│  Framework / Runtime            │
╞═════════════════════════════════╡
│  Compiler                       │  ← Triton/Inductor codegen, tiling, fusion configs
├─────────────────────────────────┤
│  Kernel / Operator Libraries    │  ← Deep kernel analysis, dispatch tracing, rewrites,
│                                 │     GEMM shape tuning, kernel fusion, framework rebuilds
├─────────────────────────────────┤
│  Communication                  │  ← Topology-aware algorithms, compute-comm overlap,
│                                 │     multi-node optimization (scales to 1000s of GPUs)
├─────────────────────────────────┤
│  GPU Runtime                    │  ← Graph strategies, stream concurrency, occupancy tuning
├─────────────────────────────────┤
│  Driver                         │
│  Hardware                       │
└─────────────────────────────────┘
```

Marathon does NOT redo Sprint-level work. If Sprint was not run, Marathon starts with
a re-profile to understand the baseline before going deep.

## The Protocol

**Execute these steps in STRICT ORDER. No skipping. No reordering.**

Each step has a GATE — you MUST record completion in `state.json` before
proceeding to the next step. Check `state.json["protocol_stages_completed"]`
at the start of each step; if the previous stage is missing, you are out
of order — STOP and go back.

```
0. WARM-START     → actions/setup.md       — ingest baseline + BENCHMARK to measure actual tput
1. RE-PROFILE     → actions/profile.md     — fresh trace on the baseline server
2. DEEP ANALYSIS  → actions/deep-kernel-analysis.md + BULK KM DISPATCH
3. BUILD STACK    → score Marathon-specific actions from profile + analysis results
4. DFS LOOP       → pop → execute → measure → re-score → dream every 3-4h → repeat
5. SWEEP          → actions/sweep.md       — extended sweep on deeply-optimized config
6. REPORT         → actions/report.md      — final report + KB contribution
7. DREAM          → final consolidation, cross-run KB contribution
```

### Stage Gate Protocol (MANDATORY — IR-20)

After completing each stage, record it in state.json:

```python
import json, datetime
def complete_stage(state_path, stage_num, stage_name, result_summary):
    with open(state_path) as f:
        state = json.load(f)
    entry = {
        "stage": stage_num,
        "name": stage_name,
        "completed_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "result": result_summary
    }
    if "protocol_stages_completed" not in state:
        state["protocol_stages_completed"] = []
    state["protocol_stages_completed"].append(entry)
    state["protocol_stage"] = stage_num + 1
    state["phase"] = stage_name
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
```

**GATE CHECK** — run this before starting any stage N:
```python
def gate_check(state, required_stage):
    completed = [s["stage"] for s in state.get("protocol_stages_completed", [])]
    for s in range(required_stage):
        if s not in completed:
            raise RuntimeError(f"STAGE GATE VIOLATION: stage {s} not completed, "
                               f"cannot start stage {required_stage}. "
                               f"Completed: {completed}")
```

### Step 0: Warm-Start

Two modes (see `actions/setup.md`):

- **Mode A: Sprint handoff** — read `handoff/config.json`, apply `handoff/patches/`,
  load `handoff/opportunities.json` as pre-scored Marathon action candidates.
- **Mode B: Pre-optimized directory** — parse launch script, extract config, set as baseline.

**Step 0 MUST end with a MEASURED baseline benchmark.** The `baseline_tput_per_gpu`
in state.json MUST be an actual benchmark result, never an assumed or inherited
value. Run the benchmark, record the result, then call `complete_stage(0, "warmstart", ...)`.

Both modes then proceed to Step 1 RE-PROFILE (no skipping).

### Step 1: Re-Profile (MANDATORY — do NOT skip)

**GATE:** Step 0 must be in `protocol_stages_completed` before starting Step 1.

After measuring the baseline in Step 0, profile the RUNNING server to get
a kernel-level GPU% breakdown. This drives everything that follows.

1. Run `actions/profile.md` on the baseline server (already running from Step 0)
2. Record the top-N kernels by GPU% in `state.json["kernel_dispatch_map"]`
3. Identify which kernels are >1% GPU time — these are your optimization targets
4. Record the profile result path in state.json
5. Call `complete_stage(1, "re-profile", {"top_kernels": [...], "profile_path": "..."})`

**DO NOT build the action stack or start DFS without a fresh profile.**
Prior session knowledge is useful context but NOT a substitute for profiling
the current running server — library versions change, optimizations get
reverted, dispatch paths shift.

### Step 2: Deep Analysis + BULK KM DISPATCH

**GATE:** Step 1 must be in `protocol_stages_completed` before starting Step 2.

Run `actions/deep-kernel-analysis.md` on AT LEAST the **top 10 kernels** by GPU%
(minimum threshold: 2.0% GPU time). Do NOT stop after finding one opportunity.

**MANDATORY analysis steps for EACH of the top 10 kernels:**

0. **Env var scan** — FIRST, scan ALL framework env vars that control dispatch
   (FP4_ASM, MFMA_PAGE_ATTN, BUFFER_OPS, CONFIG_FMOE, CONFIG_GEMM_BF16, etc.)
   Check which are set vs unset in the serve script. Each unset high-impact flag
   is a candidate action.
1. **Dispatch path trace** — Python call chain → compiled extension → GPU kernel
2. **Variant discovery** — search for alternative implementations, check platform branching
3. **Config verification** — check shape-specific tuning configs vs generic defaults
4. **Build system trace** — how is the kernel compiled? Patch vs rebuild required?
5. **Opportunity report** — classifies each kernel as `self-fix` or `oob-rewrite`:
   - **Self-fix** (dispatch bugs, env var flags, config changes): → action_stack
   - **OOB-rewrite** (kernel rewrites, multi-file changes): → KM work queue

The action_stack after Step 2 should have AT LEAST 5-10 candidate actions.
If you have fewer than 5, you haven't analyzed deeply enough — go back and
check more kernels and env var flags.

**BULK KM DISPATCH (MANDATORY after deep analysis):**

After completing deep analysis, IMMEDIATELY write work_queue entries for ALL
OOB-rewrite targets in bulk. Do NOT trickle them one-at-a-time as you pop DFS
actions. The KM runs asynchronously — feeding it all targets up front maximizes
parallelism and ensures kernel optimizations are in-flight while you do other DFS work.

For EACH kernel with GPU% >= 1 that needs OOB optimization:
```python
entry = {
    "id": f"km-{kernel_name_slug}",
    "kernel_name": kernel_name,
    "source_file": source_file,
    "source_type": source_type,  # "triton" | "cpp_hip" | "python"
    "strategy": strategy,        # "oob-rewrite" | "triton-rewrite" | "hip-kernel" | etc.
    "priority": int(gpu_pct * 10),
    "gpu_pct": gpu_pct,
    "dispatch_analysis": {
        "active_path": current_dispatch_path,
        "optimal_path": best_known_path,
        "dispatch_bug": bool,
    },
    "trace_shapes": [...],       # actual GEMM/op shapes from profiler
    "constraints": {
        "head_dim": 64,          # from model config
        "hidden_size": 2880,
        "target_vgprs": 64,     # for 4-wave occupancy on gfx950
        "fp8_type": "e4m3fnuz", # MI355X native type
    },
    "status": "pending",
    "timestamp": now_utc_iso,
}
# Append to $RESULT_DIR/kernel_manager/work_queue.jsonl
```

Target: up to 25 entries per analysis cycle (`KERNEL_OPT_MAX_SUBMISSIONS = 25`).
The KM will process them in priority order (highest gpu_pct first).

### Step 3: Build Stack

**GATE:** Step 2 must be in `protocol_stages_completed` before starting Step 3.

Score Marathon-specific actions based on the profile and deep analysis results.
The action stack MUST be derived from actual profiling data — not from prior
session knowledge alone. Prior sessions provide context for scoring, but the
actions themselves come from the current profile + deep analysis.

1. For each kernel from the profile: score based on GPU%, dispatch analysis,
   and prior session outcomes (what worked/failed before)
2. Push all scored actions onto `action_stack`, highest score first
3. Verify the KM work queue has entries (from Step 2 bulk dispatch)
4. Call `complete_stage(3, "build-stack", {"stack_size": N, "km_queue_size": M})`

### Step 4: The DFS Loop (Marathon Core)

**GATE:** Step 3 must be in `protocol_stages_completed` before entering the DFS loop.

```
WHILE NOT stopping_criteria_met():

  ** STACK EMPTY → IMMEDIATE RE-ANALYSIS (IR-24): **
     If action_stack is empty, do NOT exit the DFS loop. Instead:
       1. Re-profile the server (actions/profile.md)
       2. Run FULL deep analysis (actions/deep-kernel-analysis.md) with
          the updated rules: top 10 kernels, env var scan, min 5 candidates
       3. Bulk-dispatch new KM targets
       4. Build fresh action stack from analysis results
       5. If the new stack is STILL empty after thorough analysis, THEN
          enter dream.md and try re-exploration strategies
       6. Only exit the DFS loop if stopping_criteria_met() (time budget
          exhausted or shutdown signal)
     The marathon has 24h — an empty stack after 1h means analysis was
     incomplete, NOT that optimization is done.

  a. Check shutdown signal: [ -f "$SESSION_DIR/STOP_PANE_orchestrator" ]

  ★★★ STEP b: MERGE-OP POLL — ALWAYS FIRST, BEFORE POPPING ANY ACTION ★★★
     KM merges ALWAYS take priority over DFS exploration. The Kernel Manager
     runs validated 4-step tests (IR-17: compile → correctness → multi-shape
     micro-benchmark → adversarial). Its merge-ready patches represent
     already-proven kernel improvements — they must be integrated immediately and test e2e.

     BEFORE popping any action, call poll_kernel_results(). For each
     merge-ready result with micro_benchmark data:
     → Push a `merge-op` action with score 10 (HIGHEST PRIORITY)
     → merge-ops auto-sort to the top of the stack
     → Next pop (step c) will pick them up

     ALSO poll every 10 minutes WITHIN long-running actions (benchmark
     waits, server startup, compilation) and after every server restart.

     DO NOT SKIP. DO NOT DEFER. If merge-ready patches exist, they are
     your next action — period. No DFS exploration until all merge-ops
     are processed.

  c. Pop highest-scored action from action_stack (merge-ops will be on top)
  d. IF action is a self-fix dispatch bug:
       → Apply fix directly (git archaeology + patch + test)
       → No Kernel Manager involvement needed
  e. IF action is a deep-kernel-opt target:
       → Write to $RESULT_DIR/kernel_manager/work_queue.jsonl
       → The Kernel Manager (tmux pane 2) processes it asynchronously
       → Continue DFS loop with other actions
  f. Execute the action (dispatch to actions/*.md)
  g. ACCURACY GATE: if accuracy_risk > 0 → run eval, revert if drop > threshold
  h. Measure: new_tput_per_gpu (MANDATORY — this is the bench result)
  i. Update state: current_tput_per_gpu, cumulative_gain_pct, best_tput_per_gpu
  j. RE-SCORE all remaining actions on the stack
  k. Push new sub-actions discovered during execution
  l. Log to completed_actions with FULL schema (IR-18: tput_before, tput_after,
     gain_pct, timestamp — never omit)
  m. If KEEP: sync $BASE_DIR/state.json (IR-19)
  n. KB ingest

  ** GPU REQUEST POLL (IR-25): ** Check for pending KM GPU requests.
     When TP == GPU_COUNT, the KM cannot micro-benchmark without borrowing GPUs.
     Check $SESSION_DIR/kernel_manager/gpu_request.json for status: "pending".
     If found:
     → Kill inference server + release_gpu_lock()
     → Set gpu_request.json status to "granted" (grant_gpu_request())
     → Wait for KM to finish: poll for status "released" (15s intervals, 30min max)
     → Once released: restart server + write_gpu_lock()
     → Delete gpu_request.json (cleanup_gpu_request())
     The KM micro-benchmark is critical — it feeds merge-ops (score 10 priority).
     Wait up to 30min for KM to finish. One server restart cycle is a small cost
     compared to losing validated benchmark data.

  ** FINDINGS POLL: ** After merge-op poll and between DFS actions, check findings.jsonl for Watchdog
     guidance. Findings may influence scoring:
     - If Watchdog says "hw-blocked" for a kernel: remove its actions from stack
     - If Watchdog provides constraints: update pending kernel targets
     - If systemic finding: re-score all affected kernel actions

  ** EVENT LOGGING: ** Write to event_log.jsonl for ANY action failure:
     - Merge-op fails (crash, regression, revert)
     - Server crashes during any action (startup, benchmark, restart)
     - Framework rebuild fails (setup_rocm.py, pip install, hipcc)
     - Operator tuning crashes or produces invalid configs
     - Communication optimization hangs or RCCL errors
     - Compiler tuning produces bad codegen or cache corruption
     - Accuracy gate fails
     - Benchmark timeout or server hang
     - Dispatch fix fast-path fails unexpectedly

  ** DREAM CADENCE: ** Every 3-4 hours wall clock:
     → actions/dream.md (consolidate + contribute to KB + re-score stack)

  ** CHECKPOINT: ** After every KEEP decision and every 30 min:
     → actions/checkpoint.md (persist state for recovery)

  ** RE-PROFILE CADENCE (every 3 hours): **
     Every 3 hours of wall clock, re-profile the running server to discover
     NEW kernel bottlenecks that emerged after prior optimizations. Prior
     KEEPs may have shifted the bottleneck from one kernel to another.
       1. Run actions/profile.md to get fresh kernel breakdown
       2. Compare with previous profile — find kernels whose GPU% increased
       3. Run deep analysis on the top 10 NEW or CHANGED candidates
       4. Bulk-dispatch new targets to KM work queue (up to 25 total)
       5. Push new DFS actions from the analysis onto the stack
     This ensures the KM always has fresh work and the orch doesn't run
     out of ideas mid-marathon.

  ** ESCALATION (failed DFS action → KM): **
     When a DFS action fails and the failure is kernel-level (not config/infra):
       → Write a work_queue entry to KM with the failure context:
         {id, kernel_name, source_file, strategy: "escalated-from-orch",
          failure_reason, prior_attempts: [...], constraints_discovered}
       → The KM may find a different approach (different backend, different
         optimization strategy) that the orch couldn't execute directly
       → Do NOT just discard kernel failures — always escalate to KM first

  ** CIRCUIT BREAKER (5 consecutive failures → re-analyze): **
     After 5 consecutive DFS action failures (DISCARD or error):
       1. STOP popping actions — the current stack is stale
       2. Re-profile the server (even if <3h since last profile)
       3. Run fresh deep analysis on the new profile
       4. Bulk-dispatch new KM targets from the analysis
       5. Replace the stale action stack with fresh actions
       6. Reset consecutive failure counter
     This prevents the orch from grinding through a stale stack of bad ideas.
```

**Dream is mandatory every 3-4 hours.** Not just at tier boundaries. Every dream
contributes to KB so future runs (on this model or others) benefit.

### Dispatch Fix Fast Path

When `deep-kernel-analysis.md` discovers a dispatch bug (wrong code path active),
the orchestrator fixes it directly without involving the Kernel Manager:

1. Read the source file with the dispatch bug
2. Run `git log -S "symbol" -- file.py` to check history
3. If correct code was removed: restore it. If new code needed: write minimal fix.
4. Clear `__pycache__` and restart server
5. E2E benchmark → KEEP/REVERT

This avoids the round-trip through the work queue for trivial routing fixes.

### Kernel Manager Work Queue Integration

**Writing targets** (orchestrator → manager):

```python
import json, os, datetime

def push_kernel_target(target, result_dir):
    """Append a kernel optimization target to the work queue."""
    queue_path = os.path.join(result_dir, "kernel_manager", "work_queue.jsonl")
    os.makedirs(os.path.dirname(queue_path), exist_ok=True)
    target["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    target["status"] = "pending"
    target["attempts"] = 0
    with open(queue_path, "a") as f:
        f.write(json.dumps(target) + "\n")
```

**Polling results** (orchestrator reads manager output):

```python
MERGE_READY_STATUSES = {"merge-ready", "patch_generated_locally"}

def poll_kernel_results(result_dir, last_seen_id=None):
    """Check for new merge-ready results from the Kernel Manager.

    Matches both 'merge-ready' (OOB-tested) and 'patch_generated_locally'
    (locally written when OOB was unavailable). Both indicate a complete
    patch directory exists under merge_ready/<task_id>/.

    QUALITY GATE: Reject results that lack micro_benchmark data or
    correctness evidence. A merge-ready result without benchmark data
    means the KM skipped the 4-step test pipeline — these are unreliable
    and should be discarded with a log message, not merged.
    """
    results_path = os.path.join(result_dir, "kernel_manager", "results.jsonl")
    if not os.path.exists(results_path):
        return []
    new_results = []
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            result = json.loads(line)
            if result.get("status") in MERGE_READY_STATUSES:
                rid = result.get("id") or result.get("task_id")
                if last_seen_id is None or (rid and rid > last_seen_id):
                    # Quality gate: require micro_benchmark data or explicit "deferred"
                    mb = result.get("micro_benchmark") or result.get("micro_speedup")
                    corr = result.get("correctness") or result.get("correctness_status")
                    if mb is None and corr not in ("passed", "gpu_smoke_pass"):
                        print(f"[MERGE-OP POLL] SKIPPING {rid}: no micro_benchmark "
                              f"and no correctness evidence — KM must run 4-step test")
                        continue
                    new_results.append(result)
    return new_results
```

**Executing a merge-op** (from the DFS stack):

```
merge-op execution:
  1. Read metadata.json from result.patch_dir
  2. Kill inference server
  3. Apply patch (per apply_instructions in metadata)
  4. Run rebuild_command if rebuild_required
  5. Run cache_clear_commands
  6. Run verification_command
  7. Restart server
  8. E2E benchmark + accuracy gate
  9. KEEP → update baseline, log to KB, write_event(type="merge-keep")
     REVERT → run rollback_command, run rollback_rebuild_command if present,
              write_event(type="merge-revert", promising=True)
     CRASH → write_event(type="crash", details include server crash log)
```

### Event Logging (Orchestrator → Watchdog)

Write events to `event_log.jsonl` when merge-ops fail or crashes occur.
The Watchdog Supervisor (tmux pane 0) reads this file and investigates
promising failures.

```python
import json, os, datetime

def write_event(event, result_dir):
    """Append an event to event_log.jsonl for Watchdog consumption."""
    event_log = os.path.join(result_dir, "kernel_manager", "event_log.jsonl")
    os.makedirs(os.path.dirname(event_log), exist_ok=True)
    event["source"] = "marathon"
    event["timestamp"] = datetime.datetime.utcnow().isoformat() + "Z"
    with open(event_log, "a") as f:
        f.write(json.dumps(event) + "\n")
```

**When to write events — ALL DFS action failures:**

| Situation | Event Type | `promising` | Details to Include |
|---|---|---|---|
| **Merge-op** crashes server | `crash` | `True` (passed micro-bench) | Server crash log, patch path, kernel name |
| **Merge-op** E2E regresses | `merge-revert` | `True` | Regression %, micro vs E2E speedup, patch path |
| **Merge-op** accuracy fails | `merge-revert` | `False` | Accuracy diff, eval output |
| **Merge-op** succeeds | `merge-keep` | N/A | Speedup achieved, kernel name |
| **Framework rebuild** fails | `rebuild-fail` | `True` | Build command, stderr, library name, exit code |
| **Framework rebuild** breaks server | `rebuild-crash` | `True` | Server crash log after rebuild, library rebuilt |
| **Operator tuning** crashes | `tuning-crash` | varies | Tuning tool, kernel, shapes attempted, stderr |
| **Operator tuning** invalid config | `tuning-fail` | `False` | Config path, invalid values, server error on load |
| **Comm optimization** RCCL hang | `comm-hang` | `True` | Topology, algorithm, timeout, node count |
| **Comm optimization** RCCL error | `comm-fail` | varies | RCCL error message, topology, algorithm |
| **Compiler tuning** bad codegen | `codegen-fail` | varies | Inductor/Triton error, cache path, kernel name |
| **Compiler tuning** cache corruption | `cache-corrupt` | `False` | Cache path, symptom, error message |
| **Server crash** during any action | `server-crash` | varies | Full crash log, current action name, last config change |
| **Server hang** / benchmark timeout | `server-hang` | `False` | Action name, timeout duration, last output |
| **Dispatch fix** fast-path fails | `dispatch-fix-fail` | `True` | Source file, fix attempted, error |
| **Accuracy gate** fails | `accuracy-fail` | `False` | Accuracy diff, eval output, action that caused it |

**Event schema for non-kernel events:**

Non-kernel events use the same schema but with adapted fields:

```json
{
  "id": "evt_rebuild_crash_001",
  "source": "marathon",
  "type": "rebuild-fail",
  "kernel_name": null,
  "task_id": null,
  "severity": "error",
  "details": {
    "action_name": "framework-rebuild",
    "action_module": "actions/framework-rebuild.md",
    "error_message": "setup_rocm.py install failed with exit code 1",
    "exit_code": 1,
    "crash_log_snippet": "first 2000 chars of build stderr",
    "library": "sgl_kernel",
    "build_command": "cd /sgl-workspace/sglang/sgl-kernel && python setup_rocm.py install",
    "last_config_change": "description of what was changed before failure",
    "gpu_pct": null,
    "session_history": null
  },
  "promising": true,
  "timestamp": "2026-04-10T14:00:00Z"
}
```

Key differences from kernel events: `kernel_name` and `task_id` may be `null`,
`details.action_name` and `details.action_module` identify which DFS action failed,
and `details.library` / `details.build_command` provide build context.

### Findings Polling (Watchdog → Orchestrator)

Check `findings.jsonl` between DFS actions for Watchdog RCA guidance.

```python
def poll_watchdog_findings(result_dir, last_seen_finding_id=None):
    """Check for new Watchdog findings that affect DFS scoring."""
    findings_path = os.path.join(result_dir, "kernel_manager", "findings.jsonl")
    if not os.path.exists(findings_path):
        return []
    findings = []
    with open(findings_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            finding = json.loads(line)
            if last_seen_finding_id is None or finding["event_id"] > last_seen_finding_id:
                findings.append(finding)
    return findings
```

**How findings affect the DFS stack:**

| Finding | Effect |
|---|---|
| `resubmit: false` (hardware) | Remove all pending actions for that kernel from stack |
| `resubmit: true` with constraints | Re-submit to Kernel Manager (see protocol below) |
| `systemic: true` | Re-score all affected kernel actions; if toolchain issue, pause kernel-opt until fixed |
| `confidence: high` + constraints | Boost score of retry actions for that kernel (Watchdog guidance increases success probability) |
| `approach: "fix-toolchain"` | Orchestrator attempts the fix directly (rebuild, env fix), then resumes |
| `approach: "retry-after-rebuild"` | Queue rebuild action first, then re-submit kernel target |
| `approach: "revert-and-retry"` | Revert the last change (`git checkout`/`git stash pop`), retry the action |
| `approach: "revert-comm-config"` | Revert comm topology/algorithm change, try alternative |
| `approach: "clear-cache-retry"` | Clear Triton/Inductor/JIT caches, retry the action |
| `approach: "restart-and-retry"` | Kill server, check GPU state, restart cleanly, retry |
| `approach: "rebuild-with-fix"` | Rebuild with specific compiler flags from finding, then retry |
| `fix_command` is set | Execute the shell command directly, then resume DFS loop |

### Re-Submission Protocol (for exhausted or failed kernels)

When a Watchdog finding has `resubmit: true` and references a `task_id` whose result
in `results.jsonl` is already `failed` or `exhausted`, the orchestrator re-submits
the kernel with the Watchdog's guidance:

```python
def handle_resubmit_finding(finding, result_dir):
    """Re-submit an exhausted kernel with Watchdog RCA constraints."""
    new_target = {
        "id": f"{finding['task_id']}_rca_retry",
        "kernel_name": finding["kernel_name"],
        "source_file": finding.get("details", {}).get("source_file"),
        "strategy": finding["actionable_guidance"].get("approach", "oob-rewrite-register-constrained"),
        "priority": 8,  # high — Watchdog-guided retries have higher success probability
        "rca_constraints": {
            "constraint": finding["actionable_guidance"].get("constraint"),
            "avoid": finding["actionable_guidance"].get("avoid", []),
            "compiler_flags": finding["actionable_guidance"].get("compiler_flags"),
            "max_rounds": 3,  # fewer rounds since we have specific guidance
        },
        "rca_event_id": finding["event_id"],
        "rca_report_path": finding.get("rca_report_path"),
    }
    push_kernel_target(new_target, result_dir)
```

The Kernel Manager sees `rca_constraints` in the work queue entry and bakes
them into the OOB prompt from round 1 (no wasted exploratory rounds).

**When NOT to re-submit:**
- `resubmit: false` — hardware issue, will never succeed
- The same kernel has already been re-submitted via this protocol (check `id` suffix `_rca_retry`)
- `confidence: low` — the Watchdog isn't sure, don't waste rounds on uncertain guidance

## Marathon Actions (deeper-stack code-level)

| Action | Module | DFS Loop? | Blocking? | Stack Layer |
|--------|--------|-----------|:---------:|-------------|
| Deep Kernel Analysis | `actions/deep-kernel-analysis.md` | **Yes** | Yes (10-30 min/kernel) | Kernel/Operator |
| Dispatch Fix (fast path) | Inline (see DFS loop) | **Yes** | Yes (5-10 min) | Kernel/Operator |
| Kernel Target → Manager | Write to `work_queue.jsonl` | **Yes** | No (async) | Kernel/Operator |
| Merge-Op (from Manager) | Read `merge_ready/<id>/` | **Yes** | Yes (5-15 min) | Kernel/Operator |
| Operator Tuning | `actions/operator-tuning.md` | **Yes** | Yes (30-60 min) | Kernel/Operator |
| Framework Rebuild | `actions/framework-rebuild.md` | **Yes** | Yes (5-15 min) | Kernel/Operator |
| Deep Kernel Opt | `actions/kernel-opt.md` | **Yes** | Async + blocking integrate | Kernel/Operator |
| Comm Optimization | `actions/comm-optimization.md` | **Yes** | Yes (15-30 min) | Communication |
| Compiler Tuning | `actions/compiler-tuning.md` | **Yes** | Yes (10-20 min) | Compiler |
| Checkpoint | `actions/checkpoint.md` | At KEEP + every 30 min | No (~30s) | Infrastructure |
| Dream | `actions/dream.md` | Every 3-4h + plateau | Yes (2-5 min) | Infrastructure |
| Recover | `actions/recover.md` | On crash | Yes | Infrastructure |
| Re-Explore | `actions/re-explore.md` | On plateau | Yes | Infrastructure |
| Sweep | `actions/sweep.md` | No — after DFS loop | Yes | — |
| Report | `actions/report.md` | No — always last | Yes | — |

## Kernel Manager (Parallel Process)

The Kernel Manager runs in a separate Claude Code CLI instance (tmux pane 2) and handles
all deep kernel optimization work: OOB backend dispatch with deep guided refinement
(up to 5 rounds), prompt engineering, local compilation testing, micro-benchmarking,
and patch generation. Communication is via file-based IPC on NFS.

## Watchdog Supervisor (Parallel Process)

The Watchdog Supervisor runs in tmux pane 0. It monitors `event_log.jsonl` for crashes,
segfaults, and promising failures. When it detects an event worth investigating, it applies
the `training-workload-rca` methodology to diagnose the root cause and writes an actionable
finding to `findings.jsonl`. Both the orchestrator and Kernel Manager read findings to
improve their decisions.

```
Watchdog (pane 0)     Orchestrator (pane 1)          Kernel Manager (pane 2)
─────────────────     ─────────────────────          ──────────────────────
                      deep-kernel-analysis
                        → classify: self-fix / oob
                        → self-fix: apply directly
                        → oob: write work_queue.jsonl ──→ read work_queue.jsonl
                                                          read findings.jsonl
                      continue DFS with other actions     deep OOB loop (5 rounds)
                                                          local test + analyze
read event_log.jsonl  write events ←──────────────────── write events
  → triage event                                         generate merge-ready patch
  → if promising:     poll results.jsonl ←────────────── write results.jsonl
    investigate (RCA)   → push merge-op onto DFS stack
  → write findings ──→ poll findings.jsonl
                        → adjust scores, hw-block
                        → kill server, apply patch
                        → E2E benchmark, KEEP/REVERT
```

**Skill references:**
- `marathon-inference-optimization/kernel-manager/SKILL.md` (manager process, pane 2)
- `marathon-inference-optimization/watchdog/SKILL.md` (watchdog process, pane 0)
- `training-workload-rca/SKILL.md` (RCA methodology, invoked by watchdog — separate repo)

**IPC paths:**
- `$RESULT_DIR/kernel_manager/work_queue.jsonl` — orchestrator writes, manager reads
- `$RESULT_DIR/kernel_manager/results.jsonl` — manager writes, orchestrator reads
- `$RESULT_DIR/kernel_manager/merge_ready/<id>/` — manager writes patch dirs
- `$RESULT_DIR/kernel_manager/event_log.jsonl` — orchestrator + manager write, watchdog reads
- `$RESULT_DIR/kernel_manager/findings.jsonl` — watchdog writes, orchestrator + manager read
- `$RESULT_DIR/kernel_manager/rca_reports/<id>/` — watchdog writes detailed reports
- `$RESULT_DIR/kernel_manager/gpu_request.json` — KM writes request, orchestrator grants/cleans up

## Time Tiers

Marathon scales its ambition based on wall clock. Dream is mandatory at each boundary.

| Tier | Wall Clock | Focus | Dream? |
|------|:----------:|-------|:------:|
| **Tier 1** | 0-3h | Re-profile + deep kernel analysis on all top-N kernels | Yes (at 3h) |
| **Tier 2** | 3-8h | Operator tuning + framework rebuilds + deep kernel-opt | Yes (at ~6h) |
| **Tier 3** | 8-24h | Communication optimization + compiler tuning + fusion | Yes (every 3-4h) |
| **Tier 4** | 24h+ | Re-explore plateaus + cross-layer optimization | Yes (every 3-4h) |

At every tier boundary: `actions/checkpoint.md` then `actions/dream.md`.

## Heuristic Scoring

Same formula as Sprint, but different priors for Marathon actions:

```
score = (expected_tput_gain_per_gpu / cost_minutes)
        × (1 - accuracy_risk)
        × (1 - crash_risk)
        × target_gap_multiplier
```

### Initial Score Priors (Marathon-specific)

| Action | Dense | MoE+MLA | MoE+SWA | MoE+MLA+NSA |
|--------|-------|---------|---------|-------------|
| deep-kernel-analysis | **9** | **8** | **8** | **8** |
| operator-tuning | 4 | **7** | **7** | **7** |
| deep-kernel-opt | **8** | **6** | **6** | **6** |
| framework-rebuild | 3 | 4 | 4 | 4 |
| comm-optimization | 2 | **5** | **5** | **6** |
| compiler-tuning | **6** | 3 | 3 | 3 |

**Score boosting from Sprint handoff:**
- Kernels tagged with `marathon-candidate` in `opportunities.json` get +3 score
- Kernels with `register-pressure-fixable` get boosted deep-kernel-opt score
- Kernels with `shape-tuning-untested` get boosted operator-tuning score
- Communication bottlenecks from Sprint profile get boosted comm-optimization score

### Score Update Rules

1. **Deep kernel analysis reveals dispatch bug:** Boost deep-kernel-opt to 10 (immediate fix).
2. **Operator tuning finds untuned shapes:** Boost operator-tuning for similar kernels.
3. **Framework rebuild succeeds:** Boost remaining rebuild candidates 1.5×.
4. **Deep kernel-opt with OOB succeeds:** Boost OOB submissions for similar kernels.
5. **Comm optimization gains >2%:** Boost to try additional comm strategies.
6. **2+ discards from ONE backend:** Reduce that backend's scores only. Other backends unaffected.
7. **2+ discards across ALL backends for same kernel:** Reduce that kernel to near-zero.
8. **NEVER zero ALL kernel-opt scores from one backend's failures.** Each backend is independent.

## State Schema

```python
state = {
    # Identity
    "model_name": "",
    "model_class": "",
    "framework": "",
    "tp": 0,
    "gpu_type": "",
    "gpu_count": 0,
    "num_nodes": 1,

    # Performance tracking
    "sprint_tput_per_gpu": 0.0,    # what Sprint achieved (our starting point)
    "baseline_tput_per_gpu": 0.0,  # = sprint_tput after warm-start
    "current_tput_per_gpu": 0.0,
    "cumulative_gain_pct": 0.0,    # gain from Marathon start, not from raw baseline

    "target_tput_per_gpu": None,
    "target_gap_pct": None,

    # Deep analysis results
    "kernel_dispatch_map": {},     # kernel_name → {dispatch_path, variants, best_variant, config_status}
    "untuned_shapes": [],          # [{kernel, shape_m, shape_n, shape_k, config_status}]
    "dispatch_bugs_found": [],     # [{kernel, description, fix_type, status}]

    # Accuracy
    "baseline_accuracy": None,
    "accuracy_threshold": 0.01,

    # DFS state
    "action_stack": [],
    "completed_actions": [],       # REQUIRED per-entry schema — see below
    "kernel_candidates": [],

    # Async kernel optimization (legacy — direct OOB dispatch)
    "pending_kernel_tasks": [],
    "kernel_results": {},

    # Kernel Manager IPC
    "kernel_manager_last_seen_id": None,     # last result ID read from results.jsonl
    "kernel_manager_targets_pushed": 0,      # count of targets written to work_queue
    "kernel_manager_merges_completed": 0,    # count of merge-ops executed
    "kernel_manager_merges_kept": 0,         # count of merge-ops that passed E2E

    # Watchdog IPC
    "watchdog_last_seen_finding_id": None,   # last finding ID read from findings.jsonl
    "watchdog_findings_consumed": 0,         # count of findings read
    "watchdog_hw_blocked_kernels": [],       # kernels marked as hardware-blocked
    "events_written": 0,                     # count of events written to event_log.jsonl

    # Marathon infrastructure
    "current_time_tier": "tier1",
    "checkpoint_path": None,
    "dream_count": 0,
    "last_dream_ts": None,
    "crash_count": 0,
    "crash_log": [],
    "strategies_tested": [],
    "tier_breakdown": {},
    "loop_signatures": [],

    # Tracking
    "total_wall_minutes": 0,
    "total_kernel_opt_submissions": 0,
    "consecutive_discards": 0,
    "backend_wins": {},
    "frameworks_rebuilt": [],       # list of libraries rebuilt during this run
}
```

## `completed_actions` Entry Schema (MANDATORY — IR-18)

Every entry appended to `completed_actions` MUST include ALL of these fields.
Sessions that omit `tput_before` / `tput_after` make gains invisible in
timeline plots and break session-to-session handoff (the 5119→10551 gap bug).

```python
{
    "id": "action_xxx",                     # unique action id
    "action": "operator-tuning",            # action type from actions/*.md
    "name": "Human-readable action name",   # short description
    "status": "KEEP",                       # KEEP | DISCARD | REVERT | INFO | DONE
    "description": "What was done",         # 1-2 sentences

    # --- THESE FOUR FIELDS ARE MANDATORY (never omit, never null) ---
    "tput_before": 10551.33,               # tok/s/GPU BEFORE this action
    "tput_after": 10551.33,                # tok/s/GPU AFTER this action (= bench result)
    "gain_pct": 0.0,                       # (tput_after - tput_before) / tput_before * 100
    "timestamp": "2026-04-21T22:01:54Z",   # ISO 8601

    # Optional but encouraged
    "bench_id": "bench_marathon_xxx",       # benchmark result directory name
    "accuracy_before": None,                # if accuracy gate was run
    "accuracy_after": None,
}
```

**IR-21 (action tracking):** Every optimization MUST be recorded in
`completed_actions` with `tput_before`, `tput_after`, `gain_pct`, and
`timestamp`. This builds the throughput-vs-time timeline.

**IR-22 (no warm-start hotfixes):** Warm-start (Step 0) MUST NOT apply any
patches, hotfixes, or optimizations. It launches the server AS-IS, runs a
baseline benchmark, and records the measured throughput. That's it.
Known hotfixes from prior sessions (block_m fix, TRITON_ROPE toggle, CK
alignment fix, BF16 GEMM tuning, etc.) are added to the action_stack during
Step 3 (Build Stack) and executed one-by-one in the DFS loop (Step 4) with
proper before/after benchmarks. This ensures every gain appears on the
throughput-vs-time plot.

**IR-23 (read-only analysis):** Steps 1 (Re-Profile) and 2 (Deep Analysis)
are strictly READ-ONLY. No code changes, no patches, no file writes to
system packages, no env var changes, no server restarts. These steps ONLY
produce findings, kernel breakdowns, and candidate actions. All candidates
go into the action_stack (Step 3). Code changes ONLY happen in the DFS
loop (Step 4), one action at a time with before/after benchmarks.

## `BASE_DIR/state.json` Sync (MANDATORY — IR-19)

After every KEEP decision and at session shutdown, the orchestrator MUST
update `$BASE_DIR/state.json` with the session's best results:

```python
import json
def sync_base_dir_state(base_dir, session_state):
    base_state_path = f"{base_dir}/state.json"
    base_state = {}
    try:
        with open(base_state_path) as f:
            base_state = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    best = session_state.get("best_tput_per_gpu", 0)
    prev_best = base_state.get("current_tput_per_gpu", 0)
    if best > prev_best:
        base_state.update({
            "current_tput_per_gpu": best,
            "sprint_baseline_tput_per_gpu": base_state.get(
                "sprint_baseline_tput_per_gpu",
                session_state.get("sprint_tput_per_gpu", 0)),
            "tp": session_state.get("tp"),
            "framework": session_state.get("framework"),
            "model": session_state.get("model_name"),
            "last_session_id": session_state.get("session_id"),
            "last_session_best_tput": best,
            "last_updated": session_state.get("last_updated",
                __import__("datetime").datetime.utcnow().isoformat()),
        })
        with open(base_state_path, "w") as f:
            json.dump(base_state, f, indent=2)
```

## Accuracy Gate

Same as Sprint. Actions with `accuracy_risk > 0` must pass the eval gate before KEEP.
Framework rebuilds have `accuracy_risk = 0.15`. Communication changes have `accuracy_risk = 0.05`.

## Stopping Criteria

| Condition | Action |
|-----------|--------|
| All action scores < 1.0 | Proceed to sweep |
| Cumulative Marathon gain > 40% | Proceed to sweep |
| 7 consecutive discards across all actions | Proceed to sweep |
| Wall clock > current tier limit | Dream + check for next tier |
| Target exceeded (gap ≤ 0%) | Proceed to sweep |
| 3+ server crashes in same session | Trigger `actions/recover.md` |

## Iron Rules

**IR-1:** Deep kernel analysis is MANDATORY before kernel optimization. Never skip dispatch tracing.

**IR-2:** Framework rebuilds require rollback plan. Always `git stash` or `cp` originals first.

**IR-3:** Dream every 3-4 hours. Every dream MUST contribute to KB. No exceptions.

**IR-4:** Checkpoint after every KEEP decision. Marathon runs WILL crash — checkpoints are recovery.

**IR-5:** Safe process management — never `pkill -f` the framework.

**IR-6:** Always kill_server + check_gpu_memory before server launch.

**IR-7:** Never zero ALL kernel-opt scores from one backend's failure. Each backend is independent.

**IR-8:** Use Python AST for source patching. Naive regex breaks module-level definitions.

**IR-9:** After framework rebuild, always verify the correct kernel path is active (dispatch check).

**IR-10:** Dispatch bugs and one-line routing fixes are self-fix targets. Apply directly
via the fast path — do NOT send them to the Kernel Manager (wastes round-trip time).

**IR-11:** Always check `git log -S` before writing new code. The correct code may have
existed and been removed by a later commit (the RoPE lesson).

**IR-12 (KM merges always first):** At the TOP of every DFS iteration,
BEFORE popping any action, poll `results.jsonl` for merge-ready KM patches.
If any exist, push them as `merge-op` with score 10 (highest). KM merges
always take priority over DFS exploration — no exceptions. The Kernel
Manager has already validated these patches through the 4-step IR-17
pipeline (compile → correctness → multi-shape benchmark → adversarial).
Deferring them wastes the KM's work and delays throughput gains.

**IR-18 (completed_actions tracking):** Every entry in `completed_actions`
MUST include `tput_before`, `tput_after`, `gain_pct`, and `timestamp`.
Never omit these — timeline plots and session handoff depend on them.
See "completed_actions Entry Schema" above.

**IR-19 (BASE_DIR state sync):** After every KEEP decision and at session
shutdown, sync `best_tput_per_gpu` to `$BASE_DIR/state.json`. This is the
cross-session handoff contract. Without it, the next session starts from a
stale sprint baseline and the throughput timeline has invisible gaps.

**IR-13 (GPU lock):** Before starting the inference server, write
`/tmp/.marathon_gpu_lock.json` with the GPUs being used. Remove it when the
server is killed. The Kernel Manager reads this lock to find free GPUs for
micro-benchmarks without contention. Schema:
```json
{
  "holder": "orchestrator",
  "gpus": [0],
  "pid": 12345,
  "since": "2026-04-21T18:00:00Z",
  "purpose": "inference-server"
}
```
Write helper (call after every server launch):
```python
import json, os, datetime
def write_gpu_lock(gpus, server_pid):
    lock = {"holder": "orchestrator", "gpus": list(gpus), "pid": server_pid,
            "since": datetime.datetime.utcnow().isoformat() + "Z",
            "purpose": "inference-server"}
    with open("/tmp/.marathon_gpu_lock.json", "w") as f:
        json.dump(lock, f)

def release_gpu_lock():
    try: os.remove("/tmp/.marathon_gpu_lock.json")
    except FileNotFoundError: pass
```
Call `write_gpu_lock(list(range(TP)), pid)` after server start.
Call `release_gpu_lock()` after `kill_server()`.

**IR-25 (GPU time-share):** When `TP == GPU_COUNT` (all GPUs run the server),
the Kernel Manager has no free GPUs for micro-benchmarks. The KM can request
temporary exclusive GPU access by writing a `gpu_request.json` file. The
orchestrator checks for this request during its DFS poll loop and, if found,
temporarily kills the server to grant GPU access.

**Protocol:**
1. KM writes `$SESSION_DIR/kernel_manager/gpu_request.json` with `status: "pending"`
2. Orchestrator sees it during GPU REQUEST POLL (DFS loop), kills the server,
   calls `release_gpu_lock()`, sets request `status: "granted"`
3. KM reads "granted", acquires lock (`holder: "kernel-manager"`), runs micro-benchmarks
4. KM releases lock, sets request `status: "released"`
5. Orchestrator reads "released", restarts the server, calls `write_gpu_lock()`,
   deletes the request file

**GPU request helpers (orchestrator side):**
```python
import json, os, time

GPU_REQUEST_PATH = os.path.join(os.environ.get("SESSION_DIR", ""),
                                "kernel_manager/gpu_request.json")

def check_gpu_request():
    """Check if KM has a pending GPU request. Returns the request dict or None."""
    try:
        with open(GPU_REQUEST_PATH) as f:
            req = json.load(f)
        if req.get("status") == "pending":
            return req
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return None

def grant_gpu_request():
    """Grant the pending GPU request (call after killing server + releasing lock)."""
    try:
        with open(GPU_REQUEST_PATH) as f:
            req = json.load(f)
        req["status"] = "granted"
        req["granted_at"] = datetime.datetime.utcnow().isoformat() + "Z"
        with open(GPU_REQUEST_PATH, "w") as f:
            json.dump(req, f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass

def wait_for_gpu_release(timeout_s=1800, poll_interval_s=15):
    """Wait for KM to finish and set status to 'released'. Returns True if released.
    Default timeout is 30min — the micro-benchmark MUST run, so we wait."""
    elapsed = 0
    while elapsed < timeout_s:
        try:
            with open(GPU_REQUEST_PATH) as f:
                req = json.load(f)
            if req.get("status") == "released":
                return True
        except (FileNotFoundError, json.JSONDecodeError):
            return True  # file gone = KM done
        time.sleep(poll_interval_s)
        elapsed += poll_interval_s
    return False  # timeout

def cleanup_gpu_request():
    """Remove the request file after server restart."""
    try: os.remove(GPU_REQUEST_PATH)
    except FileNotFoundError: pass
```

**GPU REQUEST POLL sequence (in DFS loop):**
```
req = check_gpu_request()
if req:
    log("GPU REQUEST: KM needs GPUs for kernel: " + req.get("kernel", "?"))
    kill_server()
    release_gpu_lock()
    grant_gpu_request()
    released = wait_for_gpu_release()  # 30min default — micro-benchmark must run
    if not released:
        log("WARNING: GPU request timed out after 30min, restarting server anyway")
    cleanup_gpu_request()
    start_server()  # normal server launch (calls write_gpu_lock internally)
    # Next DFS iteration will re-benchmark if needed
```

## Agent Creativity

Marathon agents SHOULD:

- **Trace dispatch paths** they haven't seen before — every kernel is a mystery until traced
- **Discover unused code** — frameworks often have compiled-but-unused optimized paths
- **Write new kernels** via OOB agents when existing ones are fundamentally suboptimal
- **Fuse kernel sequences** identified during per-layer analysis
- **Challenge "already tuned" assumptions** — verify shape configs, run tuning tools
- **Cross-pollinate** — apply learnings from one model to another via KB

## Autonomy

**Execute autonomously.** No human confirmation needed for:
- Running benchmarks, profiling, tracing dispatch paths
- Killing/restarting servers
- Writing kernel targets to the work queue (Kernel Manager handles dispatch)
- Applying merge-ready patches from the Kernel Manager
- Applying dispatch-fix fast path changes directly
- Patching and rebuilding framework libraries
- Creating/stopping RayJobs (claw mode)

## Constants

### Marathon-specific

| Constant | Value | Description |
|----------|-------|-------------|
| `DREAM_CADENCE_MIN` | 210 | Dream every 3.5 hours (210 min) |
| `CHECKPOINT_CADENCE_MIN` | 30 | Auto-checkpoint every 30 min |
| `DEEP_ANALYSIS_TOP_N` | 10 | Analyze top N kernels by GPU% |
| `KERNEL_OPT_MAX_SUBMISSIONS` | 25 | Raised budget for deep optimization |
| `KERNEL_OPT_CONSECUTIVE_DISCARDS` | 7 | Raised tolerance for wider exploration |
| `KERNEL_OPT_WALL_CLOCK_MIN` | 180 | Raised kernel-opt wall-clock |

### Shared with Sprint

| Constant | Value | Description |
|----------|-------|-------------|
| `KERNEL_OPT_BACKENDS` | `geak,codex,claude` | All backends active for Marathon |
| `MIN_GPU_PCT` | 3 | Min GPU% to consider a kernel |
| `SERVER_KILL_WAIT_S` | 10 | Seconds between kill and relaunch |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  tmux pane 0: Watchdog Supervisor (Claude Code CLI)         │
│                                                             │
│  watchdog/SKILL.md              — Event monitor + triage    │
│  watchdog/actions/triage.md     — Per-event decision tree   │
│  watchdog/actions/investigate.md— RCA for inference crashes  │
│                                                             │
│  References: training-workload-rca/SKILL.md (RCA methodology)│
│                                                             │
│  IPC:  reads  ← event_log.jsonl (from pane 1 + pane 2)     │
│        writes → findings.jsonl (consumed by pane 1 + 2)     │
│        writes → rca_reports/<event_id>/ (detailed reports)  │
├─────────────────────────────────────────────────────────────┤
│  tmux pane 1: Orchestrator (this skill)                     │
│                                                             │
│  SKILL.md (this file)       — Marathon DFS orchestrator     │
│  actions/*.md                — Action modules               │
│    setup.md                  — Warm-start                   │
│    deep-kernel-analysis.md   — Dispatch tracing + classify  │
│    framework-rebuild.md      — Library rebuild + rollback   │
│    operator-tuning.md        — GEMM shape tuning            │
│    comm-optimization.md      — Communication topology       │
│    kernel-opt.md             — Direct kernel opt (legacy)   │
│    checkpoint.md / dream.md  — Multi-day infrastructure     │
│    recover.md / re-explore.md                               │
│  kernel-opt/                 — Per-backend references        │
│  kb/                         — Shared knowledge base        │
│  scripts/                    — Shell + Python helpers        │
│  modes/                      — Mode-specific details        │
│                                                             │
│  IPC:  writes → work_queue.jsonl, event_log.jsonl           │
│        reads  ← results.jsonl, merge_ready/<id>/,           │
│                  findings.jsonl                              │
├─────────────────────────────────────────────────────────────┤
│  NFS filesystem ($RESULT_DIR/kernel_manager/)               │
│    work_queue.jsonl    — orchestrator → manager             │
│    results.jsonl       — manager → orchestrator             │
│    merge_ready/<id>/   — patch dirs (manager → orchestrator)│
│    event_log.jsonl     — orchestrator + manager → watchdog  │
│    findings.jsonl      — watchdog → orchestrator + manager  │
│    rca_reports/<id>/   — watchdog detailed reports          │
├─────────────────────────────────────────────────────────────┤
│  tmux pane 2: Kernel Manager (separate Claude Code CLI)     │
│                                                             │
│  kernel-manager/SKILL.md        — Manager skill             │
│  kernel-manager/actions/        — dispatch (deep guidance    │
│                                   loop), local-test, patch  │
│                                                             │
│  IPC:  reads  ← work_queue.jsonl, findings.jsonl            │
│        writes → results.jsonl, merge_ready/<id>/,           │
│                  event_log.jsonl                             │
│        dispatches to: GEAK, Codex, Claude, LLM Proxy       │
└─────────────────────────────────────────────────────────────┘
```

## KB Integration

Before each action:
```bash
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME $ACTION_NAME" --top-k 5 --compact
```

After each action with new findings:
```bash
python3 $SKILL_ROOT/kb/kb_ingest.py \
    --category $CATEGORY --model "$MODEL_NAME" \
    --action "$WHAT_WAS_DONE" --lesson "$KEY_TAKEAWAY" \
    --tags $TAGS --gain $GAIN --status $STATUS
```

**Dream KB contribution is mandatory** — every dream writes new entries and updates existing ones.

## Critical Lessons (Marathon-Specific)

1. **Deep kernel analysis finds dispatch bugs.** The RoPE investigation found 3 kernel variants
   with the wrong one being dispatched — 9.5× speedup from routing fix alone.
2. **Framework rebuilds unlock code-level gains.** Patching + rebuilding libraries (not just
   config files) is required for C++/HIP kernel optimization.
3. **"GEAK 0%" does not mean "kernel-opt exhausted."** A single backend's failure says nothing
   about other backends or strategies. Always check tags before skipping.
4. **Register-constrained optimization works.** When unconstrained optimization causes E2E
   regression via register spilling, submit with explicit VGPR/occupancy constraints.
5. **GEMM shape tuning is separate from kernel rewriting.** Models often use generic default
   configs for shapes that could have specialized tuning.
6. **Communication optimization scales with GPU count.** At >1 node, comm becomes the
   dominant bottleneck. Topology-aware algorithm selection is critical.
7. **Dream prevents knowledge loss.** In multi-hour runs, consolidating learnings every 3-4h
   ensures the agent doesn't repeat mistakes and future runs benefit.
8. **Every kernel is a mystery until traced.** Never assume you know which code path
   is executing — always verify with dispatch tracing.
9. **Check git history before writing new code.** The RoPE kernel had its correct
   `sgl_kernel` path removed by a later commit. `git log -S` would have caught it
   instantly instead of wasting hours on an OOB rewrite.
10. **The Kernel Manager handles OOB dispatch.** The orchestrator writes targets to
    `work_queue.jsonl` and continues the DFS loop. The manager runs asynchronously
    in tmux pane 2, handling prompt engineering, local testing, and patch generation.
    The orchestrator only blocks when applying a merge-op from `results.jsonl`.
11. **Crashes are diagnostic gold.** A segfault after 1.45x micro-speedup means
    the optimization direction was right but hit a constraint (register spill, OOB
    access). The Watchdog investigates and produces constraints for retry.
12. **Accumulated context across OOB rounds prevents repeat mistakes.** The Kernel
    Manager's session_history ensures round 3 knows what rounds 1-2 tried and why
    they failed. This is more valuable than a fresh prompt.
