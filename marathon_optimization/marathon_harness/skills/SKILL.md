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

**Execute these steps in order.**

```
0. WARM-START     → actions/setup.md       — ingest Sprint handoff or pre-optimized baseline
1. RE-PROFILE     → actions/profile.md     — fresh trace on the optimized baseline
2. DEEP ANALYSIS  → actions/deep-kernel-analysis.md — per-kernel dispatch tracing + variant discovery
3. BUILD STACK    → score Marathon-specific actions, push highest-first
4. DFS LOOP       → pop → execute → measure → re-score → dream every 3-4h → repeat
5. SWEEP          → actions/sweep.md       — extended sweep on deeply-optimized config
6. REPORT         → actions/report.md      — final report + KB contribution
7. DREAM          → final consolidation, cross-run KB contribution
```

### Step 0: Warm-Start

Two modes (see `actions/setup.md`):

- **Mode A: Sprint handoff** — read `handoff/config.json`, apply `handoff/patches/`,
  load `handoff/opportunities.json` as pre-scored Marathon action candidates.
- **Mode B: Pre-optimized directory** — parse launch script, extract config, set as baseline.

Both modes skip basic setup and go directly to re-profile.

### Step 2: Deep Analysis

For every kernel consuming >MIN_GPU_PCT of GPU time, run `actions/deep-kernel-analysis.md`:

1. **Dispatch path trace** — Python call chain → compiled extension → GPU kernel
2. **Variant discovery** — search for alternative implementations, check platform branching
3. **Config verification** — check shape-specific tuning configs vs generic defaults
4. **Build system trace** — how is the kernel compiled? Patch vs rebuild required?
5. **Opportunity report** — classifies each kernel as `self-fix` or `oob-rewrite`:
   - **Self-fix** (dispatch bugs, one-line routing changes): orchestrator applies directly
   - **OOB-rewrite** (kernel rewrites, multi-file changes): written to the Kernel Manager's
     work queue at `$RESULT_DIR/kernel_manager/work_queue.jsonl`

### Step 4: The DFS Loop (Marathon Core)

```
WHILE action_stack is not empty AND NOT stopping_criteria_met():

  a. Pop highest-scored action from action_stack
  b. IF action is a self-fix dispatch bug:
       → Apply fix directly (git archaeology + patch + test)
       → No Kernel Manager involvement needed
  c. IF action is a deep-kernel-opt target:
       → Write to $RESULT_DIR/kernel_manager/work_queue.jsonl
       → The Kernel Manager (tmux pane 2) processes it asynchronously
       → Continue DFS loop with other actions
  d. Execute the action (dispatch to actions/*.md)
  e. ACCURACY GATE: if accuracy_risk > 0 → run eval, revert if drop > threshold
  f. Measure: new_tput_per_gpu
  g. Update state: current_tput_per_gpu, cumulative_gain_pct
  h. RE-SCORE all remaining actions on the stack
  i. Push new sub-actions discovered during execution
  j. Log to completed_actions
  k. KB ingest

  ** MERGE-OP POLL: ** Between DFS actions, check results.jsonl for completed
     kernel optimizations from the Kernel Manager. For each merge-ready result:
     → Push a `merge-op` action onto the stack with score 9 (high priority)
     → merge-op: kill server → apply patch → rebuild if needed → restart →
       E2E benchmark → KEEP/REVERT
     → On REVERT or crash during merge-op:
       write event to event_log.jsonl (Watchdog will investigate)

  ** FINDINGS POLL: ** Between DFS actions, check findings.jsonl for Watchdog
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
def poll_kernel_results(result_dir, last_seen_id=None):
    """Check for new merge-ready results from the Kernel Manager."""
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
            if result["status"] == "merge-ready":
                if last_seen_id is None or result["id"] > last_seen_id:
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
    "completed_actions": [],
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
