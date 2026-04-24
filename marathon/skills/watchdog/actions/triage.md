# Action: Triage Events

Determines whether an event from `event_log.jsonl` warrants a full RCA investigation,
should be tracked for pattern detection, or can be safely skipped.

## Decision Tree

For each new event, walk through this tree top-to-bottom. First matching rule wins.

### Level 1: Event Type

```
event.type
  │
  │ ── Kernel optimization events ──
  ├── "segfault"            → Level 2A (always interesting)
  ├── "crash"               → Level 2B (check if promising)
  ├── "regression"          → Level 2C (check if occupancy-related)
  ├── "compilation-fail"    → Level 2D (check for patterns)
  ├── "merge-revert"        → Level 2E (check what was reverted)
  ├── "merge-fail"          → Level 2E (same as revert)
  ├── "exhausted"           → Level 2F (all OOB rounds failed)
  ├── "merge-keep"          → SKIP (success)
  │
  │ ── Framework / build events ──
  ├── "rebuild-fail"        → Level 2G (build system failure)
  ├── "rebuild-crash"       → Level 2G (server broke after rebuild)
  │
  │ ── Operator tuning events ──
  ├── "tuning-crash"        → Level 2H (tuning tool crashed)
  ├── "tuning-fail"         → Level 2H (invalid config produced)
  │
  │ ── Communication events ──
  ├── "comm-hang"           → Level 2I (RCCL/NCCL hang)
  ├── "comm-fail"           → Level 2I (communication error)
  │
  │ ── Compiler events ──
  ├── "codegen-fail"        → Level 2J (bad codegen)
  ├── "cache-corrupt"       → Level 2J (cache corruption)
  │
  │ ── Server lifecycle events ──
  ├── "server-crash"        → Level 2K (server died unexpectedly)
  ├── "server-hang"         → Level 2K (server unresponsive)
  │
  │ ── Other ──
  ├── "dispatch-fix-fail"   → Level 2L (fast-path fix failed)
  ├── "accuracy-fail"       → SKIP (clean failure, handled by orchestrator)
  └── unknown               → PATTERN-WATCH (log it, investigate if recurs)
```

### Level 2A: Segfault (exit code 139)

Segfaults always deserve investigation. They indicate either a hardware fault or
a specific software bug that RCA can diagnose.

```
event.type == "segfault" OR event.details.exit_code == 139
  │
  ├── event.details.micro_speedup_before_crash > 1.0
  │   → INVESTIGATE (priority: HIGH)
  │     Reason: optimization was producing gains before crash.
  │     The root cause is likely register spill, stack overflow,
  │     or out-of-bounds memory access at specific shapes.
  │     RCA should produce register/shape constraints.
  │
  ├── event.details.session_history has prior successful rounds
  │   → INVESTIGATE (priority: HIGH)
  │     Reason: earlier rounds worked, this round broke something.
  │     Compare the failing round's changes against prior successes.
  │
  └── otherwise
      → INVESTIGATE (priority: MEDIUM)
        Reason: segfaults may indicate hardware issues.
        RCA should check ECC errors, dmesg, GPU state.
```

### Level 2B: Crash (non-segfault)

```
event.type == "crash" AND event.details.exit_code != 139
  │
  ├── event.details.micro_speedup_before_crash > 1.0
  │   → INVESTIGATE (priority: HIGH)
  │     Reason: promising optimization crashed — worth saving.
  │
  ├── event.details.exit_code == 134 (SIGABRT)
  │   → INVESTIGATE (priority: MEDIUM)
  │     Reason: assertion failure or double-free. The RCA classification
  │     tree handles these well (check for heap corruption vs library bug).
  │
  ├── event.details.micro_speedup_before_crash <= 1.0
  │   AND event.details.round_number <= 1
  │   → SKIP
  │     Reason: first-round crash with no improvement — normal OOB noise.
  │
  └── event.details.round_number >= 3
      → PATTERN-WATCH
        Reason: persistent crash across rounds suggests a deeper issue
        but may just be a hard kernel. Track for systemic patterns.
```

### Level 2C: Regression

```
event.type == "regression"
  │
  ├── event.details.micro_speedup_before_crash > 1.5
  │   AND E2E regressed
  │   → INVESTIGATE (priority: HIGH)
  │     Reason: classic register pressure / occupancy drop.
  │     RCA should determine VGPR count, occupancy, and produce
  │     explicit register constraints for retry.
  │
  ├── event.details.micro_speedup_before_crash > 1.05
  │   AND E2E regressed
  │   → INVESTIGATE (priority: MEDIUM)
  │     Reason: moderate improvement lost to system effects.
  │     RCA should check CUDA graph capture, scheduling impact.
  │
  └── otherwise
      → SKIP
        Reason: marginal improvement that regressed — not worth deep analysis.
```

### Level 2D: Compilation Failure

```
event.type == "compilation-fail"
  │
  ├── pattern_tracker[error_signature] >= 3
  │   → INVESTIGATE (priority: HIGH, systemic: true)
  │     Reason: same compilation error across 3+ kernels.
  │     This is a toolchain or environment issue, not a per-kernel problem.
  │     RCA should diagnose the build system.
  │
  ├── "register allocation" in error_message
  │   → INVESTIGATE (priority: MEDIUM)
  │     Reason: Triton register allocation failure is actionable.
  │     RCA can determine the register budget and produce constraints.
  │
  ├── "hipcc" in error_message
  │   → PATTERN-WATCH
  │     Reason: HIP compilation errors may indicate flag issues.
  │     Track and investigate if pattern emerges.
  │
  └── otherwise
      → PATTERN-WATCH
        Reason: one-off compilation failures are common with OOB outputs.
        Track for patterns but don't investigate individual failures.
```

### Level 2E: Merge Revert / Merge Fail

```
event.type in ("merge-revert", "merge-fail")
  │
  ├── event.details.get("rebuild_required") == true
  │   → INVESTIGATE (priority: HIGH)
  │     Reason: compiled extension change failed at integration.
  │     Could be build system issue, ABI mismatch, or wrong flags.
  │
  ├── server crashed after patch application
  │   → INVESTIGATE (priority: HIGH)
  │     Reason: the patch broke the server — need to understand why
  │     so the Kernel Manager doesn't produce similar patches.
  │
  ├── E2E regression without crash
  │   → PATTERN-WATCH
  │     Reason: performance regression at E2E is common and usually
  │     means the micro-benchmark didn't capture system effects.
  │
  └── accuracy gate failure
      → SKIP
        Reason: accuracy failures are clean — the patch affected
        numerical output. The Kernel Manager handles this via
        correctness testing.
```

### Level 2F: Exhausted (all OOB rounds failed)

```
event.type == "exhausted"
  │
  ├── event.details.gpu_pct > 5.0
  │   → INVESTIGATE (priority: MEDIUM)
  │     Reason: high-impact kernel that all backends failed on.
  │     RCA should analyze WHY all attempts failed and whether
  │     a different approach (strategy change) would help.
  │
  └── event.details.gpu_pct <= 5.0
      → SKIP
        Reason: low-impact kernel not worth further investigation.
```

### Level 2G: Framework Rebuild Failure

```
event.type in ("rebuild-fail", "rebuild-crash")
  │
  ├── event.type == "rebuild-crash" (server broke after rebuild)
  │   → INVESTIGATE (priority: HIGH)
  │     Reason: a rebuild that passes compilation but breaks the server
  │     indicates ABI mismatch, missing symbol, or wrong compiler flags.
  │     RCA should check build output, library versions, and ABI compatibility.
  │
  ├── event.type == "rebuild-fail"
  │   AND "hipcc" in error_message
  │   → INVESTIGATE (priority: MEDIUM)
  │     Reason: HIP compilation failure during rebuild may indicate
  │     wrong GPU target, missing headers, or ROCm version mismatch.
  │
  ├── event.type == "rebuild-fail"
  │   AND "setup_rocm" in error_message
  │   → INVESTIGATE (priority: MEDIUM)
  │     Reason: sgl_kernel build system failure. Check if the build
  │     script changed or if a new kernel has unsupported constructs.
  │
  ├── pattern_tracker[error_signature] >= 2
  │   → INVESTIGATE (priority: HIGH, systemic: true)
  │     Reason: repeated rebuild failures suggest an environment issue,
  │     not a one-off problem.
  │
  └── otherwise (e.g., pip install failure)
      → PATTERN-WATCH
        Reason: one-off build issues often resolve on retry.
```

### Level 2H: Operator Tuning Failure

```
event.type in ("tuning-crash", "tuning-fail")
  │
  ├── event.type == "tuning-crash"
  │   AND event.details.exit_code == 139
  │   → INVESTIGATE (priority: HIGH)
  │     Reason: tuning tool segfault — may indicate GPU issue or
  │     malformed kernel that the tuner can't handle.
  │
  ├── event.type == "tuning-crash"
  │   AND "OutOfMemory" in error_message
  │   → INVESTIGATE (priority: MEDIUM)
  │     Reason: tuning tool exhausted GPU memory. RCA should determine
  │     which shapes/configs caused the OOM and produce shape limits.
  │
  ├── event.type == "tuning-fail"
  │   AND server crashed on config load
  │   → INVESTIGATE (priority: HIGH)
  │     Reason: tuning produced an invalid config that breaks the server.
  │     RCA should identify which config values are invalid.
  │
  └── event.type == "tuning-fail" (config didn't improve perf)
      → SKIP
        Reason: tuning didn't find better config — normal outcome.
```

### Level 2I: Communication Failure

```
event.type in ("comm-hang", "comm-fail")
  │
  ├── event.type == "comm-hang"
  │   → INVESTIGATE (priority: HIGH)
  │     Reason: RCCL/NCCL hangs are serious — they can indicate network
  │     topology issues, dead links, or algorithm incompatibility.
  │     RCA should use RCCL debug logs, check topology, inspect RDMA errors.
  │     This maps directly to training-workload-rca's infra deep dive.
  │
  ├── event.type == "comm-fail"
  │   AND "timeout" in error_message.lower()
  │   → INVESTIGATE (priority: HIGH)
  │     Reason: collective timeout indicates a participating rank is stuck.
  │
  ├── event.type == "comm-fail"
  │   AND pattern_tracker[error_signature] >= 2
  │   → INVESTIGATE (priority: HIGH, systemic: true)
  │     Reason: repeated comm failures suggest infrastructure problem.
  │
  └── event.type == "comm-fail" (one-off)
      → PATTERN-WATCH
        Reason: isolated comm errors may be transient. Track for patterns.
```

### Level 2J: Compiler / Codegen Failure

```
event.type in ("codegen-fail", "cache-corrupt")
  │
  ├── event.type == "cache-corrupt"
  │   → INVESTIGATE (priority: MEDIUM)
  │     Reason: cache corruption can cause cascading failures across
  │     multiple kernels. RCA should determine which caches are affected
  │     (Triton, Inductor, __pycache__) and whether clearing fixes it.
  │     Produce systemic finding if multiple kernels affected.
  │
  ├── event.type == "codegen-fail"
  │   AND same codegen error across 3+ kernels
  │   → INVESTIGATE (priority: HIGH, systemic: true)
  │     Reason: Inductor or Triton is generating bad code systematically.
  │     This is a toolchain issue, not a per-kernel problem.
  │
  ├── event.type == "codegen-fail" (single kernel)
  │   → PATTERN-WATCH
  │     Reason: one kernel triggering a codegen edge case is common.
  │
  └── otherwise
      → PATTERN-WATCH
```

### Level 2K: Server Lifecycle Failure

```
event.type in ("server-crash", "server-hang")
  │
  ├── event.type == "server-crash"
  │   AND event.details.exit_code == 139
  │   → INVESTIGATE (priority: HIGH)
  │     Reason: server segfault — could be GPU issue, corrupted model,
  │     or broken library after a recent change.
  │
  ├── event.type == "server-crash"
  │   AND "OutOfMemory" in error_message
  │   → INVESTIGATE (priority: MEDIUM)
  │     Reason: OOM during model loading or inference. Check if a recent
  │     config change increased memory usage (larger batch, different TP).
  │
  ├── event.type == "server-crash"
  │   AND event.details.get("last_config_change") is not None
  │   → INVESTIGATE (priority: HIGH)
  │     Reason: server crashed after a config change — RCA should
  │     determine if the change caused the crash.
  │
  ├── event.type == "server-hang"
  │   → INVESTIGATE (priority: MEDIUM)
  │     Reason: server unresponsive — could be GPU deadlock, RCCL hang,
  │     or infinite loop. Check if GPU utilization is 0% (deadlock)
  │     or 100% (stuck kernel).
  │
  └── event.type == "server-crash" (no recent change)
      → PATTERN-WATCH
        Reason: sporadic server crashes happen. Track for patterns.
```

### Level 2L: Dispatch Fix Failure

```
event.type == "dispatch-fix-fail"
  │
  ├── event.details.get("fix_type") == "git-revert"
  │   → INVESTIGATE (priority: MEDIUM)
  │     Reason: attempted to restore code from git history but it
  │     didn't work. RCA should check if the surrounding code changed
  │     too, making the old code incompatible.
  │
  └── otherwise
      → PATTERN-WATCH
        Reason: dispatch fixes are usually straightforward. If one
        fails, it may indicate the dispatch analysis was wrong.
```

---

## Output Format

The triage function returns a verdict object:

```python
def triage(event, pattern_tracker):
    """Triage an event. Returns verdict dict."""
    verdict = {
        "event_id": event["id"],
        "action": "investigate" | "skip" | "pattern-watch",
        "priority": "high" | "medium" | "low",
        "systemic": False,
        "reason": "string explaining the decision",
    }
    return verdict
```

## Priority Queue

When multiple events need investigation, process in this order:

1. **HIGH priority + promising** (segfault with micro improvement) — these directly
   unblock better optimization attempts
2. **HIGH priority + systemic** (compilation pattern across 3+ kernels) — these
   affect the entire optimization session
3. **MEDIUM priority** — investigate when no HIGH items pending
4. **Pattern promotions** — events that crossed the pattern threshold

Never investigate more than `MAX_CONCURRENT_RCA` (2) events simultaneously.
Queue excess investigations and process them FIFO within priority bands.
