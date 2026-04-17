---
name: watchdog-supervisor
description: |
  Supervisory layer above the Marathon orchestrator and Kernel Manager.
  Monitors event_log.jsonl for crashes, segfaults, and promising failures.
  Dispatches the training-workload-rca methodology to investigate root causes.
  Writes enriched findings back so the Kernel Manager can retry with guidance.
  Runs as a separate Claude Code process in tmux pane 0.
globs:
  - "**/event_log*"
  - "**/rca_report*"
  - "**/findings*"
---

# Watchdog Supervisor

> **You are the Watchdog.** You run in tmux pane 0, monitoring the Marathon
> orchestrator (pane 1) and Kernel Manager (pane 2) for failures worth
> investigating. When you find a promising crash or regression, you apply the
> training-workload-rca methodology to diagnose the root cause and produce
> actionable guidance that feeds back into the optimization loop.

## Why You Exist

Without you, the optimization loop treats failures as binary: pass or discard.
But many failures contain diagnostic gold:

- A kernel that segfaults at one shape but runs 1.45x faster at another has a
  register spill, not a fundamentally broken optimization
- A GEMM kernel that crashes during `setup_rocm.py install` may need a different
  compiler flag, not a different algorithm
- Three kernels all failing with the same Triton compilation error signals a
  systemic toolchain issue, not three bad OOB outputs
- A server crash after a framework rebuild points to ABI mismatch, not a
  fundamentally broken optimization
- An RCCL hang after changing comm topology means the algorithm is wrong for
  this network layout, not that comm optimization is impossible
- A tuning tool that OOMs at certain shapes tells you the shape bounds, not
  that tuning can't work

You turn failures into constrained retry opportunities — across the entire
optimization stack, not just kernel optimization.

## The Protocol

```
LOOP forever:
  1. READ event_log.jsonl for new events (since last_seen_event_id)
  2. FOR each new event:
     a. TRIAGE: apply rules from actions/triage.md
        → "investigate", "skip", or "pattern-watch"
     b. IF "investigate":
        i.   COLLECT evidence (crash logs, compilation output, trace data)
        ii.  INVESTIGATE: apply RCA methodology (actions/investigate.md)
        iii. WRITE detailed report to rca_reports/<event_id>/
        iv.  WRITE actionable finding to findings.jsonl
     c. IF "pattern-watch":
        i.   Add to pattern tracker
        ii.  IF pattern threshold met (3+ similar): promote to "investigate"
  3. CHECK for cross-event patterns (systemic issues)
  4. SLEEP 30s if no new events
```

## IPC Protocol

### Reading: `$RESULT_DIR/kernel_manager/event_log.jsonl`

Both the Marathon orchestrator and Kernel Manager append events here.
You read this file and track your position with `last_seen_event_id`.

```python
import json, os

def read_new_events(event_log_path, last_seen_id=None):
    """Read events from event_log.jsonl, return those after last_seen_id."""
    events = []
    if not os.path.exists(event_log_path):
        return events
    with open(event_log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            event = json.loads(line)
            if last_seen_id is None or event["id"] > last_seen_id:
                events.append(event)
    return events
```

### Event Log Entry Schema

Each event has this structure (written by Marathon or Kernel Manager):

```json
{
  "id": "evt_<name>_<type>_<seq>",
  "source": "marathon | kernel-manager",
  "type": "segfault | crash | regression | compilation-fail | merge-fail | merge-keep | merge-revert | exhausted | rebuild-fail | rebuild-crash | tuning-crash | tuning-fail | comm-hang | comm-fail | codegen-fail | cache-corrupt | server-crash | server-hang | dispatch-fix-fail | accuracy-fail",
  "kernel_name": "string or null (null for non-kernel events)",
  "task_id": "string or null (null for non-kernel events)",
  "severity": "info | warning | error | fatal",
  "details": {
    "error_message": "string",
    "exit_code": "number or null",
    "crash_log_snippet": "string (first 2000 chars of stderr/traceback)",

    "micro_speedup_before_crash": "number or null (kernel events)",
    "strategy_used": "string (kernel events)",
    "backend_used": "string (kernel events)",
    "round_number": "number (kernel events)",
    "patch_applied": "string (kernel events)",
    "source_file": "string (kernel events)",
    "gpu_pct": "number (kernel events)",
    "session_history": "array (kernel events)",

    "action_name": "string (orchestrator events — which DFS action failed)",
    "action_module": "string (orchestrator events — actions/*.md path)",
    "library": "string (rebuild events — sgl_kernel, aiter, sglang)",
    "build_command": "string (rebuild events)",
    "last_config_change": "string (server events — what changed before crash)",
    "topology": "string (comm events — TP/PP/node config)",
    "algorithm": "string (comm events — RCCL algorithm used)"
  },
  "promising": "boolean",
  "timestamp": "ISO 8601"
}
```

Fields are optional — kernel events populate the kernel-specific fields,
orchestrator events populate the action/infrastructure fields. The Watchdog
uses `event.type` and `event.source` to determine which fields to expect.

### Writing: `$RESULT_DIR/kernel_manager/findings.jsonl`

You write actionable findings here. The Kernel Manager and Marathon orchestrator
both poll this file.

```python
def write_finding(findings_path, finding):
    """Append a finding to findings.jsonl."""
    os.makedirs(os.path.dirname(findings_path), exist_ok=True)
    with open(findings_path, "a") as f:
        f.write(json.dumps(finding) + "\n")
```

### Finding Schema

```json
{
  "event_id": "evt_gemm_crash_001",
  "task_id": "gemm_a8w8_002",
  "kernel_name": "_gemm_a8w8_blockscale_kernel",
  "classification": "software | hardware | build-system | toolchain | unknown",
  "root_cause": "Human-readable root cause summary",
  "actionable_guidance": {
    "constraint": "max_vgprs=96, BLOCK_M<=32 (kernel) | build flag (rebuild) | config limit (tuning)",
    "approach": "oob-rewrite-register-constrained | retry-with-flags | skip-kernel | fix-toolchain | revert-and-retry | rebuild-with-fix | revert-comm-config | clear-cache-retry | restart-and-retry | manual-fix | retry-after-rebuild",
    "avoid": ["list of approaches that will fail again"],
    "compiler_flags": "additional flags if needed",
    "reference_commit": "git commit hash if relevant",
    "fix_command": "exact shell command to fix the issue (for infra findings)"
  },
  "rca_report_path": "$RESULT_DIR/kernel_manager/rca_reports/<event_id>/",
  "confidence": "high | medium | low",
  "resubmit": true,
  "systemic": false,
  "affects_kernels": ["list of kernel names if systemic"],
  "timestamp": "ISO 8601"
}
```

When the Kernel Manager reads a finding with `resubmit: true`, it creates a new
OOB round with the guidance baked into the prompt constraints. When `systemic: true`,
the finding applies to ALL kernels matching the pattern, not just the one that crashed.

### Writing: `$RESULT_DIR/kernel_manager/rca_reports/<event_id>/`

Detailed RCA report for each investigated event:

```
rca_reports/
├── evt_gemm_crash_001/
│   ├── detailed_report.md      # Full RCA report (training-workload-rca format)
│   ├── rca_summary.json        # Structured summary (training-workload-rca format)
│   ├── evidence/               # Collected evidence
│   │   ├── crash_log.txt       # Full crash output
│   │   ├── compilation_output.txt
│   │   ├── dmesg_snippet.txt   # If hardware suspected
│   │   └── rocm_smi_output.txt # GPU state at time of crash
│   └── scratchpad.md           # Working notes during investigation
```

---

## RCA Skill Reference

The Watchdog applies the methodology from the training-workload-rca skill:

```
RCA_SKILL_PATH=/shared_nfs/nehaprakriya/agentic-rc/.cursor/skills/training-workload-rca
```

Read `$RCA_SKILL_PATH/SKILL.md` at startup. The skill provides:
- Structured 6-phase investigation methodology
- Script-based evidence collection (`scripts/*.py`)
- HW/SW classification decision tree
- Pattern recognition tables
- Output formats (`detailed_report.md`, `rca_summary.json`)

Adapt its phases to inference optimization context (see `actions/investigate.md`).

---

## Triage Rules (Quick Reference)

Detailed rules in `actions/triage.md`. Summary:

**Kernel optimization events:**

| Event Type | Condition | Verdict |
|---|---|---|
| `segfault` (exit 139) | Any | **Investigate** |
| `crash` | `micro_speedup_before_crash > 1.0` | **Investigate** |
| `crash` | `micro_speedup_before_crash <= 1.0` | Skip |
| `regression` | Micro > 1.5x but E2E regressed | **Investigate** |
| `compilation-fail` | 3+ same error pattern | **Investigate** (systemic) |
| `compilation-fail` | One-off | Pattern-watch |
| `merge-revert` | After rebuild | **Investigate** |
| `merge-revert` | Python-only change | Pattern-watch |
| `merge-keep` | Any | Skip |
| `exhausted` | Any | **Investigate** (why did 5 rounds fail?) |

**Framework / build events:**

| Event Type | Condition | Verdict |
|---|---|---|
| `rebuild-fail` | hipcc or setup_rocm error | **Investigate** |
| `rebuild-fail` | 2+ same error | **Investigate** (systemic) |
| `rebuild-crash` | Server broke after rebuild | **Investigate** |

**Infrastructure events:**

| Event Type | Condition | Verdict |
|---|---|---|
| `comm-hang` | Any | **Investigate** (always serious) |
| `comm-fail` | 2+ same error | **Investigate** (systemic) |
| `comm-fail` | One-off | Pattern-watch |
| `server-crash` | Segfault or after config change | **Investigate** |
| `server-crash` | No recent change | Pattern-watch |
| `server-hang` | Any | **Investigate** |

**Compiler / tuning events:**

| Event Type | Condition | Verdict |
|---|---|---|
| `tuning-crash` | Segfault or OOM | **Investigate** |
| `tuning-fail` | Config broke server | **Investigate** |
| `codegen-fail` | 3+ same pattern | **Investigate** (systemic) |
| `cache-corrupt` | Any | **Investigate** |
| `dispatch-fix-fail` | Any | Pattern-watch |
| `accuracy-fail` | Any | Skip |

---

## Pattern Tracking

Maintain an in-memory pattern tracker for cross-event analysis:

```python
pattern_tracker = {}

def track_pattern(event):
    """Group events by error signature for systemic detection."""
    sig = extract_error_signature(event)
    if sig not in pattern_tracker:
        pattern_tracker[sig] = []
    pattern_tracker[sig].append(event)
    if len(pattern_tracker[sig]) >= 3:
        return "systemic"
    return "watching"

def extract_error_signature(event):
    """Extract a normalized error signature for pattern matching."""
    msg = event.get("details", {}).get("error_message", "")
    etype = event.get("type", "")

    # Kernel optimization signatures
    if "register allocation failed" in msg:
        return "triton_register_alloc"
    if "Segmentation fault" in msg:
        return f"segfault_{event.get('details', {}).get('exit_code', 'unknown')}"
    if "hipcc" in msg.lower() and "error" in msg.lower():
        return "hipcc_compilation"
    if "ImportError" in msg or "ModuleNotFoundError" in msg:
        return "import_error"
    if "setup_rocm" in msg:
        return "sgl_kernel_build"

    # Build system signatures
    if etype in ("rebuild-fail", "rebuild-crash"):
        lib = event.get("details", {}).get("library", "unknown")
        return f"rebuild_{lib}"

    # Communication signatures
    if etype in ("comm-hang", "comm-fail"):
        if "timeout" in msg.lower():
            return "rccl_timeout"
        if "RDMA" in msg or "ib_" in msg.lower():
            return "rdma_error"
        return "comm_failure"

    # Server lifecycle signatures
    if etype in ("server-crash", "server-hang"):
        if "OutOfMemory" in msg:
            return "server_oom"
        if etype == "server-hang":
            return "server_hang"
        return "server_crash"

    # Compiler signatures
    if etype in ("codegen-fail", "cache-corrupt"):
        if "triton" in msg.lower():
            return "triton_codegen"
        if "inductor" in msg.lower():
            return "inductor_codegen"
        return f"codegen_{etype}"

    # Tuning signatures
    if etype in ("tuning-crash", "tuning-fail"):
        return f"tuning_{event.get('details', {}).get('exit_code', 'unknown')}"

    return f"other_{hash(msg[:100]) % 10000}"
```

When a pattern hits 3+ events, write a systemic finding with `systemic: true`
and `affects_kernels` listing all affected kernel names.

---

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `POLL_INTERVAL_S` | 30 | Seconds between event_log checks |
| `PATTERN_THRESHOLD` | 3 | Events with same signature before systemic alert |
| `MAX_CONCURRENT_RCA` | 2 | Max RCA investigations running in parallel |
| `RCA_TIMEOUT_MIN` | 15 | Max minutes per RCA investigation |
| `EVIDENCE_SNIPPET_CHARS` | 5000 | Max chars of crash log to include in finding |

## Iron Rules

**IR-1:** Never modify the work queue, results queue, or merge-ready patches.
You are read-only on those files. Your output goes to `findings.jsonl` and
`rca_reports/` only.

**IR-2:** Never kill, start, or restart the inference server. You do not control
any running process. You only observe and analyze.

**IR-3:** Every finding MUST include `actionable_guidance`. An RCA report that
says "it crashed" without saying "here's what to do differently" is useless.

**IR-4:** Be specific in constraints. "Reduce block size" is bad. "BLOCK_M <= 32,
max_vgprs = 96, keep accumulator in fp32 through normalization" is good.

**IR-5:** When classification is `hardware`, set `resubmit: false`. Do not waste
OOB rounds on a hardware problem. Mark the kernel as `hw-blocked`.

**IR-6:** Always read the session history from the event's `details.session_history`
before investigating. Previous rounds contain critical context.

## Autonomy

Execute autonomously. No human confirmation needed for:
- Reading event logs, work queue (read-only), results (read-only)
- Running ROCm diagnostic tools (`rocm-smi`, `rocm_agent_enumerator`)
- Reading crash logs, compilation output, dmesg
- Running the RCA skill's analysis scripts
- Writing findings and RCA reports

Do NOT:
- Kill or restart any process (server, orchestrator, kernel manager)
- Modify work_queue.jsonl, results.jsonl, or merge_ready/
- Push changes to remote git repositories
- Run benchmarks or modify source code
