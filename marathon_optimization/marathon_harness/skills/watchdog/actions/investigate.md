# Action: Investigate (RCA for Inference Optimization Failures)

Applies the `training-workload-rca` methodology to ALL inference optimization failures —
kernel crashes, framework rebuild failures, communication hangs, server lifecycle issues,
and more. Adapts each RCA phase from training context to inference optimization context.

## Reference Skill

```
RCA_SKILL_PATH=/shared_nfs/nehaprakriya/agentic-rc/.cursor/skills/training-workload-rca
```

Read `$RCA_SKILL_PATH/SKILL.md` before your first investigation. Use its structured
methodology but adapt the phases as described below.

---

## Phase Mapping: Training RCA → Inference RCA

| Training RCA Phase | Inference Adaptation |
|---|---|
| Phase 1: Discovery | Read event context — kernel target OR action details; optimization history; crash context |
| Phase 2: Log Analysis | Parse crash output (compiler stderr, server traceback, HIP errors, Triton errors, RCCL logs, build logs) |
| Phase 3: Metrics Analysis | Read trace data, micro-benchmark results, rocm-smi GPU state, server resource usage |
| Phase 4: HW/SW Classification | Same decision tree — crash address analysis, ECC checks, stack trace analysis |
| Phase 5: Infrastructure Deep Dive | ROCm tools, RCCL debug, network diagnostics, build system audit |
| Phase 6: Root Cause Synthesis | Produce actionable finding — constraints for kernel retry OR fix instructions for infra issues |

---

## Phase 1: Discovery — Gather Context

### 1.1 Read the event

```python
event = current_event  # from triage
kernel_name = event.get("kernel_name")  # may be null for non-kernel events
task_id = event.get("task_id")          # may be null for non-kernel events
source = event["source"]
event_type = event["type"]
action_name = event.get("details", {}).get("action_name")  # for orchestrator events
```

### 1.2 Gather context based on event source

**For kernel events** (source=kernel-manager, task_id is set):

```bash
grep "$task_id" "$RESULT_DIR/kernel_manager/work_queue.jsonl"
```

Extract: `source_file`, `strategy`, `dispatch_analysis`, `gpu_pct`, `shapes`.

**For orchestrator events** (source=marathon, action_name is set):

Read the event's `details.action_name` and `details.action_module` to understand
what the orchestrator was doing when it failed. Key fields:
- `details.library` — which library was being rebuilt (for rebuild events)
- `details.build_command` — exact build command that failed
- `details.last_config_change` — what changed before the failure
- `details.crash_log_snippet` — the actual error output

### 1.3 Read session history

The event's `details.session_history` contains summaries of all prior optimization
rounds for this kernel. This is critical context:

```
Session history example:
  Round 1 (codex): COMPILE_FAIL — register allocation failed at BLOCK_M=128
  Round 2 (claude): CORRECTNESS_FAIL — max diff=0.15 (tolerance=0.01)
  Round 3 (geak):   SEGFAULT — exit 139, micro speedup was 1.45x at shapes [1,16384]
```

### 1.4 Read related findings

Check if the watchdog already produced findings for related kernels:

```bash
grep "$kernel_name" "$RESULT_DIR/kernel_manager/findings.jsonl"
```

---

## Phase 2: Log Analysis — Parse Crash Output

### 2.1 Identify the crash artifact

| Event Type | Where to Find Logs |
|---|---|
| `segfault` | `details.crash_log_snippet`, server stderr, `dmesg` |
| `crash` | `details.crash_log_snippet`, Python traceback |
| `compilation-fail` | `details.error_message` (compiler stderr) |
| `regression` | Micro-benchmark logs, trace data |
| `merge-revert` | Server restart logs, E2E benchmark output |
| `merge-fail` | Build output, server crash log |
| `rebuild-fail` | `details.crash_log_snippet` (build stderr), `details.build_command` |
| `rebuild-crash` | Server crash log after restart, `details.library` |
| `tuning-crash` | `details.crash_log_snippet` (tuning tool stderr) |
| `tuning-fail` | `details.error_message` (config load error) |
| `comm-hang` | RCCL debug logs (set `NCCL_DEBUG=INFO`), network `dmesg` |
| `comm-fail` | RCCL error output, `details.error_message` |
| `codegen-fail` | Inductor/Triton compilation output, `~/.triton/cache/` |
| `cache-corrupt` | `details.error_message`, cache directory listing |
| `server-crash` | Server stderr, `dmesg`, `details.last_config_change` |
| `server-hang` | `rocm-smi` output (GPU util 0%=deadlock, 100%=stuck kernel) |
| `dispatch-fix-fail` | `details.error_message`, git log/show output |
| `accuracy-fail` | Eval output, accuracy diff |

### 2.2 Error pattern extraction

Parse the crash output for diagnostic signatures:

#### Triton Compilation Errors

| Pattern | Meaning | Constraint to Produce |
|---|---|---|
| `register allocation failed` | Block sizes exceed VGPR budget | `max_vgprs=N, reduce BLOCK_M/BLOCK_N` |
| `CompilationError: invalid LLVM IR` | Malformed Triton code | `avoid: [specific construct]` |
| `KeyError: 'xnumel'` | Missing grid dimension | `constraint: preserve grid dims` |
| `cannot use store to ptr with different type` | Type mismatch in store | `constraint: match pointer types` |
| `triton.compiler.errors.CompilationError` | General Triton error | Parse specific sub-error |

#### HIP/ROCm Errors

| Pattern | Meaning | Constraint to Produce |
|---|---|---|
| `hipErrorNoBinaryForGpu` | Wrong GPU target | `compiler_flags: --amdgpu-target=gfx950` |
| `hipErrorOutOfMemory` | LDS or global memory exceeded | `constraint: reduce shared memory, smaller tiles` |
| `LLVM ERROR: out of memory` | Compiler ran out of memory | `constraint: simplify kernel, reduce inlining` |
| `error: undefined identifier` | Missing HIP symbol | Check ROCm version, add missing headers |
| `Segmentation fault` in `hipcc` | Compiler crash | `constraint: simplify code, try -O2 instead of -O3` |

#### Python/Framework Errors

| Pattern | Meaning | Constraint to Produce |
|---|---|---|
| `ImportError: cannot import name 'X' from 'sgl_kernel'` | Missing or renamed export | Check git history for rename, fix import |
| `RuntimeError: CUDA error: illegal memory access` | Out-of-bounds GPU access | Analyze index computation, check bounds |
| `torch.cuda.OutOfMemoryError` | GPU memory exceeded | Reduce batch/tile sizes |
| `AssertionError` | Violated invariant | Check assertion message for constraint |

#### Build System Errors

| Pattern | Meaning | Guidance to Produce |
|---|---|---|
| `setup_rocm.py` + `hipcc error` | sgl_kernel build failed | Check GPU target, ROCm version, missing headers |
| `undefined reference to` | Linker error — missing symbol | Check library version, rebuild order |
| `error: 'X' was not declared` | C++ source incompatible with headers | Check ROCm/HIP header version |
| `pip install` + `CalledProcessError` | Python package build failed | Check build deps, wheel compatibility |
| `ModuleNotFoundError` after rebuild | Editable install path broken | `pip install -e .` to refresh |

#### Communication / RCCL Errors

| Pattern | Meaning | Guidance to Produce |
|---|---|---|
| `NCCL WARN` + `Timeout` | Collective op timed out | Check if all ranks are alive, network topology |
| `RCCL Error` + `unhandled system error` | RCCL internal failure | Check RDMA interfaces, IB link state |
| `ib_mlx5` errors in dmesg | InfiniBand hardware issue | `approach: "skip"`, alert human |
| `NET/IB : Got completion with error` | RDMA completion error | Check network cables, switch config |
| Hang with GPU util at 0% | All GPUs waiting on collective | Check if one rank crashed, topology mismatch |

#### Server Lifecycle Errors

| Pattern | Meaning | Guidance to Produce |
|---|---|---|
| `torch.cuda.OutOfMemoryError` | GPU OOM during model load or inference | Reduce batch size, check TP config |
| `RuntimeError: CUDA error` + `device-side assert` | Kernel assertion failure | Check last code change, input shapes |
| Server exits 0 but no output | Silent crash or config parse error | Check launch command, env vars |
| `Address already in use` | Port conflict from previous instance | Kill stale process, use different port |
| Repeated crash-restart loop | Persistent failure | Check if a patched file is corrupted |

### 2.3 Crash address analysis (for segfaults)

If the crash log includes a stack trace with addresses:

```bash
# Check if crash is in user code vs system library
# User code crash → software classification
# System library crash (libhsa, librocm) → investigate hardware

# Check dmesg for GPU-related kernel messages
dmesg | tail -100 | grep -i -E "amdgpu|drm|gpu|fault|error|xnack"
```

---

## Phase 3: Metrics Analysis

### 3.1 Micro-benchmark data

Read the event's `details.micro_speedup_before_crash` and any shape-by-shape
results from the session history. Look for patterns:

- **Fast at small shapes, crashes at large shapes** → register spill or LDS overflow
- **Fast at all shapes, crashes on exit** → cleanup/dealloc bug
- **Consistent crash regardless of shape** → fundamental code issue

### 3.2 GPU state check

```bash
rocm-smi --showuse --showtemp --showecc
```

Look for:
- ECC errors (indicates hardware issue)
- GPU temperature anomalies
- Abnormal memory utilization
- GPU in error state / needs reset

### 3.3 Previous round comparison

If session history shows some rounds succeeded and this one failed, diff the
approaches:

```
Working round: BLOCK_M=32, num_warps=2, accumulator in fp32
Failing round: BLOCK_M=128, num_warps=4, accumulator cast early to bf16
                              ^^^^^^^^^            ^^^^^^^^^^^^^^^^^^^^^^
                              register pressure    precision loss
```

---

## Phase 4: Classification — HW vs SW

Apply the training-workload-rca decision tree, adapted:

```
START
  │
  ├── ECC errors in rocm-smi?
  │   └── YES → HARDWARE (GPU memory fault)
  │       Action: resubmit=false, mark kernel as hw-blocked
  │
  ├── dmesg shows amdgpu fault/reset?
  │   └── YES → HARDWARE (GPU needs reset)
  │       Action: resubmit=false, alert human
  │
  ├── Crash in libhsa-runtime or librocm?
  │   └── YES → likely HARDWARE or DRIVER
  │       Check: other kernels also failing? → HARDWARE
  │       Check: only this kernel? → SOFTWARE (bad memory access pattern)
  │
  ├── Crash in compiled extension (.so)?
  │   └── YES → SOFTWARE (compiled kernel bug)
  │       Determine: wrong flags? missing symbol? logic error?
  │
  ├── Crash in Python layer?
  │   └── YES → SOFTWARE (dispatch/import bug)
  │       Determine: wrong import? shape mismatch? dtype error?
  │
  ├── Compilation failure?
  │   └── YES → SOFTWARE or TOOLCHAIN
  │       If same error across 3+ kernels → TOOLCHAIN
  │       If unique to this kernel → SOFTWARE (bad code from OOB agent)
  │
  └── Performance regression (no crash)?
      └── SOFTWARE (register pressure / occupancy issue)
          Determine register count and occupancy from Triton IR or rocprof
```

---

## Phase 5: Deep Dive

### 5.1 Software classification → Register analysis

For register pressure / occupancy issues:

```bash
# Check Triton kernel register usage (if Triton kernel)
python3 -c "
import triton
# Compile the kernel and inspect the IR
# Look for VGPR count in the compiled PTX/GCN assembly
"

# Check occupancy via rocprof (if available)
# rocprof --stats <benchmark_command>
```

Determine:
- Current VGPR usage per thread
- Current occupancy (waves per CU)
- Target: occupancy >= 4 waves requires VGPR <= 64 on gfx950

Produce constraint: `max_vgprs=N` where N = floor(256 / target_waves) for gfx950.

### 5.2 Software classification → Compilation analysis

For compilation failures:

```bash
# Re-run compilation with verbose output to get full error context
cd /sgl-workspace/sglang/sgl-kernel
python setup_rocm.py install --verbose 2>&1 | tail -100

# For Triton, check the compilation cache for IR dumps
ls ~/.triton/cache/
```

### 5.3 Hardware classification → Full hardware check

```bash
rocm-smi --showuse --showtemp --showecc --showmemuse

/opt/rocm/bin/rocm_agent_enumerator

dmesg | grep -i -E "amdgpu|drm|gpu|fault|error|xnack|ecc" | tail -50
```

If hardware is confirmed, the finding should set `resubmit: false` and
`approach: "skip-kernel"`. The Kernel Manager will not waste more rounds.

### 5.4 Toolchain classification → Environment audit

```bash
# Check ROCm version
cat /opt/rocm/.info/version

# Check hipcc version and targets
/opt/rocm/bin/hipcc --version
/opt/rocm/bin/hipcc --print-targets

# Check Python/Triton versions
python3 -c "import triton; print(triton.__version__)"
python3 --version

# Check if sgl_kernel is properly built
python3 -c "import sgl_kernel; print(dir(sgl_kernel))"
```

---

## Phase 6: Root Cause Synthesis

### 6.1 Write the detailed report

Use the training-workload-rca output format at
`$RESULT_DIR/kernel_manager/rca_reports/<event_id>/detailed_report.md`:

```markdown
# RCA Report: <event_id>

## Summary
- Kernel: <kernel_name>
- Event: <type> at round <N>
- Classification: <HW/SW/toolchain>
- Root cause: <one-line summary>
- Confidence: <high/medium/low>

## Timeline
1. Round 1: <what happened>
2. Round 2: <what happened>
3. Round N (crash): <what happened>

## Evidence
<crash logs, compiler output, rocm-smi output>

## Analysis
<step-by-step reasoning from evidence to root cause>

## Root Cause
<detailed explanation>

## Recommended Constraints
- <specific constraint 1>
- <specific constraint 2>

## Recommended Approach
<which strategy/backend to use for retry>
```

### 6.2 Write the structured summary

At `$RESULT_DIR/kernel_manager/rca_reports/<event_id>/rca_summary.json`:

```json
{
  "event_id": "<event_id>",
  "kernel_name": "<name>",
  "classification": "software",
  "root_cause": "<one-line>",
  "confidence": "high",
  "evidence_quality": "strong",
  "timestamp": "ISO 8601"
}
```

### 6.3 Write the actionable finding to findings.jsonl

This is the most important output — it feeds back into the optimization loop.

```python
finding = {
    "event_id": event["id"],
    "task_id": event["task_id"],
    "kernel_name": event["kernel_name"],
    "classification": "software",
    "root_cause": "Register spill at BLOCK_M=128 causes stack overflow on gfx950",
    "actionable_guidance": {
        "constraint": "max_vgprs=96, BLOCK_M<=32",
        "approach": "oob-rewrite-register-constrained",
        "avoid": ["BLOCK_M>64", "num_warps>4", "early bf16 cast"],
        "compiler_flags": None,
        "reference_commit": None,
    },
    "rca_report_path": f"{result_dir}/kernel_manager/rca_reports/{event['id']}/",
    "confidence": "high",
    "resubmit": True,
    "systemic": False,
    "affects_kernels": [],
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
}
write_finding(findings_path, finding)
```

---

## Investigation Templates by Failure Type

### Template A: Segfault with Prior Improvement

1. Read session history — identify which round's changes caused the segfault
2. Diff the working and crashing kernel code
3. Check for out-of-bounds indexing, especially at boundary shapes
4. Check for register spill → stack overflow (common on gfx950)
5. Produce register constraint + shape limit for retry

### Template B: Compilation Failure (Register Allocation)

1. Extract the exact VGPR count from the error message
2. Calculate max block sizes for target occupancy on gfx950:
   - 256 VGPRs per CU, 65536 total
   - Occupancy 4 waves: max 64 VGPRs per thread
   - Occupancy 2 waves: max 128 VGPRs per thread
3. Determine which construct caused the pressure (large block sizes, excessive
   temporaries, unrolled loops)
4. Produce explicit `max_vgprs`, `max_block_m`, `max_block_n` constraints

### Template C: Correctness Failure with Partial Improvement

1. Read the correctness test output — which elements diverge?
2. Check for precision issues: fp32 → bf16 cast too early, rsqrt accumulation
3. Check for race conditions in parallel reductions
4. Produce constraint: "keep accumulator in fp32 through step X"

### Template D: E2E Regression Despite Micro Improvement

1. Confirm the micro-benchmark improvement
2. Check register pressure via Triton IR or rocprof
3. Calculate occupancy impact: faster kernel with lower occupancy may hurt
   overall throughput due to scheduling stalls
4. Produce register budget constraint for retry

### Template E: Systemic Build Failure

1. Identify the common error signature across affected kernels
2. Check ROCm/Triton/Python version compatibility
3. Check if a recent environment change caused the issue
4. Produce a systemic finding with `affects_kernels` listing all affected kernels
   and `approach: "fix-toolchain"` guidance

### Template F: Exhausted (All Rounds Failed)

1. Read all round summaries from session history
2. Categorize each failure type (compile, correctness, regression, crash)
3. Identify if there's a common constraint all attempts violate
4. If diverse failures: the kernel may need a different strategy entirely
5. Produce finding with `approach` suggesting strategy change, not just retry

---

## Non-Kernel Investigation Templates

These templates cover failures from the Marathon orchestrator that aren't
related to kernel optimization rounds.

### Template G: Framework Rebuild Failure

1. Read `details.build_command` and `details.crash_log_snippet`
2. Identify the failing compilation unit (which .cu/.hip file, which target)
3. Check environment:
   ```bash
   /opt/rocm/bin/hipcc --version
   /opt/rocm/bin/hipcc --print-targets
   cat /opt/rocm/.info/version
   ```
4. If the build was working before: check what changed (`git diff` in the library repo)
5. If it's a new kernel being added: check if it uses unsupported HIP features for gfx950
6. Classify: build-system (wrong flags, missing dep) vs toolchain (ROCm bug) vs code (bad source)
7. Produce finding:
   - `approach: "retry-with-flags"` if compiler flags need changing
   - `approach: "fix-toolchain"` if ROCm/env issue
   - `approach: "revert-and-retry"` if a source change caused it

### Template H: Server Crash After Config/Rebuild Change

1. Read `details.last_config_change` — what was changed before the crash
2. Read the server crash log (`details.crash_log_snippet`)
3. Determine if the crash is:
   - **Import-time** (server won't even start): missing module, ABI mismatch
   - **Load-time** (crashes during model loading): OOM, shape mismatch, dtype error
   - **Inference-time** (crashes on first request): kernel bug, dispatch error
4. Check if reverting the change fixes it:
   ```bash
   # For editable installs, check git status
   cd /sgl-workspace/sglang && git diff --stat
   cd /sgl-workspace/aiter && git diff --stat
   ```
5. Produce finding with:
   - `approach: "revert-and-retry"` if the change is clearly the cause
   - `approach: "rebuild-with-fix"` if ABI mismatch (need to rebuild dependent libs)
   - Specific `constraint` about what the change broke

### Template I: Communication Hang / RCCL Failure

This maps most directly to the training-workload-rca skill's infrastructure
deep dive. Use the infra-deep-dive sub-skill.

1. Check RCCL debug output:
   ```bash
   # Look for RCCL initialization and topology info
   grep -i "NCCL" server_logs | head -50
   grep -i "RCCL" server_logs | head -50
   ```
2. Check network infrastructure:
   ```bash
   # RDMA interface status
   ibstat 2>/dev/null || echo "no IB"
   # Check for link errors
   dmesg | grep -i -E "ib_|mlx5|rdma|infiniband" | tail -20
   ```
3. Check if topology changed:
   - Was a new comm optimization algorithm applied?
   - Did TP/PP configuration change?
   - Is this a multi-node issue? (check if all nodes are reachable)
4. Classify: infrastructure (network, hardware) vs software (wrong algorithm, bad config)
5. Produce finding:
   - `approach: "revert-comm-config"` if a comm change caused it
   - `approach: "skip"` + `resubmit: false` if hardware network issue
   - `systemic: true` if affecting all comm operations

### Template J: Compiler / Codegen Failure

1. Identify the codegen backend (Triton, Inductor, custom)
2. Check cache state:
   ```bash
   ls -la ~/.triton/cache/ | wc -l
   ls -la /tmp/torchinductor_root/ 2>/dev/null | wc -l
   ```
3. If cache corruption suspected: check file timestamps, look for truncated files
4. If codegen bug: identify the input pattern that triggers bad code generation
5. Produce finding:
   - `approach: "clear-cache-retry"` if cache corruption
   - `approach: "fix-toolchain"` + `systemic: true` if codegen bug
   - `constraint` about which Triton/Inductor features to avoid

### Template K: Server Hang / Benchmark Timeout

1. Check GPU state during hang:
   ```bash
   rocm-smi --showuse --showtemp
   # GPU util 0% on all GPUs → collective deadlock
   # GPU util 100% on one GPU → stuck kernel
   # GPU util 0% on one, 100% on others → asymmetric hang
   ```
2. Check if the server process is alive:
   ```bash
   ps aux | grep sglang
   # If alive but unresponsive → deadlock or infinite loop
   # If dead → crash that wasn't caught
   ```
3. Check dmesg for GPU faults that might explain the hang
4. Determine what was happening when the hang started (benchmark, warmup, idle)
5. Produce finding:
   - `approach: "restart-and-retry"` if transient
   - `approach: "revert-last-change"` if hang started after a specific change
   - `resubmit: false` if GPU hardware fault confirmed

### Template L: Dispatch Fix Failure

1. Read the source file the fix was applied to
2. Read `details.error_message` — what went wrong after the fix
3. Check git history to understand the full context:
   ```bash
   git log --oneline -10 -- <source_file>
   ```
4. If the fix was a git revert: check if surrounding code changed, making the
   old code incompatible
5. Produce finding:
   - `approach: "manual-fix"` with detailed guidance on what the correct code should be
   - `constraint` about which code patterns must be preserved
