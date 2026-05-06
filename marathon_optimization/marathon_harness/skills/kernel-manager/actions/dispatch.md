# Action: Dispatch to OOB Backends

Selects the right backend(s) for each kernel target, builds hardware-specific prompts
with trace-derived shapes, submits tasks, polls for completion, and runs iterative
refinement loops with local feedback.

## Backend Selection Matrix

| Kernel Type / Strategy | GEAK | Codex | Claude | LLM Proxy | Notes |
|----------------------|:----:|:-----:|:------:|:---------:|-------|
| Triton rewrite (`@triton.jit`) | Yes | Yes | Yes | Yes | Race all, LLM Proxy returns first |
| HIP/C++ kernel (`.cu`, `.hip`) | **Yes** | Limited | Limited | No | GEAK has GPU for hipcc |
| Inductor-generated (`triton_heuristics`) | Yes | Yes | Yes | Yes | Race all |
| Register-constrained rewrite | Skip | **Yes** | **Yes** | Yes | GEAK doesn't take VGPR limits well |
| Multi-file framework change | No | Partial | **Yes** | No | Claude has multi-turn reasoning |
| Config/dispatch fix | — | — | — | — | Self-fix, don't dispatch externally |
| Kernel fusion (new kernel) | **Yes** | Yes | **Yes** | No | GEAK can compile+test HIP on pod |

**Default**: submit to ALL active backends in parallel unless the kernel type narrows the selection.

## MCP Tool References

### GEAK — `geak_*` tools (remote GPU pod)

| Tool | Purpose |
|------|---------|
| `geak_set_model_config` | Configure LLM backend (once per session) |
| `geak_create_task` | Create task with source + prompt |
| `geak_submit_task` | Start optimization |
| `geak_get_task` | Poll status (every 30s) |
| `geak_get_outputs` | List output files |
| `geak_download_file` | Download optimized code |
| `geak_list_tasks` | Debug: list all tasks |

**Critical parameters for `geak_create_task`:**
- `input_type`: `"file"` (required)
- `prompt`: optimization instructions (not `instructions`)
- `step_limit`: **100** (GEAK needs room for analyze→write→compile→fix→bench→iterate)
- `workspace_id`: `GEAK_WORKSPACE` (default `"control-plane-moe"`)
- `files`: array of `{filename, content}` — full kernel source embedded
- `gpu_count`: 1 (default)

**GEAK image selection:**

| Framework | Local Mode | Claw Mode |
|-----------|-----------|-----------|
| sglang | `GEAK_IMAGE_SGLANG` | `GEAK_IMAGE_SGLANG_RAY` |
| vllm | `GEAK_IMAGE_VLLM` | `GEAK_IMAGE_VLLM` |

**Latency:** 10–30 min (pod scheduling 2-15 min + docker pull 1-5 min + agent 3-10 min).

### Codex — `agent_*` tools (OOB Agent MCP)

| Tool | Purpose |
|------|---------|
| `agent_create_task` | Create task with `agent="codex"` |
| `agent_submit_task` | Start execution |
| `agent_get_task` | Poll status (every 15s) |
| `agent_get_outputs` | List output files |
| `agent_download_file` | Download optimized kernel |
| `agent_cancel_task` | Cancel if stuck |

**Critical parameters for `agent_create_task`:**
- `agent`: `"codex"`
- `prompt`: optimization instructions
- `files`: array of `{filename, content}`
- No `image`, `workspace_id`, or `gpu_count` — Codex has no GPU pod

**Latency:** 30–120s per iteration, 2–6 min for 3-iteration round.

### Claude — `agent_*` tools (OOB Agent MCP)

Same tools as Codex, with `agent="claude"`.

- `max_turns`: 30 (higher than Codex — Claude benefits from multi-step reasoning)
- `system_prompt`: recommended — Claude responds well to detailed persona prompts
- **Latency:** 1–5 min per iteration, 3–15 min for 3-iteration round
- **Fallback:** If `agent="claude"` returns error, retry with `agent="codex"`

### LLM Proxy — Direct OpenAI API

```python
from openai import OpenAI
import httpx, os

http_client = httpx.Client(verify=False, timeout=180)
client = OpenAI(
    base_url="https://core42.example-internal-host.invalid/api/v1/llm-proxy/v1",
    api_key=os.environ["LLM_PROXY_API_KEY"],
    http_client=http_client,
)

response = client.chat.completions.create(
    model="claude-opus-4-6",  # or "claude-opus-4.5", "gpt-4.1"
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
    max_tokens=8192,
    temperature=0.0,
)
```

**Available models:**

| Model | Status | Latency | Best for |
|-------|--------|---------|----------|
| `claude-opus-4-6` | Working | ~24s | Complex structural optimizations |
| `claude-opus-4.5` | Working | ~1s | Quick block-size tuning |
| `gpt-4.1` | Working | ~2s | Fast but may use invalid Triton APIs |
| `gpt-5.2` | Broken | — | Do not use |

**Latency:** 1–30s (immediate return). Run micro-benchmark locally right away.

---

## Prompt Engineering

### What OOB can and cannot do (READ FIRST)

OOB (the `oob-gpu-optimizer` MCP, backends `claude` / `codex`) runs a
single-turn agent inside a SaFE workload with **exactly these resources**:

- `gpu_count: 1` (default; single-GPU micro-benchmarking only)
- `cpu: 4`, `memory: 16Gi`, `ephemeral_storage: 50Gi` (defaults)
- `timeout: 1800s` (30 min hard wall-clock, enforced by MCP)
- No model weights mounted, no multi-node, no RDMA fabric

**OOB CAN**: rewrite a single kernel source file, compile it with the
toolchain present in the pod, run a micro-benchmark against shapes
embedded in the prompt, return the optimised file.

**OOB CANNOT** — and if you ask it to, the pod will stall until the MCP
timeout fires, costing the marathon up to `MAX_OOB_ROUNDS × 30min` per
bad target:

- Launch an inference server (`vllm serve`, `sglang.launch_server`, …)
- Load full model weights (no weights are mounted)
- Run end-to-end serving benchmarks (`benchmark_serving`, ShareGPT, …)
- Multi-GPU / tensor-parallel / distributed work (only 1 GPU)
- NCCL / RCCL collective tests
- Anything requiring more than 50 GiB of disk

### Prompt Hygiene Guard (MANDATORY before every submission)

Before calling `agent_create_task` (OOB) or `geak_create_task` (GEAK),
the kernel-manager MUST run the prompt string through this guard:

```python
import re

# Patterns that indicate a pollution target — things OOB physically
# cannot do. Matching ANY of these aborts the submission.
POLLUTION_PATTERNS = [
    # Inference server launch
    r"\bvllm\s+serve\b",
    r"sglang[\w\.]*\.?launch_server",
    r"python\s+-m\s+vllm\.entrypoints",
    r"\buvicorn\b",
    r"\btrtllm-serve\b",

    # Multi-GPU / distributed
    r"tensor[-_ ]parallel[-_ ]size\s*[=:]?\s*[2-9]",
    r"--tp[ =][2-9]",
    r"torchrun\s+[^\n]*--nproc[-_]per[-_]node\s*[=]?\s*[2-9]",
    r"\bNCCL_[A-Z_]+",
    r"\bRCCL_[A-Z_]+",

    # End-to-end serving benchmark
    r"\bbenchmark_serving\b",
    r"\bsharegpt\b",
    r"--num-prompts\s+\d{3,}",
    r"\bTTFT\b",
    r"\bTPOT\b",
    r"end[- ]to[- ]end\s+(benchmark|throughput|latency)",

    # Full-model weight loading
    r"--model\s+[^ \n]+\.safetensors",
    r"MODEL_PATH\s*=\s*[^\s]+/[A-Za-z0-9_.-]+-\d+B",
]
_POLLUTION_RE = re.compile("|".join(POLLUTION_PATTERNS), re.IGNORECASE)


def sanitise_oob_prompt(prompt: str, target: dict, write_event) -> bool:
    """Return True if prompt is safe to submit; False if polluted."""
    m = _POLLUTION_RE.search(prompt)
    if m is None and len(prompt) <= 8 * 1024:
        return True

    reason = (
        f"prompt-too-long ({len(prompt)} bytes > 8KB)"
        if m is None else f"banned-pattern: {m.group(0)!r}"
    )
    write_event({
        "type": "prompt-pollution",
        "kernel_name": target.get("kernel_name"),
        "task_id": target.get("id"),
        "reason": reason,
        "prompt_snippet": prompt[:2000],
    })
    return False


# Usage at submission site:
if not sanitise_oob_prompt(prompt, target, write_event):
    # Do NOT submit. Mark the round as failed; let the deep-guidance
    # loop either rewrite the prompt on the next round with accumulated
    # context, or exhaust and move on.
    return "prompt-pollution"
```

### Hardware Context Block (include in every prompt)

```
Hardware: AMD MI355X (gfx950, CDNA4).
  - 304 CUs, 256 VGPR/CU, wavefront size 64
  - HBM3e ~8 TB/s bandwidth
  - MFMA bf16 instructions
  - 256KB LDS per CU, 65536 VGPRs per CU
Context: LLM inference serving (decode path).
```

### Mandatory Constraints Block (include in every prompt)

```
MANDATORY CONSTRAINTS (violation = rejected):
0. You are a SINGLE-KERNEL OPTIMIZER running on 1 GPU with no model
   weights mounted. Do NOT launch vllm/sglang/any inference server.
   Do NOT run end-to-end benchmarks or ShareGPT traces. Do NOT attempt
   tensor-parallel / multi-GPU / NCCL work. Your ONLY deliverable is
   the rewritten kernel source file, validated by a single-kernel
   micro-benchmark on the shapes given below.
1. The output function name MUST be EXACTLY: {original_function_name}. Do NOT rename it.
2. The function signature (parameter names, order, types) MUST be IDENTICAL to the original.
3. Decorators MUST be preserved (@triton_heuristics, @triton.jit, etc.).
4. Block size limits: BLOCK_M <= 16, BLOCK_N <= 128, BLOCK_K <= 256.
   Do NOT increase any block dimension beyond 2x its original value.
5. Do NOT add @triton.autotune or change existing decorators.
6. The kernel MUST be optimized to at least 1.5x speedup.
7. Do NOT search the filesystem with find / or grep -r /.
```

### Strategy-Specific Prompt Templates

#### Triton Kernel Rewrite

```
Optimize this Triton kernel for AMD MI355X (gfx950).

{hardware_context}
Input shapes: {shapes_from_trace}
Data types: {dtypes} (e.g., bf16 activations, fp8_e4m3 weights/KV cache)
Currently: {kernel_time_ms}ms per call, {gpu_pct}% of total GPU time.
Called {count} times per batch of {batch_size} requests.

{mandatory_constraints}

OPTIMIZATION TARGETS (prioritized):
1. STRUCTURAL: Hoist loop-invariant computations out of loops.
2. STRUCTURAL: Merge dual-pass into single-pass where possible.
3. TUNING: Adjust BLOCK sizes to match exact dimensions (e.g., BLOCK_M=M when M is small).
4. TUNING: Simplify grid indexing when dimensions are small.
5. MICRO: Use libdevice.rsqrt (NOT tl.math.rsqrt), multiply by reciprocal.

```python
{kernel_source}
```

Write the COMPLETE optimized file to optimized_kernel.py.
```

#### Register-Constrained Rewrite (Strategy B')

```
{base_triton_prompt}

CRITICAL REGISTER CONSTRAINT:
The previous optimization attempt achieved {micro_speedup}x micro-benchmark speedup but
REGRESSED E2E throughput by {e2e_regression}% because of register pressure causing
occupancy to drop from {old_occupancy} to {new_occupancy}.

You MUST:
1. Keep VGPR usage under {max_vgprs} registers per thread (current: {current_vgprs})
2. Target occupancy >= {min_occupancy} waves per CU
3. Use shared memory (LDS) instead of extra registers where possible
4. Prefer smaller tile sizes that fit within the register budget
5. If using Triton: set num_warps and num_stages conservatively

The kernel MUST be both faster in micro-benchmark AND maintain occupancy.
```

#### HIP/C++ Kernel (GEAK-specific)

```
Optimize this HIP kernel for AMD MI355X (gfx950).

{hardware_context}
The kernel source file is at {source_file_path}.
The kernel repo is at {repo_path}.

Input shapes: {shapes_from_trace}
Currently: {kernel_time_ms}ms per call, {gpu_pct}% of GPU time.

MANDATORY CONSTRAINTS:
1. Function name and signature MUST be identical to original.
2. Compile with: hipcc -O3 --amdgpu-target=gfx950
3. Use homogeneous mode. Set max_rounds to 1.
4. The kernel MUST be optimized to at least 1.5x speedup.
5. Do NOT search the filesystem with find / or grep -r /.

```cpp
{kernel_source}
```
```

#### Multi-File Framework Change (Claude-specific)

```
You are analyzing the inference serving framework for optimization.

{hardware_context}
Context: {description_of_the_scheduling_or_dispatch_issue}

Files involved:
{list_of_files_with_contents}

Think step by step:
1. Analyze the current dispatch/scheduling logic — identify the bottleneck.
2. Trace the data flow across the files.
3. Design a minimal change that addresses the bottleneck.
4. Write the COMPLETE modified files.
5. Verify the changes handle edge cases.

Write each modified file to the output directory with its original filename.
```

#### RMSNorm / Reduction Kernel (highest-impact template)

```
Optimize this Triton kernel for AMD MI355X (gfx950).

{hardware_context}
SHAPES: xnumel={xnumel} (batch rows), r0_numel={r0_numel} (hidden_dim).
CURRENT: {gpu_pct}% of GPU time, called {call_count} times per forward pass in LLM decode.

{mandatory_constraints}
R0_BLOCK <= {r0_block}. Do NOT increase beyond original value.
MUST produce numerically identical output.

CRITICAL OPTIMIZATION — TRUE SINGLE-PASS:
The original kernel has TWO loops that BOTH read from in_ptr0:
  Loop 1: load input → compute sum of squares
  Loop 2: RE-LOAD input → normalize with rsqrt → multiply weight → store

Since R0_BLOCK = r0_numel = {r0_numel}, each loop executes exactly ONCE. The data fits
in registers. ELIMINATE THE SECOND LOOP ENTIRELY:
  1. Load ALL inputs ONCE
  2. Compute sum of squares + rsqrt
  3. Normalize and store
  4. ZERO for-loops in the result

WARNING: This optimization is ONLY valid when R0_BLOCK = r0_numel.

```python
{kernel_source}
```

Return the COMPLETE optimized file.
```

---

## GEAK-Specific Prompt Rules

These rules apply ONLY to GEAK submissions (in addition to the mandatory constraints):

1. **Kernel path — conditional on image availability:**
   - If source exists in the Docker image (`/sgl-workspace/`, `/opt/`): include absolute
     file path and repo path in prompt
   - If source is runtime-generated (`/tmp/torchinductor_*`): omit path, rely on
     `files[].content` only
2. **Always include:** `"Use homogeneous mode. Set max_rounds to 1."`
3. **Always say:** `"Do NOT search the filesystem with find / or grep -r /"`
4. **Always pass framework image** (see GEAK Image Selection table above)
5. **Always embed full source** in `files[].content`

---

## Submission Workflow

### Fire-and-Forget Submission

For each work queue target, submit to all selected backends in parallel and record
task IDs. Do NOT wait for results during submission.

```
candidate kernel ────┬─ GEAK:   geak_create_task + geak_submit_task → record task_id
                     ├─ Codex:  agent_create_task + agent_submit_task → record task_id
                     ├─ Claude: agent_create_task + agent_submit_task → record task_id
                     └─ LLM:    openai API call → result arrives immediately
```

**LLM exception:** LLM Proxy returns immediately (1–30s). If the result compiles
correctly, store it as the best-so-far while waiting for slower backends.

### Multi-Model LLM Strategy

Submit to all working LLM models in parallel for maximum coverage:

```python
import concurrent.futures

models = ["claude-opus-4-6", "claude-opus-4.5", "gpt-4.1"]

def optimize_with_model(model_name):
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        max_tokens=8192,
        temperature=0.0,
    )
    return model_name, resp.choices[0].message.content

with concurrent.futures.ThreadPoolExecutor() as pool:
    results = list(pool.map(optimize_with_model, models))
```

---

## Polling and Collection

### MANDATORY: Defensive polling template

**Prior production bug** — the polling loop used to only recognise two
failure strings:

```python
elif result["status"] in ("failed", "cancelled"):
    return None
```

Real OOB / GEAK workloads also emit `error`, `errored`, `terminated`,
`crashed`, `timeout`, `timed_out`, `exhausted`, `aborted`, `hw_error`,
etc. Any status outside that tiny whitelist sent the loop to `sleep` and
kept polling for the full `POLL_TIMEOUT_MIN`; with 5 deep-guidance rounds
that wastes **~75 minutes per already-dead target**. This happened in
practice and consumed entire marathons.

You MUST use the template below verbatim. Do NOT simplify it to a single
`if status == "completed"` branch.

```python
import time

# Terminal states — break out immediately. No sleep, no retry.
TERMINAL_OK = {"completed", "succeeded", "success", "done", "finished"}
TERMINAL_FAIL = {
    "failed", "cancelled", "canceled", "error", "errored",
    "terminated", "crashed", "timeout", "timed_out", "exhausted",
    "aborted", "hw_error", "oom", "killed",
}

# Active states — legitimate reasons to keep polling.
ACTIVE = {"running", "pending", "queued", "in_progress", "starting",
          "scheduling", "initializing"}


def poll_backend(get_task_fn, cancel_task_fn, get_outputs_fn,
                 download_file_fn, task_id, *,
                 poll_interval_s, task_timeout_min,
                 backend_name, kernel_name, write_event):
    """Unified polling used for GEAK / Codex / Claude / OOB.

    Returns downloaded code string on success, None on any failure.
    Writes an event_log.jsonl entry on every non-success exit so the
    Watchdog sees the failure mode.
    """
    start = time.time()
    unknown_strikes = 0
    transport_strikes = 0
    last_status = None

    while True:
        elapsed = time.time() - start

        # Hard per-task wall-clock budget.
        if elapsed >= task_timeout_min * 60:
            try:
                cancel_task_fn(task_id=task_id)
            except Exception:
                pass
            write_event({
                "type": "oob-timeout",
                "backend": backend_name,
                "kernel_name": kernel_name,
                "task_id": task_id,
                "elapsed_min": round(elapsed / 60, 1),
                "last_status": last_status,
            })
            return None

        # Resilient status fetch — MCP transport can hiccup.
        try:
            result = get_task_fn(task_id=task_id)
            transport_strikes = 0
        except Exception as e:
            transport_strikes += 1
            if transport_strikes >= 3:
                write_event({
                    "type": "oob-transport-fail",
                    "backend": backend_name,
                    "kernel_name": kernel_name,
                    "task_id": task_id,
                    "error": str(e)[:500],
                })
                return None
            time.sleep(poll_interval_s)
            continue

        status = (result.get("status") or "").strip().lower()
        last_status = status

        # ① Terminal failure — STOP, do not retry in this round.
        if status in TERMINAL_FAIL:
            write_event({
                "type": "oob-failed",
                "backend": backend_name,
                "kernel_name": kernel_name,
                "task_id": task_id,
                "status": status,
                "elapsed_min": round(elapsed / 60, 1),
                "error": (result.get("error") or result.get("message") or "")[:2000],
            })
            return None

        # ② Terminal success — fetch outputs. Empty output = failure.
        if status in TERMINAL_OK:
            try:
                outputs = get_outputs_fn(task_id=task_id)
            except Exception as e:
                write_event({
                    "type": "oob-output-fetch-fail",
                    "backend": backend_name,
                    "kernel_name": kernel_name,
                    "task_id": task_id,
                    "error": str(e)[:500],
                })
                return None

            files = outputs.get("files") or []
            candidate = next(
                (f for f in files
                 if "optimized" in f["filename"]
                 or f["filename"].endswith((".py", ".cu", ".hip", ".cuh"))),
                None,
            )
            if candidate is None:
                write_event({
                    "type": "oob-empty-output",
                    "backend": backend_name,
                    "kernel_name": kernel_name,
                    "task_id": task_id,
                    "files_seen": [f["filename"] for f in files][:10],
                })
                return None

            return download_file_fn(
                task_id=task_id, filename=candidate["filename"],
            )

        # ③ Unknown (neither terminal nor active) — strike-and-cancel.
        #    This catches new states the backend may introduce over time.
        if status not in ACTIVE:
            unknown_strikes += 1
            if unknown_strikes >= 5:
                try:
                    cancel_task_fn(task_id=task_id)
                except Exception:
                    pass
                write_event({
                    "type": "oob-unknown-status-stuck",
                    "backend": backend_name,
                    "kernel_name": kernel_name,
                    "task_id": task_id,
                    "status": status,
                    "elapsed_min": round(elapsed / 60, 1),
                })
                return None
        else:
            unknown_strikes = 0

        time.sleep(poll_interval_s)
```

### Per-backend invocation

```python
GEAK_POLL_INTERVAL_S = 30
GEAK_POLL_TIMEOUT_MIN = 30

OOB_POLL_INTERVAL_S = 15
OOB_POLL_TIMEOUT_MIN = {"codex": 10, "claude": 15}

# GEAK
code = poll_backend(
    get_task_fn=geak_get_task,
    cancel_task_fn=lambda task_id: None,     # GEAK has no cancel; fall through
    get_outputs_fn=geak_get_outputs,
    download_file_fn=geak_download_file,
    task_id=task_id,
    poll_interval_s=GEAK_POLL_INTERVAL_S,
    task_timeout_min=GEAK_POLL_TIMEOUT_MIN,
    backend_name="geak",
    kernel_name=target["kernel_name"],
    write_event=write_event,
)

# OOB (Codex / Claude via oob-gpu-optimizer MCP)
code = poll_backend(
    get_task_fn=agent_get_task,
    cancel_task_fn=agent_cancel_task,
    get_outputs_fn=agent_get_outputs,
    download_file_fn=agent_download_file,
    task_id=task_id,
    poll_interval_s=OOB_POLL_INTERVAL_S,
    task_timeout_min=OOB_POLL_TIMEOUT_MIN[backend],
    backend_name=backend,
    kernel_name=target["kernel_name"],
    write_event=write_event,
)
```

### Circuit breakers on top of the template

| Counter | Trigger | Effect |
|---|---|---|
| `per_target_backend_fails[target_id][backend]` | 3 consecutive non-`TERMINAL_OK` returns from `poll_backend` for the same `(target, backend)` | Skip that backend for the remainder of the target's deep-guidance loop. |
| `per_session_backend_fails[backend]` | 8 cumulative failures across all targets this session | Deprioritise the backend for the rest of the session (submit only if no other active backend). |
| `OOB_TASK_TOTAL_BUDGET_MIN` (default 30) | Sum of `elapsed_min` across all rounds for the same target exceeds the budget | Abort the target entirely, mark `failed / reason=oob-budget-exceeded` in `results.jsonl`, write `exhausted` event. |

These are enforced in the Deep Guidance Loop (§below), not inside
`poll_backend` itself.

---

## Deep Guidance Loop

The core optimization loop for each kernel target. Replaces fire-and-forget with
multi-round guided refinement using accumulated context, failure analysis, and
Watchdog RCA findings.

```
FOR round in 1..MAX_OOB_ROUNDS (5):

  1. BUILD PROMPT with accumulated context:
     - Round 1: full kernel source + hardware context + shapes
     - Round 2+: ALL of the above PLUS:
       * Previous attempt(s) code summary
       * Specific failure analysis (not just "COMPILE_FAIL" but WHY)
       * What approaches were tried and their outcomes
       * Cross-backend learnings (if Codex found X, tell Claude)
       * Watchdog findings (if RCA produced guidance, bake it in)
       * Session history block (see Accumulated Context Format)

  2. SELECT backend for this round:
     - Round 1: submit to all active backends in parallel
     - Round 2+: select based on what worked best
       * If backend A got closest: retry with backend A + feedback
       * If all backends failed similarly: try a different backend
       * If Watchdog finding specifies approach: match to best backend

  3. SUBMIT to backend(s)

  4. COLLECT result (poll + download)

  5. DEEP TEST (compile → correctness → multi-shape benchmark)
     See actions/local-test.md

  6. DEEP ANALYZE result:
     a. If COMPILE_FAIL:
        - Parse the exact compiler error (line number, construct)
        - Identify root cause: Triton API issue? Missing import?
          Register pressure? Type mismatch?
        - Record: "Round N: COMPILE_FAIL — {specific_cause}"
        - If register allocation: calculate VGPR budget, produce
          explicit max_vgprs constraint for next round

     b. If CORRECTNESS_FAIL:
        - Identify which output elements diverge (max_diff, mean_diff)
        - Determine: precision issue (fp32→bf16 too early) vs logic error
        - If precision: identify which computation step loses precision
        - Record: "Round N: CORRECTNESS_FAIL — {divergence_analysis}"

     c. If PERF_REGRESSION (some shapes improve, some regress):
        - Log shape-by-shape speedups/regressions
        - Calculate register pressure from block sizes
        - Determine if occupancy dropped (VGPR > 64 → < 4 waves)
        - Record: "Round N: REGRESSION — shapes {X} improved {Y}x,
          shapes {Z} regressed {W}x, likely occupancy drop"

     d. If SEGFAULT / CRASH:
        - Write event to event_log.jsonl with full session_history
        - Mark as "promising" if any shape showed improvement
        - Include crash_log_snippet (first 2000 chars)
        - Record: "Round N: SEGFAULT — {crash_context}"
        - WAIT up to FINDINGS_POLL_INTERVAL_S for Watchdog guidance

     e. If PASS (speedup > threshold, all shapes correct):
        - Verify improvement is consistent across all shapes
        - Record: "Round N: PASS — {speedup}x, all shapes correct"
        - Generate merge-ready patch → DONE

  7. CHECK findings.jsonl for Watchdog guidance on this kernel:
     - If RCA found root cause with resubmit=true:
       Bake constraints into next round's prompt
     - If RCA said resubmit=false (hardware):
       STOP immediately, mark as hw-blocked
     - If systemic finding applies:
       Apply constraints from the systemic finding

  8. RECORD this round in session_history

  9. IF max rounds exhausted:
     - Write "exhausted" event to event_log.jsonl with full session_history
     - Mark task as failed in results.jsonl with reason: "exhausted after N rounds"
     - Include the best partial result (closest to passing) for reference
```

### Accumulated Context Format

Each round builds a session history included in the prompt from round 2 onward:

```
== OPTIMIZATION SESSION HISTORY ==

Round 1 (codex): COMPILE_FAIL
  Attempt: merged dual-loop into single-pass, BLOCK_M=128
  Error: "triton.compiler.errors.CompilationError: register allocation failed"
  Analysis: BLOCK_M=128 exceeds register budget on gfx950 (256 VGPRs per CU).
  At BLOCK_M=128, the kernel requires ~180 VGPRs which limits occupancy to 1 wave.

Round 2 (claude): CORRECTNESS_FAIL
  Attempt: single-pass with BLOCK_M=32, vectorized loads
  Error: out_ptr0 max diff=0.15 (tolerance=0.01)
  Analysis: rsqrt precision loss — accumulator cast to bf16 before normalization.
  The intermediate sum needs to stay in fp32 through the rsqrt step.

Round 3 (this attempt):
  Constraints from previous rounds:
  - BLOCK_M must be <= 32 (register budget, round 1)
  - Keep accumulator in fp32 through normalization (precision, round 2)
  - Vectorized loads are good (round 2 was 1.3x faster before correctness fail)

  Watchdog RCA finding (evt_xyz):
  - Max VGPRs = 96 for stable occupancy >= 4 waves
  - Avoid: num_warps > 4
```

### Failure Analysis Guide

Detailed analysis procedures for each failure type:

#### Compilation Failure Analysis

```python
def analyze_compilation_failure(error_output, kernel_source):
    """Extract actionable constraints from compilation failure."""
    analysis = {"cause": "", "constraint": "", "avoid": []}

    if "register allocation" in error_output:
        # Extract VGPR count if available
        # gfx950: 256 VGPRs per CU, 65536 total
        # Occupancy targets: 4 waves → max 64 VGPRs, 2 waves → max 128
        analysis["cause"] = "Register allocation failure"
        analysis["constraint"] = "max_vgprs=64, reduce block sizes"
        analysis["avoid"] = ["BLOCK_M > current_value", "excessive unrolling"]

    elif "invalid LLVM IR" in error_output:
        analysis["cause"] = "Malformed Triton IR"
        analysis["constraint"] = "Simplify kernel structure"

    elif "cannot use store" in error_output and "different type" in error_output:
        analysis["cause"] = "Type mismatch in store operation"
        analysis["constraint"] = "Match pointer and value types explicitly"

    elif "KeyError" in error_output:
        analysis["cause"] = "Missing grid dimension or constant"
        analysis["constraint"] = "Preserve all grid dimensions from original"

    return analysis
```

#### Correctness Failure Analysis

```python
def analyze_correctness_failure(test_output, kernel_source):
    """Determine whether correctness failure is precision or logic."""
    analysis = {"type": "", "fix_guidance": ""}

    max_diff = extract_max_diff(test_output)
    mean_diff = extract_mean_diff(test_output)

    if max_diff < 0.5 and mean_diff < 0.01:
        analysis["type"] = "precision"
        analysis["fix_guidance"] = (
            "Keep intermediate accumulators in fp32. "
            "Cast to output dtype only at the final store."
        )
    elif max_diff > 1.0:
        analysis["type"] = "logic_error"
        analysis["fix_guidance"] = (
            "Output is significantly wrong. Check index computation, "
            "reduction logic, and boundary handling."
        )
    else:
        analysis["type"] = "mixed"
        analysis["fix_guidance"] = (
            "Check both precision (accumulator dtype) and "
            "boundary conditions (edge shapes)."
        )

    return analysis
```

### Cross-Backend Learning

When multiple backends attempt the same kernel, share insights:

```
IF codex_round_1 found: BLOCK_M=32 works but vectorization caused correctness issue
AND claude_round_1 found: vectorization is correct but BLOCK_M=64 caused register spill

THEN round_2_prompt includes:
  "Cross-backend learnings:
   - BLOCK_M=32 is within register budget (codex, round 1)
   - Vectorized loads need fp32 accumulator for correctness (codex, round 1)
   - BLOCK_M=64 exceeds register budget (claude, round 1)
   Use BLOCK_M=32 with vectorized loads AND fp32 accumulator."
```

Track cross-backend learnings in the session_history and include them in prompts
to avoid repeating the same mistakes.

### Watchdog Finding Integration

When the Watchdog produces a finding for the current kernel:

```python
def integrate_watchdog_finding(prompt, finding):
    """Inject Watchdog RCA guidance into the OOB prompt."""
    guidance = finding.get("actionable_guidance", {})

    constraint_block = "\n\nWATCHDOG RCA CONSTRAINTS (from root cause analysis):\n"
    if guidance.get("constraint"):
        constraint_block += f"  MUST: {guidance['constraint']}\n"
    if guidance.get("avoid"):
        constraint_block += f"  MUST NOT: {', '.join(guidance['avoid'])}\n"
    if guidance.get("approach"):
        constraint_block += f"  Recommended approach: {guidance['approach']}\n"
    if finding.get("root_cause"):
        constraint_block += f"  Root cause: {finding['root_cause']}\n"

    return prompt + constraint_block
```

### Key Rules for the Deep Guidance Loop

- Test inputs MUST come from traces, never fabricated
- If GPU is busy (server running), defer micro-benchmark — feed compilation
  success/failure as feedback instead
- After MAX_OOB_ROUNDS rounds with no passing result: write `exhausted` event,
  mark as `failed`, move to next target
- If a backend produces 5 consecutive failures across all kernels: deprioritize it
- Always include session_history from round 2 onward — accumulated context is
  the most valuable input to the next attempt
- Check findings.jsonl between every round, not just at the start
- Write events for every crash and segfault, not just exhausted targets

---

## Output Validation Checklist

Before passing any backend result to `actions/local-test.md`:

- [ ] Function name matches original exactly
- [ ] Function signature (parameters, constexprs) matches original
- [ ] Decorators preserved (`@triton_heuristics`, `@triton.jit`, etc.)
- [ ] No new imports that don't exist in target environment
- [ ] Block sizes within constraints (not exceeding 2x original)
- [ ] Source code is actual code, not comments or path references
- [ ] Complete file (not truncated)

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| GEAK task stuck >30 min | Cancel with `geak_list_tasks`, retry with alternate `workspace_id` |
| Codex returns no output file | Feedback: "Write the COMPLETE optimized file to optimized_kernel.py" |
| Claude spends all turns analyzing | Feedback: "Skip analysis. Write optimized code immediately." |
| LLM uses invalid Triton APIs | Known issue with `gpt-4.1`. Verify compilation before accepting. |
| All backends produce wrong signature | Re-submit all with bolded constraint: "**FUNCTION NAME MUST BE: {name}**" |
| GEAK pod scheduling slow | Use `workspace_id: "control-plane-moe"`, check cluster load |
| LLM Proxy returns 400 | Model may be broken (e.g., `gpt-5.2`). Skip and try others. |
