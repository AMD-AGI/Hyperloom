---
name: codex-inference-kernel-reference
description: Codex backend for kernel optimization via `oob_ray_submit.py run` (Ray-scheduled CLI). Code generation with optional GPU — verification done by the calling skill. Referenced by actions/kernel-opt.md Step 2.
---

# Codex — Kernel Optimization Backend

Codex backend for kernel optimization. In this remote-only skill, Codex is invoked
inside the RayJob through the Ray-scheduled OOB CLI transport:

| Runtime | How Codex is invoked |
|---------|----------------------|
| RayJob | `oob_ray_submit.py run -a codex ...` CLI (single blocking subprocess per iteration) |

Generates optimized kernel code. The calling skill is
responsible for compilation checking, correctness verification, and micro-benchmarking.

## Status: Stable

Validated 2026-04-02. Typical completion: 1 turn, ~60 seconds. Output quality is
good for Triton structural rewrites (dual-loop to single-pass, block-size tuning).

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `OOB_ROUND_ITERATIONS` | 3 | Iterations per round (submit → benchmark → feedback) |
| `CODEX_MAX_TURNS` | 20 | Max agent turns per task (per iteration) |
| `CODEX_POLL_INTERVAL_S` | 10 | Seconds between status polls |
| `CODEX_POLL_TIMEOUT_MIN` | 5 | Max minutes to poll before cancel |

## Comparison with OOB Backends

| | Codex (this) | Claude |
|---|---|---|
| **Invocation** | `oob_ray_submit.py run -a codex` | `oob_ray_submit.py run -a claude` |
| **Latency (per iter)** | 30–120s | 1–5 min |
| **Latency (full round)** | 2–6 min | 3–15 min |
| **GPU on pod** | No | No |
| **Output** | Verified by calling skill | Verified by calling skill |
| **Tool use** | File I/O, shell | File I/O, shell |
| **Best for** | Fast Triton rewrites | Multi-step autonomous edits |

## Tracing Setup

At the **start** of OOB Codex usage (before the first `oob_ray_submit.py run`), record
the start timestamp for message-level cost correlation:

```bash
python3 $SCRIPTS_DIR/trace_action.py --component oob --action start --agent codex
```

After ALL OOB Codex tasks complete (all iterations done), record the end:

```bash
python3 $SCRIPTS_DIR/trace_action.py --component oob --action end
```

**NOTE:** Provider header injection is handled automatically by `auth_proxy.py`
inside the OOB workload pod when configured by bootstrap. No manual header
configuration is needed. The timestamps allow correlating OOB spend to specific
messages by querying provider spend logs with time ranges.

## CLI Sequence

| Step | Command | Purpose |
|------|---------|---------|
| 0 | `python3 $SCRIPTS_DIR/trace_action.py --component oob --action start --agent codex` | Record start timestamp (once) |
| 1 | `$OOB_RAY_CLI run -a codex -p "$PROMPT" -f kernel.py -o "$OUT_DIR" --max-turns 20 --timeout 300 --no-live --json` | Create, submit, poll, and return result |
| 2 | `jq -r .workspace` | Locate task workspace |
| 3 | `cp "$WORKSPACE/optimized_kernel.py" ...` | Collect optimized code |
| 4 | `python3 $SCRIPTS_DIR/trace_action.py --component oob --action end` | Record end timestamp |

## Prompt Template

The core optimization prompt follows the shared OOB rules in `actions/kernel-opt.md`.
Codex-specific differences:

- **No `mode` or `max_rounds`** — Codex has no concept of optimization modes
- **No `image` in prompt text** — image/environment is provided by the RayJob/CLI environment, not mentioned in the prompt body
- **Explicit output filename** — must tell Codex where to write

```
Optimize this Triton kernel for AMD MI355X (gfx950, CDNA4).

Hardware: 304 CUs, 256 VGPR/CU, HBM3e ~8 TB/s, MFMA instructions.
Context: LLM inference serving (decode path).
Input shapes: [exact shapes from TraceLens profile]
Data types: bf16 activations, fp8_e4m3 weights/KV cache.
Currently: {kernel_time_ms}ms per call, {gpu_pct}% of total GPU time.

MANDATORY CONSTRAINTS (violation = rejected):
1. The output function name MUST be EXACTLY: {original_function_name}. Do NOT rename it.
2. The function signature (parameter names, order, types) MUST be IDENTICAL to the original.
3. Block size limits: BLOCK_M <= 16, BLOCK_N <= 128, BLOCK_K <= 256.
4. Do NOT increase any block dimension beyond 2x its original value.
5. Do NOT add @triton.autotune or change @triton_heuristics decorators.

The kernel MUST be optimized to at least 1.5x speedup.
Do NOT search the filesystem with find / or grep -r /.

OPTIMIZATION TARGETS (prioritized):
1. STRUCTURAL: Hoist loop-invariant computations out of loops.
2. STRUCTURAL: Merge dual-pass into single-pass where possible.
3. TUNING: Adjust BLOCK sizes to match exact dimensions.
4. TUNING: Simplify grid indexing when dimensions are small.
5. MICRO: Use libdevice.rsqrt, multiply by reciprocal.

Write the COMPLETE optimized file to optimized_kernel.py.
```

### Optional System Prompt

```
You are an expert GPU kernel engineer specializing in AMD ROCm and Triton.
Target hardware: AMD MI355X (gfx950, CDNA4, wavefront size 64, 256KB LDS per CU,
304 CUs, HBM3e ~8TB/s, MFMA bf16).
Focus on: eliminating redundant memory loads, optimal block sizes for wavefront 64,
vectorized loads for HBM3e, register pressure management.
Return the COMPLETE optimized file — do not return partial snippets.
```

## Output Convention

- Codex writes the optimized kernel to `optimized_kernel.py` in its workspace
- The prompt MUST instruct Codex to use this filename
- Read directly from `<output-dir>/tasks/<user>/<task_id>/workspace/optimized_kernel.py`. The `oob_ray_submit.py run --json` result already exposes this via `.workspace`; use `$WORKSPACE/optimized_kernel.py` directly. There is **no** `output/` subdir.

## RayJob Execution

Each iteration is a **single blocking `oob_ray_submit.py run` invocation** submitted
inside the RayJob. Ray assigns a GPU, provisions a workspace, copies input files,
spawns the `codex` subprocess with the right env vars, and blocks until the task
reaches a terminal status.

### Single iteration

```bash
# $OOB_RAY_CLI = "python3 $SKILL_ROOT/scripts/oob_ray_submit.py" (set by setup.md)
OUT_DIR="$WORK_DIR/oob_codex_${KERNEL_NAME}_iter${ITER}"

RESULT_JSON=$($OOB_RAY_CLI run \
    -a codex \
    -p "$PROMPT" \
    -f "$WORK_DIR/kernel.py" \
    -o "$OUT_DIR" \
    --max-turns 20 \
    --timeout $((CODEX_POLL_TIMEOUT_MIN * 60)) \
    --no-live --json)

TASK_ID=$(echo "$RESULT_JSON"  | jq -r .task_id)
STATUS=$(echo "$RESULT_JSON"   | jq -r .status)
WORKSPACE=$(echo "$RESULT_JSON" | jq -r .workspace)

if [ "$STATUS" = "completed" ]; then
    OPTIMIZED="$WORKSPACE/optimized_kernel.py"
    [ -f "$OPTIMIZED" ] || { echo "MISSING_OUTPUT"; exit 1; }
fi
```

### Things that do NOT apply to CLI mode

- `image` and `workspace_id` — no additional SaFE workload is created; the agent CLI runs inside the RayJob.
  These args (and the `KERNEL_OPT_IMAGE` / `KERNEL_OPT_WORKSPACE` env vars) are silently ignored.
- `gpu_count`, `cpu`, `memory`, `ephemeral_storage`, `replicas`, `rdma` — these are
  K8s-workload knobs; the subprocess uses the RayJob resources.
- Service-side cancellation tools — use process signals instead.

## Iterative Refinement Loop

Codex and Claude have **no GPU** — they cannot compile or benchmark their own output.
The calling skill runs an iterative loop: submit → RayJob benchmark/validation → feed results back.

### Overview

```
One Round = OOB_ROUND_ITERATIONS (default 3) iterations.
Each iteration:
  1. Submit optimization task (with accumulated feedback context)
  2. Poll + download optimized kernel
  3. RayJob verification: compile → correctness → micro-benchmark
  4. Record result: {iteration, speedup, status, error_if_any}
  5. Append result to feedback context for next iteration

After all iterations: pick the result with the best verified speedup.
```

### Iteration Flow (pseudocode)

The same outer loop runs for Codex and Claude. `submit_and_wait()` is a single
`oob_ray_submit.py run` subprocess.

```python
best_result = None
feedback_context = ""

for i in range(OOB_ROUND_ITERATIONS):
    # 1. Build prompt with accumulated feedback
    prompt = base_prompt
    if feedback_context:
        prompt += f"\n\n--- PREVIOUS ITERATION RESULTS ---\n{feedback_context}"
        prompt += "\nUse these results to improve your optimization. Avoid repeating failed approaches."

    # 2. Submit + wait. Single blocking CLI call writes results to <workspace>.
    result = oob_run_blocking(
        agent="codex",  # or "claude"
        prompt=prompt,
        input_file=("kernel.py", original_kernel_source),
        max_turns=CODEX_MAX_TURNS,
        output_dir=f"{WORK_DIR}/oob_codex_iter{i+1}",
        timeout=CODEX_POLL_TIMEOUT_MIN * 60,
    )

    if result["status"] == "failed":
        # When the user passes --timeout to `oob run`, the agent subprocess is
        # killed after the deadline — but it may have already written output files.
        # OOB CLI (--json) returns "partial_outputs" listing workspace files.
        workspace = result.get("workspace")
        partial = result.get("partial_outputs", [])
        if "optimized_kernel.py" in partial and workspace:
            optimized_code = read_file(f"{workspace}/optimized_kernel.py")
            if optimized_code:
                # Fall through to verification below (compile → correctness → bench)
                pass
            else:
                feedback_context += f"\nIteration {i+1}: FAILED (timeout, output unreadable)"
                continue
        else:
            feedback_context += f"\nIteration {i+1}: FAILED — task error: {result.get('error')}"
            continue
    else:
        # 3. Read optimized kernel from the OOB workspace.
        optimized_code = read_optimized_kernel(result)
        if not optimized_code:
            feedback_context += f"\nIteration {i+1}: FAILED — no output file produced"
            continue

    # 4. RayJob verification (on the inference server / GPU runtime)
    compile_ok, compile_err = check_compilation(optimized_code)
    if not compile_ok:
        feedback_context += f"\nIteration {i+1}: COMPILE_FAIL — {compile_err}"
        continue

    correct, correctness_err = check_correctness(optimized_code)
    if not correct:
        feedback_context += f"\nIteration {i+1}: CORRECTNESS_FAIL — {correctness_err}"
        continue

    speedup = run_micro_benchmark(optimized_code, original_code)
    feedback_context += f"\nIteration {i+1}: speedup={speedup:.2f}x"

    if speedup > 1.0 and (best_result is None or speedup > best_result["speedup"]):
        best_result = {"iteration": i+1, "speedup": speedup, "code": optimized_code}

# 5. Return best result from the round
return best_result  # None if all iterations failed
```

### Feedback Context Format

Each iteration appends exactly one line to the feedback context:

```
Iteration 1: speedup=1.32x
Iteration 2: COMPILE_FAIL — NameError: name 'libdevice' is not defined
Iteration 3: CORRECTNESS_FAIL — torch.allclose failed: max diff=0.15
Iteration 4: speedup=1.51x
Iteration 5: speedup=1.48x
```

This gives the agent visibility into what worked and what failed, enabling it to:
- Avoid repeating compilation errors (e.g., add missing imports)
- Try different optimization strategies after low-speedup results
- Build on successful approaches from prior iterations

### Key Rules

1. **Always use the ORIGINAL kernel source** in `files[].content` or `-f` — never pass a previous iteration's output as the source. The
   agent should generate each attempt from scratch based on the original + feedback.
2. **Each iteration is a NEW task** — a fresh `oob_ray_submit.py run` invocation.
   Do not try to resume or modify a previous task.
3. **Verification runs inside the RayJob** (the environment with GPU access).
4. **Stop early** if `speedup >= 2.0x` — no need to exhaust all iterations.
5. **Stop early** if all iterations produce compilation errors — likely
   a fundamental issue with the kernel type for this backend.

### Verification Steps (per iteration)

| Step | Method | Failure → feedback |
|------|--------|-------------------|
| Compilation | `exec(compile(code, "kernel.py", "exec"))` | `COMPILE_FAIL — {error}` |
| Correctness | `torch.allclose(orig_out, opt_out, atol=1e-2, rtol=1e-2)` | `CORRECTNESS_FAIL — max diff={diff}` |
| Micro-benchmark | Time kernel execution (median of 100 runs, 10 warmup) | `speedup={x:.2f}x` |

## Behavioral Notes

- Codex typically completes in **1 turn** (~60 seconds) per iteration
- Output quality is strong for **Triton structural rewrites**: dual-loop to single-pass
  merges, block-size tuning, loop-invariant hoisting
- Handles `libdevice.rsqrt` import fallback correctly
- May struggle with complex HIP/C++ kernels; prefer Claude for deeper multi-step edits,
  or discard if neither OOB backend can produce a verifiable patch
- With feedback, later iterations often fix compilation issues from early iterations
- Best results typically appear in iterations 2–5 (after initial feedback)

## Troubleshooting

### Task completes but no optimized_kernel.py in outputs
- `ls <workspace>/` (the `oob_ray_submit.py run --json` result's `.workspace`) to see what Codex actually wrote
- Next iteration prompt will include this failure, prompting explicit file output

### Task fails immediately
- Inspect `<output-dir>/<task_id>/execution.log` and the JSON `error_message`
- Verify prompt is not empty and the input file (`-f`) is non-empty / readable

### All iterations produce compilation errors
- Kernel may be too complex for this backend (e.g., HIP/C++)
- Try the Claude OOB backend if enabled; otherwise discard this candidate with the
  compile errors recorded as feedback

### Best speedup is < 1.0x across all iterations
- Log as `discard` for this backend
- The other enabled OOB backend may still produce a better result
