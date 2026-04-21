---
name: codex-inference-kernel-reference
description: Codex backend for kernel optimization via OOB GPU Optimizer MCP. Code generation with optional GPU — verification done by the calling skill. Referenced by actions/kernel-opt.md Step 2.
---

# Codex — Kernel Optimization Backend

Codex backend for kernel optimization via the OOB GPU Optimizer MCP (`oob-gpu-optimizer`).
Generates optimized kernel code. The calling skill is responsible for compilation
checking, correctness verification, and micro-benchmarking.

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

## Comparison with Other Backends

| | Codex (this) | GEAK | Claude | LLM Proxy |
|---|---|---|---|---|
| **MCP** | OOB GPU Optimizer | GEAK | OOB GPU Optimizer | Direct API |
| **Latency (per iter)** | 30–120s | N/A | 1–5 min | 1–30s |
| **Latency (full round)** | 2–6 min | 10–30 min | 3–15 min | 1–30s |
| **GPU on pod** | No | Yes | No | No |
| **Output** | Locally verified | Verified on pod | Locally verified | Unverified |
| **Tool use** | File I/O, shell | Bash, profiling, submit | File I/O, shell | None |
| **Best for** | Fast Triton rewrites | Complex HIP, final polish | Multi-step autonomous | Quick iteration |

## Tracing Setup

At the **start** of OOB Codex usage (before the first `agent_create_task`), record
the start timestamp for message-level cost correlation:

```bash
python3 $SCRIPTS_DIR/trace_action.py --component oob --action start --agent codex
```

After ALL OOB Codex tasks complete (all iterations done), record the end:

```bash
python3 $SCRIPTS_DIR/trace_action.py --component oob --action end
```

**NOTE:** LLM header injection (`x-litellm-tags`, `x-litellm-spend-logs-metadata`)
is handled automatically by `auth_proxy.py` inside the OOB workload pod — no manual
header configuration needed (unlike GEAK). The timestamps allow correlating OOB's
LLM spend to specific messages by querying `LiteLLM_SpendLogs` with time ranges.

## Tool Sequence

| Step | Tool | Purpose |
|------|------|---------|
| 0 | `bash: trace_action.py --component oob --action start --agent codex` | Record start timestamp (once) |
| 1 | `agent_create_task` | Create task with kernel source + prompt |
| 2 | `agent_submit_task` | Start execution |
| 3 | `agent_get_task` | Poll status (every `CODEX_POLL_INTERVAL_S`) |
| 4 | `agent_get_outputs` | List output files |
| 5 | `agent_download_file` | Download optimized kernel |
| 6 | `bash: trace_action.py --component oob --action end` | Record end timestamp (once, after all iterations) |
| - | `agent_cancel_task` | Cancel if stuck past `CODEX_POLL_TIMEOUT_MIN` |

## agent_create_task — Critical Details

- `agent`: `"codex"` (required)
- `prompt`: kernel optimization instructions (see Prompt Template below)
- `files`: array of `{filename, content}` — full kernel source embedded here
- `max_turns`: use `CODEX_MAX_TURNS` (default 20)
- `system_prompt`: optional — GPU-expert persona can improve output quality
- `image`: optional — use `KERNEL_OPT_IMAGE` (shared with GEAK)
- `workspace_id`: optional — use `KERNEL_OPT_WORKSPACE` (shared with GEAK, default `"control-plane-moe"`)

### Example

```
Tool: agent_create_task
Args: {
    "agent": "codex",
    "prompt": "<optimization instructions — see template below>",
    "files": [
        {"filename": "kernel.py", "content": "<full kernel source>"}
    ],
    "max_turns": 20,
    "image": "KERNEL_OPT_IMAGE",
    "workspace_id": "KERNEL_OPT_WORKSPACE"
}
```

Then:
```
Tool: agent_submit_task
Args: { "task_id": "<task_id from create>" }
```

### Polling

```python
for attempt in range(CODEX_POLL_TIMEOUT_MIN * 60 // CODEX_POLL_INTERVAL_S):
    result = agent_get_task(task_id=TASK_ID)
    if result["status"] == "completed":
        break
    elif result["status"] == "failed":
        raise RuntimeError(f"Codex task failed: {result.get('error')}")
    time.sleep(CODEX_POLL_INTERVAL_S)
```

### Downloading Results

```python
outputs = agent_get_outputs(task_id=TASK_ID)
for f in outputs["files"]:
    if f["path"].endswith(".py") and "optimized" in f["path"]:
        result = agent_download_file(task_id=TASK_ID, file_path=f["path"])
        optimized_code = result["content"]
```

## Prompt Template

The core optimization prompt is shared with GEAK (see `actions/kernel-opt.md` for
the shared prompt rules). Codex-specific differences:

- **No `mode` or `max_rounds`** — Codex has no concept of optimization modes
- **No `image` in prompt text** — image is passed as MCP parameter (`image`), not mentioned in the prompt body
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
- Use `agent_get_outputs` to list files, then `agent_download_file` to retrieve

## Iterative Refinement Loop

Codex and Claude have **no GPU** — they cannot compile or benchmark their own output.
The calling skill runs an iterative loop: submit → local benchmark → feed results back.

### Overview

```
One Round = OOB_ROUND_ITERATIONS (default 3) iterations.
Each iteration:
  1. Submit optimization task (with accumulated feedback context)
  2. Poll + download optimized kernel
  3. LOCAL verification: compile → correctness → micro-benchmark
  4. Record result: {iteration, speedup, status, error_if_any}
  5. Append result to feedback context for next iteration

After all iterations: pick the result with the best verified speedup.
```

### Iteration Flow (pseudocode)

```python
best_result = None
feedback_context = ""

for i in range(OOB_ROUND_ITERATIONS):
    # 1. Build prompt with accumulated feedback
    prompt = base_prompt
    if feedback_context:
        prompt += f"\n\n--- PREVIOUS ITERATION RESULTS ---\n{feedback_context}"
        prompt += "\nUse these results to improve your optimization. Avoid repeating failed approaches."

    # 2. Submit task
    task = agent_create_task(
        agent="codex",  # or "claude"
        prompt=prompt,
        files=[{"filename": "kernel.py", "content": original_kernel_source}],
        max_turns=CODEX_MAX_TURNS,
        image=KERNEL_OPT_IMAGE,
        workspace_id=KERNEL_OPT_WORKSPACE,
    )
    agent_submit_task(task_id=task["task_id"])

    # 3. Poll until done
    result = poll_until_complete(task["task_id"])
    if result["status"] == "failed":
        feedback_context += f"\nIteration {i+1}: FAILED — task error: {result.get('error')}"
        continue

    # 4. Download optimized kernel
    optimized_code = download_optimized_kernel(task["task_id"])
    if not optimized_code:
        feedback_context += f"\nIteration {i+1}: FAILED — no output file produced"
        continue

    # 5. LOCAL verification (on the inference server or RayJob)
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

# 6. Return best result from the round
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

1. **Always use the ORIGINAL kernel source** in `files[].content` — never pass a
   previous iteration's output as the source. The agent should generate each attempt
   from scratch based on the original + feedback.
2. **Each iteration is a NEW task** (`agent_create_task` + `agent_submit_task`).
   Do not try to resume or modify a previous task.
3. **Verification runs locally** (on the machine with GPU access — the inference
   server in local mode, or the RayJob in claw mode).
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
- May struggle with complex HIP/C++ kernels — use GEAK for those
- With feedback, later iterations often fix compilation issues from early iterations
- Best results typically appear in iterations 2–5 (after initial feedback)

## Troubleshooting

### Task completes but no optimized_kernel.py in outputs
- Check `agent_get_outputs` — file may have a different name
- Next iteration prompt will include this failure, prompting explicit file output

### Task fails immediately
- Check `agent_get_task` `error` field
- Verify prompt is not empty and `files` array contains valid content

### All iterations produce compilation errors
- Kernel may be too complex for this backend (e.g., HIP/C++)
- Fall back to GEAK which has on-pod GPU compilation

### Best speedup is < 1.0x across all iterations
- Log as `discard` for this backend
- The other parallel backends (GEAK, LLM) may produce better results
