---
name: codex-inference-kernel-reference
description: Codex backend for kernel optimization via OOB GPU Optimizer CLI. Code generation with optional GPU — verification done by the calling skill. Referenced by actions/kernel-opt.md Step 2.
---

# Codex — Kernel Optimization Backend

Codex backend for kernel optimization via the OOB GPU Optimizer CLI (`oob_client.py`).
Generates optimized kernel code. The calling skill is responsible for compilation
checking, correctness verification, and micro-benchmarking.

OOB is accessed via direct REST API calls using the `oob_client.py` CLI wrapper.
The script lives at `$SKILL_ROOT/../shared/scripts/oob_client.py` and requires
no dependencies beyond Python stdlib.

### Authentication

Requires environment variables (set in `.env`):
- `OOB_API_URL` — OOB service base URL
  - Remote: `https://oci-slc.primus-safe.amd.com/control-plane/control-plane-sandbox/agent-mcp-server-zr29p`
  - Local: `http://localhost:8003`
- `OOB_AUTH_KEY` — Bearer token (shares the SaFE ak- key with GEAK)

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
| **Interface** | OOB CLI | GEAK CLI | OOB CLI | Direct API |
| **Latency (per iter)** | 30–120s | N/A | 1–5 min | 1–30s |
| **Latency (full round)** | 2–6 min | 10–30 min | 3–15 min | 1–30s |
| **GPU on pod** | No | Yes | No | No |
| **Output** | Locally verified | Verified on pod | Locally verified | Unverified |
| **Tool use** | File I/O, shell | Bash, profiling, submit | File I/O, shell | None |
| **Best for** | Fast Triton rewrites | Complex HIP, final polish | Multi-step autonomous | Quick iteration |

## Command Sequence

```bash
OOB_CLI="python3 $SKILL_ROOT/../shared/scripts/oob_client.py"
```

| Step | Command | Purpose |
|------|---------|---------|
| 1 | `$OOB_CLI create-task --agent codex --file kernel.py --prompt "..." --workspace-id $KERNEL_OPT_WORKSPACE --image $KERNEL_OPT_IMAGE --max-turns 20` | Create task with kernel source + prompt |
| 2 | `$OOB_CLI submit-task TASK_ID` | Start execution |
| 3 | `$OOB_CLI poll-task TASK_ID --interval 10 --timeout 300` | Poll status (every `CODEX_POLL_INTERVAL_S`) |
| 4 | `$OOB_CLI get-outputs TASK_ID` | List output files |
| 5 | `$OOB_CLI download-file TASK_ID FILE_PATH --output-dir $WORK_DIR/kernels` | Download optimized kernel |
| - | `$OOB_CLI cancel-task TASK_ID` | Cancel if stuck past `CODEX_POLL_TIMEOUT_MIN` |

## create-task — Critical Details

- `--agent codex` (required for this backend)
- `--prompt` — kernel optimization instructions (see Prompt Template below)
- `--file` — repeatable; full kernel source embedded here
- `--max-turns` — use `CODEX_MAX_TURNS` (default 20)
- `--system-prompt` — optional, GPU-expert persona can improve output quality
- `--image` — use `$KERNEL_OPT_IMAGE` (shared with GEAK)
- `--workspace-id` — required; use `$KERNEL_OPT_WORKSPACE` (shared with GEAK, default `"control-plane-moe"`)

### Example

```bash
$OOB_CLI create-task \
  --agent codex \
  --prompt "<optimization instructions — see template below>" \
  --file kernel.py \
  --max-turns 20 \
  --image "$KERNEL_OPT_IMAGE" \
  --workspace-id "$KERNEL_OPT_WORKSPACE"

# Then submit using the task_id from the response:
$OOB_CLI submit-task <TASK_ID>
```

### Polling

```bash
$OOB_CLI poll-task <TASK_ID> \
  --interval $CODEX_POLL_INTERVAL_S \
  --timeout $((CODEX_POLL_TIMEOUT_MIN * 60))
```

`poll-task` loops `get-task` and exits when the task reaches `completed`,
`failed`, or `cancelled`. Exit code 2 on timeout.

### Downloading Results

```bash
$OOB_CLI get-outputs <TASK_ID>
# Find the optimized file and download:
$OOB_CLI download-file <TASK_ID> "optimized_kernel.py" --output-dir "$WORK_DIR/kernels"
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
- Use `$OOB_CLI get-outputs` to list files, then `$OOB_CLI download-file` to retrieve

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

### Iteration Flow (pseudocode, shell)

```bash
best_speedup=0
best_iter=""
feedback_context=""

for i in $(seq 1 $OOB_ROUND_ITERATIONS); do
    # 1. Build prompt with accumulated feedback
    prompt="$base_prompt"
    if [ -n "$feedback_context" ]; then
        prompt+="\n\n--- PREVIOUS ITERATION RESULTS ---\n$feedback_context\n"
        prompt+="Use these results to improve your optimization. Avoid repeating failed approaches."
    fi

    # 2. Submit task
    RESULT=$($OOB_CLI create-task \
        --agent codex \
        --prompt "$prompt" \
        --file kernel.py \
        --max-turns $CODEX_MAX_TURNS \
        --image "$KERNEL_OPT_IMAGE" \
        --workspace-id "$KERNEL_OPT_WORKSPACE")
    TASK_ID=$(echo "$RESULT" | jq -r '.task_id // .id')
    $OOB_CLI submit-task "$TASK_ID"

    # 3. Poll until done
    $OOB_CLI poll-task "$TASK_ID" --interval 10 --timeout 300 > /tmp/poll_result.json || {
        feedback_context+="\nIteration $i: FAILED — task timed out"
        continue
    }
    STATUS=$(jq -r '.status' /tmp/poll_result.json)
    [ "$STATUS" != "completed" ] && {
        feedback_context+="\nIteration $i: FAILED — $STATUS"
        continue
    }

    # 4. Download optimized kernel
    $OOB_CLI download-file "$TASK_ID" "optimized_kernel.py" --output-dir "$WORK_DIR/iter_$i"

    # 5. LOCAL verification (on the inference server or RayJob)
    # check_compilation / check_correctness / run_micro_benchmark are skill-side helpers
    # Record result in feedback_context per iteration:
    #   "Iteration $i: speedup=1.32x"
    #   "Iteration $i: COMPILE_FAIL — NameError: ..."
done

# 6. Return best result from the round
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

1. **Always use the ORIGINAL kernel source** via `--file` — never pass a
   previous iteration's output as the source. The agent should generate each attempt
   from scratch based on the original + feedback.
2. **Each iteration is a NEW task** (`$OOB_CLI create-task` + `$OOB_CLI submit-task`).
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
- Check `$OOB_CLI get-outputs` — file may have a different name
- Next iteration prompt will include this failure, prompting explicit file output

### Task fails immediately
- Check `$OOB_CLI get-task` `error` field
- Verify prompt is not empty and at least one `--file` argument is provided

### All iterations produce compilation errors
- Kernel may be too complex for this backend (e.g., HIP/C++)
- Fall back to GEAK which has on-pod GPU compilation

### Best speedup is < 1.0x across all iterations
- Log as `discard` for this backend
- The other parallel backends (GEAK, LLM) may produce better results
