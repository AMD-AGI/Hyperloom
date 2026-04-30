# OOB Backends Guide (kernel agent reference)

> Migrated + merged from `src/inference_optimizer/kernel_opt/{codex,claude,llm}.md`
> as part of Plan A (kernel agent ownership). The kernel agent reads
> this on demand from `actions/run_optimization.md` whenever an OOB
> (codex / claude / llm) backend is invoked.
>
> Three previously-separate documents now live as sections below:
> §A Codex backend, §B Claude Code backend, §C LLM (PRISM/SAFE) backend.
> Cross-references in the original docs that point at GEAK live in
> `geak_guide.md`; references to the executor-side `actions/kernel-opt.md`
> now route through the kernel agent's `actions/run_optimization.md`.

---

# §A Codex Backend


# Codex — Kernel Optimization Backend

Codex backend for kernel optimization. Two transport modes:

| Mode | How Codex is invoked |
|------|----------------------|
| `local` / `claw` | OOB GPU Optimizer MCP (`agent_create_task` etc.) |
| `fully-local` | `oob_ray_submit.py run -a codex ...` CLI (single blocking subprocess per iteration) |

Tool surface and prompt template are identical across modes; only the call
mechanism differs. Generates optimized kernel code. The calling skill is
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
- **MCP modes:** Use `agent_get_outputs` to list files, then `agent_download_file` to retrieve
- **Fully-local mode:** Read directly from `<output-dir>/tasks/<user>/<task_id>/workspace/optimized_kernel.py`. The `oob_ray_submit.py run --json` result already exposes this via `.workspace`; use `$WORKSPACE/optimized_kernel.py` directly. There is **no** `output/` subdir.

## Fully-Local Execution

In fully-local mode each iteration is a **single blocking `oob_ray_submit.py run`
invocation**. Ray assigns a GPU, provisions a workspace, copies input files, spawns
the `codex` subprocess with the right env vars, and blocks until the task reaches
a terminal status.

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

### CLI ↔ MCP mapping

| MCP tool | CLI equivalent |
|----------|----------------|
| `agent_create_task(agent="codex", prompt, files, max_turns, ...)` | `oob_ray_submit.py run -a codex -p ... -f ... --max-turns ...` (single call) |
| `agent_submit_task` | (folded into `oob_ray_submit.py run`) |
| `agent_get_task` (poll) | (folded into `oob_ray_submit.py run`, polls internally) |
| `agent_get_outputs` | `ls <workspace>/` (the `oob_ray_submit.py run --json` result's `.workspace` field already points at the live dir) |
| `agent_download_file` | `cp <workspace>/<file>` (file is already on local disk) |
| `agent_cancel_task` | `kill -INT <oob-pid>` (graceful) |

### Things that do NOT apply to fully-local

- `image` and `workspace_id` — no SaFE workload is created; the agent CLI runs in-container.
  These args (and the `KERNEL_OPT_IMAGE` / `KERNEL_OPT_WORKSPACE` env vars) are silently ignored.
- `gpu_count`, `cpu`, `memory`, `ephemeral_storage`, `replicas`, `rdma` — these are
  K8s-workload knobs; in fully-local mode the subprocess uses whatever the container has.
- `agent_cancel_task` MCP tool — use process signals instead.

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

The same outer loop runs in all modes; only `submit_and_wait()` differs.
For fully-local, `submit_and_wait()` is a single `oob_ray_submit.py run` subprocess.
For MCP modes, it is the `agent_create_task` + `agent_submit_task` + poll sequence.

```python
best_result = None
feedback_context = ""

for i in range(OOB_ROUND_ITERATIONS):
    # 1. Build prompt with accumulated feedback
    prompt = base_prompt
    if feedback_context:
        prompt += f"\n\n--- PREVIOUS ITERATION RESULTS ---\n{feedback_context}"
        prompt += "\nUse these results to improve your optimization. Avoid repeating failed approaches."

    # 2. Submit + wait (mode-specific)
    if MODE == "fully-local":
        # Single blocking CLI call writes results to <workspace> (== task workspace dir)
        result = oob_run_blocking(
            agent="codex",  # or "claude"
            prompt=prompt,
            input_file=("kernel.py", original_kernel_source),
            max_turns=CODEX_MAX_TURNS,
            output_dir=f"{WORK_DIR}/oob_codex_iter{i+1}",
            timeout=CODEX_POLL_TIMEOUT_MIN * 60,
        )
    else:
        task = agent_create_task(
            agent="codex",  # or "claude"
            prompt=prompt,
            files=[{"filename": "kernel.py", "content": original_kernel_source}],
            max_turns=CODEX_MAX_TURNS,
            image=KERNEL_OPT_IMAGE,
            workspace_id=KERNEL_OPT_WORKSPACE,
        )
        agent_submit_task(task_id=task["task_id"])
        result = poll_until_complete(task["task_id"])

    if result["status"] == "failed":
        feedback_context += f"\nIteration {i+1}: FAILED — task error: {result.get('error')}"
        continue

    # 3. Read optimized kernel (already on local disk in fully-local;
    #    requires agent_download_file in MCP modes)
    optimized_code = read_optimized_kernel(result)
    if not optimized_code:
        feedback_context += f"\nIteration {i+1}: FAILED — no output file produced"
        continue

    # 4. LOCAL verification (on the inference server or RayJob)
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

1. **Always use the ORIGINAL kernel source** in `files[].content` (or `-f` in
   fully-local) — never pass a previous iteration's output as the source. The
   agent should generate each attempt from scratch based on the original + feedback.
2. **Each iteration is a NEW task** — `agent_create_task` + `agent_submit_task` in
   MCP modes, a fresh `oob_ray_submit.py run` invocation in fully-local. Do not try to resume
   or modify a previous task.
3. **Verification runs locally** (on the machine with GPU access — the inference
   server in local / fully-local mode, or the RayJob in claw mode).
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
- **MCP modes:** Check `agent_get_outputs` — file may have a different name
- **Fully-local:** `ls <workspace>/` (the `oob_ray_submit.py run --json` result's `.workspace`) to see what Codex actually wrote
- Next iteration prompt will include this failure, prompting explicit file output

### Task fails immediately
- **MCP modes:** Check `agent_get_task` `error` field
- **Fully-local:** Inspect `<output-dir>/<task_id>/execution.log` and the JSON `error_message`
- Verify prompt is not empty and the input file (`-f`) is non-empty / readable

### All iterations produce compilation errors
- Kernel may be too complex for this backend (e.g., HIP/C++)
- Fall back to GEAK which has on-pod GPU compilation

### Best speedup is < 1.0x across all iterations
- Log as `discard` for this backend
- The other parallel backends (GEAK, LLM) may produce better results

---

# §B Claude Code Backend


# Claude Code — Kernel Optimization Backend

Claude Code backend for kernel optimization. Two transport modes:

| Mode | How Claude is invoked |
|------|-----------------------|
| `local` / `claw` | OOB GPU Optimizer MCP (`agent_create_task(agent="claude")` etc.) |
| `fully-local` | `oob_ray_submit.py run -a claude ...` CLI (single blocking subprocess per iteration) |

Multi-turn agent with tool-use capability (file I/O, shell commands).
The calling skill is responsible for compilation checking, correctness verification,
and micro-benchmarking.

## Status: Experimental

Claude Code support in the OOB GPU Optimizer MCP is under active development. The tool
interface is identical to Codex (`agent_create_task(agent="claude")`), but availability
may be intermittent. **Fallback to `codex` if `claude` is unavailable.**

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `OOB_ROUND_ITERATIONS` | 3 | Iterations per round — shared with Codex (see [`codex.md`](codex.md)) |
| `CLAUDE_MAX_TURNS` | 30 | Max agent turns per task (per iteration) |
| `CLAUDE_POLL_INTERVAL_S` | 15 | Seconds between status polls |
| `CLAUDE_POLL_TIMEOUT_MIN` | 10 | Max minutes to poll before cancel |

## Comparison with Other Backends

| | Claude (this) | Codex | GEAK | LLM Proxy |
|---|---|---|---|---|
| **MCP** | OOB GPU Optimizer | OOB GPU Optimizer | GEAK | Direct API |
| **Latency (per iter)** | 1–5 min | 30–120s | N/A | 1–30s |
| **Latency (full round)** | 3–15 min | 2–6 min | 10–30 min | 1–30s |
| **GPU on pod** | No | No | Yes | No |
| **Tool use** | File I/O, shell, multi-step | File I/O, shell | Bash, profiling, submit | None |
| **Multi-turn** | Yes (up to 30 turns) | Yes (typically 1) | Yes (up to 100 steps) | No |
| **Best for** | Multi-step: analyze, write, verify compilation | Fast single-pass rewrites | Complex HIP, hardware-verified | Quick iteration |

## Tracing Setup

At the **start** of OOB Claude usage (before the first `agent_create_task`), record
the start timestamp for message-level cost correlation:

```bash
python3 $SCRIPTS_DIR/trace_action.py --component oob --action start --agent claude
```

After ALL OOB Claude tasks complete (all iterations done), record the end:

```bash
python3 $SCRIPTS_DIR/trace_action.py --component oob --action end
```

**NOTE:** LLM header injection (`x-litellm-tags`, `x-litellm-spend-logs-metadata`)
is handled automatically by `auth_proxy.py` inside the OOB workload pod — no manual
header configuration needed (unlike GEAK). The timestamps allow correlating OOB's
LLM spend to specific messages by querying `LiteLLM_SpendLogs` with time ranges.

## Tool Sequence

Identical to Codex — same MCP, different `agent` parameter.

| Step | Tool | Purpose |
|------|------|---------|
| 0 | `bash: trace_action.py --component oob --action start --agent claude` | Record start timestamp (once) |
| 1 | `agent_create_task` | Create task with kernel source + prompt |
| 2 | `agent_submit_task` | Start execution |
| 3 | `agent_get_task` | Poll status (every `CLAUDE_POLL_INTERVAL_S`) |
| 4 | `agent_get_outputs` | List output files |
| 5 | `agent_download_file` | Download optimized kernel |
| 6 | `bash: trace_action.py --component oob --action end` | Record end timestamp (once, after all iterations) |
| - | `agent_cancel_task` | Cancel if stuck past `CLAUDE_POLL_TIMEOUT_MIN` |

## agent_create_task — Critical Details

- `agent`: `"claude"` (required)
- `prompt`: kernel optimization instructions (same core as Codex — see [`codex.md`](codex.md))
- `files`: array of `{filename, content}` — full kernel source embedded here
- `max_turns`: use `CLAUDE_MAX_TURNS` (default 30, higher than Codex because Claude
  benefits from multi-step reasoning)
- `system_prompt`: recommended — Claude responds well to detailed persona prompts
- `image`: optional — use `KERNEL_OPT_IMAGE` (shared with GEAK)
- `workspace_id`: optional — use `KERNEL_OPT_WORKSPACE` (shared with GEAK, default `"control-plane-moe"`)

### Example

```
Tool: agent_create_task
Args: {
    "agent": "claude",
    "prompt": "<optimization instructions — same template as codex.md>",
    "files": [
        {"filename": "kernel.py", "content": "<full kernel source>"}
    ],
    "max_turns": 30,
    "system_prompt": "<GPU expert persona — see codex.md>",
    "image": "KERNEL_OPT_IMAGE",
    "workspace_id": "KERNEL_OPT_WORKSPACE"
}
```

Then:
```
Tool: agent_submit_task
Args: { "task_id": "<task_id from create>" }
```

### Polling and Downloading

Same pattern as Codex. See [`codex.md`](codex.md) for polling loop and download examples.
Use `CLAUDE_POLL_INTERVAL_S` and `CLAUDE_POLL_TIMEOUT_MIN` instead of Codex constants.

## Prompt Template

Same core prompt as Codex (see [`codex.md`](codex.md) Prompt Template section). All shared
prompt rules from `actions/kernel-opt.md` apply. No GEAK-specific directives (no `mode`,
no `max_rounds`). Image is passed as MCP parameter, not in prompt text.

### MANDATORY CONSTRAINTS (must appear verbatim in every Claude prompt)

Claude is multi-turn with `Bash` access; without explicit constraints it will explore the
filesystem and try to run benchmarks itself, burning turns and producing no `optimized_kernel.py`.
Always include the following block at the top of the prompt (in addition to the shared
constraints from `codex.md`):

```
MANDATORY CONSTRAINTS (violation = rejected):
1. The output function name MUST be EXACTLY: {original_function_name}. Do NOT rename it.
2. The function signature (parameter names, order, types) MUST be IDENTICAL to the original.
3. Block size limits: BLOCK_M <= 16, BLOCK_N <= 128, BLOCK_K <= 256.
4. Do NOT increase any block dimension beyond 2x its original value.
5. Do NOT add @triton.autotune or change @triton_heuristics decorators.
6. Do NOT search the filesystem with `find /` or `grep -r /`. The kernel source
   is the file passed via `-f` (or `files[].content`) — work only from that.
7. Write the COMPLETE optimized file to `optimized_kernel.py` in the current
   working directory. Do NOT write anywhere else.
```

Claude also benefits from a brief reasoning scaffold appended after the constraints:

```
Think step by step:
1. Analyze the original kernel structure — identify redundant memory loads and loop patterns.
2. Determine if a single-pass merge is safe (check R0_BLOCK vs r0_numel).
3. Write the optimized kernel preserving the exact function signature.
4. Verify edge cases hold (mask boundaries, dtypes).
Then write the COMPLETE optimized file to optimized_kernel.py and exit.
```

## Output Convention

Same as Codex: optimized kernel written to `optimized_kernel.py` in the task workspace.

- **MCP modes:** Use `agent_get_outputs` to list, then `agent_download_file`.
- **Fully-local mode:** The `oob_ray_submit.py run --json` result's `.workspace` field points at the
  task workspace. Read `$WORKSPACE/optimized_kernel.py` directly. There is no
  `output/` subdir.

## Fully-Local Execution

Identical to Codex's "Fully-Local Execution" section in [`codex.md`](codex.md);
swap `-a codex` → `-a claude` and use Claude-specific constants:

```bash
# $OOB_RAY_CLI = "python3 $SKILL_ROOT/scripts/oob_ray_submit.py" (set by setup.md)
OUT_DIR="$WORK_DIR/oob_claude_${KERNEL_NAME}_iter${ITER}"

RESULT_JSON=$($OOB_RAY_CLI run \
    -a claude \
    -p "$PROMPT" \
    -f "$WORK_DIR/kernel.py" \
    -o "$OUT_DIR" \
    --max-turns 30 \
    --timeout $((CLAUDE_POLL_TIMEOUT_MIN * 60)) \
    --no-live --json)
```

The same CLI ↔ MCP mapping table from [`codex.md`](codex.md) applies; Claude's
longer per-iteration latency just means a larger `--timeout` budget.

## Iterative Refinement Loop

Claude uses the **same iterative refinement loop** as Codex. See [`codex.md`](codex.md)
"Iterative Refinement Loop" section for the full flow, pseudocode, feedback context
format, and key rules.

The only difference: use `agent="claude"` (or `oob_ray_submit.py run -a claude` in fully-local) and
Claude-specific constants (`CLAUDE_MAX_TURNS`, `CLAUDE_POLL_INTERVAL_S`,
`CLAUDE_POLL_TIMEOUT_MIN`).

Claude's multi-turn capability means it may produce higher quality output per
iteration (at the cost of higher latency). With feedback from prior iterations,
Claude is particularly effective at fixing compilation errors and avoiding
previously-failed approaches.

## Behavioral Notes

- **Multi-turn capable**: Claude may spend 2–5 turns analyzing the kernel before
  producing output. This is expected and usually produces higher-quality results.
- **Tool use**: Claude can read the input file, write intermediate analysis, and
  iterate on the solution within a single task. This makes it better at complex
  optimizations that require structural understanding.
- **Higher latency**: 1–5 minutes vs Codex's 30–120 seconds per iteration. Total
  round time: ~3–15 min for 3 iterations (vs Codex ~2–6 min).
- **Feedback responsive**: Claude excels at incorporating iteration feedback —
  typically fixes compilation errors within 1–2 feedback iterations.
- **Experimental**: If `agent_create_task(agent="claude")` returns an error, fall
  back to `codex` automatically.

## Troubleshooting

### agent_create_task returns error
- Claude backend may be unavailable. Fall back to Codex:
  `agent_create_task(agent="codex", ...)` with the same prompt and files.

### Task runs but produces no output file
- Claude may have spent all turns on analysis without writing a file.
- Feedback context will capture this, and the next iteration prompt will be more
  directive. If persistent, the early-stop rule (5 consecutive failures) triggers.

### Other issues
- Same troubleshooting as Codex (see [`codex.md`](codex.md) Troubleshooting section).

---

# §C LLM Backend (PRISM SAFE LLM proxy)


# LLM Inference Kernel Optimization — Deep Reference

This document provides detailed reference material for LLM-based kernel optimization in the inference optimization loop defined in `SKILL.md`. LLM and GEAK are **run in parallel** for each candidate kernel — same candidate selection, same integration paths, but the result with the best micro-benchmark speedup wins. See also [`geak.md`](geak.md) for GEAK-specific details.

## Relationship to GEAK

| | GEAK | LLM (this skill) |
|---|---|---|
| **Backend** | GEAK MCP → remote GPU pod with AI agent | PRISM SAFE LLM proxy → Claude / GPT |
| **Latency** | 10–30 min (pod scheduling + agent) | 1–30s (direct API call) |
| **GPU access** | Yes — hardware-in-loop micro-benchmark | No — LLM writes code, you benchmark in your serving env |
| **Output** | Verified kernel (compiled + benchmarked on pod) | Unverified kernel (must compile + benchmark locally) |
| **Best for** | Complex HIP kernels, final polish, high-confidence | Fast iteration, Triton rewrites, GEAK pods overloaded |
| **Cost** | GPU pod time (shared cluster) | API tokens (~$0.01–0.50 per call) |

**Strategy: always run both in parallel.** For every candidate kernel, submit to GEAK and LLM simultaneously. LLM results arrive in seconds; GEAK results arrive in minutes. While waiting for GEAK, verify + micro-benchmark LLM results locally. When both are done, pick the winner by micro-benchmark speedup.

**Where each backend shines** (both still run; this explains which tends to win):
- **LLM wins more often on**: Triton structural rewrites (dual-loop → single-pass), simple block-size tuning, kernels where multi-model diversity finds creative solutions
- **GEAK wins more often on**: Complex HIP/C++ kernels, cases needing compile → test → fix iteration on real hardware, kernels with subtle correctness constraints

**Fallback behavior:**
- If GEAK fails (pod timeout, all 3 retries exhausted) → use best LLM result
- If all LLM models fail (compilation errors after 3 reflection rounds) → use GEAK result
- If both fail → skip kernel, move to next candidate

## Prerequisites

- `pip install openai httpx` (likely already installed)
- `LLM_PROXY_API_KEY` set in `.env` (starts with `ak-`)
- A profiling trace analyzed (TraceLens or manual kernel breakdown from SKILL.md Phase 3-5)
- Kernel source code extracted (from Inductor cache or framework source)

## Available Models (validated 2026-03-28)

Gateway: `https://oci-slc.primus-safe.amd.com/api/v1/llm-proxy/v1`

| Model | Provider | Status | Latency | Recommendation |
|---|---|---|---|---|
| `claude-opus-4-6` | Anthropic | **Working** | ~24s | Best for complex structural optimizations. Produces multi-variant solutions. |
| `claude-opus-4.5` | Anthropic | **Working** | ~1s | Good for quick block-size tuning, simple rewrites. |
| `gpt-4.1` | OpenAI | **Working** | ~2s | Fast but may use invalid Triton APIs. Always verify. |
| `gpt-5.2` | OpenAI | **Broken** | — | 400 BadRequest. Do not use. |

## Inference Kernel Categories (Same as GEAK)

| Kernel pattern | Framework | Source available? | LLM target? |
|----------------|-----------|-------------------|-------------|
| `Cijk_Ailk_Bljk_*` | hipBLASLt | No (compiled) | No — vendor BLAS |
| `aiter::fmha_v3_fwd` | aiter | No (.so) by default | No by default — **Yes if user provides source** |
| `moe_ck2stages_gemm*` | aiter | No (.so) by default | No by default — **Yes if user provides source** |
| `aiter::fmoe_*`, `moe_sorting_*` | aiter | No (.so) by default | No by default — **Yes if user provides source** |
| `triton_*` from SGLang | SGLang | Yes (Python) | **Yes** |
| `triton_poi_*`, `triton_red_*` | torch.compile | Yes (Inductor cache) | **Yes** — primary target |
| `vectorized_elementwise_kernel` | PyTorch | No (C++) | Maybe — try torch.compile first |
| Custom HIP `__global__` | User code | Yes | **Yes** |

**User-provided source override:** When the user specifies kernel source paths (e.g.,
`/opt/aiter/csrc/`, `/opt/sglang/`), kernels found at those paths are LLM targets
regardless of the default classification above. Map trace kernel names back to source
files using `rg` in the provided repo. Include full source in the prompt.

## Tracing Setup

Before the first LLM API call, record the start timestamp:

```bash
python3 $SCRIPTS_DIR/trace_action.py --component llm --action start
```

After all LLM calls and reflection rounds complete, record the end:

```bash
python3 $SCRIPTS_DIR/trace_action.py --component llm --action end
```

Additionally, inject tracing headers into the `OpenAI` client so LLM spend is
attributed to the correct session. See Step 3 below for the `default_headers`
parameter in the client constructor.

## LLM Optimization Flow

### Step 1: Extract kernel source (same as GEAK)

**Strategy A (torch.compile mode):** Extract from STANDALONE Inductor files.

```bash
find /tmp/torchinductor_root -name "*.py" | while read f; do
    if grep -q "@triton_heuristics" "$f" && \
       ! grep -q "async_compile\|def call(" "$f"; then
        echo "STANDALONE: $f"
    fi
done
```

**Strategy B (no torch.compile):** Find framework source kernels.

```bash
find /opt/venv -path "*/sglang/srt/layers/*.py" -exec grep -l "@triton.jit" {} \;
find /sgl-workspace/aiter -name "*.py" -exec grep -l "@triton.jit" {} \;
```

### Step 2: Build the prompt

The prompt quality directly determines output quality. Include all available context:

```python
SYSTEM_PROMPT = (
    "You are an expert GPU kernel engineer specializing in AMD ROCm and Triton. "
    "Target hardware: AMD MI355X (gfx950, CDNA4, wavefront size 64, 256KB LDS per CU, "
    "304 CUs, HBM3e ~8TB/s, MFMA bf16). "
    "Context: LLM inference serving (decode path). "
    "When optimizing kernels, return the COMPLETE optimized file in a ```python code block. "
    "Focus on: eliminating redundant memory loads, optimal block sizes for wavefront 64, "
    "vectorized loads for HBM3e, register pressure management."
)
```

**For RMSNorm / reduction kernels (highest-impact target):**

```python
USER_PROMPT = f"""Optimize this Triton kernel for AMD MI355X (gfx950).

HARDWARE: gfx950, 304 CUs, HBM3e ~8TB/s, MFMA bf16, wavefront 64, 65536 VGPRs per CU.
SHAPES: xnumel={xnumel} (batch rows), r0_numel={r0_numel} (hidden_dim).
CURRENT: {gpu_pct}% of GPU time, called {call_count} times per forward pass in LLM decode.

MANDATORY CONSTRAINTS:
1. Function name MUST be EXACTLY: `{original_function_name}`. Do NOT rename.
2. Function signature MUST be IDENTICAL to original.
3. The decorator MUST be preserved.
4. R0_BLOCK <= {r0_block}. Do NOT increase beyond original value.
5. MUST produce numerically identical output.

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

Return the COMPLETE optimized file."""
```

**For general kernels:**

```python
USER_PROMPT = f"""Optimize this Triton kernel for AMD MI355X (gfx950).

HARDWARE: gfx950, 304 CUs, HBM3e ~8TB/s, MFMA bf16, wavefront 64.
SHAPES: {shapes_from_trace}
CURRENT: {gpu_pct}% of GPU time, called {call_count} times per forward pass.

MANDATORY CONSTRAINTS:
1. Function name MUST be EXACTLY: `{original_function_name}`. Do NOT rename.
2. Function signature MUST be IDENTICAL to original.
3. Decorators MUST be preserved.
4. Do NOT increase block sizes beyond 2x original values.

OPTIMIZATION TARGETS (prioritized):
1. Eliminate redundant memory loads (merge dual-pass into single-pass)
2. Hoist loop-invariant computations
3. Adjust BLOCK sizes to match exact dimensions
4. Use libdevice.rsqrt (NOT tl.math.rsqrt)
5. Simplify grid indexing when dimensions are small

```python
{kernel_source}
```

Return the COMPLETE optimized file."""
```

### Step 3: Call the LLM

```python
import json
import os
from openai import OpenAI
import httpx

http_client = httpx.Client(verify=False, timeout=180)

session_id = os.environ.get("SESSION_ID", "")
tracing_headers = {
    "x-litellm-tags": "product:primus-claw,component:llm",
}
if session_id:
    tracing_headers["x-litellm-spend-logs-metadata"] = json.dumps({
        "session_id": session_id,
        "component": "llm",
    })

client = OpenAI(
    base_url="https://oci-slc.primus-safe.amd.com/api/v1/llm-proxy/v1",
    api_key=os.environ["LLM_PROXY_API_KEY"],
    http_client=http_client,
    default_headers=tracing_headers,
)

response = client.chat.completions.create(
    model="claude-opus-4-6",   # or claude-opus-4.5 / gpt-4.1
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ],
    max_tokens=8192,
    temperature=0.0,
)

optimized_code = response.choices[0].message.content
```

### Step 4: Extract and save the code

```python
import re

code_match = re.search(r'```python\n(.*?)```', optimized_code, re.DOTALL)
if code_match:
    with open("solution.py", "w") as f:
        f.write(code_match.group(1))
```

### Step 5: Verify compilation

```python
try:
    exec(open("solution.py").read())
    print("Compilation: PASS")
except Exception as e:
    print(f"Compilation: FAIL — {e}")
    # Feed error back for reflection (see below)
```

### Step 6: Parallel multi-model optimization (optional)

For maximum coverage, try multiple models on the same kernel simultaneously:

```python
import concurrent.futures

models = ["claude-opus-4-6", "claude-opus-4.5", "gpt-4.1"]

def optimize_with_model(model):
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
        max_tokens=8192,
        temperature=0.0,
    )
    return model, resp.choices[0].message.content

with concurrent.futures.ThreadPoolExecutor() as pool:
    results = list(pool.map(optimize_with_model, models))

# Test each result for compilation + correctness, pick the best
for model, code in results:
    try:
        # extract code, compile, correctness check, micro-benchmark
        ...
    except Exception:
        continue
```

## Integration Paths (Same as GEAK)

After the LLM returns optimized code, integrate using the same paths as [`geak.md`](geak.md):

### Strategy A: Standalone File Patching (torch.compile mode)

Use the `patch_standalone_kernels()` function from SKILL.md Phase 8a. The function:
- Finds all standalone kernel files matching the kernel name
- Adapts xnumel per file
- Checks r0_numel safety (single-pass only when R0_BLOCK = r0_numel)
- Backs up originals, patches, clears binary cache

```python
patch_standalone_kernels(kernel_name, "solution.py", target_signature_pattern)
```

Then kill and restart the server.

### Strategy B: Direct Source Edit (no torch.compile)

Use AST-based function replacement from SKILL.md Phase 8a Strategy B:

```python
replace_function_ast(original_source, func_name, geak_source)
```

Then clear `__pycache__` and restart.

## Reflection Loop

If the first attempt fails or regresses, feed results back to the LLM:

| Issue | Append to conversation |
|-------|----------------------|
| Compilation error | `Your solution failed to compile: {error}. Fix it.` |
| Correctness failure | `Your solution produces wrong output (max diff={diff}). Fix the computation logic.` |
| Performance regression | `Your solution is correct but {X}% slower. The bottleneck is {detail}. Eliminate redundant memory loads.` |
| Improvement < target | `Your solution achieves {Z}% speedup. Target is 10%+. Try: {suggestion}.` |

Max 3 reflection iterations per kernel. Use `temperature=0.0` for deterministic output.

## Parallel Race with GEAK

**Both backends run simultaneously for every candidate kernel.** The winner is determined by micro-benchmark speedup after correctness verification.

### Why parallel is better than sequential

1. **No wall-clock penalty**: GEAK pod scheduling (5–30 min) is the bottleneck. LLM calls (1–30s) complete during GEAK's wait time, so running both costs no extra time.
2. **Diversity wins**: LLM multi-model (3 models) + GEAK = 4 independent optimization attempts per kernel. More attempts = higher chance of finding a good structural optimization.
3. **Complementary strengths**: LLM excels at structural rewrites (dual-loop → single-pass); GEAK excels at hardware-verified tuning. The better result wins.
4. **Graceful degradation**: If one backend fails (GEAK pod timeout, LLM compilation error), the other still produces a result.

### LLM advantages (tend to make LLM the winner for Triton kernels)

1. **Speed**: 1–30s vs 10–30 min. Enables 20+ iterations in the time GEAK does 1.
2. **Multi-model**: Try Claude + GPT in parallel, pick the best.
3. **Reflection loop**: Feed compilation/correctness errors back instantly.
4. **Cheap experimentation**: Test wild ideas (aggressive fusion, entirely new algorithms) without GPU cost.

### GEAK advantages (tend to make GEAK the winner for complex kernels)

1. **Hardware verification**: GEAK compiles and benchmarks on real GPU — output is pre-validated.
2. **Iteration on hardware**: GEAK's mini-swe-agent does compile → test → fix cycles on the actual target GPU.
3. **HIP/C++ support**: GEAK can optimize non-Python kernels.

### Workflow (Phase 7 of SKILL.md)

For each candidate kernel:
1. Submit to GEAK **and** LLM (all 3 models) simultaneously
2. LLM results arrive first → verify compilation + correctness + micro-benchmark each
3. GEAK result arrives later → verify + micro-benchmark
4. **Compare**: pick the result with the best micro-benchmark speedup
5. Patch the winner → E2E benchmark → keep/revert

If one backend produces no valid result, the other wins by default. If neither produces a valid result, skip the kernel.

## Knowledge Base: LLM Inference Kernel Lessons

### Gateway Configuration (validated 2026-03-28)

```
Base URL: https://oci-slc.primus-safe.amd.com/api/v1/llm-proxy/v1
Key format: ak-... (PRISM SAFE API key, set as LLM_PROXY_API_KEY in .env)
SSL: Must use httpx.Client(verify=False)
```

### Model-Specific Observations

- **claude-opus-4-6**: Produces thorough solutions with multiple variants (single-pass + multi-block fallback). Uses `tl.math.rsqrt` correctly. Best for RMSNorm dual-loop → single-pass merges. ~24s latency.
- **claude-opus-4.5**: Concise, correct, fast (~1s). Good enough for block-size tuning.
- **gpt-4.1**: Fast (~2s) but uses invalid Triton APIs (`tl.shared_memory`, `tl.barrier()`). Always test compilation before benchmarking. Better for brainstorming optimization ideas than producing runnable code.

### Same Integration Caveats as GEAK

- **MUST patch STANDALONE files, NOT graph module inline source** (Strategy A)
- **Clear ALL binary caches** (.so, .json, ~/.triton/cache) after patching
- **Kill server and restart** — SGLang loads kernels at startup
- **Wait 10+ seconds** between server kill and relaunch
- **r0_numel > R0_BLOCK**: single-pass is UNSAFE — do NOT eliminate loops
- **Always `unset PROFILE SGLANG_TORCH_PROFILER_DIR`** after profiling

### When LLM Kernel Optimization Is Not Worth It

Same criteria as GEAK (see [`geak.md`](geak.md)):
- Kernel is <3% of total GPU time
- Kernel is from vendor library (aiter, hipBLASLt, CK)
- All compute is in vendor C++/ASM (>50% GPU time)
- Model is GEMM-dominated with vendor BLAS
