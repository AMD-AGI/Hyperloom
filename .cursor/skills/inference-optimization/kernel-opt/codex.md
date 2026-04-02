---
name: codex-inference-kernel-reference
description: Codex backend for kernel optimization via OOB Agent MCP. Code generation without GPU — verification done by the calling skill. Referenced by actions/kernel-opt.md Step 2.
---

# Codex — Kernel Optimization Backend

Codex backend for kernel optimization via the OOB Agent MCP (`oci-oob-agent`).
Generates optimized kernel code without GPU access. The calling skill is responsible
for compilation checking, correctness verification, and micro-benchmarking.

## Status: Stable

Validated 2026-04-02. Typical completion: 1 turn, ~60 seconds. Output quality is
good for Triton structural rewrites (dual-loop to single-pass, block-size tuning).

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `CODEX_MAX_TURNS` | 20 | Max agent turns per task |
| `CODEX_POLL_INTERVAL_S` | 10 | Seconds between status polls |
| `CODEX_POLL_TIMEOUT_MIN` | 5 | Max minutes to poll before cancel |

## Comparison with Other Backends

| | Codex (this) | GEAK | Claude | LLM Proxy |
|---|---|---|---|---|
| **MCP** | OOB Agent | GEAK | OOB Agent | Direct API |
| **Latency** | 30–120s | 10–30 min | 1–5 min | 1–30s |
| **GPU on pod** | No | Yes | No | No |
| **Output** | Unverified | Verified on pod | Unverified | Unverified |
| **Tool use** | File I/O, shell | Bash, profiling, submit | File I/O, shell | None |
| **Best for** | Fast Triton rewrites | Complex HIP, final polish | Multi-step autonomous | Quick iteration |

## Tool Sequence

| Step | Tool | Purpose |
|------|------|---------|
| 1 | `agent_create_task` | Create task with kernel source + prompt |
| 2 | `agent_submit_task` | Start execution |
| 3 | `agent_get_task` | Poll status (every `CODEX_POLL_INTERVAL_S`) |
| 4 | `agent_get_outputs` | List output files |
| 5 | `agent_download_file` | Download optimized kernel |
| - | `agent_cancel_task` | Cancel if stuck past `CODEX_POLL_TIMEOUT_MIN` |

## agent_create_task — Critical Details

- `agent`: `"codex"` (required)
- `prompt`: kernel optimization instructions (see Prompt Template below)
- `files`: array of `{filename, content}` — full kernel source embedded here
- `max_turns`: use `CODEX_MAX_TURNS` (default 20)
- `system_prompt`: optional — GPU-expert persona can improve output quality
- There is NO `image`, `workspace_id`, or `gpu_count` — Codex has no GPU pod

### Example

```
Tool: agent_create_task
Args: {
    "agent": "codex",
    "prompt": "<optimization instructions — see template below>",
    "files": [
        {"filename": "kernel.py", "content": "<full kernel source>"}
    ],
    "max_turns": 20
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
- **No `image` reference** — Codex has no Docker image context
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

## Verification (Caller Responsibility)

Codex has no GPU — it cannot compile or benchmark. The calling skill MUST:

1. **Compilation check**: `exec(open("solution.py").read())`
2. **Correctness check**: `torch.allclose(original_output, optimized_output, atol=1e-2, rtol=1e-2)`
3. **Micro-benchmark**: compare kernel latency against original
4. **Integration**: patch via `patch_inductor.py` (see `actions/integrate.md`)

## Behavioral Notes

- Typically completes in **1 turn** (~60 seconds) — Codex treats kernel optimization
  as a single code-generation task
- Output quality is strong for **Triton structural rewrites**: dual-loop to single-pass
  merges, block-size tuning, loop-invariant hoisting
- Handles `libdevice.rsqrt` import fallback correctly
- May struggle with complex HIP/C++ kernels — use GEAK for those
- No reflection loop built-in — if compilation fails, create a new task with the
  error message appended to the prompt

## Troubleshooting

### Task completes but no optimized_kernel.py in outputs
- Check `agent_get_outputs` — file may have a different name
- Re-submit with explicit instruction: "Write the COMPLETE file to optimized_kernel.py"

### Task fails immediately
- Check `agent_get_task` `error` field
- Verify prompt is not empty and `files` array contains valid content

### Output does not compile
- Create a new task with the compilation error appended:
  `"Previous attempt failed to compile: {error}. Fix the issue."`
- Include both the original kernel and the failed attempt in `files`

### Output compiles but is slower
- Log as `discard`, try next backend result
- Consider providing more specific hardware constraints in the prompt
