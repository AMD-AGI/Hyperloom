---
name: claude-inference-kernel-reference
description: Claude Code backend for kernel optimization. In local/claw modes uses the OOB GPU Optimizer MCP; in fully-local mode uses the `oob run` CLI. Multi-turn agent with tool-use capability. Verification done by the calling skill. Referenced by actions/kernel-opt.md Step 2.
---

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
OOB_CLI="${OOB_CLI:-oob}"
OUT_DIR="$WORK_DIR/oob_claude_${KERNEL_NAME}_iter${ITER}"

RESULT_JSON=$($OOB_CLI run \
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
