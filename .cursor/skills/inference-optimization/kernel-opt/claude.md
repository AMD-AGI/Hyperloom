---
name: claude-inference-kernel-reference
description: Claude Code backend for kernel optimization via OOB Agent MCP. Multi-turn agent with tool-use capability. No GPU — verification done by the calling skill. Referenced by actions/kernel-opt.md Step 2.
---

# Claude Code — Kernel Optimization Backend

Claude Code backend for kernel optimization via the OOB Agent MCP (`oci-oob-agent`).
Multi-turn agent with tool-use capability (file I/O, shell commands). No GPU access —
the calling skill is responsible for compilation checking, correctness verification,
and micro-benchmarking.

## Status: Experimental

Claude Code support in the OOB Agent MCP is under active development. The tool
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
| **MCP** | OOB Agent | OOB Agent | GEAK | Direct API |
| **Latency (per iter)** | 1–5 min | 30–120s | N/A | 1–30s |
| **Latency (full round)** | 3–15 min | 2–6 min | 10–30 min | 1–30s |
| **GPU on pod** | No | No | Yes | No |
| **Tool use** | File I/O, shell, multi-step | File I/O, shell | Bash, profiling, submit | None |
| **Multi-turn** | Yes (up to 30 turns) | Yes (typically 1) | Yes (up to 100 steps) | No |
| **Best for** | Multi-step: analyze, write, verify compilation | Fast single-pass rewrites | Complex HIP, hardware-verified | Quick iteration |

## Tool Sequence

Identical to Codex — same MCP, different `agent` parameter.

| Step | Tool | Purpose |
|------|------|---------|
| 1 | `agent_create_task` | Create task with kernel source + prompt |
| 2 | `agent_submit_task` | Start execution |
| 3 | `agent_get_task` | Poll status (every `CLAUDE_POLL_INTERVAL_S`) |
| 4 | `agent_get_outputs` | List output files |
| 5 | `agent_download_file` | Download optimized kernel |
| - | `agent_cancel_task` | Cancel if stuck past `CLAUDE_POLL_TIMEOUT_MIN` |

## agent_create_task — Critical Details

- `agent`: `"claude"` (required)
- `prompt`: kernel optimization instructions (same core as Codex — see [`codex.md`](codex.md))
- `files`: array of `{filename, content}` — full kernel source embedded here
- `max_turns`: use `CLAUDE_MAX_TURNS` (default 30, higher than Codex because Claude
  benefits from multi-step reasoning)
- `system_prompt`: recommended — Claude responds well to detailed persona prompts
- There is NO `image`, `workspace_id`, or `gpu_count` — Claude has no GPU pod

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
    "system_prompt": "<GPU expert persona — see codex.md>"
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
no `max_rounds`, no `image`).

Claude benefits from more detailed reasoning prompts. Consider adding:

```
Think step by step:
1. Analyze the original kernel structure — identify redundant memory loads and loop patterns.
2. Determine if a single-pass merge is safe (check R0_BLOCK vs r0_numel).
3. Write the optimized kernel preserving the exact function signature.
4. Verify the optimized kernel handles edge cases (mask boundaries, data types).
Write the COMPLETE optimized file to optimized_kernel.py.
```

## Output Convention

Same as Codex: optimized kernel written to `optimized_kernel.py`.

## Iterative Refinement Loop

Claude uses the **same iterative refinement loop** as Codex. See [`codex.md`](codex.md)
"Iterative Refinement Loop" section for the full flow, pseudocode, feedback context
format, and key rules.

The only difference: use `agent="claude"` and Claude-specific constants
(`CLAUDE_MAX_TURNS`, `CLAUDE_POLL_INTERVAL_S`, `CLAUDE_POLL_TIMEOUT_MIN`).

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
