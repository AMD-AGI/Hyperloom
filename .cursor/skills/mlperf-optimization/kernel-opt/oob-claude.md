---
name: oob-claude-mlperf-kernel-reference
description: Claude Code backend for kernel optimization via OOB Agent MCP. Multi-turn agent with tool-use capability. No GPU — verification done by the calling skill. Referenced by actions/kernel-opt.md Step 2.
---

# OOB-Claude — Kernel Optimization Backend

Claude Code backend for kernel optimization via the OOB Agent MCP (`oob-optimizer-dev`).
Multi-turn agent with tool-use capability (file I/O, shell commands). No GPU access —
the calling skill is responsible for compilation checking, correctness verification,
and micro-benchmarking.

## Status: Experimental

Claude Code support in the OOB Agent MCP is under active development. The tool
interface is identical to Codex (`agent_create_task(agent="claude")`), but availability
may be intermittent. **If `claude` is unavailable, the other parallel backends
(GEAK, Codex) continue independently.**

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `OOB_ROUND_ITERATIONS` | 3 | Iterations per round — shared with Codex (see [`oob-codex.md`](oob-codex.md)) |
| `CLAUDE_MAX_TURNS` | 30 | Max agent turns per task (per iteration) |
| `CLAUDE_POLL_INTERVAL_S` | 15 | Seconds between status polls |
| `CLAUDE_POLL_TIMEOUT_MIN` | 10 | Max minutes to poll before cancel |

## Comparison with Other Backends

| | OOB-Claude (this) | OOB-Codex | GEAK |
|---|---|---|---|
| **MCP** | OOB Agent | OOB Agent | GEAK |
| **Latency (per iter)** | 1–5 min | 30–120s | N/A |
| **Latency (full round)** | 3–15 min | 2–6 min | 10–30 min |
| **GPU on pod** | No | No | Yes |
| **Tool use** | File I/O, shell, multi-step | File I/O, shell | Bash, profiling, submit |
| **Multi-turn** | Yes (up to 30 turns) | Yes (typically 1) | Yes (up to 100 steps) |
| **Best for** | Multi-step: analyze, write, verify compilation | Fast single-pass rewrites | Complex HIP, hardware-verified |

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
- `prompt`: kernel optimization instructions (same core as Codex — see [`oob-codex.md`](oob-codex.md))
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
    "prompt": "<optimization instructions — same template as oob-codex.md>",
    "files": [
        {"filename": "kernel.py", "content": "<full kernel source>"}
    ],
    "max_turns": 30,
    "system_prompt": "<GPU expert persona — see oob-codex.md>"
}
```

Then:
```
Tool: agent_submit_task
Args: { "task_id": "<task_id from create>" }
```

### Polling and Downloading

Same pattern as Codex. See [`oob-codex.md`](oob-codex.md) for polling loop and download examples.
Use `CLAUDE_POLL_INTERVAL_S` and `CLAUDE_POLL_TIMEOUT_MIN` instead of Codex constants.

## Prompt Template

Same core prompt as Codex (see [`oob-codex.md`](oob-codex.md) Prompt Template section). All shared
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

Claude uses the **same iterative refinement loop** as Codex. See [`oob-codex.md`](oob-codex.md)
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
- **Experimental**: If `agent_create_task(agent="claude")` returns an error, the
  other parallel backends (GEAK, Codex) continue independently.

## Troubleshooting

### agent_create_task returns error
- Claude backend may be unavailable. This does NOT block the run — GEAK and Codex
  continue in parallel. Log the error and proceed.

### Task runs but produces no output file
- Claude may have spent all turns on analysis without writing a file.
- Feedback context will capture this, and the next iteration prompt will be more
  directive. If persistent, the early-stop rule (all iterations fail) triggers.

### Other issues
- Same troubleshooting as Codex (see [`oob-codex.md`](oob-codex.md) Troubleshooting section).
