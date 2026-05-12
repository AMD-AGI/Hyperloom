---
name: claude-inference-kernel-reference
description: Claude Code backend for kernel optimization via `oob_ray_submit.py run` (Ray-scheduled CLI). Multi-turn agent with tool-use capability. Verification done by the calling skill. Referenced by actions/kernel-opt.md Step 2.
---

# Claude Code — Kernel Optimization Backend

Claude Code backend for kernel optimization. In this remote-only skill, Claude is
invoked inside the RayJob through the Ray-scheduled OOB CLI transport:

| Runtime | How Claude is invoked |
|---------|-----------------------|
| RayJob | `oob_ray_submit.py run -a claude ...` CLI (single blocking subprocess per iteration) |

Multi-turn agent with tool-use capability (file I/O, shell commands).
The calling skill is responsible for compilation checking, correctness verification,
and micro-benchmarking.

## Status: Experimental

Claude Code support depends on the `claude` CLI being installed in the runtime
image. **Fallback to `codex` if `claude` is unavailable.**

## Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `OOB_ROUND_ITERATIONS` | 3 | Iterations per round — shared with Codex (see [`codex.md`](codex.md)) |
| `CLAUDE_MAX_TURNS` | 30 | Max agent turns per task (per iteration) |
| `CLAUDE_POLL_INTERVAL_S` | 15 | Seconds between status polls |
| `CLAUDE_POLL_TIMEOUT_MIN` | 10 | Max minutes to poll before cancel |

## Comparison with OOB Backends

| | Claude (this) | Codex |
|---|---|---|
| **Invocation** | `oob_ray_submit.py run -a claude` | `oob_ray_submit.py run -a codex` |
| **Latency (per iter)** | 1–5 min | 30–120s |
| **Latency (full round)** | 3–15 min | 2–6 min |
| **GPU on pod** | No | No |
| **Tool use** | File I/O, shell, multi-step | File I/O, shell |
| **Multi-turn** | Yes (up to 30 turns) | Yes (typically 1) |
| **Best for** | Multi-step autonomous edits | Fast single-pass rewrites |

## Tracing Setup

At the **start** of OOB Claude usage (before the first `oob_ray_submit.py run`), record
the start timestamp for message-level cost correlation:

```bash
python3 $SCRIPTS_DIR/trace_action.py --component oob --action start --agent claude
```

After ALL OOB Claude tasks complete (all iterations done), record the end:

```bash
python3 $SCRIPTS_DIR/trace_action.py --component oob --action end
```

**NOTE:** Provider header injection is handled automatically by `auth_proxy.py`
inside the OOB workload pod when configured by bootstrap. No manual header
configuration is needed. The timestamps allow correlating OOB spend to specific
messages by querying provider spend logs with time ranges.

## CLI Sequence

Identical to Codex — same `oob_ray_submit.py run` flow, different `-a` value.

```bash
$OOB_RAY_CLI run \
  -a claude \
  -p "$PROMPT" \
  -f "$WORK_DIR/kernel.py" \
  -o "$WORK_DIR/oob_claude_${KERNEL_NAME}" \
  --max-turns 30 \
  --timeout $((CLAUDE_POLL_TIMEOUT_MIN * 60)) \
  --no-live --json
```

## Prompt Template

Same core prompt as Codex (see [`codex.md`](codex.md) Prompt Template section). All shared
OOB prompt rules from `actions/kernel-opt.md` apply. Image/environment is provided by
the RayJob/CLI environment, not in prompt text.

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

- The `oob_ray_submit.py run --json` result's `.workspace` field points at the
  task workspace. Read `$WORKSPACE/optimized_kernel.py` directly. There is no
  `output/` subdir.

## RayJob Execution

Identical to Codex's "RayJob Execution" section in [`codex.md`](codex.md);
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

The same CLI output convention from [`codex.md`](codex.md) applies; Claude's
longer per-iteration latency just means a larger `--timeout` budget.

## Iterative Refinement Loop

Claude uses the **same iterative refinement loop** as Codex. See [`codex.md`](codex.md)
"Iterative Refinement Loop" section for the full flow, pseudocode, feedback context
format, and key rules.

The only difference: use `oob_ray_submit.py run -a claude` and
Claude-specific constants (`CLAUDE_MAX_TURNS`, `CLAUDE_POLL_INTERVAL_S`,
`CLAUDE_POLL_TIMEOUT_MIN`).

**Timeout recovery:** When the user passes `--timeout` to `oob run` and the agent
exceeds that deadline, OOB kills the subprocess and marks the task as `failed` — but
the agent may have already written output files. The OOB JSON result includes a
`partial_outputs` field listing workspace files; check it before discarding the
iteration. Claude's multi-turn nature makes timeout more likely, but it often writes
intermediate files before the deadline. See `codex.md` pseudocode for the exact
salvage logic.

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
- **Experimental**: If the `claude` CLI/backend is unavailable, fall back to
  `codex` automatically.

## Troubleshooting

### Claude CLI/backend unavailable
- Fall back to Codex with the same prompt and files:
  `oob_ray_submit.py run -a codex ...`
- In Core42 RayJob remote mode, a valid Claude model can appear unavailable if
  the OOB auth proxy returns `404`. To repair the OOB/Claude environment,
  use the Core42 Anthropic pass-through smoke-test guidance from
  [`modes/REMOTE.md`](../modes/REMOTE.md), then rerun OOB; do not treat this
  as a replacement for `oob_ray_submit.py run`. Set
  `ANTHROPIC_BASE_URL=https://core42.example-internal-host.invalid/api/v1/llm-proxy` and
  `ANTHROPIC_CUSTOM_HEADERS="Authorization: Bearer ${ANTHROPIC_API_KEY}"`.

### Task runs but produces no output file
- Claude may have spent all turns on analysis without writing a file.
- Feedback context will capture this, and the next iteration prompt will be more
  directive. If persistent, the early-stop rule (5 consecutive failures) triggers.

### Other issues
- Same troubleshooting as Codex (see [`codex.md`](codex.md) Troubleshooting section).
