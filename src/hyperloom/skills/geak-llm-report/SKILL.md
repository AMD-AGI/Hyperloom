---
name: geak-llm-report
description: Render the phase → agent → API call → tool call report for a GEAK run, standalone or inside a Hyperloom session — every call's ISL/OSL, tokens (input/thinking/output/cache), wall-clock, derived USD and tool calls. Use when asked where GEAK's time or money went, which GEAK phase or agent dominates a run, whether a task is cheap enough to delegate to a smaller model, or to compare two models' GEAK runs.
---

# GEAK per-LLM-call report

Answers: *for every LLM API call a GEAK run made, what did it cost in tokens,
seconds and dollars, and which phase and agent did it belong to?*

## Where the data comes from

GEAK issues almost no LLM calls itself. `interface/run_e2e.py` opens one
`ClaudeSDKClient` and hands a single prompt to Claude Code, which runs
`e2e_workflow/e2e_workflow.js` — a Workflow script that declares its phases and
tags every `agent()` call with `{phase, label}`. **Claude Code already records
that structure**, so this report reads existing files; no GEAK code writes a
ledger and none needs to.

| File | Holds |
| --- | --- |
| `<claude_home>/projects/<slug>/<session>/workflows/wf_*.json` | phases, `args.eval_dir`, `workflowProgress` — one entry per agent with label, phase, model, tool calls, `durationMs` |
| `<session>/subagents/workflows/<runId>/agent-<agentId>.jsonl` | one row per API call: `usage`, `model`, `apiBlockIndex`, `timestamp`, `tool_use` blocks |

## Steps

1. **Find the run.** Selection is an identity match on `args.eval_dir`, never a
   guess by mtime:

   ```bash
   PYTHONPATH=src python3 -m hyperloom.inference_optimizer.tools.dump_geak_call_report \
       --session-dir <HYPERLOOM_SESSION_DIR> --list
   ```

   A run made by a GEAK build that mirrors its own ledger also carries a copy at
   `<eval_dir>/llm_trace/` — pass that as `--claude-home` and everything below
   works unchanged. Prefer it: it is the copy that survives the container the
   run's Claude home lived in.

   `--eval-dir <path>` and `--run-id wf_...` also select; `--eval-dir` matches
   `args.eval_dir` first and falls back to `args.exp_root`, so it finds a
   standalone GEAK run as well as a Hyperloom-driven one. `--list` with no
   selector at all lists every record in every home, which is how to find a run
   whose paths you do not know. Archived runs sit in another user's home — add
   `--claude-home /shared_nfs/<user>/.claude` (repeatable). Confirm the
   candidate you meant before rendering.

2. **Render.** From the Hyperloom checkout:

   ```bash
   PYTHONPATH=src python3 -m hyperloom.inference_optimizer.tools.dump_geak_call_report \
       --run-id wf_ec8b57b0-1a7 --claude-home /shared_nfs/chaox/.claude
   ```

   Writes `geak_call_report.md` and `geak_call_report.json` to
   `$USER_DATA_PATH/reports/<runId>/`. `--max-depth 2` gives the
   phase → agent table alone, which is what a "where did the time go" question
   usually wants; the full depth bottoms out at tool names and runs to
   thousands of rows.

3. **For the cross-hierarchy view**, add `--join-hyperloom <SESSION_DIR>`. The
   GEAK tree is nested under the session's `KERNEL_AGENT` phase, so one document
   carries Hyperloom phases on top and GEAK phases inside. Rows the GEAK
   harvester already wrote into that session's `ext/` shard are dropped first,
   so the run is not billed twice.

4. **Read the Coverage section before quoting a number.** It is printed first
   for that reason.

## Options that widen the report

| Flag | Effect |
| --- | --- |
| `--include-orchestrator` | Folds in the launching SDK conversation from `<session>.jsonl` as an `(orchestrator)` phase. It is a couple of calls against the workflow's thousands — include it for a complete bill, omit it to keep the tree about the workflow. |
| `--include-text` | Also writes `geak_calls.jsonl`: one row per API call with `prompt_text`, `thinking_text` and `output_text` beside every metric, tagged with its phase, agent and call index. This is the artifact to read when deciding *which* tasks are cheap enough to delegate — the tree says where the money went, the sidecar says what was being asked. |
| `--text-chars N` | Caps each captured field (default 4000; `0` uncaps). Truncation is marked inline as `[+N chars]`, never silent. Uncapped on a large run produces a multi-gigabyte file. |
| `--no-nested` | Suppresses grafting of nested workflow records. |

`prompt_text` is the *increment* the model was handed since it last spoke — the
agent prompt on call 0, tool results after that — not the full context. The full
context is what `isl` counts.

## Reading the numbers

- **ISL** = `input_tokens + cache_read + cache_creation`. Cache read dominates
  by two orders of magnitude in these runs; a token total that ignores it
  understates the bill enormously, and one that includes it is not comparable
  to a "tokens" figure quoted from anywhere else. Say which you mean.
- **Thinking is a subset of output here** — the opposite of Hyperloom's own
  ledger. `OSL = output_tokens` alone. Adding thinking on top double-counts it.
- **Cost is always derived** from the shipped rate card; Claude Code records no
  provider cost. An unpriced model is excluded from the USD total, never counted
  as free, and Coverage says how many calls that was.
- **Thinking seconds are always 0.** Only per-message timestamps exist. Thinking
  *tokens* are exact.
- **Per-call time is inferred** from consecutive timestamps, so an agent's first
  call absorbs its queue wait. Agent-level time is the runtime's own
  `durationMs` and is authoritative.
- **Summed agent durations exceed the run's wall-clock** wherever the script
  fanned out. The record's `durationMs` is the wall-clock; quote that for
  elapsed time and the sum for compute-time attribution.
- **A repeated label gets `#2`, `#3`** suffixes — those are separate agents
  (a retry, or an unnumbered fan-out), not duplicates to merge.

## Reconciliation

The report prints the record's own figures beside the leaf-derived ones. Tool
calls, agent count and per-agent duration reconcile exactly; **tokens do not,
and are not expected to** — `workflowProgress[].tokens` is a streamed snapshot
that lands on either side of the transcript sums agent by agent. The transcripts
are authoritative for tokens. A tool-call or agent-count mismatch, by contrast,
means a transcript is missing and should be investigated.

## Do not

- Do not modify anything under a GEAK checkout to produce this report; nothing
  needs instrumenting.
- Do not run a report against `/shared_nfs/chaox/GEAK` expecting it to match a
  local checkout's phase list — the deployed build carries phases no local
  checkout has, so a missing phase there is a build difference, not a bug.
- Do not treat a nested workflow's numbers as certain. Records carry no parent
  linkage, so nesting is joined by time containment within the session and
  Coverage labels it as such. Every archived run inspected so far has
  `spawnDepth: 1` and no nesting at all.
- Do not present the `--join-hyperloom` total as the session bill when the
  session's own ledger has no `KERNEL_AGENT` rows; in that case Hyperloom never
  harvested GEAK, and the two hierarchies were measured independently.
