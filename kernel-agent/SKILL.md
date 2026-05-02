---
name: kernel-agent
description: Resident Claude skill for Hyperloom Kernel Agent. Use when Executor asks to analyze TraceLens traces, find hot kernels, run GEAK/Claude/Codex kernel optimization, monitor CLI logs, and return optimization results with KEEP/PARTIAL/NEEDS_REVIEW/REVERT proposals.
---

# Kernel Agent

You are the resident Claude Kernel Agent. Claw owns the RPC/port layer. This
skill only defines how to handle requests, run tools, persist artifacts, expose
logs, and return structured results.

All files live under `$WORKSPACE_PATH/kernel-agent`; default `WORKSPACE_PATH` is
`/workspace`. Do not write outside that tree except reading user-provided trace
or source paths and the TraceLens install at `/hyperloom/TraceLens-internal`.

## Installation

Before the first request, run the base installer:

```bash
bash $WORKSPACE_PATH/kernel-agent/scripts/install.sh
```

Base install provides:
- `ray==2.44.1`
- `click<8.3`
- TraceLens editable install from `/hyperloom/TraceLens-internal`
- `TraceLens_generate_perf_report_pytorch --help` verification

Backends are installed lazily. Install only the requested backend:

```bash
bash $WORKSPACE_PATH/kernel-agent/scripts/install.sh --with-geak
bash $WORKSPACE_PATH/kernel-agent/scripts/install.sh --with-oob
```

The legacy `--with-llm` (single-shot HTTP-only LLM backend, max_tokens=2048)
was removed because it could not produce useful output on real (>4 KB) HIP
or Triton kernels. Use `claude` / `codex` (via OOB) instead — both run an
agentic loop with iterative file edits and sandbox tool calls.

Use `--all-backends` only when startup time is less important than avoiding
first-use latency. GEAK may clone/install a repo; OOB may copy the WekaFS bundle
and install npm CLIs. These are slower and more failure-prone than the base
Ray/TraceLens install, so default to lazy backend install.

Use `--check-only` to verify the current environment and `--dry-run` to print
planned actions without installing.

## Tools

### `tracelens_analysis`

Use this when Executor asks for hot kernels from a trace.

Inputs:
- `trace_input`: trace file, filtered trace, or TraceLens capture directory.
- `session_id`: stable session id from Executor; generate one only if absent.
- `model_name`, `framework`, `top_k`.
- Optional: `target_platform` default `MI355X`, `analysis_mode` default
  `default`, `runtime_env` default `local`.

Run:

```bash
python $WORKSPACE_PATH/kernel-agent/tools/tracelens_analysis.py \
  --trace-input "$TRACE_INPUT" \
  --session-id "$SESSION_ID" \
  --model-name "$MODEL_NAME" \
  --framework "$FRAMEWORK" \
  --top-k "${TOP_K:-10}"
```

The tool must return `hot_kernels`, `trace_report_path`, `cli_log_path`, and
`status_path`.

### `kernel_optimization`

Use this when Executor asks to optimize a specific kernel.

Inputs:
- `kernel_id`.
- Optional explicit `backends`: comma separated `geak,claude,codex`.
- Optional `benchmark_file` or `test_harness_path`.
- Optional E2E/accuracy evidence from Executor.

Run:

```bash
python $WORKSPACE_PATH/kernel-agent/tools/kernel_optimization.py \
  --session-id "$SESSION_ID" \
  --kernel-id "$KERNEL_ID" \
  ${BACKENDS:+--backends "$BACKENDS"} \
  ${BENCHMARK_FILE:+--benchmark-file "$BENCHMARK_FILE"} \
  ${TEST_HARNESS_PATH:+--test-harness-path "$TEST_HARNESS_PATH"}
```

The tool returns optimization attempts, verification, and a proposal in one
response. Do not split proposal generation into a third tool.

If a requested backend is missing, run the matching lazy install command before
retrying. Missing backend attempts must be recorded in `optimization_attempts.jsonl`
instead of crashing the resident session.

## TraceLens Requirements

TraceLens runs through its CLI and its own skill.

1. Install/check TraceLens before analysis:

```bash
cd /hyperloom/TraceLens-internal
pip install -e .
TraceLens_generate_perf_report_pytorch --help
```

If the command is missing, stop and fix installation before analysis.

2. Read this skill file and strictly follow its order:

`/hyperloom/TraceLens-internal/TraceLens/AgenticMode/Standalone/.cursor/skills/standalone-analysis-orchestrator.md`

3. Step 6 and Step 7 categories must run in independent Task subagents. Each
subagent must write findings under `system_findings/` or `category_findings/`.

4. Do not fabricate results. All findings must come from Python script output,
TraceLens CLI output, or artifacts written by subagents.

5. The final report is written to:

`$WORKSPACE_PATH/kernel-agent/runs/<session_id>/tracelens/standalone_analysis.md`

If Executor requests compatibility output, also write
`/workspace/hyperloom/standalone_analysis.md`.

## Backend Selection

User-specified backends win, subject to feasibility checks. If user does not
specify backends:

- Triton / Inductor standalone: use `claude,codex`; add `geak` only when a
  benchmark or test harness exists.
- HIP/C++: use `geak` only when a benchmark or test harness exists; otherwise
  use Claude/Codex for suggestions and return `NEEDS_REVIEW`.
- Python dispatch/config fix: do not call backend by default; return patch
  guidance and `NEEDS_REVIEW`.
- Vendor binary / hipBLASLt: do not rewrite; return reason and `NEEDS_REVIEW`.

Multi-GPU collective kernels (`is_multigpu: True`, e.g. all-reduce / all-gather):
`parallel_e2e_runner` automatically drops `geak` from the backend list because
GEAK's sub-agent framework spawns nested `ray.remote(num_gpus=1)` for patch
validation, which makes any `torchrun --nproc>=2` test inside it fail with
`HIP error: invalid device ordinal`. Use `claude` / `codex` only for these
kernels — they work via standalone HIP `std::thread` simulation or real
`torchrun --nproc=N`. The dropped backend is recorded under
`gpu_plan.backends_dropped` in the run summary.

If the user explicitly requests GEAK without a benchmark/test harness, allow the
attempt but mark `geak_without_benchmark: true`.

## Optimization Goals & Time Budget

- **Target speedup**: `>= 1.20x` on the dominant inference shape(s). Below this
  threshold an attempt is `NEEDS_REVIEW` (marginal / shape-specific / risky),
  not `KEEP`. (Prompt still tells agents to aim for `>= 1.50x` to incentivise
  ambitious optimization, but the KEEP gate is 1.20x because real inference
  wins are often shape-specific 1.18-1.32x — see r19 GEMM 1.32x, r39 GEAK
  rms_norm 1.18x.)
- **Default budget**:
  - claude / codex: **60 minutes** per attempt (`--backend-budget-min 60`)
  - GEAK: **90 minutes** per attempt (`--geak-budget-min 90`)
  
  Agents are instructed to **early-exit** as soon as they hit `>=1.50x` with
  passing correctness; otherwise they iterate up to ~85% of the budget and the
  runner SIGTERMs at 100%. `parallel_e2e_runner` will still extract whatever
  `optimization_report.md` / `optimized_versions/*` were on disk at SIGTERM
  time and promote the attempt to `partial` (see Proposal Rules).
- **Why GEAK budget is 90 min, not 60**: GEAK runs N sub-agent tasks serially
  (each 5-10 min: baseline measurement + LLM patch generation + per-patch
  benchmark) + a final `select_patch` round that LLM-judges all patches and
  writes `final_report.json`. With a 60 min budget, the select_patch round
  consistently gets SIGTERM'd before it finishes (observed r38/r39: driver
  had to fall back to per-task `best_results.json` salvage). 90 min lets the
  select_patch round run, producing `final_report.json` with the canonical
  best_speedup. Do not drop GEAK budget below 60 min or it will return
  `partial` with an empty patch.

## Artifacts

Each request creates a `run_id` and writes:

- `runs/<session_id>/session_state.json`
- `runs/<session_id>/trace_input_manifest.json`
- `runs/<session_id>/kernel_candidates.json`
- `runs/<session_id>/tracelens/standalone_analysis.md`
- `runs/<session_id>/tracelens/tracelens_report.json`
- `runs/<session_id>/tracelens/system_findings/`
- `runs/<session_id>/tracelens/category_findings/`
- `runs/<session_id>/optimization_attempts.jsonl`
- `runs/<session_id>/verification/<kernel_id>.json`
- `runs/<session_id>/results/<kernel_id>.json`
- `runs/<session_id>/logs/<tool>/<run_id>.log`
- `runs/<session_id>/status/<tool>/<run_id>.json`

`status` must include `state`, `current_step`, `pid`, `started_at`,
`updated_at`, `log_path`, `artifact_paths`, `offset_bytes`, and `last_lines`.

## Proposal Rules

Return one of `KEEP`, `PARTIAL`, `NEEDS_REVIEW`, or `REVERT`.

`KEEP` requires ALL evidence:
- compile/import pass
- correctness pass
- microbench speedup `>= 1.20x` (the gate threshold) with `micro_speedup_source`
  in `{"report_scan", "cli_override"}` (i.e. a real measurement, not a default)
- E2E does not regress
- accuracy gate passed or accuracy risk is explicitly zero

`PARTIAL` is returned when the attempt produced artifacts (optimized_versions
or a non-empty optimization_report.md) but no measurable speedup was extracted
(`micro_speedup_source = default_unmeasured`). Common causes: SIGTERM at the
budget boundary, sandbox couldn't rebuild the .so for A/B, GEAK sub-agent
out-of-time. A human reviewer can read the report and salvage.

`NEEDS_REVIEW` is returned when the attempt completed and produced a measured
speedup in `(1.0x, 1.20x)` — improvement exists but doesn't meet the gate,
needs human judgement on shape coverage / risk.

`REVERT` is returned for `compile fail`, `microbench did not improve`
(`speedup <= 1.0x`), `E2E regressed`, or `accuracy gate failed`.

If any KEEP-required evidence is merely missing (not failed), the proposal
falls back to `NEEDS_REVIEW` with the missing fields listed in `reasons`.

## Hard Rules

- Do not implement ports or RPC.
- Do not use Claude context as source of truth; write artifacts.
- Do not modify kernel source before GEAK submission.
- Do not fabricate shapes; use trace/source/work queue data only.
- Do not apply patches to the original repo. Produce patch/artifacts/proposal.
- Do not auto-retry unknown side effects.
- Process tool requests serially in the resident Claude session; backend race can
  be parallel inside a single request.
