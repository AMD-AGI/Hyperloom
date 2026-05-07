---
name: kernel-agent
description: Resident Claude skill for Hyperloom Kernel Agent. Use when Coordinator or Orchestration requests TraceLens trace analysis, hot-kernel discovery, GEAK/Claude/Codex kernel optimization, CLI log monitoring, and KEEP/PARTIAL/NEEDS_REVIEW/REVERT proposals.
---

# Kernel Agent

You are the resident Claude Kernel Agent. Claw owns the RPC/port layer. This
skill only defines how to handle requests, run tools, persist artifacts, expose
logs, and return structured results.

All files live under `$WORKSPACE_PATH/kernel-agent`; default `WORKSPACE_PATH` is
`/workspace`. Do not write outside that tree except reading user-provided trace
or source paths and the TraceLens install at
`/wekafs/hyperloom/TraceLens-internal`.

## LLM Environment

The canonical LLM endpoint is LiteLLM-compatible and is configured by:

- `OPENAI_BASE_URL`
- `SAFE_API_KEY`

If either variable is missing, load `.env` from the repository that launched the
skill. Do not print the key. Before running installers or tools, export
compatibility aliases used by OOB, GEAK, Claude/Codex CLIs, and older scripts:

```bash
if [ -n "${REPO_ROOT:-}" ] && [ -f "$REPO_ROOT/.env" ]; then
  set -a
  . "$REPO_ROOT/.env"
  set +a
elif [ -f .env ]; then
  set -a
  . ./.env
  set +a
fi

: "${OPENAI_BASE_URL:?OPENAI_BASE_URL must be set in env or .env}"
: "${SAFE_API_KEY:?SAFE_API_KEY must be set in env or .env}"

export OPENAI_API_KEY="${OPENAI_API_KEY:-$SAFE_API_KEY}"
export ANTHROPIC_AUTH_TOKEN="${ANTHROPIC_AUTH_TOKEN:-$SAFE_API_KEY}"
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-$SAFE_API_KEY}"
export OOB_API_KEY="${OOB_API_KEY:-$SAFE_API_KEY}"
export GEAK_API_KEY="${GEAK_API_KEY:-$SAFE_API_KEY}"
export LLM_API_KEY="${LLM_API_KEY:-$SAFE_API_KEY}"
export AMD_LLM_API_KEY="${AMD_LLM_API_KEY:-$SAFE_API_KEY}"

export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-$OPENAI_BASE_URL}"
export OOB_BASE_URL="${OOB_BASE_URL:-$OPENAI_BASE_URL}"
export GEAK_BASE_URL="${GEAK_BASE_URL:-$OPENAI_BASE_URL}"
export LLM_API_BASE="${LLM_API_BASE:-$OPENAI_BASE_URL}"

python3 - <<'PY'
import os
missing = [k for k in ("OPENAI_BASE_URL", "SAFE_API_KEY") if not os.environ.get(k)]
print("llm_env_present=", not missing)
if missing:
    print("missing=", ",".join(missing))
PY
```

## Installation

Before the first request, run the installer:

```bash
bash $WORKSPACE_PATH/kernel-agent/scripts/install.sh
```

The installer always installs everything in one shot:
- `ray==2.44.1` + `click<8.3.0`
- TraceLens editable install from `/wekafs/hyperloom/TraceLens-internal`
- GEAK CLI + `geak-config/local.yaml` (model resolution: `GEAK_MODEL_NAME` /
  `GEAK_API_KEY` / `GEAK_BASE_URL` from env)
- OOB CLI + claude/codex npm CLIs + `/root/.claude/config.json` /
  `/root/.codex/auth.json`
- OOB auth-proxy on `127.0.0.1:4002` (header rewrite for the AMD LLM
  gateway), supervised by `scripts/ensure_auth_proxy.sh`

The previous lazy `--with-geak / --with-oob / --with-llm` selectivity was
removed because it caused recurring "OOB proxy not running, request 401'd,
discovered the missing service after the fact" issues. Those flags are still
accepted as no-ops for backwards compatibility; new call sites should drop
them.

Use `--check-only` to verify the current environment and `--dry-run` to print
planned actions without installing.

If a pod or venv was rebuilt, repair the Ray/Click pair before any kernel
optimization request. `click>=8.3` is incompatible with the Ray 2.44 CLI in this
environment and can make backend submission fail before any kernel code compiles:

```bash
pip install --quiet 'click<8.3.0' 'ray[default]==2.44.1'
ray --version
```

Start Ray with every visible GPU before OOB/GEAK kernel optimization. OOB and
GEAK submit Ray tasks with `num_gpus>=1`; if Ray is started with
`--num-gpus=0`, tasks stay pending even when the node has idle GPUs.

```bash
RAY_NUM_GPUS="${RAY_NUM_GPUS:-$(python3 - <<'PY'
try:
    import torch
    print(torch.cuda.device_count() or 1)
except Exception:
    print(1)
PY
)}"
ray stop --force || true
ray start --head --disable-usage-stats --num-gpus="$RAY_NUM_GPUS" --include-dashboard=false
ray status
```

### Auth-proxy supervision

The OOB auth-proxy on `:4002` rewrites `x-api-key` -> `Authorization: Bearer`
for the AMD LLM gateway; without it, every claude/codex CLI request returns
HTTP 401 "token not present". It is started by `install.sh` and can be re-
verified at any time via:

```bash
bash $WORKSPACE_PATH/kernel-agent/scripts/ensure_auth_proxy.sh
```

The script is idempotent: it TCP-probes :4002, then HTTP-probes via curl. If
the port is open but the probe times out (a stuck proxy), it kills the
existing `auth_proxy.py` process and relaunches. If port :4002 is healthy,
it noops. Run it before any kernel-agent tool that talks to claude/codex,
or simply re-run `install.sh` (which calls it as part of normal install).

## Tools

### `tracelens_analysis`

Use this when Coordinator or Orchestration requests hot kernels from a trace.

Inputs:
- `trace_input`: trace file, filtered trace, or TraceLens capture directory.
- `session_id`: stable session id from Coordinator; generate one only if absent.
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

Use this when Coordinator or Orchestration requests optimization for a specific kernel.

Inputs:
- `kernel_id`.
- Optional explicit `backends`: comma separated `geak,claude,codex`.
- Optional `benchmark_file` or `test_harness_path`.
- Optional E2E/accuracy evidence from Coordinator or Orchestration.

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
export TRACELENS_ROOT="${TRACELENS_ROOT:-/wekafs/hyperloom/TraceLens-internal}"
cd "$TRACELENS_ROOT"
pip install -e .
# TraceLens #124: prefer the `_inference` perf-report CLI for vLLM/SGLang
# traces (correct execution mode = graph-replay). The legacy
# `TraceLens_generate_perf_report_pytorch` is acceptable on older builds.
TraceLens_generate_perf_report_pytorch_inference --help \
  || TraceLens_generate_perf_report_pytorch --help
```

If neither CLI is on PATH, stop and fix installation before analysis. Do not
fall back to the open-source TraceLens clone when the internal mount exists;
the internal mount contains the standalone skills expected by this tool.

`tools/tracelens_analysis.py` picks the right CLI at runtime (preferring
`_inference`, falling back to the legacy name) — see `select_perf_report_cli`.

2. Read this skill file and strictly follow its order:

`/wekafs/hyperloom/TraceLens-internal/TraceLens/AgenticMode/Standalone/.cursor/skills/standalone-analysis-orchestrator.md`

3. Step 6 and Step 7 categories must run in independent Task subagents. Each
subagent must write findings under `system_findings/` or `category_findings/`.

4. Do not fabricate results. All findings must come from Python script output,
TraceLens CLI output, or artifacts written by subagents.

5. The final report is written to:

`$WORKSPACE_PATH/kernel-agent/runs/<session_id>/tracelens/standalone_analysis.md`

If Executor requests compatibility output, also write
`/workspace/hyperloom/standalone_analysis.md`.

### Debug Mode (#78)

Set `DEBUG_TRACELENS=true` (or pass `--debug-tracelens` to
`tracelens_analysis.py`) to capture the orchestrator agent's complete event
stream as JSONL for offline replay.

When enabled, `tools/tracelens_analysis.py` reserves the artifact path
`$WORKSPACE_PATH/kernel-agent/runs/<session_id>/tracelens/tracelens_agent_stream.jsonl`
and exposes it under `artifact_paths.tracelens_agent_stream`. The streamJSON
file itself is produced by the upstream `cursor-agent run --output-format
stream-json --output-file <path>` invocation — `build_orchestrator_invocation`
in the same module returns the corresponding argv for callers.

```bash
DEBUG_TRACELENS=true python $WORKSPACE_PATH/kernel-agent/tools/tracelens_analysis.py \
  --trace-input "$TRACE_INPUT" --session-id "$SESSION_ID" ...
```

Each line of the resulting `tracelens_agent_stream.jsonl` is a single
event (`type` ∈ {`system`, `assistant`, `user`, `result`}); see
`https://cursor.com/docs/cli/reference/output-format` for schema.

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
