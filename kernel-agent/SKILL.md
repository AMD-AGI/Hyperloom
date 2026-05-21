---
name: kernel-agent
description: Resident Claude skill for Hyperloom Kernel Agent. Use when Coordinator or Orchestration requests TraceLens trace analysis, hot-kernel discovery, GEAK/Claude/Codex kernel optimization, CLI log monitoring, and KEEP/PARTIAL/NEEDS_REVIEW/REVERT proposals.
---

# Kernel Agent

You are the resident Claude Kernel Agent. Claw owns the RPC/port layer. This
skill only defines how to handle requests, run tools, persist artifacts, expose
logs, and return structured results.

All tool-call artefacts (TraceLens runs, optimization_attempts.jsonl,
verification JSON, per-tool logs) live under
`$USER_DATA_PATH/kernel-agent/runs/<session_id>/` — the per-session output
namespace. The sibling `$USER_DATA_PATH/kernel-agent-workspace/<kernel_id>/`
tree (created by Coordinator) holds cross-task GEAK/OOB work artefacts
keyed by `kernel_id`. Default `USER_DATA_PATH` is `/workspace/hyperloom`.
Do not write outside `$USER_DATA_PATH` except for reading user-provided
trace/source paths and the read-only TraceLens source at
`$TRACELENS_ROOT` (default `/wekafs/hyperloom/TraceLens-internal`).
The legacy `WORKSPACE_PATH` env was retired during the
all-artefacts-under-`USER_DATA_PATH` migration; rename launchers that
still set it.

## Setup

This skill is **two commands**. The installer is idempotent and brings up
everything in one shot — do NOT pip install ray / claude-agent-sdk / Magpie
manually from chat, do NOT manually export auth aliases, do NOT manually
edit `~/.claude/config.json`. All of that is what `install.sh` and the
auth-proxy supervisor are for.

### Credentials (env > .env, env always wins)

`SAFE_API_KEY` and `OPENAI_BASE_URL` are the only credentials needed.
`install.sh` resolves them in this order:

1. If both are in env → use them, do not touch `.env`.
2. Otherwise, source `$REPO_ROOT/.env` for **missing** keys only; keys
   already in env are protected and never overwritten by `.env`.

Caller either `export REPO_ROOT=<hyperloom_repo_root>` or invokes
`install.sh` from the repo root (so `$(pwd)` becomes the fallback). Do
NOT manually `source .env` from chat — `install.sh` does it with the
correct env-wins semantics.

### Step 1 — Install (one-time per pod / venv rebuild)

```bash
export REPO_ROOT="$(pwd)"   # hyperloom repo root that owns .env (or `cd` there)
bash "$REPO_ROOT/kernel-agent/scripts/install.sh"
# install.sh writes the pod-local env at $HYPERLOOM_RUNTIME_DIR/kernel-agent.env.sh
# (= $USER_DATA_PATH/runtime/kernel-agent.env.sh by default).
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}"
```

`install.sh` always installs everything (no `--with-*` flags any more — those
are accepted as no-ops for backwards compat):

- `ray==2.44.1` + `click<8.3.0`
- Node.js 20 + npm when they are missing (required for the `claude` /
  `codex` npm CLIs)
- TraceLens editable install from `/wekafs/hyperloom/TraceLens-internal` and
  verifies `TraceLens_generate_perf_report_pytorch_inference --help`
  (Hyperloom is inference-only since v0.4; the training-mode CLI is no
  longer accepted)
- GEAK CLI from `GEAK_REF` (default `v3.1.0`) +
  `${HYPERLOOM_ROOT}/geak-config/local.yaml` (model resolution:
  `GEAK_MODEL_NAME` / `GEAK_API_KEY` / `GEAK_BASE_URL` from env, default
  `claude-opus-4-7`)
- GEAK MCP tools — installed as five pip packages from
  `${HYPERLOOM_ROOT}/geak/mcp_tools/`. The bundled `minisweagent` imports
  these at preprocess + run time; missing any of them fails the GEAK
  attempt fast (observed on Qwen3-32B 2026-05-15: `profiler_mcp` not
  installed → 4-minute aborts with zero-byte baselines).
    - `rag-mcp` — knowledge-base retrieval; gated by `tools.rag: true`.
      The first RAG index build writes to
      `~/.cache/amd-ai-devtool/semantic-index/` and may download the
      ~1.3 GB BGE embedding model. The installer builds this index with
      `GEAK_RAG_INDEX_DEVICE=cuda` by default because CPU embedding can
      take hours; set `GEAK_RAG_INDEX_DEVICE=cpu` only for CPU-only
      environments.
    - `profiler-mcp` — Metrix-backed instrumented profiling
      (`preprocessor.py` Step 5/7); produces `profile.json` per attempt.
    - `metrix-mcp` — AMD Metrix backend for `profiler-mcp`.
    - `cross-session-memory-mcp` — SQLite-backed cross-session memory
      retriever; points at `GEAK_MEMORY_STORE_PATH` (default
      `/wekafs/hyperloom/geak-memory/memory.db`).
    - `automated-test-discovery` — pre-fills the eval_command harness so
      GEAK gets a runnable baseline benchmark.
- GEAK cross-session memory env; by default Hyperloom stores GEAK's SQLite
  memory DB at `/wekafs/hyperloom/geak-memory/memory.db`, enables
  `GEAK_SAVE_TO_KNOWLEDGE_BASE=1`, and aligns
  `GEAK_MEMORY_MIN_SPEEDUP=1.20` with the KEEP gate.
- OOB CLI + claude/codex npm CLIs + `@cursor/sdk` global install +
  `~/.claude/config.json` + `~/.codex/auth.json`
- **OOB auth-proxy on `127.0.0.1:4002`**, supervised by
  `scripts/ensure_auth_proxy.sh`. The proxy rewrites `x-api-key` →
  `Authorization: Bearer` for the AMD primus-safe gateway; without it
  every claude/codex CLI request returns HTTP 401 "token not present".
  The cursor backend talks to Cursor's own gateway via `@cursor/sdk` and
  does not go through this proxy; it requires `CURSOR_API_KEY` (separate
  Cursor account, prefix `crsr_...`).

`env.sh` is regenerated by `install.sh` and contains the proxy-rewritten
URLs (`ANTHROPIC_BASE_URL=http://127.0.0.1:4002/...`) plus auth aliases.
Source it instead of trying to derive these by hand.

Use `--check-only` to verify the current environment without installing,
and `--dry-run` to print planned actions:

```bash
bash "$REPO_ROOT/kernel-agent/scripts/install.sh" --check-only
```

### Step 2 — Start Ray with all visible GPUs

GEAK and OOB submit Ray tasks with `num_gpus>=1`. If Ray is started with
`--num-gpus=0`, tasks stay pending forever even when the node has idle GPUs:

```bash
RAY_NUM_GPUS="${RAY_NUM_GPUS:-$(python3 -c 'import torch; print(torch.cuda.device_count() or 1)')}"
ray stop --force || true
ray start --head --disable-usage-stats --num-gpus="$RAY_NUM_GPUS" --include-dashboard=false
ray status
```

`inference_optimizer.cli` does NOT auto-start ray, so this step is required
both standalone and under the inference-optimizer entry point.

### Recovery

The auth-proxy is the most common failure point. If a tool fails with
HTTP 401 / `Primus.00009 token not present` / `Claude SDK exit code 1`,
re-run the supervisor (idempotent, noop if healthy):

```bash
bash "$REPO_ROOT/kernel-agent/scripts/ensure_auth_proxy.sh"
```

It TCP-probes `:4002`, then HTTP-probes via `curl`. If the port is open
but the probe times out (stuck proxy), it kills the existing
`auth_proxy.py` process and relaunches. If `:4002` is healthy, it noops.

If a pod or venv was rebuilt and `ray --version` fails / Ray CLI rejects
`--num-gpus`, repair the Ray/Click pair (Click >= 8.3 is incompatible
with the Ray 2.44 CLI in this environment):

```bash
pip install --quiet 'click<8.3.0' 'ray[default]==2.44.1'
ray --version
```

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
python "$REPO_ROOT/kernel-agent/tools/tracelens_analysis.py" \
  --trace-input "$TRACE_INPUT" \
  --session-id "$SESSION_ID" \
  --model-name "$MODEL_NAME" \
  --framework "$FRAMEWORK" \
  --top-k "${TOP_K:-10}"
```

`--workspace-path` defaults to `${USER_DATA_PATH:-/workspace/hyperloom}` so
TraceLens artifacts land alongside the other Hyperloom session data
(`storage/`, `runs/`, `agents/`, ...) under `$USER_DATA_PATH`. Legacy launchers
that export `$WORKSPACE_PATH` must rename to `$USER_DATA_PATH`; override
explicitly with `--workspace-path` if needed.

The tool must return `hot_kernels`, `trace_report_path`, `cli_log_path`, and
`status_path`.

### `kernel_optimization`

Use this when Coordinator or Orchestration requests optimization for a specific kernel.

Inputs:
- `kernel_id`.
- Optional explicit `backends`: comma separated `geak,claude,codex,cursor`.
- Optional `benchmark_file` or `test_harness_path`.
- Optional E2E/accuracy evidence from Coordinator or Orchestration.
- Optional `enable_rag: false` to pass `--disable-rag` for this request only.
- Optional `enable_xs_memory: false` to pass `--disable-xs-memory` for this
  request only.

Run:

```bash
python "$REPO_ROOT/kernel-agent/tools/kernel_optimization.py" \
  --session-id "$SESSION_ID" \
  --kernel-id "$KERNEL_ID" \
  ${BACKENDS:+--backends "$BACKENDS"} \
  ${BENCHMARK_FILE:+--benchmark-file "$BENCHMARK_FILE"} \
  ${TEST_HARNESS_PATH:+--test-harness-path "$TEST_HARNESS_PATH"}
```

The tool returns optimization attempts, verification, and a proposal in one
response. Do not split proposal generation into a third tool.
The response always includes `rag_hits` and `xs_memory_hits` arrays for
observability. They may be empty when GEAK does not emit structured retrieval
metadata.

If a requested backend is missing after `install.sh` succeeded, this is a
real bug; record the missing backend attempt in `optimization_attempts.jsonl`
and report it instead of crashing the resident session.

#### Auto-generated unittest harness (GEAK pre-step)

Before each `backend=geak` attempt, `invoke_backend` calls
`tools/unittest_agent.py::generate_unittest(candidate, ...)` to materialise
an AgentKernelArena-compatible unittest task right next to the GEAK run dir.
This repo carries the full contract below; do not read or depend on a
developer-local AgentKernelArena checkout when generating or reviewing these
harnesses. The harness reflects the **live vLLM/SGLang runtime** the kernel was
profiled in:

| Field                | Source                                                                                  |
|----------------------|-----------------------------------------------------------------------------------------|
| `source/<kernel>` | Python/Triton: symlink/copy of the live source. HIP/C++: writable mirror copied from the live source; the runner overlays it onto the live path only while tests run. |
| `_baseline_snapshot/`| Frozen copy of the original bytes — the **golden reference** for `correctness`.        |
| `TEST_SHAPES`        | `candidate["input_shapes"]` (TraceLens-resolved per-arg shapes from real traffic).      |
| `TEST_DTYPES`        | `candidate["input_dtypes"]` with `float16` fallback (warned in `unittest_meta.json`).   |
| `RUNTIME_ENV`        | `candidate["env_vars"]` ∪ `os.environ` matching `SGLANG_* / VLLM_* / AITER_* / TRITON_* / HIP_* / ROCR_* / CUDA_*` (KEY/TOKEN/SECRET-redacted). |
| `HOST_ENTRY`         | First non-`@triton.jit` top-level def whose name matches `kernel_name` / `<base>_triton` / `<base>_launcher` / `run_<base>`. |

The harness lands at
`$USER_DATA_PATH/kernel-agent/unittests/<session_id>/<prompt_stem>/` with the
canonical layout:

```text
<out_dir>/
├── config.yaml
├── scripts/task_runner.py
├── source/<kernel>
├── source/_baseline_snapshot/<kernel>
└── unittest_meta.json
```

`config.yaml` must name the target kernel and contain `compile_command`,
`correctness_command`, and `performance_command` entries that call
`python3 scripts/task_runner.py <mode>`. The runner exposes `compile` /
`correctness` / `performance` modes; correctness imports *both* the live source
and the snapshot under distinct module names and asserts tensor equality within
natural fp tolerance (`fp8 -> 5e-2`, `bf16/fp16 -> 1e-2`, `fp32 -> 1e-4`).

After generation we **self-verify** on the unmodified source (compile +
correctness MUST both pass). The manifest's `status` field reports:

* `ok` — both passed; harness becomes GEAK's `--test-command`
  (single-GPU compute kernels only; multi-GPU collectives still take the
  legacy `torchrun` path because the in-process harness can't drive
  `init_process_group`).
* `degraded` — compile passed but correctness was skipped unexpectedly
  (e.g. shapes were not captured) OR self-verify failed; Hyperloom records
  `unittest_status=degraded` but does NOT use the harness in GEAK. It falls
  back to the previous `candidate.benchmark_files` / `test_harness_path` mode.
* `ok` for HIP/C++ — compile precheck passed and at least one existing
  `benchmark_file` was captured. Correctness is intentionally deferred to
  GEAK runtime because HIP tests can take minutes; the generated runner will
  overlay `source/<kernel>` onto the live source, invalidate likely aiter JIT
  `.so` files, run the benchmark, then restore the live tree.
* `skipped` — unsupported source suffix with no runner strategy.
* `failed` — missing source file or unparseable; fall through without a
  harness. Import errors, generation exceptions, and non-`ok` manifests must
  never block GEAK dispatch; they always fall back to the legacy benchmark path.

Control: set `HYPERLOOM_UNITTEST_AGENT=off` or pass
`--unittest-agent off` to bypass the whole pre-step (debugging only — GEAK then
reverts to its legacy benchmark path). `auto` is the default; `force` attempts
best-effort generation whenever a source file and benchmark surface exist. The
older `HYPERLOOM_DISABLE_UNITTEST_AGENT=1` remains a compatibility alias for
`off`.

Result surfaces (visible in `optimization_attempts.jsonl[].backend_paths`):

* `unittest_status`        — `ok` / `degraded` / `skipped` / `failed`.
* `unittest_out_dir`       — the harness workspace (inspect on debug).
* `unittest_test_command`  — the exact `correctness` command GEAK ran.

#### Outer-timeout contract (do NOT skip)

When a unittest harness is in play, GEAK's `--test-command` stops being a
fast `python bench_<kernel>.py` (seconds) and becomes
`python3 scripts/task_runner.py correctness`, which transparently
triggers an aiter JIT recompile (~51s) + a multi-shape benchmark and
routinely takes minutes. mini-swe-agent's `LocalEnvironmentConfig.timeout`
defaults to **30 seconds**, so a custom `local.yaml` that omits the `env`
block — or copies an old `timeout: 30` — silently SIGKILLs every patch
test with `"Test command timed out"`. select_patch then falls back to
the unmodified baseline and the whole GEAK attempt looks like a no-op.

`unittest_agent` owns BOTH halves of the test-command contract:

| Knob                            | Where it lives                       | Wired by                                   |
|---------------------------------|--------------------------------------|--------------------------------------------|
| `--test-command`                | `unittest_test_command`              | `_append_unittest_context_to_prompt`       |
| Outer `env.timeout` (GEAK side) | `harness_timeout_*_sec` in manifest  | `_geak_config_for_run(unittest_manifest=)` |

`tools/kernel_optimization.py::_harness_outer_timeout(manifest)` returns
`max(harness_timeout_correctness_sec, harness_timeout_performance_sec) +
buffer` (buffer = 300s) whenever `status == "ok"`; the caller writes the
result into the per-run GEAK config so mini-swe-agent sees a matching
`env.timeout`. The fallback path (no manifest) still injects 3600s so a
config without an `env` block never silently inherits the 30s default.

Regression checklist — if `Test command timed out` reappears:
1. Read `patch_*_test.txt` under `<session>/geak/run-*/` — confirm it's a
   process-killed timeout, not a Python `pytest` failure.
2. `grep -A4 '^env:' $GEAK_CONFIG` — confirm `env.timeout` >= the
   manifest's `harness_timeout_correctness_sec`.
3. Inspect `<harness>/unittest_meta.json` — confirm `status: ok` and that
   `harness_timeout_*_sec` is in the manifest.
4. Re-run `kernel_optimization.py` so `_apply_geak_env_overrides` writes
   a fresh per-prompt `*.geak-config.yaml`; never edit `local.yaml` by
   hand from a tight loop — Hyperloom rewrites it per attempt.

## TraceLens Requirements

TraceLens runs through its CLI and its own skill.

`install.sh` already installs TraceLens (editable install) and verifies the
perf-report CLI is on PATH. If `tracelens_analysis` fails with "CLI not
found", re-run `install.sh` instead of cloning the open-source TraceLens
repo from chat — the open-source clone does NOT contain the standalone
skills required by `tools/tracelens_analysis.py`.

When `$TRACELENS_ROOT` is on a read-only mount (the WekaFS default at
`/wekafs/hyperloom/TraceLens-internal`), `ensure_tracelens` automatically
mirrors the source tree to `${HYPERLOOM_ROOT}/TraceLens-internal` (parallel
to `${HYPERLOOM_ROOT}/geak` / `${HYPERLOOM_ROOT}/OOB/oob_cli`) via `cp -r`,
runs `pip install -e` against the writable mirror, and `write_env_file`
re-exports `TRACELENS_ROOT` pointing at the mirror so subsequent CLI
subprocesses inherit it. This prevents the `select_kernels` failure loop
caused by `tools/tracelens_analysis.py` re-running `pip install -e .`
inside `cwd=$TRACELENS_ROOT` on every request. No manual `rsync` is
needed for the single-node read-only case.

If `install.sh` did not finish or the CLI is unexpectedly missing, run a
manual editable install + smoke test before analysis:

```bash
export TRACELENS_ROOT="${TRACELENS_ROOT:-/wekafs/hyperloom/TraceLens-internal}"
cd "$TRACELENS_ROOT"
pip install -e .
# TraceLens #124: only the `_inference` perf-report CLI is accepted
# (correct execution mode = graph-replay for vLLM/SGLang traces). Hyperloom
# is inference-only since v0.4; the legacy training-mode CLI is no
# longer accepted — bump TraceLens-internal if the inference CLI is
# missing.
TraceLens_generate_perf_report_pytorch_inference --help
```

If neither CLI is on PATH, stop and fix installation before analysis. Do not
fall back to the open-source TraceLens clone when the internal mount exists;
the internal mount contains the standalone skills expected by this tool.
`tools/tracelens_analysis.py` picks the right CLI at runtime (preferring
`_inference`, falling back to the legacy name) — see `select_perf_report_cli`.

When running TraceLens analysis, read this skill file and strictly follow
its order:

`/wekafs/hyperloom/TraceLens-internal/TraceLens/Agent/Analysis/.cursor/skills/analysis-orchestrator.md`

Step 6 and Step 7 categories must run in independent Task subagents. Each
subagent must write findings under `system_findings/` or `category_findings/`.

Do not fabricate results. All findings must come from Python script output,
TraceLens CLI output, or artifacts written by subagents.

The final report is the TraceLens v0.3 SDK orchestrator's `analysis.md`,
written by the upstream skill to:

`$USER_DATA_PATH/kernel-agent/runs/<session_id>/tracelens/analysis.md`

Hyperloom does not alias, copy, or wrap this file (#203 removed the
legacy `standalone_analysis.md` / `tracelens_report.md` copies and the
`--compat-report-path` argument; the `${USER_DATA_PATH:-/workspace/hyperloom}/`
compatibility output is gone with them). Downstream consumers read the
canonical upstream path returned in `analysis_report_path` from
`select_kernels_handler`.

## Backend Selection

User-specified backends win, subject to feasibility checks. If user does not
specify backends:

- **Default ladder (all rewritable kernels)**: `geak,claude,codex,cursor` —
  GEAK first because every kernel Claude/Codex/Cursor can rewrite, GEAK can
  rewrite too. Claude/Codex stay on as fallbacks if GEAK times out or
  rejects; `cursor` is appended when `$CURSOR_API_KEY` is provisioned (auto-
  dropped otherwise to avoid wasted 401 attempts). The kernel type
  (Triton / HIP-C++ / Python / unknown) does NOT change the ladder; the
  capability differences are GEAK-side, not Hyperloom-side, so we let GEAK
  decide what to handle.
- **No-benchmark case**: still attempt GEAK but flag
  `geak_without_benchmark: true` so the KEEP gate downstream knows
  verification confidence is reduced (matches the existing user-specified
  behaviour — the auto-pick now follows the same contract).
- **Vendor binary / hipBLASLt**: do not rewrite; return reason and
  `NEEDS_REVIEW`. Only case that yields an empty backend list upstream.

Multi-GPU collective kernels (`is_multigpu: True`, e.g. all-reduce / all-gather):
`parallel_e2e_runner` automatically drops `geak` from the backend list because
GEAK's sub-agent framework spawns nested `ray.remote(num_gpus=1)` for patch
validation, which makes any `torchrun --nproc>=2` test inside it fail with
`HIP error: invalid device ordinal`. Use `claude` / `codex` / `cursor` only
for these kernels — they work via standalone HIP `std::thread` simulation
or real `torchrun --nproc=N`. The dropped backend is recorded under
`gpu_plan.backends_dropped` in the run summary.

If the user explicitly requests GEAK without a benchmark/test harness, allow the
attempt but mark `geak_without_benchmark: true`.

**Cursor auto-skip when `CURSOR_API_KEY` is unset.** Cursor backend talks to
Cursor's own gateway and requires `CURSOR_API_KEY` (separate Cursor account,
prefix `crsr_...`; never inherited from `SAFE_API_KEY`). When the env var is
empty, `kernel_optimization.choose_backends`,
`tracelens_analysis.recommend_backends`,
`kernel_request_handlers._backend_order`, and `parallel_e2e_runner`'s
`--backends` default all drop `cursor` from the auto-derived list so
Hyperloom doesn't waste attempt slots on guaranteed 401s. The selection
`notes` carry `cursor_key_present: bool` for observability. Explicit user
input (`--backends cursor`, payload `backends="cursor"` /
`backend_order="...,cursor,..."`, or `KERNEL_OPT_BACKEND_ORDER`) is always
honored; a missing key surfaces as a single failed cursor attempt rather
than a silent skip.

## Optimization Goals & Time Budget

- **Target speedup**: `>= 1.20x` on the dominant inference shape(s). Below this
  threshold an attempt is `NEEDS_REVIEW` (marginal / shape-specific / risky),
  not `KEEP`. (Prompt still tells agents to aim for `>= 1.50x` to incentivise
  ambitious optimization, but the KEEP gate is 1.20x because real inference
  wins are often shape-specific 1.18-1.32x — see r19 GEMM 1.32x, r39 GEAK
  rms_norm 1.18x.)
- **Default budget**:
  - claude / codex / cursor: **60 minutes** per attempt (`--backend-budget-min 60`)
  - GEAK: **120 minutes** per attempt (`--geak-budget-min 120`)
- **GEAK task parameters** (prompt-injected, align with GEAK team defaults):
  - `max_rounds`: **5** (multi-round heterogeneous optimization)
  - `step_limit`: **200** (GEAK recommended; 100 limits multi-round runs)
  
  Agents are instructed to **early-exit** as soon as they hit `>=1.50x` with
  passing correctness; otherwise they iterate up to ~85% of the budget and the
  runner SIGTERMs at 100%. `parallel_e2e_runner` will still extract whatever
  `optimization_report.md` / `optimized_versions/*` were on disk at SIGTERM
  time and promote the attempt to `partial` (see Proposal Rules).
- **Why GEAK budget is 120 min, not 60**: GEAK runs N sub-agent tasks serially
  (each 5-10 min: baseline measurement + LLM patch generation + per-patch
  benchmark) + a final `select_patch` round that LLM-judges all patches and
  writes `final_report.json`. With a 60 min budget, the select_patch round
  consistently gets SIGTERM'd before it finishes (observed r38/r39: driver
  had to fall back to per-task `best_results.json` salvage). 120 min gives the
  optimizer room for multi-round patch attempts plus the select_patch round,
  producing `final_report.json` with the canonical best_speedup. Do not drop
  GEAK budget below 60 min or it will return
  `partial` with an empty patch.

## Artifacts

Every path below is relative to `$USER_DATA_PATH/kernel-agent/` (the
per-session tool-output namespace). Each request creates a `run_id` and
writes:

- `runs/<session_id>/session_state.json`
- `runs/<session_id>/trace_input_manifest.json`
- `runs/<session_id>/kernel_candidates.json`
- `runs/<session_id>/tracelens/analysis.md` (TraceLens v0.3 final report; owned by upstream, not copied by Hyperloom)
- `runs/<session_id>/tracelens/tracelens_report.json`
- `runs/<session_id>/tracelens/system_findings/`
- `runs/<session_id>/tracelens/category_findings/`
- `runs/<session_id>/optimization_attempts.jsonl`
- `runs/<session_id>/prompts/<attempt_id>.md` (one per attempt; the GEAK/OOB prompt that was actually sent)
- `runs/<session_id>/optimized/<attempt_id>_stdout.log` (real backend runs; see "Per-attempt stdout file naming" below)
- `runs/<session_id>/optimized/<attempt_id>_optimized<source_suffix>` (dry-run only; synthetic placeholder for smoke tests)
- `runs/<session_id>/verification/<kernel_id>.json`
- `runs/<session_id>/results/<kernel_id>.json`
- `runs/<session_id>/logs/<tool>/<run_id>.log`
- `runs/<session_id>/status/<tool>/<run_id>.json`

### Per-attempt stdout file naming

`run_attempt` (in `kernel-agent/tools/kernel_optimization.py`) materialises
one file per attempt under `runs/<session_id>/optimized/`:

| Mode             | Filename                                                  | Contents                                                                 |
|------------------|-----------------------------------------------------------|--------------------------------------------------------------------------|
| Real backend run | `<attempt_id>_stdout.log`                                 | The raw subprocess stdout (mini-swe-agent / OOB / GEAK conversation log) |
| `--dry-run`      | `<attempt_id>_optimized<source_suffix>` (e.g. `.cu`/`.py`) | A small synthetic placeholder kernel (smoke-test only)                   |

**Backward compatibility note**: prior to 2026-05 the real-backend file was
also named `<attempt_id>_optimized<source_suffix>` and contained subprocess
stdout. That caused `_source_text_looks_complete` to false-positive match
generic English (e.g. transcripts containing `"void "` / `"extern "`) and
promote the conversation log to `artifact_source = source_file` — observed
on Qwen3-8B k007 rmsnorm_quant and k013 silu_and_mul. The `.log` rename
routes the stdout through `_extract_source_block` (which only returns an
artifact when a real fenced code block is present) and never lets a
transcript impersonate a `.cu` / `.py` source.

Downstream consumers (`breakdown/collectors.py`, dashboards, etc.) should
either:
1. read `attempt["optimized_path"]` from `optimization_attempts.jsonl`
   (already the canonical pointer — it tracks whichever name was used), or
2. glob `runs/<session_id>/optimized/<attempt_id>*` to pick up both
   historical `_optimized.<suffix>` files and the new `_stdout.log`
   (`breakdown/collectors.py` does this — older session dirs keep working).

Never assume a fixed `_optimized.cu` / `_optimized.py` suffix on
real-backend runs after 2026-05.

Cross-task GEAK/OOB work artefacts keyed by `kernel_id` live in the
sibling tree `$USER_DATA_PATH/kernel-agent-workspace/<kernel_id>/`
(populated by Coordinator and reused across multiple `kernel_optimization.py`
invocations on the same kernel).

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
