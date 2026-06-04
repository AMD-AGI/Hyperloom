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
trace/source paths, the TraceLens public source at
`$TRACELENS_ROOT` (default `/wekafs/hyperloom/TraceLens-internal`), and — when enabled —
the optional TraceLens-internal source at `$TRACELENS_INTERNAL_ROOT`
(no default; set it to opt in, otherwise open-source-only).
The legacy `WORKSPACE_PATH` env was retired during the
all-artefacts-under-`USER_DATA_PATH` migration; rename launchers that
still set it.

## Setup

This skill is **two commands**. The installer is idempotent and brings up
everything in one shot — do NOT pip install ray / claude-agent-sdk / Magpie
manually from chat, do NOT manually export auth aliases, do NOT manually
edit `~/.claude/config.json`. All of that is what `install.sh` is for.

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
- TraceLens public editable install from `$TRACELENS_ROOT`
  (default `/wekafs/hyperloom/TraceLens-internal`), plus the optional internal extension
  from `$TRACELENS_INTERNAL_ROOT` only when that var is set (no default;
  unset => open-source-only), and verifies
  `TraceLens_generate_perf_report_pytorch_inference --help`
  (Hyperloom is inference-only since v0.4; the training-mode CLI is no
  longer accepted)
- GEAK CLI from `GEAK_REF` (default `v3.2.0`) +
  `${HYPERLOOM_ROOT}/geak-config/local.yaml` (model resolution:
  `GEAK_MODEL_NAME` / `GEAK_API_KEY` / `GEAK_BASE_URL` from env, default
  `claude-opus-4-7`). Run-mode default for the generated yaml is
  controlled by `GEAK_RUN_MODE` (`quick` or `full`; defaults to `full`,
  which selects the 2 h / 5-round `run.budgets.full` preset). Set
  `GEAK_RUN_MODE=quick` before `install.sh` for the 1 h / 2-round smoke
  preset. Other yaml budget knobs are not env-overridable on purpose —
  edit `$GEAK_CONFIG` directly if you need to tune them per pod.
- GEAK MCP tools — installed as four pip packages from
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
    - `profiler-mcp` — unified profiling MCP (Metrix + rocprof-compute);
      produces `profile.json` per attempt. Metrix is no longer a separate
      `metrix-mcp` folder in v3.2.0 — it is pulled in transitively via
      `profiler-mcp/pyproject.toml` (`dependencies = ["metrix>=0.1.0"]`).
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
  `~/.claude/config.json` (`customApiUrl` pointed at the upstream
  Anthropic URL derived from `$OPENAI_BASE_URL` with a trailing `/v1`
  stripped) + `~/.codex/auth.json`. The AMD primus-safe gateway accepts
  both `x-api-key` (what claude/codex CLIs send) and
  `Authorization: Bearer` natively, so the legacy auth-proxy on
  `127.0.0.1:4002` is no longer in the loop. The cursor backend talks to
  Cursor's own gateway via `@cursor/sdk` and requires `CURSOR_API_KEY`
  (separate Cursor account, prefix `crsr_...`).

`env.sh` is regenerated by `install.sh` and contains the upstream gateway
URLs (`ANTHROPIC_BASE_URL` / `OPENAI_BASE_URL` derived from
`$OPENAI_BASE_URL`) plus auth aliases. Source it instead of trying to
derive these by hand.

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

If a tool fails with HTTP 401 / `Primus.00009 token not present` /
`Claude SDK exit code 1`, the gateway rejected the request. Re-source
`kernel-agent.env.sh` and curl-probe the gateway directly:

```bash
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}"
curl -sS -H "Authorization: Bearer $SAFE_API_KEY" "$OPENAI_BASE_URL/models" | head
```

A stale `customApiUrl=http://127.0.0.1:4002/...` in
`~/.claude/config.json` left over from a previous install can also
trigger this — re-run `install.sh` to have `ensure_llm_auth_files`
rewrite the file to the upstream URL.

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
  ${TEST_HARNESS_PATH:+--test-harness-path "$TEST_HARNESS_PATH"} \
  ${TEST_COMMAND:+--test-command "$TEST_COMMAND"}
```

The tool returns optimization attempts, verification, and a proposal in one
response. Do not split proposal generation into a third tool.
The response always includes `rag_hits` and `xs_memory_hits` arrays for
observability. They may be empty when GEAK does not emit structured retrieval
metadata.

If a requested backend is missing after `install.sh` succeeded, this is a
real bug; record the missing backend attempt in `optimization_attempts.jsonl`
and report it instead of crashing the resident session.

#### Pre-GEAK Unittest Harness (unittest skill)

Before `backend=geak` attempts, the main agent generates a GEAK-compatible
test harness by following `kernel-agent/skills/unittest/SKILL.md`. The skill
searches for existing tests, collects shapes/dtypes from TraceLens, and
generates a 4-mode harness (`--correctness`/`--profile`/`--benchmark`/`--full-benchmark`).

The resulting `test_command` is passed via `--test-command` to
`kernel_optimization.py`, which forwards it to GEAK's `--test-command`.

If the skill fails to produce a valid harness, omit `--test-command` and
GEAK falls back to its own test discovery.

Validation uses `kernel-agent/skills/unittest/validate_harness.py` for
static checks (argparse + 4 flags + output markers) and runtime verification
(run correctness + benchmark modes, check exit codes and markers).

#### Merging same-kernel candidates

TraceLens often emits **multiple `kernel_id`s for the same kernel function
called at different shapes** — e.g. `aiter::rmsnorm` shows up as `k006`,
`k007`, `k008` because the model invokes it at `(256,128)`, `(1024,128)`,
and `(64,4096)`. These all resolve to the same `source_file` and the same
underlying op.

**Default (do-not-merge) behavior**: dispatch each `kernel_id` as a
separate `run_optimization` request. The orchestrator picks them off
`reusable_native_kernel_ids` one by one, GEAK runs its full
preprocessing pipeline (`~4 min`) per task, and each task generates
patches for one shape.

**Merge optimization** (preferred when the candidates share
`(name, source_file)`): batch them into a single `run_optimization`
request whose harness covers all shapes. Concrete benefits:
- **Preprocessing amortization**: GEAK preprocessing (discovery →
  resolution → testcase selection → baseline/profile collection) takes
  ~4 min regardless of shape count. Merging 3 candidates skips 8 min of
  duplicated work.
- **Better patch quality**: the sub-agent sees ALL shapes when reasoning
  about the optimization, so it can pick a strategy that wins on
  small-and-large shapes simultaneously instead of overfitting to one.
- **Cross-shape correctness gate**: a patch that breaks any shape fails
  `--correctness` immediately rather than being merged later and
  discovered to regress.

Mechanically:
1. Group `reusable_native_kernel_ids` by `(candidate.name,
   candidate.source_file)`.
2. For each group with size > 1, build a synthetic merged candidate:
   - `kernel_id`: `<name>_merged` (or any unique id)
   - `name`, `source_file`, `kernel_repo`, `benchmark_files`: copied
     from any member (they're equal within the group)
   - `input_shapes`: concatenation of all members' `input_shapes`
     (the unittest skill will dedupe by ndim + dtype when building
     `ALL_CONFIGS`)
   - `call_count`: sum of members' `call_count`
   - `gpu_pct`: sum of members' `gpu_pct`
   - `kernel_params` / `env_vars`: copied from the member with the
     highest `call_count`
3. Write the merged candidate(s) to a new `candidates_path` (do not
   mutate the original `kernel_candidates.json`) and dispatch
   `kernel_optimization.py --candidates-path <merged>.json
   --kernel-id <merged_id>`.

When **not** to merge:
- Candidates share a `source_file` but resolve to different functions
  inside it (different `kernel_params.kernel_name`).
- Shapes span ndim or dtype boundaries that would force the harness
  to special-case at runtime (e.g. mixing 1-D `(128,)` weight tensors
  with 2-D activations is OK because they're separate args; mixing
  `bf16` and `fp8` activations is not — generate two harnesses).
- One member is the bottleneck (`gpu_pct >> others`) and the others
  are negligible — the LLM may be distracted by tiny shapes.

## TraceLens Requirements

TraceLens runs through its CLI and its own skill. The **public** repo is the
required base; the **internal** repo is an **optional** rehydration-module
extension. Install the public repo inside the GPU container before Hyperloom
bootstrap (see README Local Mode step 1); install the internal repo only if you
need the internal extension:

```bash
# Required: public repo
git clone https://github.com/AMD-AGI/TraceLens.git
cd TraceLens && pip install -e .

TraceLens_generate_perf_report_pytorch_inference --help
```

OPTIONAL internal extension (internal users only): if you have access to it,
install your own checkout (`pip install -e .`) and set `$TRACELENS_INTERNAL_ROOT`
to its path. Hyperloom keeps no internal URL/path and never clones it.

Default base repo: `/wekafs/hyperloom/TraceLens-internal` (shared cluster checkout)
(`TRACELENS_ROOT`). The internal extension has **no default path**: it is used
ONLY when `$TRACELENS_INTERNAL_ROOT` is set to an existing checkout you provide.
Presence of that env var is the sole switch —
there is no separate toggle. When set, `TL_EXTENSION` is exported and added to
the orchestrator prompt so the analysis skill can locate the rehydration module;
when unset, Hyperloom stays on the open-source-only report.

`install.sh` re-runs the public editable install (and the internal one when
`$TRACELENS_INTERNAL_ROOT` is set) and verifies the perf-report CLI is on PATH.
If `tracelens_analysis` fails with "CLI not found", re-run `install.sh` or
repeat the manual install above.

When `$TRACELENS_ROOT` or `$TRACELENS_INTERNAL_ROOT` is on a read-only mount,
`ensure_tracelens` automatically mirrors the checkout to
`${HYPERLOOM_ROOT}/TraceLens` or `${HYPERLOOM_ROOT}/TraceLens-internal`
respectively (parallel to `${HYPERLOOM_ROOT}/geak` /
`${HYPERLOOM_ROOT}/OOB/oob_cli`) via `cp -r`, runs `pip install -e` against
the writable mirror, and `write_env_file` re-exports the resolved root so
subsequent CLI subprocesses inherit the mirror.

If `install.sh` did not finish or the CLI is unexpectedly missing, run a
manual editable install + smoke test before analysis:

```bash
export TRACELENS_ROOT="${TRACELENS_ROOT:-/wekafs/hyperloom/TraceLens-internal}"
cd "$TRACELENS_ROOT" && pip install -e .
# OPTIONAL internal extension — only if TRACELENS_INTERNAL_ROOT is set:
[ -n "${TRACELENS_INTERNAL_ROOT:-}" ] && cd "$TRACELENS_INTERNAL_ROOT" && pip install -e .
```

If the CLI is not on PATH, stop and fix installation before analysis.
`tools/tracelens_analysis.py` picks the right CLI at runtime (preferring
`_inference`, falling back to the legacy name) — see `select_perf_report_cli`.

When running TraceLens analysis, read this skill file and strictly follow
its order:

`$TRACELENS_ROOT/TraceLens/Agent/Analysis/.cursor/skills/analysis-orchestrator.md`

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
`trace_analyze_handler`.

## Backend Selection

User-specified backends win, subject to feasibility checks. If user does not
specify backends:

- **Default ladder (all rewritable kernels)**: `geak,claude,codex,cursor` —
  GEAK first because every kernel Claude/Codex/Cursor can rewrite, GEAK can
  rewrite too. Claude/Codex stay on as fallbacks if GEAK times out or
  rejects; `cursor` is appended when `$CURSOR_API_KEY` is provisioned (auto-
  dropped otherwise to avoid wasted 401 attempts). The kernel type
  (Triton / HIP-C++ / FlyDSL / Python / unknown) does NOT change the ladder;
  the capability differences are GEAK-side, not Hyperloom-side, so we let
  GEAK decide what to handle.
- **No-benchmark case**: still attempt GEAK but flag
  `geak_without_benchmark: true` so the KEEP gate downstream knows
  verification confidence is reduced (matches the existing user-specified
  behaviour — the auto-pick now follows the same contract).
- **Vendor binary / hipBLASLt**: do not rewrite; return reason and
  `NEEDS_REVIEW`. Only case that yields an empty backend list upstream.

FlyDSL kernels (`source_type=flydsl`, detected by content-sniffing
`@flyc.kernel` / `flydsl.compiler` / `flydsl.expr` markers) are sent to
GEAK with `kernel_type=flydsl`. GEAK's `task_parser.py` routes that to
its `skills/flydsl/SKILL.md` (write / optimize / debug workflows for
`@flyc.kernel` tile programs); Hyperloom does not maintain its own copy
of the FlyDSL guidance.

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
  - GEAK: tracks `$GEAK_RUN_MODE` (set by `install.sh`, exported via
    `kernel-agent.env.sh`). `full` (default) → **130 min**, `quick` → **70 min**.
    Both `kernel_optimization.py` and `parallel_e2e_runner.py` read
    `$GEAK_RUN_MODE` to pick the `--geak-budget-min` default; override by
    passing the flag explicitly.
- **GEAK task parameters** (prompt-injected, align with GEAK team defaults):
  - `max_rounds`: **5** for full / **2** for quick (driven by yaml
    `run.presets.<mode>.orchestrator.max_rounds`).
  - `step_limit`: **200** (GEAK recommended; 100 limits multi-round runs).

  Agents are instructed to **early-exit** as soon as they hit `>=1.50x` with
  passing correctness; otherwise they iterate up to ~85% of the budget and the
  runner SIGTERMs at 100%. `parallel_e2e_runner` will still extract whatever
  `optimization_report.md` / `optimized_versions/*` were on disk at SIGTERM
  time and promote the attempt to `partial` (see Proposal Rules).
- **Why GEAK budget tracks $GEAK_RUN_MODE**: GEAK v3.2.0 yaml ships two
  presets — `run.budgets.quick.total_s=3600` (1 h, 2 rounds) and
  `run.budgets.full.total_s=7200` (2 h, 5 rounds). GEAK's mini.py:435
  resolves mode by LLM-parsing the prompt-quoted budget: <120 min → quick,
  >=120 min → full. 130 min ≥ full.total_s (7200s) + finalize_grace_s (300s)
  + kill_buffer_s (60s) + safety, and 70 min ≥ quick.total_s (3600s) + the
  same finalize_grace + kill_buffer + safety. Defaults sit one tier above
  their respective yaml total_s so the matching mode's last round +
  select_patch can complete (vs the old uniform 90 min default which fell
  between GEAK quick (60) and full (120) and silently downgraded every run
  to mode=quick). 60 min is still the floor — do not drop GEAK budget below
  it; r38/r39 SIGTERM'd the select_patch round and had to fall back to
  per-task `best_results.json` salvage.

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

## Multi-node mode

When the workload runs with `--nodes >= 2`
(`/tmp/multi_node_state.json` has `nodes >= 2`), the sandbox is
CPU-only and the inference server lives only on RayJob pods. The
kernel-agent flow adapts transparently:

* Reading source — `/sgl-workspace/{aiter,sglang,vllm}/` is image-baked
  on both sandbox and pods, same commit on all sides, so `cat` /
  `read_file` work as usual.
* Applying patches — `apply_kernel_patch.py` detects multi-node via
  `_is_multi_node()` and, after writing the sandbox-local copy,
  fans the SAME patch bytes to every pod via
  `python3 -m inference_optimizer.multi_node apply-patch`. Per-host
  backup paths are persisted into the manifest so revert can hit the
  same pods. Pod fan-out failure → sandbox copy is auto-restored from
  the source backup (strict 3-way transaction: sandbox + head + workers
  all on v1, or all on v0; no partial state).
* Reverting — `revert_kernel_patch` reads `manifest.multinode.host_backup_map`
  and dispatches `python3 -m inference_optimizer.multi_node revert-patch`
  before returning.
* Compiling + benchmarking — the sandbox has no GPU. Backend prompts
  (Claude/Codex/GEAK) get a `MULTI-NODE SANDBOX` block in their safety
  instructions directing them to
  `python3 -m inference_optimizer.multi_node kernel-bench` instead of
  local `hipcc` / `torch.cuda.*` / `torch.utils.cpp_extension.load`.
  The CLI base64-encodes any helper files, stages them on a
  GPU-bearing pod, runs `bash --bench-command`, and returns
  stdout/stderr plus matching `result*.json` artifacts.
* Integrating — `integrate_handler` invokes
  `restart_server_for_round(force_full_restart=True)` after a
  successful apply so the resume fast-path is bypassed and sglang
  re-imports the patched modules. Without this the re-baseline would
  measure pre-patch behaviour and integrate decisions become noise.
* RayJob recreate — when a fresh RayJob is provisioned (after OOM /
  manual recreate), `_replay_kernel_patches_for_multi_node` in
  `inference_optimizer/cli.py` scans the session's kernel-agent
  workspace for applied manifests and replays each via `apply-patch`
  so the new pods start in the post-stack state.

## Hard Rules

- Do not implement ports or RPC.
- Do not use Claude context as source of truth; write artifacts.
- Do not modify kernel source before GEAK submission.
- Do not fabricate shapes; use trace/source/work queue data only.
- Do not apply patches to the original repo. Produce patch/artifacts/proposal.
- Do not auto-retry unknown side effects.
- Process tool requests serially in the resident Claude session; backend race can
  be parallel inside a single request.
