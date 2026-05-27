---
name: inference_optimizer
description: |
  Launches and monitors Hyperloom's multi-agent inference optimizer for LLM
  serving on AMD GPUs. Use when the user asks to optimize an inference model,
  run Magpie benchmarks/profiles, resume an inference_optimizer session, tune
  SGLang/vLLM serving parameters, run TraceLens/kernel-agent, or validate
  end-to-end throughput gains in a new inference environment.
globs:
  - "**/inference*optim*"
  - "**/inference_optimizer*"
---

# Inference Optimizer Skill

You are the launcher and monitor. The optimizer itself is the Python
`inference_optimizer` runtime under this repository. Do not manually optimize
inside chat unless debugging; launch the CLI, poll persisted state, and report
objective progress.

## What This Skill Runs

The CLI starts a Python Coordinator that coordinates:

- Orchestration: decides next actions (`baseline`, `profile`, `backends`, `params`, `sweep`, Kernel requests, `report`).
- Kernel: responder path for `select_kernels`, `run_optimization`, `integrate`.
- Critic: proposal review (default: `--critic-agent` — drives the
  `critic-agent/` skill runtime with KB priors / session memory /
  `review_constraints`-gated verdicts). `--critic-mock` for offline /
  smoke tests; `--critic-codex-bare` for debugging the LLM layer
  without the runtime layer.
- Robustness: default `--robustness-agent` — drives the `robustness-agent/`
  subprocess runtime for health monitoring, RCA, and scheduling-police
  intents. `--robustness-mock` for offline / smoke tests.
  - **Multi-node auto-downgrade (`--nodes >= 2`)**: the agent backend's
    `LocalProbeSource` targets sandbox-local resources only (ray status,
    inference server, GPU, FD, disk, shm). On multi-node every
    such resource lives in a separate pod (head / worker / RayJob), so each
    probe surfaces as a HIGH false positive that floods the bus. The CLI
    auto-downgrades to `--robustness-mock` (heartbeat only) and prints a
    WARNING; pass `--robustness-mock` explicitly to suppress it. See
    `multi_node/SKILL.md` (Robustness limitation in multi-node mode).

State lives under a **session directory** (per optimization run).
The **workspace root** is ``$USER_DATA_PATH`` (default
``/workspace/hyperloom``) — it holds shared ``runtime/`` and ``logs/``.

### Layout (N17 default: ``per_model_ts``)

```text
$USER_DATA_PATH/                          # workspace_root — set by operator / Claw / SaFE
├── runtime/                              # workspace-shared (install.sh, Magpie, kernel-agent.env.sh)
│   ├── kernel-agent.env.sh
│   ├── geak-config/local.yaml
│   ├── Magpie/
│   └── source-mirrors/{geak,OOB,TraceLens-internal}/
├── logs/                                 # workspace-shared launcher stdout
└── <model_basename>/                     # e.g. DeepSeek-R1-0528, deepseek-ai-DeepSeek-V3
    └── <UTC_YYYYMMDDTHHMMSSZ>/           # session_dir — manifest.json, state.json, runs/, …
        ├── manifest.json
        ├── state.json
        ├── storage/coordinator.db
        ├── agents/{orchestration,kernel,critic,robustness}/
        ├── runs/{baseline,profile,roofline,backends,params,...}/<task_id>/
        ├── kernel-agent/runs/<session_id>/
        ├── kernel-agent-workspace/<kernel_id>/
        ├── optimizer_runs/               # per-session launcher logs / PID / monitor
        ├── reports/
        └── …
```

**Claw / SaFE pods:** the launcher often sets ``$USER_DATA_PATH`` to a
run-scoped path *before* the optimizer starts, e.g.
``/hyperloom/users/<uid>/deepseek-ai-DeepSeek-V3-20260522_034024/``.
That outer directory is **platform isolation** (one Claw job). The
optimizer then creates ``<model_basename>/<UTC_ts>/`` inside it. Full
session path example::

    /hyperloom/users/<uid>/deepseek-ai-DeepSeek-V3-20260522_034024/   ← USER_DATA_PATH (Claw)
        deepseek-ai-DeepSeek-V3/20260522T035359Z/                      ← session_dir (optimizer)

**Legacy flat layout:** set ``INFERENCE_OPTIMIZER_SESSION_LAYOUT=flat``
so ``session_dir == workspace_root`` (no ``<model>/<ts>`` subdirs).

### Path resolution (do not guess)

| Concept | Env / helper | Meaning |
|---|---|---|
| Workspace root | ``$USER_DATA_PATH`` → ``paths.workspace_root()`` | Shared ``runtime/``, parent of all sessions |
| Session dir | ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`` → ``paths.session_dir()`` | Where ``manifest.json`` / ``state.json`` live |
| Session id | ``manifest.json`` → ``session_id`` | Logical label only — **not** a directory name |

Resolution order for ``paths.session_dir()``:

1. ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`` (pin from CLI — **authoritative**)
2. ``$USER_DATA_PATH`` (legacy flat / tests without pin)
3. ``/workspace/hyperloom``

**Iron rule for agents:** never treat ``$USER_DATA_PATH`` as the session
dir when ``$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`` is set. Read
``manifest.json`` / ``state.json`` from the **session dir** (CLI prints
``Session dir : …`` at startup). For monitoring after launch, parse that
line or walk ``$USER_DATA_PATH/<model_basename>/`` for the latest
``*T*Z/`` timestamp dir.

Path helpers (don't string-concat):

| Helper | Returns |
|---|---|
| `paths.workspace_root()` | `$USER_DATA_PATH` (workspace root) |
| `paths.session_dir()` | Pinned session dir (see resolution order above) |
| `paths.make_session_dir(model_name=…)` | Creates `<workspace>/<model>/<ts>/` + pin |
| `paths.find_latest_per_session_dir(model_name=…)` | Latest `*T*Z/` under workspace (for `--resume`) |
| `paths.db_path_for(sd)` | `<sd>/storage/coordinator.db` |
| `session_paths.runs_dir(sd, kind, task_id)` | `<sd>/runs/<kind>/<task_id>/` |
| `session_paths.kernel_workspace(sd, kernel_id)` | `<sd>/kernel-agent-workspace/<kernel_id>/` |
| `session_paths.patches_dir(sd, kernel_id)` | `<sd>/patches/<kernel_id>/` |
| `session_paths.agent_log(sd, role)` | `<sd>/logs/<role>.log` |
| `session_paths.agent_prompt_snapshot(sd, role)` | `<sd>/agents/<role>/system_prompt.snapshot.md` |
| `manifest.write_manifest(sd, args)` / `load_manifest(sd)` | manifest.json read/write |

Inputs that stay outside `$USER_DATA_PATH` by design (read-only sources
or warm-start caches): `$TRACELENS_ROOT` (default `/wekafs/hyperloom/
TraceLens-internal`; **must** be at tag `Hyperloom_integration_v0.3.1`
or the matching `release/hyperloom_integration_v0.3.1` branch — the
per-version `sglang_roofline_patches/sglang_<minor>_<patch>/` layout is
required by `_server_patcher`),
`$OOB_SRC` / `$HYPERLOOM_BUNDLE`,
`/sgl-workspace/{aiter,sglang,vllm}/`, `~/.claude/config.json` +
`~/.codex/auth.json`, `~/.cache/amd-ai-devtool/semantic-index/`
(GEAK RAG embedding cache), `/wekafs/hyperloom/geak-memory/memory.db`
(GEAK cross-session memory). Each is overridable via its own env if
you want a fully self-contained session.

Paths emitted by agents must resolve under the **session dir** — PolicyGate
enforces this (with a framework-source allowlist for `source_file`:
`/sgl-workspace/{aiter,sglang,vllm}/` plus any paths in
`$INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS` — colon-separated, unioned
with defaults; auto-probed by `inference_optimizer/scripts/install.sh`).

Always prefer `manifest.json` / `state.json` / `coordinator.db` under the
**session dir** over guessing from terminal logs.

## Iron Rules

SKILL-level constraints the launcher MUST satisfy before `Coordinator`
is allowed to boot. These IronRULEs are the gate
that runs **before** `inference_optimizer optimize` is even spawned.

### IR-1 — GPU MUST be unoccupied before every launch

Before every `inference_optimizer optimize` invocation (fresh start OR
`--resume`), verify that every visible GPU on this pod has **zero
foreign serving PIDs and ≲ 500 MiB VRAM in use**. A leftover
`sglang.launch_server` / `vllm.entrypoints` / `Magpie` from a previous
run silently degrades the next `baseline` by 5–30 % (shares VRAM +
schedules on the same XCD); neither `current_best` nor
`validate_stack` can detect this pollution after the fact.
> Inside a running session, the equivalent guard is Kernel-agent IR-4
> (`kill_server` + `check_gpu_memory` before every server (re)start —
> see `orchestrator/system_prompts/kernel.md`). IR-1 above is the
> *outer* gate that fires before the optimizer process exists.

### IR-2 — install.sh MUST succeed before every launch

Run `bash "$REPO_ROOT/inference_optimizer/scripts/install.sh"` and
source the regenerated
`${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}`
in the **same shell** that will spawn `inference_optimizer optimize`.
Skipping install strikes silently *after* `baseline` succeeds: missing
TraceLens/GEAK/OOB CLI → `select_kernels` / `kernel_opt` fail; no live
Ray head → `kernel_opt` tasks hang; missing `kernel-agent.env.sh` →
first claude/codex call returns `401`. `install.sh --check-only` is a
*diagnostic*, never a substitute.

**Resume carve-out.** `... optimize --resume` may skip install only when
ALL hold: (1) `install.sh` exited 0 earlier in the *same shell*; (2)
`kernel-agent.env.sh` is still sourced; (3)
`${USER_DATA_PATH:-/workspace/hyperloom}/manifest.json` exists. Any
failure → treat as fresh launch and re-run `install.sh`.

> The in-loop equivalent is `_preflight()` steps 1–12 (drift repair, not
> a substitute for this outer gate).

### IR-3 — KB + PR Monitor reachability (in-loop, soft degrade)

`_preflight()` invokes:

```
bash "$REPO_ROOT/inference_optimizer/scripts/preflight_kb.sh"
```

Exit codes (soft degrade — IR-3 never aborts launch):

- `0` → KB + PR Monitor both reachable. `cortex_enabled` / `pr_monitor_enabled` stay `True`.
- `1` → at least one branch unreachable. The cli automatically enables the
  matching `--degraded-*` and continues; `manifest.json` records
  `kb_degraded_reason=ir3_auto` (or `pr_degraded_reason=ir3_auto`).

Operator opt-out: pass `--degraded-kb` / `--degraded-pr` to skip the
corresponding probe (one round-trip saved); `manifest.json` then
records `reason=explicit_flag`. Both flags together short-circuit the
entire IR-3 step.

### IR-4 — EXPLORE is specialist-first (PR-A9 Arbor-into-Hyperloom)

PolicyGate's `explore_requires_specialist_provenance` rule denies any
`delegate{action_name='explore'}` whose grid is entirely the legacy
`provenance='llm_direct'`. Every EXPLORE round must trace its variants
to one of:

- `provenance='specialist:<domain>'` — variant came from a
  `specialist_done.proposal_set` entry. The canonical path. **At most
  ONE such variant per explore round** (rule
  `explore_specialist_grid_max_one`); pick the strongest proposal and
  defer the runners-up to a subsequent round.
- `provenance='default_grid'` — cold-start fallback when no specialist
  has produced a proposal_set yet. The executor uses its built-in grid;
  uncapped (several `default_grid` variants in one round is fine).

The orchestration LLM is taught the specialist-first contract via
`actions/_meta/specialist.yaml` (PR-A1) plus the orchestration prompt's
EXPLORE section. Inv-5.1 (`specialist 不出 patch`) was relaxed by PR-A2 +
PR-A4: specialists MAY author source patches into their isolated
worktree (`runs/specialist/<task_id>/worktree/`), but the actual
`git apply` against `framework_source_roots` is the sole job of the
`integrate_patch` action (PR-A4) which holds the serving lanes and
runs the throughput + accuracy gate.

This rule is the final shape of the Arbor-into-Hyperloom porting
effort (see `Agent-deligate-gap.MD` and the `arbor-dispatch-into-hyperloom`
plan). Cold-start sessions can still proceed via `default_grid`; every
subsequent round should be specialist-derived for Arbor-grade gains.

### IR-6 — EXPLORE HARD force-exit on low budget

`phase_state.should_force_exit_explore` exits EXPLORE the moment EITHER
of the following holds:

- total wall-clock remaining (`SharedState.remaining_minutes()`) is below
  `--explore-force-exit-hours-remaining` (default **3.0 h**), OR
- EXPLORE's remaining phase budget is below
  `--explore-force-exit-budget-pct` (default **20%** of its allotted
  slice).

The gate is non-negotiable — the steward / plateau judge / LLM
proposals cannot extend EXPLORE past either threshold. Routes
EXPLORE → KERNEL (or → SWEEP when `--no-kernel`) via the standard
`compute_next_phase` plumbing; the new exit reason
`explore_force_exit_low_budget` lands in both `PHASE_EXIT_REASONS`
and `STOP_REASON_VOCAB` so resume + breakdown collectors see it.

Rationale (report iter 19 lesson): leave at least 3 h of buffer
for the downstream KERNEL → SWEEP → CLOSE sequence so the session
can produce a clean report + recipe write-back. EXPLORE that
consumes the entire budget loses the value of every KEEP because
the report never lands.

### IR-7 — Honest self-stop via session_steward_specialist

On EXPLORE plateau (the canonical `compute_plateau_explore` judge —
real plateau, not the legacy m2_proxy), Coordinator enqueues an
internal `session_steward_specialist` task BEFORE permitting the
EXPLORE→KERNEL transition. The steward reads the full session state
(`optimization_stack`, `explore_search.rejected`,
`specialist_rounds`, `gaps[]`, `policy_denial_history`) and returns
one of:

- `recommendation='stop_session'` → Coordinator sets
  `stop_reason='no_more_leverage'`; CLOSE phase runs next.
- `recommendation='advance_to_kernel'` → Coordinator writes
  `pending_escalate_hint='skip_to_kernel'`; the next
  `compute_next_phase` advances to KERNEL (or SWEEP under
  `--no-kernel`).
- `recommendation='continue_explore'` → Coordinator injects
  `next_gap_canonical_id` into `gaps[]`, resets
  `params_no_promote_streak` + per-domain empty streaks, sets
  `steward_continuation_used=True`. **Only one continuation per
  session**: a second `continue_explore` is coerced to
  `advance_to_kernel`.

The steward is purely advisory at the SOFT layer — IR-6 still wins
when wall-clock budget drops below the threshold, regardless of
any steward verdict. Operators can disable the steward entirely
via `--steward-disabled`; the plateau judge then exits EXPLORE
directly without consulting it.

LLM-side `propose_action{action_name='assess_remaining_gaps'}` is
allowed when the LLM thinks plateau is imminent but the
Coordinator hasn't fired yet. PolicyGate
`assess_remaining_gaps_throttle` denies back-to-back proposals
within `INFERENCE_OPTIMIZER_ASSESSMENT_MIN_INTERVAL_SEC`
(default 1800s).

### FRAMEWORK_PR phase

Inserted between PRELUDE and EXPLORE. Gated by
`SharedState.framework_phase_enabled` (CLI `--no-framework` opts
out; default on). Coordinator owns the loop end-to-end — the LLM
never proposes the `framework_pr` action; PolicyGate
`framework_pr_action_not_llm_proposable` denies any attempt.

Flow per tick (`_pump_framework_pr_phase`):
1. If no pending/running `framework_pr` task and no current batch:
   call `fa phase-discover` for a fresh candidate batch (model +
   framework + gpu_type + `gaps[]`).
2. Pop the next candidate; enqueue a `framework_pr` task with
   `requires_lanes=[server_lifecycle, workspace_mutation, benchmark_lane]`.
3. `FrameworkPrExecutor` (a) calls `fa phase-fetch` to apply the PR
   into an isolated worktree, (b) shells `fa phase-emit-proposal`
   to build a `specialist_done`-shaped envelope, (c) runs
   `run_grid([single_variant])` for benchmarking.
4. Result lands via `_promote_to_shared_state['framework_pr']` —
   KEEP triggers a `cumulative_gain_validated` update + watermark
   refresh (`_maybe_enqueue_watermark_roofline(reason="framework_pr_keep_watermark")`).
5. Per-candidate row recorded in `framework_pr_phase_progress`;
   batch totals in `framework_pr_batches`.

Exit (`exit_normal_framework_pr`, 3-way precedence):
- `framework_pr_force_exit_low_budget` — remaining wall-clock <
  `0.6 × max_hours`.
- `framework_pr_plateau` — 3 consecutive batches with
  `max_gain_pct_observed_in_batch < 1.0`.
- `framework_pr_phase_done` — `framework_pr_phase_done=True`
  (set when `fa phase-discover` returns an empty batch).

Resume: same shape as EXPLORE — completed candidates skip via the
task registry idempotency key (`framework_pr:<batch_id>:<cand_id>`),
in-flight ones are dropped + redone.

## Retired modules and rules (do not re-introduce)

These orchestrator modules were intentionally removed; the
`actions/_meta/*.yaml` registry + `_grid_runner.py` + specialist-first
EXPLORE flow replaced them. Re-adding them re-creates conflicting
decision paths:

- `orchestrator/backends.py` (the action-routing one — distinct from
  the LLM-adapter directory `orchestrator/backends/`)
- `orchestrator/params.py`
- `orchestrator/validate_stack.py`
- `orchestrator/scoring.py`

Related rules that look reasonable but break things:

- **No `framework_pr first-explore priority` rule** in
  `system_prompts/orchestration.md` — conflicts with **IR-4**.
  Framework-agent runs in the dedicated **FRAMEWORK_PR** phase
  before EXPLORE; the LLM never proposes the `framework_pr`
  action (PolicyGate denies it via
  `framework_pr_action_not_llm_proposable`). Use `--no-framework`
  to skip the phase entirely.
- **No `sequence_denial` rule** consuming `backends_attempts` /
  `params_attempts` — those fields have no writers and would
  permanently deny `kernel_opt`. Use
  `explore_attempts_minimum_before_kernel_opt`.

## Setup

Two commands: Step 1 implements **IR-2** (install gate), Step 2 launches.
Both are idempotent; do not replicate them inside chat.

### Credentials (env only)

`SAFE_API_KEY` and `OPENAI_BASE_URL` are the only credentials this skill
needs and must be exported in the calling shell before running install
or the CLI (typically by sourcing `$HYPERLOOM_KERNEL_AGENT_ROOT/env.sh`
after Step 1). `install.sh` and the CLI's `_preflight()` read them from
`os.environ` only — no `.env` files are loaded.


### Step 1 — Install (one-time per pod / venv rebuild)

```bash
export REPO_ROOT="$(pwd)"   # repo root containing kernel-agent/ + inference_optimizer/ + .env
bash "$REPO_ROOT/inference_optimizer/scripts/install.sh"
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}"   # pod-local runtime env
```

`inference_optimizer/scripts/install.sh` is the only install entrypoint for
full inference optimization. It installs the optimizer / Magpie / InferenceX
first, then chains to `kernel-agent/scripts/install.sh` for the kernel
optimization environment. `kernel-agent/scripts/install.sh` remains valid for
standalone kernel-agent debugging, but should not be the main entrypoint for a
full inference optimizer session.

The install phase always initializes the full Hyperloom runtime. Even if the
user later passes `--no-kernel` at runtime, the installer still prepares
kernel-agent / TraceLens / GEAK / OOB CLI auth; `--no-kernel` only means
that this `optimize` run skips the kernel optimization phase.

`install.sh` installs everything in one shot (no `--with-*` flags to
remember). Direct steps in `inference_optimizer/scripts/install.sh`:

| Component | Provided by |
|---|---|
| `inference_optimizer` pkg + `claude_agent_sdk` extras (`pip install -e .[test]`) | `ensure_inference_optimizer` |
| **Magpie** (`git clone --depth 1 $MAGPIE_REPO $MAGPIE_DIR` + `pip install -e`; default `$MAGPIE_DIR=$HYPERLOOM_RUNTIME_DIR/Magpie`) | `ensure_magpie` |
| `INFERENCEX_PATH` auto-detection (scans `$MAGPIE_DIR/InferenceX` → `$HYPERLOOM_RUNTIME_DIR/InferenceX` → WekaFS fallbacks) | `ensure_inferencex` |
| `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS` appended to `kernel-agent.env.sh` | `_probe_framework_source_roots` |

Chained from `kernel-agent/scripts/install.sh` (single chain at the end
of `inference_optimizer/install.sh`):

| Component | Provided by |
|---|---|
| `ray==2.44.1` + `click<8.3.0` | pip |
| TraceLens internal (perf-report CLI) | `ensure_tracelens` (`cp -r` from read-only WekaFS mount to `${HYPERLOOM_ROOT}/TraceLens-internal` = `$USER_DATA_PATH/runtime/source-mirrors/TraceLens-internal`) |
| GEAK CLI + `${HYPERLOOM_RUNTIME_DIR}/geak-config/local.yaml` | `ensure_geak` |
| Node.js/npm + OOB CLI + claude/codex npm CLIs + `@cursor/sdk` global install + `~/.claude/config.json` + `~/.codex/auth.json` | `ensure_node` + `ensure_oob` (mirrors `${HYPERLOOM_BUNDLE}/OOB` → `${HYPERLOOM_ROOT}/OOB/oob_cli`) |
| `CURSOR_API_KEY` / `CURSOR_DEFAULT_MODEL` exported to `kernel-agent.env.sh` if set in env (cursor backend uses Cursor's own gateway). When `CURSOR_API_KEY` is unset, `cursor` is auto-skipped from default backend selection (`choose_backends` / `recommend_backends` / batch fallback ladder / `parallel_e2e_runner --backends` default); explicit user-supplied backends are still honored. | `write_env_file` |

`${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}` is
regenerated by `install.sh` and contains the proxy-rewritten URLs, auth aliases,
GEAK config path, and InferenceX path. Source it (don't try to derive these by
hand). Generated env/config state is written to the pod-local runtime directory,
not back into a shared WekaFS source checkout.

**Multi-node escape hatch**: if `$TRACELENS_ROOT` / `$OOB_SRC` / `$GEAK_REPO` /
`$WORKSPACE_ROOT/Magpie` / `$INFERENCEX_PATH` may move or differ across nodes,
`rsync -a` them into `$SESSION_DIR/vendor/<name>/` and override the matching
env vars BEFORE running `install.sh`. Single-node WekaFS-mount setups (the
production default) need none of this — `ensure_tracelens` / `ensure_oob`
already handle the read-only-source case.

### Step 2 — Launch

**Multi-node (`nodes >= 2`):** [`multi_node/SKILL.md`](multi_node/SKILL.md).

```bash
inference_optimizer optimize \
  --model "$MODEL_PATH" \
  --framework vllm \           # or sglang (default)
  --gpu-type MI300X \          # or omit for rocm-smi auto-detect
  --model-class moe_mla \      # dense / moe_mla / moe_swa / moe_mla_nsa; biases per-action curated priors
  --max-hours 2 \
  --compare-against-gpu B200   # optional — when set, fetches real InferenceX reference; when unset, target_analysis still runs and writes a 'no_target_gpu_configured' marker JSON
```

**Caller responsibility (post-classify-removal)**: the in-loop `setup` /
`classify` actions were deleted; the SKILL caller is now expected to
supply session metadata directly via CLI flags / env vars:

| Surface | CLI flag | Env var | Notes |
|---|---|---|---|
| Model path | `--model` | — | required |
| Framework | `--framework` | `FRAMEWORK` | `vllm` / `sglang` |
| GPU type | `--gpu-type` | `GPU_TYPE` | rocm-smi auto-detect when unset |
| Model class | `--model-class` | `MODEL_CLASS` | drives `orchestrator/scoring.MODEL_CLASS_ACTION_PRIORS`; defaults to `moe_mla` when unset |
| External reference GPU | `--compare-against-gpu` | — | Coordinator *always* hard-gates `target_analysis` as TODO 0 so `$SESSION_DIR/target_analysis/target_baseline.json` exists before `baseline` runs. When this flag is set the JSON carries the InferenceX reference (`reason="ok"`); when unset the JSON carries a structured `reason="no_target_gpu_configured"` marker. The report renders the "External baseline" section from this JSON in both cases (heading switches to "(not requested)" for the marker variant) |

A user request to optimize a model is approval to run Step 1 on a fresh
node; do not stop for an extra confirmation. After IR-2, smoke-test the
CLI:

```bash
export HYPERLOOM_KERNEL_AGENT_ROOT="$REPO_ROOT/kernel-agent"
export KERNEL_AGENT_ROOT="$HYPERLOOM_KERNEL_AGENT_ROOT"
export WORKSPACE_PATH="${WORKSPACE_PATH:-/workspace}"
export TRACELENS_ROOT="${TRACELENS_ROOT:-/wekafs/hyperloom/TraceLens-internal}"

export PYTHON="${PYTHON:-$(command -v python3)}"
export PATH="$(dirname "$PYTHON"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

bash "$REPO_ROOT/inference_optimizer/scripts/install.sh"
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}"
"$PYTHON" -m inference_optimizer.cli --help
```

Quirks: with `set -u`, assign dependent vars on separate lines (chained
`export A=... B=$A` can fail with `unbound variable`). The installer
leaves a live Ray head; `ray status` must succeed because `select_kernels`
submits tasks with `num_gpus>=1` — never restart Ray with `--num-gpus=0`.

`_preflight()` runs every launch as the in-loop counterpart of IR-2.
Steps 1–9 run before it returns; 10–12 run before Coordinator boots.
Cite the linked section for fixes:

| #  | Check | On-fail / Reference |
|----|---|---|
| 1  | Re-export auth aliases (`ANTHROPIC_AUTH_TOKEN`, `OPENAI_API_KEY`, ...) from `SAFE_API_KEY`. `OOB_BASE_URL` / `GEAK_BASE_URL` / `LLM_API_BASE` inherit `OPENAI_BASE_URL` directly (Bearer-native, no proxy). | — |
| 2  | Auto-`pip install` missing `claude-agent-sdk>=0.1.65`, `openai>=1.50`, `httpx>=0.27` into `sys.executable` | `## Failure Handling` — `claude-agent-sdk not installed` |
| 3  | Derive `ANTHROPIC_BASE_URL` from `OPENAI_BASE_URL` (strip trailing `/v1`); force-override both env vars to keep them consistent (overriding any shell/`.env`/k8s value, logged on stdout). | `### Recovery` |
| 4  | Reset `~/.claude/config.json` `customApiUrl` to the upstream `ANTHROPIC_BASE_URL` so any stale `127.0.0.1:4002` value is replaced. | `### Recovery` |
| 5  | ROCm hygiene (WARN-only): pop `HIP_VISIBLE_DEVICES` if `ROCR_VISIBLE_DEVICES` also set; visible-GPU count vs `$TP` via `rocm-smi --showid`; `/dev/shm` free ≥ 16 GiB | — |
| 6  | Auto-install missing `ray` / `Magpie` / `InferenceX` (pod rebuild recovery) | — |
| 7  | Auto-detect `--gpu-type` if not given | `## GPU Runner Type` |
| 8  | WARN-only presence check: `node` / `claude` / `codex` CLIs + `@cursor/sdk` (resolved via `node -e "require.resolve('@cursor/sdk')"` against `$(npm root -g)`) | — |
| 9  | Emit canonical `Preflight diagnostics:` block (`asset_root`, `session_dir` + resolving env var, `magpie_python`, `INFERENCEX_PATH`, aiter jit cache WARM/COLD + `.so` count + path, cold/warm timeouts, resolved `ANTHROPIC_BASE_URL`). Paste verbatim into status reports. | `## Cold-start Discipline` |
| 10 | Hard model gate: `--claude-model` ∈ {`claude-opus-4-7` (preferred), `claude-opus-4-6` (fallback)}; probe `GET <OPENAI_BASE_URL>/models` with Bearer (3 retries 1s/3s/5s); rewrite to `4-6` if `4-7` missing; abort if neither present or gateway unreachable | `## Failure Handling` — model-gate errors |
| 11 | Codex smoke-test (WARN-only): `--codex-model` checked when codex actually used (`--critic-agent` / `--critic-codex-bare` / `--kernel-codex`) | — |
| 12 | Critic-agent runtime probe (when `--critic-agent` active): resolve `CRITIC_AGENT_ROOT` (env > sibling `$REPO_ROOT/critic-agent/` > abort), `python -m runtime.cli --help` (5s timeout); abort rc=2 if it fails. Default-sets `WORKSPACE_PATH` / `CRITIC_SESSION_MEMORY_DIR` / `CRITIC_KB_CLIENT_MODE`. | `## Critic Backend Selection` |

Don't manually pip-install SDKs, edit `~/.claude/config.json`, start Ray,
or `curl /v1/models` — `_preflight()` owns these. See `kernel-agent/SKILL.md`
for the chained installer truth.

### Recovery

If the CLI exits with `Claude SDK exit code 1` or `Primus.00009 token not present`,
the gateway rejected the request. Check that `OPENAI_BASE_URL` / `SAFE_API_KEY`
are set in `.env` (or the calling shell) and that the gateway is reachable:

```bash
curl -sS -H "Authorization: Bearer $SAFE_API_KEY" "$OPENAI_BASE_URL/models" | head
```

If `_preflight()` itself fails, run install in `--check-only` mode to see
which piece is missing, then re-run full install:

```bash
bash "$REPO_ROOT/inference_optimizer/scripts/install.sh" --check-only
bash "$REPO_ROOT/inference_optimizer/scripts/install.sh"
```

In sandboxes where `/workspace/hyperloom` is unwritable, override the
**workspace root** with `USER_DATA_PATH` (not the per-session subdir):

```bash
export USER_DATA_PATH="/wekafs/xiaofei/sessions"   # workspace root
mkdir -p "$USER_DATA_PATH"
```

The CLI calls `make_session_dir(model_name=…)` once at startup; that
creates `$USER_DATA_PATH/<model_basename>/<UTC_ts>/` and pins
`$INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR`.

## Portable Preflight

Implements **IR-1**. Run order is always **IR-2 → IR-1 → launch**:
without IR-2 the script below has no `torch` to import. Verify the
model path, GPU visibility, and that no stale serving process holds
VRAM; exit non-zero on any violation so the calling shell aborts
before `inference_optimizer optimize` is spawned. Never print tokens.

```bash
export MODEL_PATH=/path/to/model
test -d "$MODEL_PATH"

"$PYTHON" - <<'PY'
import os
try:
    import torch
    print("torch_cuda_available=", torch.cuda.is_available())
    print("torch_cuda_device_count=", torch.cuda.device_count())
except Exception as exc:
    print("torch_check_error=", type(exc).__name__, str(exc)[:300])

patterns = ("inference_optimizer.cli", "Magpie", "sglang.launch_server")
for pid in filter(str.isdigit, os.listdir("/proc")):
    try:
        cmd = open(f"/proc/{pid}/cmdline", "rb").read()
    except Exception:
        continue
    text = cmd.replace(b"\0", b" ").decode("utf-8", "ignore")
    if text and any(p in text for p in patterns):
        print(f"existing_process {pid}: {text[:300]}")
PY
```

## Benchmark Config

Default configs live here:

```bash
inference_optimizer/scripts/configs/baseline_sglang.yaml
inference_optimizer/scripts/configs/baseline_vllm.yaml
inference_optimizer/scripts/configs/profile_sglang.yaml
inference_optimizer/scripts/configs/profile_vllm.yaml
```

Two fields in each YAML are **fallback only** — the optimizer overrides
them at runtime:

- `benchmark.model` <- `--model` / `$MODEL_PATH`
- `benchmark.runner_type` <- `--gpu-type` / `$GPU_TYPE` / rocm-smi auto-detect

`benchmark.benchmark_script` is deliberately NOT set in the shipped
YAMLs. At materialize time Hyperloom pins it to
`{framework}_{runner_type}.sh` (e.g. `sglang_mi300x.sh` /
`sglang_mi355x.sh`) so Magpie's resolver hits priority 1 (explicit
user override) and uses the generic script — which respects
`RESULT_DIR` and `EXTRA_*_ARGS`. Each shipped YAML has a commented
`# benchmark_script: ...` template right under `framework:` for manual
debug overrides; Orchestration can also route per-task via
`params.benchmark_script` (sanitized).

Before a new model run, verify these fields match the environment:

- `benchmark.model`: model path.
- `benchmark.envs.TP`: tensor parallel size.
- `benchmark.envs.CONC`, `ISL`, `OSL`: workload.
- `benchmark.envs.ROCR_VISIBLE_DEVICES`: GPU pinning.
- `benchmark.envs.PATH`: must lead with the launcher Python's bin dir
  (`$(dirname "$PYTHON")`).

### Magpie leak-path salvage (`INFERENCE_OPTIMIZER_RESCUE_PATHS`)

The default benchmark path force-pins `{framework}_{runner_type}.sh` at
materialize time so Magpie's resolver picks the generic script that
respects `RESULT_DIR` and `EXTRA_*_ARGS`. The salvage logic below is
defense-in-depth; it primarily fires when an operator explicitly opts
into a leaky script via `params.benchmark_script` whose underlying
`*.sh` hardcodes a `--result-dir` outside the per-task workspace.

When the in-workspace search produces no usable measurement,
`extract_benchmark_measurement` runs a second-chance salvage pass over
documented leak destinations:

1. `$INFERENCE_OPTIMIZER_RESCUE_PATHS` — colon-separated list of files
   and/or directories. Directories are scanned for
   `inferencex_result*.json`.
2. Default: `/workspace/inferencex_result.json`.

Salvage is mtime-gated: only files written *after* the executor
captured `subprocess_started_unix = time.time()` (right before
`subprocess.run`) are eligible, so a stale leak can never masquerade
as the current run's result. Adopted leaks are tagged in the result's
`nonfatal_warnings` as `rescued_from_leaked_path:<path>`.

When salvage adopts a leaked file, `_materialize_rescue_into_workspace`
also `shutil.copy2`s it into the task workspace (preserving the
basename — `inferencex_result.json`, `inferencex_result_eval.json`,
etc.) so the canonical artifact lives alongside `benchmark_report.json`
and the NFS clone of `<session>/runs/<action>/<task_id>/` is
self-contained. `raw_result_path` advertises the in-workspace copy;
the original leak location is preserved verbatim in the
`rescued_from_leaked_path:<path>` warning. The copy is best-effort:
on a permission / disk error the salvage falls back to the leak path
and additionally emits `rescued_copy_into_workspace_failed:<path>`.

`inferencex_result.json` is one of several artifacts Magpie's shell
wrappers hardcode under `/workspace/`. The single_node `*.sh` scripts
also redirect:

* `SERVER_LOG=/workspace/server.log` — sglang/vllm server stdout+stderr
  (the smoking gun for GPU OOM / checkpoint-load failures).
* `GPU_METRICS_CSV=/workspace/gpu_metrics.csv` — per-second
  power/temp/utilisation from `start_gpu_monitor`.
* `/workspace/profile_*.trace.json.gz` — the PROFILE relay copy of
  the torch profiler trace (`benchmark_lib.sh:540`).

`harvest_leaked_artifacts` runs in `BaselineExecutor.__call__` and in
`_grid_runner._run_magpie` **unconditionally** after the subprocess
returns (including the timeout / no-workspace branches) and copies
every fresh match (mtime-gated against `subprocess_started_unix`) into
the task workspace alongside `benchmark_report.json`. The leak source
files stay in place; each harvested artifact is tagged in
`nonfatal_warnings` as `harvested_leaked_artifact:<source>`. The scan
root list is `$INFERENCE_OPTIMIZER_LEAK_ROOTS` (colon-separated, with
`/workspace` as the default); operators can extend it without touching
code, and the test suite pins it to a sandbox via an autouse fixture
so unit tests stay isolated from the host's real `/workspace`.

The same mtime gate is applied per variant in `_grid_runner.py`
(`variant_started_unix = time.time()` is captured immediately before
each `_run_magpie` call and forwarded to
`extract_benchmark_measurement(subprocess_started_unix=...)`), so
variants in a `backends` / `params` / `sweep` grid never adopt
another variant's `/workspace/inferencex_result.json`. The same
salvage helpers apply to validate_stack runs.

Orchestration can route per-task via two `task.params` knobs
(descriptive `params_schema` blocks live in each
`actions/_meta/<action>.yaml`):

* `params.benchmark_script` — bare `*.sh` file name (sanitized at every
  executor boundary via `sanitize_script_name`; path separators / shell
  metacharacters are rejected with `error_class=bad_param`). When set,
  Magpie's `benchmark.benchmark_script` is rewritten in the materialized
  YAML *after* the gpu_type → script auto-selection runs, so the
  operator pick wins. Honored by baseline / profile / validate_stack /
  backends / params / sweep.
* `params.result_dir` — absolute or workspace-relative path (sanitized
  via `sanitize_result_dir`). Forwarded as `$RESULT_DIR` for that
  subprocess. The executors ALWAYS set `$RESULT_DIR`, defaulting to the
  per-task workspace (baseline) or the per-variant slot (grid_runner),
  so Magpie scripts that respect the env var write into the optimizer's
  workspace by default; operators only override when redirecting at a
  known leak destination already on `$INFERENCE_OPTIMIZER_RESCUE_PATHS`.

Coordinator stamps the canonical `_baseline_params_fingerprint` (a
projection over `benchmark_script` / `result_dir` / `extra_sglang_args` /
`extra_envs` / `model_path` / `gpu_type` / `config_path` /
`disable_run_eval`) on every baseline audit entry (success path in
`_promote_to_shared_state`, failure path in `_handle_unpromotable_result`).
PolicyGate enforces a `baseline_self_loop` denial when two consecutive
failed baseline attempts carry the same fingerprint AND Orchestration
proposes a third attempt with that same fingerprint; the denial's
`hint` points at the next override surface so the prompt's FAILURE
RECOVERY block has a deterministic recovery path.

### Workload-contract reuse (baseline → params/backends/sweep)

`baseline` materializes its YAML once with the operator's process env (`CONC` /
`ISL` / `OSL` / `TP` / `MAX_MODEL_LEN` / `PRECISION` / `RUN_EVAL` /
`ROCR_VISIBLE_DEVICES` plus adaptive `NUM_PROMPTS` / `NUM_WARMUPS`), saves it
as `baseline_config.with_envs.yaml`, and forwards the path on
`SharedState.baseline_config_path` as `task.params["config_path"]` to every
`params` / `backends` / `sweep` task. Variants thus benchmark the **same
workload baseline ran**; without this contract they would render from the
YAML's smoke defaults (`TP=1` / `CONC=8` / `ISL=256` / `OSL=256`) and produce
~10x lower throughput. Downstream actions
re-materialize on top of `config_path`; per-variant `extra_envs` (e.g. sweep's
explicit `CONC`/`ISL`/`OSL`) still win because `_grid_runner._build_variant_yaml`
applies them last.

## Critic Backend Selection

The Critic role has three backend modes, picked by mutually-exclusive
CLI flags. Default is `--critic-agent` (no flag needed).

| Flag | Backend class | Behaviour |
|---|---|---|
| (none) / `--critic-agent` | `CriticAgentBackend` | Drives the standalone `critic-agent/` skill runtime via `python -m runtime.cli prepare-review` → Codex chat completion → `python -m runtime.cli commit-review`. Adds KB priors lookup (with circuit-breaker for unreachable services), per-session memory + idempotent `reviewed_msg_ids` (no double-verdict), `judge_bundle.review_constraints` injected into the LLM prompt, and `needs_review` / `critic_unavailable` source when context is missing. |
| `--critic-mock` | `MockCriticBackend` | Always-approve adapter. Use for offline / smoke tests when Codex creds aren't available. |
| `--critic-codex-bare` | `CodexBackend` | Legacy direct chat-completion path with no KB / session memory / `review_constraints`. Available for debugging the LLM layer in isolation. (`--critic-real` is a hidden back-compat alias.) |

Default is overridable per pod via
`INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND` (one of `mock` / `agent` /
`codex_bare`).

### Required env when `--critic-agent` is active

| Var | Purpose | Default |
|---|---|---|
| `CRITIC_AGENT_ROOT` | Path to the directory containing `runtime/cli.py`. | sibling `$REPO_ROOT/critic-agent/` |
| `CRITIC_KB_CLIENT_MODE` | `inmemory` keeps KB writes / reads off the wire. `live` requires `KB_BASE_URL`. | `inmemory` |
| `KB_BASE_URL` | KB service URL when `CRITIC_KB_CLIENT_MODE=live`. | unset (live mode aborts at start if absent) |
| `KB_TIMEOUT_MS` / `KB_RETRY_MAX` / `KB_DEAD_LETTER_DIR` | Forwarded to the runtime; see `critic-agent/AGENTS.md`. | runtime defaults |
| `CRITIC_SESSION_MEMORY_DIR` | Where the runtime persists per-session decisions / reviewed_msg_ids. | `$SESSION_DIR/critic-session-memory` (auto-set by the optimizer; co-located with the Coordinator session and cleaned up alongside it). |
| `WORKSPACE_PATH` | Skill root the critic-agent runtime resolves prompt assets against. | `$REPO_ROOT` (auto-set). |

`_preflight()` checks `CRITIC_AGENT_ROOT` resolves to a real directory
with `runtime/cli.py`, then runs `python -m runtime.cli --help` (5s
timeout) before the Coordinator boots. Missing or broken runtime
aborts the run with a clear error pointing at `--critic-mock` /
`--critic-codex-bare` as bypasses.

### Per-turn artefacts (audit trail)

Each Critic turn writes:

```text
$SESSION_DIR/critic-workdir/<turn_idx 6-digit>/
├── request.json         # raw_prompt + session_id passed to runtime.cli
├── judge_bundle.json    # output of prepare-review (proposals, KB priors,
│                          review_constraints, kb_read_skipped_reason)
├── review.json          # LLM's verdicts (extracted JSON envelope)
└── emit.json            # output of commit-review (intent_envelope +
                           kb_writes); the Coordinator consumes
                           intent_envelope verbatim.

$SESSION_DIR/critic-session-memory/<session_id>/
├── context.json          decisions.jsonl   events.jsonl
└── kb_priors_cache.json  reviewed_msg_ids.json
```

The backend prunes everything older than the latest 50 turn workdirs
on every tick to avoid unbounded growth.


## Framework Selection

A session is single-framework. Pick `sglang` (default) or `vllm` via
`--framework` or `$FRAMEWORK`:

```bash
inference_optimizer optimize --framework vllm --model "$MODEL_PATH" --max-hours 2
FRAMEWORK=vllm inference_optimizer optimize --model "$MODEL_PATH" --max-hours 2
```

Resolution order: `--framework` > `$FRAMEWORK` > `sglang` (default).

What this controls:
- Which Magpie YAML the executors default to
  (`baseline_sglang.yaml` / `baseline_vllm.yaml`,
  `profile_sglang.yaml` / `profile_vllm.yaml`)
- Which framework-specific seed grid the `explore` action falls
  back to when no `params.grid` is supplied
- Which extra-args env name `_grid_runner` writes
  (`EXTRA_VLLM_ARGS` vs `EXTRA_SGLANG_ARGS`)
- Which Marathon KB partition orchestration reads for hints

Mixing sglang and vllm in a single session is not supported; the CLI
locks `$FRAMEWORK` for the run. Resume re-reads `$FRAMEWORK` from the
shell — set it when you resume a vLLM session.

## GPU Runner Type

Pick the GPU explicitly with `--gpu-type` or `$GPU_TYPE`; without
either, the optimizer auto-detects via `rocm-smi --showproductname`
(falling back to `torch.cuda.get_device_properties(0).gcnArchName`).

```bash
inference_optimizer optimize --gpu-type mi355x --model "$MODEL_PATH" --max-hours 2
GPU_TYPE=mi300x inference_optimizer optimize --model "$MODEL_PATH" --max-hours 2
```

Accepted values: `mi300x`, `mi325x`, `mi355x`. **`mi325x` is mapped to
`mi300x`** with a warning, since the two GPUs share the same arch and
Magpie has not shipped `sglang_mi325x.sh` / `vllm_mi325x.sh` yet. If you
need a true MI325X-specific script, uncomment the `benchmark_script:`
template in the relevant YAML and point it at your script under
`InferenceX/benchmarks/...`.

Do not set `HIP_VISIBLE_DEVICES` on the known ROCm stack unless the user asks;
it can make `torch.cuda.is_available()` return false. Use
`ROCR_VISIBLE_DEVICES` for GPU pinning.

## SGLang Parameter Search

Validate SGLang first, add vLLM once SGLang is stable. `params` writes
candidates through `EXTRA_SGLANG_ARGS` and `benchmark.envs`; don't hard-code
a flag as default unless A/B keeps it across the target workload.

Default grid covers cuda graph batch caps, continuous decode steps, memory
fraction, scheduling conservativeness, chunked prefill, and max prefill tokens.
Also test the InferenceX-derived candidates:

- Cache/scheduler: `--disable-radix-cache`, `--max-running-requests 128/256`.
- Tokenization/streaming: `--tokenizer-worker-num 8/16`, `--stream-interval 30/50`.
- ROCm/TileLang envs: `SGLANG_OPT_USE_MULTI_STREAM_OVERLAP=1`,
  `SGLANG_HACK_FLASHMLA_BACKEND=tilelang`, `SGLANG_OPT_USE_TILELANG_INDEXER=true`.

Speculative decoding is model-specific — only enable `SGLANG_ENABLE_SPEC_V2=1` /
`--speculative-*` when the model has the required draft path or MTP support,
and benchmark with chat-formatted prompts (`--dsv4` for DeepSeek-V4 style)
because random prompts skew acceptance-rate results.

Judge candidates over **{1k/1k, 8k/1k} × {low CONC, high CONC}** (high-CONC
only when the model fits); KEEP only when throughput improves without
unacceptable TTFT/E2E or correctness regression. Coordinator long runs default
`max_candidates_per_round=3` (aligned with the per-specialist `proposal_set`
cap — see `DEFAULT_SPECIALIST_MAX_PROPOSALS` in `orchestrator/policy.py`);
direct runner calls may pass `0` for the full grid.

### Per-Run Asset Override (advanced)

To override shipped configs without editing them, materialize a per-run asset
root and pass `--asset-root`. `mkdir -p "$ASSET_ROOT/scripts/configs"`,
`ln -sfn` `actions/` / `kernel_opt/` / `orchestrator/` and the two
`scripts/ab_torch_compile_*.py` from `$REPO_ROOT/inference_optimizer/`, then
copy + edit the relevant `baseline_*.yaml` / `profile_*.yaml`. Reach for this
only when `_workload_envs.materialize_config_with_envs` defaults don't fit
(e.g. per-yaml `profiler.torch_profiler.enabled`); otherwise `--model` /
`--gpu-type` overrides are enough.

## Launch a New Optimization

Assumes Step 1 (install) already ran. Set `$USER_DATA_PATH` to the **workspace
root** (parent of per-session dirs). The CLI creates
`$USER_DATA_PATH/<model_basename>/<UTC_ts>/` via `make_session_dir`.
Launcher stdout / PID files go under that session's `optimizer_runs/`.
For sandboxes that don't persist `export`s across shell calls (Cursor agents),
copy `inference_optimizer/scripts/setup_env.sh.example` to
`$USER_DATA_PATH/optimizer_runs/setup_env.sh`, fill in the workload block,
and `.` it each call.
After `setsid nohup ... &`, locate the optimizer via
`pgrep -af 'inference_optimizer.*optimize'` — `$!` may be a wrapper PID.

```bash
cd "$REPO_ROOT"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}"
export PATH="$(dirname "$PYTHON"):/usr/local/bin:$PATH"
export RUN_TAG="$(basename "$MODEL_PATH")-$(date +%Y%m%d_%H%M%S)"
# RUN_LOG/PID under workspace until session_dir is known; move or re-tail
# from $session_dir/optimizer_runs/ after parsing "Session dir" from stdout.
export RUN_DIR="${USER_DATA_PATH:-/workspace/hyperloom}/optimizer_runs"
export RUN_LOG="$RUN_DIR/run_${RUN_TAG}.log"
export PID_FILE="$RUN_DIR/run_${RUN_TAG}.pid"
mkdir -p "$RUN_DIR"

setsid nohup inference_optimizer --verbose optimize \
  --model "$MODEL_PATH" \
  --framework "${FRAMEWORK:-sglang}" \
  --target-gain "${TARGET_GAIN:-10}" \
  --max-hours "${MAX_HOURS:-5}" \
  --tick-interval-sec 30 \
  --kernel-claude \
  > "$RUN_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
```

`setsid nohup ... &` is required for runs > 5 min — Cursor's background
shell can die on SSH disconnect.

Critic defaults to `--critic-agent`; Robustness defaults to `--robustness-agent`.
See [Critic Backend Selection](#critic-backend-selection) for `--critic-mock` /
`--critic-codex-bare` overrides; pod-level overrides via
`INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND` /
`INFERENCE_OPTIMIZER_DEFAULT_ROBUSTNESS_BACKEND`.

After launching, do a short health check:

```bash
sleep 30
pid="$(cat "$PID_FILE")"
test -d "/proc/$pid" && echo "optimizer_alive=true pid=$pid"
# Parse session dir from RUN_LOG or resolve latest timestamp subdir:
session_dir="$(grep -m1 '^Session dir' "$RUN_LOG" 2>/dev/null | sed 's/^Session dir[[:space:]]*:[[:space:]]*//')"
if [ -z "$session_dir" ]; then
  model_base="$(basename "$MODEL_PATH")"
  session_dir="$(ls -d "${USER_DATA_PATH:-/workspace/hyperloom}/$model_base/"*T*Z 2>/dev/null | sort | tail -1)"
fi
test -f "$session_dir/manifest.json" && echo "manifest_present=true session_dir=$session_dir"
test -f "$session_dir/state.json" && echo "state_exists=true" \
  && python3 -c "import json; print(json.load(open('$session_dir/state.json')).get('stop_reason'))"
```

Healthy = optimizer process alive + `manifest.json` + `state.json`
exist + no early `stop_reason`.

## Resume Existing Session

`--resume` auto-picks the latest `$USER_DATA_PATH/<model>/<UTC_ts>/`
(without `--resume-from`) or an explicit path via `--resume-from`.
`$USER_DATA_PATH` must stay at the **workspace root** so
`runtime/kernel-agent.env.sh` resolves. The CLI refuses to start if
`manifest.json` or `state.json` is missing in the picked session dir.

Reuse the Launch template above with these diffs: drop `--model`, add
`--resume`, set `RUN_TAG="resume-$(date +%Y%m%d_%H%M%S)"`. Resume preserves
baseline, current best, params-search state, event history, and kernel-agent
artifacts; the CLI clears stale `stop_reason` and `crash_count` before retrying.

## Robustness Monitor for Long Runs

For runs > 5 min, start a monitor in its own `setsid nohup` process. It polls
`state.json` every 5 min, exits without resuming when the session is terminal
(any `stop_reason` in `STOP_REASON_VOCAB`, `phase=CLOSE`, or
`reports/final.md` present — including failure sentinels like
`baseline_failed`), and resumes via `--resume` only when the optimizer dies
without those markers (unexpected crash).

```bash
export RUN_DIR="${USER_DATA_PATH:-/workspace/hyperloom}/optimizer_runs"
mkdir -p "$RUN_DIR"
cp "$REPO_ROOT/optimizer_runs/robustness_monitor.sh.example" \
   "$RUN_DIR/robustness_monitor.sh"
chmod +x "$RUN_DIR/robustness_monitor.sh"
setsid nohup bash "$RUN_DIR/robustness_monitor.sh" \
  > "$RUN_DIR/robustness_monitor_$(date +%Y%m%d_%H%M%S).log" \
  2>&1 < /dev/null &
```

Reads `$PID_FILE` and (optional) `$USER_DATA_PATH` / `$MAX_HOURS` /
`$TARGET_GAIN`. Edit the example before copying if defaults need to
change. `stop_reason` interpretation matches the `## Monitoring` reader.

## Monitoring

Poll at most every 5 minutes unless debugging a startup failure.

```bash
export SESSION="${USER_DATA_PATH:-/workspace/hyperloom}"
python3 - <<'PY'
import json, os, pathlib
s = json.loads((pathlib.Path(os.environ["SESSION"]) / "state.json").read_text())
for k in ("stop_reason", "baseline_tput", "cumulative_gain", "current_best",
          "last_kernel_opt", "last_select_kernels", "last_sweep"):
    print(f"{k}: {s.get(k)}")
print("params_search_last_round:", s.get("params_search", {}).get("last_round"))
print("backends_search_last_round:", s.get("backends_search", {}).get("last_round"))
PY
```

Recent action counts from SQLite (last 500 events grouped by category):

```bash
python3 "$REPO_ROOT/inference_optimizer/scripts/event_counts.py" "$SESSION"
```

## Expected Flow

The optimizer should:

1. Establish or reuse `baseline_tput`.
2. **Coordinator** auto-enqueues an analysis task at the end of
  PRELUDE (after baseline) and again whenever validated tput crosses
  the watermark (`current_tput / last_roofline_tput >= 1.10`;
  compound 10% → 21% → 33% …). The task is `roofline` (composite
  profile + trace_analyze + analysis.md snapshot) by default;
  `--no-enable-roofline` switches it to plain `profile` (trace only,
  no analysis.md) with otherwise-identical semantics. The LLM CANNOT
  propose `roofline` or `profile` — PolicyGate denies both with
  `rule='analysis_action_not_llm_proposable'`. While an analysis task
  is in flight, `specialist` / `explore` / `kernel_opt` / `integrate`
  / `deep_kernel_analysis` / `operator_tuning` / `vendor_kernel_config`
  dispatches are blocked by PolicyGate
  (`rule='wait_for_auto_roofline'`) until it lands.
3. Run `select_kernels` once per trace/config and cache the result in
  `last_select_kernels`.
4. Pick only `reusable_native_kernel_ids` for `run_optimization`.
5. Require compile + correctness + microbench/E2E evidence before KEEP.
6. Use `explore_search` to test parameters incrementally and remember
  rejected candidates across resume. The ledger keys entries by
  **content fingerprint** (a sha1 hash of sorted `extra_sglang_args` +
  sorted `extra_envs`), so renaming an already-tested variant does not
  bypass dedup — LLM-supplied `params.grid` is filtered through the same
  ledger as the default seed grid.
7. Use `optimization_stack` so backend + params + kernel changes do not
  overwrite each other.
8. Use `sweep` to understand workload-specific results beyond the smoke
  workload.

## Cache Topology & Cold-start Discipline

SGLang/vLLM on ROCm route hot fused kernels (RMSNorm / attention / MoE / GEMM /
RoPE) through `aiter`, which JIT-compiles per-shape variants on first sight
and caches `.so` on disk. First launch of a fresh (model, dtype, TP,
`max_model_len`, `max_num_seqs`, `gpu_memory_utilization`) signature can spend
30+ min in `hipcc` for 671B FP8 MoE; later launches reuse the cache in seconds.

### Cache locations

| Cache | Path | Clear |
|---|---|---|
| aiter JIT (primary cold-start cost) | `<aiter pkg root>/jit/` (resolved via `import aiter`; wheel installs hold ~80 pre-built `.so` here, plus runtime-JIT staging under `jit/build/<module>/build/`) | `rm -rf <aiter pkg root>/jit/build/` (clears JIT staging only; do NOT delete `jit/*.so` — those are wheel-bundled) |
| Triton | `~/.triton/cache/` (resolves via `$HOME`) | `rm -rf ~/.triton/cache` |
| torch.compile / Inductor | `/tmp/torchinductor_<user>/` (override `$TORCHINDUCTOR_CACHE_DIR`) | `rm -rf /tmp/torchinductor_root` |

`sgl_kernel` (`site-packages/sgl_kernel/common_ops.*.so`) is build-time only;
only `kernel_opt` / `integrate` may rebuild it.

### Cold-start triggers

First launch on this pod; change to `--max-model-len` / `--max-num-seqs` /
`--gpu-memory-utilization` / `--cuda-graph-max-bs` / `--quantization` /
`--enable-torch-compile`; pod rebuild; manual cache `rm`; aiter source patch.

### Auto-detection + timeout

`BaselineExecutor` (and `ProfileExecutor`, which subclasses it) counts `.so`
files recursively under aiter's `jit/` dir, resolved via
`importlib.util.find_spec("aiter")` (with `jit/build/` as secondary
candidate and `baseline.py:AITER_JIT_PROBE_PATHS` as legacy fallback).
Threshold: **< 20 = COLD**. Override via `INFERENCE_OPTIMIZER_AITER_JIT_DIR`.

| Condition | `subprocess.run(timeout=...)` |
|---|---|
| `task.params['timeout_sec']` set | task value (always wins) |
| Probe `found` + COLD | `BASELINE_COLD_START_TIMEOUT_SEC` (3600s; override `INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC=N`) |
| Probe `found` + WARM | `BASELINE_DEFAULT_TIMEOUT_SEC` (2400s) |
| Probe `not_found` / `error` | 2400s + WARN |

Every launch logs one `baseline_executor: ...` marker (COLD_START / WARM /
explicit / not located); grep `optimizer_runs/run_*.log` to verify. Resolved
cache state also lands in the boot `Preflight diagnostics:` block. If
COLD_START repeats across retries, JIT was killed mid-`hipcc`; bump
`INFERENCE_OPTIMIZER_COLD_START_TIMEOUT_SEC=5400` instead of relaunching.

## Kernel Apply Safety

Kernel optimization may modify `/sgl-workspace/aiter`, `/sgl-workspace/sglang`,
or compiled artifacts. Before applying a patch:

- Back up source files.
- Back up compiled `.so` / `.co` artifacts when available.
- On REVERT, restore compiled artifacts first, then source files, then restart
  the server. Avoid a rebuild on revert when the original compiled artifact was
  backed up.
- Only KEEP when correctness and E2E are acceptable.

If the user has not explicitly approved environment mutation, stop before real
apply/rebuild and ask. Dry-run and analysis are safe.

## Kernel E2E Retry Discipline

Microbench speedups are not enough. After `run_optimization` returns a candidate
kernel patch, `integrate` must validate the patch with E2E Magpie throughput and
record every attempt in `state.json`.

For the same `kernel_id + patch_path + EXTRA_SGLANG_ARGS`:

- `KEEP`: accept only when E2E gain clears the configured threshold.
- `REVERT`: reject that patch immediately and do not run it again.
- `NEEDS_REVIEW`: allow at most 3 E2E attempts. If none clears the KEEP
  threshold, reject that patch and move on to params search or a different
  reusable native kernel.

Do not repeatedly integrate the same patch because its microbench was strong.
If E2E results are unstable around zero gain, the correct action is to mark the
patch rejected, preserve the artifacts for human review, and spend the remaining
budget on untested params/backend candidates or the next kernel.

## Failure Handling

Auth / SDK drift (`Claude SDK exit code 1`, `Primus.00009 token not present`,
`ANTHROPIC_AUTH_TOKEN not set`, `BackendError: claude-agent-sdk not installed`,
`Fatal error in message reader`) is owned by `_preflight()`; see
`## Setup → Recovery` for the supervisor + install rerun loop. Manual SDK
fallback if frozen pip blocks `_ensure_python_sdks()`:
`python -m pip install 'claude-agent-sdk>=0.1.65' 'openai>=1.50' 'httpx>=0.27'`.
Transient SDK errors retry/resume up to the Coordinator emergency threshold.

### Model-gate errors (preflight #10)

Allowlist: `claude-opus-4-7` (preferred) → `claude-opus-4-6` (fallback). The
gate is intentional — opus-4-5 / haiku silently degraded prior runs.

| Symptom | Fix |
|---|---|
| `--claude-model=... is not allowed` | Drop `--claude-model` / `$CLAUDE_MODEL`. Update `_CLAUDE_ALLOWED_MODELS` in `cli.py` only when a successor is blessed. |
| `gateway catalog unreachable after retries` (4 probes at 0/1/3/5s) | Reproduce: `curl -k -H "Authorization: Bearer $SAFE_API_KEY" "$OPENAI_BASE_URL/models" \| jq '.data[].id'`. Gateway answers → proxy/SSL is wrong; gateway down → fix gateway. Fail-fast is intentional vs. 401 mid-baseline. |

### Critic-agent runtime errors

Inspect `$SESSION_DIR/critic-workdir/<latest>/{request,judge_bundle,review,emit}.json`.
Bypass with `--critic-mock` (offline / smoke) or `--critic-codex-bare` (legacy
direct Codex). See `## Critic Backend Selection`.

| Symptom | Fix |
|---|---|
| `--critic-agent selected but critic-agent runtime not found` | `export CRITIC_AGENT_ROOT=/path/to/critic-agent`, or `git -C "$REPO_ROOT" submodule update --init critic-agent`. |
| `runtime.cli prepare-review/commit-review exited rc=2` | Schema/validation bug (per `critic-agent/AGENTS.md` §Exit codes). Inspect workdir payload; retry with `--critic-mock` while fixing. |
| `runtime.cli ... timed out after 30s` | KB stuck. If `CRITIC_KB_CLIENT_MODE=live`, drop to `inmemory`. Reproducing in `inmemory` is a bug — that path must not block on I/O. |
| All verdicts `('needs_review','critic_unavailable')` + `kb_skipped=missing_critical_context` | Static context load failed. Check `manifest.json` has non-empty `model_name`/`framework`; grep `logs/cli.log` for `critic_agent_backend static_context`. |

### Run-time signals

- `No accelerator` (Magpie): subprocess `PATH` must lead with `$(dirname "$PYTHON")` (or set `MAGPIE_PYTHON`); use `ROCR_VISIBLE_DEVICES`, not `HIP_VISIBLE_DEVICES`.
- Repeated `select_kernels` with unchanged trace/config: bug — reuse `last_select_kernels`.
- `correctness_passed=false`: do not integrate; the kernel-agent report must contain explicit correctness evidence.
- `stop_reason=no_more_leverage`: stop and report; only resume if the user changes workload / search space / model / strategy.
- `stop_reason=policy_loop`: Coordinator hit ≥10 consecutive `policy_denied` events for the same action/rule pair; all top actions may be locked or pruned. Inspect `SharedState.policy_denial_history` and the per-tick `Policy denials` block. To recover: manually edit `state.json` to remove the action from `pruned_families`, clear `policy_denial_streak` / `stop_reason`, and re-propose with fresh `params.grid` content (omit stale `idempotency_key`).
- `stop_reason=time_exhausted`: resume same session (`--resume`); do not start fresh.

## Report Back To User

Report concise status:

- session id (from `manifest.json`) and log path
- `cumulative_gain` and `current_best`
- params accepted/rejected summary
- last kernel optimized, correctness, micro speedup, E2E gain, decision
- whether the process is still running or stopped and why
