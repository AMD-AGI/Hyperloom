<!--
SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: MIT
-->

# AGENTS.md

Single source of truth for anyone (human or AI) working in this repo: **what the
system is** (architecture, commands, the core pipeline) and **how to change it**
(authoring rules of engagement).

- **Workflow & PR process** → [`CONTRIBUTING.md`](CONTRIBUTING.md)
- **Formatting, naming, testing, REUSE conventions** → [`docs/contributing/style-guide.md`](docs/contributing/style-guide.md)
- **What a review will flag** → [`.github/PULL_REQUEST_TEMPLATE.md`](.github/PULL_REQUEST_TEMPLATE.md) and the Copilot instructions under [`.github/`](.github/)

## What this is

Hyperloom is an autonomous agentic system that optimizes end-to-end LLM inference workloads (host code and GPU kernels) on AMD GPUs. A Python Coordinator drives an iterative loop (Think → Decide → Implement → Benchmark) across serving frameworks (SGLang / vLLM / atom), delegating to LLM roles and programmatic handlers. It integrates external tools: Magpie (benchmark/profile collection), TraceLens (trace analysis), GEAK (kernel optimization backend), and AMD Quark (quantization).

The whole package lives under `src/hyperloom/`. The pip distribution name is `hyperloom-inference_optimizer`.

## Common commands

```bash
pip install -e .[test]            # dev + test extras
pip install -e ".[test,ci]"       # mirror the coverage CI job locally

pytest                            # full suite (testpaths = src/hyperloom/**/tests)
pytest src/hyperloom/inference_optimizer/tests
pytest src/hyperloom/inference_optimizer/tests/test_prompt_builder.py -k subset   # single file / subset

ruff check .                      # lint (E, F, W; line-length 120, E501/E741 ignored)
ruff format --check .
mypy src/hyperloom                 # optional/local only
```

## Architecture

The system is a single persistent Coordinator loop plus reactive LLM roles and programmatic handlers.

### The phase machine

The Coordinator advances through a fixed chain (with cyclic reloop while budget/leverage remain):

```
PRELUDE -> FRAMEWORK_AGENT -> EXPLORE -> KERNEL_AGENT -> SWEEP -> CLOSE
```

- **PRELUDE** — `target_analysis`, `baseline`, then `roofline`/`profile`.
- **FRAMEWORK_AGENT** — Coordinator-owned framework enablement (discovery/ranking/audit + authoring specialists + the Rung 0–5 enablement escalation ladder for un-runnable `(model, backend)` combos). `--no-framework-agent` skips it.
- **EXPLORE** — serving-parameter and source-patch search via the unified `explore` ledger + `specialist` + `integrate_patch`.
- **KERNEL_AGENT** — bridge to programmatic kernel work (`kernel_opt`, `integrate`, `gemm_tuning`, `roofline`, `profile`, `recover`); handler kinds like `trace_analyze`/`run_optimization` dispatch inside the request channel.
- **SWEEP** — validates the optimized stack across concurrency / ISL-OSL frontiers.
- **CLOSE** — idempotent drain: `report` then `session_breakdown`.

### Roles (LLM) vs handlers (programmatic)

- **Orchestration** — a single persistent multi-turn conversation across ticks (delta prompts + periodic compaction into `orchestration_memory`, rebuilt from `SharedState` on resume). Decides next actions. Kernel work is **not** LLM-driven — the Coordinator dispatches it to Python handlers.
- **Critic** — reviews proposals/patches before apply. Default `--critic-agent` (drives `hyperloom.agents.critic` runtime); `--critic-mock` for offline.
- **Robustness** — health monitoring, RCA, stall/crash detection. Default `--robustness-agent`; `--robustness-mock` (auto-selected on `--nodes >= 2`).

All writes flow through `emit_intent` → the Coordinator intent handler → Critic → accuracy gate → PolicyGate → runtime state.

### Key package map

| Area | Location |
|---|---|
| CLI (`optimize`/`--resume`, preflight, model gate) | `src/hyperloom/inference_optimizer/cli/` |
| Coordinator main loop | `src/hyperloom/orchestrator/loop/coordinator.py` |
| Phase machine + allowlists | `src/hyperloom/orchestrator/phases/` |
| PolicyGate | `src/hyperloom/orchestrator/policy/gate.py` |
| LLM role backends | `src/hyperloom/orchestrator/roles/` |
| Action executors (baseline, profile, explore, ...) | `src/hyperloom/orchestrator/actions/executors/` |
| Action catalogue (source of truth) | `src/hyperloom/inference_optimizer/protocol/action_surfaces.py` (`ACTION_CATALOGUE`) |
| Run state | `src/hyperloom/orchestrator/state/` (`shared_state.py`, journals) |
| Message bus / GPU pool / leases | `src/hyperloom/orchestrator/bus/` |
| Knowledge / RecipeKB / PR monitor | `src/hyperloom/orchestrator/knowledge/` |
| Specialist dispatch | `src/hyperloom/orchestrator/specialists/` |
| Orchestration prompts | `src/hyperloom/orchestrator/prompts/` |
| Path resolution (single authority) | `src/hyperloom/inference_optimizer/session/paths.py` |
| Sub-agents | `src/hyperloom/agents/` (`critic`, `robustness`, `framework`, `kernel`, `quantization`) |
| Downstream JSON contract producer | `src/hyperloom/inference_optimizer/breakdown/` |
| Cross-cutting helpers | `src/hyperloom/common/` |

### Session artifacts (observable contracts)

The private helper names and prompt wording are **not** contracts; the session artifacts are. Under the per-run session dir: `manifest.json`, `state.json`, `storage/coordinator.db`, `runs/<action>/<task_id>/`, `reports/`, and `session_breakdown.json` (the external dashboard contract, shape documented in `docs/reference/session-breakdown.md`). Prefer reading these over parsing terminal logs. `session/paths.py` is the single authority — do not hand-build session paths.

## The profiling → TraceLens → kernel pipeline

This is the core value chain and the part most worth understanding before touching analyzer or kernel code. Full reference: `docs/conceptual/kernel-execution-path.md`.

### 1. Profile capture (Magpie)

`profile`/`roofline` executors (`orchestrator/actions/executors/profile.py`, `roofline.py`) run the workload through **Magpie**, which drives the framework's benchmark script with profiling enabled and emits a torch/ROCm trace (`*.pt.trace.json.gz`). Magpie relies on **IntelliKit** for the low-level GPU profiling primitives. `roofline` is the preferred composite: it wraps profile + trace-analysis + `analysis.md` publication in one action. Coordinator auto-enqueues an analysis at the end of PRELUDE and at each +10% validated-throughput watermark.

### 2. Trace analysis (TraceLens)

The `trace_analyze` request kind is dispatched by the Coordinator's kernel request channel to a Python handler → `agents/kernel/tools/tracelens_analysis.py`, which runs **TraceLens**. TraceLens is an external repo (`$TRACELENS_ROOT`, cloned + SHA-pinned by `install.sh`). It consumes the trace and produces `tracelens/analysis.md` + `tracelens_report.json` + system/category findings, ranks hot kernels, and derives roofline targets. Results cache in `state.last_trace_analyze`. `roofline_ceiling.py` stamps a roofline ceiling for the report.

### 3. Kernel optimization

Kernel work is **programmatic, never LLM-driven**. Orchestration emits a `request{target_agent:"kernel_agent", kind:...}`; `IntentRouter._handle_request` (`orchestrator/loop/intent_router.py`) intercepts it *before* any agent backend runs and dispatches to a registered handler in `orchestrator/kernel/request_handlers.py`. The RESPONSE is written straight to the bus (`source="programmatic_handler"`) and never sees PolicyGate.

Request kinds: `trace_analyze`, `run_gemm_tuning`, `run_optimization`, `integrate`/`apply_patch`. Only `reusable_native_kernel_ids` surfaced by trace analysis are eligible for `run_optimization`. Backends:

- **GEAK** owns the KERNEL phase by default (`KERNEL_OPT_BACKEND_ORDER=geak`) and picks kernel strategy internally (Triton / HIP / FlyDSL). GEAK RAG/memory caches live outside the session (`~/.cache/amd-ai-devtool/semantic-index/`, `/shared/hyperloom/geak-memory/memory.db`).
- **Forge** is the opt-in per-kernel backend (`KERNEL_OPT_BACKEND_ORDER=forge` exactly). FlyDSL kernels and multi-GPU collective kernels route to Forge when enabled.

### 4. Integration & KEEP gate

`integrate_handler` applies the patch, re-baselines, and decides KEEP/REVERT on E2E Magpie throughput not microbench alone. Discipline: KEEP only when E2E clears threshold; REVERT rejects a patch permanently; NEEDS_REVIEW allows at most 3 E2E attempts. After every KEEP the full stack is revalidated end-to-end so cumulative gain is real, not a sum of per-round deltas. Kernel apply may mutate `/sgl-workspace/{aiter,sglang}` — back up source + compiled `.so`/`.co` before apply; restore artifacts first on REVERT.

Kernel artifacts land under `$USER_DATA_PATH/kernel-agent/runs/<session_id>/` and cross-task GEAK output under `$USER_DATA_PATH/kernel-agent-workspace/<kernel_id>/`.


## Authoring rules of engagement

Rules for changing this code. They complement the conventions above; where a rule
names a boundary, the architecture section is the authority on where that boundary lives.

### One concern per change
A PR fixes one issue or adds one capability. If you must bundle, say why in the PR
description. Split unrelated refactors out; don't ride them in on a fix unless unavoidable.

### Size budget
Prefer reviewable diffs. A large diff is a signal to stop and split, not to push
harder. Cleanup in an unrelated file is a separate PR.

### Fix upstream, not around it
Hyperloom is includes over components (GEAK, Magpie, TraceLens and the frameworks). When the root 
cause is in one of those, suggest fix it there and pin the fix — do not paper over it with a local workaround.

### Don't Grow Technical Debt
- **No new broad `except Exception` / bare `except`.** Catch the specific error, or let it
  raise. There is a backlog being ratcheted down; don't add to it.
- **No new feature flags / env toggles** to route around a design problem. A flag is a
  decision deferred; make the decision.

### Delete, don't comment out
Dead code goes. Version control is the archive. Commented-out blocks and `# removed …`
tombstones rot and mislead.

### Clean design preferences
- One boundary rule per concern, owned by one module.
- Derive over hardcode: prefer a single computed source over duplicated constants.
- Minimal typed interfaces; push validation to the system boundary, trust internal callers.

### New framework or platform
Building a new framework (e.g. AgentX) or platform (e.g. world models)? Work bring-up
through the owning-component leads **before** integration code lands — see
[`CONTRIBUTING.md`](CONTRIBUTING.md) § *Proposing a new framework or platform*.
