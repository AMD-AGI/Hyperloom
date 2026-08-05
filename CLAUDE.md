# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Hyperloom is an agentic system that autonomously optimizes LLM inference on AMD GPUs (MI300X/MI325X/MI355X). It treats optimization as a search problem: given a workload, it explores candidate optimizations one change at a time — backend swaps, server parameters, GEMM tuning, kernel rewrites, parallelism configs — always benchmarking against the real workload before keeping a change. The single user-facing entry point is the `inference_optimizer` skill/CLI.

## Commands

Install (editable) with test + coverage extras — mirrors the CI coverage job:

```bash
pip install -e ".[test,ci]"
# When OOB/ is present in the clone, also: pip install -e "./OOB"
```

Testing (pytest, `asyncio_mode = auto`):

```bash
pytest                                              # full suite (testpaths in pyproject.toml)
pytest inference_optimizer/tests                    # one tree
pytest inference_optimizer/tests/test_prompt_builder.py -k subset   # single file / filtered
pytest -m "not critic_agent_e2e and not robustness_agent_e2e"       # skip e2e that shell out to sub-agent runtimes
```

`testpaths` spans multiple packages (`inference_optimizer/tests`, `robustness-agent/tests`, `critic-agent/runtime/tests`, `quantization_agent/tests`, selected `kernel-agent/tools/test_*.py`, and `ci`). `pythonpath` includes `.`, `ci`, `robustness-agent/src`, `critic-agent`, `framework-agent/src`, so imports resolve without installing every sub-package.

Lint / format / type (Ruff pins `E`,`F`,`W`; line-length 120; `E501`/`E741` ignored):

```bash
ruff check .
ruff format --check .
mypy inference_optimizer kernel-agent robustness-agent
pre-commit run --all-files    # ruff, bandit, shellcheck, yamllint, actionlint
```

Coverage note: the authoritative UT coverage is only `.github/workflows/tests-coverage.yml` (config in `pyproject.toml` `[tool.coverage.*]` and `[tool.hyperloom.tests_coverage]`). Do not treat ad-hoc `pytest --cov` runs as the headline metric. Doc-only changes (`**/*.md`, `docs/**`, `LICENSE*`, etc.) skip CI via `paths-ignore` — keep that list in sync when adding workflows.

Running the optimizer (normally launched via the skill, not by hand):

```bash
inference_optimizer optimize --model <path> --framework sglang --gpu-type MI300X ...
inference_optimizer optimize --resume     # requires manifest.json + state.json in the session dir
```

## Runtime architecture (the big picture)

Hyperloom is a **single-mode, 3-role agent runtime** driven by a Python **Coordinator**. The roles are:

- **Orchestration** — decides the next action. Runs as a *single persistent multi-turn conversation* that continues across coordinator ticks (not a fresh stateless call each tick). First turn gets a full state seed; later turns get only a delta plus new inbox events, and pull more via read-only context tools. Periodically checkpointed/compacted into `state.json` (`orchestration_memory`) and re-seeded to bound context. The EXPLORE-phase orchestration is published as "Arbor."
- **Critic** — reviews proposals/patches before apply (`critic-agent/` subprocess runtime).
- **Robustness** — health monitoring, RCA, scheduling-police (`robustness-agent/` subprocess runtime). Auto-downgrades to mock on multi-node (`--nodes >= 2`).

Critic and Robustness are reactive/stateless per tick; only Orchestration is stateful.

**Kernel work is not an LLM role.** Every kernel `REQUEST` emitted by Orchestration is intercepted inline by the Coordinator's `IntentRouter` and routed to a registered Python handler (`orchestrator/kernel/request_handlers.py`). No LLM turn is consumed. See `docs/conceptual/kernel-execution-path.md` for the full dispatch flow.

**Phase chain** (Coordinator advances monotonically; `phase_state.PHASE_ALLOWED_ACTIONS` + `PolicyGate` enforce which actions are legal per phase):

```
PRELUDE -> FRAMEWORK -> EXPLORE -> KERNEL -> SWEEP -> CLOSE
```

- **PRELUDE**: `target_analysis` → `baseline` → `roofline`/`profile`.
- **FRAMEWORK**: Coordinator-managed framework-agent candidate discovery/apply (Critic-gated). No separate LLM framework role.
- **EXPLORE**: searches config + source-patch levers via the `explore` ledger — `explore` (server-arg/env variants), `specialist` (unified research/patch sub-agent with `scope`/`mode`/`bench`/`lane` dials, incl. `scope=freeform`), `integrate_patch`. After each KEEP the full stack is revalidated end-to-end.
- **KERNEL**: bridge to kernel-agent; Coordinator owns request handlers + safety gates.
- **SWEEP**: concurrency and ISL/OSL frontier checks (`conc_sweep` optional).
- **CLOSE**: `report` → `session_breakdown` → CLI finally-block safety-net breakdown. Must be idempotent (sessions can end via phase transition, deadline, interrupt, or resume).

**Write path**: every write action flows through `emit_intent` → the Coordinator's intent handler, so Critic review, the accuracy gate, Robustness escalation, and PolicyGate invariants (path sandbox, resource leases, phase ordering, data dependencies, single-writer) always apply.

Retired action names that must NOT appear as live positive instructions: `setup`, `classify`, `backends`, `params`, `validate_stack`, `select_kernels` (they survive only in migration readers, archived breakdown aliases, or rejection tests).

## Key directories

- `inference_optimizer/` — the sole skill and CLI. `cli*.py` (entry `inference_optimizer.cli:main`), `orchestrator/` (Coordinator, agent roles, `action_executors/`, `backends/` Claude/Codex/Critic adapters, `system_prompts/`), `actions/_meta/` (action metadata + scheduling policy), `breakdown/` (produces `session_breakdown.json`), `baseline_comparison/`, `kb/` + `recipe_kb/` (RecipeKB durable lessons), `multi_node/`, `SKILL.md` (the agent's operating instructions — source of truth for the protocol).
- `src/hyperloom/agents/kernel/` — kernel tool scripts (TraceLens/GEAK/OOB); `tools/` + `tools/backends/` (Ray-scheduled GEAK/OOB submission). Installed via `scripts/install.sh`; resolved at runtime via `HYPERLOOM_KERNEL_AGENT_ROOT`.
- `critic-agent/`, `robustness-agent/`, `quantization_agent/` — subprocess sub-agent runtimes with their own `tests/`.
- `ci/` — inference-optimization CI pipeline (PR submitter, AB test, matrix generation, session-summary transform). On the import path and collected by pytest.
- `docs/` — architecture docs and case studies. Start with `docs/conceptual/optimization-loop.md` and `docs/reference/environment-variables.md`.

## Session artifacts (the real contracts)

Private helper names and prompt wording are *not* contracts; the observable session artifacts and subprocess JSON bridges are. A run writes into a **session directory** under `$USER_DATA_PATH` (workspace root, default `/workspace/hyperloom`):

- `manifest.json`, `state.json`, `storage/coordinator.db`
- `runs/<action>/<task_id>/`, `agents/{orchestration,kernel,critic,robustness}/`, `reports/`
- `session_breakdown.json` — the stable downstream contract (consumers: `claw-stats-service`, dashboards). Producer lives in `inference_optimizer/breakdown/`; shape documented in `docs/INTEGRATION_SESSION_BREAKDOWN.md`.

## Environment

Config is env-driven (`.env` from `.env.template`; full list in `docs/CONFIGURATION_REFERENCE.md`). Key vars: `SAFE_API_KEY` + `OPENAI_BASE_URL` (LLM gateway; GEAK and OOB claude/codex inherit these automatically), `USER_DATA_PATH` (runtime dir for deps/logs/state/results — NOT the source dir), optional `TRACELENS_INTERNAL_ROOT` (internal roofline extension; unset = open-source-only report), `QUARK_ROOT` (only for the `--quantize` prelude), `CURSOR_API_KEY` (optional Cursor kernel-opt backend). Local Mode bootstrap: `bash inference_optimizer/scripts/local_setup.sh`.
