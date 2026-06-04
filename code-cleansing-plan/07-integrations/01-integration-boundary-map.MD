# Integration Boundary Map

## Core Principle

This cleansing effort exists to sharply reduce the engineering and logical complexity of the codebase. The cleanup only needs to preserve end-to-end behavior; it does not need to preserve every unit-level behavior, helper shape, historical test assertion, or internal implementation detail. The codebase is already large enough that code should be cut aggressively when it does not protect a real end-to-end capability or external contract.

## Purpose

Record the single live entry per peripheral integration so later cleanup keeps boundaries clean and does not delete a live subprocess/JSON/artifact bridge.

This is a cleanup guide, not a second source of truth. Runtime source remains the code.

## Kernel Integration

- Single entry: `orchestrator/kernel_request_handlers.py` (`KERNEL_REQUEST_HANDLERS`).
- Live request kinds: `trace_analyze`, `run_gemm_tuning`, `run_optimization`, `integrate`, `apply_patch`.
- Main-package responsibility: route kernel requests, validate shape/path, enforce safety gates, read kernel-agent artifacts, update state.
- kernel-agent responsibility: the actual optimization tools under `kernel-agent/tools/`.
- Contract surfaces: request payload JSON, tool stdout JSON, artifact paths under `kernel-agent/runs/<sid>/` and `kernel-agent-workspace/<kid>/`.
- Keep: handler registry, shape/path validation, safety gates, artifact readers.

## Critic Integration

- Single entry: `orchestrator/backends/critic_agent.py` (prepare-review / commit-review subprocess bridge).
- Main-package responsibility: build the proposal packet, run prepare/commit, validate the verdict envelope, degrade on missing context.
- critic-agent responsibility: the review bundle and model judgment.
- Contract surfaces: `request.json` / `judge_bundle.json` / `review.json` / `emit.json`, `review_verdict` envelope.
- Modes: `mock`, `agent` (CLI `_VALID_CRITIC_BACKENDS`). Default `agent`.
- Removed: legacy `codex_bare` direct chat-completion critic path. Use
  `--critic-agent` for production or `--critic-mock` for offline smoke.

## Robustness Integration

- Single entry: `orchestrator/backends/robustness_agent.py` (tick subprocess bridge).
- Contract mirror: `robustness-agent/src/robustness_agent/role/envelope.py` mirrors the intent vocabulary without importing the main package; contract tests keep them in sync.
- Main-package responsibility: send the Coordinator prompt/state, validate the emitted intent envelope, enforce robustness-only intents through PolicyGate.
- Multi-node: CLI auto-downgrades to `--robustness-mock` when `--nodes >= 2` without a robustness-server.
- Decision pending: robustness legacy mode (drop if production uses reactor mode only).

## Framework Integration

- Single entry: `orchestrator/framework_agent_client.py` (`phase_discover`); only `fa phase-discover` is wired into the Coordinator.
- Executor: `orchestrator/action_executors/framework_pr.py` applies the diff, benchmarks, KEEP/REVERT.
- Main-package responsibility: Coordinator-owned FRAMEWORK_PR phase — discovery call, Critic gate, apply, benchmark, rollback.
- framework-agent responsibility: PR candidate discovery.
- Not live: `fa candidates` / `fa explore` (standalone legacy tools) and `fa agent` / 5th-role (future, not wired). `framework_agent_client` already documents that `phase_fetch` / `phase_emit_proposal` shims were removed.
- Corrected in this step: `install.sh` no longer calls the framework path a "bandit arm"; it is the FRAMEWORK_PR phase.

## Benchmark Integration (Magpie)

- Entry: action executors materialize config and parse results; `_grid_runner.py` shared helpers; `_subprocess_kill.py` for timeout/kill.
- Main-package responsibility: config materialization, result parsing, timeout/kill/cleanup.
- Magpie responsibility: actually running the benchmark.
- Note: `benches/` registry is a disabled scaffold (`BENCH_TOOL_ENABLED_V1 = False`); not a live integration today.

## TraceLens / InferenceX

- TraceLens: reached via kernel-agent tools (trace analysis); not a separate main-package bridge.
- InferenceX: `baseline_comparison/` target analysis client; read-only reference fetch with a `no_target_gpu_configured` marker fallback.

## Multi-node Integration

- Entry: `multi_node/cli.py` (RayJob create/bootstrap), CLI provisioning hooks.
- Main-package responsibility: provision RayJob; action executors read unified env + session artifacts (e.g. shared profile trace dir).
- Backup artifacts already removed in Step 02.

## Single-Entry Summary

| Integration | Single live entry | Peripheral owner |
|---|---|---|
| Kernel | `kernel_request_handlers.KERNEL_REQUEST_HANDLERS` | kernel-agent tools |
| Critic | `backends/critic_agent.py` prepare/commit | critic-agent runtime |
| Robustness | `backends/robustness_agent.py` tick | robustness-agent runtime |
| Framework | `framework_agent_client.phase_discover` | framework-agent `fa` |
| Benchmark | action executors + `_grid_runner` | Magpie |
| Multi-node | `multi_node/cli.py` | SaFE RayJob / Ray |
