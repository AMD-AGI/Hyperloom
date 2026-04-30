# Inference Optimizer — Cursor Skill

Cursor entrypoint for the standalone `inference_optimizer` Python package: an
autonomous multi-agent LLM-inference optimization runtime targeting AMD MI355X
GPUs.

For the full launch procedure (env vars, preflight, CLI invocation, transport
modes), see [`SKILL.md`](./SKILL.md). For a concrete prompt that invokes the
optimizer end-to-end, see [`prompt_sample.MD`](./prompt_sample.MD).

## v0.4 MVP roster (current)

Four Claude-backed reactor agents, all wired through one `Conductor` + a
SQLite-backed `MessageBus`:

| Agent      | Mode coverage                       | Always-on | Role |
| ---------- | ----------------------------------- | --------- | ---- |
| `executor` | quick / guided / marathon           | —         | sole proposer + delegator |
| `critic`   | guided / marathon                   | —         | KEEP/REVERT verdicts via `send_message` |
| `triage`   | quick / guided / marathon (tick=60s)| ✅        | cross-layer health watcher; only role allowed to emit `kill_task` |
| `kernel`   | guided / marathon                   | —         | Plan-A `kernel_opt` / `integrate` responder (executor → kernel REQUEST/RESPONSE) |

Parliament mode, Sage role, and the Codex backend were removed in v0.4 — see
`src/inference_optimizer/docs/standalone_agent_design.md` §13 for the design
contract and decision log.

## `src/inference_optimizer/` layout (briefly)

```text
src/inference_optimizer/
├── __main__.py                 module entrypoint (`python -m inference_optimizer`)
├── cli.py                      CLI flag parsing → Conductor wiring
├── paths.py                    asset-root resolution (env-overridable)
├── requirements.txt            runtime deps (claude-agent-sdk, openai, PyYAML, …)
│
├── orchestrator/               core runtime — single owner of the run
│   ├── conductor.py              Conductor: bootstraps reactors + clock + dispatcher
│   ├── agent_role.py             4-role registry (executor/critic/triage/kernel)
│   ├── policy.py                 PolicyGate: intent allow-list + KILL_TASK source guard
│   ├── intent_parser.py          envelope schema + 11 IntentTypes (incl. KILL_TASK)
│   ├── message_bus.py            SQLite events bus + topic allowlist
│   ├── feature_flags.py          per-mode flag matrix (quick / guided / marathon)
│   ├── execution_mode.py         mode selection from MAX_HOURS
│   ├── shared_state.py           SharedState (CORE_STATE_FIELDS guarded)
│   ├── task_registry.py          DelegatedTask state machine (queued→cancelled handled)
│   ├── resource_lock.py          SQLite lease backend for resource lanes
│   ├── cursor_store.py           per-agent inbox cursor persistence
│   ├── checkpoint.py             snapshot + LegacySessionRejected resume guard
│   ├── sub_agent_runner.py       ephemeral OOB sub-agent runner
│   ├── action_registry.py        action catalogue loader (markdown + YAML metadata)
│   ├── action_executors/         per-action ActionExecutor classes (baseline,
│   │                             bench_runner, param_sweep_run, profile)
│   ├── backends/                 Backend interface + Mock/Claude/Codex impls
│   ├── multi_cli/                router · launcher · agent_card · envelope · mock_agent
│   ├── system_prompts/           canonical role prompts (executor/critic/triage/kernel)
│   ├── scheduler.py              BudgetAwareScheduler (action scoring)
│   ├── iron_rules.py             IR-1..IR-7 prompt block
│   ├── kb.py / persona.py        KB + persona helpers (KB stub in v0.4)
│   ├── brier.py                  prediction-quality bookkeeping
│   └── …                         accuracy_gate, early_stop, env_probe, etc.
│
├── agents/                     per-agent skill bundles (loaded by multi-cli launcher)
│   ├── PROTOCOL.md               wire schema + Bash recipe (shared by every agent)
│   ├── executor/                 SKILL.md + actions/ + reference/ + scripts/
│   ├── critic/                   agent_card.yaml + system_prompt.md
│   ├── triage/                   agent_card.yaml + system_prompt.md + scripts/
│   └── kernel/                   SKILL.md + actions/ + reference/ + scripts/
│
├── actions/                    action prompts + YAML metadata (read by ActionRegistry)
│
├── scripts/                    GPU + benchmark shell helpers
│   ├── preflight.sh / bootstrap.sh    pre-launch environment validation
│   ├── run_baseline.sh / executor.sh  baseline / sweep wrappers
│   ├── eval_accuracy.sh               GSM8K accuracy gate
│   ├── run_profile.sh / trace_action.py   profiling helpers
│   ├── geak_ray_submit.py / oob_ray_submit.py    kernel-opt Ray submitters
│   ├── patch_inductor.py              Inductor flag patcher
│   ├── monitor.sh / monitor.py        live event_log tailer
│   └── multi_cli_smoke.sh             4-pane multi-cli dry-run script
│
├── bootstrap/                  CLI/SDK install + env probe + orchestrator boot
├── storage/                    SQLite connection + schema + backup helpers
├── kb/                         KB schema + jsonl stores + ingest/query (stub in v0.4)
├── docs/                       design + checklist + KB
│   ├── standalone_agent_design.md     ★ the v0.4 SoT (§13 = MVP plan + decisions)
│   ├── inference-optimizer-DESIGN.md  full system design
│   ├── IMPLEMENTATION-CHECKLIST.md    phase-by-phase implementation log
│   └── KNOWLEDGE-BASE.md              KB architecture
└── tests/                      pytest suite (822 tests; e2e/ has subprocess + dry-run)
```

## Key entrypoints by intent

| You want to…                                  | Read / run                                   |
| --------------------------------------------- | -------------------------------------------- |
| Launch the optimizer end-to-end               | [`SKILL.md`](./SKILL.md) — Launch Procedure  |
| Understand the agent contract                 | `src/inference_optimizer/docs/standalone_agent_design.md` |
| Add a new action                              | `src/inference_optimizer/actions/README.md` + `action_registry.py` |
| Add / modify an agent                         | `src/inference_optimizer/agents/<name>/agent_card.yaml` + `agent_role.py` |
| Trace what an agent saw / emitted             | `$SESSION_DIR/agents/<name>/{inbox,outbox}.jsonl` (or `monitor.sh`) |
| Run the test suite                            | `pytest src/inference_optimizer/tests/`      |
| Smoke-test multi-cli (4 panes, no GPU)        | `bash src/inference_optimizer/scripts/multi_cli_smoke.sh` |

## Status

- **v0.4 MVP** — 4-Claude-agent roster, `kill_task` triage power, parliament
  removed, Sage merged into Critic placeholder. Pending real GPU shake-down run.
- Full design SoT: `src/inference_optimizer/docs/standalone_agent_design.md` §13.
- Test coverage: 822 tests passing (`pytest -q src/inference_optimizer/tests/`).
