# KB Design

> Working design document for upcoming modifications to `./inference_optimizer`.
> Each chapter captures one cohesive layer of the design conversation.

---

## 1. current_overview

This chapter records my reading of the existing `inference_optimizer/` codebase
(version `0.6.0`, see `inference_optimizer/__init__.py`) so that subsequent
modification chapters have a stable reference point.

### 1.1 Mission & Boundaries

`inference_optimizer` is a **multi-agent, LLM-driven runtime** that drives an
end-to-end optimization loop for LLM inference serving on AMD GPUs (MI300X /
MI325X / MI355X) under SGLang or vLLM. It is invoked via a single CLI
(`inference_optimizer optimize …`, see `cli.py`) and runs until a target gain /
target throughput / wall-clock budget is reached.

It does **not** itself decide what to do — instead it hosts a Coordinator that
brokers messages between four persistent agent reactors and a registry of
deterministic Python "action executors" that wrap shell tools (Magpie /
TraceLens / GEAK / OOB / kernel-agent). The Coordinator is a *protocol manager*,
not a strategist.

Out of scope for this package (delegated to sibling repositories — all live next
to it under the same repo root):

- `kernel-agent/` — TraceLens analysis, GEAK kernel synthesis, OOB CLI, patch
  applier; `inference_optimizer` calls it via `KERNEL_REQUEST_HANDLERS`
  programmatic handlers (`orchestrator/kernel_request_handlers.py`) plus
  Hyperloom CLI subprocesses.
- `critic-agent/` — KB-aware proposal review skill; driven by
  `CriticAgentBackend` over `python -m runtime.cli prepare-review` /
  `commit-review` (see `orchestrator/backends/critic_agent.py`).
- `robustness-agent/` — health-monitoring / RCA / scheduling-police skill;
  driven by `RobustnessAgentBackend` over `python -m
  robustness_agent.runtime.cli`.

### 1.2 Top-Level Layout

```
inference_optimizer/
├── __init__.py                  # version + role roster docstring
├── SKILL.md                     # operator-facing launch + monitoring playbook
├── cli.py                       # entry point (~2.3 kLOC)
├── manifest.py                  # session resume tag (manifest.json)
├── paths.py                     # session/asset path resolution
├── session_paths.py             # per-action workspace path helpers
├── actions/                     # action metadata + agent-facing playbooks
│   ├── _meta/<name>.yaml        # declarative ActionMetadata source
│   ├── pmc_roofline.md          # standalone in-flight playbooks
│   ├── recover.md
│   └── validate_stack.md
├── baseline_comparison/         # InferenceX external-baseline fetcher
├── breakdown/                   # session_breakdown.json exporter (~2.6 kLOC collectors)
├── orchestrator/                # protocol layer, the heart of the package
│   ├── action_executors/        # 18 concrete executors (baseline, profile, …)
│   ├── backends/                # LLM transports (Claude / Codex / Critic / Robustness / mocks)
│   ├── system_prompts/          # rules fragments + prompt_builder.py
│   ├── coordinator.py           # ~3.2 kLOC main loop / dispatcher / promotion logic
│   ├── shared_state.py          # ~1.9 kLOC state.json schema + helpers
│   ├── kernel_request_handlers.py # ~1.2 kLOC programmatic kernel responders
│   ├── policy.py                # PolicyGate (intent validation chokepoint)
│   ├── action_registry.py       # loads _meta/*.yaml into ActionMetadata
│   ├── scoring.py               # action prior + scoreboard
│   ├── intent_parser.py         # IntentEnvelope schema (Claude tool / Codex JSON)
│   ├── message_bus.py           # SQLite-backed events table
│   ├── resource_lock.py         # 4-lane lease manager
│   ├── task_registry.py         # DelegatedTask state machine
│   ├── cursor_store.py          # per-agent last_processed_seq
│   ├── sub_agent_runner.py      # delegate → executor dispatcher
│   ├── objective.py             # gain / tput / baseline / time-only objectives
│   ├── roofline_integration.py  # ~36 kLOC PMC roofline analysis
│   └── …                        # framework_paths / pmc_workload_params / kb_digest …
├── storage/                     # SQLite schema + connection wrapper
├── scripts/                     # install.sh, configs/*.yaml, dump_*.py
├── examples/                    # P0–P2 demo scripts
└── tests/                       # ~50 test files, mostly Pn_* milestone-driven
```

### 1.3 Runtime Topology — the 4-agent v0.6 design

Per `orchestrator/agent_role.py`, every run instantiates a fixed 4-agent
roster (no mode gating; ADR-34 collapsed the previous quick/full split):

| Role | Backend | Allowed intents (high-level) | Notes |
|---|---|---|---|
| **orchestration** | Claude (tool-using) | `propose_action` / `delegate` / `request` / `update_state` / send/ask/answer/alert/persona | Only role with REQUEST authority (target=`kernel`) |
| **kernel** | Claude (or Codex via `--kernel-codex`) | `response` only / send / alert / `update_state` for own metrics | Responder-only; owns 5 deep-kernel actions (`kernel_opt`, `integrate`, `deep_kernel_analysis`, `operator_tuning`, `vendor_kernel_config`) |
| **critic** | Codex no-tools (default external critic-agent skill) | `review_verdict` only + base intents | Devil's-advocate review of every proposal; owns KB |
| **robustness** | Claude (or external robustness-agent skill) | `kill_task` / `force_dispatch` / `prune_branch` / `escalate_strategy_change` + base | Always-on tick; cross-layer health watcher + scheduling police |

Each role has a system prompt assembled per run. For Orchestration / Critic the
prompt is built from typed inputs (`system_prompts/prompt_builder.py` /
`critic_prompt_builder.py`) by combining the rules fragment in `*.md` with the
ActionRegistry catalogue. For Kernel / Robustness the prompt is the markdown
file plus a small CLI-side header. Effective prompts are snapshotted to
`agents/<role>/system_prompt.snapshot.md`.

### 1.4 Communication & Persistence Substrate

All inter-agent communication is event-sourced through a single SQLite WAL
database at `<session_dir>/storage/coordinator.db` (`storage/schema.py`):

| Table | Purpose |
|---|---|
| `events` | A2A message bus, AUTOINCREMENT `seq` provides monotonic order |
| `cursors` | Per-agent `last_processed_seq` for idempotent replay across resume |
| `tasks` | DelegatedTask state machine (`queued → running → succeeded/failed/cancelled/needs_manual_review`); UNIQUE `idempotency_key` |
| `leases` | 4 mutex lanes — `server_lifecycle` / `workspace_mutation` / `benchmark_lane` / `profile_lane`, with documented cross-lane conflicts |

Persistent **session state** (separate from the event log) lives in
`<session_dir>/state.json` (`SharedState` dataclass in
`orchestrator/shared_state.py`). It is the only writer-Coordinator-owned
mutable summary: baseline tput / accuracy, `current_best`, `cumulative_gain`,
optimization_stack, ledgers for backends / params / sweep, audit attempts,
action_scores, kernel-opt history, etc. Writes are atomic (`tmp` + `os.replace`).

Two transport layers feed the bus:

1. **Claude transport** — `emit_intent` MCP tool (`mcp_emit_intent.py` +
   `intent_parser.EMIT_INTENT_TOOL_SCHEMA`). Each call carries one Intent.
2. **Codex transport** — `validated_json_output`: a single JSON object whose
   `intents[]` array is parsed by `parse_codex_validated_json`.

Both flow through `validate_envelope` → typed `Intent` list → **PolicyGate**
(`orchestrator/policy.py`), the single intent-validation chokepoint:

- Role permission (does this role allow this `intent_type`?).
- Source allowlists for verdicts / scheduling-police / `kill_task`.
- REQUEST routing (only `orchestration → kernel` in v0.6).
- Kernel ownership: 5 KERNEL_OWNED_ACTIONS cannot be `delegate`d — must be
  `request{target_agent="kernel", kind=…}`.
- Core SharedState field guard (only Coordinator may mutate
  `current_best`, `stop_reason`, `baseline_tput`, …).
- Path containment (`PATH_LIKE_FIELDS` must resolve under SESSION_DIR or, for
  `source_file`, under the framework source allowlist `/sgl-workspace/{aiter,sglang,vllm}/`).

A denial fires a `policy_denied` observation, which lands in the offending
agent's inbox so the LLM self-corrects on the next reactor turn.

### 1.5 Action Registry & Executors

Actions are declared in `actions/_meta/<name>.yaml` and loaded by
`orchestrator/action_registry.ActionRegistry` into immutable `ActionMetadata`
records. Schema (DESIGN §16.2) covers `family`, `cost_minutes_p50/p75`,
`expected_gain_pct`, `accuracy_risk`, `crash_risk`, `requires_lanes`,
`allowed_tools`, `side_effects`, `preferred_backend/model`, `lease_ttl_sec`, plus
prompt-builder hints (`description`, `pipeline_phase`, `typical_runtime_min`).

Currently registered actions (21 yaml files):

| Pipeline phase | Actions |
|---|---|
| prep | `target_analysis`, `baseline` |
| measure | `baseline` (gates explore) |
| analysis | `profile`, `pmc_roofline`, `deep_kernel_analysis` |
| explore | `backends`, `params`, `sweep` |
| deep (kernel-owned) | `kernel_opt`, `integrate`, `operator_tuning`, `vendor_kernel_config` |
| validate | `validate_stack` |
| finalize | `report` |
| support | `recover`, `session_breakdown`, `dream`, `re_explore`, `comm_optimization`, `compiler_tuning` |

Real Python executors live in `orchestrator/action_executors/` and are wired
into `SubAgentRunner.executor_registry` by `cli._register_executors`. The
`_REAL_EXECUTORS_FULL` map shows which actions have actually-implemented
executors today; `dream` / `re_explore` / `comm_optimization` / `compiler_tuning`
are still `_noop_prep` stubs. Per-task workspaces are auto-created at
`runs/<kind>/<task_id>/`.

Two recurring patterns in executors:

- **Magpie wrapper** — `BaselineExecutor` / `ProfileExecutor` /
  `BackendsExecutor` / `ParamsExecutor` / `SweepExecutor` /
  `ValidateStackExecutor` materialize a YAML config (with workload envs from
  `_workload_envs.py`), spawn `subprocess.run(["magpie", …])`, parse
  `benchmark_report.json`, and return a structured measurement. They share a
  `_grid_runner.py` for variant fan-out and `benchmark_result.py` for parsing,
  including an mtime-gated **leak-path salvage** (`harvest_leaked_artifacts`)
  for legacy Magpie scripts that hardcode `/workspace/inferencex_result.json`.
- **External-comparison wrapper** — `TargetAnalysisExecutor` calls
  `baseline_comparison.target_analyzer.analyze`, which fetches public
  InferenceX rows over HTTPS to write `target_analysis/target_baseline.json`.
  This is the only network egress the optimizer makes by default.

### 1.6 Coordinator Main Loop

`orchestrator/coordinator.py:Coordinator` (~3.2 kLOC) is the central piece.
Responsibilities, in roughly the order they fire each tick:

1. **Reactor pump** — for each role in canonical order
   (`orchestration → kernel → critic → robustness`), build a per-tick prompt
   (`_compose_prompt`), call `backend.run(...)`, validate the returned intents
   through PolicyGate, and persist them onto the bus.
2. **REQUEST routing** — if Orchestration emitted `request{target_agent="kernel",
   kind=…}` and `kind` is in `KERNEL_REQUEST_HANDLERS`, run the deterministic
   handler (e.g. `select_kernels`, `run_optimization`, `integrate`, `apply_patch`)
   instead of routing to the Kernel LLM. Otherwise mirror to Kernel's inbox.
3. **Critic Review gate (§18)** — every `propose_action` becomes a
   `PendingProposal`; Critic must emit a matching `review_verdict` before the
   proposal is materialized. `verdict ∈ {approve, reject, redirect, advise,
   needs_review}`.
4. **Dispatcher** — `_pump_dispatcher_once` pulls queued tasks, runs
   `SubAgentRunner.run_task(task)` (which acquires lanes, calls the executor,
   and transitions the task), then routes the result through
   `_promote_to_shared_state` / `_handle_unpromotable_result` to update
   SharedState (current_best, optimization_stack, audit attempts, action_scores,
   …).
5. **Robustness scheduling-police** — applies `kill_task`, `prune_branch`,
   `force_dispatch`, `escalate_strategy_change`.
6. **Stop check** — `objective.reached(state)` /
   wall-clock deadline / `stop_reason` / max-tick guard.

The Coordinator is also the **only** writer of `CORE_STATE_FIELDS`
(`current_best`, `stop_reason`, `cumulative_gain`, `baseline_tput`,
`baseline_accuracy`, `session_id`, `model_path`, `model_class`, …). Its
promotion rules carry significant policy:

- 1-shot KEEP threshold of **0.2 %** gain over current_best for grid actions
  (`backends` / `params`), with an **accuracy gate** for high-precision-risk
  variants (GSM8K drop ≤ 5 %).
- **Cross-round consistent-winner** path (≥ 2 of last 3 rounds, ≥ 0.1 % avg
  gain) so genuinely small but real gains can stack up.
- A **baseline self-loop guard** (`_BASELINE_FINGERPRINT_KEYS`,
  `_BASELINE_SELF_LOOP_THRESHOLD = 2`): two failed baselines with identical
  fingerprints + a third proposal with that same fingerprint → PolicyDenied
  with hint pointing at next override surface.
- Per-action **streak-aware scoring** (`scoring.STREAK_THRESHOLD = 3`,
  `STREAK_PENALTY_MULT = 0.85`, `STREAK_PENALTY_FLOOR = 0.2`) replaces
  hard cooldowns; Marathon priors per `model_class` seed initial scores.

### 1.7 Filesystem & Session Layout

A run produces one fixed, flat session directory (no `session_id` subdir).
Resolution order in `paths.session_dir()`: `$USER_DATA_PATH` →
`/workspace/hyperloom`. `make_session_dir()` materializes the full skeleton
declared in `_SESSION_SKELETON`:

```
<session_dir>/
├── manifest.json                         # written first; resume tag
├── state.json                            # SharedState
├── storage/coordinator.db                # SQLite WAL
├── agents/{orchestration,kernel,critic,robustness}/
│   ├── inbox.jsonl  outbox.jsonl
│   ├── persona.md   system_prompt.snapshot.md
├── personas/  checkpoints/  findings/  kb/
├── runs/<action>/<task_id>/              # executor outputs
├── kernel-agent-workspace/<kernel_id>/   # cross-task GEAK/OOB artefacts
├── kernel-agent/                         # CLI tool runs (one tree per invocation)
├── patches/<kernel_id>/                  # KEEP'd patches + backup
├── optimizer_runs/                       # launcher stdout / pid / monitor logs
├── runtime/                              # pod-local writable runtime
│   ├── kernel-agent.env.sh               # generated; sourced by every CLI invocation
│   ├── geak-config/local.yaml            # generated GEAK litellm config
│   ├── Magpie/                           # cloned by install.sh
│   └── source-mirrors/{geak,OOB,TraceLens-internal}/
├── reports/                              # `report` action output
└── logs/                                 # cli + reactor + auth-proxy logs
```

Read-only sources stay outside `$USER_DATA_PATH`: `$TRACELENS_ROOT`,
`$OOB_SRC` / `$HYPERLOOM_BUNDLE`, `/sgl-workspace/{aiter,sglang,vllm}/`,
`~/.claude/config.json`, `~/.codex/auth.json`, `~/.cache/amd-ai-devtool/`,
`/wekafs/hyperloom/geak-memory/memory.db`. Each is overridable via env.

### 1.8 CLI Entry & Setup Discipline

`cli.py` exposes one subcommand (`optimize`) and orchestrates an opinionated
preflight before the Coordinator boots:

- **Iron Rules** (encoded in `SKILL.md`): IR-1 (GPU unoccupied before launch)
  and IR-2 (`scripts/install.sh` must succeed before launch) are operator
  responsibilities; the in-loop equivalent is `_preflight()` steps 1–12 inside
  `cli.py` (auth aliases, SDK auto-install, OOB auth-proxy bootstrap, ROCm
  hygiene, gpu-type autodetect, model-allowlist gate, critic-agent runtime
  probe, …).
- **Hard model gate**: only `claude-opus-4-7` (preferred) or `claude-opus-4-6`
  (fallback) are accepted (`_CLAUDE_ALLOWED_MODELS`); the preflight probes
  `GET <OPENAI_BASE_URL>/models` with retries before allowing the run.
- **Backend selection**: `_build_backends` picks per-role backend instances
  (`ClaudeBackend`, `CodexBackend`, `CriticAgentBackend`, `MockCriticBackend`,
  `RobustnessAgentBackend`, `MockRobustnessBackend`). Defaults are
  `--critic-agent` and `--robustness-agent`; mocks are reserved for offline
  smoke tests. A `--no-kernel` switch removes the kernel role from the run
  registry while leaving the install layer intact.
- **Session metadata via flags** (`--model`, `--framework`,
  `--gpu-type`, `--model-class`, `--compare-against-gpu`); the previously
  in-loop `setup` / `classify` actions were deleted so the SKILL caller
  supplies these directly. `target_analysis` is *always* hard-gated as TODO 0
  so a marker JSON exists even when no external GPU was requested.
- **Configs**: shipped Magpie YAMLs under `scripts/configs/` (one
  `baseline_<framework>.yaml` and one `profile_<framework>.yaml`). The
  optimizer overrides `benchmark.model` / `benchmark.runner_type` /
  `benchmark.benchmark_script` at materialize time; per-task YAMLs are
  written via `_workload_envs.materialize_config_with_envs`.
- **Resume**: `--resume` is a flag (no argument). It demands an existing
  `manifest.json` + `state.json`, clears stale `stop_reason` / `crash_count`,
  and lets `MessageBus.replay_for` + `Coordinator.replay_for_resume` rebuild
  pending proposals from the event log.

### 1.9 Observability Surfaces

Three primary on-disk artefacts back the user-facing reports:

1. **state.json** — live structured state used by both the Coordinator and any
   monitoring script (`scripts/event_counts.py`,
   `scripts/dump_session_breakdown.py`).
2. **events table** — full audit log in SQLite; queryable for monitoring and
   the basis for resume.
3. **session_breakdown.json** — a single self-contained dashboard payload
   produced by `breakdown/exporter.py` (build / write_breakdown_json) at
   end-of-session and on-demand via the `session_breakdown` action. Schema
   `hyperloom.session_breakdown.v1` (~14 sections covering session, workload,
   baseline, final, phase_timeline, capability_summary, geak/oob invocations,
   kernel_lifecycle, param_search, sweep, critic_robustness, telemetry,
   attribution, warnings, source_files). Consumed by `claw-stats-service` and
   `hyperloom-results-service`.

The `report` action assembles a human-facing markdown / JSON report using both
state.json and the per-action workspaces.

### 1.10 Key Design Invariants (for the upcoming modifications)

These constraints come from the design notes embedded throughout the code and
should be respected unless the modification explicitly intends to change them:

1. **Coordinator is the only writer of `CORE_STATE_FIELDS`.** LLMs go through
   `UPDATE_STATE` intents; PolicyGate enforces the field-name allowlist.
2. **Kernel is responder-only.** Orchestration must use
   `request{target_agent="kernel", kind=…}`; kernel-owned actions cannot be
   `delegate`d.
3. **Single intent chokepoint = PolicyGate.** Every transport (Claude tool,
   Codex JSON) lands here before anything is persisted.
4. **One session dir, flat.** Each sandbox is single-use; no `session_id`
   subdirectory under `$USER_DATA_PATH`. All writable artefacts live under it.
5. **Action metadata is the source of truth.** `actions/_meta/<name>.yaml`
   feeds both the runtime (lanes, lease TTL, allowed_tools) and the
   prompt builder (description, pipeline_phase). Adding/changing an action
   means touching the yaml, the executor wiring (`cli._REAL_EXECUTORS_FULL`),
   and `system_paths._runs_actions()` in lockstep.
6. **Idempotent everything.** `tasks.idempotency_key` UNIQUE,
   `make_session_dir()` re-runnable, `events.msg_id` UNIQUE, `cursors` per
   agent, manifest writes atomic.
7. **Path containment.** Every path-like payload field must resolve under
   `$SESSION_DIR` or the framework-source allowlist; otherwise PolicyGate
   denies and the LLM self-corrects.
8. **External services are optional.** `--critic-mock` /
   `--robustness-mock` keep the optimizer self-contained for tests; live
   modes shell out to sibling skill runtimes.

### 1.11 Open Questions / Surface Areas Likely to Be Touched

Based on TODO comments, `_noop_prep` stubs, and recently-added scaffolding,
these are the spots most likely to be touched by upcoming modifications and
worth keeping in mind for chapters 2+:

- **Stub executors**: `dream`, `re_explore`, `comm_optimization`,
  `compiler_tuning` — yaml metadata exists but executors are no-ops.
- **`recover` action**: re-enabled in 2026-05; behind
  `HYPERLOOM_RECOVER_ALLOW_GPU_RESET=1` for the destructive path.
- **Action scoring**: streak-based dampening replaced cooldowns; marathon
  priors are model-class-specific and `MARATHON_PRIORS` is the only place
  to extend per-class behaviour.
- **Critic / Robustness external skills**: the optimizer treats them as
  subprocess transports; broader contract changes in those skills bubble
  through the `*_agent.py` backends.
- **Kernel REQUEST surface**: 4 well-defined kinds today (`select_kernels`,
  `run_optimization`, `integrate`, `apply_patch`). Adding kinds means
  extending `KERNEL_REQUEST_HANDLERS` *and* the system prompt's "Hard rules"
  section in `system_prompts/orchestration.md`.
- **Breakdown exporter**: schema v1 is a stable contract for downstream
  dashboards; any new SharedState fields should be reflected in
  `breakdown/schema.py` and the corresponding collector.
- **Magpie leak-path salvage**: `harvest_leaked_artifacts` +
  `INFERENCE_OPTIMIZER_RESCUE_PATHS` are defense-in-depth; the canonical fix
  is pinning the `*_<runner_type>.sh` script at materialize time.

---

## 2. tbo_vs_hyperloom_kb_gap

This chapter compares `/wekafs/zgong/TBO`(Arbor)的 KB + 本地源码探索机制 与
Hyperloom v0.6 的现状, 然后说明 `/wekafs/haiskong/cortex-for-hyperloom-2026-05-18.md`
所描述的 Cortex KB 服务能补足哪些缺口.

### 2.1 TBO/Arbor 的探索机制(参考实现)

Arbor 的"自学习"并非靠运行时调度算法, 而是靠**写在文件里的领域知识 + 跨
session 的 recipe + 动态构造 specialist prompt** 三件事拼出来的.

**(a) 文件式 KB —— `skills/kb/`**

- 8 个领域目录: `kernel/` `communication/` `compiler/` `framework/`
  `fusion/` `systems/` `pr_intelligence/` `recipes/`.
- 每个域两份核心 markdown: `empirical_kb.md`(已验证经验)+
  `shared_pitfalls.md`(已知坑). 体量在 20–35 KB, 直接给 LLM 当 context.
- 选择器 `arbor.kb.select_kb(task_description)`(`src/arbor/kb.py`)—— 关键词
  → 域映射(`DOMAIN_KEYWORDS`)→ 文件优先序读取, 全部内容拼到 specialist
  prompt 的 "Domain Knowledge" 节.
- 写回 `arbor.kb.contribute_to_kb(domain, finding, source, session_id)`
  以时间戳追加到对应域的 `empirical_kb.md`, append-only 日志即审计.

**(b) Recipe —— 跨 session 的 warm start**

`src/arbor/recipes.py` 定义了一个 dataclass-shaped JSON, 一文件对应一个
`(model, hardware)` 对, 落在 `skills/kb/recipes/<slug>.json`. 字段:

| 字段 | 含义 |
|---|---|
| `best_config` | 上次 session 留下的最佳服务配置(env + flags) |
| `best_throughput` | 已知最佳吞吐 |
| `what_worked` (`Finding[]`) | 已验证为 KEEP 的优化 + 影响 |
| `what_failed` (`Failure[]`) | 已验证为 REVERT 的尝试 + 原因 |
| `remaining_gaps` (`Gap[]`) | 上次没解决的瓶颈 |
| `prs_tested` (`PRResult[]`) | 已经试过的 GitHub PR + outcome |
| `pitfalls` (`Pitfall[]`) | 已知地雷 |
| `stack_fingerprint` | `vllm_version`/`aiter_commit`/`rocm_version`, 用来判断 recipe 是否过期 |
| `sessions[]` | 历史 session 摘要 |

`load_recipe / save_recipe / update_from_session / is_stale` 是接口的全部.
新 session 起手:`load_recipe → 应用 best_config → 检查 stack_fingerprint`,
然后只在 `remaining_gaps` 上发力, 主动跳过 `what_failed` / `pitfalls`.

**(c) GitHub PR 一等公民**

`src/arbor/github_search.py` 直连 GitHub REST: `search_prs` /
`fetch_pr_diff` / `find_relevant_prs` / `search_code`, 默认仓库
`ROCm/aiter` `ROCm/vllm` `sgl-project/sglang` `triton-lang/triton`
`ROCm/hip` `ROCm/FlyDSL`. specialist prompt 中明确鼓励 "看 NVIDIA 在
TensorRT-LLM / vLLM 的同名 PR, 适配到 AMD".

**(d) 本地源码 —— 全工具开放给 specialist**

specialist 是 LLM 而非 Python 函数; `prompt_builder.build_agent_prompt`
告诉它 `Bash / Read / Write / Edit / Grep / Glob / WebSearch / WebFetch`
全部可用, 加上 "你可以读 / 改 framework 源码、提 patch、跑 micro-bench".
KB 文件直接告诉它 *该去 framework 哪个文件*(例如 `opt_flags.py` 编辑
`w_cache_modifier=None`). 真正的本地源码探索靠 LLM agent 自己用
Grep/Read/Edit 完成.

**(e) Introspection —— 闭环写回**

每个 KEEP/REVERT 决策落两个表:
`introspection.save_lesson(model, hw, action, outcome, delta, reason)` 和
`save_research(...)`. 失败也算研究产出. 最后 `update_from_session(recipe,
session)` 把这些汇总进 recipe, 关闭循环.

**(f) Optimization Loop 形态(`skills/SKILL.md`)**

Profile → Score gaps → Search(Recipe ▸ KB ▸ GitHub ▸ Web ▸ novel)→
Dispatch dynamic specialists in parallel(每个带定制 prompt + 该 gap 的
KB 子集 + recipe + pitfalls)→ Integrate one-at-a-time + benchmark →
KEEP/REVERT → update recipe + KB.

### 2.2 Hyperloom v0.6 现状 —— 已有的 KB 触点

Hyperloom 不是没有 KB, 但**只在 Critic 层有, 且能力很窄**:

| 触点 | 位置 | 说明 |
|---|---|---|
| `kb_digest.format_kb_digest_for_orchestration` | `orchestrator/kb_digest.py` | 通过 `INFERENCE_OPTIMIZER_KB_ROOT`(默认 `/wekafs/xiaofei/Hyperloom/marathon/skills/kb`)动态加载 `kb_query.py` + `entries.jsonl`, 用 3 个固定 query pack(torch.compile / sglang params / kernel)取 14 行摘要塞进 Orchestration prompt. 不按 model / framework 过滤. |
| `MARATHON_PRIORS` | `orchestrator/scoring.py` | 4 个 `model_class`(`dense`/`moe_mla`/`moe_swa`/`moe_mla_nsa`)各自的动作初始分(1–10), 仅用于 scoreboard 排序. |
| Critic-agent KB 客户端 | `orchestrator/backends/critic_agent.py` + 外置 `critic-agent/` skill | `CRITIC_KB_CLIENT_MODE ∈ {inmemory, live}`, `KB_BASE_URL` / `KB_TIMEOUT_MS` / `KB_RETRY_MAX` / `KB_DEAD_LETTER_DIR`. Critic 在 `prepare-review` 时拉 `kb_priors_by_proposal`, 在 `commit-review` 时回写 `kb_writes`. 触达面**只在 Critic 评审一条路**, Orchestration 不直接读写 KB(PolicyGate 明确禁止其它角色操作 KB). |
| Kernel REQUEST 路径 | `orchestrator/kernel_request_handlers.py` | shell 出去调用 GEAK / OOB / TraceLens, 自带 KB 但**与 inference_optimizer 不共享**, 各自有独立 memory(GEAK 走 `/wekafs/hyperloom/geak-memory/memory.db`). |
| 会话内审计 | `state.json` + `coordinator.db` | 每个 session 的全部 attempt 都在 `<action>_attempts` / `last_action_failures` / `optimization_stack` 里, **但仅本会话生命周期可见**, 资源单次使用后整个 session_dir 即可销毁. |

**关键差距**:Hyperloom 没有 Arbor 的(a)(b)(c)(d)(e)(f)中的几乎任何
一个等价物 —— 详见 §2.3.

### 2.3 能力差距矩阵

> 横向:Arbor 已有的能力点;纵向:Hyperloom v0.6 是否具备等价物.

| 能力维度 | Arbor 实现 | Hyperloom 现状 | 差距 |
|---|---|---|---|
| 跨 session 的 `(model, hw)` warm-start recipe | `recipes.Recipe` JSON + `load_recipe` / `is_stale` | 无. 每个 sandbox 单次性 (`session_dir` 一次性); `manifest.json` 只是当前 session 的 resume tag | **完全缺失 1 ——没有持久"上次怎么样"的状态** |
| `best_config` / `what_worked` / `what_failed` / `remaining_gaps` / `pitfalls` 结构化语义 | `recipes.Recipe` 字段 | 仅有 `last_action_failures` / `optimization_stack`, 不持久;`MARATHON_PRIORS` 只是数值优先级 | **完全缺失 2 ——没有"已验证经验/坑"的结构化沉淀** |
| Stack fingerprint 失效检查 | `StackFingerprint(vllm/aiter/rocm)` + `is_stale` | 无;`manifest.json` 记录 `code_revision` 但不参与决策 | **完全缺失 3** |
| 领域 KB 文件 (kernel / framework / comm / compiler / fusion / systems) | 8 个域 markdown, 直接拼进 prompt | 单一聚合 `entries.jsonl` 经 `kb_digest` 输出 14 行;无领域分区, 无 pitfalls 文件 | **部分覆盖, 维度不足** |
| 关键词 → 领域选择器 (`select_kb`) | `DOMAIN_KEYWORDS` + 文件优先序 | 无;`kb_digest` 用固定 3 个 query pack | **缺失 select_kb 等价机制** |
| KB write-back (lessons + research) | `contribute_to_kb` / `save_lesson` / `save_research` | Critic 通过 `commit-review` 写 `kb_writes`(只有 verdict 维度);Orchestration / Kernel / Robustness 无写权 | **缺失 4 ——只有 Critic 一条窄写入通道** |
| GitHub PR 检索 | `github_search` 模块, specialist 默认可用 | 无 | **完全缺失 5** |
| WebSearch / WebFetch | specialist prompt 显式开放 | 通过 `cli` 启 Claude/Codex backend, 但 Orchestration / Kernel 系统提示**未开放** Web 工具(PolicyGate 没列入 allowed_tools) | **能力存在但未授权** |
| 动态构造 specialist + 注入领域 KB | `prompt_builder.build_agent_prompt(task, kb_content, recipe, pitfalls, …)` 每个 gap 一份 prompt | Sub-agent 是确定性 Python executor (`baseline_executor` / `params_executor` …), 没有 LLM specialist, 也没有"按 gap 注入 KB" 的能力 | **架构性差距** |
| 跨 session 的 attempt / verdict 检索 | `session.sessions[]` + `cortex-kb traverse`(规划中) | session 销毁后只剩 `session_breakdown.json` 文件, 无统一查询面 | **缺失 6** |
| Negation / refutation 显式建模 | recipe.what_failed + pitfalls 每条带 `reason` | `last_action_failures` 仅滚动 10 条本 session | **缺失 7** |

总结:Hyperloom 的"知识"几乎全部活在**单一 session 内**;Arbor 的"知识"
有 `KB / recipes / lessons` 三层, 跨 session 累积. 这是 V2 升级的核心方向.

### 2.4 Cortex KB 能补足哪些缺口

`cortex-for-hyperloom-2026-05-18.md` 描述的是**一个集中式、图结构、带因果
推理与 hypothesize-verify-commit 协议的 KB 服务**(已 2026-05-18 端到端
跑通, session 36 / promoted_edge 15782). 它直接覆盖 §2.3 的多数空缺,
并且在 Arbor 文件式实现之外提供更强的语义.

**(a) 数据模型对齐到 Arbor 的概念**

| Arbor 概念 | Cortex 概念 | 对应 |
|---|---|---|
| `recipes/<model_hw>.json` | `recipe_node` (point) | `find-recipe --workload --chip` 直接拿到 |
| `what_worked` (Finding) | `optimization_node` + `EXPERIENTIAL` 边 | confirm 时 `promoted-authority EXPERIENTIAL` |
| `what_failed` / `pitfalls` | `negation` edge | `verify --outcome refuted` 自动创建 |
| `remaining_gaps` | `issue_node` (gap) | `propose-point --kind issue_node` |
| `prs_tested` | `pr_node` + `evidence_ref kind:url` | bulk ingest 历史 PR |
| 一次 session | `session` 对象 | `sessions/begin → hypothesize → verify → commit` |
| `stack_fingerprint` | session attrs.`stack_fingerprint` | 任何 query 都能按 fingerprint 过滤 |

**(b) 比 Arbor 更强的部分**

1. **集中服务 + 图查询**: `traverse(start=<gap>, steps=3, branches=4)` 一
   次拉回 gap → optimization → attempt 子图, 不再像 Arbor 那样"全文件
   读进来再让 LLM 自己挑". 也是给 LLM prompt 注入子图的标准方式.
2. **因果链显式建模**: 7 类边(`structural` / `causal` / `evolutionary` /
   `empirical` / `investigation` / `hypothetical` / `negation`)+ 三档
   `authority`(`HYPOTHESIZED → EXPERIENTIAL → AUTHORITATIVE` /
   `INFERRED`). Arbor 的 KB 是平面 markdown, 没有 "假设 → 验证 → 提级"
   的状态机.
3. **Hypothesize-Verify-Commit 协议**: T2 写 hypothetical → T3 verify
   → confirmed 自动提级 / refuted 自动开 negation 边. 与 Hyperloom
   现有的 KEEP / REVERT 决策语义天然对齐, 几乎可以零侵入接到
   Coordinator 的 `_promote_to_shared_state` / `_handle_unpromotable_result`
   分叉点.
4. **跨集群 + crash-safe**: NDJSON 兜底队列(`.kb_pending.ndjson`)+
   `cortex_kb_flusher` daemon, T2/T3 异步, T0/T1/T4 同步. Cortex 短暂
   不可达不阻塞 marathon, 恢复后补齐;canonical_id 自然去重不会双写.
5. **幂等键内置**: `canonical_id` 即幂等键, session_id 跨重启 resume,
   完全契合 Hyperloom "idempotency_key + manifest.json resume" 现有约定.

**(c) 与 Hyperloom v0.6 的对接成本(取自 §5)**

文档里给的最小改动量:

```
HyperloomV2/
├── inference_optimizer/
│   ├── cortex_kb_client.py      [NEW]  ~200 LoC subprocess wrapper of `cortex-kb`
│   └── cortex_kb_flusher.py     [NEW]  ~150 LoC NDJSON flush daemon
├── marathon_optimization/.../scripts/
│   ├── run_baseline.sh          [EDIT] T0 head + T4 trap EXIT
│   ├── run_iter.sh              [EDIT] T1 head
│   └── propose_change.sh        [EDIT] T2 hook
└── .cursor/skills/inference-optimization/
    └── SKILL.md                 [EDIT] add Phase 0 warm start
```

注意:Hyperloom v0.6 已经把 marathon shell 脚本路径替换成了 Magpie 包装器
+ Coordinator action executor, 所以**接入面的具体落点要重新映射**(参见
§2.7).

**(d) 与现有 critic-agent KB 客户端的关系**

Cortex 的 HTTP 端点(`KB_BASE_URL=http://kb-service…`)与 critic-agent 的
`CRITIC_KB_CLIENT_MODE=live` 走的是同一种连接方式. 现有的 `kb_priors` /
`kb_writes` 通路可以直接复用 Cortex 后端;Loop 1 真正新增的是 **将 KB
访问从"只在 Critic 时点"扩展到"T0 / T1 / T2 / T3 / T4 全链路"**.

### 2.5 Primus Cortex PR Monitor 能补足哪些缺口

`primus-cortex-pr-monitor-access.md` 描述的是**一个集中式、内集群部署
的多仓库 GitHub PR 索引服务**(`primus-cortex-pr-api`, 2 副本 Running;
配套 `primus-cortex-pr-monitor` 抓取器 + 50Gi `wekafs-storage-csi` PVC
作 cache, 2026-05-07 上线). 同时暴露 **REST** (`/v1/*`, 含 Swagger 在
`/v1/docs`) 与 **MCP** (`/mcp/`, streamable-http) 两条接入面. 它直接
覆盖 §2.3 中"6. GitHub PR 检索"这一项空缺, 并且在 Arbor 的
`arbor.github_search` 之外提供更强的语义.

**(a) 数据模型对齐到 Arbor 的概念**

| Arbor 概念 | PR Monitor 概念 | 对应 |
|---|---|---|
| `arbor.github_search.search_prs(repo, kw, since)` | REST `GET /v1/repos/{repo}/prs?state=&since=&limit=` + MCP `pr_list` | 同语义, 多了 `label` / `author` / `file_path` / `before` 过滤面 |
| `arbor.github_search.fetch_pr(repo, num)` | REST `GET /v1/repos/{repo}/prs/{n}` + MCP `pr_get` | metadata + commits + 文件列表一次返回 |
| `arbor.github_search.fetch_pr_diff(repo, num)` | REST `GET /v1/repos/{repo}/prs/{n}/patches` + MCP `pr_patches` | Arbor 返回 unified diff 字符串, PR Monitor 返回结构化 patch 列表 |
| `arbor.github_search.find_relevant_prs(desc, repos)` | REST `GET /v1/search/prs?q=&repo=&state=&limit=` + MCP `pr_search` | 全文搜索 PR 标题/正文 |
| `arbor.github_search.search_code(query)` | (无直接对应, 见 §c) | PR Monitor 只索引 PR 范围内的文件, 不做仓库全局 code search |
| Arbor recipe.`prs_tested[]` | Cortex KB `pr_node` + `evidence_ref kind:url` | PR Monitor 给 url + sha, Coordinator mint pr_node 落 Cortex |
| (Arbor 没有的) | MCP `pr_files` / `pr_file_patch` / `pr_pr_file_baseline` / `pr_blob` / `pr_commit_files` / `pr_commit_file` | 按 commit 取文件树快照 / 按 sha256 取 blob / 拿 PR baseline 文件全文 — Arbor 这些都得直连 GitHub 自行拼 |

**(b) 比 Arbor 更强的部分**

1. **集中索引 + 持久 cache**: 50Gi PVC 缓存 PR / commit / blob, **不打
   GitHub API**, 不再受 GitHub rate limit(Arbor `_github_api` 中
   `X-RateLimit-Reset` 退避逻辑直接退役). 6 个监控仓库
   (`ROCm/aiter` / `ROCm/FlyDSL` / `ROCm/hip` / `ROCm/vllm` /
   `sgl-project/sglang` / `triton-lang/triton`) 全部预抓.
2. **commit-level 文件树 + baseline**: `pr_commit_files(repo, sha)` +
   `pr_pr_file_baseline(repo, num, path)` 让 specialist 既能看到 "PR
   改了什么" 也能看到 "PR 改之前长什么样" — 这是判断 "能否在我们当前
   sglang/aiter 版本上 cherry-pick" 的关键. Arbor 只能拿 unified
   diff, 看不到 PR 改前的完整源.
3. **MCP 原生集成**: specialist (LLM sub-agent) 直接挂 MCP server URL
   就能用全套 12 个 readonly tool, **不需要 Hyperloom 写任何 wrapper**.
   Arbor 形态下 `arbor.github_search` 是 Python 模块, agent 还得自己
   写 import + try/except + token 管理.
4. **认证模式简单**: 内集群直连 svc DNS, 不需要 `GITHUB_TOKEN` (PR
   Monitor 后端自己持有 GitHub 凭证). Arbor 需要每个 agent / 每个 pod
   都配 `GITHUB_TOKEN`, 同时 rate limit 还是按 token 算.
5. **与 Cortex KB 同集群**: PR Monitor (`primus-cortex-pr-api`) 与
   Cortex KB (`kb-service.primus-cortex`) 同集群同 namespace, 这意味
   着同一份 KnowledgePlane facade 抽象一次接入, 两个服务共享相同的
   "可达性 / 跨集群 ingress / 监控 / 故障域", 操作员只需处理一处。
6. **`recommended 流程`** : 文档给出标准 PR 分析序列 (`pr_get →
   pr_patches → pr_pr_file_baseline → pr_blob(head_content_sha256) →
   pr_commit_file`), 等价于 Arbor SKILL.md 里只能口述的 "看 NV PR
   再适配 AMD" 流程, 现在是端点级别的明确链路.

**(c) 与 Hyperloom v0.6 的对接成本(基于文档 §"REST API" / §"MCP 接入")**

文档没列改动表, 但实际接入面非常薄, 因为 PR Monitor 自身已是 service:

```
HyperloomV2/
├── inference_optimizer/
│   ├── knowledge_plane/
│   │   └── pr_monitor_client.py        [NEW]   REST 客户端: pr_list / pr_get / search_prs
│   │                                            (urllib stdlib, 不引新依赖)
│   ├── orchestrator/
│   │   └── system_prompts/
│   │       └── specialist.md           [EDIT]  prompt 段 6 (pr_feed) +
│   │                                            specialist 工具集声明 mcp__pr_monitor__*
│   ├── actions/_meta/
│   │   └── _domain_repos.yaml          [NEW]   domain → 仓库映射
│   │                                            (kernel: ROCm/aiter, triton-lang/triton ...)
│   └── cli.py                          [EDIT]  --pr-monitor-url / --no-pr-monitor /
│                                                --pr-feed-window-days
```

具体落点参见 §2.7 (T1 specialist 装配前的 pr_feed 预热) + §3.6 (
KnowledgePlane facade 中 read 侧的 `pr_feed_warm(domain)` 能力) + §3.13
M4 实施稿. 主要 caveat 与 Cortex KB 相同 — **跨集群可达性**: 当前
Hyperloom marathon 跑在 data-plane pod, PR Monitor 在 primus-cortex
集群, 文档明确建议跨集群 ingress 或 sidecar 转发, 与 KB 服务一并解决.

**值得注意的 gap (Arbor 有但 PR Monitor 暂无)**:

- `arbor.github_search.search_code(query)` 的 *仓库全局 code search*
  在 PR Monitor 没有直接对应端点 (`pr_search` 只搜 PR 标题/正文). 真
  需要时可走 Cortex KB 的 evidence_ref 关联, 或者 specialist 用 Read
  工具直接读本地 source-mirrors. 罕见场景, 不阻塞 v0.8.
- PR Monitor 仅索引 6 个仓库; Arbor 默认列表多 `ROCm/composable_kernel`
  (Arbor `DEFAULT_REPOS`). 需要扩仓库走 PR Monitor 文档 §"维护配置"
  的 SQL flow, 平台侧操作即可加, 不是代码改动.

**(d) 与现有 KnowledgePlane facade / critic-agent 客户端的关系**

PR Monitor 与 Cortex KB 在 KnowledgePlane facade (§3.6) 里是**并列的
两个 backend**, 调用面共享同一抽象:

- **读侧 (Coordinator 用)**: facade 暴露 `pr_feed_warm(domain) ->
  list[PR_summary]`, 内部走 REST. 仿照 Cortex KB 的 `find_recipe` /
  `traverse` 风格, 失败时 fallback 返回空列表 + breakdown.warnings 加
  条目, **不阻塞**主流程.
- **specialist 侧 (LLM 直接用)**: 通过 MCP server 注入, 工具白名单含
  `mcp__pr_monitor__*` 全套 12 readonly tool. specialist 看到的"工具
  集"统一由 §3.11 R5 控制 (Web / PR / Cortex MCP 三档分级).
- **写回 Cortex KB 的桥**: specialist 在 `specialist_done.proposal_set[i].
  pr_evidence` 字段引用 PR (url + sha) → Coordinator 在 T2 hypothesize
  时把该 PR 作为 `pr_node` mint 入 Cortex (`canonical_id =
  pr.<repo>#<number>`, 自然幂等), 并把 hypothetical 边的 evidence 多挂
  一条 `kind:url`. 长期看 Cortex 里会累积一份"本组实际尝试过的 PR
  索引", 替代 Arbor recipe 中 `prs_tested[]` 的功能.
- 与 critic-agent 当前 `KB_BASE_URL` 通路**互不冲突**: critic-agent
  评审走的是 Cortex KB, PR Monitor 不参与评审流程; 但 critic 在
  `commit-review` 的 kb_writes 里若引用 PR url, 通过同样的 evidence
  机制让 Cortex 端能反查到 PR Monitor 中的 pr_node, 形成 *KB / PR /
  本地源码*三者通过 Cortex 拼起来的统一证据图.

一句话: PR Monitor 把 Arbor 自带的 `github_search` 模块**外置到独立
服务 + 加上 commit-level 文件树 + MCP 原生集成**, 与 Cortex KB 共
享同一份 KnowledgePlane facade 抽象, 是 §2.4 Cortex 之外的第二个"知
识源"接入. §3.6 / §3.13 M4 给出详细落点.

### 2.6 Hyperloom v0.6 → V2 的差距闭合表

把 §2.3 的差距和 §2.4 / §2.5 两段能力一一对照:

| 差距 | Cortex 能否覆盖 | V2 还需要做什么(Cortex 之外) |
|---|---|---|
| 1. 跨 session warm-start | ✅ `find-recipe` + `traverse` | Coordinator 在 `_seed_shared_state` 后追加 T0 拉取并写入 `SharedState.warm_start_*` 字段 |
| 2. `what_worked` / `what_failed` 结构化沉淀 | ✅ `optimization_node` + edges/negation | 在 `_promote_to_shared_state` / `_handle_unpromotable_result` 各加一个 hook, 调 `hypothesize → verify` |
| 3. Stack fingerprint 失效检查 | ✅ session attrs + query 过滤 | `manifest.py` 增加 `stack_fingerprint`(rocm/aiter/sglang/vllm 版本), 写入 session begin attrs |
| 4. 领域 KB 文件 + select_kb | ✳️ Cortex 用图替代 markdown 检索 | 仍需要保留少量"操作手册"型 markdown(放 `actions/<name>.md`), 但 "经验/坑" 全转 Cortex point/edge |
| 5. KB write-back 多通道 | ✅ propose-point / propose-edge / ingest-attempt | PolicyGate 需要新增 `kb_write` intent type, 把权限按角色细分(Orchestration: hypothesize / Kernel: ingest-attempt / Critic: verify-promote / Robustness: 只读) |
| 6. GitHub PR 检索 | ✅ Primus Cortex PR Monitor (REST + MCP, 见 §2.5) — 6 个监控仓库已预抓, 含 commit-level 文件树 + PR baseline | KnowledgePlane facade 加 `pr_feed_warm(domain)`(REST) + specialist 工具集挂 MCP server URL; 引用过的 PR 由 Coordinator 在 T2 mint 为 Cortex `pr_node` |
| 7. WebSearch / WebFetch 授权 | ➖ 与 Cortex 无关 | PolicyGate 在 Orchestration / Kernel 的 `allowed_tools` 里加入 `WebSearch` / `WebFetch`(必要时为 `dream` / `re_explore` 这类创造性 action 单独放权) |
| 8. 动态 specialist + 注入 KB | ✳️ Cortex 提供 KB, specialist 框架仍需补 | 把 `dream` / `re_explore` / `comm_optimization` / `compiler_tuning` 四个 stub executor 改造成 LLM specialist, prompt 走 `build_agent_prompt(kb=traverse(gap))` 风格 |
| 9. 跨 session attempt 检索 | ✅ `query --kind attempt_node --attrs '{"session_id":...}'` | breakdown exporter 末尾追加 "Cortex session_id" 字段, 让历史能反查 |
| 10. Negation 显式建模 | ✅ refuted → negation edge 自动 | `_handle_unpromotable_result` 触发 `verify refuted` |
| 11. Crash-safe / 跨集群 | ✅ NDJSON `.kb_pending.ndjson` | 在 `<session_dir>/runtime/` 下复用同一目录约定;`cortex_kb_flusher` 作为 robustness-monitor 的一个子任务挂起 |
| 12. 幂等键 | ✅ canonical_id | 复用现有 `idempotency_key`(`approved-<msg_id>` 等), 直接喂给 propose-point 的 canonical_id |

**一句话**:Cortex KB 把 Arbor 的 "KB+recipes+lessons" 三层文件存储升级
为带因果语义的图服务, PR Monitor 把 Arbor 的 `github_search` 模块外
置为内集群索引服务 (含 commit-level 文件树 + MCP), 两者共享同一份
KnowledgePlane facade, 合起来直接覆盖 Hyperloom 当前 12 项缺失中的 6
项 (含原本算 Arbor 自补的 PR 检索); 剩下的
**`领域 markdown 操作手册`(部分保留)**、**`Web 工具授权`** 和
**`动态 LLM specialist 框架`** 是 V2 必须自己补的非 KB 能力.

### 2.7 V2 接入 Cortex 的最小落点(给后续章节铺垫)

按照 Hyperloom v0.6 现有架构, T0–T4 时序契约的具体落点应当是:

| Cortex 时机 | Hyperloom 落点 | 备注 |
|---|---|---|
| **T0** session begin + find-recipe + traps | `cli._run_optimize` 的 `make_session_dir()` 之后, `Coordinator.__init__` 之前 | 拉 warm-start 写 `SharedState.warm_start_recipe` / `warm_start_pitfalls`;Orchestration prompt 用 `build_orchestration_prompt(... warm_start=...)` 注入 |
| **T1** traverse 子图入 prompt | `Coordinator._compose_prompt` 每 tick(可缓存 5 min LRU) | 替代当前 `format_kb_digest_for_orchestration` 的固定 query pack |
| **T2** hypothesize | `Coordinator._handle_propose_action`(propose_action 通过 PolicyGate 之后, 落 PendingProposal 的同时调一次)| 拿 `tentative_edge_id` 写到 `PendingProposal.kb_edge_id` |
| **T3** ingest-attempt + verify | `_promote_to_shared_state` (KEEP → confirmed) / `_handle_unpromotable_result`(REVERT → refuted) | 复用 audit-trail 的 `record_action_attempt` 同位置 |
| **T4** session commit | `Coordinator.stop()` 末尾, `session_breakdown` 之前 | breakdown JSON 新增 `cortex_session_id` 字段 |
| 异步 NDJSON flusher | 作为 `optimizer_runs/` 下的常驻进程, 由 robustness-monitor 检活 | 落点借用现有 `robustness_monitor.sh.example` 启动模式 |

后续章节(第 3 章起)将基于该落点表展开具体的 schema、PolicyGate 规则
扩展、Coordinator hook、breakdown 字段、CLI flag 与回归测试设计.

---

## 3. v0.8_implementation_plan

> 目标版本: `inference_optimizer` v0.8 (从 v0.6 直接跳号, 跳过 0.7 以
> 标记这是一次架构层面的非平凡演进).
> 设计风格: 概念层. 本章只描述 *what / why / where*,
> 不写具体函数签名 / 数据结构字段表 / 测试用例; 这些放在后续 v0.8
> 实施 PR 的设计稿中.
> 范围: 把 TBO/Arbor 的并行 specialist + KB-warm-start + PR 探索能力
> 移植到 Hyperloom, 同时把 v0.6 的 backends/params/scoring 三件遗产
> 折叠掉. 不动 kernel 阶段, 不动 Magpie / TraceLens / GEAK / OOB 的子
> 进程接口.

### 3.1 设计哲学与三条主轴

v0.8 想同时满足三条主轴, 任何设计选择都先对照这三条评价:

1. **流程固化优先于评分动态调度**. v0.6 的 scoreboard + Marathon priors
   解决的是"在 12 个 action 中下一步选哪个"的问题, 而真实的优化收益结
   构是 *先填配置面 → 再填内核面 → 最后跨 workload 验证*, 选择面其实
   是确定性的. 把决策权从代码的数值排序还给 LLM(让它在固定 phase 内
   按 gap + KB 自由判断), 同时让 phase 之间的转移由 Coordinator 强制
   执行 — 这是 TBO 的双层设计 (DFS phase + kernel phase).
2. **知识从单 session 抽出, 拉到 Cortex**. 所有"上次的 best_config / 已
   验证的 optimization / 已知的 pitfall / 关心的 gap" 都不再活在
   `state.json` 里, 而是 Cortex 图 + NDJSON 兜底队列. v0.8 session 启
   动时从 Cortex 拉 warm-start, 退出时 commit 整段 hypothesize-verify
   链. session_dir 仍然单次性, 但**知识是跨 session 持久的**.
3. **执行体一分为二: deterministic executor + LLM specialist**. v0.6 的
   sub-agent 全是确定性 Python(跑 Magpie / 解析 benchmark_report).
   v0.8 在不取代它们的前提下, 新增一类 LLM 驱动的 specialist sub-agent,
   专门承担"看 KB / 读 PR / 翻框架源码 / 提出参数候选" 的工作. 真正跑
   bench 的环节仍然是 deterministic executor, 串行抢 serving GPU.

### 3.2 目标流程: 五段固化管线

v0.8 把 v0.6 的 `prep / measure / explore / analysis / deep / validate /
finalize / support` 八个 pipeline_phase 合并/重排成一条线性管线; 任何
被允许的 action 都必须落到下面五段之一, 不允许 phase 之间反向跳转
(robustness `recover` 例外).

```
                      ┌────────────────────────── time ──────────────────────────►

  Phase 0  PRELUDE      Phase 1  EXPLORE          Phase 2  KERNEL         Phase 3  SWEEP        Phase 4  CLOSE
  ─────────────       ────────────────────       ────────────────         ──────────────         ─────────────
  • Cortex T0          • parallel specialists    • profile (only here)    • workload sweep      • report
  • find-recipe        • merged 'explore' run    • select_kernels         • cross-CONC/ISL/OSL  • Cortex T4 commit
  • traps/pitfalls     • serial integrate        • run_optimization       • validate stability  • session_breakdown
  • baseline           • plateau→ next phase     • integrate              • plateau→ next phase
  • target_analysis    • specialists generate    • E2E retry discipline
  • SharedState seed     hypotheses, NOT patches • freeze server config
```

每段的进入条件、退出条件、允许 action 集合在 §3.8 详细给出.
关键变化:

- `profile` 从 v0.6 的"任意时刻可发"变成 *只在 Phase 2 入口跑一次*,
  作为 kernel 阶段 trace 的来源. EXPLORE 阶段不再需要 trace.
- `sweep` 从 v0.6 的"任意时刻可作 explore 候选"变成 *只在 Phase 3 跑*,
  以 `current_best` 为锚, 验证它在多个工作负载点的稳定性.
- `validate_stack` 不再是独立 action, 它的语义被吸收进 EXPLORE 的串行
  integrate 步骤(每次 KEEP 自动重 bench 当前 stack).
- `recover` 在任何 phase 都可由 robustness 触发(IR-1/IR-4 不变), 但
  不计入 phase 进度.

### 3.3 角色重对齐

v0.8 仍然是 4 角色, 协议层(MessageBus / PolicyGate / TaskRegistry /
ResourceLockManager)不变, 但每个角色的边界根据新管线收紧:

| 角色 | v0.6 职责 | v0.8 职责 | 变化点 |
|---|---|---|---|
| Orchestration | 在 12 action 里按 scoreboard 选 → propose → delegate | 维护 phase 状态机, 在当前 phase 允许的 action 集合内, **基于 gaps + KB traverse 子图自由决策**, 派发 specialist, 汇总结果, 串行触发 integrate | 决策从"选下一个 action"退化为"在当前 phase 内决定派多少个 specialist + 各探什么 domain". 不再读 scoreboard. |
| Kernel | 5 个 deep-kernel action 的 responder | 不变. Phase 2 之外的 kernel REQUEST 由 PolicyGate 直接拒绝 | phase-aware 的 PolicyGate 增量规则 |
| Critic | 评审每个 propose_action | **新增评审 specialist 提议的"探索假设"** + 维持现有 `propose_action` 评审; KB 写入由 Critic 在 commit-review 时点合并到 Cortex 的 `verify` 调用 | 评审面扩大到 specialist 假设 |
| Robustness | 健康监控 + scheduling police | 不变, 新增对 specialist sub-agent 的 stale-heartbeat 检测(模仿 TBO `gpu_pool.cleanup_dead_agents`) | sub-agent 探活范围扩大 |

特别强调:**Orchestration 不直接读 KB, 也不直接调 cortex-kb CLI**. 它通过
Coordinator 注入的 prompt 字段(KB warm-start / traverse 子图 / pitfalls)
间接消费 KB. 这保留了 v0.6 "Orchestration 不持有 KB 写权" 的不变量,
也避免它绕过 Critic 闸把未评审假设直接写入 Cortex.

### 3.4 EXPLORE 阶段: backends 与 params 合并为统一 explore

#### 3.4.1 合并后的 explore action

v0.6 的 `backends` 和 `params` 两个 action 在执行体上几乎重合 — 都走
`_grid_runner._run_magpie` 跑 variant, 都写 `EXTRA_*_ARGS` / `extra_envs`,
都进 `_promote_to_shared_state` 的同一个分叉. 区别只在搜索空间(`DEFAULT_BACKENDS_GRID`
vs `DEFAULT_PARAMS_GRID`)和落点 ledger(`backends_search` vs
`params_search`). v0.8 把它们合并成单一 `explore` action:

- 一个统一执行体, 接收一个 grid (一组 variant), 每个 variant 既可以是
  backend flag 又可以是 param flag, 不再做二分.
- 一个统一 ledger `explore_search`, 取消 `backends_search` 和
  `params_search` 两份 ledger; `winner_history` / `discovered_flags` /
  `synergy_attempted` 全部在新 ledger 里维护.
- variant 来源由 specialist 提供(见 §3.4.2), 不再有"默认 grid"的概念
  — 默认 grid 只作为冷启动兜底(Cortex 没有 recipe 的全新 model).
- accuracy gate 移植不变(高风险 flag 触发 GSM8K diff).

#### 3.4.2 specialist → orchestration → explore 的三段流水

EXPLORE 阶段的核心节奏:

```
            ┌─────────────────────────── one EXPLORE round ───────────────────────────┐
            │                                                                          │
   ┌────────┴────────┐                                                                  │
   │ Orchestration   │  T-0  partition gaps into N domains                              │
   │ phase=EXPLORE   │  T-1  for each domain: build specialist prompt                   │
   │                 │       (gap + KB traverse subgraph + recipe + pitfalls + PRs)    │
   │                 │  T-2  emit N parallel `delegate{action='specialist', ...}`      │
   └────────┬────────┘                                                                  │
            │                                                                          │
            ▼                                                                          │
   ┌─────────────────────────────────────────┐                                         │
   │ N specialists run in parallel           │   each specialist:                       │
   │ (LLM sub-agents, no serving GPU)        │   - reads KB + PR feed                   │
   │                                         │   - reads framework source              │
   │                                         │   - writes a 'proposal_set' result      │
   │                                         │     containing 1..K candidate variants  │
   └────────┬────────────────────────────────┘                                         │
            │                                                                          │
            ▼                                                                          │
   ┌─────────────────────────────────────────┐                                         │
   │ Orchestration aggregates proposal_sets  │   - dedup by canonical fingerprint       │
   │                                         │   - rank by specialist confidence       │
   │                                         │   - pick the next batch (size = M)      │
   │                                         │   - emit propose_action 'explore'       │
   └────────┬────────────────────────────────┘                                         │
            │                                                                          │
            ▼                                                                          │
   ┌─────────────────────────────────────────┐                                         │
   │ Critic Review (existing gate)           │   approve / reject / advise              │
   └────────┬────────────────────────────────┘                                         │
            │                                                                          │
            ▼                                                                          │
   ┌─────────────────────────────────────────┐                                         │
   │ explore executor runs the M variants    │   serial real bench, KEEP/REVERT,       │
   │ (deterministic, holds benchmark_lane)   │   stack onto optimization_stack          │
   └────────┬────────────────────────────────┘                                         │
            │                                                                          │
            └──→ feedback into next round prompt (winners / losers / new gaps)         │
                                                                                        │
   loop until: cumulative_gain plateau OR specialists return empty proposal_set        │
   OR explore time budget exhausted                                                     │
   ──────────────────────────────────────────────────────────────────────────────────┘
```

要点:

- **specialist 不动 serving GPU**, 不抢 `benchmark_lane`. 它的产出是
  *提案* (一组 variant 描述 + 选择理由 + 引用的 KB/PR/源码片段),
  没有 patch, 没有 micro-bench. 与 TBO 的差异: TBO specialist 自己跑
  micro-bench + 出 patch, v0.8 specialist 只出"假设描述" 让
  Coordinator 走统一的 explore executor 跑真 E2E. 这避免了 Hyperloom
  必须再造一套 micro-bench 工具链, 也保留了 PolicyGate / Critic 闸.
- **integrate 严格串行**. 一轮 specialist 派发完, Orchestration 一次
  propose_action 拿到一批 M 个 variant, explore executor 在
  benchmark_lane 上**按顺序**跑, 每个 variant 跑完即 KEEP/REVERT.
  TBO 的 "one change at a time" 在这里等价于 "explore executor 内部
  顺序跑 M 个 variant, 每个独立 stack 决策".
- **plateau 退出**. 当 ≥ K 轮 specialist 全部返回空 proposal_set, 或
  最近 K 轮的 KEEP 累计 gain 低于阈值 ε, EXPLORE 自动结束并进入
  Phase 2.

### 3.4.3 与 v0.6 同名 action 的兼容

`backends_executor` / `params_executor` 在 v0.8 仍以 *internal mode* 存
在, 用于 explore executor 内部分流(根据 variant 是 backend flag 还是
sglang/vllm runtime param 走不同的 EXTRA_*_ARGS 渲染); 但它们不再作
为顶层 action 暴露给 LLM, 也不写自己的 ledger. action_registry 删除
两个 yaml, 加入新的 `explore.yaml`. 旧 session 的 resume 会做一次
ledger 迁移: `backends_search` ∪ `params_search` → `explore_search`,
迁移在 SharedState 加载时一次性完成.


### 3.5 Specialist Sub-Agent 框架

v0.8 引入第二类 sub-agent. 命名上保留 "specialist" 以与 TBO 对齐.

#### 3.5.1 与 deterministic executor 的差异

| 维度 | deterministic executor (v0.6 沿用) | LLM specialist (v0.8 新增) |
|---|---|---|
| 执行体 | 一个 async Python 函数 | 一个 LLM 后端调用 (Claude / Codex / Cursor) 在受限 prompt + 工具集下跑若干轮 |
| 触发 | `delegate{action_name=...}` + PolicyGate | 同样走 `delegate`, 但 action_name 进入 specialist 命名空间(见下) |
| 生命周期 | 单 turn, 输入 RunnerContext, 输出 result dict | 多 turn 直到 LLM 输出"specialist_done" intent 或超时 |
| 工具开放 | 仅 subprocess.run + 读 session_dir | Read / Grep / Glob / Bash(白名单) / WebSearch / WebFetch / Cortex MCP / PR Monitor MCP |
| 写权限 | 只写自己的 task workspace | 同左 + 可向 KB 提 hypothesis(走 Critic 评审) |
| 资源 | 抢 lane(默认 benchmark_lane) | **不抢** benchmark/profile/server lane; 只占新增的 research_lane(见 §3.7) |
| 失败模式 | exception 直接 fail | stale heartbeat / token overflow / tool error 由 robustness 检活 |

#### 3.5.2 Specialist 类型与命名空间

v0.8 不再把 specialist 类型写死在 yaml meta 里(TBO 也是这样: 类型由
LLM prompt 决定, 不在系统侧建固定 taxonomy). 但为了让 PolicyGate +
Cortex KB 能按 domain 分流, Coordinator 维护一份 *运行时* specialist
domain 清单, 至少包含:

- `kernel_switch_specialist` — 关心 attention / MoE / GEMM / fusion
- `serving_specialist` — sglang / vllm 调度器、cuda_graph、kv_cache
- `comm_specialist` — RCCL / NCCL / QuickReduce / AllReduce 算法
- `compiler_specialist` — torch.compile / inductor / triton / AMDGCN
- `system_specialist` — KFD / driver / 内存碎片 / dispatch overhead
- `pr_intel_specialist` — 专做 PR-monitor 跨仓库历史检索(给其他 specialist 提引用)

这份清单等价于 TBO `DOMAIN_KEYWORDS` 的 KB 分区, 并与 Cortex 的
`workload_node` / `issue_node` 的 `layer` 属性对齐. 所有 specialist
共享同一个 sub-agent runner, 区别只在 prompt 注入的 KB 子集 / 工具
白名单 / canonical_id 前缀.

#### 3.5.3 Specialist Prompt 装配契约

每个 specialist prompt 由 Coordinator 现场拼装, 字段固定:

1. *Identity & autonomy* — 你是 X 类 specialist, 不要碰 serving GPU.
2. *Hardware context* — gpu_type / TP / HBM / 计算峰值.
3. *Gap statement* — 当前 gap 描述 (Coordinator 从 SharedState 取出
   "由 specialist 上轮发现 + 由 baseline_failure 暴露 + Cortex
   `issue_node` traverse 给的"三方合并的 gap 列表).
4. *Cortex KB subgraph* — 以本次 gap canonical_id 为锚 traverse
   3 步 / 4 分支的子图, 含历史 attempt + 验证状态 + negation 边.
5. *Recipe summary* — find-recipe 拿到的 best_config + what_worked +
   what_failed + remaining_gaps + pitfalls(由 Coordinator 在 T0 一次性
   缓存到 SharedState.warm_start).
6. *PR feed* — primus-cortex-pr-monitor 上当前 domain 关心仓库的近期
   PR 列表(标题 + 链接 + 标签), specialist 可调 `pr_get` / `pr_patches`
   下钻.
7. *Local source navigation hint* — framework_source_roots 列表 +
   "这些目录可读不可写"提示.
8. *Output protocol* — 必须以一条 specialist_done intent 收尾, 内含
   1..K 个候选 variant + 每个 variant 的 reason / kb_evidence /
   pr_evidence / source_evidence 引用.
9. *Iron rules* — 不动 serving GPU / 不跑 E2E / 不直接写 KB(写要走
   Critic 评审) / 不修改 framework 源码(只读) / 必须在
   max_specialist_turns 内收尾.

contract 上把 specialist 的 *输出* 限定为 "提案 + 证据", 这样它的产
物天然就是 Cortex `optimization_node` 的种子, 也是 Critic 评审的对象.

#### 3.5.4 Specialist 退出协议

specialist 通过一种新的 intent `specialist_done` 收尾, payload 至少包
含: gap canonical_id 引用 / proposal_set / 引用证据列表 / 自评信心
分数 / 残留疑问. 当 specialist 输出空 proposal_set, 也必须显式
`specialist_done{empty: true, reason: ...}` 而不是沉默 — 用于让
Orchestration 判 plateau.

stale 检测: robustness 在每个 tick 扫 `tasks` 表中状态 = running 且
specialist kind 的任务, 比对 last update 时间, 超过阈值即 `kill_task`.
被杀的 specialist 视为输出空 proposal_set.

### 3.6 知识平面: Cortex KB + PR Monitor + 本地源码

v0.8 把 v0.6 的"KB 只在 Critic 时点用"扩展为一个统一的 *知识平面*,
任何 specialist / Critic / Orchestration prompt 都从这个平面消费数据.

#### 3.6.1 三类知识源

| 源 | 提供者 | 主要消费者 | 触达方式 |
|---|---|---|---|
| **Cortex KB(图)** | `kb-service` (HTTP + cortex-kb CLI) | Coordinator (T0/T1/T2/T3/T4) + Critic (priors/writes) + 所有 specialist | HTTP REST 或 CLI 子进程; 失败走 NDJSON 兜底 |
| **PR Monitor** | `primus-cortex-pr-api` (REST + MCP) | 所有 specialist (尤其 pr_intel_specialist) | REST 直接调用 *或* 通过 MCP 客户端注入到 specialist 工具集 |
| **本地框架源码** | `framework_source_roots`(`/sgl-workspace/{aiter,sglang,vllm}/`)+ `runtime/source-mirrors/` | 所有 specialist | Read/Grep/Glob 工具读, 严格只读 |

三者在 Coordinator 一侧由一个新的 *KnowledgePlane* 抽象统一对外, 内
部各自走自己的传输. 调用者(specialist 装配 / Critic prompt 装配)
不感知后端是 HTTP 还是 MCP 还是文件系统.

#### 3.6.2 Cortex 时序锚点的 v0.8 落点

第二章 §2.7 给出 T0–T4 的初步落点; v0.8 在新管线下做以下细化:

- **T0** 落在 Phase 0 PRELUDE 的开头, *manifest 写入之后, baseline 跑之前*.
  返回的 warm_start 不直接进 prompt, 先快照到 SharedState.warm_start_*
  字段(Coordinator 写, 别的角色只读), Phase 0 后续动作 + Phase 1
  specialist 装配都从这里取.
- **T1** 在 EXPLORE 每轮 specialist 装配前调一次 `traverse`(per gap 一
  次, 5 min LRU). 不再每 tick 调用 `kb_digest` 那种全局 query.
- **T2** specialist 提出 proposal_set 时, Coordinator 把每个 variant
  在 KB 中 propose-point 为 `optimization_node` (HYPOTHESIZED) +
  hypothesize 一条 `hypothetical` 边 from `issue_node(gap)` to
  `optimization_node`. tentative_edge_id 落在 `PendingProposal`
  上面, 让后续 verify 找回. *propose_action* 通过 Critic 闸之前的
  这一步是 KB 的"事前留痕".
- **T3** explore executor 跑完 KEEP → `verify confirmed +
  promoted-authority EXPERIENTIAL`; REVERT → `verify refuted` 自动开
  negation 边. ingest-attempt 在每个 variant 跑完时调用一次, 与
  `record_action_attempt` 同位置触发.
- **T4** 在 Phase 4 CLOSE 的 session_breakdown 之前 commit. 整段
  session 的 hypothesize-verify 链在 Cortex 落地, breakdown JSON
  新增 `cortex_session_id` 字段做反查.

#### 3.6.3 NDJSON 兜底与 robustness 监控

`<session_dir>/runtime/cortex/.kb_pending.ndjson` 与
`.kb_flushed.ndjson` 复用 cortex 文档定义的语义. flusher 进程作为
robustness-monitor 的子任务被拉起(模仿 v0.6 已有的
`robustness_monitor.sh.example` 启动样式), 每 5s/50 行 batch POST.
crash recovery 读 `.kb_sid`, resume session, drain pending. cortex
不可达期间 EXPLORE 不阻塞, 但 phase 不能跨 T4 commit — Coordinator
在退出前必须等 flusher drain 或显式 abort session.

#### 3.6.4 PR Monitor 接入形态

primus-cortex-pr-monitor 给出 REST + MCP 两条路, v0.8 选择以下接法:

- **specialist 一侧**: 注入为 MCP server. specialist 的 LLM 子进程
  直接用 `mcp__pr_get` / `mcp__pr_patches` / `mcp__pr_search` 等工具.
  这复用了 LLM 后端原生的 MCP 通路, 不需要在 Hyperloom 写额外封装.
- **Coordinator 一侧**: 启动 specialist 前预热一次 REST 调用拉 *gap
  关心仓库的近期 N 条 PR 摘要*, 写到 prompt 注入字段 `pr_feed`. 这给
  specialist 一个起点, 不必从零搜起.
- **Cortex KB 一侧**: 当 specialist 引用某个 PR (在 specialist_done
  intent 里给出 url + sha), Coordinator 在 T2 propose-point 时把
  PR 作为 `pr_node` 同时 mint, 用 `evidence_ref kind:url` + 通过
  `evidence_ref kind:commit` 引该 PR 的 head sha. 长期看 Cortex 会
  累积一份"本组实际尝试过的 PR 索引", 取代 v0.6 完全没有的
  "prs_tested" 概念.

### 3.7 资源 lane 重设计

v0.6 的四条 lane 在 v0.8 下保留, 新增第五条:

- `server_lifecycle`(沿用)— integrate / kernel apply / 重启服务
- `benchmark_lane`(沿用)— 真 E2E bench, 串行
- `profile_lane`(沿用)— Phase 2 profile 一次性
- `workspace_mutation`(沿用)— kernel patch apply
- `research_lane`(**新增**)— specialist 占用此 lane

`research_lane` 的关键特性:

- 与 `benchmark_lane` / `profile_lane` / `server_lifecycle` 互**不**冲
  突 — specialist 不碰 serving GPU.
- 自身允许并发(N specialist 各占一份 *holder*, lane 容量不为 1).
  v0.6 的 ResourceLockManager 是单 holder lane; v0.8 在抽象层面把 lane
  扩为 (lane_name, capacity), 默认 1, research_lane 由 CLI 配置(默认
  6, 与 TBO `DYNAMIC_GPUS=2,3,4,5,6,7` 对齐).
- 与 `workspace_mutation` 兼容互斥 — specialist 不应在 kernel apply 期
  间读源码, 否则可能读到不一致的源.

dispatcher 改造: `_pump_dispatcher_once` 改为**按 lane 容量并发**调度.
当队列里有多个 specialist 任务且 research_lane 有余量时, 同时启动;
explore / baseline / kernel 任务因为 benchmark_lane 容量为 1 自然串行.
具体并发实现细节 (asyncio.gather / TaskGroup / per-lane 限流) 留给
v0.8 实施稿; 概念上等价于"per-lane semaphore".

### 3.8 phase 状态机与停机条件

#### 3.8.1 phase 转移图

```
                     [enter]
                        │
                        ▼
              ┌──────────────────┐
              │   Phase 0        │
              │   PRELUDE        │  T0 + baseline + target_analysis
              └────────┬─────────┘
                       │ baseline_tput > 0
                       ▼
              ┌──────────────────┐
              │   Phase 1        │
              │   EXPLORE        │  N specialists → propose → integrate → loop
              └────────┬─────────┘
                       │ plateau OR empty proposals OR explore budget hit
                       ▼
              ┌──────────────────┐
              │   Phase 2        │
              │   KERNEL         │  profile (1×) + select_kernels + kernel_opt + integrate
              └────────┬─────────┘
                       │ plateau OR retry budget exhausted OR kernel budget hit
                       ▼
              ┌──────────────────┐
              │   Phase 3        │
              │   SWEEP          │  cross-workload validation of current_best
              └────────┬─────────┘
                       │ sweep complete OR budget hit
                       ▼
              ┌──────────────────┐
              │   Phase 4        │
              │   CLOSE          │  report + Cortex T4 commit + breakdown
              └────────┬─────────┘
                       │
                       ▼
                     [exit]

  At any phase, robustness may emit `recover` (out-of-band, doesn't change phase)
  or `escalate_strategy_change` (priority-0 advisory; orchestration may early-exit)
```

phase 由 Coordinator 在 SharedState 持有, 写入受 PolicyGate 保护(只
有 Coordinator 自己能改, 别的角色 propose `phase_change` 视为非法).

#### 3.8.2 每段允许的 action

| Phase | 允许 action | 备注 |
|---|---|---|
| 0 PRELUDE | target_analysis, baseline, recover | 顺序固定; recover 仅作 IR-1 失败时的恢复 |
| 1 EXPLORE | explore (合并版), specialist (新), recover | profile / kernel_opt / integrate / sweep / report 在此 phase 全部被 PolicyGate 拒绝 |
| 2 KERNEL | profile (1 次), kernel_opt, integrate, deep_kernel_analysis, operator_tuning, vendor_kernel_config, recover | 由 v0.6 KERNEL_OWNED_ACTIONS 加 profile 组成; explore/specialist 拒绝 |
| 3 SWEEP | sweep, recover | sweep 在此独占 |
| 4 CLOSE | report, session_breakdown, recover | 任何其他 action 拒绝 |

PolicyGate 在 v0.8 增加一条 *phase 兼容性* 校验, 失败仍走
`policy_denied` 观察事件让 Orchestration 自我修正.

#### 3.8.3 plateau / 终止判定

每个 phase 退出条件均由 Coordinator (而非 LLM) 判定; LLM 可以提前
emit `escalate_strategy_change` 建议跳 phase, 但不能直接触发. 判定
信号:

- EXPLORE plateau: 连续 K 轮 specialist 调度, 累计 KEEP 增益
  < ε_explore (默认 0.5%)且最近一轮 proposal_set 全部为空.
- KERNEL plateau: 连续 K 个 kernel_id 都触发 REVERT 或 NEEDS_REVIEW
  (与 v0.6 现有 retry discipline 一致).
- SWEEP 完整性: 预设 workload 网格全部跑完即结束; 失败 ≤ 阈值.
- 全局: 总 wall-clock budget hit / 用户请求 stop / robustness
  `escalate_strategy_change` 三连击.

`stop_reason` 词表沿用 v0.6 (`target_reached` / `no_more_leverage` /
`time_exhausted` / `max_ticks` / `policy_loop` / `baseline_failed`)
并新增 `plateau_explore` / `plateau_kernel` 两个细化原因, 便于 Cortex
端反查为什么 session 这么早结束.


### 3.9 决策面变化: 砍掉 scoreboard

v0.6 的 `orchestrator/scoring.py` + `MARATHON_PRIORS` + 流式 streak 评分 +
prompt 渲染 top-12 全部从 v0.8 移除. 移除后:

- Orchestration prompt 不再包含 "Action scores" 区块.
- 没有 `apply_keep / apply_failure / apply_no_promote / apply_lock` /
  cooldown 概念, 也没有 base_score / score_mult / locked_reason 字段.
- 决策面仅由三个东西驱动: 当前 phase 允许 action 集合 + Cortex KB
  traverse 给出的 gap 子图 + specialist 输出的 proposal_set.
- "什么 action 优先" 完全交给 LLM, prompt 只告诉它"当前 phase 允许
  做这些, 这是 gaps, 这是 KB". 与 TBO `skills/SKILL.md` 的 "Scoring
  Gaps" 一节一致 — 评分由 LLM 完成, 系统只供原料.

兼容: 老 session 的 `state.json.action_scores` 字段在 SharedState 加
载时被静默丢弃(写入一次警告日志). breakdown 导出删除 attribution 中
的 score 派生字段.

注意: scoreboard 砍掉后, 仍然保留以下两类**软 cooldown**:

- specialist 域级: 每个 domain 在 EXPLORE 内连续 3 轮空 proposal_set
  即被本 phase 暂时锁(下一 phase 不影响), 防止反复白召唤.
- kernel_opt 失败 cooldown: 同一 kernel_id 跑出 N 次 PARTIAL 后熔断,
  与 v0.6 `_DEFAULT_KERNEL_OPT_MAX_PARTIAL` 等价, 这是数据保护不是评分.

这两个不是基于"分值排序"的, 是基于**事件触发的硬熔断**, 与 scoreboard
精神不同, 故保留.

### 3.10 SharedState 演进概要

v0.8 在 SharedState 上做加减如下(具体字段表留实施稿):

新增:

- `phase`: 当前 phase 标识
- `phase_started_ts`: 进入当前 phase 的时间
- `phase_history[]`: phase 转移日志, 含进入/退出 reason
- `warm_start_recipe`: T0 拉到的 best_config 快照
- `warm_start_pitfalls`: T0 拉到的已知坑列表
- `explore_search`: 合并后的统一 ledger (替代 backends_search + params_search)
- `specialist_rounds[]`: 每轮 specialist 派发的 metadata (gap, domains, proposal_count, kept_count)
- `cortex_session_id`: T0 返回的 sid
- `pending_kb_edges[]`: T2 创建但 T3 未 verify 的 edge_id, 用于 crash recovery 补 verify

删除 / 弃用 (resume 时静默迁移):

- `action_scores`(整个字典)
- `backends_search` / `params_search`(并入 `explore_search`)
- `params_no_promote_streak` / `score_violation` 等评分派生字段

保留不动:

- `baseline_tput` / `baseline_accuracy` / `current_best` /
  `cumulative_gain` / `optimization_stack` / `gain_per_stack_entry` /
  `last_profile_trace` / `last_select_kernels` / `last_kernel_opt`
  / `last_action_failures` / `<action>_attempts` 系列等所有本质事
  实层数据.

这里的核心原则: **事实层不动, 决策层(评分 + cooldown)清空, 知识层
(warm_start + cortex_session_id + pending_kb_edges)外接 Cortex**.

### 3.11 PolicyGate 演进概要

PolicyGate 是 v0.8 唯一仍然由代码硬性把守的边界. 以下规则**新增** (不改
v0.6 已有规则):

1. **Phase 兼容性**: 任何 `propose_action` / `delegate` / `request` 必
   须在当前 SharedState.phase 允许集合内, 否则 `policy_denied`.
2. **Specialist 来源**: `delegate{action='specialist', ...}` 仅
   Orchestration 可发. 此外还会校验 `params.domain ∈ 已知 domain 集`.
3. **Specialist 输出**: `specialist_done` intent 仅 specialist 自身可
   发(`from_agent` 必须以 `specialist:` 前缀 + 该 task 的 task_id).
4. **KB 写权角色矩阵**:
   - hypothesize / propose-point / propose-edge: Coordinator (代 Orchestration)
   - ingest-attempt: Coordinator (代 explore executor 结果)
   - verify (confirmed/refuted) + promote: Coordinator (代 Critic Review 结果)
   - 任何 LLM 角色直接调 cortex-kb: 拒绝
5. **Web 工具白名单分级**:
   - WebSearch / WebFetch: 仅 specialist 角色, 仅在 EXPLORE phase
   - PR Monitor MCP: 仅 specialist 角色, 任何 phase
   - Cortex MCP(只读 traverse / find-recipe): 仅 specialist 角色

注意 Orchestration / Kernel / Critic / Robustness 主反应器的工具集**保
持收紧** — 它们看到的世界仍由 Coordinator 注入的 prompt 字段中介.
"开放工具"的扩张面只放给 specialist 这个新型 sub-agent, 风险面有限.

### 3.12 可观测性与 breakdown 演进

session_breakdown.json schema bump 到 `hyperloom.session_breakdown.v2`,
保持向后兼容(`v1` reader 仍可读, 多余字段忽略).

新增 / 调整段:

- `phase_timeline`: 扩展为带 phase 边界的 timeline, 每段可独立可视化.
- `specialist_runs`(新): 每轮 specialist 派发的 (gap, domains,
  parallelism, kept_proposals, kb_edge_ids).
- `capability_summary`: 把原来的 `backends` + `params` 行合并为
  `explore`; 新增 `specialist` 行.
- `kb_provenance`(新): cortex_session_id + 本次 session 在 KB 里
  promote 的 edge_id 列表 + 创建的 negation_edge_id 列表.
- `attribution`: gain 按 phase 归因(Phase 1 EXPLORE / Phase 2 KERNEL /
  Phase 3 SWEEP), 内部细分到 specialist domain.

老 dashboard(claw-stats-service / hyperloom-results-service)按现有
读取顺序一律读 v2 字段, 缺失则读 v1 兼容字段, 不强制下游升级.

### 3.13 迁移路径与里程碑

v0.6 → v0.8 不是单 PR, 按下表分阶段推进, 每阶段独立可发布:

| 阶段 | 范围 | 验收 |
|---|---|---|
| **M1: Cortex 接入(纯写)** | T0/T2/T3/T4 全链路接 cortex-kb CLI; NDJSON 兜底; SharedState `cortex_session_id` 与 `pending_kb_edges`; breakdown v2 加 `kb_provenance` 段 | 端到端跑一次 v0.6 流程, KB 中能看到 hypothesize → verify → commit 链 |
| **M2: phase 状态机 + scoreboard 砍除** | 引入 5-phase 管线, PolicyGate 加 phase 兼容性, SharedState 加 phase / phase_history, 删除 action_scores, prompt 删除 scoreboard | v0.6 老 yaml session 能 resume; phase 自动推进; 不再依赖 MARATHON_PRIORS 也能跑完整 loop |
| **M3: explore 合并** | 新增 explore action / explore_search ledger; 老 backends/params 降级为内部 mode; ledger 迁移逻辑 | 单次 EXPLORE 跑出与 v0.6 同等吞吐, breakdown 显示统一 explore 行 |
| **M4: PR Monitor 接入** | KnowledgePlane 抽象; specialist 注入 PR MCP; Coordinator 预热 pr_feed; cortex pr_node 写回 | specialist prompt 中能看到 pr_feed; Cortex 中 pr_node + url evidence 能查到 |
| **M5: Specialist 框架(MVP)** | 新 sub-agent runner 类型; research_lane(单容量起步); 一类 specialist (`serving_specialist`) 跑通 | EXPLORE 中能并发出 1 个 specialist + 1 个 explore executor, 不抢锁 |
| **M6: Specialist 多 domain + 并发** | research_lane 容量提到 6; 6 类 specialist 全部接入; dispatcher per-lane 并发 | EXPLORE 中并发 N 个 specialist (N≤研究 lane 容量), serial integrate 不变 |
| **M7: SWEEP / KERNEL 收敛 + plateau 调参** | plateau 阈值字段, escalate_strategy_change 软退路径, robustness specialist 探活 | 整段端到端跑完会自然进入 SWEEP 然后 CLOSE, breakdown 中 phase_history 完整 |

每个里程碑都不破坏 v0.6 的"手工 fallback 路径": 例如 specialist 暂时
不可用, Orchestration 仍能直接 `propose_action='explore'` 走默认 grid.
这样保证 v0.8 永远是 *additive*, 在线上有问题时可以一键回退到上一个
里程碑.

### 3.14 已识别风险与应对

| 风险 | 影响面 | 应对 |
|---|---|---|
| Cortex 服务不可达期 EXPLORE 误判 plateau | 误进 KERNEL phase | NDJSON 兜底 + plateau 判定多看 1 轮 + 设最低 EXPLORE 时长保护 |
| specialist 输出 proposal_set 与本地框架版本不匹配 (PR 还没合) | KEEP 后 server 启不来 | explore executor 增加"启动失败立即 REVERT + 标 patch_unresolved 写 Cortex"路径 |
| research_lane 并发暴涨, 占满 LLM 配额 | 反噬主 4-agent loop | per-lane capacity 默认保守 (start with 2), CLI 可调; 同时全局 LLM 调用速率限流 |
| 砍掉 scoreboard 后 LLM 决策不稳定 | EXPLORE 反复尝试同一类失败 variant | dedup 仍由 explore_search.canonical_fingerprint 处理; specialist domain 软 cooldown 兜底 |
| Critic 评审吞吐成为瓶颈 (specialist 大量提案 → Critic 限速) | EXPLORE 阶段被 Critic 节流 | 评审改批量(一次评一组 variant)+ 必要时 specialist 提议合并 K 个 variant 为单个 proposal |
| 多 specialist 同时读 framework 源码触发文件系统压力 | I/O 抖动 | source-mirrors 已落在 wekafs, 单 session 单挂载点; 必要时 add LRU file cache |
| PR Monitor MCP 跨集群访问失败 | specialist 退化为只看 KB | specialist prompt 显式说明 "PR feed 可空", 不强依赖 |

### 3.15 与 v0.6 的语义对照速查

最后给一份"同名概念的语义跃迁"备查表, 给后续读者校准:

| v0.6 | v0.8 | 备注 |
|---|---|---|
| `pipeline_phase` (yaml meta) | `phase` (SharedState, by Coordinator) | yaml 字段保留作展示但不再驱动决策 |
| `MARATHON_PRIORS` | (removed) | 被 KB warm_start + LLM 决策替代 |
| `backends` action | `explore` action (internal mode = backend) | yaml 删除, ledger 合并 |
| `params` action | `explore` action (internal mode = param) | 同上 |
| `validate_stack` action | EXPLORE serial integrate 内置语义 | 不再独立 action |
| `kb_digest.format_kb_digest_for_orchestration` | KnowledgePlane 中 traverse 装配 | 全局 query pack 退役 |
| `delegate` 派发 deterministic executor | `delegate` 派发 deterministic executor 或 specialist (按 action_name 分流) | sub-agent runner 双形态 |
| `propose_action` | 不变, 但 EXPLORE 阶段的 propose 来源是 specialist 汇总 | 走的还是 Critic 闸 |
| Critic Review (KB priors / writes) | 不变 + 评审 specialist 假设 | 评审面扩大 |
| `state.json.action_scores` | (removed) | resume 时静默丢弃 |
| `last_action_failures` / `<action>_attempts` | 保留 | 事实层数据不动 |
| breakdown schema v1 | breakdown schema v2 (附带 v1 兼容) | 新增 specialist_runs / kb_provenance |
| `optimizer_runs/robustness_monitor.sh` | 同 + flusher 子任务 | 复用现有 launcher 模板 |

—— 第 3 章结. 后续章节 (第 4 章起) 将基于 §3.13 的 M1–M7 里程碑各
出一份**实施级**设计稿(字段表 / PolicyGate 规则增量 / 测试矩阵).
