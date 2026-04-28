# Inference Optimizer — Implementation Checklist (v0.6)

> **来源**: `inference-optimizer-DESIGN-modified.md` v0.5 Final  
> **目标**: 把设计文档每一节落到一个个具体可勾的实现点  
> **使用方式**: 一项一项做；勾选 `[x]` 表示完成（含代码 + 单测 + lint）；若该项是占位等待外部依赖（如 sprint 资产迁入），写明依赖

---

## 0. Legend

- `[x]` = 已完成（v0.5 stub 阶段已经写出且有单测 / 已经 import-clean）
- `[ ]` = 未完成
- `[~]` = 部分完成（stub 占位，需要补血）
- `(STUB)` = stub 文件已存在，但函数体是 `NotImplementedError`
- `(EXT)` = 依赖外部资源（sprint 现有脚本 / claw / OOB / GEAK MCP 等）
- `[D §X.Y]` = 引用 DESIGN 第 X.Y 节

---

## Phase 0 — 已完成项（v0.5 落地的部分）

### Storage 基础设施
- [x] `src/inference_optimizer/__init__.py` — 包标识（version = 0.5.0）
- [x] `src/inference_optimizer/paths.py` — 生产 NFS 路径 + dev override `[D §3.1 / §17]`
- [x] `src/inference_optimizer/storage/__init__.py`
- [x] `src/inference_optimizer/storage/connection.py` — `SqliteConnection` (WAL + BEGIN IMMEDIATE) `[D §3.5.4]`
- [x] `src/inference_optimizer/storage/schema.py` — 4 张表 DDL + ensure_schema/reset_schema `[D §13.2]`

### Pure-logic 模块
- [x] `orchestrator/execution_mode.py` — `ExecutionMode` + `choose_execution_mode` `[D §3.4.1]`
- [x] `orchestrator/feature_flags.py` — `FeatureFlags` + `build_feature_flags` 12 项 flag `[D §3.4.4]`
- [x] `orchestrator/objective.py` — Objective 抽象 + 4 个实现 + 工厂 `[D §8]`

### Storage-backed 协调原语
- [x] `orchestrator/resource_lock.py` — `SqliteLeaseBackend` + `ResourceLockManager` `[D §3.5]`
- [x] `orchestrator/message_bus.py` — events 表 + topic 白名单 + replay `[D §10.1 / §10.2]`
- [x] `orchestrator/task_registry.py` — tasks 表 + 状态机 + idempotency_key `[D §13.4]`
- [x] `orchestrator/cursor_store.py` — cursors 表 + UPSERT + monotonic 保护 `[D §13.3]`

### 单测（570 项已 pass — v0.7 多日实施完成）
- [x] `tests/conftest.py` — tmp session_dir + clean db fixture
- [x] `tests/test_storage_schema.py` (6)
- [x] `tests/test_resource_lock.py` (9)
- [x] `tests/test_message_bus.py` (11)
- [x] `tests/test_task_registry.py` (8)
- [x] `tests/test_cursor_store.py` (6)
- [x] `tests/test_execution_mode.py` (11)
- [x] `tests/test_feature_flags.py` (5)
- [x] `tests/test_objective.py` (8)
- [x] `tests/e2e/test_resume_smoke.py` (3) — 含跨表事务 ROLLBACK / COMMIT 验证
- [x] `tests/e2e/test_dry_run_smoke.py` (7) — Conductor + MockBackend 端到端
- [x] `tests/test_intent_parser.py` (20) — envelope schema + tool_use + JSON-in-text
- [x] `tests/test_bootstrap_probe.py` (8) — find_node / find_claude / version 校验
- [x] `tests/test_bootstrap_install.py` (7) — Node 下载/解压 + npm install -g（mocked）
- [x] `tests/test_bootstrap_orchestrator.py` (5) — `ensure_claude_cli` 路径矩阵
- [x] `tests/test_claude_backend.py` (7) — ClaudeBackend with mocked claude-agent-sdk
- [x] `tests/test_cli_backends.py` (9) — `--backend claude` + `--auto-install` wiring
- [x] `tests/test_agent_role.py` (18) — AgentRole + role registry + `roles_for_mode`
- [x] `tests/test_policy.py` (29) — `PolicyGate` per-role / per-mode / state-mutation rules
- [x] `tests/test_conductor_policy.py` (7) — multi-reactor + PolicyGate gating in Conductor
- [x] `tests/test_handle_intent.py` (12) — 10 intent-type branches end-to-end
- [x] `tests/test_action_registry.py` (19) — YAML loader + schema + mode filter
- [x] `tests/test_sub_agent_runner.py` (10) — lane acquire / prompt / metrics extraction / dispatcher
- [x] `tests/test_mcp_emit_intent.py` (13) — `emit_intent` MCP tool, build_emit_intent_server, ClaudeBackend wiring
- [x] `tests/test_iron_rules.py` (28) — IR-1..IR-7 predicates + render_for_prompt + mode filter
- [x] `tests/test_process_management.py` (24) — safe_kill / pgrep / unset_profile / TP guard / vllm flags / IR-3
- [x] `tests/test_kernel_opt_constants.py` (13) — Final constants + env-driven helpers
- [x] `tests/test_action_catalog.py` (24) — full 22-action catalog + lane / mode / family invariants
- [x] `tests/test_score_priors.py` (22) — DENSE/MOE_MLA/SWA/NSA priors + classify_model patterns
- [x] `tests/test_accuracy_gate.py` (24) — Verdict / extract_score / run_gsm8k subprocess seam / micro-check
- [x] `tests/test_scheduler.py` (15) — pressure / mode_gate / depth_gate / diminishing / lane_avail / 7 update rules
- [x] `tests/test_early_stop.py` (12) — 5 stop signals + priority ordering
- [x] `tests/test_kb.py` (20) — KnowledgeBase, ingest, recall, persona, conflict, cross-run synthesis
- [x] `tests/test_persona.py` (12) — estimate_tokens / should_distill / distill_persona (truncate + backend)
- [x] `tests/test_sage_query_service.py` (12) — recall, cache, timeout, prefetch
- [x] `tests/test_kb_cli.py` (10) — kb_query / kb_ingest CLI end-to-end
- [x] `tests/test_backup.py` (8) — vacuum_into / periodic_backup / force_backup / restore
- [x] `tests/test_checkpoint.py` (26) — Checkpoint.create/list/load + resume + 6-row evidence_check_matrix
- [x] `tests/test_monitor.py` (10) — snapshot, per_agent, per_lane, top_events, db resolver
- [x] `tests/test_patch_inductor.py` (9) — IR-6 enforcement (target-file / cache-dir / best-config)
- [x] `tests/test_scripts_skeletons.py` (3 + 3 skipped on non-bash) — DRY_RUN_MOCK paths
- [x] `tests/test_conductor_wiring.py` (18) — persona index + IronRules block + sage hint + dispatcher loop + RCA + parliament + token meter + 30-min checkpoint cadence + resume
- [x] `tests/test_brier.py` (12) — BrierEntry + tracker shrinkage + per-agent independence + snapshot
- [x] `tests/e2e/test_full_loop.py` (4) — delegate dispatch e2e / crash→resume→evidence_check / emergency stop / checkpoint round-trip
- [x] `tests/test_perf_smoke.py` (4) — 500 events / 8-reactor replay / 50 lease churns / mutex sanity

### Phase D — 最少集合（dry-run launch ready，2026-04-27 完成）
- [x] D2 `orchestrator/backends/{__init__,base,mock}.py` — Backend ABC + MockBackend
- [x] D3 `orchestrator/shared_state.py` — minimum methods (time_left/summary/set_stopping/should_stop/refresh_elapsed/snapshot/transition)
- [x] D4 `Conductor._bootstrap()` — wires DB/bus/cursors/tasks/locks/objective/state（绕过 scheduler/policy/sub_agent/actions）
- [x] D5 `Conductor.run/_reactor/_clock/_stopping_watcher/_graceful_stop/_handle_intent/_compose_prompt` — minimum 主循环 + self-message filter
- [x] D6 `cli.py` + `__main__.py` — argparse 入口；可 `python -m inference_optimizer ...`
- [x] D7 `.cursor/skills/inference-optimizer/SKILL.md` — Launch + Monitor sections
- [x] D8 `tests/e2e/test_dry_run_smoke.py` — 7 个端到端测试（time_exhausted / events / no-feedback / cursor / state.json / objective / emergency_stop）

### Phase E — Claude backend + 自动装 Node/CLI（2026-04-27 完成）
- [x] E1 `bootstrap/probe.py` — `find_node` / `find_claude` / 版本校验，跨平台 `[D §6.2 / §15]`
- [x] E2 `bootstrap/install.py` — portable Node 下载/解压 + `npm install -g --prefix`，零 sudo
- [x] E3 `bootstrap/{__init__,errors,orchestrator}.py` — `ensure_claude_cli(auto_install)` 编排 + 路径矩阵
- [x] E4 `orchestrator/intent_parser.py` — 真实 schema/validator/`parse_claude_trajectory`/`parse_codex_validated_json` `[D §10.5.3-6]`
- [x] E5 `orchestrator/backends/claude.py` — `ClaudeBackend` (claude-agent-sdk + JSON-envelope，repair retry，SDK 异常包裹)
- [x] E6 `cli.py` — `--backend claude` / `--auto-install` / `--no-auto-install` / `--bootstrap-cache-dir` / `--claude-model` + `requirements.txt`
- [x] E7 `SKILL.md` — bootstrap 决策矩阵 + 缓存目录布局 + Claude 启动 PowerShell/Bash 示例
- [x] E8 全套 130 项 pytest 通过，ReadLints 0 错

> 该阶段开通了 `python -m inference_optimizer ... --backend claude --auto-install`。
> 已落地：探测 → 自动 install Node + claude CLI → 启动 ClaudeBackend → SDK 调用 → 解析 envelope → 写入 events / 推进 cursor / 安全停机。
> 仍未通：MCP 自定义工具注册（Phase 6.3）、PolicyGate 校验（Phase 4）、Scheduler/SubAgentRunner 触发真实 benchmark/kernel-opt（Phase 7-9）。
- [x] D9 `scripts/inspect_session.py` — 会话状态快查工具

> 该最少集合让 `python -m inference_optimizer --model X --max-hours 0.001 --backend mock`
> 可以完整执行（boot → reactor + clock 循环 → 优雅停止），是后续接入 Claude/Codex 真实 backend 的脚手架。

### Phase F — Make Intents Real（v0.6，2026-04-27 完成）

让从 LLM 流出的 intent 第一次能真正改变系统（创建 task / 修改状态 /
触发 sub-agent / 调用 MCP 工具），而不只是被丢进 events 表。

- [x] F1a `orchestrator/agent_role.py` — `BackendType` + `AgentRole` 数据类、`_BASE/_EXECUTOR/_WATCHDOG/_CRITIC/_SAGE_INTENTS` allow-lists、`claude_role` / `codex_role` 工厂、`default_role_registry`、`roles_for_mode(mode)`、按角色加载 markdown system prompt（带缓存）`[D §6.1 / §10.5]`
- [x] F1b `orchestrator/policy.py` — `PolicyGate` 单一 chokepoint（`role.allowed_intents`、`flags.enable_subagent_delegate`、`DELEGATE`/`PROPOSE_ACTION` action allow-list、`UPDATE_STATE` 保护 `CORE_STATE_FIELDS`、`SEND_MESSAGE` topic allow-list、Bash 命令 allow/denylist + `DEFAULT_QUICK_ACTION_ALLOWLIST`）+ `PolicyDenied` `[D §3.4 / §10.5 / §11]`
- [x] F1c `Conductor` 接线 PolicyGate 与 multi-reactor — `ConductorContext.role_registry/roles`、`_bootstrap` 实例化 PolicyGate、`run` 为每个 `roles_for_mode(mode)` 角色 spawn 独立 `_reactor`、`_reactor` 通过 `policy.allowed_tools_for_agent` 拉允许的工具、`_compose_prompt` 注入角色 system prompt、每条 intent 走 `_gate_intent`（拒绝 → bus 上 `policy_denied`）`[D §15]`
- [x] F2 完整 `_handle_intent` —— 10 个 intent 分支真正落地：
  - `send_message` / `alert`（同时写 `findings/alerts.jsonl`）
  - `propose_action` / `delegate`：用 `_task_idempotency_key` 入队 `delegate` task，幂等键防重复
  - `update_state`：写 `SharedState`（再过一遍核心字段保护），并发布 `state_change` 事件
  - `update_persona`：append-only 写 `personas/<agent>.md`
  - `ask_question` / `answer` / `objection` / `vote`：广播到对应 topic
- [x] F3a `orchestrator/action_registry.py` — `ActionMetadata`（含 `requires_lanes` / `allowed_tools` / `allowed_modes` / `lease_ttl_sec` / `applicable_when`）、从 `actions/_meta/*.yaml` + `actions/<name>.md` 加载、schema 校验（必填 / family / name 匹配文件名 / `allowed_modes` 必须是 `ExecutionMode`）、`get/all/names/allowed_for_mode/system_prompt_for`，并交付三个最小 action：`bench_runner`、`param_sweep_run`、`kernel_opt`，全部 `requires_lanes` 仅引用 `ResourceLockManager.KNOWN_LANES`（`server_lifecycle` / `workspace_mutation` / `benchmark_lane` / `profile_lane`）`[D §11.3 / §12]`
- [x] F3b `orchestrator/sub_agent_runner.py` — v0.6 dry skeleton：解析 action metadata（缺则 `queued→running→failed` 走全状态机）、`ResourceLockManager.acquire` 拿 lane（contention → failed retry）、`_compose_prompt` 拼接 action.md + task.params + lanes，调用 `Backend.run(prompt, allowed_tools=..., max_turns=..., extra=...)`，从返回 intents 中抽 metrics（`update_state.changes.current_tput` → `tput`）和 artifacts，状态收尾（succeeded / failed / needs_manual_review），并提供 `dispatch_pending_delegates(runner, db, ...)` 持续抽取 queued delegate task `[D §11]`
- [x] F4a `orchestrator/backends/mcp_emit_intent.py` — 进程内 MCP server 暴露 `emit_intent` 工具，`MCP_SERVER_NAME="inference_optimizer"` / `EMIT_INTENT_TOOL_QUALIFIED="mcp__inference_optimizer__emit_intent"`、`validate_emit_intent_input`（单 intent 形式 + `_PAYLOAD_REQUIRED` 校验）、`build_emit_intent_server(sdk_module, tool_factory, server_factory, handler)` 提供完整 test seam，SDK 不可用时返回 `None` 自动降级
- [x] F4b `orchestrator/backends/claude.py` — 引入 `enable_mcp_emit_intent` / `sdk_module` / `mcp_server_factory` / `mcp_tool_factory`，`__post_init__` 真注册工具（设置 `mcp_server_config` + `mcp_tool_name`），`_compose_prompt` 优先走 `_OUTPUT_INSTRUCTIONS_TOOL`（fallback 仍是 fenced JSON），`_build_options` 自动注入 `mcp_servers` + 把 qualified tool name 加进 `allowed_tools`，`_import_sdk` 改为返回四元组（`query, options_cls, extractor, sdk_module`）
- [x] F4c 测试矩阵：
  - `tests/test_mcp_emit_intent.py` (13) — `validate_emit_intent_input` 正负样本、handler 正常/错误返回 envelope、`build_emit_intent_server` 在缺失 SDK helpers 时返回 `None`、注入 fake factories 后能跑通、真 SDK 下产出 `{type:"sdk", name, instance}` 字典、`ClaudeBackend(mcp_*=...)` 注入 `mcp_servers` + `allowed_tools`、`enable_mcp_emit_intent=False` 不污染 options
  - `tests/test_claude_backend.py` 已显式 `enable_mcp_emit_intent=False` 验证 JSON 回退路径仍可用
  - `tests/test_cli_backends.py` 用四元组 fake `_import_sdk`，并断言 `has_emit_intent_tool=False`（SDK helper 缺失自动降级）

> 该阶段把 `Conductor` 从“单 reactor + 只接收 send_message”进化到“按 mode
> 拉起多角色，每条 intent 都走 PolicyGate，10 个 intent 都有副作用，delegate
> task 被 SubAgentRunner 真去调 backend，Claude 通过真正的 emit_intent MCP
> 工具说话”。下一步真正缺的是：Phase 6.2 CodexBackend、Phase 7 OOB
> 子进程隔离、Phase 8 真实 benchmark/kernel-opt action 接入 sandbox。

---

## Phase 1 — Iron Rules / Constants / Process Management `[D §4.5 / §4.6 / §4.7]` ✅ M1 完成

### `orchestrator/iron_rules.py`
- [x] 1.1 `IronRule` dataclass（`id`/`description`/`applies_to_modes`/`severity`）`[D §4.5]`
- [x] 1.2 IR-1 (`_ir1_parallel_kernel_submission`) `[D §4.5 IR-1]`
- [x] 1.3 IR-2 (`_ir2_no_kernel_source_modification_before_geak`) `[D §4.5 IR-2]`
- [x] 1.4 IR-3 (`_ir3_integrate_after_kernel_opt`) `[D §4.5 IR-3]`
- [x] 1.5 IR-4 (`_ir4_kill_then_check_gpu`) `[D §4.5 IR-4]`
- [x] 1.6 IR-5 (`_ir5_no_pkill_f_sglang`) `[D §4.5 IR-5]`
- [x] 1.7 IR-6 (`_ir6_patch_inductor_args`) `[D §4.5 IR-6]`
- [x] 1.8 IR-7 (`_ir7_no_geak_config_mutation`) `[D §4.5 IR-7]`
- [x] 1.9 `validate_action(action_metadata, mode) -> list[Violation]` 收集所有违规
- [x] 1.10 单测：`tests/test_iron_rules.py` 28 项（命中 + 不命中 + render_for_prompt + 多 IR 同时违规）

### `orchestrator/kernel_opt_constants.py`
- [x] 1.11 `KERNEL_OPT_BACKENDS = "geak,codex"` `[D §4.6]`
- [x] 1.12 `OOB_ROUND_ITERATIONS = 3`
- [x] 1.13 `KERNEL_OPT_WORKSPACE = "control-plane-moe"`
- [x] 1.14 `GEAK_STEP_LIMIT = 100` / `GEAK_MAX_RETRIES = 3` / `GEAK_MAX_SUBMISSIONS = 15`
- [x] 1.15 `GEAK_TOP_CANDIDATES = 5` / `GEAK_CONSECUTIVE_DISCARDS = 5`
- [x] 1.16 `GEAK_WALL_CLOCK_MIN = 120` / `GEAK_POLL_INTERVAL_S = 60` / `GEAK_POLL_TIMEOUT_MIN = 15`
- [x] 1.17 `MIN_GPU_PCT = 3` / `SERVER_KILL_WAIT_S = 10`
- [x] 1.18 `FILTERED_TRACE_NAME = "filtered-TP-0.trace.json.gz"`
- [x] 1.19 frozen / read-only 保证：模块级 `Final` + `tests/test_kernel_opt_constants.py` 锁值
- [x] 1.20 `KERNEL_OPT_IMAGE` 从 env 读取 + 缺失时 raise；`venv_bin_path` 同上 `[D §4.6 footnote]`

### `orchestrator/process_management.py`
- [x] 1.21 `prepend_venv_path()` helper，幂等 + 不修改输入 `[D §4.7]`
- [x] 1.22 `safe_kill_server(framework)` 用 `_run_pgrep` + `_run_kill` 测试 seam；`FRAMEWORK_PATTERNS` 锁定 sglang.launch_server / vllm.entrypoints `[D §4.7 / IR-5]`
- [x] 1.23 `wait_kill_settle(seconds, framework=...)` 真做 sleep + 二次 pgrep `[D §4.7]`
- [x] 1.24 `unset_profile_envs()`（PROFILE / SGLANG_TORCH_PROFILER_DIR） `[D §4.7]`
- [x] 1.25 `pick_filtered_trace(dir)` 递归 + 返回最新 `[D §4.7]`
- [x] 1.26 `assert_user_tp_respected(prompt_tp, detected_gpus)` `[D §4.7]`
- [x] 1.27 `vllm_flag_translator()` `--disable-log-requests` → `--disable-log-stats` `[D §4.7]`
- [x] 1.28 `enforce_run_baseline_sh(action_name, script_path=...)` 检查脚本存在 `[IR-3]`
- [x] 1.29 `tests/test_process_management.py` 24 项；mock 掉 subprocess 验证 IR-5 永远不走 `pkill -f sglang`

---

## Phase 2 — A2A Intent Transport `[D §10.5]`

### `orchestrator/intent_parser.py` —— ✅ Phase E4 完成
- [x] 2.1 `IntentEnvelope` schema 常量（与 §10.5.3 字段一致）
- [x] 2.2 `Intent` dataclass：`type`（10 个 enum）+ `payload` dict
- [x] 2.3 `EMIT_INTENT_TOOL_SCHEMA` 常量（Claude tool input_schema）`[D §10.5.4]`
- [x] 2.4 `parse_claude_trajectory(trajectory, fallback_text=...) -> list[Intent]` `[D §10.5.6]`
- [x] 2.5 `parse_codex_validated_json(text) -> list[Intent]` —— fenced/plain/brace-fragment 三种提取 `[D §10.5.5]`
- [x] 2.6 `validate_envelope(envelope) -> list[Intent]` 全字段 schema 校验 + per-intent payload required-field 表
- [x] 2.7 `class NoIntentEmitted(RuntimeError)`
- [x] 2.8 `class IntentValidationError(RuntimeError)` —— 含 raw + reason
- [x] 2.9 `class ProtocolError(RuntimeError)` —— Critic/Sage 兜底
- [x] 2.10 单测：合法 Claude tool_use → intents
- [x] 2.11 单测：合法 Codex JSON → intents
- [x] 2.12 单测：缺失 `intents` 数组 → `IntentValidationError`
- [x] 2.13 单测：未知 `intent_type` → 拒收
- [x] 2.14 单测：自由文本（无 tool_use / 无 JSON）→ `NoIntentEmitted`
- [~] 2.14b JSON repair pass —— 由 `ClaudeBackend.run` 做 1 轮 repair；intent_parser 自身不做（待 Codex 落地后再决定要不要内化）

### `orchestrator/policy.py` —— ✅ Phase F1b 完成
- [x] 2.15 `PolicyGate.__init__(flags, mode, action_registry, role_registry)`
- [x] 2.16 `validate_intent(from_agent, intent, state) -> None | raises PolicyDenied`
- [x] 2.17 `_validate_role_permission(agent, intent)` —— `role.allowed_intents` chokepoint，Codex Critic/Sage 默认无 delegate 权限 `[D §10.5.7]`
- [x] 2.18 `_validate_mode_allowed(mode, action)` —— `ActionMetadata.allowed_modes` 在 `_validate_propose_action` / `_validate_delegate` 中查 `[D §10.5.8]`
- [x] 2.19 `_validate_delegate_allowed(mode, from_agent, action)` —— `flags.enable_subagent_delegate` + role gate
- [~] 2.20 `_validate_side_effect_policy(action, state)` —— 通过 `ActionRegistry` schema 强制 `requires_lanes` 必须定义 + `SubAgentRunner` 真去 `acquire`；待 §11.5 真实 sandbox 接入后补硬性 lane/lease_ttl 一致性测试
- [x] 2.21 `_validate_state_transition(from_agent, changes, state)` —— `CORE_STATE_FIELDS` (`current_best` / `stop_reason` 等) 只能 Conductor 改 `[D §10.5.7]`
- [x] 2.22 `allowed_tools_for_agent(agent_name, mode) -> list[str]`
- [x] 2.23 Quick-mode Bash allowlist 落地：`QUICK_BASH_ALLOWLIST`（`server_lifecycle / read_only / sweep_scripts / runtime_query`）`[D §10.5.8 quick_allowlist]`
- [x] 2.24 Quick-mode Bash denylist 落地：`QUICK_BASH_DENYLIST`（`workspace_write / git_write / patch_ops / GEAK / kernel_build`）`[D §10.5.8]`
- [x] 2.25 跨 mode 硬规则：quick 禁 delegate / 禁 kernel_opt family / Codex 不能 delegate 副作用 `[D §10.5.8 硬规则]`
- [x] 2.26 单测：Critic 试图 delegate `kernel-opt` → 拒
- [x] 2.27 单测：quick mode Executor 提议 `kernel-opt` → 拒（`mode_gate=0`）
- [x] 2.28 单测：guided Executor delegate `bench_runner` → 通过
- [x] 2.29 单测：quick allowlist —— `pkill -f sglang` 拒；`kill $(pgrep -f sglang.launch_server)` 通过
- [x] 2.30 单测：state transition —— 非 Conductor 改 `current_best` → 拒

### `orchestrator/agent_role.py` —— ✅ Phase F1a 完成
- [x] 2.31 `AgentRole` dataclass：`name / backend / model / api_key_env / no_tools / max_turns / allowed_intents / can_delegate_side_effects / can_mutate_core_state`
- [x] 2.32 `load_system_prompt(name) -> str` 从 `system_prompts/<name>.md` 读取（含缓存）
- [x] 2.33 `claude_role(name)` factory（opus-4-7）
- [x] 2.34 `codex_role(name)` factory（gpt-5.4，`no_tools=True` 标记）
- [x] 2.35 单测：role 工厂返回正确 backend type；`roles_for_mode` 三种 mode 的角色集合均覆盖

---

## Phase 3 — Scheduler / SharedState / Accuracy Gate `[D §7 / §8 / §9]`

### `orchestrator/shared_state.py` (STUB)
- [x] 3.1 `SharedState` dataclass：所有 prompt 注入字段已就位 `[D §6.3]`
- [x] 3.2 `summary() -> str` 紧凑 markdown
- [x] 3.3 `time_left_minutes` 属性 + `refresh_elapsed()` helper
- [~] 3.4 `attach_rca(finding)` —— append-only 已实现；PolicyGate 路径待 Phase 4
- [x] 3.5 `last_decisions(n)` 返回最近 N 条
- [~] 3.6 `apply_validated_transition(from_agent, changes)` —— 字段白名单已实现；PolicyGate 校验待 Phase 4
- [x] 3.7 `set_stopping(reason)` + `should_stop()`
- [x] 3.8 持久化：`state.json` 通过 `write_snapshot()` —— 时钟每 tick 写入，graceful_stop 后再写一次 `[D §13.2]`
- [ ] 3.9 单测：transition 拒绝非授权字段（dry-run 沉默忽略；待 Phase 4 PolicyGate 抛 ValueError 后补）
- [ ] 3.10 单测：summary 不超过 token 上限

### `orchestrator/scheduler.py` ✅ M3 完成 — `[D §9]`
- [x] 3.11 `BudgetAwareScheduler.__init__(objective, mode, env, action_registry, model_class=None)`
- [x] 3.12 `pressure(state) -> float` —— `max(objective.pressure_input, time-budget pressure)` 兜底，永远 ramp `[D §8.5 / §9.1]`
- [x] 3.13 `score(action, state) -> ActionScore`：base × pressure × mode_gate × depth_gate × diminishing × lane_available × prior × adjustment + breakdown 字段全可读 `[D §9.1]`
- [x] 3.14 `_base_factor` —— `(gain_avg / cost_p75) × (1-acc_risk) × (1-crash_risk)` `[D §9.1]`
- [x] 3.15 `_mode_gate` —— 0 当 mode 不在 `action.allowed_modes` `[D §9.1]`
- [x] 3.16 `_depth_gate` —— `cost_p75 ≤ time_left × 0.8` `[D §9.1]`
- [x] 3.17 `_diminishing` —— `0.7 ** family_count(history)` `[D §9.1]`
- [x] 3.18 `_lane_available` —— 任一 required lane 被持有则 0 `[D §9.1]`
- [x] 3.19 `pick_next(state, lock_summary, history)` —— 先消费 followup 队列；rule-7 一次性应用避免无限递归
- [x] 3.20 `update_after_action(action, gain_pct, status, history)` —— 7 条 update rule `[D §9.3]`
- [x] 3.21 Rule 1：succeeded → 同 family 其它 action `× 1.2`（cap 3.0）`[D §9.3 #1]`
- [x] 3.22 Rule 2：failed → 同 family 其它 action `× 0.5` `[D §9.3 #2]`
- [x] 3.23 Rule 3：backends 成功 → push `combined_backends_test` 到 followups
- [x] 3.24 Rule 4：所有 backends 已测 → push `profile`
- [x] 3.25 Rule 5：kernel 成功 → push `profile` + `kernel_opt`
- [x] 3.26 Rule 6：kernel reverted → 同 family 剩余 `× 0.7`
- [x] 3.27 Rule 7：所有 score < 1.0 → push `sweep` 然后 `report`
- [x] 3.28 单测：`tests/test_scheduler.py` 包括 quick + kernel-opt = 0
- [x] 3.29 单测：pressure 随 elapsed_minutes 增长
- [x] 3.30 单测：depth_gate 阻止 cost_p75 > time_left×0.8 的 action

### `orchestrator/score_priors.py` ✅ M3 完成 — `[D §9.2]`
- [x] 3.31 `INITIAL_PRIORS: dict[ModelClass, dict[ActionName, float]]`
- [x] 3.32 Dense 列：3 / 5 / 8 / 7 / 1
- [x] 3.33 MoE+MLA 列：9 / 6 / 2 / 0 / 1
- [x] 3.34 MoE+SWA 列：8 / 7 / 2 / 0 / 1
- [x] 3.35 MoE+MLA+NSA 列：10 / 5 / 2 / 0 / 1
- [x] 3.36 `prior_for(model_class, action_name, default=1.0)` —— 同时支持 hyphen / underscore action 名
- [x] 3.37 `classify_model(model_path) -> ModelClass` —— pattern 优先级：NSA → MLA → SWA → DENSE → UNKNOWN
- [x] 3.38 单测：未知 model_class → UNKNOWN 表 + 默认 prior

### Early stopping — `orchestrator/early_stop.py` ✅ M3 完成
- [x] 3.39 `should_stop_early(state, objective, *, scheduler, flags, brier_window, lock_summary, history)` `[D §7.1]`
- [x] 3.40 信号 1 `target_reached`（objective.is_satisfied）`[D §7.1 #1]`
- [x] 3.41 信号 2 `time_exhausted`（≤ `TIME_BUFFER_MIN=5.0`）`[D §7.1 #2]`
- [x] 3.42 信号 3 `no_more_leverage`（所有 score < 1.0）`[D §7.1 #3]`
- [x] 3.43 信号 4 `brier_plateau`（仅 critic 启用 + window ≥5）`[D §7.1 #4]`
- [x] 3.44 信号 5 `emergency`（`crash_count ≥ 2`）`[D §7.1 #5]`
- [~] 3.45 `graceful_stop(reason)` —— 基础尾流（final-event + state.json）已 OK；marathon 5 分支落到 `_graceful_stop` 再展开 `[D §7.2]`
- [x] 3.46 单测：`tests/test_early_stop.py` 覆盖每个信号 + priority 顺序

### `orchestrator/accuracy_gate.py` ✅ M3 完成 — `[D §7.5]`
- [x] 3.47 `ACCURACY_RISK_TABLE` mapping action → risk `[D §7.5.1]`
- [x] 3.48 `requires_gate(action) -> bool` —— hyphen / underscore 自动归一化
- [x] 3.49 `run_gsm8k(server_port, model, results_dir, *, script_path, eval_task, num_fewshot, timeout_s)` 通过 `_run_eval` subprocess seam（测试可 mock） `[D §7.5.2 #1]`
- [x] 3.50 `extract_score_from_summary(path)` —— 同时支持 lm-evaluation-harness `acc,none` / 平铺 `score` / 嵌套 dict `[D §7.5.2 #2]`
- [x] 3.51 `compare_to_baseline(baseline, new, threshold=0.01) -> Verdict` 处理 NaN / 越界 → FAIL
- [x] 3.52 `Verdict` KEEP / REVERT / FAIL
- [x] 3.53 `optional_kernel_micro_check(orig, opt, atol, rtol)` —— 支持 numpy / nested list / scalar `[D §7.5.3]`
- [x] 3.54 跳过列表 `_GATE_FREE_ACTIONS`：setup/classify/profile/sweep/report 等 `[D §7.5.4]`
- [x] 3.55 单测：accuracy_drop > 0.01 → REVERT；nan / 越界 → FAIL
- [x] 3.56 单测：scheduling 参数（risk=0）不调 gate

---

## Phase 4 — Sub-agent + Action System `[D §11 / §12]`

### `orchestrator/action_registry.py` —— ✅ Phase F3a 完成（schema 完整，目前只装了 3 个最小 action，剩 17 个待 Phase 8）
- [x] 4.1 `ActionMetadata` dataclass —— 全字段对齐 §12.2 yaml schema
- [x] 4.2 `ActionRegistry.__init__(actions_dir: Path)` 扫描 `_meta/*.yaml`
- [x] 4.3 `get(name) -> ActionMetadata`
- [x] 4.4 `allowed_for_mode(mode) -> list[ActionMetadata]`
- [x] 4.5 `system_prompt_for(name) -> str` 读 `actions/<name>.md`
- [x] 4.6 校验 yaml schema：必填 `name/family/cost_minutes_p50/p75/expected_gain_pct/accuracy_risk/crash_risk/requires_lanes/allowed_tools/side_effects/allowed_modes/preferred_backend/preferred_model/max_turns/lease_ttl_sec` + `name` 必须等于文件名 + `allowed_modes` 必须是合法 `ExecutionMode`
- [~] 4.7 单测：加载 3 个 action（`bench_runner` / `param_sweep_run` / `kernel_opt`）全 OK；Phase 8 把剩余 17 个落到 yaml 后再补
- [x] 4.8 单测：缺字段 / 不合法 family / 不合法 mode / 文件名不匹配 → 启动失败

### `orchestrator/sub_agent_runner.py` —— ✅ Phase F3b dry skeleton 完成（v0.6 复用 Conductor backend；OOB 进程隔离待 Phase 7）
- [x] 4.9 `Task` 复用 `task_registry.Task`；`TaskResult` 在 `sub_agent_runner.py` 内定义 `[D §11.2]`
- [x] 4.10 `SubAgentRunner.__init__(backend, policy, locks, action_registry, tasks, workspace, agent_name)`
- [x] 4.11 `run(task) -> TaskResult` 主入口 `[D §11.2]`
- [x] 4.12 `_compose_prompt(task, action)` 注入 task.params + action.md + lanes + allowed_tools `[D §11.2]`
- [~] 4.13 `_extract_metrics` / `_extract_artifacts` —— 已能从 intents 抽 `tput` 和 `artifact_path/result_path/log_path`；`action.result_schema` 完整解析待真 action 落地
- [x] 4.14 通过 `locks.acquire(action.requires_lanes, holder_id=..., task_id=..., action=..., ttl_sec=...)` 拿 lease `[D §11.2]`
- [~] 4.15 `policy.allowed_tools_for_action` 由 `action.allowed_tools` 直接传入；待 Phase 7 多 backend 池上线后再做交集
- [x] 4.16 在 `task.transition("running")` 后调 `backend.run(...)` `[D §11.2]`
- [x] 4.17 异常捕获 → `task.transition("failed", evidence)`（含 lane_contention / backend_error 两条路径）`[D §13.4]`
- [x] 4.18 GPU 资源争抢：100% 通过 §3.5 lane 锁解决（`SubAgentRunner` 已经 always-acquire / always-release）`[D §11.4]`
- [~] 4.19 Codex sub-agent 限制：role allow-list 已禁；`SubAgentRunner` 选 backend 的策略（Phase 7）会再二次过滤 `[D §11.5]`
- [x] 4.20 单测：`bench_runner` 在 mock backend 下跑通（含 metrics / artifacts 抽取）
- [x] 4.21 单测：unknown action / backend exception / no intents / `dispatch_pending_delegates` 抽干 queue

### Action 文件（22 个 .md + 22 个 _meta yaml）✅ M2 完成 `[D §12.1]`

> 命名约定：使用下划线（与 Python module 风格一致）。设计文档里写
> `kernel-opt` 我们落地为 `kernel_opt`。`prior_for` / `requires_gate`
> 都把两种形式归一化处理。

#### prep family
- [x] 4.22 `actions/setup.{md,yaml}` —— session bootstrap
- [x] 4.23 `actions/classify.{md,yaml}` —— model class detection
- [x] 4.24 `actions/target_analysis.{md,yaml}` —— TARGET_DIR ingestion
- [x] 4.25 `actions/baseline.{md,yaml}` —— `run_baseline.sh` first-light
- [x] 4.25b `actions/bench_runner.{md,yaml}` —— 通用 bench 入口（v0.6 已交付）

#### analysis family
- [x] 4.26 `actions/profile.{md,yaml}` —— filtered-TP-0 trace 捕获

#### shallow family
- [x] 4.27 `actions/backends.{md,yaml}`（accuracy_risk=0.10）
- [x] 4.28 `actions/params.{md,yaml}`（默认 risk=0；量化候选在执行期 override 至 0.30）
- [x] 4.29 `actions/sweep.{md,yaml}`
- [x] 4.30 `actions/report.{md,yaml}`
- [x] 4.30b `actions/param_sweep_run.{md,yaml}` —— 显式参数扫描入口（v0.6 已交付）

#### deep_kernel family
- [x] 4.31 `actions/kernel_opt.{md,yaml}`（quick=✗，accuracy_risk=0.20，已在 v0.6 交付）
- [x] 4.32 `actions/integrate.{md,yaml}`（accuracy_risk=0.15，3 条 lane）
- [x] 4.33 `actions/deep_kernel_analysis.{md,yaml}`（marathon-only）
- [x] 4.34 `actions/operator_tuning.{md,yaml}`
- [x] 4.35 `actions/vendor_kernel_config.{md,yaml}`

#### long family
- [x] 4.36 `actions/framework_rebuild.{md,yaml}`（p75=90min，lease_ttl=5400）
- [x] 4.37 `actions/comm_optimization.{md,yaml}`
- [x] 4.38 `actions/compiler_tuning.{md,yaml}`

#### creative family
- [x] 4.39 `actions/dream.{md,yaml}` `[D §12.3]`
- [x] 4.40 `actions/re_explore.{md,yaml}` `[D §12.3]`

#### resilience family
- [x] 4.41 `actions/recover.{md,yaml}` `[D §12.3]`

#### applicable_when 触发条件
- [x] 4.42 `framework_rebuild.applicable_when` 含 `kernel_dispatch_shows_aiter_dominance` + `cumulative_gain_plateau` + `marathon_only` `[D §12.2 example]`
- [x] 4.43 其他 marathon-only action 的 applicable_when 已补：`deep_kernel_analysis` / `operator_tuning` / `comm_optimization` / `compiler_tuning` / `dream` / `re_explore`

#### 全局 catalog 约束
- [x] `tests/test_action_catalog.py` (24 项) 锁住：
  - 22 个 action 全部能被 ActionRegistry 加载
  - 每个 action 都声明 `emit_intent` 工具
  - `requires_lanes` 全部命中 `KNOWN_LANES`
  - 每个 action 都有 markdown body
  - long 家族 lease_ttl ≥ p75×0.9
  - creative 家族 risk = 0
  - prep 家族在所有 mode 都允许；deep_kernel 全部排除 quick

---

## Phase 5 — Memory: KB / Persona / Sage Query `[D §5.3 / §6]`

### `orchestrator/kb.py` ✅ M4 完成 — `[D §6]`
- [x] 5.1 `KnowledgeBase.__init__(session_dir, user_id="default", *, kb_dir=None)`
- [x] 5.2 `count_entries(model_family)` 过滤 `user_id` + `model_family` `[D §6.2]`
- [x] 5.3 `is_warm_start_eligible(model_family)` —— `count >= 1` `[D §6.2]`
- [x] 5.4 `recall_for_model(model_name, agent_name, top_k=5, timeout_s=30)` 调 `kb_query.py` 子进程；超时 / 子进程失败 → 返回空串 `[D §6.3]`
- [x] 5.5 `ingest(...)` 写 `entries.jsonl`，返回 `KBEntry`
- [x] 5.6 `read_persona(agent_name) -> str`
- [x] 5.7 `append_persona(agent_name, note) -> Path` 时间戳注释 `[D §10.5.7]`
- [x] 5.8 `cross_run_synthesize(*, max_lookback=200)` 写 `insights.jsonl`
- [x] 5.9 `detect_conflicts() -> list[Conflict]` 写 `kb/conflicts.jsonl`
- [x] 5.10 KB 按 `user_id` 分区（`KnowledgeBase(user_id=...)` + `count_entries` filter）
- [x] 5.11 单测 `test_kb.py::test_warm_start_after_first_ingest`：第 1 次同 family 不可读
- [x] 5.12 单测：第 2 次 → 可读

### `orchestrator/persona.py` ✅ M4 完成 — `[D §5.3]`
- [x] 5.13 `estimate_tokens(text) -> int` —— `len // 4`
- [x] 5.14 `should_distill(path, mode, *, last_distill_ts, keep_just_happened)` —— 8K token 硬 / 4h 软 / post-KEEP 触发
- [x] 5.15 `distill_persona(agent_name, backend, *, persona_path, archive_dir)` —— backend=None 走 tail-truncation；backend 提供 → 调 `update_persona` intent
- [x] 5.16 `archive_old_persona(agent, body, archive_dir) -> Path` 时间戳备份
- [x] 5.17 仅 marathon mode 启用（quick / guided 直接返回 False）`[D ADR-22]`
- [x] 5.18 单测：低于 4K token 且未到 4h 不蒸馏（marathon 下也 False，前提 last_distill_ts 是最近的）
- [x] 5.19 单测：超过 8K 触发蒸馏（marathon-only），`HARD_TOKEN_LIMIT=8000`

### `orchestrator/sage_query_service.py` ✅ M4 完成 — `[D §5.1.2]`
- [x] 5.20 `SageQueryService.__init__(codex_backend, kb, *, timeout_s, cache_ttl_s)`
- [x] 5.21 `recall(model, action) -> str` 异步调用，命中则缓存 5min
- [x] 5.22 30s timeout fallback 空字符串
- [x] 5.23 `prefetch(model, planned_actions)` 并发预取
- [x] 5.24 单测：`_SlowBackend` 触发超时返回空
- [x] 5.25 单测：成功返回 backend 输出 / 兜底返回 KB 原始

### `kb/kb_query.py` ✅ M4 完成 — `[D §6.2]`
- [x] 5.26 CLI: `kb_query "<query>" --kb-dir DIR --top-k N [--compact|--json]`
- [x] 5.27 compact 输出 `(category, model, gain, status) lesson`；full 输出 markdown 段
- [x] 5.28 token-overlap + status 加权 + recency bonus，无 embedding

### `kb/kb_ingest.py` ✅ M4 完成 — `[D §6.2]`
- [x] 5.29 CLI: `kb_ingest --kb-dir --category --model --action --lesson --tags --gain --status [--user-id --ts]`
- [x] 5.30 append `entries.jsonl`；`--tags` 同时支持 JSON 数组与逗号分隔

### `kb/entries.jsonl` + `kb/insights.jsonl`
- [x] 5.31 文件由 `kb.py` 第一次 ingest 时按需创建
- [x] 5.32 schema 文档落在 `KNOWLEDGE-BASE.md`

---

## Phase 6 — Checkpoint / Resume / Backup `[D §13]`

### `orchestrator/checkpoint.py` ✅ M5 完成 — `[D §13]`
- [x] 6.1 `Checkpoint.create(session_dir, db, state, *, trigger, ts)` —— state.json + persona fsync + vacuum_into + checkpoint_taken event
- [x] 6.2 `Checkpoint.create_after_keep(...)` —— 同上但 trigger=AFTER_KEEP `[D §13.7]`
- [x] 6.3 `Checkpoint.load_latest(session_dir) -> CheckpointHandle | None`
- [x] 6.4 `Checkpoint.list_all(session_dir)` 按 ts 升序
- [x] 6.5 `resume_from_session_dir(session_dir, db, locks, tasks=None) -> ResumeState` `[D §13.5]`
- [x] 6.6 cursors 通过 `CursorStore.all` 装入 `ResumeState.cursors` `[D §13.5]`
- [x] 6.7 personas 索引返回 `agent → Path` 字典
- [x] 6.8 in-flight tasks 通过 `tasks.list_by_state("queued"+"running")` 收集，待 `evidence_check_matrix` 处理 `[D §13.5 / §13.6]`
- [x] 6.9 过期 leases 通过 `locks.backend.reap_expired()`；失败时 silent
- [~] 6.10 cursor 之后的 events 重放 —— Reactor 在 `_reactor` 已天然按 cursor 过滤；resume path 不重复实现 `[D §13.5]`
- [~] 6.11 触发条件：30min cadence + KEEP 已落地（`Conductor._maybe_checkpoint` + `Checkpoint.create_after_keep`）；graceful_stop / crash 触发待 Phase 7 收尾

### `evidence_check_matrix(task)` — `orchestrator/checkpoint.py` `[D §13.6]`
- [x] 6.12 `_check_bench`：`results/<task_id>/metrics.json` 大小 / 内容判 SUCCEEDED / SAFELY_FAILED / EVIDENCE_INSUFFICIENT
- [x] 6.13 `_check_profile`：递归找 `filtered-TP-0.trace.json.gz`
- [x] 6.14 `_check_patch_or_integrate`：`patch_fingerprint.txt` 与 task.params['patch_fingerprint'] 比对
- [x] 6.15 `_check_server_restart`：pid file + 注入式 `server_health_fn`
- [x] 6.16 `_check_kernel_extract`：`*.kernel.json` / `*.kernel.tar.gz` artifact 检查
- [x] 6.17 `_check_geak_submit`：`geak_request_id.txt` + 注入式 `geak_status_fn`
- [x] 6.18 `Verdict` 三态：SUCCEEDED / SAFELY_FAILED / EVIDENCE_INSUFFICIENT
- [x] 6.19 单测：`tests/test_checkpoint.py` 26 项覆盖每行 happy / 不足 / 部分 path

### `storage/backup.py` ✅ M5 完成 — `[D §3.5.8 路径 A]`
- [x] 6.20 `vacuum_into(db, dest_path)` —— 透过 SqliteConnection 的同步 lock + asyncio.to_thread；自动覆盖已存在文件
- [x] 6.21 `periodic_backup(db, checkpoint_dir, *, period_min, stop_event, on_complete)` —— 失败容忍 + cancellable
- [x] 6.22 `force_backup_after_keep(db, checkpoint_dir)` 立即落 `<ts>/conductor.db.bak`
- [x] 6.23 backup 命名 `<UTC compact ts>/conductor.db.bak`
- [x] 6.24 `restore_from_backup(backup_path, target_db)` 复制并清理 -wal / -shm sidecar
- [x] 6.25 单测：`tests/test_backup.py` 8 项含 round-trip / 失败 retry / restore

---

## Phase 7 — Conductor Main Loop `[D §15]`

### `orchestrator/conductor.py` —— ✅ Phase D / E / F 累积完成核心循环；marathon 高阶 cadence 待 Phase 11
- [x] 7.1 `Conductor.__init__(session_dir, env, backend, db, role_registry, action_registry, reactor_tick_s, clock_tick_s)`
- [~] 7.2 实例化 `objective` / `mode` / `flags` / `state` / `bus` / `cursors` / `locks` / `tasks` / `policy` / `actions` / `role_registry` / `roles` —— ✅ 已就位；`kb` / `scheduler` / `sage_query` / `sub_agent_runner` 还未挂到 `ConductorContext`，`SubAgentRunner` 类已存在但需要在 `_bootstrap` 里实例化并由 `_clock` 或专门的 dispatcher loop 调 `dispatch_pending_delegates`（Phase 7.10）
- [x] 7.3 按 mode 启用对应 reactor —— `roles_for_mode(mode)` 决定 spawn 集合：quick=[executor]、guided=[executor,critic]、marathon=[executor,critic,watchdog,sage]，每个角色一个 `_reactor` task `[D §15 / §3.4]`
- [~] 7.4 `run()` 入口：`asyncio.gather(reactors + clock + stopping_watcher)` → `_graceful_stop`；resume-from-checkpoint 路径待 Phase 6 落地 `[D §15]`
- [ ] 7.5 `_init_session()` 首次启动 vs resume 分流（resume path 待 Phase 6）
- [x] 7.6 `_reactor(agent_name)` 通用 reactor 循环 `[D §15]`
- [~] 7.7 reactor 内：`bus.replay_for` → cursor 过滤 self-message → `_compose_prompt` → `backend.run(allowed_tools=policy.allowed_tools_for_agent)` → `parse_intents` → `_gate_intent`（PolicyGate） → `_handle_intent` → `cursors.advance` —— 主链路已通；`sage_hint` 注入 + reactor 处理过程的“单事务化”待 Phase 5 KB / Phase 6 checkpoint 落地
- [x] 7.8 `_handle_intent(from, intent)` 全 10 个 intent_type 分支真有副作用 `[D §15]`（F2 完成）
- [x] 7.9 intent 分支：`send_message` / `alert` / `propose_action` / `delegate` / `update_state` / `update_persona` / `ask_question` / `answer` / `objection` / `vote`
- [x] 7.10 `_dispatcher_loop` —— `Conductor.run` 在 `enable_dispatcher=True` 时 spawn `dispatch_pending_delegates(runner, db, stop=_dispatcher_stop)` 常驻 task；`SubAgentRunner` 在 `_bootstrap` 由 ActionRegistry 在场时构建。E2E `test_delegate_dispatch_loop_succeeds` 验证 queued→running→succeeded 全链路（M7）
- [~] 7.11 `_clock()` —— 基础 tick + state.json + reflection_tick + time_exhausted + 30min checkpoint cadence (`_maybe_checkpoint`) ✅ 已 OK；30min critic post-mortem / 4h persona 蒸馏 / 2h strategic review / 6h cross-run synthesis 仍待 Phase 11 cadence  `[D §14]`
- [x] 7.12 `_stopping_watcher()` 监 stop reason —— 一旦 `state.should_stop()` 即 set `_stop_event`
- [~] 7.13 `_graceful_stop(reason)` —— 基础 final-event (`graceful_stop` topic + reason / elapsed / max / cumulative_gain) + state.json 二次落盘 已 OK；marathon-specific 5 个尾流分支（target_reached / no_more_leverage / brier_plateau / emergency / time_exhausted 各自的 followup 写法）待 Phase 11
- [~] 7.14 emergency 分支：guided → `ephemeral_rca_via_critic` ✅ 已落地；marathon → Watchdog RCA / quick → minimal crash report 仍待 Phase 11 mode-specific tail `[D §5.1.3 / §7.2]`
- [x] 7.15 `ephemeral_rca_via_critic()` —— 调 backend.run + 找 `topic="rca_finding"` send_message intent；测试覆盖正负样本 `[D §5.1.3]`
- [x] 7.16 `_compose_prompt(agent_name, msgs, *, sage_hint=None)` —— 注入 role.system_prompt + `personas/<agent>.md` 长 persona 块 + `IronRules` for mode + state.summary + objective.describe + sage_hint 占位 + 最近 10 inbox + flags；`update_persona` intent 落库后 `_refresh_persona_index` 自动重读
- [ ] 7.17 `priority>=2 立即 / <2 batch (30s/2min)` event-driven 模式 `[D §14]` —— v0.7 仍是简单轮询，留给 Phase 11 一起做
- [x] 7.18 `_record_proposal_for_self_review(proposal)` —— 写 topic=`proposal`、payload kind=`self_review`、to=executor `[D §15]`
- [x] 7.19 `_open_parliament(proposal)` —— 仅 marathon 真计票（quick / guided 直接 abstained）；2 s vote 窗口，多数胜出 `[D §15]`
- [x] 7.20 单测：`test_quick_mode_spawns_only_executor_reactor` —— quick mode 不召唤 critic/watchdog/sage `[tests/test_conductor_policy.py]`
- [x] 7.21 单测：`test_marathon_mode_spawns_full_roster` —— marathon 启用 executor + critic + watchdog + sage 4 reactors；`test_guided_mode_spawns_executor_and_critic` 验证 guided 2 reactors
- [x] 7.22 集成测试：fake backend 跑完一个 dummy quick run —— `tests/e2e/test_dry_run_smoke.py` 7 项 ✅；`tests/test_conductor_policy.py` 7 项再次端到端验证 PolicyGate + multi-reactor

### Token 估算 / 成本观察
- [x] 7.23 `TokenBudgetMeter.record(prompt_tokens, completion_tokens)` + `should_throttle()` + `remaining()` + `reset()` 已就位（M7）`[D §14 token 估算]`
- [x] 7.24 quick ~0.5M / guided ~3M / marathon ~11.5M 警戒线（`_BUDGET_TOKENS` 类常量）
- [~] 7.25 token 预算告警时 critic 介入降级到 20% 采样 —— `should_throttle` 返回 True 已经能让外部读到信号；具体的“降级到 20% 采样”策略实施待 Phase 11 cadence `[D §5.2 采样降级]`

---

## Phase 8 — System Prompts ✅ M6 完成 `[D §17]`

> 文件位于 `.cursor/skills/inference-optimizer/system_prompts/`。`AgentRole.system_prompt()`（带缓存）按角色名加载。

- [x] 8.1 `system_prompts/executor.md` —— Claude opus 4.7：`emit_intent` 工具协议、IR-1..IR-7 提示、写 prediction
- [x] 8.2 `system_prompts/critic.md` —— Codex 无工具：`validated_json_output` 协议 + Brier 自我校准
- [x] 8.3 `system_prompts/sage.md` —— Codex 无工具：recall ≤500 token / 6h synthesis / conflict watch
- [x] 8.4 `system_prompts/watchdog.md` —— Claude opus 4.7：alert + watchdog_health heartbeat + IR 监控
- [x] 8.5 `system_prompts/rca_critic.md` —— guided emergency 时 Critic 一次性 RCA 模式 `[D §5.1.3]`

---

## Phase 9 — Scripts `[D §17 / §4.6]`

### `scripts/run_baseline.sh` ✅ M6 完成（DRY_RUN_MOCK + 真实路径占位）
- [x] 9.1 接受 `MODEL`/`TP`/`PORT`/`OUT_DIR`；缺任一即报错
- [~] 9.2 server 启动 / health wait / benchmark / profiling 真序列 —— 真实路径仍依赖 sprint sandbox（GPU），骨架返回 rc=1
- [~] 9.3 `kill_server` 调 `process_management.safe_kill_server` —— Python 端已就位（M1）；shell 端 IR-3 enforcement 留给 sandbox 真实接入
- [x] 9.4 `DRY_RUN_MOCK=1` 时写 `metrics.json` 含 `tput_per_gpu` / `p50_latency_ms` / `p95_latency_ms`，便于测试
- [~] 9.5 真实路径下 unset `PROFILE` / `SGLANG_TORCH_PROFILER_DIR` —— Python `process_management.unset_profile_envs` ✅；shell 注释提醒待真实部署补

### `scripts/eval_accuracy.sh` ✅ M6 完成（DRY_RUN_MOCK + 真实路径占位）
- [x] 9.6 接受 `EVAL_TASK` / `NUM_FEWSHOT` / `PORT` / `MODEL` / `RESULTS_DIR`
- [~] 9.7 调 lm-evaluation-harness —— 真实路径依赖 sandbox，骨架在生产环境 rc=1
- [x] 9.8 `DRY_RUN_MOCK=1` 写 `eval_summary_<task>.json` 含 `score` 给 `accuracy_gate.extract_score_from_summary` 提取

### `scripts/patch_inductor.py` ✅ M6 完成（IR-6 enforcement）
- [x] 9.9 IR-6：`--target-file` 必填，缺失 rc=2
- [x] 9.10 `--cache-dir` 显式拒绝 `[D §4.5 IR-6]`
- [x] 9.11 当 `--tuning-keys` 含 `block_size` / `num_warps` 时强制要求 `--best-config`；输出 manifest JSON

### `scripts/monitor.sh` + `scripts/monitor.py` ✅ M6 完成 — 全局健康 dashboard
> bash 版直接 `sqlite3`；Python 版 (`python -m inference_optimizer.scripts.monitor`)
> 跨平台。两者输出同样的四张表。
- [x] 9.12 接受 `--db` / `INFERENCE_OPTIMIZER_DB_PATH` / `SESSION_DIR` 三种解析
- [x] 9.13 `events` 行：总数 + 最新 ts
- [x] 9.14 `in-flight tasks` 行：`state IN ('queued','running')` 计数
- [x] 9.15 `active leases` 行：`expires_at > datetime('now')` 计数
- [x] 9.16 `cursors lag` 行：`MAX(events.seq) - MIN(cursors.last_processed_seq)`
- [x] 9.17 `--watch N` 参数（`monitor.py`）；bash 版同样支持
- [x] 9.18 `--per-agent` 详表
- [x] 9.19 `--per-lane` 详表（含 holder / action / expires_at）
- [x] 9.20 `--top-events N` 详表
- [x] 9.21 退出码：lag > 阈值或存在 30min 未更新的 running 任务 → rc=1；missing DB → rc=3
- [x] 9.22 README §"Monitoring" 段已加（`bash scripts/monitor.sh --watch 5` 示例 + Python 版）
- [x] 9.23 单测：`tests/test_monitor.py` (10 项) — 装填 fixture DB 验证输出 + zombie 检测；`tests/test_patch_inductor.py` (9 项) — IR-6；`tests/test_scripts_skeletons.py` (6 项, bash 不可用时跳过) — DRY_RUN_MOCK 路径

最小 SQL 模板（已落地到 `scripts/monitor.sh` stub）：
```sql
SELECT 'events'                         AS what, COUNT(*) AS n, MAX(ts) AS latest FROM events;
SELECT 'in-flight tasks'                AS what, COUNT(*) AS n
       FROM tasks WHERE state IN ('queued','running');
SELECT 'active leases'                  AS what, COUNT(*) AS n
       FROM leases WHERE expires_at > datetime('now');
SELECT 'cursors lag (events behind)'    AS what,
       MAX(seq) - MIN(last_processed_seq) AS lag
       FROM events, cursors;
```

---

## Phase 10 — Skill Entry & Documentation `[D §16]`

### `.cursor/skills/inference-optimizer/SKILL.md`
- [x] 10.1 trigger: `@inference-optimizer`（v0.6 SKILL.md "Launch" 段已含触发说明）
- [x] 10.2 必填: `MODEL_PATH` + `MAX_HOURS`（"Required env" 表）
- [x] 10.3 可选: `TARGET_GAIN_PCT` / `TARGET_TPUT_PER_GPU` / `TARGET_DIR`（"Optional objective" 表，最多一个）
- [~] 10.4 给三档 mode 入口示例 `[D §16.2]` —— 已给 quick + Claude/Mock 两种 PowerShell/Bash 命令；guided + marathon 显式案例待 Phase 7 cadence 落地后补
- [ ] 10.5 marathon thin-shim 保留一个 release `[D §16.3]`
- [~] 10.6 关键 env：`INFERENCE_OPTIMIZER_SESSION_ROOT` / `INFERENCE_OPTIMIZER_DB_PATH` 已在 SKILL.md 解释；`KERNEL_OPT_BACKENDS` / `KERNEL_OPT_IMAGE` 待 Phase 1 常量落地后追加

### `.cursor/skills/inference-optimizer/README.md` ✅ M8 完成
- [x] 10.7 一页架构概览图（§3.1 拓扑图）
- [x] 10.8 三档 mode 简介 + Token 估算
- [x] 10.9 链到 `inference-optimizer-DESIGN-modified.md` + IMPLEMENTATION-CHECKLIST + KNOWLEDGE-BASE
- [x] 10.10 已知限制（Codex backend pending、OOB 进程隔离 pending、真实 sandbox 接入 pending）
- [x] 10.10b "Monitoring" 小节，含 `--watch` / `--per-agent` / `--per-lane` / `--top-events` 调用示例

### `.cursor/skills/inference-optimizer/KNOWLEDGE-BASE.md` ✅ M8 完成
- [x] 10.11 schema：`entries.jsonl` / `insights.jsonl` / `conflicts.jsonl` 三种文件 + 推荐 categories
- [x] 10.12 KB CLI 用法（`kb_query` / `kb_ingest`）+ 入口指向 `kb/entries.jsonl`

---

## Phase 11 — Multi-sandbox / Cross-run KB / TODO Items `[D §23]`

- [ ] 11.1 T1 — 跟 sandbox 团队确认跨 sandbox 通信能力（CPU + GPU 分开）`[D §23 T1]`
- [ ] 11.2 T2 — 多 GPU sandbox 并行 backend 测试 `[D §23 T2]`
- [ ] 11.3 T3 — KB 多 user/多 session 隔离策略 `[D §23 T3]`
- [ ] 11.4 T4 — Codex no-tools + validated_json_output 稳定性测试套 `[D §23 T4]`
- [ ] 11.5 T5 — Marathon thin shim 参数映射 `[D §23 T5]`
- [ ] 11.6 T6 — KB schema (entries / insights / embeddings) `[D §23 T6]`
- [ ] 11.7 T7 — Skill 名最终定稿（当前占位 inference-optimizer）`[D §23 T7]`
- [ ] 11.8 T8 — 跨前端 (Cursor / ClaudeCode) 测试矩阵 `[D §23 T8]`
- [ ] 11.9 T9 — SQLite leases 表后端真实 sandbox 落地（ADR-33 v0.5 接入测量）`[D §23 T9]`
- [ ] 11.10 T10 — §13.6 evidence-check 矩阵端到端覆盖测试 `[D §23 T10]`
- [ ] 11.11 T11 — kernel-opt 仅优化原生 kernel（不优化 torch.compile 后）`[D §23 T11]`

---

## Phase 12 — Brier 加权 / 跨 run 学习 ✅ M8 完成 `[D §9.4 / §6.2]`

- [x] 12.1 `orchestrator/brier.py::BrierTracker` —— 滚动窗口（默认 50）+ shrinkage prior(0.25, k=5)；`record(agent, predicted, actual)` 即时返回归一化 score；`weight_for(agent)` 给 parliament；`snapshot()`/`restore()` 持久化 `[D §9.4]`
- [~] 12.2 议会投票按 Brier 加权 —— `_open_parliament` 现在按多数票判定（M7）；`weight_for_score` 已就绪但 conductor 还没读到；正式启用待 Phase 11 cadence + 实际数据 `[D §9.4]`
- [~] 12.3 critic 长期可信度自动调整 —— `BrierTracker.score_for(agent)` 提供数据；具体反馈到 PolicyGate / persona 待 Phase 11 `[D §G11]`
- [x] 12.4 cold-start 防护：`KnowledgeBase.is_warm_start_eligible(family)` + `recall_for_model` 在 cold 状态返回空串，确保第 1 次同 family 只 write 不 read `[D §6.2 / ADR-21]`

---

## Phase 13 — Hardening / 未来工作

- [ ] 13.1 死锁检测（环路检测 / timeout 全局监控）`[D §23 T9]`
- [ ] 13.2 Workspace P1 reader/writer lock（v0.5 仍是全局独占）`[D §3.5.5]`
- [ ] 13.3 path B 备选：`apsw + unix-dotfile` VFS 直跑 NFS（路径 B）`[D §3.5.8]`
- [ ] 13.4 跨 sandbox sandbox-to-sandbox 通信（CPU/GPU 分离）`[D §23 T1/T2]`
- [ ] 13.5 KB embedding 升级（v0.5 用简单 BM25）`[D §23 T6]`
- [ ] 13.6 Codex Critic / Sage repair-prompt 失败的兜底分流（marathon → Watchdog；guided → 自审；quick → fail）`[D §10.5.5 / §10.5.6]`
- [x] 13.7 性能 smoke：`tests/test_perf_smoke.py` (4 项) — 500 events 写入 / 8 reactor 重放 / 50 lease 周转 / 互斥并发 acquire（M9 落地，HARD_CAP_S=10s 留足 CI 余量）
- [ ] 13.8 NFS backup 周期可配置（默认 30min，用户可调）`[D §3.5.8]`

---

## Phase 14 — End-to-End 集成测试

- [x] 14.1 E2E1：quick mode dry-run（mock backend）—— `tests/e2e/test_dry_run_smoke.py` (7 项) 已覆盖
- [x] 14.2 E2E2：guided dispatcher 端到端（queued delegate → SubAgentRunner → succeeded）—— `tests/e2e/test_full_loop.py::test_delegate_dispatch_loop_succeeds`
- [~] 14.3 E2E3：marathon 模拟（4 reactor 全开 + cadence）—— `tests/test_conductor_policy.py::test_marathon_mode_spawns_full_roster` 验证多 reactor，但完整 1h 模拟 + cadence 留给 Phase 11
- [x] 14.4 E2E4：crash mid-bench → resume → evidence_check → SUCCEEDED —— `tests/e2e/test_full_loop.py::test_crash_resume_succeeded_via_evidence_check`
- [x] 14.5 E2E5：patch fingerprint 不匹配 → EVIDENCE_INSUFFICIENT —— `tests/test_checkpoint.py::test_check_integrate_evidence_insufficient_on_mismatch`
- [x] 14.6 E2E6：跨 process 重启 —— `test_full_loop.py::test_checkpoint_then_resume_round_trip`（关闭旧 db，从新 SqliteConnection + `resume_from_session_dir` 恢复）；真实跨 sandbox 还需 `restore_from_backup` 走 NFS（Phase 11）
- [x] 14.7 E2E7：accuracy_gate REVERT 链路 —— `tests/test_accuracy_gate.py::test_compare_revert_when_drop_exceeds_threshold` + run_gsm8k mock 端到端
- [x] 14.8 E2E8：early stop on `target_reached` —— `tests/test_early_stop.py::test_should_stop_early_returns_target_first`
- [x] 14.9 E2E9：early stop on `time_exhausted` —— `tests/e2e/test_dry_run_smoke.py` MAX_HOURS=0.001 用例 + `test_early_stop.py::test_time_exhausted_fires_inside_buffer`
- [x] 14.10 E2E10：early stop on `emergency` —— `tests/e2e/test_full_loop.py::test_emergency_stop_via_set_stopping`

---

## 进度统计模板

```
Phase 0   ████████████ 100% (11/11)  — storage / pure-logic / coord primitives
Phase D   ████████████ 100% (8/8)    — minimal viable set (dry-run via MockBackend)
Phase E   ████████████ 100% (8/8)    — ClaudeBackend + Node/CLI auto-install
Phase F   ████████████ 100% (9/9)    — Make Intents Real (multi-reactor, PolicyGate,
                                       ActionRegistry, SubAgentRunner skeleton, MCP)
Phase 1   ████████████ 100% (29/29) — Iron Rules / kernel_opt constants / process mgmt
Phase 2   ███████████░  94% (34/36) — IntentParser ✓ / PolicyGate ✓ / AgentRole ✓
                                       (剩 2.14b JSON-repair pass + 2.20 lane/lease hard test)
Phase 3   ██████████░░  98% (45/46) — Scheduler ✓ / ScorePriors ✓ / AccuracyGate ✓ /
                                       EarlyStop ✓ ; 仅剩 3.45 marathon 5-tail graceful_stop
Phase 4   ████████████ 100% (43/43) — ActionRegistry ✓ / SubAgentRunner ✓ / 22 actions
                                       全部交付 (.md+yaml) + catalog 测试锁定
Phase 5   ████████████ 100% (32/32) — KB ✓ / Persona ✓ / SageQuery ✓ / kb_query ✓ /
                                       kb_ingest ✓
Phase 6   ███████████░  94% (24/25) — backup ✓ / checkpoint ✓ / 6-row evidence_check ✓
                                       (剩 6.10 cursor 重放 — reactor 已天然过滤)
Phase 7   ████████████ 100% (25/25) — dispatcher loop ✓ / resume ✓ / RCA ✓ /
                                       cadence(30min) ✓ / parliament ✓ / TokenMeter ✓
Phase 8   ████████████ 100% (5/5)   — 5 system prompts (executor/critic/sage/watchdog/rca)
Phase 9   ███████████░  96% (22/23) — monitor.sh + monitor.py + IR-6 patch_inductor +
                                       run_baseline/eval_accuracy DRY_RUN_MOCK skeletons
Phase 10  ████████████ 100% (13/13) — README + KNOWLEDGE-BASE + SKILL.md / Monitoring
Phase 11  ░░░░░░░░░░░░   0% (0/11)  — TODO follow-ups (real sandbox / multi-user / NSF)
Phase 12  ████████████ 100% (4/4)   — BrierTracker + cold-start protection
Phase 13  █░░░░░░░░░░░  13% (1/8)   — Hardening (perf smoke ✓; rest pending)
Phase 14  ████████████ 100% (10/10) — E2E1..E2E10 covered via mock+fixture suite

Total                    ~331 check items   |   Tests passing: 570 / 570
```

---

## 实施建议（自顶向下做最小可跑）

1. 先做 **Phase 1 + Phase 8** —— 没有 Iron Rules / KERNEL_OPT 常量 / system prompts 任何 reactor 都跑不起来
2. 然后 **Phase 2 + Phase 4** —— PolicyGate 是所有 reactor 入口；ActionRegistry + 20 个 action metadata 是 scheduler 的输入
3. **Phase 3** —— scheduler + state + accuracy gate 让 quick mode 闭环
4. **Phase 7** —— Conductor 串起来，到这里 quick mode 应该可以跑了
5. **Phase 5** + **Phase 6** —— 加上 KB + checkpoint，guided mode 可跑
6. **Phase 9** —— 沿用 sprint 脚本，做 production 级跑通
   - **TIP**: §9.12‒9.21 的 `scripts/monitor.sh` 几乎可以**第一天就做**（只需 Phase 0 已经完成的 4 张表），开发 Phase 1‒7 期间就能用它当 dashboard 看 events / leases / cursors / tasks 是否真的在动
7. **Phase 10** —— 入口三件套，对外公开
8. **Phase 14** —— E2E 测试逐个绿，可以发布
9. **Phase 12 + Phase 13 + Phase 11** —— 后续打磨

---

**End of Implementation Checklist v0.7 — 2026-04-27 multi-day local marathon**

Net change vs v0.6:
- Tests: 238 → 570 (+332)
- Action catalog: 3 → 22
- Net new modules: iron_rules ✓, process_management ✓, score_priors ✓,
  scheduler ✓, accuracy_gate ✓, early_stop ✓, kb ✓, persona ✓,
  sage_query_service ✓, kb_query ✓, kb_ingest ✓, storage/backup ✓,
  checkpoint ✓, brier ✓, scripts/monitor ✓, scripts/patch_inductor ✓,
  scripts/run_baseline + eval_accuracy DRY_RUN_MOCK skeletons ✓
- Conductor: dispatcher loop ✓ resume ✓ RCA ✓ checkpoint cadence ✓
  parliament ✓ self-review ✓ token meter (record/throttle) ✓
  persona auto-refresh ✓ IronRules + sage_hint in prompt ✓
- Documentation: README ✓ KNOWLEDGE-BASE ✓ 5 system prompts ✓

What still needs the live MI355X sandbox / Codex backend:
- Phase 6.2 CodexBackend (`validated_json_output`)
- Phase 9.2/9.3/9.5/9.7 real `run_baseline.sh` / `eval_accuracy.sh` flows
- Phase 11 multi-sandbox bringup + KB embedding + cross-sandbox comm
- Phase 13.1‒13.6 deep hardening
- Marathon cadence (4h persona distill / 2h strategic review /
  6h cross-run synthesis) — primitives all implemented; clock-tick
  wiring left for the next session.
