# §3.15 v0.6 → v0.8 速查 (cheatsheet)

## 1. 设计目标

给操作员 / 老 v0.6 用户 / 文档 reader 一份**1 页可贴在墙上**的对照表,
让 "我以前怎么做 / 现在怎么做" 一目了然。本节不引入新概念, 只把
§3.1–§3.14 已有的结论凝聚成翻译表。

## 2. 概念跃迁速查

### 2.1 整体架构

| v0.6 | v0.8 | 备注 |
|---|---|---|
| 8 个 pipeline_phase 字段 (yaml 展示) | 5 个 phase 状态机 (PRELUDE / EXPLORE / KERNEL / SWEEP / CLOSE), Coordinator 持有 | yaml 字段保留作展示, 不再驱动决策 (§3.2) |
| LLM 自由选 12 个 action 中之一 | LLM 在当前 phase 允许集合内决策 | phase 决定可用 action 集 (§3.2 §5) |
| 4 个反应器角色 (Orch / Kernel / Critic / Robustness) | 4 个反应器角色 (相同) + 一类 ephemeral specialist sub-agent | 角色不增不减 (Inv-3.1) |
| sub-agent = deterministic Python executor only | sub-agent = deterministic + LLM specialist 双形态 | 主轴 C |
| KB 仅 Critic 评审时点用 | KB 全链路 (T0/T1/T2/T3/T4) 用 + specialist 工具集 | 主轴 B |
| PR 检索不存在 | PR Monitor REST + MCP 接入 | §3.6 / §3.13 M4 |
| Web 工具未授权 | specialist 工具集含 WebSearch/WebFetch (EXPLORE only) | §3.11 R5 |

### 2.2 决策面 (评分体系)

| v0.6 | v0.8 |
|---|---|
| `MARATHON_PRIORS[model_class][action]` (1–10 数值) | (removed) |
| `scoring.py` ActionScore / base_score / score_mult / streak / cooldown | (removed) |
| Orchestration prompt "Action scores" top-12 区块 | (removed) |
| `target_gap_multiplier` | (removed) |
| `cooldown_until_tick` (legacy) | (removed) |
| `params_no_promote_streak` | derived from explore_search.winners_history (不持久化) |
| `effective_score` 排序 | (removed) — LLM 直接判 gap + KB |

decisional input 改为:

| v0.6 字段 | v0.8 字段 |
|---|---|
| Action scores top-12 | phase + phase_allowed_actions + phase_budget_remaining_pct |
| MARATHON_PRIORS 隐式描述 | warm_start_recipe (Cortex T0) |
| streak 字段 | last_action_failures + winners_history (事实层) |
| (无) | gaps[] + kb_subgraph_per_gap |
| (无) | pr_feed (specialist) + specialist_round_summary |

### 2.3 action 名映射

| v0.6 action | v0.8 行为 |
|---|---|
| `target_analysis` | 不变 (PRELUDE) |
| `baseline` | 不变 (PRELUDE) |
| `profile` | 仅 KERNEL 入口 1 次, 不再任意 phase 可发 |
| `pmc_roofline` | 不变 (KERNEL phase) |
| `deep_kernel_analysis` | 不变 (KERNEL phase) |
| `backends` | **removed** → 合并入 `explore` (M3) |
| `params` | **removed** → 合并入 `explore` (M3) |
| `validate_stack` | **removed** → 内嵌进 explore 的 KEEP-后-stack-rebench (M3) |
| `sweep` | 仅 SWEEP phase 允许 |
| `kernel_opt` | 仅 KERNEL phase 允许 |
| `integrate` | 仅 KERNEL phase 允许 |
| `operator_tuning` | 仅 KERNEL phase 允许 |
| `vendor_kernel_config` | 仅 KERNEL phase 允许 |
| `report` | 仅 CLOSE phase 允许 |
| `session_breakdown` | 仅 CLOSE phase 允许 (其它 phase Coordinator 内部触发不算 LLM propose) |
| `recover` | phase-orthogonal, 任意 phase 可由 robustness 触发 |
| `dream` / `re_explore` / `comm_optimization` / `compiler_tuning` | **removed**: 此前 stub executor, 没有真实执行体. v0.8 由 specialist 类型替代 (kernel/comm/compiler 等 domain) |
| (新) `explore` | EXPLORE phase 唯一 grid 跑 variant 的 action |
| (新) `specialist` | EXPLORE phase 内派 specialist 的入口 (delegate{action='specialist'}) |

### 2.4 ledger 字段映射

| v0.6 字段 | v0.8 字段 |
|---|---|
| `backends_search` | `explore_search` (合并) |
| `params_search` | `explore_search` (合并) |
| `backend_winners_history` | `explore_search.winners_history` |
| `last_validate_stack` | (removed) |

### 2.5 stop_reason 词表

| v0.6 (开放字符串) | v0.8 (闭合 ENUM) |
|---|---|
| `target_reached` / `no_more_leverage` / `time_exhausted` / `max_ticks` / `policy_loop` | 沿用 |
| `baseline_failed` | 沿用 (用于 EXPLORE 之后的 baseline 失败) |
| (无) | `prelude_baseline_failed` (PRELUDE 内退出) |
| (无) | `prelude_policy_loop` |
| (无) | `time_exhausted_during_prelude` |
| (无) | `cortex_t0_failed` |
| (无) | `cortex_drain_failed` |
| (无) | `plateau_explore` (主动结束 EXPLORE 而非接转) |
| (无) | `plateau_kernel` |
| (无) | `no_kernel_skipped` |
| (无) | `sweep_done` |
| (无) | `crash_threshold_exceeded` |
| (无) | `robustness_escalated` |
| (无) | `user_stop_requested` |

### 2.6 SharedState 字段

| v0.6 字段 (类别 1 会话身份) | v0.8 |
|---|---|
| session_id / claw_session_id / sandbox_user_id / model_name / model_path / model_class / framework / gpu_type / start_ts / max_minutes / kernel_enabled | **不变** |

| v0.6 字段 (类别 2 事实层) | v0.8 |
|---|---|
| baseline_tput / baseline_accuracy / baseline_failure_streak / current_best / cumulative_gain / cumulative_gain_validated_* / optimization_stack / gain_per_stack_entry / last_action_failures / <action>_attempts / last_profile_trace / last_select_kernels / last_kernel_opt / last_sweep | **不变** (Inv-10.1) |
| `last_validate_stack` | (removed) |

| v0.6 字段 (类别 3 评分) | v0.8 |
|---|---|
| `action_scores` | (removed) |
| `params_no_promote_streak` | (removed; 派生计算) |
| `score_violation` | (removed) |
| `cooldown_until_tick` (legacy) | (removed) |
| `locked_reason` | (removed) |
| `streak_*` | (removed; 替换为 specialist_domain_empty_streak + rejected_kernel_partial_overflow) |

| (新) v0.8 字段 | 来源 |
|---|---|
| `phase` / `phase_started_ts` / `phase_history[]` / `phase_budget_pct` | §3.2 / §3.10 |
| `warm_start_recipe` / `warm_start_pitfalls` | §3.6 T0 |
| `cortex_session_id` / `cortex_session_summary` / `pending_kb_edges[]` | §3.6 |
| `explore_search` (合并 ledger) | §3.4 |
| `specialist_rounds[]` / `specialist_domain_empty_streak{}` | §3.5 / §3.9 |
| `rejected_kernel_partial_overflow{}` | §3.9 |
| `research_lane_capacity` | §3.7 / manifest mirror |

## 3. 操作员翻译指南

### 3.1 监控脚本对照

| v0.6 操作员习惯 | v0.8 替代 |
|---|---|
| `cat state.json | jq .action_scores` 看下一步 | `cat state.json | jq .phase` + `jq '.gaps' (M5+)`; 决策由 LLM 做, scoreboard 不存在 |
| `cat state.json | jq .params_no_promote_streak` | `jq '.explore_search.winners_history | length'` (作 plateau 近似); 真正的 plateau 见 phase_history |
| `cat state.json | jq .backends_search.tested` | `jq '.explore_search.tested'` |
| 按 cooldown 字段查 next propose | 不存在; LLM prompt 中 phase + gaps 决定 |

### 3.2 breakdown.json 字段对照

| v0.6 段 / 字段 | v0.8 |
|---|---|
| `capability_summary.backends` | `capability_summary.explore` (alias `backends` 仍提供, 数据派生) |
| `capability_summary.params` | 同上 (alias) |
| `capability_summary.validate_stack` | (removed; 内嵌 explore) |
| `param_search` | `explore_search` (alias `param_search` 仍提供) |
| (无) | `specialist_runs[]` |
| (无) | `kb_provenance{}` |
| `phase_timeline` (action 序列) | `phase_timeline` 重新定义为 phase 边界外层 + 内嵌 actions; 同时提供 `action_timeline` 顶层字段供 v1 reader |
| (无) | `attribution.phase_breakdown` |
| (无) | `telemetry.lane_timeline` |

### 3.3 CLI flag 速查

| v0.6 / 通用 | v0.8 新增 |
|---|---|
| `--model` / `--framework` / `--gpu-type` / `--max-hours` / `--target-gain` ... | 沿用 |
| `--critic-mock` / `--critic-agent` / `--critic-codex-bare` | 沿用 |
| `--robustness-mock` / `--robustness-agent` | 沿用 |
| `--no-kernel` | 沿用 |
| (无) | `--no-cortex` (M1 起) |
| (无) | `--cortex-kb-url` (M1 起) |
| (无) | `--cortex-strict-fingerprint` |
| (无) | `--no-pr-monitor` (M4 起) |
| (无) | `--pr-monitor-url` |
| (无) | `--pr-feed-window-days` |
| (无) | `--research-lane-capacity` (M5 起) |
| (无) | `--legacy-action-scores=drop|warn` |
| (无) | `--migration-mode=strict|lenient` |
| (无) | `--reset-state` |
| (无) | `--max-minutes-prelude / -explore / -kernel / -sweep / -close` (M2 起) |
| (无) | `--plateau-explore-keep-gain / -empty-streak` (M7 起) |
| (无) | `--plateau-kernel-revert-streak / -keep-gain` (M7 起) |
| (无) | `--specialist-aggregation=wait_all|partial_k` (M6/M7) |
| (无) | `--legacy-explore-split` (M3 紧急回退) |
| (无) | `--no-stack-rebench` (M3 紧急 degrade) |
| (无) | `--force-phase=PRELUDE|EXPLORE|KERNEL|SWEEP|CLOSE` (resume 时强制) |

## 4. resume 行为对照

| 场景 | v0.6 → v0.8 行为 |
|---|---|
| v0.6 session 有 `action_scores` | drop 默认; `--legacy-action-scores=warn` 写 log warning |
| v0.6 session 没有 `phase` | 推断 (§3.10 §5.2); 若推不出默认 `EXPLORE` 并写 evidence 标 `inference_uncertain` |
| v0.6 session 有 `backends_search` + `params_search` | union → `explore_search`; 老字段保留至少一个版本周期作 deprecated |
| v0.6 session 有 `last_validate_stack` | drop |
| v0.6 session 没有 `cortex_session_id` | 第一次 v0.8 resume 时跑一次 T0 begin, 写入 sid; 之前 session 的 KB 写不可补 (那段 session 视为 KB-orphan) |
| v0.6 session 有 in-flight task | 沿用 v0.6 task resume (TaskRegistry 状态机不变); 完成后正常进 audit |
| v0.6 session resume 时 `--reset-state` | 完全清空 SharedState, baseline 重跑; Cortex KB 中已有的跨 session 知识仍可用 |
| v0.8 session resume 时 phase=CLOSE 但未 commit | 重新走 CLOSE 5 步 (drain + commit); 幂等 |

## 5. 部署 / 监控 提醒

- v0.8 上线先**灰度跑同 model 的 v0.6 / v0.8 双 session 对比**, 看
  cumulative_gain 不退步。
- breakdown.warnings 是日常 review 必看段; 每条 warning 在本 cheatsheet
  或 §3.14 中有解释。
- 任何 stop_reason 不在 §3.8 §6 词表内的 v0.6 老 session 都用
  `--migration-mode=lenient` 启动一次, 让迁移函数把它映射到合理的新词
  表项 (具体映射列在迁移函数内部, 见 §3.10 §5.2)。

## 6. 哲学回引

cheatsheet 本身不引入新约束, 但每行翻译都对应 §3.1–§3.14 中的某条结
论。如果操作员发现某条对照与系统实际行为不一致, 应当回到对应章节 +
里程碑 PR review 找根因, 不要直接在系统里"打补丁让它符合 cheatsheet"。
本表是**结果文档**, 不是源真相。
