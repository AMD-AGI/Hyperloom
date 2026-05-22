# KB_design_gaps — v0.8 设计意图 vs 现行代码差距分析

> 父文档: `KB_design.MD` (概念) + `KB_design/3.1`–`3.15` (展开).
> 本文件: 把 KB_design 的"设计应当如何"和 `inference_optimizer/` 的
> "代码实际怎样"做一次对账, 列出会**阻断 v0.8 设计目标达成**的具体 gap,
> 并给出修复路径建议.
>
> 阅读对象: v0.8 推进负责人 + 各 milestone PR reviewer + 操作员 (准备
> 上线前需要知道哪些环节实际还没接通).
>
> **结论提前**:
>
> - **主轴 A (流程固化)** — 字段层 / phase 状态机 / PolicyGate 都已落,
>   但 *phase 入口的副作用 hook* (自动 profile / 自动 sweep / 自动 CLOSE
>   顺序器) 缺失, 导致 phase 切换"只换标签不换行为".
> - **主轴 B (知识外接 Cortex)** — T0/T2/T3/T4 hook 都有, 但 KnowledgePlane
>   未在生产路径 bootstrap; T2/T3 粒度仍是 per-proposal/per-task, 没升到
>   M5 设计的 per-variant. 主轴 B 在事实层 50% 落地, 在 KB 视图层未达 GA.
> - **主轴 C (sub-agent 双形态)** — **SpecialistRunner 完全未注册**,
>   `delegate{action='specialist'}` 会 fail 到 `no_executor`. 主轴 C 在
>   生产路径 **0% 落地** (代码 + 测试都齐, 只缺 cli `_register_executors`
>   一处接线).
>
> 三主轴的差距叠加 → 一个真实新 session 在 CLI 启动后, 行为接近于
> "v0.7 + phase enum + scoreboard 删除", 还**没有**进入 v0.8 设计声称的
> "三主轴齐"状态. KB_design §3.13 中的 M5 / M6 在生产路径上**未启用**,
> 即便 M5 的代码全数 merged.

---

## 1. 设计目标 (gap analysis 自身)

把 `KB_design/` 中每条设计承诺翻译为"如果今天 cold-start 一个新 session,
会发生什么", 然后逐条对比代码事实, 给出:

- **现状**: 实际跑起来的行为.
- **设计意图**: KB_design 对应章节 (§3.x / Mn) 的 *should-be*.
- **差距**: 现状缺哪一段 + 为什么这条 gap 重要.
- **严重度**: P0 (设计目标不可达) / P1 (degraded behavior) / P2 (cosmetic /
  observability only).
- **关联章节 / 文件 / 行**: 便于 reviewer 复现.
- **修复路径建议**: 落地的最小集合.

本节**不是新设计**, 是体检报告. 任何"在 gap 里现场提新方案"的内容,
都必须先回到对应 §3.x 章节走正式更新.

---

## 2. 现状回顾 — 一次新 session 的完整 trace

假设操作员跑:

```
python -m inference_optimizer optimize \
  --model llama-3.1-70B --framework sglang --gpu-type mi300x \
  --max-hours 6 --target-gain 12
```

按 KB_design §3.2 五段管线**应当**发生的事 vs **实际**发生的事:

### 2.1 启动 (CLI `_run_optimize`)

| 步骤 | KB_design | 实际行为 |
|---|---|---|
| Preflight + manifest | §3.13 M1 / M2 | 全部 ✅ |
| 路径解析 + session_dir | M1 §3 | ✅ |
| ENV 切换 (`INFERENCE_OPTIMIZER_STRICT_PHASE` / `STRICT_PATHS` / `MIGRATION_MODE` / `LEGACY_ACTION_SCORES` / `BREAKDOWN_INCLUDE_TRANSCRIPTS`) | §3.10 / §3.12 | ✅ |
| ResetState 备份 | §3.10 §5.3 | ✅ |
| Cortex T0 `session begin` + warm_start_recipe + warm_start_pitfalls | §3.6 / M1 §5.1 | ✅ (`_bootstrap_cortex_kb`) |
| KnowledgePlane bootstrap (Cortex 客户端 + PR Monitor 客户端 + domain_repos) | §3.6 / M4 §6 | ❌ **`_bootstrap_knowledge_plane` 永不调用** |
| Coordinator 构造 + 注册 executor | §3.3 / §3.13 | 部分 — SpecialistRunner **未注册** |
| 第一次 phase 转入 PRELUDE | §3.2 §5.1 | ✅ (`_ensure_phase_initialised`) |

### 2.2 PRELUDE phase

| 步骤 | KB_design | 实际行为 |
|---|---|---|
| `target_analysis` 强制门 (现有 hard-gate) | §3.2 §5.1 | ✅ |
| `baseline` (现有) | §3.2 §5.1 | ✅ |
| `baseline_tput > 0` → 转 EXPLORE | §3.2 §5.1 | ✅ (`compute_next_phase`) |
| `baseline_failure_streak ≥ 3` → `stop_reason='baseline_failed'` → CLOSE | §3.2 §5.1 / §3.8 | ✅ |

### 2.3 EXPLORE phase (核心差距集中地)

| 步骤 | KB_design (§3.2 §5.2 + §3.5) | 实际行为 |
|---|---|---|
| Coordinator 切 gap 列表 (baseline / Cortex `issue_node` traverse / 上轮 KEEP/REVERT) | §3.2 §5.2 step 1 | ❌ `SharedState.gaps[]` 字段不存在; prompt 仅渲染 `last_action_failures` + `explore_search.winners_history` 作 proxy |
| 并行 `delegate{action='specialist'}` 派 N 个 specialist | §3.2 §5.2 step 2 | ⚠️ **能 emit**, 但: ① policy R2 通过 ② dispatcher 找 specialist executor 时 fail (cli `_register_executors` 不注册 SpecialistRunner) |
| KnowledgePlane.pr_feed_warm + kb_subgraph 装到 delegate 的 params | §3.6 + M5 §5 | ❌ KnowledgePlane 没接 → 装 params 也无源 |
| SpecialistRunner 拉 prompt 9 段 → LLM 跑 → emit `specialist_done` | §3.5 §6 / §3.5 §10 | ⚠️ 代码 + 测试齐, 生产无 entry |
| Coordinator 接收 `specialist_done` → 解析 proposal_set → `record_specialist_round` | §3.5 §7 / §3.11 R3 | ❌ `_handle_intent` 路由表无 `SPECIALIST_DONE` 分支, 走 `else` 仅记 observation |
| `propose_action{action='explore', grid=...}` (来自 Orchestration 汇总 specialist 提议) | M5 §5 step 4 | ⚠️ LLM 能 propose `explore`; **但没有 specialist 上游**, 提议只能源自 LLM 自身判断 |
| T2 per-variant hypothesize | M5 §5 step 6 | ❌ 仍 per-proposal 一条边 |
| Critic Review batch (`verdict_map: {variant_name → verdict}`) | §3.5 §5 + M5 §5 step 5 | ❌ `REVIEW_VERDICT` payload 仅含单 `verdict` 字段, per-proposal 一刀切 |
| Explore executor 串行跑 + 每 KEEP 后 stack rebench | §3.4 / M3 | ✅ (M3 已落) |
| T3 per-variant ingest_attempt + verify | M5 §5 step 7 | ❌ 仍 per-task |
| plateau_explore 判定 + budget exhausted | §3.8 / M7 | ✅ |

### 2.4 KERNEL phase

| 步骤 | KB_design (§3.2 §5.3) | 实际行为 |
|---|---|---|
| 进入即自动 enqueue 一次 `profile` | §3.2 §5.3 step 1 | ❌ `_advance_phase_if_needed` 只切 phase 标签; `_required_next_step` 是 prompt 提示, 非自动 enqueue |
| `select_kernels` 缓存 trace_input | §3.13 M2 | ✅ |
| `kernel_opt` / `integrate` / 5 个 KERNEL_OWNED_ACTIONS | §3.2 §5.3 step 2 | ✅ (现有, 不变) |
| 同 kernel_id PARTIAL ≤ 2 → 入 `rejected_kernel_partial_overflow` | §3.9 §5.1 | ✅ |
| plateau_kernel + kernel_phase_budget_exhausted | §3.8 / M7 | ✅ |

### 2.5 SWEEP phase

| 步骤 | KB_design (§3.2 §5.4) | 实际行为 |
|---|---|---|
| 进入即自动构造 sweep grid (默认 + `recipe.sweep_grid` 覆盖) | §3.2 §5.4 step 1 | ❌ 阶段切换不 enqueue; sweep executor 只读 task params, 不读 warm_start_recipe |
| 自动 enqueue `sweep` action | §3.2 §5.4 step 2 | ❌ 必须 LLM `propose_action='sweep'` |
| `sweep_done` → CLOSE | §3.8 / phase_state | ✅ |

### 2.6 CLOSE phase

| 步骤 | KB_design (§3.2 §5.5) | 实际行为 |
|---|---|---|
| Coordinator 内部触发 `report` | §3.2 §5.5 step 1 | ⚠️ 仅 `_enter_closing_phase` 在 wall-clock deadline 自动 enqueue; 不挂在 `phase=CLOSE` 转入上 |
| Coordinator 触发 `session_breakdown` action | §3.2 §5.5 step 2 | ❌ 不在 Coordinator; 走 cli.py finally 兜底写 `session_breakdown.json` |
| NDJSON flusher drain | §3.2 §5.5 step 3 | ⚠️ 走 `Coordinator.stop()` T4 hook, 不是 CLOSE 阶段内固定 step 3 |
| Cortex `session commit` | §3.2 §5.5 step 4 | ⚠️ 同上, `stop()` 调; 时序与 report 之间无序保证 |
| Exit reactor loop | §3.2 §5.5 step 5 | ✅ |

---

## 3. 不变量审查

| 不变量 | 设计 (§3.1 §4) | 现状 | 评估 |
|---|---|---|---|
| Inv-1 事实层单写者 | CORE_STATE_FIELDS 锁 LLM 越权写 | 锁了 v0.8 加的 phase / cortex / specialist / lane 字段; **`explore_search` / `gaps` 不在锁内** | ⚠️ 部分: explore_search 可被 LLM update_state 改 |
| Inv-2 KB 写经 Coordinator 中转 | specialist 不直写 KB | SpecialistRunner 工具白名单不含 KB 写 (✅), 但 specialist 整体没生产入口 | ✅ (vacuously true — 没 specialist 就没违规面) |
| Inv-3 serving GPU 单租户 | benchmark_lane=1, research_lane 收 specialist | benchmark_lane=1 ✅, research_lane 容量字段 ✅; **但 research_lane 实际无任务进入** | ⚠️ vacuous: research_lane 始终空 |
| Inv-2.1 phase 单调性 | DAG, 不回退 | `compute_next_phase` 强制 forward; ✅ | ✅ |
| Inv-2.2 phase 退出必须有 reason | reason ∈ 词表 | `set_stop_reason` ENUM 校验 ✅; phase_history append 强制 reason 字段 ✅ | ✅ |
| Inv-4.1 explore 单 ledger | backends_search + params_search → explore_search | ledger 已合并 ✅; **但 EXPLORE allowlist 仍含 backends/params/validate_stack legacy action**, 双轨可用 | ⚠️ 部分: 设计声明合并但生产入口未关旧 |
| Inv-5.1 specialist 不出 patch | proposal_set only | SpecialistRunner 工具白名单已禁 Edit/Write/Bash apply (✅); 测试覆盖 (✅) | ✅ (vacuously true) |
| Inv-5.2 specialist 经 Coordinator 中转写 KB | T2/T3 by Coordinator only | ✅ (设计) | ✅ |
| Inv-5.3 单退出协议 (specialist_done) | one terminal intent | Intent enum 含 ✅; PolicyGate R3 ✅; **`_handle_intent` 不分发** | ❌ R3 通过但下游无人处理 |
| Inv-6.1 KB 写者唯一 (Coordinator) | provenance.generator = coordinator | ✅ 设计如此; 实际无 specialist KB 写 (Inv-2 vacuous), 安全 | ✅ |
| Inv-7.1 lane 容量配置时锁 | research_lane_capacity in CORE_STATE_FIELDS | ✅ | ✅ |
| Inv-9.1 决策层无评分 | scoring.py / action_scores 删除 | ✅ M2 已落 | ✅ |
| Inv-10.1 事实层 v0.6→v0.8 不变 | 字段位级一致 | ✅ M2 已落 (`test_v08_shared_state_evolution`) | ✅ |
| Inv-10.2 Coordinator 单写者 | LLM 经 intent | ✅ 设计层; CORE_STATE_FIELDS 已扩展 | ✅ |
| Inv-10.3 迁移幂等 | from_dict(from_dict(x)) ≡ from_dict(x) | ✅ | ✅ |
| Inv-11.1 PolicyGate pure | 无副作用 | ✅ | ✅ |
| Inv-11.2 拒绝可恢复 | rule + hint | ✅ R1-R5 都遵守 | ✅ |
| Inv-11.3 规则正交 | 一 intent 唯一命中 | ✅ test_v08_policy_gate_evolution 覆盖 | ✅ |
| Inv-12.1 breakdown schema 向后兼容 | v1 reader 不崩 | ✅ M2 已落; schema v2 + alias 字段 | ✅ |
| Inv-12.2 段落数据来源唯一 | 单一 SharedState 字段派生 | ✅ specialist_runs / capability_summary.specialist 共源 | ✅ |
| Inv-12.3 写入时机分级 | end-of-session / on-demand | ⚠️ 分散于 wall-clock deadline + cli finally + stop() | ⚠️ 部分 |

**总结**: 字段 / 数据形态 / PolicyGate / breakdown 层面 (§3.9–§3.12 主要内容) 的不变量几乎全部满足. **主轴 C 整链 (§3.5 specialist 框架) 的所有 Inv 都是 vacuous true** — 因为 specialist 实际不跑.

---

## 4. P0 阻断级 gap (设计目标不可达)

### Gap-1 — SpecialistRunner 未注册 → `delegate{action='specialist'}` 整链不通

**设计意图** (KB_design §3.5 §10 / §3.13 M5):
EXPLORE phase 内 Orchestration 通过 `delegate{action='specialist', params={domain, gap, ...}}`
派 specialist; Coordinator dispatcher 取出该 task → 走
`SpecialistRunner.run_task` → LLM 多 turn → emit `specialist_done` → 解析
proposal_set → 进入 explore round.

**实际行为**:

- `inference_optimizer/cli.py::_register_executors` (~625–730) 只注册了
  `_REAL_EXECUTORS_FULL` (deterministic executors), **没有任何
  `SpecialistRunner` 实例化或注册到 SubAgentRunner**.
- `orchestrator/coordinator.py` 全文 **无任何 `SpecialistRunner` /
  `KnowledgePlane` 引用**.
- 派发流到 `SubAgentRunner.run_task` 时 fail with `no_executor` (~113–124).

**差距**: 主轴 C 在生产路径 **0% 落地**. 测试齐, 代码齐, 接线缺.

**严重度**: **P0** — v0.8 三大主轴之一完全不可用; KB_design §3.13 M5 / M6
的所有验收标准都不能在生产环境验证.

**关联**: §3.5 全章; §3.13 M5 §3 / §8; `cli.py:625–730`; `coordinator.py`
全文; `orchestrator/specialist_runner.py` 全文 (本身就绪).

**修复路径建议**:

1. `cli.py` 中先 `_bootstrap_knowledge_plane()` (见 Gap-2),
   再构造 `SpecialistRunner(backend_factory=..., session_dir=...,
   knowledge_plane=plane, ...)`.
2. 在 `_register_executors` 末尾把 SpecialistRunner 暴露到
   `SubAgentRunner.executor_registry['specialist']` (或新增一条等价路径).
3. 同时落地 Gap-3 (specialist_done 路由) 才能闭环.

---

### Gap-2 — KnowledgePlane 在生产路径不 bootstrap → pr_feed / kb_subgraph 永空

**设计意图** (KB_design §3.6 §6 / §3.13 M4 §6):
KnowledgePlane 是 specialist 的"外接知识门面". 启动时由 CLI 构造
(Cortex 客户端 + PR Monitor 客户端 + domain_repos yaml), 注入到
Coordinator → SpecialistRunner. 每次 specialist 派发前
`pr_feed_warm(domain=..., window_days=N)` 预热 PR 列表.

**实际行为**:

- `inference_optimizer/cli.py::_bootstrap_knowledge_plane` (~1918–1975)
  函数已写好.
- `_run_optimize` 整个函数体 **从未调用** `_bootstrap_knowledge_plane`.
- `Coordinator` 构造参数 **没有 `knowledge_plane`** 字段; 全文无引用.
- 结果: 即使 Gap-1 修好让 SpecialistRunner 跑起来, `params.pr_feed` 永
  远为空; specialist 只能依靠 KB subgraph 和源码探索.

**差距**: 主轴 B 在 *跨集群知识入口* 这一段未接通 (Cortex T0/T2/T3/T4 接通,
但 PR / KB readonly traverse 未接通到 specialist).

**严重度**: **P0** — KB_design §3.6 的"具体特性 = pr_feed + kb_subgraph"
完全失效; pr_intel_specialist domain 退化为"看 KB + 源码" (相当于
serving_specialist 子集), 失去 §3.6 §5 提供的 PR 视野.

**关联**: §3.6 §5 / §6; §3.13 M4 §6; `cli.py:1918–1975` 已写函数; 调用
点 0.

**修复路径建议**:

1. 在 `_run_optimize` 中 manifest 写完后调一次
   `plane = _bootstrap_knowledge_plane(args, ...)`.
2. Coordinator 加 `knowledge_plane: KnowledgePlane | None = None` 构造
   参数, 透传给 SpecialistRunner.
3. `_handle_delegate` 检测 `action='specialist'` 时, 在 enqueue task 前
   调一次 `plane.pr_feed_warm(domain)` + `plane.cortex_traverse(...)`
   把结果写到 task.params 里.

---

### Gap-3 — `IntentType.SPECIALIST_DONE` 在 Coordinator `_handle_intent` 路由表中缺失

**设计意图** (KB_design §3.5 §7 / §3.11 R3):
specialist 跑完一定通过 `specialist_done` intent 关闭 lifecycle.
Coordinator 接收后:
- 解析 payload.proposal_set
- 调 `SharedState.record_specialist_round(...)`
- 更新 `specialist_domain_empty_streak`
- 触发该 task 的 `delegated_result` 事件

**实际行为**:

- `coordinator.py::_handle_intent` (~1971–2009) 的 if/elif 链 **不包含
  `IntentType.SPECIALIST_DONE`** 分支.
- 走 `else` 分支 (~2003–2008), 只记一条 observation 到 inbox.
- `record_specialist_round` 方法存在 (`shared_state.py:1953+`) 但
  **只在 tests 调用**.

**差距**: 即使 Gap-1 + Gap-2 修好让 specialist 真跑, 它 emit 的
`specialist_done` 也不会驱动任何行为; proposal_set 永远不会进入
explore round.

**严重度**: **P0** — specialist 整链最后一公里断开.

**关联**: §3.5 §7 / §10; §3.11 R3; `coordinator.py:1971–2009`;
`shared_state.py:1953+`.

**修复路径建议**:

1. 在 `_handle_intent` 添加 `elif it == IntentType.SPECIALIST_DONE:
   await self._handle_specialist_done(source, intent)` 分支.
2. 新建 `_handle_specialist_done`:
   - 找到对应 task (`from_agent='specialist:<task_id>'`)
   - 调 `shared_state.record_specialist_round({...})`
   - 把 proposal_set 转换为 `SubAgentResult.result.proposal_set` 触发
     `delegated_result` 事件
   - kill task / mark succeeded
3. R3 已经做了来源 + schema 校验, handler 不需要重复.

---

## 5. P1 主要 gap (degraded 但项目仍可跑)

### Gap-4 — KERNEL phase 入口不自动 profile

**设计意图** (KB_design §3.2 §5.3 step 1):
"进入即跑一次 `profile` (固定动作, 不需要 LLM propose). 该次 profile
写入 `last_profile_trace`, 锚定 KERNEL 阶段所有 `select_kernels` 的
trace_input."

**实际行为**:

- `coordinator.py::_advance_phase_if_needed` (~697–772): 只更新 `phase`
  + `phase_history`, **不 enqueue task**.
- `_required_next_step` (~1631–1635): 全局 nag "profile required",
  但只影响 prompt 提示文本, 不强制 enqueue.
- 结果: KERNEL 入口时 `last_profile_trace==""`, LLM 必须自觉 propose
  profile. 若 LLM 漏掉, 后续 `select_kernels` 没有 trace, kernel_opt
  pipeline 受阻.

**严重度**: **P1** — 不阻断, 但容易在 LLM 失误时让整个 KERNEL phase
空跑.

**关联**: §3.2 §5.3; `coordinator.py:697–772, 1631–1635`.

**修复路径建议**:
`_advance_phase_if_needed` 检测到 `from=='EXPLORE' and to=='KERNEL'` 且
`last_profile_trace==""` 时, 在切 phase 后立即调
`_enqueue_internal_task('profile', ...)` (类似 closing phase 的
`_enqueue_report_task` 模式).

---

### Gap-5 — SWEEP phase 入口不自动 sweep + 不读 `recipe.sweep_grid`

**设计意图** (KB_design §3.2 §5.4 step 1–2):
"自动构造 sweep grid (来自 SKILL.md 默认 grid + Cortex `recipe.sweep_grid`
字段, 后者优先). sweep action 串行跑每个组合."

**实际行为**:

- phase 切换 EXPLORE/KERNEL → SWEEP 自动 (`compute_next_phase` ✅), 但
  **不 enqueue sweep task**.
- `action_executors/sweep.py::_build_grid` (~59–94) 只读 task params
  (`conc_values` / `isl_osl_configs`); **不读 `state.warm_start_recipe`**.

**严重度**: **P1** — SWEEP phase 必须 LLM 主动 `propose_action='sweep'`,
跨 session 学到的 sweep grid 不被复用 (主轴 B 的"跨 session 知识"在
SWEEP 阶段失效).

**关联**: §3.2 §5.4; §3.6 (recipe); `coordinator.py:697–772`;
`sweep.py:59–94`.

**修复路径建议**:

1. `_advance_phase_if_needed` 检测到 `to=='SWEEP'` 时 enqueue sweep task,
   params 从 `state.warm_start_recipe.sweep_grid` 取 (fallback SKILL.md
   默认).
2. 或退一步: 让 sweep executor 在 task params 缺失时回退读
   `warm_start_recipe` (更小的改动面).

---

### Gap-6 — CLOSE phase 无统一 5 步顺序器, 三处零散触发

**设计意图** (KB_design §3.2 §5.5):
固定顺序 (1) report → (2) session_breakdown → (3) NDJSON drain → (4)
Cortex `session commit` → (5) exit. **不允许 LLM 跳序**.

**实际行为**:

- (1) `report`: `coordinator._enter_closing_phase` (~1076–1135) 在
  wall-clock deadline (`max_minutes` 接近用尽) 自动 enqueue report,
  **与 phase 切换无关**.
- (2) `session_breakdown`: `cli.py finally` (~2465–2474) 兜底写
  `session_breakdown.json`, **不绑定 phase=CLOSE**.
- (3) NDJSON drain: `Coordinator.stop()::_cortex_t4_hook` (~549–598),
  **在进程退出时一次性跑**, 不在 CLOSE phase 内 step 3 位置.
- (4) `session commit`: 同 (3) 在 stop() 内.
- (5) Exit: 自然结束.

**严重度**: **P1** — 单次 fresh session 跑到 deadline 时基本能产出
report + breakdown + commit, 但顺序不固定 (尤其用户提前 Ctrl-C 时
仅触发 (3)+(4), 缺 (1)+(2)).

**关联**: §3.2 §5.5; `coordinator.py:1076–1135, 549–598`;
`cli.py:2465–2474`.

**修复路径建议**:
新增 `_run_close_phase_sequencer()` 在 `_advance_phase_if_needed` 检测到
`to=='CLOSE'` 时按 1→5 顺序触发. wall-clock deadline 路径
(`_enter_closing_phase`) 也复用同一 sequencer.

---

### Gap-7 — T2 hypothesize 仍是 per-proposal, 未升级到 per-variant (M5 设计)

**设计意图** (KB_design §3.13 M5 §5 step 6):
"for each variant in grid: propose-point optimization_node ...
hypothesize hypothetical edge issue_node→opt_node attrs={variant: ...} ...
PendingProposal.kb_edge_ids[variant.name] = tentative_edge_id"

**实际行为**:

- `coordinator._cortex_t2_hook` (~2050–2124): 每 PendingProposal 一条
  `propose_point` + 一条 `hypothesize`, 用 `proposal_msg_id` 作 anchor.
- `PendingProposal` 字段 `kb_edge_id: str` (~218) — **单 edge_id, 非
  per-variant map**.

**严重度**: **P1** — KB 跨 session 检索时无法定位"具体哪个 variant
confirmed/refuted"; specialist confidence 评估缺粒度.

**关联**: §3.5 §10 / M5 §5 step 6; `coordinator.py:204–222, 2050–2124`.

**修复路径建议**: 把 `PendingProposal.kb_edge_id` 改为
`kb_edge_ids: dict[variant_name → edge_id]`, T2 hook 遍历 grid 中每
variant 各发一条 hypothesize. 现有 `explore.py:129, 157` 已经支持读
per-variant `kb_edge_id`.

---

### Gap-8 — T3 verify 仍是 per-task, 未升级到 per-variant (M5 设计)

**设计意图** (KB_design §3.13 M5 §5 step 7):
"for each variant: KEEP → ingest_attempt PASS + verify edge=kb_edge_ids[variant]
outcome=confirmed promote=EXPERIENTIAL; REVERT → ingest_attempt FAIL + verify
outcome=refuted"

**实际行为**:

- dispatcher 每 task 跑完调一次 `_cortex_t3_hook` (~3133–3137).
- `_cortex_t3_hook` (~3177–3268) 用 task-level `promoted` (来自
  `_is_promotable_result`) 决定整批 confirmed/refuted.
- explore.py 内部已经 per-variant KEEP/REVERT (~10–21), 但 **不上送
  Cortex**.

**严重度**: **P1** — 与 Gap-7 配对; KB negation edge 只能基于"整批是否
有 ≥ 1 KEEP"判断, 无法学到 "variant X refuted 但 variant Y confirmed".

**关联**: §3.5 / M5 §5 step 7; `coordinator.py:2929–2936, 3123–3137,
3177–3268`; `explore.py:10–21`.

**修复路径建议**:
explore executor 把 per-variant 结果列表 (含 variant_name, outcome,
metrics) 写入 `SubAgentResult.result.per_variant_outcomes`. T3 hook 检
测到该字段时遍历 per-variant 发 ingest_attempt + verify.

---

### Gap-9 — `SharedState.gaps[]` 字段不存在; prompt 用 proxy

**设计意图** (KB_design §3.3 / §3.5 / §3.9 §6):
Orchestration prompt 的核心决策原料字段; 也是 specialist domain selection
的主要 anchor.
"`gaps[]` 当前未解决的瓶颈 (canonical_id + symptom + layer + 已尝试历史)"

**实际行为**:

- `shared_state.py` 全文 **无 `gaps:` dataclass 字段**.
- `prompt_builder.py:500–504` 注释明说: 暂用 `last_action_failures` +
  `explore_search.winners_history` 作 proxy.
- M2 milestone 仍把 gaps 留作 future work (`M2_phase_and_scoreboard.md`
  ~108–109).

**严重度**: **P1** — LLM 决策依据降级到字面 failure 历史; specialist
domain selection (§3.5 §11) 失去主要锚点; KB negation 边过滤的反向作用
(把已 refuted 的 gap 从 LLM 视野滤掉) 失效.

**关联**: §3.3 / §3.5 / §3.9; `shared_state.py` (字段缺); `prompt_builder.py:500–504`.

**修复路径建议**:

1. 在 `SharedState` 加 `gaps: list[dict[str, Any]]` 字段 (CORE).
2. Coordinator 新增 `_refresh_gaps()` 在以下时点更新:
   - baseline 完成后初始化 (从 baseline metrics 抽 symptom)
   - 每 KEEP/REVERT 后更新对应 gap.attempts
   - Cortex `traverse` `issue_node` 时合并外部已知 gap (跨 session)
3. PolicyGate 把 `gaps` 加入 CORE_STATE_FIELDS.

---

### Gap-10 — EXPLORE phase 仍允许 legacy `backends` / `params` / `validate_stack` 双轨

**设计意图** (KB_design §3.4 / §3.15 §2.3):
v0.8 把 backends / params / validate_stack 合并为单一 `explore` action.
legacy 名应当移除.

**实际行为**:

- `phase_state.py:91–97` `PHASE_ALLOWED_ACTIONS[PHASE_EXPLORE]` 仍包含
  `backends`, `params`, `validate_stack` (注释标 "M2 transition").
- `actions/_meta/` 目录下 `backends.yaml` / `params.yaml` /
  `validate_stack.yaml` 仍在 (`glob inference_optimizer/actions/_meta/*.yaml`).
- `cli.py::_register_executors` (~627–639) 注册 `backends_executor` /
  `params_executor` / `validate_stack_executor`.

**严重度**: **P1** — Inv-4.1 (explore 单 ledger) 在数据层满足 (M3 已
合并), 但 *入口层* 双轨可用; LLM 可绕过新 `explore` 直接发 `backends`,
导致 ledger 仍会被 *两个 executor* 写, 长期有口径分裂风险.

**关联**: §3.4 / §3.15 §2.3; `phase_state.py:87–115`;
`cli.py:625–730`; `actions/_meta/*.yaml`.

**修复路径建议**:

1. `PHASE_ALLOWED_ACTIONS[PHASE_EXPLORE]` 移除三个 legacy 名.
2. `actions/_meta/{backends,params,validate_stack}.yaml` 移除.
3. `cli.py::_register_executors` 移除三个 legacy executor.
4. PolicyGate 新增 deprecation hint (R1 phase_incompatible 命中时附带
   "use 'explore' instead").

(注意: 在 M3 设计明确 GA 之前可能想保留 legacy 作回退. 设计 §3.13 M3
"灰度方式 = `--legacy-explore-split` off" 已有此意, 但 flag 尚未实现.)

---

### Gap-11 — Critic Review 仍 per-proposal verdict, 未升 per-variant `verdict_map`

**设计意图** (KB_design §3.5 §5 / M5 §5 step 5):
"Critic 收到一组 K variant + judge_bundle (含 KB priors 各 variant);
返回 verdict map {variant_name → verdict}"

**实际行为**:

- `intent_parser.py:115` `IntentType.REVIEW_VERDICT` payload 字段:
  `("target_proposal_msg_id", "verdict")` — `verdict` 单数, 单值.
- Critic agent prompt 模板基于 v0.6 review-one-proposal 模式.
- 结果: 一组 K variant 收一个 verdict, 整批 KEEP 或整批 REVERT.

**严重度**: **P1** — explore round 内 partial-success (3 KEEP + 2
REVERT) 场景下无法精确传 verdict; 退化为"一刀切批准/驳回".

**关联**: §3.3 §4.3 / §3.5 §5 / M5 §5 step 5; `intent_parser.py:115`;
critic prompt builder.

**修复路径建议**:

1. `REVIEW_VERDICT` payload schema 加可选字段
   `verdict_map: dict[str, dict[verdict_fields]]`.
2. critic prompt 改输出 per-variant verdict; v0.6 单 verdict 路径作
   fallback (单 variant case).
3. Coordinator `_handle_review_verdict` 检测到 `verdict_map` 时按 variant
   拆分 explore grid (approved-only).

---

## 6. P2 次要 gap (cosmetic / observability)

### Gap-12 — T0 落在 CLI 不在 Coordinator

**设计**: §3.2 §5.1 暗示在 PRELUDE phase 内由 Coordinator 跑.
**实际**: `cli.py::_bootstrap_cortex_kb` 在 Coordinator 构造前跑.
**影响**: SDK / 测试 / 非 CLI 入口启动时 T0 不跑 → warm_start 字段空.
实际生产路径 (CLI) 不受影响.
**严重度**: **P2**.

### Gap-13 — Legacy stub yamls 未清理

设计 §3.15 §2.3 列 `dream` / `re_explore` / `comm_optimization` /
`compiler_tuning` 为 removed.
实际 yaml 仍在 `actions/_meta/`, executor 已未注册 (走 _NOOP_KINDS).
**影响**: ActionRegistry 加载这些 yaml, prompt 中 action 枚举噪声 +4 行.
**严重度**: **P2**.

### Gap-14 — `explore_search` 未加入 CORE_STATE_FIELDS

设计 §3.10 §6.2 列 explore_search 为 Coordinator-only.
实际 `policy.py CORE_STATE_FIELDS` 不含 `explore_search`.
**影响**: 理论上 LLM `update_state{changes: {explore_search: {...}}}` 可越权
篡改 ledger (现实中 LLM 不会; 是防御性 gap).
**严重度**: **P2**.

### Gap-15 — Phase budget proxy 与真 plateau 双轨 (R-09)

§3.14 R-09 已识别. M7 已实装真 plateau, 但 legacy resume 下仍依赖
`params_no_promote_streak` proxy.
**影响**: v0.6 → v0.8 灰度期短暂双轨, 操作员 confusion.
**严重度**: **P2** (设计明知).

### Gap-16 — KnowledgePlane help 文字提示其存在但路径不通

cli.py `--no-pr-monitor` flag 描述 "switches the KnowledgePlane.pr_feed_warm
behavior" (~2736), 但 KnowledgePlane 没被 bootstrap (Gap-2). 操作员
看 help 会以为该 flag 有效.
**影响**: 文档与行为不一致.
**严重度**: **P2**.

---

## 7. 影响面与修复路径建议

### 7.1 影响项目目标达成的判断

KB_design 顶层设计目标 (引自 KB_design.MD 第 2 章 概念目标):

| 目标 | 现状评估 |
|---|---|
| **流程固化** (主轴 A) | ⚠️ phase 字段 + 状态机 ✅, 但 phase 入口 hook 缺 (Gap-4, Gap-5, Gap-6) → 操作员看 phase 已切换, 但行为没切换 |
| **跨 session 知识** (主轴 B) | ⚠️ T0 ✅, T2/T3/T4 ✅ (粒度不对 Gap-7, Gap-8), KnowledgePlane 未接 (Gap-2) → 部分落地 |
| **sub-agent 双形态** (主轴 C) | ❌ 完全未落地 (Gap-1, Gap-3) → 生产路径 0 个 specialist |
| 决策落 LLM, 砍 scoreboard | ✅ Inv-9.1 完整满足 |
| breakdown v2 可观测 | ✅ Inv-12 完整满足 |
| schema 迁移 + 兼容 | ✅ Inv-10 完整满足 |
| PolicyGate 5 条新规则 | ✅ R1–R5 完整 + 正交 |

**结论**: 当前代码状态下, fresh session 跑出来的 trace 大致是:

```
v0.6 行为
  + phase enum (但 phase 切换不带新副作用)
  + scoreboard 删除 (LLM 决策无评分)
  + KB hypothesize/verify (但粒度粗)
  + breakdown v2 字段
```

**主轴 C 完全没启用**. 项目"v0.8 三主轴齐" 的设计目标 **目前不可达**.

### 7.2 修复优先级建议

**Sprint-1 (1–2 周, 解 P0)**:

1. Gap-2 → Gap-1 → Gap-3 这条链是 specialist 整链, 必须一起做.
   预计 4–6 个 PR (KnowledgePlane bootstrap → SpecialistRunner 注册 →
   `_handle_specialist_done` → 端到端 smoke test).
2. 落 §3.13 M5 / M6 验收清单的 fresh-session 烟测.

**Sprint-2 (1–2 周, 解 P1)**:

3. Gap-4 / Gap-5 / Gap-6: phase 入口 hook 三连. 可分独立 PR (互不依赖).
4. Gap-7 / Gap-8: per-variant T2/T3. 配对 PR.
5. Gap-9: `gaps[]` 字段 + Coordinator `_refresh_gaps`. 单独 PR.

**Sprint-3 (清理 / cosmetic)**:

6. Gap-10: legacy backends/params/validate_stack 入口关闭 (需要先确认
   M3 实际 GA, 不再需要 fallback).
7. Gap-11: Critic per-variant verdict_map (Critic agent 协作大改动).
8. Gap-13 / Gap-15 / Gap-16: 清理 / 文档对齐.

---

## 8. 边界条件 / 失败模式 (gap 跑起来会怎样)

| 场景 | 现状 | 应当 |
|---|---|---|
| Orch LLM emit `delegate{action='specialist', ...}` | task enqueue → dispatcher → `SubAgentRunner.run_task` raise `no_executor` → task `failed` → empty `delegated_result` 进 inbox | task 跑成功, specialist transcript 完整, proposal_set 进下一 explore round |
| 进入 KERNEL phase 时 LLM 漏掉 `profile` | `_required_next_step` 持续 nag; 若 LLM 一直忽略, `select_kernels` 永远缺 trace, kernel_opt 不可发起 | Coordinator 自动 enqueue profile, LLM 进入 KERNEL 即看到 trace 已就绪 |
| 进入 SWEEP phase 时 LLM 漏掉 `sweep` | LLM 必须主动 propose; 若 max_minutes 用尽前没 propose, `_enter_closing_phase` 强制 enqueue report 跳过 sweep | Coordinator 自动 enqueue sweep, grid 来自 warm_start_recipe |
| 用户提前 Ctrl-C 在 EXPLORE 中段 | wall-clock 未到 closing_phase → `_enter_closing_phase` 不触发; cli `finally` 写 breakdown; `stop()` 跑 T4 drain+commit | CLOSE 5 步 sequencer 顺序跑, 即使提前停 |
| Cortex T2 一次 hypothesize 4 variants | 一条 edge attrs={proposal=...}; KB 看不到具体哪个 variant | 4 条 edge, attrs={variant: ...}; KB negation 边可指向具体 variant |
| Critic 收到 5 variants 想批准 3 个 | 只能一刀切 KEEP 或 REVERT | verdict_map per-variant 拆 |

---

## 9. 验收标准 (gap 修完后该看到什么)

- [ ] fresh session 启动后, breakdown.specialist_runs 至少 1 行 (M5
      验收清单).
- [ ] EXPLORE phase 内 `specialist_rounds` ledger 非空; `proposals_total
      > 0`; `kb_provenance.points_created` 含 ≥ 1 个
      `kind=optimization_node` per variant.
- [ ] KERNEL 入口的 `phase_history[entered=KERNEL]` 后 1 个 tick 内,
      `last_profile_trace != ""`.
- [ ] SWEEP 入口的 `phase_history[entered=SWEEP]` 后 1 个 tick 内,
      有一条 `sweep` task 入队.
- [ ] CLOSE phase 转入后, `report → session_breakdown → drain → commit`
      四步在 phase_history evidence 中按时序可见.
- [ ] breakdown.attribution.phase_breakdown.explore.by_specialist_domain
      非空 (per Inv-12.2 → Gap-9 `gaps[]` 完整后).
- [ ] PolicyGate denial 中无任何 `kb_write_unauthorized` 或
      `tool_whitelist_*` 命中 from specialist (R4 / R5 vacuous true →
      actual true).
- [ ] cli `--no-pr-monitor` 关闭后, breakdown.warnings 含
      `pr_monitor_unreachable` 或等价 marker (R-03 探测信号).

---

## 10. 依赖与影响面

- **上游影响**: 修复需要 KnowledgePlane / SpecialistRunner / Critic
  per-variant 协作 — 牵涉 cli.py, coordinator.py, intent_parser.py,
  knowledge_plane.py, specialist_runner.py, shared_state.py.
- **下游影响**:
  - claw-stats-service / hyperloom-results-service dashboard 升级路径
    (M4 cheatsheet § 3.2 已规划).
  - Cortex KB schema (R-13 已识别): per-variant 边数量上升 → 视一次
    explore round produce K 条 hypothetical edges 而非 1 条.
  - LLM API quota (R-05): specialist 实际跑起来 + max_turns × N domain
    × research_lane_capacity 会显著抬 QPS; M6 默认 capacity=6 实际未
    生效 (Gap-1 阻断), 修复后需要重新评估配额.

---

## 11. 哲学回引

本 gap 文档不引入新设计, 只是 §3.1 三主轴的"履约审计".

- **主轴 A** 在字段层 / 状态机层落地, 但**副作用层**没接 (Gap-4/5/6).
  这与 §3.1 §3.1 "phase 转移由 Coordinator 强制" 的口号一致, 但
  *Coordinator 强制了什么* 其实只到"切标签", 没到"切行为". 设计意图
  是"切 KERNEL 就该跑 profile", 现实是"切了但等 LLM".
- **主轴 B** 的 KnowledgePlane 是把 KB / PR / source 统一收口的 facade,
  现在收口了但没接到生产入口 (Gap-2). T2/T3 粒度问题 (Gap-7/8) 是
  *实现细节落后于设计*, 不是设计错.
- **主轴 C** 的 specialist 框架, 代码 + 测试都齐, 缺一行
  `_register_executors` 的注册 — 这是最讽刺的 gap: §3.5 全章 + M5 / M6
  全部 PR 加起来 ~3000 行代码, 因为最后一步接线没接, **全部不生效**.
  这条 gap 也是 P0 中**修复成本最低**的一条.

修这 3 个 P0 + 4 个 P1 (Gap-4/5/6/9) 后, v0.8 三主轴可达"GA candidate"
状态. 余下 P2 是后续清理.

最后一句:

> KB_design 没有错, 代码也没有 bug. 缺的是**把代码接线接到底**.
> v0.8 的设计目标和当前代码之间的距离, 用工程语言量化 ≈ 3 个 PR.

---

# 12. 废止 / 死代码清单 — 干扰 LLM 理解的部分

> 本章为体检报告的第二部分: 列出**物理上仍在 tree 里、但 KB_design v0.8
> 已明确废止**的代码工件. 主要目的是让未来 *AI coding assistant* 或人类
> reviewer 看 repo 时不会被这些工件**误导**, 以为它们仍是 canonical path.
>
> 与第 4–6 节"功能 gap"的区别:
> - 第 4–6 节: 设计要 X 行为, 代码没接 X. ➜ 需要**新增**接线.
> - 本节: 设计要砍 Y, 代码 Y 还在. ➜ 需要**删 / 标 deprecated**.
>
> 两类问题在 fresh-session 端到端跑出来的 trace 上**互相加强**: 死代码
> 让 LLM "看见"旧路径 → 拉走 token / 选择老 action → 进一步遮蔽功能 gap.
>
> 风险分级 (与第 4–6 节独立):
> - **HIGH-misleading**: 看着像 v0.8 canonical, LLM 极可能误选
> - **MEDIUM-noise**: 需要 reader 主动忽略
> - **LOW-historical**: 纯历史痕迹, 影响 ≈ 0

## 12.1 总览矩阵 — 7 大类废止物

| 类别 | 件数 | 顶级风险 | 影响主轴 |
|---|---|---|---|
| A. 已废止 action yaml + executor 仍 registered | 11 | HIGH | A / B (explore 合并形同虚设) |
| B. scoreboard 残骸 (stub 方法 + prompt 残句 + denial hint) | 12 | HIGH | A (主轴 A 的支撑) |
| C. validate_stack 死路径 (prompt + sequence gate + executor) | 8 | HIGH | A (M3 inline rebench) |
| D. `_KERNEL_OPT_PIPELINE_BODY` 仍说 "scoreboard decides" | 1 | HIGH | A / B |
| E. cortex_kb_flusher 单文件存在但未被 cli 拉起 | 1 | MEDIUM | B |
| F. orchestration / kernel / robustness `.md` 系统提示仍含 v0.6 vocab | ~10 | HIGH | 全部 |
| G. tests 仍在测已废止行为 (反向锁住 deprecation) | 8 | MEDIUM | 全部 |

## 12.2 类 A — 已废止 action 工件链 (yaml + executor + 注册)

KB_design §3.4 / §3.15 §2.3 / M3 明确: `backends` / `params` /
`validate_stack` 三个 action 合并入 `explore`; `dream` /
`re_explore` / `comm_optimization` / `compiler_tuning` 四个动作由
specialist domain 替代. 但当前 tree 仍**全链 active**:

### A.1 backends 链 (HIGH-misleading)

| 件 | 路径 | 状态 |
|---|---|---|
| yaml meta | `inference_optimizer/actions/_meta/backends.yaml` | 仍在; ActionRegistry 加载 |
| executor 模块 | `orchestrator/action_executors/backends.py` (~500+ 行) | 仍 import 自 `__init__.py` |
| executor 注册 | `cli.py:627` `_REAL_EXECUTORS_FULL["backends"] = backends_executor` | **每个 fresh session 注册** |
| phase 允许集 | `phase_state.py:91-97` EXPLORE 仍含 `"backends"` | **PolicyGate R1 仍放行** |
| prompt 可见 | `prompt_builder.py:59-65 FULL_ENABLED_ACTIONS` 含 `"backends"` | **LLM 看得见** |
| orchestration.md 规则 | `orchestration.md:32-42` 把 backends 列为 EXPLORE 合法 action | **LLM 看得见** |
| orchestration.md 示例 | `orchestration.md:95-96` `delegate{action_name='backends'...}` | **LLM 看得见 + 模仿** |
| coordinator promote 分支 | `coordinator.py:_promote_to_shared_state` 内含 `task_kind == "backends"` 分支 (~3585+) | **接收 backends 结果并改 state** |
| 测试锁死 | `tests/test_backends_search.py` 全文 | **测试通过, 让人误以为是 v0.8 canonical** |

### A.2 params 链 (HIGH-misleading)

同 A.1 结构, 文件: `params.yaml`, `params.py`,
`tests/test_params_search.py`, `tests/test_p2_3_param_executors.py`,
`cli.py:628`, coordinator promote `task_kind == "params"` 分支 (~3703+).

### A.3 validate_stack 链 (HIGH-misleading)

| 件 | 路径 |
|---|---|
| yaml | `validate_stack.yaml` |
| executor | `action_executors/validate_stack.py` |
| 注册 | `cli.py:639` |
| phase 允许 | `phase_state.py:96` |
| **prompt MANDATORY** | `prompt_builder.py:336-339` "MANDATORY `validate_stack` after KEEP" |
| **decision rule** | `prompt_builder.py:479-481` 决策第 3 步: `validate_stack required` TODO |
| orchestration.md | `:59, 100-103` 同样 mandatory |
| coordinator gate | `coordinator.py:1586-1660, 1852-1868` `_required_next_step` TODO 4 检测 unvalidated keeps |
| shared_state field | `last_validate_stack` (~238) 仍被读写 |
| shared_state helper | `optimization_stack_has_unvalidated_keeps` (~2383+) 驱动上述 TODO |
| 测试锁死 | `tests/test_validate_stack.py`, `tests/test_phase2_mission_and_validate_stack.py` |
| agent-facing doc | `inference_optimizer/actions/validate_stack.md` 全文 |

7 大类中**矛盾最大**的一组: M3 设计明说 "validate_stack 内嵌进 explore
的 KEEP-后-stack-rebench", 而 prompt 仍 mandatory 它, coordinator
仍 enforce TODO 4. LLM 看到 prompt → propose `validate_stack` → 命中
TODO → 跑 legacy executor → ledger 写 `last_validate_stack` → 完全
绕过 explore inline rebench.

### A.4 dream / re_explore / comm_optimization / compiler_tuning yaml (MEDIUM-noise)

| 件 | 路径 | 状态 |
|---|---|---|
| yaml | `actions/_meta/{dream,re_explore,comm_optimization,compiler_tuning}.yaml` | 仍在; ActionRegistry 加载 |
| executor | **不存在** | (好) |
| `_REAL_EXECUTORS_FULL` | **不含** | (好) |
| `FULL_ENABLED_ACTIONS` | **不含** | (好) |
| 测试 | `tests/test_p1_2_full_action_catalogue.py` 仍断言它们在 registry | 锁住了 noise |

cli.py 注释 (662-669) 解释这些动作的 noop_prep 已移除, 实际派遣会产生
`no_executor` fail. 但 yaml 仍占用 ActionRegistry 视野, breakdown
capability_summary 一些渲染逻辑仍按 family 分类时把它们当未尝试 action.

### A.5 explore 入口存在但 promote 缺失 (HIGH-misleading)

最讽刺的一项: `explore` 是 v0.8 canonical action, executor 已注册,
但:

- `coordinator._promote_to_shared_state` 全文**没有** `task_kind ==
  "explore"` 分支 (~3356-3886). 落到 generic fallback.
- `SharedState.apply_explore_search_update` (1997-2053) +
  `record_explore_accepted` (2055+): **生产代码 0 个调用点**. 只测试
  用 (`rg "apply_explore_search_update|record_explore_accepted"` 仅命中
  `shared_state.py` + `tests/test_v08_m3_explore.py`).
- `_sequence_denial_for_action` 的 `sequence_actions` 集合 (1757-1764)
  **不含** `explore`, 所以 explore 派发跳过所有顺序门 (validate_stack /
  profile). 与 A.3 prompt mandatory validate_stack 直接冲突.

LLM 读 KB_design 会以为 explore 是 GA. 实际看代码: M3 PR 只接了
*executor 入口*, *promote 出口* 没接. 一条只有起点和终点没有桥的路.

## 12.3 类 B — scoreboard 残骸

KB_design §3.9 / M2 明确删除"action_scores / scoring.py / MARATHON_PRIORS
/ Action scores prompt 块". 实际仍残留:

### B.1 Coordinator 的 no-op stub 方法仍在生产路径被调 (MEDIUM-misleading)

`coordinator.py:601-641` 保留一组 no-op 占位方法: `_score_action_keep`,
`_score_action_failure`, `_score_action_no_promote`, `_score_action_lock`,
`_apply_action_score_update`. 函数体都是 `return None`. **但调用方仍
在调**:

- `coordinator.py:2556-2591`: kernel_opt / integrate KEEP/REVERT 后调
  `_score_action_keep` / `_score_action_no_promote`
- `coordinator.py:3667-3673, 3814-3821, 3865`: sweep / params / backends
  完成后调 `_apply_action_score_update`

死路径 (no-op 返回) ≈ 5 处. LLM 读 coordinator 会认定这是真的 scoring
行为 (函数名 + 调用频率都暗示活跃).

### B.2 `_ensure_action_scores_seeded` 在 `__init__` 中被调 (MEDIUM-misleading)

`coordinator.py:421-426`: `self._ensure_action_scores_seeded()`. 函数
本身已退化为 no-op stub (§3.9 实现); 但调用点保留 + 上下文注释提到
"scoring.py" / "scoreboard renderer", 误导效果强.

### B.3 `all_top_actions_policy_locked` 死分支 (LOW)

`shared_state.py:1357-1365` 永远返回 False. `coordinator.py:1013-1018`
`_has_no_more_leverage` 内的分支永远不进入. (`coordinator.py:1002-1074`
整个函数仍用 `params_search` / `backends_search` 做 plateau, 与 M3
explore_search 双轨.)

### B.4 Live policy denial hint 仍说 "Action scores" (HIGH-misleading)

`coordinator.py:2292-2295`: delegate 被 deny 时, hint 文本仍含
"pick another action from Action scores". 这是 LLM **每次 R1 deny
都看见**的 inbox 字符串. 明显矛盾 prompt_builder 中 §3.9 通告
"scoreboard 已退场".

### B.5 prompt_builder 内部矛盾 (HIGH-misleading)

`prompt_builder.py:493-495` 写 "v0.8 retired the v0.6 Action scores"
(对). 但同一份 prompt 在:

- `336-339` 仍要求 validate_stack mandatory (类 C)
- `479-481` 仍把 validate_stack 当决策第 3 步
- `697, 741-744` `_KERNEL_OPT_PIPELINE_BODY` 内仍说 "scoreboard
  surfaces kernel_opt", "scoreboard decides" (类 D)

LLM 读起来会得出"评分被砍, 但 scoreboard 还在驱动 kernel_opt"的内部
矛盾结论.

### B.6 prompt_builder 提到 `backends_search` / `params_search` (MEDIUM-noise)

`prompt_builder.py:600-614` 的 ledger 描述用 `backends_search` /
`params_search` 字段名. v0.8 canonical 是 `explore_search`.

## 12.4 类 C — validate_stack 死路径

已在 A.3 详列, 此处不再展开数据点. 强调**跨文件矛盾**:

| 文件 | 说什么 |
|---|---|
| KB_design §3.4 / M3 | validate_stack 已内嵌 explore |
| `actions/_meta/validate_stack.yaml` | 仍存在 |
| `actions/validate_stack.md` (agent-facing) | 仍存在 |
| `phase_state.py:96` | EXPLORE 仍允许 |
| `prompt_builder.py:336-339, 479-481` | MANDATORY |
| `orchestration.md:59, 100-103` | MANDATORY |
| `coordinator.py:1586-1660, 1852-1868` | TODO 4 enforce |
| `shared_state.last_validate_stack + optimization_stack_has_unvalidated_keeps` | 仍写读 |
| `tests/test_validate_stack.py` | 锁定行为 |
| `tests/test_phase2_mission_and_validate_stack.py` | 锁定 prompt |

8 处至少. 删除任一处都会让现有测试红, 但保留全部就意味着 v0.8 设计
*在生产路径上不存在*.

## 12.5 类 D — `_KERNEL_OPT_PIPELINE_BODY` 仍把 scoreboard 当真 (HIGH-misleading)

`prompt_builder.py:697, 741-744`: kernel_opt 段说明仍写

```
"... the scoreboard surfaces `kernel_opt` ..."
"... the scoreboard decides ..."
```

`prompt_builder.py:846-847`: kernel 阶段开启时这段会被注入 system
prompt. 与 §3.9 矛盾.

## 12.6 类 E — `scripts/cortex_kb_flusher.py` 单飞 (MEDIUM-noise)

| 件 | 路径 |
|---|---|
| daemon 实现 | `inference_optimizer/scripts/cortex_kb_flusher.py` (~190 LoC) |
| 文档承诺 | KB_design §3.6 把 flusher 当 production-required daemon |
| 实际接线 | `cli.py` 全文**未 spawn / import** `cortex_kb_flusher` (验证: `rg cortex_kb_flusher cli.py` 0 命中) |
| 路径常量 | `paths.py:~88` `.kb_flusher.pid` 仍定义 |

结果: 一次 fresh session 启动**不会**有 flusher 进程; T2/T3 写入仅
靠 in-process NDJSON write (Coordinator T2/T3 hook 内同步执行). 如果
Cortex 临时不可达, NDJSON 兜底正常, 但**没有后台 drain**, 等到下次
session 启动才有可能补送 — 不符合 §3.6 / §3.14 R-01/R-02 缓解方案.

## 12.7 类 F — 静态 prompt `.md` 仍含 v0.6 vocab

`orchestrator/system_prompts/*.md` 是 system prompt fragment, 每个角色
每 tick 注入. **比 prompt_builder.py 还危险**, 因为 reviewer 看代码
(`prompt_builder.py`) 时容易忽略静态 md.

| 件 | 行 | 问题 |
|---|---|---|
| `orchestration.md` | 32-42 | EXPLORE 允许集列 `backends, params, validate_stack` |
| `orchestration.md` | 59, 100-103 | 强制 validate_stack |
| `orchestration.md` | 95-96 | 示例 `delegate{action_name='backends'}` |
| `critic.md` | 15-18 | EXPLORE 列 backends/params/validate_stack |
| `kernel.md` | 1, 5-6 | 头部声明 "v0.6", 引用 "DESIGN §7.2" (DESIGN.md 不存在) |
| `kernel.md` | 36-42 | "v0.6 invariant unchanged" — 与 v0.8 phase 不变量混淆 |
| `robustness.md` | 1, 5 | 自称 "v0.7" |
| `robustness.md` | 86 | `prune_branch` families 含 `backends`, `validate_stack` |

这些 md 是 LLM 角色的"宪法". v0.6 vocab 残留 → 角色按 v0.6 行事.

## 12.8 类 G — 锁住废止行为的测试

| 测试 | 锁住什么 | 是否应保留 |
|---|---|---|
| `tests/test_backends_search.py` | backends ledger 仍工作 | M3 GA 时 delete |
| `tests/test_params_search.py` | params ledger 仍工作 | M3 GA 时 delete |
| `tests/test_p2_3_param_executors.py` | params executor 接口 | M3 GA 时 delete |
| `tests/test_validate_stack.py` | validate_stack E2E | M3 GA 时 delete |
| `tests/test_phase2_mission_and_validate_stack.py` | mission TODO 4 | M3 GA 时 重构 |
| `tests/test_prompt_builder.py` | 断言 `validate_stack` 在 enabled actions | M3 GA 时 重构 |
| `tests/test_prompt_assets.py` | 断言 `validate_stack` 在两种模式 prompt 中 | M3 GA 时 重构 |
| `tests/test_critic_prompt_builder.py` | critic prompt 含 validate_stack | M3 GA 时 重构 |

每个都是当前**绿**的, 因此 v0.8 完整删除时会变红. 这是"测试反向锁住
废止"模式 — 测试存在的唯一作用是不让删除发生.

## 12.9 类 H — 其他细节 (LOW-historical, 列出供清理)

| 件 | 路径 | 性质 |
|---|---|---|
| 注释自称 v0.6 | `cli.py:2488` `description="Inference Optimizer v0.6"` | argparse description, 用户首屏可见 |
| 模块 doc v0.6 | `shared_state.py:1-11` "DESIGN v0.6 §8.3" | DESIGN.md 不存在 |
| 模块 doc v0.6 | `action_registry.py:1-51` "Budget-Aware Scheduler scoring" | |
| 模块 doc v0.6 | `action_executors/__init__.py:1` "DESIGN v0.6 §15.2" | |
| 模块 doc v0.6 | `cli.py:1` "DESIGN v0.6 §22" | |
| 模块 doc | `coordinator.py:19` "scheduler scoring" | §3.9 退场 |
| `phase_state.py:85-86` | 注释承诺 "M3 will inline" 但 allowlist 仍含 validate_stack | TODO 没兑现 |
| `session_paths.py:69-91` | workspace fallback 含 dream/re_explore 目录名 | 兜底死路径 |
| `paths.py:~169, 198-207` | 引用 "DESIGN §20", legacy $HYPERLOOM_ROOT / Magpie paths | 环境兼容, 仍 active |
| `prompt_builder.py:77-81, 80` | 指 `remain_todo.md` 但**文件不存在** | 断指针 |
| `cli.py` 多处 | 同上断指针 | |
| `examples/p0_main_loop.py`, `p2_full_optimize_demo.py` | demo 文件用 v0.6 phase 概念 | 不影响生产, 影响新人 onboarding |
| `breakdown/SKILL.md:34` | capability_summary 列 backends/validate_stack | §3.12 alias OK, 不算 dead |
| `breakdown/reporters/_renderers/param_search.py:16+` | 渲染 `backend_winners_history` | 仅 breakdown alias |

## 12.10 跨文件矛盾热点 (LLM 读起来最迷惑的地方)

| # | 矛盾 | 位置 |
|---|---|---|
| 1 | `prompt_builder.py:493-495` 说"scoreboard 已退场", 但 `:336-339` 说 validate_stack mandatory, `:697-744` 说 scoreboard decides kernel_opt | 同一文件内部三处 |
| 2 | KB_design §3.4 M3 说"删除 validate_stack", 但 `orchestration.md:59, 100-103` 仍 mandatory, `coordinator.py:1586-1660` 仍 enforce TODO 4 | 跨 design ↔ static prompt ↔ coordinator |
| 3 | KB_design §3.9 M2 说"删除 action_scores", 但 `coordinator.py:601-641` 仍有 `_score_action_*` stub 函数, 仍被 5 处调用, denial hint 仍说 "Action scores" | 跨 design ↔ coordinator stub ↔ live denial hint |
| 4 | KB_design §3.6 M1 说"flusher 后台 drain", 但 `cortex_kb_flusher.py` 单文件存在, cli 不拉起 | design ↔ script 接线 |
| 5 | `cli.py:2488` argparse description 自称 v0.6; `kernel.md:1` 自称 v0.6; `robustness.md:1` 自称 v0.7 | 三个角色版本号都不一致 |

## 12.11 优先清理路径

按"对 LLM 误导面"排序:

### Sprint-A (HIGH-misleading, 应优先):

1. **删除 / 关闭 backends + params + validate_stack 入口** (类 A.1-A.3):
   - PolicyGate R1: `phase_state.py:91-97` 移除三个 legacy action 名,
     或新增 `action_deprecated` 规则 (M3 §PR7 设计 — 设计有, 代码没)
   - cli.py:627-639: 删除三个 executor 注册
   - `prompt_builder.py`: `FULL_ENABLED_ACTIONS` / `NO_KERNEL_ENABLED_ACTIONS`
     去掉, 删除 validate_stack mandatory 段, 删除决策 step 3 TODO
   - `orchestration.md` / `critic.md`: 同步删除 validate_stack / backends /
     params 提及
   - `coordinator.py`: `_required_next_step` TODO 4 删除;
     `_promote_to_shared_state` 三个 task_kind 分支删除
   - 同步重构 8 个测试 (类 G)
   - **接上类 A.5**: `_promote_to_shared_state` 加 `task_kind ==
     "explore"` 分支, 调 `apply_explore_search_update` +
     `record_explore_accepted`

2. **删除 scoreboard 残骸** (类 B):
   - `_score_action_*` / `_apply_action_score_update` /
     `_ensure_action_scores_seeded` 全部删
   - 所有调用方 (5 处) 也删
   - denial hint "pick another action from Action scores" 改为
     "pick a phase-allowed action"
   - `_KERNEL_OPT_PIPELINE_BODY` 修文字
   - `shared_state.all_top_actions_policy_locked` 死分支删

3. **接 cortex_kb_flusher** (类 E) — 这是真接线 gap, 跨入第 4-6 节
   范畴; 优先级在死代码清理之上, 因为 §3.14 R-01/R-02 缓解依赖它

### Sprint-B (MEDIUM-noise):

4. **删除 / 标 deprecated dream / re_explore / comm_optimization /
   compiler_tuning yaml** (类 A.4); 修复 `test_p1_2_full_action_catalogue.py`

5. **修 kernel.md / robustness.md / orchestration.md 头部 version
   字段** + 删 v0.6 模块 docstring 中的 DESIGN.md 引用 (类 F + H)

6. **修 prompt_builder.py 内部矛盾** (类 D + B.5)

### Sprint-C (LOW-historical, 后续清理):

7. **删除 `actions/validate_stack.md` agent-facing doc**

8. **更新 examples/*.py 用 v0.8 phase 概念**

9. **修 `remain_todo.md` 断指针** (要么创建文件要么删引用)

10. **统一 argparse description 到 v0.8** (`cli.py:2488`)

## 12.12 删除工作量预估

| Sprint | 估时 | 文件数 | 测试需重构 |
|---|---|---|---|
| A | 1-2 周 | ~12 个文件 + 8 个测试 | 8 |
| B | 1 周 | ~6 个文件 | 1 |
| C | < 1 周 | ~10 个文件 | 0 |

**总计**: 约 28-30 个文件, 9 个测试需要重构 (主要是删除 backends /
params / validate_stack 锁定测试).

## 12.13 与第 4-6 节 (功能 gap) 的合并视图

| 关联项 | 死代码 (本章) | 功能 gap (第 4-6 节) |
|---|---|---|
| explore 整链 | promote 缺失 → executor 注册了但 outcome 不写 (类 A.5) | Gap-7/8 per-variant T2/T3 缺失 |
| specialist 整链 | (无死代码; runner 真没接) | Gap-1/2/3 |
| validate_stack | prompt mandatory + sequence gate + executor 全在 (类 C) | 不是 gap, 是反 gap (设计要删, 代码仍 enforce) |
| scoreboard | stub + denial hint + prompt 残句 (类 B) | 不是 gap (设计已退场, 残骸应清理) |
| flusher | 文件存在但 cli 不拉 (类 E) | 关联 R-01/R-02 缓解能力 |
| dream / re_explore | yaml + 测试锁住 (类 A.4) | 不是 gap |

**整体观感**: 项目当前像"v0.6 + v0.7 + v0.8 的并集" — 每一代功能都
有部分代码留下来, 没有一代被完整退役. LLM agent 读这个 repo 时, 第
一感觉会是"v0.6 仍是主路径, v0.8 是上层 patch", 与 KB_design 的
"v0.8 是新 GA, v0.6 已废" **完全相反**.

修这两章 (功能 gap + 死代码清理) 后, repo 才真正进入 "v0.8 single
source of truth" 状态.

## 12.14 哲学回引

本章实际上是 §3.13 milestone "additive 原则" 的反面教材: KB_design
要求"每个里程碑可单独回退", 实施时往往保留 legacy 入口以方便灰度.
M3 文档预设了 `--legacy-explore-split` 这种 flag (§3.13 §2.3) 来管理
灰度, 但**flag 没实现**, 灰度逻辑 = "yaml + executor + 注册全保留",
这就把灰度时间窗永久化了.

> 灰度永远是临时方案. 没有删除日历的 deprecation 等于 v0.6 的另一个
> 名字.

具体到清理顺序, §3.1 §5 决策仲裁表中"加新 deprecated rule vs 让 LLM
自律" 优先选 PolicyGate 规则 — Sprint-A 第 1 项的核心就是落地 M3 §PR7
的 `action_deprecated` PolicyGate 规则, 让 deprecation "硬"起来, 不再
依赖 prompt 文字劝阻.
