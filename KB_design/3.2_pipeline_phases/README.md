# §3.2 五段固化管线 — 进入/退出条件与边界

## 1. 设计目标

把 v0.6 的"任意 phase 之间自由跳"演化为线性 5 段管线
`PRELUDE → EXPLORE → KERNEL → SWEEP → CLOSE`,
每段都有**机器可判定的进入条件**和**机器可判定的退出条件**, 让
Coordinator (而非 LLM) 拥有 phase 的所有权。

## 2. 现状回顾

v0.6 中的 8 个 `pipeline_phase` (`prep / measure / explore / analysis /
deep / validate / finalize / support`) 只是 yaml 的展示字段, 不参与
任何控制流。LLM 决定下一步做哪个 action 时只看 scoreboard 和 prompt
中的"DECISION FRAMEWORK", 这意味着:

- 一个失败的 baseline 后, LLM 可能直接 `propose_action='kernel_opt'` —
  现在靠 PolicyGate 的"先验"和 prompt 文字劝阻; 不可靠。
- `validate_stack` 和 `sweep` 的差异完全靠 LLM 理解, 经常被混用。
- 没有"我们到底走到哪一步了"的概念, 用户无法在 monitor 中说出"还有
  10% 在 EXPLORE 阶段"这种话。

v0.8 把 phase 升级为一等公民, 由 Coordinator 持有并强制执行。

## 3. 不变量

继承 §3.1 全部三主轴 + 三不变量。本节额外引入两条 phase 专属不变量:

### Inv-2.1 — phase 单调性

phase 一旦进入就**只能往后走或退到 CLOSE**, 不能回退到更早的阶段。
即转移图是一棵**单根有向无环图**, 任意 (src, dst) 边数 ≤ 1, 且
`dst.index ≥ src.index`。例外: `recover` 是 phase-orthogonal, 不算
转移。

### Inv-2.2 — phase 退出必须有原因

退出当前 phase 时必须写一条 phase_history 记录, 包含 `from_phase`,
`to_phase`, `reason`, `evidence`, `ts`。reason 必须是 `phase_exit_reasons`
词表中的一员 (词表见 §6)。无 reason 不允许转移。

## 4. 五段管线总览

```
              PRELUDE              EXPLORE                KERNEL              SWEEP              CLOSE
              ───────              ───────                ──────              ─────              ─────
  长度          固定短              主体                  主体                 短/可选             固定短
  并发          单 task             specialist 并行       单 task             单 task             单 task
  serving GPU  独占                 整段独占              整段独占             整段独占            空闲
  Cortex 写入  T0                   T2/T3 (高频)         T2/T3 (低频)         T3 (低频)          T4
  退出由       数据齐               LLM judge + plateau  retry budget       grid 完成           固定步
  允许 actions target_analysis,    explore, specialist  profile (1×),      sweep              report,
                baseline, recover                       kernel_opt,                            session_breakdown
                                                       integrate,
                                                       deep_kernel_*
```

## 5. 各段详细设计

### 5.1 PRELUDE — 数据齐备 + Cortex warm-start

**目标**: 让所有后续 phase 拥有需要的元数据 (model_class / framework /
gpu_type / TP), 确定 baseline_tput 和 baseline_accuracy, 从 Cortex 拉
warm_start 写到 SharedState。

**进入条件**:

- session 启动 (一次性)
- resume 时若 `phase` 字段尚未存在 (来自 v0.6) 也进入此段, 但跳过
  baseline (依赖现有 `baseline_tput > 0` 判定)。

**phase 内动作**:

1. T0 Cortex `session begin` + `find-recipe` + `traps`。
2. `target_analysis` (现有 hard-gate, 不动)。
3. `baseline` (现有, 不动)。
4. 若 baseline 成功且 `baseline_tput > 0` → 触发退出。

**T0 位置约定 (v0.8 KB_gaps/Gap-12 — 落地)**:

T0 在 session boot 时跑, 而不是在 Coordinator reactor 的第一次 tick
内. 两条入口都通过同一个 helper
(`inference_optimizer.orchestrator.cortex_t0.run_t0_anchor`) 跑同样的
4 步, 区别仅在失败语义:

| 入口 | 触发时机 | 失败语义 | banner 输出 |
|---|---|---|---|
| **CLI** (canonical, `cli._bootstrap_cortex_kb`) | `Coordinator(...)` 构造**前**, 在 `_seed_shared_state` 之后 | fail-fast (`sys.exit(2)`) — 操作员立即看到 Cortex 故障 | `print(...)` 到 stdout, 操作员可见 boot banner |
| **Coordinator fallback** (`Coordinator._ensure_cortex_t0_anchored`) | `Coordinator.__init__` 内, `_ensure_phase_initialised` **之后** | fail-soft (warning + 空 warm_start) — 长时 reactor 不会因 Cortex 故障崩 | `log.info` 写 session 日志, 不污染 stdout |

Coordinator fallback 仅在以下条件**全部满足**时实际跑:

- `cortex_kb` 非 None (构造时传入了 client)
- `cortex_kb.enabled` (非 `--no-cortex`)
- `shared_state.cortex_session_id` 为空 (CLI 没 T0 过, resume 也没
  pick 到 sid)

设计意图: CLI 路径是**生产**入口, fail-fast 保留. Coordinator
fallback 服务于 SDK / 集成测 / 直接构造 `Coordinator(...)` 的场景,
保证 warm_start 字段不会因为绕过 CLI 而永空 (此前 KB_gaps/Gap-12
描述的 P2 表现).

**退出条件 (任一即转 EXPLORE)**:

- `baseline_tput > 0 ∧ stop_reason == None` → 正常退出, reason =
  `prelude_done`。

**退出条件 (任一即转 CLOSE, 跳过 EXPLORE/KERNEL/SWEEP)**:

- `baseline_failure_streak ≥ 3` → reason = `prelude_baseline_failed`。
- 用户 SIGTERM / 时间超限 → reason = `time_exhausted_during_prelude`。
- 任何**致命** `policy_loop` (>10 连 deny 同一 rule) → reason =
  `prelude_policy_loop`。

**phase 内允许 actions**: `target_analysis`, `baseline`, `recover`。
其他全部被 PolicyGate 拒绝并返回 `policy_denied: phase_incompatible`。

**回退策略**: 不回退。失败直接转 CLOSE。

### 5.2 EXPLORE — 并行 specialist + 串行 integrate

**目标**: 在不动 kernel 源码的前提下, 通过 server flag / sglang 参数 /
env 变量 / 运行时配置, 把 `current_best` 推到 *配置面的局部最优*。

**进入条件**:

- `phase == PRELUDE ∧ baseline_tput > 0` → 进入 EXPLORE,
  `phase_started_ts = now()`。

**phase 内节奏 (一轮 ≈ 数 specialist + 一批 explore variant)**:

1. Coordinator 把当前 gap 列表 (来自 baseline 表现 / Cortex
   `issue_node` traverse / 上轮 KEEP/REVERT 暴露的 gap) 切到 N 个
   domain。
2. 并行 `delegate{action='specialist'}` 派发 N 个 specialist (受
   research_lane 容量限制)。
3. specialist 各自跑 KB + PR + 源码探索, 返回 `specialist_done` 内
   的 proposal_set。
4. Orchestration 汇总, 去重 (按 explore_search.canonical_fingerprint),
   选 top-M 作为下一批 explore variant。
5. `propose_action='explore'` → Critic Review → KEEP/REJECT/REDIRECT。
6. explore executor 串行跑 M 个 variant, 每个 KEEP 进 stack /
   `_lift_to_current_best`, 每个 REVERT 写 `last_action_failures`。
7. 每个 KEEP 触发 Cortex `verify confirmed`, 每个 REVERT 触发
   `verify refuted` (negation edge)。
8. 一轮收尾后, 检查 plateau 判定 (§3.8)。

**退出条件 (任一即转 KEERNL)**:

- `plateau_explore` 信号触发 (§3.8 详细)。
- `explore_phase_budget_exhausted` (默认 wall-clock 60% 上限, CLI 可
  调) 即使没 plateau 也强制转。

**退出条件 (任一即转 CLOSE, 跳过 KERNEL/SWEEP)**:

- 用户请求 stop / 全局 `stop_reason` 设置为终态 (`time_exhausted` /
  `target_reached` / `no_more_leverage`)。
- robustness 持续报严重 alert (`crash_count ≥ 5` 或同类 alert 5 次内
  连续触发) → reason = `robustness_escalated`。

**phase 内允许 actions**: `explore`, `specialist`, `recover`。
所有 kernel-owned action / `profile` / `sweep` / `report` 拒绝。

**回退策略**: 不回退。EXPLORE 内 specialist 派发失败 → 该轮空; 连 K
轮空 → 触发 plateau 退出。

### 5.3 KERNEL — 一次 profile + 多轮 kernel_opt + integrate

**目标**: 在 EXPLORE 已经稳定的 `current_best` 之上, 进行内核级优化。

**进入条件**:

- `phase == EXPLORE ∧ (plateau_explore ∨ explore_phase_budget_exhausted)`。
- 且 `kernel_enabled == True` (即 `--no-kernel` 未开)。
- 否则**直接跳到 SWEEP**, 不进 KERNEL。

**phase 内节奏**:

1. **进入即跑一次 `profile`** (固定动作, 不需要 LLM propose)。该次
   profile 写入 `last_profile_trace`, 锚定 KERNEL 阶段所有
   `select_kernels` 的 trace_input。
2. LLM 可在 5 个 KERNEL_OWNED_ACTIONS + `deep_kernel_analysis` 内
   propose: `kernel_opt`, `integrate`, `operator_tuning`,
   `vendor_kernel_config`。
3. 每个 `kernel_opt` 触发 GEAK / OOB 候选生成, `integrate` 走 E2E
   bench + KEEP/REVERT 闸 + Cortex verify。
4. 同一 kernel_id 的 retry 上限保留 v0.6 的 `_DEFAULT_KERNEL_OPT_MAX_PARTIAL = 2`。

**退出条件 (任一即转 SWEEP)**:

- `plateau_kernel` (§3.8): 连续 K 个 kernel_id 都返回 REVERT 或
  NEEDS_REVIEW。
- `kernel_phase_budget_exhausted` (默认 wall-clock 25% 上限)。

**退出条件 (任一即转 CLOSE)**:

- 全局 `stop_reason` 终态触发 (同 EXPLORE)。
- robustness 严重 alert (同 EXPLORE)。

**phase 内允许 actions**: `profile` (仅 phase 入口的 1 次), `kernel_opt`,
`integrate`, `deep_kernel_analysis`, `operator_tuning`,
`vendor_kernel_config`, `recover`。

**回退策略**: 不回退到 EXPLORE。如果 KERNEL 阶段发现一个新的
configuration gap 应当在 EXPLORE 解决, LLM 把它作为 *finding* 写到
specialist 风格的提议, **存到下一次 session 的 KB warm_start** 而不是
回退当前 phase。

### 5.4 SWEEP — 跨 workload 验证 current_best

**目标**: 验证 EXPLORE/KERNEL 累积的 `current_best` 在不同
(CONC, ISL, OSL) 组合下的稳定性, 输出 pareto_front 给 report。

**进入条件**:

- `phase == KERNEL ∧ (plateau_kernel ∨ kernel_phase_budget_exhausted)`。
- 或 `phase == EXPLORE ∧ kernel_enabled == False ∧ explore 退出`。

**phase 内节奏**:

1. 自动构造 sweep grid (来自 SKILL.md 默认 grid + Cortex
   `recipe.sweep_grid` 字段, 后者优先)。
2. `sweep` action 串行跑每个组合, 每个组合都是 1 次 E2E bench。
3. 失败 ≤ ε 个组合时仍标 SUCCESS; > ε 标 PARTIAL。

**退出条件 (任一即转 CLOSE)**:

- grid 跑完 (无论 SUCCESS/PARTIAL) → reason = `sweep_done`。
- `sweep_budget_exhausted` 强制结束 (剩余组合标记 SKIPPED)。
- 全局 stop / robustness alert (同 EXPLORE/KERNEL)。

**phase 内允许 actions**: `sweep`, `recover`。

**回退策略**: 不回退。

### 5.5 CLOSE — 报告 + Cortex commit + breakdown

**目标**: 把本 session 的活动事实层 + 知识层完整落地。

**进入条件**: 上一个 phase 的退出。

**phase 内节奏 (固定顺序, 不允许 LLM 跳序)**:

1. Coordinator 内部触发 `report` action (生成 markdown / json 报告)。
2. Coordinator 触发 `session_breakdown` action (写
   session_breakdown.json v2)。
3. NDJSON flusher drain (等待异步 `_enqueue` 全部 POST)。
4. Cortex `session commit` 调用。
5. 退出 reactor loop, 进程结束。

**退出条件**: 上述 5 步成功 → 进程 exit 0; 任一步失败 → exit 非零并
保留 session_dir 供人工检查。

**phase 内允许 actions**: `report`, `session_breakdown`, `recover`。

**回退策略**: 不回退。CLOSE 失败留现场, 由人工或 robustness 监控处理。

## 6. phase_exit_reasons 词表 (闭合枚举)

phase_history 中 reason 字段必须取值于:

```
prelude_done                 → PRELUDE  → EXPLORE
prelude_baseline_failed      → PRELUDE  → CLOSE
plateau_explore              → EXPLORE  → KERNEL  (kernel_enabled)
plateau_explore              → EXPLORE  → SWEEP   (no_kernel)
explore_phase_budget_exhausted  → EXPLORE → KERNEL/SWEEP (同上)
plateau_kernel               → KERNEL   → SWEEP
kernel_phase_budget_exhausted → KERNEL  → SWEEP
sweep_done                   → SWEEP    → CLOSE
sweep_budget_exhausted       → SWEEP    → CLOSE
robustness_escalated         → any      → CLOSE
target_reached               → any      → CLOSE
no_more_leverage             → any      → CLOSE
time_exhausted               → any      → CLOSE
time_exhausted_during_prelude → PRELUDE → CLOSE
prelude_policy_loop          → PRELUDE  → CLOSE
user_stop_requested          → any      → CLOSE
```

任何其他 reason 字符串都视为非法, PolicyGate 拒绝写入 phase_history。

## 7. 接口/契约

phase 状态机有以下 *观察者*, 调用面均通过 SharedState 而不是
function call:

- **Orchestration prompt 装配**: 每 tick 读 `SharedState.phase`,
  按 §5 各段的"允许 actions"集合渲染 prompt。
- **PolicyGate**: 校验 `propose_action`/`delegate`/`request` 时, 看
  `SharedState.phase`, 不在允许集 → `policy_denied: phase_incompatible`。
- **Knowledge Plane**: T2 hypothesize 调用必须带 `phase` 字段, 写入
  edge attrs, 便于跨 session 按 phase 追溯。
- **Observability (§3.12)**: breakdown 的 `phase_timeline` 段从
  `phase_history` 反序列化。

phase 转移本身不通过 intent 触发, 由 Coordinator 在每 tick 末尾扫描
退出条件。LLM 只能用 `escalate_strategy_change` *建议*跳 phase, 是
否接受由 Coordinator 决定 (默认接受 SEVERITY=high 的建议)。

## 8. 实施步骤

1. **词表先行**: 把 §6 的 `phase_exit_reasons` 写入一份独立的
   contract 文档 (设计稿阶段不写代码, 但词表要锁定)。
2. **退出条件预算化**: 把"60% / 25%" 等 phase 预算转换成具体的可读
   字段 (`max_minutes_explore_pct = 0.60` 等) 并加 CLI flag 让运维可
   覆盖。
3. **进入/退出判定 unit 化**: 概念上每个 phase 配 4 个判定: enter /
   exit_normal / exit_terminal / abort, 各为 pure function of
   SharedState; 实施层就是 4 个无副作用判定。
4. **resume 兼容**: 老 v0.6 session 在 resume 时 SharedState 没有
   `phase`, 需要从 `current_best` / `last_profile_trace` /
   `optimization_stack` 反推 phase, 推不出来时默认 `EXPLORE`。
5. **CLI flag**: 至少新增 `--max-minutes-prelude / -explore / -kernel /
   -sweep / -close` 5 个百分比 flag, 默认值 5 / 60 / 25 / 8 / 2。
6. **prompt 联动**: §3.3 角色重对齐章节负责把 phase 注入到每个角色
   prompt; 本节只锁定字段。

## 9. 边界条件 / 失败模式

| 场景 | 行为 |
|---|---|
| EXPLORE 中 specialist 全部失败 (空 proposal_set) | 计入 `specialist_empty_streak`, 达 K 即触发 `plateau_explore` |
| KERNEL 中 profile 失败 | reason = `kernel_phase_aborted_no_trace`, 直接转 SWEEP |
| SWEEP 中 sweep_grid 取不到 | 用 SKILL.md 默认 grid 兜底, 写 warning |
| CLOSE 中 Cortex commit 失败 | NDJSON 保留 pending; 进程退出 1; 下次 resume 检测到 pending 自动 commit |
| resume 时 `phase==CLOSE` 但未 commit | 检测到即重新走 CLOSE 5 步 |
| 用户跨 phase 手改 state.json | 启动时校验 phase 与 baseline_tput / current_best 兼容性, 不兼容拒绝启动 |

## 10. 验收标准

- [ ] 在一次冷启动 session 内, breakdown.phase_timeline 恰好出现 5
      段, 顺序严格 `PRELUDE → EXPLORE → KERNEL → SWEEP → CLOSE`,
      或在 `--no-kernel` 时 `PRELUDE → EXPLORE → SWEEP → CLOSE`。
- [ ] 任意 phase 内 propose 不允许的 action 一律返回
      `policy_denied: phase_incompatible`, prompt 下一 tick 能看到
      denial 并自我修正。
- [ ] phase_history 中所有 reason 都属于 §6 词表。
- [ ] 用户在 monitoring 中能看到"当前 phase 已用时长 /  phase budget 比例"。
- [ ] resume 一次完整 session 后, phase_history 中的转移不被复制 (幂等)。

## 11. 依赖与影响面

- **上游**: §3.1 (主轴 A — phase 由 Coordinator 强制)。
- **下游**:
  - §3.3 (角色重对齐) 把 phase 注入 prompt。
  - §3.4 (explore 合并) 仅在 EXPLORE 内有效。
  - §3.5 (specialist) 仅在 EXPLORE 内派发。
  - §3.7 (research_lane) 仅在 EXPLORE 占用; 与 benchmark_lane 在
    各 phase 边界释放。
  - §3.8 (state machine) 提供 plateau 判定细节。
  - §3.10 (SharedState) 字段表新增 `phase` / `phase_history` /
    `phase_started_ts` / `*_phase_budget_*`。
  - §3.11 (PolicyGate) 新增 `phase_incompatible` 拒绝规则。
  - §3.12 (observability) `phase_timeline` 段。
  - §3.13 milestone M2 实施本节。

## 12. 哲学回引

本节是**主轴 A** 的核心落地: phase 转移由 Coordinator 强制, LLM 只在
phase 内自由决策, scoreboard 不再决定 action 顺序。同时尊重 **Inv-1**
(phase 字段属于 CORE_STATE_FIELDS, 仅 Coordinator 写) 与 **Inv-2.1
phase 单调性**。
