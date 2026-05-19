# §3.10 SharedState 演进 — 加减字段、迁移、不变量

## 1. 设计目标

把 §3.2–§3.9 提到的 SharedState 字段加减汇总到一处, 给出**统一的迁移
规则** + **事实层不变量保证**。本节不替代具体字段表 (那是实施稿),
但锁定 *字段类别 / 迁移行为 / 写者权限 / 验收口径*。

成功标准:

- 新增字段集 / 删除字段集 / 迁移规则 三件事在一处可查。
- 老 v0.6 session 在 v0.8 启动时, state.json 自动迁移; 行为可观测,
  失败有兜底。
- 事实层 (baseline_tput / current_best / cumulative_gain /
  optimization_stack 等) 在迁移过程中保持位级一致。

## 2. 现状回顾

v0.6 SharedState 大体分四类字段:

1. **会话身份**: session_id / claw_session_id / sandbox_user_id /
   model_name / model_path / model_class / framework / gpu_type /
   start_ts / max_minutes / kernel_enabled
2. **事实层度量**: baseline_tput / baseline_accuracy /
   baseline_failure_streak / current_best / cumulative_gain /
   cumulative_gain_validated_* / optimization_stack /
   gain_per_stack_entry / last_action_failures /
   <action>_attempts / last_profile_trace / last_select_kernels /
   last_kernel_opt / last_sweep / last_validate_stack
3. **决策面 (评分相关)**: action_scores / params_no_promote_streak /
   score_violation / cooldown_until_tick / locked_reason / streak_*
4. **辅助**: tick / pruned_families / crash_count / target_summary /
   policy_denial_history / policy_denial_streak /
   discovered_flags / synergy_attempted / backend_winners_history /
   backends_search / params_search

v0.8 在这 4 类基础上做加减。

## 3. 不变量

继承 §3.1 三主轴 + 三不变量。本节核心不变量:

### Inv-10.1 — 事实层不变

迁移过程中, **类别 2 "事实层度量"** 的所有字段必须位级一致 (相同的
key 同一个 type 同一个 value 序列化)。任何破坏性 schema 改动需要
显式的 schema_version bump + 操作员同意。

### Inv-10.2 — Coordinator 单写者

新增的字段全部归 Coordinator 写, LLM 角色经 intent 申请。
继承 §3.1 Inv-1。

### Inv-10.3 — 迁移幂等

state.json 迁移函数对同一份输入 multi-call 必须返回同一份输出 (无副
作用); 如果迁移已完成, 二次启动跳过迁移。

## 4. 字段加减汇总

### 4.1 新增字段 (按章节归口)

| 字段 | 来源章节 | 类型 / 含义 (概念) | 写者 |
|---|---|---|---|
| `phase` | §3.2 | enum: PRELUDE / EXPLORE / KERNEL / SWEEP / CLOSE | Coordinator |
| `phase_started_ts` | §3.2 | iso 时间戳 | Coordinator |
| `phase_history[]` | §3.2 | list of {from, to, reason, evidence, ts} | Coordinator |
| `phase_budget_pct` | §3.2 / §3.8 | dict {phase → 比例上限} (manifest 写, state 同步) | Coordinator |
| `warm_start_recipe` | §3.6 | T0 拉到的 best_config + what_worked + what_failed (dict) | Coordinator (T0) |
| `warm_start_pitfalls` | §3.6 | T0 拉到的已知坑列表 | Coordinator (T0) |
| `cortex_session_id` | §3.6 | T0 返回的 sid | Coordinator (T0) |
| `cortex_session_summary` | §3.6 | T4 commit 返回的 promoted_edges / negation_edges 摘要 | Coordinator (T4) |
| `pending_kb_edges[]` | §3.6 | T2 创建但尚未 verify 的 edge_id 集合 (用于 crash recovery) | Coordinator |
| `explore_search` | §3.4 | 合并后的统一 ledger (替换 backends_search + params_search) | Coordinator |
| `specialist_rounds[]` | §3.5 | 每轮 specialist 派发的元数据 | Coordinator |
| `specialist_domain_empty_streak` | §3.5 / §3.9 | dict {domain → int} | Coordinator |
| `rejected_kernel_partial_overflow` | §3.9 | set of kernel_id (kernel_opt PARTIAL 上限触发) | Coordinator |
| `research_lane_capacity` | §3.7 | int (manifest mirror, 启动时锁定) | Coordinator (启动时) |
| `stop_reason` (ENUM 校验) | §3.8 | 沿用字段, 加 ENUM 约束 | Coordinator |

(部分字段 v0.6 已存在或类似名存在; 这里列入是因为 v0.8 会改它们的
schema 或写者约束。)

### 4.2 删除字段

| 字段 | 来源章节 | 删除理由 |
|---|---|---|
| `action_scores` | §3.9 | 评分体系移除 |
| `params_no_promote_streak` | §3.9 | 评分派生; 用 `explore_search.winners_history` 替代 |
| `score_violation` | §3.9 | 评分派生 |
| `cooldown_until_tick` | §3.9 | 评分派生 (legacy) |
| `locked_reason` | §3.9 | 评分派生 |
| `streak_*` | §3.9 | 评分派生 (替换为 `specialist_domain_empty_streak` + kernel PARTIAL 集合) |
| `backends_search` | §3.4 | 合并到 `explore_search` |
| `params_search` | §3.4 | 合并到 `explore_search` |
| `last_validate_stack` | §3.4 | validate_stack 内嵌进 explore, 不再独立 |
| `backend_winners_history` | §3.4 | 合并到 `explore_search.winners_history` |

### 4.3 不动字段 (Inv-10.1 保护)

事实层全部不动: baseline_tput / baseline_accuracy /
baseline_failure_streak / current_best / cumulative_gain /
cumulative_gain_validated_* / optimization_stack /
gain_per_stack_entry / last_action_failures /
<action>_attempts (audit) / last_profile_trace /
last_select_kernels / last_kernel_opt / last_sweep。

辅助字段也保留: tick / pruned_families / crash_count /
target_summary / policy_denial_history / policy_denial_streak /
discovered_flags / synergy_attempted。

## 5. 迁移规则

state.json 在 v0.8 启动时调用一次 *migration step*, 行为如下:

### 5.1 schema_version 字段

- v0.6 没有 schema_version, 视作 schema_version = 1。
- v0.8 schema_version = 2; migration 把 1 → 2 升级。
- migration 是单向的; 升 2 之后不允许降回 1 (避免老 reader 误读 v0.8
  字段)。

### 5.2 字段映射

```
v0.6 → v0.8 字段映射 (concept 层):

  action_scores / cooldown_until_tick / streak_* / locked_reason
        → drop (不写到 v0.8)

  params_no_promote_streak
        → derive from explore_search.winners_history.length
          (不持久化, 计算时拿)

  backends_search ∪ params_search
        → explore_search:
              tested      = union(backends_search.tested,
                                  params_search.tested)
              accepted    = union(.accepted, .accepted)
              rejected    = union(.rejected, .rejected)
              winners_history = merged_by_round_id_then_ts
              discovered_flags = backends_search.discovered_flags
                                 ∪ params_search.discovered_flags
              synergy_attempted = same union
              domains_round_summary = []  (老 session 没 specialist 概念)

  backend_winners_history
        → explore_search.winners_history (合流后排序)

  last_validate_stack
        → drop (validate_stack 已并入 explore)

  无 phase 字段
        → 推断:
              baseline_tput == 0  → PRELUDE
              kernel_enabled and last_kernel_opt is set → KERNEL
              optimization_stack 非空 and not in KERNEL/SWEEP → EXPLORE
              否则 → EXPLORE (默认)
              并写 phase_history 一行 {to: <inferred>, reason:
                  "resumed_from_v06_inferred", evidence: {...}}

  无 cortex_session_id / pending_kb_edges
        → 留空 (本 session 不与 v0.6 的过去 KB 关联)

  无 specialist_rounds
        → []

  无 research_lane_capacity
        → 取 CLI flag 当前值 (CLI 不传则取默认 6)
```

### 5.3 迁移失败处理

- migration 在 SQLite WAL 内做事务; 任一步失败回滚。
- 失败导致 v0.8 进程退出 1; 操作员可强制 `--migration-mode=lenient`
  接受部分丢失字段, 但事实层 (类别 2) 字段缺失永远视为 fatal。
- 兜底: 操作员可手动备份 v0.6 state.json, 然后 `--reset-state` 完全
  从头跑 (此时所有事实层从 0 开始, 但 Cortex KB 中的跨 session 知识
  仍可用)。

## 6. 接口/契约

### 6.1 SharedState 写者权限

| 字段类别 | 写者 |
|---|---|
| 会话身份 | Coordinator (一次性 在 PRELUDE 入口) |
| 事实层度量 | Coordinator only (Inv-1) |
| phase / phase_history / stop_reason | Coordinator only |
| explore_search / specialist_rounds | Coordinator only |
| warm_start_* / cortex_* / pending_kb_edges | Coordinator only |
| pruned_families | Robustness 经 PRUNE_BRANCH intent → Coordinator 代写 |
| policy_denial_history / streak | PolicyGate 触发后 Coordinator 写 |

### 6.2 LLM 写权限路径

LLM 唯一的写入路径仍然是 `UPDATE_STATE` intent + PolicyGate 校验。
v0.6 已经把 `CORE_STATE_FIELDS` 列表锁定; v0.8 把以下字段加入
CORE_STATE_FIELDS:

- `phase`
- `phase_started_ts`
- `phase_history`
- `cortex_session_id`
- `stop_reason` (本来已经在)
- `optimization_stack` (本来已经在)
- `current_best` (本来已经在)

允许 LLM 写但仅限自身 metric 字段:

- Kernel 角色: 仅自己 action 的 metric 字段 (沿用 v0.6 §7.6 ※5)
- Robustness: `crash_count` / `current_action`
- Orchestration: 无 (除非 v0.8 的某个新动作有特定 metric)

## 7. 实施步骤

1. **schema_version 字段加入**: 一个 PR 加 schema_version=1 的兼容
   read; 老 session 不写 schema_version 时按 1 处理。
2. **新字段空写**: 一个 PR 把 §4.1 的所有新字段加入 SharedState 默
   认值, 但**不写入 state.json** (持久化时 skip), 仅内存可见。这步
   降低 schema 改动面。
3. **迁移函数**: 一个 PR 实现 §5 的字段映射, 在 SharedState.load
   入口调用; 加 unit 测试覆盖每条 mapping。
4. **新字段持久化**: 一个 PR 把新字段写入 state.json (schema_version
   bump 到 2)。
5. **删字段**: 一个 PR 把 §4.2 的字段彻底从 SharedState 类删除 (此时
   迁移函数仍 drop 它们)。
6. **CLI flag**: `--migration-mode=strict|lenient`, `--reset-state`,
   `--legacy-action-scores=drop|warn`。

每步独立可发布, 任一步问题局限于该 PR 范围。

## 8. 边界条件 / 失败模式

| 场景 | 行为 |
|---|---|
| v0.6 session 中 `action_scores` 字段 ≈ 100KB | drop, 写迁移日志, breakdown.warnings 列出 |
| v0.6 session 没有 `phase` | 按 §5.2 推断, 写 phase_history 一行 |
| v0.6 session 中 `last_validate_stack` 仍指向最近的 validate_stack 任务 | drop; 但保留 `validate_stack_attempts` (audit) 不动 |
| v0.6 session 在 KERNEL 阶段 crash, resume 时 inferred 推不出 KERNEL | 默认 EXPLORE, 写 evidence 标 `inference_uncertain`; 操作员可手动 `--force-phase=KERNEL` 覆盖 |
| migration 跑到一半 SQLite 异常 | 事务回滚, state.json 不变, 进程 exit 1 |
| 用户用 v0.6 reader 读 v0.8 state.json | 由于多了字段, 老 reader 只读自己关心的字段, 不会崩溃 (json 字段宽松); 但事实层语义保持一致, 这是 Inv-10.1 |

## 9. 验收标准

- [ ] 一个 fresh v0.8 session, state.json 包含 §4.1 所有新字段, 不含
      §4.2 任何字段。
- [ ] resume 一个 v0.6 session, 迁移日志显示每条 mapping 处理结果,
      事实层字段位级一致 (md5 / json equal 单测覆盖)。
- [ ] schema_version=2 在 state.json 顶层可见。
- [ ] CORE_STATE_FIELDS 在 PolicyGate 校验时拒绝 LLM 越权写入新字段
      (例如 LLM 想 update_state phase=KERNEL, 拒)。
- [ ] migration 函数对同一份输入两次执行结果一致 (Inv-10.3)。

## 10. 依赖与影响面

- **上游**: §3.2 (phase 字段), §3.4 (explore_search), §3.5
  (specialist_rounds), §3.6 (warm_start / cortex_*), §3.7
  (research_lane_capacity), §3.8 (stop_reason ENUM), §3.9 (评分字段
  删除)。
- **下游**:
  - §3.11 PolicyGate 的 CORE_STATE_FIELDS 扩展。
  - §3.12 breakdown.session 段读 schema_version / phase 字段。
  - §3.13 milestone M2 / M3 / M5 实施。

## 11. 哲学回引

本节是**Inv-1 (事实层单写者)** 的字段化落地; **Inv-10.1 (事实层不变)**
保证迁移不破坏数据; **Inv-10.3 (迁移幂等)** 保证 resume 可重复执行。
**主轴 B** (知识外接 Cortex) 通过新字段 `warm_start_*` / `cortex_*` 体
现; **主轴 C** (sub-agent 双形态) 通过新字段 `specialist_rounds` /
`research_lane_capacity` 体现。
