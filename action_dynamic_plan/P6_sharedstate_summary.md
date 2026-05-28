# P6 — SharedState 聚合视图与状态机

> 第六阶段的对外可见产物：`SharedState` 上出现 `dynamic_actions` 字
> 段，作为聚合视图供 orchestration 在下一 tick 看到自己派过的 dynamic
> action 的命运；状态机有 11 个明确状态，每个状态由 P5 的失败模式与
> 成功模式自然推出；该字段写保护，仅 Coordinator 写。
>
> 对应 dynamic_action.MD §3.7。

---

## 1. 目标

让 orchestration 在下一 tick 能"看见"自己派过的 dynamic action 的
结果，但不被原始日志吃光 prompt token。本阶段后：

1. `SharedState.dynamic_actions: dict[dyn_id, DynamicActionSummary]`
   存在，字段集闭合；
2. 状态机覆盖 P5 §6 列出的所有失败 / 成功模式；
3. LLM 通过 `UPDATE_STATE` 修改 `dynamic_actions` 被 PolicyGate 拒绝
   （D-C 决策落实）；
4. 注入到 orchestration prompt 时只放最近 N=5 条 summary，超出的只
   附 dyn_id + verdict + artifact_path 指针。

---

## 2. 触及的架构平面

| 平面 | 改动性质 | 中心思想 |
|---|---|---|
| SharedState dataclass | 加 1 个字段 + 1 个嵌套类型 | 视图层与事实层分离 |
| 受保护字段集 (`CORE_STATE_FIELDS`) | 加 `dynamic_actions` | D-C 决策落实 |
| 状态机 | 新建（11 个状态） | 由 P5 §6 失败模式推导 |
| Coordinator 写入点 | 4 个明确节点 | 见 §4 |
| Prompt 注入 helper | 新建（截断到最近 N） | token 预算可控 |
| 序列化 / 持久化 | 沿用 SharedState 既有 save 机制 | 不动主链 |

---

## 3. DynamicActionSummary 字段集（封闭）

每条 dynamic action 在 SharedState 上对应一个 summary 对象。字段集
**完全枚举**：

| 字段 | 类型 | 含义 |
|---|---|---|
| `dyn_id` | str | 与 artifact 目录中 dyn_id 一致 |
| `status` | enum | 状态机当前状态（见 §4） |
| `dispatched_at` | timestamp | 派发时间（与 spec.json 一致） |
| `round_index` | int | 派发时的 EXPLORE round |
| `scope_domains` | list[str] | 复制自 spec.json，便于 prompt 直接展示 |
| `motivation_gap_short` | str | motivation_gap_text 的截断版（≤ 200 chars），prompt 用 |
| `verdict` | enum or null | Critic verdict（APPROVE/REJECT/REVISE/null） |
| `cumulative_gain` | float or null | 仅 KEPT 状态有值；patch 落地后的累计 gain |
| `last_outcome` | enum | 最近一次 lifecycle 结果（与 status 配合，简化 prompt 显示） |
| `artifact_path` | str | 指向 `agents/orchestration/dynamic_actions/<dyn_id>/` |
| `updated_at` | timestamp | 最近一次 Coordinator 写入时间 |

### 3.1 字段集封闭的中心思想

- **不允许出现的字段类别**：micro-bench 数字、详细 rationale 文本、
  内部 journal 摘录、sub-agent 工具调用计数等。这些都属于"原始日志"，
  事实层 artifact 已有完整记录；视图层只需汇总。
- **一旦出现"自由备注"或"扩展元数据"字段**，dynamic action 的命运信
  号就会通过这个字段无控制地反向影响 orchestration prompt——红线模
  糊。

### 3.2 motivation_gap_short 的取舍

为什么 prompt 显示需要 `motivation_gap_short` 这一冗余字段（spec.json
里已有完整 motivation_gap_text）？

- 完整 motivation_gap_text 可能很长（DEFAULT 上限 1K tokens），多条
  叠加进 prompt 体积爆炸；
- 截断版（≤ 200 chars）让 orchestration 在下一 tick 能"快速回忆这条
  dynamic action 是为什么派的"，而不需要去读 artifact 文件。

---

## 4. 状态机（DEFAULT，11 个状态，待 review）

### 4.1 状态列表

```
                         dispatch
                            |
                            v
                  +---- DISPATCHED ----+
                  |                    |
        sub-agent failed/timed-out     |
                  v                    | sub-agent running
            FAILED / TIMED_OUT         v
                                  SUB_AGENT_RUNNING
                                       |
                          sub-agent emit / empty
                                       v
                       +-- COMPLETED_EMPTY (terminal) --
                       |
                       v
                 SUB_AGENT_DONE
                       |
                       v
                 AWAITING_CRITIC
                       |
            +----------+----------+
            |                     |
       APPROVE                REJECT/REVISE
            v                     v
      INTEGRATING        CRITIC_REJECTED (terminal)
            |
            +----------+----------+
            |                     |
       integrate fail         apply ok, grid run
            v                     |
   INTEGRATE_FAILED              v
   (terminal)              KEEP/REVERT 决策
                                  |
                       +----------+----------+
                       |                     |
                     KEPT                REVERTED
                   (terminal)             (terminal)
```

### 4.2 状态枚举（11 个）

非终态（4 个）：

1. `DISPATCHED` — Coordinator 已 mkdir + 写 spec.json，task 已派单但
   sub-agent 尚未启动；
2. `SUB_AGENT_RUNNING` — runner 已启动 sub-agent；
3. `AWAITING_CRITIC` — sub-agent 完成且产出有效 proposal_set，等待
   critic 审查；
4. `INTEGRATING` — critic APPROVE，integrate_patch 任务派单到 grid。

终态（7 个）：

5. `COMPLETED_EMPTY` — sub-agent 主动声明无可行方案；
6. `TIMED_OUT` — sub-agent budget 耗尽；
7. `FAILED` — sub-agent 运行失败（崩溃 / proposal validation 多次失
   败 / 解析失败等）；
8. `CRITIC_REJECTED` — Critic 返回 REJECT 或 REVISE（v1 等同处理）；
9. `INTEGRATE_FAILED` — patch apply 冲突或 grid run crash；
10. `KEPT` — KEEP 阈值通过，patch 落地到 optimization_stack；
11. `REVERTED` — KEEP 阈值未过 / accuracy gate 未过，patch 已 revert。

外加 1 个特殊终态（在 P8 重启时使用）：

12. `ABANDONED` — Coordinator 重启时未完成的状态被强制标记。

### 4.3 转移规则

- 转移**只能由 Coordinator 触发**，不能由 LLM 触发；
- 终态不可转出；
- DISPATCHED → SUB_AGENT_RUNNING 之间允许卡较短时间（task 在 lane 队
  列中等待）；
- 任何阶段如果 Coordinator 重启 → 当时处于非终态的全部转 ABANDONED
  （详见 P8）。

---

## 5. Coordinator 写入点（4 个明确节点）

只有这 4 个节点允许写 `dynamic_actions` 字段：

### 节点 A — Dispatch 时

- 触发：P1/P2 dispatch 完成；
- 写入：创建新 summary，status = DISPATCHED，初始化所有字段；
- `verdict / cumulative_gain` = null。

### 节点 B — Sub-agent 终态时

- 触发：runner 退出（COMPLETED / COMPLETED_EMPTY / TIMED_OUT / FAILED）；
- 写入：
  - COMPLETED → status = SUB_AGENT_DONE → AWAITING_CRITIC（一次性
    转两步，因为 critic 派单是同步的）；
  - COMPLETED_EMPTY → status = COMPLETED_EMPTY (终态)；
  - TIMED_OUT → status = TIMED_OUT；
  - FAILED → status = FAILED；
- last_outcome / updated_at 同步更新。

### 节点 C — Critic verdict 时

- 触发：Critic 写入 verdict.json；
- 写入：
  - APPROVE → status = INTEGRATING；verdict = APPROVE；
  - REJECT 或 REVISE → status = CRITIC_REJECTED；verdict = 对应值。

### 节点 D — Integrate / grid 终态时

- 触发：integrate_patch / grid run / accuracy gate 完成；
- 写入：
  - apply 成功 + KEEP → status = KEPT；cumulative_gain = grid 测得的
    gain；
  - apply 成功 + REVERT → status = REVERTED；
  - apply 失败 / grid crash → status = INTEGRATE_FAILED。

### 中心思想

- **写入点最小化**——4 个节点而非"哪里需要哪里写"；这条让审计某次
  状态变化的来源时只有 4 个候选位置；
- **写入是事务性的**——每次写入要原子地更新 `status` /
  `last_outcome` / `updated_at` / `verdict` / `cumulative_gain` 五个
  相关字段；不允许"先写 status 再写 cumulative_gain"留下中间状态；
- **Coordinator 之外的代码不得直接写**——critic backend / integrate
  executor / grid runner 都不能直接修改 SharedState；它们通过返回信号
  让 Coordinator 在 §5 的某个节点写入。

---

## 6. 写保护（D-C 决策落实）

### 6.1 加入受保护字段集

`dynamic_actions` 被加入 `SharedState.CORE_STATE_FIELDS`（与
`optimization_stack` / `current_best` / `gaps` 同级）。

### 6.2 PolicyGate 拒绝逻辑

LLM 发出 `UPDATE_STATE` intent 试图修改 `dynamic_actions`：

- PolicyGate 在 `_validate_update_state` 路径上检测到目标字段在
  `CORE_STATE_FIELDS` 内 → 直接拒绝；
- 拒绝原因 code = `core_state_write_violation`（沿用既有错误码）。

### 6.3 不允许的"半写保护"

- 不允许 LLM 修改 `dynamic_actions[<dyn_id>].cumulative_gain`（即使
  它声称"我有更好的 gain 估算"）；
- 不允许 LLM 添加新的 dyn_id（只能由 Coordinator 在 dispatch 时创建）；
- 不允许 LLM 删除 dyn_id（artifact 一经派发即永久保留）。

### 中心思想

写保护是"事实记录 vs 工作区"分离的具象——`dynamic_actions` 是
Coordinator 的事实记录，LLM 不能 negotiate 这些事实。

---

## 7. Prompt 注入策略

### 7.1 注入位置

Coordinator 的 prompt composer 在拼装 orchestration system prompt 时，
把 `dynamic_actions` summary 注入一个固定段落（DEFAULT 段落标题：
"Dynamic Action History"）。

### 7.2 截断策略（DEFAULT，待 review）

- 取最近 N = 5 条（按 `updated_at` 倒序）；
- 每条用紧凑格式（约 ≤ 50 tokens 单条）：

  ```
  - dyn-3-1 [KEPT, gain=+2.3%] scope=[kv_cache,scheduler]
    motivation: "trade-off between cache layout and scheduler ..."
    artifact: agents/orchestration/dynamic_actions/dyn-3-1/
  ```

- 超出 5 条的：以 `... (X more older entries; full list in
  $SESSION_DIR/agents/orchestration/dynamic_actions/)` 占位；
- 全部 0 条 → 不展示该段落。

### 7.3 中心思想

- **总 token 预算 ≤ ~250 tokens**，确保 dynamic_actions summary 不喧
  宾夺主；
- **指针 + 摘要**：完整 motivation 在 spec.json，prompt 仅放截断的
  motivation_short；orchestration 需要完整内容时显式 follow-up（即在
  下一轮自己请求读 artifact）；
- **不显示终态名以外的细节**：不在 prompt 里展示 sub_agent_journal
  的任何摘录——避免 LLM 学习到"上次失败的原因，这次绕开"的元学习信号
  （这与 §1.8 "跨 session 学习暂不处理"的同源约束一致）。

---

## 8. last_outcome 字段的语义

`last_outcome` 是 status 的"语义化别名"，主要用于 prompt 显示时让
orchestration 一眼看懂结果，避免 prompt 里塞状态机枚举的全名。

| status | last_outcome (DEFAULT) |
|---|---|
| DISPATCHED | `running` |
| SUB_AGENT_RUNNING | `running` |
| AWAITING_CRITIC | `awaiting_review` |
| INTEGRATING | `evaluating` |
| COMPLETED_EMPTY | `empty` |
| TIMED_OUT | `timeout` |
| FAILED | `failed` |
| CRITIC_REJECTED | `rejected` |
| INTEGRATE_FAILED | `apply_failed` |
| KEPT | `success` |
| REVERTED | `no_gain` |
| ABANDONED | `abandoned` |

### 中心思想

last_outcome 是 prompt-friendly 的"扁平化"标签；status 是状态机内
部状态。两个字段冗余但用途不同——status 是事实记录、last_outcome 是
显示。LLM 不能修改 status 也不能修改 last_outcome（同 D-C 写保护）。

---

## 9. 依赖与前置条件

P6 必须在 P5 完成后实施（依赖 P5 §6 的失败模式归属表）。

P6 与 P7 / P8 的关系：

- P7 prompt 注入策略依赖 P6 的字段集与 last_outcome；
- P8 重启逻辑依赖 P6 的状态机（明确"非终态"集合）。

---

## 10. 验收信号

| # | 测试场景 | 期望结果 |
|---|---|---|
| 1 | 完整 happy path（dispatch → KEPT） | summary 经过 DISPATCHED → SUB_AGENT_RUNNING → ... → KEPT 全程；每次写入只有 1 个原子事务 |
| 2 | LLM `UPDATE_STATE` 修改 dynamic_actions | PolicyGate 拒绝，code `core_state_write_violation` |
| 3 | LLM `UPDATE_STATE` 修改单个 dyn_id 的 cumulative_gain | 拒绝（即使是修改字典内嵌字段） |
| 4 | LLM 试图通过 `UPDATE_STATE` 添加新 dyn_id | 拒绝 |
| 5 | 6 条 dynamic action 已 dispatch（5 终态 + 1 进行中） | prompt 注入仅最近 5 条，第 6 条以占位符表示 |
| 6 | 0 条 dynamic action | prompt 不展示该段落 |
| 7 | summary 字段中出现未声明字段（异常） | SharedState save 时 fail-fast（schema 校验） |
| 8 | 状态从终态试图转出（异常） | Coordinator 内部断言失败，记 alert，不更新 state |
| 9 | dispatch 与 sub-agent 终态同时落（race condition） | 写入串行化（既有 SharedState save 机制保证），最终 status 一致 |

---

## 11. DEFAULT / 待 review

| # | 条目 | DEFAULT | 备注 |
|---|---|---|---|
| 1 | summary 字段集 | §3 列出 | 待 review，特别是是否需要 `proposal_count` / `runner_turns` 等运维统计字段 |
| 2 | 状态机 11 个状态 | §4 | 待 review |
| 3 | 写入点数量 | 4 个 | §5 |
| 4 | prompt 注入 N | 5 | §7.2，待 review |
| 5 | last_outcome 取值映射 | §8 表 | 待 review |
| 6 | motivation_gap_short 截断长度 | 200 chars | 待 review |
| 7 | summary 是否记录 critic reason_codes | 不记录（指向 artifact） | 减少视图体积 |
| 8 | summary 是否分轮归档（旧 round 不展示） | 不分轮，全部保留 | 待 review |

---

## 12. 与 §1.2 红线的对应关系

| 红线 | 在 P6 的落点 |
|---|---|
| 不能改 SharedState 受保护字段 | §6 写保护逻辑 + PolicyGate 拒绝路径 |
| 视图层与事实层分离 | §3 字段集封闭 + §7.3 指针策略 |
| 不能让 dynamic action 之间互相学习 | §7.3 prompt 不展示 journal 摘录；事实层数据需要事后审计才可见 |
| 不能用聚合视图绕过其他红线 | summary 中绝不包含 micro-bench 数字、metric 自定义、kernel 操作记录等 §1.2 红线相关字段 |

P6 是 §1.2 红线"SharedState 受保护字段"的具体物化——写保护机制把
LLM 的"自我评价"通道彻底封堵，确保 orchestration 看到的 dynamic action
命运是 Coordinator 的事实裁定。
