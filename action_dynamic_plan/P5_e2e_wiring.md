# P5 — 端到端连线（Happy Path）

> 第五阶段的对外可见产物：从 orchestration 派发 → sub-agent → critic
> → integrate_patch → grid → KEEP/REVERT，全链路打通。dynamic action
> 与 specialist patch 在评估通道完全汇合。
>
> 对应 dynamic_action.MD §3.6。
>
> 本阶段以"**不改主链**"为成功标准。如果发现需要在主链上再开一条平
> 行 promote 路径，应回头修 P3，**不**在 P5 打补丁。

---

## 1. 目标

把 P1 / P2 / P3 / P4 已经各自完成的环节**串起来**：dynamic action
能完整地走完一条派发的 lifecycle，从派发到 KEEP/REVERT 落地，与
specialist patch 在 `_grid_runner` / `integrate_patch` /
`_promote_to_shared_state` / `record_intervention` 这些下游通路上完全
等价。

具体可观测的产物：

1. 一条端到端 happy path 跑通：dispatch → runner → COMPLETED → critic
   APPROVE → integrate_patch apply → grid run → 单 variant 评估 →
   KEEP（或 REVERT）→ SharedState 更新；
2. 各种失败模式有明确的终态归属（不悬空、不跨阶段污染）；
3. dynamic 与 specialist 的并发运行不冲突。

---

## 2. 触及的架构平面

| 平面 | 改动性质 | 中心思想 |
|---|---|---|
| Coordinator dispatch 路由 | 在 `_handle_delegate` 内分流到 dynamic task kind | P1 的占位现已对接真 runner |
| Specialist done 镜像（或 dynamic done） | 复用 specialist done 路径 OR 平行 dynamic done 路径 | 决策见 §3 |
| Materialize approved proposal | 沿用 specialist materialize 路径 | 关键：不改主链 |
| ExploreExecutor variant 路由 | 接受 `provenance == "dynamic"` 的 variant | 与 default_grid / specialist:* 同处理 |
| Grid 调度 fairness | 处理 dynamic 与 specialist 在同 round 内的派单顺序 | §5 唯一允许的局部改动 |
| Failure 归属 | 各失败模式映射到 P6 状态机 | 状态不悬空 |
| Ledger / intervention 记录 | dynamic action 的成败结果写入既有 ledger | 与 specialist 区分 source 字段 |

---

## 3. Specialist done 路径的复用 vs 平行

P3 runner 输出的 `proposal_set.json` 与 specialist `specialist_done`
schema 同构。这里有一个设计选择：

### 选项 (i) 复用 `specialist_done` intent 通路

P3 runner 完成后让 sub-agent 发出 `specialist_done` 形态的 intent，
携带 `provenance: "dynamic"`，复用既有的 `_handle_specialist_done` /
`_record_specialist_result` / `_materialize_approved_proposal` 全套路径。

优点：

- 下游路径**真的不动**——critic 入口、integrate_patch 入口、grid 入口
  完全沿用；
- specialist 路径已经经过打磨，bug 面较小；
- "dynamic 与 specialist 在评估通道汇合"在代码层就是字面意义上的同一
  通道。

缺点：

- intent 名字看起来怪（dynamic action 发了 `specialist_done`）；
- 既有 `_handle_specialist_done` 内可能已经用 `domain` 字段做了 specialist
  专属逻辑，dynamic 没有 domain 概念（有 `scope_domains` list）。

### 选项 (ii) 平行 `dynamic_action_done` intent 通路

新加一个 `dynamic_action_done` intent，挂独立的 handler。

优点：

- 概念清晰；
- 处理逻辑独立演化。

缺点：

- 立刻引入并行链路（与 D-A 决策"用 DELEGATE 复用 specialist"的精神冲
  突）；
- 容易导致同样的 bug 在两条 handler 里都要修。

### 决策（DEFAULT，待 review）

**v1 取选项 (i)**。理由：

- D-A 决策已经选择"复用 specialist 框架"；如果在 done 阶段又开平行
  通路，半途而废；
- "intent 名字怪"是命名问题，可以在 P5 实施时把 intent 名换成更中性
  的名字（如 `subagent_done`），这样 specialist 与 dynamic 都用此 intent
  ——但要权衡这个命名变更对既有 specialist 路径的回归影响；
- "既有 handler 内 domain 字段假设"——审视 `_handle_specialist_done`
  的具体逻辑，如果它用 `domain` 字段做了路径分流，需要在 P5 适配
  `provenance` 字段；如果只是写元数据，dynamic 可以按需填一个
  `domain="<dynamic-multi>"` 占位（不参与下游逻辑判断）。

**注**：上述决策待 P5 实施前 review；如果实施时发现选项 (i) 适配代价
比预期高，再切换到 (ii)。

---

## 4. 端到端流程（happy path）

按时序展开 dynamic action 的完整 lifecycle：

```
[orchestration]
    |
    v
delegate(action_name="dynamic_action", payload=...)
    |
    v
[PolicyGate] -- pass --> [Coordinator._handle_delegate]
    |
    v
生成 dyn_id; 创建 artifact 目录; 装配 seed_kit.json
    |
    v
派发 task → research_lane (acquire 1 槽)
    |
    v
[DynamicActionRunner] (P3)
    multi-turn ReAct loop
    emit_proposal → COMPLETED
    |
    v
回收 proposal_set.json + journal.md
    |
    v
release lane; 发出 subagent_done intent
    |
    v
[Coordinator handler] → 触发 critic 派单
    |
    v
[Critic] (P4) 审查 → APPROVE
    |
    v
[_materialize_approved_proposal]
    |
    v
[integrate_patch task]
    apply patch → bench 单 variant → accuracy gate
    |
    v
KEEP / REVERT 决策
    |
    v
[_promote_to_shared_state] (KEEP) or [_revert_patches] (REVERT)
    |
    v
更新 SharedState.dynamic_actions[dyn_id].status / cumulative_gain / last_outcome
    |
    v
record_intervention 写 ledger
```

### 中心思想

- **dispatch / runner / critic / integrate / grid / promote 是同一条
  链路**，只在 dispatch 入口和最终 SharedState summary 这两个端点上
  有 dynamic 专属逻辑；中段路径完全复用 specialist。
- **每一步的失败都有明确归属**——见 §6。
- **artifact 目录是中转站**：每一步把自己的输出写入
  `agents/orchestration/dynamic_actions/<dyn_id>/`，下一步从同一目录
  读输入；这条让端到端调试可以在任意一步停下来检查 artifact 状态。

---

## 5. Grid 调度上的 fairness 处理

P5 唯一允许的"主链局部改动"在 grid 层：dynamic variant 与 specialist
variant 在同一 EXPLORE round 内的**派单顺序**。

### 5.1 现状

specialist sourced variant 在 round 内可能多个（受 `MAX_SPECIALIST_SOURCED_EXPLORE_VARIANTS`
约束）；dynamic sourced variant 上限 = 1（Q3 决策）。

### 5.2 派单顺序问题

如果一个 round 同时有 specialist variant 与 dynamic variant 等待派单，
应当先派哪个？

### 5.3 选项与决策

- **(a) FIFO**：按提交时间先后；
- **(b) dynamic 优先**：dynamic variant 提交时立即派，specialist 等；
- **(c) specialist 优先**：specialist 先全部派完，再派 dynamic；
- **(d) 比例分摊**：lane 容量按比例预留。

**DEFAULT = (a) FIFO**。理由：

- dynamic round-cap = 1，每 round 最多 1 个 dynamic variant，与 specialist
  比例上"少数派"，不会饿死 specialist；
- FIFO 实现最简，不引入策略层判断；
- 比例分摊（d）需要预测未来 round 内还有多少 specialist variant 要派
  ——预测错就过度预留，引入 lane 利用率下降。

### 5.4 不在此范围内的"调度策略"

- **不**做"dynamic 失败多次后降权"——§1.7 已明确不引入显式 cooldown
  / kill switch；
- **不**做"dynamic 优先抢 lane"——避免 dynamic 的偶发派发把 specialist
  关键探索阻塞；
- **不**做"per-domain lane reservation"——lane 是 specialist + dynamic
  共享 research_lane 的物理通道，不细分。

---

## 6. 失败模式归属

下表列出端到端链路上每一类失败的归属与下游处理：

| 失败位置 | 失败原因 | runner 输出 | SharedState 状态 | 下游 |
|---|---|---|---|---|
| Dispatch | PolicyGate 拒绝 | — | 不创建 summary | round-cap 不增；prompt 反馈拒绝原因 |
| Sub-agent | TIMED_OUT | proposal_set 空 | TIMED_OUT 终态 | 不进 critic；释放 lane；artifact 保留 |
| Sub-agent | FAILED | proposal_set 空 | FAILED 终态 | 同上 |
| Sub-agent | COMPLETED_EMPTY | proposal_set 空 | COMPLETED_EMPTY 终态 | 不进 critic；与 specialist empty 路径等价 |
| Critic | REJECT | proposal_set 有 | CRITIC_REJECTED 终态 | 不进 integrate_patch；artifact 保留 |
| Critic | REVISE | proposal_set 有 | CRITIC_REJECTED 终态（v1） | v1 等同 REJECT |
| Integrate | patch apply 冲突 | proposal_set 有 | INTEGRATE_FAILED 终态 | record_intervention 标 source=dynamic, fail |
| Grid run | bench crash | — | INTEGRATE_FAILED 终态 | 同上 |
| Accuracy gate | accuracy < threshold | — | REVERTED 终态 | revert patch；ledger 记 fail |
| Grid run | gain < KEEP threshold | — | REVERTED 终态 | revert patch；ledger 记 fail |
| Grid run | gain ≥ KEEP threshold | — | KEPT 终态 | promote；optimization_stack 更新 |

### 中心思想

- **每个失败模式都有终态名**——P6 状态机的状态值即由此表导出；
- **失败不冒泡到 dispatch 之外的层**——dispatch 之后的一切失败都收
  在 dynamic action 自己的 lifecycle 内，不影响其他 specialist 的派
  发或 baseline 状态；
- **artifact 永远保留**——失败 / 成功都不删除 artifact，事后审计可读。

---

## 7. 资源 lane 共存的并发场景

dynamic 与 specialist 共享 `research_lane`（容量 6）。典型并发模式：

| 场景 | 行为 |
|---|---|
| dynamic 和多个 specialist 同时排队 | 按 §5.3 FIFO 派单 |
| dynamic 已占 1 槽，specialist 排队 | specialist 在余下 5 槽内 FIFO 抢占 |
| research_lane 满，新 dispatch 进来 | 同 specialist：等待 lane 释放 |
| dynamic sub-agent 异常崩溃，未释放 lane | 由 ResourceLockManager 的超时机制兜底（既有逻辑） |

### 关键

dynamic 不应引入新的 lane 类型，也不应给自己保留独占槽位。共享
research_lane 是 D-A 决策"复用 specialist 框架"在物理层的具体落实。

---

## 8. 不改主链的边界

P5 实施期，每出现一次"想动主链"的诱惑，按以下决策树检查：

1. **改动是否在 `_grid_runner` 内？** 是 → 仅允许 §5 fairness 处理；
   否则回头修 P3。
2. **改动是否在 `integrate_patch` 内？** 不允许。dynamic patch 与
   specialist patch 在 integrate_patch 视角下等价；如果 integrate_patch
   对 dynamic 报错，问题在 P3 输出 schema（patch 不规范）或 P4 critic
   未拦住。
3. **改动是否在 `_handle_specialist_done` / `_record_specialist_result`
   内？** 仅允许加 dynamic 的元数据字段读写（如 `provenance` 透传），
   不允许加分支逻辑。
4. **改动是否在 `_promote_to_shared_state` 内？** 仅允许调用 dynamic
   summary 更新 helper（即 P6 的对外接口）；不允许重构 promote 逻辑
   本身。

任何一条违反 → 触发 §3.11 设计变更流程。

---

## 9. 依赖与前置条件

P5 必须在 P1 / P2 / P3 / P4 全部完成后实施。

P5 依赖：

- P1 dispatch 通路；
- P2 artifact 目录与 seed kit；
- P3 runner 输出 `proposal_set.json` 满足 schema；
- P4 critic 能正确分类与审查；
- 既有 `integrate_patch` / `_grid_runner` / `_promote_to_shared_state`
  / `record_intervention` 路径。

P5 完成后，P6 / P7 / P8 可以并行启动（P5 的状态终态是 P6 状态机定义
的输入；P5 的 lifecycle 是 P7 prompt 引用的依据；P5 的 artifact 是 P8
重启扫描的对象）。

---

## 10. 验收信号

| # | 测试场景 | 期望结果 |
|---|---|---|
| 1 | 完整 happy path：dispatch → COMPLETED → APPROVE → integrate apply → grid → KEEP | SharedState dynamic_actions[dyn_id].status = KEPT；optimization_stack 更新；ledger 写记录 |
| 2 | sub-agent COMPLETED_EMPTY | 不触发 critic；status = COMPLETED_EMPTY；与 specialist empty 路径行为一致 |
| 3 | sub-agent TIMED_OUT | 不触发 critic；status = TIMED_OUT；artifact 保留 journal 完整 |
| 4 | critic REJECT | 不触发 integrate_patch；status = CRITIC_REJECTED |
| 5 | integrate_patch apply 失败 | status = INTEGRATE_FAILED；其他 specialist 派发不受影响 |
| 6 | accuracy gate 不过 | status = REVERTED；patch 已 revert；baseline 状态恢复 |
| 7 | grid run gain 不达标 | status = REVERTED |
| 8 | dynamic + 2 specialist 同 round 并发 | 三者按 FIFO 派单；research_lane 利用率正常；互不干扰 |
| 9 | dynamic dispatch 时 lane 已满 | dispatch 等待，PolicyGate 不拒（lane 等待是软约束） |

---

## 11. DEFAULT / 待 review

| # | 条目 | DEFAULT | 备注 |
|---|---|---|---|
| 1 | done intent 通路选择 | 选项 (i) 复用 specialist done | §3，待实施期 review |
| 2 | grid 派单顺序 | FIFO | §5.3 |
| 3 | REVISE 在 v1 是否引入二次派发 | 否 | P4 §5.2 已决策 |
| 4 | record_intervention 中 source 字段 | 加 `dynamic` 取值 | 待 review |
| 5 | dynamic patch 在 ledger 中是否独立分类 | 是（与 specialist 区分） | 便于事后统计 dynamic 命中率 |
| 6 | integrate_patch 是否需要识别 dynamic | 否，dynamic patch 与 specialist patch 在 integrate 层等价 | 不改主链原则 |

---

## 12. 与 §1.2 红线的对应关系

| 红线 | 在 P5 的落点 |
|---|---|
| 不能落 patch 不经 integrate_patch | dynamic patch 必须走 integrate_patch 路径；P5 不提供任何旁路 |
| 不能起独立 server / 跑 Magpie | dynamic variant 在 grid 层与 specialist variant 共用 grid 派单链路（同一 grid runner / Magpie 调用） |
| 不能声明自己的 metric | KEEP/REVERT 由 grid runner + accuracy gate 决定；dynamic action 看不到 metric 计算逻辑 |
| 输出严格度与 specialist 同 | 全程沿用 specialist 下游；任何放宽都会被 §8 决策树拒绝 |

P5 是端到端打通的"组装关节"，所有 §1.2 红线在 P1–P4 的设计中已被
独立守住，P5 的工作是**保持这些守住的状态**，不在组装时退化。
