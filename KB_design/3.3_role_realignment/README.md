# §3.3 角色重对齐 — 4 角色在 v0.8 的边界

## 1. 设计目标

在保持 v0.6 协议层 (MessageBus / PolicyGate / TaskRegistry /
ResourceLockManager) 不变的前提下, 调整 4 个角色的**职责边界**和
**信息可见性**, 以匹配 §3.2 的固化管线 + §3.5 的 specialist 框架。

不增加新的反应器角色; specialist 是 sub-agent (一次性、无 inbox 持
久状态), 不计入这 4 个反应器之列。

## 2. 现状回顾

v0.6 的 4 个角色职责见 `orchestrator/agent_role.py`:

| 角色 | 现有职责简表 |
|---|---|
| Orchestration | 在 12 action 内 propose / delegate / request, 唯一可发 REQUEST 给 Kernel |
| Kernel | 5 个 KERNEL_OWNED_ACTIONS 的 responder, 不主动发任何 propose/delegate/request |
| Critic | 评审每个 propose_action, KB 读 + 评审专属写 |
| Robustness | 健康监控 + RCA + scheduling police, 唯一可发 kill_task / prune_branch / force_dispatch / escalate_strategy_change |

v0.8 的核心调整在 Orchestration 与 Critic, Kernel / Robustness 仅做
"phase-aware" 的边界扩展, 不动核心责任。

## 3. 不变量

继承 §3.1 三主轴 + 三不变量。本节额外引入:

### Inv-3.1 — 角色不增不减

v0.8 持久反应器仍然是 4 个 (orchestration / kernel / critic /
robustness), 与 v0.6 完全一致。任何"再加一个 N+1 角色"的提议不属于
v0.8 范围, 走 v0.9 RFC 流程。

### Inv-3.2 — 信息单向流

specialist sub-agent 的输出经 Coordinator 中转 → 进入 Orchestration
inbox。specialist 不直接给 Kernel / Critic / Robustness 发消息。
反过来 Kernel / Critic / Robustness 也不直接给 specialist 发消息。
specialist 与世界的交互窗口 = `delegate→specialist_done` 一对。

理由: 否则 PolicyGate 的源源 (source-source) 校验矩阵会爆炸。

## 4. 各角色 v0.8 重对齐

### 4.1 Orchestration (Claude, tool-using)

**核心变化**: 决策面从"按 scoreboard 选 action"退化为"在当前 phase
允许集合内自由组合 specialist + explore + request"。

**新增职责**:

- 持有 phase 内 *本轮调度计划* 的状态 (in-flight specialist 数 /
  待 Critic Review 的 propose 数 / explore 已 stack 的 KEEP 数)。
- 在 EXPLORE phase 决定**派多少个 specialist + 各自的 domain + 各自
  关心的 gap canonical_id**。决策依据: 当前 gaps + KB warm_start
  pitfalls + 上轮 specialist 是否空 proposal_set。
- 在 KERNEL phase 沿用 v0.6 行为 (REQUEST kernel `select_kernels` /
  `run_optimization` / `integrate` / `apply_patch`)。
- 在 SWEEP phase 仅作 `delegate(sweep)` 触发, sweep grid 由 Coordinator
  从 Cortex recipe / SKILL 默认中选。

**移除职责**:

- 不再读 scoreboard / `MARATHON_PRIORS` / cooldown 字段 (它们已被砍除
  — §3.9)。
- 不再 propose `validate_stack` (语义并入 EXPLORE 串行 integrate)。
- 不再 propose `backends` / `params` (合并为 `explore` — §3.4)。

**Prompt 注入字段 (新)**:

- `phase`: `PRELUDE | EXPLORE | KERNEL | SWEEP | CLOSE`
- `phase_allowed_actions`: 当前 phase 允许集合 (来自 §3.2)
- `phase_budget_remaining_pct`: 0–1, 当前 phase 预算剩余比例
- `warm_start_recipe_summary`: T0 拉到的 best_config + what_worked +
  what_failed 的 markdown 摘要
- `gaps`: 当前未解决 gap 列表 (canonical_id + symptom + layer)
- `kb_subgraph_per_gap`: T1 traverse 子图, gap 索引
- `pr_feed`: pr-monitor 预热摘要 (EXPLORE phase)
- `specialist_round_summary`: 上轮 specialist 派发结果 (空 / 提了几个
  / 哪些被 KEEP / 哪些被 REVERT)

**Prompt 移除字段**:

- `action_scores` 表 (整个 top-12 区块)
- `cooldown_until_tick` 提示
- `MARATHON_PRIORS` 描述

### 4.2 Kernel (Claude, responder-only)

**核心变化**: 仅在 KERNEL phase 接 REQUEST, 其它 phase 一律收到的是
heartbeat。

**新增职责**:

- 接收 phase 信息后, 在 PRELUDE / EXPLORE / SWEEP / CLOSE 期间只发
  `send_message{topic="heartbeat"}`, 不期待任何 REQUEST 进来。
- 在 KERNEL phase 入口的固定 profile 跑完后, 接 `select_kernels`
  REQUEST 时, 必须使用 `last_profile_trace` (不允许构造 trace 路径)。

**移除职责**: 无 (Kernel 在 v0.6 已经只接 REQUEST, v0.8 仅时间窗收紧)。

**Prompt 注入字段 (新)**:

- `phase`: 同上
- `phase_window_active`: 布尔, "我现在该不该接活" 的简化指示

### 4.3 Critic (Codex / 外置 critic-agent skill)

**核心变化**: 评审面**扩大** — 除原有 `propose_action` 评审外, 新增
对 specialist 提出的 *探索假设* (即 specialist_done.proposal_set 中
每个 variant) 进行**预评审**。

**新增职责**:

- *预评审 specialist 假设*: Coordinator 在收到 specialist_done 后, 把
  proposal_set 整体打包成一条 `propose_action='explore'` 提交给
  Critic; Critic 评审依据已有 KB priors + 本次 session 内已 KEEP/REVERT
  的相邻 variant + judge_bundle.review_constraints。verdict 直接对整
  组 variant 生效 (允许部分 redirect)。
- *KB 写代发*: 维持 v0.6 现状, Critic 通过 `commit-review` 让
  Coordinator 把 verdict 落到 Cortex `verify`。

**移除职责**: 无。

**Prompt 注入字段 (新)**:

- `phase`: 同上
- `phase_specific_rules`: phase 相关的 review_constraints (例如
  EXPLORE 阶段 reject 任何修改 kernel 源码的 variant)。

**容量 / 节流**: specialist 并发提议会让 Critic 评审请求量数倍上升,
v0.8 必须支持**批量评审** (一次评一组 K 个 variant), 否则 Critic 会
成为 EXPLORE 瓶颈 — 见 §3.5 / §3.14 风险条目。

### 4.4 Robustness (Claude / 外置 robustness-agent skill)

**核心变化**: 监控面**扩大** — 新增 specialist sub-agent 探活, 与已
有的 stall / lease / crash 检测并列。

**新增职责**:

- *specialist stale 检测*: 扫 `tasks` 表中 `kind='specialist'` 且
  `state='running'` 超过阈值 (默认 max_specialist_turns × 单轮上限,
  通常 ≤ 10 min) 的任务, 触发 `kill_task`。
- *phase-budget 警示*: 当任一 phase 已用预算 > 90% 但 plateau 信号
  尚未触发, 发 `alert{severity='medium'}` 提醒 Orchestration 该考虑
  收尾。
- *Cortex 不可达探测*: NDJSON pending 文件持续累积超过阈值 (例如 >
  500 行 / 30 min) 时, 发 high alert; 由 Coordinator 决定是否中止
  session。

**移除职责**: 无。

**Prompt 注入字段 (新)**:

- `phase`: 同上
- `phase_budget_telemetry`: 每段 phase 的预算用量
- `specialist_running_count`: 当前在跑的 specialist 数 (用于 stale
  扫描)

## 5. 接口/契约

### 5.1 角色 × 意图 兼容矩阵 (v0.8)

仅列改动行, 未列出的行与 v0.6 一致 (`agent_role.py` 中的
`_*_INTENTS` frozenset)。

| 角色 | 新增允许 intent | 新增禁止 intent |
|---|---|---|
| Orchestration | (无 — `delegate{action='specialist'}` 复用现有 DELEGATE) | propose `backends` / `params` / `validate_stack` (PolicyGate 拒) |
| Kernel | 无 | 无 |
| Critic | 无 (预评审 specialist 提议复用 REVIEW_VERDICT) | 无 |
| Robustness | 无 (specialist stale 复用 KILL_TASK) | 无 |

→ **结论**: v0.8 不引入新 IntentType, 只在 PolicyGate 加 phase / source
约束 (见 §3.11)。

### 5.2 角色之间的消息流 (v0.8)

```
                  Orchestration
                  ▲   │   │   │
        review    │   │   │   │
        verdict   │   │   │   │ delegate
                  │   │   │   ▼
   Critic ◄───────┘   │   │  specialist (sub-agent, ephemeral)
                      │   │   │
                  request │   │ specialist_done
                      │   │   │
                      ▼   │   ▼
                   Kernel  │  Coordinator (mediator, not a role)
                           │
                  alert    │   recover
                      ▼    │      ▲
                  Robustness ─────┘
```

**变化**:

- specialist 是 Orchestration 派出的临时 sub-agent, 在 §3.5 中详述, 这
  里仅作信息流位置标注。
- Critic / Kernel / Robustness 之间无直接消息 (与 v0.6 一致)。

### 5.3 phase 信息的注入路径

| 角色 | phase 来源 | 频率 |
|---|---|---|
| Orchestration | Coordinator 每 tick prompt 装配 | 每 tick |
| Kernel | Coordinator 每 tick prompt 装配 | 每 tick |
| Critic | judge_bundle 中带 `phase` | 每个 review |
| Robustness | Coordinator 每 tick prompt 装配 | 每 tick |
| specialist | prompt 装配时一次性带入 | 一次 (生命周期内 phase 不会变) |

## 6. 实施步骤

1. **角色 prompt 文件升级**: `orchestrator/system_prompts/{orchestration,
   kernel, critic, robustness}.md` 各加一段 "phase awareness" 段落,
   说明该角色对 phase 的反应行为。**只改 markdown, 不改 Python**。
2. **prompt_builder 升级**: 把 `phase` / `phase_allowed_actions` /
   `phase_budget_*` / `warm_start_*` / `specialist_round_summary` 加
   入 prompt 装配 (复用现有 `build_orchestration_prompt` /
   `build_critic_prompt`)。
3. **Critic 批量评审契约**: 设计 review packet 中 "组" 概念 (一个
   `propose_action='explore'` 的 payload 含 K 个 variant, Critic 对
   组返回每个 variant 的 verdict 字典)。这要求 critic-agent skill
   一侧也升级支持 — 走 critic-agent 自己的 RFC, 不阻塞本节。
4. **Robustness specialist stale 探测**: 把"扫 tasks 表"列入
   robustness 的 tick 步骤; 阈值参数化 (CLI flag)。
5. **policy 矩阵更新**: §3.11 详述。
6. **resume 兼容**: 老 session 的 prompt snapshot 仍可读, 新 prompt
   字段缺失时使用合理默认 (phase 推断默认 EXPLORE)。

## 7. 边界条件 / 失败模式

| 场景 | 行为 |
|---|---|
| Orchestration 在 PRELUDE 阶段 propose `explore` | PolicyGate 拒, prompt 下一 tick 看到 denial 自纠 |
| Critic 评审超时 / 不可达 | 现有 `--critic-mock` / `--critic-codex-bare` 路径不变; 此外 v0.8 增加 "Critic 全 needs_review 模式" 的 phase budget 减半保护 |
| Robustness 的 stale 阈值打到刚启动的 specialist | 阈值 = max_specialist_turns × 单轮 timeout × 1.5, 留 50% 余量, 几乎不会误杀 |
| Kernel 在 EXPLORE 收到 REQUEST | 不会发生 (PolicyGate 拒在 Orchestration 一侧); 若发生, Kernel 应回 `response{status='failed', reason='phase_incompatible'}` |
| specialist 跨 phase 仍在跑 (Coordinator 已切到下一 phase) | 上一 phase 退出时, Coordinator 主动 kill_task 所有 in-flight specialist; 写 phase_history evidence |

## 8. 验收标准

- [ ] 4 份 system_prompt 文件 + 1 份 specialist prompt 模板 在新 session
      启动时被正确装配, snapshot 落到 `agents/<role>/system_prompt.snapshot.md`。
- [ ] 任意角色看到 `phase` 字段;无 phase 字段一律视为 PRELUDE。
- [ ] Critic 在收到一组 K 个 variant 时, 返回的 verdict map 必须 K 个
      key 全部存在, 缺失视为 needs_review。
- [ ] Robustness tick 中可观察到一条 "specialist stale scan: N alive,
      M killed" 日志线 (或 metric)。
- [ ] resume v0.6 session 时, prompt 装配不报错, phase 推断逻辑覆盖
      到。

## 9. 依赖与影响面

- **上游**: §3.1 (三主轴), §3.2 (phase 五段)。
- **下游**:
  - §3.4 specialist→explore 流水中 Orchestration 行为依据本节。
  - §3.5 specialist 框架的 prompt 契约 + stale 检测落到本节角色。
  - §3.11 PolicyGate 5 条新规则的"角色 × phase"关联依据。
  - §3.12 breakdown 段落 `agents` / `critic_robustness` 中的角色字段。

## 10. 哲学回引

本节是**主轴 A** 的角色侧落地 (Orchestration 不再读 scoreboard),
**主轴 C** 的两侧 (Orchestration 派发 deterministic + specialist 双形
态 sub-agent) 与 **Inv-3.2 信息单向流** 的核心约束源。
