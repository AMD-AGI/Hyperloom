# P0 — 设计决策（v1 lock-down）

> 本文是 dynamic action v1 已经**锁定**的设计决策的单一事实源。任何
> PX 文档若与本文冲突，以本文为准。
>
> 决策变更需走 dynamic_action.MD §3.11 的设计变更流程；不允许在 PX
> 实施期通过"小改一下就好"的方式悄悄突破。

---

## 1. 决策来源

| 决策 ID | 来源 | 状态 |
|---|---|---|
| D-A / D-B / D-C / D-D | dynamic_action.MD §1.9 在初稿基础上加章节 | 已锁定 |
| Q1 / Q2 / Q3 | 实施路径 review 期间补充 | 已锁定 |

---

## 2. 决策清单

### D-A · Dispatch 通道

**选择**：不引入新 IntentType；以新 `action_name="dynamic_action"` 复用
现有 `DELEGATE` 通路。

**关键含义**：

- dynamic action 的派发在 intent envelope 层与 specialist delegate 同形：
  `{type: DELEGATE, payload: {action_name: "dynamic_action", ...}}`。
- PolicyGate 主链上**只新增一条 delegate 子分支**（针对
  `action_name=="dynamic_action"` 的 payload 早期校验），其余 envelope/
  schema/IntentType 校验完全沿用现有逻辑。
- Coordinator 的 intent dispatch 表无需新建新 case；只需在 delegate
  分支内根据 `action_name` 分流到 dynamic action 的 task kind。

**为何不开新 IntentType**：

新 IntentType 会同时拉出三条独立链路（envelope schema、PolicyGate
主分支、Coordinator dispatch case），三处各自的演化容易脱钩。复用
DELEGATE 让 dynamic action 在"系统视角"上与 specialist 同构，未来 review
任何"派发到 sub-agent 的动作"时只需关注一条分支链。

**对其他阶段的影响**：

- P1：PolicyGate 新增的校验方法挂在 `_validate_delegate` 链上，不挂在
  intent envelope 校验上。
- P5：Coordinator 在 `_handle_delegate` 内根据 `action_name` 分流。
- P7：orchestration prompt 在 enabled action 列表里加一项即可，不改
  intent emit hint 的整体结构。

---

### D-B · Critic 分类

**选择**：复用现有 `patch_landing` 审查类，仅在 `review_constraints`
上挂 `cross_domain=true` 标记并附加少量跨域审查规则。

**关键含义**：

- dynamic action 输出的 patch 与 specialist patch 在 Critic 视角下属
  同一审查类（`patch_landing`），共享既有四 checklist。
- "跨域"特性通过 review_constraints 上的 flag 表达，并触发额外的 review
  rule（详见 P4），但**不**新立独立审查类。
- Critic 的输出形态（APPROVE / REJECT / REVISE）不变。

**为何不立新审查类**：

新审查类（如 `cross_domain_proposal`）会要求 Critic 维护一份独立的
checklist；而 dynamic patch 的"严格性要求"实际上是 patch_landing 四
checklist 的**超集**——只是多了几条跨域专属规则。立新类只会拷贝维护
patch_landing checklist，无新信息量。

**对其他阶段的影响**：

- P3：sub-agent 输出的 proposal 必须携带 `cross_domain_rationale` 字段
  以满足 Critic 的额外检查；这是 P3 prompt 装配的硬约束。
- P4：Critic 端 classifier 根据 `provenance == "dynamic"` 自动挂 flag，
  不依赖 LLM 在 prompt 里"声明"自己是 dynamic。
- P5：Critic verdict 的下游路径（写 SharedState / 触发 integrate_patch）
  与 specialist 完全一致。

---

### D-C · SharedState 聚合视图保护

**选择**：`dynamic_actions` 字段进入 `SharedState` 的受保护字段集，
仅 Coordinator 写入；LLM 不能通过 `UPDATE_STATE` intent 修改其内容。

**关键含义**：

- `dynamic_actions` 与 `optimization_stack` / `current_best` 同性质：
  是 Coordinator 仲裁后的事实记录，不是 LLM 的工作区。
- LLM 试图通过 `UPDATE_STATE` 修改 `dynamic_actions` 应被 PolicyGate
  直接拒绝。
- Coordinator 只在以下 4 个明确节点写入：
  1. dispatch 时（创建 summary）
  2. critic verdict 落地时（更新 `verdict` 字段）
  3. KEEP / REVERT 时（更新 `cumulative_gain` / `last_outcome`）
  4. 重启 abandoned 时（更新 `status`）

**为何写保护**：

如果 LLM 能修改自己派过的 dynamic action 的 `cumulative_gain` 或
`last_outcome`，下一 tick 它读到的就是自己的"自我评价"而非 Coordinator
的事实裁定——orchestration 会陷入自我催眠。写保护把"事实"和"愿望"分
离开。

**对其他阶段的影响**：

- P6：`SharedState` 字段定义需把 `dynamic_actions` 加入 `CORE_STATE_FIELDS`
  保护集合。
- P9：必须有显式回归用例验证 LLM 通过 `UPDATE_STATE` 修改
  `dynamic_actions` 被拒绝。

---

### D-D · Sub-agent 形态

**选择**：直接采用真正的 multi-turn ReAct loop（多次 LLM 调用 + 工具
调度），不走 single-shot prompt 过渡形态。

**关键含义**：

- "single-shot prompt 教 LLM 一次性多步思考"被明确否决。
- 实施期一步到位 multi-turn，后续不需要重写 runner。

**为何不走 single-shot 过渡**：

设计稿 §1.3 把 sub-agent 的探索性作为核心特征。single-shot prompt 会
让 LLM 在单次回复里"想象"工具调用结果，而不是真正调用工具——micro-bench
反诱导（§1.2 红线）的物理基础就消失了。先做 single-shot 再升级
multi-turn 等于"先做错的，再做对的"。

**对其他阶段的影响**：

- P3：runner 必须从 v1 第一版就支持多次 LLM 调用 + 工具循环。

---

### Q1 · Multi-turn 物理形态

**选择**：runner 多次起 `claude` 子进程，每轮把上一轮的 journal + 新
观察作为 context 重新调用；状态由 runner 在外部 journal 文件中维护，
sub-agent 本身是无状态的多次调用。

**关键含义**：

- 每轮 sub-process 是短寿、独立、可观察的——失败/超时 kill 当前
  sub-process 即可，无长寿进程需要管理。
- Journal（`sub_agent_journal.md`）是 multi-turn 状态的唯一持久化载体；
  下一轮的 prompt 由 runner 用 journal + tool result 重新拼装。
- Token 累加得很快（每轮 prompt 都包含全部历史），需要明确的轮次上限
  与单轮 token cap（见 README DEFAULT #9 / #12）。

**为何不选 in-process Python loop（选项 a）**：

In-process loop 需要 runner 直接调 LLM API（绕过 `claude` CLI 的
agentic 行为契约），而 specialist runner 的 sub-process 派单链路是经过
打磨的、与 worktree 隔离 / 资源 lane / journal 写入都已对齐。in-process
方案要新建一条平行调度链路，与"复用 specialist 框架"目标冲突。

**为何不选 claude native multi-turn（选项 c）**：

把 claude CLI 当作内部 ReAct 引擎用，turn 边界由 claude 自己控制——
runner 失去对每轮工具调用的可见性，无法在轮次粒度上 enforce budget /
回收路径白名单 / journal 写入。这是 §1.2 红线的物理基础，不能让出。

**对其他阶段的影响**：

- P3：runner 的循环骨架是"启 sub-process → 解析输出 → 执行 tool →
  写 journal → 决定是否再起一轮"。
- P3：每轮 prompt 由 runner 拼装，包含 seed kit + journal + 上轮
  tool result。
- P3：超时 / 失败的恢复粒度是单轮，不需要进程级重启逻辑。

---

### Q2 · Micro-bench 边界

**选择**：

- 允许跑 ≤60s 的简化 inference 路径作为 micro-bench（少量 prompt、短
  `max_tokens`）；
- 单次 bench 由 runner 用 subprocess timeout 强制终止；
- sub-agent 整体 wall-clock budget ≤ 15 分钟（含所有 LLM call + bench
  + tool call）。

**关键含义**：

- "micro-bench" 不仅限于 kernel-level 单 op timing，但也不放行完整
  inference benchmark。
- 上限由 runner 侧硬编码（不由 LLM 在 prompt 里自由决定）。
- 整体 budget 用 wall-clock 计帐，而不是 turn 计帐——避免 sub-agent 在
  少量 turn 内堆积长 bench。

**为何不选 (a) 仅秒级 kernel-level**：

跨域 patch 的关键往往涉及 KV cache + scheduler + attention 的端到端
交互，纯 kernel-level bench 验证不了组合假设。完全禁止 inference 路径
会让 sub-agent 的"内部假设验证"变得太弱，最终回退到 LLM 凭空推理。

**为何不选 (c) 不限 bench 类型**：

不限类型 = 容许跑完整 inference benchmark = 容易被 sub-agent 当作
"我已经验证了 X% 的提升"——红线"micro-bench 不进 promote 链路"在
prompt 反诱导面前会被攻破。中等边界（≤60s 简化 inference）是物理可控
的最大尺度。

**对其他阶段的影响**：

- P3：工具白名单中 `run_bench` 必须由 runner 注册一组**预先定义的**
  bench script（sub-agent 不能任意指定脚本路径），每个脚本在 runner
  侧标注其 wall-clock 上限。
- P3：runner 用 wall-clock timer 在 sub-agent 整体超 15 分钟时强制
  终止，状态标 `TIMED_OUT`。
- P9：必须有不变量测试验证：bench 输出**绝不**进入 `proposal_set.json`
  的 `expected_gain` 字段（见 P3 §6 输出 schema）；如出现则 runner
  reject。

---

### Q3 · Proposal_set 上限

**选择**：一个 dynamic action 单次运行的 `proposal_set.length` 硬上限
= 1（一个跨域组合 patch）。

**关键含义**：

- dynamic action 的语义就是"探索那一个跨域组合"，多个候选会让 grid
  并发评估若干跨域 patch，token 与算力代价过大。
- grid 上的 dynamic-sourced variant cap 自然 = 1（无须额外 cap）。
- sub-agent prompt 必须明确告知"你只能产 1 个 patch，多个会被截断"。
  runner 侧在解析 `emit_proposal` 时若收到第二条直接 reject 当前
  sub-process 的输出（见 P3 §5）。

**为何不选 (b) 上限 = 3（与 specialist 对齐）**：

specialist 的 3 候选是"同 domain 内的 alternatives"——同代价不同实现。
dynamic 的候选是"跨域组合"，每个候选都涉及 ≥2 个 domain 的改动，3 个
就是 6+ 个 domain 改动同时进入 grid 评估，资源代价极高，且 §1.7 已经
明确"dynamic action 是补充通道，不是默认通道"——不应给它 specialist
3 倍的 grid 占用空间。

**为何不选 (c) 上限 = 2**：

折中没意义——上限 = 2 仍然要在 P5 grid 调度中处理 dynamic-sourced
variant 的并发，复杂度与 = 3 几乎一样。要么 1（最简、与设计意图一
致），要么 ≥3（值得做才做）。

**对其他阶段的影响**：

- P1：`MAX_DYNAMIC_SOURCED_VARIANTS = 1`（直接由 cap=1 推出）。
- P3：sub-agent prompt 明确提示 "1 patch only"；runner 截断逻辑落实。
- P5：grid 派单层不需要为 dynamic 做任何"同源多 variant 的 fairness"
  处理。
- P6：`DynamicActionSummary.cumulative_gain` 是单 patch 维度的标量，
  不需要 list 结构。

---

## 3. 决策变更流程

任何对本文中决策的修改：

1. 由 dynamic action owner 起草变更说明（哪条决策、改成什么、为什么、
   影响哪些 PX 文档）；
2. 评审通过后**先**更新本文 + dynamic_action.MD §1.9 与 §3 相关章节；
3. **再**更新受影响的 PX 文档；
4. **最后**才允许动代码。

不允许在 PX 实施期通过"代码先行、文档后跟"的方式绕过。
