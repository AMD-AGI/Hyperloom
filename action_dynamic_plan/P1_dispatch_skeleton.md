# P1 — 合法派发（Dispatch Skeleton）

> 第一阶段的对外可见产物：orchestration 能合法发出一条 dynamic action
> 派发请求；PolicyGate 能在所有不应该派的情况下拒绝；派发后挂一个
> stub executor 立即返回空 proposal_set。
>
> 对应 dynamic_action.MD §3.2。

---

## 1. 目标

用最小代价证明 dynamic action 这条**派发通路**与现有 specialist 通路
同构、且 §1.2 红线在派发瞬间已经守住。本阶段**不**真正运行 sub-agent，
不写 seed kit，不接入 critic / integrate_patch / grid。

具体可观测的产物：

1. orchestration 在 EXPLORE 阶段发出
   `DELEGATE{action_name="dynamic_action", payload=...}`，PolicyGate 校
   验通过、Coordinator 接住、stub executor 落一条空记录。
2. orchestration 在错误条件下（错 phase / 错 source / payload 缺字段
   / side_effects 触红线 / round-cap 超限）发出同样意图，PolicyGate
   拒绝并返回结构化原因。

---

## 2. 触及的架构平面

| 平面 | 改动性质 | 中心思想 |
|---|---|---|
| Action 注册 | 新增 1 条声明式条目 | 与 specialist 同构地登记新 action 的能力面 |
| PolicyGate | 新增 1 条 delegate 子分支 + 若干 helper | 把 §1.2 红线全部翻译成派发瞬间的拒绝条件 |
| Coordinator dispatch | 新增 1 条 action_name 分流 | 命中 dynamic action 时调 stub executor |
| Stub executor | 全新 1 个最小 executor | 仅记录派发事件，不做任何探索 |
| 容量常量 | 新增 2 条独立 cap | round-cap 与 grid sourced cap 与 specialist 解耦 |
| IR-4 provenance 白名单 | 新增 1 个字面量 `dynamic` | 单字面量 stamp，不引复合 |

---

## 3. Payload schema 的方向性设计

dynamic action 的 payload 必须强制声明三个**意图字段**和一个**资源
字段**。schema 的设计原则：**字段集封闭、每字段语义有且只有一种解释、
任何"备注/扩展"字段都不允许**。

### 意图字段（决定 §1.2 红线检查）

- **motivation_gap_text**：自然语言描述"为什么 specialist 单 domain
  prompt 不能覆盖此 patch 组合"。审计用，PolicyGate 不解析其语义，但
  作为日后 review 的事实记录强制存在；空字符串 = 拒绝。
- **scope_domains**：声明此 dynamic action 将影响的 domain 集合（如
  `["kv_cache", "scheduler", "attention"]`）。**硬约束** `length ≥ 2`，
  且每个值必须是已注册 specialist domain 集合的成员。这条同时回答
  "跨域"的字面要求与"side effect 落点是哪几条 lane"的问题。
- **side_effects_declared**：声明此 dynamic action 预期改动的 action
  范畴列表（如 `["framework_source"]`）。PolicyGate 根据此字段做红线
  检查（见 §4.2）。

### 资源字段

- **budget_hint**（可选）：orchestration 对 sub-agent 工作量的提示
  （low / medium / high），影响 P3 runner 的 turn cap 分级。可选，缺
  省 = medium。

### 字段集为何封闭

任何"自由备注"字段最终会变成 LLM 的逃生口——orchestration 会通过它把
red-line 信息（如 "我希望 sub-agent 试一下 metric override"）塞进派发，
PolicyGate 难以静态检查。封闭字段集 + 每字段一种解释 = 检查面收敛。

### 不在 payload 中的字段

- **不**允许 orchestration 在 payload 里直接给 seed kit 内容；seed kit
  完全由 Coordinator 装配（见 P2）。这条是 P2 §3 "seed kit 是
  orchestration 的承诺，不是 sub-agent 的自由输入" 的具体落实。
- **不**允许 orchestration 指定 sub-agent 的 budget 数值（只能给
  hint）；具体 budget 由 runner 配置决定（见 P3）。
- **不**允许 orchestration 直接指定 dyn_id；dyn_id 由 Coordinator 生成
  （见 P2 §3）。

---

## 4. PolicyGate 校验链的方向性设计

PolicyGate 在 delegate 校验链上新增一条针对 `action_name=="dynamic_action"`
的子分支。这条子分支负责把 §1.2 红线全部翻译成静态可拒条件。

### 4.1 校验项分组

按触发的红线类别分四组：

**组 A — 阶段与 source 准入**

- phase 必须 == EXPLORE（非 EXPLORE phase 直接拒）；
- intent source 必须是 orchestration（其他 sub-agent 不允许派发 dynamic
  action）；
- action 在 phase 当前 enabled action 列表中存在。

**组 B — payload schema 完整性**

- 三个意图字段（motivation_gap_text / scope_domains /
  side_effects_declared）非空；
- `scope_domains.length ≥ 2`；
- `scope_domains` 每个值在 specialist domain 注册表中存在；
- `budget_hint`（如有）∈ {low, medium, high}。

**组 C — 红线物体边界**

- `side_effects_declared` 不允许包含 kernel_owned action（与 specialist
  同款禁区，复用既有禁区列表）；
- `side_effects_declared` 不允许包含 metric / accuracy_gate / server
  类操作（这些是 §1.2 显式禁止 dynamic action 自定义的能力）；
- `scope_domains` 不允许全为 `kernel`（即不允许 dynamic action 实质
  退化为 kernel-only patch）。

**组 D — 容量与并发**

- 当前 EXPLORE round 内已派发 dynamic action 数 < `MAX_DYNAMIC_PER_ROUND`
  （DEFAULT = 1）；
- research_lane 容量当前可用（这条由 ResourceLockManager 在 dispatch
  阶段最终保证，PolicyGate 只做软检查）。

### 4.2 校验链的中心思想

- **静态先于动态**：四组校验全部是 payload + SharedState 静态可读字段
  的检查，不依赖 sub-agent 已经跑出什么。这让"违规意图"在 dispatch 瞬
  间就被拒，不会污染 worktree / artifact / 资源 lane。
- **拒绝原因结构化**：每条失败返回独立 reason code（如
  `dynamic_phase_violation` / `dynamic_scope_too_narrow` /
  `dynamic_side_effects_red_line` 等），prompt 反馈给 orchestration 时
  能精确指出哪一条拒了，让 LLM 修正而不是盲改。
- **拒绝不消耗 round-cap**：即被拒绝的派发不计入 `MAX_DYNAMIC_PER_ROUND`，
  避免 LLM 用一次失败拒绝把当前 round 的 dynamic 名额消耗掉。

### 4.3 容量常量的独立计帐

不复用 `MAX_SPECIALIST_SOURCED_EXPLORE_VARIANTS` 等 specialist 计帐变
量，新立两条：

- `MAX_DYNAMIC_PER_ROUND`（DEFAULT = 1）：每 EXPLORE round 允许 dispatch
  的 dynamic action 数。
- `MAX_DYNAMIC_SOURCED_VARIANTS`（DEFAULT = 1）：grid 上 provenance 为
  `dynamic` 的 variant 在同 round 内的最大数量。由 Q3 决策（proposal_set
  cap = 1）自然导出。

独立计帐的理由：dynamic 与 specialist 共享物理 lane（`research_lane`
6 槽），但**不共享配额池**——避免 dynamic 一次派发把 specialist 那一
轮的探索机会挤掉，也避免 specialist 大量派发把 dynamic 间接饿死。

### 4.4 IR-4 provenance 白名单的最小扩展

EXPLORE phase 允许的 variant provenance 集合从
`{specialist:<domain>, default_grid}` 扩为
`{specialist:<domain>, default_grid, dynamic}`。

要点：

- `dynamic` 是**单字面量**，不带 sub-tag（不存在 `dynamic:kv_cache+scheduler`
  这种复合形式）；
- IR-4 校验逻辑（即 PolicyGate 检查 explore variant 的 provenance）
  对此字面量原样接受；
- 如果实际跑出来的 variant 用了非白名单的 provenance（如 LLM 试图伪
  造 `specialist:dynamic`），仍被现有 IR-4 校验拒绝。

---

## 5. Action 注册的方向性设计

参考 specialist 的注册方式，dynamic action 也以**声明式元数据**登记，
最小字段集：

| 字段 | 含义 | DEFAULT |
|---|---|---|
| `name` | 动作名 | `"dynamic_action"` |
| `category` | 范畴标签 | `"explore"`（与 specialist 同 category） |
| `enabled_in_phases` | 合法 phase | `["EXPLORE"]` |
| `requires_lanes` | 资源 lane | `["research_lane"]` |
| `llm_proposable` | 是否允许 LLM propose | `true`（前提是 source=orchestration） |
| `internal_only` | 是否仅限内部分析 | `false` |

### 注册的中心思想

- **声明先于实现**：先有这条声明，PolicyGate 与 Coordinator 才能识别
  此 action 的存在；stub executor 是后续接入实际逻辑的占位。
- **元数据驱动校验**：组 A（phase / lane）的校验大部分可以从这条声明
  推出来，不需要在 PolicyGate 里硬编码 phase 列表。
- **不引入 trust tier 字段**：dynamic_action.MD §1.8 已明确不采用 trust
  tier；注册条目里**不**给"已通过试用期"等字段，避免概念蠕变。

---

## 6. Coordinator dispatch 分流的方向性设计

`_handle_delegate` 在收到 `action_name=="dynamic_action"` 的 delegate
intent 后：

1. 调 PolicyGate 完成 §4 的所有校验；
2. 校验通过 → 生成 dyn_id（DEFAULT 格式 `dyn-<round>-<seq>`，详见 P2
   §3）；
3. 创建 artifact 目录骨架（详见 P2，本阶段可只 mkdir，不写 seed kit）；
4. 派发到 stub executor（本阶段固定一个 task kind = `"dynamic_action"`）；
5. round-cap 计数 +1。

### 中心思想

- **dispatch 与 stub executor 解耦**：本阶段把"派发链路"和"探索逻辑"
  完全分开。后续 P2/P3 替换的只是 stub executor 内部，dispatch 链路一
  动不动。
- **失败的 dispatch 不污染 round 状态**：PolicyGate 拒绝 → round-cap
  不递增、artifact 不创建、ledger 不写记录。

---

## 7. Stub executor 的方向性设计

最小 stub 行为：

- 接收 task → 立即返回空 `proposal_set`（即 `proposal_set.length == 0`）；
- 在 `dispatch_history.jsonl` 写一条 `{ts, dyn_id, outcome:
  "stub_empty"}` 记录；
- 释放 research_lane。

### 中心思想

- **空 proposal_set 走 specialist 既有的 empty 路径**——validate 通过
  → 写入 SharedState empty_streak / specialist 类似计数（dynamic 单独
  计数，详见 P6 §3）→ 不触发 critic / grid。
- **stub 必须能与下游链路通信**：本阶段的 stub 必须输出与 specialist
  同构的"空 proposal_set"信号，下一阶段（P2/P3）替换为真 runner 时不
  改下游接收逻辑。

---

## 8. 依赖与前置条件

P1 不依赖任何前置阶段（除了 P0 决策 lock-down）。

P1 的产物会被 P2 / P3 / P5 直接复用：

- P2 在 dispatch 钩子里追加 seed kit 装配；
- P3 替换 stub executor 为真 sub-agent runner；
- P5 把 stub 路径换成完整端到端路径。

---

## 9. 验收信号

P1 完成的判据，全部可手动构造请求验证：

| # | 测试场景 | 期望结果 |
|---|---|---|
| 1 | 在 EXPLORE phase、orchestration source、payload 完整、scope_domains=2 个合法 domain → dispatch | PolicyGate 通过；artifact 目录创建；stub executor 写空记录；round-cap +1 |
| 2 | 在 PRELUDE phase 派发 | 拒绝，reason `dynamic_phase_violation` |
| 3 | 由非 orchestration source 派发 | 拒绝，reason `dynamic_source_violation` |
| 4 | scope_domains.length == 1 | 拒绝，reason `dynamic_scope_too_narrow` |
| 5 | scope_domains 含未注册 domain | 拒绝，reason `dynamic_scope_unknown_domain` |
| 6 | side_effects_declared 含 kernel_owned | 拒绝，reason `dynamic_side_effects_red_line` |
| 7 | 当前 round 已派发 1 条 dynamic（cap 满） | 拒绝，reason `dynamic_round_cap_exhausted` |
| 8 | scope_domains 全为 kernel | 拒绝，reason `dynamic_kernel_only_disallowed` |
| 9 | LLM 试图通过 `UPDATE_STATE` 修改 dynamic_actions | 拒绝（CORE_STATE_FIELDS 保护，由 P6 落实，本阶段提前布点测试位） |

P9 测试矩阵中"单元层 PolicyGate"小节会把以上场景结构化为回归 case。

---

## 10. DEFAULT / 待 review

| # | 条目 | DEFAULT | 备注 |
|---|---|---|---|
| 1 | `MAX_DYNAMIC_PER_ROUND` | 1 | 设计稿 §1.4 已规定 |
| 2 | `MAX_DYNAMIC_SOURCED_VARIANTS` | 1 | 由 Q3 推出 |
| 3 | scope_domains 最小数量 | 2 | 待 review |
| 4 | side_effects_declared 红线集合 | 复用 specialist kernel_owned 列表 + 加 metric/accuracy/server 三类 | 待 review |
| 5 | budget_hint 取值 | low/medium/high | 待 review |
| 6 | round-cap 计数粒度 | 仅成功 dispatch 计数 | 待 review |
| 7 | reason code 命名前缀 | `dynamic_*` | 沿用 specialist 风格 |

---

## 11. 与 §1.2 红线的对应关系

| 红线 | 在 P1 的落点 |
|---|---|
| 不能改 SharedState 受保护字段 | `UPDATE_STATE` 校验路径（P6 落实，P1 仅占位测试） |
| 不能走 internal analysis lane | 由 `_validate_action_not_llm_proposable` 阻断（既有逻辑），P1 不重复 |
| 不能动 kernel-owned actions | 组 C，`side_effects_declared` 红线检查 |
| 不能起独立 server / 跑 Magpie | 组 C，`side_effects_declared` 红线检查 |
| 不能声明自己的 metric | 组 C，`side_effects_declared` 红线检查 |
| 不能落 patch 不经 integrate_patch | 由 P3 runner 输出 schema + P5 端到端连线保证；P1 不直接负责 |

P1 的作用是把红线中**派发瞬间可静态拒**的部分全部前置到 dispatch
gate，让违规意图根本没有机会进入 worktree。
