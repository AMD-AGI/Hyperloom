# Dynamic Action — 设计稿

> 在 EXPLORE 阶段开一条 "cross-domain dynamic sub-agent" 通道，由
> orchestration 在运行时按需派发。本设计吸收
> `feature/zhenggong/action-formulation` 的设计精神，但 **放弃** 其
> Foundry / Smoke / Critic 整套合成-审核-入池闭环；改为复用现有
> specialist sub-agent 运行框架 + `integrate_patch` 落地通道，仅在
> EXPLORE 阶段补 specialist 单 domain prompt 无法覆盖的跨域 patch
> 探索能力。
>
> 本文 v1 阶段只覆盖**设计意图与边界**；详细的接口字段、状态机、文件
> 布局放后续章节。

---

## 1. 设计动机与核心约束（讨论结论梳理）

### 1.1 解决的问题

唯一的 motivation：**specialist 单 domain prompt 提不出跨域深度 patch**。

具体场景：orchestration 在分析 profile / roofline / 已采纳 patch 历史
之后，认为存在一组跨多个 domain 的 patch 组合（例如同时改 KV cache
布局 + scheduler 调度策略 + attention kernel 的调用方式），该组合不可
能由任何单个 specialist sub-agent 在自己的 domain prompt 边界内提
出来。

显式**不**包含的动机：

- 简单的 backend / env / serving-param 组合 —— 已被 `explore.params.grid`
  覆盖，prompt 端补一个 "把已有 patch A + B 拼成同一 variant" 的入口
  即可，不需要新通路；
- 让 LLM "现写代码即时执行" —— 这是 `action-formulation` 设计稿明文
  反对的「逃生口」模式；
- 替代 specialist 体系 —— specialist 仍是 EXPLORE 的默认通路。

### 1.2 「比 specialist 更大的职权」的精确定义

「职权大」翻译成 4 条具体的**输入/输出能力差异**。注意全部体现在
*输入侧*与*patch 范围*，**不**体现在能影响哪些 SharedState 字段或
能否绕过哪些 gate。

| 能力维度 | specialist | dynamic action |
|---|---|---|
| 输入语料 | 仅 own-domain KB + profile slice | 跨 domain 的 KB + 全 profile + 全 roofline + 已尝试 patches 摘要 |
| Prompt 装配 | 静态 domain prompt | orchestration 在 runtime 按需拼装 |
| 工作模式 | single-shot proposal_set | ReAct 多轮探索 + 工具调用 |
| Patch 范围 | 单 domain 内 | 跨 domain 组合 patch |
| 可发起的诊断 | 仅读 own-domain artifact | 可读全 session artifact + 跑轻量 micro-bench（仅自用） |

**红线**——下列「职权」dynamic action **不**获得：

- 不能改 SharedState 受保护字段（`optimization_stack` / `current_best`
  / `gaps[]`），只能通过 Coordinator 标准 promote 路径影响；
- 不能走 Coordinator-internal analysis lane（roofline / profile 仍由
  `_enqueue_internal_analysis_task` 自动派；PolicyGate
  `analysis_action_not_llm_proposable` 仍生效）；
- 不能动 kernel-owned actions（CLAUDE.md IR-6，原设计 §11 同款禁区）；
- 不能起独立 server / 跑 Magpie —— success metric 仍由 `_grid_runner`
  + Magpie + `_accuracy_gate` 决定，dynamic action **不允许声明自己的
  metric**；
- 不能落 patch 不经 `integrate_patch` —— framework source 的写入仍然
  只走唯一通道。

### 1.3 Sub-agent 形态：种子 kit + 探索式

不采取「知识全交给模型 → 模型一次性输出 patch」的 single-shot 路线，
也不采取「完全自主探索」的开放路线，取折中：

**种子阶段**（orchestration 派发时附 "kit"）

- motivation gap 描述
- 当前 roofline + 关键 profile 摘要
- 已采纳 patches 列表 + 已 revert patches 摘要
- KB pitfalls 命中（相关 domain × motivation 关键词）
- 相关源码路径提示（受 `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS`
  约束的目录指针）

**探索阶段**（dynamic sub-agent 在隔离 worktree 内）

工具白名单：

- 读源码（受 framework_source_roots 约束）；
- 读 session artifact（自身可见路径内）；
- 跑 micro-bench（**仅作内部假设验证**，产出**不进** SharedState、
  **不进** `optimization_stack`、**不进** KEEP 判定）；
- 多轮调用 LLM（ReAct 风格）。

**产出**

proposal_set（与 specialist 同 schema）→ critic gate → `integrate_patch`。

依据：跨域 patch 的难点恰好是「哪两个 domain 该联动」—— orchestration
若已知答案，已可用 `explore.params.grid` 显式拼 combo；正是因为不知道
才把这一步的探索负担下放给 sub-agent。

### 1.4 落地通路（与现有架构的接口）

| 接口点 | 处理方式 |
|---|---|
| 派发 intent | 新 `IntentType.PROPOSE_DYNAMIC_ACTION`，仅 orchestration 可发 |
| PolicyGate | 保留 PolicyGate（不删），新增早期 gate 校验 payload schema、side_effects 白名单、kernel_owned 禁区 |
| Sub-agent runner | 复用 specialist sub-agent runner（隔离 worktree、critic gate 链路），仅 prompt 装配逻辑不同（multi-step） |
| Patch 落地 | 经 `integrate_patch`，与 specialist patch 共用 `_baseline_params_fingerprint` 重放、patch_apply、git checkout 回滚 |
| Critic 审查 | 新 `review_kind=cross_domain_proposal`（或 `action_proposal` + dynamic 标记），prompt 强调"跨域改动越界风险高于 single-domain" |
| Grid 派单 | dynamic action 提交的 variant 进同一 `_grid_runner`，KEEP / REVERT / accuracy gate 路径不变 |
| 资源边界 | 受 `ResourceLockManager` 约束，与 specialist 共享 `MAX_RESEARCH_LANE_CAPACITY=6`；每 EXPLORE round 至多 1 条 dynamic action（`MAX_DYNAMIC_PER_ROUND=1`） |
| IR-4 provenance | EXPLORE variant provenance 集合扩为 `{specialist:<domain>, default_grid, dynamic}`；`dynamic` 计入 sourced variant 上限 |

### 1.5 状态存储（双层）

**完整数据落磁盘**（与 specialist artifact 同构）

```
$SESSION_DIR/agents/orchestration/dynamic_actions/<dyn_id>/
├── spec.json               # 派发时 orchestration 描述的 motivation + side_effects 声明
├── seed_kit.json           # 种子阶段附给 sub-agent 的输入
├── sub_agent_journal.md    # sub-agent ReAct 探索过程
├── proposal_set.json       # 跨域 patch 候选（与 specialist 同 schema）
├── critic_verdict.json
├── dispatch_history.jsonl  # 每次 grid run 的结果
└── telemetry.json          # 累计成功 / revert / 增益（聚合视图源）
```

**SharedState 加聚合视图**

`dynamic_actions: dict[dyn_id, DynamicActionSummary]`，仅放
`id / status / cumulative_gain / last_outcome` 等少量字段，让
orchestration 在下一 tick 看得到自己派过的 dynamic action 的命运；不
放原始日志，避免吃 prompt token budget。

### 1.6 阶段适用性

**仅 EXPLORE**。其他阶段经评估后均不开放：

| 阶段 | 是否开放 | 主要理由 |
|---|---|---|
| PRELUDE | 否 | 确定性 setup 不变量；baseline 未建立时无参照系 |
| FRAMEWORK_PR | 否 | framework-agent (Arbor) 已是同语义的自我扩展通路，再开会双轨 |
| **EXPLORE** | **是** | specialist 体系 + IR-4 + critic gate 均在位，可承接新通道 |
| KERNEL | 否 | IR-1..IR-7 patch-level 不变量，运行时合成无法保证 |
| SWEEP | 否（v1） | 不在本期范围；未来可考虑「动态选 sweep 维度」受限版本 |
| CLOSE | 否 | 无动作面 |

### 1.7 与 specialist 体系的关系

**dynamic action 不取代 specialist**。specialist 仍是 EXPLORE 的默认
通路；dynamic action 是 orchestration 在判断 specialist 不能覆盖某个
跨域机会时**显式**调用的补充入口。两条通路在 `_grid_runner` 层汇合，
对外行为（KEEP / REVERT / cooldown / accuracy gate）完全一致。

orchestration 的 prompt 中**仅声明入口存在**，不附详细使用指引：

> "如果你认为存在一组跨多个 domain 的 patch 组合，且任何单个
> specialist 在其 domain prompt 边界内都不可能提出来，可派发一条
> dynamic action。一般情况下应当依赖 specialist 体系；dynamic action
> 是补充通道，不是默认通道。"

理由：保持 orchestration prompt 体积可控；并依靠 dynamic action 多次
失败时 orchestration 自然回退到 specialist 通路，不引入显式 cooldown
/ kill switch 机制（见 §1.8）。

### 1.8 显式不在 v1 范围内的事项

下列项目在讨论中明确**搁置**，以避免设计 scope 膨胀；将来再开章节
讨论：

- **跨 session 学习**：dynamic action 的 fingerprint / KB 写回 / 跨
  session 复用——暂不处理；
- **多次 revert 后的自动 cooldown / kill switch**：不实现。理由：
  orchestration 自己有 specialist 默认通路可走，dynamic action 多次
  无效时让 orchestration 自主决定回退，不需要外部熔断；
- **prompt 反诱导（防 LLM 用 micro-bench 数字声称增益）**：不写专门
  段落。micro-bench 已在 §1.2 红线明确为 sub-agent **内部**诊断工具，
  其产出不接任何 promote 链路，机械约束足够；
- **SWEEP / 其他阶段扩展**：仅记入未来工作清单；
- **trust tier / probationary→trusted 升级机制**：原 `action-formulation`
  设计的核心治理机制之一，本设计 v1 不采用（dynamic action 每次都是
  即派即用，不入静态 ActionRegistry，无须 tier 概念）。

### 1.9 Phase 0 决策（v1 lock-down）

§1.4–1.5 在初稿中留有若干歧义点；为避免实施期反复返工，v1 现锁定
如下 4 项决策。这些决策不再属于"待讨论"范围，后续若需回滚需走单独
的设计变更流程。

| 决策 | 选择 | 取舍说明 |
|---|---|---|
| **D-A** Dispatch 通道 | **不**引入新 IntentType；以新的 `action_name="dynamic_action"` 复用 `DELEGATE` 通路 | 与 specialist 同构（specialist 也无独立 IntentType），PolicyGate 的 delegate 校验链是天然的拦截点；新 IntentType 会在 intent envelope / 校验 / dispatch 三处都拉一条平行链，反而违背"复用 specialist 框架"的设计意图 |
| **D-B** Critic 分类 | 复用现有 `patch_landing` 审查类，仅在 review_constraints 上挂 `cross_domain=true` 标记并附加跨域审查规则 | dynamic action 的产出本质仍是 patch；新立独立审查类会把 patch_landing 的四 checklist 拷贝一份独立维护，与"specialist patch 与 dynamic patch 在 grid 层汇合"的统一性冲突 |
| **D-C** SharedState 聚合视图保护 | `dynamic_actions` 字段进入受保护字段集，仅 Coordinator 写入；LLM 不能通过 `UPDATE_STATE` 修改 | 与 `optimization_stack` / `current_best` 同性质——是 Coordinator 仲裁后的事实记录而非 LLM 的工作区；写保护避免 orchestration 自我催眠"我的 dynamic action 全都成功了" |
| **D-D** Sub-agent 形态 | 直接采用真正的 multi-turn ReAct loop（多次 LLM 调用 + 工具调度），不走 single-shot 过渡形态 | §1.3 的设计意图就是"探索式"，single-shot 形态在多轮工具调用这一条上不达标；一步到位避免后续 runner 重写 |

---

## 2. 详细设计（待补）

下列子章节将在后续迭代中展开。**章节列表已按 §1.9 的 P0 决策更新**：

- 2.1 `delegate(action_name="dynamic_action")` 的 payload schema
- 2.2 PolicyGate 早期校验规则清单（含 IR-4 provenance 白名单扩展）
- 2.3 Dynamic sub-agent 的 ReAct 工具集、回收路径白名单与 budget
- 2.4 Seed kit 装配逻辑（Coordinator 侧 helper）
- 2.5 Critic `patch_landing` + `cross_domain=true` 附加审查规则
- 2.6 `integrate_patch` 对 dynamic 来源 patch 的额外校验
- 2.7 `_grid_runner` 调度对 dynamic variant 的 fairness 处理
- 2.8 SharedState `dynamic_actions` summary 字段定义与受保护字段集成员关系
- 2.9 Resume 语义（非终态 dynamic action 在 Coordinator 重启后的处理）
- 2.10 Orchestration prompt 增量 + 入口声明文本

---

## 3. 实施路径（v1 概念设计）

> 本节是实施级别的**概念设计**：描述每一阶段的目标、所触及的架构平面
> 与中心思想；不下沉到具体代码 diff。详细的接口字段（§2.1–2.10）按本
> 节阶段顺序逐步展开。

### 3.1 实施分期总览

dynamic action 的接入面横跨 7 个架构平面：

1. **Action 注册** — 新增声明，告知系统这条新通路存在并具备何种能力面；
2. **Intent 校验** — PolicyGate 在派发瞬间守住 §1.2 红线；
3. **Session 目录** — artifact-first，所有派发的输入/输出可复检；
4. **Sub-agent runner** — multi-turn ReAct 探索的物理载体与隔离边界；
5. **Critic** — 跨域 patch 的强校验入口；
6. **SharedState** — orchestration 看到自己派过的 dynamic action 命运的聚合视图；
7. **Orchestration prompt** — 入口声明，仅声明存在、不诱导默认使用。

强行一次性铺到 7 个平面会让"先派得出去"和"先跑得通端到端"互相阻塞，
因此 v1 分为 9 个阶段（P0 决策 + P1–P8 实施 + P9 测试），按
"骨架 → 接通 → 反馈环"递进，每阶段都要求**前一阶段产物已经端到端
可观测**。

### 3.2 阶段 P1：合法派发（"派得出去"）

**目标**：orchestration 能合法发出一条 dynamic action 派发请求，
PolicyGate 能在错误 phase / 错误 source / kernel-owned 禁区等情况下
拒绝它；派发后**不需要真跑**——此阶段仅挂一个 stub executor 即返回
空 proposal_set 即可。

**触及平面**：Action 注册、Intent 校验。

**中心思想**：

- **复用 specialist 的 dispatch 形态**——新 action 以声明式元数据登记
  能力面（执行 lane、合法 phase），不开新 IntentType；这样 PolicyGate
  主链上**只新增一条 delegate 分支**，其余 gate 沿用现有 delegate 路径。
- **校验体现"职权大但有红线"**——payload schema 强制声明
  `motivation_gap` / `scope_domains[]` / `side_effects_declared[]` 三组
  字段。前两者用于审计 motivation 是否构成"specialist 不可能覆盖"；
  第三者用于让 PolicyGate 在派发时就拒掉越过 §1.2 红线（kernel-owned
  / metric 自定义 / 独立 server / 绕过 integrate_patch 等）的请求，
  不让违规意图污染下游。
- **容量与并发独立计帐**——dynamic action 与 specialist 共享
  `research_lane` 物理通道（即 `MAX_RESEARCH_LANE_CAPACITY=6` 不变），
  但**新立**两条独立 cap：每 EXPLORE round 最多 1 条 dynamic dispatch，
  每 round 最多 1 条 dynamic-sourced variant。这避免 dynamic 与
  specialist 互相挤占 IR-4 配额，导致 specialist 体系被边缘化。
- **IR-4 白名单仅做最小扩展**——`provenance` 字面量集合从
  `{specialist:<domain>, default_grid}` 扩为
  `{specialist:<domain>, default_grid, dynamic}`，单字面量 stamp，不引
  入复合 provenance（如 `dynamic+specialist:foo`）以保持审计简单。

**P1 完成的可观测信号**：在 EXPLORE 阶段手动构造一条非法 dispatch
（如 phase=PRELUDE、或 side_effects 中出现 kernel-owned action）应被
PolicyGate 拒绝；合法 dispatch 进入 stub executor 并落地一条空记录。

### 3.3 阶段 P2：Session 目录 + Seed kit 落盘

**目标**：派发瞬间所有"orchestration 看到的世界"都能复检。

**触及平面**：Session 目录、Coordinator dispatch 钩子。

**中心思想**：

- **artifact-first**——dynamic action 的**所有**输入/输出必须在
  `$SESSION_DIR/agents/orchestration/dynamic_actions/<dyn_id>/` 下原样
  落盘（`spec.json` + `seed_kit.json` + 后续阶段产物），SharedState
  只放摘要（见 P6）。这样任何一条派发可独立复盘、可在事后审计
  orchestration 的 motivation 是否成立。
- **seed kit 是 orchestration 的承诺，不是 sub-agent 的自由输入**——
  seed kit 装配逻辑放在 **Coordinator 侧**（不是 sub-agent runner 侧）。
  原因：跨域 KB 命中、profile slice、已采纳/revert patch 摘要这些信息
  的"哪些算 relevant"是**任务调度问题**，不是探索问题；放 sub-agent
  内会让 sub-agent 自己定义自己的输入边界，破坏隔离。
- **seed kit 的字段完全可枚举**——motivation gap、roofline 摘要、
  profile 关键切片、已采纳 patches 列表、已 revert patches 摘要、
  KB pitfalls 命中、源码路径白名单。**不允许**未来悄悄加"任意
  orchestration 觉得有用的东西"——一旦 seed kit schema 退化为开放集，
  就等价于"orchestration 把 prompt 拼好直接喂模型"，§1.2 红线模糊。

### 3.4 阶段 P3：Sub-agent runner（multi-turn ReAct）

**目标**：dynamic sub-agent 在隔离 worktree 内多轮探索，产出
proposal_set。

**触及平面**：Sub-agent runner、Prompt builder、worktree 隔离。

**中心思想**：

- **runner 与 specialist runner 平级、不是子类**——独立模块，与
  specialist runner 平行存在；共享底层任务派单与 worktree 隔离机制，
  但 prompt 装配、回收策略、journal 形态各自独立。强行继承会让
  "specialist 是 single-shot、dynamic 是 multi-turn"这条主要差异藏到
  if/else 里，runner 演化耦合。
- **multi-turn 的物理边界由 runner 强制，不是 prompt 引导**——
  sub-agent 单次运行的 LLM call 数、工具调用数、wall-clock budget 全部
  在 runner 配置里硬编码上限；超限直接终止并落 journal 记录"超 budget
  终止"。这是 §1.2 红线"micro-bench 仅自用、不进 promote 链路"的物理
  基础——若 budget 失控，sub-agent 可以靠跑 N 次 micro-bench 编织数字
  支撑 proposal。
- **回收路径白名单**——runner 从 worktree 只回收**预先定义的固定
  文件**（`proposal_set.json` + `sub_agent_journal.md`），不接受
  sub-agent 自由声明"请把我的 X.json 也带回去"。这是 §1.4
  "micro-bench 产出不进 SharedState" 的机械保障：worktree 之外的
  SharedState / optimization_stack / grid 都看不见 sub-agent 的内部
  诊断文件。
- **proposal_set schema 与 specialist 同构**——dynamic action 的 patch
  与 specialist 的 patch 在评估通道汇合的前提是 schema 一致；唯一新增
  的是 `provenance: "dynamic"` stamp 与可选的 `cross_domain_rationale`
  字段（供 Critic 跨域审查使用）。
- **工具白名单是名词级、不是动词级**——白名单声明可访问的资源
  （"读 framework_source_roots 内文件"、"读 session artifact 路径白
  名单"、"在 worktree 内执行 bench 脚本"），而**不是**声明可以调用
  什么 LLM tool name。这层映射由 runner 侧维护，sub-agent prompt 看到
  的是"你有这些资源"而不是"你有这些 tool name"，避免 prompt 被反向
  利用扩权。

### 3.5 阶段 P4：Critic 跨域审查

**目标**：cross-domain patch 在保留 patch_landing 严格四 checklist 的
前提下，新增跨域风险审查。

**触及平面**：Critic 分类器、review_constraints。

**中心思想**：

- **复用 patch_landing 而不是另立分类**（D-B 决策）——dynamic patch
  与 specialist patch 在落地通道（`integrate_patch`）、在评估通道
  （grid runner + accuracy gate）上完全等价；Critic 侧若另立分类只会
  拷贝维护 checklist，无新信息量。
- **跨域强校验只加少量 review rule**：
  - proposal 必须显式列出**每个被改动的 domain** 与每个 domain 的
    rationale；
  - 必须显式说明**跨 domain 的耦合点与潜在副作用**；
  - 必须说明**此组合不可能由单 domain specialist 提出**的依据，即
    motivation gap 在 critic 视角下仍成立。
- **Critic 的判断面不变**——这些是 review_constraints 在 prompt 里附加
  的检查项，不是新的硬 gate；Critic 仍然给 APPROVE / REJECT / REVISE
  三选一，下游路径与 specialist patch 一致。

### 3.6 阶段 P5：落地通路接通（端到端 happy path）

**目标**：orchestration 派发 → runner 跑 → critic → integrate_patch
→ grid runner → KEEP/REVERT 全链路打通。

**触及平面**：Coordinator dispatch 与 promotion 钩子、ExploreExecutor
variant 路由。

**中心思想**：

- **此阶段以"不改主链"为目标**——P1–P4 完成后，dynamic action 的
  patch 应当能**沿用 specialist patch 的现有路径**进入 critic、
  integrate_patch、grid runner、promote 与 record_intervention。如果在
  P5 发现需要在主链上再开一条平行 promote 路径，说明 P3 输出的
  proposal_set 没有真正与 specialist 同构，应**回头修 P3** 而不是在
  P5 打补丁。
- **唯一允许的局部改动**是 grid 调度上的 fairness 处理（§2.7 待补），
  即"dynamic variant 与 specialist variant 在同一 round 内的派单顺序"
  ——这是调度策略问题，不是落地路径问题。

### 3.7 阶段 P6：SharedState 聚合视图

**目标**：orchestration 在下一 tick 看得到自己派过的 dynamic action 的
命运，但不被原始日志吃光 prompt token。

**触及平面**：SharedState、Coordinator promotion 钩子。

**中心思想**：

- **聚合视图 ≠ 原始事实**——`spec.json` / `seed_kit.json` /
  `proposal_set.json` / `dispatch_history.jsonl` 是事实层（落盘），
  SharedState 上的 `dynamic_actions: dict[dyn_id, DynamicActionSummary]`
  是视图层；视图字段集**严格闭合**（id / status / cumulative_gain /
  last_outcome / dispatched_at），不允许通过 dynamic action 的命运反
  向影响视图 schema。
- **写保护**（D-C 决策）——`dynamic_actions` 进受保护字段集，仅
  Coordinator 在 critic verdict / KEEP / REVERT / abandoned 这四个明确
  节点写入；LLM 不能通过 `UPDATE_STATE` 影响其内容。这条与
  `optimization_stack` 写保护同源——是事实记录，不是工作区。
- **prompt 注入的截断策略**——把 `dynamic_actions` 注入 orchestration
  prompt 时只放最近 N 条（建议 N=5），且在每条上附"完整路径在
  `<artifact_path>`"指针，需要更多上下文时由 orchestration 显式
  follow-up（读 artifact），保持 prompt token 预算可控。

### 3.8 阶段 P7：Orchestration prompt 入口

**目标**：让 LLM 知道 dynamic action 入口存在，但不被诱导默认使用。

**触及平面**：Orchestration system prompt、enabled action 列表、
emit hint。

**中心思想**：

- **prompt 文案严格遵循 §1.7**——"补充通道，不是默认通道"；只声明
  入口存在，不附使用范例、不附启发式触发条件。LLM 越是被 prompt 教
  "什么时候用 dynamic action"，dynamic action 越容易被滥用为
  specialist 失败时的兜底，反而违背 §1.7。
- **emit hint 与 specialist 对称**——指示 orchestration 在合法 phase
  内可发出此 action、payload 字段表、与 specialist 的边界（"不要重复
  specialist 已能覆盖的 single-domain patch"），其余文案极简。

### 3.9 阶段 P8：Resume 语义

**目标**：Coordinator 重启时未完成的 dynamic action 不悬空。

**触及平面**：Coordinator 启动 hook。

**中心思想**：

- **不做"续跑"**——dynamic sub-agent 的 ReAct 探索强依赖 worktree
  状态、外部模型 session、micro-bench 结果，跨进程恢复成本高且不可靠。
- **统一标 abandoned**——重启时扫 `dynamic_actions/` 目录，凡是处于
  "已派发未完成"状态的，全部标 `status=ABANDONED`，写回 SharedState
  summary，artifact 保留，由 orchestration 自行决定是否重新派发。
  这与 §1.7 "多次失败时让 orchestration 自主决定回退" 的设计哲学
  一致，无需引入续跑机制。

### 3.10 阶段 P9：测试矩阵

**目标**：以可重复的方式守住 §1.2 红线。

**测试层级**：

- **单元层**——PolicyGate 新 gate（phase 限制、source 限制、
  kernel-owned 禁区、payload schema、容量上限、IR-4 provenance 白名单）
  必须有显式 negative test；seed kit 装配在缺 profile / 缺 roofline /
  无已采纳 patch 等降级场景下的行为也应覆盖。
- **集成层**——mocked sub-agent 输出固定 proposal_set，验证 critic →
  integrate_patch → grid runner 全链路打通；dynamic + specialist 在同
  一 EXPLORE round 内共存且资源 lane 不冲突。
- **不变量层（最关键）**——以下三条是 §1.2 红线的**机械证据**，必须
  有专门的回归用例：
  1. dynamic sub-agent 的 micro-bench 产出不进 SharedState、不进
     `optimization_stack`、不进 KEEP 判定；
  2. SharedState `dynamic_actions` 不可由 LLM 通过 `UPDATE_STATE`
     修改；
  3. dynamic variant 的 provenance 必须为字面量 `dynamic`，不能是
     复合或伪装为 `specialist:<domain>`。

### 3.11 阶段间依赖与关键路径

P0 → P1 → P2 → P3 → P4 → P5 是关键路径，必须严格顺序；P6 / P7 / P8
在 P5 完成后可并行；P9 测试用例随各阶段同步编写，P5 之后做集成层与
不变量层补全。

任何阶段如果在实施中发现需要"反向修改前一阶段"，触发设计变更流程而
不是在当前阶段打补丁——这是 dynamic action 设计的非功能性约束
（§1 列出的红线全靠"接入面收敛"来维持，一旦在低层阶段做妥协，红线
将在高层阶段以"小改一下就好"的形式被悄悄突破）。
