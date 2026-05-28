# P7 — Orchestration Prompt 入口声明

> 第七阶段的对外可见产物：orchestration LLM 知道 dynamic action 这条
> 入口存在，但 prompt 不诱导其使用。文案严格遵循 dynamic_action.MD
> §1.7：声明存在，不附详细使用指引。
>
> 对应 dynamic_action.MD §3.8。

---

## 1. 目标

在 orchestration 的 system prompt 中加入 dynamic action 入口的最小
声明，让 LLM 在合适时机能想到这条通路；同时**不让** prompt 体积膨胀、
**不让** dynamic action 变成 specialist 失败时的默认兜底。

具体可观测的产物：

1. orchestration system prompt 中出现一节"Dynamic Action"声明，文案
   ≤ 100 tokens；
2. enabled action 列表中包含 `dynamic_action`；
3. emit hint 中描述 payload 字段表，与 specialist 的 emit hint 风格
   对齐；
4. P6 注入的 `dynamic_actions` summary 段落与 §3 文案良好衔接。

---

## 2. 触及的架构平面

| 平面 | 改动性质 | 中心思想 |
|---|---|---|
| Orchestration system prompt | 增加 1 节声明（极简） | §1.7 文案严格不诱导 |
| Enabled action 列表 | 加 1 条 `dynamic_action` | 注册位置与 specialist 平级 |
| Per-action emit hint | 加 1 条 hint | 与 specialist hint 对称 |
| Prompt composer | 配合 P6 summary 段落 | 段落顺序与体积控制 |

---

## 3. 入口声明文案（DEFAULT，待 review）

### 3.1 文案原文（节选自设计稿 §1.7，可微调）

> ## Dynamic Action
>
> 如果你认为存在一组跨多个 domain 的 patch 组合，且任何单个 specialist
> 在其 domain prompt 边界内都不可能提出来，可在 EXPLORE 阶段派发一条
> dynamic action（`delegate(action_name="dynamic_action", ...)`）。
>
> 一般情况下应当依赖 specialist 体系；dynamic action 是补充通道，不是
> 默认通道。每 round 至多 1 条。

### 3.2 文案的设计原则

- **只声明存在 + 边界**——告诉 LLM "这条入口存在 / 何时是合理的 /
  不是默认"；
- **不附触发启发式**——不写"什么时候应该用 dynamic action"的具体例
  子；理由：example 会被 LLM 当作模式匹配，凡是对得上 example 的就发
  dynamic action，反而被滥用；
- **不附 specialist 失败的引导**——不写"当 specialist 多次失败可以考
  虑 dynamic"，避免 dynamic 被定位为兜底通道；
- **明确 round-cap**——让 LLM 知道节奏。

### 3.3 不在文案中出现的内容

- 任何"dynamic action 比 specialist 强"的暗示；
- 任何关于 sub-agent 内部工作方式的描述（multi-turn / micro-bench 等）
  ——LLM 不需要知道实现细节，知道了只会催生越界尝试；
- 任何关于 KEEP / REVERT 阈值的细节；
- 任何关于成本的暗示（"dynamic action 很贵慎用"——这种 negative
  guidance 反而强化了 dynamic 的"特殊感"，导致两极反应：要么不敢用、
  要么 token 多就敢用）。

---

## 4. Enabled action 列表

dynamic action 加入 orchestration 的 enabled action 列表（在 EXPLORE
phase 启用）。注册条目与 specialist 风格对齐。

### 4.1 注册位置

- **静态层**：在 prompt 系统的 enabled action 常量集合中加 `dynamic_action`；
- **运行时层**：phase_state 的 `PHASE_ALLOWED_ACTIONS[EXPLORE]` 加
  `dynamic_action`（这条已在 P1 §2 / P1 §5 完成，P7 不重复）。

### 4.2 与其他 enabled action 的并列关系

| Action | 触发场景 |
|---|---|
| `explore` (grid) | orchestration 主导的 grid 探索 |
| `specialist` | 单 domain 探索（默认通路） |
| **`dynamic_action`** | **跨 domain 探索（补充通路）** |
| 其他既有 actions | ... |

### 4.3 不引入"互斥"约束

不通过 prompt 文案告诉 LLM "用了 specialist 就不能用 dynamic"。互斥
是配额问题（round-cap 已在 P1 守住），不是逻辑问题——orchestration
理论上可以在同 round 内同时派 specialist 和 dynamic（共享 lane，FIFO
派单）。

---

## 5. Per-action emit hint

### 5.1 hint 内容（DEFAULT）

emit hint 是 prompt 中"如何 emit 此 intent"的字段说明，对 LLM 是结构
化的字段表 + 简短约束描述。dynamic action 的 emit hint 至少包含：

- **payload 必填字段表**：
  - `motivation_gap_text` (string, non-empty)
  - `scope_domains` (list of string, length ≥ 2)
  - `side_effects_declared` (list of string)
  - `budget_hint` (optional, in {low, medium, high})
- **关键约束**（≤ 50 tokens）：
  - "scope_domains 必须 ≥ 2 个 domain"
  - "side_effects_declared 不允许包含 kernel / metric / server 类操作"
  - "每 EXPLORE round 最多 1 条"
- **失败 reason code 列表**（让 LLM 看到错误反馈时能解读）：
  - `dynamic_phase_violation` / `dynamic_scope_too_narrow` /
    `dynamic_side_effects_red_line` / `dynamic_round_cap_exhausted` 等
    （来自 P1 §4）。

### 5.2 hint 的中心思想

- **payload 字段表是 LLM 唯一可信赖的字段集来源**——不引导 LLM 通过
  其他字段"扩展"payload；
- **关键约束极简**——只列硬约束（PolicyGate 会拒绝的）；不写"建议性
  约束"（如"建议每个 scope_domain 给 30 字以上 rationale"——这种
  guidance 在 prompt 里是 noise）；
- **reason code 列表让 LLM 自学纠错**——LLM 看到拒绝信号时能直接
  识别哪条规则不过，不需要 trial-and-error。

### 5.3 与 specialist emit hint 的对称性

specialist emit hint 已有的结构是 payload 字段表 + 约束 + reason
codes；dynamic 的 hint 完全沿用此结构。这条对称性让 LLM 能复用对
specialist hint 的理解模型，降低 prompt 学习成本。

---

## 6. Prompt 段落顺序与体积

orchestration system prompt 的段落顺序（DEFAULT，待 review）：

```
[既有 system prompt 头部]
[既有 phase 描述]
[既有 EXPLORE 通用规则]
[Specialist 段落 — 默认通道，详细引导]
[Dynamic Action 段落 — 补充通道，§3 极简文案]    ← P7 新增
[Dynamic Action History — P6 注入的 summary]    ← P6 已设计
[既有结尾 / 工具说明]
```

### 中心思想

- **顺序上 dynamic 段落紧跟 specialist 段落**——让 LLM 在阅读完
  specialist 引导后立刻看到 dynamic 是 specialist 的补充；
- **summary 段落紧跟 dynamic 声明**——LLM 派发 dynamic action 之前
  能看到自己历史命运；
- **总体积控制**：dynamic 段落 + summary 段落合计 ≤ 500 tokens（声明
  约 100 + emit hint 约 150 + summary 约 250）。

---

## 7. 入口诱导风险的具体规避

dynamic action 容易被滥用的情景与本阶段的对应规避：

| 滥用情景 | 规避机制 |
|---|---|
| LLM 把 dynamic 当 specialist 失败时的兜底 | §3 文案不写 specialist 失败引导 |
| LLM 看 example 模式匹配派发 | §3 文案不附 example |
| LLM 用 dynamic 试图绕过 round-cap | round-cap 在 P1 PolicyGate 强制 |
| LLM 用 dynamic 试图绕过 kernel 禁区 | side_effects_declared 红线在 P1 强制 |
| LLM 把 dynamic 作为 trial / 探索新功能用 | §3 文案明确"不是默认通道"+ 历史 summary 让重复试错可见 |

---

## 8. 依赖与前置条件

P7 与 P6 在 P5 完成后可并行启动。P7 依赖：

- P1 的 PolicyGate 拒绝 reason codes（用于 emit hint 中的错误反馈
  字段）；
- P6 的 summary 字段结构（用于段落顺序设计）；
- 既有 orchestration prompt builder 与 enabled action 注册机制。

---

## 9. 验收信号

| # | 测试场景 | 期望结果 |
|---|---|---|
| 1 | 静态 prompt 渲染 | "Dynamic Action" 段落出现；文案与 §3.1 一致；体积 ≤ 100 tokens |
| 2 | 静态 prompt 渲染 | enabled action 列表含 `dynamic_action` |
| 3 | emit hint 渲染 | payload 字段表完整；约束列表与 §5.1 一致 |
| 4 | LLM 在 EXPLORE phase 派发 dynamic action | PolicyGate 通过（与 P1 §9 #1 一致） |
| 5 | LLM 在 PRELUDE phase 试图派发 | PolicyGate 拒绝；reason code `dynamic_phase_violation` 反馈到 LLM；下一 turn LLM 能正确识别错误 |
| 6 | 0 条 dynamic action 历史 | summary 段落不展示（与 P6 §7.2 一致） |
| 7 | 多条 dynamic action 历史 | summary 段落展示最近 5 条 |
| 8 | prompt 总体积回归测试 | 与未引入 dynamic action 前对比，体积增量 ≤ 500 tokens |

---

## 10. DEFAULT / 待 review

| # | 条目 | DEFAULT | 备注 |
|---|---|---|---|
| 1 | §3 文案具体措辞 | §3.1 | 待 review；中文 vs 英文也需确定 |
| 2 | 是否在 prompt 中包含 reason code 表 | 是 | §5.1 |
| 3 | 段落顺序 | §6 | 待 review |
| 4 | 总体积上限（dynamic 相关） | ≤ 500 tokens | 待 review |
| 5 | 是否在文案中包含示例 motivation_gap_text | 否 | 关键设计选择，强烈建议保留"否" |
| 6 | reason code 是给"全部"还是"top N" | 全部（约 8 条，体积可控） | 待 review |

---

## 11. 与 §1.7 / §1.8 的对应关系

| §1.7 / §1.8 设计哲学 | 在 P7 的落点 |
|---|---|
| "补充通道，不是默认通道" | §3.1 文案直接引用 |
| "仅声明入口存在，不附详细使用指引" | §3.2 设计原则 + §3.3 不在文案中的内容 |
| "依靠 dynamic action 多次失败时 orchestration 自然回退到 specialist 通路" | §3.2 不附触发启发式；P6 summary 让历史失败对 LLM 可见，自然影响下次决策；§3.3 不附 negative guidance |
| "不引入显式 cooldown / kill switch 机制" | P7 不在 prompt 中加任何"已失败 N 次后停止"的引导 |
| "保持 orchestration prompt 体积可控" | §6 体积控制 |

P7 是设计哲学层（§1.7 / §1.8）在 prompt 层的最后落实——任何 prompt
诱导都会反向破坏"orchestration 自主决定"这条核心约束。
