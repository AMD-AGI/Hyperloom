# P4 — Critic 跨域审查

> 第四阶段的对外可见产物：dynamic action 产出的 patch 在 Critic 端被
> 自动识别为 cross-domain，沿用 `patch_landing` 的全部既有 checklist
> 并附加少量跨域强校验；审查未通过 → REJECT/REVISE，下游不进
> integrate_patch。
>
> 对应 dynamic_action.MD §3.5。

---

## 1. 目标

让 Critic 能正确审查 dynamic action 的 patch，且**不弱化**任何既有
patch_landing 严格性。本阶段后，Critic 对 dynamic patch 的判定必须
满足：

1. 既有 `patch_landing` 四 checklist 全部通过（与 specialist patch
   等价严格度）；
2. 额外的"跨域专属"3 条 review rule 通过；
3. proposal 中 `cross_domain_rationale` 字段缺失或不充分时，Critic
   返回 REVISE（让 sub-agent 修补）而非 REJECT（彻底拒绝），尽量保留
   修复机会；
4. 最终 verdict 与 specialist patch 同结构（APPROVE / REJECT / REVISE）。

---

## 2. 触及的架构平面

| 平面 | 改动性质 | 中心思想 |
|---|---|---|
| Critic 分类器 | 在既有 `classify_proposal_action` 上增加分支 | 识别 `provenance == "dynamic"` 自动挂 `cross_domain` flag |
| Critic review_constraints | 加 `cross_domain` flag 后追加 3 条 rule | 不动既有 patch_landing 四 checklist |
| Critic prompt 模板 | 增量 1 段跨域审查指引 | 措辞与 patch_landing 部分对齐 |
| Critic verdict 落盘 | 沿用 specialist 路径 | verdict 写入 dynamic action artifact 目录 |

---

## 3. 分类器扩展（中心思想）

### 3.1 自动识别

dynamic action 的 patch 在 Critic 收到时携带 `provenance == "dynamic"`
（来自 P3 runner 输出 schema）。分类器据此：

- bundle_action_class = `patch_landing`（D-B 决策，复用既有审查类）；
- review_constraints 上加一对 KV：`cross_domain = true`；
- 不引入新的 bundle_action_class。

**关键**：识别基于 `provenance`，而**不是**基于 LLM 在 prompt 里"声称"
自己是 dynamic。这条让 LLM 没法通过伪装 specialist provenance 绕过
跨域审查规则——provenance 字面量在 P1 IR-4 白名单与 P3 runner 输出
schema 中都被强制为 `"dynamic"`。

### 3.2 与 specialist 分类的对照

| 来源 | bundle_action_class | review_constraints |
|---|---|---|
| specialist patch | `patch_landing` | `{}` |
| dynamic patch | `patch_landing` | `{cross_domain: true}` |

Critic 在 `cross_domain == true` 时**追加**审查规则，而不是替换。

---

## 4. 跨域审查规则（增量）

`cross_domain == true` 触发以下 3 条额外规则。每条都用自然语言措辞，
作为 review_constraints 的一部分注入 Critic prompt。

### 规则 1 — 每 domain rationale

> proposal 必须显式列出每个被改动的 domain，并对每个 domain 给出
> 独立的 rationale（即在该 domain 边界内为什么这个改动是必要的）。

**Critic 检查**：

- `cross_domain_rationale` 文本中是否对 `scope_domains` 列出的每个
  domain 都有独立段落或显式提及；
- 如果缺一个 domain → verdict = REVISE（reason
  `cross_domain_rationale_incomplete`）。

### 规则 2 — 跨 domain 耦合点与副作用

> proposal 必须说明**为什么这些 domain 的改动需要一起发生**——即
> 跨 domain 的耦合点（如"改 KV cache 布局后 attention kernel 必须改
> 调用方式"），以及**潜在副作用**（如"改 scheduler 后 prompt cache
> 命中率可能下降"）。

**Critic 检查**：

- `cross_domain_rationale` 是否显式提及"耦合 / 联动 / 依赖" 类语义；
- 是否提及至少一项潜在副作用；
- 缺失 → verdict = REVISE（reason `cross_domain_coupling_unspecified`）。

**这条规则的意图**：单 domain specialist 不需要思考跨 domain 副作用；
dynamic action 的核心价值就是 *因为* 想到了这一层耦合才存在。如果
sub-agent 提不出耦合点的论述，说明这个 patch 实际上是"两个独立 single
domain patch 拼在一起"——本应由 specialist 的两次独立派发覆盖，不该
吃 dynamic action 的额度。

### 规则 3 — Motivation gap 在 critic 视角下成立

> proposal 必须说明：**为什么任何单个 specialist 在自己的 domain
> prompt 边界内不可能提出来这个组合**。

**Critic 检查**：

- `cross_domain_rationale` 中是否显式回应 "specialist 边界内不可能
  覆盖"的论点（可以与 spec.json 的 `motivation_gap_text` 对照，但
  Critic 视角下要重新成立）；
- 如果论证为"specialist A 提的 X + specialist B 提的 Y 简单拼接"——
  这种组合 orchestration 用 `explore.params.grid` 拼 combo 即可（设计
  稿 §1.1 显式排除的场景），不应该走 dynamic 通路 → verdict = REJECT
  （reason `cross_domain_motivation_invalid`）。

**这条规则的意图**：是防止 dynamic action 退化为 grid combo 的 LLM 版。
如果 motivation 本身不成立，这条 dynamic dispatch 从一开始就不该存在；
critic 是最后一道关卡。

---

## 5. Verdict 形态与下游路径

### 5.1 verdict 三态

dynamic patch 的审查 verdict 与 specialist patch 完全对齐：

| verdict | 语义 | 下游处理 |
|---|---|---|
| APPROVE | 通过审查 | 进入 P5 的 integrate_patch 路径 |
| REJECT | 不可修复的违规 | 不进 integrate_patch；artifact 保留；SharedState summary 标 verdict=REJECT |
| REVISE | 可修复，给出修正建议 | 在 v1 中处理同 REJECT（不开 sub-agent 二次派发循环），但 reason 单独标记 |

### 5.2 v1 不开"二次派发"

REVISE 是 critic 给 sub-agent 的"修正建议"。理论上可以让 runner 启
另一轮 sub-agent 改 patch；v1 不做这条。理由：

- 引入二次派发 = 引入新的 sub-agent lifecycle 边界（"哪轮算重派？"
  "原 dyn_id 如何继承？"），状态机复杂度爆炸；
- §1.7 的设计哲学是"多次失败时让 orchestration 自主决定回退"，REVISE
  本质上也是失败信号——让 orchestration 看到 verdict=REVISE 后自己
  决定是否再发一条新的 dynamic action（**新 dyn_id**），更符合整体
  设计。

未来如果 dynamic action 命中率证明值得做"二次修补"，再在设计变更流程
中加 REVISE 二次派发循环。

### 5.3 verdict 落盘

`critic_verdict.json` 写入
`agents/orchestration/dynamic_actions/<dyn_id>/`，字段集（封闭）：

- `verdict`：APPROVE / REJECT / REVISE；
- `reason_codes`：list（如 `["cross_domain_rationale_incomplete"]`）；
- `reviewer_notes`：自然语言审查笔记；
- `applied_rules`：本次审查实际触发的规则名（包含基础 patch_landing
  四 checklist 与本文 §4 的 3 条跨域规则）；
- `cross_domain_flag`：true（用于事后审计是否走对了路径）。

---

## 6. Critic prompt 增量

Critic 系统 prompt 中已有 patch_landing 段落。本阶段在该段落**末尾**
追加一节"跨域审查"：

- 出现条件：`review_constraints.cross_domain == true`；
- 文案要点：
  - 引用 §4 的 3 条规则原文；
  - 说明 dynamic patch 与 specialist patch 在 patch_landing 四 checklist
    上的等价严格度；
  - 强调"REJECT 是用于不可修复的违规（如 motivation 退化为 grid
    combo），REVISE 用于可修复"。

**不在此处展开 prompt 全文**——具体措辞留给实施期实地编写并 review。

---

## 7. 审查严格度的对称性

为避免审查偏差，关键性原则：

- **patch_landing 四 checklist 在 dynamic 上不弱化**——dynamic patch
  必须满足"测试通过 / 不破坏 baseline / 不引入未声明依赖 / 改动有
  contract" 这些既有要求，与 specialist patch 同严格；
- **跨域 3 条规则只在 cross_domain == true 时启用**——specialist patch
  不受这 3 条影响（避免单 domain patch 被"应该说说跨 domain"误伤）；
- **不引入"dynamic 因为更难所以放宽"的逻辑**——dynamic action 的
  "更高职权"体现在输入侧（§1.2 表格），不体现在输出审查侧。

---

## 8. 依赖与前置条件

P4 必须在 P3 之后实施。P4 依赖：

- P3 输出的 `proposal_set.json` 中 `provenance == "dynamic"` 与
  `cross_domain_rationale` 字段已存在；
- Critic 既有的 `patch_landing` 审查类与四 checklist。

P4 的产出会被 P5 直接消费：

- P5 在 `_handle_review_verdict` 路径上根据 verdict 决定是否进入
  integrate_patch；
- P5 在派单 dynamic variant 进 grid 之前必须已收到 APPROVE。

---

## 9. 验收信号

| # | 测试场景 | 期望结果 |
|---|---|---|
| 1 | sub-agent 输出 cross_domain_rationale 完整 + patch_landing 全过 | verdict = APPROVE |
| 2 | cross_domain_rationale 漏一个 domain | verdict = REVISE，reason `cross_domain_rationale_incomplete` |
| 3 | cross_domain_rationale 未提耦合点 | verdict = REVISE，reason `cross_domain_coupling_unspecified` |
| 4 | motivation 实质是 specialist combo | verdict = REJECT，reason `cross_domain_motivation_invalid` |
| 5 | patch_landing 中某 checklist 不过 + cross_domain 完整 | verdict 取严：REJECT（patch_landing 不过） |
| 6 | provenance 被伪造为 `specialist:foo` 的 dynamic patch（异常路径） | 应在 P1 IR-4 白名单或 P3 runner 输出 schema 处已被拒；如漏到 critic → fail-fast 拒绝并报警 |
| 7 | dynamic patch 携带 `expected_gain` 数字字段（异常） | 应在 P3 runner validation 已 reject；如漏到 critic → critic 直接 REJECT，reason `dynamic_quantitative_claim_violation` |
| 8 | specialist patch（非 dynamic）走 critic | 不触发跨域 3 条规则；与既有路径完全等价 |
| 9 | proposal_set 为空（empty=true） | critic 跳过审查，直接走 specialist empty 等效路径（详见 P5 §6） |

---

## 10. DEFAULT / 待 review

| # | 条目 | DEFAULT | 备注 |
|---|---|---|---|
| 1 | REVISE 是否触发 sub-agent 二次派发 | 否（v1 等同 REJECT） | 待 review |
| 2 | cross_domain_rationale 中 domain 提及检查方式 | 字符串包含或自然语言段落识别 | 措辞细节交给 Critic prompt + LLM judgement |
| 3 | reason_codes 命名前缀 | `cross_domain_*` | 一致风格 |
| 4 | applied_rules 是否记录全部规则名 | 是（含通过的） | 便于审计 |
| 5 | provenance 校验落点 | P1 + P3 + P4 三层（每层独立校验） | 多层防御 |

---

## 11. 与 §1.2 红线的对应关系

| 红线 | 在 P4 的落点 |
|---|---|
| Patch 范围：跨 domain 组合 | 规则 1 + 规则 2（缺一不可）确保 patch 真是跨域而非单 domain 伪装 |
| 不能简单拼 specialist combo | 规则 3 在 critic 视角下重新论证 motivation gap |
| 输出严格度与 specialist 同 | §7 对称性原则；patch_landing 四 checklist 不弱化 |
| 不能声明自己的 metric | §9 #7 双层防御（P3 + P4） |
| provenance 字面量 = `dynamic` | §9 #6 三层校验（P1 IR-4 + P3 runner schema + P4 critic 入口） |

P4 的作用是 §1.2 输出侧红线的最后一道关卡——dispatch 时（P1）+
sub-agent 输出时（P3）+ critic 审查时（P4）三层独立校验，任一层失守
红线就被破坏。
