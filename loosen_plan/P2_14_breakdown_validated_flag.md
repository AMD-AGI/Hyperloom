# P2_14 — 用 `validated=false` 标注替代 stack_rebench deny,保产物诚实

- **Phase**: P2 · **风险**: 中 · **依赖**: 无 · **前置于**: P2_10(必须先于或同 PR) · **后继**: 无

## 目标

删除 `stack_rebench_required` 等"为保证报告增益诚实"的 deny(P2_10)之前,先建立**诚实性的替代机制**:在 `session_breakdown.json` / SharedState 里**显式标注**每个 KEEP / 累计增益是否经过 full-stack rebench 验证(`validated: true/false`),而不是靠阻止动作来保证。这样既放权,又不破坏下游契约(README/pr.md 的黄金契约)。

## 不变量 vs 策略判定

- "报告的累计增益必须诚实/可被下游正确解读" = **产物契约 INVARIANT**(保留)。
- 但"用 deny 阻止 report/explore 直到 rebench 完成"是用**策略 deny** 实现诚实性。改为**数据标注**实现诚实性,既保契约又不阻断 LLM。

## 关键背景

- `explore` 每次 KEEP 内联 full-stack rebench,`cumulative_gain_validated` 随之推进(已有机制)。
- 当前 `stack_rebench_required` deny(`coordinator.py` `_sequence_denial_for_action`)在有未验证 KEEP 时只放行 explore/baseline/report,逼迫先 rebench。
- 下游消费者(`claw-stats-service`、dashboards)读 `session_breakdown.json`(`docs/INTEGRATION_SESSION_BREAKDOWN.md` 契约)。

## 改动清单

### 1. SharedState/stack 标注 validated 状态
- 在 `optimization_stack` 条目与累计增益上,确保有字段区分:
  - `cumulative_gain_validated`(已有)= 经 full-stack rebench 的诚实累计。
  - 每个 KEEP 条目带 `validated: bool` + `validated_gain_pct`(若已 rebench)。
- 若 SharedState 已有等价字段,仅补齐"未验证 KEEP 也带显式 `validated=false`"。

### 2. `session_breakdown` 写出 validated 标注(`breakdown/`)
- `breakdown/collectors.py` / `breakdown/schema.py`:在 stack / best-config / cumulative-gain 输出中**显式**写 `validated` 标志与"validated vs raw"两个数值。
- **保持向后兼容**:不改既有键的语义/形状,只**新增** `validated` 标注字段(下游旧消费者忽略新字段即可)。这是契约扩展而非破坏。

### 3. report 渲染标注(`action_executors/report.py`)
- 最终报告里对未验证 KEEP/增益明确标注"unvalidated (no full-stack rebench)",与 P2_10 删除 hot_kernel_unfinished 后的"未尝试 kernel"标注一致。

## 连带测试

- `breakdown/` 相关测试 + `docs/INTEGRATION_SESSION_BREAKDOWN.md` 契约:新增 `validated` 字段的断言;确认既有键形状不变。
- 集成烟测:`session_breakdown.json` schema 校验通过(黄金契约)。
- 若有 `test_session_breakdown_*` / breakdown collector 测试,补 validated 标注用例。

## 验证
- 含未验证 KEEP 的 session,其 `session_breakdown.json`:既有键形状不变 + 新增 `validated=false` 标注;`cumulative_gain_validated` 与 raw 分列。
- 下游契约校验(`docs/INTEGRATION_SESSION_BREAKDOWN.md`)通过。
- 与 P2_10 联调:删 stack_rebench deny 后,未验证 KEEP 仍被如实标注。

## 回退
- 移除新增 `validated` 字段(因是新增字段,回退不影响既有消费者)。

## 残留风险
- 中。必须确保 `validated=false` 在所有产出路径(正常结束 / deadline / interrupt / resume)都正确落盘,否则 P2_10 放权后会出现"未验证却显示为已验证"的失真。**P2_14 必须先于或同 PR 落地 P2_10。**
