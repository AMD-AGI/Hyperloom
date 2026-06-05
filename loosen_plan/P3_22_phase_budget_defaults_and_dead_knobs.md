# P3_22 — 调高/简化 phase 预算默认 + 删死旋钮 `steward_continuation_cap`

- **Phase**: P3 · **风险**: 低 · **依赖**: P2_13/P3_17(steward 相关) · **后继**: 无

## 目标

收尾两件事:(1) 让默认预算分配更利于"探得多/深";(2) 清理放权过程中产生或暴露的**死旋钮**与冗余预算逻辑。

## 改动清单

### 1. phase 预算默认(`phase_state.py`)
- `DEFAULT_PHASE_BUDGET_PCT`(278–293:PRELUDE 5% / EXPLORE 40% / KERNEL 35% / SWEEP 18% / CLOSE 2%):
  - **方案 A(温和)**:调高 EXPLORE/KERNEL 比例(探索/优化是主战场),压缩 SWEEP/PRELUDE。
  - **方案 B(简化,偏删除)**:去掉**软**的 per-phase pct 分配,只保留**总 `max_hours` + IR-6 硬 force-exit**作为时间约束,让 LLM 在总预算内自由分配各 phase 时间(更符合放权)。phase 预算耗尽硬墙(1169–1175 等)在方案 B 下退化为"总预算"驱动。
- **保留**:IR-6 EXPLORE force-exit(581–659)、终止态路由、CLOSE 顺序器(产物契约)。
- `ESCALATE_HINT_BUDGET_BUMP_CAP`(400,80%)/ `DELTA`(399,+5pp):若保留 per-phase pct,放宽 cap;若走方案 B,这些随 per-phase pct 一并简化。

### 2. 删死旋钮 `steward_continuation_cap`(`cli.py`)
- 审计确认:`--steward-continuation-cap`(cli 5531–5537)写入 `plateau_overrides["steward_continuation_cap"]` 但**全代码无读取点**(死旋钮)。
- P2_13/P3_17 后 steward 降级/删除,该旋钮彻底无意义 → **删除** CLI flag 与 override key。

### 3. 清理其它放权后产生的死旋钮/死开关(触类旁通)
- `--depth-gate` 及阈值(P2_13 删 depth_gate 后变死)→ 删除或标注 no-op。
- `INFERENCE_OPTIMIZER_EXPLORE_ROOFLINE_HARD_GATE` / `--explore-roofline-hard-gate`(P2_15 删硬门后)→ 删除。
- `--steward-disabled`(P3_17 删 steward 后)→ 删除或 no-op。
- `INFERENCE_OPTIMIZER_ALLOW_EARLY_KERNEL_OPT`(P2_10 删 hot_kernel_unfinished 后,该 escape hatch 失去意义)→ 复核并删除。
- `--plateau-*` overrides(P3_17 后 plateau 仅 advisory):保留为"影响 advisory 计算"或删除——按是否仍展示 plateau advisory 决定。

> 死旋钮清理必须在对应删除步落地**之后**做,避免误删仍在用的旋钮。本步是"放权完成后的统一清扫"。

### 4. 补缺失 CLI(可选)
- `framework_pr_lookback` / `framework_pr_keep_gain_pct` / `framework_pr_force_exit_hours_ratio` 已在 override schema 但无 CLI——若 P3_17 后 framework_pr plateau 改 advisory,这些可一并删除或补 CLI(按是否保留 framework_pr 软预算决定)。

## 连带测试

| 文件 | 动作 |
|---|---|
| `test_phase_state_machine.py`(budget 109–387)、`test_phase_force_exit.py` | 默认 pct 调整 / 方案 B 简化后更新;**保留** IR-6 force-exit 用例 |
| `test_phase_state_plateau.py`(438–456 budget) | 同步 |
| CLI 测试(steward_continuation_cap / depth-gate / roofline-hard-gate 等死旋钮) | 删除对应断言 |
| `test_specialist_concurrent_dispatch.py`(research_lane 默认) | 若 P1_05 已改默认,这里复核一致 |

## 验证
- 默认预算更偏向 EXPLORE/KERNEL(方案 A)或仅受总预算约束(方案 B)。
- IR-6 / 终止态 / CLOSE 顺序器仍生效(预算耗尽/deadline 能收尾)。
- `--steward-continuation-cap` 等死旋钮移除后 CLI help 干净;`inference_optimizer --help` 黄金对比更新。
- grep 确认无死旋钮残留引用。

## 回退
- 恢复默认值与删除的旋钮。

## 残留风险
- 低。方案 B(去 per-phase pct)更激进:确认在无 per-phase 软预算时,IR-6 + 总 max_hours + plateau advisory 足以让 LLM 合理分配各 phase 时间且能在预算内收尾(建议 A/B 验证后再选 B)。
