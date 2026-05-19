# M7 — SWEEP / KERNEL 收敛 + plateau 阈值参数化

## 1. 设计目标

把 M2 暂用的 plateau proxy (`params_no_promote_streak >= 5`) 替换为
§3.8 真正的 plateau 计算, 同时把所有 plateau 阈值 / phase budget /
specialist 汇总策略参数化为 CLI flag, 完成 v0.8 GA 收尾。

落地后用户应当看到: EXPLORE 在合适时机自然 plateau 转 KERNEL; KERNEL
在合适时机 plateau 转 SWEEP; SWEEP grid 跑完进 CLOSE; breakdown 中
attribution.phase_breakdown 段完整 (每段 phase 各自贡献的 gain)。

## 2. 范围

**包含**:

- plateau_explore 真定义 (§3.8 §5.1: AND of 累计 KEEP 增益 < ε +
  连续 K 轮空 proposal_set)。
- plateau_kernel 真定义 (§3.8 §5.2: OR of 连续 REVERT streak + 累计
  KEEP 增益 < ε)。
- robustness `escalate_strategy_change` 与 phase 状态机的合流逻辑
  (`hint=skip_to_kernel/skip_to_close/extend_explore_budget`)。
- specialist 汇总策略 partial_k 模式调参 (M6 已落 flag, M7 给默认值
  调优)。
- attribution.phase_breakdown 段 collector (按 specialist domain 拆解
  EXPLORE 段的 gain 贡献)。
- `stop_reason` 全词表锁定 (含 plateau_explore / plateau_kernel /
  no_kernel_skipped 等新条目)。
- v0.8 GA 收尾 doc / changelog / migration guide。

**不包含**:

- 任何新功能模块 (本里程碑是收尾调参, 不引入概念变更)。

## 3. 与 M2/M5/M6 的关系

- M2 暂用 proxy 退出 EXPLORE, 本里程碑替换为真 plateau。
- M5/M6 已提供 specialist 派发 + 汇总数据 (specialist_rounds /
  explore_search.winners_history); 本里程碑 plateau 计算从这两个数据
  派生。

## 4. 概念交付物

| 交付物 | 说明 |
|---|---|
| plateau_explore 函数 | pure function over (winners_history, specialist_rounds) |
| plateau_kernel 函数 | pure function over (kernel_opt_attempts, kernel_integrate_attempts) |
| escalate hint 词表 | skip_to_kernel / skip_to_close / extend_explore_budget / extend_kernel_budget / pause_specialist_<domain> |
| CLI flags | `--plateau-explore-keep-gain` (默认 0.5%), `--plateau-explore-empty-streak` (默认 3), `--plateau-kernel-revert-streak` (默认 3), `--plateau-kernel-keep-gain` (默认 0.5%), `--specialist-aggregation` (默认 wait_all), `--specialist-partial-k-ratio` (默认 0.5) |
| attribution.phase_breakdown | 见 §3.12 §4.6 |
| stop_reason 词表锁定 | §3.8 §6 词表加入 PolicyGate ENUM |
| GA 收尾文档 | CHANGELOG.md 加入 v0.6→v0.8 概览, migration guide 引用 §3.15 |

## 5. plateau 计算细节

### 5.1 plateau_explore (§3.8 §5.1 真定义)

输入数据:

- `explore_search.winners_history`: 最近 K 轮 (K = `ε_explore_lookback`,
  默认 5) 的 winner 列表
- `specialist_rounds`: 最近 K 轮的 round summary (含 each round 的
  proposals_total, kept_count)

判定:

```
recent_rounds = last(K) of specialist_rounds
recent_keeps = sum(round.kept_count for round in recent_rounds)
recent_keep_gain = sum(winner.gain_pct for winner in
                       last(K) of explore_search.winners_history)
recent_empty = streak of "round.proposals_total == 0 ∧
                          round.proposals_kept == 0"
                from end of specialist_rounds

plateau_explore = (
    recent_keep_gain < ε_explore_keep_gain
    AND recent_empty >= ε_explore_empty_streak
)
```

注意: `wait_all` 模式下 round 概念清晰; `partial_k` 模式下 round 划分
按 propose_action 落地的轮 (而非每个 specialist 完成的轮)。

### 5.2 plateau_kernel (§3.8 §5.2 真定义)

```
recent_kernels = last(K) of kernel_opt_attempts
                  (按 kernel_id 去重, 取每 kernel_id 最后一条)

recent_revert_streak = streak of "kernel.outcome ∈ {REVERT, NEEDS_REVIEW}"
                        from end

recent_keep_gain = sum(integrate.gain_pct for integrate in
                        last(K) of kernel_integrate_attempts
                        where integrate.outcome == KEEP)

plateau_kernel = (
    recent_revert_streak >= ε_kernel_revert_streak
    OR recent_keep_gain < ε_kernel_keep_gain
)
```

### 5.3 escalate_strategy_change 合流

robustness emit `escalate_strategy_change{hint=...}` 后, Coordinator
在下 tick 扫退出条件时优先识别:

- `hint=skip_to_kernel` 在 EXPLORE 内 → 视作 plateau_explore 立即触
  发, phase_history.evidence 标 `llm_escalation`。
- `hint=skip_to_close` 任意 phase 内 → 视作 robustness_escalated 立
  即触发, stop_reason 设置。
- `hint=extend_explore_budget` 在 EXPLORE 内 → 把 phase_budget_pct.explore
  动态提高 5% (上限 80%); 写 phase_history.evidence。
- `hint=extend_kernel_budget` 类似。
- `hint=pause_specialist_<domain>` 把 `specialist_domain_empty_streak[domain]`
  强制 ≥ 阈值, 当前 phase 内不再派该 domain。

未识别的 hint 仅入 log, 不动 phase。

## 6. attribution.phase_breakdown 详细

数据派生:

- 遍历 `optimization_stack` (有序列表, KEEP 时间戳保留)。
- 每个 stack entry 看其入 stack 的时间戳落到哪段 phase_history。
- explore 段内再按 `winners_history.<entry>.specialist_origin` 字段
  归 domain (M6 起 winners_history 应当带 source 字段, 本里程碑确认)。
- kernel 段按 kernel_id 拆。
- sweep 段一般 0 (sweep 不入 stack)。

输出字段示例:

```
attribution.phase_breakdown = {
  "prelude": 0.0,
  "explore": {
    "total_gain_pct": 18.4,
    "by_domain": {
      "kernel_specialist": 5.2,
      "framework_specialist": 9.7,
      "comm_specialist": 1.0,
      "default_grid": 2.5
    }
  },
  "kernel": {
    "total_gain_pct": 7.1,
    "by_kernel_id": {
      "fmoe_fp8_blockscale_g1u1": 4.3,
      "rms_norm_decode": 2.8
    }
  },
  "sweep": 0.0
}
```

## 7. 实施步骤 (PR 拆分)

| PR | 内容 |
|---|---|
| 1 | plateau_explore / plateau_kernel pure function + 单测 |
| 2 | 替换 M2 中 EXPLORE 退出条件的 proxy (params_no_promote_streak) 为真 plateau_explore |
| 3 | 替换 KERNEL 退出条件 proxy 为真 plateau_kernel (M2 已落 KERNEL phase 推断, 本 PR 接真信号) |
| 4 | escalate_strategy_change hint 识别逻辑 |
| 5 | CLI flag 全集 (plateau 阈值 + specialist aggregation) |
| 6 | winners_history.specialist_origin 字段 (M6 起应已写, 本 PR 验证 + 补齐) |
| 7 | attribution.phase_breakdown 段 collector |
| 8 | stop_reason ENUM 校验 (PolicyGate 写入路径) |
| 9 | GA 收尾文档 / CHANGELOG / migration guide |

## 8. 验收清单

- [ ] 一次完整 v0.8 session 跑通 PRELUDE → EXPLORE → KERNEL → SWEEP →
      CLOSE; phase_history 显示真 plateau reason 而非 proxy。
- [ ] 调小 `--plateau-explore-keep-gain` 后 EXPLORE 提前结束;
      phase_history 看到 plateau_explore reason。
- [ ] robustness emit `escalate_strategy_change{hint=skip_to_kernel}`
      后 EXPLORE 立即转 KERNEL, 写 evidence 标 llm_escalation。
- [ ] attribution.phase_breakdown 字段完整, gain 加和 ≈ cumulative_gain
      (允许 < 0.5% 浮点误差)。
- [ ] stop_reason 写入非词表值时 PolicyGate 拒, log 警告。
- [ ] resume v0.8 session, plateau 函数对历史 winners_history 计算正
      确, 不重复触发 plateau。
- [ ] CHANGELOG / migration guide / §3.15 cheatsheet 三处文档一致。

## 9. 风险与回退

主风险:

- **plateau 阈值默认值不合适** → 出现 EXPLORE 永远不退 (gain 永不达
  阈值) 或秒退. 缓解: 双向监控, 在灰度时观察 plateau 命中分布; flag
  默认值有保底 (0.5% / 3 streak)。
- **escalate hint 误用导致 phase 频繁跳**: 缓解: hint 词表闭合, 未识
  别 hint 不动 phase; phase_history 记录 evidence 便于复盘。
- **partial_k 模式下 round 边界混乱**: 由 Orchestration 控制 (它 propose
  explore 时 round_id 自增), 系统层不需要复杂逻辑。

回退:

- 切回 wait_all 模式; 调大 plateau 阈值; 关闭 escalate hint 识别。
- 极端: 把 EXPLORE phase budget % 改大 / KERNEL budget % 改小, 强制
  早转 SWEEP。

## 10. 哲学回引

本里程碑收尾 §3.8 phase 状态机的 plateau 真定义; **Inv-8.2 (退出条件
机器可判定)** 在 plateau pure function + escalate hint 识别 +
stop_reason ENUM 三处共同保证。**主轴 A (流程固化)** 至此完成全部落
地 — 决策完全交给 LLM, 系统只在 phase 转移与 plateau 判定上提供 *机
器化* 的边界。
