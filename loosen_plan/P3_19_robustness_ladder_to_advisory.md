# P3_19 — robustness ActionLadder 自动 escalate/prune → alert+hint;kill 仅留资源

- **Phase**: P3 · **风险**: 高 · **依赖**: P3_17(plateau advisory)建议先行 · **后继**: 无

## 目标

`robustness-agent/` 是一个**确定性子进程 `ActionLadder`**:在 HIGH 症状下**自动发** `escalate_strategy_change` / `prune_branch` / `kill_task`,直接覆盖 Orchestration 的策略。这是"代码替 LLM 决策"在主 plan 之外最大的一处。本步把策略性自动动作降级为 **alert + advisory hint**,只保留**资源/安全**性质的自动 kill。

## 不变量 vs 策略判定

| 项 | 位置 | 当前 | 性质 | 处理 |
|---|---|---|---|---|
| 症状 cooldown(5 tick 抑制重复) | `decision/action_ladder.py` 66–171 | FORCE | INVARIANT-ish(防 inbox 洪水) | 保留 |
| HIGH 症状自动 escalate/prune/kill | `decision/action_ladder.py` 192–343 | FORCE | STRATEGY(prune/escalate)/ INVARIANT(资源 kill) | **拆分**:prune/escalate → alert+hint;kill 仅留 stale-lease/资源 |
| `gain_plateau`(6-tick,ε=0.5%)→ escalate | `signals/progress.py` 51–187 | FORCE | STRATEGY | 降级为 medium `alert` |
| `no_levers_found`(45min/8 tick)→ delegate(report) | `signals/progress.py` 189–243 | FORCE | STRATEGY | 降级 advisory;删强制 report 路径 |
| aiter_jit stale-build → escalate skip-baseline | `signals/aiter_jit.py` 83–183 | FORCE | STRATEGY | 降级 alert |
| kernel_pipeline 自动 `prune_branch(kernel_opt)` | `signals/kernel_pipeline.py` 172–381 | FORCE | STRATEGY | alert only |
| preflight 自动 prune on feasibility | `signals/preflight.py` 496 | FORCE | STRATEGY | advisory feasibility |
| cooldown/streak 阈值 | `config.py` 97, 167–172, 307–308 | FORCE | STRATEGY | 调高/暴露 operator |
| 30s 子进程 timeout/tick | `backends/robustness_agent.py` 53–54, 176–177 | FORCE | INVARIANT | 保留 |
| coordinator prune→pruned_families + cancel | `coordinator.py` 6273, 9252–9262 | DENY | STRATEGY(源自 robustness prune) | prune 改 advisory flag |
| coordinator escalate hint → 改 phase/预算 | `coordinator.py` 9273–9351 | MUTATE | 混合 | 保留 hint 校验/上限;phase-skip 默认仅 advisory(除非操作者确认) |

## 改动清单(降级优先;robustness 是独立包)

### 1. ActionLadder 拆分策略 vs 资源(`robustness-agent/src/robustness_agent/decision/action_ladder.py`)
- 192–343:HIGH 症状不再自动 `escalate_strategy_change` / `prune_branch`;改发 `alert`(带建议 hint 文本)让 Orchestration/操作者决定。
- **保留** `kill_task` 仅用于**资源/安全**触发(stale lease、runaway 子进程、卡死任务),不用于策略性 prune。
- 保留 cooldown(66–171)防洪水。

### 2. 信号降级(`robustness-agent/src/robustness_agent/signals/`)
- `progress.py`:`gain_plateau` → medium `alert`(不 escalate);`no_levers_found` → advisory(删强制 `delegate(report)`)。
- `aiter_jit.py` / `kernel_pipeline.py` / `preflight.py`:自动 prune/escalate → `alert` / advisory feasibility。

### 3. coordinator 侧联动(`coordinator.py`)
- `prune_branch` → `pruned_families` 的 **DENY 派发**(6273)与 cancel(9252–9262):由于 robustness 不再自动 prune,这里主要剩 orchestration 主动 prune(P0 已给 orchestration PRUNE_BRANCH)。把"family 被 prune 后硬挡 dispatch"降级为 advisory flag(LLM 可重新选择),或仅在**操作者/显式**prune 时硬挡。
- `_handle_escalate_strategy_change`(9273–9351):**保留** hint vocab 校验与预算 bump 上限(防滥用);robustness 来源减少后,主要服务 P3_18 的 orchestration hint。

### 4. prompt 对齐(`system_prompts/robustness.md`)
- robustness.md(24–34, 60–74,legacy prose)与实际子进程 ladder 行为对齐:robustness 现在主要"观察 + alert + 建议",不再自动改 phase/prune。

## 连带测试

| 文件 | 动作 |
|---|---|
| `robustness-agent/tests/test_signals_progress.py`(`test_plateau_*` 58–130) | 改写:`gain_plateau`/`no_levers` 产出 alert/advisory 而非 escalate/forced report |
| `robustness-agent/tests/`(action ladder / kernel_pipeline / preflight / aiter_jit) | 自动 prune/escalate 用例改为 alert 用例 |
| coordinator prune 测试(`pruned_families` / dispatch block) | prune 改 advisory 后更新 |
| `test_robustness_storm_and_mix.py`(storm 部分) | 复核 cooldown/storm 行为不变 |

## 验证
- HIGH 症状下 robustness 产 alert + 建议,**不**自动改 phase / prune / kill(策略性)。
- 资源性 kill(stale lease / runaway)仍自动执行。
- Orchestration 收到 alert 后可自行决定(配合 P3_18 的显式前进/prune 能力)。
- 烟测:制造一个 plateau/no-lever 场景,确认 run 不被 robustness 强制改道,但 alert 可见。

## 回退
- 恢复 ladder 自动动作与信号 escalate、coordinator 硬 prune;robustness 独立包,单独 revert。

## 残留风险
- **高**。robustness 原本是"卡死/失控"的自动救援。降级后:
  - 真·卡死 → 资源 kill(保留)+ 子进程 timeout(保留)+ IR-6/预算硬墙(保留)兜底。
  - 策略性停滞 → 改由 LLM 看 alert 自行决断(放权目标)。
  - 建议保留一个 operator 可配置开关:在无人值守长跑中,可选择性恢复某些自动救援(如 `no_levers_found` 强制 report),以防长跑挂死。
