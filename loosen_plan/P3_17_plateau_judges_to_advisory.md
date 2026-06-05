# P3_17 — plateau 判据全降级 advisory + 删 steward 内部 enqueue

- **Phase**: P3 · **风险**: 高 · **依赖**: P2_13 · **后继**: P3_18

## 目标

把所有 plateau 判据(EXPLORE/KERNEL/FRAMEWORK_PR + GEMM 捷径)从"自动驱动 phase 转换"降级为**纯 advisory 事实**(注入 prompt:"系统认为已 plateau")。phase 前进只由**硬墙**(IR-6 force-exit / phase 预算耗尽 / 终止态)+ **显式 hint**(robustness escalate,或 P3_18 的 LLM hint)触发。同时**彻底停用 steward 内部 enqueue**(P2_13 的激进版收尾)。

## 不变量 vs 策略判定

| 项 | 位置 | 性质 | 处理 |
|---|---|---|---|
| `compute_plateau_explore`(AND 低增益+空轮) | `phase_state.py` 749–850 | STRATEGY | 计算保留,**仅作 advisory**,不驱动 phase |
| `compute_plateau_kernel`(OR revert/低增益) | `phase_state.py` 853–951 | STRATEGY | 同上 |
| FRAMEWORK_PR plateau(3 批<1%) | `phase_state.py` 1489–1525 | STRATEGY | 同上;force-exit ratio 0.6 保留为软预算(或 P3_22 调) |
| GEMM 完成即判 plateau_kernel 捷径 | `phase_state.py` 1290–1302 | STRATEGY | **删除**(GEMM 完成只是事实) |
| `_maybe_enqueue_steward` | `coordinator.py` 3849–3869, 847–851 | STRATEGY | **删除**(不再内部派 steward) |
| `_enqueue_internal_steward_task` | `coordinator.py` 3553–3641 | STRATEGY | 删除(若 steward 不再用) |
| IR-6 force-exit / phase 预算耗尽 / 终止态 | `phase_state.py` 581–659, 1169–1175 等 | INVARIANT | **保留**(唯一硬兜底) |

## 改动清单(删除优先)

### 1. plateau 不再驱动 phase(`phase_state.py`)
- `exit_normal_explore`(1041–1176):删除 plateau 触发即返回 `plateau_explore` 的路径(1103–1168 已在 P2_13 删 steward 部分;此处删 plateau→exit)。只保留:IR-6 force-exit、显式 hint、phase 预算耗尽。
- `exit_normal_kernel`(1252–1323):删除 `compute_plateau_kernel` 驱动退出(1303–1314)与 GEMM 捷径(1290–1302)。只保留显式 hint + phase 预算耗尽。
- `exit_normal_framework_pr`(1427–1534):plateau(1489–1525)降级——不再 plateau 即退;保留 force-exit ratio(软预算)+ `framework_pr_phase_done`(完成信号,INVARIANT)。

### 2. plateau 作为 advisory 注入(`coordinator.py` / `shared_state.py`)
- 仍调用 `compute_plateau_*` 计算结果,但只**写入 advisory 字段**并注入 orchestration prompt(中性事实:"explore plateau detected: low gain + N empty rounds"),不改 phase。

### 3. 删 steward 内部 enqueue(`coordinator.py`)
- 删 `_maybe_enqueue_steward`(3849–3869)与其调用(847–851)。
- 删 `_enqueue_internal_steward_task`(3553–3641)。
- `session_steward_specialist` domain、`assess_remaining_gaps` action:若不再有任何派发路径,清理(domain 目录 `specialist_domains.py` 142–154、action meta、`phase_state.py` 113 的 EXPLORE allowlist 项)。
- `wants_steward_assessment`(`phase_state.py` 1179–1249):删除(无消费者)。

## 连带测试(大面积)

| 文件 | 函数 | 动作 |
|---|---|---|
| `test_phase_state_plateau.py` | `test_plateau_explore_*`(81–140)、`test_plateau_kernel_*`(162–237) | `compute_plateau_*` 计算用例**保留**(函数还在);但 `test_exit_normal_explore_triggers_via_real_plateau`(254)、`test_exit_normal_kernel_triggers_via_real_plateau`(301)、`test_exit_normal_kernel_after_gemm_*`(321) **删除/反转**(plateau 不再驱动) |
| `test_phase_state_plateau.py` | `test_compute_next_phase_honors_plateau_overrides`(538) | 改写(overrides 影响 advisory 计算,不影响 phase) |
| `test_phase_state_framework_pr.py` | `test_exit_normal_framework_pr_plateau_*`(114–296) | 删除/反转 plateau 驱动用例;保留 force-exit/phase_done |
| `test_assess_remaining_gaps.py` | 整体(steward) | **删除**(steward 没了);保留与其它功能无关的部分 |
| `test_decision_framework.py` | `test_kernel_entry_*gemm*`(218–268) | GEMM 自动跑保留(`_on_enter_kernel`),但 GEMM→plateau 捷径删除 |
| `test_sweep_phase_auto.py` / `test_close_phase_sequencer.py` | phase 推进相关 | 复核:确认无依赖 plateau 驱动的断言失效 |

## 验证
- plateau 出现在 prompt 作 advisory,phase 不因 plateau 自动前进。
- EXPLORE/KERNEL 在预算内由硬墙推进;deadline 时仍走 closing→自动 report(CLOSE 顺序器保留)。
- steward 相关代码/domain/action 清理干净,无残留引用。
- **A/B 必做**:同 workload,新(plateau advisory)vs 旧(plateau 驱动)对比——确认收敛性不退化、不超预算、产物形状不变。

## 回退
- 恢复 plateau 驱动、GEMM 捷径、steward enqueue 与测试。强烈建议单独 PR + feature 验证。

## 残留风险
- **高**。这是最接近"把 phase 节奏交给 LLM"的一步。核心担忧与兜底:
  - EXPLORE/KERNEL 拖到 deadline → IR-6 / phase 预算耗尽硬墙(保留)强制收尾;deadline→closing→report(保留)。
  - 没有 steward 第二意见 → plateau advisory + robustness escalate(P3_19 前仍可推进)兜底。
  - 与 P3_18 配合时,LLM 显式前进 hint 成为主要推进手段——两步需协同验证。
