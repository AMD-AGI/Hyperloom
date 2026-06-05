# P2_13 — steward 降级 advisory + 删 depth_gate + 删 assess 节流;不再驱动 phase

- **Phase**: P2 · **风险**: 高 · **依赖**: 无 · **后继**: P3_17(再删 steward 内部 enqueue)

## 目标

Session Steward 是"代码替 LLM 做战略决定"最集中的一处:它的 `stop_session`/`advance_to_kernel` 经 `pending_escalate_hint` **直接驱动 phase 转换**,还用 `_apply_depth_gate_to_verdict` **改写 steward 自己的 LLM 判断**。本步把 steward 降为**纯 advisory**(只进 prompt,不驱动 phase),并删除 depth_gate 与 assess 节流。

> 注:本步保留 steward 的内部 enqueue(仍会产出一个第二意见 advisory)。**完全停用 steward enqueue 在 P3_17**(更激进)。

## 不变量 vs 策略判定

- "EXPLORE 何时算探够、是否前进 KERNEL" = **STRATEGY**。当前由 steward + depth_gate 决定并驱动 phase = 越界。降级。
- EXPLORE 不会无限探的**兜底是 IR-6 硬 force-exit + phase 预算耗尽硬墙**(保留),不需要 steward 驱动。

## 改动清单(删除优先)

### 1. steward 不再驱动 phase(`coordinator.py`)
- `_route_steward_verdict`(8704–8903):删除 8844–8863 的 `set_pending_escalate_hint('skip_to_sweep'/'skip_to_kernel')`。steward recommendation 改为**写入一个 advisory 字段**(如 `last_remaining_gaps_assessment`,已存在)仅供 prompt 展示。
- `continue_explore` 分支(8865–8887)中的副作用(reset 计数、`steward_continuation_used`)按需保留为中性,但不再有"continuation 上限"语义(见 phase_state 改动)。

### 2. 删 depth_gate 改写(`coordinator.py` + `phase_state.py`)
- 删 `_apply_depth_gate_to_verdict`(`coordinator.py` 8502–8564)整方法及其调用(8813–8816)。
- 删 `depth_gate`(`phase_state.py` 662–745)、`depth_tracker`/`depth_gate_enabled()`/`depth_snapshot()`(`shared_state.py` 883–899, 3106–3194)若仅服务该改写。
- CLI `--depth-gate` 及阈值(`cli.py` 458, 1318–1333, 3817, 5097–5112):删除或标记为 no-op(死旋钮,见 P3_22 一并清理)。

### 3. 删 assess_remaining_gaps 节流 deny(`coordinator.py`)
- 删 `_assess_remaining_gaps_throttle_denial`(5048–5137,reason: assess_remaining_gaps_phase / min_stack / throttle)。LLM 若想主动请求评估不再被节流;内部 steward enqueue 本就 bypass 此 deny(5053–5055)。

### 4. phase_state 的 steward 门改动(`phase_state.py`)
- `exit_normal_explore`(1041–1176)：删除 1110–1168 的 steward 路由(plateau→等 steward→按 recommendation 决定)。EXPLORE 退出只保留:IR-6 force-exit(1069–1081)、显式 hint `skip_to_kernel`/`skip_to_sweep`(1083–1097,来自 robustness 或 P3_18 的 LLM hint)、phase 预算耗尽(1169–1175)。plateau 本身在 P3_17 降级为 advisory;本步先断开 steward 对 phase 的驱动。
- `wants_steward_assessment`(1179–1249)：保留(仍可产出 advisory),但其结果不再门控 phase。

## 连带测试(大面积)

| 文件 | 函数 | 动作 |
|---|---|---|
| `test_assess_remaining_gaps.py` | `test_route_stop_session`(271)、`test_route_advance_to_kernel`(302) | **删除/反转**:steward 不再写 skip hint |
| `test_assess_remaining_gaps.py` | `test_exit_normal_explore_holds_when_steward_pending`(139)、`test_exit_normal_explore_routes_on_*`(157–181)、`test_exit_normal_explore_depth_gate_*`(197–210) | **删除**(steward 不再门控 phase) |
| `test_assess_remaining_gaps.py` | `test_assess_remaining_gaps_throttle_*`(490–519)、`test_assess_remaining_gaps_denied_*`(532–546) | **删除**(节流没了) |
| `test_assess_remaining_gaps.py` | `test_record_steward_assessment_*`(223–253)、`test_wants_steward_assessment_*`(87–127) | 保留(仍产 advisory) |
| `test_depth_gate.py` | 全文件 | **删除**(depth_gate 没了) |
| `test_phase_state_plateau.py` | `test_exit_normal_explore_skip_to_*`(283,408,418) | 保留(显式 hint 路径仍在);steward 来源的断言删除 |

## 验证
- steward verdict 出现在 orchestration prompt 作为 advisory,但**不**改变 phase。
- EXPLORE 退出只由 IR-6 / 预算硬墙 / 显式 hint 触发。
- depth_gate / assess 节流相关代码与 CLI 旋钮移除,无残留引用。
- 烟测:EXPLORE plateau 时 run 不卡死(靠硬墙推进),steward 建议可见但不强制。

## 回退
- 恢复 steward 驱动、depth_gate、节流及测试;较大,建议单独 PR。

## 残留风险
- **高**。去掉 steward 驱动后,EXPLORE→KERNEL 的推进更依赖 IR-6/预算墙与 LLM 显式 hint。必须确认:
  - IR-6 force-exit(`phase_state.py` 581–659)与 EXPLORE 预算耗尽(1169–1175)**保留且生效**,否则 EXPLORE 可能拖到 deadline。
  - robustness 的 `escalate_strategy_change{skip_to_kernel}` 路径(P3_19 降级前)仍可推进 phase。
- 建议与 P3_17 一起规划:本步先断 steward 驱动,P3_17 决定是否彻底停用 steward enqueue。
