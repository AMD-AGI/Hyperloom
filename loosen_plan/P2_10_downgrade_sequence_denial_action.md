# P2_10 — `_sequence_denial_for_action`:保 baseline 门,删其余顺序 deny

- **Phase**: P2 · **风险**: 高 · **依赖**: P2_14(stack_rebench 删除前需 validated 标注) · **后继**: 无

## 目标

`_sequence_denial_for_action`(`coordinator.py` 5139–5313)把多条"下一步该做什么"的顺序判断变成了硬 **DENY**。除 baseline 门(不变量)外,其余都是 STRATEGY,删除。

## 逐子规则判定

| 子规则(行) | 当前 deny | 性质 | 处理 |
|---|---|---|---|
| `target_analysis` 必须最先(5159–) | execution_order | STRATEGY(且廉价) | **删除** |
| `baseline` 先于一切(除 target_analysis) | execution_order | **INVARIANT**(无 baseline 无意义) | **保留** |
| baseline 自环(委托 `_baseline_self_loop_denial`) | baseline_self_loop | INVARIANT(去重/反 loop) | **保留**(见 P0 保留清单) |
| `last_profile_trace` required(除 profile 外都挡) | execution_order | STRATEGY(何时重测应归 LLM) | **删除** |
| KEEP 后强制 `integrate`(只放行 integrate/report) | execution_order | STRATEGY | **删除**(改 advisory 事实:有未集成 KEEP) |
| `report` 被未跑热 kernel 挡(`hot_kernel_unfinished`) | hot_kernel_unfinished | STRATEGY(质量门) | **删除**(改为 report 内标注未尝试 kernel,见下) |
| `stack_rebench_required`(只放行 explore/baseline/report) | stack_rebench_required | STRATEGY(关联产物诚实性) | **删除**,改由 P2_14 的 `validated=false` 标注承担诚实性 |

## 改动清单(删除优先)

### 1. `coordinator.py` `_sequence_denial_for_action`(5139–5313)
- **只保留** baseline-first(无 target_analysis 时仍要求 baseline)与对 `_baseline_self_loop_denial` 的委托。
- 删除:target_analysis-first、last_profile_trace-required、KEEP-forces-integrate、hot_kernel_unfinished、stack_rebench_required 五段。
- 调用点(`_handle_propose_action` 5528–5533、`_handle_delegate` 6283–6291)保留对精简后函数的调用。

### 2. report 质量信号改 advisory(触类旁通)
- 删 hot_kernel_unfinished 后,为不损失"还有热 kernel 没试"的信息:在 `action_executors/report.py` 渲染时**标注** `untried_hot_reusable_kernels`(以及 `pending_keep_kernels`)为报告中的一节,而非用 deny 阻止 report。让 LLM/操作者看到"报告时仍有 N 个未尝试 reusable kernel"。

### 3. 与 P2_14 协同
- stack_rebench deny 删除后,未验证 KEEP 的诚实性由 `session_breakdown` / SharedState 的 `validated=false` 标注承担(P2_14)。**本步必须在 P2_14 之后或同 PR 落地**,否则下游可能把未验证 KEEP 当作已验证。

## 连带测试

| 文件 | 函数 | 动作 |
|---|---|---|
| `test_required_step_gates.py` | `test_target_analysis_denial_blocks_baseline_*`(179–205) | target_analysis 门保留还是删?本步删 target_analysis-first → 删除这些 |
| `test_required_step_gates.py` | `test_integrate_denial_blocks_explore_*`(311) | 删除(integrate 不再硬挡) |
| `test_required_step_gates.py` | `test_report_denied_when_hot_reusable_kernels_untried`(345)、`test_report_allowed_after_*`(377)、`test_report_gate_*`(467–502) | 删除/反转为"report 不再被 hot kernel 挡;报告含未尝试 kernel 节" |
| `test_required_step_gates.py` | `test_trace_analyze_gate_does_not_block_explore_actions`(525) | 保留(本就允许) |
| `test_baseline_self_loop_policy.py` | 全部(100–248) | **保留**(baseline 门 + 自环不变量未动) |
| `test_coordinator_runtime.py` | `test_execution_order_denies_backends_before_profile`(556) | **删除/反转**:explore 不再被 last_profile_trace 挡 |
| `test_coordinator_runtime.py` | `test_execution_order_does_not_deny_backends_when_trace_analyze_stale`(590) | 保留(更强:本就不挡) |
| (`stack_rebench_required` rule 无专门测试) | — | 无需改 |

## 验证
- baseline 仍必须先行(门保留);其余动作不再因顺序被拒。
- report 可在有未跑热 kernel/未验证 KEEP 时发出,报告里如实标注。
- 黄金契约:`session_breakdown.json` 形状不变;未验证 KEEP 标 `validated=false`(P2_14)。
- 烟测 + 跑最小测试集。

## 回退
- 恢复被删五段子规则与测试。

## 残留风险
- **高**。这是放权的核心一步。风险点:
  - 未验证增益污染下游 → 由 P2_14 兜底(必须先做)。
  - kernel KEEP 未集成就 report → 报告标注 + KERNEL phase 预算/plateau(P3_17)仍在;且 integrate 仍是 LLM 可选动作。
  - 建议单独成 PR,配 A/B 验证收敛性不退化。
