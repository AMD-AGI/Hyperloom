# P1_06 — 删 `explore_attempts_minimum_before_kernel_opt` + Web 工具 phase 限制

- **Phase**: P1 · **风险**: 低 · **依赖**: 无 · **后继**: 无

## 目标

删除两条纯策略性 deny:
1. `explore_attempts_minimum_before_kernel_opt`(`policy.py` 1075–1082):KERNEL phase 里若没有任何被接受的 explore round,就拒绝 `kernel_opt`。强迫"先 explore 再 kernel"。
2. `tool_whitelist_phase`(`policy.py` 1460–1473):WebSearch/WebFetch 仅 EXPLORE 可用,其它 phase 拒。

## 不变量 vs 策略判定

- "便宜的 explore 必须先于昂贵的 kernel_opt" = **STRATEGY**(成本判断,应归 LLM)。删除。
- "研究类 Web 工具只能在 EXPLORE" = **STRATEGY**(phase 适用性判断)。删除——研究在 KERNEL/SWEEP 也可能有用。**保留** `tool_whitelist_role`(1475–1485,specialist-only 工具的角色隔离 = INVARIANT)。

## 改动清单(删除优先)

### 1. 删 kernel_opt 最小门(`policy.py`)
- 删 `_validate_explore_minimum_before_kernel_opt`(1075–1082)及其在 delegate/propose/request 校验链中的调用。
- 无对应常量需清理(逻辑内联)。

### 2. 删 Web 工具 phase 限制(`policy.py`)
- 删 `_validate_tool_whitelist_collision` 中 phase 部分(1460–1473,`tool_whitelist_phase`)。**保留** role 部分(1475–1485,`tool_whitelist_role`)。

## 连带测试

| 文件 | 函数 | 动作 |
|---|---|---|
| (#8 无测试) | — | `explore_attempts_minimum_before_kernel_opt` 无测试,无需改 |
| `test_policy_gate_evolution.py` | `test_phase_restricted_tools_only_target_web_tools`(105)、`test_specialist_web_tools_denied_outside_explore_phase`(169)、`test_validate_tool_invocation_phase_override_argument`(187) | **删除/反转**:Web 工具不再被 phase 限制 |
| `test_policy_gate_evolution.py` | `test_specialist_pr_monitor_allowed_in_any_phase`(178) | 保留(PR 工具本就不限) |

## 验证
- KERNEL phase 可直接提 `kernel_opt`(无需先 explore),其余安全门(trace_analyze 数据依赖见 P2_11、kernel-owned 经 request)仍在。
- specialist 在任意 phase 可用 Web 工具;主 agent 仍不能直接用 specialist-only 工具(role 隔离保留)。

## 回退
- 恢复两个校验块与测试。

## 残留风险
- 低。删 kernel_opt 最小门后,模型理论上可在 KERNEL 直奔 kernel_opt;但 `trace_analyze` → `run_optimization` 的**数据依赖**(P2_11 保留)与 reusable kernel id 校验仍保证 kernel_opt 输入有效。
