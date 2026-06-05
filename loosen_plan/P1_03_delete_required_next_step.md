# P1_03 — 删除 `_required_next_step` 执行清单

- **Phase**: P1 · **风险**: 低 · **依赖**: 无 · **后继**: 无

## 目标

删除 Coordinator 替 LLM 排好的"下一步必须做 X"清单。`_required_next_step`(`coordinator.py` 4808–4973+)计算一份命令式 TODO(target_analysis→baseline→等 analysis→trace_analyze→GEMM→integrate→未跑热 kernel→stack rebench),注入为 `=== Execution checklist (Coordinator-enforced) ===`(4612–4615)。这是"代替 LLM 思考"的典型——这些结论本应由 LLM 读 SharedState 自行得出。

## 不变量 vs 策略判定

- `_required_next_step` 是 **ADVISE-only**(不 deny),但措辞命令式且替模型决定优先级 = **STRATEGY**。
- 它列出的"事实"(有未集成 KEEP、stack 有未验证 KEEP、有未跑热 kernel)**已存在于 SharedState** 的 `last_*` / `gaps` / `pending_keep_kernels` / `untried_hot_reusable_kernels` 字段中,LLM 已能看到。→ 清单是冗余的"代思考"层,**删除**。

## 改动清单(删除优先)

### 1. 删除方法与注入(`coordinator.py`)
- 删除 `_required_next_step`(4808–4973+)整个方法。
- 删除注入点 4612–4615(`=== Execution checklist (Coordinator-enforced) ===` 段)。

### 2. 确认事实在 SharedState 可见(触类旁通,不新增"代思考")
- 核对 `to_prompt_summary` / `to_mission_summary` / `to_gaps_summary` 已覆盖:`pending_keep_kernels`、`untried_hot_reusable_kernels`、未验证 KEEP 标记、`last_trace_analyze` 缓存状态。
  - 若 `untried_hot_reusable_kernels` / `pending_keep_kernels` 当前**仅**通过 `_required_next_step` 暴露,则把它们作为**中性事实**加入 `to_mission_summary`(客观列出,不带"你应该做 X"),避免删清单后 LLM 看不到这些事实。
- 不引入任何"建议下一步"的措辞。

## 连带测试

| 文件 | 函数 | 动作 |
|---|---|---|
| `test_required_step_gates.py` | 全文件(`test_trace_analyze_cache_*`、`test_integrate_gate_*`、`test_required_next_step_surfaces_untried_hot_kernels`、`test_analyze_gate_*` 等) | **删除** `_required_next_step` 相关用例;其中**纯事实可见性**(如 untried hot kernels 出现在 prompt)若改由 `to_mission_summary` 承担,则改测 mission summary |
| `test_no_llm_propose_profile_hints.py` | `test_required_next_step_does_not_tell_llm_to_propose_profile`(84) | 删除(方法没了);保留 `test_sequence_denial_*`(那些属 P2_10/11) |
| `test_coordinator_runtime.py` | `test_execution_checklist_is_in_orchestration_prompt`(629) | **删除或反转**为"prompt 不含 Execution checklist 段" |
| `conftest.py`(17,41) | `_required_next_step` 相关 fixture | 清理 |

## 验证
- orchestration prompt 不再含 `Execution checklist` 段;关键事实(未集成 KEEP / 未跑热 kernel / stack 未验证)仍可在 SharedState 摘要中看到。
- 烟测:一次完整 run,确认 LLM 在没有清单的情况下仍能正确推进(baseline→explore→…)。

## 回退
- 恢复方法与注入块及相关测试。

## 残留风险
- 低。注意 `_required_next_step` 内含的"事实计算"(如 untried hot kernels 排序)若有别处复用,需迁移为中性 helper。其"硬强制"由 P2_10/11 的序列守卫单独处理,本步只删 advisory 清单,不动 deny 逻辑。
