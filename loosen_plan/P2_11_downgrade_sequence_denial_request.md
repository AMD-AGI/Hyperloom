# P2_11 — `_sequence_denial_for_request`:保数据依赖,删其余;依赖改 handler 输入校验

- **Phase**: P2 · **风险**: 中 · **依赖**: 无 · **后继**: 无

## 目标

`_sequence_denial_for_request`(`coordinator.py` 5315–5368)对 orchestration→kernel 的 request 做顺序 DENY。保留**真数据依赖**,删除策略性顺序。

## 逐子规则判定

| 子规则(行) | 当前 deny | 性质 | 处理 |
|---|---|---|---|
| `baseline` 先于 kernel request(trace_analyze 直通除外,5328–5329) | execution_order | INVARIANT(需基准) | **保留** |
| `last_profile_trace` 先于 kernel request | execution_order | STRATEGY(何时重测归 LLM) | **删除** |
| `trace_analyze` 先于其它 kernel kind(GEMM 除外 5356) | execution_order | **数据依赖**(run_optimization 消费 trace_analyze 的 candidates_path / reusable ids) | **保留为数据契约**,但改实现(见下) |
| `run_gemm_tuning` 先于 `run_optimization`(FP8) | execution_order | STRATEGY | **删除** |

## 改动清单(删除优先 + 依赖改实现)

### 1. `coordinator.py` `_sequence_denial_for_request`(5315–5368)
- **保留** baseline-before-kernel。
- **删除** last_profile_trace-before-kernel、run_gemm_tuning-before-run_optimization。
- trace_analyze→run_optimization:不再用**前置硬 deny**,改为**handler 输入校验**(见 2),让边界"所见即所得"(失败发生在 handler,带明确数据错误,而非提前在 policy 层拦)。

### 2. kernel handler 输入校验(触类旁通,`kernel_request_handlers.py`)
- `run_optimization` handler 在执行前校验 `candidates_path` / `kernel_id` 是否来自有效的 `last_trace_analyze`;缺失则**优雅失败并返回明确错误**(data error),而不是被 coordinator 提前 deny。这保留了数据正确性(INVARIANT),同时把"先 trace_analyze"变成可被 LLM 理解的数据契约而非顺序策略。
- 已有的 reusable kernel id 校验(拒绝 `non_reusable_kernel` / 操作名)保留。

## 连带测试

| 文件 | 函数 | 动作 |
|---|---|---|
| `test_required_step_gates.py` | `test_trace_analyze_gate_still_blocks_run_optimization_request`(597) | 改写:从"policy 前置 deny"改为"handler 输入校验失败"(若改实现);或保留 deny 但记为数据契约——二选一 |
| `test_required_step_gates.py` | `test_trace_analyze_request_itself_passes`(614)、`test_trace_analyze_gate_clears_run_opt_request_when_cache_fresh`(652) | 相应调整 |
| `test_decision_framework.py` | `test_run_optimization_denied_until_fp8_gemm_tuning_terminal`(194) | **删除/反转**:gemm 不再前置硬挡 run_optimization |
| (last_profile_trace-before-kernel) | — | 删除其断言(若有,多在 required_step_gates / coordinator_runtime) |

## 验证
- kernel request 不再因 last_profile_trace / gemm 顺序被 policy 拦;baseline 门保留。
- `run_optimization` 缺有效 candidates 时在 handler 优雅失败(明确数据错误),不崩溃、可恢复。
- 烟测:FP8 SGLang 路径上,不先 gemm 直接 run_optimization 时行为合理(handler 校验或正常跑)。

## 回退
- 恢复被删子规则与 handler 改动、测试。

## 残留风险
- 中。把"先 trace_analyze"从 policy 前置改到 handler 后,需确保 handler 错误被正确记入 `last_action_failures` 并可被 LLM 看到(否则 LLM 不知为何失败)。删 gemm-before-run_optimization 后,FP8 GEMM 调优变为 LLM 可选(KERNEL phase 仍可自动跑 gemm,见 `_on_enter_kernel`)。
