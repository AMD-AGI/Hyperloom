# P3_18 — 打破线性 phase:EXPLORE↔KERNEL 交错 + Orchestration 可显式请求前进

- **Phase**: P3 · **风险**: 最高 · **依赖**: P1_01, P3_17 · **后继**: 收尾回扫 P1_09

## 目标

当前 phase 严格单向 `PRELUDE→FRAMEWORK_PR→EXPLORE→KERNEL→SWEEP→CLOSE`,且 Orchestration **不能决定切 phase**(只能经 robustness 间接 escalate)。这把"kernel 收益解锁新 config 空间、反之亦然"的真实优化回路切断了。本步在**保留可恢复性/产物契约**的前提下,提供两种放权(可分别或组合落地):

- **18A — LLM 显式前进**:让 Orchestration 能直接发"请求前进/回退 phase"的 hint(不再只能借道 robustness)。
- **18B — EXPLORE↔KERNEL 交错**:放宽 `PHASE_LLM_PROPOSABLE_ACTIONS`,允许 EXPLORE 内发 kernel request、KERNEL 内发 explore,实现交错优化。

> 用 env/flag `INFERENCE_OPTIMIZER_PHASE_INTERLEAVE`(默认 off)包裹 18B,验证后再翻默认。

## 不变量 vs 策略判定

- phase **单调拓扑 + 终止路由 + CLOSE 顺序器 + 产物契约** = INVARIANT(保留)。"严格不可交错 + LLM 不可影响前进" = STRATEGY(放权)。

## 改动清单

### 18A — Orchestration 显式前进 hint
- `agent_role.py`:给 orchestration 增加发 `escalate_strategy_change`(或新增轻量 `request_phase_advance`)的能力。当前 `escalate_strategy_change` 是 robustness-only(`_ROBUSTNESS_INTENTS`);PRUNE_BRANCH 已先例给了 orchestration。
- `policy.py` `_validate_robustness_only`(2550–2568):把 `ESCALATE_STRATEGY_CHANGE`(或新 intent)的 source allowlist 扩展含 orchestration(类似 PRUNE_BRANCH 在 473–475 的处理)。**保留** FORCE_DISPATCH robustness-only。
- `coordinator.py` `_handle_escalate_strategy_change`(9273–9351):接受 orchestration 来源,映射 `skip_to_kernel`/`skip_to_sweep`/`skip_to_close` 到 `set_pending_escalate_hint`。**保留**对 hint vocab 的校验(`is_valid_escalate_hint`)与预算 bump 上限(防滥用)。
- prompt(`orchestration.md` 16–22 / `prompt_builder` 460):把"你不能决定切 phase"改为"你可发 `escalate_strategy_change{hint=...}` 建议前进"。

### 18B — EXPLORE↔KERNEL 交错(flag 包裹)
- `phase_state.py` `PHASE_LLM_PROPOSABLE_ACTIONS`(P1_01 新增):当 `INTERLEAVE` 开启:
  - EXPLORE 集合并入 kernel-owned 的可 request kinds(`trace_analyze`/`run_optimization`/`integrate`/`run_gemm_tuning`)。
  - KERNEL 集合并入 `explore`/`specialist`/`integrate_patch`。
- `policy.py` `_validate_phase_action`:对交错放行(R1 基于扩展后的集合)。
- `coordinator.py`:`_on_enter_kernel` / `_on_enter_explore` 的自动行为(GEMM 自动跑等)在交错模式下保持幂等(可重入)。
- **保留**:kernel-owned 仍只能经 `request{target_agent='kernel'}`(角色契约);trace_analyze→run_optimization 数据依赖(P2_11 的 handler 校验);integrate_patch Critic 门;sweep 单例。

### 不变量保护
- phase 仍**单调**记录用于 resume/审计;交错只放宽"某 phase 内可提的动作集",不改 phase 链拓扑。SWEEP/CLOSE 仍是终点,终止态路由不变。

## 连带测试

| 文件 | 动作 |
|---|---|
| `test_phase_state_machine.py` | 新增交错模式下的 allowed-set 断言;保留非交错(默认)行为 |
| `test_agent_roles_and_policy.py` / policy robustness-only 测试 | orchestration 可发 escalate 的新断言;FORCE_DISPATCH 仍 robustness-only |
| `test_phase_state_plateau.py` | 显式 hint 推进路径(skip_to_kernel/sweep)断言更新 |
| `test_delegate_denial_loop.py` / `test_prompt_*` | 交错模式下 phase allowlist 渲染 |
| 大量 phase 相关测试 | 需在 flag off(默认)下保持原行为绿;flag on 下新增用例 |

## 验证
- 18A:Orchestration 发 `escalate_strategy_change{skip_to_kernel}` 能推进 phase(经保留的 hint 校验)。
- 18B(flag on):EXPLORE 内可发 kernel request 并回到 explore,KERNEL 内可发 explore;角色/数据/Critic/sweep 不变量全保留。
- 默认(flag off):全部既有 phase 测试绿。
- **A/B**:交错 vs 线性,长预算下收敛性与 tok/s 对比;确认产物契约、resume、stop_reason 顺序不破。

## 回退
- 18B 由 flag 包裹,翻默认即回退;18A 恢复 robustness-only 与 prompt 措辞。

## 残留风险
- **最高**。交错放大了搜索空间与调度复杂度。务必:
  - 默认关闭 18B,先在受控 workload A/B;
  - 复核 `_on_enter_*` 自动逻辑在重入下幂等(避免重复自动 GEMM/sweep);
  - 确认 resume 在交错历史下仍能正确恢复(phase 单调记录保留是关键)。
- 本步落地后回扫 P1_09,使 prompt 完整反映新的 phase 自由度。
