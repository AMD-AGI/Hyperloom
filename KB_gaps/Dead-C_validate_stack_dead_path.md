# Dead-C — validate_stack 死路径

> 风险等级: **HIGH-misleading** (8 处跨文件矛盾)
> 体检报告: `../KB_design_gaps.MD` §12.4
> 关联功能 gap: `Gap-10` (legacy actions still allowed)

## 1. 问题描述

KB_design §3.4 / M3 明确: `validate_stack` 内嵌进 `explore` 的 KEEP-后-
stack-rebench, 不再作独立 action.

实际: 8 处跨文件 (yaml + executor + cli + phase allowlist + prompt
mandatory + orchestration.md mandatory + coordinator TODO 4 + 测试)
仍把 validate_stack 当**强制 action**.

7 大类废止物中**矛盾最大**: prompt 强制 validate_stack, coordinator 等
LLM propose validate_stack 才推进; 同时 KB_design 说"已内嵌". LLM 看
代码会发现"内嵌"根本没发生.

## 2. 详细位置清单 (8 处)

| 件 | 路径 | 说什么 |
|---|---|---|
| KB_design §3.4 / M3 | `KB_design/3.4_explore_consolidation/README.md` | validate_stack 已内嵌 explore |
| yaml | `actions/_meta/validate_stack.yaml` | 仍存在 |
| executor | `action_executors/validate_stack.py` | 仍存在 |
| cli 注册 | `cli.py:639` | `validate_stack_executor` 注册 |
| phase 允许 | `phase_state.py:96` | EXPLORE 仍允许 |
| **prompt MANDATORY** | `prompt_builder.py:336-339` | "MANDATORY `validate_stack` after KEEP" |
| **decision rule** | `prompt_builder.py:479-481` | 决策第 3 步 `validate_stack required` TODO |
| orchestration.md | `:59, 100-103` | 同样 mandatory |
| coordinator gate | `coordinator.py:1586-1660, 1852-1868` | TODO 4 enforce |
| shared_state field | `shared_state.py:~238` `last_validate_stack` | 仍读写 |
| shared_state helper | `shared_state.py:~2383+` `optimization_stack_has_unvalidated_keeps` | 驱动 TODO |
| 测试 | `tests/test_validate_stack.py` | 全文锁住 |
| 测试 | `tests/test_phase2_mission_and_validate_stack.py` | 锁住 prompt |
| agent-facing doc | `actions/validate_stack.md` | 全文 |

8 处矛盾, 等价于"validate_stack 在 prompt + coordinator 仍是 first-class
action, KB_design 设计悬空".

## 3. 实际跑起来发生什么

LLM (orchestration) 跑 EXPLORE round:

1. prompt 第 3 步说 "若 `optimization_stack_has_unvalidated_keeps` →
   MUST propose validate_stack first"
2. LLM 读 SharedState, 看到 KEEP 已入 stack 但未 validate
3. LLM emit `propose_action='validate_stack'`
4. PolicyGate R1 通过 (EXPLORE allowlist 仍含)
5. coordinator 派 `validate_stack_executor` → 跑独立 validate_stack
   bench (不是 explore inline rebench)
6. 跑完写 `state.last_validate_stack`
7. coordinator TODO 4 解锁; LLM 才能继续 explore

完全绕过 explore inline rebench 的设计.

## 4. 设计意图

§3.4 / M3:

> validate_stack 内嵌进 explore 的 KEEP-后-stack-rebench. explore
> executor 在每个 variant KEEP 后立即重跑 stack (含当前所有 KEEP'd
> variants), 检测是否 keep_unstable_in_stack. 不再需要独立 validate_stack
> action.

设计目的:
- 减少 prompt 复杂度 (LLM 不用决策何时 validate)
- KEEP-rebench loop 在 explore executor 内串行, 避免 phase 内 round-trip
- breakdown 中 `keep_unstable_in_stack` 计数 = explore round 内, 不是
  独立 action

## 5. 根本原因

M3 PR 实施时, *executor 内嵌* (explore.py 加 stack rebench 逻辑) 已经
完成, **但**:
- prompt mandatory 段没改 (prompt_builder.py)
- coordinator TODO 4 没改 (`_required_next_step`)
- orchestration.md mandatory 段没改
- 测试锁住 (Dead-G)

所以即使 explore.py 已经内嵌, **LLM 在 explore 之外仍被强制 propose
validate_stack**, 走老路径.

## 6. 修复路径

### Phase 0 — 验证 explore inline rebench 真工作

prerequisite: `Dead-A.5` (explore promote 修复) 完成. 跑 fresh session,
验证 explore round KEEP variant 后, executor 内部自动跑 stack rebench,
不需要 LLM propose validate_stack.

### Phase 1 — 删除 prompt mandatory

#### PR 1.1 — `prompt_builder.py:336-339`

删除 "MANDATORY validate_stack after KEEP" 段.

#### PR 1.2 — `prompt_builder.py:479-481`

删除决策第 3 步 `validate_stack required` TODO.

#### PR 1.3 — `orchestration.md:59, 100-103`

删除 MANDATORY 文字.

### Phase 2 — 删除 coordinator TODO 4

#### PR 2.1 — `_required_next_step` 删除 validate_stack 检查

`coordinator.py:1586-1660, 1852-1868` 删除 TODO 4 (validate_stack
required gate).

#### PR 2.2 — `_sequence_denial_for_action` 删除 validate_stack pre-req

`coordinator.py:1757-1764` 中 sequence_actions 不含 validate_stack 作
为 *gate*. 它仍可以是 *被 gate 的 action* (与 baseline 一样).

### Phase 3 — 删除 SharedState helpers

#### PR 3.1 — 移除 `optimization_stack_has_unvalidated_keeps`

`shared_state.py:~2383+` 删除 helper. 仅 TODO 4 用 (Phase 2 已删).

#### PR 3.2 — 标 `last_validate_stack` 为 deprecated (保留字段, 不删)

字段保留供 v0.6 resume (Inv-10.1 事实层不变), 但不再写新值.

### Phase 4 — 物理删除 (与 Dead-A 协同)

依赖 Gap-10. validate_stack yaml + executor + cli 注册一并删.

### Phase 5 — 测试重构

依赖 Dead-G. `tests/test_validate_stack.py` 删除; `tests/test_phase2_mission_and_validate_stack.py`
重构为"explore inline rebench 验收".

新增 `tests/test_v08_explore_inline_rebench.py`:

- explore round with 1 KEEP variant
- 验证 executor 在 KEEP 后自动重跑 stack
- 验证 result 中 keep_unstable_count 字段计数

## 7. 验收口径

- [ ] fresh session breakdown 中 `capability_summary.validate_stack`
      不存在或为空 (M3 §8 验收)
- [ ] LLM 在 EXPLORE phase 不 propose validate_stack (prompt 不要求)
- [ ] coordinator TODO 4 不再出现在 prompt
- [ ] explore round 内 KEEP variant 后立即触发 stack rebench (无 round-trip)
- [ ] breakdown.capability_summary.explore.keep_unstable_count 非零 (有
      stack rebench 发生)

## 8. 风险 / 回退

- **stack rebench 抖动** (R-14): explore 内 inline rebench 受 noise 影响,
  可能把刚 KEEP 的 variant 误判 unstable. 阈值默认保守 (0.5%); 必要时
  `--no-stack-rebench` flag degrade (M3 §PR5 设计).
- **回退**: 还原 prompt mandatory + coordinator TODO 4 = 重新走 legacy
  路径. 不影响 explore.py 内部 rebench.

## 9. 关联

- `Gap-10` (legacy actions allowed) — 关闭 validate_stack 入口
- `Dead-A.3` — 同一项, 不同视角
- `Dead-A.5` — 必须先做 (explore promote)
- `Dead-G` — 测试同步重构
