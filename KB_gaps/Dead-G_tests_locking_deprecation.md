# Dead-G — 测试反向锁住废止行为

> 风险等级: **MEDIUM-noise** (绿测试 = "deprecation 不允许"的看门狗)
> 体检报告: `../KB_design_gaps.MD` §12.8

## 1. 问题描述

8 个测试当前**绿**, 但断言的是 v0.8 应当废止的行为. 它们存在的唯一作
用就是 *不让 deprecation PR 通过*. 测试反向锁住已废止行为是 v0.8 上线
最大的隐形阻力.

```
"测试存在的唯一作用是不让删除发生" — 这是 §3.13 additive 原则的反面教材
```

## 2. 详细位置 + 锁住什么

| 测试 | 锁住什么 | 应当 |
|---|---|---|
| `tests/test_backends_search.py` | backends ledger 仍工作 | M3 GA 时 **delete** |
| `tests/test_params_search.py` | params ledger 仍工作 | M3 GA 时 **delete** |
| `tests/test_p2_3_param_executors.py` | params executor 接口 | M3 GA 时 **delete** |
| `tests/test_validate_stack.py` | validate_stack E2E | M3 GA 时 **delete** |
| `tests/test_phase2_mission_and_validate_stack.py` | mission TODO 4 | **重构** 为 explore inline rebench 验收 |
| `tests/test_prompt_builder.py` | 断言 `validate_stack` 在 enabled actions | **重构** |
| `tests/test_prompt_assets.py` | 断言 `validate_stack` 在两种模式 prompt 中 | **重构** |
| `tests/test_critic_prompt_builder.py` | critic prompt 含 validate_stack | **重构** |

## 3. 实际跑起来发生什么

任何尝试删除 backends/params/validate_stack 的 PR 会:
1. 删除 yaml + executor + cli 注册
2. 跑 `pytest` → 上述 8 个测试**全红**
3. CI fail → PR 不能 merge

理论上每个测试都应在删除 PR 内**一起改**, 但 PR 大小受限, reviewer 担
心改动过大. 结果: 8 个测试永远存在, 反向锁死.

## 4. 设计意图

§3.13 milestone "additive 原则" 第 3 项 "回退可性":

> 仅回退本里程碑的 PR 即可恢复上一里程碑的 session 行为, 不依赖 SharedState /
> Cortex 数据回滚.

设计上, **测试应当与代码 PR 同期改**, 不应当**长期**锁住废止行为. 实际:
- M3 PR9 (delete legacy) 没合, 测试仍在
- M2 PR (delete scoreboard) 合了, 但测试 (`test_action_scoring.py`)
  确实删了 — 这是正确做法

`test_action_scoring.py` (已删) 是好例子; `test_backends_search.py`
等 (没删) 是坏例子.

## 5. 根本原因

M3 PR 拆解时, "删 legacy executor" 和 "删测试" 是两个不同的 PR. M3 PR9
"delete legacy" 没合, 测试 PR (M3 PR10?) 自然也没合.

附加: 部分测试 (test_phase2_mission_and_validate_stack) 是 *验收测试*,
内容广, 改造比删除更复杂. 多个团队互相等 — 谁也没动.

## 6. 修复路径

### Phase 1 — 与 Dead-A / Dead-C 同 PR 删除纯锁定测试

#### PR 1.1 — 删除 4 个测试

```
inference_optimizer/tests/test_backends_search.py        → DELETE
inference_optimizer/tests/test_params_search.py          → DELETE
inference_optimizer/tests/test_p2_3_param_executors.py   → DELETE
inference_optimizer/tests/test_validate_stack.py         → DELETE
```

这些测试 100% 测 legacy 行为, 删除 = legacy 入口关闭后 CI 跑通.

### Phase 2 — 重构 4 个混合测试

#### PR 2.1 — `tests/test_phase2_mission_and_validate_stack.py` 重构

文件名改为 `tests/test_v08_phase2_mission.py`. 把 validate_stack TODO 4
断言删除, 改为 explore inline rebench 断言:

```text
def test_explore_round_keep_triggers_inline_rebench():
    # explore round with 1 KEEP variant
    # verify executor calls stack rebench internally
    # verify breakdown.capability_summary.explore.keep_unstable_count counts
```

#### PR 2.2 — `tests/test_prompt_builder.py` 重构

把断言 `validate_stack` / `backends` / `params` 在 enabled actions 中
的项改为断言 `explore` / `specialist`:

```text
def test_full_enabled_actions_v08():
    assert "explore" in FULL_ENABLED_ACTIONS
    assert "specialist" in FULL_ENABLED_ACTIONS
    # M3-removed actions should NOT be in enabled
    for legacy in ("backends", "params", "validate_stack"):
        assert legacy not in FULL_ENABLED_ACTIONS

def test_full_enabled_actions_excludes_deprecated():
    # also dream/re_explore/comm_optimization/compiler_tuning (Gap-13)
    for removed in ("dream", "re_explore", "comm_optimization", "compiler_tuning"):
        assert removed not in FULL_ENABLED_ACTIONS
```

#### PR 2.3 — `tests/test_prompt_assets.py` 重构

类似 PR 2.2, 把 `validate_stack` 在 prompt 中的断言改为 explore 主旋律.

#### PR 2.4 — `tests/test_critic_prompt_builder.py` 重构

critic prompt 升级到 verdict_map (Gap-11) 时一并改 — 断言 critic 看到
explore action 时返回 verdict_map.

### Phase 3 — CI lint 防止再次锁定

加 `tests/test_no_legacy_action_assertions.py`:

```text
def test_no_test_asserts_legacy_action_in_enabled():
    """v0.8 §3.15 §2.3 — backends/params/validate_stack must not appear
    in *positive* test assertions (they're deprecated, should only
    appear in negative assertions like 'not in')."""
    forbidden = ("backends", "params", "validate_stack")
    base = Path(__file__).parent
    for test_file in base.glob("test_*.py"):
        content = test_file.read_text()
        # Very rough heuristic — assert presence in enabled / allowed
        # patterns should not target legacy names
        for line in content.splitlines():
            for tok in forbidden:
                if (f'"{tok}" in' in line or f"'{tok}' in" in line) and \
                   "not in" not in line and "deprecated" not in line.lower():
                    if "FULL_ENABLED" in line or "PHASE_ALLOWED" in line:
                        pytest.fail(
                            f"{test_file.name}: legacy {tok!r} positive "
                            f"assertion at line: {line}"
                        )
```

(粗糙启发式 — 实际 CI 可能需要更精确.)

## 7. 验收口径

- [ ] 4 个 legacy 测试文件物理删除
- [ ] 4 个混合测试重构后跑绿
- [ ] CI lint 防止 legacy 名在 positive assertions 中再次出现
- [ ] M3 GA 后, 整套 `pytest` 不含任何 backends/params/validate_stack
      行为锁定 (除明确 "deprecated" 标记)

## 8. 风险 / 回退

- **删除测试丢失覆盖**: legacy executor / ledger 代码已删 (依赖 Dead-A /
  Gap-10), 测试丢失不影响覆盖.
- **重构混合测试引入 bug**: 重构本身可能引入测试 bug. 缓解: 重构 PR 单
  独 review, 每个测试改一行就跑一次本地 pytest.
- **回退**: revert PR 即恢复测试 (但 legacy 代码已删 → 测试红 → 必须连
  legacy 代码也 revert).

## 9. 关联

- `Dead-A` (legacy action 链) — 必须先做或同期做
- `Dead-C` (validate_stack 死路径) — 必须先做或同期做
- `Gap-10` (legacy actions allowed) — 同链
- `Gap-13` (legacy stub yamls) — Phase 3 CI lint 一并覆盖
